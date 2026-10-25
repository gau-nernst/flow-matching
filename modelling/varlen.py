import dataclasses

import torch
import torch.nn.functional as F
from torch import Tensor


# packed sequences, concatenated along the sequence dim. blocks run on [1, T, ...].
# blocks must not read lens, otherwise torch.compile recompiles for every batch composition.
@dataclasses.dataclass(frozen=True)
class Varlen:
    lens: tuple[int, ...]
    cu_seqlens: Tensor  # [N+1] int32
    seq_ids: Tensor  # [T] int32, sample of each token
    max_seqlen: int  # rounded up to reduce recompiles
    num_seqs: int

    @staticmethod
    def from_lens(lens: list[int] | tuple[int, ...], device: torch.types.Device = None) -> "Varlen":
        lens_t = torch.tensor(lens, dtype=torch.int32)
        cu_seqlens = F.pad(lens_t.cumsum(0, dtype=torch.int32), (1, 0))
        seq_ids = torch.repeat_interleave(torch.arange(len(lens), dtype=torch.int32), lens_t)
        if device is not None:
            cu_seqlens, seq_ids = cu_seqlens.to(device), seq_ids.to(device)
        max_seqlen = -(-max(lens) // 256) * 256
        return Varlen(tuple(lens), cu_seqlens, seq_ids, max_seqlen, len(lens))


# two packed streams, interleaved per sample: [a_0, b_0, a_1, b_1, ...]
@dataclasses.dataclass(frozen=True)
class Joint:
    a: Varlen
    b: Varlen
    varlen: Varlen  # of the joint sequence
    gather: Tensor  # [Ta + Tb] index into cat([a, b])
    a_pos: Tensor  # [Ta] positions of a tokens in the joint sequence
    b_pos: Tensor  # [Tb]

    @staticmethod
    def from_lens(a_lens: list[int], b_lens: list[int], device: torch.types.Device = None) -> "Joint":
        Ta = sum(a_lens)
        gather, a_pos, b_pos = [], [], []
        a_off = b_off = joint_off = 0
        for La, Lb in zip(a_lens, b_lens, strict=True):
            gather += [torch.arange(a_off, a_off + La), torch.arange(Ta + b_off, Ta + b_off + Lb)]
            a_pos.append(torch.arange(joint_off, joint_off + La))
            b_pos.append(torch.arange(joint_off + La, joint_off + La + Lb))
            a_off, b_off, joint_off = a_off + La, b_off + Lb, joint_off + La + Lb
        a, b = Varlen.from_lens(a_lens, device), Varlen.from_lens(b_lens, device)
        varlen = Varlen.from_lens([La + Lb for La, Lb in zip(a_lens, b_lens)], device)
        return Joint(a, b, varlen, *(torch.cat(x).to(device) for x in (gather, a_pos, b_pos)))


# helpers below take None for a dense batch. a single packed sample also takes the dense path
def is_packed(varlen: Varlen | Joint | None) -> bool:
    if isinstance(varlen, Joint):
        varlen = varlen.varlen
    return varlen is not None and varlen.num_seqs > 1


def merge(a: Tensor, b: Tensor, joint: Joint | None, dim: int = 1) -> Tensor:
    out = torch.cat([a, b], dim=dim)
    return out.index_select(dim, joint.gather) if is_packed(joint) else out


def split(x: Tensor, a_len: int, joint: Joint | None, dim: int = 1) -> tuple[Tensor, Tensor]:
    if is_packed(joint):
        return x.index_select(dim, joint.a_pos), x.index_select(dim, joint.b_pos)
    return x.narrow(dim, 0, a_len), x.narrow(dim, a_len, x.shape[dim] - a_len)


def seq_ids(varlen: Varlen | None) -> Tensor | None:
    return varlen.seq_ids if is_packed(varlen) else None


# [N, 1, D] -> [1, T, D] for packed input
def per_token(cond: Tensor, varlen: Varlen | None) -> Tensor:
    if not is_packed(varlen):
        return cond
    return cond[:, 0].index_select(0, varlen.seq_ids).unsqueeze(0)


def varlen_attn(q: Tensor, k: Tensor, v: Tensor, varlen: Varlen) -> Tensor:
    # CUDA only
    from torch.nn.attention.varlen import varlen_attn as _varlen_attn

    cu, max_len = varlen.cu_seqlens, varlen.max_seqlen
    return _varlen_attn(q[0], k[0], v[0], cu, cu, max_len, max_len).unsqueeze(0)


# [..., H, W, C] -> [..., H/p * W/p, p * p * C]. channels_first: (C, p, p) order, for Flux.1
def patchify(x: Tensor, p: int, channels_first: bool = False) -> Tensor:
    *B, H, W, C = x.shape
    n = len(B)
    x = x.reshape(*B, H // p, p, W // p, p, C)
    order = (0, 2, 4, 1, 3) if channels_first else (0, 2, 1, 3, 4)
    x = x.permute(*range(n), *(n + i for i in order))
    return x.reshape(*B, (H // p) * (W // p), p * p * C)


def unpatchify(x: Tensor, nH: int, nW: int, p: int, channels_first: bool = False) -> Tensor:
    *B, _, D = x.shape
    n = len(B)
    C = D // (p * p)
    if channels_first:
        x = x.reshape(*B, nH, nW, C, p, p)
        order = (0, 3, 1, 4, 2)
    else:
        x = x.reshape(*B, nH, nW, p, p, C)
        order = (0, 2, 1, 3, 4)
    x = x.permute(*range(n), *(n + i for i in order))
    return x.reshape(*B, nH * p, nW * p, C)


# returns (img, kwargs) for model(img, t, **kwargs). ropes: per-sample joint ropes
def pack(
    imgs: list[Tensor], txts: list[Tensor], ropes: list[Tensor], txt_first: bool
) -> tuple[Tensor, dict[str, Tensor | Joint]]:
    img_lens, txt_lens = [x.shape[0] for x in imgs], [x.shape[0] for x in txts]
    lens = (txt_lens, img_lens) if txt_first else (img_lens, txt_lens)
    joint = Joint.from_lens(*lens, imgs[0].device)
    return torch.cat(imgs), dict(txt=torch.cat(txts), rope=torch.cat(ropes), varlen=joint)
