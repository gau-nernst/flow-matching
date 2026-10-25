import dataclasses
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


# packed sequences, concatenated along the sequence dim. blocks run on [1, L, ...].
# blocks must not read lens, otherwise torch.compile recompiles for every batch composition.
@dataclasses.dataclass(frozen=True)
class Varlen:
    cu_seqlens: Tensor  # [N+1] int32
    seq_ids: Tensor  # [L] int32, sample of each token
    max_seqlen: int  # rounded up to reduce recompiles

    @property
    def num_seqs(self) -> int:
        return self.cu_seqlens.shape[0] - 1

    @staticmethod
    def from_lens(lens: list[int], device: torch.types.Device = None) -> "Varlen":
        lens_t = torch.tensor(lens, dtype=torch.int32)
        cu_seqlens = F.pad(lens_t.cumsum(0, dtype=torch.int32), (1, 0))
        seq_ids = torch.repeat_interleave(torch.arange(len(lens), dtype=torch.int32), lens_t)
        max_seqlen = -(-max(lens) // 256) * 256
        return Varlen(cu_seqlens.to(device), seq_ids.to(device), max_seqlen)


# packed txt and img streams, and the joint sequence interleaving them per sample:
# [txt_0, img_0, txt_1, img_1, ...] if txt_first, else [img_0, txt_0, ...]
@dataclasses.dataclass(frozen=True)
class VarlenInfo:
    txt: Varlen
    img: Varlen
    joint: Varlen
    txt_first: bool
    gather: Tensor  # index into cat([first, second])
    txt_pos: Tensor  # positions of txt tokens in the joint sequence
    img_pos: Tensor

    @staticmethod
    def from_lens(
        txt_lens: list[int], img_lens: list[int], txt_first: bool, device: torch.types.Device = None
    ) -> "VarlenInfo":
        first, second = (txt_lens, img_lens) if txt_first else (img_lens, txt_lens)
        T1 = sum(first)
        gather, pos1, pos2 = [], [], []
        off1 = off2 = joint_off = 0
        for L1, L2 in zip(first, second, strict=True):
            gather += [torch.arange(off1, off1 + L1), torch.arange(T1 + off2, T1 + off2 + L2)]
            pos1.append(torch.arange(joint_off, joint_off + L1))
            pos2.append(torch.arange(joint_off + L1, joint_off + L1 + L2))
            off1, off2, joint_off = off1 + L1, off2 + L2, joint_off + L1 + L2
        gather, pos1, pos2 = [torch.cat(x).to(device) for x in (gather, pos1, pos2)]
        txt_pos, img_pos = (pos1, pos2) if txt_first else (pos2, pos1)

        return VarlenInfo(
            txt=Varlen.from_lens(txt_lens, device),
            img=Varlen.from_lens(img_lens, device),
            joint=Varlen.from_lens([L1 + L2 for L1, L2 in zip(first, second)], device),
            txt_first=txt_first,
            gather=gather,
            txt_pos=txt_pos,
            img_pos=img_pos,
        )


# helpers below take None for a dense batch. a single packed sample also takes the dense path
def is_packed(varlen: Varlen | VarlenInfo | None) -> bool:
    if isinstance(varlen, VarlenInfo):
        varlen = varlen.joint
    return varlen is not None and varlen.num_seqs > 1


# first, second: txt and img streams, in the model's order
def merge(first: Tensor, second: Tensor, varlen: VarlenInfo | None, dim: int = 1) -> Tensor:
    out = torch.cat([first, second], dim=dim)
    return out.index_select(dim, varlen.gather) if is_packed(varlen) else out


def split(x: Tensor, first_len: int, varlen: VarlenInfo | None, dim: int = 1) -> tuple[Tensor, Tensor]:
    if is_packed(varlen):
        pos = (varlen.txt_pos, varlen.img_pos) if varlen.txt_first else (varlen.img_pos, varlen.txt_pos)
        return tuple(x.index_select(dim, p) for p in pos)
    return x.narrow(dim, 0, first_len), x.narrow(dim, first_len, x.shape[dim] - first_len)


def seq_ids(varlen: Varlen | None) -> Tensor | None:
    return varlen.seq_ids if is_packed(varlen) else None


# [N, 1, D] -> [1, L, D] for packed input
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


# make_rope(nH, nW, L_txt): one sample's joint rope
def prepare_inputs(
    latents: Tensor | list[Tensor],
    txt: Tensor | list[Tensor],
    make_rope: Callable[[int, int, int], Tensor],
    p: int,
    txt_first: bool,
    channels_first: bool = False,
) -> tuple[Tensor, dict]:
    if isinstance(latents, Tensor):
        _, H, W, _ = latents.shape
        return patchify(latents, p, channels_first), dict(txt=txt, rope=make_rope(H // p, W // p, txt.shape[1]))

    txts = list(txt.unbind(0)) if isinstance(txt, Tensor) else txt
    imgs = [patchify(x, p, channels_first) for x in latents]
    ropes = [make_rope(x.shape[0] // p, x.shape[1] // p, y.shape[0]) for x, y in zip(latents, txts, strict=True)]
    txt_lens, img_lens = [x.shape[0] for x in txts], [x.shape[0] for x in imgs]
    varlen = VarlenInfo.from_lens(txt_lens, img_lens, txt_first, imgs[0].device)
    return torch.cat(imgs), dict(txt=torch.cat(txts), rope=torch.cat(ropes), varlen=varlen)


# sizes: latent (H, W) for a dense batch, or a list per sample
def unpatchify_outputs(
    out: Tensor, sizes: tuple[int, int] | list[tuple[int, int]], p: int, channels_first: bool = False
) -> Tensor | list[Tensor]:
    if isinstance(sizes, tuple):
        H, W = sizes
        return unpatchify(out, H // p, W // p, p, channels_first)
    outs = out.split([(H // p) * (W // p) for H, W in sizes])
    return [unpatchify(x, H // p, W // p, p, channels_first) for x, (H, W) in zip(outs, sizes)]
