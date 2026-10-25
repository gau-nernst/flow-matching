from collections.abc import Callable
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor


# packed batch: samples concatenated along the first dim. a plain container, not a Tensor subclass.
# blocks get it with data=None, and must not read Python lengths, otherwise torch.compile recompiles
# for every batch composition.
class VarlenTensor(NamedTuple):
    data: Tensor | None  # [L, ...]
    cu_seqlens: Tensor  # [N+1] int32
    seq_ids: Tensor  # [L] int32, sample of each token
    max_seqlen: int  # rounded up to reduce recompiles

    @staticmethod
    def pack(xs: list[Tensor]) -> "VarlenTensor":
        lens = torch.tensor([x.shape[0] for x in xs], dtype=torch.int32)
        cu_seqlens = F.pad(lens.cumsum(0, dtype=torch.int32), (1, 0))
        seq_ids = torch.repeat_interleave(torch.arange(len(xs), dtype=torch.int32), lens)
        max_seqlen = -(-max(lens.tolist()) // 256) * 256
        device = xs[0].device
        return VarlenTensor(torch.cat(xs), cu_seqlens.to(device), seq_ids.to(device), max_seqlen)

    def meta(self) -> "VarlenTensor":
        return self._replace(data=None)


# (joint, gather, first_pos, second_pos)
Zipped = tuple[VarlenTensor, Tensor, Tensor, Tensor]


# interleave two packed sequences per sample: [first_0, second_0, first_1, second_1, ...]. computed on device.
# returns the joint sequence, the index into cat([first, second]) and positions of each stream in the joint
def zip_varlen(first: VarlenTensor, second: VarlenTensor) -> Zipped:
    L1, L2 = first.seq_ids.shape[0], second.seq_ids.shape[0]
    cu_seqlens = first.cu_seqlens + second.cu_seqlens
    device = cu_seqlens.device
    ids = torch.arange(cu_seqlens.shape[0] - 1, dtype=torch.int32, device=device)
    seq_ids = torch.repeat_interleave(ids, cu_seqlens.diff(), output_size=L1 + L2)
    pos1 = torch.arange(L1, device=device) + second.cu_seqlens.index_select(0, first.seq_ids)
    pos2 = torch.arange(L2, device=device) + first.cu_seqlens[1:].index_select(0, second.seq_ids)
    gather = torch.empty(L1 + L2, dtype=torch.long, device=device)
    gather.index_copy_(0, pos1, torch.arange(L1, device=device))
    gather.index_copy_(0, pos2, torch.arange(L1, L1 + L2, device=device))
    joint = VarlenTensor(None, cu_seqlens, seq_ids, first.max_seqlen + second.max_seqlen)
    return joint, gather, pos1, pos2


# helpers below take None for a dense batch. a single packed sample also takes the dense path
def is_packed(varlen: VarlenTensor | None) -> bool:
    return varlen is not None and varlen.cu_seqlens.shape[0] > 2


def merge(first: Tensor, second: Tensor, zipped: Zipped | None, dim: int = 1) -> Tensor:
    out = torch.cat([first, second], dim=dim)
    return out.index_select(dim, zipped[1]) if zipped is not None and is_packed(zipped[0]) else out


def split(x: Tensor, first_len: int, zipped: Zipped | None, dim: int = 1) -> tuple[Tensor, Tensor]:
    if zipped is not None and is_packed(zipped[0]):
        return x.index_select(dim, zipped[2]), x.index_select(dim, zipped[3])
    return x.narrow(dim, 0, first_len), x.narrow(dim, first_len, x.shape[dim] - first_len)


def seq_ids(varlen: VarlenTensor | None) -> Tensor | None:
    return varlen.seq_ids if is_packed(varlen) else None


# [N, 1, D] -> [1, L, D] for packed input
def per_token(cond: Tensor, varlen: VarlenTensor | None) -> Tensor:
    if not is_packed(varlen):
        return cond
    return cond[:, 0].index_select(0, varlen.seq_ids).unsqueeze(0)


def varlen_attn(q: Tensor, k: Tensor, v: Tensor, varlen: VarlenTensor) -> Tensor:
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


# returns (img, kwargs) for model(img, t, **kwargs). make_rope(nH, nW, L_txt): one sample's joint rope.
# a list of latents is packed: img and txt become VarlenTensor
def prepare_inputs(
    latents: Tensor | list[Tensor],
    txt: Tensor | list[Tensor],
    make_rope: Callable[[int, int, int], Tensor],
    p: int,
    channels_first: bool = False,
) -> tuple[Tensor | VarlenTensor, dict]:
    if isinstance(latents, Tensor):
        _, H, W, _ = latents.shape
        return patchify(latents, p, channels_first), dict(txt=txt, rope=make_rope(H // p, W // p, txt.shape[1]))

    txts = list(txt.unbind(0)) if isinstance(txt, Tensor) else txt
    imgs = [patchify(x, p, channels_first) for x in latents]
    ropes = [make_rope(x.shape[0] // p, x.shape[1] // p, y.shape[0]) for x, y in zip(latents, txts, strict=True)]
    return VarlenTensor.pack(imgs), dict(txt=VarlenTensor.pack(txts), rope=torch.cat(ropes))


# sizes: latent (H, W) for a dense batch, or a list per sample
def unpatchify_outputs(
    out: Tensor, sizes: tuple[int, int] | list[tuple[int, int]], p: int, channels_first: bool = False
) -> Tensor | list[Tensor]:
    if isinstance(sizes, tuple):
        H, W = sizes
        return unpatchify(out, H // p, W // p, p, channels_first)
    outs = out.split([(H // p) * (W // p) for H, W in sizes])
    return [unpatchify(x, H // p, W // p, p, channels_first) for x, (H, W) in zip(outs, sizes)]
