from collections.abc import Callable
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor


# layout of a packed batch: samples concatenated along the sequence dim. blocks run on [1, L, ...].
# blocks must not read Python lengths, otherwise torch.compile recompiles for every batch composition.
class Varlen(NamedTuple):
    cu_seqlens: Tensor  # [N+1] int32
    seq_ids: Tensor  # [L] int32, sample of each token
    max_seqlen: int  # rounded up to reduce recompiles
    perm: Tensor | None = None  # from cat_varlen(): token order with each sample contiguous, for attention

    @staticmethod
    def from_lens(lens: list[int], device: torch.types.Device = None) -> "Varlen":
        lens_t = torch.tensor(lens, dtype=torch.int32)
        cu_seqlens = F.pad(lens_t.cumsum(0, dtype=torch.int32), (1, 0))
        seq_ids = torch.repeat_interleave(torch.arange(len(lens), dtype=torch.int32), lens_t)
        return Varlen(cu_seqlens.to(device), seq_ids.to(device), -(-max(lens) // 256) * 256)


# layout of torch.cat([a, b]) of two packed streams e.g. [all txt, all img]. computed on device
def cat_varlen(a: Varlen, b: Varlen) -> Varlen:
    La, Lb = a.seq_ids.shape[0], b.seq_ids.shape[0]
    device = a.seq_ids.device
    # position of each token when samples are contiguous: [a_0, b_0, a_1, b_1, ...]
    a_pos = torch.arange(La, device=device) + b.cu_seqlens.index_select(0, a.seq_ids)
    b_pos = torch.arange(Lb, device=device) + a.cu_seqlens[1:].index_select(0, b.seq_ids)
    perm = torch.empty(La + Lb, dtype=torch.long, device=device)
    perm.index_copy_(0, a_pos, torch.arange(La, device=device))
    perm.index_copy_(0, b_pos, torch.arange(La, La + Lb, device=device))
    seq_ids = torch.cat([a.seq_ids, b.seq_ids])
    return Varlen(a.cu_seqlens + b.cu_seqlens, seq_ids, a.max_seqlen + b.max_seqlen, perm)


# helpers below take None for a dense batch. a single packed sample also takes the dense path
def is_packed(varlen: Varlen | None) -> bool:
    return varlen is not None and varlen.cu_seqlens.shape[0] > 2


def seq_ids(varlen: Varlen | None) -> Tensor | None:
    return varlen.seq_ids if is_packed(varlen) else None


# [N, 1, D] -> [1, L, D] for packed input. idx: from seq_ids()
def per_token(cond: Tensor, idx: Tensor | None) -> Tensor:
    return cond if idx is None else cond[:, 0].index_select(0, idx).unsqueeze(0)


def varlen_attn(q: Tensor, k: Tensor, v: Tensor, varlen: Varlen) -> Tensor:
    # q, k, v: [1, L, H, D]. CUDA only
    from torch.nn.attention.varlen import varlen_attn as _varlen_attn

    if varlen.perm is not None:
        q, k, v = [x.index_select(1, varlen.perm) for x in (q, k, v)]
    cu, max_len = varlen.cu_seqlens, varlen.max_seqlen
    out = _varlen_attn(q[0], k[0], v[0], cu, cu, max_len, max_len).unsqueeze(0)
    if varlen.perm is not None:
        out = torch.empty_like(out).index_copy(1, varlen.perm, out)
    return out


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
# a list of latents is packed: samples are concatenated, and kwargs include txt_varlen and img_varlen
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
    # joint rope in the same layout as the joint sequence: [all first stream, all second stream]
    firsts, seconds = [], []
    for x, img, txt_ in zip(latents, imgs, txts, strict=True):
        lens = [txt_.shape[0], img.shape[0]] if txt_first else [img.shape[0], txt_.shape[0]]
        first, second = make_rope(x.shape[0] // p, x.shape[1] // p, txt_.shape[0]).split(lens)
        firsts.append(first)
        seconds.append(second)

    device = imgs[0].device
    return torch.cat(imgs), dict(
        txt=torch.cat(txts),
        rope=torch.cat(firsts + seconds),
        txt_varlen=Varlen.from_lens([x.shape[0] for x in txts], device),
        img_varlen=Varlen.from_lens([x.shape[0] for x in imgs], device),
    )


# sizes: latent (H, W) for a dense batch, or a list per sample
def unpatchify_outputs(
    out: Tensor, sizes: tuple[int, int] | list[tuple[int, int]], p: int, channels_first: bool = False
) -> Tensor | list[Tensor]:
    if isinstance(sizes, tuple):
        H, W = sizes
        return unpatchify(out, H // p, W // p, p, channels_first)
    outs = out.split([(H // p) * (W // p) for H, W in sizes])
    return [unpatchify(x, H // p, W // p, p, channels_first) for x, (H, W) in zip(outs, sizes)]
