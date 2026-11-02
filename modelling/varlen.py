from collections.abc import Callable
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor


# where each sample sits in a packed tensor [1, L, ...], whose samples are concatenated along L
class VarlenMetadata(NamedTuple):
    cu_seqlens: Tensor  # [N+1] int32
    seq_ids: Tensor  # [L] int32, sample of each token
    max_seqlen: int  # upper bound on sample length
    perm: Tensor | None = None  # from concat(), for attention

    @staticmethod
    def from_lens(lens: list[int], device: torch.types.Device = None) -> "VarlenMetadata":
        lens_t = torch.tensor(lens, dtype=torch.int32)
        cu_seqlens = F.pad(lens_t.cumsum(0, dtype=torch.int32), (1, 0))
        seq_ids = torch.repeat_interleave(torch.arange(len(lens), dtype=torch.int32), lens_t)
        return VarlenMetadata(cu_seqlens.to(device), seq_ids.to(device), max(lens))

    # metadata of torch.cat([self_tensor, other_tensor], dim=1). the tensor keeps the two streams apart,
    # so perm is needed for attention, which wants each sample contiguous:
    #   self lens [2, 3], other lens [1, 2]
    #   tensor order:     a0 a0 a1 a1 a1 b0 b1 b1
    #   attention order:  a0 a0 b0 a1 a1 a1 b1 b1
    #   perm = [0, 1, 5, 2, 3, 4, 6, 7], cu_seqlens = [0, 3, 8]
    def concat(self, other: "VarlenMetadata") -> "VarlenMetadata":
        La, Lb = self.seq_ids.shape[0], other.seq_ids.shape[0]
        device = self.seq_ids.device
        a_pos = torch.arange(La, device=device) + other.cu_seqlens.index_select(0, self.seq_ids)
        b_pos = torch.arange(Lb, device=device) + self.cu_seqlens[1:].index_select(0, other.seq_ids)
        perm = torch.empty(La + Lb, dtype=torch.long, device=device)
        perm.index_copy_(0, a_pos, torch.arange(La, device=device))
        perm.index_copy_(0, b_pos, torch.arange(La, La + Lb, device=device))
        seq_ids = torch.cat([self.seq_ids, other.seq_ids])
        cu_seqlens = self.cu_seqlens + other.cu_seqlens
        return VarlenMetadata(cu_seqlens, seq_ids, self.max_seqlen + other.max_seqlen, perm)


# [N, 1, D] -> [1, L, D] for packed input. idx: VarlenMetadata.seq_ids, or None for a dense batch
def per_token(cond: Tensor, idx: Tensor | None) -> Tensor:
    return cond if idx is None else cond[:, 0].index_select(0, idx).unsqueeze(0)


def varlen_attn(q: Tensor, k: Tensor, v: Tensor, varlen: VarlenMetadata) -> Tensor:
    # q, k, v: [1, L, H, D]. CUDA only
    from torch.nn.attention.varlen import varlen_attn as _varlen_attn

    if varlen.perm is not None:
        q, k, v = [x.index_select(1, varlen.perm) for x in (q, k, v)]
    cu, max_len = varlen.cu_seqlens, varlen.max_seqlen
    out = _varlen_attn(q[0], k[0], v[0], cu, cu, max_len, max_len).unsqueeze(0)
    if varlen.perm is not None:
        out = torch.empty_like(out).index_copy(1, varlen.perm, out)
    return out


# returns (img, kwargs) for model(img, t, **kwargs). a single sample stays a dense batch, otherwise samples are packed.
# imgs: patchified [L_i, D]. grids: patchified (H, W) of each sample. make_rope(H, W, L_txt): one sample's joint rope
def pack_inputs(
    imgs: list[Tensor],
    txt: Tensor | list[Tensor],
    grids: list[tuple[int, int]],
    make_rope: Callable[[int, int, int], Tensor],
    txt_first: bool,
) -> tuple[Tensor, dict]:
    txts = list(txt)
    if len(imgs) == 1:
        return imgs[0][None], dict(txt=txts[0][None], rope=make_rope(*grids[0], txts[0].shape[0]))

    # joint rope in the same order as the joint sequence: [all first stream, all second stream]
    firsts, seconds = [], []
    for img, txt_, (H, W) in zip(imgs, txts, grids, strict=True):
        lens = [txt_.shape[0], img.shape[0]] if txt_first else [img.shape[0], txt_.shape[0]]
        first, second = make_rope(H, W, txt_.shape[0]).split(lens)
        firsts.append(first)
        seconds.append(second)

    device = imgs[0].device
    return torch.cat(imgs), dict(
        txt=torch.cat(txts),
        rope=torch.cat(firsts + seconds),
        txt_varlen=VarlenMetadata.from_lens([x.shape[0] for x in txts], device),
        img_varlen=VarlenMetadata.from_lens([x.shape[0] for x in imgs], device),
    )
