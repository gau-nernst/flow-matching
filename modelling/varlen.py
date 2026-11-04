from collections.abc import Callable
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .utils import list_to_tensor_async


# a batch is either uniform, with shape [B, L, ...], or varlen, with shape [1, L, ...]: B samples concatenated
# along L. this describes where each sample sits in a varlen batch
class VarlenMetadata(NamedTuple):
    cu_seqlens: Tensor  # [B+1] int32
    seq_ids: Tensor  # [L] int32, sample of each token
    max_seqlen: int  # upper bound on sample length
    perm: Tensor | None = None  # from concat(), for attention

    @staticmethod
    def from_lens(lens: list[int], device: torch.types.Device) -> "VarlenMetadata":
        lens_t = list_to_tensor_async(lens, device, torch.int32)
        cu_seqlens = F.pad(lens_t.cumsum(0, dtype=torch.int32), (1, 0))
        # output_size: otherwise it reads the sum back from the GPU
        ids = torch.arange(len(lens), dtype=torch.int32, device=device)
        seq_ids = torch.repeat_interleave(ids, lens_t, output_size=sum(lens))
        return VarlenMetadata(cu_seqlens, seq_ids, max(lens))

    # metadata of torch.cat([self_tensor, *other_tensors], dim=-2). the tensor keeps the streams apart,
    # so perm is needed for attention, which wants each sample contiguous:
    #   self lens [2, 3], other lens [1, 2]
    #   tensor order:     a0 a0 a1 a1 a1 b0 b1 b1
    #   attention order:  a0 a0 b0 a1 a1 a1 b1 b1
    #   perm = [0, 1, 5, 2, 3, 4, 6, 7], cu_seqlens = [0, 3, 8]
    def concat(self, *others: "VarlenMetadata") -> "VarlenMetadata":
        parts = (self, *others)
        assert all(x.perm is None for x in parts), "concat of a concat is not supported"
        device = self.seq_ids.device
        cu_seqlens = sum(x.cu_seqlens for x in parts)
        perm = torch.empty(sum(x.seq_ids.shape[0] for x in parts), dtype=torch.long, device=device)
        start = 0  # tensor index of this part's first token
        offset = cu_seqlens[:-1]  # attention index of each sample's first token of this part
        for x in parts:
            L = x.seq_ids.shape[0]
            pos = torch.arange(L, device=device) + (offset - x.cu_seqlens[:-1]).index_select(0, x.seq_ids)
            perm.index_copy_(0, pos, torch.arange(start, start + L, device=device))
            offset = offset + x.cu_seqlens.diff()
            start += L
        seq_ids = torch.cat([x.seq_ids for x in parts])
        return VarlenMetadata(cu_seqlens, seq_ids, sum(x.max_seqlen for x in parts), perm)


# [B, 1, D] -> [1, L, D] for a varlen batch. idx: VarlenMetadata.seq_ids, or None for a uniform batch
def expand_to_tokens(cond: Tensor, idx: Tensor | None) -> Tensor:
    return cond if idx is None else cond[:, 0].index_select(0, idx).unsqueeze(0)


# imgs: list[LD] (patchified), txt: list[LD], grids: (H, W) of each patchified image
# make_rope(H, W, L_txt): one sample's joint rope
# -> img [1, sum(L), D], kwargs: txt [1, sum(L), D], rope, txt_varlen, img_varlen, joint_varlen
# imgs: patchified [L_i, D]. grids: patchified (H, W) of each sample. make_rope(H, W, L_txt): one sample's joint rope
def pack_inputs(
    imgs: list[Tensor],
    txt: Tensor | list[Tensor],
    grids: list[tuple[int, int]],
    make_rope: Callable[[int, int, int], Tensor],
    txt_first: bool,
) -> tuple[Tensor, dict]:
    txts = list(txt)
    assert len(imgs) == len(txts) == len(grids), (len(imgs), len(txts), len(grids))

    # joint rope in the same order as the joint sequence: [all first stream, all second stream]
    firsts, seconds = [], []
    for img, txt_, (H, W) in zip(imgs, txts, grids):
        lens = [txt_.shape[0], img.shape[0]] if txt_first else [img.shape[0], txt_.shape[0]]
        first, second = make_rope(H, W, txt_.shape[0]).split(lens)
        firsts.append(first)
        seconds.append(second)

    device = imgs[0].device
    txt_varlen = VarlenMetadata.from_lens([x.shape[0] for x in txts], device)
    img_varlen = VarlenMetadata.from_lens([x.shape[0] for x in imgs], device)
    return torch.cat(imgs)[None], dict(
        txt=torch.cat(txts)[None],
        rope=torch.cat(firsts + seconds),
        txt_varlen=txt_varlen,
        img_varlen=img_varlen,
        joint_varlen=txt_varlen.concat(img_varlen) if txt_first else img_varlen.concat(txt_varlen),
    )
