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

    # metadata of torch.cat([self_tensor, other_tensor], dim=-2). the tensor keeps the two streams apart,
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


# [B, 1, D] -> [1, L, D] for a varlen batch. idx: VarlenMetadata.seq_ids, or None for a uniform batch
def expand_to_tokens(cond: Tensor, idx: Tensor | None) -> Tensor:
    return cond if idx is None else cond[:, 0].index_select(0, idx).unsqueeze(0)


# returns (img, kwargs) for model(img, t, **kwargs) of a varlen batch, even when all samples have the same size: the
# layout follows the input type (a list), not its contents, so calls with different texts for the same images agree.
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
