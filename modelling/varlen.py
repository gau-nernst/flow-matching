import dataclasses

import torch
import torch.nn.functional as F
from torch import Tensor


# packed batch of variable-length sequences. tokens of all samples are concatenated along the sequence dim,
# so activations have shape [1, T, ...] with T = sum(lens). this keeps existing [B, L, ...] kernels working.
#
# code inside model blocks should only use the tensor fields and max_seqlen, not lens. torch.compile guards
# on the Python values a block reads, so reading lens would recompile for every new batch composition.
@dataclasses.dataclass(frozen=True)
class Varlen:
    lens: tuple[int, ...]
    cu_seqlens: Tensor  # [N+1] int32
    seq_ids: Tensor  # [T] int32, sample index of each token. used to look up per-sample conditioning
    max_seqlen: int  # rounded up, to reduce recompiles. only used to size attention kernels
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

    def split(self, x: Tensor, dim: int = 1) -> tuple[Tensor, ...]:
        return x.split(self.lens, dim=dim)


# joint sequence made of two packed streams a and b, concatenated per sample: [a_0, b_0, a_1, b_1, ...].
# e.g. [txt, img] in Flux, [img, txt] in Z-Image and MOSS.
@dataclasses.dataclass(frozen=True)
class Joint:
    a: Varlen
    b: Varlen
    varlen: Varlen  # of the joint sequence
    gather: Tensor  # [Ta + Tb] index into cat([a, b]) for each joint token
    a_pos: Tensor  # [Ta] positions of a tokens in the joint sequence
    b_pos: Tensor  # [Tb]

    @staticmethod
    def from_varlens(a: Varlen, b: Varlen, device: torch.types.Device = None) -> "Joint":
        Ta = sum(a.lens)
        gather, a_pos, b_pos = [], [], []
        a_off = b_off = joint_off = 0
        for La, Lb in zip(a.lens, b.lens):
            gather += [torch.arange(a_off, a_off + La), torch.arange(Ta + b_off, Ta + b_off + Lb)]
            a_pos.append(torch.arange(joint_off, joint_off + La))
            b_pos.append(torch.arange(joint_off + La, joint_off + La + Lb))
            a_off, b_off, joint_off = a_off + La, b_off + Lb, joint_off + La + Lb
        varlen = Varlen.from_lens([La + Lb for La, Lb in zip(a.lens, b.lens)], device)
        return Joint(a, b, varlen, *(torch.cat(x).to(device) for x in (gather, a_pos, b_pos)))

    # a single sample is just [a, b], so skip the index copies. lengths come from tensor shapes, not Python
    # values, so torch.compile doesn't specialize on them.
    def merge(self, a: Tensor, b: Tensor, dim: int = 1) -> Tensor:
        out = torch.cat([a, b], dim=dim)
        return out if self.varlen.num_seqs == 1 else out.index_select(dim, self.gather)

    def split(self, x: Tensor, dim: int = 1) -> tuple[Tensor, Tensor]:
        if self.varlen.num_seqs == 1:
            Ta = self.a_pos.shape[0]
            return x.narrow(dim, 0, Ta), x.narrow(dim, Ta, x.shape[dim] - Ta)
        return x.index_select(dim, self.a_pos), x.index_select(dim, self.b_pos)


# per-sample conditioning [N, 1, D] -> per-token [1, T, D] for packed input. no-op without varlen
def per_token(cond: Tensor, varlen: Varlen | None) -> Tensor:
    if varlen is None:
        return cond
    return cond[:, 0].index_select(0, varlen.seq_ids).unsqueeze(0)


def varlen_attn(q: Tensor, k: Tensor, v: Tensor, varlen: Varlen) -> Tensor:
    # q, k, v: [1, T, H, D] packed
    if q.is_cuda:
        from torch.nn.attention.varlen import varlen_attn as _varlen_attn

        cu, max_len = varlen.cu_seqlens, varlen.max_seqlen
        return _varlen_attn(q[0], k[0], v[0], cu, cu, max_len, max_len).unsqueeze(0)

    # reference: block-diagonal mask, so tokens only attend within their sample. O(T^2) memory
    mask = varlen.seq_ids[:, None] == varlen.seq_ids[None, :]
    q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2)
