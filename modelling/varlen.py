import dataclasses

import torch
import torch.nn.functional as F
from torch import Tensor


# packed batch of variable-length sequences. tokens of all samples are concatenated along the sequence dim,
# so activations have shape [1, T, ...] with T = sum(lens). this keeps existing [B, L, ...] kernels working.
@dataclasses.dataclass(frozen=True)
class Varlen:
    lens: tuple[int, ...]
    cu_seqlens: Tensor  # [N+1] int32
    seq_ids: Tensor  # [T] int32, sample index of each token. used to look up per-sample conditioning

    @staticmethod
    def from_lens(lens: list[int] | tuple[int, ...], device: torch.types.Device = None) -> "Varlen":
        lens_t = torch.tensor(lens, dtype=torch.int32)
        cu_seqlens = F.pad(lens_t.cumsum(0, dtype=torch.int32), (1, 0))
        seq_ids = torch.repeat_interleave(torch.arange(len(lens), dtype=torch.int32), lens_t)
        if device is not None:
            cu_seqlens, seq_ids = cu_seqlens.to(device), seq_ids.to(device)
        return Varlen(tuple(lens), cu_seqlens, seq_ids)

    @property
    def max_seqlen(self) -> int:
        return max(self.lens)

    def split(self, x: Tensor, dim: int = 1) -> tuple[Tensor, ...]:
        return x.split(self.lens, dim=dim)


# x: [1, T, ...] packed. returns [1, T, ...] where x[:, segments of A] and y[:, segments of B] are interleaved
# per sample: [a_0, b_0, a_1, b_1, ...]. e.g. joint [img, txt] sequence from separately packed img and txt.
def interleave(a: Tensor, a_varlen: Varlen, b: Tensor, b_varlen: Varlen, dim: int = 1) -> Tensor:
    chunks = [t for pair in zip(a_varlen.split(a, dim), b_varlen.split(b, dim)) for t in pair]
    return torch.cat(chunks, dim=dim)


# per-sample conditioning [N, ...] -> per-token [1, T, ...]
def expand_per_token(cond: Tensor, varlen: Varlen) -> Tensor:
    return cond[varlen.seq_ids.long()].unsqueeze(0)


def varlen_attn(q: Tensor, k: Tensor, v: Tensor, varlen: Varlen) -> Tensor:
    # q, k, v: [1, T, H, D] packed
    if q.is_cuda:
        from torch.nn.attention.varlen import varlen_attn as _varlen_attn

        cu, max_len = varlen.cu_seqlens, varlen.max_seqlen
        return _varlen_attn(q[0], k[0], v[0], cu, cu, max_len, max_len).unsqueeze(0)

    # reference: attend within each sample separately
    outs = [
        F.scaled_dot_product_attention(q_.transpose(1, 2), k_.transpose(1, 2), v_.transpose(1, 2)).transpose(1, 2)
        for q_, k_, v_ in zip(varlen.split(q), varlen.split(k), varlen.split(v))
    ]
    return torch.cat(outs, dim=1)
