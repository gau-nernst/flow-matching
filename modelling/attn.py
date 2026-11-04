import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.varlen import varlen_attn

from .varlen import VarlenMetadata


def dispatch_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    impl: str = "pt",
    varlen: VarlenMetadata | None = None,
    kv_varlen: VarlenMetadata | None = None,
) -> Tensor:
    # q, k, v: [B, L, H, D], or [1, L, H, D] if varlen. kv_varlen: for k, v of another sequence (cross-attention),
    # otherwise they follow varlen
    if varlen is not None:
        if impl != "pt":
            raise NotImplementedError(f"varlen attention is not supported for {impl=}")
        kv_varlen = kv_varlen or varlen
        # varlen_attn is CUDA only, and needs each sample contiguous
        if varlen.perm is not None:
            q = q.index_select(1, varlen.perm)
        if kv_varlen.perm is not None:
            k, v = [x.index_select(1, kv_varlen.perm) for x in (k, v)]
        cu_q, cu_k = varlen.cu_seqlens, kv_varlen.cu_seqlens
        out = varlen_attn(q[0], k[0], v[0], cu_q, cu_k, varlen.max_seqlen, kv_varlen.max_seqlen).unsqueeze(0)
        if varlen.perm is not None:
            out = torch.empty_like(out).index_copy_(1, varlen.perm, out)
        return out

    if impl == "pt":
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
        return F.scaled_dot_product_attention(q, k, v).transpose(1, 2)

    elif impl == "qk-fp8":
        from gn_kernels.cutedsl.sm120 import sm120_attn

        return sm120_attn.attn(q.to(torch.float8_e4m3fn), k.to(torch.float8_e4m3fn), v)

    raise NotImplementedError(f"{impl=}")
