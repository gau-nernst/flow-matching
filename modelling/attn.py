import torch
import torch.nn.functional as F
from torch import Tensor

from .varlen import Varlen, is_packed, varlen_attn


def dispatch_attn(q: Tensor, k: Tensor, v: Tensor, impl: str = "pt", varlen: Varlen | None = None) -> Tensor:
    # q, k, v: [B, L, H, D], or [1, T, H, D] packed
    if is_packed(varlen):
        if impl != "pt":
            raise NotImplementedError(f"varlen attention is not supported for {impl=}")
        return varlen_attn(q, k, v, varlen)

    if impl == "pt":
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
        return F.scaled_dot_product_attention(q, k, v).transpose(1, 2)

    elif impl == "qk-fp8":
        from gn_kernels.cutedsl.sm120 import sm120_attn

        return sm120_attn.attn(q.to(torch.float8_e4m3fn), k.to(torch.float8_e4m3fn), v)

    raise NotImplementedError(f"{impl=}")
