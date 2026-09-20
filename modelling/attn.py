import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from gn_kernels.cutedsl.sm120 import sm120_attn_qkfp8

except ImportError:
    pass


def dispatch_attn(q: Tensor, k: Tensor, v: Tensor, impl: str = "pt") -> Tensor:
    if impl == "pt":
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
        return F.scaled_dot_product_attention(q, k, v, enable_gqa=True).transpose(1, 2)

    elif impl == "qk-fp8":
        return sm120_attn_qkfp8.attn(q.to(torch.float8_e4m3fn), k.to(torch.float8_e4m3fn), v)

    raise NotImplementedError(f"{impl=}")
