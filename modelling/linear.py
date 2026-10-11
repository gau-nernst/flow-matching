import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from gn_kernels import quantize_fp8_1d, quantize_nvfp4_triton
from gn_kernels.cutedsl.sm120 import sm120_mm_fp8_1d2d
from torch import Tensor, nn


def nvfp4_mm(x: Tensor, xs: Tensor, xs2: Tensor, w: Tensor, ws: Tensor, ws2: Tensor, bias: Tensor | None = None):
    w = w.view(torch.float4_e2m1fn_x2)
    scale_type = [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise]
    swizzle = [F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE]
    return F.scaled_mm(x, w.T, [xs, xs2], scale_type, [ws, ws2], scale_type, swizzle, swizzle, bias)


class Linear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.empty(out_dim)) if bias else None

        def hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            if f"{prefix}weight_scale_2" in state_dict:
                # nvfp4 ckpt
                del self.weight
                self.register_buffer("input_scale", torch.empty((), dtype=torch.float32))
                self.register_buffer("weight", torch.empty(out_dim, in_dim // 2, dtype=torch.uint8))
                self.register_buffer("weight_scale", torch.empty(out_dim, in_dim // 16, dtype=torch.float8_e4m3fn))
                self.register_buffer("weight_scale_2", torch.empty((), dtype=torch.float32))

            elif f"{prefix}weight_scale" in state_dict:
                # fp8 tensor
                self.register_buffer("input_scale", torch.empty((), dtype=torch.float32))
                self.register_buffer("weight_scale", torch.empty((), dtype=torch.float32))

            elif f"{prefix}weight_scale_inv" in state_dict:
                # fp8 1d2d
                N, K = state_dict[f"{prefix}weight"].shape
                self.register_buffer("weight_scale_inv", torch.empty(N // 128, K // 128))

        self.register_load_state_dict_pre_hook(hook)

    def to_fp8_1d2d(self):
        w = self.weight.detach()
        assert w.dtype in (torch.bfloat16, torch.float32)
        N, K = w.shape
        w = w.view(N // 128, 128, K // 128, 128)
        scale = 448.0 * w.abs().amax(dim=(1, 3), keepdim=True).float().clip(1e-12).reciprocal()  # [N/128, 1, K/128, 1]
        w = (w * scale).clip(-448.0, 448.0).to(torch.float8_e4m3fn).view(N, K)

        self.weight = nn.Parameter(w, requires_grad=False)
        self.register_buffer("weight_scale_inv", scale.reciprocal().view(N // 128, K // 128))

    def init_lora(self, rank: int, scale: float = 1.0):
        self.lora_scale = scale

        out_dim, in_dim = self.weight.shape
        self.lora_a = nn.Parameter(torch.empty(rank, in_dim))
        self.lora_b = nn.Parameter(torch.empty(out_dim, rank))
        nn.init.kaiming_normal_(self.lora_a, a=5**0.5)
        nn.init.zeros_(self.lora_b)

    def forward(self, x: Tensor, add: Tensor | None = None) -> Tensor:
        *dims, in_dim = x.shape
        x = x.reshape(-1, in_dim)
        if add is not None:
            add = add.view(-1, add.shape[-1])

        if self.is_nvfp4():
            xq, xs = quantize_nvfp4_triton(x, self.input_scale)
            out = nvfp4_mm(xq, xs, self.input_scale, self.weight, self.weight_scale, self.weight_scale_2, self.bias)
            if add is not None:
                out = out + add

        elif self.is_fp8_tensor():
            xq = fp8_tensor_quantize(x, self.input_scale)
            out = F.scaled_mm(
                xq,
                self.weight.T,
                self.input_scale,
                F.ScalingType.TensorWise,
                self.weight_scale,
                F.ScalingType.TensorWise,
                bias=self.bias,
            )
            if add is not None:
                out = out + add

        elif self.is_fp8_1d2d():
            xq, xs = quantize_fp8_1d(x)
            out = sm120_mm_fp8_1d2d.mm(xq, xs, self.weight, self.weight_scale_inv, self.bias, add)

        else:
            # bf16
            out = F.linear(x, self.weight, self.bias)
            if add is not None:
                out = out + add

        if hasattr(self, "lora_a"):
            alpha = self.lora_scale / self.lora_a.shape[0]
            h = (x @ self.lora_a.to(x.dtype).T) * alpha
            out = out + h @ self.lora_b.to(x.dtype).T

        return out.view(*dims, -1)

    def is_nvfp4(self):
        return hasattr(self, "weight_scale_2")

    def is_fp8_tensor(self):
        ws = getattr(self, "weight_scale", None)
        return ws is not None and ws.shape == ()

    def is_fp8_1d2d(self):
        return hasattr(self, "weight_scale_inv")

    def extra_repr(self):
        extra = f"W={tuple(self.weight.shape)}"
        if self.is_nvfp4():
            extra += ", quant=nvfp4"
        return extra


@triton.jit(do_not_specialize=["M"])
def _fp8_tensor_quantize_kernel(
    x_ptr,
    o_ptr,
    s_ptr,
    M,
    N,
    stride_xm,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # x and o: [M, N]
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    scale = tl.load(s_ptr)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m < M) & (offs_n < N)
    x = tl.load(x_ptr + offs_m * stride_xm + offs_n, mask)  # [BLOCK_M, BLOCK_N]
    x /= scale
    tl.store(o_ptr + offs_m * stride_om + offs_n, x, mask)


def fp8_tensor_quantize(x: Tensor, scale: Tensor):
    M, N = x.shape
    BLOCK_M = 128
    BLOCK_N = 128
    out = x.new_empty(M, N, dtype=torch.float8_e4m3fn)
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M), 1)
    _fp8_tensor_quantize_kernel[grid](x, out, scale, M, N, x.stride(0), out.stride(0), BLOCK_M, BLOCK_N)
    return out
