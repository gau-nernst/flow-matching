import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from gn_kernels import quantize_fp8_1d, quantize_nvfp4_triton
from gn_kernels.cutedsl.sm120 import sm120_mm_fp8_1d2d, sm120_mm_fp8_rowwise
from torch import Tensor, nn
from torch.utils.flop_counter import register_flop_formula


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

            elif f"{prefix}weight_scale" in state_dict and state_dict[f"{prefix}weight_scale"].ndim == 1:
                # fp8 row-wise weight, dynamic row-wise activation
                self.register_buffer("weight_scale", torch.empty(out_dim, dtype=torch.float32))

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

        elif self.is_fp8_rowwise():
            xq, xs = fp8_rowwise_quantize(x)
            out = sm120_mm_fp8_rowwise.mm(xq, xs, self.weight, self.weight_scale)
            if self.bias is not None:
                out = out + self.bias
            if add is not None:
                out = out + add

        elif self.is_fp8_1d2d():
            out = FP8_1d2dFunc.apply(x, self.weight, self.weight_scale_inv, self.bias)
            if add is not None:
                out = out + add

        else:
            # bf16
            out = F.linear(x, self.weight, self.bias)
            if add is not None:
                out = out + add

        if hasattr(self, "lora_a"):
            alpha = self.lora_scale / self.lora_a.shape[0]
            h = (x @ self.lora_a.to(x.dtype).T) * alpha
            out = LoraBFunc.apply(h, self.lora_b, out)

        return out.view(*dims, -1)

    def is_nvfp4(self):
        return hasattr(self, "weight_scale_2")

    def is_fp8_tensor(self):
        ws = getattr(self, "weight_scale", None)
        return ws is not None and ws.shape == ()

    def is_fp8_rowwise(self):
        ws = getattr(self, "weight_scale", None)
        return ws is not None and ws.ndim == 1

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


@triton.jit(do_not_specialize=["M"])
def _fp8_rowwise_quantize_kernel(x_ptr, o_ptr, s_ptr, M, N, stride_xm, stride_om, BLOCK_N: tl.constexpr):
    # each program handles 1 row. assume the whole row fits.
    pid_m = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < N
    x = tl.load(x_ptr + pid_m * stride_xm + offs_n, mask, other=0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x)) / 448.0, 1e-12)
    tl.store(s_ptr + pid_m, scale)
    tl.store(o_ptr + pid_m * stride_om + offs_n, tl.clamp(x / scale, -448.0, 448.0), mask)


# returns scale with shape [M, 1]
def fp8_rowwise_quantize(x: Tensor) -> tuple[Tensor, Tensor]:
    M, N = x.shape
    out = x.new_empty(M, N, dtype=torch.float8_e4m3fn)
    scale = x.new_empty(M, 1, dtype=torch.float32)
    BLOCK_N = triton.next_power_of_2(N)
    num_warps = 4 if BLOCK_N <= 4096 else 8
    _fp8_rowwise_quantize_kernel[(M,)](x, out, scale, M, N, x.stride(0), out.stride(0), BLOCK_N, num_warps=num_warps)
    return out, scale


class FP8_1d2dFunc(torch.autograd.Function):
    @staticmethod
    def forward(x: Tensor, w: Tensor, ws: Tensor, bias: Tensor | None):
        xq, xs = _quantize_fp8_1d(x)
        return fp8_1d2d_mm(xq, xs, w, ws, bias)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, w, ws, _ = inputs
        ctx.save_for_backward(w, ws)

    @staticmethod
    def backward(ctx, grad_y: Tensor):
        grad_x = grad_w = grad_ws = grad_bias = None
        w, ws = ctx.saved_tensors

        if ctx.needs_input_grad[0]:
            # TODO: support N-major weight
            grad_yq, grad_ys = _quantize_fp8_1d(grad_y)
            grad_x = fp8_1d2d_mm(grad_yq, grad_ys, w.T.contiguous(), ws.T)
        if ctx.needs_input_grad[3]:
            grad_bias = grad_y.sum(0)
        return grad_x, grad_w, grad_ws, grad_bias


@torch.library.custom_op("gn_flow_matching::fp8_1d2d_mm", mutates_args=(), device_types="cuda")
def fp8_1d2d_mm(x: Tensor, xs: Tensor, w: Tensor, ws: Tensor, bias: Tensor | None = None) -> Tensor:
    return sm120_mm_fp8_1d2d.mm(x, xs, w, ws, bias)


@fp8_1d2d_mm.register_fake
def _(x: Tensor, xs: Tensor, w: Tensor, ws: Tensor, bias: Tensor | None = None) -> Tensor:
    return x.new_empty(x.shape[0], w.shape[0], dtype=torch.bfloat16)


# required for correct interaction with activation checkpointing budget
@register_flop_formula(torch.ops.gn_flow_matching.fp8_1d2d_mm)
def _(x_shape, xs_shape, w_shape, ws_shape, bias_shape=None, *, out_shape=None, **kwargs) -> int:
    M, K = x_shape
    N, _ = w_shape
    return 2 * M * N * K // 2  # divide by 2 due to FP8


def _quantize_fp8_1d(x: Tensor):
    # decompose for torch.compile
    if torch.compiler.is_compiling():
        M, N = x.shape
        x = x.reshape(M, N // 128, 128)
        inv_scale = 448.0 * x.abs().amax(dim=2, keepdim=True).float().clip(1e-12).reciprocal()
        xq = (x * inv_scale).clip(-448.0, 448.0).to(torch.float8_e4m3fn).view(M, N)
        pad_M = triton.cdiv(M, 4) * 4 - M  # alignment
        scale = F.pad(inv_scale.reciprocal().squeeze(2).T, (0, pad_M)).contiguous().T[:M]  # alignment
        return xq, scale

    # hand-written Triton kernel
    return quantize_fp8_1d(x)


class LoraBFunc(torch.autograd.Function):
    @staticmethod
    def forward(x: Tensor, w: Tensor, add: Tensor) -> Tensor:
        return small_k_mm(x, w, add)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, w, _ = inputs
        ctx.save_for_backward(x, w)

    @staticmethod
    def backward(ctx, grad_y: Tensor):
        grad_x = grad_w = grad_add = None
        x, w = ctx.saved_tensors

        if ctx.needs_input_grad[0]:
            grad_x = grad_y @ w.to(grad_y.dtype)  # reduction dim = N
        if ctx.needs_input_grad[1]:
            grad_w = torch.mm(grad_y.T, x, out_dtype=w.dtype)  # reduction dim = M
        if ctx.needs_input_grad[2]:
            grad_add = grad_y
        return grad_x, grad_w, grad_add


@triton.jit
def _small_k_mm_kernel(
    x_ptr,  # [M, K]
    w_ptr,  # [N, K]
    add_ptr,  # [M, N]
    y_ptr,  # [M, N]
    stride_xm,
    stride_wn,
    stride_am,
    stride_ym,
    M,
    N,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0).to(tl.int64)
    pid_m = tl.program_id(1).to(tl.int64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + tl.arange(0, 2)
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + tl.arange(0, 2)

    mask = mask_m[:, None] & mask_n
    acc = tl.load(add_ptr + (offs_m[:, None] * stride_am + offs_n), mask).to(tl.float32)

    for _ in range(K // 2):
        x = tl.load(x_ptrs, mask_m[:, None])
        w = tl.load(w_ptrs, mask_n[:, None])
        x0, x1 = x.split()
        w0, w1 = w.split()
        acc += x0.to(tl.float32)[:, None] * w0.to(tl.float32)
        acc += x1.to(tl.float32)[:, None] * w1.to(tl.float32)
        x_ptrs += 2
        w_ptrs += 2

    tl.store(y_ptr + (offs_m[:, None] * stride_ym + offs_n), acc, mask)


def small_k_mm(x: Tensor, w: Tensor, add: Tensor):
    # there are 2 problems (x @ w.T + add) has with torch.compile
    # - the op is rewritten as torch.addmm(add, x, w.T). this performs a memcpy on add before calling CuBLAS,
    #   which is wasteful.
    # - the GEMM kernel itself is slow for small K (due to low-rank).
    # right now this kernel is good up to K=16
    assert x.stride(1) == 1 and w.stride(1) == 1 and add.stride(1) == 1
    M, K = x.shape
    N, _ = w.shape

    out = x.new_empty(M, N)
    BLOCK_M, BLOCK_N = 128, 128
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M), 1)
    _small_k_mm_kernel[grid](
        x, w, add, out, x.stride(0), w.stride(0), add.stride(0), out.stride(0), M, N, K, BLOCK_M, BLOCK_N
    )
    return out
