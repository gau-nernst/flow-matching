import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from gn_kernels import quantize_nvfp4_triton
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

    def forward(self, x: Tensor, add: Tensor | None = None) -> Tensor:
        *dims, in_dim = x.shape
        x = x.view(-1, in_dim)
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
            x, xs = fp8_1d_quantize(x)
            out = sm120_mm_fp8_1d2d.mm(x, xs, self.weight, self.weight_scale_inv, self.bias, add)

        else:
            # bf16
            if self.bias is None and add is not None:
                out = torch.addmm(add, x, self.weight.T)
            else:
                out = F.linear(x, self.weight, self.bias)
                if add is not None:
                    out = out + add

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
def _fp8_1d2d_kernel(
    x_ptr,  # [M, K]
    xs_ptr,  # [M, K/128]
    w_ptr,  # [N, K]
    ws_ptr,  # [N/128, K/128]
    b_ptr,  # [N]
    add_ptr,  # [M, N]
    o_ptr,
    M,
    K: tl.constexpr,
    stride_xm,
    stride_xsm,
    stride_xsk,
    stride_wn,
    stride_wsn,
    stride_wsk,
    stride_am,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_DOT_SCALED: tl.constexpr,
):
    tl.static_assert(128 % BLOCK_N == 0)
    tl.static_assert(128 % BLOCK_K == 0)
    tl.static_assert(K % BLOCK_K == 0)
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, tl.num_programs(1), tl.num_programs(0), 8)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M

    x_ptrs = x_ptr + offs_m * stride_xm + offs_k
    w_ptrs = w_ptr + offs_n * stride_wn + offs_k[:, None]
    xs_ptrs = xs_ptr + offs_m * stride_xsm
    ws_ptrs = ws_ptr + (pid_n * BLOCK_N // 128) * stride_wsn

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    if USE_DOT_SCALED:
        sfx = tl.full((BLOCK_M, BLOCK_K // 32), 127, tl.uint8)
        sfw = tl.full((BLOCK_N, BLOCK_K // 32), 127, tl.uint8)

    for k in range(K // BLOCK_K):
        x = tl.load(x_ptrs, mask_m)  # [BLOCK_M, BLOCK_K]
        w = tl.load(w_ptrs)  # [BLOCK_K, BLOCK_N]
        xs = tl.load(xs_ptrs + (k * BLOCK_K // 128) * stride_xsk, mask_m)  # [BLOCK_M, 1]
        ws = tl.load(ws_ptrs + (k * BLOCK_K // 128) * stride_wsk)  # [BLOCK_N]

        if USE_DOT_SCALED:
            tmp = tl.dot_scaled(x, sfx, "e4m3", w, sfw, "e4m3")
        else:
            tmp = tl.dot(x, w)
        acc += tmp * (xs.to(tl.float32) * ws.to(tl.float32))

        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M

    if add_ptr is not None:
        acc += tl.load(add_ptr + offs_m * stride_am + offs_n, mask_m).to(tl.float32)

    if b_ptr is not None:
        acc += tl.load(b_ptr + offs_n).to(tl.float32)

    o_ptrs = o_ptr + offs_m * stride_om + offs_n
    tl.store(o_ptrs, acc, mask_m)


def fp8_1d2d_mm(x: Tensor, xs: Tensor, w: Tensor, ws: Tensor, b: Tensor | None = None, add: Tensor | None = None):
    M, K = x.shape
    N, _ = w.shape

    out = x.new_empty(M, N, dtype=torch.bfloat16)
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 128
    USE_DOT_SCALED = "GeForce RTX 50" in torch.cuda.get_device_name()

    grid = (N // BLOCK_N, triton.cdiv(M, BLOCK_M), 1)
    _fp8_1d2d_kernel[grid](
        x,
        xs,
        w,
        ws,
        b,
        add,
        out,
        M,
        K,
        x.stride(0),
        *xs.stride(),
        w.stride(0),
        *ws.stride(),
        add.stride(0) if add is not None else 0,
        out.stride(0),
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        USE_DOT_SCALED,
        num_warps=8,
    )
    return out


@triton.jit(do_not_specialize=["M"])
def _fp8_1d_quantize_kernel(
    x_ptr,
    o_ptr,
    os_ptr,
    M,
    stride_xm,
    stride_osm,
    stride_osn,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # x and o: [M, N]
    # os: [M, N/BLOCK_N]
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x = tl.load(x_ptr + offs_m * stride_xm + offs_n, mask=offs_m < M)  # [BLOCK_M, BLOCK_N]
    amax = tl.max(tl.abs(x), axis=1, keep_dims=True)  # [BLOCK_M, 1]

    scale = 448.0 * (1.0 / amax.to(tl.float32))
    inv_scale = 1.0 / scale
    x_fp8 = x.to(tl.float32) * scale

    tl.store(o_ptr + offs_m * stride_om + offs_n, x_fp8, mask=offs_m < M)
    tl.store(os_ptr + offs_m * stride_osm + pid_n * stride_osn, inv_scale, mask=offs_m < M)


def fp8_1d_quantize(x: Tensor):
    M, N = x.shape
    BLOCK_M = 1 if M < 8 else 4
    BLOCK_N = 128
    assert N % BLOCK_N == 0

    out = x.new_empty(M, N, dtype=torch.float8_e4m3fn)
    pad_M = triton.cdiv(M, 4) * 4  # 16B alignment. M-major
    scale = x.new_empty(N // BLOCK_N, pad_M, dtype=torch.float32).T[:M]

    grid = (N // BLOCK_N, triton.cdiv(M, BLOCK_M), 1)
    _fp8_1d_quantize_kernel[grid](x, out, scale, M, x.stride(0), *scale.stride(), out.stride(0), BLOCK_M, BLOCK_N)
    return out, scale


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
