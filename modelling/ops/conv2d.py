import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor, nn


@triton.jit
def _conv2d_kernel(
    x_ptr,  # [N, Hin, Win, Cin]
    w_ptr,  # [Cout, kH, kW, Cin]
    b_ptr,  # [Cou]
    add_ptr,  # [N, Hout, Wout, Cout]
    o_ptr,  # [N, Hout, Wout, Cout]
    stride_xn,
    stride_xh,
    stride_xw,
    stride_an,
    stride_ah,
    stride_aw,
    stride_on,
    stride_oh,
    stride_ow,
    Hin,
    Win,
    Cin: tl.constexpr,
    Cout: tl.constexpr,
    Kh: tl.constexpr,
    Kw: tl.constexpr,
    PADDING: tl.constexpr,
    UPSAMPLE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    Hout = Hin * UPSAMPLE + PADDING * 2 - (Kh - 1)
    Wout = Win * UPSAMPLE + PADDING * 2 - (Kw - 1)

    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    batch_id = tl.program_id(2)

    # coordinates in output fmap
    num_pid_w = tl.cdiv(Wout, BLOCK_M)
    pid_w = pid_m % num_pid_w
    hout = pid_m // num_pid_w

    x_ptr += batch_id * stride_xn
    o_ptr += batch_id * stride_on + hout * stride_oh
    if add_ptr is not None:
        add_ptr += batch_id * stride_an + hout * stride_ah

    offs_cout = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    w_ptrs = w_ptr + offs_cout * Kh * Kw * Cin + offs_k[:, None]
    mask_cout = offs_cout < Cout

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    tl.static_assert(Cin % BLOCK_K == 0)
    num_cin_tiles = Cin // BLOCK_K
    num_iters = num_cin_tiles * Kh * Kw
    for k in range(num_iters):
        cin_tile = k % num_cin_tiles
        hw_tile = k // num_cin_tiles
        w_tile = hw_tile % Kw
        h_tile = hw_tile // Kw

        offs_w = -PADDING + w_tile + pid_w * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        off_h = -PADDING + h_tile + hout
        offs_cin = cin_tile * BLOCK_K + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + (offs_w // UPSAMPLE) * stride_xw + (off_h // UPSAMPLE) * stride_xh + offs_cin
        mask = (0 <= offs_w) & (offs_w < Win * UPSAMPLE) & (0 <= off_h) & (off_h < Hin * UPSAMPLE)

        x = tl.load(x_ptrs, mask, other=0.0)  # [BLOCK_M, BLOCK_K]
        w = tl.load(w_ptrs, mask_cout, other=0.0)  # [BLOCK_K, BLOCK_N]
        acc = tl.dot(x, w, acc)

        w_ptrs += BLOCK_K

    offs_w = pid_w * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    mask = (offs_w < Wout) & mask_cout

    if add_ptr is not None:
        add = tl.load(add_ptr + offs_w * stride_aw + offs_cout, mask)
        acc += add.to(tl.float32)

    if b_ptr is not None:
        acc += tl.load(b_ptr + offs_cout, mask_cout).to(tl.float32)

    o_ptrs = o_ptr + offs_w * stride_ow + offs_cout
    tl.store(o_ptrs, acc, mask)


def conv2d_triton(
    x: Tensor,
    w: Tensor,
    b: Tensor | None = None,
    add: Tensor | None = None,
    padding: int = 0,
    upsample: int = 1,
):
    """When upsample>1, the input is upsampled with nearest-neighbor before performing convolution."""
    assert x.stride(-1) == 1
    assert w.is_contiguous()
    if b is not None:
        assert b.is_contiguous()
    if add is not None:
        assert add.stride(-1) == 1

    N, Hin, Win, Cin = x.shape
    Cout, Kh, Kw, _ = w.shape
    Hout = Hin * upsample + padding * 2 - (Kh - 1)
    Wout = Win * upsample + padding * 2 - (Kw - 1)

    out = x.new_empty(N, Hout, Wout, Cout)
    BLOCK_M = 64 if Hout * Wout <= 256 * 256 else 128
    BLOCK_N = 64 if Cout > 128 and Cout % 128 == 64 else min(max(triton.next_power_of_2(Cout), 16), 128)
    BLOCK_K = 32

    grid = (triton.cdiv(Cout, BLOCK_N), Hout * triton.cdiv(Wout, BLOCK_M), N)
    _conv2d_kernel[grid](
        x,
        w,
        b,
        add,
        out,
        *x.stride()[:-1],
        *(add.stride()[:-1] if add is not None else (0, 0, 0)),
        *out.stride()[:-1],
        Hin,
        Win,
        Cin,
        Cout,
        Kh,
        Kw,
        padding,
        upsample,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )
    return out


class Conv2d(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, kernel_size, kernel_size, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.stride = stride
        self.padding = padding

        def hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            key = f"{prefix}weight"
            state_dict[key] = state_dict[key].permute(0, 2, 3, 1).contiguous()

        self.register_load_state_dict_pre_hook(hook)

    def forward(self, x: Tensor, add: Tensor | None = None):
        Cout, K, _, Cin = self.weight.shape
        if K == 1:
            out = F.linear(x.view(-1, Cin), self.weight.squeeze(), self.bias)
            out = out.view(*x.shape[:-1], Cout)
            if add is not None:
                out = out + add

        elif self.stride == 1 and Cin % 32 == 0:
            out = conv2d_triton(x, self.weight, self.bias, add, self.padding)

        else:
            out = F.conv2d(
                x.permute(0, 3, 1, 2),
                self.weight.permute(0, 3, 1, 2),
                self.bias,
                self.stride,
                self.padding,
            ).permute(0, 2, 3, 1)
            if add is not None:
                out = out + add

        return out
