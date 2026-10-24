import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor, nn


@triton.jit(do_not_specialize=["L"])
def _group_norm_kernel1(
    x_ptr,  # [B, L, num_groups, DIM]
    mean_ptr,  # [B, num_groups, num_tiles]
    M2_ptr,
    L,
    stride,
    NUM_GROUPS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    tile_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    offs_l = tile_id * BLOCK_L + tl.arange(0, BLOCK_L)[:, None, None]
    offs_d = tl.arange(0, NUM_GROUPS)[:, None] * DIM + tl.arange(0, DIM)
    mask = offs_l < L
    x_ptrs = x_ptr + (batch_id * L + offs_l) * NUM_GROUPS * DIM + offs_d
    x = tl.load(x_ptrs, mask, other=0.0).to(tl.float32)  # [BLOCK_L, NUM_GROUPS, DIM]

    cnt = tl.minimum(L - tile_id * BLOCK_L, BLOCK_L) * DIM
    mean = tl.sum(tl.sum(x, axis=2), axis=0) / cnt  # [NUM_GROUPS]
    xc = tl.where(mask, x - mean[:, None], 0.0)
    x2 = xc * xc
    M2 = tl.sum(tl.sum(x2, axis=2), axis=0)

    offs = (batch_id * NUM_GROUPS + tl.arange(0, NUM_GROUPS)) * stride + tile_id
    tl.store(mean_ptr + offs, mean)
    tl.store(M2_ptr + offs, M2)


@triton.jit(do_not_specialize=["L"])
def _group_norm_kernel2(
    mean_ptr,  # [B, num_groups, num_tiles]
    M2_ptr,
    L,
    stride,
    NUM_GROUPS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_L: tl.constexpr,
    eps: float = 1e-6,
):
    group_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    offs = tl.arange(0, BLOCK)
    mean_ptrs = mean_ptr + (batch_id * NUM_GROUPS + group_id) * stride + offs
    M2_ptrs = M2_ptr + (batch_id * NUM_GROUPS + group_id) * stride + offs

    cnt = tl.zeros((BLOCK,), tl.int32)
    mean = tl.zeros((BLOCK,), tl.float32)
    M2 = tl.zeros((BLOCK,), tl.float32)

    num_tiles = tl.cdiv(L, BLOCK_L)

    for i in range(tl.cdiv(num_tiles, BLOCK)):
        offs = i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < num_tiles
        other_cnt = tl.where(mask, tl.minimum(L - offs * BLOCK_L, BLOCK_L) * DIM, 0)
        other_mean = tl.load(mean_ptrs, mask, other=0.0)
        other_M2 = tl.load(M2_ptrs, mask, other=0.0)

        delta = other_mean - mean
        cnt += other_cnt
        mean += delta * other_cnt / tl.maximum(cnt, 1)
        M2 += other_M2 + delta * (other_mean - mean) * other_cnt

        mean_ptrs += BLOCK
        M2_ptrs += BLOCK

    final_cnt = L * DIM
    final_mean = tl.sum(mean * cnt) / final_cnt
    delta = mean - final_mean
    final_M2 = tl.sum(M2 + cnt * delta * delta)
    rrms = tl.rsqrt(final_M2 / final_cnt + eps)

    tl.store(mean_ptr + (batch_id * NUM_GROUPS + group_id) * stride, final_mean)
    tl.store(M2_ptr + (batch_id * NUM_GROUPS + group_id) * stride, rrms)


@triton.jit(do_not_specialize=["L"])
def _group_norm_kernel3(
    x_ptr,  # [B, L, num_groups, DIM]
    mean_ptr,  # [B, num_groups, num_tiles]
    M2_ptr,
    w_ptr,  # [num_groups, DIM]
    b_ptr,  # [num_groups, DIM]
    y_ptr,  # [B, L, num_groups, DIM]
    L,
    stride,
    NUM_GROUPS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_L: tl.constexpr,
    act: tl.constexpr,
):
    tile_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    offs_l = tile_id * BLOCK_L + tl.arange(0, BLOCK_L)[:, None, None]
    offs_d = tl.arange(0, NUM_GROUPS)[:, None] * DIM + tl.arange(0, DIM)
    mask = offs_l < L
    x_ptrs = x_ptr + (batch_id * L + offs_l) * NUM_GROUPS * DIM + offs_d
    x = tl.load(x_ptrs, mask, other=0.0)  # [BLOCK_L, NUM_GROUPS, DIM]
    w = tl.load(w_ptr + offs_d)  # [NUM_GROUPS, DIM]
    b = tl.load(b_ptr + offs_d)

    offs_g = tl.arange(0, NUM_GROUPS)[:, None]
    mean = tl.load(mean_ptr + (batch_id * NUM_GROUPS + offs_g) * stride)
    rrms = tl.load(M2_ptr + (batch_id * NUM_GROUPS + offs_g) * stride)

    y = (x.to(tl.float32) - mean) * rrms * w.to(tl.float32) + b.to(tl.float32)
    y = y.to(y_ptr.dtype.element_ty).to(tl.float32)
    if act == "silu":
        y *= tl.sigmoid(y)
    else:
        assert act == "none"
    y_ptrs = y_ptr + (batch_id * L + offs_l) * NUM_GROUPS * DIM + offs_d
    tl.store(y_ptrs, y, mask)


def group_norm(x: Tensor, w: Tensor, b: Tensor, num_groups: int, eps: float = 1e-6, act: str = "none"):
    assert x.is_contiguous() and w.is_contiguous() and b.is_contiguous()
    y = torch.empty_like(x)
    B, H, W, C = x.shape
    assert C % num_groups == 0
    L = H * W
    G = num_groups
    DIM = C // num_groups

    # heuristic
    # BLOCK_L must be small enough to have sufficient active CTAs
    MIN_GRID = 1024
    BLOCK_L = triton.next_power_of_2(int(B * L / MIN_GRID))
    BLOCK_L = max(min(BLOCK_L, 128), 1)
    num_tiles = triton.cdiv(L, BLOCK_L)

    BLOCK = min(512, triton.next_power_of_2(num_tiles))
    num_tiles_pad = triton.cdiv(num_tiles, 16) * 16

    tmp_mean = x.new_empty(B, G, num_tiles_pad, dtype=torch.float32)
    tmp_M2 = torch.empty_like(tmp_mean)

    _group_norm_kernel1[(num_tiles, B)](x, tmp_mean, tmp_M2, L, num_tiles_pad, G, DIM, BLOCK_L)
    _group_norm_kernel2[(G, B)](tmp_mean, tmp_M2, L, num_tiles_pad, G, DIM, BLOCK, BLOCK_L, eps)
    _group_norm_kernel3[(num_tiles, B)](x, tmp_mean, tmp_M2, w, b, y, L, num_tiles_pad, G, DIM, BLOCK_L, act)
    return y


class GroupNorm(nn.GroupNorm):
    def forward(self, x: Tensor, act: str = "none"):
        if not torch.is_grad_enabled():
            return group_norm(x, self.weight, self.bias, self.num_groups, self.eps, act)

        out = super().forward(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        if act == "silu":
            out = F.silu(out)
        else:
            assert act == "none"
        return out


class BatchNorm2d(nn.BatchNorm2d):
    def forward(self, x: Tensor):
        return super().forward(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
