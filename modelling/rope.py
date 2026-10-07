import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor, nn


def compute_rope(
    length: int,
    dim: int,
    theta: float,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.types.Device = None,
) -> Tensor:
    # initial computations in fp64
    omega = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64, device=device) / dim))
    timestep = torch.arange(length, device=device, dtype=torch.float64)
    freqs = (timestep[:, None] * omega).to(dtype)
    return torch.polar(torch.ones_like(freqs), freqs)


@triton.jit
def _rope_kernel(
    x_ptr,  # [B, L, H, D]
    rope_ptr,  # [L, D]
    norm_ptr,  # [D]
    o_ptr,
    stride_xb,
    stride_xl,
    stride_ob,
    stride_ol,
    H,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    eps=1e-6,
):
    pid_l = tl.program_id(0)
    pid_b = tl.program_id(1)

    BLOCK_D: tl.constexpr = triton.next_power_of_2(D)
    offs_h = tl.arange(0, BLOCK_H)[:, None]
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    x_ptrs = x_ptr + (pid_b * stride_xb + pid_l * stride_xl + offs_h * D + offs_d)
    o_ptrs = o_ptr + (pid_b * stride_ob + pid_l * stride_ol + offs_h * D + offs_d)

    if norm_ptr is not None:
        norm = tl.load(norm_ptr + offs_d, mask_d).to(tl.float32)
    rope = tl.load(rope_ptr + (pid_l * D + offs_d), mask_d)
    rope0, rope1 = rope.reshape(BLOCK_D // 2, 2).split()

    for h in range(tl.cdiv(H, BLOCK_H)):
        mask_h = offs_h < H - h * BLOCK_H
        x = tl.load(x_ptrs, mask_h & mask_d, other=0.0).to(tl.float32)

        if norm_ptr is not None:
            mean_sq = tl.sum(x * x, axis=1, keep_dims=True) * (1.0 / D)
            x = x * tl.rsqrt(mean_sq + eps) * norm
            # x = x.to(tl.bfloat16).to(tl.float32)

        x0, x1 = x.reshape(BLOCK_H, BLOCK_D // 2, 2).split()
        r0 = x0 * rope0 - x1 * rope1
        r1 = x0 * rope1 + x1 * rope0
        r = tl.join(r0, r1).reshape(BLOCK_H, BLOCK_D)
        tl.store(o_ptrs, r, mask_h & mask_d)

        x_ptrs += BLOCK_H * D
        o_ptrs += BLOCK_H * D


def apply_rope(
    x: Tensor,
    rope: Tensor,
    norm: Tensor | None = None,
    eps: float = 1e-6,
    *,
    out: Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> Tensor:
    # x: [B, L, nH, D] in real
    # rope: [L, D/2] in complex
    if torch.is_grad_enabled():
        if norm is not None:
            x = F.rms_norm(x, x.shape[-1:], norm, eps)
        cos, sin = torch.view_as_real(rope).unsqueeze(-3).unbind(-1)  # [L, 1, D/2] each
        x0, x1 = x.float().unflatten(-1, (-1, 2)).unbind(-1)  # [B, L, nH, D/2] each
        x_ = torch.stack(
            [
                torch.addcmul(x0 * cos, x1, sin, value=-1),
                torch.addcmul(x0 * sin, x1, cos),
            ],
            dim=-1,
        ).flatten(-2)
        if out is not None:
            out.copy_(x_)
        else:
            out = x_.type_as(x)
        return out

    assert x[0, 0].is_contiguous() and rope.is_contiguous()
    if norm is not None:
        assert norm.is_contiguous()
    rope_real = torch.view_as_real(rope)
    if out is not None:
        assert out[0, 0].is_contiguous()
    else:
        out = torch.empty_like(x, dtype=out_dtype)
    B, L, H, D = x.shape
    BLOCK_H = 8  # 8x128 = 1024. so 4 warps issue 16B
    grid = (L, B)
    _rope_kernel[grid](x, rope_real, norm, out, *x.stride()[:2], *out.stride()[:2], H, D, BLOCK_H, eps)
    return out


class RopeND(nn.Module):
    def __init__(
        self,
        dims: tuple[int, ...],
        max_lens: tuple[int, ...],
        theta: float,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        assert len(dims) == len(max_lens)
        self.dims = dims
        self.max_lens = max_lens
        self.theta = theta
        self.dtype = dtype
        self.precompute_rope()

    def precompute_rope(self, device: torch.types.Device = None) -> None:
        # don't create things on meta device to avoid weird cases...
        device = device or torch.get_default_device()
        if torch.device(device) == torch.device("meta"):
            device = "cpu"

        for i, (dim, length) in enumerate(zip(self.dims, self.max_lens)):
            # always compute on CPU, then move to the requested device
            rope = compute_rope(length, dim, self.theta, dtype=self.dtype, device="cpu")
            self.register_buffer(f"rope{i}", rope.to(device), persistent=False)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)

        # recompute rope if dtype is changed
        dtype = self.dtype.to_complex()
        if any(getattr(self, f"rope{i}").dtype != dtype for i in range(len(self.dims))):
            self.precompute_rope(self.rope0.device)

        return self

    def create(self, start_list: tuple[int, ...], length_list: tuple[int, ...]) -> Tensor:
        pos_list = [
            torch.arange(start, start + length, device=self.rope0.device)
            for start, length in zip(start_list, length_list)
        ]
        grids = torch.meshgrid(pos_list, indexing="ij")  # this returns list[Tensor]

        rope_list = [getattr(self, f"rope{i}")[grid.flatten()] for i, grid in enumerate(grids)]
        return torch.cat(rope_list, dim=-1)
