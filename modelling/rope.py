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
    D: tl.constexpr,
    GEMMA_NORM: tl.constexpr = False,
    eps: float = 1e-6,
):
    pid_l = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    BLOCK_DIM: tl.constexpr = triton.next_power_of_2(D)
    offs = tl.arange(0, BLOCK_DIM)
    mask = offs < D
    x_ptrs = x_ptr + (pid_b * stride_xb + pid_l * stride_xl + pid_h * D + offs)
    x = tl.load(x_ptrs, mask, other=0.0).to(tl.float32)

    if norm_ptr is not None:
        norm = tl.load(norm_ptr + offs, mask)
        rrms = tl.rsqrt(tl.sum(x * x) * (1.0 / D) + eps)
        if GEMMA_NORM:
            scale = rrms + rrms * norm.to(tl.float32)
        else:
            scale = rrms * norm.to(tl.float32)
        x = (x * scale).to(tl.bfloat16).to(tl.float32)

    x_lo, x_hi = x.reshape(BLOCK_DIM // 2, 2).split()

    rope = tl.load(rope_ptr + (pid_l * D + offs), mask)
    rope_lo, rope_hi = rope.reshape(BLOCK_DIM // 2, 2).split()

    r_lo = x_lo * rope_lo - x_hi * rope_hi
    r_hi = x_lo * rope_hi + x_hi * rope_lo
    r = tl.join(r_lo, r_hi).reshape(BLOCK_DIM)

    o_ptrs = o_ptr + (pid_b * stride_ob + pid_l * stride_ol + pid_h * D + offs)
    tl.store(o_ptrs, r, mask)


def apply_rope(
    x: Tensor,
    rope: Tensor,
    norm: Tensor | None = None,
    gemma_norm: bool = False,
    eps: float = 1e-6,
    *,
    out: Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> Tensor:
    # x: [B, L, nH, D] in real
    # rope: [L, D/2] in complex
    if torch.is_grad_enabled():
        if norm is not None:
            if gemma_norm:
                x = F.rms_norm(x.float(), x.shape[-1:], norm.float() + 1.0, eps).to(x.dtype)
            else:
                x = F.rms_norm(x, x.shape[-1:], norm, eps)
        dtype = rope.dtype.to_real()
        x_ = torch.view_as_complex(x.to(dtype).unflatten(-1, (-1, 2)))  # [B, L, nH, D/2]
        x_ = torch.view_as_real(x_ * rope.unsqueeze(-2)).flatten(-2)  # [B, L, nH, D]
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
    _rope_kernel[(L, H, B)](x, rope_real, norm, out, *x.stride()[:2], *out.stride()[:2], D, gemma_norm, eps)
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
