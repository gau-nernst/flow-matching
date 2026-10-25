# https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/blob/f972bd4/transformer/pipeline_moss_image.py

from typing import NamedTuple

import torch
import triton
import triton.language as tl
from torch import Tensor, nn

from ..attn import dispatch_attn
from ..flux1.model import timestep_embedding
from ..flux1.modulate import modulate
from ..linear import Linear
from ..registry import MODELS
from ..rope import apply_rope, compute_rope
from ..utils import create_remap_hook, load_hf_state_dict
from ..varlen import Joint, Varlen, merge, seq_ids, split

HEAD_DIM = 128


@triton.jit
def _situ_kernel(
    hidden_ptr,  # [M, D]
    gate_ptr,  # [M, D]
    o_ptr,  # [M, D]
    stride_m,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < D
    hidden = tl.load(hidden_ptr + pid_m * stride_m + offs, mask).to(tl.float32)
    gate = tl.load(gate_ptr + pid_m * stride_m + offs, mask).to(tl.float32)

    out = 4.0 * tl.extra.libdevice.tanh(gate / 4.0) * tl.sigmoid(gate)
    out *= 25.0 * tl.extra.libdevice.tanh(hidden / 25.0)
    tl.store(o_ptr + pid_m * D + offs, out, mask)


# soft-capped SwiGLU: (4 tanh(gate/4) sigmoid(gate)) * (25 tanh(hidden/25))
def situ(hidden: Tensor, gate: Tensor) -> Tensor:
    if torch.is_grad_enabled():
        return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))

    D = hidden.shape[-1]
    hidden_, gate_ = hidden.flatten(0, -2), gate.flatten(0, -2)  # strided views for chunk() outputs
    assert hidden_.stride() == gate_.stride() and hidden_.stride(1) == 1
    out = hidden.new_empty(hidden.shape)
    BLOCK_D = 1024
    _situ_kernel[(hidden_.shape[0], triton.cdiv(D, BLOCK_D))](hidden_, gate_, out, hidden_.stride(0), D, BLOCK_D)
    return out


# text refiner block: pre-norm, no modulation
class TextBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.q_norm = nn.RMSNorm(HEAD_DIM, eps=1e-6)
        self.k_norm = nn.RMSNorm(HEAD_DIM, eps=1e-6)
        self.qkv = Linear(dim, 3 * dim)
        self.proj = Linear(dim, dim)
        self.ffn_up = Linear(dim, ffn_dim)
        self.ffn_gate = Linear(dim, ffn_dim)
        self.ffn_down = Linear(ffn_dim, dim)

    def forward(self, x: Tensor, rope: Tensor, varlen: Varlen | None) -> Tensor:
        # attention
        q, k, v = self.qkv(self.norm1(x)).unflatten(-1, (3, -1, HEAD_DIM)).unbind(2)
        q = apply_rope(q, rope, self.q_norm.weight, interleaved=False)
        k = apply_rope(k, rope, self.k_norm.weight, interleaved=False)
        x = self.proj(dispatch_attn(q, k, v, varlen=varlen).flatten(-2), x)

        # FFN
        y = self.norm2(x)
        return self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)), x)


class Block(nn.Module):
    def __init__(self, dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.q_norm = nn.RMSNorm(HEAD_DIM, eps=1e-6)
        self.k_norm = nn.RMSNorm(HEAD_DIM, eps=1e-6)
        self.q = Linear(dim, dim, bias=False)
        self.k = Linear(dim, dim, bias=False)
        self.v = Linear(dim, dim, bias=False)
        self.o = Linear(dim, dim, bias=False)
        self.ffn_up_gate = Linear(dim, 2 * ffn_dim, bias=False)
        self.ffn_down = Linear(ffn_dim, dim, bias=False)

        remap_pairs = [(f"proj.{name}.weight", f"{name}.weight") for name in ("q", "k", "v", "o")]
        remap_pairs += [("proj.ffn_up_gate.weight", "ffn_up_gate.weight"), ("proj.ffn_down.weight", "ffn_down.weight")]
        self.register_load_state_dict_pre_hook(create_remap_hook(remap_pairs))

    # FFN residual is deferred to the next block's modulate(). modulation is shared, so gate_f is the same.
    def forward(
        self, x: Tensor, ffn_out: Tensor | None, rope: Tensor, mod: Tensor, varlen: Varlen | None
    ) -> tuple[Tensor, Tensor]:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod.split(1, dim=1)  # [N, 1, D] each
        idx = seq_ids(varlen)

        # attention
        res_gate = gate_f if ffn_out is not None else None
        y, x = modulate(x, shift_a, scale_a, ffn_out, res_gate, 1e-6, self.norm1.weight, True, idx)
        q = self.q(y).unflatten(-1, (-1, HEAD_DIM))
        k = self.k(y).unflatten(-1, (-1, HEAD_DIM))
        v = self.v(y).unflatten(-1, (-1, HEAD_DIM))
        q = apply_rope(q, rope, self.q_norm.weight, interleaved=False)
        k = apply_rope(k, rope, self.k_norm.weight, interleaved=False)
        attn_out = self.o(dispatch_attn(q, k, v, varlen=varlen).flatten(-2))

        # FFN
        y, x = modulate(x, shift_f, scale_f, attn_out, gate_a, 1e-6, self.norm2.weight, True, idx)
        hidden, gate = self.ffn_up_gate(y).chunk(2, dim=-1)
        return x, self.ffn_down(situ(hidden, gate))


class MOSSImageConfig(NamedTuple):
    dim: int = 3072
    ffn_dim: int = 8192
    num_layers: int = 42
    text_dim: int = 4096
    latent_channels: int = 128


class MOSSImage(nn.Module):
    def __init__(self, cfg: MOSSImageConfig = MOSSImageConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        dim = cfg.dim
        self.video_embedder = Linear(cfg.latent_channels, dim)
        self.text_embedder = nn.Module()
        self.text_embedder.proj = Linear(cfg.text_dim, dim)
        self.text_embedder.blocks = nn.ModuleList([TextBlock(dim, cfg.ffn_dim) for _ in range(2)])
        self.time_embedding = nn.Sequential(Linear(256, dim), nn.SiLU(), Linear(dim, dim))
        self.shared_block_modulation = Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([Block(dim, cfg.ffn_dim) for _ in range(cfg.num_layers)])
        self.final_norm_video = nn.RMSNorm(dim, eps=1e-6)
        self.final_linear_video = Linear(dim, cfg.latent_channels, bias=False)
        self.video_head_modulation = Linear(dim, 2 * dim)

    # blocks to add LoRA to
    def lora_blocks(self) -> list[nn.Module]:
        return list(self.layers)

    def make_rope(self, H: int, W: int, L_txt: int, device: torch.types.Device = None) -> Tensor:
        # 4D RoPE (r, f, h, w). image tokens (0, L_txt, h - (H-1)/2, w - (W-1)/2), then text tokens (0, l, 0, 0)
        h, w = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
        img_ids = torch.stack([torch.zeros_like(h), torch.full_like(h, L_txt), h, w], dim=-1).flatten(0, 1).double()
        img_ids[:, 2] -= (H - 1) / 2
        img_ids[:, 3] -= (W - 1) / 2
        txt_ids = torch.zeros(L_txt, 4, device=device, dtype=torch.float64)
        txt_ids[:, 1] = torch.arange(L_txt, device=device)

        ids = torch.cat([img_ids, txt_ids])
        return torch.cat([compute_rope(ids[:, i], HEAD_DIM // 4, 10_000.0) for i in range(4)], dim=-1)

    def forward(
        self,
        img: Tensor,
        time: Tensor,
        txt: Tensor,
        rope: Tensor,
        varlen: Joint | None = None,
        skip_layers: tuple[int, ...] = (),
    ) -> Tensor:
        # img: [B, L, C], or [T, C] packed with varlen. txt: unpadded. rope: [img, txt]
        # skip_layers: blocks to skip, for the negative branch of CFG
        packed = varlen is not None
        if packed:
            img, txt = img[None], txt[None]
            N = varlen.varlen.num_seqs
            txt_varlen, joint_varlen = varlen.b, varlen.varlen
        else:
            N = -1  # keep t's shape
            txt_varlen = joint_varlen = None
        dtype = self.video_embedder.weight.dtype

        vec = self.time_embedding(timestep_embedding(time.expand(N), 256).to(dtype))
        mod = self.shared_block_modulation(vec).unflatten(1, (6, -1))  # [N, 6, D]
        shift, scale = self.video_head_modulation(vec).unflatten(1, (2, -1)).unbind(1)  # [N, D] each

        L_img = img.shape[1]
        txt_rope = split(rope, L_img, varlen, dim=0)[1]
        x_img = self.video_embedder(img.to(dtype))
        x_txt = self.text_embedder.proj(txt.to(dtype))
        for block in self.text_embedder.blocks:
            x_txt = block(x_txt, txt_rope, txt_varlen)

        x = merge(x_img, x_txt, varlen)
        ffn_out = None
        for i, block in enumerate(self.layers):
            if i not in skip_layers:
                x, ffn_out = block(x, ffn_out, rope, mod, joint_varlen)

        gate_f = mod[:, 5:6]
        norm = self.final_norm_video.weight
        y, _ = modulate(x, shift[:, None], scale[:, None], ffn_out, gate_f, 1e-6, norm, True, seq_ids(joint_varlen))
        out = self.final_linear_video(split(y, L_img, varlen)[0])
        return out[0] if packed else out


def load_moss_image(name: str = "moss-image-sft"):
    spec = MODELS[name]
    state_dict = load_hf_state_dict(spec.repo_id, spec.filename)
    state_dict.pop("rope.bands")  # recomputed in make_rope()

    with torch.device("meta"):
        model = MOSSImage()

    model.load_state_dict(state_dict, assign=True)
    return model
