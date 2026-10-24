# https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/blob/f972bd4/transformer/pipeline_moss_image.py

from typing import NamedTuple

import torch
import triton
import triton.language as tl
from torch import Tensor, nn

from ..attn import dispatch_attn
from ..flux1.modulate import modulate
from ..linear import Linear
from ..rope import apply_rope
from ..utils import create_remap_hook, load_hf_state_dict


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
    hidden_, gate_ = hidden.flatten(0, -2), gate.flatten(0, -2)  # views. rows may be strided
    assert hidden_.stride() == gate_.stride() and hidden_.stride(1) == 1
    out = hidden.new_empty(hidden.shape)
    BLOCK_D = 1024
    _situ_kernel[(hidden_.shape[0], triton.cdiv(D, BLOCK_D))](hidden_, gate_, out, hidden_.stride(0), D, BLOCK_D)
    return out


# text refiner block: pre-norm, no modulation
class TextBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, num_heads: int) -> None:
        super().__init__()
        self.head_dim = dim // num_heads
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.q_norm = nn.Parameter(torch.empty(self.head_dim))
        self.k_norm = nn.Parameter(torch.empty(self.head_dim))
        self.qkv = Linear(dim, 3 * dim)
        self.proj = Linear(dim, dim)
        self.ffn_up = Linear(dim, ffn_dim)
        self.ffn_gate = Linear(dim, ffn_dim)
        self.ffn_down = Linear(ffn_dim, dim)
        self.register_load_state_dict_pre_hook(
            create_remap_hook([("q_norm.weight", "q_norm"), ("k_norm.weight", "k_norm")])
        )

    def forward(self, x: Tensor, rope: Tensor) -> Tensor:
        # attention
        q, k, v = self.qkv(self.norm1(x)).unflatten(-1, (3, -1, self.head_dim)).unbind(2)
        q = apply_rope(q, rope, self.q_norm, interleaved=False)
        k = apply_rope(k, rope, self.k_norm, interleaved=False)
        x = self.proj(dispatch_attn(q, k, v).flatten(-2), x)

        # FFN
        y = self.norm2(x)
        return self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)), x)


class Block(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, num_heads: int) -> None:
        super().__init__()
        self.head_dim = dim // num_heads
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.q_norm = nn.Parameter(torch.empty(self.head_dim))
        self.k_norm = nn.Parameter(torch.empty(self.head_dim))
        self.q = Linear(dim, dim, bias=False)
        self.k = Linear(dim, dim, bias=False)
        self.v = Linear(dim, dim, bias=False)
        self.o = Linear(dim, dim, bias=False)
        self.ffn_up_gate = Linear(dim, 2 * ffn_dim, bias=False)
        self.ffn_down = Linear(ffn_dim, dim, bias=False)

        remap_pairs = [(f"proj.{name}.weight", f"{name}.weight") for name in ("q", "k", "v", "o")]
        remap_pairs += [("proj.ffn_up_gate.weight", "ffn_up_gate.weight"), ("proj.ffn_down.weight", "ffn_down.weight")]
        remap_pairs += [("q_norm.weight", "q_norm"), ("k_norm.weight", "k_norm")]
        self.register_load_state_dict_pre_hook(create_remap_hook(remap_pairs))

    # the gated residual of the FFN is deferred to the next block (or the final layer), so that it can be fused
    # with the next norm. modulation is shared across blocks, so gate_f is the same.
    # official residual stream is FP32 (FP32 modulation promotes it). using BF16 barely changes the output.
    def forward(self, x: Tensor, ffn_out: Tensor | None, rope: Tensor, mod: Tensor) -> tuple[Tensor, Tensor]:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod.split(1, dim=1)  # [B, 1, D] each, FP32

        # attention. also adds the previous block's FFN residual
        res_gate = gate_f if ffn_out is not None else None
        y, x = modulate(x, shift_a, scale_a, ffn_out, res_gate, 1e-6, self.norm1.weight, True)
        q = self.q(y).unflatten(-1, (-1, self.head_dim))
        k = self.k(y).unflatten(-1, (-1, self.head_dim))
        v = self.v(y).unflatten(-1, (-1, self.head_dim))
        q = apply_rope(q, rope, self.q_norm, interleaved=False)
        k = apply_rope(k, rope, self.k_norm, interleaved=False)
        attn_out = self.o(dispatch_attn(q, k, v).flatten(-2))

        # FFN. its residual is added in the next block
        y, x = modulate(x, shift_f, scale_f, attn_out, gate_a, 1e-6, self.norm2.weight, True)
        hidden, gate = self.ffn_up_gate(y).chunk(2, dim=-1)
        return x, self.ffn_down(situ(hidden, gate))


class MOSSImageConfig(NamedTuple):
    dim: int = 3072
    ffn_dim: int = 8192
    num_heads: int = 24
    num_layers: int = 42
    text_dim: int = 4096
    freq_dim: int = 256
    latent_channels: int = 128
    rope_base: float = 10_000.0


class MOSSImage(nn.Module):
    def __init__(self, cfg: MOSSImageConfig = MOSSImageConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        dim = cfg.dim
        self.video_embedder = Linear(cfg.latent_channels, dim)
        self.text_embedder = nn.Module()
        self.text_embedder.proj = Linear(cfg.text_dim, dim)
        self.text_embedder.blocks = nn.ModuleList([TextBlock(dim, cfg.ffn_dim, cfg.num_heads) for _ in range(2)])
        self.time_embedding = nn.Sequential(Linear(cfg.freq_dim, dim), nn.SiLU(), Linear(dim, dim))
        self.shared_block_modulation = Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([Block(dim, cfg.ffn_dim, cfg.num_heads) for _ in range(cfg.num_layers)])
        self.final_norm_video = nn.RMSNorm(dim, eps=1e-6)
        self.final_linear_video = Linear(dim, cfg.latent_channels, bias=False)
        self.video_head_modulation = Linear(dim, 2 * dim)

        # official stores RoPE bands as a buffer, but always recomputes them in FP32
        def hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            state_dict.pop(f"{prefix}rope.bands", None)

        self.register_load_state_dict_pre_hook(hook)

    # 4D RoPE (r, f, h, w), each axis uses head_dim/8 bands, with sequence order image then text
    # - image: (0, L_txt, h - (H-1)/2, w - (W-1)/2)
    # - text:  (0, l, 0, 0)
    # returns [H*W + L_txt, head_dim/2] complex
    def make_rope(self, H: int, W: int, L_txt: int, device: torch.types.Device = None) -> Tensor:
        num_bands = self.cfg.dim // self.cfg.num_heads // 8
        bands = 1 / (
            self.cfg.rope_base ** (torch.arange(num_bands, device=device) / num_bands)
        )  # FP32, same as official

        h, w = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
        img_ids = torch.stack([torch.zeros_like(h), torch.full_like(h, L_txt), h, w], dim=-1).flatten(0, 1).float()
        img_ids[:, 2] -= (H - 1) / 2
        img_ids[:, 3] -= (W - 1) / 2
        txt_ids = torch.zeros(L_txt, 4, device=device)
        txt_ids[:, 1] = torch.arange(L_txt, device=device)

        freqs = (torch.cat([img_ids, txt_ids])[..., None] * bands).flatten(-2)
        return torch.polar(torch.ones_like(freqs), freqs)

    def time_embed(self, t: Tensor) -> Tensor:
        # official: sinusoid in FP64, [cos, sin], t in [0, 1000]
        half = self.cfg.freq_dim // 2
        freqs = 10_000 ** (-torch.arange(half, device=t.device, dtype=torch.float64) / half)
        phase = t.double()[:, None] * freqs
        emb = torch.cat([phase.cos(), phase.sin()], dim=-1).float()
        return self.time_embedding(emb.to(self.video_embedder.weight.dtype)).float()

    # img: [B, H, W, C] (Flux.2 VAE latents, patchified + normalized)
    # time: [B] in [0, 1], 1 is noise
    # txt: [B, L, text_dim]. no padding: all prompts in a batch must have the same length.
    # skip_layers: blocks to skip, used for the negative branch of CFG.
    def forward(self, img: Tensor, time: Tensor, txt: Tensor, skip_layers: tuple[int, ...] = ()) -> Tensor:
        _, H, W, _ = img.shape
        L_img = H * W
        dtype = self.video_embedder.weight.dtype

        vec = self.time_embed(time * 1000)
        mod = self.shared_block_modulation(vec.to(dtype)).float().unflatten(1, (6, -1))
        shift, scale = self.video_head_modulation(vec.to(dtype)).float().unflatten(1, (2, -1)).unbind(1)

        rope = self.make_rope(H, W, txt.shape[1], img.device)
        x_img = self.video_embedder(img.flatten(1, 2).to(dtype))
        x_txt = self.text_embedder.proj(txt.to(dtype))
        for block in self.text_embedder.blocks:
            x_txt = block(x_txt, rope[L_img:])

        x = torch.cat([x_img, x_txt], dim=1)
        ffn_out = None
        for i, block in enumerate(self.layers):
            if i not in skip_layers:
                x, ffn_out = block(x, ffn_out, rope, mod)

        gate_f = mod[:, 5:6]
        y, _ = modulate(x, shift[:, None], scale[:, None], ffn_out, gate_f, 1e-6, self.final_norm_video.weight, True)
        return self.final_linear_video(y[:, :L_img]).unflatten(1, (H, W))

    # online block FP8 for the DiT blocks' linear layers
    def to_fp8(self):
        for block in self.layers:
            for m in block.modules():
                if isinstance(m, Linear):
                    m.to_fp8_1d2d()
        return self


def load_moss_image(name: str = "sft"):
    repo_id = {
        "base": "OpenMOSS-Team/MOSS-Image-Base",
        "sft": "OpenMOSS-Team/MOSS-Image-SFT",
    }[name]
    state_dict = load_hf_state_dict(repo_id, "transformer/diffusion_pytorch_model.safetensors.index.json")

    with torch.device("meta"):
        model = MOSSImage()

    model.load_state_dict(state_dict, assign=True)
    return model
