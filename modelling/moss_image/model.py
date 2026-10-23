# https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/blob/f972bd4/transformer/pipeline_moss_image.py

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..attn import dispatch_attn
from ..flux1.modulate import modulate
from ..linear import Linear
from ..rope import apply_rope
from ..utils import create_remap_hook, load_hf_state_dict
from .situ import situ


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # official: normalize in FP32, cast to input dtype, then multiply weight
        return F.rms_norm(x.float(), x.shape[-1:], eps=self.eps).to(x.dtype) * self.weight


# official RoPE rotates pairs (i, i + D/2), while our apply_rope() rotates pairs (2i, 2i + 1).
# permute q/k rows within each head so that they are equivalent.
def _rope_perm(head_dim: int) -> Tensor:
    return torch.arange(head_dim, device="cpu").view(2, head_dim // 2).T.flatten()


def _permute_heads(x: Tensor, perm: Tensor) -> Tensor:
    return x.unflatten(0, (-1, perm.shape[0]))[:, perm].flatten(0, 1)


# text refiner block: pre-norm, no modulation
class TextBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, num_heads: int) -> None:
        super().__init__()
        self.head_dim = dim // num_heads
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.q_norm = nn.Parameter(torch.empty(self.head_dim))
        self.k_norm = nn.Parameter(torch.empty(self.head_dim))
        self.qkv = Linear(dim, 3 * dim)
        self.proj = Linear(dim, dim)
        self.ffn_up_gate = Linear(dim, 2 * ffn_dim)
        self.ffn_down = Linear(ffn_dim, dim)

        perm = _rope_perm(self.head_dim)

        def hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            for suffix in ("weight", "bias"):
                if f"{prefix}qkv.{suffix}" in state_dict:
                    q, k, v = state_dict[f"{prefix}qkv.{suffix}"].chunk(3)
                    state_dict[f"{prefix}qkv.{suffix}"] = torch.cat(
                        [_permute_heads(q, perm), _permute_heads(k, perm), v]
                    )
                if f"{prefix}ffn_gate.{suffix}" in state_dict:
                    gate = state_dict.pop(f"{prefix}ffn_gate.{suffix}")
                    up = state_dict.pop(f"{prefix}ffn_up.{suffix}")
                    state_dict[f"{prefix}ffn_up_gate.{suffix}"] = torch.cat([up, gate])
            for name in ("q_norm", "k_norm"):
                if f"{prefix}{name}.weight" in state_dict:
                    state_dict[f"{prefix}{name}"] = state_dict.pop(f"{prefix}{name}.weight")[perm]

        self.register_load_state_dict_pre_hook(hook)

    def forward(self, x: Tensor, rope: Tensor) -> Tensor:
        q, k, v = self.qkv(self.norm1(x)).unflatten(-1, (3, -1, self.head_dim)).unbind(2)
        q = apply_rope(q, rope, self.q_norm)
        k = apply_rope(k, rope, self.k_norm)
        x = self.proj(dispatch_attn(q, k, v).flatten(-2), x)
        return self.ffn_down(situ(self.ffn_up_gate(self.norm2(x))), x)


class Block(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, num_heads: int) -> None:
        super().__init__()
        self.head_dim = dim // num_heads
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
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

        perm = _rope_perm(self.head_dim)

        def hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            for name in ("q.weight", "k.weight"):
                if f"{prefix}{name}" in state_dict:
                    state_dict[f"{prefix}{name}"] = _permute_heads(state_dict[f"{prefix}{name}"], perm)
            for name in ("q_norm", "k_norm"):
                if f"{prefix}{name}" in state_dict:
                    state_dict[f"{prefix}{name}"] = state_dict[f"{prefix}{name}"][perm]

        # must run after remap
        self.register_load_state_dict_pre_hook(hook)

    # the gated residual of the FFN is deferred to the next block (or the final layer), so that it can be fused
    # with the next norm. modulation is shared across blocks, so gate_f is the same.
    # official residual stream is FP32 (FP32 modulation promotes it). using BF16 barely changes the output.
    def forward(self, x: Tensor, ffn_out: Tensor | None, rope: Tensor, mod: Tensor) -> tuple[Tensor, Tensor]:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod.split(1, dim=1)  # [B, 1, D] each, FP32

        y, x = modulate(
            x, shift_a, scale_a, ffn_out, gate_f if ffn_out is not None else None, 1e-6, self.norm1.weight, True
        )
        q = apply_rope(self.q(y).unflatten(-1, (-1, self.head_dim)), rope, self.q_norm)
        k = apply_rope(self.k(y).unflatten(-1, (-1, self.head_dim)), rope, self.k_norm)
        v = self.v(y).unflatten(-1, (-1, self.head_dim))
        attn_out = self.o(dispatch_attn(q, k, v).flatten(-2))

        y, x = modulate(x, shift_f, scale_f, attn_out, gate_a, 1e-6, self.norm2.weight, True)
        return x, self.ffn_down(situ(self.ffn_up_gate(y)))


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
        self.final_norm_video = RMSNorm(dim)
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
