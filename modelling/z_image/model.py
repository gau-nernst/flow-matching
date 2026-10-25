# https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/models/transformers/transformer_z_image.py

import dataclasses

import torch
import torch.nn.functional as F
from gn_kernels.cutedsl.sm120 import sm120_gated_gemm_nvfp4
from gn_kernels.quant_utils import quantize_nvfp4_triton
from torch import Tensor, nn

from ..attn import dispatch_attn
from ..flux1.model import timestep_embedding
from ..linear import Linear, nvfp4_mm
from ..registry import MODELS
from ..rope import RopeND, apply_rope
from ..utils import create_weight_merge_hook, load_hf_state_dict
from .gated_norm import gated_norm


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels):
        super().__init__()
        self.linear = Linear(hidden_size, out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), Linear(256, hidden_size))

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        scale = self.adaLN_modulation(c).unsqueeze(1)
        x = (1.0 + scale) * F.layer_norm(x, x.shape[-1:], eps=1e-6)
        return self.linear(x)


class Attention(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.head_dim = 128
        self.qkv = Linear(dim, dim * 3, bias=False)
        self.norm_q = nn.RMSNorm(self.head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=eps)
        self.to_out = nn.Sequential(Linear(dim, dim, bias=False))
        self.eps = eps
        self.attn_impl = "pt"

        self.register_load_state_dict_pre_hook(create_weight_merge_hook(["to_q", "to_k", "to_v"], "qkv"))

    def forward(self, x: Tensor, pe: Tensor):
        q, k, v = self.qkv(x).unflatten(2, (-1, self.head_dim)).chunk(3, dim=2)
        qk_dtype = torch.float8_e4m3fn if self.attn_impl == "qk-fp8" else x.dtype
        q = apply_rope(q, pe, self.norm_q.weight, self.eps, out=torch.empty_like(q, dtype=qk_dtype))
        k = apply_rope(k, pe, self.norm_k.weight, self.eps, out=torch.empty_like(k, dtype=qk_dtype))
        out = dispatch_attn(q, k, v, self.attn_impl).flatten(2)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = Linear(dim, hidden_dim, bias=False)
        self.w3 = Linear(dim, hidden_dim, bias=False)
        self.w2 = Linear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        if all(w.is_nvfp4() for w in (self.w1, self.w3, self.w2)):
            # assume w1.input_scale == w3.input_scale
            B, L, C = x.shape
            xq, xs = quantize_nvfp4_triton(x.view(B * L, C), self.w1.input_scale)
            xq, xs = sm120_gated_gemm_nvfp4.mm(
                xq,
                xs,
                self.w1.input_scale,
                self.w1.weight.view(torch.float4_e2m1fn_x2),
                self.w1.weight_scale,
                self.w1.weight_scale_2,
                self.w3.weight.view(torch.float4_e2m1fn_x2),
                self.w3.weight_scale,
                self.w3.weight_scale_2,
                self.w2.input_scale,
            )
            out = nvfp4_mm(xq, xs, self.w2.input_scale, self.w2.weight, self.w2.weight_scale, self.w2.weight_scale_2)
            out = out.unflatten(0, (B, L))

        else:
            out = self.w2(F.silu(self.w1(x)) * self.w3(x))

        return out


class Block(nn.Module):
    def __init__(self, dim: int, mod_dim: int, mlp_ratio: float, eps: float = 1e-5) -> None:
        super().__init__()
        self.adaLN_modulation = nn.Sequential(Linear(mod_dim, 4 * dim)) if mod_dim > 0 else None
        self.attention = Attention(dim)
        self.feed_forward = FeedForward(dim, int(dim * mlp_ratio))
        self.attention_norm1 = nn.RMSNorm(dim, eps=eps)
        self.attention_norm2 = nn.RMSNorm(dim, eps=eps)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=eps)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=eps)
        self.eps = eps

    def forward(self, x: Tensor, adaln_input: Tensor | None, pe: Tensor) -> Tensor:
        if self.adaLN_modulation is not None:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)

            attn = self.attention(gated_norm(x, self.attention_norm1.weight, gate=scale_msa, eps=self.eps), pe)
            x = gated_norm(attn, self.attention_norm2.weight, gate=gate_msa, add=x, gate_type="tanh", eps=self.eps)

            ffn = self.feed_forward(gated_norm(x, self.ffn_norm1.weight, gate=scale_mlp, eps=self.eps))
            x = gated_norm(ffn, self.ffn_norm2.weight, gate=gate_mlp, add=x, gate_type="tanh", eps=self.eps)

        else:
            attn = self.attention(self.attention_norm1(x), pe)
            x = gated_norm(attn, self.attention_norm2.weight, add=x, eps=self.eps)

            ffn = self.feed_forward(self.ffn_norm1(x))
            x = gated_norm(ffn, self.ffn_norm2.weight, add=x, eps=self.eps)

        return x


@dataclasses.dataclass
class ZImageConfig:
    img_dim: int = 16
    txt_dim: int = 2560
    mod_dim: int = 256
    dim: int = 3840
    n_refiner_layers: int = 2
    n_layers: int = 30
    patch_size: int = 2
    mlp_ratio: float = 8 / 3
    rope_dims: tuple[int, int, int] = (32, 48, 48)


class ZImage(nn.Module):
    def __init__(self, cfg: ZImageConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or ZImageConfig()
        self.cfg = cfg

        self.t_embedder = nn.Sequential()
        self.t_embedder.mlp = nn.Sequential(Linear(256, 1024), nn.SiLU(), Linear(1024, cfg.mod_dim))
        self.pos_embed = RopeND(cfg.rope_dims, (1536, 512, 512), theta=256.0)
        self.x_pad_token = nn.Parameter(torch.zeros(1, cfg.dim))
        self.cap_pad_token = nn.Parameter(torch.zeros(1, cfg.dim))

        # image-only processing
        patchified_dim = cfg.patch_size * cfg.patch_size * cfg.img_dim
        self.all_x_embedder = nn.ModuleDict()
        self.all_x_embedder["2-1"] = Linear(patchified_dim, cfg.dim)
        self.noise_refiner = nn.ModuleList(
            [Block(cfg.dim, cfg.mod_dim, cfg.mlp_ratio) for _ in range(cfg.n_refiner_layers)]
        )

        # text-only processing
        self.cap_embedder = nn.Sequential(nn.RMSNorm(cfg.txt_dim, eps=1e-5), Linear(cfg.txt_dim, cfg.dim))
        self.context_refiner = nn.ModuleList([Block(cfg.dim, 0, cfg.mlp_ratio) for _ in range(cfg.n_refiner_layers)])

        # joint processing
        self.layers = nn.ModuleList([Block(cfg.dim, cfg.mod_dim, cfg.mlp_ratio) for _ in range(cfg.n_layers)])
        self.all_final_layer = nn.ModuleDict()
        self.all_final_layer["2-1"] = FinalLayer(cfg.dim, patchified_dim)

    @staticmethod
    def _pad_tokens(x: Tensor, pad_token: Tensor):
        """Pad to a multiple of 32"""
        pad_len = (-x.shape[1]) % 32
        pad_tokens = pad_token.view(1, 1, -1).expand(x.shape[0], pad_len, -1)
        return torch.cat([x, pad_tokens], dim=1)

    def forward(self, img: Tensor, timesteps: Tensor, txt: Tensor) -> Tensor:
        # inputs/outputs follow the repo convention: t=1 is noise, predict velocity (noise - data).
        # the original Z-Image uses t=0 as noise and predicts (data - noise), so flip both t and v.
        B, H, W, C = img.shape
        t_embeds = timestep_embedding(1 - timesteps, 256)
        t_embeds = self.t_embedder(t_embeds.to(self.t_embedder[0][0].weight.dtype))

        # patchify
        patch_size = self.cfg.patch_size
        nH = H // patch_size
        nW = W // patch_size
        img = img.view(B, nH, patch_size, nW, patch_size, C)
        img = img.transpose(2, 3)  # (B, nH, nW, 2, 2, C)
        img = img.reshape(B, nH * nW, patch_size * patch_size * C)

        # RoPE embedding has 3 components:
        # - time: text embeds stay at pos=[1,L+1), img embeds stay at pos=L+1
        # - height: all text embeds stay at pos=0
        # - width: all text embeds stay at pos=0

        # text-only processing
        txt = self.cap_embedder(txt)
        txt = self._pad_tokens(txt, self.cap_pad_token)
        txt_rope = self.pos_embed.create((1, 0, 0), (txt.shape[1], 1, 1))
        for layer in self.context_refiner:
            txt = layer(txt, None, txt_rope)

        # image-only processing
        img = img.to(self.all_x_embedder["2-1"].weight.dtype)
        img = self.all_x_embedder["2-1"](img)
        img = self._pad_tokens(img, self.x_pad_token)
        img_rope = self.pos_embed.create((txt.shape[1] + 1, 0, 0), (1, nH, nW))
        for layer in self.noise_refiner:
            img = layer(img, t_embeds, img_rope)

        # joint processing
        unified = torch.cat([img, txt], dim=1)
        unified_rope = torch.cat([img_rope, txt_rope], dim=0)
        for layer in self.layers:
            unified = layer(unified, t_embeds, unified_rope)
        unified = unified[:, : nH * nW]

        unified = self.all_final_layer["2-1"](unified, t_embeds)

        # unpatchify
        img = unified.view(B, nH, nW, patch_size, patch_size, C)
        img = img.transpose(2, 3)  # (B, nH, 2, nW, 2, C)
        img = img.reshape(B, H, W, C)
        return -img


def load_zimage(name: str = "z-image-turbo"):
    spec = MODELS[name]
    repo_id = spec.repo_id
    state_dict = load_hf_state_dict(repo_id, spec.filename)

    # remap comfy ckpt
    if repo_id.startswith("Comfy-Org/"):
        state_dict = {
            k.replace("q_norm", "norm_q")
            .replace("k_norm", "norm_k")
            .replace("out", "to_out.0")
            .replace("x_embedder", "all_x_embedder.2-1")
            .replace("final_layer", "all_final_layer.2-1"): v
            for k, v in state_dict.items()
        }

    with torch.device("meta"):
        model = ZImage()

    model.load_state_dict(state_dict, assign=True)
    return model
