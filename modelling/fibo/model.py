# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_bria_fibo2.py
# Z-Image blocks, plus a Perceiver that compresses the text encoder's hidden states into "gist" tokens, and gated
# cross-attention to the text encoder in a few blocks.

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..attn import dispatch_attn
from ..flux1.model import timestep_embedding
from ..linear import Linear
from ..rope import RopeND
from ..utils import load_hf_state_dict
from ..z_image.gated_norm import gated_norm
from ..z_image.model import Attention, FeedForward, FinalLayer


# attention without RoPE, with optional different K/V input
class CrossAttention(nn.Module):
    def __init__(self, dim: int, kv_dim: int | None = None, eps: float = 1e-5) -> None:
        super().__init__()
        self.head_dim = 128
        kv_dim = kv_dim or dim
        self.to_q = Linear(dim, dim, bias=False)
        self.to_k = Linear(kv_dim, dim, bias=False)
        self.to_v = Linear(kv_dim, dim, bias=False)
        self.norm_q = nn.RMSNorm(self.head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=eps)
        self.to_out = nn.Sequential(Linear(dim, dim, bias=False))

    def forward(self, x: Tensor, kv: Tensor | None = None) -> Tensor:
        kv = x if kv is None else kv
        q = self.norm_q(self.to_q(x).unflatten(-1, (-1, self.head_dim)))
        k = self.norm_k(self.to_k(kv).unflatten(-1, (-1, self.head_dim)))
        v = self.to_v(kv).unflatten(-1, (-1, self.head_dim))
        return self.to_out(dispatch_attn(q, k, v).flatten(-2))


class Block(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, eps: float = 1e-5, cross_dim: int | None = None) -> None:
        super().__init__()
        self.adaLN_modulation = nn.Sequential(Linear(256, 4 * dim))
        self.attention = Attention(dim, eps)
        self.feed_forward = FeedForward(dim, mlp_dim)
        self.attention_norm1 = nn.RMSNorm(dim, eps=eps)
        self.attention_norm2 = nn.RMSNorm(dim, eps=eps)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=eps)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=eps)
        self.eps = eps

        # gated cross-attention to the text encoder
        if cross_dim is not None:
            self.cross_attn_norm1 = nn.RMSNorm(dim, eps=eps)
            self.cross_attention = CrossAttention(dim, cross_dim, eps)
            self.cross_attn_norm2 = nn.RMSNorm(dim, eps=eps)
            self.cross_attn_gate = nn.Parameter(torch.empty(1))
        else:
            self.cross_attention = None

    # vec: [B, S, 256] timestep embedding of each segment of x. with S=2, tokens [0, split) use vec[:, 0]
    # (gist tokens, t=0), the rest use vec[:, 1] (image tokens).
    def forward(
        self, x: Tensor, vec: Tensor, pe: Tensor, txt: Tensor | None = None, split: int | None = None
    ) -> Tensor:
        scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(vec).chunk(4, dim=-1)  # [B, S, D] each
        eps = self.eps

        attn = self.attention(gated_norm(x, self.attention_norm1.weight, scale_msa, eps=eps, split=split), pe)
        x = gated_norm(attn, self.attention_norm2.weight, gate_msa, x, "tanh", eps, split)

        if self.cross_attention is not None:
            cross = self.cross_attention(self.cross_attn_norm1(x), txt)
            x = gated_norm(cross, self.cross_attn_norm2.weight * self.cross_attn_gate.tanh(), add=x, eps=eps)

        ffn = self.feed_forward(gated_norm(x, self.ffn_norm1.weight, scale_mlp, eps=eps, split=split))
        return gated_norm(ffn, self.ffn_norm2.weight, gate_mlp, x, "tanh", eps, split)


# gist tokens read the text encoder (cross-attention), then each other (self-attention), then an FFN
class PerceiverLayer(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, cross_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.cross_attn_norm = nn.RMSNorm(dim, eps=eps)
        self.cross_attention = CrossAttention(dim, cross_dim, eps)
        self.self_attn_norm = nn.RMSNorm(dim, eps=eps)
        self.self_attention = CrossAttention(dim, eps=eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=eps)
        self.ffn = FeedForward(dim, mlp_dim)

    def forward(self, x: Tensor, txt: Tensor) -> Tensor:
        x = x + self.cross_attention(self.cross_attn_norm(x), txt)
        x = x + self.self_attention(self.self_attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


@dataclasses.dataclass
class FiboConfig:
    img_dim: int = 32
    patch_size: int = 2
    dim: int = 3584
    n_layers: int = 44
    n_refiner_layers: int = 2
    txt_dim: int = 7680  # 3 Qwen3-VL-4B layers concatenated
    injection_layer_ids: tuple[int, ...] = (6, 14, 22, 30, 38)
    perceiver_num_layers: int = 2
    min_num_gist_tokens: int = 256
    max_num_gist_tokens: int = 512
    gist_step: int = 64
    gist_min_text_len: int = 256
    gist_max_text_len: int = 2048
    rope_dims: tuple[int, int, int] = (32, 48, 48)
    rope_lens: tuple[int, int, int] = (2048, 512, 512)


class Fibo(nn.Module):
    def __init__(self, cfg: FiboConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or FiboConfig()
        self.cfg = cfg
        dim = cfg.dim
        mlp_dim = (dim * 8 // 3) // 64 * 64
        patchified_dim = cfg.patch_size * cfg.patch_size * cfg.img_dim

        self.x_embedder = Linear(patchified_dim, dim)
        self.t_embedder = nn.Sequential()
        self.t_embedder.mlp = nn.Sequential(Linear(256, 1024), nn.SiLU(), Linear(1024, 256))
        self.pos_embed = RopeND(cfg.rope_dims, cfg.rope_lens, theta=256.0)

        self.noise_refiner = nn.ModuleList([Block(dim, mlp_dim) for _ in range(cfg.n_refiner_layers)])
        self.perceiver = nn.Module()
        self.perceiver.latent_queries = nn.Parameter(torch.empty(cfg.max_num_gist_tokens, dim))
        self.perceiver.layers = nn.ModuleList(
            [PerceiverLayer(dim, mlp_dim, cfg.txt_dim) for _ in range(cfg.perceiver_num_layers)]
        )
        self.perceiver.norm_out = nn.RMSNorm(dim, eps=1e-5)
        self.layers = nn.ModuleList(
            [
                Block(dim, mlp_dim, cross_dim=cfg.txt_dim if i in cfg.injection_layer_ids else None)
                for i in range(cfg.n_layers)
            ]
        )
        self.final_layer = FinalLayer(dim, patchified_dim)

    # number of gist tokens from text length: linear between min and max, rounded up to a multiple of gist_step
    def num_gist_tokens(self, txt_len: int) -> int:
        cfg = self.cfg
        ratio = min(1.0, max(0.0, (txt_len - cfg.gist_min_text_len) / (cfg.gist_max_text_len - cfg.gist_min_text_len)))
        n = cfg.min_num_gist_tokens + ratio * (cfg.max_num_gist_tokens - cfg.min_num_gist_tokens)
        n = math.ceil(n / cfg.gist_step) * cfg.gist_step
        return max(cfg.min_num_gist_tokens, min(cfg.max_num_gist_tokens, n))

    # img: [B, H, W, C] (Flux.2 VAE latents, patchified + normalized)
    # time: [B] in [0, 1], 1 is noise
    # txt: [B, 6, L, txt_dim]. text encoder bundles: the Perceiver's first, then one per injection block.
    def forward(self, img: Tensor, time: Tensor, txt: Tensor) -> Tensor:
        B, H, W, _ = img.shape
        dtype = self.x_embedder.weight.dtype

        # image tokens get the real timestep, clean tokens (gist) get t=0
        t_mlp = self.t_embedder.mlp
        vec = t_mlp(timestep_embedding(time, 256).to(dtype))
        vec_clean = t_mlp(timestep_embedding(torch.zeros_like(time), 256).to(dtype))

        # RoPE axes (plane, h, w). gist token i is on plane 1+i, the image on plane max_num_gist_tokens+1.
        img_pe = self.pos_embed.create((self.cfg.max_num_gist_tokens + 1, 0, 0), (1, H, W))
        x = self.x_embedder(img.flatten(1, 2).to(dtype))
        for block in self.noise_refiner:
            x = block(x, vec[:, None], img_pe)

        # Perceiver
        txt = txt.to(dtype)
        num_gist = self.num_gist_tokens(txt.shape[2])
        gist = self.perceiver.latent_queries[:num_gist].unsqueeze(0).expand(B, -1, -1)
        for layer in self.perceiver.layers:
            gist = layer(gist, txt[:, 0])
        gist = self.perceiver.norm_out(gist)
        gist_pe = self.pos_embed.create((1, 0, 0), (num_gist, 1, 1))

        # one stream: gist tokens, then image tokens. injection blocks also read their own text bundle.
        x = torch.cat([gist, x], dim=1)
        pe = torch.cat([gist_pe, img_pe], dim=0)
        vecs = torch.stack([vec_clean, vec], dim=1)
        bundle_ids = {layer_id: i + 1 for i, layer_id in enumerate(self.cfg.injection_layer_ids)}
        for i, block in enumerate(self.layers):
            x = block(x, vecs, pe, txt[:, bundle_ids[i]] if i in bundle_ids else None, num_gist)

        x = self.final_layer(x[:, num_gist:], vec)
        return x.unflatten(1, (H, W))

    # online block FP8 (1d2d) for the linear layers of the image-stream blocks.
    # FFN hidden dim (9536) is not a multiple of 128: zero-pad it, which doesn't change the output.
    def to_fp8(self):
        for block in [*self.noise_refiner, *self.layers]:
            ffn = block.feed_forward
            pad = -ffn.w1.weight.shape[0] % 128
            for w in (ffn.w1, ffn.w3):
                w.weight = nn.Parameter(F.pad(w.weight, (0, 0, 0, pad)), requires_grad=False)
            ffn.w2.weight = nn.Parameter(F.pad(ffn.w2.weight, (0, pad)), requires_grad=False)

            for m in (block.attention.qkv, block.attention.to_out[0], ffn.w1, ffn.w3, ffn.w2):
                m.to_fp8_1d2d()
        return self


def load_fibo(name: str = "dev"):
    repo_id = {"dev": "briaai/Fibo-2.0-dev"}[name]
    state_dict = load_hf_state_dict(repo_id, "transformer/diffusion_pytorch_model.safetensors.index.json")

    with torch.device("meta"):
        model = Fibo()

    model.load_state_dict(state_dict, assign=True)
    return model
