# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_bria_fibo2.py
# Z-Image blocks, plus a Perceiver that compresses the text encoder's hidden states into "gist" tokens, and gated
# cross-attention to the text encoder in a few blocks.

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn

from ..attn import dispatch_attn
from ..flux1.model import timestep_embedding
from ..linear import Linear
from ..registry import MODELS
from ..rope import RopeND
from ..utils import load_hf_state_dict
from ..z_image.gated_norm import gated_norm
from ..z_image.model import Attention, FeedForward, FinalLayer

HEAD_DIM = 128
MAX_NUM_GIST_TOKENS = 512
INJECTION_LAYER_IDS = (6, 14, 22, 30, 38)


# attention without RoPE, with optional different K/V input
class CrossAttention(nn.Module):
    def __init__(self, dim: int, kv_dim: int | None = None, eps: float = 1e-5) -> None:
        super().__init__()
        kv_dim = kv_dim or dim
        self.to_q = Linear(dim, dim, bias=False)
        self.to_k = Linear(kv_dim, dim, bias=False)
        self.to_v = Linear(kv_dim, dim, bias=False)
        self.norm_q = nn.RMSNorm(HEAD_DIM, eps=eps)
        self.norm_k = nn.RMSNorm(HEAD_DIM, eps=eps)
        self.to_out = nn.Sequential(Linear(dim, dim, bias=False))

    def forward(self, x: Tensor, kv: Tensor | None = None) -> Tensor:
        kv = x if kv is None else kv
        q = self.norm_q(self.to_q(x).unflatten(-1, (-1, HEAD_DIM)))
        k = self.norm_k(self.to_k(kv).unflatten(-1, (-1, HEAD_DIM)))
        v = self.to_v(kv).unflatten(-1, (-1, HEAD_DIM))
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


class FiboConfig(NamedTuple):
    dim: int = 3584
    num_layers: int = 44
    txt_dim: int = 7680  # 3 Qwen3-VL-4B layers concatenated
    img_dim: int = 128  # patchified latents


class Fibo(nn.Module):
    def __init__(self, cfg: FiboConfig = FiboConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        dim = cfg.dim
        mlp_dim = (dim * 8 // 3) // 64 * 64

        self.x_embedder = Linear(cfg.img_dim, dim)
        self.t_embedder = nn.Sequential()
        self.t_embedder.mlp = nn.Sequential(Linear(256, 1024), nn.SiLU(), Linear(1024, 256))
        self.pos_embed = RopeND((32, 48, 48), (2048, 512, 512), theta=256.0)

        self.noise_refiner = nn.ModuleList([Block(dim, mlp_dim) for _ in range(2)])
        self.perceiver = nn.Module()
        self.perceiver.latent_queries = nn.Parameter(torch.empty(MAX_NUM_GIST_TOKENS, dim))
        self.perceiver.layers = nn.ModuleList([PerceiverLayer(dim, mlp_dim, cfg.txt_dim) for _ in range(2)])
        self.perceiver.norm_out = nn.RMSNorm(dim, eps=1e-5)
        self.layers = nn.ModuleList(
            [
                Block(dim, mlp_dim, cross_dim=cfg.txt_dim if i in INJECTION_LAYER_IDS else None)
                for i in range(cfg.num_layers)
            ]
        )
        self.final_layer = FinalLayer(dim, cfg.img_dim)

    # number of gist tokens: 256 for text length <= 256, linear up to 512 at text length 2048, rounded up to a
    # multiple of 64
    @staticmethod
    def num_gist_tokens(txt_len: int) -> int:
        ratio = min(max((txt_len - 256) / (2048 - 256), 0.0), 1.0)
        return math.ceil((256 + ratio * (MAX_NUM_GIST_TOKENS - 256)) / 64) * 64

    # blocks to add LoRA to
    def lora_blocks(self) -> list[nn.Module]:
        return list(self.layers)

    # one sample's [gist, context, image] rope. axes (plane, h, w). gist token i is on plane 1+i, the image on plane
    # MAX_NUM_GIST_TOKENS+1, context image k on the k-th plane after it.
    # reference order is [gist, image, context]. we put context before the image, so that clean and noisy tokens
    # are 2 contiguous segments. positions come from RoPE, so this only changes rounding.
    def make_rope(self, H: int, W: int, L_txt: int, ctx_sizes: list[tuple[int, int]] = ()) -> Tensor:
        img_plane = MAX_NUM_GIST_TOKENS + 1
        gist_rope = self.pos_embed.create((1, 0, 0), (self.num_gist_tokens(L_txt), 1, 1))
        ctx_rope = [self.pos_embed.create((img_plane + 1 + k, 0, 0), (1, *s)) for k, s in enumerate(ctx_sizes)]
        return torch.cat([gist_rope, *ctx_rope, self.pos_embed.create((img_plane, 0, 0), (1, H, W))])

    # latents: [B, H, W, C] (Flux.2 VAE latents, patchified + normalized), or a list of [H_i, W_i, C] of one size
    # txt: [B, 6, L, txt_dim], or a list of [6, L, txt_dim] of one length. text encoder bundles: the Perceiver's first,
    # then one per injection block.
    # ctx: latents of the images to edit, [B, h, w, C] each, sizes can differ. clean tokens (t=0), not predicted.
    def prepare(
        self, latents: Tensor | list[Tensor], txt: Tensor | list[Tensor], ctx: list[Tensor] | None = None
    ) -> tuple[Tensor, dict]:
        if not isinstance(latents, Tensor):
            if len({x.shape for x in latents}) > 1 or len({x.shape for x in txt}) > 1:
                raise NotImplementedError("Fibo needs a batch of one image size and one prompt length")
            latents, txt = torch.stack(latents), torch.stack(list(txt))

        _, H, W, _ = latents.shape
        ctx = ctx or []
        rope = self.make_rope(H, W, txt.shape[2], [c.shape[1:3] for c in ctx])
        kwargs = dict(txt=txt, rope=rope)
        if ctx:
            kwargs["ctx"] = torch.cat([c.flatten(1, 2) for c in ctx], dim=1)
        return latents.flatten(1, 2), kwargs

    def unpatchify(self, out: Tensor, sizes: tuple[int, int] | list[tuple[int, int]]) -> Tensor | list[Tensor]:
        # sizes: (H, W) of a uniform batch, or a list per sample
        if isinstance(sizes, tuple):
            return out.unflatten(1, sizes)
        outs = out.flatten(0, -2).split([H * W for H, W in sizes])
        return [x.unflatten(0, size) for x, size in zip(outs, sizes)]

    # inputs are from prepare(). img: [B, L_img, C]. time: [B] in [0, 1], 1 is noise. ctx: [B, L_ctx, C]
    def forward(self, img: Tensor, time: Tensor, txt: Tensor, rope: Tensor, ctx: Tensor | None = None) -> Tensor:
        B, L_img, _ = img.shape
        dtype = self.x_embedder.weight.dtype

        # image tokens get the real timestep, clean tokens (gist, context) get t=0
        t_mlp = self.t_embedder.mlp
        vec = t_mlp(timestep_embedding(time, 256).to(dtype))
        vec_clean = t_mlp(timestep_embedding(torch.zeros_like(time), 256).to(dtype))
        vecs = torch.stack([vec_clean, vec], dim=1)

        num_gist = self.num_gist_tokens(txt.shape[2])
        x = self.x_embedder((img if ctx is None else torch.cat([ctx, img], dim=1)).to(dtype))
        num_ctx = x.shape[1] - L_img
        for block in self.noise_refiner:
            x = block(x, vecs, rope[num_gist:], split=num_ctx) if num_ctx else block(x, vec[:, None], rope[num_gist:])

        # Perceiver
        txt = txt.to(dtype)
        gist = self.perceiver.latent_queries[:num_gist].unsqueeze(0).expand(B, -1, -1)
        for layer in self.perceiver.layers:
            gist = layer(gist, txt[:, 0])
        gist = self.perceiver.norm_out(gist)

        # one stream: gist tokens, context tokens, then image tokens. injection blocks also read their own text bundle.
        x = torch.cat([gist, x], dim=1)
        bundle_ids = {layer_id: i + 1 for i, layer_id in enumerate(INJECTION_LAYER_IDS)}
        for i, block in enumerate(self.layers):
            x = block(x, vecs, rope, txt[:, bundle_ids[i]] if i in bundle_ids else None, num_gist + num_ctx)

        return self.final_layer(x[:, num_gist + num_ctx :], vec)


def load_fibo(name: str = "fibo-2.0-dev"):
    spec = MODELS[name]
    state_dict = load_hf_state_dict(spec.repo_id, spec.filename)

    with torch.device("meta"):
        model = Fibo()

    model.load_state_dict(state_dict, assign=True)
    return model
