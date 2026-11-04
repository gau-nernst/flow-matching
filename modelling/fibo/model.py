# https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/transformers/transformer_bria_fibo2.py
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
from ..varlen import VarlenMetadata
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

    # varlen, kv_varlen: of x and kv, for a varlen batch
    def forward(
        self,
        x: Tensor,
        kv: Tensor | None = None,
        varlen: VarlenMetadata | None = None,
        kv_varlen: VarlenMetadata | None = None,
    ) -> Tensor:
        kv = x if kv is None else kv
        q = self.norm_q(self.to_q(x).unflatten(-1, (-1, HEAD_DIM)))
        k = self.norm_k(self.to_k(kv).unflatten(-1, (-1, HEAD_DIM)))
        v = self.to_v(kv).unflatten(-1, (-1, HEAD_DIM))
        return self.to_out(dispatch_attn(q, k, v, varlen=varlen, kv_varlen=kv_varlen).flatten(-2))


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

    # x: [B, L, D], or [1, L, D] if varlen. vec: [N, 1, 256] timestep embeddings. token l of the flattened x uses
    # vec[mod_idx[l]]. txt_varlen: of txt
    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        mod_idx: Tensor,
        pe: Tensor,
        txt: Tensor | None = None,
        varlen: VarlenMetadata | None = None,
        txt_varlen: VarlenMetadata | None = None,
    ) -> Tensor:
        scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(vec).chunk(4, dim=-1)  # [N, 1, D] each

        # per-token modulation on the flattened tokens, as a varlen batch
        def norm(x: Tensor, w: Tensor, gate: Tensor, add: Tensor | None = None, gate_type: str = "plus_one"):
            flat = [None if t is None else t.view(1, -1, t.shape[-1]) for t in (x, add)]
            return gated_norm(flat[0], w, gate, flat[1], gate_type, self.eps, mod_idx).view(x.shape)

        y = norm(x, self.attention_norm1.weight, scale_msa)
        x = norm(self.attention(y, pe, varlen), self.attention_norm2.weight, gate_msa, x, "tanh")

        if self.cross_attention is not None:
            cross = self.cross_attention(self.cross_attn_norm1(x), txt, varlen, txt_varlen)
            x = gated_norm(cross, self.cross_attn_norm2.weight * self.cross_attn_gate.tanh(), add=x, eps=self.eps)

        ffn = self.feed_forward(norm(x, self.ffn_norm1.weight, scale_mlp))
        return norm(ffn, self.ffn_norm2.weight, gate_mlp, x, "tanh")


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

    def forward(
        self, x: Tensor, txt: Tensor, varlen: VarlenMetadata | None = None, txt_varlen: VarlenMetadata | None = None
    ) -> Tensor:
        x = x + self.cross_attention(self.cross_attn_norm(x), txt, varlen, txt_varlen)
        x = x + self.self_attention(self.self_attn_norm(x), varlen=varlen)
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

    # one sample's [gist, ref, image] rope. axes (plane, h, w). gist token i is on plane 1+i, the image on plane
    # MAX_NUM_GIST_TOKENS+1, ref image k on the k-th plane after it.
    # diffusers' order is [gist, image, ref]. we put refs before the image, so that clean and noisy tokens
    # are 2 contiguous segments. positions come from RoPE, so this only changes rounding.
    def make_rope(self, H: int, W: int, L_txt: int, ref_sizes: list[tuple[int, int]] = ()) -> Tensor:
        img_plane = MAX_NUM_GIST_TOKENS + 1
        gist_rope = self.pos_embed.create((1, 0, 0), (self.num_gist_tokens(L_txt), 1, 1))
        ref_rope = [self.pos_embed.create((img_plane + 1 + k, 0, 0), (1, *s)) for k, s in enumerate(ref_sizes)]
        return torch.cat([gist_rope, *ref_rope, self.pos_embed.create((img_plane, 0, 0), (1, H, W))])

    # uniform: latents BHWC, txt [B, 6, L, D], ref list[list[HWC]] -> img [B, HW, C]
    # varlen: latents list[HWC], txt list[6LD], ref list[list[HWC]] -> img [1, sum(HW), C]
    # latents: Flux.2 VAE latents, patchified + normalized. txt: 6 text encoder bundles, the Perceiver's first, then
    # one per injection block. ref[i]: sample i's images to edit, clean tokens (t=0). a uniform batch needs the same
    # ref shapes in each sample
    def prepare(
        self,
        latents: Tensor | list[Tensor],
        txt: Tensor | list[Tensor],
        ref: list[list[Tensor]] | None = None,
    ) -> tuple[Tensor, dict]:
        refs = ref or [[] for _ in range(len(latents))]
        assert len(latents) == len(txt) == len(refs), (len(latents), len(txt), len(refs))
        if not isinstance(latents, Tensor):
            return self._pack_varlen(latents, list(txt), refs)

        assert isinstance(txt, Tensor), (
            "latents is a tensor, so txt must be too. for a varlen batch, pass both as lists"
        )
        assert len({tuple(c.shape for c in cs) for cs in refs}) == 1, "a uniform batch needs the same ref shapes"
        ref = [torch.stack(cs) for cs in zip(*refs)]  # [B, h, w, C] for each image to edit

        _, H, W, _ = latents.shape
        rope = self.make_rope(H, W, txt.shape[2], [c.shape[1:3] for c in ref])
        kwargs = dict(txt=txt, rope=rope)
        if ref:
            kwargs["ref"] = torch.cat([c.flatten(1, 2) for c in ref], dim=1)
        return latents.flatten(1, 2), kwargs

    # varlen batch: [all gist tokens, all ref tokens, all image tokens], each part sample after sample.
    # the txt bundles are concatenated along L: [1, 6, sum L_i, txt_dim]
    def _pack_varlen(self, latents: list[Tensor], txts: list[Tensor], refs: list[list[Tensor]]) -> tuple[Tensor, dict]:
        txt_lens = [x.shape[1] for x in txts]
        gist_lens = [self.num_gist_tokens(L) for L in txt_lens]
        ref_lens = [sum(c.shape[0] * c.shape[1] for c in cs) for cs in refs]
        img_lens = [x.shape[0] * x.shape[1] for x in latents]
        ropes = [[], [], []]  # gist, ref, img
        for x, L, cs, lens in zip(latents, txt_lens, refs, zip(gist_lens, ref_lens, img_lens)):
            sample_rope = self.make_rope(x.shape[0], x.shape[1], L, [c.shape[:2] for c in cs])
            for part, rope in zip(ropes, sample_rope.split(lens)):
                part.append(rope)

        device = latents[0].device
        gist_varlen = VarlenMetadata.from_lens(gist_lens, device)
        img_varlen = VarlenMetadata.from_lens(img_lens, device)
        kwargs = dict(
            txt=torch.cat(txts, dim=1)[None],
            rope=torch.cat(sum(ropes, [])),
            txt_varlen=VarlenMetadata.from_lens(txt_lens, device),
            gist_varlen=gist_varlen,
            img_varlen=img_varlen,
            joint_varlen=gist_varlen.concat(img_varlen),
        )
        if any(ref_lens):
            ref_varlen = VarlenMetadata.from_lens(ref_lens, device)
            kwargs["ref"] = torch.cat([c.flatten(0, 1) for cs in refs for c in cs])[None]
            kwargs["ref_img_varlen"] = ref_varlen.concat(img_varlen)
            kwargs["joint_varlen"] = gist_varlen.concat(ref_varlen, img_varlen)
        return torch.cat([x.flatten(0, 1) for x in latents])[None], kwargs

    def unpatchify(self, out: Tensor, sizes: tuple[int, int] | list[tuple[int, int]]) -> Tensor | list[Tensor]:
        # sizes: (H, W) of a uniform batch, or a list per sample
        if isinstance(sizes, tuple):
            return out.unflatten(1, sizes)
        outs = out.flatten(0, -2).split([H * W for H, W in sizes])
        return [x.unflatten(0, size) for x, size in zip(outs, sizes)]

    # inputs are from prepare(). img: [B, L_img, C]. time: [B] in [0, 1], 1 is noise. ref: [B, L_ref, C]
    # varlen batch: img [1, L_img, C], txt [1, 6, L_txt, txt_dim], ref [1, L_ref, C]
    def forward(
        self,
        img: Tensor,
        time: Tensor,
        txt: Tensor,
        rope: Tensor,
        ref: Tensor | None = None,
        txt_varlen: VarlenMetadata | None = None,
        gist_varlen: VarlenMetadata | None = None,
        img_varlen: VarlenMetadata | None = None,
        joint_varlen: VarlenMetadata | None = None,
        ref_img_varlen: VarlenMetadata | None = None,
    ) -> Tensor:
        varlen = img_varlen is not None
        B = img_varlen.cu_seqlens.shape[0] - 1 if varlen else img.shape[0]
        dtype = self.x_embedder.weight.dtype

        # image tokens get the real timestep, clean tokens (gist, ref) get t=0, which is the same for every sample
        t_mlp = self.t_embedder.mlp
        vec = t_mlp(timestep_embedding(time.expand(B), 256).to(dtype))  # [B, 256]
        vec_clean = t_mlp(timestep_embedding(time.new_zeros(1), 256).to(dtype))  # [1, 256]

        x = self.x_embedder((img if ref is None else torch.cat([ref, img], dim=1)).to(dtype))
        num_ref = x.shape[1] - img.shape[1]

        # token l is modulated by vecs[mod_idx[l]]: its sample's row for image tokens, the extra t=0 row B for clean
        # tokens (gist, ref)
        vecs = torch.cat([vec, vec_clean])[:, None]  # [B+1, 1, 256]
        device = img.device
        if varlen:
            # gist token i of each sample is latent query i
            num_gist = gist_varlen.seq_ids.shape[0]
            gist_pos = torch.arange(num_gist, device=device) - gist_varlen.cu_seqlens[gist_varlen.seq_ids]
            gist = self.perceiver.latent_queries[gist_pos][None]

            def mod_idx(num_clean: int) -> Tensor:  # [clean tokens of all samples, image tokens of all samples]
                return torch.cat([torch.full((num_clean,), B, dtype=torch.int32, device=device), img_varlen.seq_ids])

            refiner_kwargs = dict(varlen=ref_img_varlen if num_ref else img_varlen)
            joint_kwargs = dict(varlen=joint_varlen, txt_varlen=txt_varlen)
        else:
            num_gist = self.num_gist_tokens(txt.shape[2])
            gist = self.perceiver.latent_queries[:num_gist].unsqueeze(0).expand(B, -1, -1)

            def mod_idx(num_clean: int) -> Tensor:  # each sample: [its clean tokens, its image tokens]
                pos = torch.arange(num_clean + img.shape[1], device=device)
                sample = torch.arange(B, dtype=torch.int32, device=device)[:, None]
                return torch.where(pos < num_clean, B, sample).flatten()

            refiner_kwargs = joint_kwargs = {}

        refiner_idx = mod_idx(num_ref)
        for block in self.noise_refiner:
            x = block(x, vecs, refiner_idx, rope[num_gist:], **refiner_kwargs)

        # Perceiver
        txt = txt.to(dtype)
        for layer in self.perceiver.layers:
            gist = layer(gist, txt[:, 0], gist_varlen, txt_varlen)
        gist = self.perceiver.norm_out(gist)

        # one stream: gist tokens, ref tokens, then image tokens. injection blocks also read their own text bundle.
        x = torch.cat([gist, x], dim=1)
        bundle_ids = {layer_id: i + 1 for i, layer_id in enumerate(INJECTION_LAYER_IDS)}
        joint_idx = mod_idx(num_gist + num_ref)
        for i, block in enumerate(self.layers):
            x = block(x, vecs, joint_idx, rope, txt[:, bundle_ids[i]] if i in bundle_ids else None, **joint_kwargs)

        return self.final_layer(x[:, num_gist + num_ref :], vec, img_varlen)


def load_fibo(name: str = "fibo-2.0-dev"):
    spec = MODELS[name]
    state_dict = load_hf_state_dict(spec.repo_id, spec.filename)

    with torch.device("meta"):
        model = Fibo()

    model.load_state_dict(state_dict, assign=True)
    return model
