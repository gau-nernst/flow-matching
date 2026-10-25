# https://github.com/black-forest-labs/flux/blob/7e14a05e/src/flux/math.py
# https://github.com/black-forest-labs/flux/blob/7e14a05e/src/flux/modules/layers.py
# https://github.com/black-forest-labs/flux/blob/7e14a05e/src/flux/model.py

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn

from ..attn import dispatch_attn
from ..linear import Linear
from ..rope import RopeND, apply_rope
from ..utils import create_remap_hook, load_hf_state_dict
from ..varlen import Joint, Varlen, merge, per_token, seq_ids, split
from .modulate import modulate


def timestep_embedding(t: Tensor, dim: int, max_period: float = 10_000.0, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t.float()
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) / half * torch.arange(half, dtype=torch.float32, device=t.device))
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([args.cos(), args.sin()], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class MLPEmbedder(nn.Sequential):
    def __init__(self, in_dim: int, hidden_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=bias)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, bias: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        self.head_dim = 128
        self.qkv = Linear(dim, dim * 3, bias=bias)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=eps)
        self.proj = Linear(dim, dim, bias=bias)
        remap_pairs = [
            ("norm.query_norm.scale", "q_norm.weight"),
            ("norm.key_norm.scale", "k_norm.weight"),
        ]
        self.register_load_state_dict_pre_hook(create_remap_hook(remap_pairs))

    def forward_qkv(self, x: Tensor):
        q, k, v = self.qkv(x).unflatten(2, (-1, self.head_dim)).chunk(3, dim=2)
        return self.q_norm(q), self.k_norm(k), v


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool, bias: bool = True) -> None:
        super().__init__()
        self.multiplier = 6 if double else 3
        self.silu = nn.SiLU()
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=bias)

    def forward(self, vec: Tensor) -> tuple[Tensor, ...]:
        return self.lin(self.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)


class DoubleStreamBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, attn_impl: str = "pt", eps: float = 1e-6) -> None:
        super().__init__()
        self.head_dim = 128
        mlp_dim = int(dim * mlp_ratio)
        self.attn_impl = attn_impl
        self.eps = eps

        self.img_mod = Modulation(dim, double=True)
        self.txt_mod = Modulation(dim, double=True)

        self.img_attn = SelfAttention(dim, eps=eps)
        self.txt_attn = SelfAttention(dim, eps=eps)

        self.img_mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(approximate="tanh"), nn.Linear(mlp_dim, dim))
        self.txt_mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(approximate="tanh"), nn.Linear(mlp_dim, dim))

    # joint: [txt, img] per sample
    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, joint: Joint | None) -> tuple[Tensor, Tensor]:
        txt_varlen, img_varlen = (joint.a, joint.b) if joint is not None else (None, None)
        img_idx, txt_idx = seq_ids(img_varlen), seq_ids(txt_varlen)
        img_shift1, img_scale1, img_gate1, img_shift2, img_scale2, img_gate2 = self.img_mod(vec)
        txt_shift1, txt_scale1, txt_gate1, txt_shift2, txt_scale2, txt_gate2 = self.txt_mod(vec)

        img_mod1 = modulate(img, img_shift1, img_scale1, eps=self.eps, idx=img_idx)[0]
        txt_mod1 = modulate(txt, txt_shift1, txt_scale1, eps=self.eps, idx=txt_idx)[0]
        img_q, img_k, img_v = self.img_attn.forward_qkv(img_mod1)
        txt_q, txt_k, txt_v = self.txt_attn.forward_qkv(txt_mod1)

        q = apply_rope(merge(txt_q, img_q, joint), pe)
        k = apply_rope(merge(txt_k, img_k, joint), pe)
        v = merge(txt_v, img_v, joint)
        attn = dispatch_attn(q, k, v, self.attn_impl, joint.varlen if joint is not None else None).flatten(2)
        txt_attn, img_attn = split(attn, txt.shape[1], joint)

        img = torch.addcmul(img, per_token(img_gate1, img_varlen), self.img_attn.proj(img_attn))
        img_mod2 = modulate(img, img_shift2, img_scale2, idx=img_idx)[0]
        img = torch.addcmul(img, per_token(img_gate2, img_varlen), self.img_mlp(img_mod2))

        txt = torch.addcmul(txt, per_token(txt_gate1, txt_varlen), self.txt_attn.proj(txt_attn))
        txt_mod2 = modulate(txt, txt_shift2, txt_scale2, idx=txt_idx)[0]
        txt = torch.addcmul(txt, per_token(txt_gate2, txt_varlen), self.txt_mlp(txt_mod2))

        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(self, dim: int, mlp_ratio: float, attn_impl: str = "pt", eps: float = 1e-6) -> None:
        super().__init__()
        self.head_dim = 128
        self.dim = dim
        self.mlp_dim = int(dim * mlp_ratio)
        self.attn_impl = attn_impl
        self.eps = eps

        self.linear1 = nn.Linear(dim, dim * 3 + self.mlp_dim)  # qkv and mlp_in
        self.linear2 = nn.Linear(dim + self.mlp_dim, dim)  # proj and mlp_out

        self.q_norm = nn.RMSNorm(self.head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=eps)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(dim, double=False)

        remap_pairs = [
            ("norm.query_norm.scale", "q_norm.weight"),
            ("norm.key_norm.scale", "k_norm.weight"),
        ]
        self.register_load_state_dict_pre_hook(create_remap_hook(remap_pairs))

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor, varlen: Varlen | None) -> Tensor:
        shift, scale, gate = self.modulation(vec)
        x_mod, _ = modulate(x, shift, scale, eps=self.eps, idx=seq_ids(varlen))
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.dim, self.mlp_dim], dim=-1)

        q, k, v = qkv.unflatten(2, (-1, self.head_dim)).chunk(3, dim=2)
        q = apply_rope(self.q_norm(q), pe)
        k = apply_rope(self.k_norm(k), pe)
        attn = dispatch_attn(q, k, v, self.attn_impl, varlen).flatten(2)

        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + per_token(gate, varlen) * output


class LastLayer(nn.Module):
    def __init__(self, dim: int, patch_size: int, out_channels: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels, bias=bias)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=bias))

    def forward(self, x: Tensor, vec: Tensor, varlen: Varlen | None = None) -> Tensor:
        shift, scale = self.adaLN_modulation(vec)[:, None, :].chunk(2, dim=-1)
        return self.linear(modulate(x, shift, scale, idx=seq_ids(varlen))[0])


# default is Flux.1-dev
class Flux1Config(NamedTuple):
    img_dim: int = 64
    txt_dim: int = 4096
    vec_dim: int = 768
    dim: int = 3072
    num_double_blocks: int = 19
    num_single_blocks: int = 38
    patch_size: int = 2
    guidance_embed: bool = True  # False for schnell


class Flux1(nn.Module):
    def __init__(self, cfg: Flux1Config = Flux1Config()) -> None:
        super().__init__()
        self.cfg = cfg

        # input projections
        self.img_in = nn.Linear(cfg.img_dim, cfg.dim)
        self.txt_in = nn.Linear(cfg.txt_dim, cfg.dim)
        self.time_in = MLPEmbedder(256, cfg.dim)
        self.vector_in = MLPEmbedder(cfg.vec_dim, cfg.dim)
        if cfg.guidance_embed:
            self.guidance_in = MLPEmbedder(256, cfg.dim)

        # 3D rope
        self.pos_embed = RopeND(dims=(16, 56, 56), max_lens=(512, 512, 512), theta=1e4)

        mlp_ratio = 4.0
        self.double_blocks = nn.ModuleList(
            [DoubleStreamBlock(cfg.dim, mlp_ratio) for _ in range(cfg.num_double_blocks)]
        )
        self.single_blocks = nn.ModuleList(
            [SingleStreamBlock(cfg.dim, mlp_ratio) for _ in range(cfg.num_single_blocks)]
        )
        self.final_layer = LastLayer(cfg.dim, 1, cfg.img_dim)

    def make_rope(self, H: int, W: int, L: int):
        # one sample's [txt, img] rope. H, W: patchified size
        # RoPE embedding has 3 components:
        # - time: text embeds stay at pos=0, img embeds stay at pos=0
        # - height: all text embeds stay at pos=0
        # - width: all text embeds stay at pos=0
        img_rope = self.pos_embed.create((0, 0, 0), (1, H, W))
        txt_rope = self.pos_embed.create((0, 0, 0), (1, 1, 1)).expand(L, -1)
        return torch.cat([txt_rope, img_rope], dim=0)

    def forward(
        self,
        img: Tensor,
        t: Tensor,
        txt: Tensor,
        rope: Tensor,
        vec: Tensor,
        guidance: Tensor | None = None,
        varlen: Joint | None = None,
    ) -> Tensor:
        # img: [B, L, C*p*p] from patchify(channels_first=True), or [T, C*p*p] packed with varlen
        packed = varlen is not None
        if packed:
            img, txt = img[None], txt[None]
            N = varlen.varlen.num_seqs
            img_varlen, joint_varlen = varlen.b, varlen.varlen
        else:
            N = -1  # keep t's shape
            img_varlen = joint_varlen = None

        img = self.img_in(img.to(self.img_in.weight.dtype))
        txt = self.txt_in(txt)
        vec = self.time_in(timestep_embedding(t.expand(N), 256).to(img.dtype)) + self.vector_in(vec)

        if guidance is not None:  # allow no guidance_embed
            vec = vec + self.guidance_in(timestep_embedding(guidance.expand(N), 256).to(img.dtype))

        for block in self.double_blocks:
            img, txt = block(img, txt, vec, rope, varlen)

        L_txt = txt.shape[1]
        x = merge(txt, img, varlen)
        for block in self.single_blocks:
            x = block(x, vec, rope, joint_varlen)
        img = split(x, L_txt, varlen)[1]

        img = self.final_layer(img, vec, img_varlen)
        return img[0] if packed else img


def _load_flux1(repo_id: str, filename: str, prefix: str | None = None):
    state_dict = load_hf_state_dict(repo_id, filename, prefix=prefix)

    # support depth-pruned FLUX, such as Flex.1-alpha
    num_double_blocks = 0
    num_single_blocks = 0

    for key in state_dict.keys():
        if key.startswith("double_blocks."):
            num_double_blocks = max(num_double_blocks, int(key.split(".")[1]) + 1)
        elif key.startswith("single_blocks."):
            num_single_blocks = max(num_single_blocks, int(key.split(".")[1]) + 1)

    cfg = Flux1Config(
        num_double_blocks=num_double_blocks,
        num_single_blocks=num_single_blocks,
    )
    with torch.device("meta"):
        model = Flux1(cfg)

    model.load_state_dict(state_dict, assign=True)
    return model


def load_flux1(name: str = "dev"):
    repo_id, filename, prefix = {
        "dev": ("black-forest-labs/FLUX.1-dev", "flux1-dev.safetensors", None),
        "schnell": ("black-forest-labs/FLUX.1-schnell", "flux1-schnell.safetensors", None),
        "krea-dev": ("black-forest-labs/FLUX.1-Krea-dev", "flux1-krea-dev.safetensors", None),
    }[name]

    # BF16
    return _load_flux1(repo_id, filename, prefix=prefix)
