import dataclasses

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..attn import dispatch_attn
from ..flux1.model import timestep_embedding
from ..linear import Linear
from ..rope import RopeND, apply_rope
from ..utils import load_hf_state_dict, make_merge_hook


class SimpleModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(2, dim))

    def forward(self, vec: Tensor):
        return (vec + self.lin).chunk(2, dim=1)


class DoubleSharedModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(6 * dim))

    def forward(self, vec: Tensor):
        return (vec + self.lin).chunk(6, dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, axdims: list[int], theta: float = 1e2, ntk: float = 1.0):
        super().__init__()
        self.axdims = axdims
        self.theta = theta
        self.ntk = ntk

    def forward(self, pos: Tensor) -> Tensor:
        return torch.cat(
            [rope(pos[..., i], d, self.theta, self.ntk) for i, d in enumerate(self.axdims)],
            dim=-3,
        )


class RMSNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-05):
        super().__init__()
        self.features = features
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(features))

    def forward(self, x: Tensor) -> Tensor:
        t, dtype = x.float(), x.dtype
        t = F.rms_norm(t, (self.features,), eps=self.eps, weight=(self.scale.float() + 1.0))
        return t.to(dtype)


class MLP(nn.Module):
    def __init__(self, dim: int, multiplier: int) -> None:
        super().__init__()
        mlp_dim = int(2 * dim / 3) * multiplier
        mlp_dim = (mlp_dim + 128 - 1) // 128 * 128

        self.gate = Linear(dim, mlp_dim, bias=False)
        self.up = Linear(dim, mlp_dim, bias=False)
        self.down = Linear(mlp_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: int = None) -> None:
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.head_dim = dim // self.heads
        self.attn_impl = "pt"

        self.qkvg = Linear(dim, self.head_dim * (heads + self.kv_heads * 2) + dim, bias=False)
        self.split_sizes = [self.head_dim * heads, self.head_dim * self.kvheads, self.head_dim * self.kvheads, dim]
        self.wo = Linear(dim, dim, bias=False)
        self.qknorm = nn.Module()
        self.qknorm.qnorm = RMSNorm(self.head_dim)
        self.qknorm.knorm = RMSNorm(self.head_dim)

        self.register_load_state_dict_pre_hook(make_merge_hook(["wq", "wk", "wv", "gate"], "qkvg"))

    def forward(self, x: Tensor, rope: Tensor) -> Tensor:
        q, k, v, gate = self.qkvg(x).split(self.split_sizes)
        q = q.unflatten(2, (-1, self.head_dim))
        k = k.unflatten(2, (-1, self.head_dim))
        v = v.unflatten(2, (-1, self.head_dim))

        q = apply_rope(q, rope, self.qknorm.qnorm.scale, gemma_norm=True, eps=1e-5)
        k = apply_rope(k, rope, self.qknorm.knorm.scale, gemma_norm=True, eps=1e-5)
        out = dispatch_attn(q, k, v, self.attn_impl).flatten(2)
        return self.wo(out * F.sigmoid(gate))


class LastLayer(nn.Module):
    def __init__(self, features: int, patch: int, channels: int):
        super().__init__()
        self.norm = RMSNorm(features)
        self.linear = Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x: Tensor, tvec: Tensor) -> Tensor:
        scale, shift = self.modulation(tvec)
        x = (1 + scale) * self.norm(x) + shift
        return self.linear(x)


class TextFusionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, multiplier: int, kvheads: int = None):
        super().__init__()
        self.prenorm = RMSNorm(dim)
        self.postnorm = RMSNorm(dim)
        self.attn = Attention(dim, heads, kvheads)
        self.mlp = MLP(dim, multiplier)

    def forward(self, x: Tensor) -> Tensor:
        # TODO: custom gemma rmsnorm
        # TODO: fuse add to output gemm
        x = x + self.attn(self.prenorm(x))
        x = x + self.mlp(self.postnorm(x))
        return x


class TextFusionTransformer(nn.Module):
    def __init__(self, n_layers: int, dim: int, n_heads: int, multiplier: int, n_kv_heads: int = None) -> None:
        super().__init__()
        self.layerwise_blocks = nn.Sequential(
            *[TextFusionBlock(dim, n_heads, multiplier, n_kv_heads) for _ in range(2)]
        )
        self.projector = Linear(n_layers, 1, bias=False)
        self.refiner_blocks = nn.Sequential(*[TextFusionBlock(dim, n_heads, multiplier, n_kv_heads) for _ in range(2)])

    def forward(self, x: Tensor) -> Tensor:
        B, L, N, D = x.shape
        x = x.reshape(B * L, N, D)
        x = self.layerwise_blocks(x)
        x = x.view(B, L, N, D).transpose(2, 3)  # [B, L, D, N]
        x = self.projector(x).squeeze(-1)  # [B, L, D]
        x = self.refiner_blocks(x)
        return x


class SingleStreamBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, multiplier: int, n_kv_heads: int = None) -> None:
        super().__init__()
        self.mod = DoubleSharedModulation(dim)
        self.prenorm = RMSNorm(dim)
        self.postnorm = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, n_kv_heads)
        self.mlp = MLP(dim, multiplier)

    def forward(self, x: Tensor, vec: Tensor, rope: Tensor) -> Tensor:
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x = x + pregate * self.attn((1 + prescale) * self.prenorm(x) + preshift, rope)
        x = x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)
        return x


@dataclasses.dataclass
class Krea2Config:
    img_dim: int
    txt_dim: int
    t_dim: int
    dim: int
    n_heads: int
    multiplier: int
    n_layers: int
    patch_size: int
    theta: float = 1e3
    n_kv_heads: int | None = None
    n_txt_layers: int = 1
    n_txt_heads: int = 20
    n_txt_kv_heads: int = 20


class Krea2(nn.Module):
    def __init__(self, cfg: Krea2Config | None = None) -> None:
        super().__init__()
        cfg = cfg or Krea2Config()
        self.cfg = cfg

        self.first = Linear(cfg.img_dim * cfg.patch_size * cfg.patch_size, cfg.dim)
        self.blocks = nn.ModuleList(
            [SingleStreamBlock(cfg.dim, cfg.n_heads, cfg.multiplier, cfg.n_kv_heads) for _ in range(cfg.n_layers)]
        )
        self.tmlp = nn.Sequential(Linear(cfg.t_dim, cfg.dim), nn.GELU(approximate="tanh"), Linear(cfg.dim, cfg.dim))
        self.tproj = nn.Sequential(nn.GELU(approximate="tanh"), Linear(cfg.features, cfg.features * 6))

        self.txtfusion = TextFusionTransformer(
            cfg.n_txt_layers, cfg.txt_dim, cfg.n_txt_heads, cfg.multiplier, cfg.n_txt_kv_heads
        )
        self.txtmlp = nn.Sequential(
            RMSNorm(cfg.txt_dim), Linear(cfg.txt_dim, cfg.dim), nn.GELU(approximate="tanh"), Linear(cfg.dim, cfg.dim)
        )
        self.last = LastLayer(cfg.dim, cfg.patch_size, cfg.img_dim)

    def forward(self, img: Tensor, txt: Tensor, t: Tensor, pos: Tensor) -> Tensor:
        img = self.first(img)
        txt = self.txtmlp(self.txtfusion(txt))

        txtlen = txt.shape[1]
        imglen = img.shape[1]
        combined = torch.cat((txt, img), dim=1)

        tvec = timestep_embedding(t, self.cfg.tdim)
        t = self.tmlp(tvec.to(self.tmlp[0].weight.dtype))
        tvec = self.tproj(t)
        freqs = self.posemb(pos)
        for block in self.blocks:
            combined = block(combined, tvec, freqs)

        final = self.last(combined, t)
        output = final[:, txtlen : txtlen + imglen, :]
        return output
