import dataclasses

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..attn import dispatch_attn
from ..flux1.model import timestep_embedding
from ..linear import Linear
from ..rope import RopeND, apply_rope
from ..utils import load_hf_state_dict, make_merge_hook


def rope(pos: Tensor, dim: int, theta: float = 1e4, ntk: float = 1.0) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / ((theta * ntk) ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = out.unflatten(-1, (2, 2))
    return out.float()


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


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-05) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x.float(), (self.dim,), eps=self.eps, weight=self.scale.float() + 1.0).to(x.dtype)


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
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.attn_impl = "pt"

        self.qkvg = Linear(dim, self.head_dim * (n_heads + n_kv_heads * 2) + dim, bias=False)
        self.split_sizes = [self.head_dim * n_heads, self.head_dim * n_kv_heads, self.head_dim * n_kv_heads, dim]
        self.wo = Linear(dim, dim, bias=False)
        self.qknorm = nn.Module()
        self.qknorm.qnorm = GemmaRMSNorm(self.head_dim)
        self.qknorm.knorm = GemmaRMSNorm(self.head_dim)

        self.register_load_state_dict_pre_hook(make_merge_hook(["wq", "wk", "wv", "gate"], "qkvg"))

    def forward(self, x: Tensor, rope: Tensor | None) -> Tensor:
        q, k, v, gate = self.qkvg(x).split(self.split_sizes, dim=-1)
        q = q.unflatten(2, (-1, self.head_dim))
        k = k.unflatten(2, (-1, self.head_dim))
        v = v.unflatten(2, (-1, self.head_dim))

        if rope is not None:
            q = apply_rope(q, rope, self.qknorm.qnorm.scale, gemma_norm=True, eps=1e-5)
            k = apply_rope(k, rope, self.qknorm.knorm.scale, gemma_norm=True, eps=1e-5)
        else:
            q = self.qknorm.qnorm(q)
            k = self.qknorm.qnorm(k)
        out = dispatch_attn(q, k, v, self.attn_impl).flatten(2)
        return self.wo(out * F.sigmoid(gate))


class TextFusionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, multiplier: int) -> None:
        super().__init__()
        self.prenorm = GemmaRMSNorm(dim)
        self.postnorm = GemmaRMSNorm(dim)
        self.attn = Attention(dim, n_heads, n_heads)
        self.mlp = MLP(dim, multiplier)

    def forward(self, x: Tensor) -> Tensor:
        # TODO: custom gemma rmsnorm
        # TODO: fuse add to output gemm
        x = x + self.attn(self.prenorm(x), rope=None)
        x = x + self.mlp(self.postnorm(x))
        return x


class TextFusionTransformer(nn.Module):
    def __init__(self, n_layers: int, dim: int, n_heads: int, multiplier: int) -> None:
        super().__init__()
        self.layerwise_blocks = nn.Sequential(*[TextFusionBlock(dim, n_heads, multiplier) for _ in range(2)])
        self.projector = Linear(n_layers, 1, bias=False)
        self.refiner_blocks = nn.Sequential(*[TextFusionBlock(dim, n_heads, multiplier) for _ in range(2)])

    def forward(self, x: Tensor) -> Tensor:
        B, L, N, D = x.shape
        x = x.reshape(B * L, N, D)
        x = self.layerwise_blocks(x)
        x = x.view(B, L, N, D).transpose(2, 3)  # [B, L, D, N]
        x = self.projector(x).squeeze(-1)  # [B, L, D]
        x = self.refiner_blocks(x)
        return x


class SingleStreamBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, multiplier: int) -> None:
        super().__init__()
        self.mod = DoubleSharedModulation(dim)
        self.prenorm = GemmaRMSNorm(dim)
        self.postnorm = GemmaRMSNorm(dim)
        self.attn = Attention(dim, n_heads, n_kv_heads)
        self.mlp = MLP(dim, multiplier)

    def forward(self, x: Tensor, vec: Tensor, rope: Tensor) -> Tensor:
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x = x + pregate * self.attn((1 + prescale) * self.prenorm(x) + preshift, rope)
        x = x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)
        return x


class SimpleModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(2, dim))

    def forward(self, vec: Tensor):
        return (vec.unsqueeze(-2) + self.lin).chunk(2, dim=-2)


class LastLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = GemmaRMSNorm(in_dim)
        self.linear = Linear(in_dim, out_dim)
        self.modulation = SimpleModulation(in_dim)

    def forward(self, x: Tensor, tvec: Tensor) -> Tensor:
        scale, shift = self.modulation(tvec)
        x = (1 + scale) * self.norm(x) + shift
        return self.linear(x)


@dataclasses.dataclass
class Krea2Config:
    img_dim: int = 16
    txt_dim: int = 2560
    t_dim: int = 256
    dim: int = 6144
    n_heads: int = 48
    n_kv_heads: int = 12
    multiplier: int = 4
    n_layers: int = 28
    patch_size: int = 2
    theta: float = 1e3
    n_txt_layers: int = 12
    n_txt_heads: int = 20


class Krea2(nn.Module):
    def __init__(self, cfg: Krea2Config | None = None) -> None:
        super().__init__()
        cfg = cfg or Krea2Config()
        self.cfg = cfg

        headdim = cfg.dim // cfg.n_heads
        axes = (
            headdim - 12 * (headdim // 16),
            6 * (headdim // 16),
            6 * (headdim // 16),
        )
        self.posemb = PositionalEncoding(axes, theta=cfg.theta, ntk=1.0)
        self.pos_embed = RopeND(axes, (1536, 512, 512), theta=cfg.theta)

        self.first = Linear(cfg.img_dim * cfg.patch_size * cfg.patch_size, cfg.dim)
        self.blocks = nn.ModuleList(
            [SingleStreamBlock(cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.multiplier) for _ in range(cfg.n_layers)]
        )
        self.tmlp = nn.Sequential(Linear(cfg.t_dim, cfg.dim), nn.GELU(approximate="tanh"), Linear(cfg.dim, cfg.dim))
        self.tproj = nn.Sequential(nn.GELU(approximate="tanh"), Linear(cfg.dim, cfg.dim * 6))

        self.txtfusion = TextFusionTransformer(cfg.n_txt_layers, cfg.txt_dim, cfg.n_txt_heads, cfg.multiplier)
        self.txtmlp = nn.Sequential(
            GemmaRMSNorm(cfg.txt_dim),
            Linear(cfg.txt_dim, cfg.dim),
            nn.GELU(approximate="tanh"),
            Linear(cfg.dim, cfg.dim),
        )
        self.last = LastLayer(cfg.dim, cfg.patch_size * cfg.patch_size * cfg.img_dim)

    def make_rope(self, H: int, W: int) -> Tensor:
        return self.pos_embed.create((0, 0, 0), (1, H, W))

    def forward(self, img: Tensor, txt: Tensor, t: Tensor, rope: Tensor) -> Tensor:
        img = self.first(img)
        txt = self.txtmlp(self.txtfusion(txt))
        combined = torch.cat((txt, img), dim=1)

        t = self.tmlp(timestep_embedding(t, self.cfg.t_dim).to(self.tmlp[0].weight.dtype))
        tvec = self.tproj(t)

        for block in self.blocks:
            combined = block(combined, tvec, rope)

        img = combined[:, txt.shape[1] :]
        return self.last(img, t)


def load_krea2(name: str = "turbo"):
    repo_id = f"krea/Krea-2-{name.capitalize()}"
    filename = f"{name}.safetensors"
    state_dict = load_hf_state_dict(repo_id, filename)

    with torch.device("meta"):
        model = Krea2()

    model.load_state_dict(state_dict, assign=True)
    return model
