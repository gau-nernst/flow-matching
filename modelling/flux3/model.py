# https://github.com/black-forest-labs/flux-3-image-klein/blob/bfc8889/src/flux3/dit.py

from typing import NamedTuple

import torch
from gn_kernels import permute_nv_sf
from torch import Tensor, nn

from ..attn import dispatch_attn
from ..flux1.model import LastLayer, MLPEmbedder, Modulation, timestep_embedding
from ..flux1.modulate import modulate
from ..flux2.model import MLP
from ..linear import Linear
from ..rope import apply_rope
from ..utils import create_remap_hook, load_hf_state_dict


# same as Flux.2's SingleStreamBlock, but with separate q/k/v/mlp_in and attn_out/mlp_out weights.
# we don't merge q/k/v: quantized checkpoints have separate scales for each, and merging only gives <1% speedup.
# xs is a list of segments, each with its own modulation (e.g. image and text in single-stream blocks).
# they are concatenated for qkv/mlp projections and attention.
class Block(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, eps: float = 1e-6) -> None:
        super().__init__()
        self.head_dim = 128
        self.eps = eps
        self.attn_impl = "pt"

        self.q_proj = Linear(dim, dim, bias=False)
        self.k_proj = Linear(dim, dim, bias=False)
        self.v_proj = Linear(dim, dim, bias=False)
        self.attn_out = Linear(dim, dim, bias=False)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        self.q_norm = nn.Parameter(torch.empty(self.head_dim))
        self.k_norm = nn.Parameter(torch.empty(self.head_dim))

        remap_pairs = [
            ("norm.query_norm.scale", "q_norm"),
            ("norm.key_norm.scale", "k_norm"),
        ]
        for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
            remap_pairs.append((f"mlp_in.{suffix}", f"mlp.0.{suffix}"))
            remap_pairs.append((f"mlp_out.{suffix}", f"mlp.2.{suffix}"))
        self.register_load_state_dict_pre_hook(create_remap_hook(remap_pairs))

    def forward(
        self,
        xs: list[Tensor],
        ress: list[Tensor | None],
        pe: Tensor,
        mods: list[tuple[Tensor, ...]],
    ) -> tuple[list[Tensor], list[Tensor]]:
        x_mods = []
        new_xs = []
        for x, res, (shift, scale, gate) in zip(xs, ress, mods):
            x_mod, x = modulate(x, shift, scale, res, gate)
            x_mods.append(x_mod)
            new_xs.append(x)
        x_mod = torch.cat(x_mods, dim=1) if len(x_mods) > 1 else x_mods[0]

        q = apply_rope(self.q_proj(x_mod).unflatten(2, (-1, self.head_dim)), pe, self.q_norm, self.eps)
        k = apply_rope(self.k_proj(x_mod).unflatten(2, (-1, self.head_dim)), pe, self.k_norm, self.eps)
        v = self.v_proj(x_mod).unflatten(2, (-1, self.head_dim))
        attn = dispatch_attn(q, k, v, impl=self.attn_impl).flatten(2)

        # TODO: fuse residual
        out = self.attn_out(attn) + self.mlp(x_mod)
        outs = list(out.split([x.shape[1] for x in xs], dim=1))
        return outs, new_xs


# default is klein (distilled)
class Flux3Config(NamedTuple):
    img_dim: int = 128
    txt_dim: int = 20480
    dim: int = 3072
    mlp_ratio: float = 3.0
    num_mode_blocks: int = 5
    num_single_blocks: int = 28
    flow_map: bool = True


class Flux3(nn.Module):
    def __init__(self, cfg: Flux3Config = Flux3Config()) -> None:
        super().__init__()
        self.cfg = cfg

        def make_blocks(n: int):
            return nn.ModuleList([Block(cfg.dim, cfg.mlp_ratio) for _ in range(n)])

        # input projections
        self.emb_in = nn.ModuleDict({k: Linear(cfg.img_dim, cfg.dim, bias=False) for k in ("image", "image_cond")})
        self.txt_in = Linear(cfg.txt_dim, cfg.dim, bias=False)
        self.time_in = MLPEmbedder(256, cfg.dim, bias=False)
        if cfg.flow_map:
            self.ref_time_in = MLPEmbedder(256, cfg.dim, bias=False)

        # use our Linear, since modulation weights are quantized in FP8/NVFP4 checkpoints
        def make_mod():
            mod = Modulation(cfg.dim, double=False, bias=False)
            mod.lin = Linear(cfg.dim, cfg.dim * 3, bias=False)
            return mod

        modes = ("image", "image_cond", "txt")
        self.early_stream_modulations = nn.ModuleDict({k: make_mod() for k in modes})
        self.single_stream_modulations = nn.ModuleDict({k: make_mod() for k in modes})

        # per-mode blocks: each mode only attends to itself
        self.content_mode_blocks = nn.ModuleDict({k: make_blocks(cfg.num_mode_blocks) for k in ("image", "image_cond")})
        self.txt_mode_blocks = make_blocks(cfg.num_mode_blocks)

        self.single_blocks = make_blocks(cfg.num_single_blocks)
        self.final_layer = nn.ModuleDict({"image": LastLayer(cfg.dim, 1, cfg.img_dim, bias=False)})

    # 4D RoPE (t, h, w, l) with theta=10_000 and float positions
    # - image:     (0, h/2, w/2, 0)
    # - image_cond (reference image i, 0-indexed): (0, h/2, w/2, -(i+1))
    # - text:      (0, 0, 0, l)
    @staticmethod
    def make_rope(ids: Tensor) -> Tensor:
        rope_list = []
        for i in range(ids.shape[-1]):
            dim = 32
            omega = 1.0 / (10_000 ** (torch.arange(0, dim, 2, dtype=torch.float64, device=ids.device) / dim))
            freqs = ids[:, i, None].double() * omega
            rope_list.append(torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64))
        return torch.cat(rope_list, dim=-1)

    @staticmethod
    def make_img_rope(H: int, W: int, l: int = 0, device: torch.types.Device = None) -> Tensor:
        h, w = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
        zeros = torch.zeros_like(h)
        ids = torch.stack([zeros, h * 0.5, w * 0.5, torch.full_like(h, l)], dim=-1).flatten(0, 1)
        return Flux3.make_rope(ids)

    @staticmethod
    def make_txt_rope(L: int, device: torch.types.Device = None) -> Tensor:
        l = torch.arange(L, device=device)
        zeros = torch.zeros_like(l)
        ids = torch.stack([zeros, zeros, zeros, l], dim=-1)
        return Flux3.make_rope(ids)

    def time_embed(self, mlp: MLPEmbedder, t: Tensor) -> Tensor:
        # official code does t * 1000 in BF16
        t = (t.bfloat16() * 1000).float()
        return mlp(timestep_embedding(t, 256, time_factor=1.0).to(self.txt_in.weight.dtype))

    # cond: tokens of all reference images, concatenated along the sequence dim
    def forward(
        self,
        img: Tensor,
        time: Tensor,
        txt: Tensor,
        img_rope: Tensor,
        txt_rope: Tensor,
        ref_time: Tensor | None = None,
        cond: Tensor | None = None,
        cond_rope: Tensor | None = None,
    ) -> Tensor:
        assert (ref_time is not None) == self.cfg.flow_map
        assert (cond is None) == (cond_rope is None)
        dtype = self.txt_in.weight.dtype

        xs = dict(image=self.emb_in["image"](img.to(dtype)), txt=self.txt_in(txt))
        ropes = dict(image=img_rope, txt=txt_rope)
        if cond is not None:
            xs["image_cond"] = self.emb_in["image_cond"](cond.to(dtype))
            ropes["image_cond"] = cond_rope

        # text and reference images are always at t=0
        vec_img = self.time_embed(self.time_in, time)
        vec_zero = self.time_embed(self.time_in, torch.zeros_like(time))
        if ref_time is not None:
            vec_ref = self.time_embed(self.ref_time_in, ref_time)
            vec_img = vec_img + vec_ref
            vec_zero = vec_zero + vec_ref
        vecs = {k: vec_img if k == "image" else vec_zero for k in xs}

        # per-mode blocks
        blocks = dict(
            image=self.content_mode_blocks["image"],
            image_cond=self.content_mode_blocks["image_cond"],
            txt=self.txt_mode_blocks,
        )
        mods = {k: self.early_stream_modulations[k](vecs[k]) for k in xs}
        ress = dict.fromkeys(xs)
        for i in range(self.cfg.num_mode_blocks):
            for k in xs:
                (ress[k],), (xs[k],) = blocks[k][i]([xs[k]], [ress[k]], ropes[k], [mods[k]])
        xs = {k: torch.addcmul(xs[k], mods[k][-1], ress[k]) for k in xs}

        # single-stream blocks. image must be the 1st segment.
        keys = list(xs)
        mods = [self.single_stream_modulations[k](vecs[k]) for k in keys]
        rope = torch.cat([ropes[k] for k in keys], dim=0)
        x_list = [xs[k] for k in keys]
        res_list = [None] * len(keys)
        for block in self.single_blocks:
            res_list, x_list = block(x_list, res_list, rope, mods)
        img = torch.addcmul(x_list[0], mods[0][-1], res_list[0])

        return self.final_layer["image"](img, vec_img)


def _load_flux3(repo_id: str, filename: str):
    state_dict = load_hf_state_dict(repo_id, filename)

    # official NVFP4 checkpoint stores unswizzled block scales. our kernels expect swizzled ones.
    for key in list(state_dict.keys()):
        if key.endswith(".weight_scale_2"):
            ws_key = key.removesuffix("_2")
            state_dict[ws_key] = permute_nv_sf(state_dict[ws_key])

    num_mode_blocks = 0
    num_single_blocks = 0

    for key in state_dict.keys():
        if key.startswith("txt_mode_blocks."):
            num_mode_blocks = max(num_mode_blocks, int(key.split(".")[1]) + 1)
        elif key.startswith("single_blocks."):
            num_single_blocks = max(num_single_blocks, int(key.split(".")[1]) + 1)

    dim, txt_dim = state_dict["txt_in.weight"].shape
    flow_map = "ref_time_in.in_layer.weight" in state_dict

    cfg = Flux3Config(
        txt_dim=txt_dim,
        dim=dim,
        num_mode_blocks=num_mode_blocks,
        num_single_blocks=num_single_blocks,
        flow_map=flow_map,
    )
    with torch.device("meta"):
        model = Flux3(cfg)

    model.load_state_dict(state_dict, assign=True)
    return model


def load_flux3(name: str = "klein"):
    repo_id, filename = {
        "klein": ("black-forest-labs/flux-3-image-klein", "model.safetensors"),
        "klein-fp8": ("black-forest-labs/flux-3-image-klein", "variants/fp8r/model.safetensors"),
        "klein-nvfp4": ("black-forest-labs/flux-3-image-klein", "variants/nvfp4/model.safetensors"),
        "klein-base": ("black-forest-labs/flux-3-image-klein-base", "model.safetensors"),
    }[name]

    return _load_flux3(repo_id, filename)
