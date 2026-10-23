# https://github.com/black-forest-labs/flux-3-image-klein/blob/bfc8889/src/flux3/dit.py

from typing import NamedTuple

import torch
from torch import Tensor, nn

from ..attn import dispatch_attn
from ..flux1.model import LastLayer, MLPEmbedder, Modulation, timestep_embedding
from ..flux1.modulate import modulate
from ..flux2.model import MLP
from ..linear import Linear
from ..rope import apply_rope
from ..utils import create_remap_hook, create_weight_merge_hook, load_hf_state_dict


# same as Flux.2's SingleStreamBlock, but with separate q/k/v/mlp_in and attn_out/mlp_out weights.
# xs is a list of segments, each with its own modulation (e.g. image and text in single-stream blocks).
# they are concatenated for qkv/mlp projections and attention.
class Block(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, eps: float = 1e-6) -> None:
        super().__init__()
        self.head_dim = 128
        self.eps = eps
        self.attn_impl = "pt"

        self.qkv = Linear(dim, dim * 3, bias=False)
        self.attn_out = Linear(dim, dim, bias=False)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        self.q_norm = nn.Parameter(torch.empty(self.head_dim))
        self.k_norm = nn.Parameter(torch.empty(self.head_dim))

        self.register_load_state_dict_pre_hook(create_weight_merge_hook(["q_proj", "k_proj", "v_proj"], "qkv"))
        remap_pairs = [
            ("mlp_in.weight", "mlp.0.weight"),
            ("mlp_out.weight", "mlp.2.weight"),
            ("norm.query_norm.scale", "q_norm"),
            ("norm.key_norm.scale", "k_norm"),
        ]
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

        q, k, v = self.qkv(x_mod).unflatten(2, (-1, self.head_dim)).chunk(3, dim=2)
        q = apply_rope(q, pe, self.q_norm, self.eps)
        k = apply_rope(k, pe, self.k_norm, self.eps)
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

        modes = ("image", "image_cond", "txt")
        self.early_stream_modulations = nn.ModuleDict({k: Modulation(cfg.dim, double=False, bias=False) for k in modes})
        self.single_stream_modulations = nn.ModuleDict({k: Modulation(cfg.dim, double=False, bias=False) for k in modes})

        # per-mode blocks: each mode only attends to itself
        self.content_mode_blocks = nn.ModuleDict({k: make_blocks(cfg.num_mode_blocks) for k in ("image", "image_cond")})
        self.txt_mode_blocks = make_blocks(cfg.num_mode_blocks)

        self.single_blocks = make_blocks(cfg.num_single_blocks)
        self.final_layer = nn.ModuleDict({"image": LastLayer(cfg.dim, 1, cfg.img_dim, bias=False)})

    # 4D RoPE (t, h, w, l) with theta=10_000 and float positions
    # - image: (0, h/2, w/2, 0)
    # - text:  (0, 0, 0, l)
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
    def make_img_rope(H: int, W: int, device: torch.types.Device = None) -> Tensor:
        h, w = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
        zeros = torch.zeros_like(h)
        ids = torch.stack([zeros, h * 0.5, w * 0.5, zeros], dim=-1).flatten(0, 1)
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

    # TODO: image_cond
    def forward(
        self,
        img: Tensor,
        time: Tensor,
        txt: Tensor,
        img_rope: Tensor,
        txt_rope: Tensor,
        ref_time: Tensor | None = None,
    ) -> Tensor:
        assert (ref_time is not None) == self.cfg.flow_map

        img = self.emb_in["image"](img.to(self.txt_in.weight.dtype))
        txt = self.txt_in(txt)

        # text is always at t=0
        vec_img = self.time_embed(self.time_in, time)
        vec_txt = self.time_embed(self.time_in, torch.zeros_like(time))
        if ref_time is not None:
            vec_ref = self.time_embed(self.ref_time_in, ref_time)
            vec_img = vec_img + vec_ref
            vec_txt = vec_txt + vec_ref

        mod_img = self.early_stream_modulations["image"](vec_img)
        mod_txt = self.early_stream_modulations["txt"](vec_txt)
        img_res = txt_res = None
        for img_block, txt_block in zip(self.content_mode_blocks["image"], self.txt_mode_blocks):
            (img_res,), (img,) = img_block([img], [img_res], img_rope, [mod_img])
            (txt_res,), (txt,) = txt_block([txt], [txt_res], txt_rope, [mod_txt])

        img = torch.addcmul(img, mod_img[-1], img_res)
        txt = torch.addcmul(txt, mod_txt[-1], txt_res)

        mod_img = self.single_stream_modulations["image"](vec_img)
        mod_txt = self.single_stream_modulations["txt"](vec_txt)
        rope = torch.cat([img_rope, txt_rope], dim=0)
        xs = [img, txt]
        ress = [None, None]
        for block in self.single_blocks:
            ress, xs = block(xs, ress, rope, [mod_img, mod_txt])
        img = torch.addcmul(xs[0], mod_img[-1], ress[0])

        return self.final_layer["image"](img, vec_img)


def _load_flux3(repo_id: str, filename: str):
    state_dict = load_hf_state_dict(repo_id, filename)

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
        "klein-base": ("black-forest-labs/flux-3-image-klein-base", "model.safetensors"),
    }[name]

    return _load_flux3(repo_id, filename)
