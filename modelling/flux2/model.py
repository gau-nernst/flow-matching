# https://github.com/black-forest-labs/flux2/blob/b56ac614/src/flux2/model.py

from typing import NamedTuple

import torch
import torch.nn.functional as F
from gn_kernels.cutedsl.sm120 import sm120_gated_gemm_nvfp4
from gn_kernels.quant_utils import quantize_nvfp4_triton
from torch import Tensor, nn

from ..attn import dispatch_attn
from ..flux1.model import LastLayer, MLPEmbedder, Modulation, timestep_embedding
from ..flux1.modulate import modulate
from ..linear import Linear, nvfp4_mm
from ..rope import RopeND, apply_rope
from ..utils import create_name_map_hook, load_hf_state_dict


class MLP(nn.ModuleList):
    def __init__(self, dim: int, mlp_dim: int) -> None:
        super().__init__()
        self.append(Linear(dim, mlp_dim * 2, bias=False))
        self.append(nn.Module())
        self.append(Linear(mlp_dim, dim, bias=False))

    def forward(self, x: Tensor) -> Tensor:
        if (
            self[0].is_nvfp4()
            and self[2].is_nvfp4()
            and not hasattr(self[0], "lora_a")
            and not hasattr(self[2], "lora_b")
        ):
            w1, w3 = self[0].weight.view(torch.float4_e2m1fn_x2).chunk(2, dim=0)
            w1_sf, w3_sf = self[0].weight_scale.chunk(2, dim=0)
            xs_2 = self[0].input_scale
            ws_2 = self[0].weight_scale_2

            B, L, C = x.shape
            xq, xs = quantize_nvfp4_triton(x.view(B * L, C), xs_2)
            xq, xs = sm120_gated_gemm_nvfp4.mm(xq, xs, xs_2, w1, w1_sf, ws_2, w3, w3_sf, ws_2, self[2].input_scale)
            out = nvfp4_mm(xq, xs, self[2].input_scale, self[2].weight, self[2].weight_scale, self[2].weight_scale_2)
            out = out.unflatten(0, (B, L))

        else:
            up, gate = self[0](x).chunk(2, dim=-1)
            out = self[2](F.silu(up) * gate)

        return out


# compared to Flux.1
# - per-layer modulation projection is replaced with a single model-wide projection.
# - no bias.
# - gelu is replaced with swiglu
class DoubleStreamBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, eps: float = 1e-6) -> None:
        super().__init__()
        self.head_dim = 128
        self.eps = eps
        mlp_dim = int(dim * mlp_ratio)
        self.attn_impl = "pt"

        self.img_attn = nn.Module()
        self.txt_attn = nn.Module()

        for m in [self.img_attn, self.txt_attn]:
            m.qkv = Linear(dim, dim * 3, bias=False)
            m.proj = Linear(dim, dim, bias=False)
            m.q_norm = nn.Parameter(torch.empty(self.head_dim))
            m.k_norm = nn.Parameter(torch.empty(self.head_dim))

        self.img_mlp = MLP(dim, mlp_dim)
        self.txt_mlp = MLP(dim, mlp_dim)

        remap_pairs = [
            ("img_attn.norm.query_norm.scale", "img_attn.q_norm"),
            ("img_attn.norm.key_norm.scale", "img_attn.k_norm"),
            ("txt_attn.norm.query_norm.scale", "txt_attn.q_norm"),
            ("txt_attn.norm.key_norm.scale", "txt_attn.k_norm"),
        ]
        self.register_load_state_dict_pre_hook(create_name_map_hook(remap_pairs))

    def forward(
        self,
        img: Tensor,
        img_res: Tensor | None,
        txt: Tensor,
        txt_res: Tensor | None,
        pe: Tensor,
        mod_img: tuple[Tensor, ...],
        mod_txt: tuple[Tensor, ...],
    ) -> tuple[Tensor, Tensor]:
        B, Limg, _ = img.shape
        _, Ltxt, _ = txt.shape

        img_shift1, img_scale1, img_gate1, img_shift2, img_scale2, img_gate2 = mod_img
        txt_shift1, txt_scale1, txt_gate1, txt_shift2, txt_scale2, txt_gate2 = mod_txt

        img_res, img = modulate(img, img_shift1, img_scale1, img_res, img_gate2)
        txt_res, txt = modulate(txt, txt_shift1, txt_scale1, txt_res, txt_gate2)
        img_q, img_k, img_v = self.img_attn.qkv(img_res).unflatten(2, (-1, self.head_dim)).chunk(3, dim=2)
        txt_q, txt_k, txt_v = self.txt_attn.qkv(txt_res).unflatten(2, (-1, self.head_dim)).chunk(3, dim=2)

        # pre-allocate buffer to avoid torch.cat()
        qk_dtype = torch.float8_e4m3fn if self.attn_impl == "qk-fp8" else img.dtype
        q = img.new_empty(B, Ltxt + Limg, *img_q.shape[2:], dtype=qk_dtype)
        apply_rope(txt_q, pe[:Ltxt], self.txt_attn.q_norm, self.eps, out=q[:, :Ltxt])
        apply_rope(img_q, pe[Ltxt:], self.img_attn.q_norm, self.eps, out=q[:, Ltxt:])

        k = torch.empty_like(q)
        apply_rope(txt_k, pe[:Ltxt], self.txt_attn.k_norm, self.eps, out=k[:, :Ltxt])
        apply_rope(img_k, pe[Ltxt:], self.img_attn.k_norm, self.eps, out=k[:, Ltxt:])

        # TODO: remove cat?
        v = torch.cat((txt_v, img_v), dim=1)
        attn = dispatch_attn(q, k, v, impl=self.attn_impl).flatten(2)
        txt_attn, img_attn = attn.split([txt.shape[1], img.shape[1]], dim=1)

        img_res, img = modulate(img, img_shift2, img_scale2, self.img_attn.proj(img_attn), img_gate1)
        txt_res, txt = modulate(txt, txt_shift2, txt_scale2, self.txt_attn.proj(txt_attn), txt_gate1)
        img_res = self.img_mlp(img_res)
        txt_res = self.txt_mlp(txt_res)
        return img_res, img, txt_res, txt


class SingleStreamBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, eps: float = 1e-6) -> None:
        super().__init__()
        self.head_dim = 128
        self.dim = dim
        self.mlp_dim = int(dim * mlp_ratio)
        self.eps = eps
        self.attn_impl = "pt"

        self.linear1 = Linear(dim, dim * 3 + self.mlp_dim * 2, bias=False)  # qkv and mlp_in
        self.linear2 = Linear(dim + self.mlp_dim, dim, bias=False)  # proj and mlp_out

        self.q_norm = nn.Parameter(torch.empty(self.head_dim))
        self.k_norm = nn.Parameter(torch.empty(self.head_dim))

        remap_pairs = [
            ("norm.query_norm.scale", "q_norm"),
            ("norm.key_norm.scale", "k_norm"),
        ]
        self.register_load_state_dict_pre_hook(create_name_map_hook(remap_pairs))

    def forward(self, x: Tensor, res: Tensor | None, pe: Tensor, mod: tuple[Tensor, ...]) -> Tensor:
        # TODO: fuse quantize with modulate
        shift, scale, gate = mod
        x_mod, x = modulate(x, shift, scale, res, gate)

        if self.linear1.is_nvfp4():
            xs_2 = self.linear1.input_scale
            ws_2 = self.linear1.weight_scale_2
            split_sizes = [self.dim * 3, self.mlp_dim, self.mlp_dim]
            wa, w1, w3 = self.linear1.weight.view(torch.float4_e2m1fn_x2).split_with_sizes(split_sizes, dim=0)
            wa_sf, w1_sf, w3_sf = self.linear1.weight_scale.split_with_sizes(split_sizes, dim=0)

            B, L, C = x_mod.shape
            xq, xs = quantize_nvfp4_triton(x_mod.view(B * L, C), xs_2)

            qkv = nvfp4_mm(xq, xs, xs_2, wa, wa_sf, ws_2).unflatten(0, (B, L))
            mlp = sm120_gated_gemm_nvfp4.mm(xq, xs, xs_2, w1, w1_sf, ws_2, w3, w3_sf, ws_2).unflatten(0, (B, L))

        else:
            qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.dim, self.mlp_dim * 2], dim=-1)
            up, gate = mlp.chunk(2, dim=-1)
            mlp = F.silu(up) * gate

        qk_dtype = torch.float8_e4m3fn if self.attn_impl == "qk-fp8" else x.dtype
        q, k, v = qkv.unflatten(2, (-1, self.head_dim)).chunk(3, dim=2)
        q = apply_rope(q, pe, self.q_norm, self.eps, out_dtype=qk_dtype)
        k = apply_rope(k, pe, self.k_norm, self.eps, out_dtype=qk_dtype)
        attn = dispatch_attn(q, k, v, impl=self.attn_impl).flatten(2)

        # TODO: pre-allocate attn+mlp buffer
        return self.linear2(torch.cat([attn, mlp], 2)), x


# default is klein-4B
class Flux2Config(NamedTuple):
    img_dim: int = 128
    txt_dim: int = 7680
    dim: int = 3072
    mlp_ratio: float = 3.0
    num_double_blocks: int = 5
    num_single_blocks: int = 20
    patch_size: int = 2
    guidance_embed: bool = False


class Flux2(nn.Module):
    def __init__(self, cfg: Flux2Config = Flux2Config()) -> None:
        super().__init__()
        self.cfg = cfg

        # input projections
        self.img_in = Linear(cfg.img_dim, cfg.dim, bias=False)
        self.txt_in = Linear(cfg.txt_dim, cfg.dim, bias=False)
        self.time_in = MLPEmbedder(256, cfg.dim, bias=False)
        if cfg.guidance_embed:
            self.guidance_in = MLPEmbedder(256, cfg.dim, bias=False)

        # 4D rope
        self.pos_embed = RopeND(dims=(32, 32, 32, 32), max_lens=(512, 512, 512, 512), theta=2e3)

        self.double_stream_modulation_img = Modulation(cfg.dim, double=True, bias=False)
        self.double_stream_modulation_txt = Modulation(cfg.dim, double=True, bias=False)
        self.single_stream_modulation = Modulation(cfg.dim, double=False, bias=False)

        self.double_blocks = nn.ModuleList(
            [DoubleStreamBlock(cfg.dim, cfg.mlp_ratio) for _ in range(cfg.num_double_blocks)]
        )
        self.single_blocks = nn.ModuleList(
            [SingleStreamBlock(cfg.dim, cfg.mlp_ratio) for _ in range(cfg.num_single_blocks)]
        )
        self.final_layer = LastLayer(cfg.dim, 1, cfg.img_dim, bias=False)

    # 4D RoPE used in Flux.2
    # main image:         https://github.com/black-forest-labs/flux2/blob/b56ac614/src/flux2/sampling.py#L93
    # conditioned images: https://github.com/black-forest-labs/flux2/blob/b56ac614/src/flux2/sampling.py#L52
    # conditioned text:   https://github.com/black-forest-labs/flux2/blob/b56ac614/src/flux2/sampling.py#L141
    # - time: main image is at t=0, conditioning images are at t=10, 20, ...
    # - height: all text embeds stay at pos=0
    # - width: all text embeds stay at pos=0
    # - text length
    def make_img_rope(self, H: int, W: int, t: int = 0) -> Tensor:
        return self.pos_embed.create((t, 0, 0, 0), (1, H, W, 1))

    def make_txt_rope(self, L: int):
        return self.pos_embed.create((0, 0, 0, 0), (1, 1, 1, L))

    def forward(
        self,
        img: Tensor,
        time: Tensor,
        txt: Tensor,
        rope: Tensor,
        guidance: Tensor | None = None,
    ) -> Tensor:
        B, Limg, _ = img.shape
        _, Ltxt, _ = txt.shape

        img = self.img_in(img.to(self.img_in.weight.dtype))
        txt = self.txt_in(txt)
        vec = self.time_in(timestep_embedding(time, 256).to(img.dtype))

        if guidance is not None:  # allow no guidance_embed
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        mod_img = self.double_stream_modulation_img(vec)
        mod_txt = self.double_stream_modulation_txt(vec)
        img_res = txt_res = None
        for block in self.double_blocks:
            img_res, img, txt_res, txt = block(img, img_res, txt, txt_res, rope, mod_img, mod_txt)

        img = torch.addcmul(img, mod_img[-1], img_res)
        txt = torch.addcmul(txt, mod_txt[-1], txt_res)
        joint = torch.cat([txt, img], dim=1)

        mod = self.single_stream_modulation(vec)
        res = None
        for block in self.single_blocks:
            res, joint = block(joint, res, rope, mod)
        img = torch.addcmul(joint[:, Ltxt:], mod[-1], res[:, Ltxt:])

        return self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)


def _load_flux2(repo_id: str, filename: str):
    state_dict = load_hf_state_dict(repo_id, filename)

    num_double_blocks = 0
    num_single_blocks = 0

    for key in state_dict.keys():
        if key.startswith("double_blocks."):
            num_double_blocks = max(num_double_blocks, int(key.split(".")[1]) + 1)
        elif key.startswith("single_blocks."):
            num_single_blocks = max(num_single_blocks, int(key.split(".")[1]) + 1)

    dim, txt_dim = state_dict["txt_in.weight"].shape
    guidance_embed = "guidance_in.in_layer.weight" in state_dict

    cfg = Flux2Config(
        txt_dim=txt_dim,
        dim=dim,
        num_double_blocks=num_double_blocks,
        num_single_blocks=num_single_blocks,
        guidance_embed=guidance_embed,
    )
    with torch.device("meta"):
        model = Flux2(cfg)

    model.load_state_dict(state_dict, assign=True)
    return model


def load_flux2(name: str = "klein-4B"):
    repo_id, filename = {
        "dev": ("black-forest-labs/FLUX.2-dev", "flux2-dev.safetensors"),
        "klein-4B": ("black-forest-labs/FLUX.2-klein-4B", "flux-2-klein-4b.safetensors"),
        "klein-4B-fp8": ("black-forest-labs/FLUX.2-klein-4b-fp8", "flux-2-klein-4b-fp8.safetensors"),
        "klein-4B-nvfp4": ("black-forest-labs/FLUX.2-klein-4b-nvfp4", "flux-2-klein-4b-nvfp4.safetensors"),
        "klein-9B": ("black-forest-labs/FLUX.2-klein-9B", "flux-2-klein-9b.safetensors"),
        "klein-9B-fp8": ("black-forest-labs/FLUX.2-klein-9b-fp8", "flux-2-klein-9b-fp8.safetensors"),
        "klein-9B-nvfp4": ("black-forest-labs/FLUX.2-klein-9b-nvfp4", "flux-2-klein-9b-nvfp4.safetensors"),
        "klein-base-4B": ("black-forest-labs/FLUX.2-klein-base-4B", "flux-2-klein-base-4b.safetensors"),
        "klein-base-9B": ("black-forest-labs/FLUX.2-klein-base-9B", "flux-2-klein-base-9b.safetensors"),
    }[name]

    return _load_flux2(repo_id, filename)
