# https://github.com/black-forest-labs/flux/blob/7e14a05e/src/flux/modules/autoencoder.py
# https://github.com/Stability-AI/sd3.5/blob/4e484e05/sd3_impls.py

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .ops import BatchNorm2d, Conv2d, GroupNorm, conv2d_triton, vae_attn
from .utils import load_hf_state_dict


class AttnBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = GroupNorm(32, dim, eps=1e-6)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)

        def hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            for name in ("q", "k", "v", "proj_out"):
                key = f"{prefix}{name}.weight"
                state_dict[key] = state_dict[key].squeeze()

        self.register_load_state_dict_pre_hook(hook)

    def forward(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape
        h = self.norm(x).view(B, H * W, C)
        h = vae_attn(self.q(h), self.k(h), self.v(h))
        h = self.proj_out(h).view(B, H, W, C)
        return x + h


class ResnetBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int | None = None) -> None:
        super().__init__()
        out_dim = out_dim or in_dim
        self.norm1 = GroupNorm(32, in_dim, eps=1e-6)
        self.norm2 = GroupNorm(32, out_dim, eps=1e-6)
        self.conv1 = Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv2 = Conv2d(out_dim, out_dim, 3, 1, 1)

        if in_dim != out_dim:
            self.nin_shortcut = Conv2d(in_dim, out_dim, 1)
        else:
            self.nin_shortcut = None

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(self.norm1(x, act="silu"))
        h = self.norm2(h, act="silu")

        if self.nin_shortcut is not None:
            return self.conv2(h) + self.nin_shortcut(x)
        else:
            return self.conv2(h, x)


# no asymmetric padding in torch conv, must do it ourselves
class Downsample(nn.Sequential):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = Conv2d(dim, dim, 3, 2, 0)

    def forward(self, x: Tensor):
        return self.conv(F.pad(x, (0, 0, 0, 1, 0, 1)))


class Upsample(nn.Sequential):
    def __init__(self, in_dim: int, out_dim: int | None = None) -> None:
        super().__init__()
        self.conv = Conv2d(in_dim, out_dim or in_dim, 3, 1, 1)

    def forward(self, x: Tensor):
        return conv2d_triton(x, self.conv.weight, self.conv.bias, padding=1, upsample=2)


class Encoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        dim: int,
        dim_mult: tuple[int, ...],
        num_res_blocks: int,
        z_dim: int,
        quant_conv: bool = False,
    ) -> None:
        super().__init__()
        self.num_resolutions = len(dim_mult)
        self.num_res_blocks = num_res_blocks
        # downsampling
        self.conv_in = Conv2d(in_dim, dim, 3, 1, 1)

        in_ch_mult = (1,) + tuple(dim_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = dim * in_ch_mult[i_level]
            block_out = dim * dim_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)

        # middle
        self.mid = nn.Sequential()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)

        # end
        self.norm_out = GroupNorm(32, block_in, eps=1e-6)
        self.conv_out = Conv2d(block_in, 2 * z_dim, 3, 1, 1)
        self.quant_conv = Conv2d(2 * z_dim, 2 * z_dim, 1) if quant_conv else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = self.mid(hs[-1])

        # end
        h = self.conv_out(self.norm_out(h, act="silu"))
        h = self.quant_conv(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int,
        dim_mult: tuple[int, ...],
        num_res_blocks: int,
        z_dim: int,
        quant_conv: bool = False,
    ) -> None:
        super().__init__()
        self.num_resolutions = len(dim_mult)
        self.num_res_blocks = num_res_blocks
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = dim * dim_mult[self.num_resolutions - 1]

        # z to block_in
        self.post_quant_conv = Conv2d(z_dim, z_dim, 1) if quant_conv else nn.Identity()
        self.conv_in = Conv2d(z_dim, block_in, 3, 1, 1)

        # middle
        self.mid = nn.Sequential()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = dim * dim_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = GroupNorm(32, block_in, eps=1e-6)
        self.conv_out = Conv2d(block_in, out_dim, 3, 1, 1)

    def forward(self, z: Tensor) -> Tensor:
        # z to block_in
        h = self.post_quant_conv(z)
        h = self.conv_in(h)

        # middle
        h = self.mid(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.conv_out(self.norm_out(h, act="silu"))
        return h


def diagonal_gaussian(z: Tensor, sample: bool) -> Tensor:
    mean, logvar = torch.chunk(z, 2, dim=-1)
    if sample:
        # SD clamps logvar to [-30,20]
        std = (0.5 * logvar).exp()
        return mean + std * torch.randn_like(mean)
    else:
        return mean


class AEConfig(NamedTuple):
    in_dim: int = 3
    dim: int = 128
    out_dim: int = 3
    dim_mult: tuple[int] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    z_dim: int = 16
    scale_shift: tuple[float, float] = (1.0, 0.0)
    quant_conv: bool = False
    use_bn: bool = False


class AutoEncoder(nn.Module):
    def __init__(self, cfg: AEConfig = AEConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.downsample = int(2 ** (len(cfg.dim_mult) - 1))

        self.encoder = Encoder(cfg.in_dim, cfg.dim, cfg.dim_mult, cfg.num_res_blocks, cfg.z_dim, cfg.quant_conv)
        self.decoder = Decoder(cfg.dim, cfg.out_dim, cfg.dim_mult, cfg.num_res_blocks, cfg.z_dim, cfg.quant_conv)

        # Flux.2
        if cfg.use_bn:
            self.bn = BatchNorm2d(4 * cfg.z_dim, eps=1e-4, affine=False)

    def encode(self, x: Tensor, sample: bool = False) -> Tensor:
        if x.dtype == torch.uint8:
            x = x.float() / 127.5 - 1

        mean_var = self.encoder(x.to(self.encoder.conv_in.weight.dtype))
        z = diagonal_gaussian(mean_var.float(), sample)

        if self.cfg.use_bn:
            # Flux.2 does patchify in AE, before BN. right now for other models, we patchify inside the model
            # we should consolidate this some how
            # patchify
            B, H, W, C = z.shape
            nH = H // 2
            nW = W // 2
            z = z.view(B, nH, 2, nW, 2, C)
            z = z.permute(0, 1, 3, 5, 2, 4)
            z = z.reshape(B, nH, nW, C * 4)

            # normalize
            z = self.bn(z)

        else:
            scale, shift = self.cfg.scale_shift
            z = (z - shift) * scale

        return z

    def decode(self, z: Tensor, uint8: bool = False) -> Tensor:
        if self.cfg.use_bn:
            # inverse normalize
            mean = self.bn.running_mean
            var = self.bn.running_var
            eps = self.bn.eps
            z = z.float() * (var + eps).sqrt() + mean

            # unpatchify
            B, nH, nW, nC = z.shape
            H = nH * 2
            W = nW * 2
            C = nC // 4
            z = z.view(B, nH, nW, C, 2, 2)
            z = z.permute(0, 1, 4, 2, 5, 3)
            z = z.reshape(B, H, W, C)

        else:
            scale, shift = self.cfg.scale_shift
            z = z.float() / scale + shift

        x = self.decoder(z.to(self.decoder.conv_in.weight.dtype))

        if uint8:
            x = x.float().add(1).mul(127.5).round().clip(0, 255).to(torch.uint8)
        return x

    def forward(self, x: Tensor, sample: bool = False) -> Tensor:
        return self.decode(self.encode(x, sample), x.dtype == torch.uint8)


def _load_vae(repo_id: str, filename: str, scale_shift: tuple[float, float] = (1.0, 0.0)):
    state_dict = load_hf_state_dict(repo_id, filename)

    cfg = AEConfig(
        z_dim=state_dict["encoder.conv_out.weight"].shape[0] // 2,
        scale_shift=scale_shift,
        quant_conv="encoder.quant_conv.weight" in state_dict,
        use_bn="bn.running_mean" in state_dict,
    )
    with torch.device("meta"):
        ae = AutoEncoder(cfg)

    ae.load_state_dict(state_dict, assign=True)
    ae.eval()
    return ae


def load_vae(name: str):
    repo_id, filename, scale_shift = dict(
        # original weight is FP32
        flux1=("black-forest-labs/FLUX.1-dev", "ae.safetensors", (0.3611, 0.1159)),
        flux2=("black-forest-labs/FLUX.2-dev", "ae.safetensors", (1.0, 0.0)),
    )[name]
    return _load_vae(repo_id, filename, scale_shift)
