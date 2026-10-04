from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from torchvision.transforms import v2

from modelling import (
    Flux2,
    Flux2Qwen3TextEncoder,
    load_flux2,
    load_vae,
    load_zimage,
)
from modelling.linear import Linear
from time_sampler import TimeSampler


def setup_model(model_name: str, lora: int, use_compile: bool):
    if model_name.startswith("flux2-"):
        model = load_flux2(model_name.removeprefix("flux2-"))
        layers = list(model.double_blocks) + list(model.single_blocks)

        ae = load_vae("flux2")

        if model_name.startswith("flux2-klein-4B"):
            text_id = "Qwen/Qwen3-4B-FP8"
        elif model_name.startswith("flux2-klein-9B"):
            text_id = "Qwen/Qwen3-8B-FP8"
        else:
            raise ValueError
        text_embedder = Flux2Qwen3TextEncoder(text_id)

    elif model_name.startswith("z-image-"):
        model = load_zimage(model_name.removeprefix("z-image-"))
        layers = list(model.layers)

        ae = load_vae("flux1")
        # TODO: text encoder
        raise ValueError

    else:
        raise ValueError(f"Unsupported {model_name=}")

    model.cuda().train().requires_grad_(False)
    ae.bfloat16().eval().cuda()
    text_embedder.cuda()

    for layer in layers:
        with torch.device("cuda"):
            for m in layer.modules():
                if isinstance(m, Linear):
                    m.init_lora(lora)

        # TODO: use selective activation checkpointing
        # https://pytorch.org/blog/activation-checkpointing-techniques/
        layer.forward = partial(checkpoint, layer.forward, use_reentrant=False)

    if use_compile:
        model.compile()

    return model, ae, text_embedder


def compute_loss(model: Flux2, latents: Tensor, time_sampler: TimeSampler, model_kwargs: dict) -> Tensor:
    bsize = latents.shape[0]
    t_vec = time_sampler(bsize, device=latents.device)
    noise = torch.randn_like(latents)
    interpolate = latents.lerp(noise, t_vec.view([-1] + [1] * (latents.ndim - 1)))

    v = model(interpolate, t_vec, **model_kwargs)

    # rectified flow loss. predict velocity from latents (t=0) to noise (t=1).
    return F.mse_loss(noise.float() - latents.float(), v.float())


def parse_img_size(img_size: str):
    out = [int(x) for x in img_size.split(",")]
    if len(out) == 1:
        out = [out[0], out[0]]
    assert len(out) == 2
    assert out[0] % 16 == 0 and out[1] % 16 == 0, out
    return tuple(out)


def random_resize(img_pil: Image.Image, min_size: int, max_size: int):
    # randomly resize while maintaining aspect ratio.
    long_edge = max(img_pil.size)
    assert long_edge >= min_size
    max_size = min(max_size, long_edge)

    target_long_edge = torch.randint(min_size, max_size + 1, size=()).item()
    scale = target_long_edge / long_edge
    target_height = round(img_pil.height * scale)
    target_width = round(img_pil.width * scale)
    img_pil = img_pil.resize((target_width, target_height), Image.Resampling.BICUBIC)

    # slightly crop the image so that each side is divisible by 16.
    # to reduce fragmentation, make it in <factor> increment.
    factor = 64
    height = target_height // factor * factor
    width = target_width // factor * factor
    img_pt = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1)  # HWC->CHW
    img_pt = v2.RandomCrop((height, width))(img_pt).permute(1, 2, 0)  # CHW->HWC
    return img_pt


class EMA:
    def __init__(self, model: nn.Module, beta: float = 0.999, num_warmup: int = 500):
        self.model = model
        self.ema_params = {
            name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad
        }
        self.beta = beta
        self.num_warmup = num_warmup

    @torch.no_grad()
    def update(self, step: int):
        if step < self.num_warmup:
            return

        ema_params, online_params = [], []
        for name, param in self.model.named_parameters():
            if name in self.ema_params:
                ema_params.append(self.ema_params[name])
                online_params.append(param)

        if step == self.num_warmup:
            torch._foreach_copy_(ema_params, online_params)
        else:
            torch._foreach_lerp_(ema_params, online_params, 1 - self.beta)

    def swap_params(self):
        for name, param in self.model.named_parameters():
            if name in self.ema_params:
                ema_p = self.ema_params[name]
                param.data, ema_p.data = ema_p.data, param.data

    def state_dict(self):
        return dict(self.ema_params)  # shallow copy

    def load_state_dict(self, state_dict: dict[str, Tensor]):
        state_dict = dict(state_dict)  # shallow copy
        for name, param in self.ema_params.items():
            param.copy_(state_dict.pop(name))
        assert len(state_dict) == 0
