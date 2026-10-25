from functools import partial

import numpy as np
import torch
import torch._functorch.config
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from torchvision.transforms import v2

from modelling import load_pipeline
from modelling.linear import Linear
from time_sampler import TimeSampler


def setup_model(
    model_name: str, lora: int, fp8: bool, use_compile: bool, mem_budget: float = 0.7, ckpt_ratio: float = 0.5
):
    pipeline = load_pipeline(model_name).cuda()
    model = pipeline.model
    model.train().requires_grad_(False)
    layers = model.lora_blocks()

    for i, layer in enumerate(layers):
        with torch.device("cuda"):
            for m in layer.modules():
                if isinstance(m, Linear):
                    if fp8:
                        m.to_fp8_1d2d()
                    m.init_lora(lora)

        # TODO: use selective activation checkpointing
        # https://pytorch.org/blog/activation-checkpointing-techniques/
        fwd = layer.forward
        if i < len(layers) * ckpt_ratio:
            fwd = partial(checkpoint, fwd, use_reentrant=False)
        if use_compile:
            fwd = torch.compile(fwd)
        layer.forward = fwd

    torch._functorch.config.activation_memory_budget = mem_budget
    torch._functorch.config.is_non_builtin_to_include = True  # without this, mem budget ignores custom ops' cost

    return pipeline


def compute_loss(model: nn.Module, latents: Tensor, time_sampler: TimeSampler, model_kwargs: dict) -> Tensor:
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
