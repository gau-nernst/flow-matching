import pytest
import torch
from torch import nn

from modelling.moss_image.model import MOSSImage, MOSSImageConfig
from modelling.moss_image.pipeline import moss_image_generate
from modelling.z_image.model import ZImage, ZImageConfig
from modelling.z_image.pipeline import zimage_generate

SAMPLERS = {
    "zimage": (ZImage, ZImageConfig(dim=256, n_layers=2, n_refiner_layers=1, txt_dim=64), 16, zimage_generate),
    "moss": (MOSSImage, MOSSImageConfig(dim=256, ffn_dim=512, num_layers=2, text_dim=64), 128, moss_image_generate),
}


def randn(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)


# guidance with prompts of different lengths and one negative prompt. the prompt pass is varlen, so the negative
# prompt pass must be too, or they disagree on the latents' layout. each sample matches sampling it alone
@pytest.mark.parametrize("name", SAMPLERS)
def test_batched_prompts_cfg(name):
    cls, cfg, C, generate = SAMPLERS[name]
    torch.manual_seed(0)
    with torch.device("cuda"):
        model = cls(cfg)
    for p in model.parameters():
        nn.init.normal_(p, std=0.1)
    model = model.bfloat16()
    timesteps = [1.0, 0.75, 0.5, 0.25, 0.0]
    noise = torch.randn(2, 16, 24, C, device="cuda")
    txts = [randn(37, 64), randn(80, 64)]
    neg = randn(10, 64)

    out = generate(model, noise, timesteps, txts, [neg] * 2, cfg_scale=3.0)
    for i in range(2):
        expected = generate(model, noise[i : i + 1], timesteps, txts[i : i + 1], [neg], cfg_scale=3.0)[0]
        assert (out[i].float() - expected.float()).norm() / expected.float().norm() < 0.05
