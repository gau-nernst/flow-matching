import pytest
import torch
from PIL import Image
from torch import nn

from modelling.fibo.pipeline import FiboPipeline
from modelling.flux2.model import Flux2, Flux2Config
from modelling.flux2.pipeline import Flux2Pipeline


def randn(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)


# ref[i] is sample i's reference images: a batch with different refs per sample matches running each sample alone
def test_flux2_per_sample_ref():
    torch.manual_seed(0)
    with torch.device("cuda"):
        model = Flux2(Flux2Config(txt_dim=64, dim=256, num_double_blocks=1, num_single_blocks=1))
    for p in model.parameters():
        nn.init.normal_(p, std=1.0)
    model = model.bfloat16()
    latents, txt = randn(2, 16, 24, 128), randn(2, 20, 64)
    ref = [[randn(8, 12, 128), randn(16, 16, 128)] for _ in range(2)]
    t = torch.tensor([0.3, 0.9], device="cuda")

    with torch.no_grad():
        img, kwargs = model.prepare(latents, txt, ref)
        out = model(img, t, **kwargs)
        for i in range(2):
            img, kwargs = model.prepare(latents[i : i + 1], txt[i : i + 1], ref[i : i + 1])
            expected = model(img, t[i : i + 1], **kwargs)[0].float()
            assert (out[i].float() - expected).norm() / expected.norm() < 0.02

    with pytest.raises(AssertionError):  # a uniform batch needs the same ref sizes in each sample
        model.prepare(latents, txt, [ref[0], ref[0][:1]])


# with ref_imgs, generate() takes a single prompt. it raises before using any weights
@pytest.mark.parametrize("cls", [Flux2Pipeline, FiboPipeline])
def test_ref_imgs_single_prompt(cls):
    pipeline = cls.__new__(cls)
    pipeline.sample_defaults = dict(cfg_scale=1.0, num_steps=1)
    with pytest.raises(ValueError, match="single prompt"):
        pipeline.generate(["a", "b"], ref_imgs=[Image.new("RGB", (64, 64))])
