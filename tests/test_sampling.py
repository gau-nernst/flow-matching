import torch
from torch import nn

from modelling.fibo.model import Fibo, FiboConfig
from modelling.fibo.pipeline import fibo_generate, fibo_timesteps


def randn(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)


# prompts of different lengths are one varlen batch, while the shared negative prompt is a uniform batch.
# each sample matches sampling it alone. some have text images (base_txt), some don't
def test_fibo_batched_prompts():
    torch.manual_seed(0)
    with torch.device("cuda"):
        model = Fibo(FiboConfig(dim=256, num_layers=8, txt_dim=64, img_dim=16))
    for p in model.parameters():
        nn.init.normal_(p, std=0.1)
    model = model.bfloat16()
    timesteps = fibo_timesteps(6, 16 * 24)
    noise = torch.randn(3, 16, 24, 16, device="cuda")
    txts = [randn(6, L, 64) for L in (37, 300, 37)]
    neg_txt = randn(6, 20, 64)
    base_txts = [None, randn(6, 297, 64), None]

    out = fibo_generate(model, noise, timesteps, txts, neg_txt, base_txt=base_txts, cfg_scale=3.0)
    for i in range(3):
        base = None if base_txts[i] is None else base_txts[i : i + 1]
        expected = fibo_generate(
            model, noise[i : i + 1], timesteps, txts[i : i + 1], neg_txt, base_txt=base, cfg_scale=3.0
        )
        # ~2% from bf16 (a uniform batch of 3 is off by the same). another prompt is ~100% off
        assert (out[i] - expected[0]).norm() / expected[0].norm() < 0.05
