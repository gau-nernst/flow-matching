import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from modelling.fibo.model import Fibo, FiboConfig
from modelling.flux2.model import Flux2, Flux2Config
from modelling.moss_image.model import MOSSImage, MOSSImageConfig
from modelling.varlen import VarlenMetadata
from modelling.z_image.model import ZImage, ZImageConfig
from time_sampler import LogitNormal
from train_lora import ImageDataset
from train_utils import compute_loss

MODELS = {
    "flux2": (Flux2, Flux2Config(txt_dim=64, dim=256, num_double_blocks=1, num_single_blocks=1), 128),
    "moss": (MOSSImage, MOSSImageConfig(dim=256, ffn_dim=512, num_layers=2, text_dim=64), 128),
    "zimage": (ZImage, ZImageConfig(dim=256, n_layers=2, n_refiner_layers=1, txt_dim=64), 16),
}
SIZES = [(16, 24), (32, 16), (24, 24)]
TXT_LENS = [37, 80, 12]


def randn(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)


@pytest.fixture(params=MODELS)
def setup(request):
    cls, cfg, C = MODELS[request.param]
    torch.manual_seed(0)
    with torch.device("cuda"):
        model = cls(cfg)
    for p in model.parameters():
        nn.init.normal_(p, std=1.0)  # small weights hide token mixing: attention barely affects the output
    return model.to(torch.bfloat16), C


def test_concat():
    a = VarlenMetadata.from_lens([2, 3], "cuda")
    b = VarlenMetadata.from_lens([1, 2], "cuda")
    joint = a.concat(b)
    assert joint.perm.tolist() == [0, 1, 5, 2, 3, 4, 6, 7]
    assert joint.cu_seqlens.tolist() == [0, 3, 8]
    assert joint.seq_ids.tolist() == [0, 0, 1, 1, 1, 0, 1, 1]


@pytest.mark.parametrize("grad", [False, True])  # Triton kernels and their reference paths
def test_varlen_matches_uniform(setup, grad):
    model, C = setup
    latents = [randn(H, W, C) for H, W in SIZES]
    txts = [randn(L, 64) for L in TXT_LENS]
    t = torch.tensor([0.3, 0.6, 0.9], device="cuda")

    with torch.set_grad_enabled(grad):
        img, kwargs = model.prepare(latents, txts)
        assert "img_varlen" in kwargs
        outs = model.unpatchify(model(img, t, **kwargs), SIZES)
        for i, size in enumerate(SIZES):
            img, kwargs = model.prepare(latents[i][None], txts[i][None])
            expected = model.unpatchify(model(img, t[i : i + 1], **kwargs), size)[0]
            # clean is below 0.6%, mixing samples is above 5%
            expected = expected.float()
            assert (outs[i].float() - expected).norm() / expected.norm() < 0.02


# a list is varlen even when all samples have the same shape. a tensor is uniform
def test_layout_follows_input_type(setup):
    model, C = setup
    img, kwargs = model.prepare([randn(16, 24, C)] * 2, [randn(20, 64)] * 2)
    assert img.shape[0] == 1 and "img_varlen" in kwargs
    img, kwargs = model.prepare(randn(2, 16, 24, C), randn(2, 20, 64))
    assert img.shape[0] == 2 and "img_varlen" not in kwargs


def test_mismatched_inputs_raise(setup):
    model, C = setup
    with pytest.raises(AssertionError):
        model.prepare([randn(16, 24, C)], [randn(20, 64)] * 2)


@pytest.mark.parametrize("budget", [1, 2500, 4096])  # 1: every image is over budget and batched alone
def test_batch_tokens(tmp_path, budget):
    sizes = [(512, 768), (1024, 1024), (640, 640), (768, 512), (512, 512)]
    for i, (W, H) in enumerate(sizes):
        Image.new("RGB", (W, H)).save(tmp_path / f"{i}.png")
    meta = pd.DataFrame(dict(filename=[f"{i}.png" for i in range(5)], prompt=list("abcde")))
    meta.to_csv(tmp_path / "meta.csv", index=False)

    ds_iter = iter(ImageDataset(tmp_path / "meta.csv", tmp_path, 512, 1024, budget))
    for _ in range(20):
        imgs, prompts = next(ds_iter)
        assert len(imgs) == len(prompts)
        assert len(imgs) == 1 or sum(x.shape[0] * x.shape[1] // 256 for x in imgs) <= budget


def test_fibo_interface():
    model = Fibo(FiboConfig(dim=256, num_layers=2, txt_dim=32, img_dim=16))
    for p in model.parameters():
        nn.init.normal_(p, std=0.1)
    latents = [torch.randn(4, 6, 16) for _ in range(2)]
    txts = [torch.randn(6, 10, 32) for _ in range(2)]

    img, kwargs = model.prepare(torch.stack(latents), torch.stack(txts))
    assert img.shape == (2, 24, 16)
    assert kwargs["rope"].shape[0] == model.num_gist_tokens(10) + 24
    compute_loss(model, latents, LogitNormal(), txts).backward()

    with pytest.raises(NotImplementedError):
        model.prepare([latents[0], torch.randn(6, 4, 16)], txts)
    with pytest.raises(NotImplementedError):
        model.prepare(latents, [txts[0], torch.randn(6, 12, 32)])
