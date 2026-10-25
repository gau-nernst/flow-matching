# PYTHONPATH=. python tests/test_varlen.py [name filter]
# without a GPU: CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 PYTHONPATH=. python tests/test_varlen.py kernel
import os

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import modelling.attn
from modelling.flux1.model import timestep_embedding
from modelling.varlen import Varlen, pack, patchify, unpatchify, varlen_attn
from modelling.z_image.gated_norm import gated_norm
from modelling.z_image.model import ZImage, ZImageConfig

TRITON_INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


def kernel_test(fn):
    def wrapper():
        if not (TRITON_INTERPRET or torch.cuda.is_available()):
            return print("  skipped: needs CUDA, or TRITON_INTERPRET=1 on CPU")
        with torch.device("cpu" if TRITON_INTERPRET else "cuda"):
            return fn()

    return wrapper


def tiny_zimage() -> ZImage:
    torch.manual_seed(0)
    cfg = ZImageConfig(txt_dim=64, dim=256, n_refiner_layers=1, n_layers=2)
    model = ZImage(cfg)
    for p in model.parameters():
        nn.init.normal_(p, std=0.05)
    return model


# dense forward from main
def dense_reference(model: ZImage, img: Tensor, timesteps: Tensor, txt: Tensor) -> Tensor:
    def pad_tokens(x, pad_token):
        pad_len = (-x.shape[1]) % 32
        return torch.cat([x, pad_token.view(1, 1, -1).expand(x.shape[0], pad_len, -1)], dim=1)

    B, H, W, C = img.shape
    t_embeds = model.t_embedder(timestep_embedding(1 - timesteps, 256))
    p = model.cfg.patch_size
    nH, nW = H // p, W // p
    img = img.view(B, nH, p, nW, p, C).transpose(2, 3).reshape(B, nH * nW, p * p * C)

    txt = pad_tokens(model.cap_embedder(txt), model.cap_pad_token)
    txt_rope = model.pos_embed.create((1, 0, 0), (txt.shape[1], 1, 1))
    for layer in model.context_refiner:
        txt = layer(txt, None, txt_rope)

    img = pad_tokens(model.all_x_embedder["2-1"](img), model.x_pad_token)
    img_rope = model.pos_embed.create((txt.shape[1] + 1, 0, 0), (1, nH, nW))
    for layer in model.noise_refiner:
        img = layer(img, t_embeds, img_rope)

    unified = torch.cat([img, txt], dim=1)
    unified_rope = torch.cat([img_rope, txt_rope], dim=0)
    for layer in model.layers:
        unified = layer(unified, t_embeds, unified_rope)
    unified = model.all_final_layer["2-1"](unified[:, : nH * nW], t_embeds)
    return -unified.view(B, nH, nW, p, p, C).transpose(2, 3).reshape(B, H, W, C)


def masked_varlen_attn(q: Tensor, k: Tensor, v: Tensor, varlen: Varlen) -> Tensor:
    mask = varlen.seq_ids[:, None] == varlen.seq_ids[None, :]
    q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2)


# model tests run on CPU
_cuda_varlen_attn = modelling.attn.varlen_attn
modelling.attn.varlen_attn = lambda q, k, v, varlen: (_cuda_varlen_attn if q.is_cuda else masked_varlen_attn)(
    q, k, v, varlen
)


def test_varlen_attn():
    if not torch.cuda.is_available():
        print("  skipped: needs CUDA")
        return
    torch.manual_seed(0)
    lens = [5, 17, 300]
    varlen = Varlen.from_lens(lens, "cuda")
    q, k, v = torch.randn(3, 1, sum(lens), 2, 64, device="cuda", dtype=torch.bfloat16).unbind(0)
    expected = masked_varlen_attn(q.float(), k.float(), v.float(), varlen)
    torch.testing.assert_close(varlen_attn(q, k, v, varlen).float(), expected, atol=1e-2, rtol=1e-2)


def test_patchify():
    x = torch.randn(2, 8, 12, 4)
    for channels_first in (False, True):
        y = patchify(x, 2, channels_first)
        assert y.shape == (2, 24, 16)
        torch.testing.assert_close(unpatchify(y, 4, 6, 2, channels_first), x)
        torch.testing.assert_close(patchify(x[1], 2, channels_first), y[1])  # no batch dim

    # layouts from main
    torch.testing.assert_close(patchify(x, 2), x.view(2, 4, 2, 6, 2, 4).transpose(2, 3).reshape(2, 24, 16))
    nchw = x.permute(0, 3, 1, 2)
    flux1 = nchw.reshape(2, 4, 4, 2, 6, 2).permute(0, 2, 4, 1, 3, 5).reshape(2, 24, 16)
    torch.testing.assert_close(patchify(x, 2, channels_first=True), flux1)


# extra: per-sample [N, ...] inputs
def assert_packed_matches_single(model, imgs, txts, ropes, t, txt_first, extra=None, **kwargs):
    extra = extra or {}
    img, model_kwargs = pack(imgs, txts, ropes, txt_first)
    out = model(img, t, **model_kwargs, **extra, **kwargs)
    assert out.shape == img.shape, (out.shape, img.shape)
    outs = out.split([x.shape[0] for x in imgs])
    for i in range(len(imgs)):
        extra_i = {k: v[i : i + 1] for k, v in extra.items()}
        (expected,) = model(imgs[i][None], t[i : i + 1], txts[i][None], ropes[i], **extra_i, **kwargs)
        torch.testing.assert_close(outs[i], expected, atol=1e-5, rtol=1e-5)


def assert_dense_matches_single(model, img, t, txt, rope, **kwargs):
    out = model(img, t, txt, rope, **kwargs)
    assert out.shape == img.shape, (out.shape, img.shape)
    for i in range(img.shape[0]):
        kwargs_i = {k: v[i : i + 1] if isinstance(v, Tensor) and v.shape[0] > 1 else v for k, v in kwargs.items()}
        t_i = t[i : i + 1] if t.shape[0] > 1 else t
        (expected,) = model(img[i : i + 1], t_i, txt[i : i + 1], rope, **kwargs_i)
        torch.testing.assert_close(out[i], expected, atol=1e-5, rtol=1e-5)


def zimage_inputs(model: ZImage, sizes, txt_lens):
    imgs = [patchify(torch.randn(H, W, 16), 2) for H, W in sizes]
    txts = [torch.randn(L, 64) for L in txt_lens]
    ropes = [model.make_rope(H // 2, W // 2, L) for (H, W), L in zip(sizes, txt_lens)]
    return imgs, txts, ropes


@torch.enable_grad()  # use the eager paths of custom ops
def test_zimage_matches_main():
    model = tiny_zimage()
    # image token counts are multiples of 32, since main's image padding is broken
    for (H, W), L in zip([(16, 16), (16, 32), (32, 16)], [5, 40, 17]):
        img = torch.randn(1, H, W, 16)
        txt = torch.randn(1, L, 64)
        t = torch.tensor([0.3])
        out = model(patchify(img, 2), t, txt, model.make_rope(H // 2, W // 2, L))
        torch.testing.assert_close(unpatchify(out, H // 2, W // 2, 2), dense_reference(model, img, t, txt))


@torch.enable_grad()
def test_zimage_packed():
    # 6x6 = 36 image tokens need padding
    model = tiny_zimage()
    imgs, txts, ropes = zimage_inputs(model, [(12, 12), (16, 24), (8, 40)], [3, 33, 64])
    assert_packed_matches_single(model, imgs, txts, ropes, torch.tensor([0.1, 0.5, 1.0]), txt_first=False)


@torch.enable_grad()
def test_zimage_dense_batch():
    model = tiny_zimage()
    imgs, txts, ropes = zimage_inputs(model, [(12, 12)] * 2, [7, 7])
    for t in (torch.tensor([0.5]), torch.tensor([0.2, 0.7])):
        assert_dense_matches_single(model, torch.stack(imgs), t, torch.stack(txts), ropes[0])


def test_zimage_packed_grads():
    model = tiny_zimage()
    imgs, txts, ropes = zimage_inputs(model, [(16, 16), (12, 20)], [9, 45])
    t = torch.tensor([0.2, 0.7])

    img, model_kwargs = pack(imgs, txts, ropes, txt_first=False)
    out = model(img, t, **model_kwargs).split([x.shape[0] for x in imgs])
    sum(x.square().mean() for x in out).backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    model.zero_grad()
    for i in range(2):
        model(imgs[i][None], t[i : i + 1], txts[i][None], ropes[i]).square().mean().backward()
    for n, p in model.named_parameters():
        if n in grads:
            torch.testing.assert_close(grads[n], p.grad, atol=1e-5, rtol=1e-4, msg=n)


@kernel_test
def test_gated_norm_kernel_varlen():
    torch.manual_seed(0)
    D = 96
    varlen = Varlen.from_lens([3, 5, 2])
    x = torch.randn(1, 10, D)
    w = torch.randn(D)
    add = torch.randn(1, 10, D)
    gate = torch.randn(3, 1, 4 * D).chunk(4, dim=2)[1]  # non-contiguous

    # interpreter doesn't support libdevice.tanh
    for gate_type in ("plus_one",) if TRITON_INTERPRET else ("plus_one", "tanh"):
        with torch.no_grad():
            out = gated_norm(x, w, gate, add, gate_type, gate_idx=varlen.seq_ids)
        with torch.enable_grad():
            expected = gated_norm(x, w, gate, add, gate_type, gate_idx=varlen.seq_ids)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


@kernel_test
def test_gated_norm_kernel_batched():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 64)
    w = torch.randn(64)
    for gate in (torch.randn(2, 1, 64), torch.randn(1, 1, 64)):
        with torch.no_grad():
            out = gated_norm(x, w, gate)
        with torch.enable_grad():
            expected = gated_norm(x, w, gate)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


@kernel_test
def test_modulate_kernel():
    from modelling.flux1.modulate import modulate

    torch.manual_seed(0)
    D = 96
    x = torch.randn(1, 10, D)
    res = torch.randn(1, 10, D)
    norm = torch.randn(D)
    shift, scale, gate = torch.randn(3, 1, 3 * D).chunk(3, dim=2)  # non-contiguous
    idx = Varlen.from_lens([3, 5, 2]).seq_ids
    for kwargs in (dict(idx=idx), dict(idx=idx, res=res, gate=gate, norm_weight=norm, rms=True)):
        with torch.no_grad():
            out = modulate(x, shift, scale, **kwargs)
        with torch.enable_grad():
            expected = modulate(x, shift, scale, **kwargs)
        for a, b in zip(out, expected):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)

    x = torch.randn(3, 4, D)
    res = torch.randn_like(x)
    with torch.no_grad():
        out = modulate(x, shift, scale, res, gate)
    with torch.enable_grad():
        expected = modulate(x, shift, scale, res, gate)
    for a, b in zip(out, expected):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def init_random(model: nn.Module) -> nn.Module:
    torch.manual_seed(0)
    for p in model.parameters():
        nn.init.normal_(p, std=0.05)
    return model


def tiny_flux2():
    from modelling.flux2.model import Flux2, Flux2Config

    cfg = Flux2Config(img_dim=32, txt_dim=64, dim=256, num_double_blocks=1, num_single_blocks=1)
    return init_random(Flux2(cfg))


def flux2_inputs(model, sizes, txt_lens):
    imgs = [torch.randn(h * w, 32) for h, w in sizes]
    txts = [torch.randn(L, 64) for L in txt_lens]
    ropes = [torch.cat([model.make_txt_rope(L), model.make_img_rope(h, w)]) for (h, w), L in zip(sizes, txt_lens)]
    return imgs, txts, ropes


@torch.enable_grad()
def test_flux2():
    model = tiny_flux2()
    imgs, txts, ropes = flux2_inputs(model, [(8, 8), (6, 10), (12, 4)], [16, 16, 16])
    t = torch.tensor([0.2, 0.5, 0.8])
    assert_packed_matches_single(model, imgs, txts, ropes, t, txt_first=True)
    img = torch.stack([imgs[0], imgs[0] * 0.5])
    assert_dense_matches_single(model, img, torch.tensor([0.3]), torch.stack(txts[:2]), ropes[0])


def tiny_flux1():
    from modelling.flux1.model import Flux1, Flux1Config

    cfg = Flux1Config(img_dim=16, txt_dim=64, vec_dim=32, dim=256, num_double_blocks=1, num_single_blocks=1)
    return init_random(Flux1(cfg))


@torch.enable_grad()
def test_flux1():
    model = tiny_flux1()
    sizes = [(8, 8), (6, 10), (4, 12)]
    imgs = [torch.randn(h * w, 16) for h, w in sizes]
    txts = [torch.randn(10, 64) for _ in sizes]
    ropes = [model.make_rope(h, w, 10) for h, w in sizes]
    vec = torch.randn(3, 32)
    t = torch.tensor([0.2, 0.5, 0.8])
    guidance = torch.tensor([3.5])
    assert_packed_matches_single(model, imgs, txts, ropes, t, txt_first=True, extra=dict(vec=vec), guidance=guidance)
    img = torch.stack([imgs[0], imgs[0]])
    assert_dense_matches_single(model, img, t[:2], torch.stack(txts[:2]), ropes[0], vec=vec[:2], guidance=guidance)


def tiny_moss():
    from modelling.moss_image.model import MOSSImage, MOSSImageConfig

    cfg = MOSSImageConfig(dim=256, ffn_dim=512, num_layers=2, text_dim=64, latent_channels=32)
    return init_random(MOSSImage(cfg))


@torch.enable_grad()
def test_moss():
    model = tiny_moss()
    sizes = [(8, 8), (6, 10), (4, 12)]
    txt_lens = [5, 23, 12]
    imgs = [torch.randn(h * w, 32) for h, w in sizes]
    txts = [torch.randn(L, 64) for L in txt_lens]
    ropes = [model.make_rope(h, w, L) for (h, w), L in zip(sizes, txt_lens)]
    t = torch.tensor([0.2, 0.5, 0.8])
    for skip_layers in ((), (1,)):
        assert_packed_matches_single(model, imgs, txts, ropes, t, txt_first=False, skip_layers=skip_layers)
    txt = torch.stack([txts[0], txts[0] * 0.5])
    assert_dense_matches_single(model, torch.stack([imgs[0], imgs[0]]), torch.tensor([0.4]), txt, ropes[0])


@torch.enable_grad()
def test_compute_loss_multires():
    from time_sampler import Uniform
    from train_utils import compute_loss

    model = tiny_zimage()
    latents, txts, ropes = zimage_inputs(model, [(16, 16), (12, 20)], [9, 45])

    torch.manual_seed(1)
    loss = compute_loss(model, latents, Uniform(), pack(latents, txts, ropes, txt_first=False)[1])
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    # same random draws, one sample at a time
    model.zero_grad()
    torch.manual_seed(1)
    t = Uniform()(2, device="cpu")
    noise = [torch.randn_like(x) for x in latents]
    losses = []
    for i in range(2):
        (v,) = model(latents[i].lerp(noise[i], t[i])[None], t[i : i + 1], txts[i][None], ropes[i])
        losses.append(F.mse_loss(noise[i] - latents[i], v))
    expected = torch.stack(losses).mean()
    expected.backward()
    torch.testing.assert_close(loss, expected)
    for n, p in model.named_parameters():
        if n in grads:
            torch.testing.assert_close(grads[n], p.grad, atol=1e-5, rtol=1e-4, msg=n)


if __name__ == "__main__":
    import sys

    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and (len(sys.argv) == 1 or sys.argv[1] in name):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                failed += 1
                print(f"FAIL {name}: {type(e).__name__}: {str(e)[:500]}")
    sys.exit(1 if failed else 0)
