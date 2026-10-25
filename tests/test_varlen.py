# CPU tests for varlen (packed) batching. run with pytest, or directly:
#   CUDA_VISIBLE_DEVICES= PYTHONPATH=. python tests/test_varlen.py [name filter]
# the Triton kernel tests need the interpreter on CPU: TRITON_INTERPRET=1 ... tests/test_varlen.py kernel
import os

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from modelling.flux1.model import timestep_embedding
import modelling.attn
from modelling.varlen import Varlen, varlen_attn
from modelling.z_image.gated_norm import gated_norm
from modelling.z_image.model import ZImage, ZImageConfig

TRITON_INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


def tiny_zimage() -> ZImage:
    torch.manual_seed(0)
    cfg = ZImageConfig(txt_dim=64, dim=256, n_refiner_layers=1, n_layers=2)
    model = ZImage(cfg)
    for p in model.parameters():
        nn.init.normal_(p, std=0.05)
    return model


# dense forward from main (before varlen), for one sample at a time
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
    # reference: block-diagonal mask, so tokens only attend within their sample
    mask = varlen.seq_ids[:, None] == varlen.seq_ids[None, :]
    q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2)


# varlen_attn is CUDA only. model tests on CPU use the masked reference instead
if not torch.cuda.is_available():
    modelling.attn.varlen_attn = masked_varlen_attn


def test_varlen_attn():
    if not torch.cuda.is_available():
        print("  skipped: needs CUDA")
        return
    torch.manual_seed(0)
    lens = [5, 17, 300]
    varlen = Varlen.from_lens(lens, "cuda")
    q, k, v = torch.randn(3, 1, sum(lens), 2, 64, device="cuda", dtype=torch.bfloat16).unbind(0)
    torch.testing.assert_close(varlen_attn(q, k, v, varlen), masked_varlen_attn(q, k, v, varlen))


@torch.enable_grad()  # use the eager paths of custom ops
def test_zimage_varlen_matches_dense():
    model = tiny_zimage()
    # image token counts are multiples of 32, since dense padding of image RoPE was broken otherwise
    sizes = [(16, 16), (16, 32), (32, 16)]
    txt_lens = [5, 40, 17]
    imgs = [torch.randn(H, W, 16) for H, W in sizes]
    txts = [torch.randn(L, 64) for L in txt_lens]
    t = torch.tensor([0.3, 0.6, 0.9])

    outs = model(imgs, t, txts)
    for i in range(len(imgs)):
        expected = dense_reference(model, imgs[i][None], t[i : i + 1], txts[i][None])[0]
        torch.testing.assert_close(outs[i], expected, atol=1e-5, rtol=1e-5)


@torch.enable_grad()
def test_zimage_varlen_matches_single():
    # includes image token counts that need padding (6x6 = 36 tokens)
    model = tiny_zimage()
    sizes = [(12, 12), (16, 24), (8, 40)]
    txt_lens = [3, 33, 64]
    imgs = [torch.randn(H, W, 16) for H, W in sizes]
    txts = [torch.randn(L, 64) for L in txt_lens]
    t = torch.tensor([0.1, 0.5, 1.0])

    outs = model(imgs, t, txts)
    for i in range(len(imgs)):
        (expected,) = model([imgs[i]], t[i : i + 1], [txts[i]])
        assert outs[i].shape == imgs[i].shape
        torch.testing.assert_close(outs[i], expected, atol=1e-5, rtol=1e-5)


@torch.enable_grad()
def test_zimage_batched_tensor_input():
    # batched tensors (same image size) with a shared t of shape [1], like sampling
    model = tiny_zimage()
    img = torch.randn(2, 16, 16, 16)
    txts = [torch.randn(7, 64), torch.randn(50, 64)]
    t = torch.tensor([0.5])

    out = model(img, t, txts)
    assert isinstance(out, Tensor) and out.shape == img.shape
    for i in range(2):
        (expected,) = model([img[i]], t, [txts[i]])
        torch.testing.assert_close(out[i], expected, atol=1e-5, rtol=1e-5)


def test_zimage_varlen_grads():
    # grads of a packed batch == sum of per-sample grads
    model = tiny_zimage()
    sizes = [(16, 16), (12, 20)]
    imgs = [torch.randn(H, W, 16) for H, W in sizes]
    txts = [torch.randn(L, 64) for L in (9, 45)]
    t = torch.tensor([0.2, 0.7])

    sum(x.square().mean() for x in model(imgs, t, txts)).backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    model.zero_grad()
    for i in range(2):
        model([imgs[i]], t[i : i + 1], [txts[i]])[0].square().mean().backward()
    for n, p in model.named_parameters():
        if n in grads:
            torch.testing.assert_close(grads[n], p.grad, atol=1e-5, rtol=1e-4, msg=n)


def test_gated_norm_kernel_varlen():
    if not (TRITON_INTERPRET or torch.cuda.is_available()):
        return print("  skipped: needs TRITON_INTERPRET=1 on CPU")
    torch.manual_seed(0)
    D = 96
    varlen = Varlen.from_lens([3, 5, 2])
    x = torch.randn(1, 10, D)
    w = torch.randn(D)
    add = torch.randn(1, 10, D)
    gate = torch.randn(3, 1, 4 * D).chunk(4, dim=2)[1]  # non-contiguous, like the modulation chunks

    # the Triton interpreter doesn't support libdevice.tanh
    for gate_type in ("plus_one",) if TRITON_INTERPRET else ("plus_one", "tanh"):
        with torch.no_grad():
            out = gated_norm(x, w, gate, add, gate_type, gate_idx=varlen.seq_ids)
        with torch.enable_grad():
            expected = gated_norm(x, w, gate, add, gate_type, gate_idx=varlen.seq_ids)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_gated_norm_kernel_batched():
    # existing [B, L, D] usage, with per-batch and broadcast gates
    if not (TRITON_INTERPRET or torch.cuda.is_available()):
        return print("  skipped: needs TRITON_INTERPRET=1 on CPU")
    torch.manual_seed(0)
    x = torch.randn(2, 4, 64)
    w = torch.randn(64)
    for gate in (torch.randn(2, 1, 64), torch.randn(1, 1, 64)):
        with torch.no_grad():
            out = gated_norm(x, w, gate)
        with torch.enable_grad():
            expected = gated_norm(x, w, gate)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_modulate_kernel():
    if not (TRITON_INTERPRET or torch.cuda.is_available()):
        return print("  skipped: needs TRITON_INTERPRET=1 on CPU")
    from modelling.flux1.modulate import modulate

    torch.manual_seed(0)
    D = 96
    x = torch.randn(1, 10, D)
    res = torch.randn(1, 10, D)
    norm = torch.randn(D)
    shift, scale, gate = torch.randn(3, 1, 3 * D).chunk(3, dim=2)  # non-contiguous, like Modulation chunks
    idx = Varlen.from_lens([3, 5, 2]).seq_ids
    for kwargs in (dict(idx=idx), dict(idx=idx, res=res, gate=gate, norm_weight=norm, rms=True)):
        with torch.no_grad():
            out = modulate(x, shift, scale, **kwargs)
        with torch.enable_grad():
            expected = modulate(x, shift, scale, **kwargs)
        for a, b in zip(out, expected):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)

    # existing [B, L, D] usage
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
def test_flux2_varlen_matches_single():
    model = tiny_flux2()
    imgs, txts, ropes = flux2_inputs(model, [(8, 8), (6, 10), (12, 4)], [16, 16, 16])
    t = torch.tensor([0.2, 0.5, 0.8])
    outs = model(imgs, t, txts, ropes)
    for i in range(3):
        (expected,) = model([imgs[i]], t[i : i + 1], [txts[i]], [ropes[i]])
        torch.testing.assert_close(outs[i], expected, atol=1e-5, rtol=1e-5)

    # batched tensor input with a shared rope
    img = torch.stack([imgs[0], imgs[0] * 0.5])
    txt = torch.stack([txts[0], txts[1]])
    out = model(img, torch.tensor([0.3]), txt, ropes[0])
    for i in range(2):
        (expected,) = model([img[i]], torch.tensor([0.3]), [txt[i]], [ropes[0]])
        torch.testing.assert_close(out[i], expected, atol=1e-5, rtol=1e-5)


def tiny_flux1():
    from modelling.flux1.model import Flux1, Flux1Config

    cfg = Flux1Config(img_dim=16, txt_dim=64, vec_dim=32, dim=256, num_double_blocks=1, num_single_blocks=1)
    return init_random(Flux1(cfg))


@torch.enable_grad()
def test_flux1_varlen_matches_single():
    model = tiny_flux1()
    imgs = [torch.randn(4, H, W) for H, W in [(16, 16), (12, 20), (8, 24)]]
    txts = [torch.randn(L, 64) for L in (10, 10, 10)]
    vec = torch.randn(3, 32)
    t = torch.tensor([0.2, 0.5, 0.8])
    guidance = torch.tensor([3.5])
    outs = model(imgs, t, txts, vec, guidance)
    for i in range(3):
        (expected,) = model([imgs[i]], t[i : i + 1], [txts[i]], vec[i : i + 1], guidance)
        assert expected.shape == imgs[i].shape
        torch.testing.assert_close(outs[i], expected, atol=1e-5, rtol=1e-5)

    out = model(torch.stack([imgs[0], imgs[0]]), t[:2], torch.stack(txts[:2]), vec[:2], guidance)
    torch.testing.assert_close(out[1], model([imgs[0]], t[1:2], [txts[1]], vec[1:2], guidance)[0])


def tiny_moss():
    from modelling.moss_image.model import MOSSImage, MOSSImageConfig

    cfg = MOSSImageConfig(dim=256, ffn_dim=512, num_layers=2, text_dim=64, latent_channels=32)
    return init_random(MOSSImage(cfg))


@torch.enable_grad()
def test_moss_varlen_matches_single():
    model = tiny_moss()
    imgs = [torch.randn(H, W, 32) for H, W in [(8, 8), (6, 10), (4, 12)]]
    txts = [torch.randn(L, 64) for L in (5, 23, 12)]
    t = torch.tensor([0.2, 0.5, 0.8])
    for skip_layers in ((), (1,)):
        outs = model(imgs, t, txts, skip_layers)
        for i in range(3):
            (expected,) = model([imgs[i]], t[i : i + 1], [txts[i]], skip_layers)
            torch.testing.assert_close(outs[i], expected, atol=1e-5, rtol=1e-5)

    out = model(torch.stack([imgs[0], imgs[0]]), torch.tensor([0.4]), [txts[0], txts[1]])
    torch.testing.assert_close(out[1], model([imgs[0]], torch.tensor([0.4]), [txts[1]])[0])


@torch.enable_grad()
def test_compute_loss_multires():
    # loss and grads on a mixed-resolution batch == mean of per-sample losses
    from time_sampler import Uniform
    from train_utils import compute_loss

    model = tiny_zimage()
    latents = [torch.randn(H, W, 16) for H, W in [(16, 16), (12, 20)]]
    txt = [torch.randn(L, 64) for L in (9, 45)]

    torch.manual_seed(1)
    loss = compute_loss(model, latents, Uniform(), dict(txt=txt))
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    # replay the same random draws one sample at a time
    model.zero_grad()
    torch.manual_seed(1)
    t = Uniform()(2, device="cpu")
    noise = [torch.randn_like(x) for x in latents]
    losses = []
    for i in range(2):
        (v,) = model([latents[i].lerp(noise[i], t[i])], t[i : i + 1], [txt[i]])
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
