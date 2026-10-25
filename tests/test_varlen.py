# CPU tests for varlen (packed) batching. run with pytest, or directly:
#   CUDA_VISIBLE_DEVICES= PYTHONPATH=. python tests/test_varlen.py [name filter]
# the Triton kernel tests need the interpreter on CPU: TRITON_INTERPRET=1 ... tests/test_varlen.py kernel
import os

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from modelling.flux1.model import timestep_embedding
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


def test_varlen_attn_reference():
    torch.manual_seed(0)
    lens = [5, 17, 32]
    varlen = Varlen.from_lens(lens)
    q, k, v = torch.randn(3, 1, sum(lens), 2, 64).unbind(0)
    out = varlen_attn(q, k, v, varlen)

    # block-diagonal mask
    seq = varlen.seq_ids
    mask = seq[:, None] == seq[None, :]
    expected = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), mask)
    torch.testing.assert_close(out, expected.transpose(1, 2))


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
