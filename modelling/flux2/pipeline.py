# https://github.com/black-forest-labs/flux2/blob/b56ac614/src/flux2/text_encoder.py

import math
from collections.abc import Callable

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from ..offload import PerLayerOffloadCUDAStream
from ..qwen3 import Qwen3ForCausalLM
from ..solvers import get_solver
from ..vae import load_vae
from .model import Flux2, load_flux2


# TODO: might unify with Z-Image
class Flux2Qwen3TextEncoder(nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        self.model = Qwen3ForCausalLM.from_pretrained(model_id).model
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id)

        # TODO: we can truncate Qwen3 to 27 layers
        self.model.eval()
        self.output_indices = [9, 18, 27]

    @torch.no_grad()
    def forward(self, texts: str | list[str]) -> Tensor:
        if isinstance(texts, str):
            texts = [texts]

        texts = [
            self.tokenizer.apply_chat_template(
                [dict(role="user", content=txt)],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for txt in texts
        ]
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        device = next(self.parameters()).device
        outputs = self.model(inputs["input_ids"].to(device), output_indices=self.output_indices)
        return torch.cat(outputs, dim=-1)


class Flux2Pipeline:
    DEV_DEFAULTS = dict(guidance=4.0, cfg_scale=1.0, num_steps=50)
    KLEIN_DEFAULTS = dict(guidance=None, cfg_scale=1.0, num_steps=4)
    KLEIN_BASE_DEFAULTS = dict(guidance=None, cfg_scale=4.0, num_steps=50)

    def __init__(
        self,
        flux: Flux2 | None = None,
        text_encoder_id: str = "Qwen/Qwen3-4B-FP8",
        offload_flux: bool = False,
    ) -> None:
        self.flux = flux or load_flux2()
        self.ae = load_vae("flux2").bfloat16()
        self.text_encoder = Flux2Qwen3TextEncoder(text_encoder_id)

        self.flux_offloader = PerLayerOffloadCUDAStream(self.flux, enable=offload_flux)

    def cpu(self):
        for m in (self.ae, self.text_encoder, self.flux_offloader):
            m.cpu()
        return self

    def cuda(self):
        for m in (self.ae, self.text_encoder, self.flux_offloader):
            m.cuda()
        return self

    @staticmethod
    def load(name: str):
        if "4B" in name:
            text_id = "Qwen/Qwen3-4B-FP8"
        elif "9B" in name:
            text_id = "Qwen/Qwen3-8B-FP8"
        else:
            raise ValueError
        return Flux2Pipeline(load_flux2(name), text_id)

    # default is for klein
    @torch.no_grad()
    def generate(
        self,
        prompt: str | list[str],
        neg_prompt: str | list[str] = "",
        ref_imgs: list[Image.Image] | None = None,
        img_size: int | tuple[int, int] = 512,
        guidance: Tensor | float | None = None,
        cfg_scale: float = 1.0,
        num_steps: int = 4,
        seed: int | None = None,
        solver: str = "euler",
        pbar: bool = False,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> Tensor:
        txt = self.text_encoder(prompt)

        if cfg_scale != 1.0:
            neg_txt = self.text_encoder(neg_prompt)
            if neg_txt.shape[0] == 1:
                neg_txt = neg_txt.expand(txt.shape[0], -1, -1)
        else:
            neg_txt = None

        bsize = txt.shape[0]
        if isinstance(img_size, int):
            height = width = img_size
        else:
            height, width = img_size

        ae = self.ae
        shape = (bsize, height // ae.downsample // 2, width // ae.downsample // 2, self.ae.cfg.z_dim * 4)
        device = ae.encoder.conv_in.weight.device

        # keep latents in FP32 for accurate .lerp()
        rng = torch.Generator(device).manual_seed(seed) if seed is not None else None
        noise = torch.randn(shape, device=device, dtype=torch.float32, generator=rng)

        timesteps = torch.linspace(1.0, 0.0, num_steps + 1)
        timesteps = flux2_time_shift(timesteps, num_steps, shape[1] * shape[2])
        timesteps = timesteps.tolist()

        if ref_imgs is not None:
            ref_latents = []
            for img in ref_imgs:
                img = crop_to_multiple(img).convert("RGB")
                img = torch.from_numpy(np.asarray(img)).to(device)
                ref_latents.append(self.ae.encode(img.unsqueeze(0)).squeeze(0))
        else:
            ref_latents = None

        latents = flux2_generate(
            flux=self.flux,
            latents=noise,
            ref_latents=ref_latents,
            timesteps=timesteps,
            txt=txt,
            neg_txt=neg_txt,
            guidance=guidance,
            cfg_scale=cfg_scale,
            solver=solver,
            pbar=pbar,
            progress_cb=progress_cb,
        )
        return self.ae.decode(latents, uint8=True)


# https://github.com/black-forest-labs/flux2/blob/b56ac614/src/flux2/sampling.py#L244
def flux2_time_shift(timesteps: Tensor | float, num_steps: int, img_seq_len: int):
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    m_10 = a1 * img_seq_len + b1
    m_200 = a2 * img_seq_len + b2

    if img_seq_len > 4300:
        mu = m_200

    else:
        a = (m_200 - m_10) / (200 - 10)
        b = m_200 - 200 * a
        mu = a * num_steps + b

    exp_mu = math.exp(mu)
    return exp_mu / (exp_mu + 1 / timesteps - 1)


def crop_to_multiple(img: Image.Image, x: int = 16):
    W, H = img.size
    newW = W // x * x
    newH = H // x * x

    left = (W - newW) // 2
    top = (H - newH) // 2
    return img.crop((left, top, left + newW, top + newH))


@torch.no_grad()
def flux2_generate(
    flux: Flux2,
    latents: Tensor,
    timesteps: list[float],
    txt: Tensor,
    neg_txt: Tensor | None = None,
    ref_latents: list[Tensor] | None = None,
    guidance: Tensor | float | None = 3.5,
    cfg_scale: float = 1.0,
    solver: str = "euler",
    pbar: bool = False,
    progress_cb: Callable[[int, int], None] | None = None,
) -> Tensor:
    B, H, W, _ = latents.shape
    img_len = H * W

    if isinstance(guidance, (int, float)):
        guidance = torch.full((B,), guidance, device=latents.device)

    num_steps = len(timesteps) - 1
    solver_ = get_solver(solver, timesteps)
    timesteps_tensor = torch.tensor(timesteps).to("cuda", non_blocking=True)

    img_rope = flux.make_img_rope(H, W)
    txt_rope = flux.make_txt_rope(txt.shape[1])

    if ref_latents is not None:
        ref_ropes = []
        flat_latents = []

        for i, ref_lat in enumerate(ref_latents):
            h, w, _ = ref_lat.shape
            ref_ropes.append(flux.make_img_rope(h, w, t=10 * (i + 1)))
            flat_latents.append(ref_lat.flatten(0, 1))

        rope = torch.cat([txt_rope, img_rope, *ref_ropes], dim=0)
        ref_imgs = torch.cat(flat_latents, dim=0).unsqueeze(0).expand(B, -1, -1)

    else:
        rope = torch.cat([txt_rope, img_rope], dim=0)

    latents = latents.flatten(1, 2)  # (B,H,W,C) -> (B,H*W,C)

    for i in tqdm(range(num_steps), disable=not pbar, dynamic_ncols=True):
        # concat reference latents (images) if needed.
        if ref_latents is not None:
            imgs = torch.cat([latents, ref_imgs], dim=1)
        else:
            imgs = latents

        # only keep the velocity for the latents being denoised.
        t = timesteps_tensor[None, i]
        v = flux(imgs, t, txt, rope, guidance)[:, :img_len].float()

        # classifier-free guidance
        if cfg_scale != 1.0:
            assert neg_txt is not None
            neg_v = flux(imgs, t, neg_txt, rope, guidance)[:, :img_len].float()
            v = neg_v.lerp(v, cfg_scale)

        latents = solver_.step(latents, v, i)
        if progress_cb is not None:
            progress_cb(i + 1, num_steps)

    return latents.unflatten(1, (H, W))
