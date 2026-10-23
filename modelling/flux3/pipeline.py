# https://github.com/black-forest-labs/flux-3-image-klein/blob/bfc8889/src/flux3/text_encoder.py
# https://github.com/black-forest-labs/flux-3-image-klein/blob/bfc8889/src/flux3/pipeline.py

import math
from collections.abc import Callable

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor, nn
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer

from ..qwen3 import Qwen3Model
from ..solvers import get_solver
from ..utils import load_hf_state_dict
from ..vae import load_vae
from .model import Flux3, load_flux3


# text-only Qwen3-VL. with text-only inputs, M-RoPE reduces to 1D RoPE, so we can reuse Qwen3Model.
class Flux3TextEncoder(nn.Module):
    # pad_multiple=1 disables padding
    def __init__(self, model_id: str = "Qwen/Qwen3-VL-4B-Instruct-FP8", pad_multiple: int = 80):
        super().__init__()
        self.output_indices = [3, 7, 11, 15, 19, 23, 27, 31]  # hidden_states[4, 8, ..., 32] in HF
        self.max_length = 4096
        self.pad_multiple = pad_multiple

        # only keep layers we need
        cfg = AutoConfig.from_pretrained(model_id).text_config
        cfg.num_hidden_layers = max(self.output_indices) + 1
        with torch.device("meta"):
            self.model = Qwen3Model(cfg)

        state_dict = load_hf_state_dict(model_id, "model.safetensors.index.json", prefix="model.language_model.")
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith("layers.") or int(k.split(".")[1]) < cfg.num_hidden_layers
        }
        self.model.load_state_dict(state_dict, assign=True)
        self.model.eval()

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id)

    # all prompts are padded to the same length (multiple of 80). the DiT attends to pad tokens, so we need to
    # match official pad embeddings: pad tokens don't attend to each other, and have position_id=1.
    # NOTE: official code encodes 1 prompt at a time. for multiple prompts of different lengths, the shorter ones
    # will have more pad tokens than official.
    @torch.no_grad()
    def forward(self, texts: str | list[str]) -> Tensor:
        if isinstance(texts, str):
            texts = [texts]

        texts = [
            self.tokenizer.apply_chat_template([dict(role="user", content=txt)], tokenize=False, add_generation_prompt=True)
            for txt in texts
        ]
        input_ids_list = [
            self.tokenizer(txt, truncation=True, max_length=self.max_length)["input_ids"] for txt in texts
        ]
        L = max(len(x) for x in input_ids_list)
        L = min(math.ceil(L / self.pad_multiple) * self.pad_multiple, self.max_length)

        device = self.model.embed_tokens.weight.device
        outputs = []
        for input_ids in input_ids_list:
            num_pads = L - len(input_ids)
            input_ids = torch.tensor(input_ids + [self.tokenizer.pad_token_id] * num_pads, device=device)

            mask = torch.arange(L, device=device) < L - num_pads
            position_ids = torch.where(mask, torch.arange(L, device=device), 1)
            attn_mask = torch.ones(L, L, dtype=torch.bool, device=device).tril() & mask

            hidden_states = self.model(
                input_ids[None],
                output_indices=self.output_indices,
                attn_mask=attn_mask[None, None],
                position_ids=position_ids,
            )
            outputs.append(torch.cat(hidden_states, dim=-1))

        return torch.cat(outputs, dim=0)


class Flux3Pipeline:
    KLEIN_DEFAULTS = dict(num_steps=16, cfg_scale=1.0)
    KLEIN_BASE_DEFAULTS = dict(num_steps=50, cfg_scale=4.0)

    def __init__(self, flux: Flux3 | None = None, text_encoder_id: str = "Qwen/Qwen3-VL-4B-Instruct-FP8") -> None:
        self.flux = flux or load_flux3()
        self.ae = load_vae("flux2").bfloat16()  # same weights as Flux.2. official code keeps it in FP32
        self.text_encoder = Flux3TextEncoder(text_encoder_id)

    def cpu(self):
        for m in (self.flux, self.ae, self.text_encoder):
            m.cpu()
        return self

    def cuda(self):
        for m in (self.flux, self.ae, self.text_encoder):
            m.cuda()
        return self

    # img_size: (height, width). default to 1024x1024 for T2I, and the (prepared) 1st reference image size for TI2I.
    # default is for klein (distilled). for klein-base, use KLEIN_BASE_DEFAULTS.
    @torch.no_grad()
    def generate(
        self,
        prompt: str | list[str],
        neg_prompt: str = "",
        ref_imgs: list[Image.Image] | None = None,
        img_size: int | tuple[int, int] | None = None,
        cfg_scale: float = 1.0,
        num_steps: int = 16,
        alpha: float = 16.0,
        seed: int | None = None,
        solver: str = "euler",
        pbar: bool = False,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> Tensor:
        txt = self.text_encoder(prompt)
        bsize = txt.shape[0]

        # neg_txt may have a different length from txt, so we can't batch them
        neg_txt = self.text_encoder(neg_prompt).expand(bsize, -1, -1) if cfg_scale != 1.0 else None
        ae = self.ae
        device = ae.encoder.conv_in.weight.device

        # official: a single reference is reflect-padded to a multiple of 16, and the output is cropped back.
        # multiple references are center-cropped instead.
        ref_imgs = list(ref_imgs or [])
        crop_box = None
        if len(ref_imgs) == 1:
            ref_imgs[0], crop_box = flux3_prepare_reference(ref_imgs[0])
        else:
            ref_imgs = [flux3_prepare_multi_reference(img) for img in ref_imgs]

        if img_size is None:
            width, height = ref_imgs[0].size if ref_imgs else (1024, 1024)
        else:
            crop_box = None
            height, width = (img_size, img_size) if isinstance(img_size, int) else img_size

        cond = cond_rope = None
        if ref_imgs:
            cond_list = []
            cond_rope_list = []
            for i, img in enumerate(ref_imgs):
                img = torch.from_numpy(np.array(img)).to(device)
                z = ae.encode(img.unsqueeze(0))  # (1, h, w, C)
                cond_list.append(z.flatten(1, 2))
                cond_rope_list.append(self.flux.make_img_rope(z.shape[1], z.shape[2], l=-(i + 1), device=device))
            cond = torch.cat(cond_list, dim=1).expand(bsize, -1, -1)
            cond_rope = torch.cat(cond_rope_list, dim=0)

        H = height // ae.downsample // 2
        W = width // ae.downsample // 2

        # same as official: CPU RNG, round to BF16, keep latents in FP32
        rng = torch.Generator().manual_seed(seed) if seed is not None else None
        noise = torch.randn(bsize, H * W, ae.cfg.z_dim * 4, generator=rng).bfloat16().float().to(device)

        latents = flux3_generate(
            flux=self.flux,
            latents=noise,
            img_size=(H, W),
            timesteps=flux3_timesteps(num_steps, alpha),
            txt=txt,
            neg_txt=neg_txt,
            cfg_scale=cfg_scale,
            cond=cond,
            cond_rope=cond_rope,
            solver=solver,
            pbar=pbar,
            progress_cb=progress_cb,
        )
        out = ae.decode(latents.unflatten(1, (H, W)), uint8=True)  # (B, H, W, 3)

        if crop_box is not None:
            left, top, right, bottom = crop_box
            out = out[:, top:bottom, left:right]
        return out


# https://github.com/black-forest-labs/flux-3-image-klein/blob/bfc8889/src/flux3/pipeline.py#L131
# fit within 4MP (sides <= 4096), then reflect-pad to a multiple of 16 (sides >= 64).
# returns the padded image, and the (left, top, right, bottom) box of the original image inside it (None if no pad).
def flux3_prepare_reference(img: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    img = ImageOps.exif_transpose(img).convert("RGB")
    max_pixels = 4 * 1024 * 1024
    ceil16 = lambda x: -(-x // 16) * 16

    scale = min(1.0, math.sqrt(max_pixels / (img.width * img.height)), 4096 / max(img.size))
    while True:
        fit = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        w, h = (min(4096, max(64, ceil16(x))) for x in fit)
        if w * h <= max_pixels:
            break
        scale *= 0.995

    if fit != img.size:
        img = img.resize(fit, Image.Resampling.LANCZOS)
    if (w, h) == img.size:
        return img, None

    left = (w - img.width) // 2
    top = (h - img.height) // 2
    padding = ((top, h - img.height - top), (left, w - img.width - left), (0, 0))
    mode = "reflect" if min(img.size) > 16 else "symmetric"  # reflect needs pad shorter than the side
    img_padded = Image.fromarray(np.pad(np.asarray(img), padding, mode=mode))
    return img_padded, (left, top, left + img.width, top + img.height)


# https://github.com/black-forest-labs/flux-3-image-klein/blob/bfc8889/src/flux3/pipeline.py#L161
# fit within 1MP, then center-crop to a multiple of 16.
def flux3_prepare_multi_reference(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    max_pixels = 1024 * 1024
    W, H = img.size
    if W * H > max_pixels:
        scale = math.sqrt(max_pixels / (W * H))
        W, H = int(W * scale), int(H * scale)
        img = img.resize((W, H), Image.Resampling.LANCZOS)

    w = W // 16 * 16
    h = H // 16 * 16
    left = (W - w) // 2
    top = (H - h) // 2
    return img.crop((left, top, left + w, top + h))


# https://github.com/black-forest-labs/flux-3-image-klein/blob/bfc8889/src/flux3/sampling.py#L9
# NOTE: official config.json says alpha=6.93, but official pipeline uses alpha=16 by default.
def flux3_timesteps(num_steps: int, alpha: float) -> list[float]:
    t = torch.linspace(1.0, 0.0, num_steps + 1)
    return (alpha * t / (1.0 + (alpha - 1.0) * t)).tolist()


@torch.no_grad()
def flux3_generate(
    flux: Flux3,
    latents: Tensor,
    img_size: tuple[int, int],
    timesteps: list[float],
    txt: Tensor,
    neg_txt: Tensor | None = None,
    cfg_scale: float = 1.0,
    cond: Tensor | None = None,
    cond_rope: Tensor | None = None,
    solver: str = "euler",
    pbar: bool = False,
    progress_cb: Callable[[int, int], None] | None = None,
) -> Tensor:
    B = latents.shape[0]
    H, W = img_size
    num_steps = len(timesteps) - 1
    solver_ = get_solver(solver, timesteps)
    timesteps_tensor = torch.tensor(timesteps).to(latents.device, non_blocking=True)

    img_rope = flux.make_img_rope(H, W, device=latents.device)
    txt_rope = flux.make_txt_rope(txt.shape[1], latents.device)
    if cfg_scale != 1.0:
        assert neg_txt is not None
        neg_txt_rope = flux.make_txt_rope(neg_txt.shape[1], latents.device)

    for i in tqdm(range(num_steps), disable=not pbar, dynamic_ncols=True):
        # flow map (distilled): the model also takes the timestep it jumps to
        t = timesteps_tensor[i].expand(B)
        t_next = timesteps_tensor[i + 1].expand(B) if flux.cfg.flow_map else None
        kwargs = dict(ref_time=t_next, cond=cond, cond_rope=cond_rope)
        v = flux(latents, t, txt, img_rope, txt_rope, **kwargs).float()

        # classifier-free guidance
        if cfg_scale != 1.0:
            neg_v = flux(latents, t, neg_txt, img_rope, neg_txt_rope, **kwargs).float()
            v = neg_v.lerp(v, cfg_scale)

        latents = solver_.step(latents, v, i)
        if progress_cb is not None:
            progress_cb(i + 1, num_steps)

    return latents
