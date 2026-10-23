# https://github.com/black-forest-labs/flux-3-image-klein/blob/bfc8889/src/flux3/text_encoder.py
# https://github.com/black-forest-labs/flux-3-image-klein/blob/bfc8889/src/flux3/pipeline.py
# https://github.com/black-forest-labs/flux-3-image-klein/blob/bfc8889/src/flux3/pup.py

import base64
import io
import json
import math
import urllib.request
from collections.abc import Callable

import numpy as np
import torch
from huggingface_hub import hf_hub_download
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
    def __init__(self, model_id: str = "Qwen/Qwen3-VL-4B-Instruct-FP8"):
        super().__init__()
        self.output_indices = [3, 7, 11, 15, 19, 23, 27, 31]  # hidden_states[4, 8, ..., 32] in HF
        self.max_length = 4096
        self.pad_multiple = 80

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

    # same as official: right-pad to a multiple of 80. pad tokens are masked as keys, and have position_id=1.
    # the DiT attends to pad tokens, so this matters. NOTE: we pad all prompts to the same length.
    @torch.no_grad()
    def forward(self, texts: str | list[str]) -> Tensor:
        if isinstance(texts, str):
            texts = [texts]

        texts = [
            self.tokenizer.apply_chat_template(
                [dict(role="user", content=txt)], tokenize=False, add_generation_prompt=True
            )
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


# Prompt upsampler (pup), a Qwen3.5 VLM, served by vLLM in a separate env (vLLM pins its own torch/transformers).
# Bounding-box prompts are not supported. Example server for a single 32GB GPU, shared with Flux3 FP8:
#   vllm serve black-forest-labs/flux-3-image-klein-pup --served-model-name pup --quantization fp8 \
#     --max-model-len 8192 --max-num-seqs 1 --kv-cache-memory-bytes 512M --no-enable-prefix-caching \
#     --limit-mm-per-prompt '{"image": 10}' --generation-config vllm --port 8001

I2I_SCAFFOLD = (
    "Reference images follow in order (the first image, the second image, ...).\n"
    "User request: {instruction}\n"
    "Output the caption."
)


class Flux3PromptUpsampler:
    def __init__(self, url: str = "http://localhost:8001", model: str = "pup") -> None:
        self.url = f"{url}/v1/chat/completions"
        self.model = model
        path = hf_hub_download("black-forest-labs/flux-3-image-klein-pup", "system_prompts.json")
        with open(path) as f:
            self.system_prompts = json.load(f)

    def __call__(self, prompt: str, ref_imgs: list[Image.Image] | None = None) -> str:
        ref_imgs = ref_imgs or []
        assert len(ref_imgs) <= 10
        task = "i2i" if ref_imgs else "t2i"
        if ref_imgs:
            content = [
                dict(type="image_url", image_url=dict(url=flux3_prepare_pup_reference(img, len(ref_imgs))))
                for img in ref_imgs
            ]
            content.append(dict(type="text", text=I2I_SCAFFOLD.format(instruction=prompt)))
        else:
            content = prompt

        body = dict(
            model=self.model,
            messages=[dict(role="system", content=self.system_prompts[task]), dict(role="user", content=content)],
            temperature=0.0,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            presence_penalty=1.5 if task == "t2i" else 0.0,
            max_tokens=3584,
            chat_template_kwargs=dict(enable_thinking=False),
        )
        req = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            caption = json.load(resp)["choices"][0]["message"]["content"].strip()
        assert caption, "empty generation"
        return caption


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

        ae = self.ae
        device = ae.encoder.conv_in.weight.device
        cond, cond_rope = self.encode_ref_imgs(ref_imgs)
        if cond is not None:
            cond = cond.expand(bsize, -1, -1)

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

    # reference image i uses RoPE l=-(i+1). returns (1, L, C) tokens and rope.
    def encode_ref_imgs(self, ref_imgs: list[Image.Image]) -> tuple[Tensor | None, Tensor | None]:
        if not ref_imgs:
            return None, None

        device = self.ae.encoder.conv_in.weight.device
        cond_list = []
        cond_rope_list = []
        for i, img in enumerate(ref_imgs):
            z = self.ae.encode(torch.from_numpy(np.array(img)).to(device).unsqueeze(0))  # (1, h, w, C)
            cond_list.append(z.flatten(1, 2))
            cond_rope_list.append(self.flux.make_img_rope(z.shape[1], z.shape[2], l_pos=-(i + 1), device=device))
        return torch.cat(cond_list, dim=1), torch.cat(cond_rope_list, dim=0)


# https://github.com/black-forest-labs/flux-3-image-klein/blob/bfc8889/src/flux3/pipeline.py#L131
# fit within 4MP (sides <= 4096), then reflect-pad to a multiple of 16 (sides >= 64).
# returns the padded image, and the (left, top, right, bottom) box of the original image inside it (None if no pad).
def flux3_prepare_reference(img: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    img = ImageOps.exif_transpose(img).convert("RGB")
    max_pixels = 4 * 1024 * 1024

    scale = min(1.0, math.sqrt(max_pixels / (img.width * img.height)), 4096 / max(img.size))
    while True:
        fit = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        w, h = (min(4096, max(64, math.ceil(x / 16) * 16)) for x in fit)
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


# official: RGB over white, LANCZOS resize to a pixel budget (depends on the number of references) with sides
# a multiple of 32, then JPEG q92
def flux3_prepare_pup_reference(img: Image.Image, num_refs: int) -> str:
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img)
    img = img.convert("RGB")

    cap = {1: 1024 * 1024, 2: 724 * 724}.get(num_refs, 512 * 512)
    w = max(32, round(img.width / 32) * 32)
    h = max(32, round(img.height / 32) * 32)
    if w * h > cap:
        beta = math.sqrt(img.width * img.height / cap)
        w = max(32, math.floor(img.width / beta / 32) * 32)
        h = max(32, math.floor(img.height / beta / 32) * 32)
    if (w, h) != img.size:
        img = img.resize((w, h), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# https://github.com/black-forest-labs/flux-3-image-klein/blob/bfc8889/src/flux3/sampling.py#L9
# NOTE: we follow official code (alpha=16). official config.json, diffusers and ComfyUI use alpha=6.93.
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
