# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/bria_fibo2/pipeline_bria_fibo2.py

import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from PIL import Image
from torch import Tensor, nn
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor

from ..qwen3 import Qwen3Model
from ..qwen3_vl import Qwen3VLVisionModel, qwen3_vl_forward, qwen3_vl_position_ids, qwen3_vl_rope
from ..solvers import get_solver
from ..utils import load_hf_state_dict
from ..vae import load_vae
from .model import Fibo, load_fibo
from .text_images import draw_text_image, load_font

# where an image stands in a Qwen3-VL prompt
VISION_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"

# the structured caption with every value empty, which Fibo 2.0 was trained with
DEFAULT_NEG_PROMPT = (
    '{"short_description":"","objects":[],"background_setting":"","lighting":{},'
    '"aesthetics":{},"photographic_characteristics":{},"style_medium":"",'
    '"context":"","artistic_style":""}'
)


# Qwen3-VL-4B. Fibo 2.0's text encoder is the stock model, so we can use the official FP8 weights.
# the vision tower (BF16 in the FP8 checkpoint) sees the images to edit and the images of the caption's text.
class FiboTextEncoder(nn.Module):
    # HF hidden_states indices behind the DiT's text inputs: the Perceiver's, then one per injection block.
    # each bundle concatenates 3 layers, each normalized to unit RMS.
    BUNDLES = ((9, 20, 31), (5, 16, 27), (7, 18, 29), (9, 20, 31), (11, 22, 33), (13, 24, 35))

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-4B-Instruct-FP8") -> None:
        super().__init__()
        self.layer_ids = sorted({i for bundle in self.BUNDLES for i in bundle})
        self.max_length = 4096

        # hidden_states[i] in HF is the output of layer i-1. only keep layers we need.
        cfg = AutoConfig.from_pretrained(model_id)
        self.cfg = cfg
        text_cfg = cfg.text_config
        text_cfg.num_hidden_layers = max(self.layer_ids)
        with torch.device("meta"):
            self.model = Qwen3Model(text_cfg)
            self.visual = Qwen3VLVisionModel(cfg.vision_config)

        state_dict = load_hf_state_dict(model_id, "model.safetensors.index.json", prefix="model.language_model.")
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith("layers.") or int(k.split(".")[1]) < text_cfg.num_hidden_layers
        }
        self.model.load_state_dict(state_dict, assign=True)
        self.visual.load_state_dict(
            load_hf_state_dict(model_id, "model.safetensors.index.json", prefix="model.visual."), assign=True
        )
        self.eval().requires_grad_(False)

        # Fibo 2.0's processor: Qwen3-VL's tokenizer, with its own image processor settings
        self.processor = AutoProcessor.from_pretrained("briaai/Fibo-2.0-dev", subfolder="processor")

    # each prompt is encoded separately without padding, with its images in the order of its vision markers.
    # returns [6, L, 7680] for each prompt.
    @torch.no_grad()
    def forward(self, texts: str | list[str], images: list[list[Image.Image]] | None = None) -> list[Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        images = images or [[]] * len(texts)

        device = self.model.embed_tokens.weight.device
        text_cfg = self.cfg.text_config
        merge_size = self.cfg.vision_config.spatial_merge_size
        outputs = []
        for text, imgs in zip(texts, images):
            text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
            inputs = self.processor(
                text=[text], images=imgs or None, truncation=True, max_length=self.max_length, return_tensors="pt"
            )
            input_ids = inputs.input_ids.to(device)

            image_mask = image_embeds = None
            deepstack = []
            grid_thw = []
            if imgs:
                grid_thw = inputs.image_grid_thw.tolist()
                image_embeds, deepstack = self.visual(inputs.pixel_values.to(device), grid_thw)
                image_mask = input_ids[0] == self.cfg.image_token_id

            pos_ids = qwen3_vl_position_ids(input_ids[0], self.cfg.image_token_id, grid_thw, merge_size)
            rope = text_cfg.rope_parameters
            pos_embeds = qwen3_vl_rope(pos_ids, text_cfg.head_dim, rope["rope_theta"], rope["mrope_section"])
            hidden_states = qwen3_vl_forward(
                self.model,
                input_ids,
                pos_embeds,
                [i - 1 for i in self.layer_ids],
                image_mask,
                image_embeds,
                deepstack,
            )
            hidden_states = {
                i: F.rms_norm(h[0].float(), h.shape[-1:], eps=1e-6).bfloat16()
                for i, h in zip(self.layer_ids, hidden_states)
            }
            bundles = [torch.cat([hidden_states[i] for i in bundle], dim=-1) for bundle in self.BUNDLES]
            outputs.append(torch.stack(bundles))
        return outputs


# a prompt's caption in the form Fibo 2.0 was trained with, and its edit instruction. a prompt that is not a JSON object
# is returned as is, with no instruction. an edit prompt's caption is its other keys, or its "structured_caption" key.
# training captions hold no empty values, and no aesthetic_score / preference_score. a caption of exactly a
# short_description and a text_render list keeps both, even when the list is empty (it says the image has no text).
def fibo_parse_prompt(prompt: str) -> tuple[dict | str | None, str | None]:
    try:
        caption = json.loads(prompt)
    except json.JSONDecodeError:
        return prompt, None
    if not isinstance(caption, dict):
        return prompt, None
    instruction = caption.pop("edit_instruction", None)
    if "structured_caption" in caption:
        caption = caption.pop("structured_caption")
        if not isinstance(caption, dict):
            return caption or None, instruction

    def empty(v):
        return math.isnan(v) if isinstance(v, float) else (v is None or (isinstance(v, (str, dict, list)) and not v))

    def clean(v):
        if isinstance(v, dict):
            v = {k: clean(x) for k, x in v.items()}
            return {k: x for k, x in v.items() if not empty(x)}
        if isinstance(v, list):
            return [x for x in map(clean, v) if not empty(x)]
        return v

    keys = set(caption)
    scores = ("aesthetic_score", "preference_score")
    out = {
        k: {kk: vv for kk, vv in v.items() if kk not in scores} if isinstance(v, dict) else v
        for k, v in clean(caption).items()
        if k not in scores
    }
    out = {k: v for k, v in out.items() if v != {}}
    if keys == {"short_description", "text_render"}:
        out = dict(short_description=out.get("short_description") or "", text_render=out.get("text_render") or [])
    return out or None, instruction


def _dumps(x) -> str:
    return json.dumps(x, separators=(",", ":"), ensure_ascii=False)


# a generation prompt in its trained form, serialized compactly. a prompt that is not a JSON caption stays as is.
def fibo_clean_prompt(prompt: str) -> str:
    caption, _ = fibo_parse_prompt(prompt)
    return caption if isinstance(caption, str) else _dumps(caption or {})


# the texts of a caption's text_render entries that have text, in order. each gets an image of its text.
def fibo_caption_texts(caption) -> list[str]:
    entries = (caption.get("text_render") or []) if isinstance(caption, dict) else []
    return [e["text"] for e in entries if isinstance(e, dict) and isinstance(e.get("text"), str) and e["text"]]


# marks, in place, each text_render entry that has text with an "image_tokens" entry: the vision marker of the image of
# its text, or empty where the image isn't shown.
def fibo_mark_text_images(caption: dict, marker: str) -> dict:
    for entry in caption.get("text_render") or []:
        if isinstance(entry, dict) and isinstance(entry.get("text"), str) and entry["text"]:
            entry["image_tokens"] = marker
    return caption


# an edit prompt is one JSON object: a vision marker per image, the caption of the result when the prompt has one,
# then the edit instruction. with text images, each text entry gets the marker of its image (after the images to edit).
def fibo_edit_prompt(prompt: str, num_images: int, text_images: bool = False) -> str:
    caption, instruction = fibo_parse_prompt(prompt)
    if instruction is None:  # the prompt is the instruction alone
        caption, instruction = None, prompt
    if text_images and isinstance(caption, dict):
        fibo_mark_text_images(caption, VISION_MARKER)

    if num_images == 1:
        edit = {"image": VISION_MARKER}
    else:
        edit = {f"image_{k}": VISION_MARKER for k in range(1, num_images + 1)}
    if caption:
        edit["structured_caption"] = caption
    edit["edit_instruction"] = instruction
    return _dumps(edit)


# the (width, height) Fibo 2.0 was trained at for an aspect ratio: of the sizes divisible by 16, at least 64 a side and
# within 10% of area, the one that crops the least from a picture with that aspect ratio, then the one closest to area.
def fibo_image_size(aspect_ratio: float, area: int = 1024 * 1024, multiple: int = 16) -> tuple[int, int]:
    min_side = 4 * multiple
    sizes = [
        (w, h)
        for w in range(min_side, 2 * area // min_side, multiple)
        for h in range(min_side, 2 * area // w, multiple)
        if 0.9 * area <= w * h <= 1.1 * area
    ]

    def crop(size):
        ratio = size[0] / size[1]
        return 1 - min(ratio / aspect_ratio, aspect_ratio / ratio)

    return min(sizes, key=lambda size: (round(crop(size), 9), abs(size[0] * size[1] - area), -size[0]))


# scale to cover width x height keeping the aspect ratio, then crop the center, as training did
def fibo_resize_crop(img: Image.Image, width: int, height: int) -> Image.Image:
    img = img.convert("RGB")
    scale = max(height / img.height, width / img.width)
    w, h = math.ceil(img.width * scale), math.ceil(img.height * scale)
    img = img.resize((w, h), resample=Image.LANCZOS)
    left, top = (w - width) // 2, (h - height) // 2
    return img.crop((left, top, left + width, top + height))


# the region the mask marks (white on black, binarized at 128) greyed out to (128, 128, 128), as training did
def fibo_grey_out(img: Image.Image, mask: Image.Image) -> Image.Image:
    grey = Image.new("RGB", img.size, (128, 128, 128))
    binary_mask = mask.convert("L").point(lambda v: 255 if v >= 128 else 0)
    return Image.composite(grey, img.convert("RGB"), binary_mask)


class FiboPipeline:
    MAX_NUM_IMAGES = 5
    TEXT_ENCODER_IMAGE_SIZE = 768  # Qwen3-VL sees each image to edit at most this many pixels a side
    # a generation shows the text images to Qwen3-VL only once the noise level is at or below this. an edit shows them
    # at every step.
    TEXT_IMAGES_MAX_NOISE_LEVEL = 0.9

    # fp8: online block FP8 for the DiT, quantized on GPU in .cuda(). the text encoder uses official FP8 weights.
    def __init__(
        self, fibo: Fibo | None = None, text_encoder_id: str = "Qwen/Qwen3-VL-4B-Instruct-FP8", fp8: bool = True
    ) -> None:
        # apply_rope() takes the slow reference path when params require grad
        self.fibo = (fibo or load_fibo()).requires_grad_(False)
        self.fp8 = fp8
        self.ae = load_vae("flux2").bfloat16()  # same weights as Flux.2. official runs the VAE in FP32
        self.text_encoder = FiboTextEncoder(text_encoder_id)

        # training drew the characters Pillow's built-in font lacks in these fonts. official uses them only if passed.
        font_dir = Path(snapshot_download("briaai/Fibo-2.0-dev", allow_patterns="fonts/*")) / "fonts"
        self.fonts = [load_font(str(font_dir / name)) for name in ("Aileron-Regular.otf", "DejaVuSans.ttf")]

    def cpu(self):
        for m in (self.fibo, self.ae, self.text_encoder):
            m.cpu()
        return self

    def cuda(self):
        for m in (self.fibo, self.ae, self.text_encoder):
            m.cuda()
        if self.fp8 and not self.fibo.layers[0].attention.qkv.is_fp8_1d2d():
            self.fibo.to_fp8()
        return self

    # prompt: a structured JSON caption (preferred), or a short description.
    #   with images: the edit instruction, or a JSON object with "edit_instruction" and the caption of the result
    #   (its other keys, or a "structured_caption" key).
    # images: up to 5 images to edit, which the instruction can refer to as <image_1>, <image_2>, ...
    # mask: the region of the single image to regenerate, white on black.
    # text_images: show Qwen3-VL an image of each text in the caption's text_render, as training did.
    # all prompts must have the same token length, since the DiT doesn't support padding.
    @torch.no_grad()
    def generate(
        self,
        prompt: str | list[str],
        neg_prompt: str = DEFAULT_NEG_PROMPT,
        images: Image.Image | list[Image.Image] | None = None,
        mask: Image.Image | None = None,
        text_images: bool = True,
        img_size: int | tuple[int, int] | None = None,
        cfg_scale: float = 5.0,
        num_steps: int = 30,
        seed: int | None = None,
        solver: str = "euler",
        pbar: bool = False,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> Tensor:
        prompts = [prompt] if isinstance(prompt, str) else prompt
        images = [images] if isinstance(images, Image.Image) else images
        if images is not None:
            assert 1 <= len(images) <= self.MAX_NUM_IMAGES
        assert mask is None or (images is not None and len(images) == 1 and mask.size == images[0].size)

        # output size. by default for an edit, the trained size for the aspect ratio of the first image.
        if img_size is None:
            img_size = (1024, 1024) if images is None else fibo_image_size(images[0].width / images[0].height)[::-1]
        height, width = (img_size, img_size) if isinstance(img_size, int) else img_size

        # the images to edit: center-cropped to the output size (a single image), or each to its own trained size.
        # the VAE encodes the crops, Qwen3-VL sees downsized copies.
        ctx, te_images = None, []
        if images is not None:
            if mask is not None:
                images = [fibo_grey_out(images[0], mask)]
            if len(images) == 1:
                sizes = [(width, height)]
            else:
                sizes = [fibo_image_size(img.width / img.height) for img in images]
            crops = [fibo_resize_crop(img, w, h) for img, (w, h) in zip(images, sizes)]
            for crop in crops:
                crop = crop.copy()
                crop.thumbnail((self.TEXT_ENCODER_IMAGE_SIZE, self.TEXT_ENCODER_IMAGE_SIZE))
                te_images.append(crop)
            device = self.ae.encoder.conv_in.weight.device
            ctx = [
                self.ae.encode(torch.from_numpy(np.array(crop)).to(device)[None])
                .bfloat16()
                .expand(len(prompts), -1, -1, -1)
                for crop in crops
            ]

        # the images of each prompt's text
        captions = [fibo_parse_prompt(p)[0] for p in prompts]
        own_text_images = [
            [draw_text_image(t, self.fonts) for t in fibo_caption_texts(c)] if text_images else [] for c in captions
        ]

        # Qwen3-VL prompts, and their images: the images to edit, then the text images
        te_prompts = []
        base_prompts = []  # a generation's first steps use the captions with each text entry's marker left empty
        for p, caption, own in zip(prompts, captions, own_text_images):
            if images is not None:
                te_prompts.append(fibo_edit_prompt(p, len(images), text_images=bool(own)))
                continue
            p = fibo_clean_prompt(p)
            base_prompts.append(_dumps(fibo_mark_text_images(json.loads(p), "")) if own else p)
            te_prompts.append(_dumps(fibo_mark_text_images(json.loads(p), VISION_MARKER)) if own else p)

        txt = self.text_encoder(te_prompts, [te_images + own for own in own_text_images])
        assert all(x.shape == txt[0].shape for x in txt), "all prompts must have the same token length"
        txt = torch.stack(txt)
        base_txt = None
        if images is None and any(own_text_images):
            base_txt = self.text_encoder(base_prompts)
            assert all(x.shape == base_txt[0].shape for x in base_txt), "all prompts must have the same token length"
            base_txt = torch.stack(base_txt)
        bsize = txt.shape[0]
        neg_txt = self.text_encoder(neg_prompt)[0].expand(bsize, -1, -1, -1) if cfg_scale != 1.0 else None

        ae = self.ae
        H = height // ae.downsample // 2
        W = width // ae.downsample // 2
        device = ae.encoder.conv_in.weight.device

        # same as official: FP32 noise in unpatchified NCHW layout, sampled on GPU
        rng = torch.Generator(device).manual_seed(seed) if seed is not None else None
        noise = torch.randn(bsize, ae.cfg.z_dim, H * 2, W * 2, device=device, generator=rng)
        noise = noise.unflatten(2, (H, 2)).unflatten(4, (W, 2)).permute(0, 2, 4, 1, 3, 5).flatten(3)

        latents = fibo_generate(
            fibo=self.fibo,
            latents=noise,
            timesteps=fibo_timesteps(num_steps, H * W),
            txt=txt,
            neg_txt=neg_txt,
            ctx=ctx,
            base_txt=base_txt,
            base_max_noise_level=self.TEXT_IMAGES_MAX_NOISE_LEVEL,
            cfg_scale=cfg_scale,
            solver=solver,
            pbar=pbar,
            progress_cb=progress_cb,
        )
        return ae.decode(latents, uint8=True)


# official: sigmas from 1 to 1/num_steps, with resolution-dependent exponential shift. ends with 0.
def fibo_timesteps(num_steps: int, img_seq_len: int) -> list[float]:
    base_len, max_len, base_shift, max_shift = 256, 4096, 0.5, 1.15
    mu = base_shift + (max_shift - base_shift) / (max_len - base_len) * (img_seq_len - base_len)
    sigmas = torch.linspace(1.0, 1 / num_steps, num_steps, dtype=torch.float64)
    sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
    return sigmas.float().tolist() + [0.0]


@torch.no_grad()
def fibo_generate(
    fibo: Fibo,
    latents: Tensor,
    timesteps: list[float],
    txt: Tensor,
    neg_txt: Tensor | None = None,
    ctx: list[Tensor] | None = None,
    base_txt: Tensor | None = None,
    base_max_noise_level: float = 0.9,
    cfg_scale: float = 1.0,
    solver: str = "euler",
    pbar: bool = False,
    progress_cb: Callable[[int, int], None] | None = None,
) -> Tensor:
    num_steps = len(timesteps) - 1
    solver_ = get_solver(solver, timesteps)
    B = latents.shape[0]

    for i in tqdm(range(num_steps), disable=not pbar, dynamic_ncols=True):
        t = torch.tensor([timesteps[i]], device=latents.device).expand(B)
        # base_txt (prompt without text images) is used while the noise level is above base_max_noise_level
        use_base = base_txt is not None and timesteps[i] > base_max_noise_level
        v = fibo(latents, t, base_txt if use_base else txt, ctx).float()

        # classifier-free guidance. the images to edit condition both passes.
        if cfg_scale != 1.0:
            neg_v = fibo(latents, t, neg_txt, ctx).float()
            v = neg_v.lerp(v, cfg_scale)

        latents = solver_.step(latents, v, i)
        if progress_cb is not None:
            progress_cb(i + 1, num_steps)

    return latents
