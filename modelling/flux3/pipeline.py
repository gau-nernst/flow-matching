# https://github.com/black-forest-labs/flux-3-image-klein/blob/bfc8889/src/flux3/text_encoder.py
# https://github.com/black-forest-labs/flux-3-image-klein/blob/bfc8889/src/flux3/pipeline.py
# https://github.com/black-forest-labs/flux-3-image-klein/blob/bfc8889/src/flux3/pup.py

import base64
import io
import json
import math
import re
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

    # right-pad to a multiple of 80. pad tokens are masked as keys, and have position_id=1.
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
# Example server for a single 32GB GPU, shared with Flux3 FP8:
#   vllm serve black-forest-labs/flux-3-image-klein-pup --served-model-name pup --quantization fp8 \
#     --max-model-len 8192 --max-num-seqs 1 --kv-cache-memory-bytes 512M --no-enable-prefix-caching \
#     --limit-mm-per-prompt '{"image": 10}' --generation-config vllm --port 8001

I2I_SCAFFOLD = (
    "Reference images follow in order (the first image, the second image, ...).\n"
    "User request: {instruction}\n"
    "Output the caption."
)
I2I_BBOX_SCAFFOLD = (
    "Reference images follow in attachment order: ref_image_0, ref_image_1, ...\n"
    "User editing instruction: {instruction}\n"
    "Decide the edit, imagine the target image, and output the annotation record."
)


class Flux3PromptUpsampler:
    def __init__(self, url: str = "http://localhost:8001", model: str = "pup") -> None:
        self.url = f"{url}/v1/chat/completions"
        self.model = model
        path = hf_hub_download("black-forest-labs/flux-3-image-klein-pup", "system_prompts.json")
        with open(path) as f:
            self.system_prompts = json.load(f)

    # boxes: see flux3_box_rows(). the upsampler may refine their desc and add its own boxes.
    def __call__(self, prompt: str, ref_imgs: list[Image.Image] | None = None, boxes: list[dict] | None = None) -> str:
        ref_imgs = ref_imgs or []
        assert len(ref_imgs) <= 10
        task = "i2i" if ref_imgs else "t2i"
        rows = flux3_box_rows(boxes) if boxes else None
        if rows:
            prompt = (
                f"ORIGINAL PROMPT:\n{prompt.strip()}\n\n"
                "LOCKED BOUNDING-BOX ANNOTATIONS ([y_min, x_min, y_max, x_max] on a 0-1000 canvas, with (0, 0) at "
                "the top-left). This is an authoritative JSON list of dictionaries; preserve every supplied identity, "
                f"provenance value, null, and coordinate exactly:\n{json.dumps(rows, ensure_ascii=False)}"
            )

        if ref_imgs:
            content = [
                dict(type="image_url", image_url=dict(url=flux3_prepare_pup_reference(img, len(ref_imgs))))
                for img in ref_imgs
            ]
            scaffold = I2I_BBOX_SCAFFOLD if rows else I2I_SCAFFOLD
            content.append(dict(type="text", text=scaffold.format(instruction=prompt)))
        else:
            content = prompt

        system_prompt = self.system_prompts[f"{task}_bbox" if rows else task]
        body = dict(
            model=self.model,
            messages=[dict(role="system", content=system_prompt), dict(role="user", content=content)],
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
        return _reconcile_boxes(caption, rows) if rows else caption


class Flux3Pipeline:
    KLEIN_DEFAULTS = dict(num_steps=16, cfg_scale=1.0)
    KLEIN_BASE_DEFAULTS = dict(num_steps=50, cfg_scale=4.0)

    def __init__(self, flux: Flux3 | None = None, text_encoder_id: str = "Qwen/Qwen3-VL-4B-Instruct-FP8") -> None:
        self.flux = flux or load_flux3()
        self.ae = load_vae("flux2").bfloat16()  # same weights as Flux.2. reference code keeps it in FP32
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

        # a single reference is reflect-padded to a multiple of 16, and the output is cropped back.
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

        # CPU RNG, round to BF16, keep latents in FP32
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


# RGB over white, LANCZOS resize to a pixel budget (depends on the number of references) with sides
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


# https://github.com/black-forest-labs/flux-3-image-klein/blob/bfc8889/src/flux3/bbox.py
# boxes are [y_min, x_min, y_max, x_max] in 0-1, top-left origin. each box is either
# - T2I/TI2I: {"id", "desc", "bbox"}
# - TI2I: {"id", "desc", "from": "ref_image_<i>" | "target", "src_bbox", "tgt_bbox"} (where it is in a reference,
#   where it goes in the output, or both)
# the DiT sees them as a JSON list on a 0-1000 canvas, appended to the caption. this returns that list.
def flux3_box_rows(boxes: list[dict]) -> list[dict]:
    def to_canvas(box):
        return [round(min(max(float(v), 0.0), 1.0) * 1000) for v in box] if box else None

    rows = []
    for i, box in enumerate(boxes):
        row_id = str(box.get("id") or f"obj_{i + 1}")
        desc = str(box.get("desc") or "")
        if box.get("bbox"):
            rows.append(dict(id=row_id, bbox=to_canvas(box["bbox"]), desc=desc))
        else:
            assert box.get("src_bbox") or box.get("tgt_bbox"), f"box {i} has neither bbox nor src_bbox/tgt_bbox"
            src, tgt = to_canvas(box.get("src_bbox")), to_canvas(box.get("tgt_bbox"))
            rows.append(
                {"id": row_id, "from": str(box.get("from") or ""), "src_bbox": src, "tgt_bbox": tgt, "desc": desc}
            )
    return rows


# caption for the DiT without the upsampler
def flux3_box_prompt(prompt: str, boxes: list[dict]) -> str:
    return f"{prompt.rstrip()} {json.dumps(flux3_box_rows(boxes), ensure_ascii=False)}"


_BOX_ID = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


# 0-1000 canvas box, or None
def _valid_box(box) -> list | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
        return None
    y1, x1, y2, x2 = box
    if y1 < 0 or x1 < 0 or y2 <= y1 or x2 <= x1 or y2 > 1000 or x2 > 1000 or max(box) <= 1:
        return None
    return list(box)


# a row the upsampler added: kept only if it is well formed and cited as <id> in the caption
def _valid_added_row(row, caption: str) -> dict | None:
    if not isinstance(row, dict):
        return None
    row_id, desc = row.get("id"), row.get("desc")
    if not isinstance(row_id, str) or not _BOX_ID.fullmatch(row_id) or f"<{row_id}>" not in caption:
        return None
    if not isinstance(desc, str) or not desc.strip():
        return None
    if (box := _valid_box(row.get("bbox"))) is not None:
        return dict(id=row_id, bbox=box, desc=desc.strip())

    origin, src, tgt = row.get("from"), row.get("src_bbox"), row.get("tgt_bbox")
    src_box = _valid_box(src) if src is not None else None
    tgt_box = _valid_box(tgt) if tgt is not None else None
    if not isinstance(origin, str) or not re.fullmatch(r"ref_image_\d+|target", origin):
        return None
    if (src is not None and src_box is None) or (tgt is not None and tgt_box is None):
        return None
    if (src_box is None and tgt_box is None) or (origin == "target") == (src_box is not None):
        return None
    return {"id": row_id, "from": origin, "src_bbox": src_box, "tgt_bbox": tgt_box, "desc": desc.strip()}


def _iou(a: list, b: list) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# the upsampler's answer is "<caption> [rows]". re-impose the user's (locked) rows:
# 1. a row re-planning a locked box under its own id (IoU >= 0.9) takes the locked id, also in the caption's <id>s.
# 2. locked rows are kept verbatim, adopting the upsampler's desc. its own rows follow if valid.
def _reconcile_boxes(text: str, locked: list[dict]) -> str:
    # rows: the last '[' from which the rest parses as a JSON list. descs may contain brackets.
    caption, rows = text.strip(), []
    for i in range(len(text) - 1, -1, -1):
        if text[i] == "[":
            try:
                parsed = json.loads(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, list):
                caption, rows = text[:i].rstrip(), parsed
                break

    def get_box(row):
        return row.get("bbox") or row.get("tgt_bbox") or row.get("src_bbox")

    locked_ids = {r["id"] for r in locked}
    row_ids = {r.get("id") for r in rows if isinstance(r, dict)}
    renamed = {}
    for lock in locked:
        lock_box = get_box(lock)
        if lock["id"] in row_ids or lock_box is None:
            continue
        for row in rows:
            if not isinstance(row, dict) or row.get("id") in renamed or row.get("id") in locked_ids:
                continue
            box = _valid_box(get_box(row))
            if box is not None and _iou(box, lock_box) >= 0.9:
                renamed[row["id"]] = lock["id"]
                break
    for row in rows:
        if isinstance(row, dict) and row.get("id") in renamed:
            row["id"] = renamed[row["id"]]
    for old, new in renamed.items():
        caption = caption.replace(f"<{old}>", f"<{new}>")

    by_id = {r.get("id"): r for r in rows if isinstance(r, dict) and isinstance(r.get("id"), str)}
    out = []
    for lock in locked:
        desc = by_id.get(lock["id"], {}).get("desc")
        out.append({**lock, "desc": desc.strip() if isinstance(desc, str) and desc.strip() else lock["desc"]})
    for row in rows:
        added = _valid_added_row(row, caption)
        if added is not None and added["id"] not in {r["id"] for r in out}:
            out.append(added)
    return f"{caption} {json.dumps(out, ensure_ascii=False)}"


# https://github.com/black-forest-labs/flux-3-image-klein/blob/bfc8889/src/flux3/sampling.py#L9
# NOTE: we follow the reference code (alpha=16). its config.json, diffusers and ComfyUI use alpha=6.93.
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
