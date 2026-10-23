# https://github.com/black-forest-labs/flux-3-image-klein/blob/bfc8889/src/flux3/pup.py
# Prompt upsampler (pup), a Qwen3.5 VLM, served by vLLM in a separate env (vLLM pins its own torch/transformers).
# Bounding-box prompts are not supported. Example server for a single 32GB GPU, shared with Flux3 FP8:
#   vllm serve black-forest-labs/flux-3-image-klein-pup --served-model-name pup --quantization fp8 \
#     --max-model-len 8192 --max-num-seqs 1 --kv-cache-memory-bytes 512M --no-enable-prefix-caching \
#     --limit-mm-per-prompt '{"image": 10}' --generation-config vllm --port 8001

import base64
import io
import json
import math
import urllib.request

from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps

I2I_SCAFFOLD = (
    "Reference images follow in order (the first image, the second image, ...).\n"
    "User request: {instruction}\n"
    "Output the caption."
)


# official: RGB over white, LANCZOS resize to a pixel budget (depends on the number of references) with sides
# a multiple of 32, then JPEG q92
def pup_prepare_reference(img: Image.Image, num_refs: int) -> str:
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


class Flux3Upsampler:
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
                dict(type="image_url", image_url=dict(url=pup_prepare_reference(img, len(ref_imgs))))
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
