# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/bria_fibo2/pipeline_bria_fibo2.py

import json
import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer

from ..qwen3 import Qwen3Model
from ..solvers import get_solver
from ..utils import load_hf_state_dict
from ..vae import load_vae
from .model import Fibo, load_fibo

# the structured caption with every value empty, which Fibo 2.0 was trained with
DEFAULT_NEG_PROMPT = (
    '{"short_description":"","objects":[],"background_setting":"","lighting":{},'
    '"aesthetics":{},"photographic_characteristics":{},"style_medium":"",'
    '"context":"","artistic_style":""}'
)


# text-only Qwen3-VL-4B. Fibo 2.0's text encoder is the stock model, so we can use the official FP8 weights.
class FiboTextEncoder(nn.Module):
    # HF hidden_states indices behind the DiT's text inputs: the Perceiver's, then one per injection block.
    # each bundle concatenates 3 layers, each normalized to unit RMS.
    BUNDLES = ((9, 20, 31), (5, 16, 27), (7, 18, 29), (9, 20, 31), (11, 22, 33), (13, 24, 35))

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-4B-Instruct-FP8") -> None:
        super().__init__()
        self.layer_ids = sorted({i for bundle in self.BUNDLES for i in bundle})
        self.max_length = 4096

        # hidden_states[i] in HF is the output of layer i-1. only keep layers we need.
        cfg = AutoConfig.from_pretrained(model_id).text_config
        cfg.num_hidden_layers = max(self.layer_ids)
        with torch.device("meta"):
            self.model = Qwen3Model(cfg)

        state_dict = load_hf_state_dict(model_id, "model.safetensors.index.json", prefix="model.language_model.")
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith("layers.") or int(k.split(".")[1]) < cfg.num_hidden_layers
        }
        self.model.load_state_dict(state_dict, assign=True)
        self.model.eval().requires_grad_(False)

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id)

    # each prompt is encoded separately without padding. returns [6, L, 7680] for each prompt.
    @torch.no_grad()
    def forward(self, texts: str | list[str]) -> list[Tensor]:
        if isinstance(texts, str):
            texts = [texts]

        device = self.model.embed_tokens.weight.device
        outputs = []
        for text in texts:
            text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
            input_ids = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")
            hidden_states = self.model(input_ids.input_ids.to(device), output_indices=[i - 1 for i in self.layer_ids])
            hidden_states = {
                i: F.rms_norm(h[0].float(), h.shape[-1:], eps=1e-6).bfloat16()
                for i, h in zip(self.layer_ids, hidden_states)
            }
            bundles = [torch.cat([hidden_states[i] for i in bundle], dim=-1) for bundle in self.BUNDLES]
            outputs.append(torch.stack(bundles))
        return outputs


# training captions hold no empty values, and no aesthetic_score / preference_score. a caption of exactly a
# short_description and a text_render list keeps both, even when the list is empty (it says the image has no text).
# a prompt that is not a JSON object is returned as is.
def fibo_clean_prompt(prompt: str) -> str:
    try:
        caption = json.loads(prompt)
    except json.JSONDecodeError:
        return prompt
    if not isinstance(caption, dict):
        return prompt

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
    return json.dumps(out, separators=(",", ":"), ensure_ascii=False)


class FiboPipeline:
    # fp8: online block FP8 for the DiT, quantized on GPU in .cuda(). the text encoder uses official FP8 weights.
    def __init__(
        self, fibo: Fibo | None = None, text_encoder_id: str = "Qwen/Qwen3-VL-4B-Instruct-FP8", fp8: bool = True
    ) -> None:
        # apply_rope() takes the slow reference path when params require grad
        self.fibo = (fibo or load_fibo()).requires_grad_(False)
        self.fp8 = fp8
        self.ae = load_vae("flux2").bfloat16()  # same weights as Flux.2. official runs the VAE in FP32
        self.text_encoder = FiboTextEncoder(text_encoder_id)

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
    # all prompts must have the same token length, since the DiT doesn't support padding.
    # NOTE: we don't show rendered images of text_render entries to the text encoder (official does it by default).
    @torch.no_grad()
    def generate(
        self,
        prompt: str | list[str],
        neg_prompt: str = DEFAULT_NEG_PROMPT,
        img_size: int | tuple[int, int] = 1024,
        cfg_scale: float = 5.0,
        num_steps: int = 30,
        seed: int | None = None,
        solver: str = "euler",
        pbar: bool = False,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> Tensor:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        txt = self.text_encoder([fibo_clean_prompt(p) for p in prompt])
        assert all(x.shape == txt[0].shape for x in txt), "all prompts must have the same token length"
        txt = torch.stack(txt)
        bsize = txt.shape[0]
        neg_txt = self.text_encoder(neg_prompt)[0].expand(bsize, -1, -1, -1) if cfg_scale != 1.0 else None

        height, width = (img_size, img_size) if isinstance(img_size, int) else img_size
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
        v = fibo(latents, t, txt).float()

        # classifier-free guidance
        if cfg_scale != 1.0:
            neg_v = fibo(latents, t, neg_txt).float()
            v = neg_v.lerp(v, cfg_scale)

        latents = solver_.step(latents, v, i)
        if progress_cb is not None:
            progress_cb(i + 1, num_steps)

    return latents
