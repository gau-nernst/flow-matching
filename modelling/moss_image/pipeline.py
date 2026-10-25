# https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/blob/f972bd4/transformer/pipeline_moss_image.py

from collections.abc import Callable

import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from ..qwen3_5 import Qwen3_5Model
from ..registry import MODELS
from ..solvers import get_solver
from ..vae import load_vae
from .model import MOSSImage, load_moss_image


class MOSSImageTextEncoder:
    def __init__(self, model_id: str = "Qwen/Qwen3.5-9B") -> None:
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = Qwen3_5Model.from_pretrained(model_id)
        self.max_length = 4096

    def cpu(self):
        self.model.cpu()
        return self

    def cuda(self):
        self.model.cuda()
        return self

    # encode each prompt separately, without padding
    @torch.no_grad()
    def __call__(self, texts: str | list[str]) -> list[Tensor]:
        if isinstance(texts, str):
            texts = [texts]

        device = self.model.embed_tokens.weight.device
        outputs = []
        for text in texts:
            text = self.tokenizer.apply_chat_template(
                [dict(role="user", content=text)], tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
            input_ids = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt").input_ids
            outputs.append(self.model(input_ids.to(device))[0])
        return outputs


class MOSSImagePipeline:
    def __init__(
        self,
        moss: MOSSImage | None = None,
        text_encoder_id: str = "Qwen/Qwen3.5-9B",
        sample_defaults: dict | None = None,  # used by generate(). default: moss-image-sft
    ) -> None:
        self.model = moss or load_moss_image()
        self.sample_defaults = sample_defaults or MODELS["moss-image-sft"].sample_defaults
        self.ae = load_vae("flux2_small").bfloat16()  # reference runs the VAE in FP32
        self.text_encoder = MOSSImageTextEncoder(text_encoder_id)

    def cpu(self):
        for m in (self.model, self.ae, self.text_encoder):
            m.cpu()
        return self

    def cuda(self):
        for m in (self.model, self.ae, self.text_encoder):
            m.cuda()
        return self

    @classmethod
    def load(cls, name: str):
        spec = MODELS[name]
        return cls(load_moss_image(name), spec.text_encoder_id, sample_defaults=spec.sample_defaults)

    # imgs: list of uint8 [H_i, W_i, 3], can have different sizes.
    # returns (latents, model_kwargs) for training: model(noisy_latents, t, **model_kwargs), all per-sample lists
    @torch.no_grad()
    def prepare_train_inputs(self, imgs: list[Tensor], prompts: list[str]) -> tuple[list[Tensor], dict]:
        latents = [self.ae.encode(x[None])[0] for x in imgs]
        return latents, dict(txt=self.text_encoder(prompts))

    # cfg_scale, num_steps, shift and uncond_skip_layers default to self.sample_defaults
    @torch.no_grad()
    def generate(
        self,
        prompt: str | list[str],
        neg_prompt: str = "",
        img_size: int | tuple[int, int] = 1024,
        cfg_scale: float | None = None,
        num_steps: int | None = None,
        shift: float | None = None,
        uncond_skip_layers: tuple[int, ...] | None = None,
        seed: int | None = None,
        solver: str = "euler",
        pbar: bool = False,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> Tensor:
        cfg_scale = self.sample_defaults["cfg_scale"] if cfg_scale is None else cfg_scale
        num_steps = self.sample_defaults["num_steps"] if num_steps is None else num_steps
        shift = self.sample_defaults["shift"] if shift is None else shift
        if uncond_skip_layers is None:
            uncond_skip_layers = self.sample_defaults["uncond_skip_layers"]
        txts = self.text_encoder(prompt)
        neg_txts = self.text_encoder(neg_prompt) * len(txts) if cfg_scale != 1.0 else None

        height, width = (img_size, img_size) if isinstance(img_size, int) else img_size
        ae = self.ae
        H = height // ae.downsample // 2
        W = width // ae.downsample // 2
        device = ae.encoder.conv_in.weight.device
        rng = torch.Generator(device).manual_seed(seed) if seed is not None else None

        # sample noise per image, so a seed gives the same image regardless of batch size
        noise = torch.cat([torch.randn(1, H, W, ae.cfg.z_dim * 4, device=device, generator=rng) for _ in txts])
        latents = moss_image_generate(
            moss=self.model,
            latents=noise,
            timesteps=moss_image_timesteps(num_steps, shift),
            txt=txts,
            neg_txt=neg_txts,
            cfg_scale=cfg_scale,
            uncond_skip_layers=uncond_skip_layers,
            solver=solver,
            pbar=pbar,
            progress_cb=progress_cb,
        )
        return ae.decode(latents, uint8=True)


# shifted grid, snapped to the nearest training timestep. ends with 0
def moss_image_timesteps(num_steps: int, shift: float = 5.0, num_train_timesteps: int = 1000) -> list[float]:
    def grid(count: int, shift: float):
        values = torch.linspace(1, shift / (num_train_timesteps + shift - 1), count)
        return shift * values / (1 + (shift - 1) * values)

    sigmas = grid(num_steps, shift)
    training = grid(num_train_timesteps, 1.0).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=num_train_timesteps - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
    return torch.unique_consecutive(training[nearest]).tolist() + [0.0]


@torch.no_grad()
def moss_image_generate(
    moss: MOSSImage,
    latents: Tensor,
    timesteps: list[float],
    txt: list[Tensor],
    neg_txt: list[Tensor] | None = None,
    cfg_scale: float = 1.0,
    uncond_skip_layers: tuple[int, ...] = (),
    solver: str = "euler",
    pbar: bool = False,
    progress_cb: Callable[[int, int], None] | None = None,
) -> Tensor:
    num_steps = len(timesteps) - 1
    solver_ = get_solver(solver, timesteps)
    B = latents.shape[0]

    for i in tqdm(range(num_steps), disable=not pbar, dynamic_ncols=True):
        t = torch.tensor([timesteps[i]], device=latents.device).expand(B)
        v = moss(latents, t, txt).float()

        # classifier-free guidance
        if cfg_scale != 1.0:
            neg_v = moss(latents, t, neg_txt, skip_layers=uncond_skip_layers).float()
            v = neg_v.lerp(v, cfg_scale)

        latents = solver_.step(latents, v, i)
        if progress_cb is not None:
            progress_cb(i + 1, num_steps)

    return latents
