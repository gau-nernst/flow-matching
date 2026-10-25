# https://github.com/Tongyi-MAI/Z-Image/blob/2151737e/src/zimage/pipeline.py

from collections.abc import Callable

import torch
from torch import Tensor, nn
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen3Model

from ..registry import MODELS
from ..solvers import get_solver
from ..vae import load_vae
from .model import ZImage, load_zimage


class ZImagePipeline:
    def __init__(
        self,
        zimage: ZImage | None = None,
        text_encoder_id: str = "Qwen/Qwen3-4B",
        sample_defaults: dict | None = None,  # used by generate(). default: z-image-turbo
    ) -> None:
        self.model = zimage or load_zimage()
        self.sample_defaults = sample_defaults or MODELS["z-image-turbo"].sample_defaults
        self.ae = load_vae("flux1").bfloat16()

        # text stuff
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_id)
        self.qwen3 = Qwen3Model.from_pretrained(text_encoder_id, dtype="auto")
        # prune unneeded parts
        self.qwen3.layers = self.qwen3.layers[:-1]
        self.qwen3.norm = nn.Identity()

    def cpu(self):
        for m in (self.ae, self.qwen3, self.model):
            m.cpu()
        return self

    def cuda(self):
        for m in (self.ae, self.qwen3, self.model):
            m.cuda()
        return self

    @classmethod
    def load(cls, name: str):
        spec = MODELS[name]
        return cls(load_zimage(name), spec.text_encoder_id, sample_defaults=spec.sample_defaults)

    def embed_text(self, prompts: str | list[str]) -> Tensor:
        if isinstance(prompts, str):
            prompts = [prompts]

        formatted_prompts = [
            self.tokenizer.apply_chat_template(
                [dict(role="user", content=prompt)],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for prompt in prompts
        ]
        inputs = self.tokenizer(
            formatted_prompts,
            # padding="max_length",
            padding="longest",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )

        device = next(self.qwen3.parameters()).device
        input_ids = inputs.input_ids.to(device)
        mask = inputs.attention_mask.to(device).bool()
        txt_embeds = self.qwen3(input_ids, attention_mask=mask, output_hidden_states=True).last_hidden_state

        return txt_embeds

        # remove padding
        # TODO: use varlen attention instead
        # return [txt_embeds[i, mask[i]] for i in range(len(formatted_prompts))]

    # returns (latents, model_kwargs) for training: model(noisy_latents, t, **model_kwargs)
    @torch.no_grad()
    def prepare_train_inputs(self, imgs: Tensor, prompts: list[str]) -> tuple[Tensor, dict]:
        latents = self.ae.encode(imgs)
        txt = self.embed_text(prompts)
        return latents, dict(txt=txt)

    # cfg_scale, num_steps and time_shift default to self.sample_defaults
    @torch.no_grad()
    def generate(
        self,
        prompt: str | list[str],
        neg_prompt: str | list[str] = "",
        img_size: int | tuple[int, int] = 512,
        latents: Tensor | None = None,
        denoise: float = 1.0,
        cfg_scale: float | None = None,
        num_steps: int | None = None,
        time_shift: float | None = None,
        seed: int | None = None,
        solver: str = "euler",
        pbar: bool = False,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> Tensor:
        cfg_scale = self.sample_defaults["cfg_scale"] if cfg_scale is None else cfg_scale
        num_steps = self.sample_defaults["num_steps"] if num_steps is None else num_steps
        time_shift = self.sample_defaults["time_shift"] if time_shift is None else time_shift
        txt_embeds = self.embed_text(prompt)
        neg_txt_embeds = self.embed_text(neg_prompt) if cfg_scale != 1.0 else None

        bsize = txt_embeds.shape[0]
        if isinstance(img_size, int):
            height = width = img_size
        else:
            height, width = img_size

        ae = self.ae
        shape = (bsize, height // ae.downsample, width // ae.downsample, self.ae.cfg.z_dim)
        device = ae.encoder.conv_in.weight.device

        # keep latents in FP32 for accurate .lerp()
        rng = torch.Generator(device).manual_seed(seed) if seed is not None else None
        noise = torch.randn(shape, device=device, dtype=torch.float32, generator=rng)

        # this is basically SDEdit https://arxiv.org/abs/2108.01073
        latents = latents.float().lerp(noise, denoise) if latents is not None else noise

        # static shift
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1)
        timesteps = time_shift / (time_shift + 1 / timesteps - 1)
        timesteps = timesteps.tolist()

        latents = zimage_generate(
            self.model,
            latents,
            timesteps,
            txt=txt_embeds,
            neg_txt=neg_txt_embeds,
            cfg_scale=cfg_scale,
            solver=solver,
            pbar=pbar,
            progress_cb=progress_cb,
        )
        return self.ae.decode(latents, uint8=True)


@torch.no_grad()
def zimage_generate(
    zimage: ZImage,
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

    for i in tqdm(range(num_steps), disable=not pbar, dynamic_ncols=True):
        t = torch.tensor([timesteps[i]], device="cuda")
        v = zimage(latents, t, txt).float()

        # classifier-free guidance
        if cfg_scale != 1.0:
            assert neg_txt is not None
            neg_v = zimage(latents, t, neg_txt).float()
            v = neg_v.lerp(v, cfg_scale)

        latents = solver_.step(latents, v, i)
        if progress_cb is not None:
            progress_cb(i + 1, num_steps)

    return latents
