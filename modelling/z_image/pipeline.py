# https://github.com/Tongyi-MAI/Z-Image/blob/2151737e/src/zimage/pipeline.py

from collections.abc import Callable

import torch
from torch import Tensor, nn
from tqdm import tqdm
from transformers import AutoTokenizer

from ..qwen3 import Qwen3ForCausalLM
from ..registry import MODELS
from ..solvers import get_solver
from ..vae import load_vae
from ..varlen import VarlenMetadata
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
        self.qwen3 = Qwen3ForCausalLM.from_pretrained(text_encoder_id).model
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

    # one unpadded [L_i, D] per prompt
    def embed_text(self, prompts: str | list[str]) -> list[Tensor]:
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
        input_ids = self.tokenizer(formatted_prompts, max_length=512, truncation=True).input_ids
        lens = [len(x) for x in input_ids]

        device = next(self.qwen3.parameters()).device
        varlen = VarlenMetadata.from_lens(lens, device) if len(lens) > 1 else None
        input_ids = torch.tensor(sum(input_ids, [])).pin_memory().to(device, non_blocking=True)
        input_ids = input_ids if varlen is not None else input_ids[None]
        return list(self.qwen3(input_ids, varlen=varlen).flatten(0, -2).split(lens))

    # imgs: uint8 [H_i, W_i, 3]. returns per-sample latents and kwargs for model.prepare()
    @torch.no_grad()
    def prepare_train_inputs(self, imgs: list[Tensor], prompts: list[str]) -> tuple[list[Tensor], dict]:
        latents = [self.ae.encode(x[None])[0] for x in imgs]
        return latents, dict(txt=self.embed_text(prompts))

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
        bsize = len(txt_embeds)
        neg_txt_embeds = self.embed_text(neg_prompt) if cfg_scale != 1.0 else None
        if neg_txt_embeds is not None and len(neg_txt_embeds) == 1:
            neg_txt_embeds = neg_txt_embeds * bsize
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
    txt: list[Tensor],
    neg_txt: list[Tensor] | None = None,
    cfg_scale: float = 1.0,
    solver: str = "euler",
    pbar: bool = False,
    progress_cb: Callable[[int, int], None] | None = None,
) -> Tensor:
    num_steps = len(timesteps) - 1
    solver_ = get_solver(solver, timesteps)
    timesteps_tensor = torch.tensor(timesteps).pin_memory().to("cuda", non_blocking=True)

    # pack, since prompts have different lengths
    imgs = list(latents)
    sizes = [x.shape[:2] for x in imgs]
    latents, pos_kwargs = zimage.prepare(imgs, txt)
    neg_kwargs = zimage.prepare(imgs, neg_txt)[1] if cfg_scale != 1.0 else None

    for i in tqdm(range(num_steps), disable=not pbar, dynamic_ncols=True):
        t = timesteps_tensor[None, i]
        v = zimage(latents, t, **pos_kwargs).float()

        # classifier-free guidance
        if cfg_scale != 1.0:
            assert neg_kwargs is not None
            neg_v = zimage(latents, t, **neg_kwargs).float()
            v = neg_v.lerp(v, cfg_scale)

        latents = solver_.step(latents, v, i)
        if progress_cb is not None:
            progress_cb(i + 1, num_steps)

    return torch.stack(zimage.unpatchify(latents, sizes))
