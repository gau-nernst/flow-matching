# https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/blob/f972bd4/transformer/pipeline_moss_image.py

from collections.abc import Callable

import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from ..qwen3_5 import Qwen3_5Model
from ..solvers import get_solver
from ..vae import load_vae
from .model import MOSSImage, load_moss_image


class MOSSImageTextEncoder:
    def __init__(self, repo_id: str = "OpenMOSS-Team/MOSS-Image-SFT", fp8: bool = True) -> None:
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder="tokenizer")
        self.model = Qwen3_5Model.from_pretrained(repo_id, "text_encoder", prefix="language_model.").requires_grad_(
            False
        )
        self.fp8 = fp8
        self.max_length = 4096

    def cpu(self):
        self.model.cpu()
        return self

    def cuda(self):
        self.model.cuda()
        # quantize on GPU since it's much faster
        if self.fp8 and not self.model.layers[0].mlp.w2.is_fp8_1d2d():
            self.model.to_fp8()
        return self

    # official encodes each prompt separately without padding. returns the last hidden state for each prompt.
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
    def __init__(self, moss: MOSSImage | None = None, text_encoder_fp8: bool = True) -> None:
        # apply_rope() takes the slow reference path when params require grad
        self.moss = (moss or load_moss_image()).requires_grad_(False)
        self.ae = load_vae("flux2_small_decoder").bfloat16()  # official runs the VAE in FP32
        self.text_encoder = MOSSImageTextEncoder(fp8=text_encoder_fp8)

    def cpu(self):
        for m in (self.moss, self.ae, self.text_encoder):
            m.cpu()
        return self

    def cuda(self):
        for m in (self.moss, self.ae, self.text_encoder):
            m.cuda()
        return self

    # all prompts must have the same token length, since the DiT doesn't support padding.
    # by default, the negative branch of CFG skips block 22.
    @torch.no_grad()
    def generate(
        self,
        prompt: str | list[str],
        neg_prompt: str = "",
        img_size: int | tuple[int, int] = 1024,
        cfg_scale: float = 3.0,
        num_steps: int = 50,
        shift: float = 5.0,
        uncond_skip_layers: tuple[int, ...] = (22,),
        seed: int | None = None,
        solver: str = "euler",
        pbar: bool = False,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> Tensor:
        txt = self.text_encoder(prompt)
        assert all(x.shape == txt[0].shape for x in txt), "all prompts must have the same token length"
        txt = torch.stack(txt)
        bsize = txt.shape[0]
        neg_txt = self.text_encoder(neg_prompt)[0].expand(bsize, -1, -1) if cfg_scale != 1.0 else None

        height, width = (img_size, img_size) if isinstance(img_size, int) else img_size
        ae = self.ae
        H = height // ae.downsample // 2
        W = width // ae.downsample // 2
        device = ae.encoder.conv_in.weight.device

        # same as official: FP32 noise in NCHW layout, sampled on GPU
        rng = torch.Generator(device).manual_seed(seed) if seed is not None else None
        noise = torch.randn(bsize, ae.cfg.z_dim * 4, H, W, device=device, generator=rng)
        noise = noise.permute(0, 2, 3, 1)

        latents = moss_image_generate(
            moss=self.moss,
            latents=noise,
            timesteps=moss_image_timesteps(num_steps, shift),
            txt=txt,
            neg_txt=neg_txt,
            cfg_scale=cfg_scale,
            uncond_skip_layers=uncond_skip_layers,
            solver=solver,
            pbar=pbar,
            progress_cb=progress_cb,
        )
        return ae.decode(latents, uint8=True)


# official: shifted grid, then snapped to the nearest of the 1000 training timesteps (shift=1). ends with 0.
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
    txt: Tensor,
    neg_txt: Tensor | None = None,
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
