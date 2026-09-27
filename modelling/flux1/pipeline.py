import math

import torch
from torch import Tensor
from tqdm import tqdm

from ..offload import PerLayerOffloadCUDAStream
from ..solvers import get_solver
from ..text_embedder import load_clip_l, load_t5
from ..vae import AutoEncoder, load_vae
from .model import Flux1, load_flux1


class Flux1TextEmbedder:
    def __init__(self, offload_t5: bool = False) -> None:
        self.t5 = load_t5()  #  9.5 GB in BF16
        self.clip = load_clip_l().bfloat16()  # 246 MB in BF16
        self.t5_offloader = PerLayerOffloadCUDAStream(self.t5, enable=offload_t5)

    def cpu(self):
        for m in (self.clip, self.t5_offloader):
            m.cpu()
        return self

    def cuda(self):
        for m in (self.clip, self.t5_offloader):
            m.cuda()
        return self

    def __call__(self, prompt: list[str]):
        return self.t5(prompt), self.clip(prompt)  # (B, 512, 4096) and (B, 768)


def prepare_inputs(
    ae: AutoEncoder,
    text_embedder: Flux1TextEmbedder,
    prompt: str | list[str],
    negative_prompt: str | list[str] | None = None,
    img_size: int | tuple[int, int] = 512,
    latents: Tensor | None = None,
    denoise: float = 1.0,
    seed: int | None = None,
):
    embeds, vecs = text_embedder(prompt)
    neg_embeds, neg_vecs = text_embedder(negative_prompt) if negative_prompt is not None else (None, None)

    bsize = embeds.shape[0]
    if isinstance(img_size, int):
        height = width = img_size
    else:
        height, width = img_size
    shape = (bsize, ae.cfg.z_dim, height // ae.downsample, width // ae.downsample)
    device = ae.encoder.conv_in.weight.device

    # keep latents in FP32 for accurate .lerp()
    rng = torch.Generator(device).manual_seed(seed) if seed is not None else None
    noise = torch.randn(shape, device=device, dtype=torch.float32, generator=rng)

    # this is basically SDEdit https://arxiv.org/abs/2108.01073
    latents = latents.float().lerp(noise, denoise) if latents is not None else noise

    return latents, embeds, vecs, neg_embeds, neg_vecs


class Flux1Pipeline:
    def __init__(self, flux: Flux1 | None = None, offload_flux: bool = False, offload_t5: bool = False) -> None:
        self.flux = flux or load_flux1()  # 23.8 GB in BF16
        self.ae = load_vae("flux1").bfloat16()  # 168 MB in BF16
        self.text_embedder = Flux1TextEmbedder(offload_t5)

        # autoencoder and clip are small, don't need to offload
        self.flux_offloader = PerLayerOffloadCUDAStream(self.flux, enable=offload_flux)

    def cpu(self):
        for m in (self.ae, self.text_embedder, self.flux_offloader):
            m.cpu()
        return self

    def cuda(self):
        for m in (self.ae, self.text_embedder, self.flux_offloader):
            m.cuda()
        return self

    @torch.no_grad()
    def generate(
        self,
        prompt: str | list[str],
        neg_prompt: str | list[str] = "",
        img_size: int | tuple[int, int] = 512,
        latents: Tensor | None = None,
        denoise: float = 1.0,
        guidance: Tensor | float | None = 3.5,
        cfg_scale: float = 1.0,
        num_steps: int = 50,
        seed: int | None = None,
        solver: str = "euler",
        pbar: bool = False,
    ) -> Tensor:
        latents, embeds, vecs, neg_embeds, neg_vecs = prepare_inputs(
            self.ae,
            self.text_embedder,
            prompt,
            neg_prompt if cfg_scale != 1.0 else None,
            img_size,
            latents,
            denoise,
            seed,
        )

        # denoise from t=1 (noise) to t=0 (latents)
        # divide by 4 due to FLUX's input patchification
        height, width = latents.shape[-2:]
        timesteps = flux1_timesteps(denoise, 0.0, num_steps, height * width // 4)

        latents = flux1_generate(
            flux=self.flux,
            latents=latents,
            timesteps=timesteps,
            txt=embeds,
            vec=vecs,
            neg_txt=neg_embeds,
            neg_vec=neg_vecs,
            guidance=guidance,
            cfg_scale=cfg_scale,
            solver=solver,
            pbar=pbar,
        )
        latents = latents.permute(0, 2, 3, 1)  # NCHW -> NHWC
        return self.ae.decode(latents, uint8=True)


def flux1_timesteps(start: float = 1.0, end: float = 0.0, num_steps: int = 50, img_seq_len: int = 0):
    # only dev version has time shift
    timesteps = torch.linspace(start, end, num_steps + 1)
    timesteps = flux1_time_shift(timesteps, img_seq_len)
    return timesteps.tolist()


# https://arxiv.org/abs/2403.03206
# Section 5.3.2 - Resolution-dependent shifting of timesteps schedules
# https://github.com/black-forest-labs/flux/blob/805da8571a0b49b6d4043950bd266a65328c243b/src/flux/sampling.py#L222-L238
def flux1_time_shift(timesteps: Tensor | float, img_seq_len: int, base_shift: float = 0.5, max_shift: float = 1.15):
    m = (max_shift - base_shift) / (4096 - 256)
    b = base_shift - m * 256
    mu = m * img_seq_len + b

    exp_mu = math.exp(mu)  # this is (m/n) in Equation (23)
    return exp_mu / (exp_mu + 1 / timesteps - 1)


@torch.no_grad()
def flux1_generate(
    flux: Flux1,
    latents: Tensor,
    timesteps: list[float],
    txt: Tensor,
    vec: Tensor,
    neg_txt: Tensor | None = None,
    neg_vec: Tensor | None = None,
    guidance: Tensor | float | None = 3.5,
    cfg_scale: float = 1.0,
    solver: str = "euler",
    pbar: bool = False,
) -> Tensor:
    B, _, H, W = latents.shape

    if isinstance(guidance, (int, float)):
        guidance = torch.full((B,), guidance, device=latents.device)

    num_steps = len(timesteps) - 1
    solver_ = get_solver(solver, timesteps)
    rope = flux.make_rope(H // 2, W // 2, txt.shape[1])

    for i in tqdm(range(num_steps), disable=not pbar, dynamic_ncols=True):
        t = torch.tensor([timesteps[i]], device="cuda")
        v = flux(latents, t, txt, vec, guidance, rope).float()

        # classifier-free guidance
        if cfg_scale != 1.0:
            assert neg_txt is not None and neg_vec is not None
            neg_v = flux(latents, t, neg_txt, neg_vec, guidance, rope).float()
            v = neg_v.lerp(v, cfg_scale)

        latents = solver_.step(latents, v, i)

    return latents
