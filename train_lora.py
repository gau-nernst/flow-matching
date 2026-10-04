import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

# improve memory usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# by default, torch.compile cache are written to /tmp/torchinductor_username,
# which does not persist across restarts. write to local dir for persistence.
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(Path(__file__).parent / "torchinductor")

import pandas as pd
import psutil
import torch
import wandb
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, IterableDataset, default_collate, get_worker_info
from tqdm import tqdm

from modelling import AutoEncoder, Flux1
from modelling.flux1.pipeline import Flux1TextEmbedder, flux1_generate, flux1_timesteps
from time_sampler import LogitNormal, Uniform
from train_utils import EMA, compute_loss, parse_img_size, random_resize, setup_model

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ImageDataset(IterableDataset):
    def __init__(
        self,
        meta_path: str,
        img_dir: str,
        min_size: int,
        max_size: int,
        batch_size: int,
        drop_tag_rate: float = 0.0,
    ) -> None:
        self.img_dir = Path(img_dir)
        self.min_size = min_size
        self.max_size = max_size
        self.batch_size = batch_size
        self.drop_tag_rate = drop_tag_rate

        if str(meta_path).endswith(".tsv"):
            meta = pd.read_csv(meta_path, sep="\t")
        else:
            meta = pd.read_csv(meta_path)
        self.filenames = meta["filename"].tolist()
        self.prompts = meta["prompt"].tolist()

        self.shuffle_rng = torch.Generator()
        self.shuffle_rng.seed()

    def __iter__(self):
        buckets = dict()
        counter = -1

        # NOTE: not handle distributed training for now
        worker_info = get_worker_info()
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        else:
            num_workers = 1
            worker_id = 0

        while True:
            # indices sequence is identical across workers
            indices = torch.randperm(len(self.filenames), generator=self.shuffle_rng)
            for idx in indices:
                counter = (counter + 1) % num_workers
                if counter != worker_id:
                    continue

                # torchvision 0.20 has memory leak when decoding webp. use PIL for robustness
                img_path = self.img_dir / self.filenames[idx]
                img_pil = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
                img_pt = random_resize(img_pil, self.min_size, self.max_size)

                prompt = self.prompts[idx]
                if self.drop_tag_rate > 0.0 and torch.rand(()).item() < self.drop_tag_rate:
                    SEPARATOR = ", "
                    tags = prompt.split(SEPARATOR)
                    if len(tags) > 5:
                        tags.pop(torch.randint(len(tags), size=()).item())
                        prompt = SEPARATOR.join(tags)

                img_size = tuple(img_pt.shape[-2:])
                if img_size not in buckets:
                    buckets[img_size] = []
                buckets[img_size].append((img_pt, prompt))

                if len(buckets[img_size]) == self.batch_size:
                    yield default_collate(buckets[img_size])
                    buckets[img_size] = []


@torch.no_grad()
def save_images(
    model: Flux1,
    ae: AutoEncoder,
    text_embedder: Flux1TextEmbedder,
    prompt_path: str,
    save_dir: Path,
    img_size: tuple[int, int],
    batch_size: int = 4,
):
    model.eval()
    prompts = [line.rstrip() for line in open(prompt_path, encoding="utf-8")]
    save_dir.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator("cuda").manual_seed(2024)

    neg_embeds, neg_vecs = text_embedder([""])

    for offset in tqdm(range(0, len(prompts), batch_size), "Generating images", dynamic_ncols=True):
        embeds, vecs = text_embedder(prompts[offset : offset + batch_size])

        shape = (embeds.shape[0], 16, img_size[0] // 8, img_size[1] // 8)
        noise = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=rng)

        if isinstance(model, Flux1):
            timesteps = flux1_timesteps(img_seq_len=shape[2] * shape[3] // 4)

            if len(model.double_blocks) == 19:  # flux-dev
                guidance_list = [(3.5, 1.0), (1.0, 3.5), (2.0, 2.0)]
            elif len(model.double_blocks) == 8:  # flex-alpha
                guidance_list = [(3.5, 1.0), (None, 3.5), (2.0, 2.0)]
            else:
                raise ValueError

            for guidance, cfg_scale in guidance_list:
                latents = flux1_generate(
                    model,
                    noise,
                    timesteps,
                    embeds,
                    vecs,
                    neg_embeds.expand_as(embeds),
                    neg_vecs.expand_as(vecs),
                    guidance=guidance,
                    cfg_scale=cfg_scale,
                )
                latents = latents.permute(0, 2, 3, 1)  # NCHW -> NHWC
                imgs = ae.decode(latents, uint8=True).cpu()
                for img_idx in range(imgs.shape[0]):
                    save_path = save_dir / f"{offset + img_idx:04d}_{guidance}-{cfg_scale}.webp"
                    Image.fromarray(imgs[img_idx].numpy()).save(save_path, lossless=True)

        else:
            raise ValueError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="flux-dev")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--min_size", type=int, default=512)
    parser.add_argument("--max_size", type=int, default=1024)
    parser.add_argument("--lora", type=int, default=8)
    parser.add_argument("--time_sampler", default="LogitNormal()")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--ema", action="store_true")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_ds", type=json.loads, required=True)
    parser.add_argument("--distill_ds", type=json.loads)
    parser.add_argument("--drop_tag_rate", type=float, default=0.0)

    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_accumulation", type=int, default=1)

    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--test_img_size", type=parse_img_size, default=(512, 512))
    parser.add_argument("--test_prompt_path", required=True)
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--run_name", default="Debug")
    parser.add_argument("--resume")
    args = parser.parse_args()

    assert args.batch_size % args.gradient_accumulation == 0
    batch_size = args.batch_size // args.gradient_accumulation
    torch._dynamo.config.cache_size_limit = 1000
    torch._dynamo.config.accumulated_cache_size_limit = 1000

    wandb.init(project="Flux finetune", name=args.run_name, dir="/tmp")

    def create_dloader(ds_config: dict, batch_size: int):
        ds = ImageDataset(
            ds_config["meta_path"], ds_config["img_dir"], args.min_size, args.max_size, batch_size, args.drop_tag_rate
        )
        dloader = DataLoader(ds, batch_size=None, num_workers=args.num_workers, pin_memory=True)
        return iter(dloader)

    if args.distill_ds is not None:
        assert batch_size % 2 == 0
        train_dloader = create_dloader(args.train_ds, batch_size // 2)
        distill_dloader = create_dloader(args.distill_ds, batch_size // 2)
    else:
        train_dloader = create_dloader(args.train_ds, batch_size)
        distill_dloader = None

    model, offloader, ae, text_embedder = setup_model(args.model, args.offload, args.lora, args.compile)
    ema = EMA(model) if args.ema else None
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True)
    logger.info(model)
    logger.info(f"No. of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logger.info(f"No. of non-trainable params: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
    time_sampler = eval(args.time_sampler, dict(Uniform=Uniform, LogitNormal=LogitNormal))

    log_dir = args.log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.run_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = log_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    img_dir = log_dir / "images"

    step = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True, mmap=True)

        if "model" in ckpt:  # training checkpoint
            logger.info("Load model: ", model.load_state_dict(ckpt["model"], strict=False))
            if ema is not None:
                ema.load_state_dict(ckpt["ema"])
            optim.load_state_dict(ckpt["optim"])
            step = ckpt["step"]

        else:  # model only
            logger.info("Load model: ", model.load_state_dict(ckpt, strict=False))

    # inference before any training
    if step == 0:
        save_images(model, ae, text_embedder, args.test_prompt_path, img_dir / f"step{step:06d}", args.test_img_size)

    pbar = tqdm(initial=step, total=args.num_steps, dynamic_ncols=True)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    time0 = time.perf_counter()
    while step < args.num_steps:
        for _ in range(args.gradient_accumulation):
            imgs, prompts = next(train_dloader)

            if distill_dloader is not None:
                distill_imgs, distill_prompts = next(distill_dloader)
                imgs = torch.cat([imgs, distill_imgs], dim=0)
                prompts = prompts + distill_prompts

            with torch.no_grad():
                # NCHW -> NHWC
                latents = ae.encode(imgs.permute(0, 2, 3, 1).cuda(), sample=True)

            if args.model == "flux-dev":
                # finetune at guidance=1.0 is better than at 3.5
                model_kwargs = dict(guidance=torch.full((imgs.shape[0],), 1.0, device="cuda", dtype=torch.bfloat16))
            else:
                model_kwargs = dict()

            loss = compute_loss(model, latents, *text_embedder(prompts), time_sampler, model_kwargs)
            with offloader.disable_forward_hook():
                loss.backward()

        if step % args.log_interval == 0:
            grad_norm = sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None) ** 0.5
            log_dict = dict(
                loss=loss.item(),
                grad_norm=grad_norm,
            )
            wandb.log(log_dict, step)

        optim.step()
        optim.zero_grad()
        if ema is not None:
            ema.update(step)
        step += 1
        pbar.update()

        if step % args.log_interval == 0:
            memory_info = psutil.virtual_memory()
            time1 = time.perf_counter()
            log_dict = dict(
                imgs_per_second=args.batch_size * args.log_interval / (time1 - time0),
                max_memory_allocated=torch.cuda.max_memory_allocated() / 1e9,
                cpu_mem_active=memory_info.active / 1e9,
                cpu_mem_used=memory_info.used / 1e9,
            )
            wandb.log(log_dict, step=step)
            time0 = time1

        if step % args.eval_interval == 0:
            # only save LoRA weights. Flux doesn't have buffers.
            ckpt = dict(
                model={name: p.detach().bfloat16() for name, p in model.named_parameters() if p.requires_grad},
                optim=optim.state_dict(),
                step=step,
            )
            if ema is not None:
                ckpt["ema"] = ema.state_dict()
            torch.save(ckpt, ckpt_dir / f"step{step:06d}.pth")

            save_images(
                model,
                ae,
                text_embedder,
                args.test_prompt_path,
                img_dir / f"step{step:06d}",
                args.test_img_size,
            )
            if ema is not None:
                ema.swap_params()
                save_images(
                    model,
                    ae,
                    text_embedder,
                    args.test_prompt_path,
                    img_dir / f"step{step:06d}_ema",
                    args.test_img_size,
                )
                ema.swap_params()
            model.train()

    wandb.finish()
