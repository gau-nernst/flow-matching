# flow-matching

Supported models:

- [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) and its derivatives: [Flex.1-alpha](https://huggingface.co/ostris/Flex.1-alpha)
- Z-Image: [Base](https://huggingface.co/Tongyi-MAI/Z-Image) and [Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo). Turbo-NVFP4 from Comfy-UI.
- [FLUX.2-klein](https://github.com/black-forest-labs/flux2). T2I and TI2I. NVFP4 checkpoints are supported.

Supported features:

- Training: LoRA fine-tuning with CPU offload, logit-normal sampler, and multi-resolution/multi-aspect-ratio training.
  - TODO: Multi-resolution via varlen attention?
- [RF-inversion](https://arxiv.org/abs/2410.10792) (FLUX-Redux is simpler and superior).
- Single-GPU model offloading for inference and training.
- Solvers: Euler, [DPM-Solver++(2M)](https://arxiv.org/abs/2211.01095), [UniPC](https://arxiv.org/abs/2302.04867)
- Web UI: `uvicorn webui.server:app --host 0.0.0.0 --port 8000`

```bash
uv sync
source .venv/bin/activate
```

Resources:

- Rectified-Flow paper: https://arxiv.org/abs/2209.03003
- Stable Diffusion 3 paper: https://arxiv.org/abs/2403.03206
- DreamBooth: https://arxiv.org/abs/2208.12242

Notes:

- Finetune with logit-normal sampler seems to be more stable than uniform sampler. The changes are less abrupt. Not necessarily mean the final result is better.
- If you only have low-res images, you can train in low-res and FLUX can extrapolate to hi-res. However, for foreign concepts, resolution extrapolation is not reliable. It's best to do inference at similar resolution as fine-tuning (or we can do multi-resolution fine-tuning).
- FLUX-Redux
  - To reduce influence of image guidance, we can just scale the embeddings down e.g. `* 0.2`. Similarly, to increase influence of text guidance, we can scale T5 embeddings up.
  - We can use multiple guidance images (>=2). Just concat the embeddings.
- Finetune FLUX with `guidance=1.0` usually leads to better results. The finetuned model can be either used with built-in distilled CFG or true CFG (the latter gives slightly better results). Using both distilled CFG and true CFG is also possible, where distilled CFG > 1.0 can help with coherence, hi-res extrapolation, and the usual strange artifacts of finetuning.

## Flux (1024 x 1024)

### Matmul shapes

Note: there is always bias

M|N|K|count
---|---|---|---
(B,)|3072|256|1
(B,)|3072|768|1
(B,)|3072|3072|2
(B,)|6144|3072|1
(B,)|9216|3072|38
(B,)|18432|3072|38
(B, 512)|3072|3072|19
(B, 512)|3072|4096|1
(B, 512)|3072|12288|19
(B, 512)|9216|3072|19
(B, 512)|12288|3072|19
(B, 4096)|64|3072|1
(B, 4096)|3072|64|1
(B, 4096)|3072|3072|19
(B, 4096)|3072|12288|19
(B, 4096)|9216|3072|19
(B, 4096)|12288|3072|19
(B, 4608)|3072|15360|38
(B, 4608)|21504|3072|38

### Attention shapes

```
num_heads = 24
seq_len = 4096 + 512 = 4608
head_dim = 128
```
