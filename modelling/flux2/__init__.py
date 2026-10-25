from ..registry import MODELS, ModelSpec
from .model import Flux2, Flux2Config, load_flux2
from .pipeline import Flux2Pipeline, Flux2Qwen3TextEncoder


def _spec(repo_id: str, filename: str, text_encoder_id: str, sample_defaults: dict):
    return ModelSpec(Flux2Pipeline, f"black-forest-labs/{repo_id}", filename, text_encoder_id, sample_defaults)


_QWEN3_4B = "Qwen/Qwen3-4B-FP8"
_QWEN3_8B = "Qwen/Qwen3-8B-FP8"
_KLEIN = dict(cfg_scale=1.0, num_steps=4)
_KLEIN_BASE = dict(cfg_scale=4.0, num_steps=50)

MODELS.update(
    {
        "flux2-klein-4B": _spec("FLUX.2-klein-4B", "flux-2-klein-4b.safetensors", _QWEN3_4B, _KLEIN),
        "flux2-klein-4B-fp8": _spec("FLUX.2-klein-4b-fp8", "flux-2-klein-4b-fp8.safetensors", _QWEN3_4B, _KLEIN),
        "flux2-klein-4B-nvfp4": _spec("FLUX.2-klein-4b-nvfp4", "flux-2-klein-4b-nvfp4.safetensors", _QWEN3_4B, _KLEIN),
        "flux2-klein-9B": _spec("FLUX.2-klein-9B", "flux-2-klein-9b.safetensors", _QWEN3_8B, _KLEIN),
        "flux2-klein-9B-fp8": _spec("FLUX.2-klein-9b-fp8", "flux-2-klein-9b-fp8.safetensors", _QWEN3_8B, _KLEIN),
        "flux2-klein-9B-nvfp4": _spec("FLUX.2-klein-9b-nvfp4", "flux-2-klein-9b-nvfp4.safetensors", _QWEN3_8B, _KLEIN),
        "flux2-klein-base-4B": _spec(
            "FLUX.2-klein-base-4B", "flux-2-klein-base-4b.safetensors", _QWEN3_4B, _KLEIN_BASE
        ),
        "flux2-klein-base-9B": _spec(
            "FLUX.2-klein-base-9B", "flux-2-klein-base-9b.safetensors", _QWEN3_8B, _KLEIN_BASE
        ),
    }
)
