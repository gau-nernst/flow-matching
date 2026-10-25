from ..registry import MODELS, ModelSpec
from .model import ZImage, ZImageConfig, load_zimage
from .pipeline import ZImagePipeline

_HF_FILENAME = "transformer/diffusion_pytorch_model.safetensors.index.json"
_QWEN3_4B = "Qwen/Qwen3-4B"
_TURBO = dict(cfg_scale=1.0, num_steps=8, time_shift=3.0)
_BASE = dict(cfg_scale=3.0, num_steps=28, time_shift=6.0)

MODELS.update(
    {
        "z-image-turbo": ModelSpec(ZImagePipeline, "Tongyi-MAI/Z-Image-Turbo", _HF_FILENAME, _QWEN3_4B, _TURBO),
        "z-image-turbo-nvfp4": ModelSpec(
            ZImagePipeline,
            "Comfy-Org/z_image_turbo",
            "split_files/diffusion_models/z_image_turbo_nvfp4.safetensors",
            _QWEN3_4B,
            _TURBO,
        ),
        "z-image-base": ModelSpec(ZImagePipeline, "Tongyi-MAI/Z-Image", _HF_FILENAME, _QWEN3_4B, _BASE),
    }
)
