from ..registry import MODELS, ModelSpec
from .model import MOSSImage, MOSSImageConfig, load_moss_image
from .pipeline import MOSSImagePipeline, MOSSImageTextEncoder

_HF_FILENAME = "transformer/diffusion_pytorch_model.safetensors.index.json"
_QWEN3_5_9B = "Qwen/Qwen3.5-9B"
_DEFAULTS = dict(cfg_scale=3.0, num_steps=50, shift=5.0, uncond_skip_layers=(22,))

MODELS.update(
    {
        "moss-image-sft": ModelSpec(
            MOSSImagePipeline, "OpenMOSS-Team/MOSS-Image-SFT", _HF_FILENAME, _QWEN3_5_9B, _DEFAULTS
        ),
        "moss-image-base": ModelSpec(
            MOSSImagePipeline, "OpenMOSS-Team/MOSS-Image-Base", _HF_FILENAME, _QWEN3_5_9B, _DEFAULTS
        ),
    }
)
