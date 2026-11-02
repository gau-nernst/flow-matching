from ..registry import MODELS, ModelSpec
from .model import Fibo, FiboConfig, load_fibo
from .pipeline import FiboPipeline, FiboTextEncoder

MODELS.update(
    {
        "fibo-2.0-dev": ModelSpec(
            FiboPipeline,
            "briaai/Fibo-2.0-dev",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "Qwen/Qwen3-VL-4B-Instruct-FP8",
            dict(cfg_scale=5.0, num_steps=30),
        ),
    }
)
