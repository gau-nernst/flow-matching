import dataclasses


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    pipeline_cls: type
    repo_id: str
    filename: str
    text_encoder_id: str | None = None  # None: text encoder is not supported
    sample_defaults: dict = dataclasses.field(default_factory=dict)  # kwargs for pipeline.generate()


# each model family adds its entries in its package __init__.py
MODELS: dict[str, ModelSpec] = {}


def load_pipeline(name: str):
    if name not in MODELS:
        raise ValueError(f"Unsupported model {name!r}. Available: {list(MODELS)}")
    return MODELS[name].pipeline_cls.load(name)
