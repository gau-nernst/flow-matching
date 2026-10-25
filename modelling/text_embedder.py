# https://github.com/black-forest-labs/flux/blob/7e14a05e/src/flux/modules/conditioner.py

import torch
from torch import Tensor, nn
from transformers import AutoTokenizer, CLIPTextModel, T5EncoderModel
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer


class TextEmbedder(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        output_key: str | int | list[str | int],
        use_attn_mask: bool = False,
    ) -> None:
        super().__init__()
        self.model = model.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.max_length = max_length  # allow overrides
        self.output_key = output_key
        self.use_attn_mask = use_attn_mask

    @staticmethod
    def unwrap_output(out, key: str | int | list[str | int]):
        if isinstance(key, str):
            return out[key]
        elif isinstance(key, int):
            return out.hidden_states[key]
        else:
            return [TextEmbedder.unwrap_output(out, k) for k in key]

    @staticmethod
    def apply_mask(inputs: Tensor | list[Tensor], mask: Tensor):
        if isinstance(inputs, Tensor):
            # add trailing dims for broadcasting
            mask = mask.view(mask.shape + (1,) * (inputs.ndim - mask.ndim))
            return inputs * mask
        return [TextEmbedder.apply_mask(x, mask) for x in inputs]

    @torch.no_grad()
    def forward(self, texts: str | list[str]) -> Tensor:
        # - Flux always pad to max_length and ignore attention mask
        # - Wan uses attention mask i.e. don't attend to padding tokens
        #   and zero-out outputs
        # TODO: sequence packing
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=self.use_attn_mask,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        out = self.model(**inputs, output_hidden_states=True)

        out = self.unwrap_output(out, self.output_key)
        if self.use_attn_mask:
            out = self.apply_mask(out, inputs["attention_mask"])

        return out


def load_t5(max_length: int = 512) -> TextEmbedder:
    model_id = "mcmonkey/google_t5-v1_1-xxl_encoderonly"
    model = T5EncoderModel.from_pretrained(model_id, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return TextEmbedder(model, tokenizer, max_length, "last_hidden_state")


def load_clip_l(output_key: str | int | list[str | int] = "pooler_output") -> TextEmbedder:
    # NOTE: OpenAI CLIP weights use FP16 for Linear, and FP32 for the rest (Embeddings, LayerNorm, pos_embed, ...)
    # but Flux loads it in BF16. TODO: investigate FP16 vs BF16
    # NOTE: FLUX only uses pooler_output
    # NOTE: use weights from Stability AI to reduce download time, since it is FP16 and doesn't include ViT's weights.
    model = CLIPTextModel.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        dtype=torch.float16,
        subfolder="text_encoder",
    )
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    return TextEmbedder(model, tokenizer, 77, output_key)
