# https://github.com/black-forest-labs/flux-3-image-klein/blob/bfc8889/src/flux3/text_encoder.py

import math

import torch
from torch import Tensor, nn
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer

from ..qwen3 import Qwen3Model
from ..utils import load_hf_state_dict


# text-only Qwen3-VL. with text-only inputs, M-RoPE reduces to 1D RoPE, so we can reuse Qwen3Model.
class Flux3TextEncoder(nn.Module):
    # pad_multiple=1 disables padding
    def __init__(self, model_id: str = "Qwen/Qwen3-VL-4B-Instruct-FP8", pad_multiple: int = 80):
        super().__init__()
        self.output_indices = [3, 7, 11, 15, 19, 23, 27, 31]  # hidden_states[4, 8, ..., 32] in HF
        self.max_length = 4096
        self.pad_multiple = pad_multiple

        # only keep layers we need
        cfg = AutoConfig.from_pretrained(model_id).text_config
        cfg.num_hidden_layers = max(self.output_indices) + 1
        with torch.device("meta"):
            self.model = Qwen3Model(cfg)

        state_dict = load_hf_state_dict(model_id, "model.safetensors.index.json", prefix="model.language_model.")
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith("layers.") or int(k.split(".")[1]) < cfg.num_hidden_layers
        }
        self.model.load_state_dict(state_dict, assign=True)
        self.model.eval()

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id)

    # all prompts are padded to the same length (multiple of 80). the DiT attends to pad tokens, so we need to
    # match official pad embeddings: pad tokens don't attend to each other, and have position_id=1.
    # NOTE: official code encodes 1 prompt at a time. for multiple prompts of different lengths, the shorter ones
    # will have more pad tokens than official.
    @torch.no_grad()
    def forward(self, texts: str | list[str]) -> Tensor:
        if isinstance(texts, str):
            texts = [texts]

        texts = [
            self.tokenizer.apply_chat_template([dict(role="user", content=txt)], tokenize=False, add_generation_prompt=True)
            for txt in texts
        ]
        input_ids_list = [
            self.tokenizer(txt, truncation=True, max_length=self.max_length)["input_ids"] for txt in texts
        ]
        L = max(len(x) for x in input_ids_list)
        L = min(math.ceil(L / self.pad_multiple) * self.pad_multiple, self.max_length)

        device = self.model.embed_tokens.weight.device
        outputs = []
        for input_ids in input_ids_list:
            num_pads = L - len(input_ids)
            input_ids = torch.tensor(input_ids + [self.tokenizer.pad_token_id] * num_pads, device=device)

            mask = torch.arange(L, device=device) < L - num_pads
            position_ids = torch.where(mask, torch.arange(L, device=device), 1)
            attn_mask = torch.ones(L, L, dtype=torch.bool, device=device).tril() & mask

            hidden_states = self.model(
                input_ids[None],
                output_indices=self.output_indices,
                attn_mask=attn_mask[None, None],
                position_ids=position_ids,
            )
            outputs.append(torch.cat(hidden_states, dim=-1))

        return torch.cat(outputs, dim=0)
