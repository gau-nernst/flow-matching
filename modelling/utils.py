import json

import safetensors.torch
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import Tensor, nn


def load_hf_state_dict(repo_id: str, filename: str, prefix: str | None = None):
    local_path = hf_hub_download(repo_id, filename)

    if local_path.endswith(".safetensors"):
        state_dict = safetensors.torch.load_file(local_path)

    elif local_path.endswith(".pth"):
        state_dict = torch.load(local_path, map_location="cpu", mmap=True)

    elif local_path.endswith(".safetensors.index.json"):
        index = json.load(open(local_path))
        names = sorted(set(index["weight_map"].values()))

        base_path = (filename.rsplit("/", 1)[0] + "/") if "/" in filename else ""
        state_dict = dict()
        for name in names:
            state_dict.update(load_hf_state_dict(repo_id, base_path + name))

    else:
        raise RuntimeError(f"Unsupported file type. path={local_path}")

    if prefix is not None:
        state_dict = {k.removeprefix(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}
    return state_dict


def create_remap_hook(pairs: list[tuple[str, str]]):
    def hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for old, new in pairs:
            if f"{prefix}{old}" in state_dict:
                state_dict[f"{prefix}{new}"] = state_dict.pop(f"{prefix}{old}")

    return hook


def create_weight_merge_hook(old_keys: list[str], new_key: str):
    def hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for suffix in ("weight", "weight_scale_inv"):
            if f"{prefix}{old_keys[0]}.{suffix}" not in state_dict:
                continue

            w_list = [state_dict.pop(f"{prefix}{key}.{suffix}") for key in old_keys]
            state_dict[f"{prefix}{new_key}.{suffix}"] = torch.cat(w_list, dim=0)

    return hook


class FP32Linear(nn.Linear):
    """Mimic torch.autocast(dtype=torch.float32) behavior"""

    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.float() if self.bias is not None else None
        return F.linear(x.float(), self.weight.float(), bias)


class FP32LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor):
        if self.elementwise_affine:
            weight = self.weight.float()
            bias = self.bias.float()
        else:
            weight = None
            bias = None
        return F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps)
