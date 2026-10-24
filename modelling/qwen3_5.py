# https://github.com/huggingface/transformers/blob/v5.14.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py
# text-only Qwen3.5, prefill only (no KV cache / recurrent state). 3 Gated DeltaNet layers + 1 gated attention layer,
# repeated. GDN kernels are from flash-linear-attention.

import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from torch import Tensor, nn
from transformers import AutoConfig

from .linear import Linear
from .qwen3 import Qwen3MLP
from .rope import apply_rope
from .utils import create_weight_merge_hook, load_hf_state_dict


# zero-centered weight: y = norm(x) * (1 + w). add 1 at load, and keep it in FP32 since (1 + w) loses precision in BF16.
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.eps = eps

        def hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            if f"{prefix}weight" in state_dict:
                state_dict[f"{prefix}weight"] = state_dict[f"{prefix}weight"].float() + 1.0

        self.register_load_state_dict_pre_hook(hook)

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x.float(), x.shape[-1:], self.weight, self.eps).type_as(x)


class Qwen3_5Attention(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.head_dim = D = cfg.head_dim
        self.num_qo_heads = H = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.eps = cfg.rms_norm_eps

        # q_proj outputs [q, gate] for each head
        self.q_proj = Linear(cfg.hidden_size, H * D * 2, bias=False)
        self.k_proj = Linear(cfg.hidden_size, self.num_kv_heads * D, bias=False)
        self.v_proj = Linear(cfg.hidden_size, self.num_kv_heads * D, bias=False)
        self.o_proj = Linear(H * D, cfg.hidden_size, bias=False)
        self.q_norm = nn.Parameter(torch.empty(D))
        self.k_norm = nn.Parameter(torch.empty(D))

        self.rot_dim = int(D * cfg.rope_parameters["partial_rotary_factor"])

        def hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            if f"{prefix}q_proj.weight" in state_dict:
                # [H, (q, gate), D, in] -> [(q, gate), H, D, in], so that q and gate are contiguous chunks
                q, gate = state_dict[f"{prefix}q_proj.weight"].unflatten(0, (H, 2, D)).unbind(1)
                state_dict[f"{prefix}q_proj.weight"] = torch.cat([q.flatten(0, 1), gate.flatten(0, 1)])
            for name in ("q_norm.weight", "k_norm.weight"):
                if f"{prefix}{name}" in state_dict:
                    state_dict[f"{prefix}{name[:6]}"] = state_dict.pop(f"{prefix}{name}").float() + 1.0  # zero-centered

        self.register_load_state_dict_pre_hook(hook)

    def forward(self, x: Tensor, pos_embeds: Tensor, add: Tensor | None = None) -> Tensor:
        q, gate = self.q_proj(x).chunk(2, dim=-1)
        q = q.unflatten(-1, (-1, self.head_dim))
        k = self.k_proj(x).unflatten(-1, (-1, self.head_dim))
        q = apply_rope(q, pos_embeds, self.q_norm, self.eps, interleaved=False, rot_dim=self.rot_dim).transpose(1, 2)
        k = apply_rope(k, pos_embeds, self.k_norm, self.eps, interleaved=False, rot_dim=self.rot_dim)
        v = self.v_proj(x).unflatten(-1, (-1, self.head_dim))
        out = F.scaled_dot_product_attention(q, k.transpose(1, 2), v.transpose(1, 2), is_causal=True, enable_gqa=True)
        out = out.transpose(1, 2).flatten(-2) * torch.sigmoid(gate)
        return self.o_proj(out, add)


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.num_k_heads = cfg.linear_num_key_heads
        self.num_v_heads = cfg.linear_num_value_heads
        self.head_k_dim = cfg.linear_key_head_dim
        self.head_v_dim = cfg.linear_value_head_dim
        self.eps = cfg.rms_norm_eps
        key_dim = self.num_k_heads * self.head_k_dim
        value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = key_dim * 2 + value_dim

        # [qkv, z] and [b, a]. the latter is kept separate since its dim is not a multiple of 128 (for block FP8)
        self.in_proj_qkvz = Linear(cfg.hidden_size, self.conv_dim + value_dim, bias=False)
        self.in_proj_ba = Linear(cfg.hidden_size, self.num_v_heads * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            cfg.linear_conv_kernel_dim,
            groups=self.conv_dim,
            padding=cfg.linear_conv_kernel_dim - 1,
            bias=False,
        )
        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads))
        self.norm = nn.Module()
        self.norm.weight = nn.Parameter(torch.empty(self.head_v_dim))
        self.out_proj = Linear(value_dim, cfg.hidden_size, bias=False)

        self.register_load_state_dict_pre_hook(create_weight_merge_hook(["in_proj_qkv", "in_proj_z"], "in_proj_qkvz"))
        self.register_load_state_dict_pre_hook(create_weight_merge_hook(["in_proj_b", "in_proj_a"], "in_proj_ba"))

    def forward(self, x: Tensor, add: Tensor | None = None) -> Tensor:
        L = x.shape[1]
        qkv, z = self.in_proj_qkvz(x).split((self.conv_dim, self.num_v_heads * self.head_v_dim), dim=-1)
        b, a = self.in_proj_ba(x).chunk(2, dim=-1)
        qkv = F.silu(self.conv1d(qkv.transpose(1, 2))[..., :L]).transpose(1, 2)
        q, k, v = qkv.split((self.num_k_heads * self.head_k_dim,) * 2 + (self.num_v_heads * self.head_v_dim,), dim=-1)

        rep = self.num_v_heads // self.num_k_heads
        q = q.unflatten(-1, (-1, self.head_k_dim)).repeat_interleave(rep, dim=2)
        k = k.unflatten(-1, (-1, self.head_k_dim)).repeat_interleave(rep, dim=2)
        v = v.unflatten(-1, (-1, self.head_v_dim))
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        out, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)

        # gated RMSNorm
        out = F.rms_norm(out.float(), out.shape[-1:], eps=self.eps).type_as(x) * self.norm.weight
        out = (out.float() * F.silu(z.unflatten(-1, (-1, self.head_v_dim)).float())).type_as(x)
        return self.out_proj(out.flatten(-2), add)


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, cfg, layer_type: str) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        if layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(cfg)
        else:
            self.self_attn = Qwen3_5Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = Qwen3MLP(cfg)

    def forward(self, x: Tensor, pos_embeds: Tensor) -> Tensor:
        if hasattr(self, "linear_attn"):
            x = self.linear_attn(self.input_layernorm(x), x)
        else:
            x = self.self_attn(self.input_layernorm(x), pos_embeds, x)
        x = self.mlp(self.post_attention_layernorm(x), x)
        return x


class Qwen3_5Model(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(cfg, layer_type) for layer_type in cfg.layer_types])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    # text-only, so M-RoPE reduces to 1D RoPE. returns the last hidden state, after the final norm.
    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed_tokens(input_ids)

        L = x.shape[1]
        D = self.cfg.head_dim
        rot_dim = int(D * self.cfg.rope_parameters["partial_rotary_factor"])
        theta = self.cfg.rope_parameters["rope_theta"]
        omega = 1.0 / (theta ** (torch.arange(0, rot_dim, 2, dtype=torch.float64, device=x.device) / rot_dim))
        freqs = torch.arange(L, dtype=torch.float64, device=x.device)[:, None] * omega
        freqs = F.pad(freqs, (0, (D - rot_dim) // 2)).float()  # non-rotated dims have angle 0
        pos_embeds = torch.polar(torch.ones_like(freqs), freqs)

        for layer in self.layers:
            x = layer(x, pos_embeds)
        return self.norm(x)

    @staticmethod
    def from_pretrained(model_id: str, subfolder: str = "", prefix: str | None = None) -> "Qwen3_5Model":
        cfg = AutoConfig.from_pretrained(model_id, subfolder=subfolder).text_config
        with torch.device("meta"):
            model = Qwen3_5Model(cfg)

        filename = f"{subfolder}/model.safetensors.index.json" if subfolder else "model.safetensors.index.json"
        state_dict = load_hf_state_dict(model_id, filename, prefix=prefix)
        model.load_state_dict(state_dict, assign=True)
        return model

    # online block-FP8 (1x128 activation, 128x128 weight) for all linear layers
    def to_fp8(self):
        for m in self.modules():
            if isinstance(m, Linear) and m.weight.shape[0] % 128 == 0:
                m.to_fp8_1d2d()
        return self
