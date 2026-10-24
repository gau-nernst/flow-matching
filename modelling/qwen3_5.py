# https://github.com/huggingface/transformers/blob/v5.14.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py
# text-only, prefill only. GDN kernels are from flash-linear-attention

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from fla.modules import FusedRMSNormGated
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from torch import Tensor, nn
from transformers import AutoConfig

from .linear import Linear
from .qwen3 import Qwen3MLP
from .rope import apply_rope, compute_rope
from .utils import create_weight_merge_hook, load_hf_state_dict


@triton.jit
def _rms_norm_kernel(x_ptr, w_ptr, o_ptr, D: tl.constexpr, eps: float):
    pid = tl.program_id(0)
    offs = tl.arange(0, triton.next_power_of_2(D))
    mask = offs < D
    x = tl.load(x_ptr + pid * D + offs, mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask).to(tl.float32)
    rrms = tl.rsqrt(tl.sum(x * x) * (1.0 / D) + eps)
    tl.store(o_ptr + pid * D + offs, x * rrms * (1.0 + w), mask)


# y = norm(x) * (1 + w)
class GemmaRMSNorm(nn.RMSNorm):
    def forward(self, x: Tensor) -> Tensor:
        if torch.is_grad_enabled():
            return F.rms_norm(x.float(), x.shape[-1:], self.weight.float() + 1.0, self.eps).type_as(x)

        assert x.is_contiguous()
        out = torch.empty_like(x)
        D = x.shape[-1]
        _rms_norm_kernel[(x.numel() // D,)](x, self.weight, out, D, self.eps)
        return out


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
                    state_dict[f"{prefix}{name[:6]}"] = state_dict.pop(f"{prefix}{name}")

        self.register_load_state_dict_pre_hook(hook)

    def forward(self, x: Tensor, pos_embeds: Tensor, add: Tensor | None = None) -> Tensor:
        q, gate = self.q_proj(x).chunk(2, dim=-1)
        q = q.unflatten(-1, (-1, self.head_dim))
        k = self.k_proj(x).unflatten(-1, (-1, self.head_dim))
        rope_kwargs = {"interleaved": False, "rot_dim": self.rot_dim, "gemma_norm": True}
        q = apply_rope(q, pos_embeds, self.q_norm, self.eps, **rope_kwargs).transpose(1, 2)
        k = apply_rope(k, pos_embeds, self.k_norm, self.eps, **rope_kwargs)
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
        key_dim = self.num_k_heads * self.head_k_dim
        value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = key_dim * 2 + value_dim

        # [qkv, z] and [b, a]. [b, a] is separate since its dim is not a multiple of 128 (block FP8)
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
        self.norm = FusedRMSNormGated(self.head_v_dim, eps=cfg.rms_norm_eps)
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

        out = self.norm(out, z.unflatten(-1, (-1, self.head_v_dim)))
        return self.out_proj(out.flatten(-2), add)


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, cfg, layer_type: str) -> None:
        super().__init__()
        self.input_layernorm = GemmaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        if layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(cfg)
        else:
            self.self_attn = Qwen3_5Attention(cfg)
        self.post_attention_layernorm = GemmaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
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
        self.norm = GemmaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    # text-only, so M-RoPE reduces to 1D RoPE
    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed_tokens(input_ids)

        D = self.cfg.head_dim
        rot_dim = int(D * self.cfg.rope_parameters["partial_rotary_factor"])
        rope = compute_rope(x.shape[1], rot_dim, self.cfg.rope_parameters["rope_theta"], device=x.device)
        pad = rope.new_ones(rope.shape[0], (D - rot_dim) // 2)  # angle 0 after rot_dim
        pos_embeds = torch.cat([rope, pad], dim=-1)

        for layer in self.layers:
            x = layer(x, pos_embeds)
        return self.norm(x)

    @staticmethod
    def from_pretrained(model_id: str) -> "Qwen3_5Model":
        cfg = AutoConfig.from_pretrained(model_id).text_config
        with torch.device("meta"):
            model = Qwen3_5Model(cfg)

        state_dict = load_hf_state_dict(model_id, "model.safetensors.index.json", prefix="model.language_model.")
        model.load_state_dict(state_dict, assign=True)
        return model
