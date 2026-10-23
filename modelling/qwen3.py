import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import Qwen3Config

from .linear import Linear
from .rope import apply_rope, compute_rope
from .utils import create_remap_hook, create_weight_merge_hook, load_hf_state_dict


class Qwen3Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.head_dim = cfg.head_dim
        self.num_qo_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.eps = cfg.rms_norm_eps
        qo_dim = cfg.num_attention_heads * cfg.head_dim
        kv_dim = cfg.num_key_value_heads * cfg.head_dim

        self.qkv_proj = Linear(cfg.hidden_size, qo_dim + kv_dim * 2, bias=False)
        self.o_proj = Linear(qo_dim, cfg.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = nn.RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)

        # HF Qwen3 rotates pairs (i, i + D/2), while our apply_rope() rotates pairs (2i, 2i + 1).
        # permute q/k rows within each head so that they are equivalent. this must run before weight merge.
        # for FP8 weights, each head is exactly 1 block of 128 rows, so weight_scale_inv stays unchanged.
        def rope_hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            D = self.head_dim
            perm = torch.stack([torch.arange(D // 2), torch.arange(D // 2) + D // 2], dim=-1).flatten()
            for name in ("q_proj.weight", "k_proj.weight"):
                if f"{prefix}{name}" in state_dict:
                    w = state_dict[f"{prefix}{name}"]
                    state_dict[f"{prefix}{name}"] = w.unflatten(0, (-1, D))[:, perm].flatten(0, 1)
            for name in ("q_norm.weight", "k_norm.weight"):
                if f"{prefix}{name}" in state_dict:
                    state_dict[f"{prefix}{name}"] = state_dict[f"{prefix}{name}"][perm]

        self.register_load_state_dict_pre_hook(rope_hook)
        self.register_load_state_dict_pre_hook(create_weight_merge_hook(["q_proj", "k_proj", "v_proj"], "qkv_proj"))

    def forward(
        self,
        x: Tensor,
        pos_embeds: Tensor,
        add: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        qkv = self.qkv_proj(x).view(*x.shape[:-1], -1, self.head_dim)
        q, k, v = qkv.split((self.num_qo_heads, self.num_kv_heads, self.num_kv_heads), dim=-2)
        q = apply_rope(q, pos_embeds, self.q_norm.weight, self.eps).transpose(1, 2)
        k = apply_rope(k, pos_embeds, self.k_norm.weight, self.eps).transpose(1, 2)
        v = v.transpose(1, 2)
        if attn_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=True)
        out = self.o_proj(out.transpose(1, 2).flatten(-2), add)
        return out


class Qwen3MLP(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.w13 = Linear(cfg.hidden_size, cfg.intermediate_size * 2, bias=False)
        self.w2 = Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

        self.register_load_state_dict_pre_hook(create_weight_merge_hook(["gate_proj", "up_proj"], "w13"))
        remap_pairs = [
            ("down_proj.weight", "w2.weight"),
            ("down_proj.weight_scale_inv", "w2.weight_scale_inv"),
        ]
        self.register_load_state_dict_pre_hook(create_remap_hook(remap_pairs))

    def forward(self, x: Tensor, add: Tensor | None = None) -> Tensor:
        gate, up = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up, add)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen3Config, layer_id: int) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = Qwen3Attention(cfg)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = Qwen3MLP(cfg)
        self.layer_id = layer_id

    def forward(self, x: Tensor, pos_embeds: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        x = self.self_attn(self.input_layernorm(x), pos_embeds, x, attn_mask)
        x = self.mlp(self.post_attention_layernorm(x), x)
        return x


class Qwen3Model(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.act_ckpt = False
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, cfg.pad_token_id)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    # attn_mask: [B, 1, L, L] bool, True means attend. default to causal mask.
    def forward(
        self,
        input_ids: Tensor,
        *,
        output_indices: tuple[int, ...] | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        x = self.embed_tokens(input_ids)

        theta = self.cfg.rope_parameters["rope_theta"]
        pos_embeds = compute_rope(x.shape[-2], self.cfg.head_dim, theta, device=x.device)

        intermediates = []
        for i, layer in enumerate(self.layers):
            x = layer(x, pos_embeds, attn_mask)
            if output_indices and i in output_indices:
                intermediates.append(x)

        return intermediates if intermediates else self.norm(x)


class Qwen3ForCausalLM(nn.Module):
    _cfg_cls = Qwen3Config

    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = Qwen3Model(cfg)
        self.lm_head = Linear(cfg.hidden_size, cfg.vocab_size, bias=False) if not cfg.tie_word_embeddings else None

    def forward(self, input_ids: Tensor) -> Tensor:
        hidden_states = self.model(input_ids)
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            w = self.model.embed_tokens.weight.to(hidden_states.dtype)
            logits = F.linear(hidden_states, w)
        return logits

    @staticmethod
    def from_pretrained(model_id: str) -> "Qwen3ForCausalLM":
        cfg = Qwen3Config.from_pretrained(model_id)
        with torch.device("meta"):
            model = Qwen3ForCausalLM(cfg)

        state_dict = load_hf_state_dict(model_id, "model.safetensors.index.json")
        if cfg.tie_word_embeddings and "lm_head.weight" in state_dict:
            state_dict.pop("lm_head.weight")
        model.load_state_dict(state_dict, assign=True)
        return model
