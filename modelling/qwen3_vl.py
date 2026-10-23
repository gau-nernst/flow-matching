# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py
# Qwen3-VL = Qwen3 text model + a ViT whose features replace the image placeholder tokens, with 3D (t, h, w) MRoPE
# for the text model and "deepstack" features added to the first few text layers' outputs at the image tokens.
# the text model is our Qwen3Model. here are the vision tower and the glue.

import itertools

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .linear import Linear
from .qwen3 import Qwen3Model


class VisionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = Linear(dim, dim * 3)
        self.proj = Linear(dim, dim)

    # x: [L, D]. rope: (cos, sin), [L, head_dim] each, rotate-half layout. attention within each image only.
    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor], seqlens: list[int]) -> Tensor:
        q, k, v = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).unbind(1)  # [L, nH, Dh] each

        cos, sin = rope

        def rotate(x: Tensor):
            x1, x2 = x.float().chunk(2, dim=-1)
            return (x.float() * cos + torch.cat([-x2, x1], dim=-1) * sin).type_as(x)

        q, k, v = (t.transpose(0, 1) for t in (rotate(q), rotate(k), v))  # [nH, L, Dh]
        out = [
            F.scaled_dot_product_attention(*args)
            for args in zip(q.split(seqlens, 1), k.split(seqlens, 1), v.split(seqlens, 1))
        ]
        return self.proj(torch.cat(out, dim=1).transpose(0, 1).flatten(1))


class VisionBlock(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = VisionAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Module()
        self.mlp.linear_fc1 = Linear(dim, mlp_dim)
        self.mlp.linear_fc2 = Linear(mlp_dim, dim)

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor], seqlens: list[int]) -> Tensor:
        x = x + self.attn(self.norm1(x), rope, seqlens)
        mlp = self.mlp
        return x + mlp.linear_fc2(F.gelu(mlp.linear_fc1(self.norm2(x)), approximate="tanh"))


# merges each 2x2 block of patches into 1 text-model token.
# postshuffle_norm: LayerNorm over the merged 4 patches (deepstack mergers) instead of over each patch.
class PatchMerger(nn.Module):
    def __init__(self, dim: int, out_dim: int, merge_size: int, postshuffle_norm: bool = False) -> None:
        super().__init__()
        merged_dim = dim * merge_size**2
        self.merged_dim = merged_dim
        self.postshuffle_norm = postshuffle_norm
        self.norm = nn.LayerNorm(merged_dim if postshuffle_norm else dim, eps=1e-6)
        self.linear_fc1 = Linear(merged_dim, merged_dim)
        self.linear_fc2 = Linear(merged_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        if self.postshuffle_norm:
            x = self.norm(x.view(-1, self.merged_dim))
        else:
            x = self.norm(x).view(-1, self.merged_dim)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class Qwen3VLVisionModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        dim = cfg.hidden_size
        self.patch_size = cfg.patch_size
        self.merge_size = cfg.spatial_merge_size
        self.head_dim = dim // cfg.num_heads

        # Conv3d with kernel = stride = (2, 16, 16) is a linear layer over each flattened patch
        self.patch_embed = nn.Module()
        self.patch_embed.proj = Linear(cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size**2, dim)
        self.pos_embed = nn.Embedding(cfg.num_position_embeddings, dim)
        self.num_grid_per_side = int(cfg.num_position_embeddings**0.5)

        self.blocks = nn.ModuleList([VisionBlock(dim, cfg.intermediate_size, cfg.num_heads) for _ in range(cfg.depth)])
        self.merger = PatchMerger(dim, cfg.out_hidden_size, self.merge_size)
        self.deepstack_visual_indexes = list(cfg.deepstack_visual_indexes)
        self.deepstack_merger_list = nn.ModuleList(
            [
                PatchMerger(dim, cfg.out_hidden_size, self.merge_size, postshuffle_norm=True)
                for _ in self.deepstack_visual_indexes
            ]
        )

        def flatten_conv(module, state_dict, prefix, *args):
            key = f"{prefix}patch_embed.proj.weight"
            if key in state_dict:
                state_dict[key] = state_dict[key].flatten(1)

        self.register_load_state_dict_pre_hook(flatten_conv)

    # patches are in 2x2-block-major order, as the HF image processor lays them out
    def _block_major(self, h: int, w: int) -> Tensor:
        m = self.merge_size
        return torch.arange(h * w).view(h // m, m, w // m, m).transpose(1, 2).flatten()

    # learned 48x48 position embeddings, bilinearly interpolated to the image's patch grid
    def _pos_embed(self, h: int, w: int, device) -> Tensor:
        side = self.num_grid_per_side
        hg = torch.linspace(0, side - 1, h, device=device)
        wg = torch.linspace(0, side - 1, w, device=device)
        h0, w0 = hg.int(), wg.int()
        h1, w1 = (h0 + 1).clamp(max=side - 1), (w0 + 1).clamp(max=side - 1)
        hf, wf = hg - h0, wg - w0

        table = self.pos_embed.weight
        out = 0
        for hi, hw in ((h0, 1 - hf), (h1, hf)):
            for wi, ww in ((w0, 1 - wf), (w1, wf)):
                idx = (hi[:, None] * side + wi[None, :]).flatten()
                weight = (hw[:, None] * ww[None, :]).flatten()
                out = out + table[idx].float() * weight[:, None]
        return out[self._block_major(h, w).to(device)].to(table.dtype)

    # 2D RoPE: first half of the rotary dims from the patch row, second half from the patch column
    def _rope(self, h: int, w: int, device) -> tuple[Tensor, Tensor]:
        dim = self.head_dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))
        order = self._block_major(h, w).to(device)
        hpos = torch.arange(h, device=device).repeat_interleave(w)[order]
        wpos = torch.arange(w, device=device).repeat(h)[order]
        freqs = torch.cat([hpos[:, None] * inv_freq, wpos[:, None] * inv_freq], dim=-1)
        emb = torch.cat([freqs, freqs], dim=-1)[:, None]  # [L, 1, head_dim]
        return emb.cos(), emb.sin()

    # pixel_values: [L, 3 * 2 * 16 * 16] from the HF image processor. grid_thw: (t, h, w) patches of each image.
    # returns the merged image tokens [L/4, out_dim], and the deepstack features, same shape, one per index.
    def forward(self, pixel_values: Tensor, grid_thw: list[tuple[int, int, int]]) -> tuple[Tensor, list[Tensor]]:
        device = pixel_values.device
        assert all(t == 1 for t, _, _ in grid_thw), "only images are supported"
        x = self.patch_embed.proj(pixel_values.to(self.pos_embed.weight.dtype))
        x = x + torch.cat([self._pos_embed(h, w, device) for _, h, w in grid_thw])
        ropes = [self._rope(h, w, device) for _, h, w in grid_thw]
        rope = (torch.cat([c for c, _ in ropes]), torch.cat([s for _, s in ropes]))
        seqlens = [h * w for _, h, w in grid_thw]

        deepstack = []
        for i, block in enumerate(self.blocks):
            x = block(x, rope, seqlens)
            if i in self.deepstack_visual_indexes:
                deepstack.append(self.deepstack_merger_list[self.deepstack_visual_indexes.index(i)](x))
        return self.merger(x), deepstack


# 3D (t, h, w) position of each token. text tokens advance all 3 together. an image's tokens sit at
# (start, start + row, start + col), and the text after it starts at start + max(rows, cols).
def qwen3_vl_position_ids(
    input_ids: Tensor, image_token_id: int, grid_thw: list[tuple[int, int, int]], merge_size: int
) -> Tensor:
    is_image = (input_ids == image_token_id).tolist()
    grids = iter(grid_thw)
    pos, out = 0, []
    for image, group in itertools.groupby(is_image):
        n = len(list(group))
        if not image:
            out.append(torch.arange(pos, pos + n).expand(3, -1))
            pos += n
            continue
        while n > 0:  # consecutive images
            _, h, w = next(grids)
            h, w = h // merge_size, w // merge_size
            hh, ww = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            out.append(torch.stack([torch.zeros(h * w, dtype=torch.long), hh.flatten(), ww.flatten()]) + pos)
            pos += max(h, w)
            n -= h * w
    return torch.cat(out, dim=1).to(input_ids.device)


# interleaved MRoPE: rotary frequency j takes its position from h if j % 3 == 1 (j < 3 * sections[1]),
# from w if j % 3 == 2 (j < 3 * sections[2]), else from t. returns [L, head_dim / 2] complex for apply_rope().
def qwen3_vl_rope(position_ids: Tensor, head_dim: int, theta: float, mrope_section: list[int]) -> Tensor:
    j = torch.arange(head_dim // 2, device=position_ids.device)
    axis = torch.zeros_like(j)
    axis[(j % 3 == 1) & (j < 3 * mrope_section[1])] = 1
    axis[(j % 3 == 2) & (j < 3 * mrope_section[2])] = 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64, device=j.device) / head_dim))
    pos = position_ids.double()[axis].T.contiguous()  # [L, head_dim / 2]
    freqs = (pos * inv_freq).float()
    return torch.polar(torch.ones_like(freqs), freqs)


# Qwen3-VL text model forward for 1 sequence. image tokens' embeddings are replaced by the vision tower's,
# and deepstack feature i is added to the output of layer i at the image tokens.
# returns the outputs of layers in output_indices.
def qwen3_vl_forward(
    model: Qwen3Model,
    input_ids: Tensor,  # [1, L]
    pos_embeds: Tensor,
    output_indices: list[int],
    image_mask: Tensor | None = None,  # [L] bool
    image_embeds: Tensor | None = None,
    deepstack: list[Tensor] = (),
) -> list[Tensor]:
    x = model.embed_tokens(input_ids)
    if image_mask is not None:
        x[0, image_mask] = image_embeds.to(x.dtype)

    outputs = []
    for i, layer in enumerate(model.layers):
        x = layer(x, pos_embeds)
        if i < len(deepstack):
            x[0, image_mask] += deepstack[i].to(x.dtype)
        if i in output_indices:
            outputs.append(x)
        if i >= max(output_indices):
            break
    return outputs
