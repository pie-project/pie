"""GPT OSS Large Language Model Architecture"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import flashinfer as ops
from adapter_utils import AdapterSubpass
from model.config import CommonArch, ModelConfig
from model.gptoss_utils import (
    FP4_VALUES,
    chunked_enumerate,
    GptOssRMSNorm,
    GptOssRotaryEmbedding,
)
from einops import einsum, rearrange


VERSION = "0.1.0"


@dataclass
class GptOssArch(CommonArch):
    """GPT OSS specific architecture configuration."""

    # MoE configuration
    num_experts: int
    experts_per_token: int

    # RoPE configuration
    rope_theta: float
    rope_scaling_factor: float
    rope_ntk_alpha: float
    rope_ntk_beta: float

    # Model specific parameters
    initial_context_length: int
    sliding_window: int
    swiglu_limit: float

    @staticmethod
    def from_config(cfg: ModelConfig) -> "GptOssArch":
        """Parse GPT OSS-specific architecture configuration."""
        # Get common architecture fields
        common_arch_dict = cfg.get_common_arch_dict()

        # Get all the fields for the architecture section to grab other
        # architecture-specific fields
        arch_dict = cfg.get_required_key(cfg.root, "architecture")

        # Get MoE configuration
        moe_dict = cfg.get_required_key(arch_dict, "moe")
        num_experts = cfg.get_required_key(moe_dict, "num_experts")
        experts_per_token = cfg.get_required_key(moe_dict, "experts_per_token")

        # Get RoPE configuration (GPT OSS uses YaRN-style RoPE)
        rope_dict = cfg.get_required_key(arch_dict, "rope")
        rope_theta = cfg.get_required_key(rope_dict, "theta")
        rope_scaling_factor = cfg.get_required_key(rope_dict, "scaling_factor")
        rope_ntk_alpha = cfg.get_required_key(rope_dict, "ntk_alpha")
        rope_ntk_beta = cfg.get_required_key(rope_dict, "ntk_beta")

        # Get model specific parameters
        initial_context_length = cfg.get_required_key(
            arch_dict, "initial_context_length"
        )
        sliding_window = cfg.get_required_key(arch_dict, "sliding_window")
        swiglu_limit = cfg.get_required_key(arch_dict, "swiglu_limit")

        return GptOssArch(
            # Common fields
            **common_arch_dict,
            # GPT OSS-specific fields
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            rope_theta=rope_theta,
            rope_scaling_factor=rope_scaling_factor,
            rope_ntk_alpha=rope_ntk_alpha,
            rope_ntk_beta=rope_ntk_beta,
            initial_context_length=initial_context_length,
            sliding_window=sliding_window,
            swiglu_limit=swiglu_limit,
        )


def create_fusion_map(model: nn.Module):
    """
    Analyzes the model and creates a map for fusing weights and handling MXFP4 tensors.

    Returns:
        A dictionary mapping {
            fused_tensor_name: {"sources": [source_names], "dim": cat_dim, "type": type}
        }.
        For MXFP4 tensors, type is "mxfp4" and sources contains [blocks_name, scales_name].
        For fusion tensors, type is "fusion" and sources contains the tensors to concatenate.
        For regular tensors, type is "regular" and sources contains the single tensor name.
    """
    fusion_map = {}
    for name, module in model.named_modules():
        # --- Rule for GptOssAttention QKV Fusion ---
        if isinstance(module, GptOssAttention):
            # Handle weights
            target_w = f"{name}.qkv_proj.weight"
            sources_w = [
                f"{name}.q_proj.weight",
                f"{name}.k_proj.weight",
                f"{name}.v_proj.weight",
            ]
            fusion_map[target_w] = {"sources": sources_w, "dim": 0, "op": "fusion"}

            # Handle biases if they exist
            if module.qkv_proj.bias is not None:
                target_b = f"{name}.qkv_proj.bias"
                sources_b = [
                    f"{name}.q_proj.bias",
                    f"{name}.k_proj.bias",
                    f"{name}.v_proj.bias",
                ]
                fusion_map[target_b] = {
                    "sources": sources_b,
                    "dim": 0,
                    "op": "fusion",
                }

        # --- Rule for GptOssExperts MXFP4 Weights ---
        elif isinstance(module, GptOssExperts):
            # Handle gate_up_proj weights (MXFP4 format)
            target_gate_up = f"{name}.gate_up_proj"
            blocks_gate_up = f"{name}.gate_up_proj_blocks"
            scales_gate_up = f"{name}.gate_up_proj_scales"
            fusion_map[target_gate_up] = {
                "sources": [blocks_gate_up, scales_gate_up],
                "op": "dequantize_mxfp4",
                "fp4_values": FP4_VALUES,
            }

            # Handle down_proj weights (MXFP4 format)
            target_down = f"{name}.down_proj"
            blocks_down = f"{name}.down_proj_blocks"
            scales_down = f"{name}.down_proj_scales"
            fusion_map[target_down] = {
                "sources": [blocks_down, scales_down],
                "op": "dequantize_mxfp4",
                "fp4_values": FP4_VALUES,
            }

    return fusion_map


class GptOssAttention(nn.Module):
    """GPT OSS attention module with attention sink."""

    def __init__(self, config: GptOssArch, layer_idx: int, rope: GptOssRotaryEmbedding):
        """Initialize the GPT OSS attention module."""
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_size
        self.num_attention_heads = config.num_query_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        # Apply sliding window to even layers and full attention to odd layers
        # This follows the GPT-OSS alternating attention pattern
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0

        # Define the output sizes for Q, K, and V for clarity
        self.q_size = config.num_query_heads * config.head_size
        self.k_size = config.num_key_value_heads * config.head_size
        self.v_size = config.num_key_value_heads * config.head_size

        # Sink tokens parameter
        self.sinks = nn.Parameter(
            torch.empty(
                config.num_query_heads,
                device=torch.device(config.device),
                dtype=config.dtype,
            )
        )

        qkv_dim = config.head_size * (
            config.num_query_heads + 2 * config.num_key_value_heads
        )
        self.qkv_proj = nn.Linear(
            config.hidden_size,
            qkv_dim,
            device=torch.device(config.device),
            dtype=config.dtype,
        )

        self.o_proj = nn.Linear(
            config.head_size * config.num_query_heads,
            config.hidden_size,
            device=torch.device(config.device),
            dtype=config.dtype,
        )

        self.scaling = self.head_dim**-0.5

        self.rope = rope

    def _ceil_div(self, a: int, b: int) -> int:
        return -(-a // b)

    def _attend_one_page(
        self,
        query: torch.Tensor,
        paged_keys: torch.Tensor,
        paged_mask: torch.Tensor,
        paged_values: torch.Tensor,
        sum_exp: torch.Tensor,
        sum_val: torch.Tensor,
        max_score: torch.Tensor,
    ):
        page_attn_scores = einsum(query, paged_keys, "q h d, s h d -> h q s")
        page_attn_scores = (page_attn_scores + paged_mask).to(torch.float32)
        page_max_score = torch.max(page_attn_scores, dim=-1, keepdim=False).values

        # Convert -inf elements to 0.0 in page_max_score
        page_max_score = torch.where(
            torch.isinf(page_max_score) & (page_max_score < 0),
            torch.tensor(0.0, dtype=page_max_score.dtype, device=page_max_score.device),
            page_max_score,
        )

        page_attn_scores = torch.exp(page_attn_scores - page_max_score.unsqueeze(-1))

        page_sum_exp = torch.sum(page_attn_scores, dim=-1, keepdim=False)
        page_sum_val = einsum(
            page_attn_scores, paged_values.to(torch.float32), "h q s, s h d -> h q d"
        )

        new_max_score = torch.max(max_score, page_max_score)
        alpha = torch.exp(max_score - new_max_score)
        beta = torch.exp(page_max_score - new_max_score)

        sum_exp = sum_exp * alpha + page_sum_exp * beta
        sum_val = sum_val * alpha.unsqueeze(-1) + page_sum_val * beta.unsqueeze(-1)
        max_score = new_max_score

        return sum_val, sum_exp, max_score

    def _paged_attention(
        self,
        queries: torch.Tensor,
        qo_indptr: torch.IntTensor,
        kv_page_indptr: torch.IntTensor,
        kv_last_page_lens: torch.IntTensor,
        kv_page_indices: torch.IntTensor,
        attention_mask: torch.Tensor,
        kv_cache_at_layer: torch.Tensor,
    ):
        output_embeds = torch.empty(
            queries.shape[0],
            self.config.hidden_size,
            dtype=queries.dtype,
            device=queries.device,
        )
        kv_page_size = kv_cache_at_layer.shape[2]
        mask_offset = 0
        batch_num = len(qo_indptr) - 1

        for batch_idx in range(batch_num):
            q_start = qo_indptr[batch_idx]
            q_end = qo_indptr[batch_idx + 1]
            query_len = int(q_end - q_start)

            kv_page_start = int(kv_page_indptr[batch_idx])
            kv_page_end = int(kv_page_indptr[batch_idx + 1])
            kv_last_page_len = int(kv_last_page_lens[batch_idx])

            seq_len = int(
                (kv_page_end - kv_page_start - 1) * kv_page_size + kv_last_page_len
            )

            mask_len = seq_len * query_len
            mask = attention_mask[mask_offset : mask_offset + mask_len].view(
                query_len, seq_len
            )
            mask_offset += mask_len

            # If this attention layer uses a sliding window, we keep only the last
            # few pages of the KV cache and the corresponding mask.
            if self.sliding_window != 0:
                attn_page_cnt = 1 + self._ceil_div(
                    self.sliding_window - kv_last_page_len, kv_page_size
                )
                kv_page_start = max(kv_page_start, kv_page_end - attn_page_cnt)

                seq_len = int(
                    (kv_page_end - kv_page_start - 1) * kv_page_size + kv_last_page_len
                )

                mask = mask[:, -seq_len:]

            query = queries[q_start:q_end, :] * self.scaling

            sum_exp = torch.zeros(
                self.num_attention_heads,
                query_len,
                device=query.device,
                dtype=torch.float32,
            )
            sum_val = torch.zeros(
                self.num_attention_heads,
                query_len,
                self.head_dim,
                device=query.device,
                dtype=torch.float32,
            )
            max_score = torch.zeros(
                self.num_attention_heads,
                query_len,
                device=query.device,
                dtype=torch.float32,
            )

            # Attend to all but the last page, processing 32 pages at a time
            for page_cnts, kv_page_idx_idxs in chunked_enumerate(
                range(kv_page_start, kv_page_end - 1), 32
            ):
                chunk_kv_page_indices = kv_page_indices[kv_page_idx_idxs]

                # Gather keys and values for all pages in the chunk at once
                # Shape: [chunk_size, page_size, num_kv_heads, head_dim]
                chunk_keys = kv_cache_at_layer[chunk_kv_page_indices, 0]
                chunk_values = kv_cache_at_layer[chunk_kv_page_indices, 1]

                # Reshape to concatenate pages as one page:
                # [chunk_size * page_size, num_kv_heads, head_dim]
                paged_keys = chunk_keys.view(
                    -1, chunk_keys.shape[-2], chunk_keys.shape[-1]
                )
                paged_values = chunk_values.view(
                    -1, chunk_values.shape[-2], chunk_values.shape[-1]
                )

                paged_keys = torch.repeat_interleave(
                    paged_keys, self.num_key_value_groups, dim=1
                )
                paged_values = torch.repeat_interleave(
                    paged_values, self.num_key_value_groups, dim=1
                )

                chunk_size = len(page_cnts)
                mask_start = page_cnts[0] * kv_page_size
                mask_end = mask_start + chunk_size * kv_page_size
                paged_mask = mask[:, mask_start:mask_end].unsqueeze(0)

                sum_val, sum_exp, max_score = self._attend_one_page(
                    query,
                    paged_keys,
                    paged_mask,
                    paged_values,
                    sum_exp,
                    sum_val,
                    max_score,
                )

            # Attend to the last page
            page_cnt = kv_page_end - kv_page_start - 1
            kv_page_idx_idx = kv_page_end - 1

            kv_page_idx = kv_page_indices[kv_page_idx_idx]
            paged_keys = kv_cache_at_layer[kv_page_idx, 0][:kv_last_page_len]
            paged_values = kv_cache_at_layer[kv_page_idx, 1][:kv_last_page_len]

            paged_keys = torch.repeat_interleave(
                paged_keys, self.num_key_value_groups, dim=1
            )
            paged_values = torch.repeat_interleave(
                paged_values, self.num_key_value_groups, dim=1
            )

            paged_mask_offset = page_cnt * kv_page_size
            paged_mask = mask[:, paged_mask_offset : paged_mask_offset + kv_page_size][
                ..., :kv_last_page_len
            ]
            paged_mask = paged_mask.unsqueeze(0)

            sum_val, sum_exp, max_score = self._attend_one_page(
                query, paged_keys, paged_mask, paged_values, sum_exp, sum_val, max_score
            )

            adjusted_sinks = self.sinks.unsqueeze(-1) - max_score
            adjusted_sinks = torch.exp(adjusted_sinks)
            sum_exp += adjusted_sinks

            attn_output = sum_val / sum_exp.unsqueeze(-1)
            attn_output = rearrange(attn_output, "h q d -> q (h d)")

            attn_output = self.o_proj(attn_output.to(queries.dtype))

            output_embeds[q_start:q_end, :] = attn_output

        return output_embeds

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.IntTensor,
        qo_indptr: torch.IntTensor,
        kv_cache_at_layer: torch.Tensor,
        kv_page_indices: torch.IntTensor,
        kv_page_indptr: torch.IntTensor,
        kv_last_page_lens: torch.IntTensor,
        batch_indices: torch.IntTensor,
        batch_positions: torch.IntTensor,
        attention_mask: torch.Tensor,
        adapter_subpass: AdapterSubpass | None,
        activation_buffers: dict,
    ) -> torch.Tensor:
        """Forward pass through the attention module."""
        n, _ = hidden_states.size()

        qkv_states = self._qkv_projection(hidden_states, activation_buffers)

        query_states, key_states, value_states = torch.split(
            qkv_states, [self.q_size, self.k_size, self.v_size], dim=-1
        )

        # apply adapters if provided
        if adapter_subpass is not None:
            adapter_subpass.execute(
                self.layer_idx,
                hidden_states,
                q_state=query_states,
                k_state=key_states,
                v_state=value_states,
            )

        # Reshape for multi-head attention
        query_states = query_states.view(n, self.num_attention_heads, self.head_dim)
        key_states = key_states.view(n, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(n, self.num_key_value_heads, self.head_dim)

        # Apply rotary embedding
        query_states, key_states = self.rope(query_states, key_states, position_ids)

        # Store current KV states in FlashInfer cache for future use
        ops.append_paged_kv_cache(
            append_key=key_states,
            append_value=value_states,
            batch_indices=batch_indices,
            positions=batch_positions,
            paged_kv_cache=kv_cache_at_layer[self.layer_idx],
            kv_indices=kv_page_indices,
            kv_indptr=kv_page_indptr,
            kv_last_page_len=kv_last_page_lens,
            kv_layout="NHD",
        )

        new_attn_output = self._paged_attention(
            query_states,
            qo_indptr,
            kv_page_indptr,
            kv_last_page_lens,
            kv_page_indices,
            attention_mask,
            kv_cache_at_layer[self.layer_idx],
        )

        return new_attn_output

    def _qkv_projection(self, hidden_states: torch.Tensor, activation_buffers: dict):
        """QKV projection."""
        n = hidden_states.shape[0]
        if n > activation_buffers["qkv_buffer"].shape[0]:
            raise RuntimeError(
                f"Batch size {n} exceeds max_batch_tokens "
                f"{activation_buffers['qkv_buffer'].shape[0]}"
            )

        qkv_buffer = activation_buffers["qkv_buffer"][:n]
        torch.addmm(
            self.qkv_proj.bias if self.qkv_proj.bias is not None
            else torch.zeros(self.qkv_proj.out_features, device=hidden_states.device, dtype=hidden_states.dtype),
            hidden_states,
            self.qkv_proj.weight.t(),
            out=qkv_buffer
        )
        return qkv_buffer


class GptOssRouter(nn.Module):
    """GPT OSS Router for selecting top-k experts."""

    def __init__(self, config: GptOssArch):
        """Initialize the GPT OSS Router."""
        super().__init__()
        self.experts_per_token = config.experts_per_token
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size

        self.weight = nn.Parameter(
            torch.empty(
                config.num_experts,
                config.hidden_size,
                device=torch.device(config.device),
                dtype=config.dtype,
            )
        )
        self.bias = nn.Parameter(
            torch.empty(
                config.num_experts,
                device=torch.device(config.device),
                dtype=config.dtype,
            )
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the router."""
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = torch.nn.functional.linear(  # pylint: disable=not-callable
            hidden_states, self.weight, self.bias
        )

        router_top_value, router_indices = torch.topk(
            router_logits, self.experts_per_token, dim=-1, sorted=True
        )

        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1)
        router_scores = torch.zeros_like(router_logits).scatter_(
            1, router_indices, router_top_value
        )
        return router_scores, router_indices


class GptOssExperts(nn.Module):
    """GPT OSS Experts layer containing the actual expert parameters."""

    def __init__(self, config: GptOssArch):
        """Initialize the GPT OSS Experts layer."""
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.swiglu_limit = config.swiglu_limit

        self.gate_up_proj = nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.intermediate_size * 2,
                    config.hidden_size,
                ),
                device=torch.device(config.device),
                dtype=config.dtype,
            )
        )
        self.gate_up_proj_bias = nn.Parameter(
            torch.empty(
                (config.num_experts, config.intermediate_size * 2),
                device=torch.device(config.device),
                dtype=config.dtype,
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.hidden_size,
                    config.intermediate_size,
                ),
                device=torch.device(config.device),
                dtype=config.dtype,
            )
        )
        self.down_proj_bias = nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                device=torch.device(config.device),
                dtype=config.dtype,
            )
        )

    def forward(self, t: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """Forward pass through the experts."""

        # Gate and Up projection
        gate_up_proj = self.gate_up_proj[expert_indices, ...]
        gate_up_proj_bias = self.gate_up_proj_bias[expert_indices, ...]

        t = torch.einsum("beck,bk->bec", gate_up_proj, t) + gate_up_proj_bias

        # Inline swiglu function
        x_glu, x_linear = t[..., ::2], t[..., 1::2]

        # Clamp the input values
        x_glu = x_glu.clamp(min=None, max=self.swiglu_limit)
        x_linear = x_linear.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        out_glu = x_glu * torch.sigmoid(1.702 * x_glu)

        # Add an extra bias of 1 to the linear layer
        t = out_glu * (x_linear + 1)

        # Down projection
        down_proj = self.down_proj[expert_indices, ...]
        down_proj_bias = self.down_proj_bias[expert_indices, ...]

        t = torch.einsum("beck,bek->bec", down_proj, t) + down_proj_bias

        return t


class GptOssMlp(nn.Module):
    """GPT OSS MLP layer with Mixture of Experts."""

    def __init__(self, config: GptOssArch):
        """Initialize the GPT OSS MLP layer."""
        super().__init__()
        self.config = config
        self.router = GptOssRouter(config)
        self.experts = GptOssExperts(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP layer."""
        # Router determines expert selection and weights
        router_scores, router_indices = self.router(x)

        # Extract the weights for selected experts
        expert_weights = torch.gather(router_scores, 1, router_indices)

        # Forward through experts
        t = self.experts(x, router_indices)

        # Weighted sum of experts
        t = torch.einsum("bec,be->bc", t, expert_weights)

        return t


class GptOssDecoderLayer(nn.Module):
    """GPT OSS decoder layer."""

    def __init__(self, config: GptOssArch, layer_idx: int):
        """Initialize the GPT OSS decoder layer."""
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = GptOssRMSNorm(config.hidden_size, device=config.device)
        self.rope = GptOssRotaryEmbedding(
            config.head_size,
            int(config.rope_theta),
            torch.float32,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            device=torch.device(config.device),
            max_position_id=131072,
        )
        self.self_attn = GptOssAttention(config, layer_idx, self.rope)
        self.mlp = GptOssMlp(config)
        self.post_attention_layernorm = GptOssRMSNorm(
            config.hidden_size, device=config.device
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_cache_at_layer: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_positions: torch.Tensor,
        full_mask: torch.Tensor,
        window_mask: torch.Tensor,
        adapter_subpass: AdapterSubpass | None,
        activation_buffers: dict,
    ) -> torch.Tensor:
        """Forward pass through the decoder layer."""
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            qo_indptr=qo_indptr,
            kv_cache_at_layer=kv_cache_at_layer,
            kv_page_indices=kv_page_indices,
            kv_page_indptr=kv_page_indptr,
            kv_last_page_lens=kv_last_page_lens,
            batch_indices=batch_indices,
            batch_positions=batch_positions,
            attention_mask=window_mask if self.layer_idx % 2 == 0 else full_mask,
            adapter_subpass=adapter_subpass,
            activation_buffers=activation_buffers,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states


class GptOssModel(nn.Module):
    """GPT OSS model with FlashInfer support."""

    def __init__(self, config: GptOssArch):
        """Initialize the GPT OSS model."""
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=0,
            device=torch.device(config.device),
            dtype=config.dtype,
        )
        self.layers = nn.ModuleList(
            [
                GptOssDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_layers)
            ]
        )
        self.norm = GptOssRMSNorm(
            config.hidden_size,
            device=config.device,
        )
        self.sliding_window = config.sliding_window

        # Pre-allocate activation buffers to reserve memory and cap maximum usage.
        # These buffers reserve GPU memory upfront, making the memory footprint predictable
        # and preventing OOM errors that would otherwise occur mid-inference.
        # Note: For GptOss, only attention buffers are pre-allocated due to the
        # complex MoE architecture with custom einsum operations.
        self.activation_buffers = self._create_activation_buffers(config)

    def _create_activation_buffers(self, config: GptOssArch) -> dict:
        """Create pre-allocated activation buffers based on max_batch_tokens.

        These buffers are sized for the maximum batch size and will be reused across
        all forward passes to prevent dynamic memory allocation.

        Returns:
            Dictionary of pre-allocated tensor buffers
        """
        max_tokens = config.max_batch_tokens
        hidden_size = config.hidden_size
        num_query_heads = config.num_query_heads
        num_kv_heads = config.num_key_value_heads
        head_size = config.head_size
        device = config.device
        dtype = config.dtype

        print(
            f"Pre-allocating activation buffers for max_batch_tokens={max_tokens}, "
            f"hidden_size={hidden_size}"
        )

        buffers = {
            # QKV projection output: [max_tokens, (num_query_heads + 2*num_kv_heads) * head_size]
            "qkv_buffer": torch.empty(
                (max_tokens, (num_query_heads + 2 * num_kv_heads) * head_size),
                device=device,
                dtype=dtype,
            ),
        }

        # Calculate total memory allocated
        total_bytes = sum(buf.numel() * buf.element_size() for buf in buffers.values())
        print(
            f"Allocated {total_bytes / 1e9:.2f}GB for activation buffers across {len(buffers)} tensors"
        )

        return buffers

    def forward(
        self,
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_cache_at_layer: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        custom_mask: torch.Tensor,
        single_token_inference_mode: bool,
        adapter_subpass: AdapterSubpass | None,
    ) -> torch.Tensor:
        """Forward pass through the GPT OSS model."""

        # The current naive implementation does not distinguish between
        # single-token inference mode and batch inference mode
        _ = single_token_inference_mode

        hidden_states = input_embeds
        n, _ = hidden_states.size()

        page_size = kv_cache_at_layer[0].shape[2]

        batch_indices, batch_positions = ops.get_batch_indices_positions(
            append_indptr=qo_indptr,
            seq_lens=ops.get_seq_lens(kv_page_indptr, kv_last_page_lens, page_size),
            nnz=n,
        )

        batch_num = len(qo_indptr) - 1

        full_mask = custom_mask
        window_mask = custom_mask.clone()

        # For window attention layers, set the mask to 0 for positions that are
        # outside the sliding window.
        mask_offset = 0
        for batch_idx in range(batch_num):
            q_start = qo_indptr[batch_idx]
            q_end = qo_indptr[batch_idx + 1]

            kv_page_start = kv_page_indptr[batch_idx]
            kv_page_end = kv_page_indptr[batch_idx + 1]

            query_len = int(q_end - q_start)
            seq_len = int(
                (kv_page_end - kv_page_start - 1) * page_size
                + kv_last_page_lens[batch_idx]
            )
            mask_len = seq_len * query_len

            mask = window_mask[mask_offset : mask_offset + mask_len]
            mask_offset += mask_len

            mask = mask.view(query_len, seq_len)

            pos_id = position_ids[q_start:q_end]
            for q_idx in range(query_len):
                mask[
                    q_idx, : max(0, int(pos_id[q_idx]) - (self.sliding_window - 1))
                ] = 0

        full_mask = torch.where(
            full_mask,
            torch.tensor(0.0, dtype=input_embeds.dtype, device=input_embeds.device),
            torch.tensor(
                float("-inf"), dtype=input_embeds.dtype, device=input_embeds.device
            ),
        )
        window_mask = torch.where(
            window_mask,
            torch.tensor(0.0, dtype=input_embeds.dtype, device=input_embeds.device),
            torch.tensor(
                float("-inf"), dtype=input_embeds.dtype, device=input_embeds.device
            ),
        )

        for decoder_layer in self.layers:

            layer_outputs = decoder_layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                qo_indptr=qo_indptr,
                kv_cache_at_layer=kv_cache_at_layer,
                kv_page_indices=kv_page_indices,
                kv_page_indptr=kv_page_indptr,
                kv_last_page_lens=kv_last_page_lens,
                batch_indices=batch_indices,
                batch_positions=batch_positions,
                full_mask=full_mask,
                window_mask=window_mask,
                adapter_subpass=adapter_subpass,
                activation_buffers=self.activation_buffers,
            )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return hidden_states


class GptOssForCausalLM(nn.Module):
    """GPT OSS model for causal language modeling."""

    def __init__(self, config: GptOssArch):
        """Initialize the GPT OSS causal LM model."""
        super().__init__()
        self.config = config
        self.model = GptOssModel(config)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=torch.device(config.device),
            dtype=config.dtype,
        )

    def forward(self):
        """
        Should not be called. Method 'forward' is abstract in class
        'torch.nn.modules.module' so must be overridden in child class.
        """
        raise NotImplementedError("Should not be called")
