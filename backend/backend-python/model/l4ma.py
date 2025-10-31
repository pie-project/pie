"""Llama-Like Large Language Model Architecture (L4MA).

Supports both FlashInfer and metal_kernels backends:
- metal_kernels: Metal-accelerated operations for macOS with Apple Silicon
- FlashInfer: CUDA-accelerated operations for other platforms
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Optional

import torch
from torch import nn

# Safe import of adapter functionality
from adapter_utils import AdapterSubpass
from model.config import CommonArch, ModelConfig
from platform_detection import is_apple_silicon
from profiler import start_profile, profile_attention

# Direct import of backend operations based on platform
if is_apple_silicon():
    try:
        import metal_kernels.ops as ops  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(f"metal_kernels backend is not available: {e}") from e
else:
    try:
        import flashinfer as ops  # type: ignore[import-not-found,no-redef]
    except ImportError as e:
        raise RuntimeError(f"flashinfer backend is not available: {e}") from e

VERSION = "0.1.0"


def calculate_activation_memory_bytes(
    max_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    dtype_size: int,
) -> int:
    """Calculate total memory needed for activation buffers.

    Args:
        max_tokens: Maximum number of tokens in a batch
        hidden_size: Hidden layer size
        intermediate_size: MLP intermediate size
        num_query_heads: Number of query attention heads
        num_kv_heads: Number of key/value attention heads
        head_size: Size of each attention head
        dtype_size: Size of data type in bytes (e.g., 2 for float16)

    Returns:
        Total bytes needed for all activation buffers
    """
    qkv_size = (
        max_tokens * (num_query_heads + 2 * num_kv_heads) * head_size * dtype_size
    )
    attn_out_size = max_tokens * num_query_heads * head_size * dtype_size
    o_proj_size = max_tokens * hidden_size * dtype_size
    gate_up_size = max_tokens * 2 * intermediate_size * dtype_size
    mlp_act_size = max_tokens * intermediate_size * dtype_size
    down_proj_size = max_tokens * hidden_size * dtype_size
    layer_out_size = 2 * max_tokens * hidden_size * dtype_size

    return (
        qkv_size
        + attn_out_size
        + o_proj_size
        + gate_up_size
        + mlp_act_size
        + down_proj_size
        + layer_out_size
    )


@dataclass
class L4maArch(CommonArch):
    """L4MA/Llama specific architecture configuration."""

    rope_factor: float
    rope_high_frequency_factor: float
    rope_low_frequency_factor: float
    rope_theta: float

    @staticmethod
    def from_config(cfg: ModelConfig) -> "L4maArch":
        """Parse L4MA/Llama-specific architecture configuration."""
        # Get common architecture fields
        common_arch_dict = cfg.get_common_arch_dict()

        # Get all the fields for the architecture section to grab other
        # architecture-specific fields
        arch_dict = cfg.get_required_key(cfg.root, "architecture")

        # Get RoPE configuration (Llama-style with factor-based scaling)
        rope_dict = cfg.get_required_key(arch_dict, "rope")
        rope_factor = cfg.get_required_key(rope_dict, "factor")
        rope_high_frequency_factor = cfg.get_required_key(
            rope_dict, "high_frequency_factor"
        )
        rope_low_frequency_factor = cfg.get_required_key(
            rope_dict, "low_frequency_factor"
        )
        rope_theta = cfg.get_required_key(rope_dict, "theta")

        return L4maArch(
            # Common fields
            **common_arch_dict,
            # L4MA-specific fields
            rope_factor=rope_factor,
            rope_high_frequency_factor=rope_high_frequency_factor,
            rope_low_frequency_factor=rope_low_frequency_factor,
            rope_theta=rope_theta,
        )


def _infer_page_size(kv_cache_at_layer) -> int:
    """Infer the page size from the KV cache tensor shape."""
    if not kv_cache_at_layer:
        raise ValueError("kv_cache_at_layer must contain at least one tensor")
    first_layer = kv_cache_at_layer[0]
    if first_layer.ndim < 3:
        raise ValueError("Unexpected KV cache tensor shape; expected >= 3 dimensions")
    return int(first_layer.shape[2])


def create_fusion_map(model: nn.Module):
    """
    Analyzes the model and creates a map for fusing weights.

    Returns:
        A dictionary mapping {fused_tensor_name: {"sources": [source_names], "dim": cat_dim}}.
    """

    fusion_map = {}
    for name, module in model.named_modules():
        # --- Rule for L4maAttention QKV Fusion ---
        if isinstance(module, L4maAttention):
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
                fusion_map[target_b] = {"sources": sources_b, "dim": 0, "op": "fusion"}

        # --- Rule for L4maMlp Gate/Up Fusion ---
        elif isinstance(module, L4maMlp):
            target_w = f"{name}.gate_up_proj.weight"
            sources_w = [f"{name}.gate_proj.weight", f"{name}.up_proj.weight"]
            fusion_map[target_w] = {"sources": sources_w, "dim": 0, "op": "fusion"}

    return fusion_map


class L4maMlp(nn.Module):
    """Feed-forward network block used in each decoder layer."""

    def __init__(self, config: L4maArch):
        super().__init__()
        self.config = config
        self.gate_up_proj = nn.Linear(
            config.hidden_size,
            2 * config.intermediate_size,  # Double the output dimension
            bias=False,
            device=config.device,
            dtype=config.dtype,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            device=config.device,
            dtype=config.dtype,
        )
        self.act_fn = nn.SiLU()

    def forward(self, x, activation_buffers: dict):
        """Forward pass through the MLP layer.

        Args:
            x: Input tensor
            activation_buffers: Pre-allocated buffers
        """
        gate_up_proj_out = self._gate_up_projection(x, activation_buffers)
        gate_proj, up_proj = gate_up_proj_out.chunk(2, dim=-1)
        interim = self._silu_activation(gate_proj, up_proj, activation_buffers)
        down_proj = self._down_projection(interim, activation_buffers)
        return down_proj

    def _gate_up_projection(self, x, activation_buffers: dict):
        """Gate/Up projection."""
        n = x.shape[0]
        if n > activation_buffers["gate_up_buffer"].shape[0]:
            raise RuntimeError(
                f"Batch size {n} exceeds max_batch_tokens "
                f"{activation_buffers['gate_up_buffer'].shape[0]}"
            )

        gate_up_buffer = activation_buffers["gate_up_buffer"][:n]
        torch.addmm(
            (
                self.gate_up_proj.bias
                if self.gate_up_proj.bias is not None
                else torch.zeros(
                    self.gate_up_proj.out_features, device=x.device, dtype=x.dtype
                )
            ),
            x,
            self.gate_up_proj.weight.t(),
            out=gate_up_buffer,
        )
        return gate_up_buffer

    def _silu_activation(self, gate_proj, up_proj, activation_buffers: dict):
        """SiLU activation."""
        n = gate_proj.shape[0]
        if n > activation_buffers["mlp_act_buffer"].shape[0]:
            raise RuntimeError(
                f"Batch size {n} exceeds max_batch_tokens "
                f"{activation_buffers['mlp_act_buffer'].shape[0]}"
            )

        mlp_act_buffer = activation_buffers["mlp_act_buffer"][:n]
        torch.mul(self.act_fn(gate_proj), up_proj, out=mlp_act_buffer)
        return mlp_act_buffer

    def _down_projection(self, interim, activation_buffers: dict):
        """Down projection."""
        n = interim.shape[0]
        if n > activation_buffers["down_proj_buffer"].shape[0]:
            raise RuntimeError(
                f"Batch size {n} exceeds max_batch_tokens "
                f"{activation_buffers['down_proj_buffer'].shape[0]}"
            )

        down_proj_buffer = activation_buffers["down_proj_buffer"][:n]
        torch.addmm(
            (
                self.down_proj.bias
                if self.down_proj.bias is not None
                else torch.zeros(
                    self.down_proj.out_features,
                    device=interim.device,
                    dtype=interim.dtype,
                )
            ),
            interim,
            self.down_proj.weight.t(),
            out=down_proj_buffer,
        )
        return down_proj_buffer


class L4maAttention(nn.Module):
    """Multi-head attention block for the decoder."""

    def __init__(self, config: L4maArch, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Define the output sizes for Q, K, and V for clarity
        self.q_size = config.num_query_heads * config.head_size
        self.k_size = config.num_key_value_heads * config.head_size
        self.v_size = config.num_key_value_heads * config.head_size

        self.qkv_proj = nn.Linear(
            config.hidden_size,
            self.q_size + self.k_size + self.v_size,
            bias=config.use_qkv_bias,
            device=config.device,
            dtype=config.dtype,
        )

        self.o_proj = nn.Linear(
            config.num_query_heads * config.head_size,
            config.hidden_size,
            bias=False,
            device=config.device,
            dtype=config.dtype,
        )

    def forward(
        self,
        wrapper,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache_at_layer: Sequence[torch.Tensor],
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_positions: torch.Tensor,
        adapter_subpass: Optional[AdapterSubpass],
        activation_buffers: dict,
    ) -> torch.Tensor:
        """Attention forward pass using FlashInfer ops directly."""

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
        query_states = query_states.view(
            n, self.config.num_query_heads, self.config.head_size
        )
        key_states = key_states.view(
            n, self.config.num_key_value_heads, self.config.head_size
        )
        value_states = value_states.view(
            n, self.config.num_key_value_heads, self.config.head_size
        )

        # Apply RoPE encoding
        ops.apply_llama31_rope_pos_ids_inplace(
            q=query_states,
            k=key_states,
            pos_ids=position_ids,
            rope_scale=self.config.rope_factor,
            rope_theta=self.config.rope_theta,
            low_freq_factor=self.config.rope_low_frequency_factor,
            high_freq_factor=self.config.rope_high_frequency_factor,
        )

        if query_states.dtype != self.config.dtype:
            query_states = query_states.to(self.config.dtype)

        # Append to KV cache
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

        with profile_attention(
            self.layer_idx, query_states, kv_cache_at_layer[self.layer_idx]
        ):
            # FlashInfer wrapper.run() creates its own output tensor
            # Future: could be optimized to write to attn_out_buffer if wrapper supports it
            attn_output = wrapper.run(query_states, kv_cache_at_layer[self.layer_idx])

        attn_output = attn_output.reshape(n, -1)
        return self._output_projection(attn_output, activation_buffers)

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
            (
                self.qkv_proj.bias
                if self.qkv_proj.bias is not None
                else torch.zeros(
                    self.qkv_proj.out_features,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
            ),
            hidden_states,
            self.qkv_proj.weight.t(),
            out=qkv_buffer,
        )
        return qkv_buffer

    def _output_projection(self, attn_output: torch.Tensor, activation_buffers: dict):
        """Output projection."""
        n = attn_output.shape[0]
        if n > activation_buffers["o_proj_buffer"].shape[0]:
            raise RuntimeError(
                f"Batch size {n} exceeds max_batch_tokens "
                f"{activation_buffers['o_proj_buffer'].shape[0]}"
            )

        o_proj_buffer = activation_buffers["o_proj_buffer"][:n]
        torch.addmm(
            (
                self.o_proj.bias
                if self.o_proj.bias is not None
                else torch.zeros(
                    self.o_proj.out_features,
                    device=attn_output.device,
                    dtype=attn_output.dtype,
                )
            ),
            attn_output,
            self.o_proj.weight.t(),
            out=o_proj_buffer,
        )
        return o_proj_buffer


class L4maDecoderLayer(nn.Module):
    """Single decoder layer consisting of attention + MLP."""

    def __init__(self, config: L4maArch, layer_idx: int):
        super().__init__()

        self.self_attn = L4maAttention(config, layer_idx)

        self.mlp = L4maMlp(config)
        self.input_layernorm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            device=config.device,
            dtype=config.dtype,
        )
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            device=config.device,
            dtype=config.dtype,
        )

    def forward(
        self,
        wrapper,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache_at_layer: Sequence[torch.Tensor],
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_positions: torch.Tensor,
        adapter_subpass: Optional[AdapterSubpass],
        activation_buffers: dict,
    ) -> torch.Tensor:
        """Run the decoder layer."""

        with start_profile("input_norm"):
            residual = hidden_states
            hidden_states = self._input_normalization(hidden_states)

        with start_profile("attention"):
            hidden_states = self.self_attn(
                wrapper=wrapper,
                hidden_states=hidden_states,
                position_ids=position_ids,
                kv_cache_at_layer=kv_cache_at_layer,
                kv_page_indices=kv_page_indices,
                kv_page_indptr=kv_page_indptr,
                kv_last_page_lens=kv_last_page_lens,
                batch_indices=batch_indices,
                batch_positions=batch_positions,
                adapter_subpass=adapter_subpass,
                activation_buffers=activation_buffers,
            )

        with start_profile("attention_residual"):
            hidden_states = residual + hidden_states

        with start_profile("post_attn_norm"):
            residual = hidden_states
            hidden_states = self._post_attention_normalization(hidden_states)

        with start_profile("mlp"):
            hidden_states = self.mlp(
                hidden_states, activation_buffers=activation_buffers
            )

        with start_profile("mlp_residual"):
            hidden_states = residual + hidden_states

        return hidden_states

    def _input_normalization(self, hidden_states):
        """Input RMSNorm for Metal normalization kernel comparison."""
        return self.input_layernorm(hidden_states)

    def _post_attention_normalization(self, hidden_states):
        """Post-attention RMSNorm for Metal normalization kernel comparison."""
        return self.post_attention_layernorm(hidden_states)


class L4maModel(nn.Module):
    """Backbone model for the L4MA architecture."""

    def __init__(self, config: L4maArch):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=0,
            device=config.device,
            dtype=config.dtype,
        )
        self.layers = nn.ModuleList(
            [
                L4maDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_layers)
            ]
        )
        self.norm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            device=config.device,
            dtype=config.dtype,
        )

        # FlashInfer wrappers for attention operations
        self.workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=config.device
        )
        self.wrapper_decode = ops.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )
        self.wrapper_append = ops.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )

        # Pre-allocate activation buffers to reserve memory and cap maximum usage.
        # These buffers reserve GPU memory upfront, making the memory footprint predictable
        # and preventing OOM errors that would otherwise occur mid-inference.
        self.activation_buffers = self._create_activation_buffers(config)

    def _create_activation_buffers(self, config: L4maArch) -> dict:
        """Create pre-allocated activation buffers based on max_batch_tokens.

        These buffers are sized for the maximum batch size and will be reused across
        all forward passes to prevent dynamic memory allocation.

        Returns:
            Dictionary of pre-allocated tensor buffers
        """
        max_tokens = config.max_batch_tokens
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        num_query_heads = config.num_query_heads
        num_kv_heads = config.num_key_value_heads
        head_size = config.head_size
        device = config.device
        dtype = config.dtype

        print(
            f"Pre-allocating activation buffers for max_batch_tokens={max_tokens}, "
            f"hidden_size={hidden_size}, intermediate_size={intermediate_size}"
        )

        buffers = {
            # QKV projection output: [max_tokens, (num_query_heads + 2*num_kv_heads) * head_size]
            "qkv_buffer": torch.empty(
                (max_tokens, (num_query_heads + 2 * num_kv_heads) * head_size),
                device=device,
                dtype=dtype,
            ),
            # Attention output buffer: [max_tokens, num_query_heads * head_size]
            "attn_out_buffer": torch.empty(
                (max_tokens, num_query_heads * head_size),
                device=device,
                dtype=dtype,
            ),
            # O projection output buffer: [max_tokens, hidden_size]
            "o_proj_buffer": torch.empty(
                (max_tokens, hidden_size), device=device, dtype=dtype
            ),
            # MLP gate_up projection output: [max_tokens, 2 * intermediate_size]
            "gate_up_buffer": torch.empty(
                (max_tokens, 2 * intermediate_size), device=device, dtype=dtype
            ),
            # MLP activation output: [max_tokens, intermediate_size]
            "mlp_act_buffer": torch.empty(
                (max_tokens, intermediate_size), device=device, dtype=dtype
            ),
            # MLP down projection output: [max_tokens, hidden_size]
            "down_proj_buffer": torch.empty(
                (max_tokens, hidden_size), device=device, dtype=dtype
            ),
            # Layer output buffers (ping-pong pattern between layers)
            "layer_out_0": torch.empty(
                (max_tokens, hidden_size), device=device, dtype=dtype
            ),
            "layer_out_1": torch.empty(
                (max_tokens, hidden_size), device=device, dtype=dtype
            ),
        }

        # Calculate total memory allocated
        total_bytes = sum(buf.numel() * buf.element_size() for buf in buffers.values())
        print(
            f"Allocated {total_bytes / 1e9:.2f}GB for activation buffers "
            f"across {len(buffers)} tensors"
        )

        return buffers

    def forward(
        self,
        # input
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        qo_indptr: torch.Tensor,
        # kv cache
        kv_cache_at_layer: Sequence[torch.Tensor],
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        # mask
        custom_mask: torch.Tensor | None,
        single_token_inference_mode: bool,
        # subpasses
        adapter_subpass: Optional[AdapterSubpass],
    ) -> torch.Tensor:
        """Forward pass through all decoder layers."""

        with start_profile("model_setup"):
            hidden_states = input_embeds
            n, _ = hidden_states.size()

            page_size = _infer_page_size(kv_cache_at_layer)

            seq_lens = ops.get_seq_lens(
                kv_page_indptr,
                kv_last_page_lens,
                page_size,
            )

            batch_indices, batch_positions = ops.get_batch_indices_positions(
                append_indptr=qo_indptr,
                seq_lens=seq_lens,
                nnz=n,
            )

            if single_token_inference_mode:
                wrapper = self.wrapper_decode
                wrapper.plan(
                    indptr=kv_page_indptr,
                    indices=kv_page_indices,
                    last_page_len=kv_last_page_lens,
                    num_qo_heads=self.config.num_query_heads,
                    num_kv_heads=self.config.num_key_value_heads,
                    head_dim=self.config.head_size,
                    page_size=page_size,
                    pos_encoding_mode="NONE",
                    q_data_type=self.config.dtype,
                )
            else:
                wrapper = self.wrapper_append
                wrapper.plan(
                    qo_indptr=qo_indptr,
                    paged_kv_indptr=kv_page_indptr,
                    paged_kv_indices=kv_page_indices,
                    paged_kv_last_page_len=kv_last_page_lens,
                    num_qo_heads=self.config.num_query_heads,
                    num_kv_heads=self.config.num_key_value_heads,
                    head_dim_qk=self.config.head_size,
                    page_size=page_size,
                    custom_mask=custom_mask,
                    q_data_type=self.config.dtype,
                )

        with start_profile("decoder_layers"):
            for layer_idx, decoder_layer in enumerate(self.layers):
                with start_profile(f"layer_{layer_idx}"):
                    hidden_states = decoder_layer(
                        wrapper=wrapper,
                        hidden_states=hidden_states,
                        position_ids=position_ids,
                        kv_cache_at_layer=kv_cache_at_layer,
                        kv_page_indices=kv_page_indices,
                        kv_page_indptr=kv_page_indptr,
                        kv_last_page_lens=kv_last_page_lens,
                        batch_indices=batch_indices,
                        batch_positions=batch_positions,
                        adapter_subpass=adapter_subpass,
                        activation_buffers=self.activation_buffers,  # Pass pre-allocated buffers
                    )

        with start_profile("final_norm"):
            hidden_states = self.norm(hidden_states)

        return hidden_states


class L4maForCausalLM(nn.Module):
    """Top-level causal language model wrapper for L4MA architecture."""

    def __init__(self, config: L4maArch):
        super().__init__()
        self.config = config
        self.model = L4maModel(config)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=config.device,
            dtype=config.dtype,
        )

    def forward(self):  # pragma: no cover - interface parity placeholder
        """The handler uses dedicated methods rather than Module.forward."""
        raise NotImplementedError("Should not be called")


__all__ = [
    "L4maForCausalLM",
    "L4maModel",
    "L4maDecoderLayer",
    "L4maAttention",
    "L4maMlp",
    "create_fusion_map",
    "VERSION",
]
