"""Qwen 3 Large Language Model Architecture (Qwen3)"""

from __future__ import annotations
from typing import List, Dict, Optional

import torch
from torch import nn

from common import AdapterSubpass, TensorLoader, Qwen3Arch
import flashinfer as ops

VERSION = "0.1.0"


def create_fusion_map(model: nn.Module):
    """
    Analyzes the model and creates a map for fusing weights.

    Returns:
        A dictionary mapping {fused_tensor_name: {"sources": [source_names], "dim": cat_dim}}.
    """
    fusion_map = {}
    for name, module in model.named_modules():
        # --- Rule for Qwen3Attention QKV Fusion ---
        if isinstance(module, Qwen3Attention):
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

        # --- Rule for Qwen3Mlp Gate/Up Fusion ---
        elif isinstance(module, Qwen3Mlp):
            # Handle weights
            target_w = f"{name}.gate_up_proj.weight"
            sources_w = [f"{name}.gate_proj.weight", f"{name}.up_proj.weight"]
            fusion_map[target_w] = {"sources": sources_w, "dim": 0, "op": "fusion"}

            # Handle biases (Qwen3 uses bias in MLP layers)
            target_b = f"{name}.gate_up_proj.bias"
            sources_b = [f"{name}.gate_proj.bias", f"{name}.up_proj.bias"]
            fusion_map[target_b] = {"sources": sources_b, "dim": 0, "op": "fusion"}

    return fusion_map


class Qwen3TensorLoader(TensorLoader):
    """
    TensorLoader implementation for Qwen3 models.

    Handles fusion of QKV projections and gate/up projections based on the
    model architecture.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize the tensor loader with a model instance.

        Args:
            model: The Qwen3 model instance
        """
        self.model = model
        self._fusion_map = create_fusion_map(model)

        # Create reverse mapping for quick lookup
        self._source_to_target = {
            source: target
            for target, details in self._fusion_map.items()
            for source in details["sources"]
        }

    def query(self, runtime_tensor_name: str) -> List[str]:
        """
        Query which checkpoint tensors are needed for a given runtime tensor.

        Args:
            runtime_tensor_name: Name of the tensor in the runtime model

        Returns:
            List of checkpoint tensor names needed to construct the runtime tensor
        """
        if runtime_tensor_name in self._fusion_map:
            # This is a fusion target, return its source tensors
            return self._fusion_map[runtime_tensor_name]["sources"]
        else:
            # This is a regular tensor, return itself
            return [runtime_tensor_name]

    def load(
        self, runtime_tensor_name: str, checkpoint_tensors: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Load and transform checkpoint tensors into the runtime tensor.

        Args:
            runtime_tensor_name: Name of the tensor in the runtime model
            checkpoint_tensors: Dictionary mapping checkpoint tensor names to their loaded tensors

        Returns:
            The constructed runtime tensor
        """
        if runtime_tensor_name in self._fusion_map:
            # This is a fusion target, concatenate the source tensors
            fusion_info = self._fusion_map[runtime_tensor_name]
            source_names = fusion_info["sources"]
            concat_dim = fusion_info["dim"]

            # Get all source tensors in order
            source_tensors = [checkpoint_tensors[name] for name in source_names]

            # Concatenate along the specified dimension
            return torch.cat(source_tensors, dim=concat_dim)
        else:
            # This is a regular tensor, return it directly
            if runtime_tensor_name not in checkpoint_tensors:
                raise KeyError(
                    f"Tensor '{runtime_tensor_name}' not found in checkpoint tensors"
                )
            return checkpoint_tensors[runtime_tensor_name]


class Qwen3Mlp(nn.Module):
    """Qwen3 MLP layer with SiLU activation function and bias in feed-forward layers."""

    def __init__(self, config: Qwen3Arch):
        """Initialize the Qwen3 MLP layer."""
        super().__init__()
        self.config = config
        self.gate_up_proj = nn.Linear(
            config.hidden_size,
            2 * config.intermediate_size,  # Double the output dimension
            bias=False,  # Qwen3 0.6B does not use bias in feed-forward layers
            device=config.device,
            dtype=config.dtype,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,  # Qwen3 0.6B does not use bias in feed-forward layers
            device=config.device,
            dtype=config.dtype,
        )
        self.act_fn = nn.SiLU()

    def forward(self, x):
        """Forward pass through the MLP layer."""
        gate_up_proj_out = self.gate_up_proj(x)
        gate_proj, up_proj = gate_up_proj_out.chunk(2, dim=-1)

        interim = self.act_fn(gate_proj) * up_proj

        down_proj = self.down_proj(interim)
        return down_proj


class Qwen3Attention(nn.Module):
    """Qwen3 attention module with FlashInfer support and QK normalization."""

    def __init__(self, config: Qwen3Arch, layer_idx: int):
        """Initialize the Qwen3 attention module."""
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Define the output sizes for Q, K, and V for clarity
        self.q_size = config.num_query_heads * config.head_size
        self.k_size = config.num_key_value_heads * config.head_size
        self.v_size = config.num_key_value_heads * config.head_size

        # Qwen3 uses attention_bias for QKV projections
        attention_bias = getattr(config, "attention_bias", False)

        self.qkv_proj = nn.Linear(
            config.hidden_size,
            self.q_size + self.k_size + self.v_size,
            bias=attention_bias,
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

        # Qwen3 uses QK normalization - critical for stability
        self.q_norm = nn.RMSNorm(
            config.head_size,
            eps=config.rms_norm_eps,
            device=config.device,
            dtype=config.dtype,
        )
        self.k_norm = nn.RMSNorm(
            config.head_size,
            eps=config.rms_norm_eps,
            device=config.device,
            dtype=config.dtype,
        )

    def forward(
        self,
        wrapper,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache_at_layer: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_positions: torch.Tensor,
        adapter_subpass: Optional[AdapterSubpass],
    ) -> torch.Tensor:
        """Forward pass through the attention module."""

        n, _ = hidden_states.size()

        qkv_states = self.qkv_proj(hidden_states)
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

        # Reshape and continue as before
        query_states = query_states.view(
            n, self.config.num_query_heads, self.config.head_size
        )
        key_states = key_states.view(
            n, self.config.num_key_value_heads, self.config.head_size
        )
        value_states = value_states.view(
            n, self.config.num_key_value_heads, self.config.head_size
        )

        # Apply QK normalization (critical for Qwen3)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Apply RoPE with Qwen3 specific parameters
        ops.apply_rope_pos_ids_inplace(
            q=query_states,
            k=key_states,
            pos_ids=position_ids,
            rope_theta=self.config.rope_theta,
        )

        # Ensure query_states matches the configured dtype for FlashInfer plan
        # if query_states.dtype != self.config.dtype:
        #     query_states = query_states.to(self.config.dtype)

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

        attn_output = wrapper.run(query_states, kv_cache_at_layer[self.layer_idx])
        attn_output = attn_output.reshape(n, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output


class Qwen3DecoderLayer(nn.Module):
    """Qwen3 decoder layer."""

    def __init__(self, config: Qwen3Arch, layer_idx: int):
        """Initialize the Qwen3 decoder layer."""
        super().__init__()

        self.self_attn = Qwen3Attention(config, layer_idx)

        self.mlp = Qwen3Mlp(config)
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
        kv_cache_at_layer: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_positions: torch.Tensor,
        adapter_subpass: Optional[AdapterSubpass],
    ) -> torch.Tensor:
        """Forward pass through the decoder layer."""
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
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
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3Model(nn.Module):
    """Qwen3 model with FlashInfer support."""

    def __init__(self, config: Qwen3Arch):
        """Initialize the Qwen3 model."""
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
                Qwen3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_layers)
            ]
        )
        self.norm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            device=config.device,
            dtype=config.dtype,
        )

        self.workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=config.device
        )
        self.wrapper_decode = ops.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )
        self.wrapper_append = ops.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )

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
        adapter_subpass: Optional[AdapterSubpass],
    ) -> torch.Tensor:
        """Forward pass through the Qwen3 model."""
        hidden_states = input_embeds
        n, _ = hidden_states.size()

        page_size = kv_cache_at_layer[0].shape[2]

        batch_indices, batch_positions = ops.get_batch_indices_positions(
            append_indptr=qo_indptr,
            seq_lens=ops.get_seq_lens(kv_page_indptr, kv_last_page_lens, page_size),
            nnz=n,
        )

        # check if its decoding (qo_indptr is )
        if single_token_inference_mode:
            self.wrapper_decode.plan(
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
            wrapper = self.wrapper_decode
        else:
            self.wrapper_append.plan(
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
            wrapper = self.wrapper_append

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
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
            )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 model for causal language modeling."""

    def __init__(self, config: Qwen3Arch):
        """Initialize the Qwen3 causal LM model."""
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=config.device,
            dtype=config.dtype,
        )

    def forward(self):
        """
        Should not be called. Method 'forward' is abstract in class
        'torch.nn.modules.module' so must be overridden in child class.
        """
        raise NotImplementedError("Should not be called")


__all__ = [
    "create_fusion_map",
    "Qwen3TensorLoader",
    "Qwen3Mlp",
    "Qwen3Attention",
    "Qwen3DecoderLayer",
    "Qwen3Model",
    "Qwen3ForCausalLM",
    "VERSION",
]
