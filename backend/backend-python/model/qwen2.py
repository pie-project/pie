"""Qwen 2 Large Language Model Architecture (Qwen2)"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from adapter_utils import AdapterSubpass
from model.config import CommonArch, ModelConfig
import flashinfer as ops

VERSION = "0.1.0"


@dataclass
class Qwen2Arch(CommonArch):
    """Qwen2 specific architecture configuration."""

    rope_theta: float

    @staticmethod
    def from_config(cfg: ModelConfig) -> "Qwen2Arch":
        """Parse Qwen2-specific architecture configuration."""
        # Get common architecture fields
        common_arch_dict = cfg.get_common_arch_dict()

        # Get all the fields for the architecture section to grab other
        # architecture-specific fields
        arch_dict = cfg.get_required_key(cfg.root, "architecture")

        # Get RoPE configuration
        rope_dict = cfg.get_required_key(arch_dict, "rope")
        rope_theta = cfg.get_required_key(rope_dict, "theta")

        return Qwen2Arch(
            # Common fields
            **common_arch_dict,
            # Qwen2-specific fields
            rope_theta=rope_theta,
        )


def create_fusion_map(model: nn.Module):
    """
    Analyzes the model and creates a map for fusing weights.

    Returns:
        A dictionary mapping {fused_tensor_name: {"sources": [source_names], "dim": cat_dim}}.
    """
    fusion_map = {}
    for name, module in model.named_modules():
        # --- Rule for Qwen2Attention QKV Fusion ---
        if isinstance(module, Qwen2Attention):
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

        # --- Rule for Qwen2Mlp Gate/Up Fusion ---
        elif isinstance(module, Qwen2Mlp):
            # Handle weights
            target_w = f"{name}.gate_up_proj.weight"
            sources_w = [f"{name}.gate_proj.weight", f"{name}.up_proj.weight"]
            fusion_map[target_w] = {"sources": sources_w, "dim": 0, "op": "fusion"}

            # Handle biases (Qwen2 typically doesn't use bias in MLP layers)
            if hasattr(module, "gate_up_proj") and module.gate_up_proj.bias is not None:
                target_b = f"{name}.gate_up_proj.bias"
                sources_b = [f"{name}.gate_proj.bias", f"{name}.up_proj.bias"]
                fusion_map[target_b] = {"sources": sources_b, "dim": 0, "op": "fusion"}

    return fusion_map


class Qwen2Mlp(nn.Module):
    """Qwen2 MLP layer with SiLU activation function."""

    def __init__(self, config: Qwen2Arch):
        """Initialize the Qwen2 MLP layer."""
        super().__init__()
        self.config = config
        self.gate_up_proj = nn.Linear(
            config.hidden_size,
            2 * config.intermediate_size,
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
            self.gate_up_proj.bias if self.gate_up_proj.bias is not None
            else torch.zeros(self.gate_up_proj.out_features, device=x.device, dtype=x.dtype),
            x,
            self.gate_up_proj.weight.t(),
            out=gate_up_buffer
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
            self.down_proj.bias if self.down_proj.bias is not None
            else torch.zeros(self.down_proj.out_features, device=interim.device, dtype=interim.dtype),
            interim,
            self.down_proj.weight.t(),
            out=down_proj_buffer
        )
        return down_proj_buffer


class Qwen2Attention(nn.Module):
    """Qwen2 attention module with FlashInfer support and Grouped Query Attention."""

    def __init__(self, config: Qwen2Arch, layer_idx: int):
        """Initialize the Qwen2 attention module."""
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
        kv_cache_at_layer: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_positions: torch.Tensor,
        adapter_subpass: Optional[AdapterSubpass],
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

        query_states = query_states.view(
            n, self.config.num_query_heads, self.config.head_size
        )
        key_states = key_states.view(
            n, self.config.num_key_value_heads, self.config.head_size
        )
        value_states = value_states.view(
            n, self.config.num_key_value_heads, self.config.head_size
        )

        ops.apply_rope_pos_ids_inplace(
            q=query_states,
            k=key_states,
            pos_ids=position_ids,
            rope_theta=self.config.rope_theta,
        )

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
            self.qkv_proj.bias if self.qkv_proj.bias is not None
            else torch.zeros(self.qkv_proj.out_features, device=hidden_states.device, dtype=hidden_states.dtype),
            hidden_states,
            self.qkv_proj.weight.t(),
            out=qkv_buffer
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
            self.o_proj.bias if self.o_proj.bias is not None
            else torch.zeros(self.o_proj.out_features, device=attn_output.device, dtype=attn_output.dtype),
            attn_output,
            self.o_proj.weight.t(),
            out=o_proj_buffer
        )
        return o_proj_buffer


class Qwen2DecoderLayer(nn.Module):
    """Qwen2 decoder layer."""

    def __init__(self, config: Qwen2Arch, layer_idx: int):
        """Initialize the Qwen2 decoder layer."""
        super().__init__()

        self.self_attn = Qwen2Attention(config, layer_idx)

        self.layer_idx = layer_idx

        self.mlp = Qwen2Mlp(config)
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
        activation_buffers: dict,
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
            activation_buffers=activation_buffers,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states, activation_buffers=activation_buffers)

        hidden_states = residual + hidden_states

        return hidden_states


class Qwen2Model(nn.Module):
    """Qwen2 model with FlashInfer support."""

    def __init__(self, config: Qwen2Arch):
        """Initialize the Qwen2 model."""
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
                Qwen2DecoderLayer(config, layer_idx)
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

        # Pre-allocate activation buffers to reserve memory and cap maximum usage.
        # These buffers reserve GPU memory upfront, making the memory footprint predictable
        # and preventing OOM errors that would otherwise occur mid-inference.
        # Note: Currently used for memory reservation. Future enhancement could use these
        # buffers directly with custom CUDA kernels or F.linear with output management.
        self.activation_buffers = self._create_activation_buffers(config)

    def _create_activation_buffers(self, config: Qwen2Arch) -> dict:
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
        adapter_subpass: Optional[AdapterSubpass],
    ) -> torch.Tensor:
        """Forward pass through the Qwen2 model."""
        hidden_states = input_embeds
        n, _ = hidden_states.size()

        page_size = kv_cache_at_layer[0].shape[2]

        batch_indices, batch_positions = ops.get_batch_indices_positions(
            append_indptr=qo_indptr,
            seq_lens=ops.get_seq_lens(kv_page_indptr, kv_last_page_lens, page_size),
            nnz=n,
        )

        # Theoretically, we should check if it's single-token inference mode and use
        # `self.wrapper_decode` instead of `self.wrapper_append`.
        # However, the current FlashInfer implementation of the decode wrapper does not support
        # does not support arbitrary ratio between the number of query and key-value heads.
        # For example, it returns errors when running the Qwen 2 14B model, where there are
        # 40 query heads and 8 key-value heads (5:1 ratio).
        #
        # For now, we just always use the append wrapper.
        _ = single_token_inference_mode

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
                activation_buffers=self.activation_buffers,
            )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return hidden_states


class Qwen2ForCausalLM(nn.Module):
    """Qwen2 model for causal language modeling."""

    def __init__(self, config: Qwen2Arch):
        """Initialize the Qwen2 causal LM model."""
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
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
    "Qwen2Mlp",
    "Qwen2Attention",
    "Qwen2DecoderLayer",
    "Qwen2Model",
    "Qwen2ForCausalLM",
    "VERSION",
]
