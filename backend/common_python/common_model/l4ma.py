"""Llama-Like Large Language Model Architecture (L4MA).

This module now focuses on the architecture itself and relies on an injected
runtime backend for kernel-specific behaviour (e.g. FlashInfer or Metal).
"""

from __future__ import annotations
from typing import Sequence, Optional

import torch
from torch import nn

from debug_utils import is_tensor_debug_enabled, checkpoint_validation

# Safe import of adapter functionality
from adapter_import_utils import AdapterSubpass
from config.l4ma import L4maArch
from .l4ma_runtime import L4maBackend, L4maForwardContext, RuntimeInputs

VERSION = "0.1.0"


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

    def forward(self, x):
        """Forward pass through the MLP layer."""
        gate_up_proj_out = self._gate_up_projection(x)
        gate_proj, up_proj = gate_up_proj_out.chunk(2, dim=-1)

        interim = self._silu_activation(gate_proj, up_proj)

        down_proj = self._down_projection(interim)
        return down_proj

    @checkpoint_validation(
        checkpoint_name="l4ma_mlp_forward",
        capture_tensors=True,
        include_metadata=True,
        tolerance=1e-5,
        backend_comparison=None,
        performance_monitoring=True,
    )
    def _gate_up_projection(self, x):
        """Gate/Up projection for Metal GEMM kernel comparison."""
        return self.gate_up_proj(x)

    @checkpoint_validation(
        checkpoint_name="l4ma_mlp_activation",
        capture_tensors=True,
        include_metadata=True,
        tolerance=1e-5,
        backend_comparison=None,
        performance_monitoring=True,
    )
    def _silu_activation(self, gate_proj, up_proj):
        """SiLU activation for Metal activation kernel comparison."""
        return self.act_fn(gate_proj) * up_proj

    @checkpoint_validation(
        checkpoint_name="l4ma_mlp_down_proj",
        capture_tensors=True,
        include_metadata=True,
        tolerance=1e-5,
        backend_comparison=None,
        performance_monitoring=True,
    )
    def _down_projection(self, interim):
        """Down projection for Metal GEMM kernel comparison."""
        return self.down_proj(interim)


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
        runtime: L4maForwardContext,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache_at_layer: Sequence[torch.Tensor],
        adapter_subpass: Optional[AdapterSubpass],
    ) -> torch.Tensor:
        """Attention forward pass that delegates runtime specifics."""

        n, _ = hidden_states.size()

        if hidden_states.numel() and is_tensor_debug_enabled():
            attn_input_min, attn_input_max = hidden_states.aminmax()
            attn_input_nan = torch.isnan(hidden_states).any().item()
            attn_input_inf = torch.isinf(hidden_states).any().item()
            print(
                "[MetalTensorDebug]",
                f"layer={self.layer_idx}",
                "stage=attn_input",
                "dtype=",
                hidden_states.dtype,
                "min=",
                float(attn_input_min),
                "max=",
                float(attn_input_max),
                "has_nan=",
                bool(attn_input_nan),
                "has_inf=",
                bool(attn_input_inf),
            )

        qkv_states = self.qkv_proj(hidden_states)
        if qkv_states.numel() and is_tensor_debug_enabled():
            qkv_min, qkv_max = qkv_states.aminmax()
            qkv_nan = torch.isnan(qkv_states).any().item()
            qkv_inf = torch.isinf(qkv_states).any().item()
            print(
                "[MetalTensorDebug]",
                f"layer={self.layer_idx}",
                "stage=qkv_proj",
                "dtype=",
                qkv_states.dtype,
                "min=",
                float(qkv_min),
                "max=",
                float(qkv_max),
                "has_nan=",
                bool(qkv_nan),
                "has_inf=",
                bool(qkv_inf),
            )

        query_states, key_states, value_states = torch.split(
            qkv_states, [self.q_size, self.k_size, self.v_size], dim=-1
        )

        if query_states.numel() and is_tensor_debug_enabled():
            q_min, q_max = query_states.aminmax()
            q_nan = torch.isnan(query_states).any().item()
            q_inf = torch.isinf(query_states).any().item()
            print(
                "[MetalTensorDebug]",
                f"layer={self.layer_idx}",
                "stage=query_states",
                "dtype=",
                query_states.dtype,
                "min=",
                float(q_min),
                "max=",
                float(q_max),
                "has_nan=",
                bool(q_nan),
                "has_inf=",
                bool(q_inf),
            )

        if key_states.numel() and is_tensor_debug_enabled():
            k_min, k_max = key_states.aminmax()
            k_nan = torch.isnan(key_states).any().item()
            k_inf = torch.isinf(key_states).any().item()
            print(
                "[MetalTensorDebug]",
                f"layer={self.layer_idx}",
                "stage=key_states",
                "dtype=",
                key_states.dtype,
                "min=",
                float(k_min),
                "max=",
                float(k_max),
                "has_nan=",
                bool(k_nan),
                "has_inf=",
                bool(k_inf),
            )

        if value_states.numel() and is_tensor_debug_enabled():
            v_min, v_max = value_states.aminmax()
            v_nan = torch.isnan(value_states).any().item()
            v_inf = torch.isinf(value_states).any().item()
            print(
                "[MetalTensorDebug]",
                f"layer={self.layer_idx}",
                "stage=value_states",
                "dtype=",
                value_states.dtype,
                "min=",
                float(v_min),
                "max=",
                float(v_max),
                "has_nan=",
                bool(v_nan),
                "has_inf=",
                bool(v_inf),
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

        runtime.apply_rope(query_states, key_states, position_ids)

        if query_states.dtype != self.config.dtype:
            query_states = query_states.to(self.config.dtype)

        runtime.append_kv_cache(
            layer_idx=self.layer_idx,
            key_states=key_states,
            value_states=value_states,
            kv_cache_layer=kv_cache_at_layer[self.layer_idx],
        )

        attn_output = runtime.run_attention(
            layer_idx=self.layer_idx,
            query_states=query_states,
            kv_cache_layer=kv_cache_at_layer[self.layer_idx],
        )

        if attn_output.numel() and is_tensor_debug_enabled():
            attn_min, attn_max = attn_output.aminmax()
            attn_nan = torch.isnan(attn_output).any().item()
            attn_inf = torch.isinf(attn_output).any().item()
            print(
                "[MetalTensorDebug]",
                f"layer={self.layer_idx}",
                "stage=attn_output",
                "dtype=",
                attn_output.dtype,
                "min=",
                float(attn_min),
                "max=",
                float(attn_max),
                "has_nan=",
                bool(attn_nan),
                "has_inf=",
                bool(attn_inf),
            )

        attn_output = self.o_proj(attn_output)

        if attn_output.numel() and is_tensor_debug_enabled():
            o_proj_min, o_proj_max = attn_output.aminmax()
            o_proj_nan = torch.isnan(attn_output).any().item()
            o_proj_inf = torch.isinf(attn_output).any().item()
            print(
                "[MetalTensorDebug]",
                f"layer={self.layer_idx}",
                "stage=o_proj",
                "dtype=",
                attn_output.dtype,
                "min=",
                float(o_proj_min),
                "max=",
                float(o_proj_max),
                "has_nan=",
                bool(o_proj_nan),
                "has_inf=",
                bool(o_proj_inf),
            )

        return attn_output


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

    @checkpoint_validation(
        checkpoint_name="l4ma_decoder_layer_forward",
        capture_tensors=True,
        include_metadata=True,
        tolerance=1e-5,
        backend_comparison=None,
        performance_monitoring=True,
    )
    def forward(
        self,
        runtime: L4maForwardContext,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache_at_layer: Sequence[torch.Tensor],
        adapter_subpass: Optional[AdapterSubpass],
    ) -> torch.Tensor:
        """Run the decoder layer using the provided runtime context."""

        residual = hidden_states

        if hidden_states.numel() and is_tensor_debug_enabled():
            pre_norm_min, pre_norm_max = hidden_states.aminmax()
            pre_norm_nan = torch.isnan(hidden_states).any().item()
            pre_norm_inf = torch.isinf(hidden_states).any().item()
            weight = self.input_layernorm.weight
            weight_min, weight_max = tuple(weight.aminmax())
            with torch.no_grad():
                manual_input = hidden_states.to(torch.float32)
                eps = self.input_layernorm.eps or 1e-6
                denom = torch.rsqrt(
                    manual_input.pow(2).mean(dim=-1, keepdim=True) + eps
                )
                manual_norm = manual_input * denom
                manual_norm = manual_norm * weight.to(torch.float32)
                manual_min, manual_max = manual_norm.aminmax()
            print(
                "[MetalTensorDebug]",
                f"layer={self.self_attn.layer_idx}",
                "stage=decoder_input_norm_weight",
                "dtype=",
                weight.dtype,
                "min=",
                float(weight_min),
                "max=",
                float(weight_max),
                "sample=",
                weight.flatten().detach().cpu().numpy().tolist()[:8],
            )
            print(
                "[MetalTensorDebug]",
                f"layer={self.self_attn.layer_idx}",
                "stage=decoder_pre_input_norm",
                "dtype=",
                hidden_states.dtype,
                "min=",
                float(pre_norm_min),
                "max=",
                float(pre_norm_max),
                "has_nan=",
                bool(pre_norm_nan),
                "has_inf=",
                bool(pre_norm_inf),
                "manual_min=",
                float(manual_min),
                "manual_max=",
                float(manual_max),
            )

        hidden_states = self._input_normalization(hidden_states)

        if hidden_states.numel() and is_tensor_debug_enabled():
            post_norm_min, post_norm_max = hidden_states.aminmax()
            post_norm_nan = torch.isnan(hidden_states).any().item()
            post_norm_inf = torch.isinf(hidden_states).any().item()
            print(
                "[MetalTensorDebug]",
                f"layer={self.self_attn.layer_idx}",
                "stage=decoder_post_input_norm",
                "dtype=",
                hidden_states.dtype,
                "min=",
                float(post_norm_min),
                "max=",
                float(post_norm_max),
                "has_nan=",
                bool(post_norm_nan),
                "has_inf=",
                bool(post_norm_inf),
            )

        hidden_states = self.self_attn(
            runtime=runtime,
            hidden_states=hidden_states,
            position_ids=position_ids,
            kv_cache_at_layer=kv_cache_at_layer,
            adapter_subpass=adapter_subpass,
        )

        hidden_states = residual + hidden_states

        if hidden_states.numel() and is_tensor_debug_enabled():
            post_attn_min, post_attn_max = hidden_states.aminmax()
            post_attn_nan = torch.isnan(hidden_states).any().item()
            post_attn_inf = torch.isinf(hidden_states).any().item()
            print(
                "[MetalTensorDebug]",
                f"layer={self.self_attn.layer_idx}",
                "stage=decoder_post_attention_residual",
                "dtype=",
                hidden_states.dtype,
                "min=",
                float(post_attn_min),
                "max=",
                float(post_attn_max),
                "has_nan=",
                bool(post_attn_nan),
                "has_inf=",
                bool(post_attn_inf),
            )

        residual = hidden_states
        hidden_states = self._post_attention_normalization(hidden_states)

        if hidden_states.numel() and is_tensor_debug_enabled():
            post_attn_norm_min, post_attn_norm_max = hidden_states.aminmax()
            post_attn_norm_nan = torch.isnan(hidden_states).any().item()
            post_attn_norm_inf = torch.isinf(hidden_states).any().item()
            weight = self.post_attention_layernorm.weight
            weight_min, weight_max = tuple(weight.aminmax())
            with torch.no_grad():
                manual_input = residual.to(torch.float32)
                eps = self.post_attention_layernorm.eps or 1e-6
                denom = torch.rsqrt(
                    manual_input.pow(2).mean(dim=-1, keepdim=True) + eps
                )
                manual_norm = manual_input * denom
                manual_norm = manual_norm * weight.to(torch.float32)
                manual_min, manual_max = manual_norm.aminmax()
            print(
                "[MetalTensorDebug]",
                f"layer={self.self_attn.layer_idx}",
                "stage=decoder_post_attn_norm_weight",
                "dtype=",
                weight.dtype,
                "min=",
                float(weight_min),
                "max=",
                float(weight_max),
                "sample=",
                weight.flatten().detach().cpu().numpy().tolist()[:8],
            )
            print(
                "[MetalTensorDebug]",
                f"layer={self.self_attn.layer_idx}",
                "stage=decoder_post_attention_norm",
                "dtype=",
                hidden_states.dtype,
                "min=",
                float(post_attn_norm_min),
                "max=",
                float(post_attn_norm_max),
                "has_nan=",
                bool(post_attn_norm_nan),
                "has_inf=",
                bool(post_attn_norm_inf),
                "manual_min=",
                float(manual_min),
                "manual_max=",
                float(manual_max),
            )

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        if hidden_states.numel() and is_tensor_debug_enabled():
            post_mlp_min, post_mlp_max = hidden_states.aminmax()
            post_mlp_nan = torch.isnan(hidden_states).any().item()
            post_mlp_inf = torch.isinf(hidden_states).any().item()
            print(
                "[MetalTensorDebug]",
                f"layer={self.self_attn.layer_idx}",
                "stage=decoder_post_mlp_residual",
                "dtype=",
                hidden_states.dtype,
                "min=",
                float(post_mlp_min),
                "max=",
                float(post_mlp_max),
                "has_nan=",
                bool(post_mlp_nan),
                "has_inf=",
                bool(post_mlp_inf),
            )

        return hidden_states

    @checkpoint_validation(
        checkpoint_name="l4ma_input_norm",
        capture_tensors=True,
        include_metadata=True,
        tolerance=1e-5,
        backend_comparison=None,
        performance_monitoring=True,
    )
    def _input_normalization(self, hidden_states):
        """Input RMSNorm for Metal normalization kernel comparison."""
        return self.input_layernorm(hidden_states)

    @checkpoint_validation(
        checkpoint_name="l4ma_post_attention_norm",
        capture_tensors=True,
        include_metadata=True,
        tolerance=1e-5,
        backend_comparison=None,
        performance_monitoring=True,
    )
    def _post_attention_normalization(self, hidden_states):
        """Post-attention RMSNorm for Metal normalization kernel comparison."""
        return self.post_attention_layernorm(hidden_states)


class L4maModel(nn.Module):
    """Backbone model for the L4MA architecture."""

    def __init__(self, config: L4maArch, backend: L4maBackend):
        super().__init__()
        self.config = config
        self.backend = backend

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

    @checkpoint_validation(
        checkpoint_name="l4ma_model_forward",
        capture_tensors=True,
        include_metadata=True,
        tolerance=1e-5,
        backend_comparison=None,
        performance_monitoring=True,
    )
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
        """Forward pass through all decoder layers using the injected backend."""

        hidden_states = input_embeds
        n, _ = hidden_states.size()

        runtime_inputs = RuntimeInputs(
            num_tokens=n,
            kv_cache_at_layer=kv_cache_at_layer,
            kv_page_indices=kv_page_indices,
            kv_page_indptr=kv_page_indptr,
            kv_last_page_lens=kv_last_page_lens,
            qo_indptr=qo_indptr,
            custom_mask=custom_mask,
            single_token_inference_mode=single_token_inference_mode,
        )

        runtime = self.backend.create_forward_context(
            config=self.config,
            inputs=runtime_inputs,
        )

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                runtime=runtime,
                hidden_states=hidden_states,
                position_ids=position_ids,
                kv_cache_at_layer=kv_cache_at_layer,
                adapter_subpass=adapter_subpass,
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states


class L4maForCausalLM(nn.Module):
    """Top-level causal language model wrapper for L4MA architecture."""

    def __init__(self, config: L4maArch, backend: L4maBackend):
        super().__init__()
        self.config = config
        self.model = L4maModel(config, backend)
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
