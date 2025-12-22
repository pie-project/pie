"""GPT OSS Large Language Model Architecture"""

from __future__ import annotations

from dataclasses import dataclass
import math


import torch
from torch import nn
import flashinfer as ops
from flashinfer.attention import BatchAttentionWithAttentionSinkWrapper
from adapter_utils import AdapterSubpass
from model.config import CommonArch, ModelConfig
from model.gptoss_utils import (
    pad_to_multiple,
    ALIGNMENT,
    TUNE_MAX_NUM_TOKENS,
)


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
    swiglu_alpha: float
    swiglu_beta: float
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
        swiglu_alpha = cfg.get_required_key(arch_dict, "swiglu_alpha")
        swiglu_beta = cfg.get_required_key(arch_dict, "swiglu_beta")
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
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
        )


def create_fusion_map(model: nn.Module):
    """
    Analyzes the model and creates a map for fusing weights and handling MXFP4 tensors.

    Returns:
        A dictionary mapping {
            fused_tensor_name: {"sources": [source_names], "dim": cat_dim, "op": op_type, ...}
        }.
        For fusion tensors, op is "fusion" and sources contains the tensors to concatenate.
        For MoE tensors, op is "prepare_moe_gate_up" or "prepare_moe_down" which dequantizes,
        preprocesses, and re-quantizes the weights for the FlashInfer kernel.
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

            # Convert sinks from bfloat16 to float32 at load time
            target_sinks = f"{name}.sinks"
            fusion_map[target_sinks] = {
                "sources": [target_sinks],
                "op": "to_float32",
            }

        # --- Rule for GptOssExperts MoE Weight Preparation ---
        elif isinstance(module, GptOssExperts):
            # Common config for MoE preparation
            moe_config = {
                "hidden_size": module.hidden_size,
                "intermediate_size": module.intermediate_size,
                "padded_hidden_size": module.padded_hidden_size,
                "padded_intermediate_size": module.padded_intermediate_size,
                "num_experts": module.num_experts,
            }

            # gate_up: MXFP4 weights + bias -> prepared gemm1 weights, scales, bias
            blocks_gate_up = f"{name}.gate_up_proj_blocks"
            scales_gate_up = f"{name}.gate_up_proj_scales"
            bias_gate_up = f"{name}.gate_up_proj_bias"

            fusion_map[f"{name}.gemm1_weights"] = {
                "sources": [blocks_gate_up, scales_gate_up, bias_gate_up],
                "op": "prepare_gptoss_moe_gate_up",
                "output_type": "weights",
                **moe_config,
            }
            fusion_map[f"{name}.gemm1_scales"] = {
                "sources": [blocks_gate_up, scales_gate_up, bias_gate_up],
                "op": "prepare_gptoss_moe_gate_up",
                "output_type": "scales",
                **moe_config,
            }
            fusion_map[f"{name}.gemm1_bias"] = {
                "sources": [blocks_gate_up, scales_gate_up, bias_gate_up],
                "op": "prepare_gptoss_moe_gate_up",
                "output_type": "bias",
                **moe_config,
            }

            # down: MXFP4 weights + bias -> prepared gemm2 weights, scales, bias
            blocks_down = f"{name}.down_proj_blocks"
            scales_down = f"{name}.down_proj_scales"
            bias_down = f"{name}.down_proj_bias"

            fusion_map[f"{name}.gemm2_weights"] = {
                "sources": [blocks_down, scales_down, bias_down],
                "op": "prepare_gptoss_moe_down",
                "output_type": "weights",
                **moe_config,
            }
            fusion_map[f"{name}.gemm2_scales"] = {
                "sources": [blocks_down, scales_down, bias_down],
                "op": "prepare_gptoss_moe_down",
                "output_type": "scales",
                **moe_config,
            }
            fusion_map[f"{name}.gemm2_bias"] = {
                "sources": [blocks_down, scales_down, bias_down],
                "op": "prepare_gptoss_moe_down",
                "output_type": "bias",
                **moe_config,
            }

    return fusion_map


class GptOssRMSNorm(nn.Module):
    """GPT OSS RMS Normalization layer, which has a scaling parameter."""

    def __init__(self, hidden_size: int, device: str, eps: float = 1e-6):
        """RMS Normalization layer."""

        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=torch.float32)
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """Forward pass through the RMS Normalization layer with scaling parameter."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class GptOssRotaryEmbedding(nn.Module):
    """Rotary Position Embedding with YaRN scaling support."""

    def __init__(
        self,
        head_dim: int,
        base: int,
        dtype: torch.dtype,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | None = None,
        max_position_id: int = 131072,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device
        self.max_position_id = max_position_id

        # Pre-compute concentration and inv_freq since they're constant
        self._concentration, self._inv_freq = self._compute_concentration_and_inv_freq()

        # Pre-compute the full cos/sin cache for all positions up to max_position_id
        self._cos_sin_cache = self._precompute_cos_sin_cache()

    def _compute_concentration_and_inv_freq(self) -> tuple[float, torch.Tensor]:
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device)
            / self.head_dim
        )
        if self.scaling_factor > 1.0:
            concentration = (
                0.1 * math.log(self.scaling_factor) + 1.0
            )  # YaRN concentration

            d_half = self.head_dim / 2
            # NTK by parts
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _precompute_cos_sin_cache(self) -> torch.Tensor:
        """Pre-compute cos/sin cache for all positions up to max_position_id."""
        # Create position indices
        position_ids = torch.arange(
            self.max_position_id, dtype=torch.float32, device=self.device
        )

        # Compute frequencies for all positions
        freqs = torch.einsum("i,j->ij", position_ids, self._inv_freq)

        # Compute cos and sin values with concentration
        cos_cache = freqs.cos() * self._concentration
        sin_cache = freqs.sin() * self._concentration

        # Concatenate cos and sin for FlashInfer format
        # Shape: [max_position_id, head_dim] where head_dim contains
        # [cos_0, cos_1, ..., sin_0, sin_1, ...]
        cos_sin_cache = torch.cat([cos_cache, sin_cache], dim=-1)

        # Ensure float32 precision for numerical accuracy
        return cos_sin_cache.to(torch.float32)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embedding to query and key tensors using position IDs."""
        # Use FlashInfer's optimized RoPE function with pre-computed cache
        ops.apply_rope_with_cos_sin_cache_inplace(
            positions=position_ids.to(torch.int32),
            query=query,
            key=key,
            head_size=self.head_dim,
            cos_sin_cache=self._cos_sin_cache,
            is_neox=True,  # GPT-OSS uses Neox-style RoPE
        )

        return query, key


class GptOssAttention(nn.Module):
    """GPT OSS attention module with attention sink using FlashInfer."""

    def __init__(self, config: GptOssArch, layer_idx: int, rope: GptOssRotaryEmbedding):
        """Initialize the GPT OSS attention module."""
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_size
        self.num_attention_heads = config.num_query_heads
        self.num_key_value_heads = config.num_key_value_heads

        # Define the output sizes for Q, K, and V for clarity
        self.q_size = config.num_query_heads * config.head_size
        self.k_size = config.num_key_value_heads * config.head_size
        self.v_size = config.num_key_value_heads * config.head_size

        # Sink tokens parameter for attention sink
        # Declared as float32 (converted from bfloat16 at load time via fusion_map)
        # FlashInfer fused MoE kernel expects float32 sinks
        self.sinks = nn.Parameter(
            torch.empty(
                config.num_query_heads,
                device=torch.device(config.device),
                dtype=torch.float32,
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

    def forward(
        self,
        wrapper: BatchAttentionWithAttentionSinkWrapper,
        hidden_states: torch.Tensor,
        position_ids: torch.IntTensor,
        kv_cache_at_layer: torch.Tensor,
        kv_page_indices: torch.IntTensor,
        kv_page_indptr: torch.IntTensor,
        kv_last_page_lens: torch.IntTensor,
        batch_indices: torch.IntTensor,
        batch_positions: torch.IntTensor,
        adapter_subpass: AdapterSubpass | None,
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

        # Run attention using FlashInfer with attention sink support
        attn_output = wrapper.run(
            query_states,
            kv_cache_at_layer[self.layer_idx],
            self.sinks,
            self.scaling,
        )
        attn_output = attn_output.reshape(n, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output


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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get raw router logits without top-k selection."""
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = torch.nn.functional.linear(  # pylint: disable=not-callable
            hidden_states, self.weight, self.bias
        )
        return router_logits


class GptOssExperts(nn.Module):
    """GPT OSS Experts layer using FlashInfer fused MoE with MXFP4 weights."""

    def __init__(self, config: GptOssArch):
        """Initialize the GPT OSS Experts layer."""
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.swiglu_alpha = config.swiglu_alpha
        self.swiglu_beta = config.swiglu_beta
        self.swiglu_limit = config.swiglu_limit
        self.experts_per_token = config.experts_per_token

        # Compute padded dimensions for FlashInfer alignment requirements
        self.padded_hidden_size = pad_to_multiple(config.hidden_size, ALIGNMENT)
        self.padded_intermediate_size = pad_to_multiple(
            config.intermediate_size, ALIGNMENT
        )

        device = torch.device(config.device)

        # Pre-processed MXFP4 weights for FlashInfer kernel
        # These are loaded already prepared (quantized, shuffled) by model_loader
        # Using register_buffer since these are inference-only weights
        # (uint8/float8 can't have gradients)
        self.register_buffer(
            "gemm1_weights",
            torch.empty(
                (
                    config.num_experts,
                    self.padded_intermediate_size * 2,
                    self.padded_hidden_size // 2,
                ),
                device=device,
                dtype=torch.uint8,
            ),
        )
        self.register_buffer(
            "gemm1_scales",
            torch.empty(
                (
                    config.num_experts,
                    self.padded_intermediate_size * 2,
                    self.padded_hidden_size // 32,
                ),
                device=device,
                dtype=torch.float8_e4m3fn,
            ),
        )
        self.register_buffer(
            "gemm1_bias",
            torch.empty(
                (config.num_experts, self.padded_intermediate_size * 2),
                device=device,
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "gemm2_weights",
            torch.empty(
                (
                    config.num_experts,
                    self.padded_hidden_size,
                    self.padded_intermediate_size // 2,
                ),
                device=device,
                dtype=torch.uint8,
            ),
        )
        self.register_buffer(
            "gemm2_scales",
            torch.empty(
                (
                    config.num_experts,
                    self.padded_hidden_size,
                    self.padded_intermediate_size // 32,
                ),
                device=device,
                dtype=torch.float8_e4m3fn,
            ),
        )
        self.register_buffer(
            "gemm2_bias",
            torch.empty(
                (config.num_experts, self.padded_hidden_size),
                device=device,
                dtype=torch.float32,
            ),
        )

        # Scale scalars for the kernel
        self._output1_scale_scalar = torch.full((self.num_experts,), 1.0, device=device)
        self._output1_scale_gate_scalar = torch.full(
            (self.num_experts,), 1.0, device=device
        )
        self._output2_scale_scalar = torch.full((self.num_experts,), 1.0, device=device)

        # Activation parameters for GPT OSS style SwiGLU
        self._gemm1_alpha = torch.full(
            (self.num_experts,),
            self.swiglu_alpha,
            device=device,
            dtype=torch.float32,
        )
        self._gemm1_beta = torch.full(
            (self.num_experts,),
            self.swiglu_beta,
            device=device,
            dtype=torch.float32,
        )
        # Clamp limit for activation inputs
        self._gemm1_clamp_limit = torch.full(
            (self.num_experts,),
            self.swiglu_limit,
            device=device,
            dtype=torch.float32,
        )

    def forward(
        self, hidden_states: torch.Tensor, expert_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the experts using FlashInfer fused MoE.

        Args:
            hidden_states: Input tensor of shape [num_tokens, hidden_size]
            expert_logits: Router logits of shape [num_tokens, num_experts]

        Returns:
            Output tensor of shape [num_tokens, hidden_size]
        """
        # Ensure hidden states are bfloat16
        hidden_states_bf16 = hidden_states.to(torch.bfloat16)

        # Pad hidden states if necessary
        if self.hidden_size != self.padded_hidden_size:
            num_tokens = hidden_states_bf16.shape[0]
            padded_hidden = torch.zeros(
                (num_tokens, self.padded_hidden_size),
                dtype=hidden_states_bf16.dtype,
                device=hidden_states_bf16.device,
            )
            padded_hidden[:, : self.hidden_size] = hidden_states_bf16
            hidden_states_bf16 = padded_hidden

        # Call FlashInfer fused MoE kernel with pre-loaded weights
        output = ops.trtllm_fp4_block_scale_moe(
            routing_logits=expert_logits.to(torch.bfloat16),
            routing_bias=None,
            hidden_states=hidden_states_bf16,
            hidden_states_scale=None,
            gemm1_weights=self.gemm1_weights,
            gemm1_weights_scale=self.gemm1_scales,
            gemm1_bias=self.gemm1_bias,
            gemm1_alpha=self._gemm1_alpha,
            gemm1_beta=self._gemm1_beta,
            gemm1_clamp_limit=self._gemm1_clamp_limit,
            gemm2_weights=self.gemm2_weights,
            gemm2_weights_scale=self.gemm2_scales,
            gemm2_bias=self.gemm2_bias,
            output1_scale_scalar=self._output1_scale_scalar,
            output1_scale_gate_scalar=self._output1_scale_gate_scalar,
            output2_scale_scalar=self._output2_scale_scalar,
            num_experts=self.num_experts,
            top_k=self.experts_per_token,
            n_group=None,
            topk_group=None,
            intermediate_size=self.padded_intermediate_size,
            local_expert_offset=0,
            local_num_experts=self.num_experts,
            routed_scaling_factor=None,
            tile_tokens_dim=None,
            routing_method_type=1,  # 1: Renormalize (TopK -> Softmax)
            gated_act_type=0,  # 0: SwiGlu
            do_finalize=True,
            tune_max_num_tokens=TUNE_MAX_NUM_TOKENS,
        )

        output = output[0]

        # Strip padding from output
        if self.hidden_size != self.padded_hidden_size:
            output = output[:, : self.hidden_size]

        return output.to(hidden_states.dtype)


class GptOssMlp(nn.Module):
    """GPT OSS MLP layer with Mixture of Experts using FlashInfer fused MoE."""

    def __init__(self, config: GptOssArch):
        """Initialize the GPT OSS MLP layer."""
        super().__init__()
        self.config = config
        self.router = GptOssRouter(config)
        self.experts = GptOssExperts(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP layer."""
        # Get router logits (the fused MoE kernel handles routing internally)
        router_logits = self.router(x)

        # Forward through experts with fused MoE
        # The FlashInfer kernel handles top-k selection, softmax, and weighted sum
        output = self.experts(x, router_logits)

        return output


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
        wrapper_window: BatchAttentionWithAttentionSinkWrapper,
        wrapper_full: BatchAttentionWithAttentionSinkWrapper,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache_at_layer: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_positions: torch.Tensor,
        adapter_subpass: AdapterSubpass | None,
    ) -> torch.Tensor:
        """Forward pass through the decoder layer."""
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Select wrapper based on layer index:
        # Even layers use sliding window attention, odd layers use full attention
        wrapper = wrapper_window if self.layer_idx % 2 == 0 else wrapper_full

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

        # Create separate workspace buffers for each FlashInfer wrapper
        self.workspace_buffer_window = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=config.device
        )
        self.workspace_buffer_full = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=config.device
        )

        # Create wrapper for even layers (with sliding window attention)
        self.wrapper_window = BatchAttentionWithAttentionSinkWrapper(
            float_workspace_buffer=self.workspace_buffer_window,
            kv_layout="NHD",
            window_left=config.sliding_window - 1,
            q_data_type=config.dtype,
            kv_data_type=config.dtype,
            head_dim_qk=config.head_size,
            head_dim_vo=config.head_size,
        )

        # Create wrapper for odd layers (full attention, no sliding window)
        self.wrapper_full = BatchAttentionWithAttentionSinkWrapper(
            float_workspace_buffer=self.workspace_buffer_full,
            kv_layout="NHD",
            window_left=-1,
            q_data_type=config.dtype,
            kv_data_type=config.dtype,
            head_dim_qk=config.head_size,
            head_dim_vo=config.head_size,
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
        adapter_subpass: AdapterSubpass | None,
    ) -> torch.Tensor:
        """Forward pass through the GPT OSS model."""

        # The current implementation does not distinguish between
        # single-token inference mode and batch inference mode
        _ = single_token_inference_mode
        # custom_mask is not used with attention sink wrapper
        _ = custom_mask

        hidden_states = input_embeds
        n, _ = hidden_states.size()

        page_size = kv_cache_at_layer[0].shape[2]

        seq_lens = ops.get_seq_lens(kv_page_indptr, kv_last_page_lens, page_size)

        batch_indices, batch_positions = ops.get_batch_indices_positions(
            append_indptr=qo_indptr,
            seq_lens=seq_lens,
            nnz=n,
        )

        # Wrapper for windowed attention (even layers)
        self.wrapper_window.plan(
            qo_indptr,
            kv_page_indptr,
            kv_page_indices,
            kv_last_page_lens,
            self.config.num_query_heads,
            self.config.num_key_value_heads,
            self.config.head_size,
            page_size,
            causal=True,
            window_left=self.sliding_window - 1,
            q_data_type=self.config.dtype,
            kv_data_type=self.config.dtype,
            non_blocking=True,
        )

        # Wrapper for full attention (odd layers)
        self.wrapper_full.plan(
            qo_indptr,
            kv_page_indptr,
            kv_page_indices,
            kv_last_page_lens,
            self.config.num_query_heads,
            self.config.num_key_value_heads,
            self.config.head_size,
            page_size,
            causal=True,
            window_left=-1,
            q_data_type=self.config.dtype,
            kv_data_type=self.config.dtype,
            non_blocking=True,
        )

        for decoder_layer in self.layers:

            layer_outputs = decoder_layer(
                wrapper_window=self.wrapper_window,
                wrapper_full=self.wrapper_full,
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
