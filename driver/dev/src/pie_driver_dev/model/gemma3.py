"""Gemma 3 Large Language Model Architecture (Text-Only).

Google's Gemma 3 model with key differences from Gemma 2:
- NO attention/final logit softcapping (removed!)
- Dual RoPE: rope_theta for global layers, rope_local_base_freq for local
- Sliding window pattern: every N-th layer is global (sliding_window_pattern)
- Query scaling via query_pre_attn_scalar (256)
- GELU activation (gelu_pytorch_tanh)
- 4 RMSNorms per layer (same as Gemma 2)

This is the text-only variant for models like google/gemma-3-1b-it.
"""


from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Any

import torch
import torch.nn.functional as fun
import torch.distributed as dist

from . import ModelConfig as ModelConfigBase
from pie_driver_dev.config import NativeRuntimeConfig as RuntimeConfig
from ..adapter import AdapterSubpass
from pie_driver_dev.utils import get_available_memory
from ..schema import Schema, Source, WeightStore

import pie_kernels as ops

from . import common
from ._base import DenseForwardPass


# =============================================================================
# GEMMA3 WEIGHT SCHEMA
# =============================================================================
# Gemma 3 has:
# - 4 RMSNorms per layer (like Gemma 2)
# - Q-Norm and K-Norm for attention (NEW in Gemma 3!)

GEMMA3_SCHEMA = (
    Schema("gemma3")
    # Embedding (row-parallel sharding, no quantization)
    .define(
        "embed_token",
        Source("model.embed_tokens.weight").shard("row"),
    )
    # Per-layer weights - 4 RMSNorms
    .define(
        "layers.*.norm_attn",
        Source("model.layers.*.input_layernorm.weight"),
    )
    .define(
        "layers.*.norm_attn_post",
        Source("model.layers.*.post_attention_layernorm.weight"),
    )
    .define(
        "layers.*.norm_mlp",
        Source("model.layers.*.pre_feedforward_layernorm.weight"),
    )
    .define(
        "layers.*.norm_mlp_post",
        Source("model.layers.*.post_feedforward_layernorm.weight"),
    )
    # QK-Norm: RMSNorm applied to Q and K after projection, before RoPE
    # This is a key difference from Gemma 2!
    .define(
        "layers.*.q_norm",
        Source("model.layers.*.self_attn.q_norm.weight"),
    )
    .define(
        "layers.*.k_norm",
        Source("model.layers.*.self_attn.k_norm.weight"),
    )
    # Fused QKV projection weight (interleaved column-parallel, quantized)
    .define(
        "layers.*.proj_qkv",
        Source.fuse(
            [
                "model.layers.*.self_attn.q_proj.weight",
                "model.layers.*.self_attn.k_proj.weight",
                "model.layers.*.self_attn.v_proj.weight",
            ],
            dim=0,
        )
        .shard("interleaved_column")
        .quantize(),
    )
    # Output projection (row-parallel, quantized)
    .define(
        "layers.*.proj_o",
        Source("model.layers.*.self_attn.o_proj.weight").shard("row").quantize(),
    )
    # Fused gate+up projection (interleaved column-parallel, quantized)
    .define(
        "layers.*.proj_gate_up",
        Source.fuse(
            [
                "model.layers.*.mlp.gate_proj.weight",
                "model.layers.*.mlp.up_proj.weight",
            ],
            dim=0,
        )
        .shard("interleaved_column")
        .quantize(),
    )
    # Down projection (row-parallel, quantized)
    .define(
        "layers.*.proj_down",
        Source("model.layers.*.mlp.down_proj.weight").shard("row").quantize(),
    )
    # Final layer norm
    .define(
        "norm_last",
        Source("model.norm.weight"),
    )
)


def create_schema(config: "ModelConfig") -> Schema:
    """Create weight schema for Gemma3, handling tied/untied embeddings."""
    schema = GEMMA3_SCHEMA

    # Handle untied embeddings (lm_head separate from embed_token)
    if not config.tie_word_embeddings:
        schema = schema.define(
            "lm_head",
            Source("lm_head.weight").shard("row"),
        )

    return schema


@dataclass
class ModelConfig(ModelConfigBase):
    """
    Gemma3-specific model architecture configuration.

    Key differences from Gemma2:
    - No softcapping (attn_logit_softcapping and final_logit_softcapping are None)
    - sliding_window_pattern: every N-th layer is global (vs layer_types list)
    - rope_local_base_freq: RoPE theta for local (sliding window) layers
    """

    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    num_vocabs: int

    dim_head: int
    dim_hidden: int
    dim_mlp: int

    rms_norm_eps: float

    # RoPE configuration
    rope_theta: float  # For global attention layers (1000000)
    rope_local_base_freq: float  # For local sliding window layers (10000)

    tie_word_embeddings: bool

    # Gemma3-specific
    query_pre_attn_scalar: float  # Scaling factor for Q
    sliding_window: int  # Sliding window size (512)
    sliding_window_pattern: int  # Every N-th layer is global (6)
    hidden_activation: str  # Activation function

    @staticmethod
    def from_dict(spec: dict) -> "ModelConfig":
        # Calculate head_dim if not present
        if "head_dim" in spec:
            head_dim = int(spec["head_dim"])
        else:
            head_dim = int(spec["hidden_size"]) // int(spec["num_attention_heads"])

        return ModelConfig(
            num_layers=int(spec["num_hidden_layers"]),
            num_q_heads=int(spec["num_attention_heads"]),
            num_kv_heads=int(spec["num_key_value_heads"]),
            dim_head=head_dim,
            dim_hidden=int(spec["hidden_size"]),
            dim_mlp=int(spec["intermediate_size"]),
            num_vocabs=int(spec["vocab_size"]),
            rms_norm_eps=float(spec["rms_norm_eps"]),
            # Dual RoPE
            rope_theta=float(spec.get("rope_theta", 1000000.0)),
            rope_local_base_freq=float(spec.get("rope_local_base_freq", 10000.0)),
            tie_word_embeddings=bool(spec.get("tie_word_embeddings", True)),
            # Gemma3-specific
            query_pre_attn_scalar=float(spec.get("query_pre_attn_scalar", 256)),
            sliding_window=int(spec.get("sliding_window", 512)),
            sliding_window_pattern=int(spec.get("sliding_window_pattern", 6)),
            hidden_activation=spec.get("hidden_activation", "gelu_pytorch_tanh"),
        )

    def is_global_layer(self, layer_idx: int) -> bool:
        """Check if a layer uses global attention (vs sliding window).
        
        Every sliding_window_pattern-th layer is global (0-indexed).
        For pattern=6: layers 5, 11, 17, 23 are global.
        """
        return (layer_idx + 1) % self.sliding_window_pattern == 0

    def get_rope_theta(self, layer_idx: int) -> float:
        """Get the RoPE theta for a specific layer.
        
        Global layers use rope_theta (1M), local layers use rope_local_base_freq (10K).
        """
        if self.is_global_layer(layer_idx):
            return self.rope_theta
        else:
            return self.rope_local_base_freq

    def eval_total_pages(self, runtime_config: RuntimeConfig) -> int:
        """Evaluate the maximum number of KV pages based on available memory."""
        available_bytes = get_available_memory(
            devices=runtime_config.devices,
            rank=runtime_config.rank,
        )
        usable_bytes = available_bytes * runtime_config.gpu_mem_utilization
        element_size_bytes = torch.empty(
            (), dtype=runtime_config.activation_dtype
        ).element_size()

        # In multi-GPU mode, KV cache is sharded across GPUs
        local_num_kv_heads = self.num_kv_heads // runtime_config.tensor_parallel_size

        total_bytes_per_page = (
            element_size_bytes
            * 2  # key + value
            * runtime_config.kv_page_size
            * local_num_kv_heads
            * self.dim_head
            * self.num_layers
        )

        max_num_pages = int(usable_bytes // total_bytes_per_page)
        return max_num_pages


def _gelu_pytorch_tanh(x: torch.Tensor) -> torch.Tensor:
    """GELU activation with tanh approximation."""
    return fun.gelu(x, approximate="tanh")


class ForwardPass(DenseForwardPass):
    """Gemma3 forward pass implementation.

    Key differences from Gemma2:
    - No softcapping in attention or lm_head
    - Uses dual RoPE (different theta for global vs local layers)
    - Sliding window pattern-based layer type detection

    Adds two precomputed scalars used in attention/embedding
    (``embed_normalizer`` and ``query_scale``).
    """

    WORKSPACE_BYTES = 1024 * 1024 * 1024

    def __init__(
        self,
        model_config: ModelConfig,
        runtime_config: RuntimeConfig,
        weights: WeightStore,
    ):
        super().__init__(model_config, runtime_config, weights)
        # Gemma3-specific scalars (computed once, used in attention/embed).
        self.embed_normalizer = math.sqrt(model_config.dim_hidden)
        self.query_scale = model_config.query_pre_attn_scalar ** -0.5

    def embed_inputs(self, batch_metadata: dict[str, Any]) -> torch.Tensor:
        """Embed input tokens into hidden states."""
        device = self.runtime_config.device
        token_ids_tensor = torch.as_tensor(
            batch_metadata["token_ids"], device=device, dtype=torch.int32
        )
        return self.embed_tokens(token_ids_tensor)

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute sampling using the model's LM head."""
        lm_head_fn = lambda x: self.lm_head(x)

        return common.sample_common(
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
            lm_head_fn=lm_head_fn,
            device=self.runtime_config.device,
            dtype=self.runtime_config.activation_dtype,
        )

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs with Tensor Parallel support.
        
        Gemma3 applies normalization: embeddings *= sqrt(hidden_size)
        """
        if self.tp_size == 1:
            embeddings = fun.embedding(token_ids, self.weights.get("embed_token"))
            return embeddings * self.embed_normalizer

        local_embeds = fun.embedding(token_ids, self.weights.get("embed_token"))
        gathered_list = [torch.empty_like(local_embeds) for _ in range(self.tp_size)]
        dist.all_gather(gathered_list, local_embeds)
        full_embeds = torch.cat(gathered_list, dim=-1)
        return full_embeds * self.embed_normalizer

    def _gemma_rms_norm(
        self, x: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """Gemma RMSNorm: uses (1 + weight) instead of weight."""
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.model_config.rms_norm_eps)
        return x_normed * (1.0 + weight)

    def _qk_norm(
        self, x: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """QK-Norm for Gemma 3: RMSNorm applied to Q/K tensors.
        
        Input shape: [seq_len, num_heads, head_dim]
        Normalizes over head_dim (last dimension).
        Uses (1 + weight) like other Gemma norms.
        """
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.model_config.rms_norm_eps)
        return x_normed * (1.0 + weight)

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits.
        
        Gemma3: NO final logit softcapping (unlike Gemma2).
        """
        normed = self._gemma_rms_norm(
            hidden_states,
            self.weights.get("norm_last"),
        )

        if self.tp_size == 1:
            weight = (
                self.weights.get("embed_token")
                if self.model_config.tie_word_embeddings
                else self.weights.get("lm_head")
            )
            logits = fun.linear(normed, weight)
        else:
            hidden_per_rank = self.model_config.dim_hidden // self.tp_size
            start_idx = self.tp_rank * hidden_per_rank
            end_idx = start_idx + hidden_per_rank
            local_normed = normed[:, start_idx:end_idx]

            weight = (
                self.weights.get("embed_token")
                if self.model_config.tie_word_embeddings
                else self.weights.get("lm_head")
            )
            logits = fun.linear(local_normed, weight)
            dist.all_reduce(logits)

        # Gemma3: NO softcapping (this is a key difference from Gemma2)
        return logits

    def mlp(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Executes the MLP block for a single layer."""
        local_mlp_size = self.model_config.dim_mlp // self.tp_size
        residual = hidden_states

        # 1. Pre-feedforward RMSNorm
        normed_input = self._gemma_rms_norm(
            hidden_states,
            self.weights.get(f"layers.{layer_idx}.norm_mlp"),
        )

        # 2. Gate+Up Projection
        gate_up = fun.linear(
            normed_input,
            self.weights.get(f"layers.{layer_idx}.proj_gate_up"),
        )

        gate, up = torch.split(gate_up, [local_mlp_size, local_mlp_size], dim=-1)

        # 3. GELU activation (GeGLU)
        hidden = _gelu_pytorch_tanh(gate) * up

        # 4. Down Projection
        down = fun.linear(
            hidden,
            self.weights.get(f"layers.{layer_idx}.proj_down"),
        )
        del hidden, gate, up, gate_up

        if self.tp_size > 1:
            dist.all_reduce(down)

        # 5. Post-feedforward RMSNorm
        down = self._gemma_rms_norm(
            down,
            self.weights.get(f"layers.{layer_idx}.norm_mlp_post"),
        )

        # 6. Residual Connection
        return residual + down

    def attention(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        position_ids: torch.Tensor,
        kv_cache_layer: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_positions: torch.Tensor,
        adapter_subpass: Optional[AdapterSubpass],
        wrapper: Any,
    ) -> torch.Tensor:
        """Executes the attention block for a single layer.
        
        Gemma3 attention:
        - Query scaling via query_pre_attn_scalar
        - NO attention logit softcapping (removed in Gemma3!)
        - Dual RoPE: different theta for global vs local layers
        - Post-attention layernorm before residual
        """
        local_num_query_heads = self.model_config.num_q_heads // self.tp_size
        local_num_key_value_heads = self.model_config.num_kv_heads // self.tp_size
        local_q_size = local_num_query_heads * self.model_config.dim_head
        local_kv_size = local_num_key_value_heads * self.model_config.dim_head

        n = hidden_states.size(0)
        residual = hidden_states

        # 1. Input RMSNorm
        normed_input = self._gemma_rms_norm(
            hidden_states,
            self.weights.get(f"layers.{layer_idx}.norm_attn"),
        )

        # 2. QKV Projection
        qkv_proj = fun.linear(
            normed_input,
            self.weights.get(f"layers.{layer_idx}.proj_qkv"),
        )

        q, k, v = torch.split(
            qkv_proj,
            [local_q_size, local_kv_size, local_kv_size],
            dim=-1,
        )

        # 3. Adapter (if any)
        if adapter_subpass is not None:
            adapter_subpass.execute(
                layer_idx,
                normed_input,
                q_state=q,
                k_state=k,
                v_state=v,
            )
        del normed_input

        # 4. Reshape QKV
        q = q.view(n, local_num_query_heads, self.model_config.dim_head)
        k = k.view(n, local_num_key_value_heads, self.model_config.dim_head)
        v = v.view(n, local_num_key_value_heads, self.model_config.dim_head)

        # 5. Apply QK-Norm (Gemma 3 specific - RMSNorm on Q and K before RoPE)
        # This is a critical difference from Gemma 2!
        q = self._qk_norm(q, self.weights.get(f"layers.{layer_idx}.q_norm"))
        k = self._qk_norm(k, self.weights.get(f"layers.{layer_idx}.k_norm"))

        # 6. Apply RoPE with layer-specific theta (dual RoPE)
        rope_theta = self.model_config.get_rope_theta(layer_idx)
        ops.apply_rope_pos_ids_inplace(
            q=q,
            k=k,
            pos_ids=position_ids,
            rope_theta=rope_theta,
        )

        # 6. Append K, V to cache
        ops.append_paged_kv_cache_with_format(
            append_key=k,
            append_value=v,
            batch_indices=batch_indices,
            positions=batch_positions,
            paged_kv_cache=kv_cache_layer,
            kv_indices=kv_page_indices,
            kv_indptr=kv_page_indptr,
            kv_last_page_len=kv_last_page_lens,
            kv_layout="NHD",
        )

        # 7. Compute Attention
        # NOTE: Gemma3 has NO attention softcapping (unlike Gemma2)
        # TODO(LIMITATION): Sliding window attention is not implemented.
        # All layers use full attention regardless of layer type.
        attn_output = wrapper.run(q, kv_cache_layer)
        del q, k, v
        del qkv_proj

        attn_output = attn_output.reshape(n, -1)

        # 8. Output Projection
        attn_proj = fun.linear(
            attn_output,
            self.weights.get(f"layers.{layer_idx}.proj_o"),
        )
        del attn_output

        if self.tp_size > 1:
            dist.all_reduce(attn_proj)

        # 9. Post-attention RMSNorm
        attn_proj = self._gemma_rms_norm(
            attn_proj,
            self.weights.get(f"layers.{layer_idx}.norm_attn_post"),
        )

        # 10. Residual Connection
        return residual + attn_proj

    def transform(
        self,
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_cache_at_layer: list[torch.Tensor],
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        custom_mask: torch.Tensor | None,
        single_token_inference_mode: bool,
        adapter_subpass: Optional[AdapterSubpass],
    ) -> torch.Tensor:
        """Main transformation pipeline through all layers.

        NOTE: Gemma 3 uses interleaved local/global attention (5 local : 1 global).
        This implementation uses full attention for all layers and ignores
        the sliding window pattern. Sliding window would reduce KV cache memory.
        """
        torch.cuda.set_device(self.runtime_config.device)

        page_size = int(kv_cache_at_layer[0].shape[2])
        n = input_embeds.shape[0]

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
        del seq_lens

        local_num_query_heads = self.model_config.num_q_heads // self.tp_size
        local_num_key_value_heads = self.model_config.num_kv_heads // self.tp_size

        if single_token_inference_mode:
            wrapper = self.wrapper_decode
            wrapper.plan(
                indptr=kv_page_indptr,
                indices=kv_page_indices,
                last_page_len=kv_last_page_lens,
                num_qo_heads=local_num_query_heads,
                num_kv_heads=local_num_key_value_heads,
                head_dim=self.model_config.dim_head,
                page_size=page_size,
                pos_encoding_mode="NONE",
                q_data_type=input_embeds.dtype,
            )
        else:
            wrapper = self.wrapper_append
            wrapper.plan(
                qo_indptr=qo_indptr,
                paged_kv_indptr=kv_page_indptr,
                paged_kv_indices=kv_page_indices,
                paged_kv_last_page_len=kv_last_page_lens,
                num_qo_heads=local_num_query_heads,
                num_kv_heads=local_num_key_value_heads,
                head_dim_qk=self.model_config.dim_head,
                page_size=page_size,
                custom_mask=custom_mask,
                q_data_type=input_embeds.dtype,
            )

        return self._run_layers(
            hidden_states=input_embeds,
            position_ids=position_ids,
            kv_cache_at_layer=kv_cache_at_layer,
            kv_page_indices=kv_page_indices,
            kv_page_indptr=kv_page_indptr,
            kv_last_page_lens=kv_last_page_lens,
            batch_indices=batch_indices,
            batch_positions=batch_positions,
            adapter_subpass=adapter_subpass,
            wrapper=wrapper,
        )

    def _run_layers(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache_at_layer: list[torch.Tensor],
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_positions: torch.Tensor,
        adapter_subpass: Optional[AdapterSubpass],
        wrapper: Any,
    ) -> torch.Tensor:
        """Execute all transformer layers sequentially."""
        for layer_idx in range(self.model_config.num_layers):
            hidden_states = self.attention(
                hidden_states=hidden_states,
                layer_idx=layer_idx,
                position_ids=position_ids,
                kv_cache_layer=kv_cache_at_layer[layer_idx],
                kv_page_indices=kv_page_indices,
                kv_page_indptr=kv_page_indptr,
                kv_last_page_lens=kv_last_page_lens,
                batch_indices=batch_indices,
                batch_positions=batch_positions,
                adapter_subpass=adapter_subpass,
                wrapper=wrapper,
            )
            hidden_states = self.mlp(
                hidden_states=hidden_states,
                layer_idx=layer_idx,
            )
        return hidden_states

def create_kv_cache(
    model_config: ModelConfig, runtime_config: RuntimeConfig
) -> list[torch.Tensor]:
    """Create KV cache tensors for all layers."""
    local_num_kv_heads = model_config.num_kv_heads // runtime_config.tensor_parallel_size

    return [
        torch.zeros(
            (
                runtime_config.total_pages + 1,
                2,
                runtime_config.kv_page_size,
                local_num_kv_heads,
                model_config.dim_head,
            ),
            dtype=runtime_config.activation_dtype,
            device=runtime_config.device,
        )
        for _ in range(model_config.num_layers)
    ]


def create_adapter_cache(
    model_config: ModelConfig, runtime_config: RuntimeConfig
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Create adapter cache tensors for all layers."""
    local_num_q_heads = model_config.num_q_heads // runtime_config.tensor_parallel_size
    local_num_kv_heads = model_config.num_kv_heads // runtime_config.tensor_parallel_size

    return [
        (
            torch.zeros(
                (
                    runtime_config.max_num_adapters,
                    model_config.dim_hidden,
                    runtime_config.max_adapter_rank * 3,
                ),
                dtype=runtime_config.activation_dtype,
                device=runtime_config.device,
            ),
            torch.zeros(
                (
                    runtime_config.max_num_adapters,
                    runtime_config.max_adapter_rank,
                    model_config.dim_head
                    * (local_num_q_heads + local_num_kv_heads * 2),
                ),
                dtype=runtime_config.activation_dtype,
                device=runtime_config.device,
            ),
        )
        for _ in range(model_config.num_layers)
    ]
