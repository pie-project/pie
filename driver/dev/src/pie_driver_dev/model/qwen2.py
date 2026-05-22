"""Qwen2 Large Language Model Architecture."""

from __future__ import annotations

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
# QWEN2 WEIGHT SCHEMA
# =============================================================================
# Declarative definition of how physical tensor names map to logical names,
# with fusion, sharding, and quantization applied.

QWEN2_SCHEMA = (
    Schema("qwen2")
    # Embedding (row-parallel sharding, no quantization)
    .define(
        "embed_token",
        Source("model.embed_tokens.weight").shard("row"),
    )
    # Per-layer weights
    .define(
        "layers.*.norm_attn",
        Source("model.layers.*.input_layernorm.weight"),
    ).define(
        "layers.*.norm_mlp",
        Source("model.layers.*.post_attention_layernorm.weight"),
    )
    # Fused QKV projection weight (interleaved column-parallel, quantized)
    .define(
        "layers.*.proj_qkv.weight",
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
    # Fused QKV projection bias (interleaved column-parallel to match weight)
    .define(
        "layers.*.proj_qkv.bias",
        Source.fuse(
            [
                "model.layers.*.self_attn.q_proj.bias",
                "model.layers.*.self_attn.k_proj.bias",
                "model.layers.*.self_attn.v_proj.bias",
            ],
            dim=0,
        ).shard("interleaved_column"),
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
    """Create weight schema for Qwen2, handling tied/untied embeddings."""
    schema = QWEN2_SCHEMA

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
    Qwen2-specific model architecture configuration.

    Inherits from the abstract ModelConfig base class and defines
    all architecture-specific parameters for Qwen2 models.
    """

    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    num_vocabs: int

    dim_head: int
    dim_hidden: int
    dim_mlp: int

    rms_norm_eps: float
    use_qkv_bias: bool

    rope_theta: float
    tie_word_embeddings: bool

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
            use_qkv_bias=spec.get("attention_bias", spec.get("use_qkv_bias", True)),
            rope_theta=float(spec.get("rope_theta", 10000.0)),
            tie_word_embeddings=bool(spec.get("tie_word_embeddings", True)),
        )

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
        # Each GPU only stores num_kv_heads // tp_size heads
        local_num_kv_heads = self.num_kv_heads // runtime_config.tensor_parallel_size

        total_bytes_per_page = (
            element_size_bytes
            * 2  # key + value
            * runtime_config.kv_page_size
            * local_num_kv_heads  # Use local (sharded) head count
            * self.dim_head
            * self.num_layers
        )

        max_num_pages = int(usable_bytes // total_bytes_per_page)
        return max_num_pages


class ForwardPass(DenseForwardPass):
    """Qwen2 forward pass implementation.

    Uses a 1 GiB attention workspace (vs the 128 MiB default) — overridden
    via ``WORKSPACE_BYTES``.
    """

    WORKSPACE_BYTES = 1024 * 1024 * 1024

    def embed_inputs(self, batch_metadata: dict[str, Any]) -> torch.Tensor:
        """
        Embed input tokens into hidden states.

        Args:
            batch_metadata: Metadata dictionary from the batch builder/packager.

        Returns:
            Tensor of input embeddings.
        """
        device = self.runtime_config.device

        # Extract token IDs from metadata
        token_ids_tensor = torch.as_tensor(
            batch_metadata["token_ids"], device=device, dtype=torch.int32
        )

        return self.embed_tokens(token_ids_tensor)

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute sampling using the model's LM head.

        Args:
            hidden_states: Output hidden states.
            sampling_metadata: Metadata for sampling.

        Returns:
            Sampling results (tokens, distributions).
        """
        # Define a lambda to call self.lm_head passing parameters correctly
        lm_head_fn = lambda x: self.lm_head(x)

        return common.sample_common(
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
            lm_head_fn=lm_head_fn,
            device=self.runtime_config.device,
            dtype=self.runtime_config.activation_dtype,
        )

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs with Tensor Parallel support (Column Parallel).

        The embedding weight is column-sharded: [vocab_size, hidden_size/tp_size]
        Each rank computes partial hidden states, then all_gather combines them.
        """
        if self.tp_size == 1:
            return fun.embedding(token_ids, self.weights.get("embed_token"))

        # Column-parallel embedding: each rank has [vocab_size, hidden_size/tp_size]
        # 1. Lookup - each rank gets partial hidden states [seq_len, hidden_size/tp_size]
        local_embeds = fun.embedding(token_ids, self.weights.get("embed_token"))

        # 2. All-gather to combine partial hidden states from all ranks
        # Output: [seq_len, hidden_size] (full hidden dimension)
        gathered_list = [torch.empty_like(local_embeds) for _ in range(self.tp_size)]
        dist.all_gather(gathered_list, local_embeds)

        # Concatenate along hidden dimension (last dim)
        full_embeds = torch.cat(gathered_list, dim=-1)

        return full_embeds

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits (weight-tied with embed_tokens).

        The embedding weight is column-sharded: [vocab_size, hidden_size/tp_size]
        For lm_head (linear projection), this is effectively [hidden_size/tp_size, vocab_size]
        when transposed.
        """
        # Apply final layer norm
        normed = fun.rms_norm(
            hidden_states,
            normalized_shape=[self.model_config.dim_hidden],
            weight=self.weights.get("norm_last"),
            eps=self.model_config.rms_norm_eps,
        )

        if self.tp_size == 1:
            # Single GPU: simple linear projection
            weight = (
                self.weights.get("embed_token")
                if self.model_config.tie_word_embeddings
                else self.weights.get("lm_head")
            )
            return fun.linear(normed, weight)

        # Multi-GPU: Column-parallel projection
        # 1. Split input along hidden dimension - each rank uses its slice
        hidden_per_rank = self.model_config.dim_hidden // self.tp_size
        start_idx = self.tp_rank * hidden_per_rank
        end_idx = start_idx + hidden_per_rank
        local_normed = normed[:, start_idx:end_idx]

        # 2. Project with local weight shard
        weight = (
            self.weights.get("embed_token")
            if self.model_config.tie_word_embeddings
            else self.weights.get("lm_head")
        )
        local_logits = fun.linear(local_normed, weight)

        # 3. All-reduce to combine partial logits
        dist.all_reduce(local_logits)

        return local_logits

    def mlp(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Executes the MLP block for a single layer, including the
        pre-norm and residual connection.
        """
        # --- Calculate local TP sizes ---
        local_mlp_size = self.model_config.dim_mlp // self.tp_size

        # Save input for residual connection
        residual = hidden_states

        # 1. MLP RMSNorm
        normed_input = fun.rms_norm(
            hidden_states,
            normalized_shape=[self.model_config.dim_hidden],
            weight=self.weights.get(f"layers.{layer_idx}.norm_mlp"),
            eps=self.model_config.rms_norm_eps,
        )

        # 2. Gate+Up Projection (Column Parallel)
        gate_up = fun.linear(
            normed_input,
            self.weights.get(f"layers.{layer_idx}.proj_gate_up"),
        )

        # Split gate and up
        gate, up = torch.split(gate_up, [local_mlp_size, local_mlp_size], dim=-1)

        # 3. SiLU activation * gate (SwiGLU)
        hidden = fun.silu(gate) * up

        # 4. Down Projection (Row Parallel)
        down = fun.linear(
            hidden,
            self.weights.get(f"layers.{layer_idx}.proj_down"),
        )
        del hidden, gate, up, gate_up

        # ALL-REDUCE: Sum partial outputs from all ranks (only if TP > 1)
        if self.tp_size > 1:
            dist.all_reduce(down)

        # 5. Residual Connection
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
        """
        Executes the attention block for a single layer, including the
        pre-norm and residual connection.
        """
        # --- Calculate local TP sizes ---
        local_num_query_heads = (
            self.model_config.num_q_heads // self.tp_size
        )
        local_num_key_value_heads = (
            self.model_config.num_kv_heads // self.tp_size
        )
        local_q_size = local_num_query_heads * self.model_config.dim_head
        local_kv_size = local_num_key_value_heads * self.model_config.dim_head

        n = hidden_states.size(0)

        # Save input for the first residual connection (replicated)
        residual = hidden_states

        # 1. Input RMSNorm (replicated input -> replicated output)
        normed_input = fun.rms_norm(
            hidden_states,
            normalized_shape=[self.model_config.dim_hidden],
            weight=self.weights.get(f"layers.{layer_idx}.norm_attn"),
            eps=self.model_config.rms_norm_eps,
        )

        # 2. QKV Projection (Column Parallel) with bias
        # Input is replicated, weight is sharded -> output is sharded
        qkv_proj = fun.linear(
            normed_input,
            self.weights.get(f"layers.{layer_idx}.proj_qkv.weight"),
            (
                self.weights.get(f"layers.{layer_idx}.proj_qkv.bias")
                if self.model_config.use_qkv_bias
                else None
            ),
        )

        # q, k, v are all LOCAL shards
        q, k, v = torch.split(
            qkv_proj,
            [
                local_q_size,
                local_kv_size,
                local_kv_size,
            ],
            dim=-1,
        )

        # 3. Adapter (if any)
        if adapter_subpass is not None:
            adapter_subpass.execute(
                layer_idx,
                normed_input,  # Adapter needs the (replicated) normed input
                q_state=q,
                k_state=k,
                v_state=v,
            )
        del normed_input

        # 4. Reshape QKV (local shapes) ------------------> input scatter
        q = q.view(n, local_num_query_heads, self.model_config.dim_head)
        k = k.view(n, local_num_key_value_heads, self.model_config.dim_head)
        v = v.view(n, local_num_key_value_heads, self.model_config.dim_head)

        # 5. Apply RoPE (in-place on local shards) - Qwen2 uses standard RoPE
        ops.apply_rope_pos_ids_inplace(
            q=q,
            k=k,
            pos_ids=position_ids,
            rope_theta=self.model_config.rope_theta,
        )

        # 6. Append K, V to cache (local shards to local cache)
        # kv_cache_layer is the LOCAL shard of the cache for this layer
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

        # 7. Compute Attention (on local shards)
        # wrapper was planned with local head counts
        attn_output = wrapper.run(q, kv_cache_layer)
        del q, k, v
        del qkv_proj

        # attn_output is a local shard
        attn_output = attn_output.reshape(n, -1)

        # 8. Output Projection (Row Parallel)
        # Input is sharded, weight is sharded -> output is partial
        attn_proj = fun.linear(
            attn_output,
            self.weights.get(f"layers.{layer_idx}.proj_o"),
        )
        del attn_output

        # ALL-REDUCE: Sum partial outputs from all ranks (only if TP > 1)
        if self.tp_size > 1:
            dist.all_reduce(attn_proj)

        # 9. First Residual Connection
        # residual (replicated) + attn_proj (now replicated)
        return residual + attn_proj

    def transform(
        self,
        # inputs
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        qo_indptr: torch.Tensor,
        # kv cache
        kv_cache_at_layer: list[torch.Tensor],  # Each tensor is a LOCAL shard
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        # mask
        custom_mask: torch.Tensor | None,
        single_token_inference_mode: bool,
        # subpasses
        adapter_subpass: Optional[AdapterSubpass],
    ) -> torch.Tensor:
        """Main transformation pipeline through all layers."""
        torch.cuda.set_device(self.runtime_config.device)

        # Calculate derived inputs
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

        # Wrapper Planning
        local_num_query_heads = (
            self.model_config.num_q_heads // self.tp_size
        )
        local_num_key_value_heads = (
            self.model_config.num_kv_heads // self.tp_size
        )

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
                runtime_config.total_pages + 1,  # +1 for scratch/padding page
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
    """Create adapter cache tensors for all layers.

    Returns a list of (down_weights, up_weights) tuples, one per layer.
    - down_weights: [max_num_adapters, dim_hidden, max_adapter_rank * 3]
    - up_weights: [max_num_adapters, max_adapter_rank, dim_head * (local_num_q_heads + local_num_kv_heads * 2)]
    """
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
