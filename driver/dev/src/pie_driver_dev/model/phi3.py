"""Phi-3 / Phi-4 dense LLM architecture.

Targets `microsoft/phi-4` and the Phi-3 family (`model_type=phi3`). A
plain Llama-style decoder with two flavour notes:

- Pre-fused `qkv_proj` and `gate_up_proj` in the checkpoint (one
  matrix each, no separate q/k/v or gate/up tensors).
- No QK-norm and no biases on attention; RMSNorm is the standard
  `weight * normalized` (init=ones), matching Llama.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import torch
import torch.nn.functional as fun

from . import ModelConfig as ModelConfigBase
from pie_driver_dev.config import NativeRuntimeConfig as RuntimeConfig
from ..adapter import AdapterSubpass
from pie_driver_dev.utils import get_available_memory
from ..schema import Schema, Source, WeightStore

import pie_kernels as ops

from . import common
from ._base import DenseForwardPass


# =============================================================================
# Schema
# =============================================================================


PHI3_SCHEMA = (
    Schema("phi3")
    .define("embed_token", Source("model.embed_tokens.weight").shard("row"))
    .define("norm_last", Source("model.norm.weight"))
    .define("layers.*.norm_attn", Source("model.layers.*.input_layernorm.weight"))
    .define(
        "layers.*.norm_mlp",
        Source("model.layers.*.post_attention_layernorm.weight"),
    )
    # Pre-fused QKV: shipped as one [(num_q+2*num_kv)*head_dim, hidden] tensor.
    .define(
        "layers.*.proj_qkv",
        Source("model.layers.*.self_attn.qkv_proj.weight"),
    )
    .define(
        "layers.*.proj_o",
        Source("model.layers.*.self_attn.o_proj.weight").shard("row").quantize(),
    )
    # Pre-fused gate+up: shipped as one [2*intermediate, hidden] tensor.
    .define(
        "layers.*.proj_gate_up",
        Source("model.layers.*.mlp.gate_up_proj.weight"),
    )
    .define(
        "layers.*.proj_down",
        Source("model.layers.*.mlp.down_proj.weight").shard("row").quantize(),
    )
)


def create_schema(config: "ModelConfig") -> Schema:
    schema = PHI3_SCHEMA
    if not config.tie_word_embeddings:
        schema = schema.define("lm_head", Source("lm_head.weight").shard("row"))
    return schema


# =============================================================================
# ModelConfig
# =============================================================================


@dataclass
class ModelConfig(ModelConfigBase):
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    num_vocabs: int

    dim_head: int
    dim_hidden: int
    dim_mlp: int

    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool

    @staticmethod
    def from_dict(spec: dict) -> "ModelConfig":
        head_dim = int(spec.get("head_dim") or spec["hidden_size"] // spec["num_attention_heads"])
        return ModelConfig(
            num_layers=int(spec["num_hidden_layers"]),
            num_q_heads=int(spec["num_attention_heads"]),
            num_kv_heads=int(spec["num_key_value_heads"]),
            dim_head=head_dim,
            dim_hidden=int(spec["hidden_size"]),
            dim_mlp=int(spec["intermediate_size"]),
            num_vocabs=int(spec["vocab_size"]),
            rms_norm_eps=float(spec["rms_norm_eps"]),
            rope_theta=float(spec.get("rope_theta", 10000.0)),
            tie_word_embeddings=bool(spec.get("tie_word_embeddings", False)),
        )

    def eval_total_pages(self, runtime_config: RuntimeConfig) -> int:
        available = get_available_memory(devices=runtime_config.devices, rank=runtime_config.rank)
        usable = available * runtime_config.gpu_mem_utilization
        elem = torch.empty((), dtype=runtime_config.activation_dtype).element_size()
        local_kv = self.num_kv_heads // runtime_config.tensor_parallel_size
        per_page = elem * 2 * runtime_config.kv_page_size * local_kv * self.dim_head * self.num_layers
        return int(usable // per_page)


# =============================================================================
# ForwardPass
# =============================================================================


class ForwardPass(DenseForwardPass):
    """Phi3 forward pass implementation.

    No adapter integration; ``embed_tokens`` / ``embed_inputs`` /
    ``sample`` use the base defaults. ``__init__`` rejects TP > 1
    (fused qkv_proj sharding isn't wired for phi3 yet).
    """

    def __init__(
        self,
        model_config: ModelConfig,
        runtime_config: RuntimeConfig,
        weights: WeightStore,
    ):
        if runtime_config.tensor_parallel_size > 1:
            # The pre-fused qkv_proj weight needs an interleaved-by-head
            # column shard to keep Q heads aligned with their matching K/V
            # groups across ranks; the loader has no path for that yet.
            raise NotImplementedError(
                "phi3: TP>1 not supported (fused qkv_proj sharding not wired)"
            )
        super().__init__(model_config, runtime_config, weights)

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = fun.rms_norm(
            hidden_states,
            normalized_shape=[self.model_config.dim_hidden],
            weight=self.weights.get("norm_last"),
            eps=self.model_config.rms_norm_eps,
        )
        weight = (
            self.weights.get("embed_token")
            if self.model_config.tie_word_embeddings
            else self.weights.get("lm_head")
        )
        return fun.linear(normed, weight)

    def mlp(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        cfg = self.model_config
        residual = hidden_states
        normed = fun.rms_norm(
            hidden_states,
            normalized_shape=[cfg.dim_hidden],
            weight=self.weights.get(f"layers.{layer_idx}.norm_mlp"),
            eps=cfg.rms_norm_eps,
        )
        gate_up = fun.linear(normed, self.weights.get(f"layers.{layer_idx}.proj_gate_up"))
        gate, up = torch.split(gate_up, [cfg.dim_mlp, cfg.dim_mlp], dim=-1)
        hidden = fun.silu(gate) * up
        down = fun.linear(hidden, self.weights.get(f"layers.{layer_idx}.proj_down"))
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
        wrapper: Any,
    ) -> torch.Tensor:
        cfg = self.model_config
        n = hidden_states.shape[0]
        residual = hidden_states

        normed = fun.rms_norm(
            hidden_states,
            normalized_shape=[cfg.dim_hidden],
            weight=self.weights.get(f"layers.{layer_idx}.norm_attn"),
            eps=cfg.rms_norm_eps,
        )

        qkv = fun.linear(normed, self.weights.get(f"layers.{layer_idx}.proj_qkv"))
        q_size = cfg.num_q_heads * cfg.dim_head
        kv_size = cfg.num_kv_heads * cfg.dim_head
        q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)
        q = q.view(n, cfg.num_q_heads, cfg.dim_head)
        k = k.view(n, cfg.num_kv_heads, cfg.dim_head)
        v = v.view(n, cfg.num_kv_heads, cfg.dim_head)

        ops.apply_rope_pos_ids_inplace(
            q=q, k=k, pos_ids=position_ids, rope_theta=cfg.rope_theta,
        )

        ops.append_paged_kv_cache(
            append_key=k, append_value=v,
            batch_indices=batch_indices, positions=batch_positions,
            paged_kv_cache=kv_cache_layer,
            kv_indices=kv_page_indices,
            kv_indptr=kv_page_indptr,
            kv_last_page_len=kv_last_page_lens,
            kv_layout="NHD",
        )

        attn_out = wrapper.run(q, kv_cache_layer).reshape(n, -1)
        attn_proj = fun.linear(attn_out, self.weights.get(f"layers.{layer_idx}.proj_o"))
        return residual + attn_proj

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
        """Execute all transformer layers sequentially.

        Phi3 has no adapter integration, so ``adapter_subpass`` is
        ignored (kept in the signature to satisfy the
        :class:`DenseForwardPass` contract).
        """
        del adapter_subpass
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
                wrapper=wrapper,
            )
            hidden_states = self.mlp(hidden_states, layer_idx)
        return hidden_states

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
        if self.runtime_config.device.type == "cuda":
            torch.cuda.set_device(self.runtime_config.device)

        cfg = self.model_config
        n = input_embeds.shape[0]
        page_size = int(kv_cache_at_layer[0].shape[2])
        local_q = cfg.num_q_heads
        local_kv = cfg.num_kv_heads

        seq_lens = ops.get_seq_lens(kv_page_indptr, kv_last_page_lens, page_size)
        batch_indices, batch_positions = ops.get_batch_indices_positions(
            append_indptr=qo_indptr, seq_lens=seq_lens, nnz=n,
        )
        del seq_lens

        decode_supported_groups = {1, 2, 4, 8, 16, 32}
        group_size = local_q // local_kv
        use_decode = single_token_inference_mode and group_size in decode_supported_groups
        if use_decode:
            wrapper = self.wrapper_decode
            wrapper.plan(
                indptr=kv_page_indptr,
                indices=kv_page_indices,
                last_page_len=kv_last_page_lens,
                num_qo_heads=local_q,
                num_kv_heads=local_kv,
                head_dim=cfg.dim_head,
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
                num_qo_heads=local_q,
                num_kv_heads=local_kv,
                head_dim_qk=cfg.dim_head,
                page_size=page_size,
                custom_mask=custom_mask,
                causal=(custom_mask is None),
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


def create_kv_cache(model_config: ModelConfig, runtime_config: RuntimeConfig) -> list[torch.Tensor]:
    local_kv = model_config.num_kv_heads // runtime_config.tensor_parallel_size
    return [
        torch.zeros(
            (
                runtime_config.total_pages + 1,
                2,
                runtime_config.kv_page_size,
                local_kv,
                model_config.dim_head,
            ),
            dtype=runtime_config.activation_dtype,
            device=runtime_config.device,
        )
        for _ in range(model_config.num_layers)
    ]


def create_adapter_cache(model_config: ModelConfig, runtime_config: RuntimeConfig):
    return []
