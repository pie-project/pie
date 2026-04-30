"""Mixtral (sparse top-k MoE) LLM architecture.

Targets `mistralai/Mixtral-8x7B-v0.1` and the `mixtral` model_type more
broadly. Llama-style attention with grouped-query KV; the MLP block is
replaced by a sparse Mixture-of-Experts: a router selects top-k experts
per token, runs each expert's SwiGLU MLP, and combines the outputs
weighted by the (renormalized) router scores.

Memory: Mixtral 8x7B is ~47 B params (~94 GB at bf16). To fit on a
single 96 GB GPU with KV cache + activation headroom, set the runtime
`weight_dtype="int8"` (~47 GB resident). The schema marks every
projection `.quantize()`; pie's loader applies the runtime's chosen
torchao config, falling through to bf16 when `weight_dtype="auto"`.

Restrictions in this first pass:
- TP > 1 not supported. Per-expert weights are stacked into 3D tensors
  whose interleaved-column shard semantics aren't wired.
- No CUDA-graph capture (the per-batch expert dispatch loop is dynamic).
- No adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import torch
import torch.nn.functional as fun

from . import ModelConfig as ModelConfigBase
from ..config import RuntimeConfig
from ..adapter import AdapterSubpass
from ..utils import get_available_memory
from ..schema import Schema, Source, WeightStore

import pie_kernels as ops

from . import common


# =============================================================================
# Schema
# =============================================================================


def create_schema(config: "ModelConfig") -> Schema:
    """One quantized weight per expert per projection, kept un-stacked.

    Stacking experts into a single 3D quantized tensor and indexing per token
    triggers torchao's "Cannot set version_counter for inference tensor"
    under `torch.inference_mode()` — torchao mutates a version counter on
    the indexed view that inference-mode forbids. Storing each expert as
    its own AffineQuantizedTensor avoids the indexing path entirely; the
    MoE forward picks weights with `weights.get(f".moe_w1_e{e}")`.
    """
    schema = (
        Schema("mixtral")
        .define("embed_token", Source("model.embed_tokens.weight").shard("row"))
        .define("norm_last", Source("model.norm.weight"))
    )
    if not config.tie_word_embeddings:
        schema = schema.define("lm_head", Source("lm_head.weight").shard("row"))

    for e in range(config.num_experts):
        for logical, expert_proj in [("moe_w1", "w1"), ("moe_w2", "w2"), ("moe_w3", "w3")]:
            schema = schema.define(
                f"layers.*.{logical}_e{e}",
                Source(f"model.layers.*.block_sparse_moe.experts.{e}.{expert_proj}.weight").quantize(),
            )

    schema = (
        schema
        .define(
            "layers.*.moe_gate",
            Source("model.layers.*.block_sparse_moe.gate.weight"),
        )
        .define(
            "layers.*.norm_attn",
            Source("model.layers.*.input_layernorm.weight"),
        )
        .define(
            "layers.*.norm_mlp",
            Source("model.layers.*.post_attention_layernorm.weight"),
        )
        .define(
            "layers.*.proj_q",
            Source("model.layers.*.self_attn.q_proj.weight").quantize(),
        )
        .define(
            "layers.*.proj_k",
            Source("model.layers.*.self_attn.k_proj.weight").quantize(),
        )
        .define(
            "layers.*.proj_v",
            Source("model.layers.*.self_attn.v_proj.weight").quantize(),
        )
        .define(
            "layers.*.proj_o",
            Source("model.layers.*.self_attn.o_proj.weight").quantize(),
        )
    )
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

    # MoE
    num_experts: int
    top_k: int

    @staticmethod
    def from_dict(spec: dict) -> "ModelConfig":
        head_dim = int(
            spec.get("head_dim") or spec["hidden_size"] // spec["num_attention_heads"]
        )
        return ModelConfig(
            num_layers=int(spec["num_hidden_layers"]),
            num_q_heads=int(spec["num_attention_heads"]),
            num_kv_heads=int(spec["num_key_value_heads"]),
            dim_head=head_dim,
            dim_hidden=int(spec["hidden_size"]),
            dim_mlp=int(spec["intermediate_size"]),
            num_vocabs=int(spec["vocab_size"]),
            rms_norm_eps=float(spec["rms_norm_eps"]),
            rope_theta=float(spec.get("rope_theta", 1_000_000.0)),
            tie_word_embeddings=bool(spec.get("tie_word_embeddings", False)),
            num_experts=int(spec["num_local_experts"]),
            top_k=int(spec["num_experts_per_tok"]),
        )

    def eval_max_num_kv_pages(self, runtime_config: RuntimeConfig) -> int:
        available = get_available_memory(devices=runtime_config.devices, rank=runtime_config.rank)
        usable = available * runtime_config.gpu_mem_utilization
        elem = torch.empty((), dtype=runtime_config.activation_dtype).element_size()
        local_kv = self.num_kv_heads // runtime_config.tensor_parallel_size
        per_page = (
            elem * 2 * runtime_config.kv_page_size * local_kv * self.dim_head * self.num_layers
        )
        return int(usable // per_page)


# =============================================================================
# ForwardPass
# =============================================================================


class ForwardPass:
    def __init__(
        self,
        model_config: ModelConfig,
        runtime_config: RuntimeConfig,
        weights: WeightStore,
        compute_process_group: Any = None,
    ):
        if runtime_config.tensor_parallel_size > 1:
            raise NotImplementedError(
                "mixtral: TP>1 not supported (stacked-expert weights need a "
                "split scheme that's not wired in the loader)."
            )
        self.model_config = model_config
        self.runtime_config = runtime_config
        self.weights = weights

        self.workspace_buffer = torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device=runtime_config.device
        )
        self.wrapper_decode = ops.BatchDecodeWithPagedKVCacheWrapper(self.workspace_buffer, "NHD")
        self.wrapper_append = ops.BatchPrefillWithPagedKVCacheWrapper(self.workspace_buffer, "NHD")

    # ------------------------------------------------------------------
    # Embedding / sampling
    # ------------------------------------------------------------------

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return fun.embedding(token_ids, self.weights.get("embed_token"))

    def embed_inputs(self, batch_metadata: dict[str, Any]) -> torch.Tensor:
        ids = torch.as_tensor(
            batch_metadata["token_ids"], device=self.runtime_config.device, dtype=torch.int32
        )
        return self.embed_tokens(ids)

    def sample(self, hidden_states: torch.Tensor, sampling_metadata: dict[str, Any]) -> dict[str, Any]:
        return common.sample_common(
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
            lm_head_fn=lambda x: self.lm_head(x),
            device=self.runtime_config.device,
            dtype=self.runtime_config.activation_dtype,
        )

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

    # ------------------------------------------------------------------
    # MoE block
    # ------------------------------------------------------------------

    def moe(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        cfg = self.model_config
        residual = hidden_states
        normed = fun.rms_norm(
            hidden_states,
            normalized_shape=[cfg.dim_hidden],
            weight=self.weights.get(f"layers.{layer_idx}.norm_mlp"),
            eps=cfg.rms_norm_eps,
        )

        # Router → top-k weights and indices, normalised so each token's
        # k weights sum to 1 (Mixtral re-softmaxes the topped-k subset).
        router_logits = fun.linear(normed, self.weights.get(f"layers.{layer_idx}.moe_gate"))
        router_probs = fun.softmax(router_logits.float(), dim=-1)
        top_k_w, top_k_idx = router_probs.topk(cfg.top_k, dim=-1)
        top_k_w = (top_k_w / top_k_w.sum(dim=-1, keepdim=True)).to(normed.dtype)

        # `expert_mask[e, k, t]` is 1 iff token t routed expert e in slot k.
        expert_mask = fun.one_hot(top_k_idx, num_classes=cfg.num_experts).permute(2, 1, 0)
        expert_hit = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero(as_tuple=False).flatten()

        out = torch.zeros_like(normed)
        for e in expert_hit.tolist():
            slot_idx, token_idx = expert_mask[e].nonzero(as_tuple=True)
            x = normed[token_idx]
            w1 = self.weights.get(f"layers.{layer_idx}.moe_w1_e{e}")
            w2 = self.weights.get(f"layers.{layer_idx}.moe_w2_e{e}")
            w3 = self.weights.get(f"layers.{layer_idx}.moe_w3_e{e}")
            gate = fun.linear(x, w1)
            up = fun.linear(x, w3)
            hidden = fun.silu(gate) * up
            y = fun.linear(hidden, w2)
            y = y * top_k_w[token_idx, slot_idx, None]
            out.index_add_(0, token_idx, y)

        return residual + out

    # ------------------------------------------------------------------
    # Attention
    # ------------------------------------------------------------------

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

        q = fun.linear(normed, self.weights.get(f"layers.{layer_idx}.proj_q"))
        k = fun.linear(normed, self.weights.get(f"layers.{layer_idx}.proj_k"))
        v = fun.linear(normed, self.weights.get(f"layers.{layer_idx}.proj_v"))
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

    # ------------------------------------------------------------------
    # Top-level transform
    # ------------------------------------------------------------------

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

        h = input_embeds
        for layer_idx in range(cfg.num_layers):
            h = self.attention(
                h, layer_idx, position_ids,
                kv_cache_layer=kv_cache_at_layer[layer_idx],
                kv_page_indices=kv_page_indices,
                kv_page_indptr=kv_page_indptr,
                kv_last_page_lens=kv_last_page_lens,
                batch_indices=batch_indices,
                batch_positions=batch_positions,
                wrapper=wrapper,
            )
            h = self.moe(h, layer_idx)
        return h


def create_kv_cache(model_config: ModelConfig, runtime_config: RuntimeConfig) -> list[torch.Tensor]:
    local_kv = model_config.num_kv_heads // runtime_config.tensor_parallel_size
    return [
        torch.zeros(
            (
                runtime_config.max_num_kv_pages + 1,
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
