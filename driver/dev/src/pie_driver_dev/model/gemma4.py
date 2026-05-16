"""Gemma 4 Large Language Model Architecture (text-only, multimodal-stripped).

Targets E2B/E4B-style checkpoints (e.g. ``google/gemma-4-E2B-it``). The
multimodal towers (vision/audio) are ignored; only the text decoder is loaded.

Notable architectural points (vs. Gemma 3):
- Per-Layer Embeddings (PLE): an auxiliary 256-dim residual signal
  injected after the MLP at every layer.
- KV-cache sharing: the last ``num_kv_shared_layers`` layers reuse K/V
  from the most recent non-shared layer of the same type. Shared layers
  store no K/V of their own and load no k_proj/v_proj weights.
- Per-layer-type head_dim: sliding layers use ``head_dim`` (256), full
  layers use ``global_head_dim`` (512). KV cache shape varies per layer.
- Double-wide MLP for shared layers when ``use_double_wide_mlp`` is set.
- Proportional RoPE on full-attention layers via a precomputed cos/sin
  cache (denominator is ``head_dim``, not ``rotary_dim``).
- Attention softmax uses ``sm_scale=1.0`` — the learnable Q/K-norm
  absorbs the usual ``1/sqrt(head_dim)`` factor. Forgetting this scale
  silently flattens attention to nearly-uniform.
- Embedding/PLE scalars are stored as activation-dtype tensors rather
  than Python floats. Multiplying a bf16 tensor by a Python float
  upcasts via fp64 and produces 1-ULP-per-element drift, which RMSNorm
  amplifies into multi-unit divergence in q_proj.
- RMSNorm: scales by ``weight`` directly (init=ones), not ``(1+weight)``
  like Gemma 2/3. Computed in fp32 for stability.
- V-Norm: pure RMSNorm (no learnable scale) on V before KV cache write.
- Per-layer learnable scalar applied to the layer output.
- Final logit softcapping (cap=30).

Not implemented:
- Sliding-window attention compute is enforced via ``window_left``, but
  the KV cache still allocates pages for the full sequence on sliding
  layers; cache-size split per layer-type is a follow-up for memory.
- ``attention_k_eq_v`` (K reused as V on full layers) is not supported.
- MoE block is not supported (``enable_moe_block=False``).
- Tensor parallelism: Q, gate/up/down, and embed are sharded across TP
  ranks; K/V projections and PLE weights are *replicated* on every rank
  (small num_kv_heads in this family — E2B has 1 — can't be split). TP
  > 1 trades extra K/V/PLE memory for parallelism on Q/MLP/LM-head.
- Full-attention prefill uses an SDPA fallback because FlashInfer's
  default prefill kernel does not support head_dim=512 in 0.6.x.
  Decode uses FlashInfer (head_dim=512 decode is supported). The
  ``trtllm_batch_context_with_kv_cache`` path does support head_dim=512
  in 0.6.9+ and is a natural follow-up.
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


# =============================================================================
# CONFIG
# =============================================================================


def _resolve_text_config(spec: dict) -> dict:
    """Gemma 4 multimodal checkpoints nest the text config; dereference."""
    if "text_config" in spec and isinstance(spec["text_config"], dict):
        return spec["text_config"]
    return spec


def _layer_type_pattern(num_layers: int) -> list[str]:
    return [
        "full_attention" if (i + 1) % 5 == 0 else "sliding_attention"
        for i in range(num_layers)
    ]


@dataclass
class ModelConfig(ModelConfigBase):
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    num_vocabs: int

    dim_head: int           # sliding head_dim
    dim_head_global: int    # full-attention head_dim
    dim_hidden: int
    dim_mlp: int

    rms_norm_eps: float

    rope_theta_full: float
    rope_theta_sliding: float
    rope_partial_factor_full: float
    rope_partial_factor_sliding: float

    tie_word_embeddings: bool

    final_logit_softcapping: float | None
    sliding_window: int
    layer_types: list[str]
    hidden_activation: str

    # PLE
    hidden_size_per_layer_input: int
    vocab_size_per_layer_input: int

    # KV sharing
    num_kv_shared_layers: int

    # Double-wide MLP for shared layers
    use_double_wide_mlp: bool

    # Tensor name prefix in safetensors. Gemma 4 multimodal checkpoints
    # nest the text decoder under ``model.language_model.`` while a hypothetical
    # text-only checkpoint would use ``model.``.
    weight_prefix: str = "model.language_model."

    @staticmethod
    def from_dict(spec: dict) -> "ModelConfig":
        text = _resolve_text_config(spec)

        head_dim = int(text.get("head_dim") or text["hidden_size"] // text["num_attention_heads"])
        head_dim_global = int(text.get("global_head_dim") or head_dim)

        rope_params = text.get("rope_parameters", {}) or {}

        def _rope_for(layer_type: str, default_theta: float) -> tuple[float, float]:
            entry = rope_params.get(layer_type)
            if isinstance(entry, dict):
                theta = float(entry.get("rope_theta", default_theta))
                partial = float(entry.get("partial_rotary_factor", 1.0))
                return theta, partial
            return default_theta, 1.0

        rope_theta_full, partial_full = _rope_for("full_attention", 1_000_000.0)
        rope_theta_sliding, partial_sliding = _rope_for("sliding_attention", 10_000.0)
        if "rope_theta" in text and "full_attention" not in rope_params:
            rope_theta_full = float(text["rope_theta"])

        num_layers = int(text["num_hidden_layers"])
        layer_types = text.get("layer_types") or _layer_type_pattern(num_layers)

        # Multimodal checkpoints have ``model.language_model.X``; text-only
        # have ``model.X``. Detect via top-level model_type.
        is_multimodal = spec.get("model_type", "") in ("gemma4",) and "text_config" in spec
        weight_prefix = "model.language_model." if is_multimodal else "model."

        return ModelConfig(
            num_layers=num_layers,
            num_q_heads=int(text["num_attention_heads"]),
            num_kv_heads=int(text["num_key_value_heads"]),
            dim_head=head_dim,
            dim_head_global=head_dim_global,
            dim_hidden=int(text["hidden_size"]),
            dim_mlp=int(text["intermediate_size"]),
            num_vocabs=int(text["vocab_size"]),
            rms_norm_eps=float(text["rms_norm_eps"]),
            rope_theta_full=rope_theta_full,
            rope_theta_sliding=rope_theta_sliding,
            rope_partial_factor_full=partial_full,
            rope_partial_factor_sliding=partial_sliding,
            tie_word_embeddings=bool(text.get("tie_word_embeddings", spec.get("tie_word_embeddings", True))),
            final_logit_softcapping=text.get("final_logit_softcapping"),
            sliding_window=int(text.get("sliding_window", 512)),
            layer_types=list(layer_types),
            hidden_activation=text.get("hidden_activation", "gelu_pytorch_tanh"),
            hidden_size_per_layer_input=int(text.get("hidden_size_per_layer_input") or 0),
            vocab_size_per_layer_input=int(text.get("vocab_size_per_layer_input") or 0),
            num_kv_shared_layers=int(text.get("num_kv_shared_layers") or 0),
            use_double_wide_mlp=bool(text.get("use_double_wide_mlp", False)),
            weight_prefix=weight_prefix,
        )

    # ---- per-layer queries -------------------------------------------------

    @property
    def first_kv_shared_layer_idx(self) -> int:
        if self.num_kv_shared_layers <= 0:
            return self.num_layers
        return self.num_layers - self.num_kv_shared_layers

    def is_full_attention(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "full_attention"

    def is_kv_shared(self, layer_idx: int) -> bool:
        return layer_idx >= self.first_kv_shared_layer_idx

    def head_dim_at(self, layer_idx: int) -> int:
        return self.dim_head_global if self.is_full_attention(layer_idx) else self.dim_head

    def mlp_dim_at(self, layer_idx: int) -> int:
        if self.use_double_wide_mlp and self.is_kv_shared(layer_idx):
            return self.dim_mlp * 2
        return self.dim_mlp

    def rope_theta_at(self, layer_idx: int) -> float:
        return self.rope_theta_full if self.is_full_attention(layer_idx) else self.rope_theta_sliding

    def rotary_dim_at(self, layer_idx: int) -> int:
        factor = (
            self.rope_partial_factor_full
            if self.is_full_attention(layer_idx)
            else self.rope_partial_factor_sliding
        )
        head_dim = self.head_dim_at(layer_idx)
        rotary_dim = int(head_dim * factor)
        return rotary_dim - (rotary_dim % 2)

    def kv_source_layer(self, layer_idx: int) -> int:
        """The layer whose K/V this layer reuses. Returns layer_idx itself
        if it's not a shared layer."""
        if not self.is_kv_shared(layer_idx):
            return layer_idx
        # Find the most recent non-shared layer with the same layer_type.
        target_type = self.layer_types[layer_idx]
        for i in range(self.first_kv_shared_layer_idx - 1, -1, -1):
            if self.layer_types[i] == target_type:
                return i
        raise ValueError(
            f"No source layer found for shared layer {layer_idx} of type {target_type}"
        )

    def eval_total_pages(self, runtime_config: RuntimeConfig) -> int:
        available_bytes = get_available_memory(
            devices=runtime_config.devices,
            rank=runtime_config.rank,
        )
        usable_bytes = available_bytes * runtime_config.gpu_mem_utilization
        element_size_bytes = torch.empty(
            (), dtype=runtime_config.activation_dtype
        ).element_size()
        # K/V are replicated across TP ranks (not sharded), so each rank holds
        # the full num_kv_heads in its KV cache.

        # Sum across non-shared layers only — shared layers alias their source
        # layer's KV tensor and contribute no extra storage.
        bytes_per_page = 0
        for i in range(self.num_layers):
            if self.is_kv_shared(i):
                continue
            bytes_per_page += (
                element_size_bytes
                * 2
                * runtime_config.kv_page_size
                * self.num_kv_heads
                * self.head_dim_at(i)
            )

        if bytes_per_page == 0:
            return 0
        return int(usable_bytes // bytes_per_page)


# =============================================================================
# WEIGHT SCHEMA
# =============================================================================


def create_schema(config: ModelConfig) -> Schema:
    """Build the per-layer weight schema for a Gemma 4 checkpoint.

    Layer-by-layer enumeration is necessary because:
    - Sliding vs full layers have different head_dim, hence different
      Q/K/V/O shapes.
    - Shared layers carry no k_proj/v_proj/k_norm.
    - Shared layers may have a 2x-wider MLP.
    """
    p = config.weight_prefix
    schema = Schema("gemma4")

    # Top-level
    schema = schema.define("embed_token", Source(f"{p}embed_tokens.weight").shard("row"))
    schema = schema.define("norm_last", Source(f"{p}norm.weight"))

    if not config.tie_word_embeddings:
        schema = schema.define("lm_head", Source("lm_head.weight").shard("row"))

    if config.hidden_size_per_layer_input > 0:
        schema = schema.define(
            "embed_token_per_layer",
            Source(f"{p}embed_tokens_per_layer.weight"),
        )
        schema = schema.define(
            "ple_model_proj",
            Source(f"{p}per_layer_model_projection.weight"),
        )
        schema = schema.define(
            "ple_model_norm",
            Source(f"{p}per_layer_projection_norm.weight"),
        )

    # Per-layer
    for i in range(config.num_layers):
        layer_p = f"{p}layers.{i}."

        # 4 RMSNorms shared by all layer kinds
        schema = schema.define(f"layers.{i}.norm_attn",       Source(f"{layer_p}input_layernorm.weight"))
        schema = schema.define(f"layers.{i}.norm_attn_post",  Source(f"{layer_p}post_attention_layernorm.weight"))
        schema = schema.define(f"layers.{i}.norm_mlp",        Source(f"{layer_p}pre_feedforward_layernorm.weight"))
        schema = schema.define(f"layers.{i}.norm_mlp_post",   Source(f"{layer_p}post_feedforward_layernorm.weight"))
        schema = schema.define(f"layers.{i}.layer_scalar",    Source(f"{layer_p}layer_scalar"))

        # Q always exists (with q_norm)
        schema = schema.define(
            f"layers.{i}.proj_q",
            Source(f"{layer_p}self_attn.q_proj.weight").shard("interleaved_column").quantize(),
        )
        schema = schema.define(
            f"layers.{i}.q_norm",
            Source(f"{layer_p}self_attn.q_norm.weight"),
        )
        schema = schema.define(
            f"layers.{i}.proj_o",
            Source(f"{layer_p}self_attn.o_proj.weight").shard("row").quantize(),
        )

        # K/V exist only for non-shared layers (shared layers reuse cached K/V).
        # K/V projections are NOT sharded across TP ranks: with num_kv_heads=1
        # on E2B (and small num_kv_heads more broadly in this family), the head
        # dimension can't be split across ranks, so each rank holds the full K/V
        # weight and computes the full K/V locally.
        if not config.is_kv_shared(i):
            schema = schema.define(
                f"layers.{i}.proj_k",
                Source(f"{layer_p}self_attn.k_proj.weight").quantize(),
            )
            schema = schema.define(
                f"layers.{i}.proj_v",
                Source(f"{layer_p}self_attn.v_proj.weight").quantize(),
            )
            schema = schema.define(
                f"layers.{i}.k_norm",
                Source(f"{layer_p}self_attn.k_norm.weight"),
            )

        # MLP (intermediate_size doubles for shared layers when use_double_wide_mlp)
        schema = schema.define(
            f"layers.{i}.proj_gate_up",
            Source.fuse(
                [
                    f"{layer_p}mlp.gate_proj.weight",
                    f"{layer_p}mlp.up_proj.weight",
                ],
                dim=0,
            ).shard("interleaved_column").quantize(),
        )
        schema = schema.define(
            f"layers.{i}.proj_down",
            Source(f"{layer_p}mlp.down_proj.weight").shard("row").quantize(),
        )

        # PLE per-layer weights
        if config.hidden_size_per_layer_input > 0:
            schema = schema.define(
                f"layers.{i}.ple_gate",
                Source(f"{layer_p}per_layer_input_gate.weight"),
            )
            schema = schema.define(
                f"layers.{i}.ple_proj",
                Source(f"{layer_p}per_layer_projection.weight"),
            )
            schema = schema.define(
                f"layers.{i}.ple_norm",
                Source(f"{layer_p}post_per_layer_input_norm.weight"),
            )

    return schema


# =============================================================================
# FORWARD PASS
# =============================================================================


def _gelu_pytorch_tanh(x: torch.Tensor) -> torch.Tensor:
    return fun.gelu(x, approximate="tanh")


class ForwardPass:
    def __init__(
        self,
        model_config: ModelConfig,
        runtime_config: RuntimeConfig,
        weights: WeightStore,
    ):
        self.model_config = model_config
        self.runtime_config = runtime_config
        self.weights = weights
        self.tp_size = runtime_config.tensor_parallel_size
        self.tp_rank = runtime_config.rank % self.tp_size
        # Q/MLP/embed are sharded across ranks; K/V are replicated (see schema).
        # With num_kv_heads=1 on E2B/E4B the KV head dim can't be split.
        if model_config.num_q_heads % self.tp_size != 0:
            raise ValueError(
                f"num_q_heads={model_config.num_q_heads} not divisible by "
                f"tp_size={self.tp_size}"
            )
        if model_config.dim_mlp % self.tp_size != 0:
            raise ValueError(
                f"dim_mlp={model_config.dim_mlp} not divisible by tp_size={self.tp_size}"
            )

        # Embedding/PLE scalars must match HF's `Gemma4TextScaledWordEmbedding`,
        # which casts the scale to the activation dtype *before* multiplying.
        # `bf16(sqrt(1536)) = 39.25`, not 39.19 — using a Python float causes
        # PyTorch to upcast to fp64 and changes the bf16 result by 1 ULP per
        # element, which RMSNorm then amplifies into a real divergence.
        adt = runtime_config.activation_dtype
        dev = runtime_config.device
        self.embed_normalizer = torch.tensor(
            math.sqrt(model_config.dim_hidden), dtype=adt, device=dev
        )
        ple_dim = model_config.hidden_size_per_layer_input
        self.has_ple = ple_dim > 0
        self.ple_dim = ple_dim
        if self.has_ple:
            self.ple_token_normalizer = torch.tensor(
                math.sqrt(ple_dim), dtype=adt, device=dev
            )
            self.ple_model_proj_scale = torch.tensor(
                1.0 / math.sqrt(model_config.dim_hidden), dtype=adt, device=dev
            )
            self.ple_combine_scale = torch.tensor(
                1.0 / math.sqrt(2.0), dtype=adt, device=dev
            )
        else:
            self.ple_token_normalizer = None
            self.ple_model_proj_scale = None
            self.ple_combine_scale = None

        # Proportional RoPE cos/sin cache for full-attention layers. The
        # frequency denominator is the *full* head_dim (not rotary_dim), so
        # we precompute cos/sin and use the cos_sin_cache RoPE kernel.
        max_seq = 131072  # max_position_embeddings; resize if exceeded
        self._full_cos_sin_cache = self._build_proportional_rope_cache(
            theta=model_config.rope_theta_full,
            head_dim=model_config.dim_head_global,
            partial_factor=model_config.rope_partial_factor_full,
            max_seq=max_seq,
            device=runtime_config.device,
        )

        # FlashInfer wrappers — one pair (decode/append) per layer-type so
        # head_dim differences (256 vs 512) are honored at plan time.
        self.workspace_buffer = torch.zeros(
            1024 * 1024 * 1024, dtype=torch.uint8, device=runtime_config.device
        )
        self.wrapper_decode_sliding = ops.BatchDecodeWithPagedKVCacheWrapper(self.workspace_buffer, "NHD")
        self.wrapper_decode_full = ops.BatchDecodeWithPagedKVCacheWrapper(self.workspace_buffer, "NHD")
        self.wrapper_append_sliding = ops.BatchPrefillWithPagedKVCacheWrapper(self.workspace_buffer, "NHD")
        self.wrapper_append_full = ops.BatchPrefillWithPagedKVCacheWrapper(self.workspace_buffer, "NHD")

        # Stash for embed→transform handoff (single-threaded inference).
        self._cur_per_layer_inputs: torch.Tensor | None = None

    @staticmethod
    def _build_proportional_rope_cache(
        theta: float, head_dim: int, partial_factor: float,
        max_seq: int, device: torch.device,
    ) -> torch.Tensor:
        """Build a cos/sin cache for Gemma 4 proportional RoPE.

        Pairing convention: HF Gemma 4 uses standard rotate_half on the *full*
        head_dim (so each pair is offset by head_dim/2), but only the first
        ``rope_angles`` of those pairs carry non-trivial frequencies; the rest
        get cos=1 / sin=0 (identity). To make FlashInfer's
        ``apply_rope_with_cos_sin_cache_inplace`` follow the same pairing, we
        build the cache at ``rotary_dim = head_dim`` and pad the upper
        ``head_dim/2 - rope_angles`` cos/sin entries with the identity. The
        frequency denominator is ``head_dim`` (not ``rope_angles*2``), which
        is the "proportional" part.
        """
        half = head_dim // 2
        rope_angles = int(partial_factor * head_dim) // 2  # e.g. 0.25*512//2 = 64
        # inv_freq[k] = 1 / theta^(2k / head_dim) for k in [0, rope_angles)
        ks = torch.arange(0, rope_angles, dtype=torch.float32, device=device)
        inv_freq_rot = 1.0 / (theta ** (2 * ks / head_dim))
        positions = torch.arange(max_seq, dtype=torch.float32, device=device)
        freqs_rot = positions[:, None] * inv_freq_rot[None, :]  # [max_seq, rope_angles]
        cos = torch.ones((max_seq, half), dtype=torch.float32, device=device)
        sin = torch.zeros((max_seq, half), dtype=torch.float32, device=device)
        cos[:, :rope_angles] = freqs_rot.cos()
        sin[:, :rope_angles] = freqs_rot.sin()
        # FlashInfer's expected layout: cos in first half, sin in second half along
        # the last dim — with rotary_dim = head_dim, "half" == head_dim // 2.
        return torch.cat([cos, sin], dim=-1)

    # ------------------------------------------------------------------
    # Embedding + PLE
    # ------------------------------------------------------------------

    def embed_inputs(self, batch_metadata: dict[str, Any]) -> torch.Tensor:
        device = self.runtime_config.device
        token_ids = torch.as_tensor(
            batch_metadata["token_ids"], device=device, dtype=torch.int32
        )
        embeds = self.embed_tokens(token_ids)

        if self.has_ple:
            self._cur_per_layer_inputs = self._compute_per_layer_inputs(token_ids, embeds)
        else:
            self._cur_per_layer_inputs = None

        return embeds

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        embeds = fun.embedding(token_ids, self.weights.get("embed_token"))
        if self.tp_size > 1:
            # embed_token is row-sharded along hidden_size; gather to full.
            gathered = [torch.empty_like(embeds) for _ in range(self.tp_size)]
            dist.all_gather(gathered, embeds)
            embeds = torch.cat(gathered, dim=-1)
        return embeds * self.embed_normalizer

    def _compute_per_layer_inputs(
        self, token_ids: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Computes the per-layer input residual signal.

        Returns shape ``[seq, num_layers, ple_dim]`` (ple_dim = 256 for E2B).
        """
        cfg = self.model_config

        # Token-identity component
        token_long = token_ids.to(torch.long)
        per_layer_token = fun.embedding(token_long, self.weights.get("embed_token_per_layer"))
        per_layer_token = per_layer_token * self.ple_token_normalizer
        per_layer_token = per_layer_token.reshape(
            -1, cfg.num_layers, cfg.hidden_size_per_layer_input
        )

        # Context component
        per_layer_proj = fun.linear(inputs_embeds, self.weights.get("ple_model_proj"))
        per_layer_proj = per_layer_proj * self.ple_model_proj_scale
        per_layer_proj = per_layer_proj.reshape(
            -1, cfg.num_layers, cfg.hidden_size_per_layer_input
        )
        per_layer_proj = self._gemma_rms_norm(
            per_layer_proj, self.weights.get("ple_model_norm")
        )

        return (per_layer_proj + per_layer_token) * self.ple_combine_scale

    # ------------------------------------------------------------------
    # Sampling / LM head
    # ------------------------------------------------------------------

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return common.sample_common(
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
            lm_head_fn=self.lm_head,
            device=self.runtime_config.device,
            dtype=self.runtime_config.activation_dtype,
        )

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self._gemma_rms_norm(hidden_states, self.weights.get("norm_last"))
        weight = (
            self.weights.get("embed_token")
            if self.model_config.tie_word_embeddings
            else self.weights.get("lm_head")
        )
        if self.tp_size == 1:
            logits = fun.linear(normed, weight)
        else:
            # weight is row-sharded along hidden_size; slice the matching
            # rank's portion of the (full) hidden state and all-reduce.
            hidden_per_rank = self.model_config.dim_hidden // self.tp_size
            start = self.tp_rank * hidden_per_rank
            local_normed = normed[..., start:start + hidden_per_rank]
            logits = fun.linear(local_normed, weight)
            dist.all_reduce(logits)
        cap = self.model_config.final_logit_softcapping
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return logits

    # ------------------------------------------------------------------
    # Norms
    # ------------------------------------------------------------------

    def _gemma_rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Gemma 4 stores `weight` already including the +1 baseline (init=ones,
        # not zeros). Multiply directly. Compute in fp32 to match HF.
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        x_normed = x_f32 * torch.rsqrt(variance + self.model_config.rms_norm_eps)
        return (x_normed * weight.float()).to(x.dtype)

    def _qk_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        x_normed = x_f32 * torch.rsqrt(variance + self.model_config.rms_norm_eps)
        return (x_normed * weight.float()).to(x.dtype)

    def _v_norm(self, x: torch.Tensor) -> torch.Tensor:
        # V-Norm has no learnable scale (with_scale=False in HF).
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        return (x_f32 * torch.rsqrt(variance + self.model_config.rms_norm_eps)).to(x.dtype)

    # ------------------------------------------------------------------
    # MLP
    # ------------------------------------------------------------------

    def mlp(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        residual = hidden_states
        local_mlp_size = self.model_config.mlp_dim_at(layer_idx) // self.tp_size

        normed = self._gemma_rms_norm(
            hidden_states, self.weights.get(f"layers.{layer_idx}.norm_mlp")
        )
        gate_up = fun.linear(normed, self.weights.get(f"layers.{layer_idx}.proj_gate_up"))
        gate, up = torch.split(gate_up, [local_mlp_size, local_mlp_size], dim=-1)
        hidden = _gelu_pytorch_tanh(gate) * up
        down = fun.linear(hidden, self.weights.get(f"layers.{layer_idx}.proj_down"))

        del hidden, gate, up, gate_up

        if self.tp_size > 1:
            dist.all_reduce(down)

        down = self._gemma_rms_norm(
            down, self.weights.get(f"layers.{layer_idx}.norm_mlp_post")
        )
        return residual + down

    # ------------------------------------------------------------------
    # PLE per-layer residual
    # ------------------------------------------------------------------

    def _ple_residual(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        per_layer_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the PLE residual block at a given layer.

        ``per_layer_inputs`` has shape ``[seq, num_layers, ple_dim]``;
        we slice off the column for this layer (``[seq, ple_dim]``).
        """
        ple_signal = per_layer_inputs[:, layer_idx, :]  # [seq, ple_dim]

        residual = hidden_states
        gate = fun.linear(hidden_states, self.weights.get(f"layers.{layer_idx}.ple_gate"))
        gate = _gelu_pytorch_tanh(gate)
        gated = gate * ple_signal
        out = fun.linear(gated, self.weights.get(f"layers.{layer_idx}.ple_proj"))
        out = self._gemma_rms_norm(
            out, self.weights.get(f"layers.{layer_idx}.ple_norm")
        )
        return residual + out

    # ------------------------------------------------------------------
    # Attention
    # ------------------------------------------------------------------

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        position_ids: torch.Tensor,
        layer_idx: int,
    ) -> None:
        cfg = self.model_config
        head_dim = cfg.head_dim_at(layer_idx)

        if k is None:
            n = q.size(0)
            k_dummy = torch.empty(
                (n, max(1, cfg.num_kv_heads), head_dim),
                dtype=q.dtype, device=q.device,
            )
            target_k = k_dummy
        else:
            target_k = k

        if cfg.is_full_attention(layer_idx):
            # Proportional RoPE via cos/sin cache. See `_build_proportional_rope_cache`
            # for the layout — pairs are at offset head_dim/2 (HF convention),
            # with identity entries for non-rotated channels.
            n = q.size(0)
            num_q_heads = q.size(1)
            num_kv_heads = target_k.size(1)
            q_flat = q.view(n, num_q_heads * head_dim)
            k_flat = target_k.view(n, num_kv_heads * head_dim)
            ops.apply_rope_with_cos_sin_cache_inplace(
                positions=position_ids,
                query=q_flat,
                key=k_flat,
                head_size=head_dim,
                cos_sin_cache=self._full_cos_sin_cache,
                is_neox=True,
            )
        else:
            # Sliding layers use full RoPE — standard kernel works.
            ops.apply_rope_pos_ids_inplace(
                q=q, k=target_k, pos_ids=position_ids,
                rope_theta=cfg.rope_theta_at(layer_idx),
            )

    def _sdpa_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qo_indptr: torch.Tensor,
    ) -> torch.Tensor:
        """SDPA fallback for full-attention prefill.

        FlashInfer 0.6.x's default prefill kernel doesn't support
        head_dim=512. ``q``/``k``/``v`` are the per-token tensors for the
        current batch (same order as the cache append).
        """
        out = torch.empty_like(q)
        qo_indptr_cpu = qo_indptr.tolist()
        for i in range(len(qo_indptr_cpu) - 1):
            start, end = qo_indptr_cpu[i], qo_indptr_cpu[i + 1]
            if end <= start:
                continue
            q_seq = q[start:end].transpose(0, 1).unsqueeze(0)
            k_seq = k[start:end].transpose(0, 1).unsqueeze(0)
            v_seq = v[start:end].transpose(0, 1).unsqueeze(0)
            attn = fun.scaled_dot_product_attention(
                q_seq, k_seq, v_seq, is_causal=True, enable_gqa=True, scale=1.0,
            )
            out[start:end] = attn.squeeze(0).transpose(0, 1).contiguous()
        return out

    def _sdpa_prefill_against_source(
        self,
        q: torch.Tensor,
        source_k: torch.Tensor,
        source_v: torch.Tensor,
        qo_indptr: torch.Tensor,
    ) -> torch.Tensor:
        """SDPA fallback for shared full-attention prefill. ``source_k``/
        ``source_v`` are the fresh K/V tensors stashed by the most-recent
        non-shared full layer (layer 14 for E2B); they cover the same
        token positions as ``q``.
        """
        out = torch.empty_like(q)
        qo_indptr_cpu = qo_indptr.tolist()
        for i in range(len(qo_indptr_cpu) - 1):
            start, end = qo_indptr_cpu[i], qo_indptr_cpu[i + 1]
            if end <= start:
                continue
            q_seq = q[start:end].transpose(0, 1).unsqueeze(0)
            k_seq = source_k[start:end].transpose(0, 1).unsqueeze(0)
            v_seq = source_v[start:end].transpose(0, 1).unsqueeze(0)
            attn = fun.scaled_dot_product_attention(
                q_seq, k_seq, v_seq, is_causal=True, enable_gqa=True, scale=1.0,
            )
            out[start:end] = attn.squeeze(0).transpose(0, 1).contiguous()
        return out

    def _attention_non_shared(
        self,
        normed_input: torch.Tensor,
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
        single_token_inference_mode: bool,
        qo_indptr: torch.Tensor,
        source_kv_stash: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        cfg = self.model_config
        n = normed_input.size(0)
        num_q_heads_local = cfg.num_q_heads // self.tp_size
        num_kv_heads = cfg.num_kv_heads  # K/V are replicated, not sharded
        head_dim = cfg.head_dim_at(layer_idx)
        is_full = cfg.is_full_attention(layer_idx)

        q = fun.linear(normed_input, self.weights.get(f"layers.{layer_idx}.proj_q"))
        k = fun.linear(normed_input, self.weights.get(f"layers.{layer_idx}.proj_k"))
        v = fun.linear(normed_input, self.weights.get(f"layers.{layer_idx}.proj_v"))

        if adapter_subpass is not None:
            adapter_subpass.execute(layer_idx, normed_input, q_state=q, k_state=k, v_state=v)

        q = q.view(n, num_q_heads_local, head_dim)
        k = k.view(n, num_kv_heads, head_dim)
        v = v.view(n, num_kv_heads, head_dim)

        q = self._qk_norm(q, self.weights.get(f"layers.{layer_idx}.q_norm"))
        k = self._qk_norm(k, self.weights.get(f"layers.{layer_idx}.k_norm"))
        v = self._v_norm(v).contiguous()

        self._apply_rope(q, k, position_ids, layer_idx)
        # Re-contiguify k, v after the in-place RoPE — SDPA needs them.
        k = k.contiguous()
        v = v.contiguous()

        ops.append_paged_kv_cache(
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

        # Stash this layer's fresh K/V so a downstream shared layer of the
        # same type can attend over them during prefill (when SDPA can't
        # read them from the paged cache).
        layer_type = "full_attention" if is_full else "sliding_attention"
        source_kv_stash[layer_type] = (k, v)

        if is_full and not single_token_inference_mode:
            attn_output = self._sdpa_prefill(q, k, v, qo_indptr)
        else:
            attn_output = wrapper.run(q, kv_cache_layer)
        attn_output = attn_output.reshape(n, -1)
        return attn_output

    def _attention_shared(
        self,
        normed_input: torch.Tensor,
        layer_idx: int,
        position_ids: torch.Tensor,
        kv_cache_source: torch.Tensor,
        wrapper: Any,
        single_token_inference_mode: bool,
        qo_indptr: torch.Tensor,
        source_kv_stash: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Q-only path: K/V come from the source layer's KV cache (decode)
        or from the source layer's freshly-computed tensors (full prefill)."""
        cfg = self.model_config
        n = normed_input.size(0)
        num_q_heads_local = cfg.num_q_heads // self.tp_size
        head_dim = cfg.head_dim_at(layer_idx)
        is_full = cfg.is_full_attention(layer_idx)

        q = fun.linear(normed_input, self.weights.get(f"layers.{layer_idx}.proj_q"))
        q = q.view(n, num_q_heads_local, head_dim)
        q = self._qk_norm(q, self.weights.get(f"layers.{layer_idx}.q_norm"))
        self._apply_rope(q, None, position_ids, layer_idx)

        if is_full and not single_token_inference_mode:
            layer_type = "full_attention"
            source_k, source_v = source_kv_stash[layer_type]
            attn_output = self._sdpa_prefill_against_source(q, source_k, source_v, qo_indptr)
        else:
            attn_output = wrapper.run(q, kv_cache_source)
        attn_output = attn_output.reshape(n, -1)
        return attn_output

    def attention(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        position_ids: torch.Tensor,
        kv_cache_at_layer: list[torch.Tensor],
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_positions: torch.Tensor,
        adapter_subpass: Optional[AdapterSubpass],
        wrapper_full: Any,
        wrapper_sliding: Any,
        single_token_inference_mode: bool,
        qo_indptr: torch.Tensor,
        source_kv_stash: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        cfg = self.model_config
        residual = hidden_states
        normed = self._gemma_rms_norm(
            hidden_states, self.weights.get(f"layers.{layer_idx}.norm_attn")
        )

        wrapper = wrapper_full if cfg.is_full_attention(layer_idx) else wrapper_sliding

        if cfg.is_kv_shared(layer_idx):
            source = cfg.kv_source_layer(layer_idx)
            attn_out = self._attention_shared(
                normed_input=normed,
                layer_idx=layer_idx,
                position_ids=position_ids,
                kv_cache_source=kv_cache_at_layer[source],
                wrapper=wrapper,
                single_token_inference_mode=single_token_inference_mode,
                qo_indptr=qo_indptr,
                source_kv_stash=source_kv_stash,
            )
        else:
            attn_out = self._attention_non_shared(
                normed_input=normed,
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
                single_token_inference_mode=single_token_inference_mode,
                qo_indptr=qo_indptr,
                source_kv_stash=source_kv_stash,
            )

        attn_proj = fun.linear(attn_out, self.weights.get(f"layers.{layer_idx}.proj_o"))
        if self.tp_size > 1:
            dist.all_reduce(attn_proj)
        attn_proj = self._gemma_rms_norm(
            attn_proj, self.weights.get(f"layers.{layer_idx}.norm_attn_post")
        )
        return residual + attn_proj

    # ------------------------------------------------------------------
    # Transform pipeline
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
        torch.cuda.set_device(self.runtime_config.device)
        cfg = self.model_config

        n = input_embeds.shape[0]

        # Plan separate wrappers per layer-type because head_dim differs.
        # Pages, indptr, last_page_len are global (per-sequence), not per-layer-type.
        page_size = int(kv_cache_at_layer[0].shape[2])
        seq_lens = ops.get_seq_lens(kv_page_indptr, kv_last_page_lens, page_size)
        batch_indices, batch_positions = ops.get_batch_indices_positions(
            append_indptr=qo_indptr, seq_lens=seq_lens, nnz=n,
        )
        del seq_lens

        # Gemma 4 uses sm_scale=1.0 (no 1/sqrt(d) factor — the learnable
        # Q/K-norm is expected to absorb that scaling). FlashInfer defaults
        # to 1/sqrt(head_dim) which would shrink the softmax inputs by ~16x
        # (sliding) or ~22x (full), producing nearly-uniform attention.
        # Sliding layers also limit attention to the most-recent
        # ``sliding_window`` tokens via window_left; full layers attend
        # over the whole sequence (window_left=-1).
        # Q heads are sharded across TP ranks; K/V are replicated.
        local_num_q_heads = cfg.num_q_heads // self.tp_size
        sliding_window_left = cfg.sliding_window
        if single_token_inference_mode:
            wrapper_sliding = self.wrapper_decode_sliding
            wrapper_full = self.wrapper_decode_full
            for w, head_dim, win in [
                (wrapper_sliding, cfg.dim_head, sliding_window_left),
                (wrapper_full, cfg.dim_head_global, -1),
            ]:
                w.plan(
                    indptr=kv_page_indptr,
                    indices=kv_page_indices,
                    last_page_len=kv_last_page_lens,
                    num_qo_heads=local_num_q_heads,
                    num_kv_heads=cfg.num_kv_heads,
                    head_dim=head_dim,
                    page_size=page_size,
                    pos_encoding_mode="NONE",
                    sm_scale=1.0,
                    window_left=win,
                    q_data_type=input_embeds.dtype,
                )
        else:
            wrapper_sliding = self.wrapper_append_sliding
            # Full-attention prefill uses an SDPA fallback (head_dim=512
            # isn't supported by FlashInfer prefill), so plan only sliding.
            wrapper_full = None
            wrapper_sliding.plan(
                qo_indptr=qo_indptr,
                paged_kv_indptr=kv_page_indptr,
                paged_kv_indices=kv_page_indices,
                paged_kv_last_page_len=kv_last_page_lens,
                num_qo_heads=local_num_q_heads,
                num_kv_heads=cfg.num_kv_heads,
                head_dim_qk=cfg.dim_head,
                page_size=page_size,
                custom_mask=custom_mask,
                causal=(custom_mask is None),
                sm_scale=1.0,
                window_left=sliding_window_left,
                q_data_type=input_embeds.dtype,
            )

        per_layer_inputs = self._cur_per_layer_inputs
        # Per-layer-type stash of fresh K/V tensors. Shared full-attention
        # layers consume these during prefill (SDPA can't pull them from
        # the paged cache without a gather pass).
        source_kv_stash: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        hidden_states = input_embeds
        for layer_idx in range(cfg.num_layers):
            hidden_states = self.attention(
                hidden_states=hidden_states,
                layer_idx=layer_idx,
                position_ids=position_ids,
                kv_cache_at_layer=kv_cache_at_layer,
                kv_page_indices=kv_page_indices,
                kv_page_indptr=kv_page_indptr,
                kv_last_page_lens=kv_last_page_lens,
                batch_indices=batch_indices,
                batch_positions=batch_positions,
                adapter_subpass=adapter_subpass,
                wrapper_full=wrapper_full,
                wrapper_sliding=wrapper_sliding,
                single_token_inference_mode=single_token_inference_mode,
                qo_indptr=qo_indptr,
                source_kv_stash=source_kv_stash,
            )
            hidden_states = self.mlp(hidden_states, layer_idx)
            if self.has_ple and per_layer_inputs is not None:
                hidden_states = self._ple_residual(hidden_states, layer_idx, per_layer_inputs)

            layer_scalar = self.weights.get(f"layers.{layer_idx}.layer_scalar")
            hidden_states = hidden_states * layer_scalar

        return hidden_states


# =============================================================================
# Buffers
# =============================================================================


def create_kv_cache(
    model_config: ModelConfig, runtime_config: RuntimeConfig
) -> list[torch.Tensor]:
    """Allocate one KV cache tensor per layer.

    Shared layers alias the source layer's tensor (same Python object) so
    appends on the source are immediately visible from the shared layer.
    """
    cfg = model_config
    # K/V are replicated across TP ranks; each rank stores the full num_kv_heads.

    kv_cache: list[torch.Tensor | None] = [None] * cfg.num_layers
    for i in range(cfg.num_layers):
        if cfg.is_kv_shared(i):
            continue
        kv_cache[i] = torch.zeros(
            (
                runtime_config.total_pages + 1,
                2,
                runtime_config.kv_page_size,
                cfg.num_kv_heads,
                cfg.head_dim_at(i),
            ),
            dtype=runtime_config.activation_dtype,
            device=runtime_config.device,
        )
    for i in range(cfg.num_layers):
        if cfg.is_kv_shared(i):
            kv_cache[i] = kv_cache[cfg.kv_source_layer(i)]

    return kv_cache  # type: ignore[return-value]


def create_adapter_cache(
    model_config: ModelConfig, runtime_config: RuntimeConfig
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Adapter cache sized for the worst-case (full-attention) head_dim.

    Adapters are layer-uniform in this codebase; using the larger head_dim
    keeps the layout consistent across layers and oversizes (rather than
    undersizes) for sliding layers.
    """
    cfg = model_config
    # Q is sharded across ranks; K/V are replicated.
    local_num_q_heads = cfg.num_q_heads // runtime_config.tensor_parallel_size
    num_kv_heads = cfg.num_kv_heads
    max_head_dim = max(cfg.dim_head, cfg.dim_head_global)

    return [
        (
            torch.zeros(
                (
                    runtime_config.max_num_adapters,
                    cfg.dim_hidden,
                    runtime_config.max_adapter_rank * 3,
                ),
                dtype=runtime_config.activation_dtype,
                device=runtime_config.device,
            ),
            torch.zeros(
                (
                    runtime_config.max_num_adapters,
                    runtime_config.max_adapter_rank,
                    max_head_dim * (local_num_q_heads + num_kv_heads * 2),
                ),
                dtype=runtime_config.activation_dtype,
                device=runtime_config.device,
            ),
        )
        for _ in range(cfg.num_layers)
    ]
