"""Qwen3.5 / Qwen3.6 (text-only, dense) — hybrid Gated DeltaNet + full-attention.

Single-batch reference implementation. The engine owns paged-KV allocation
for the full-attention layers via flashinfer; linear-attention SSM and conv
state live on the ForwardPass instance and are reset on every prefill.
Multi-request batching, MoE, vision, MTP, TP, and CUDA graphs are out of
scope in this first pass.
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


# =============================================================================
# Weight schema — built per-config because layer-types are heterogeneous.
# =============================================================================

_LM_PREFIX = "model.language_model"


def _define_full_attention_layer(schema: Schema, i: int) -> Schema:
    p = f"{_LM_PREFIX}.layers.{i}"
    return (
        schema
        .define(f"layers.{i}.norm_attn", Source(f"{p}.input_layernorm.weight"))
        .define(f"layers.{i}.norm_mlp", Source(f"{p}.post_attention_layernorm.weight"))
        # q_proj produces [2 * num_q_heads * head_dim] — Q concatenated with gate.
        .define(f"layers.{i}.proj_q_gate", Source(f"{p}.self_attn.q_proj.weight"))
        .define(f"layers.{i}.proj_k", Source(f"{p}.self_attn.k_proj.weight"))
        .define(f"layers.{i}.proj_v", Source(f"{p}.self_attn.v_proj.weight"))
        .define(f"layers.{i}.proj_o", Source(f"{p}.self_attn.o_proj.weight"))
        .define(f"layers.{i}.q_norm", Source(f"{p}.self_attn.q_norm.weight"))
        .define(f"layers.{i}.k_norm", Source(f"{p}.self_attn.k_norm.weight"))
        .define(f"layers.{i}.proj_gate", Source(f"{p}.mlp.gate_proj.weight"))
        .define(f"layers.{i}.proj_up", Source(f"{p}.mlp.up_proj.weight"))
        .define(f"layers.{i}.proj_down", Source(f"{p}.mlp.down_proj.weight"))
    )


def _define_linear_attention_layer(schema: Schema, i: int) -> Schema:
    p = f"{_LM_PREFIX}.layers.{i}"
    return (
        schema
        .define(f"layers.{i}.norm_attn", Source(f"{p}.input_layernorm.weight"))
        .define(f"layers.{i}.norm_mlp", Source(f"{p}.post_attention_layernorm.weight"))
        # GatedDeltaNet projections — separate qkv / b / a / z, NOT fused.
        .define(f"layers.{i}.lin_in_qkv", Source(f"{p}.linear_attn.in_proj_qkv.weight"))
        .define(f"layers.{i}.lin_in_b", Source(f"{p}.linear_attn.in_proj_b.weight"))
        .define(f"layers.{i}.lin_in_a", Source(f"{p}.linear_attn.in_proj_a.weight"))
        .define(f"layers.{i}.lin_in_z", Source(f"{p}.linear_attn.in_proj_z.weight"))
        .define(f"layers.{i}.lin_conv", Source(f"{p}.linear_attn.conv1d.weight"))
        .define(f"layers.{i}.lin_A_log", Source(f"{p}.linear_attn.A_log"))
        .define(f"layers.{i}.lin_dt_bias", Source(f"{p}.linear_attn.dt_bias"))
        .define(f"layers.{i}.lin_norm", Source(f"{p}.linear_attn.norm.weight"))
        .define(f"layers.{i}.lin_out", Source(f"{p}.linear_attn.out_proj.weight"))
        .define(f"layers.{i}.proj_gate", Source(f"{p}.mlp.gate_proj.weight"))
        .define(f"layers.{i}.proj_up", Source(f"{p}.mlp.up_proj.weight"))
        .define(f"layers.{i}.proj_down", Source(f"{p}.mlp.down_proj.weight"))
    )


def create_schema(config: "ModelConfig") -> Schema:
    schema = (
        Schema("qwen3_5")
        .define("embed_token", Source(f"{_LM_PREFIX}.embed_tokens.weight"))
        .define("norm_last", Source(f"{_LM_PREFIX}.norm.weight"))
    )
    if not config.tie_word_embeddings:
        schema = schema.define("lm_head", Source("lm_head.weight"))
    for i in range(config.num_layers):
        if config.is_full_attention(i):
            schema = _define_full_attention_layer(schema, i)
        else:
            schema = _define_linear_attention_layer(schema, i)
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
    partial_rotary_factor: float
    tie_word_embeddings: bool

    # Layer pattern
    layer_types: tuple[str, ...]
    full_attention_interval: int

    # Linear-attention specific
    lin_num_k_heads: int
    lin_num_v_heads: int
    lin_key_head_dim: int
    lin_value_head_dim: int
    lin_conv_kernel_dim: int

    @staticmethod
    def from_dict(spec: dict) -> "ModelConfig":
        text = spec.get("text_config", spec)
        # `head_dim` is required on Qwen3.5; default-derive only as a safety net.
        head_dim = int(text.get("head_dim", text["hidden_size"] // text["num_attention_heads"]))
        rope_params = text.get("rope_parameters", {})
        rope_theta = float(rope_params.get("rope_theta", text.get("rope_theta", 10000.0)))
        partial = float(
            rope_params.get(
                "partial_rotary_factor",
                text.get("partial_rotary_factor", 1.0),
            )
        )
        return ModelConfig(
            num_layers=int(text["num_hidden_layers"]),
            num_q_heads=int(text["num_attention_heads"]),
            num_kv_heads=int(text["num_key_value_heads"]),
            dim_head=head_dim,
            dim_hidden=int(text["hidden_size"]),
            dim_mlp=int(text["intermediate_size"]),
            num_vocabs=int(text["vocab_size"]),
            rms_norm_eps=float(text["rms_norm_eps"]),
            rope_theta=rope_theta,
            partial_rotary_factor=partial,
            tie_word_embeddings=bool(text.get("tie_word_embeddings", spec.get("tie_word_embeddings", False))),
            layer_types=tuple(text.get("layer_types", ())),
            full_attention_interval=int(text.get("full_attention_interval", 4)),
            lin_num_k_heads=int(text["linear_num_key_heads"]),
            lin_num_v_heads=int(text["linear_num_value_heads"]),
            lin_key_head_dim=int(text["linear_key_head_dim"]),
            lin_value_head_dim=int(text["linear_value_head_dim"]),
            lin_conv_kernel_dim=int(text.get("linear_conv_kernel_dim", 4)),
        )

    def is_full_attention(self, layer_idx: int) -> bool:
        if self.layer_types:
            return self.layer_types[layer_idx] == "full_attention"
        return ((layer_idx + 1) % self.full_attention_interval) == 0

    def num_full_attention_layers(self) -> int:
        return sum(1 for i in range(self.num_layers) if self.is_full_attention(i))

    def eval_total_pages(self, runtime_config: RuntimeConfig) -> int:
        """Sized for full-attn layers only; linear-attn layers carry tiny placeholder caches."""
        avail = get_available_memory(devices=runtime_config.devices, rank=runtime_config.rank)
        # Reserve ~10% of the budget for SSM state and conv state.
        usable = avail * runtime_config.gpu_mem_utilization * 0.9
        elem = torch.empty((), dtype=runtime_config.activation_dtype).element_size()
        local_kv_heads = self.num_kv_heads // runtime_config.tensor_parallel_size
        n_full = max(1, self.num_full_attention_layers())
        per_page = elem * 2 * runtime_config.kv_page_size * local_kv_heads * self.dim_head * n_full
        return int(usable // per_page)


# =============================================================================
# Helpers
# =============================================================================


def _l2_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _qwen35_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3.5 RMSNorm — `(1 + weight) * normalized` (Gemma-style).

    Distinct from Qwen3 / Llama which uses `weight * normalized`. The stored
    weight tensor is initialised to zeros, so adding 1 recovers the identity
    scale at init.
    """
    input_dtype = x.dtype
    x32 = x.float()
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    x32 = x32 * torch.rsqrt(var + eps)
    x32 = x32 * (1.0 + weight.float())
    return x32.to(input_dtype)


def _torch_chunk_gated_delta_rule(
    query, key, value, g, beta, chunk_size=64, initial_state=None,
    output_final_state=True, use_qk_l2norm_in_kernel=True,
):
    """Verbatim port of `transformers.qwen3_next.torch_chunk_gated_delta_rule`."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2_norm(query, dim=-1)
        key = _l2_norm(key, dim=-1)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = fun.pad(query, (0, 0, 0, pad_size))
    key = fun.pad(key, (0, 0, 0, pad_size))
    value = fun.pad(value, (0, 0, 0, pad_size))
    beta = fun.pad(beta, (0, pad_size))
    g = fun.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_chunk = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_chunk @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def _torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state=True,
    use_qk_l2norm_in_kernel=True,
):
    """Verbatim port of `transformers.qwen3_next.torch_recurrent_gated_delta_rule`."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2_norm(query, dim=-1)
        key = _l2_norm(key, dim=-1)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size, num_heads, sequence_length, v_head_dim,
        dtype=value.dtype, device=value.device,
    )
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim,
                    dtype=value.dtype, device=value.device)
        if initial_state is None else initial_state.to(value)
    )
    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# =============================================================================
# ForwardPass
# =============================================================================


class ForwardPass:
    """Single-batch hybrid forward pass for Qwen3.5/3.6 dense text models.

    Notes / restrictions:
    - tensor_parallel_size MUST equal 1.
    - Only one in-flight request is supported. SSM state is reset on every
      prefill (single_token_inference_mode=False).
    - Vision tokens are not handled; pass text-only token ids.
    - MTP head is not loaded.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        runtime_config: RuntimeConfig,
        weights: WeightStore,
    ):
        if runtime_config.tensor_parallel_size > 1:
            raise NotImplementedError(
                "qwen3_5 native: TP>1 not supported (linear-attention SSM "
                "state would need head-axis sharding plus per-rank reset)."
            )
        self.model_config = model_config
        self.runtime_config = runtime_config
        self.weights = weights

        self.workspace_buffer = torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device=runtime_config.device
        )
        self.wrapper_decode = ops.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )
        self.wrapper_append = ops.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )

        # model layer idx → index into kv_cache_at_layer (full-attn layers only)
        self.full_attn_layer_lookup: dict[int, int] = {}
        full_idx = 0
        for i in range(model_config.num_layers):
            if model_config.is_full_attention(i):
                self.full_attn_layer_lookup[i] = full_idx
                full_idx += 1

        self._ssm_state: dict[int, torch.Tensor] = {}    # layer_idx → [1, h, dk, dv]
        self._conv_state: dict[int, torch.Tensor] = {}   # layer_idx → [1, conv_dim, k-1]

    # ------------------------------------------------------------------
    # Embedding / sampling
    # ------------------------------------------------------------------

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return fun.embedding(token_ids, self.weights.get("embed_token"))

    def embed_inputs(self, batch_metadata: dict[str, Any]) -> torch.Tensor:
        device = self.runtime_config.device
        token_ids = torch.as_tensor(batch_metadata["token_ids"], device=device, dtype=torch.int32)
        return self.embed_tokens(token_ids)

    def sample(self, hidden_states: torch.Tensor, sampling_metadata: dict[str, Any]) -> dict[str, Any]:
        return common.sample_common(
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
            lm_head_fn=lambda x: self.lm_head(x),
            device=self.runtime_config.device,
            dtype=self.runtime_config.activation_dtype,
        )

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = _qwen35_rms_norm(
            hidden_states,
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
    # MLP (SwiGLU, shared across layer types)
    # ------------------------------------------------------------------

    def mlp(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        residual = hidden_states
        normed = _qwen35_rms_norm(
            hidden_states,
            weight=self.weights.get(f"layers.{layer_idx}.norm_mlp"),
            eps=self.model_config.rms_norm_eps,
        )
        gate = fun.linear(normed, self.weights.get(f"layers.{layer_idx}.proj_gate"))
        up = fun.linear(normed, self.weights.get(f"layers.{layer_idx}.proj_up"))
        hidden = fun.silu(gate) * up
        down = fun.linear(hidden, self.weights.get(f"layers.{layer_idx}.proj_down"))
        return residual + down

    # ------------------------------------------------------------------
    # Full attention with QK norm + partial RoPE + output gating
    # ------------------------------------------------------------------

    def attention_full(
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

        normed = _qwen35_rms_norm(
            hidden_states,
            weight=self.weights.get(f"layers.{layer_idx}.norm_attn"),
            eps=cfg.rms_norm_eps,
        )

        # q_proj fans out to 2× num_q_heads × head_dim — first half is Q,
        # second half is the per-element output gate (sigmoid'd post-attn).
        q_gate = fun.linear(normed, self.weights.get(f"layers.{layer_idx}.proj_q_gate"))
        q_gate = q_gate.view(n, cfg.num_q_heads, 2 * cfg.dim_head)
        q, gate = q_gate.split([cfg.dim_head, cfg.dim_head], dim=-1)
        gate = gate.reshape(n, cfg.num_q_heads * cfg.dim_head)

        k = fun.linear(normed, self.weights.get(f"layers.{layer_idx}.proj_k"))
        v = fun.linear(normed, self.weights.get(f"layers.{layer_idx}.proj_v"))
        k = k.view(n, cfg.num_kv_heads, cfg.dim_head)
        v = v.view(n, cfg.num_kv_heads, cfg.dim_head)

        q = _qwen35_rms_norm(q, weight=self.weights.get(f"layers.{layer_idx}.q_norm"), eps=cfg.rms_norm_eps)
        k = _qwen35_rms_norm(k, weight=self.weights.get(f"layers.{layer_idx}.k_norm"), eps=cfg.rms_norm_eps)

        # Partial RoPE: only the first `partial_rotary_factor * head_dim`
        # dims are rotated; the tail is left unchanged.
        rotary_dim = int(cfg.dim_head * cfg.partial_rotary_factor)
        ops.apply_rope_pos_ids_inplace(
            q=q,
            k=k,
            pos_ids=position_ids,
            rotary_dim=rotary_dim,
            interleave=False,
            rope_theta=cfg.rope_theta,
        )

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

        attn_out = wrapper.run(q, kv_cache_layer)
        attn_out = attn_out.reshape(n, -1) * torch.sigmoid(gate)
        attn_proj = fun.linear(attn_out, self.weights.get(f"layers.{layer_idx}.proj_o"))
        return residual + attn_proj

    # ------------------------------------------------------------------
    # Linear attention (Gated DeltaNet)
    # ------------------------------------------------------------------

    def attention_linear(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        is_prefill: bool,
    ) -> torch.Tensor:
        cfg = self.model_config
        n = hidden_states.shape[0]
        residual = hidden_states

        normed = _qwen35_rms_norm(
            hidden_states,
            weight=self.weights.get(f"layers.{layer_idx}.norm_attn"),
            eps=cfg.rms_norm_eps,
        )

        # HF's reference fuses qkvz into one projection and ba into another;
        # Qwen3.5 ships these as four separate matrices.
        key_dim = cfg.lin_num_k_heads * cfg.lin_key_head_dim
        value_dim = cfg.lin_num_v_heads * cfg.lin_value_head_dim
        head_v_per_k = cfg.lin_num_v_heads // cfg.lin_num_k_heads
        conv_dim = 2 * key_dim + value_dim

        qkv = fun.linear(normed, self.weights.get(f"layers.{layer_idx}.lin_in_qkv"))
        z = fun.linear(normed, self.weights.get(f"layers.{layer_idx}.lin_in_z"))
        b_proj = fun.linear(normed, self.weights.get(f"layers.{layer_idx}.lin_in_b"))
        a_proj = fun.linear(normed, self.weights.get(f"layers.{layer_idx}.lin_in_a"))

        # Causal depthwise conv1d. We carry (kernel-1) tokens of pre-conv
        # input across decode steps in `_conv_state` so the kernel sees the
        # right left-context — equivalent to nn.Conv1d's padding=k-1 on
        # the first prefill, but supports cross-step continuation.
        x_conv_in = qkv.unsqueeze(0).transpose(1, 2)   # [1, conv_dim, n]
        conv_state = self._conv_state.get(layer_idx)
        if conv_state is None or is_prefill:
            conv_state = torch.zeros(
                1, conv_dim, cfg.lin_conv_kernel_dim - 1,
                dtype=qkv.dtype, device=qkv.device,
            )
        x_with_ctx = torch.cat([conv_state, x_conv_in], dim=-1)
        x_post = fun.conv1d(
            x_with_ctx,
            self.weights.get(f"layers.{layer_idx}.lin_conv"),
            bias=None, padding=0, groups=conv_dim,
        )
        x_post = fun.silu(x_post[:, :, -n:])
        self._conv_state[layer_idx] = x_with_ctx[:, :, -(cfg.lin_conv_kernel_dim - 1):].clone()

        x_post = x_post.transpose(1, 2).reshape(n, conv_dim)
        q_l, k_l, v_l = x_post.split([key_dim, key_dim, value_dim], dim=-1)
        q_l = q_l.view(1, n, cfg.lin_num_k_heads, cfg.lin_key_head_dim)
        k_l = k_l.view(1, n, cfg.lin_num_k_heads, cfg.lin_key_head_dim)
        v_l = v_l.view(1, n, cfg.lin_num_v_heads, cfg.lin_value_head_dim)

        beta = b_proj.view(1, n, cfg.lin_num_v_heads).sigmoid()
        A_log = self.weights.get(f"layers.{layer_idx}.lin_A_log")
        dt_bias = self.weights.get(f"layers.{layer_idx}.lin_dt_bias")
        a_h = a_proj.view(1, n, cfg.lin_num_v_heads).float()
        g = -A_log.float().exp() * fun.softplus(a_h + dt_bias.float())

        if head_v_per_k > 1:
            q_l = q_l.repeat_interleave(head_v_per_k, dim=2)
            k_l = k_l.repeat_interleave(head_v_per_k, dim=2)

        initial_state = None if is_prefill else self._ssm_state.get(layer_idx)
        # Chunked path for prefill (parallel scan), recurrent for single-token decode.
        rule = _torch_chunk_gated_delta_rule if n > 1 else _torch_recurrent_gated_delta_rule
        core_out, new_state = rule(
            q_l, k_l, v_l, g, beta,
            initial_state=initial_state, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        self._ssm_state[layer_idx] = new_state.detach()

        # RMSNormGated (uses standard `weight * normalized`, not `(1+w) *`):
        # norm per value-head, then multiply by silu(z) per element.
        core_2d = core_out.reshape(-1, cfg.lin_value_head_dim)
        normed_core = fun.rms_norm(
            core_2d,
            normalized_shape=[cfg.lin_value_head_dim],
            weight=self.weights.get(f"layers.{layer_idx}.lin_norm"),
            eps=cfg.rms_norm_eps,
        )
        normed_core = normed_core.reshape(1, n, cfg.lin_num_v_heads, cfg.lin_value_head_dim)
        z_view = z.view(1, n, cfg.lin_num_v_heads, cfg.lin_value_head_dim)
        gated = normed_core * fun.silu(z_view.to(normed_core.dtype))

        out = fun.linear(gated.reshape(n, value_dim), self.weights.get(f"layers.{layer_idx}.lin_out"))
        return residual + out

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

        # On prefill, reset the SSM and conv state so a fresh request starts
        # from zeros. (Single-request restriction.)
        is_prefill = not single_token_inference_mode
        if is_prefill:
            self._ssm_state.clear()
            self._conv_state.clear()

        # Plan flashinfer wrapper for full-attention layers (one wrapper, reused).
        if any(cfg.is_full_attention(i) for i in range(cfg.num_layers)):
            page_size = int(kv_cache_at_layer[0].shape[2])
            local_q = cfg.num_q_heads // self.runtime_config.tensor_parallel_size
            local_kv = cfg.num_kv_heads // self.runtime_config.tensor_parallel_size

            seq_lens = ops.get_seq_lens(kv_page_indptr, kv_last_page_lens, page_size)
            batch_indices, batch_positions = ops.get_batch_indices_positions(
                append_indptr=qo_indptr, seq_lens=seq_lens, nnz=n,
            )
            del seq_lens

            # The flashinfer cubin only ships precompiled BatchDecode kernels
            # for these group sizes. For others (e.g. Qwen3.6-27B has 24/4=6)
            # we fall back to BatchPrefill, which is general — decode is a
            # degenerate prefill with one query token per request.
            decode_supported_groups = {1, 2, 4, 8, 16, 32}
            group_size = local_q // local_kv
            use_decode_kernel = (
                single_token_inference_mode
                and group_size in decode_supported_groups
            )
            if use_decode_kernel:
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
                    q_data_type=input_embeds.dtype,
                )
        else:
            wrapper = None
            batch_indices = batch_positions = None

        h = input_embeds
        for layer_idx in range(cfg.num_layers):
            if cfg.is_full_attention(layer_idx):
                full_idx = self.full_attn_layer_lookup[layer_idx]
                h = self.attention_full(
                    h, layer_idx, position_ids,
                    kv_cache_layer=kv_cache_at_layer[full_idx],
                    kv_page_indices=kv_page_indices,
                    kv_page_indptr=kv_page_indptr,
                    kv_last_page_lens=kv_last_page_lens,
                    batch_indices=batch_indices,
                    batch_positions=batch_positions,
                    wrapper=wrapper,
                )
            else:
                h = self.attention_linear(h, layer_idx, is_prefill=is_prefill)
            h = self.mlp(h, layer_idx)
        return h


# =============================================================================
# Cache factories — exported under the standard registry contract
# =============================================================================


def create_kv_cache(model_config: ModelConfig, runtime_config: RuntimeConfig) -> list[torch.Tensor]:
    """Paged-KV cache *only for full-attention layers* — the linear-attention
    layers' SSM and conv state live on the ForwardPass instance. The list is
    indexed by `ForwardPass.full_attn_layer_lookup[model_layer_idx]`.
    """
    local_kv_heads = model_config.num_kv_heads // runtime_config.tensor_parallel_size
    n_full = model_config.num_full_attention_layers()
    return [
        torch.zeros(
            (
                runtime_config.total_pages + 1,
                2,
                runtime_config.kv_page_size,
                local_kv_heads,
                model_config.dim_head,
            ),
            dtype=runtime_config.activation_dtype,
            device=runtime_config.device,
        )
        for _ in range(n_full)
    ]


def create_adapter_cache(model_config: ModelConfig, runtime_config: RuntimeConfig):
    # Adapters are not supported on this path; engine contract requires a list.
    return []
