"""Olmo3 Large Language Model Architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import torch
import torch.nn.functional as fun
import torch.distributed as dist
import math

from . import ModelConfig as ModelConfigBase
from ..config import RuntimeConfig
from ..adapter import AdapterSubpass
from ..utils import is_apple_silicon, get_available_memory
from ..loader import Schema, Source, WeightStore

if is_apple_silicon():
    import flashinfer_metal as ops  # type: ignore[import-not-found]
else:
    import flashinfer as ops  # type: ignore[import-not-found,no-redef]

from . import common


# =============================================================================
# OLMO3 WEIGHT SCHEMA
# =============================================================================
# Declarative definition of how physical tensor names map to logical names,
# with fusion, sharding, and quantization applied.
#
# Key differences from Qwen3:
# - Post-norm architecture (norm applied AFTER attention/MLP blocks)
# - Uses post_attention_layernorm and post_feedforward_layernorm
# - Has QK normalization weights (q_norm, k_norm per layer)

OLMO3_SCHEMA = (
    Schema("olmo3")
    # Embedding (row-parallel sharding, no quantization)
    .define(
        "embed_token",
        Source("model.embed_tokens.weight").shard("row"),
    )
    .define(
        "lm_head",
        Source("lm_head.weight").shard("column"),
    )
    # Per-layer weights - POST normalization (Olmo3-specific)
    .define(
        "layers.*.norm_attn",
        Source("model.layers.*.post_attention_layernorm.weight"),
    )
    .define(
        "layers.*.norm_ffn",
        Source("model.layers.*.post_feedforward_layernorm.weight"),
    )
    # Attention (fused QKV, column-parallel)
    .define(
        "layers.*.attn.qkv",
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
    # Attention Norms (Olmo3-specific)
    .define(
        "layers.*.attn.q_norm",
        Source("model.layers.*.self_attn.q_norm.weight"),
    )
    .define(
        "layers.*.attn.k_norm",
        Source("model.layers.*.self_attn.k_norm.weight"),
    )
    # Attention Output (row-parallel, quantized)
    .define(
        "layers.*.attn.proj",
        Source("model.layers.*.self_attn.o_proj.weight").shard("row").quantize(),
    )
    # MLP (Gate+Up fused, column-parallel, quantized)
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


@dataclass
class ModelConfig(ModelConfigBase):
    """
    Olmo3-specific model architecture configuration.

    Inherits from the abstract ModelConfig base class and defines
    all architecture-specific parameters for Olmo3 models.

    Key differences from other models:
    - Post-norm architecture (norm after attention/MLP)
    - Has QK normalization (q_norm, k_norm)
    - Supports sliding window attention via layer_types
    - Uses YaRN (Yet another RoPE extensioN) for rope scaling
    """

    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    num_vocabs: int

    dim_head: int
    dim_hidden: int
    dim_mlp: int

    rms_norm_eps: float

    rope_theta: float

    # Sliding window support
    sliding_window: int | None
    layer_types: list[str] | None  # Per-layer attention types

    # YaRN rope scaling parameters
    rope_scaling_type: str | None  # "yarn", "linear", or None
    rope_scaling_factor: float
    rope_scaling_original_max_position_embeddings: int
    rope_scaling_attention_factor: float
    rope_scaling_beta_fast: float
    rope_scaling_beta_slow: float

    @staticmethod
    def from_dict(spec: dict) -> "ModelConfig":
        # Calculate head_dim if not present
        if "head_dim" in spec:
            head_dim = int(spec["head_dim"])
        else:
            head_dim = int(spec["hidden_size"]) // int(spec["num_attention_heads"])

        # Extract layer_types for sliding attention support
        layer_types = spec.get("layer_types", None)

        # Parse rope_scaling config
        rope_scaling = spec.get("rope_scaling", {})
        rope_scaling_type = rope_scaling.get("rope_type", None) if rope_scaling else None
        rope_scaling_factor = float(rope_scaling.get("factor", 1.0)) if rope_scaling else 1.0
        rope_scaling_original_max = int(rope_scaling.get("original_max_position_embeddings", 8192)) if rope_scaling else 8192
        rope_scaling_attention_factor = float(rope_scaling.get("attention_factor", 1.0)) if rope_scaling else 1.0
        rope_scaling_beta_fast = float(rope_scaling.get("beta_fast", 32.0)) if rope_scaling else 32.0
        rope_scaling_beta_slow = float(rope_scaling.get("beta_slow", 1.0)) if rope_scaling else 1.0

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
            sliding_window=spec.get("sliding_window", None),
            layer_types=layer_types,
            rope_scaling_type=rope_scaling_type,
            rope_scaling_factor=rope_scaling_factor,
            rope_scaling_original_max_position_embeddings=rope_scaling_original_max,
            rope_scaling_attention_factor=rope_scaling_attention_factor,
            rope_scaling_beta_fast=rope_scaling_beta_fast,
            rope_scaling_beta_slow=rope_scaling_beta_slow,
        )

    @property
    def internal_num_kv_heads(self) -> int:
        """
        Returns number of KV heads used internally by the engine.
        If GQA ratio is 5 (40/8), flashinfer 0.6.1 crashes with 'Unsupported group_size: 5'.
        We expand to MHA (num_kv_heads = num_q_heads) for compatibility.
        """
        # if self.num_kv_heads > 0 and self.num_q_heads // self.num_kv_heads == 5:
        #      # Workaround for GQA 5 crash: Expand to MHA
        #      return self.num_q_heads
        return self.num_kv_heads

    def eval_max_num_kv_pages(self, runtime_config: RuntimeConfig) -> int:
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
        # Use internal_num_kv_heads to account for expansion
        local_num_kv_heads = self.internal_num_kv_heads // runtime_config.tensor_parallel_size

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


class ForwardPass:
    """
    Olmo3 forward pass implementation.

    Stores model config, runtime config, and weights internally.

    Key differences from Qwen3:
    - POST-normalization: Applies layer norm AFTER attention/MLP blocks
    - Applies QK normalization before RoPE (similar to Qwen3)
    """

    def __init__(
        self,
        model_config: ModelConfig,
        runtime_config: RuntimeConfig,
        weights: WeightStore,
        compute_process_group: dist.ProcessGroup | None = None,
    ):
        """Initialize the forward pass with weights and attention wrappers."""
        self.model_config = model_config
        self.runtime_config = runtime_config
        self.weights = weights
        self.compute_process_group = compute_process_group
        self.tp_size = runtime_config.tensor_parallel_size
        self.tp_rank = runtime_config.rank % self.tp_size

        # Create workspace buffer for attention operations
        self.workspace_buffer = torch.zeros(
            2048 * 1024 * 1024, dtype=torch.uint8, device=self.runtime_config.device
        )
        self.wrapper_decode = ops.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD", use_tensor_cores=True
        )
        self.wrapper_append = ops.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )

        # Pre-compute YaRN RoPE cos/sin cache if using rope scaling
        self.cos_sin_cache = self._build_yarn_cos_sin_cache(
            device=self.runtime_config.device,
            dtype=self.runtime_config.activation_dtype,
        )

        # --- CUDA Graph Setup (Bins + Padding) ---
        self.use_cuda_graphs = runtime_config.use_cuda_graphs

        # Dynamic bins: Powers of 2 up to 16, then steps of 16
        limit = runtime_config.max_batch_size or 512
        bins = [1, 2, 4, 8, 16]
        bins = [b for b in bins if b <= limit]
        if limit > 16:
            bins.extend(range(24, limit + 1, 16))
            if bins[-1] < limit:  # Ensure we cover the limit if not multiple of 8
                bins.append(limit)
        self.cuda_graph_bins = sorted(list(set(bins)))
        max_bin = self.cuda_graph_bins[-1]

        self.cuda_graph_wrappers = {}
        device = runtime_config.device

        # Alloc shared static buffers
        self.shared_static_hidden = torch.zeros(
            (max_bin, model_config.dim_hidden),
            dtype=runtime_config.activation_dtype,
            device=device,
        )
        self.shared_static_indptr = torch.zeros(
            max_bin + 1, dtype=torch.int32, device=device
        )
        self.shared_static_last_len = torch.zeros(
            max_bin, dtype=torch.int32, device=device
        )
        self.shared_static_position_ids = torch.zeros(
            max_bin, dtype=torch.int32, device=device
        )
        self.shared_static_batch_indices = torch.zeros(
            max_bin, dtype=torch.int32, device=device
        )
        self.shared_static_batch_positions = torch.zeros(
            max_bin, dtype=torch.int32, device=device
        )

        self.cuda_graph_aux_buffers = {}  # Maps bin -> (indptr_view, last_len_view)

        # Scratch page index for padding (use the extra page we allocated)
        self.scratch_page_idx = runtime_config.max_num_kv_pages

        # Initialize wrappers for each bin using VIEWS of shared buffers
        max_num_pages = runtime_config.max_num_kv_pages + 1
        self.shared_kv_indices_buffer = torch.zeros(
            max_num_pages, dtype=torch.int32, device=device
        )

        if self.use_cuda_graphs:
            for b in self.cuda_graph_bins:
                # Create views
                indptr_view = self.shared_static_indptr[: b + 1]
                last_len_view = self.shared_static_last_len[:b]
                self.cuda_graph_aux_buffers[b] = (indptr_view, last_len_view)

                # Wrapper with usage_cuda_graph=True
                self.cuda_graph_wrappers[b] = ops.BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_cuda_graph=True,
                    use_tensor_cores=True,
                    paged_kv_indptr_buffer=indptr_view,
                    paged_kv_indices_buffer=self.shared_kv_indices_buffer,
                    paged_kv_last_page_len_buffer=last_len_view,
                )

        # CUDA Graph cache for the layer loop: bin_size -> (graph, static_inputs..., static_output)
        self.cuda_graph_img: dict[int, tuple] = {}

        # Fallback Decode wrapper
        self.wrapper_decode_fallback = ops.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD", use_tensor_cores=True
        )

    def warmup_cuda_graphs(self, kv_cache_at_layer: list[torch.Tensor]):
        """
        Pre-capture CUDA graphs for all defined bins using shared static buffers.
        This avoids lag during the first few inference steps.
        """
        if not self.use_cuda_graphs:
            return

        from tqdm import tqdm

        device = self.runtime_config.device

        # Local head counts for planning
        local_num_query_heads = self.model_config.num_q_heads // self.tp_size
        local_num_key_value_heads = self.model_config.num_kv_heads // self.tp_size
        page_size = self.runtime_config.kv_page_size

        print(f"Warmup: Capturing CUDA graphs for bins {self.cuda_graph_bins}...")

        for b in tqdm(self.cuda_graph_bins, desc="CUDA Graphs"):
            # 1. Get Views of Shared Buffers
            indptr_view, last_len_view = self.cuda_graph_aux_buffers[b]
            hidden_view = self.shared_static_hidden[:b]

            # 2. Plan Wrapper
            indptr_view.copy_(torch.arange(b + 1, dtype=torch.int32, device=device))
            last_len_view.fill_(1)

            wrapper = self.cuda_graph_wrappers[b]
            wrapper.plan(
                indptr=indptr_view,
                indices=self.shared_kv_indices_buffer,
                last_page_len=last_len_view,
                num_qo_heads=local_num_query_heads,
                num_kv_heads=local_num_key_value_heads,
                head_dim=self.model_config.dim_head,
                page_size=page_size,
                pos_encoding_mode="NONE",
                q_data_type=self.runtime_config.activation_dtype,
                window_left=self.model_config.sliding_window or -1,
            )

            # 3. Create Dummy Inputs for Capture
            dummy_indices = torch.full(
                (b,), self.scratch_page_idx, device=device, dtype=torch.int32
            )
            self.shared_kv_indices_buffer[:b].copy_(dummy_indices)

            # Dummy Position IDs (View)
            pos_view = self.shared_static_position_ids[:b]
            pos_view.zero_()

            # Dummy Batch Indices (View)
            batch_indices_view = self.shared_static_batch_indices[:b]
            batch_indices_view.copy_(torch.arange(b, device=device, dtype=torch.int32))

            # Dummy Batch Positions (View)
            batch_pos_view = self.shared_static_batch_positions[:b]
            batch_pos_view.zero_()

            # 4. Capture
            graph = torch.cuda.CUDAGraph()
            # Run once before capture (warmup caches)
            self._run_layers(
                hidden_states=hidden_view,
                position_ids=pos_view,
                kv_cache_at_layer=kv_cache_at_layer,
                kv_page_indices=self.shared_kv_indices_buffer,
                kv_page_indptr=indptr_view,
                kv_last_page_lens=last_len_view,
                batch_indices=batch_indices_view,
                batch_positions=batch_pos_view,
                adapter_subpass=None,
                wrapper=wrapper,
            )

            with torch.cuda.graph(graph):
                output = self._run_layers(
                    hidden_states=hidden_view,
                    position_ids=pos_view,
                    kv_cache_at_layer=kv_cache_at_layer,
                    kv_page_indices=self.shared_kv_indices_buffer,
                    kv_page_indptr=indptr_view,
                    kv_last_page_lens=last_len_view,
                    batch_indices=batch_indices_view,
                    batch_positions=batch_pos_view,
                    adapter_subpass=None,
                    wrapper=wrapper,
                )

            self.cuda_graph_img[b] = (graph,)

        torch.cuda.synchronize()
        print("Warmup complete.")

    def _build_yarn_cos_sin_cache(
        self,
        device: torch.device,
        dtype: torch.dtype,
        max_seq_len: int = 65536,
    ) -> torch.Tensor:
        """
        Build YaRN (Yet another RoPE extensioN) cos/sin cache.

        YaRN interpolates between linear interpolation and no interpolation
        based on frequency, with a smooth ramp function controlled by beta_fast/beta_slow.

        Returns:
            cos_sin_cache: [max_seq_len, head_dim] where first half is cos, second half is sin
        """
        cfg = self.model_config
        head_dim = cfg.dim_head
        dim = head_dim  # Full head dim for RoPE
        base = cfg.rope_theta

        # Helper functions from HuggingFace transformers
        def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
            """Inverse dimension formula to find the dimension based on the number of rotations"""
            return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

        def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
            """Find dimension range bounds based on rotations"""
            low = find_correction_dim(low_rot, dim, base, max_position_embeddings)
            high = find_correction_dim(high_rot, dim, base, max_position_embeddings)
            low = math.floor(low)
            high = math.ceil(high)
            return max(low, 0), min(high, dim - 1)

        def linear_ramp_factor(min_val, max_val, dim):
            if min_val == max_val:
                max_val += 0.001  # Prevent singularity
            linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
            return torch.clamp(linear_func, 0, 1)

        # Base frequencies
        pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (cfg.rope_scaling_factor * pos_freqs)

        # Apply YaRN scaling if configured
        if cfg.rope_scaling_type == "yarn" and cfg.rope_scaling_factor > 1.0:
            factor = cfg.rope_scaling_factor
            original_max = cfg.rope_scaling_original_max_position_embeddings
            beta_fast = cfg.rope_scaling_beta_fast
            beta_slow = cfg.rope_scaling_beta_slow

            # Find correction range using HuggingFace formula
            low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max)

            # Get n-dimensional rotational scaling corrected for extrapolation
            # inv_freq_extrapolation_factor = 1 means use extrapolation (no scaling)
            # inv_freq_extrapolation_factor = 0 means use interpolation (full scaling)
            inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2)

            inv_freq = (
                inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
                + inv_freq_extrapolation * inv_freq_extrapolation_factor
            )
        else:
            inv_freq = inv_freq_extrapolation

        # Compute position embeddings
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # [max_seq_len, dim/2]

        # Apply YaRN attention scaling factor (matches HuggingFace reference)
        attention_scaling = cfg.rope_scaling_attention_factor
        
        cos = torch.cos(freqs) * attention_scaling
        sin = torch.sin(freqs) * attention_scaling

        # FlashInfer expects cos_sin_cache shape: [max_seq_len, rotary_dim]
        # For is_neox=True (GPT-NeoX style), cos is first half, sin is second half
        cos_sin_cache = torch.cat([cos, sin], dim=-1)  # [max_seq_len, dim]

        return cos_sin_cache.to(device=device, dtype=torch.float32)

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs with Tensor Parallel support (Column Parallel).

        The embedding weight is column-sharded: [vocab_size, hidden_size/world_size]
        Each rank computes partial hidden states, then all_gather combines them.
        """
        world_size = self.runtime_config.world_size

        if self.tp_size == 1:
            return fun.embedding(token_ids, self.weights.get("embed_token"))

        # Column-parallel embedding: each rank has [vocab_size, hidden_size/tp_size]
        # 1. Lookup - each rank gets partial hidden states [seq_len, hidden_size/tp_size]
        local_embeds = fun.embedding(token_ids, self.weights.get("embed_token"))

        # 2. All-gather to combine partial hidden states from all ranks
        # Output: [seq_len, hidden_size] (full hidden dimension)
        gathered_list = [torch.empty_like(local_embeds) for _ in range(self.tp_size)]
        dist.all_gather(gathered_list, local_embeds, group=self.compute_process_group)

        # Concatenate along hidden dimension (last dim)
        full_embeds = torch.cat(gathered_list, dim=-1)

        return full_embeds

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

        embeds = self.embed_tokens(token_ids_tensor)
        
        return embeds

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits (weight-tied with embed_tokens).

        The embedding weight is column-sharded: [vocab_size, hidden_size/world_size]
        For lm_head (linear projection), this is effectively [hidden_size/world_size, vocab_size]
        when transposed.

        Column-parallel lm_head:
        1. Split input hidden_states along hidden dimension
        2. Each rank computes partial logits with its weight shard
        3. All-reduce sums the partial logits to get full result
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
            return fun.linear(normed, self.weights.get("lm_head"))

        # Multi-GPU: Column-parallel projection
        # 1. Split input along hidden dimension - each rank uses its slice
        #    (Wait, lm_head is column-parallel which means:
        #     Input is replicated [N, H], Weight is [H, V/TP] -> Output [N, V/TP]
        #     Then we gather output)
        
        # Actually Olmo3 usage of lm_head should be Column Parallel:
        # Input [N, H] -> Linear(H, V/TP) -> Output [N, V/TP]
        # Then Gather(dim=-1) -> Output [N, V]
        
        # Checking how fun.linear handles sharded weights:
        # If weight is [Output, Input], checking source...
        # Source("lm_head.weight").shard("column") implies split on dim 0.
        # This means each rank gets [V/TP, H].
        # fun.linear(x, w) computes x @ w.T
        # x=[N,H], w=[V/TP, H]. w.T=[H, V/TP]. x@w.T = [N, V/TP].
        # So each rank produces partial vocabulary logits.
        # We need to gather them.
        
        local_logits = fun.linear(
            normed, 
            self.weights.get("lm_head")
        )

        return self.compute_process_group.all_gather(local_logits, dim=-1)

    def mlp(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Executes the MLP block for a single layer.

        Olmo3-specific: MLP is applied BEFORE the post-MLP norm.
        The residual connection and norm are handled by the caller (_run_layers).
        """
        # --- Calculate local TP sizes ---
        local_mlp_size = self.model_config.dim_mlp // self.tp_size

        # 1. Gate+Up Projection (Column Parallel)
        # Input is replicated, weight is sharded -> output is sharded
        gate_up = fun.linear(
            hidden_states,
            self.weights.get(f"layers.{layer_idx}.proj_gate_up"),
            None,
        )

        # Split gate and up
        gate, up = torch.split(gate_up, [local_mlp_size, local_mlp_size], dim=-1)

        # 2. SiLU activation * gate (SwiGLU)
        hidden = fun.silu(gate) * up

        # 3. Down Projection (Row Parallel)
        # Input is sharded, weight is sharded -> output is sharded
        down = fun.linear(
            hidden,
            self.weights.get(f"layers.{layer_idx}.proj_down"),
            None,
        )
        del hidden, gate, up, gate_up

        # ALL-REDUCE: Sum partial outputs from all ranks (only if TP > 1)
        if self.tp_size > 1:
            dist.all_reduce(down, group=self.compute_process_group)

        return down

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
        Executes the attention block for a single layer.

        Olmo3-specific:
        - No pre-norm (Olmo3 uses post-norm architecture)
        - Applies QK normalization before RoPE
        - Returns raw attention output (residual + post-norm handled by caller)
        """
        # --- Calculate local TP sizes ---
        local_num_query_heads = self.model_config.num_q_heads // self.tp_size
        local_num_key_value_heads = self.model_config.num_kv_heads // self.tp_size
        local_q_size = local_num_query_heads * self.model_config.dim_head
        local_kv_size = local_num_key_value_heads * self.model_config.dim_head

        n = hidden_states.size(0)

        # 1. QKV Projection (Column Parallel) - NO bias, NO pre-norm in Olmo3
        # Input is replicated, weight is sharded -> output is sharded
        qkv_proj = fun.linear(
            hidden_states,
            self.weights.get(f"layers.{layer_idx}.attn.qkv"),
            None,
        )

        # q, k, v are all LOCAL shards (FLAT: [n, local_q_size] etc.)
        q, k, v = torch.split(
            qkv_proj,
            [
                local_q_size,
                local_kv_size,
                local_kv_size,
            ],
            dim=-1,
        )



        # 2. QK Normalization (Olmo3-specific)
        # Applied to FLAT tensors BEFORE reshape, matching HuggingFace reference
        q_norm_w = self.weights.get(f"layers.{layer_idx}.attn.q_norm")  # [local_q_size]
        k_norm_w = self.weights.get(f"layers.{layer_idx}.attn.k_norm")  # [local_kv_size]

        # Apply RMSNorm to flat Q and K
        q = fun.rms_norm(
            q,
            normalized_shape=[local_q_size],
            weight=q_norm_w,
            eps=self.model_config.rms_norm_eps,
        )
        k = fun.rms_norm(
            k,
            normalized_shape=[local_kv_size],
            weight=k_norm_w,
            eps=self.model_config.rms_norm_eps,
        )

        # 3. Adapter (if any)
        if adapter_subpass is not None:
            adapter_subpass.execute(
                layer_idx,
                hidden_states,  # Adapter needs the input
                q_state=q,
                k_state=k,
                v_state=v,
                rank=self.runtime_config.rank,
                world_size=self.runtime_config.world_size,
            )

        # 4. Reshape QKV to [n, heads, head_dim]
        q = q.view(n, local_num_query_heads, self.model_config.dim_head)
        k = k.view(n, local_num_key_value_heads, self.model_config.dim_head)
        v = v.view(n, local_num_key_value_heads, self.model_config.dim_head)



        # 5. Workaround for GQA-5 crash: Expand K/V to MHA if needed
        # We did the earlier split/norm using original heads (8).
        # Now we expand to internal heads (40) for FlashInfer/Cache.
        local_kv_heads_internal = self.model_config.internal_num_kv_heads // self.tp_size
        if local_kv_heads_internal != local_num_key_value_heads:
             ratio = local_kv_heads_internal // local_num_key_value_heads
             k = k.repeat_interleave(ratio, dim=1)
             v = v.repeat_interleave(ratio, dim=1)

        # 6. Apply RoPE (in-place on local shards)
        # Use precomputed YaRN cos/sin cache if rope scaling is enabled
        if self.model_config.rope_scaling_type == "yarn":
            # FlashInfer apply_rope_with_cos_sin_cache expects flattened q/k
            # q shape: [n, heads, head_dim] -> [n, heads * head_dim]
            n_tokens = q.shape[0]
            q_flat = q.view(n_tokens, -1)
            k_flat = k.view(n_tokens, -1)

            ops.apply_rope_with_cos_sin_cache_inplace(
                positions=position_ids,
                query=q_flat,
                key=k_flat,
                head_size=self.model_config.dim_head,
                cos_sin_cache=self.cos_sin_cache,
                is_neox=True,  # GPT-NeoX style RoPE (non-interleaved)
            )

            # Reshape back
            q = q_flat.view(n, -1, self.model_config.dim_head)
            k = k_flat.view(n, -1, self.model_config.dim_head)
        else:
            ops.apply_rope_pos_ids_inplace(
                q=q,
                k=k,
                pos_ids=position_ids,
                rope_theta=self.model_config.rope_theta,
            )

        # 6. Append K, V to cache (local shards to local cache)
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




        # 7. Compute Attention (on local shards)
        attn_output = wrapper.run(q, kv_cache_layer)
        del q, k, v
        del qkv_proj

        # attn_output is a local shard
        attn_output = attn_output.reshape(n, -1)

        # 8. Output Projection (Row Parallel)
        # Input is sharded, weight is sharded -> output is sharded
        attn_proj = fun.linear(
            attn_output,
            self.weights.get(f"layers.{layer_idx}.attn.proj"),
            None,
        )
        del attn_output

        # ALL-REDUCE: Sum partial outputs from all ranks (only if TP > 1)
        if self.tp_size > 1:
            dist.all_reduce(attn_proj, group=self.compute_process_group)

        return attn_proj

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
        total_pages_cpu: int = 0,
    ) -> torch.Tensor:
        """Main transformation pipeline through all layers."""
        # Only set CUDA device for CUDA tensors (not MPS)
        if self.runtime_config.device.type == "cuda":
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
        local_num_query_heads = self.model_config.num_q_heads // self.tp_size
        local_num_key_value_heads = self.model_config.internal_num_kv_heads // self.tp_size

        if single_token_inference_mode:
            if self.use_cuda_graphs:
                return self._run_layers_graphed(
                    hidden_states=input_embeds,
                    position_ids=position_ids,
                    kv_cache_at_layer=kv_cache_at_layer,
                    kv_page_indices=kv_page_indices,
                    kv_page_indptr=kv_page_indptr,
                    kv_last_page_lens=kv_last_page_lens,
                    batch_indices=batch_indices,
                    batch_positions=batch_positions,
                    total_pages_cpu=total_pages_cpu,
                )
            # Normal decode fallback
            wrapper = self.wrapper_decode_fallback
            # NOTE: wrapper.plan() is now called per-layer inside _run_layers
        else:
            wrapper = self.wrapper_append
            # NOTE: wrapper.plan() is now called per-layer inside _run_layers

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
            # Per-layer planning parameters
            is_decode=single_token_inference_mode,
            page_size=page_size,
            qo_indptr=qo_indptr,
            custom_mask=custom_mask,
            input_dtype=input_embeds.dtype,
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
        # Added planning parameters for per-layer window
        is_decode: bool,
        page_size: int,
        qo_indptr: torch.Tensor | None,
        custom_mask: torch.Tensor | None,
        input_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Execute all transformer layers sequentially.

        Olmo3 uses POST-normalization architecture:
        - hidden = hidden + post_attn_norm(attention(hidden))
        - hidden = hidden + post_mlp_norm(mlp(hidden))

        Per-layer attention types: Each layer can be "sliding_attention" or "full_attention"
        based on model_config.layer_types[layer_idx].
        """
        local_num_query_heads = self.model_config.num_q_heads // self.tp_size
        local_num_key_value_heads = self.model_config.internal_num_kv_heads // self.tp_size



        for layer_idx in range(self.model_config.num_layers):
            # --- Per-layer attention type: determine window_left ---
            layer_types = self.model_config.layer_types
            if layer_types and layer_idx < len(layer_types):
                attention_type = layer_types[layer_idx]
                # sliding_attention uses sliding_window, full_attention uses -1 (no window)
                if attention_type == "sliding_attention":
                    window_left = self.model_config.sliding_window or -1
                else:
                    window_left = -1  # full_attention
            else:
                # Fallback to global setting
                window_left = self.model_config.sliding_window or -1

            # --- Plan wrapper for this layer with per-layer window ---
            if is_decode:
                wrapper.plan(
                    indptr=kv_page_indptr,
                    indices=kv_page_indices,
                    last_page_len=kv_last_page_lens,
                    num_qo_heads=local_num_query_heads,
                    num_kv_heads=local_num_key_value_heads,
                    head_dim=self.model_config.dim_head,
                    page_size=page_size,
                    pos_encoding_mode="NONE",
                    q_data_type=input_dtype,
                    window_left=window_left,
                )
            else:
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
                    q_data_type=input_dtype,
                    window_left=window_left,
                )

            # Save input for residual connection
            residual = hidden_states

            # Attention block
            attn_output = self.attention(
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

            # Post-attention norm (applied BEFORE residual connection)
            attn_output = fun.rms_norm(
                attn_output,
                normalized_shape=[self.model_config.dim_hidden],
                weight=self.weights.get(f"layers.{layer_idx}.norm_attn"),
                eps=self.model_config.rms_norm_eps,
            )

            # Residual connection (added AFTER norm)
            hidden_states = residual + attn_output

            # Save for MLP residual
            residual = hidden_states

            # MLP block
            mlp_output = self.mlp(
                hidden_states=hidden_states,
                layer_idx=layer_idx,
            )

            # Post-MLP norm (applied BEFORE residual connection)
            mlp_output = fun.rms_norm(
                mlp_output,
                normalized_shape=[self.model_config.dim_hidden],
                weight=self.weights.get(f"layers.{layer_idx}.norm_ffn"),
                eps=self.model_config.rms_norm_eps,
            )

            # Store normalization for next layer
            hidden_states = residual + mlp_output
            


            # Apply residual (Post-Norm architecture: out = residual + norm(block))
            # However, looking at HF:
            # hidden_states = hidden_states + residual
            # But earlier in loop: residual = hidden_states
            # And: hidden_states = attention...
            # The structure is:
            #   residual = hidden_states
            #   hidden_states = layernorm(hidden_states)
            #   hidden_states = attention(hidden_states)
            #   hidden_states = residual + hidden_states
            #
            # My current codestructure in _run_layers:
            # hidden_states calculated in self.layers[idx] (which is Olmo3DecoderLayer)
            # Wait, Olmo3DecoderLayer applies the residual?
            # Let's check Olmo3DecoderLayer.forward/transform
            
        return hidden_states

    def _get_bin(self, batch_size: int) -> int | None:
        """Find the smallest bin >= batch_size."""
        for b in self.cuda_graph_bins:
            if b >= batch_size:
                return b
        return None

    def _run_layers_graphed(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache_at_layer: list[torch.Tensor],
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_positions: torch.Tensor,
        total_pages_cpu: int = 0,
    ) -> torch.Tensor:
        """
        Execute layers with CUDA graph replay using shared buffers.
        Optimized for zero-overhead launch.
        """
        batch_size = hidden_states.shape[0]
        bin_size = self._get_bin(batch_size)

        if bin_size is None or bin_size not in self.cuda_graph_img:
            # Fallback
            wrapper = self.wrapper_decode_fallback
            page_size = self.runtime_config.kv_page_size
            wrapper.plan(
                indptr=kv_page_indptr,
                indices=kv_page_indices,
                last_page_len=kv_last_page_lens,
                num_qo_heads=self.model_config.num_q_heads
                // self.runtime_config.world_size,
                num_kv_heads=self.model_config.internal_num_kv_heads
                // self.runtime_config.world_size,
                head_dim=self.model_config.dim_head,
                page_size=page_size,
                pos_encoding_mode="NONE",
                q_data_type=hidden_states.dtype,
                window_left=self.model_config.sliding_window or -1,
            )
            return self._run_layers(
                hidden_states,
                position_ids,
                kv_cache_at_layer,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_lens,
                batch_indices,
                batch_positions,
                None,
                wrapper,
            )

        # 1. Copy data to Shared Static Buffers
        self.shared_static_hidden[:batch_size].copy_(hidden_states)

        indptr_view = self.cuda_graph_aux_buffers[bin_size][0]
        indptr_view[: batch_size + 1].copy_(kv_page_indptr)

        last_len_view = self.cuda_graph_aux_buffers[bin_size][1]
        last_len_view[:batch_size].copy_(kv_last_page_lens)

        num_indices = kv_page_indices.numel()
        self.shared_kv_indices_buffer[:num_indices].copy_(kv_page_indices)

        self.shared_static_position_ids[:batch_size].copy_(position_ids)
        self.shared_static_batch_indices[:batch_size].copy_(batch_indices)
        self.shared_static_batch_positions[:batch_size].copy_(batch_positions)

        # 2. Handle Padding
        if batch_size < bin_size:
            # Pad Indptr
            remainder = bin_size - batch_size
            padding = torch.arange(
                1, remainder + 1, device=self.runtime_config.device, dtype=torch.int32
            )
            padding.add_(total_pages_cpu)
            indptr_view[batch_size + 1 :].copy_(padding)

            # Pad Last Len
            last_len_view[batch_size:].fill_(1)

            # Pad Position IDs
            self.shared_static_position_ids[batch_size:].zero_()

            # Pad Batch Indices / Positions
            self.shared_static_batch_indices[batch_size:].zero_()
            self.shared_static_batch_positions[batch_size:].zero_()

        # 3. Replay
        graph = self.cuda_graph_img[bin_size][0]
        graph.replay()

        # 4. Return Output Slice
        return self.shared_static_hidden[:batch_size].clone()


def create_kv_cache(
    model_config: ModelConfig, runtime_config: RuntimeConfig
) -> list[torch.Tensor]:
    """Create KV cache tensors for all layers."""
    local_num_kv_heads = (
        model_config.internal_num_kv_heads // runtime_config.tensor_parallel_size
    )

    return [
        torch.zeros(
            (
                runtime_config.max_num_kv_pages + 1,  # +1 for scratch/padding page
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
      (Sharded: each rank stores only its portion of the up-projection)
    """
    local_num_q_heads = model_config.num_q_heads // runtime_config.world_size
    local_num_kv_heads = model_config.num_kv_heads // runtime_config.world_size

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
