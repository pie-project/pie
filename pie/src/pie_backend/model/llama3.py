from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Any

import torch
import torch.nn.functional as fun
import torch.distributed as dist


from . import ModelConfig as ModelConfigBase
from ..config import RuntimeConfig
from ..adapter import AdapterSubpass
from ..quantization import quantize
from ..utils import is_apple_silicon, get_available_memory
from ..loader import Schema, Source, WeightStore

if is_apple_silicon():
    import flashinfer_metal as ops  # type: ignore[import-not-found]
else:
    import flashinfer as ops  # type: ignore[import-not-found,no-redef]

from . import common


# =============================================================================
# LLAMA3 WEIGHT SCHEMA
# =============================================================================
# Declarative definition of how physical tensor names map to logical names,
# with fusion, sharding, and quantization applied.


def create_schema(config: dict) -> Schema:
    schema = (
        Schema("llama3")
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
        # Fused QKV projection weight: fused from [Q, K, V] and sharded INTERLEAVED
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
        # Fused gate+up projection: fused from [Gate, Up] and sharded INTERLEAVED
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

    # Handle untied embeddings (lm_head separate from embed_token)
    if not config.tie_word_embeddings:
        # LM Head (column-parallel, like embed_token but for output)
        # Actually in `lm_head` method we treat it as linear: [hidden, vocab]
        # In column parallel: split hidden? No.
        # Check `lm_head` method:
        #   local_logits = fun.linear(local_normed, weight)
        #   weight shape: [vocab, hidden/world_size]
        # So we want to shard the weight along dim 1 (hidden dim).
        # Column sharding usually means sharding dim 0?
        # Let's check Source.shard:
        #   dim = 1 if source.sharding == "row" else 0
        # So 'column' -> dim 0 (vocab dim). 'row' -> dim 1 (hidden dim).
        # We need `hidden` dim sharded. That corresponds to 'row' sharding for the weight tensor [vocab, hidden].
        # wait, Weight is [out_features, in_features] for Linear.
        #   lm_head weight: [vocab, hidden]
        #   We want to split `hidden` (in_features). That is dim 1.
        #   So we use .shard("row") which maps to dim 1.
        schema.define(
            "lm_head",
            Source("lm_head.weight").shard("row"),
        )

    return schema


@dataclass
class ModelConfig(ModelConfigBase):
    """
    Llama3-specific model architecture configuration.

    Inherits from the abstract ModelConfig base class and defines
    all architecture-specific parameters for Llama3 models.
    """

    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    num_vocabs: int

    dim_head: int
    dim_hidden: int
    dim_mlp: int

    rms_norm_eps: float

    rope_factor: float
    rope_high_frequency_factor: float
    rope_low_frequency_factor: float
    rope_original_max_position_embeddings: int
    rope_theta: float
    tie_word_embeddings: bool

    @staticmethod
    def from_dict(spec: dict) -> "ModelConfig":
        # Handle rope scaling
        rope = spec.get("rope_scaling") or {}
        # Llama 3 style rope scaling might look different, check for 'rope_type' or just use params

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
            rope_factor=float(rope.get("factor", 8.0)),
            rope_high_frequency_factor=float(rope.get("high_freq_factor", 4.0)),
            rope_low_frequency_factor=float(rope.get("low_freq_factor", 1.0)),
            rope_original_max_position_embeddings=int(
                rope.get("original_max_position_embeddings", 8192)
            ),
            rope_theta=float(spec.get("rope_theta", 500000.0)),
            tie_word_embeddings=bool(spec.get("tie_word_embeddings", False)),
        )

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


class ForwardPass:
    """
    Llama3 forward pass implementation.

    Stores model config, runtime config, and weights internally.
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

        device = self.runtime_config.device

        # Create workspace buffer for attention operations
        self.workspace_buffer = torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )

        # --- CUDA Graph Setup (Bins + Padding) ---
        # Binned batch sizes for graph capture
        # Reduce bins to avoid OOM on smaller GPUs
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
        # Aux buffers derived from shared static buffers (views)
        # Allocate ONE set of maximal buffers
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
                    paged_kv_indptr_buffer=indptr_view,
                    paged_kv_indices_buffer=self.shared_kv_indices_buffer,
                    paged_kv_last_page_len_buffer=last_len_view,
                )

        # CUDA Graph cache for the layer loop: bin_size -> (graph, static_inputs..., static_output)
        self.cuda_graph_img: dict[int, tuple] = {}
        self.use_cuda_graphs = runtime_config.use_cuda_graphs

        # Fallback/Prefill wrapper
        self.wrapper_append = ops.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )
        # Fallback Decode wrapper (for batches > max bin or disabled graphs)
        self.wrapper_decode_fallback = ops.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD"
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
        dim = self.model_config.dim_hidden

        # Local head counts for planning
        local_num_query_heads = self.model_config.num_q_heads // self.tp_size
        local_num_key_value_heads = self.model_config.num_kv_heads // self.tp_size
        page_size = self.runtime_config.kv_page_size

        print(f"Warmup: Capturing CUDA graphs for bins {self.cuda_graph_bins}...")

        for b in tqdm(self.cuda_graph_bins, desc="CUDA Graphs"):
            # 1. Get Views of Shared Buffers
            indptr_view, last_len_view = self.cuda_graph_aux_buffers[b]
            hidden_view = self.shared_static_hidden[:b]

            # 2. Plan Wrapper (Critical: Must be done before capture for each bin config)
            # Update views with valid DUMMY content for planning/capture
            # indptr: 0, 1, 2, ... b (total pages = b)
            indptr_view.copy_(torch.arange(b + 1, dtype=torch.int32, device=device))
            # last_len: 1 (safe)
            last_len_view.fill_(1)

            wrapper = self.cuda_graph_wrappers[b]
            wrapper.plan(
                indptr=indptr_view,
                indices=self.shared_kv_indices_buffer,  # Full buffer view (shared)
                last_page_len=last_len_view,
                num_qo_heads=local_num_query_heads,
                num_kv_heads=local_num_key_value_heads,
                head_dim=self.model_config.dim_head,
                page_size=page_size,
                pos_encoding_mode="NONE",
                q_data_type=self.runtime_config.activation_dtype,
            )

            # 3. Create Dummy Inputs for Capture
            # Indices: point to scratch page
            dummy_indices = torch.full(
                (b,), self.scratch_page_idx, device=device, dtype=torch.int32
            )
            # We must copy dummy indices to the shared buffer because wrapper reads from it!
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

            # Capture
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
        # lm_head expects (hidden_states) and returns logits
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
        # tp_size = self.tp_size (already available)

        if self.tp_size == 1:
            return fun.embedding(token_ids, self.weights.get("embed_token"))

        # Column-parallel embedding: each rank has [vocab_size, hidden_size/tp_size]
        # 1. Lookup - each rank gets partial hidden states [seq_len, hidden_size/tp_size]
        local_embeds = fun.embedding(token_ids, self.weights.get("embed_token"))

        # 2. All-gather to combine partial hidden states from all ranks
        # Output: [seq_len, hidden_size] (full hidden dimension)
        # Output: [seq_len, hidden_size] (full hidden dimension)
        gathered_list = [torch.empty_like(local_embeds) for _ in range(self.tp_size)]
        dist.all_gather(gathered_list, local_embeds, group=self.compute_process_group)

        # Concatenate along hidden dimension (last dim)
        full_embeds = torch.cat(gathered_list, dim=-1)

        return full_embeds

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
        # world_size = self.runtime_config.world_size
        # rank = self.runtime_config.rank

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
        local_normed = normed[:, start_idx:end_idx]  # [seq, hidden/world_size]

        # 2. Project with local weight shard: [seq, hidden/world_size] @ [hidden/world_size, vocab]
        # embed_token has shape [vocab, hidden/world_size] (row-sharded/dim=1)
        # lm_head also has [vocab, hidden/world_size] (row-sharded/dim=1)
        # so we use linear which transposes
        weight = (
            self.weights.get("embed_token")
            if self.model_config.tie_word_embeddings
            else self.weights.get("lm_head")
        )
        local_logits = fun.linear(local_normed, weight)  # [seq, vocab]

        # 3. All-reduce to combine partial logits (sum of partial projections = full projection)
        dist.all_reduce(local_logits, group=self.compute_process_group)

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
            dist.all_reduce(down, group=self.compute_process_group)

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
        local_num_query_heads = self.model_config.num_q_heads // self.tp_size
        local_num_key_value_heads = self.model_config.num_kv_heads // self.tp_size
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

        # 2. QKV Projection (Column Parallel)
        # Input is replicated, weight is sharded -> output is sharded
        qkv_proj = fun.linear(
            normed_input,
            self.weights.get(f"layers.{layer_idx}.proj_qkv"),
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

        # 5. Apply RoPE (in-place on local shards)
        ops.apply_llama31_rope_pos_ids_inplace(
            q=q,
            k=k,
            pos_ids=position_ids,
            rope_scale=self.model_config.rope_factor,
            rope_theta=self.model_config.rope_theta,
            low_freq_factor=self.model_config.rope_low_frequency_factor,
            high_freq_factor=self.model_config.rope_high_frequency_factor,
            old_context_len=self.model_config.rope_original_max_position_embeddings,
        )

        # gather where?

        # 6. Append K, V to cache (local shards to local cache)
        # kv_cache_layer is the LOCAL shard of the cache for this layer
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
            dist.all_reduce(attn_proj, group=self.compute_process_group)

        # 9. First Residual Connection
        # residual (replicated) + attn_proj (now replicated)
        return residual + attn_proj

    def transform(
        self,
        # inputs
        input_embeds: torch.Tensor,  # Replicated [n, hidden_size]
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
        total_pages_cpu: int = 0,  # Added for CUDA graph (avoid .item() sync)
    ) -> torch.Tensor:

        # Ensure we're running on the correct CUDA device (critical for Triton kernels)
        if self.runtime_config.device.type == "cuda":
            torch.cuda.set_device(self.runtime_config.device)

        # --- Calculate local TP sizes ---
        # <-- These are still needed here for planning the wrapper
        local_num_query_heads = self.model_config.num_q_heads // self.tp_size
        local_num_key_value_heads = self.model_config.num_kv_heads // self.tp_size

        hidden_states = input_embeds
        n, _ = hidden_states.size()

        page_size = int(kv_cache_at_layer[0].shape[2])

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
        del seq_lens  # No longer needed

        if single_token_inference_mode:
            if not adapter_subpass:
                # We handle planning inside _run_layers_graphed for graph mode
                # Or pass dummy wrapper if using graph mode
                # But for fallback or pre-graph execution logic verification?
                pass

            # For standard execution (fallback) we need to plan the fallback wrapper
            wrapper = self.wrapper_decode_fallback
            wrapper.plan(
                indptr=kv_page_indptr,
                indices=kv_page_indices,
                last_page_len=kv_last_page_lens,
                num_qo_heads=local_num_query_heads,  # Use local head count
                num_kv_heads=local_num_key_value_heads,  # Use local head count
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

        # Execute layers
        if single_token_inference_mode and self.use_cuda_graphs and not adapter_subpass:
            # Use CUDA graphs for decode mode (no adapters, fixed shapes via padding)
            hidden_states = self._run_layers_graphed(
                hidden_states=hidden_states,
                position_ids=position_ids,
                kv_cache_at_layer=kv_cache_at_layer,
                kv_page_indices=kv_page_indices,
                kv_page_indptr=kv_page_indptr,
                kv_last_page_lens=kv_last_page_lens,
                batch_indices=batch_indices,
                batch_positions=batch_positions,
                total_pages_cpu=total_pages_cpu,
            )
        else:
            # Execute layers (FlashInfer handles CUDA graphs internally for decode mode)
            hidden_states = self._run_layers(
                hidden_states=hidden_states,
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

        # Returns replicated hidden_states
        return hidden_states

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
                num_kv_heads=self.model_config.num_kv_heads
                // self.runtime_config.world_size,
                head_dim=self.model_config.dim_head,
                page_size=page_size,
                pos_encoding_mode="NONE",
                q_data_type=hidden_states.dtype,
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
        model_config.num_kv_heads // runtime_config.tensor_parallel_size
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
    """
    local_num_q_heads = model_config.num_q_heads // runtime_config.tensor_parallel_size
    local_num_kv_heads = (
        model_config.num_kv_heads // runtime_config.tensor_parallel_size
    )

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
