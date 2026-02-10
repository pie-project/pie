from __future__ import annotations

import math
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
# NATIVE FP8 SUPPORT
# =============================================================================

# FP8 configuration
FP8_DTYPE = torch.float8_e4m3fn
FP8_FINFO = torch.finfo(FP8_DTYPE)


def fp8_linear(input: torch.Tensor, 
               weight_fp8: torch.Tensor, 
               weight_scale: torch.Tensor,
               activation_scale: torch.Tensor | None = None,
               bias: torch.Tensor | None = None) -> torch.Tensor:
    """
    Native FP8 linear using torch._scaled_mm.
    
    Performs: output = input @ weight.T
    With FP8 quantization for both input and weight.
    
    Args:
        input: BF16 input tensor (*, in_features)
        weight_fp8: FP8 weight tensor (out_features, in_features), stored row-major
        weight_scale: Inverse scale for weight (scalar)
        activation_scale: Optional scale for input quantization. If None, compute dynamically.
        bias: Optional bias
        
    Returns:
        BF16 output tensor (*, out_features)
    """
    # Handle batched input - flatten to 2D
    orig_shape = input.shape
    input_2d = input.view(-1, orig_shape[-1])
    
    # Quantize input to FP8
    if activation_scale is not None:
        # Use provided activation scale (inverse scale)
        inv_act_scale = activation_scale.reciprocal()
        input_scaled = (input_2d * inv_act_scale).clamp(FP8_FINFO.min, FP8_FINFO.max)
        input_fp8 = input_scaled.to(FP8_DTYPE)
        act_scale_for_mm = activation_scale.to(torch.float32)
    else:
        # Dynamic quantization
        input_max = input_2d.abs().max().clamp(min=1e-12)
        scale = FP8_FINFO.max / input_max
        input_fp8 = (input_2d * scale).clamp(FP8_FINFO.min, FP8_FINFO.max).to(FP8_DTYPE)
        act_scale_for_mm = scale.reciprocal().to(torch.float32)
    
    # FP8 matmul: input @ weight.T
    # cuBLAS needs: row-major A @ column-major B
    # weight_fp8 is (out, in) row-major, so weight_fp8.t() is (in, out) col-major
    output = torch._scaled_mm(
        input_fp8,              # (batch, in_features) row-major
        weight_fp8.t(),         # (in_features, out_features) column-major view
        scale_a=act_scale_for_mm,
        scale_b=weight_scale.to(torch.float32),
        out_dtype=torch.bfloat16
    )
    
    if bias is not None:
        output = output + bias
    
    # Restore original shape
    output = output.view(*orig_shape[:-1], -1)
    
    return output




def _fuse_fp8_qkv(tensors: list[torch.Tensor], kwargs: dict) -> torch.Tensor:
    """Fuse Q, K, V FP8 weights into single tensor."""
    q_w, k_w, v_w = tensors[0], tensors[1], tensors[2]
    device = kwargs.get("device", "cuda")
    return torch.cat([q_w.to(device), k_w.to(device), v_w.to(device)], dim=0).contiguous()


def _fuse_fp8_gate_up(tensors: list[torch.Tensor], kwargs: dict) -> torch.Tensor:
    """Fuse gate and up FP8 weights into single tensor."""
    gate_w, up_w = tensors[0], tensors[1]
    device = kwargs.get("device", "cuda")
    return torch.cat([gate_w.to(device), up_w.to(device)], dim=0).contiguous()


def _max_scale(tensors: list[torch.Tensor], kwargs: dict) -> torch.Tensor:
    """Return the maximum scale (most conservative for quantization)."""
    device = kwargs.get("device", "cuda")
    scales = torch.stack([t.to(device) for t in tensors])
    return scales.max()


def _to_device(tensors: list[torch.Tensor], kwargs: dict) -> torch.Tensor:
    """Simple pass-through to device."""
    device = kwargs.get("device", "cuda")
    return tensors[0].to(device)


# Global permutation cache (shared across all weight loading)
_permutation_cache: dict = {}


def create_schema(config: "ModelConfig") -> Schema:
    """
    Create weight schema for Ministral 3B with native FP8 support.
    
    Loads FP8 weights and their scales as SEPARATE tensor keys.
    This allows the loader to handle each tensor individually.
    """
    prefix = "language_model."
    
    schema = (
        Schema("mistral3")
        # === FP8 Weights (stored separately from scales) ===
        # Fused QKV weight (FP8)
        .define(
            "layers.*.proj_qkv",
            Source.gather([
                f"{prefix}model.layers.*.self_attn.q_proj.weight",
                f"{prefix}model.layers.*.self_attn.k_proj.weight",
                f"{prefix}model.layers.*.self_attn.v_proj.weight",
            ]).transform(_fuse_fp8_qkv),
        )
        # Fused QKV weight scale
        .define(
            "layers.*.proj_qkv_scale",
            Source.gather([
                f"{prefix}model.layers.*.self_attn.q_proj.weight_scale_inv",
                f"{prefix}model.layers.*.self_attn.k_proj.weight_scale_inv",
                f"{prefix}model.layers.*.self_attn.v_proj.weight_scale_inv",
            ]).transform(_max_scale),
        )
        # QKV activation scale
        .define(
            "layers.*.act_scale_qkv",
            Source.gather([
                f"{prefix}model.layers.*.self_attn.q_proj.activation_scale",
                f"{prefix}model.layers.*.self_attn.k_proj.activation_scale",
                f"{prefix}model.layers.*.self_attn.v_proj.activation_scale",
            ]).transform(_max_scale),
        )
        # Output projection weight (FP8)
        .define(
            "layers.*.proj_o",
            Source.gather([
                f"{prefix}model.layers.*.self_attn.o_proj.weight",
            ]).transform(_to_device),
        )
        .define(
            "layers.*.proj_o_scale",
            Source.gather([
                f"{prefix}model.layers.*.self_attn.o_proj.weight_scale_inv",
            ]).transform(_to_device),
        )
        .define(
            "layers.*.act_scale_o",
            Source.gather([
                f"{prefix}model.layers.*.self_attn.o_proj.activation_scale",
            ]).transform(_to_device),
        )
        # Fused gate+up weight (FP8)
        .define(
            "layers.*.proj_gate_up",
            Source.gather([
                f"{prefix}model.layers.*.mlp.gate_proj.weight",
                f"{prefix}model.layers.*.mlp.up_proj.weight",
            ]).transform(_fuse_fp8_gate_up),
        )
        .define(
            "layers.*.proj_gate_up_scale",
            Source.gather([
                f"{prefix}model.layers.*.mlp.gate_proj.weight_scale_inv",
                f"{prefix}model.layers.*.mlp.up_proj.weight_scale_inv",
            ]).transform(_max_scale),
        )
        .define(
            "layers.*.act_scale_gate_up",
            Source.gather([
                f"{prefix}model.layers.*.mlp.gate_proj.activation_scale",
                f"{prefix}model.layers.*.mlp.up_proj.activation_scale",
            ]).transform(_max_scale),
        )
        # Down projection weight (FP8)
        .define(
            "layers.*.proj_down",
            Source.gather([
                f"{prefix}model.layers.*.mlp.down_proj.weight",
            ]).transform(_to_device),
        )
        .define(
            "layers.*.proj_down_scale",
            Source.gather([
                f"{prefix}model.layers.*.mlp.down_proj.weight_scale_inv",
            ]).transform(_to_device),
        )
        .define(
            "layers.*.act_scale_down",
            Source.gather([
                f"{prefix}model.layers.*.mlp.down_proj.activation_scale",
            ]).transform(_to_device),
        )
        # === Non-FP8 tensors ===
        # Per-layer norms
        .define(
            "layers.*.norm_attn",
            Source(f"{prefix}model.layers.*.input_layernorm.weight"),
        )
        .define(
            "layers.*.norm_mlp",
            Source(f"{prefix}model.layers.*.post_attention_layernorm.weight"),
        )
        # Embedding - large tensor LAST
        .define(
            "embed_token",
            Source(f"{prefix}model.embed_tokens.weight").shard("row"),
        )
        # Final layer norm
        .define(
            "norm_last",
            Source(f"{prefix}model.norm.weight"),
        )
    )

    # Handle untied embeddings (lm_head separate from embed_token)
    if not config.tie_word_embeddings:
        schema.define(
            "lm_head",
            Source(f"{prefix}lm_head.weight").shard("row"),
        )

    return schema





@dataclass
class ModelConfig(ModelConfigBase):
    """
    Mistral3-specific model architecture configuration.
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
    tie_word_embeddings: bool
    sliding_window: int | None
    
    # YARN RoPE scaling parameters
    rope_scaling_factor: float = 1.0
    rope_original_max_position: int = 16384
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0

    @classmethod
    def from_dict(cls, spec: dict[str, Any]) -> "ModelConfig":
        # Handle nested text_config if present (common in multimodal models like Pixtral/Ministral)
        if "text_config" in spec:
            spec = spec["text_config"]
            
        theta = 10000.0
        if "rope_theta" in spec:
            theta = float(spec["rope_theta"])
        
        # Determine head dim
        if "head_dim" in spec:
            head_dim = int(spec["head_dim"])
        else:
            head_dim = int(spec["hidden_size"]) // int(spec["num_attention_heads"])

        # Handle RoPE parameters
        rope = spec.get("rope_parameters") or {}
        # Ministral 3 uses 'yarn' with 'rope_theta' inside parameters
        # If rope_parameters is present, use theta from there, else try simple rope_theta
        theta = float(rope.get("rope_theta", spec.get("rope_theta", 10000.0)))
        

        
        # YARN scaling parameters
        rope_scaling_factor = float(rope.get("factor", 1.0))
        rope_original_max_position = int(rope.get("original_max_position_embeddings", 16384))
        rope_beta_fast = float(rope.get("beta_fast", 32.0))
        rope_beta_slow = float(rope.get("beta_slow", 1.0))

        # Workaround: Ministral 3B config says 1M theta + YaRN, but short-context inference
        # fails with that configuration (garbage output).
        # Empirically, Standard RoPE with theta=10k works perfectly.
        if spec.get("model_type") == "ministral3" and theta > 100000.0:
             theta = 10000.0

        return ModelConfig(
            num_layers=int(spec["num_hidden_layers"]),
            num_q_heads=int(spec["num_attention_heads"]),
            num_kv_heads=int(spec["num_key_value_heads"]),
            dim_head=head_dim,
            dim_hidden=int(spec["hidden_size"]),
            dim_mlp=int(spec["intermediate_size"]),
            num_vocabs=int(spec["vocab_size"]),
            rms_norm_eps=float(spec["rms_norm_eps"]),
            rope_theta=theta,
            tie_word_embeddings=bool(spec.get("tie_word_embeddings", False)),
            sliding_window=spec.get("sliding_window"),
            rope_scaling_factor=rope_scaling_factor,
            rope_original_max_position=rope_original_max_position,
            rope_beta_fast=rope_beta_fast,
            rope_beta_slow=rope_beta_slow,
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


class ForwardPass:
    """
    Mistral3 forward pass implementation.
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
        
        # Pre-compute YARN RoPE cos/sin cache
        self._rope_cos_sin_cache = self._compute_rope_cache()

    def _fp8_linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        FP8 matrix multiplication using FlashInfer's optimized kernel.
        
        Args:
            input: Input tensor in bf16/fp16, shape (m, k)
            weight: Pre-processed FP8 weight from prepare_low_latency_gemm_weights
            weight_scale: Weight scale factor (weight_scale_inv from model)
            
        Returns:
            Output tensor in bf16, shape (m, n)
        """
        # Quantize activations to FP8
        x_abs_max = input.abs().max()
        # Use 224.0 as safe max for e4m3 (less than 448 to prevent saturation)
        fp8_max = 224.0
        scale = fp8_max / (x_abs_max + 1e-12)  # Avoid div by zero
        
        x_scaled = input * scale
        x_fp8 = x_scaled.to(torch.float8_e4m3fn)
        x_scale_inv = 1.0 / scale
        
        # Combined scale for output: x_scale_inv * weight_scale
        alpha = x_scale_inv * weight_scale
        
        # Call FlashInfer's FP8 matmul
        return ops.mm_fp8(
            x_fp8, weight, alpha,
            out_dtype=torch.bfloat16,
        )

    def _compute_rope_cache(self) -> torch.Tensor:
        """Pre-compute YARN RoPE cos/sin cache for all positions."""
        cfg = self.model_config
        device = self.runtime_config.device
        head_dim = cfg.dim_head
        max_position_id = 131072  # Max sequence length

        # Compute base frequencies
        freq = cfg.rope_theta ** (
            torch.arange(0, head_dim, 2, dtype=torch.float, device=device) / head_dim
        )

        if cfg.rope_scaling_factor > 1.0:
            # YaRN concentration
            concentration = 0.1 * math.log(cfg.rope_scaling_factor) + 1.0

            d_half = head_dim / 2
            # NTK by parts
            low = (
                d_half
                * math.log(
                    cfg.rope_original_max_position / (cfg.rope_beta_fast * 2 * math.pi)
                )
                / math.log(cfg.rope_theta)
            )
            high = (
                d_half
                * math.log(
                    cfg.rope_original_max_position / (cfg.rope_beta_slow * 2 * math.pi)
                )
                / math.log(cfg.rope_theta)
            )

            interpolation = 1.0 / (cfg.rope_scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (torch.arange(d_half, dtype=torch.float32, device=device) - low) / (
                high - low
            )
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq
            
        # Compute positions and frequencies
        position_ids = torch.arange(max_position_id, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", position_ids, inv_freq)

        # Compute cos/sin with concentration scaling
        cos_cache = freqs.cos() * concentration
        sin_cache = freqs.sin() * concentration

        # Concatenate for FlashInfer format: [max_pos, head_dim]
        cos_sin_cache = torch.cat([cos_cache, sin_cache], dim=-1)

        return cos_sin_cache.to(torch.float32)

    def warmup_cuda_graphs(self, kv_cache_at_layer: list[torch.Tensor]):
        """
        Pre-capture CUDA graphs for all defined bins using shared static buffers.
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
            dummy_indices = torch.full(
                (b,), self.scratch_page_idx, device=device, dtype=torch.int32
            )
            self.shared_kv_indices_buffer[:b].copy_(dummy_indices)

            pos_view = self.shared_static_position_ids[:b]
            pos_view.zero_()

            batch_indices_view = self.shared_static_batch_indices[:b]
            batch_indices_view.copy_(torch.arange(b, device=device, dtype=torch.int32))

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
        """Embed input tokens into hidden states."""
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
        """Embed token IDs with Tensor Parallel support (Column Parallel)."""
        if self.tp_size == 1:
            return fun.embedding(token_ids, self.weights.get("embed_token"))

        # Column-parallel embedding
        local_embeds = fun.embedding(token_ids, self.weights.get("embed_token"))
        gathered_list = [torch.empty_like(local_embeds) for _ in range(self.tp_size)]
        dist.all_gather(gathered_list, local_embeds, group=self.compute_process_group)
        full_embeds = torch.cat(gathered_list, dim=-1)
        return full_embeds

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits."""
        # Apply final layer norm
        normed = fun.rms_norm(
            hidden_states,
            normalized_shape=[self.model_config.dim_hidden],
            weight=self.weights.get("norm_last"),
            eps=self.model_config.rms_norm_eps,
        )

        if self.tp_size == 1:
            weight = (
                self.weights.get("embed_token")
                if self.model_config.tie_word_embeddings
                else self.weights.get("lm_head")
            )
            return fun.linear(normed, weight)

        # Multi-GPU: Column-parallel projection
        hidden_per_rank = self.model_config.dim_hidden // self.tp_size
        start_idx = self.tp_rank * hidden_per_rank
        end_idx = start_idx + hidden_per_rank
        local_normed = normed[:, start_idx:end_idx]

        weight = (
            self.weights.get("embed_token")
            if self.model_config.tie_word_embeddings
            else self.weights.get("lm_head")
        )
        local_logits = fun.linear(local_normed, weight)

        dist.all_reduce(local_logits, group=self.compute_process_group)

        return local_logits

    def mlp(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Executes the MLP block for a single layer using native FP8."""
        local_mlp_size = self.model_config.dim_mlp // self.tp_size
        residual = hidden_states

        # 1. MLP RMSNorm
        normed_input = fun.rms_norm(
            hidden_states,
            normalized_shape=[self.model_config.dim_hidden],
            weight=self.weights.get(f"layers.{layer_idx}.norm_mlp"),
            eps=self.model_config.rms_norm_eps,
        )

        # 2. Gate+Up Projection (Column Parallel) - FP8
        gate_up_weight = self.weights.get(f"layers.{layer_idx}.proj_gate_up")
        gate_up_scale = self.weights.get(f"layers.{layer_idx}.proj_gate_up_scale")
        gate_up_act_scale = self.weights.get(f"layers.{layer_idx}.act_scale_gate_up")
        gate_up = fp8_linear(normed_input, gate_up_weight, gate_up_scale, gate_up_act_scale)

        # Split gate and up
        gate, up = torch.split(gate_up, [local_mlp_size, local_mlp_size], dim=-1)

        # 3. SiLU activation * gate (SwiGLU)
        hidden = fun.silu(gate) * up

        # 4. Down Projection (Row Parallel) - FP8
        down_weight = self.weights.get(f"layers.{layer_idx}.proj_down")
        down_scale = self.weights.get(f"layers.{layer_idx}.proj_down_scale")
        down_act_scale = self.weights.get(f"layers.{layer_idx}.act_scale_down")
        down = fp8_linear(hidden, down_weight, down_scale, down_act_scale)
        del hidden, gate, up, gate_up

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
        """Executes the attention block for a single layer using native FP8."""
        local_num_query_heads = self.model_config.num_q_heads // self.tp_size
        local_num_key_value_heads = self.model_config.num_kv_heads // self.tp_size
        local_q_size = local_num_query_heads * self.model_config.dim_head
        local_kv_size = local_num_key_value_heads * self.model_config.dim_head

        n = hidden_states.size(0)
        residual = hidden_states

        # 1. Input RMSNorm
        normed_input = fun.rms_norm(
            hidden_states,
            normalized_shape=[self.model_config.dim_hidden],
            weight=self.weights.get(f"layers.{layer_idx}.norm_attn"),
            eps=self.model_config.rms_norm_eps,
        )

        # 2. QKV Projection (Column Parallel) - FP8
        qkv_weight = self.weights.get(f"layers.{layer_idx}.proj_qkv")
        qkv_scale = self.weights.get(f"layers.{layer_idx}.proj_qkv_scale")
        qkv_act_scale = self.weights.get(f"layers.{layer_idx}.act_scale_qkv")
        qkv_proj = fp8_linear(normed_input, qkv_weight, qkv_scale, qkv_act_scale)

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

        # 5. Apply YaRN RoPE using pre-computed cos/sin cache
        ops.apply_rope_with_cos_sin_cache_inplace(
            positions=position_ids.to(torch.int32),
            query=q,
            key=k,
            head_size=self.model_config.dim_head,
            cos_sin_cache=self._rope_cos_sin_cache,
            is_neox=True,
        )

        # 6. Append K, V to cache
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

        # 7. Compute Attention
        attn_output = wrapper.run(q, kv_cache_layer)
        del q, k, v
        del qkv_proj

        attn_output = attn_output.reshape(n, -1)

        # 8. Output Projection - FP8
        o_weight = self.weights.get(f"layers.{layer_idx}.proj_o")
        o_scale = self.weights.get(f"layers.{layer_idx}.proj_o_scale")
        o_act_scale = self.weights.get(f"layers.{layer_idx}.act_scale_o")
        attn_proj = fp8_linear(attn_output, o_weight, o_scale, o_act_scale)
        del attn_output

        if self.tp_size > 1:
            dist.all_reduce(attn_proj, group=self.compute_process_group)

        # 9. Residual Connection
        return residual + attn_proj

    def transform(
        self,
        # inputs
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        qo_indptr: torch.Tensor,
        # kv cache
        kv_cache_at_layer: list[torch.Tensor],
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

        if self.runtime_config.device.type == "cuda":
            torch.cuda.set_device(self.runtime_config.device)

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
        del seq_lens

        if single_token_inference_mode:
            # For standard execution (fallback) we need to plan the fallback wrapper
            wrapper = self.wrapper_decode_fallback
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

        # Execute layers
        if single_token_inference_mode and self.use_cuda_graphs and not adapter_subpass:
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
                runtime_config.max_num_kv_pages + 1,
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
