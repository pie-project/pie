"""GPT OSS Large Language Model Architecture"""

from __future__ import annotations
from typing import List, Dict

import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

import flashinfer as ops
from adapter import AdapterSubpass

from config.gptoss import GPTOSSArch
from config.common import TensorLoader
from .utils import convert_mxfp4_to_bf16, extract_kv_from_paged_cache

VERSION = "0.1.0"


def create_fusion_map(model: nn.Module):
    """
    Analyzes the model and creates a map for fusing weights and handling MXFP4 tensors.

    Returns:
        A dictionary mapping {
            fused_tensor_name: {"sources": [source_names], "dim": cat_dim, "type": type}
        }.
        For MXFP4 tensors, type is "mxfp4" and sources contains [blocks_name, scales_name].
        For fusion tensors, type is "fusion" and sources contains the tensors to concatenate.
        For regular tensors, type is "regular" and sources contains the single tensor name.
    """
    fusion_map = {}
    for name, module in model.named_modules():
        # --- Rule for GPTOSSAttention QKV Fusion ---
        if isinstance(module, GPTOSSAttention):
            # Handle weights
            target_w = f"{name}.qkv_proj.weight"
            sources_w = [
                f"{name}.q_proj.weight",
                f"{name}.k_proj.weight",
                f"{name}.v_proj.weight",
            ]
            fusion_map[target_w] = {"sources": sources_w, "dim": 0, "type": "fusion"}

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
                    "type": "fusion",
                }

        # --- Rule for GPTOSSExperts MXFP4 Weights ---
        elif isinstance(module, GPTOSSExperts):
            # Handle gate_up_proj weights (MXFP4 format)
            target_gate_up = f"{name}.gate_up_proj"
            blocks_gate_up = f"{name}.gate_up_proj_blocks"
            scales_gate_up = f"{name}.gate_up_proj_scales"
            fusion_map[target_gate_up] = {
                "sources": [blocks_gate_up, scales_gate_up],
                "dim": None,
                "type": "mxfp4",
            }

            # Handle down_proj weights (MXFP4 format)
            target_down = f"{name}.down_proj"
            blocks_down = f"{name}.down_proj_blocks"
            scales_down = f"{name}.down_proj_scales"
            fusion_map[target_down] = {
                "sources": [blocks_down, scales_down],
                "dim": None,
                "type": "mxfp4",
            }

            # Biases are regular tensors (not MXFP4)
            # No special handling needed for biases

    return fusion_map


class GPTOSSTensorLoader(TensorLoader):
    """
    TensorLoader implementation for GPT-OSS models.

    Handles fusion of QKV projections and conversion of MXFP4 MLP weights
    to bfloat16 format based on the model architecture.
    """

    def __init__(self, model: GPTOSSForCausalLM):
        """
        Initialize the tensor loader with a model instance.

        Args:
            model: The GPT-OSS model instance
        """
        self.model = model
        self._fusion_map = create_fusion_map(model)

        # Create reverse mapping for quick lookup
        self._source_to_target = {
            source: target
            for target, details in self._fusion_map.items()
            for source in details["sources"]
        }

    def query(self, runtime_tensor_name: str) -> List[str]:
        """
        Query which checkpoint tensors are needed for a given runtime tensor.

        Args:
            runtime_tensor_name: Name of the tensor in the runtime model

        Returns:
            List of checkpoint tensor names needed to construct the runtime tensor
        """
        if runtime_tensor_name in self._fusion_map:
            # This tensor requires special handling (fusion or MXFP4 conversion)
            return self._fusion_map[runtime_tensor_name]["sources"]
        else:
            # This is a regular tensor, return itself
            return [runtime_tensor_name]

    def load(
        self, runtime_tensor_name: str, checkpoint_tensors: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Load and transform checkpoint tensors into the runtime tensor.

        Args:
            runtime_tensor_name: Name of the tensor in the runtime model
            checkpoint_tensors: Dictionary mapping checkpoint tensor names to their loaded tensors

        Returns:
            The constructed runtime tensor
        """
        if runtime_tensor_name in self._fusion_map:
            fusion_info = self._fusion_map[runtime_tensor_name]
            tensor_type = fusion_info["type"]
            source_names = fusion_info["sources"]

            if tensor_type == "fusion":
                # Handle QKV fusion - concatenate the source tensors
                concat_dim = fusion_info["dim"]
                source_tensors = [checkpoint_tensors[name] for name in source_names]
                return torch.cat(source_tensors, dim=concat_dim)

            elif tensor_type == "mxfp4":
                # Handle MXFP4 conversion - convert blocks and scales to bfloat16
                if len(source_names) != 2:
                    raise ValueError(
                        f"MXFP4 tensor {runtime_tensor_name} must have exactly 2 sources "
                        f"(blocks and scales), got {len(source_names)}: {source_names}"
                    )

                blocks_name, scales_name = source_names
                if blocks_name not in checkpoint_tensors:
                    raise KeyError(
                        f"Blocks tensor '{blocks_name}' not found in checkpoint tensors"
                    )
                if scales_name not in checkpoint_tensors:
                    raise KeyError(
                        f"Scales tensor '{scales_name}' not found in checkpoint tensors"
                    )

                blocks = checkpoint_tensors[blocks_name]
                scales = checkpoint_tensors[scales_name]

                # Convert MXFP4 to bfloat16 using our conversion function
                return convert_mxfp4_to_bf16(
                    blocks,
                    scales,
                    dtype=torch.bfloat16,
                    target_device=self.model.config.device,
                )

            else:
                raise ValueError(f"Unknown tensor type: {tensor_type}")
        else:
            # This is a regular tensor, return it directly
            if runtime_tensor_name not in checkpoint_tensors:
                raise KeyError(
                    f"Tensor '{runtime_tensor_name}' not found in checkpoint tensors"
                )
            return checkpoint_tensors[runtime_tensor_name]


class GPTOSSRMSNorm(nn.Module):
    """RMS Normalization layer."""

    def __init__(
        self, num_features: int, eps: float = 1e-05, device: torch.device | None = None
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(num_features, device=device, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the RMSNorm layer."""
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.weight).to(dtype)


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embedding to input tensor."""
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


class GPTOSSRotaryEmbedding(nn.Module):
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

        # Cache for cos/sin values to avoid recomputation
        self._cos_sin_cache = {}
        self._max_cached_position = 0

        # Pre-compute and cache concentration and inv_freq since they're constant
        self._concentration, self._inv_freq = self._compute_concentration_and_inv_freq()

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

    def _get_cos_sin_for_positions(self, position_ids: torch.Tensor):
        """Get cosine and sine values for specific position IDs with caching."""
        # Convert position_ids to a hashable key for caching
        position_ids = position_ids.to(dtype=torch.int64, device=self.device)
        max_pos = position_ids.max().item()

        # Check if we need to extend our cache
        if max_pos > self._max_cached_position:
            # Extend cache to cover new positions
            new_positions = torch.arange(
                self._max_cached_position + 1,
                max_pos + 1,
                dtype=torch.float32,
                device=self.device,
            )

            if len(new_positions) > 0:
                freqs = torch.einsum("i,j->ij", new_positions, self._inv_freq)
                cos_new = freqs.cos() * self._concentration
                sin_new = freqs.sin() * self._concentration

                # Store in cache
                for i, pos in enumerate(new_positions):
                    pos_key = int(pos.item())
                    self._cos_sin_cache[pos_key] = (cos_new[i], sin_new[i])

                self._max_cached_position = max_pos

        # Retrieve cached values for the requested positions
        cos_list = []
        sin_list = []
        for pos in position_ids:
            pos_key = pos.item()
            if pos_key in self._cos_sin_cache:
                cos_val, sin_val = self._cos_sin_cache[pos_key]
                cos_list.append(cos_val)
                sin_list.append(sin_val)
            else:
                # Fallback for position 0 or negative positions
                pos_float = torch.tensor(
                    pos_key, dtype=torch.float32, device=self.device
                )
                freqs = torch.einsum("j->j", pos_float * self._inv_freq)
                cos_val = freqs.cos() * self._concentration
                sin_val = freqs.sin() * self._concentration
                cos_list.append(cos_val)
                sin_list.append(sin_val)

        cos = torch.stack(cos_list, dim=0)
        sin = torch.stack(sin_list, dim=0)
        return cos, sin

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embedding to query and key tensors using position IDs."""
        cos, sin = self._get_cos_sin_for_positions(position_ids)

        query_shape = query.shape
        query = query.view(position_ids.shape[0], -1, self.head_dim)
        query = _apply_rotary_emb(query, cos, sin)
        query = query.reshape(query_shape)

        key_shape = key.shape
        key = key.view(position_ids.shape[0], -1, self.head_dim)
        key = _apply_rotary_emb(key, cos, sin)
        key = key.reshape(key_shape)
        return query, key


def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    """SwiGLU activation function with clamping."""
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)


class GPTOSSAttention(nn.Module):
    """GPT OSS attention module with FlashInfer support and sink tokens."""

    def __init__(self, config: GPTOSSArch, layer_idx: int):
        """Initialize the GPT OSS attention module."""
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_size
        self.num_attention_heads = config.num_query_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        # Apply sliding window to even layers and full attention to odd layers
        # This follows the GPT-OSS alternating attention pattern
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0

        # Define the output sizes for Q, K, and V for clarity
        self.q_size = config.num_query_heads * config.head_size
        self.k_size = config.num_key_value_heads * config.head_size
        self.v_size = config.num_key_value_heads * config.head_size

        # Sink tokens parameter
        self.sinks = nn.Parameter(
            torch.empty(
                config.num_query_heads,
                device=torch.device(config.device),
                dtype=config.dtype,
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

        self.rope = GPTOSSRotaryEmbedding(
            config.head_size,
            int(config.rope_theta),
            torch.float32,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            device=torch.device(config.device),
        )

    def _pytorch_attention_with_sinks(
        self,
        query: torch.Tensor,  # [n, num_q_heads, head_dim]
        key: torch.Tensor,  # [seq_len, num_kv_heads, head_dim] or [batch_size * seq_len, num_kv_heads, head_dim]
        value: torch.Tensor,  # [seq_len, num_kv_heads, head_dim] or [batch_size * seq_len, num_kv_heads, head_dim]
        position_ids: torch.Tensor,  # [n]
        use_sliding_window: bool = False,
        num_batches: int = 1,
    ) -> torch.Tensor:
        """
        Compute attention using PyTorch with attention sinks support.
        Based on the reference implementation.

        Args:
            query: Query tensor [n, num_q_heads, head_dim]
            key: Key tensor [seq_len, num_kv_heads, head_dim]
            value: Value tensor [seq_len, num_kv_heads, head_dim]
            position_ids: Position IDs for current tokens [n]
            use_sliding_window: Whether to apply sliding window attention

        Returns:
            Attention output [n, num_q_heads, head_dim]
        """
        n, num_q_heads, head_dim = query.shape
        seq_len, num_kv_heads, _ = key.shape

        # Determine batch size from the input tensors
        # Current format: query [n, num_q_heads, head_dim], key/value [seq_len, num_kv_heads, head_dim]
        # Target format: [batch, num_heads, seq_len, head_dim]

        # Handle all batch sizes uniformly
        # key and value are [batch_size * seq_len, num_kv_heads, head_dim]
        # We need to reshape them to [batch_size, num_kv_heads, seq_len, head_dim]
        total_seq_len = key.shape[0]
        seq_len_per_batch = total_seq_len // num_batches

        # Reshape key and value to proper batch format
        key = key.reshape(num_batches, seq_len_per_batch, num_kv_heads, head_dim)
        key = key.transpose(1, 2)  # [batch_size, num_kv_heads, seq_len, head_dim]

        value = value.reshape(num_batches, seq_len_per_batch, num_kv_heads, head_dim)
        value = value.transpose(1, 2)  # [batch_size, num_kv_heads, seq_len, head_dim]

        # For query, we assume it's distributed across batches
        # query is [n, num_q_heads, head_dim] where n might be batch_size * tokens_per_batch
        tokens_per_batch = n // num_batches
        query = query.reshape(num_batches, tokens_per_batch, num_q_heads, head_dim)
        query = query.transpose(
            1, 2
        )  # [batch_size, num_q_heads, tokens_per_batch, head_dim]

        batch_size = num_batches
        seq_len = seq_len_per_batch
        n = tokens_per_batch

        # Handle grouped query attention by repeating KV heads - exactly like reference
        key_states = torch.repeat_interleave(
            key, self.num_key_value_groups, dim=1
        )  # [batch_size, num_q_heads, seq_len, head_dim]
        value_states = torch.repeat_interleave(
            value, self.num_key_value_groups, dim=1
        )  # [batch_size, num_q_heads, seq_len, head_dim]

        # Compute attention weights - exactly like reference
        attn_weights = (
            torch.matmul(query, key_states.transpose(2, 3)) * self.scaling
        )  # [batch_size, num_q_heads, n, seq_len]

        # Apply sliding window attention if needed (critical for GPT OSS alternating attention)
        attention_mask = None
        if use_sliding_window and self.sliding_window > 0 and seq_len > 0:
            # Create sliding window mask
            # For each query position, mask out keys that are too far in the past
            query_positions = position_ids.unsqueeze(1)  # [n, 1]

            # Key positions in cache: they have absolute positions from 0 to seq_len-1
            # But we need to know their actual positions relative to the current sequence
            # The cached keys represent positions from (current_pos - seq_len) to (current_pos - 1)
            current_pos = (
                position_ids.max().item()
            )  # Current position of the last query token
            key_start_pos = (
                current_pos - seq_len + 1
            )  # Starting position of first cached key
            key_positions = torch.arange(
                key_start_pos, current_pos + 1, device=query.device
            )[:seq_len].unsqueeze(
                0
            )  # [1, seq_len]

            # Sliding window: can only attend to keys within sliding_window distance
            distances = query_positions - key_positions  # [n, seq_len]
            sliding_mask = (
                distances > self.sliding_window
            )  # [n, seq_len] - True where we should mask

            # Convert to attention mask format
            sliding_mask = sliding_mask.masked_fill(sliding_mask, float("-inf"))
            attention_mask = sliding_mask.unsqueeze(0).unsqueeze(
                0
            )  # [1, 1, n, seq_len]

        # Apply attention mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Add attention sinks - exactly like reference
        sinks = self.sinks.reshape(1, -1, 1, 1).expand(
            query.shape[0], -1, query.shape[-2], -1
        )  # [batch_size, num_q_heads, n, 1]
        combined_logits = torch.cat(
            [attn_weights, sinks], dim=-1
        )  # [batch_size, num_q_heads, n, seq_len + 1]

        # Prevent overflow - exactly like reference
        combined_logits = (
            combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        )

        # Apply softmax - exactly like reference
        probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)

        # Drop sinks - exactly like reference
        scores = probs[..., :-1]  # [batch_size, num_q_heads, n, seq_len]

        # Apply dropout - exactly like reference
        attn_weights = nn.functional.dropout(scores, p=0.0, training=self.training)

        # Compute attention output - exactly like reference
        attn_output = torch.matmul(
            attn_weights, value_states
        )  # [batch_size, num_q_heads, n, head_dim]

        # Reshape back - handle all batch sizes uniformly
        attn_output = attn_output.transpose(
            1, 2
        ).contiguous()  # [batch_size, n, num_q_heads, head_dim]

        # Reshape back to concatenated format for compatibility
        original_n = (
            attn_output.shape[0] * attn_output.shape[1]
        )  # batch_size * tokens_per_batch
        attn_output = attn_output.reshape(
            original_n, num_q_heads, head_dim
        )  # [original_n, num_q_heads, head_dim]

        return attn_output

    def forward(
        self,
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

        # Extract all KV cache (including newly added tokens) for PyTorch attention
        # Handle all batch sizes uniformly
        num_batches = len(kv_last_page_lens)

        # Extract each batch and find max length
        all_keys = []
        all_values = []
        max_seq_len = 0

        for batch_idx in range(num_batches):
            keys, values = extract_kv_from_paged_cache(
                kv_cache_at_layer[self.layer_idx],
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_lens,
                batch_idx=batch_idx,
                num_key_value_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
            )
            all_keys.append(keys)
            all_values.append(values)
            max_seq_len = max(max_seq_len, keys.shape[0])

        # Pad all sequences to max length
        padded_keys = []
        padded_values = []
        for keys, values in zip(all_keys, all_values):
            if keys.shape[0] < max_seq_len:
                # Pad with zeros
                pad_len = max_seq_len - keys.shape[0]
                key_padding = torch.zeros(
                    pad_len,
                    keys.shape[1],
                    keys.shape[2],
                    device=keys.device,
                    dtype=keys.dtype,
                )
                value_padding = torch.zeros(
                    pad_len,
                    values.shape[1],
                    values.shape[2],
                    device=values.device,
                    dtype=values.dtype,
                )
                keys = torch.cat([keys, key_padding], dim=0)
                values = torch.cat([values, value_padding], dim=0)
            padded_keys.append(keys)
            padded_values.append(values)

        # Stack into batch dimension: [batch_size, seq_len, num_kv_heads, head_dim]
        cached_keys = torch.stack(padded_keys, dim=0)
        cached_values = torch.stack(padded_values, dim=0)

        # Reshape to [batch_size * seq_len, num_kv_heads, head_dim] for compatibility
        batch_size, seq_len = cached_keys.shape[:2]
        cached_keys = cached_keys.reshape(
            batch_size * seq_len, cached_keys.shape[2], cached_keys.shape[3]
        )
        cached_values = cached_values.reshape(
            batch_size * seq_len, cached_values.shape[2], cached_values.shape[3]
        )

        # Determine if we should use sliding window attention
        # Even layers use sliding window, odd layers use full attention
        use_sliding_window = self.layer_idx % 2 == 0

        # Compute attention using PyTorch with attention sinks
        # Pass the number of batches for proper handling
        attn_output = self._pytorch_attention_with_sinks(
            query_states,
            cached_keys,
            cached_values,
            position_ids,
            use_sliding_window=use_sliding_window,
            num_batches=num_batches,
        )

        attn_output = attn_output.reshape(n, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output


class GPTOSSRouter(nn.Module):
    """GPT OSS Router for selecting top-k experts."""

    def __init__(self, config: GPTOSSArch):
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

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the router."""
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = torch.nn.functional.linear(  # pylint: disable=not-callable
            hidden_states, self.weight, self.bias
        )
        router_top_value, router_indices = torch.topk(
            router_logits, self.experts_per_token, dim=-1, sorted=True
        )
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1)
        router_scores = torch.zeros_like(router_logits).scatter_(
            1, router_indices, router_top_value
        )
        return router_scores, router_indices


class GPTOSSExperts(nn.Module):
    """GPT OSS Experts layer containing the actual expert parameters."""

    def __init__(self, config: GPTOSSArch):
        """Initialize the GPT OSS Experts layer."""
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.swiglu_limit = config.swiglu_limit
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        assert config.intermediate_size % self.world_size == 0
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.intermediate_size * 2 // self.world_size,
                    config.hidden_size,
                ),
                device=torch.device(config.device),
                dtype=config.dtype,
            )
        )
        self.gate_up_proj_bias = nn.Parameter(
            torch.empty(
                (config.num_experts, config.intermediate_size * 2 // self.world_size),
                device=torch.device(config.device),
                dtype=config.dtype,
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.hidden_size,
                    config.intermediate_size // self.world_size,
                ),
                device=torch.device(config.device),
                dtype=config.dtype,
            )
        )
        self.down_proj_bias = nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                device=torch.device(config.device),
                dtype=config.dtype,
            )
        )

    def forward(self, t: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """Forward pass through the experts."""
        # Gate and Up projection
        gate_up_proj = self.gate_up_proj[expert_indices, ...]
        gate_up_proj_bias = self.gate_up_proj_bias[expert_indices, ...]
        t = torch.einsum("beck,bk->bec", gate_up_proj, t) + gate_up_proj_bias
        t = swiglu(t, limit=self.swiglu_limit)

        # Down projection
        down_proj = self.down_proj[expert_indices, ...]
        down_proj_bias = self.down_proj_bias[expert_indices, ...]
        t = torch.einsum("beck,bek->bec", down_proj, t)
        if self.world_size > 1:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t += down_proj_bias

        return t


class GPTOSSMlp(nn.Module):
    """GPT OSS MLP layer with Mixture of Experts."""

    def __init__(self, config: GPTOSSArch):
        """Initialize the GPT OSS MLP layer."""
        super().__init__()
        self.config = config
        self.router = GPTOSSRouter(config)
        self.experts = GPTOSSExperts(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP layer."""
        # Router determines expert selection and weights
        router_scores, router_indices = self.router(x)

        # Extract the weights for selected experts
        expert_weights = torch.gather(router_scores, 1, router_indices)

        # Forward through experts
        t = self.experts(x, router_indices)

        # Weighted sum of experts
        t = torch.einsum("bec,be->bc", t, expert_weights)

        return t


class GPTOSSDecoderLayer(nn.Module):
    """GPT OSS decoder layer."""

    def __init__(self, config: GPTOSSArch, layer_idx: int):
        """Initialize the GPT OSS decoder layer."""
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = GPTOSSRMSNorm(
            config.hidden_size, device=torch.device(config.device)
        )
        self.self_attn = GPTOSSAttention(config, layer_idx)
        self.mlp = GPTOSSMlp(config)
        self.post_attention_layernorm = GPTOSSRMSNorm(
            config.hidden_size, device=torch.device(config.device)
        )

    def forward(
        self,
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

        # Self Attention
        hidden_states = self.self_attn(
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

        # MLP
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GPTOSSModel(nn.Module):
    """GPT OSS model with PyTorch attention and attention sinks."""

    def __init__(self, config: GPTOSSArch):
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
                GPTOSSDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_layers)
            ]
        )
        self.norm = GPTOSSRMSNorm(
            config.hidden_size,
            device=torch.device(config.device),
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
        hidden_states = input_embeds
        n, _ = hidden_states.size()

        page_size = kv_cache_at_layer[0].shape[2]

        batch_indices, batch_positions = ops.get_batch_indices_positions(
            append_indptr=qo_indptr,
            seq_lens=ops.get_seq_lens(kv_page_indptr, kv_last_page_lens, page_size),
            nnz=n,
        )

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
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


class GPTOSSForCausalLM(nn.Module):
    """GPT OSS model for causal language modeling."""

    def __init__(self, config: GPTOSSArch):
        """Initialize the GPT OSS causal LM model."""
        super().__init__()
        self.config = config
        self.model = GPTOSSModel(config)
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
