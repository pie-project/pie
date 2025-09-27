"""
Metal-based L4MA Model Implementation

This replaces the flashinfer-dependent model.forward() call with Metal kernels
while maintaining the same interface as the original L4MA model.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Any, Tuple

import torch
import numpy as np

from config.l4ma import L4maArch

# Import Metal backend
try:
    from metal_backend import MetalBackend
    METAL_BACKEND_AVAILABLE = True
except ImportError:
    METAL_BACKEND_AVAILABLE = False


class MetalL4maModel:
    """
    Metal-based L4MA model that replaces flashinfer with Metal kernels.

    This maintains the same interface as the original PyTorch model
    but uses Metal kernels for all computations.
    """

    def __init__(self, config: L4maArch):
        """Initialize Metal L4MA model."""
        self.config = config

        if not METAL_BACKEND_AVAILABLE:
            raise RuntimeError("Metal backend not available")

        # Initialize Metal backend with model configuration
        self.metal_backend = MetalBackend(model_metadata={
            'architecture': {
                'num_query_heads': config.num_query_heads,
                'num_key_value_heads': config.num_key_value_heads,
                'head_size': config.head_size,
                'hidden_size': config.hidden_size,
                'vocab_size': config.vocab_size,
                'num_layers': config.num_layers,
                'intermediate_size': config.intermediate_size,
            }
        })

        if not self.metal_backend.initialize():
            raise RuntimeError("Failed to initialize Metal backend")

        # Load model weights (this would come from actual model file)
        self._initialize_weights()

        print(f"✅ MetalL4maModel initialized")
        print(f"   Device: {self.metal_backend.get_capabilities()['device_info']}")
        print(f"   Layers: {config.num_layers}")
        print(f"   Hidden size: {config.hidden_size}")

    def _initialize_weights(self):
        """Initialize model weights. In production, load from model file."""
        # For testing, create random weights matching the architecture
        np.random.seed(42)  # Reproducible

        config = self.config

        # Embedding weights
        self.embed_tokens_weight = np.random.randn(
            config.vocab_size, config.hidden_size
        ).astype(np.float32) * 0.1

        # Layer weights for each transformer layer
        self.layer_weights = []
        for _ in range(config.num_layers):
            layer_weights = {
                'input_layernorm_weight': np.ones(config.hidden_size, dtype=np.float32),
                'post_attention_layernorm_weight': np.ones(config.hidden_size, dtype=np.float32),
                # QKV and MLP weights would be loaded here
                # For now, Metal kernels will handle the computation
            }
            self.layer_weights.append(layer_weights)

        # Final layer norm and LM head
        self.norm_weight = np.ones(config.hidden_size, dtype=np.float32)
        self.lm_head_weight = np.random.randn(
            config.vocab_size, config.hidden_size
        ).astype(np.float32) * 0.1

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Token embedding using Metal kernels.

        Args:
            input_ids: Input token IDs tensor of shape (batch_size, seq_len) or (seq_len,)

        Returns:
            Embedded tokens tensor of shape (batch_size, seq_len, hidden_size) or (seq_len, hidden_size)
        """
        # Convert to numpy for Metal backend processing
        if input_ids.dim() == 1:
            # Single sequence: (seq_len,)
            token_ids_np = input_ids.cpu().numpy().astype(np.int32)
            batch_size = 1
            seq_len = len(token_ids_np)
        else:
            # Batch: (batch_size, seq_len)
            token_ids_np = input_ids.cpu().numpy().astype(np.int32)
            batch_size, seq_len = token_ids_np.shape

        # Use Metal backend for embedding lookup
        try:
            result = self.metal_backend.run_embedding(
                token_ids_np.flatten(),  # Metal backend expects flat array
                embedding_table=self.embed_tokens_weight
            )

            if hasattr(result, 'success') and result.success:
                # Reshape back to expected dimensions
                if input_ids.dim() == 1:
                    # Return (seq_len, hidden_size)
                    embeddings = result.output.reshape(seq_len, self.config.hidden_size)
                else:
                    # Return (batch_size, seq_len, hidden_size)
                    embeddings = result.output.reshape(batch_size, seq_len, self.config.hidden_size)

                # Convert back to torch tensor on the same device as input
                return torch.from_numpy(embeddings).to(device=input_ids.device)
            elif hasattr(result, 'output') and result.output is not None:
                # Metal operation succeeded but different result structure
                if input_ids.dim() == 1:
                    # Return (seq_len, hidden_size)
                    embeddings = result.output.reshape(seq_len, self.config.hidden_size)
                else:
                    # Return (batch_size, seq_len, hidden_size)
                    embeddings = result.output.reshape(batch_size, seq_len, self.config.hidden_size)

                # Convert back to torch tensor on the same device as input
                return torch.from_numpy(embeddings).to(device=input_ids.device)
            else:
                # Fallback: simple numpy-based embedding lookup
                error_msg = getattr(result, 'error', 'unknown result structure')
                print(f"⚠️ Metal embedding failed, using fallback: {error_msg}")
                return self._fallback_embed_tokens(input_ids)

        except Exception as e:
            print(f"⚠️ Metal embedding error, using fallback: {e}")
            return self._fallback_embed_tokens(input_ids)

    def _fallback_embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Fallback token embedding using simple numpy lookup.
        """
        # Convert to numpy
        token_ids_np = input_ids.cpu().numpy().astype(np.int32)

        # Clamp token IDs to valid range
        token_ids_np = np.clip(token_ids_np, 0, self.config.vocab_size - 1)

        # Simple embedding lookup
        if input_ids.dim() == 1:
            # Single sequence
            embeddings = self.embed_tokens_weight[token_ids_np]  # (seq_len, hidden_size)
        else:
            # Batch
            batch_size, seq_len = token_ids_np.shape
            embeddings = self.embed_tokens_weight[token_ids_np.flatten()].reshape(
                batch_size, seq_len, self.config.hidden_size
            )

        # Convert back to torch tensor
        return torch.from_numpy(embeddings).to(device=input_ids.device)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        kv_cache_at_layer: Optional[List[torch.Tensor]] = None,
        kv_page_indices: Optional[torch.Tensor] = None,
        kv_page_indptr: Optional[torch.Tensor] = None,
        kv_last_page_lens: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through the model using Metal kernels.

        This maintains the same interface as the original PyTorch model.
        """
        # Convert PyTorch tensors to numpy for Metal backend with proper dtype handling
        if input_ids is not None:
            input_ids_np = input_ids.cpu().numpy().astype(np.int32)
        elif input_embeds is not None:
            # Handle pre-computed embeddings - preserve precision for bfloat16
            if input_embeds.dtype == torch.bfloat16:
                # bfloat16 -> float32 conversion (unavoidable for NumPy compatibility)
                hidden_states_np = input_embeds.cpu().to(torch.float32).numpy()
            else:
                hidden_states_np = input_embeds.cpu().numpy().astype(np.float32)
            input_ids_np = None
        else:
            raise ValueError("Either input_ids or input_embeds must be provided")

        # Step 1: Embedding lookup (if needed)
        if input_ids_np is not None:
            embed_result = self.metal_backend.run_embedding(
                input_ids_np,
                embedding_table=self.embed_tokens_weight
            )
            hidden_states_np = embed_result.output

        # Convert KV cache to numpy if provided
        kv_cache_np_layers = []
        if kv_cache_at_layer is not None:
            for kv_cache_layer in kv_cache_at_layer:
                if kv_cache_layer is not None:
                    kv_cache_np_layers.append(kv_cache_layer.cpu().numpy().astype(np.float32))
                else:
                    # Create empty KV cache for this layer
                    batch_size = hidden_states_np.shape[0]
                    kv_cache_shape = (
                        1,  # num_pages
                        2,  # K and V
                        16, # page_size
                        self.config.num_key_value_heads,
                        self.config.head_size
                    )
                    empty_kv_cache = np.zeros(kv_cache_shape, dtype=np.float32)
                    kv_cache_np_layers.append(empty_kv_cache)
        else:
            # Create empty KV caches for all layers
            batch_size = hidden_states_np.shape[0]
            for _ in range(self.config.num_layers):
                kv_cache_shape = (
                    1,  # num_pages
                    2,  # K and V
                    16, # page_size
                    self.config.num_key_value_heads,
                    self.config.head_size
                )
                empty_kv_cache = np.zeros(kv_cache_shape, dtype=np.float32)
                kv_cache_np_layers.append(empty_kv_cache)

        # Convert page indices to numpy if provided
        if kv_page_indices is not None:
            kv_page_indices_np = kv_page_indices.cpu().numpy().astype(np.int32)
        else:
            kv_page_indices_np = np.array([0], dtype=np.int32)

        if kv_page_indptr is not None:
            kv_page_indptr_np = kv_page_indptr.cpu().numpy().astype(np.int32)
        else:
            kv_page_indptr_np = np.array([0, 1], dtype=np.int32)

        if kv_last_page_lens is not None:
            kv_last_page_lens_np = kv_last_page_lens.cpu().numpy().astype(np.int32)
        else:
            seq_len = hidden_states_np.shape[0]
            kv_last_page_lens_np = np.array([seq_len], dtype=np.int32)

        # Step 2: Forward pass through all transformer layers
        for layer_idx in range(self.config.num_layers):
            hidden_states_np = self._forward_layer(
                hidden_states_np,
                layer_idx,
                kv_cache_np_layers[layer_idx],
                kv_page_indices_np,
                kv_page_indptr_np,
                kv_last_page_lens_np
            )

        # Step 3: Final normalization
        norm_result = self.metal_backend.run_normalization(hidden_states_np)
        hidden_states_np = norm_result.output

        # Step 4: LM head projection
        # For now, use numpy matrix multiplication
        # In production, this would use a specialized Metal kernel
        logits_np = np.dot(hidden_states_np, self.lm_head_weight.T)

        # Convert back to PyTorch tensor
        logits_tensor = torch.from_numpy(logits_np).to(
            device=self.config.device,
            dtype=self.config.dtype
        )

        return logits_tensor

    def _forward_layer(
        self,
        hidden_states: np.ndarray,
        layer_idx: int,
        kv_cache: np.ndarray,
        kv_page_indices: np.ndarray,
        kv_page_indptr: np.ndarray,
        kv_last_page_lens: np.ndarray
    ) -> np.ndarray:
        """Forward pass through a single transformer layer."""

        # Pre-attention LayerNorm
        norm_result = self.metal_backend.run_normalization(hidden_states)
        normed_hidden_states = norm_result.output

        # Self-attention with KV cache
        attn_result = self.metal_backend.run_attention_with_kv_cache(
            normed_hidden_states,
            kv_cache,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_lens
        )
        attn_output = attn_result.output

        # Residual connection
        hidden_states = hidden_states + attn_output

        # Pre-MLP LayerNorm
        norm_result = self.metal_backend.run_normalization(hidden_states)
        normed_hidden_states = norm_result.output

        # MLP
        mlp_result = self.metal_backend.run_mlp(normed_hidden_states)
        mlp_output = mlp_result.output

        # Residual connection
        hidden_states = hidden_states + mlp_output

        return hidden_states


class MetalL4maForCausalLM:
    """
    Metal-based L4MA model for causal language modeling.

    This is the top-level model class that matches the interface
    expected by the handler.
    """

    def __init__(self, config: L4maArch):
        """Initialize the causal LM model."""
        self.config = config
        self.model = MetalL4maModel(config)

    def forward(self, **kwargs):
        """Forward pass delegation to the model."""
        return self.model.forward(**kwargs)

    def get_capabilities(self) -> Dict[str, Any]:
        """Get model capabilities."""
        return {
            'model_type': 'metal_l4ma',
            'num_layers': self.config.num_layers,
            'hidden_size': self.config.hidden_size,
            'vocab_size': self.config.vocab_size,
            'metal_backend_available': METAL_BACKEND_AVAILABLE,
            'device': self.config.device
        }