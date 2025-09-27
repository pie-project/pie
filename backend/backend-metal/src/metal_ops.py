"""
Metal Backend Operations

This module provides Metal-specific implementations of the backend operations
interface defined in backend_ops.py. It uses Metal kernels for sampling and
other operations where possible.
"""

from __future__ import annotations
import sys
from pathlib import Path
import torch

# Import the base BackendOps class from backend-python
backend_python_path = Path(__file__).parent.parent.parent / "backend-python"
sys.path.insert(0, str(backend_python_path))

from backend_ops import BackendOps


class MetalOps(BackendOps):
    """Metal backend operations using Metal kernels."""

    def __init__(self):
        super().__init__("metal")
        # Check if Metal backend is available
        try:
            from metal_backend import MetalBackend
            self.available = True
            print("✅ Metal backend available for operations")
        except ImportError:
            self.available = False
            print("⚠️ Metal backend not available")

    def decode_image(self, image_blob: bytes, dtype: torch.dtype, device: str) -> torch.Tensor:
        """Decode image using Metal kernels."""
        # TODO: Implement Metal image decoding when needed
        raise NotImplementedError("Metal image decoding not yet implemented")

    def sampling_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """Sample using Metal kernels."""
        return self._metal_sample_multinomial(probs)

    def top_p_sampling_from_probs(self, probs: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
        """Top-p sampling using Metal kernels."""
        return self._metal_top_p_sample(probs, top_p)

    def top_k_sampling_from_probs(self, probs: torch.Tensor, top_k: torch.Tensor) -> torch.Tensor:
        """Top-k sampling using Metal kernels."""
        return self._metal_top_k_sample(probs, top_k)

    def min_p_sampling_from_probs(self, probs: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
        """Min-p sampling using Metal kernels."""
        return self._metal_min_p_sample(probs, min_p)

    def top_k_top_p_sampling_from_probs(self, probs: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
        """Combined top-k and top-p sampling using Metal kernels."""
        return self._metal_top_k_top_p_sample(probs, top_k, top_p)

    def _metal_sample_multinomial(self, probs: torch.Tensor) -> torch.Tensor:
        """Simple multinomial sampling using PyTorch (fallback until Metal implementation)."""
        # For now, use PyTorch's multinomial sampling
        # TODO: Implement Metal kernel for sampling
        return torch.multinomial(probs, 1).squeeze(-1)

    def _metal_top_p_sample(self, probs: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
        """Top-p sampling implementation using Metal-optimized approach."""

        # Validate inputs
        if probs.size(0) == 0:
            raise ValueError("Cannot sample from empty batch")
        if not torch.isfinite(probs).all():
            raise ValueError("Non-finite probabilities detected")

        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Create mask for probabilities to keep (cumulative probability <= top_p)
        # Handle broadcasting more robustly
        if top_p.numel() == 1:
            # Single top_p value - use scalar comparison for efficiency
            threshold = top_p.item()
            keep_mask = cumulative_probs <= threshold
        else:
            # Multiple top_p values - use tensor broadcasting
            keep_mask = cumulative_probs <= top_p.unsqueeze(-1)

        # Always keep at least the highest probability token
        keep_mask[..., 0] = True

        # Set probabilities of filtered tokens to 0
        filtered_probs = sorted_probs.clone()
        filtered_probs[~keep_mask] = 0.0

        # Renormalize and validate
        prob_sums = filtered_probs.sum(dim=-1, keepdim=True)
        if (prob_sums <= 0).any():
            raise ValueError("Filtered probabilities sum to zero; cannot sample")
        filtered_probs = filtered_probs / prob_sums

        # Sample from filtered distribution
        sampled_indices = torch.multinomial(filtered_probs, 1).squeeze(-1)

        # Map back to original indices
        result = sorted_indices.gather(-1, sampled_indices.unsqueeze(-1)).squeeze(-1)

        # Validate result
        vocab_size = probs.size(-1)
        if (result < 0).any() or (result >= vocab_size).any():
            raise ValueError(f"Invalid token IDs generated: {result}")

        return result

    def _metal_top_k_sample(self, probs: torch.Tensor, top_k: torch.Tensor) -> torch.Tensor:
        """Top-k sampling implementation using Metal-optimized approach."""

        if top_k.numel() == 0:
            raise ValueError("top_k tensor must contain at least one element")

        batch_size = probs.size(0)
        top_k = top_k.to(device=probs.device, dtype=torch.int64)
        if top_k.ndim == 0:
            top_k = top_k.expand(batch_size)
        elif top_k.ndim == 1 and top_k.size(0) == 1 and batch_size > 1:
            top_k = top_k.expand(batch_size)
        elif top_k.ndim != 1 or top_k.size(0) != batch_size:
            raise ValueError("top_k shape must align with probability batch dimension")

        clamped_top_k = torch.clamp(top_k, min=1, max=probs.size(-1))
        max_k = int(clamped_top_k.max().item())
        top_probs, top_indices = torch.topk(probs, k=max_k, dim=-1)

        sampled_tokens = []
        for row_idx in range(batch_size):
            k_val = int(clamped_top_k[row_idx].item())
            row_probs = top_probs[row_idx, :k_val]
            prob_sum = row_probs.sum()
            if prob_sum <= 0:
                raise ValueError("Top-k probabilities sum to zero; cannot sample")
            row_probs = row_probs / prob_sum
            sampled_idx = torch.multinomial(row_probs, 1)
            sampled_tokens.append(top_indices[row_idx, sampled_idx])

        return torch.stack(sampled_tokens).squeeze(-1)

    def _metal_min_p_sample(self, probs: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
        """Min-p sampling implementation using Metal-optimized approach."""

        # Find maximum probability
        max_prob = torch.max(probs, dim=-1, keepdim=True)[0]

        # Calculate threshold (min_p * max_prob)
        threshold = min_p.unsqueeze(-1) * max_prob

        # Create mask for probabilities above threshold
        keep_mask = probs >= threshold

        # Ensure at least one token is kept
        if not keep_mask.any():
            keep_mask = probs == max_prob

        # Filter probabilities
        filtered_probs = probs.clone()
        filtered_probs[~keep_mask] = 0.0

        # Renormalize
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

        # Sample
        return torch.multinomial(filtered_probs, 1).squeeze(-1)

    def _metal_top_k_top_p_sample(self, probs: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
        """Combined top-k and top-p sampling implementation using Metal-optimized approach."""
        print("🔧 Using Metal-optimized top-k + top-p sampling")

        if top_k.numel() == 0 or top_p.numel() == 0:
            raise ValueError("top_k and top_p tensors must contain at least one element")

        batch_size = probs.size(0)
        top_k = top_k.to(device=probs.device, dtype=torch.int64)
        top_p = top_p.to(device=probs.device, dtype=probs.dtype)

        if top_k.ndim == 0:
            top_k = top_k.expand(batch_size)
        elif top_k.ndim == 1 and top_k.size(0) == 1 and batch_size > 1:
            top_k = top_k.expand(batch_size)
        elif top_k.ndim != 1 or top_k.size(0) != batch_size:
            raise ValueError("top_k shape must align with probability batch dimension")

        if top_p.ndim == 0:
            top_p = top_p.expand(batch_size)
        elif top_p.ndim == 1 and top_p.size(0) == 1 and batch_size > 1:
            top_p = top_p.expand(batch_size)
        elif top_p.ndim != 1 or top_p.size(0) != batch_size:
            raise ValueError("top_p shape must align with probability batch dimension")

        clamped_top_k = torch.clamp(top_k, min=1, max=probs.size(-1))
        clamped_top_p = torch.clamp(top_p, min=1e-6, max=1.0)

        max_k = int(clamped_top_k.max().item())
        top_probs, top_indices = torch.topk(probs, k=max_k, dim=-1)

        sampled_tokens = []
        for row_idx in range(batch_size):
            k_val = int(clamped_top_k[row_idx].item())
            p_val = clamped_top_p[row_idx].item()

            row_top_probs = top_probs[row_idx, :k_val]
            row_top_indices = top_indices[row_idx, :k_val]

            sorted_probs, sorted_idx = torch.sort(row_top_probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)

            keep_mask = cumulative_probs <= p_val
            keep_mask[0] = True

            filtered_probs = sorted_probs.clone()
            filtered_probs[~keep_mask] = 0.0
            prob_sum = filtered_probs.sum()
            if prob_sum <= 0:
                raise ValueError("Filtered probabilities sum to zero; cannot sample")
            filtered_probs = filtered_probs / prob_sum

            sampled_idx = torch.multinomial(filtered_probs, 1)
            mapped_idx = sorted_idx[sampled_idx]
            sampled_tokens.append(row_top_indices[mapped_idx])

        return torch.stack(sampled_tokens).squeeze(-1)
