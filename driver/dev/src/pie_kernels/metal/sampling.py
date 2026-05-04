"""
Sampling operations on MPS.

PyTorch-on-MPS implementations of the symbols that upstream `flashinfer.sampling`
exposes. Consumed via the `pie_kernels.sampling` dispatcher.
"""

import torch

__all__ = [
    "sampling_from_probs",
    "top_p_sampling_from_probs",
    "top_k_sampling_from_probs",
    "min_p_sampling_from_probs",
    "top_k_top_p_sampling_from_probs",
]


def sampling_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """Sample from probability distribution."""
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def top_p_sampling_from_probs(probs: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    """Top-p (nucleus) sampling."""
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_probs <= top_p.unsqueeze(-1)
    mask[..., 0] = True
    sorted_probs[~mask] = 0
    sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
    return torch.gather(sorted_indices, -1, sampled_sorted_idx.unsqueeze(-1)).squeeze(
        -1
    )


def top_k_sampling_from_probs(probs: torch.Tensor, top_k: torch.Tensor) -> torch.Tensor:
    """Top-k sampling: sample from the top k tokens."""
    k = int(top_k.max().item()) if top_k.numel() > 0 else 50
    k = min(k, probs.shape[-1])

    topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    sampled_indices = torch.multinomial(topk_probs, num_samples=1).squeeze(-1)
    return topk_indices.gather(-1, sampled_indices.unsqueeze(-1)).squeeze(-1)


def min_p_sampling_from_probs(probs: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
    """Min-p sampling: filter tokens below min_p * max_prob."""
    max_probs = probs.max(dim=-1, keepdim=True).values
    threshold = min_p.unsqueeze(-1) * max_probs
    filtered_probs = probs.clone()
    filtered_probs[probs < threshold] = 0
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True).clamp(
        min=1e-8
    )
    return torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)


def top_k_top_p_sampling_from_probs(
    probs: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor
) -> torch.Tensor:
    """Combined top-k and top-p sampling."""
    k = int(top_k.max().item()) if top_k.numel() > 0 else 50
    k = min(k, probs.shape[-1])

    topk_probs, topk_indices = torch.topk(probs, k, dim=-1)

    sorted_probs, sorted_order = torch.sort(topk_probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_probs <= top_p.unsqueeze(-1)
    mask[..., 0] = True
    sorted_probs[~mask] = 0

    sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
    sampled_topk_idx = sorted_order.gather(
        -1, sampled_sorted_idx.unsqueeze(-1)
    ).squeeze(-1)
    return topk_indices.gather(-1, sampled_topk_idx.unsqueeze(-1)).squeeze(-1)
