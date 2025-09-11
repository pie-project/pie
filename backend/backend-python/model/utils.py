"""Model utility functions shared across different architectures."""

import math
import torch
from typing import Optional


# MXFP4 conversion constants (from official GPT-OSS implementation)
FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def convert_mxfp4_to_bf16(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    target_device: str,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 16384 * 512,
) -> torch.Tensor:
    """
    Convert MXFP4 format tensors (blocks and scales) to bfloat16 format.

    This implementation is based on the official GPT-OSS weights.py implementation.

    Args:
        blocks: The packed FP4 values tensor (uint8)
        scales: The block scales tensor
        dtype: Target dtype for conversion (default: torch.bfloat16)
        rows_per_chunk: Number of rows to process per chunk for memory efficiency

    Returns:
        Converted tensor in the target dtype
    """
    # Convert scales to int32 and subtract bias (from official implementation)
    scales = scales.to(torch.int32) - 127

    assert (
        blocks.shape[:-1] == scales.shape
    ), f"{blocks.shape=} does not match {scales.shape=}"

    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=target_device)

    *prefix_shape, g, b = blocks.shape
    rows_total = math.prod(prefix_shape) * g

    blocks = blocks.reshape(rows_total, b).to(target_device)
    scales = scales.reshape(rows_total, 1).to(target_device)

    out = torch.empty(rows_total, b * 2, dtype=dtype, device=target_device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)

        blk = blocks[r0:r1]
        exp = scales[r0:r1]

        # nibble indices -> int64
        idx_lo = (blk & 0x0F).to(torch.long)
        idx_hi = (blk >> 4).to(torch.long)

        sub = out[r0:r1]
        sub[:, 0::2] = lut[idx_lo]
        sub[:, 1::2] = lut[idx_hi]

        torch.ldexp(sub, exp, out=sub)
        del idx_lo, idx_hi, blk, exp

    return out.reshape(*prefix_shape, g, b * 2).view(*prefix_shape, g * b * 2)


def extract_kv_from_paged_cache(
    kv_cache_at_layer: torch.Tensor,
    kv_page_indices: torch.Tensor,
    kv_page_indptr: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    batch_idx: int = 0,
    num_key_value_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract keys and values from FlashInfer's paged KV cache format.

    Args:
        kv_cache_at_layer: Paged KV cache tensor [num_pages, 2, page_size, num_kv_heads, head_dim]
        kv_page_indices: Page indices for each batch
        kv_page_indptr: Pointer to start of each batch's pages
        kv_last_page_lens: Length of the last page for each batch
        batch_idx: Which batch to extract (default 0 for single batch)
        num_key_value_heads: Number of KV heads (for empty cache fallback)
        head_dim: Head dimension (for empty cache fallback)

    Returns:
        Tuple of (keys, values) tensors with shape [seq_len, num_kv_heads, head_dim]
    """
    # Get the page information for this batch
    page_start = kv_page_indptr[batch_idx].item()
    page_end = kv_page_indptr[batch_idx + 1].item()
    batch_page_indices = kv_page_indices[page_start:page_end]
    last_page_len = kv_last_page_lens[batch_idx].item()

    # Collect keys and values from all pages
    keys_list = []
    values_list = []

    for i, page_idx in enumerate(batch_page_indices):
        page_idx_int = int(page_idx.item())

        # Extract K and V from this page (layout: [num_pages, 2, page_size, num_kv_heads, head_dim])
        # Index 0 is keys, index 1 is values
        k_page = kv_cache_at_layer[
            page_idx_int, 0
        ]  # [page_size, num_kv_heads, head_dim]
        v_page = kv_cache_at_layer[
            page_idx_int, 1
        ]  # [page_size, num_kv_heads, head_dim]

        if i == len(batch_page_indices) - 1:  # Last page
            # Only use the valid tokens in the last page
            keys_list.append(k_page[:last_page_len])
            values_list.append(v_page[:last_page_len])
        else:
            # Use the full page
            keys_list.append(k_page)
            values_list.append(v_page)

    # Concatenate all pages to get the full sequence
    if keys_list:
        keys = torch.cat(keys_list, dim=0)  # [seq_len, num_kv_heads, head_dim]
        values = torch.cat(values_list, dim=0)  # [seq_len, num_kv_heads, head_dim]
    else:
        # Empty cache - need dimensions for fallback
        if num_key_value_heads is None or head_dim is None:
            # Try to infer from cache tensor shape
            if kv_cache_at_layer.numel() > 0:
                num_key_value_heads = kv_cache_at_layer.shape[3]
                head_dim = kv_cache_at_layer.shape[4]
            else:
                raise ValueError(
                    "Cannot create empty cache without num_key_value_heads and head_dim"
                )

        keys = torch.empty(
            0,
            num_key_value_heads,
            head_dim,
            device=kv_cache_at_layer.device,
            dtype=kv_cache_at_layer.dtype,
        )
        values = torch.empty(
            0,
            num_key_value_heads,
            head_dim,
            device=kv_cache_at_layer.device,
            dtype=kv_cache_at_layer.dtype,
        )

    return keys, values
