"""KV cache allocation for the vllm driver.

Pie cedes physical bytes to vllm: vllm's selected attention backend picks the
KV layout (FlashAttn: `(2, num_blocks, block_size, kv_heads, head)`, FlashInfer:
`(num_blocks, 2, block_size, kv_heads, head)`, etc.). We allocate one raw
buffer per attention layer, view it through the backend's preferred shape, and
bind each `Attention` layer's `self.kv_cache` attribute via `bind_kv_cache`.

Pie's Rust scheduler still owns block IDs — they're just the integer indices
into the block dimension of these tensors. The CLI handshake already fixed
`kv_page_size` (via vllm's preferred block size) before Rust bootstrap, so
the per-block stride agrees on both sides.

Host pool: skipped in Phase 1.2 (returns empty). Phase 1.5+ will allocate it
in vllm's chosen layout and add per-backend block-dim awareness to the swap
RPCs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pie_driver.config import RuntimeConfig

if TYPE_CHECKING:
    from .loader import LoadedModel


def _compute_num_blocks(
    *,
    page_size_bytes: int,
    num_layers: int,
    device: torch.device,
    gpu_mem_utilization: float,
) -> int:
    """How many blocks we can afford given the post-load free memory budget.

    We measure free memory *after* the model has been loaded, then take a
    fraction (`gpu_mem_utilization`) of it as the KV budget. Total bytes per
    block across all layers = `page_size_bytes * num_layers`.
    """
    free_bytes, _total = torch.cuda.mem_get_info(device)
    budget = int(free_bytes * gpu_mem_utilization)
    per_block_total = page_size_bytes * num_layers
    if per_block_total <= 0:
        return 0
    num_blocks = budget // per_block_total
    # We need at least a handful for the runtime to schedule anything; fail
    # loudly if the budget is insufficient.
    if num_blocks < 8:
        raise RuntimeError(
            f"Insufficient KV cache budget: {budget} bytes / "
            f"{per_block_total} bytes per block × {num_layers} layers = {num_blocks} blocks. "
            f"Increase gpu_mem_utilization or reduce model size."
        )
    return int(num_blocks)


def allocate_and_bind_kv_cache(
    loaded: "LoadedModel",
    config: RuntimeConfig,
    driver_config,
) -> list[torch.Tensor]:
    """Allocate the GPU KV cache and bind it to every attention layer.

    Returns the per-layer tensor list in layer-index order, which pie's worker
    treats as `engine.kv_cache_at_layer` for the swap RPCs.
    """
    from ._vllm_compat import (
        AttentionLayerBase,
        bind_kv_cache,
        extract_layer_index,
        set_current_vllm_config,
    )

    vllm_config = loaded.vllm_config
    fc = vllm_config.compilation_config.static_forward_context

    attn_layers: dict[str, "AttentionLayerBase"] = {
        name: layer for name, layer in fc.items()
        if isinstance(layer, AttentionLayerBase)
    }

    if not attn_layers:
        raise RuntimeError("No attention layers found in vllm forward context.")

    with set_current_vllm_config(vllm_config):
        first_spec = next(iter(attn_layers.values())).get_kv_cache_spec(vllm_config)
    page_size_bytes = first_spec.page_size_bytes
    block_size = first_spec.block_size
    num_kv_heads = first_spec.num_kv_heads
    head_size = first_spec.head_size
    spec_dtype = first_spec.dtype

    device = torch.device(config.device)
    num_blocks = _compute_num_blocks(
        page_size_bytes=page_size_bytes,
        num_layers=len(attn_layers),
        device=device,
        gpu_mem_utilization=driver_config.gpu_memory_utilization,
    )
    config.max_num_kv_pages = num_blocks
    # vLLM's FlexAttention metadata builder asserts `cache_config.num_gpu_blocks`
    # is set; pie manages its own KV pool, so we sync the count over here so
    # backends that consult vllm_config see the right value.
    vllm_config.cache_config.num_gpu_blocks = num_blocks

    # For each layer, allocate a raw int8 buffer and view it through the
    # backend's preferred shape. We don't pool buffers across layers (vllm's
    # runner does for memory locality, but with one tensor per layer we keep
    # pie's swap RPC math simple).
    kv_caches: dict[str, torch.Tensor] = {}
    layer_to_index: dict[str, int] = {}

    for layer_name, layer in attn_layers.items():
        backend = layer.attn_backend
        kv_shape = backend.get_kv_cache_shape(
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )

        # Stride order (some backends want a permutation of the canonical shape).
        try:
            with set_current_vllm_config(vllm_config):
                stride_order = backend.get_kv_cache_stride_order()
        except (AttributeError, NotImplementedError):
            stride_order = tuple(range(len(kv_shape)))

        permuted_shape = tuple(kv_shape[i] for i in stride_order)
        inv_order = [stride_order.index(i) for i in range(len(stride_order))]

        total_bytes = num_blocks * page_size_bytes
        raw = torch.zeros(total_bytes, dtype=torch.int8, device=device)
        # view int8 → spec dtype → backend shape → permute back to canonical
        kv = raw.view(spec_dtype).view(permuted_shape).permute(*inv_order)
        kv_caches[layer_name] = kv
        layer_to_index[layer_name] = extract_layer_index(layer_name)

    # bind_kv_cache writes `layer.kv_cache = tensor` for every attention layer
    # and fills `runner_kv_caches` in layer-index order (which we want).
    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_caches, fc, runner_kv_caches)

    return runner_kv_caches


def allocate_host_pool(
    gpu_kv: list[torch.Tensor],
    swap_budget_bytes: int,
) -> tuple[list[torch.Tensor], int]:
    """CPU swap pool. Skipped in Phase 1.2.

    Phase 1.5+ will mirror vllm's GPU layout (block dim depends on backend),
    pin where supported (CUDA / ROCm / XPU; not TPU), and update pie's swap
    RPCs to be block-dim aware.
    """
    if swap_budget_bytes <= 0 or not gpu_kv:
        return [], 0
    # Until per-backend block-dim awareness is wired, refuse non-zero budgets
    # rather than silently producing wrong tokens after a swap.
    raise NotImplementedError(
        "CPU swap pool not yet supported on the vllm driver (Phase 1.5). "
        "Set swap_budget_bytes=0 / cpu_mem_budget_in_gb=0 for now."
    )
