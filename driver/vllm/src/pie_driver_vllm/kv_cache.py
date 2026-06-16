"""KV cache allocation for the vllm driver.

Pie cedes physical bytes to vllm: vllm's selected attention backend picks the
KV layout (FlashAttn: `(2, num_blocks, block_size, kv_heads, head)`, FlashInfer:
`(num_blocks, 2, block_size, kv_heads, head)`, etc.). We allocate one raw
buffer per attention layer, view it through the backend's preferred shape, and
bind each `Attention` layer's `self.kv_cache` attribute via `bind_kv_cache`.

Pie's Rust scheduler still owns block IDs — they're just the integer indices
into the block dimension of these tensors. The CLI handshake already fixed
`kv_page_size` from vllm's preferred block size before Rust bootstrap, so
the per-block stride agrees on both sides.

Host pool: skipped in Phase 1.2 (returns empty). Phase 1.5+ will allocate it
in vllm's chosen layout and add per-backend block-dim awareness to the swap
RPCs.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from ._bridge.config import RuntimeConfig
from .hybrid_blocks import select_kernel_block_size

if TYPE_CHECKING:
    from .loader import LoadedModel


def _compute_num_blocks(
    *,
    bytes_per_block: int,
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
    if bytes_per_block <= 0:
        return 0
    num_blocks = budget // bytes_per_block
    # We need at least a handful for the runtime to schedule anything; fail
    # loudly if the budget is insufficient.
    if num_blocks < 8:
        raise RuntimeError(
            f"Insufficient KV cache budget: {budget} bytes / "
            f"{bytes_per_block} bytes per global block = {num_blocks} blocks. "
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
        set_current_vllm_config,
    )
    from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec
    from vllm.utils.torch_utils import get_dtype_size

    vllm_config = loaded.vllm_config
    fc = vllm_config.compilation_config.static_forward_context

    cache_layers: dict[str, "AttentionLayerBase"] = {
        name: layer for name, layer in fc.items()
        if isinstance(layer, AttentionLayerBase)
    }

    if not cache_layers:
        raise RuntimeError("No cache-bearing layers found in vllm forward context.")

    with set_current_vllm_config(vllm_config):
        layer_specs = {
            name: layer.get_kv_cache_spec(vllm_config)
            for name, layer in cache_layers.items()
        }
    layer_specs = {
        name: spec for name, spec in layer_specs.items()
        if spec is not None
    }
    if not layer_specs:
        raise RuntimeError("No KV/Mamba cache specs found in vllm forward context.")

    def _layer_backend(layer):
        backend = getattr(layer, "attn_backend", None)
        if backend is not None:
            return backend
        return layer.get_attn_backend()

    layer_kernel_block_sizes: dict[str, int] = {}
    for layer_name, spec in layer_specs.items():
        layer = cache_layers[layer_name]
        layer_kernel_block_sizes[layer_name] = select_kernel_block_size(
            spec,
            [_layer_backend(layer)],
            vllm_config,
        )

    # GPUModelRunner normally initializes the global Mamba SSU dispatcher from
    # the resolved KV cache config. Pie allocates caches directly, so provide
    # the minimal group view the dispatcher needs.
    try:
        from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
            initialize_mamba_ssu_backend,
        )

        initialize_mamba_ssu_backend(
            vllm_config.mamba_config,
            SimpleNamespace(
                kv_cache_groups=[
                    SimpleNamespace(kv_cache_spec=spec)
                    for spec in layer_specs.values()
                ]
            ),
        )
    except ImportError:
        pass

    bytes_per_block = sum(spec.page_size_bytes for spec in layer_specs.values())

    device = torch.device(config.device)
    physical_num_blocks = _compute_num_blocks(
        bytes_per_block=bytes_per_block,
        device=device,
        gpu_mem_utilization=driver_config.gpu_memory_utilization,
    )
    # vLLM reserves physical block 0 as the NULL/padding block. Pie's runtime
    # sees only schedulable pages, and attn_metadata maps page id N to physical
    # block N+1 before handing metadata to vLLM.
    num_blocks = physical_num_blocks - 1
    if num_blocks < 8:
        raise RuntimeError(
            f"Insufficient KV cache budget after reserving vLLM null block: "
            f"{num_blocks} schedulable blocks."
        )
    config.total_pages = num_blocks
    # vLLM's FlexAttention metadata builder asserts `cache_config.num_gpu_blocks`
    # is set; pie manages its own KV pool, so we sync the count over here so
    # backends that consult vllm_config see the right value.
    vllm_config.cache_config.num_gpu_blocks = physical_num_blocks

    def _backend_kv_shape(
        backend,
        spec: AttentionSpec,
        kernel_block_size: int,
    ) -> tuple[int, ...]:
        num_blocks_per_kv_block = spec.block_size // kernel_block_size
        kernel_num_blocks = physical_num_blocks * num_blocks_per_kv_block
        if spec.storage_block_size != spec.block_size:
            shape_block_size = spec.storage_block_size
        else:
            shape_block_size = kernel_block_size
        kwargs = dict(
            num_blocks=kernel_num_blocks,
            block_size=shape_block_size,
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
        )
        try:
            return backend.get_kv_cache_shape(
                **kwargs,
                cache_dtype_str=vllm_config.cache_config.cache_dtype,
            )
        except TypeError:
            return backend.get_kv_cache_shape(**kwargs)

    def _backend_block_dim(
        backend,
        spec: AttentionSpec,
        kernel_block_size: int,
    ) -> int:
        try:
            return backend.get_kv_cache_block_dim(
                kernel_block_size,
                spec.num_kv_heads,
                spec.head_size,
                cache_dtype_str=vllm_config.cache_config.cache_dtype,
            )
        except TypeError:
            return backend.get_kv_cache_block_dim(
                kernel_block_size,
                spec.num_kv_heads,
                spec.head_size,
            )

    def _allocate_attention_cache(
        layer,
        spec: AttentionSpec,
        kernel_block_size: int,
    ) -> torch.Tensor:
        backend = _layer_backend(layer)
        kv_shape = _backend_kv_shape(backend, spec, kernel_block_size)

        # Stride order (some backends want a permutation of the canonical shape).
        try:
            with set_current_vllm_config(vllm_config):
                stride_order = backend.get_kv_cache_stride_order()
        except (AttributeError, NotImplementedError):
            stride_order = tuple(range(len(kv_shape)))

        permuted_shape = tuple(kv_shape[i] for i in stride_order)
        inv_order = [stride_order.index(i) for i in range(len(stride_order))]

        total_bytes = physical_num_blocks * spec.page_size_bytes
        raw = torch.zeros(total_bytes, dtype=torch.int8, device=device)
        # view int8 → spec dtype → backend shape → permute back to canonical
        raw_typed = raw.view(spec.dtype)
        if spec.page_size_padded is not None:
            dtype_size = get_dtype_size(spec.dtype)
            page_stride = spec.page_size_bytes // dtype_size
            strides = list(torch.empty(permuted_shape).stride())
            strides[inv_order[0]] = page_stride
            kv = torch.as_strided(
                raw_typed,
                size=permuted_shape,
                stride=tuple(strides),
            )
        else:
            kv = raw_typed.view(permuted_shape)
        return kv.permute(*inv_order)

    def _allocate_mamba_cache(spec: MambaSpec) -> list[torch.Tensor]:
        raw = torch.zeros(
            physical_num_blocks * spec.page_size_bytes,
            dtype=torch.int8,
            device=device,
        )
        state_tensors: list[torch.Tensor] = []
        storage_offset_bytes = 0
        for shape, dtype in zip(spec.shapes, spec.dtypes):
            dtype_size = get_dtype_size(dtype)
            elements_per_page = spec.page_size_bytes // dtype_size
            target_shape = (physical_num_blocks, *shape)
            stride = torch.empty(target_shape).stride()
            target_stride = (elements_per_page, *stride[1:])
            assert storage_offset_bytes % dtype_size == 0
            state_tensors.append(
                torch.as_strided(
                    raw.view(dtype),
                    size=target_shape,
                    stride=target_stride,
                    storage_offset=storage_offset_bytes // dtype_size,
                )
            )
            storage_offset_bytes += stride[0] * dtype_size
        return state_tensors

    # For each layer, allocate a raw int8 buffer and view it through the
    # backend's preferred shape. We don't pool buffers across layers (vllm's
    # runner does for memory locality, but with one object per layer we keep
    # pie's swap RPC math simple).
    kv_caches: dict[str, object] = {}
    has_attn = False
    has_mamba = False

    for layer_name, spec in layer_specs.items():
        layer = cache_layers[layer_name]
        if isinstance(spec, AttentionSpec):
            has_attn = True
            kv_caches[layer_name] = _allocate_attention_cache(
                layer,
                spec,
                layer_kernel_block_sizes[layer_name],
            )
        elif isinstance(spec, MambaSpec):
            has_mamba = True
            kv_caches[layer_name] = _allocate_mamba_cache(spec)
        else:
            raise NotImplementedError(f"Unsupported vLLM KV cache spec: {spec!r}")

    # vLLM changes the attention cache strides in hybrid attention+Mamba
    # models so the physical block is the contiguous transfer unit even for
    # backends whose logical shape is (K/V, block, ...). Mirror that layout so
    # Mamba and attention groups agree on block IDs.
    if has_attn and has_mamba:
        for layer_name, spec in layer_specs.items():
            if not isinstance(spec, AttentionSpec):
                continue
            kv_cache = kv_caches[layer_name]
            assert isinstance(kv_cache, torch.Tensor)
            backend = _layer_backend(cache_layers[layer_name])
            block_dim = _backend_block_dim(
                backend,
                spec,
                layer_kernel_block_sizes[layer_name],
            )
            if block_dim == 0:
                continue
            assert block_dim == 1
            hidden_size = kv_cache.shape[2:].numel()
            kv_cache.as_strided_(
                size=kv_cache.shape,
                stride=(hidden_size, 2 * hidden_size, *kv_cache.stride()[2:]),
            )

    # bind_kv_cache writes `layer.kv_cache = tensor` for every attention layer
    # and fills `runner_kv_caches` in layer-index order (which we want).
    runner_kv_caches: list[object] = []
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
