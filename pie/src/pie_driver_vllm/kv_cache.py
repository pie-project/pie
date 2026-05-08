"""KV cache allocation for the vllm driver.

For pure-attention models (Llama, Qwen3, …) pie cedes physical bytes to
vllm: the resolved attention backend picks the KV layout (FlashAttn:
`(2, num_blocks, block_size, kv_heads, head)`, FlashInfer:
`(num_blocks, 2, block_size, kv_heads, head)`, etc.). We allocate one raw
buffer per attention layer, view it through the backend's preferred shape,
and bind each `Attention` layer's `self.kv_cache` attribute via
`bind_kv_cache`.

For hybrid Transformer + Mamba models (Qwen3.5-MoE, Qwen3-Next: alternating
`full_attention` + `linear_attention` layers) we additionally allocate a
per-mamba-layer `[conv_state, ssm_state]` pair sized to the same `num_blocks`
as the attention pool. The state slot for a request is the request's first
KV page id (see `gdn_metadata.build_gdn_metadata`); since pie's allocator
gives a request at least one page, that page id doubles as the state slot
index. We follow vllm's `mamba_cache_mode="none"` (1 block per request) for
budget computation.

Pie's Rust scheduler still owns block IDs — they're integer indices into
the block dimension of these tensors. The CLI handshake fixes `kv_page_size`
(via vllm's preferred block size for attention layers) before Rust bootstrap.

Host pool: skipped on the vllm driver (Phase 1.5+). For hybrid models we
explicitly refuse a non-zero swap budget — the per-layer copy logic in
`pie_driver/worker.py` doesn't know about mamba state, so swap-out would
silently lose recurrent state. Capabilities advertises `supports_kv_fork=False`
on hybrid so pie's runtime routes fork-using inferlets to a different driver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from pie_driver.config import RuntimeConfig

if TYPE_CHECKING:
    from .loader import LoadedModel


@dataclass
class KVCacheLayout:
    """Result of `allocate_and_bind_kv_cache` — splits attention vs mamba.

    `attention_layers_in_order` is what pie's worker treats as
    `engine.kv_cache_at_layer` (one tensor per attention layer, in
    layer-index order). The existing `_handle_copy_d2d/h2d/d2h` handlers in
    `pie_driver/worker.py` iterate this list to do per-page index_copy_;
    they are correct for attention but cannot rebuild mamba's recurrent
    state, hence `is_hybrid` propagates to capabilities and engine init.

    `mamba_layers` is `(layer_name, conv_state, ssm_state)` tuples; the
    state tensors are sized `(num_blocks, *spec.shapes[i])` so any pie page
    id is a valid state-slot index. `mamba_layer.kv_cache = [conv, ssm]` is
    set during binding so vllm's GatedDeltaNet `_forward` finds it via
    `self.kv_cache[0]`/`[1]`.
    """

    attention_layers_in_order: list[torch.Tensor]
    mamba_layers: list[tuple[str, torch.Tensor, torch.Tensor]]
    is_hybrid: bool


def _compute_num_blocks(
    *,
    attn_page_bytes: int,
    num_attn_layers: int,
    mamba_state_bytes: int,
    num_mamba_layers: int,
    device: torch.device,
    gpu_mem_utilization: float,
) -> int:
    """How many blocks the post-load free memory budget can afford.

    For pure-attention: total_per_block = attn_page_bytes * num_attn_layers.
    For hybrid: each "block" of the shared pool also costs one mamba state
    slot per mamba layer (one (conv_state + ssm_state) per slot). Since
    we size the mamba pool to num_blocks (so any page id is a valid state
    slot), each block contributes mamba state bytes too.
    """
    free_bytes, _total = torch.cuda.mem_get_info(device)
    budget = int(free_bytes * gpu_mem_utilization)
    per_block_total = attn_page_bytes * num_attn_layers
    per_block_total += mamba_state_bytes * num_mamba_layers
    if per_block_total <= 0:
        return 0
    num_blocks = budget // per_block_total
    if num_blocks < 8:
        raise RuntimeError(
            f"Insufficient KV cache budget: {budget} bytes / "
            f"{per_block_total} bytes per block "
            f"({num_attn_layers} attn + {num_mamba_layers} mamba) "
            f"= {num_blocks} blocks. Increase gpu_mem_utilization or reduce model size."
        )
    return int(num_blocks)


def _mamba_state_bytes_per_slot(spec) -> int:
    """Bytes consumed by one mamba state slot across all states (conv+ssm).

    Mirrors `MambaSpec.page_size_bytes` directly — vllm computes it the
    same way for budget accounting in `_get_kv_cache_config_*`. We use this
    rather than `page_size_bytes` to keep budget math explicit (and so a
    future `page_size_padded` doesn't sneak into pie's per-block accounting
    without us noticing).
    """
    from math import prod

    total = 0
    for shape, dtype in zip(spec.shapes, spec.dtypes):
        total += prod(shape) * torch.empty((), dtype=dtype).element_size()
    return total


def allocate_and_bind_kv_cache(
    loaded: "LoadedModel",
    config: RuntimeConfig,
    driver_config,
) -> KVCacheLayout:
    """Allocate the GPU KV cache and bind it to every attention/mamba layer.

    Returns a `KVCacheLayout` with both the attention-only list (what pie's
    worker iterates as `kv_cache_at_layer`) and the mamba state tensors
    (held separately on the engine; not exposed to pie's per-page
    copy handlers).
    """
    from ._vllm_compat import (
        AttentionLayerBase,
        FullAttentionSpec,
        KVCacheSpec,
        MambaSpec,
        bind_kv_cache,
        extract_layer_index,
        get_mamba_base,
        set_current_vllm_config,
    )

    MambaBase = get_mamba_base()

    vllm_config = loaded.vllm_config
    fc = vllm_config.compilation_config.static_forward_context

    # Partition `static_forward_context` into attention layers (KV pool)
    # and mamba layers (recurrent state). Both kinds register themselves in
    # this dict via their __init__ — see `Attention` and `MambaBase`
    # subclasses (Qwen3NextGatedDeltaNet, etc.).
    #
    # NB: in vllm 0.20.0 `MambaBase` extends `AttentionLayerBase` (so
    # GDN/Mamba layers pass the AttentionLayerBase isinstance check too).
    # The mamba check has to come first, and the attention list must
    # explicitly EXCLUDE mamba layers — otherwise we'd treat mamba layers
    # as attention and call `get_kv_cache_spec` on them, which raises
    # MambaSpec downstream of the attention-only flow.
    mamba_layers: dict[str, MambaBase] = {
        name: layer for name, layer in fc.items()
        if isinstance(layer, MambaBase)
    }
    attn_layers: dict[str, "AttentionLayerBase"] = {
        name: layer for name, layer in fc.items()
        if isinstance(layer, AttentionLayerBase)
        and not isinstance(layer, MambaBase)
    }

    if not attn_layers and not mamba_layers:
        raise RuntimeError(
            "No attention or mamba layers found in vllm forward context."
        )

    # Resolve every attention layer's spec up front — pie only handles
    # `FullAttentionSpec`. Sliding-window / chunked / MLA / cross-attention
    # remain unsupported (the existing FlashInfer metadata path doesn't
    # carry their fields). Mamba spec is collected separately below.
    layer_specs: dict[str, "KVCacheSpec"] = {}
    with set_current_vllm_config(vllm_config):
        for name, layer in attn_layers.items():
            layer_specs[name] = layer.get_kv_cache_spec(vllm_config)

    non_full_attention = {
        name: type(spec).__name__
        for name, spec in layer_specs.items()
        if not isinstance(spec, FullAttentionSpec)
    }
    if non_full_attention:
        sample = ", ".join(f"{n}={t}" for n, t in list(non_full_attention.items())[:5])
        raise NotImplementedError(
            f"pie_driver_vllm only supports models whose attention layers all "
            f"return FullAttentionSpec; got {len(non_full_attention)} layer(s) "
            f"with a different spec ({sample}{'...' if len(non_full_attention) > 5 else ''}). "
            f"Sliding-window / chunked / MLA / cross-attention need a separate "
            f"per-spec metadata path — not yet implemented."
        )

    is_hybrid = bool(mamba_layers)

    # On hybrid, swap is unsupported (the existing copy_h2d/d2h handlers
    # in pie_driver/worker.py only iterate `kv_cache_at_layer` and have no
    # way to migrate mamba state). Refuse a non-zero swap budget upfront so
    # the user sees a clear error instead of silent corruption mid-swap.
    if is_hybrid and config.swap_budget_bytes > 0:
        raise NotImplementedError(
            "vllm driver does not support KV swap (CPU offload) on hybrid "
            "Transformer+Mamba models. Set `swap_budget_bytes=0` / "
            "`cpu_mem_budget_in_gb=0` for these models."
        )

    # Resolve mamba spec (one per mamba layer; in current Qwen3-Next /
    # Qwen3.5-MoE all mamba layers share the same shape, but we don't
    # assume that — each layer's spec is queried individually).
    mamba_specs: dict[str, "MambaSpec"] = {}
    with set_current_vllm_config(vllm_config):
        for name, layer in mamba_layers.items():
            spec = MambaSpec(
                shapes=layer.get_state_shape(),
                dtypes=layer.get_state_dtype(),
                # block_size for mamba is irrelevant in our addressing scheme
                # (we use page id as state slot, not block_size-token chunks);
                # pick max_model_len to match vllm's "1 block per request"
                # convention for "none" mode.
                block_size=int(vllm_config.model_config.max_model_len),
                mamba_type=layer.mamba_type,
                num_speculative_blocks=0,
            )
            mamba_specs[name] = spec

    # Page bytes for attention layers (first FullAttentionSpec; merged
    # specs guarantee they all match).
    if attn_layers:
        first_attn_spec = next(iter(layer_specs.values()))
        attn_page_bytes = first_attn_spec.page_size_bytes
        block_size = first_attn_spec.block_size
        num_kv_heads = first_attn_spec.num_kv_heads
        head_size = first_attn_spec.head_size
        spec_dtype = first_attn_spec.dtype
    else:
        # Pure-mamba model — no attention pool. Not in pie's smoke test
        # target, but the math falls through cleanly.
        attn_page_bytes = 0
        block_size = 0
        num_kv_heads = 0
        head_size = 0
        spec_dtype = None

    # Mamba state bytes per slot (sum over all states). All mamba layers
    # are assumed to share the same per-slot byte cost in current Qwen3
    # variants; if they ever diverge, we'd need to size each pool
    # independently.
    if mamba_layers:
        mamba_state_bytes = max(
            _mamba_state_bytes_per_slot(spec) for spec in mamba_specs.values()
        )
    else:
        mamba_state_bytes = 0

    device = torch.device(config.device)
    num_blocks = _compute_num_blocks(
        attn_page_bytes=attn_page_bytes,
        num_attn_layers=len(attn_layers),
        mamba_state_bytes=mamba_state_bytes,
        num_mamba_layers=len(mamba_layers),
        device=device,
        gpu_mem_utilization=driver_config.gpu_memory_utilization,
    )
    config.max_num_kv_pages = num_blocks
    # vLLM's metadata builders read num_gpu_blocks off cache_config — pie
    # manages its own pool but we sync the count so anything reading vllm's
    # config sees the right value.
    vllm_config.cache_config.num_gpu_blocks = num_blocks

    # ── Allocate attention KV ────────────────────────────────────────────
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

        try:
            with set_current_vllm_config(vllm_config):
                stride_order = backend.get_kv_cache_stride_order()
        except (AttributeError, NotImplementedError):
            stride_order = tuple(range(len(kv_shape)))

        permuted_shape = tuple(kv_shape[i] for i in stride_order)
        inv_order = [stride_order.index(i) for i in range(len(stride_order))]

        total_bytes = num_blocks * attn_page_bytes
        raw = torch.zeros(total_bytes, dtype=torch.int8, device=device)
        kv = raw.view(spec_dtype).view(permuted_shape).permute(*inv_order)
        kv_caches[layer_name] = kv
        layer_to_index[layer_name] = extract_layer_index(layer_name)

    # `bind_kv_cache` writes `attn_layer.kv_cache = tensor` for every
    # attention layer and fills `runner_kv_caches` in layer-index order.
    runner_kv_caches: list[torch.Tensor] = []
    if attn_layers:
        bind_kv_cache(kv_caches, fc, runner_kv_caches)

    # ── Allocate mamba state pools ───────────────────────────────────────
    mamba_layers_out: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for layer_name, spec in mamba_specs.items():
        # Shape contract per `_reshape_kv_cache_tensors` (gpu_model_runner.py):
        # state_tensor[i] has shape (num_blocks, *spec.shapes[i]) and
        # dtype spec.dtypes[i]. Allocated as separate tensors here (no
        # shared raw buffer) since the mamba kernel reads each state by its
        # own dtype and stride; pie doesn't gain anything from packing.
        states = []
        for shape, dtype in zip(spec.shapes, spec.dtypes):
            t = torch.zeros((num_blocks, *shape), dtype=dtype, device=device)
            states.append(t)
        # vllm's GDN `_forward` reads `self.kv_cache[0]` (conv) and `[1]`
        # (ssm) without an outer virtual_engine wrapping (see
        # `vllm/model_executor/layers/mamba/gdn_linear_attn.py:794-802`).
        # bind_kv_cache only handles attention; assign directly.
        fc[layer_name].kv_cache = states
        # MambaSpec orders shapes as (conv, ssm) per
        # MambaStateShapeCalculator.gated_delta_net_state_shape; assert
        # length so a future spec change fails loud rather than miswiring.
        if len(states) != 2:
            raise NotImplementedError(
                f"Expected MambaSpec with 2 states (conv, ssm); got {len(states)} "
                f"for layer {layer_name!r} (mamba_type={spec.mamba_type!r})."
            )
        mamba_layers_out.append((layer_name, states[0], states[1]))

    return KVCacheLayout(
        attention_layers_in_order=runner_kv_caches,
        mamba_layers=mamba_layers_out,
        is_hybrid=is_hybrid,
    )


def allocate_host_pool(
    gpu_kv: list[torch.Tensor],
    swap_budget_bytes: int,
) -> tuple[list[torch.Tensor], int]:
    """CPU swap pool. Skipped on the vllm driver (Phase 1.5).

    Pie's per-layer copy_h2d/d2h handlers iterate `kv_cache_at_layer` and
    have no path for mamba state migration, so even a future host-pool
    implementation must be hybrid-aware before we re-enable it.
    """
    if swap_budget_bytes <= 0 or not gpu_kv:
        return [], 0
    raise NotImplementedError(
        "CPU swap pool not yet supported on the vllm driver (Phase 1.5). "
        "Set swap_budget_bytes=0 / cpu_mem_budget_in_gb=0 for now."
    )
