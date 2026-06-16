"""Helpers for vLLM hybrid scheduler/kernel block sizes."""

from __future__ import annotations

from typing import Iterable


def select_kernel_block_size(spec, backends: Iterable, vllm_config) -> int:
    """Return the attention-kernel block size for a KV-manager block.

    vLLM can allocate KV cache in large scheduler blocks while letting the
    attention backend view each scheduler block as smaller kernel blocks. Hybrid
    attention/Mamba models depend on this: the scheduler block is sized to match
    the Mamba state page, while FlashAttention still runs on 16-token blocks.
    """
    from ._vllm_compat import set_current_vllm_config
    from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

    if isinstance(spec, AttentionSpec):
        from vllm.v1.worker.utils import select_common_block_size

        with set_current_vllm_config(vllm_config):
            return int(select_common_block_size(int(spec.block_size), list(backends)))
    if isinstance(spec, MambaSpec):
        return int(spec.block_size)
    return int(spec.block_size)
