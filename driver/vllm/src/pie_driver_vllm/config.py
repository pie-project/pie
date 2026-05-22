"""vLLM driver config — typed view of a curated
`vllm.engine.arg_utils.EngineArgs` subset. Field names mirror EngineArgs
where Pie exposes a backend policy knob:

    [model.driver.options]    # with [model.driver].type = "vllm"
    attention_backend = "FLASHINFER"      → EngineArgs.attention_backend
    enforce_eager = false                 → EngineArgs.enforce_eager
    gpu_memory_utilization = 0.85         → EngineArgs.gpu_memory_utilization
    ...

Batch capacity knobs are optional: when omitted, vLLM resolves them at
startup and Pie reports the result through DriverCapabilities.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VllmDriverConfig:
    """vLLM-specific knobs, expressed in vllm's vocabulary."""

    # Attention backend selection (FLASHINFER / FLASH_ATTN / TRITON_ATTN /
    # FLEX_ATTENTION / etc.). None = let vllm auto-pick per platform.
    attention_backend: str | None = None

    # If True, vllm runs eager (no torch.compile, no CUDA graphs).
    # None lets the driver choose a safe default for the local topology.
    enforce_eager: bool | None = None

    # If True, vllm uses NCCL collectives instead of its custom all-reduce.
    # None lets the driver auto-disable custom all-reduce on system-level PCIe
    # topologies where NCCL P2P must also be disabled to avoid TP startup hangs.
    disable_custom_all_reduce: bool | None = None

    # Fraction of free GPU memory to use for KV cache + activations.
    gpu_memory_utilization: float = 0.9

    # Capacity knobs forwarded to vLLM EngineArgs. None = let vLLM choose.
    max_model_len: int | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None

    # ---- Speculative decoding (NGRAM, driver-supplied drafts) ----
    # When True, VllmEngine.spec_step proposes linear draft continuations.
    # Verification + splice run in the shared `._bridge.batching.Batch`
    # path; the engine only owns drafting. Field names mirror
    # SGLangDriverConfig so configs port across drivers.
    spec_ngram_enabled: bool = False
    # Drafts proposed per accepted iteration.
    spec_ngram_num_drafts: int = 4
    # n-gram match window (mirrors vllm's
    # SpeculativeConfig.prompt_lookup_min/max). Requires 1 ≤ min ≤ max.
    spec_ngram_min_n: int = 2
    spec_ngram_max_n: int = 4
