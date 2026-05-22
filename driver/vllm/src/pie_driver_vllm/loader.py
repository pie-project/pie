"""Resolve a vllm model class for the requested HF repo, build a VllmConfig,
load weights, and surface the backend choice.

This module owns the "what kind of vllm model are we running" decisions so
`engine.py` and `worker.py` stay backend-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from ._bridge.config import RuntimeConfig


# `RuntimeConfig.activation_dtype` is a string identifier; vllm's
# EngineArgs accepts the same vocabulary. This set gates the supported
# values.
_SUPPORTED_DTYPES = {"bfloat16", "float16", "float32"}


def _resolve_hf_snapshot_dir(hf_repo: str) -> str | None:
    """Find the local cache path for an HF repo's tokenizer/config.

    Uses `huggingface_hub.snapshot_download` with `local_files_only=True` —
    if vllm has already loaded weights we know the snapshot is on disk. If
    nothing is cached, fall back to scanning HF_HUB_CACHE for the repo's dir.
    """
    from pathlib import Path

    local = Path(hf_repo)
    if local.is_dir():
        return str(local)

    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(hf_repo, local_files_only=True)
    except Exception:
        pass

    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        repo_dir = Path(HF_HUB_CACHE) / f"models--{hf_repo.replace('/', '--')}" / "snapshots"
        if repo_dir.is_dir():
            snapshots = sorted(repo_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if snapshots:
                return str(snapshots[0])
    except Exception:
        pass

    return None


@dataclass
class LoadedModel:
    """Bundle of everything `Engine.load` needs after vllm finishes loading."""

    model: torch.nn.Module
    vllm_config: Any           # vllm.config.VllmConfig
    attn_backend: Any          # resolved AttentionBackend class (or None — resolved lazily)
    model_config: Any          # vllm's ModelConfig — exposes vocab_size, num_layers, etc.
    arch_type: str             # HF architecture string, e.g. "LlamaForCausalLM"
    info: dict
    snapshot_dir: str | None


def _build_vllm_config(config: RuntimeConfig, driver_config) -> Any:
    """Build a VllmConfig from pie's `RuntimeConfig` (universal slice) and
    `VllmDriverConfig` (vllm-native knobs).

    Exposed driver config field names mirror EngineArgs, so we splat policy
    knobs into EngineArgs as kwargs. Capacity limits are resolved by vLLM and
    reported later through DriverCapabilities.
    """
    from dataclasses import asdict
    from ._vllm_compat import EngineArgs

    if config.activation_dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"Unsupported activation_dtype for vllm driver: {config.activation_dtype}. "
            f"Expected one of {sorted(_SUPPORTED_DTYPES)}."
        )

    # Universal fields go through pie's RuntimeConfig.
    engine_kwargs = dict(
        model=config.hf_repo,
        dtype=config.activation_dtype,
        tensor_parallel_size=config.tensor_parallel_size,
        seed=config.random_seed,
        skip_tokenizer_init=True,        # pie owns tokenization
        download_dir=config.cache_dir,
    )

    # Driver-specific policy fields splat verbatim — names match EngineArgs.
    # `None` means "leave default" (vllm picks). Pie-only knobs that don't
    # correspond to vllm EngineArgs are filtered out: spec_ngram_* drive
    # pie's NGRAM drafter (engine-side, see VllmEngine.spec_step).
    _PIE_ONLY_KEYS = {
        "spec_ngram_enabled",
        "spec_ngram_num_drafts",
        "spec_ngram_min_n",
        "spec_ngram_max_n",
    }
    for k, v in asdict(driver_config).items():
        if k in _PIE_ONLY_KEYS:
            continue
        if v is not None:
            engine_kwargs[k] = v

    # When piggybacking on vllm's CUDA-graph dispatch (enforce_eager=False),
    # vllm bakes a fixed `compile_ranges` into the piecewise backend at load
    # time and asserts at fire time that `num_tokens` falls inside one of
    # them — chunked prefill's default 2048 is far below pie's batched
    # peak. Default `max_num_batched_tokens` to the model's context length
    # so any pie batch fits the compile range.
    if (
        not engine_kwargs.get("enforce_eager", False)
        and engine_kwargs.get("max_num_batched_tokens") is None
    ):
        from transformers import AutoConfig
        try:
            hf_cfg = AutoConfig.from_pretrained(config.hf_repo, trust_remote_code=True)
            mml = int(getattr(hf_cfg, "max_position_embeddings", 0)) or None
        except Exception:
            mml = None
        if mml is not None:
            engine_kwargs["max_num_batched_tokens"] = mml

    args = EngineArgs(**engine_kwargs)
    vllm_config = args.create_engine_config()

    # Pie launches each DP replica as an independent single-rank subprocess.
    # vLLM names those compile-cache leaves `rank_0_0` in every replica, so
    # two DP replicas on different CUDA devices can otherwise reuse the same
    # TorchInductor/static-launcher artifacts. Keep compile + CUDA graphs on,
    # but isolate the cache per process/device.
    if (
        not engine_kwargs.get("enforce_eager", False)
        and not vllm_config.compilation_config.cache_dir
    ):
        import os

        safe_device = config.device.replace(":", "_")
        vllm_config.compilation_config.cache_dir = (
            f"/tmp/pie_vllm_compile_cache/pid_{os.getpid()}_{safe_device}"
        )

    return vllm_config


def _ensure_vllm_distributed(vllm_config: Any, rank: int, local_rank: int) -> None:
    """Bring vllm's parallel state up on top of pie's torch.distributed init.

    Pie's worker calls `torch.distributed.init_process_group` only when
    world_size > 1 (single-GPU path skips it for latency). vllm's parallel
    state machinery requires *some* process group, so for the single-rank
    case we bring up a tcp://localhost rendezvous of size 1 here.

    `init_distributed_environment` is idempotent — if dist is already
    initialized (multi-GPU path), it just records the world/rank.
    `ensure_model_parallel_initialized` then constructs vllm's TP/PP/DP
    groups that model layers consult during construction.
    """
    import tempfile, datetime
    from ._vllm_compat import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    parallel_config = vllm_config.parallel_config

    if not dist.is_initialized():
        # Single-rank fallback. FileStore avoids picking a port (no contention).
        store_path = tempfile.mktemp(prefix="pie_vllm_singlerank_")
        store = dist.FileStore(store_path, parallel_config.world_size)
        device_id = (
            torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else None
        )
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            store=store,
            rank=rank,
            world_size=parallel_config.world_size,
            timeout=datetime.timedelta(seconds=300),
            device_id=device_id,
        )

    init_distributed_environment(
        world_size=parallel_config.world_size,
        rank=rank,
        distributed_init_method="env://",
        local_rank=local_rank,
    )

    ensure_model_parallel_initialized(
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size,
        parallel_config.prefill_context_parallel_size,
        parallel_config.decode_context_parallel_size,
    )


def load_vllm_model(
    config: RuntimeConfig,
    driver_config,
    log_queue: object = None,
) -> LoadedModel:
    """Construct a vllm model on the local rank's device and load its weights."""

    def _log(msg: str, level: str = "INFO"):
        if log_queue is not None:
            log_queue.put({"message": msg, "level": level})

    # Pin the device for this rank — vllm reads current_device() during
    # model construction.
    device_str = config.device
    if device_str.startswith("cuda"):
        torch.cuda.set_device(device_str)
        local_rank = int(device_str.split(":")[1]) if ":" in device_str else 0
    else:
        local_rank = config.rank

    _log(f"Building VllmConfig for {config.hf_repo}", "DEBUG")
    vllm_config = _build_vllm_config(config, driver_config)

    # vllm's parallel state and model construction both consult
    # `get_current_vllm_config()` — they must run inside `set_current_vllm_config`.
    from ._vllm_compat import get_model_loader, set_current_vllm_config

    with set_current_vllm_config(vllm_config):
        _log("Bringing up vllm parallel state", "DEBUG")
        _ensure_vllm_distributed(vllm_config, rank=config.rank, local_rank=local_rank)

        _log(f"Loading model weights ({config.hf_repo})", "INFO")
        loader = get_model_loader(vllm_config.load_config)
        model = loader.load_model(
            vllm_config=vllm_config,
            model_config=vllm_config.model_config,
        )
        _log("Model weights loaded", "INFO")

    # Architecture string (first arch in the HF config). Used for telemetry
    # and as the `arch_type` reported back through pie's ready handshake.
    # Multimodal HF configs (e.g. Qwen3.5) carry `architectures` only on the
    # top-level config; the inner `text_config` is None there. Fall back to
    # the outer `hf_config` so multimodal text-only loads still report a name.
    text_arches = vllm_config.model_config.hf_text_config.architectures
    outer_arches = getattr(vllm_config.model_config.hf_config, "architectures", None)
    arches = list(text_arches or outer_arches or [])
    if not arches:
        raise RuntimeError(
            f"vllm's HF config for {config.hf_repo} has no `architectures` field. "
            "Pie needs at least one HF architecture string."
        )
    arch_type = arches[0]

    info = {
        "architecture": {"type": arch_type, "all": arches},
        "vocab_size": vllm_config.model_config.get_vocab_size(),
        "max_model_len": vllm_config.model_config.max_model_len,
        "num_hidden_layers": vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        ),
    }

    # Snapshot dir: pie's Rust runtime needs a local filesystem path that
    # contains tokenizer.json (and the HF config). vllm's `model_config.model`
    # is just the HF repo name like "Qwen/Qwen3-0.6B", so we resolve it to
    # the cached snapshot via huggingface_hub. (We avoid `pie_driver_dev.hf_utils`
    # because it transitively imports `pie_kernels` which JIT-compiles CUDA
    # extensions; that's heavy and irrelevant on the vllm path.)
    snapshot_dir = _resolve_hf_snapshot_dir(config.hf_repo)
    if snapshot_dir is None:
        raise RuntimeError(
            f"Could not resolve a local snapshot dir for {config.hf_repo!r}. "
            "vllm has loaded weights but huggingface_hub.snapshot_download "
            "(local_files_only) found no cached snapshot. Pie's Rust runtime "
            "needs the snapshot path for tokenizer.json. Verify the HF cache "
            "is populated (e.g., run `huggingface-cli download <repo>`)."
        )

    # `attn_backend` is resolved lazily inside vllm's Attention layers — we
    # don't pin it here; metadata builder reads it from the layer at first
    # forward.
    attn_backend = None

    return LoadedModel(
        model=model,
        vllm_config=vllm_config,
        attn_backend=attn_backend,
        model_config=vllm_config.model_config,
        arch_type=arch_type,
        info=info,
        snapshot_dir=snapshot_dir,
    )
