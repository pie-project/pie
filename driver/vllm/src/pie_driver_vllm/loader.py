"""Resolve a vllm model class for the requested HF repo, build a VllmConfig,
load weights, and surface the backend choice.

This module owns the "what kind of vllm model are we running" decisions so
`engine.py` and `worker.py` stay backend-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import os
import sys
import torch
import torch.distributed as dist

from ._bridge.config import RuntimeConfig
from .utils import configure_distributed_environment


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
        # Pie still owns request tokenization, but some vLLM model classes
        # need a tokenizer during construction to pre-tokenize multimodal
        # sentinels (e.g. Nemotron-H Nano Omni). Keep this aligned with
        # standalone vLLM and pass token IDs at runtime as before.
        skip_tokenizer_init=False,
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
        "text_only_mm",
        "decode_lookahead_tokens",
    }
    for k, v in asdict(driver_config).items():
        if k in _PIE_ONLY_KEYS:
            continue
        if v is not None:
            engine_kwargs[k] = v

    if getattr(driver_config, "text_only_mm", False):
        engine_kwargs["limit_mm_per_prompt"] = {
            "image": 0,
            "video": 0,
            "audio": 0,
        }

    system_topology = configure_distributed_environment(
        int(engine_kwargs["tensor_parallel_size"]), config.devices
    )
    if (
        system_topology
        and "disable_custom_all_reduce" not in engine_kwargs
        and os.environ.get("PIE_VLLM_DISABLE_CUSTOM_AR_ON_SYSTEM")
    ):
        engine_kwargs["disable_custom_all_reduce"] = True
    # When piggybacking on vllm's CUDA-graph dispatch, vllm bakes a fixed
    # `compile_ranges` into the piecewise backend at load time and asserts at
    # fire time that `num_tokens` falls inside one of them. Keep Pie's default
    # at vLLM's standard 8192-token compile range unless the user explicitly
    # sets a different batch-token capacity.
    if (
        not engine_kwargs.get("enforce_eager", False)
        and engine_kwargs.get("max_num_batched_tokens") is None
    ):
        engine_kwargs["max_num_batched_tokens"] = 8192

    args = EngineArgs(**engine_kwargs)
    vllm_config = args.create_engine_config()
    vllm_config.parallel_config.rank = config.rank

    if (
        system_topology
        and not engine_kwargs.get("enforce_eager", False)
        and os.environ.get("PIE_VLLM_DISABLE_CUDAGRAPH_ON_SYSTEM")
    ):
        from ._vllm_compat import CUDAGraphMode

        # Keep torch.compile enabled, but allow an explicit escape hatch for
        # SYS-level TP pairs if CUDA graph capture wedges NCCL collectives.
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        vllm_config.compilation_config.cudagraph_capture_sizes = []
        vllm_config.compilation_config.max_cudagraph_capture_size = None

    # Pie launches each DP replica as an independent single-rank subprocess.
    # vLLM names those compile-cache leaves `rank_0_0` in every replica, so
    # two DP replicas on different CUDA devices can otherwise reuse the same
    # TorchInductor/static-launcher artifacts. Keep TP-only runs on vLLM's
    # normal cache path so they can reuse warmed standalone compile artifacts.
    if (
        not engine_kwargs.get("enforce_eager", False)
        and not vllm_config.compilation_config.cache_dir
        and config.num_groups > 1
    ):
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
    import os
    import sys
    from ._vllm_compat import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    def _debug_stage(msg: str) -> None:
        if os.environ.get("PIE_VLLM_DEBUG_LOAD"):
            print(
                f"[pie-vllm-load rank={rank}] ensure: {msg}",
                file=sys.stderr,
                flush=True,
            )

    parallel_config = vllm_config.parallel_config

    # vLLM's normal GPUWorker path applies this before constructing model
    # parallel groups. Pie loads the model directly, so mirror that hook here;
    # otherwise `disable_custom_all_reduce=True` is present in config but the
    # TP communicator still builds vLLM's custom all-reduce path.
    from vllm.distributed.parallel_state import set_custom_all_reduce

    _debug_stage("set_custom_all_reduce: begin")
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)
    _debug_stage("set_custom_all_reduce: done")

    if not dist.is_initialized() and parallel_config.world_size == 1:
        # Single-rank fallback. FileStore avoids picking a port (no contention).
        store_path = tempfile.mktemp(prefix="pie_vllm_singlerank_")
        store = dist.FileStore(store_path, parallel_config.world_size)
        device_id = (
            torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else None
        )
        _debug_stage("torch init_process_group: begin")
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            store=store,
            rank=rank,
            world_size=parallel_config.world_size,
            timeout=datetime.timedelta(seconds=300),
            device_id=device_id,
        )
        _debug_stage("torch init_process_group: done")

    _debug_stage("init_distributed_environment: begin")
    init_distributed_environment(
        world_size=parallel_config.world_size,
        rank=rank,
        distributed_init_method="env://",
        local_rank=local_rank,
    )
    _debug_stage("init_distributed_environment: done")

    _debug_stage("ensure_model_parallel_initialized: begin")
    ensure_model_parallel_initialized(
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size,
        parallel_config.prefill_context_parallel_size,
        parallel_config.decode_context_parallel_size,
    )
    _debug_stage("ensure_model_parallel_initialized: done")

    # Reuse vLLM's CPU world group for Pie's metadata broadcasts. Creating
    # an extra Gloo group before vLLM initializes can deadlock group creation
    # order; assigning this after vLLM setup keeps both systems aligned.
    try:
        from vllm.distributed.parallel_state import get_world_group
        from . import utils as runtime_utils

        runtime_utils._cpu_group = get_world_group().cpu_group
    except Exception:
        pass


def load_vllm_model(
    config: RuntimeConfig,
    driver_config,
    log_queue: object = None,
) -> LoadedModel:
    """Construct a vllm model on the local rank's device and load its weights."""

    def _log(msg: str, level: str = "INFO"):
        if log_queue is not None:
            log_queue.put({"message": msg, "level": level})

    def _debug_stage(msg: str) -> None:
        if os.environ.get("PIE_VLLM_DEBUG_LOAD"):
            print(
                f"[pie-vllm-load rank={config.rank}] {msg}",
                file=sys.stderr,
                flush=True,
            )

    # Pin the device for this rank — vllm reads current_device() during
    # model construction.
    device_str = config.device
    if device_str.startswith("cuda"):
        torch.cuda.set_device(device_str)
        local_rank = int(device_str.split(":")[1]) if ":" in device_str else 0
    else:
        local_rank = config.rank

    # GPUModelRunner normally initializes this before any MoE forward path.
    # Pie embeds the model directly, so mirror the hook here; otherwise
    # vLLM's modular MoE kernels fail on first request with an uninitialized
    # workspace manager.
    try:
        from vllm.v1.worker.workspace import (
            init_workspace_manager,
            is_workspace_manager_initialized,
        )

        if not is_workspace_manager_initialized():
            init_workspace_manager(torch.device(device_str))
    except Exception:
        pass

    _log(f"Building VllmConfig for {config.hf_repo}", "DEBUG")
    _debug_stage("build_vllm_config: begin")
    vllm_config = _build_vllm_config(config, driver_config)
    _debug_stage("build_vllm_config: done")

    # vllm's parallel state and model construction both consult
    # `get_current_vllm_config()` — they must run inside `set_current_vllm_config`.
    from ._vllm_compat import get_model_loader, set_current_vllm_config

    with set_current_vllm_config(vllm_config):
        _log("Bringing up vllm parallel state", "DEBUG")
        _debug_stage("ensure_vllm_distributed: begin")
        _ensure_vllm_distributed(vllm_config, rank=config.rank, local_rank=local_rank)
        _debug_stage("ensure_vllm_distributed: done")

        _log(f"Loading model weights ({config.hf_repo})", "INFO")
        _debug_stage("loader.load_model: begin")
        loader = get_model_loader(vllm_config.load_config)
        model = loader.load_model(
            vllm_config=vllm_config,
            model_config=vllm_config.model_config,
        )
        if getattr(driver_config, "text_only_mm", False):
            language_model = getattr(model, "language_model", None)
            if language_model is not None:
                _debug_stage(
                    "loader.load_model: using inner language_model for text-only"
                )
                model = language_model
        _debug_stage("loader.load_model: done")

        # Hybrid attention/Mamba models can only finalize the KV block geometry
        # after layers resolve their concrete attention backend. Standalone vLLM
        # runs this platform hook before KV cache initialization; Pie loads the
        # model directly, so mirror it here before allocate_and_bind_kv_cache()
        # asks each layer for its cache spec.
        _debug_stage("update_block_size_for_backend: begin")
        from vllm.platforms import current_platform

        current_platform.update_block_size_for_backend(vllm_config)
        vllm_config.validate_block_size()
        _debug_stage(
            "update_block_size_for_backend: done "
            f"block_size={vllm_config.cache_config.block_size} "
            f"mamba_block_size={vllm_config.cache_config.mamba_block_size} "
            "mamba_page_size_padded="
            f"{vllm_config.cache_config.mamba_page_size_padded}"
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
    # the cached snapshot via huggingface_hub.
    snapshot_dir = _resolve_hf_snapshot_dir(config.hf_repo)
    _debug_stage("resolve_snapshot_dir: done")
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
