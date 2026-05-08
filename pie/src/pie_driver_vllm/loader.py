"""Resolve a vllm model class for the requested HF repo, build a VllmConfig,
load weights, and surface the backend choice.

This module owns the "what kind of vllm model are we running" decisions so
`engine.py` and `worker.py` stay backend-agnostic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from pie_driver.config import RuntimeConfig

logger = logging.getLogger(__name__)


# Map pie's RuntimeConfig.activation_dtype (torch.dtype) to vllm's string form.
# vllm's EngineArgs wants strings; we work in torch.dtype internally.
_DTYPE_TO_STR = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}


def _resolve_hf_snapshot_dir(hf_repo: str) -> str | None:
    """Find the local cache path for an HF repo's tokenizer/config.

    Uses `huggingface_hub.snapshot_download` with `local_files_only=True` —
    if vllm has already loaded weights we know the snapshot is on disk. If
    nothing is cached, fall back to scanning HF_HUB_CACHE for the repo's dir.
    """
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(hf_repo, local_files_only=True)
    except Exception:
        pass

    try:
        from pathlib import Path
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
    arch_type: str             # pie internal arch name (e.g. "llama3"); resolved via pie_driver.model.resolve()
    info: dict
    snapshot_dir: str | None


def _build_vllm_config(config: RuntimeConfig, driver_config) -> Any:
    """Build a VllmConfig from pie's `RuntimeConfig` (universal slice) and
    `VllmDriverConfig` (vllm-native knobs).

    Driver config field names mirror EngineArgs exactly, so we splat them
    into EngineArgs as kwargs.
    """
    from dataclasses import asdict
    from ._vllm_compat import EngineArgs

    if config.activation_dtype not in _DTYPE_TO_STR:
        raise ValueError(
            f"Unsupported activation_dtype for vllm driver: {config.activation_dtype}. "
            f"Expected one of {list(_DTYPE_TO_STR)}."
        )
    dtype_str = _DTYPE_TO_STR[config.activation_dtype]

    # Universal fields go through pie's RuntimeConfig.
    engine_kwargs = dict(
        model=config.hf_repo,
        dtype=dtype_str,
        tensor_parallel_size=config.tensor_parallel_size,
        seed=config.random_seed,
        skip_tokenizer_init=True,        # pie owns tokenization
        download_dir=config.cache_dir,
    )

    # Driver-specific fields splat verbatim — names match EngineArgs.
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

    # Pin attention backend to FLASHINFER when cudagraph capture is going
    # to be active (Lever 5 / ticket #100). The decode-side cudagraph path
    # captures `model.forward` whole; pie passes per-call attention
    # metadata (slot_mapping, block_table, paged_kv_*) that lands at fresh
    # GPU addresses each step, but the captured graph aliases capture-time
    # data_ptrs. FlashInfer's BatchDecodeWithPagedKVCacheWrapper runs
    # `plan()` per call and copies our metadata into ITS persistent
    # cudagraph buffers (paged_kv_indptr/indices/last_page_len), so the
    # captured graph reads stable addresses on replay. FLASH_ATTN/TRITON
    # have no such wrapper-internal persistence — they read block_table /
    # seq_lens directly off pie's per-call tensors, so the captured graph
    # silently reads stale memory and decode output degenerates. Promotion
    # is gated on (a) cudagraph mode is going to be active, AND (b) the
    # user did not pin a different backend. Eager mode keeps vllm's
    # default backend selection.
    if not driver_config.enforce_eager and driver_config.attention_backend is None:
        engine_kwargs["attention_backend"] = "FLASHINFER"

    # vllm's torch.compile + V1 attention backends (FlashInfer, FlashAttn, …)
    # size their compile ranges and per-call CPU/GPU buffers to
    # `scheduler_config.max_num_batched_tokens`. vllm defaults this to 2048
    # for chat models with chunked prefill enabled — but pie drives the model
    # outside vllm's GPUModelRunner, so its scheduler chunking does NOT
    # constrain pie's batches. Pie can fire any single-request prefill up to
    # the model's max sequence length, and a 2052-token prefill against a
    # (1, 2048) compile range trips:
    #   AssertionError: Shape: N out of considered ranges: [(1, 2048)]
    #     in vllm/compilation/piecewise_backend.py
    # On the FlashInfer side the same misconfiguration cascades into a
    # `max() iterable argument is empty` from `flashinfer/decode.py:plan` once
    # subsequent calls find FlashInfer's persistent metadata corrupted.
    #
    # Pie sets vllm's compile ceiling to the HF model's max position
    # embeddings so any single-request prefill is within compile range. We
    # probe HF config directly (rather than letting vllm resolve max_model_len
    # then mutate) because vllm derives `compile_ranges_endpoints` inside
    # `VllmConfig.__post_init__` against the scheduler value at that time;
    # setting it before EngineArgs avoids touching vllm's private
    # `_set_compile_ranges` after the fact.
    #
    # Important: we do NOT also raise pie's BatchAccumulator cap. `engine.capabilities()`
    # reads `driver_config.max_num_batched_tokens` first; only when that's None
    # does it fall back to `vc.scheduler_config.max_num_batched_tokens`. If we let
    # capabilities fall through to the raised vllm value, pie's BatchAccumulator
    # would also widen to max_position_embeddings — bringing 16-context bursts
    # into single 32K-token forwards and OOM'ing at activation time. Pin the
    # driver-side cap to vllm's chat-model default (2048) so pie's BatchAccumulator
    # behaves the same as before this change. Memory cost is therefore bounded
    # to the extra inductor compile range, which is one extra graph bucket.
    # Users who want bigger pie batches set `[model.X.driver.vllm].max_num_batched_tokens`
    # explicitly; we respect that override.
    if engine_kwargs.get("max_num_batched_tokens") is None:
        mpe = 0
        probe_err: Exception | None = None
        try:
            from transformers import AutoConfig

            hf_cfg = AutoConfig.from_pretrained(
                config.hf_repo, trust_remote_code=True, cache_dir=config.cache_dir
            )
            # max_position_embeddings is the universal field. Some models also
            # expose `model_max_length` or rope_scaling-extended context;
            # max_position_embeddings is the safe lower bound that always
            # exists for HF causal LMs.
            mpe = int(getattr(hf_cfg, "max_position_embeddings", 0) or 0)
        except Exception as _e:
            probe_err = _e
        if mpe > 0:
            engine_kwargs["max_num_batched_tokens"] = mpe
            # Pin pie's BatchAccumulator cap to vllm's pre-fix chat-model default
            # so memory pressure does not regress. driver_config is mutated in
            # place (it is not frozen); capabilities() will return this value.
            if driver_config.max_num_batched_tokens is None:
                driver_config.max_num_batched_tokens = 2048
            logger.info(
                "setting vllm max_num_batched_tokens=%d "
                "(from HF max_position_embeddings) for compile-range coverage; "
                "pinning pie BatchAccumulator cap to %d so per-batch memory stays bounded. "
                "Set [model.X.driver.vllm].max_num_batched_tokens explicitly to override either.",
                mpe,
                driver_config.max_num_batched_tokens,
            )
        else:
            # Probe failed (offline cache miss, malformed config.json, network
            # error with cache_dir unset, transformers version skew on
            # trust_remote_code archs, …). Without engine_kwargs[…] set,
            # vllm derives compile_ranges_endpoints against its 2048 chat
            # default and any single-request prefill > 2048 trips the
            # piecewise_backend assertion. The post-`create_engine_config`
            # backstop below catches this for all models with
            # `max_model_len > 2048`, but only after the operator sees what
            # looks like a regression in this PR's fix. Emit a loud,
            # grep-able warning so it's not silent.
            logger.warning(
                "HF max_position_embeddings probe for %r failed (%r); "
                "engine_kwargs['max_num_batched_tokens'] left unset. "
                "vllm will use its chat-model default (2048) and the "
                "post-create_engine_config backstop will widen the compile "
                "range based on model_config.max_model_len after construction. "
                "If you see Shape:N out-of-(1,2048) crashes after this, the "
                "backstop also failed — re-check vllm version and pie's "
                "_set_compile_ranges integration.",
                config.hf_repo,
                probe_err,
            )

    args = EngineArgs(**engine_kwargs)
    vllm_config = args.create_engine_config()

    # Defensive backstop: if vllm's resolved `max_model_len` exceeds the
    # value we passed in (e.g. rope scaling extends context beyond HF's
    # max_position_embeddings, or the user lowered max_num_batched_tokens
    # explicitly), the compile range can still be too narrow. Raise the
    # scheduler budget and recompute compile ranges. `_set_compile_ranges`
    # is internal to vllm but the only documented path that re-derives
    # compile_ranges_endpoints from scheduler_config without rebuilding
    # the whole VllmConfig — if a vllm version refactor renames or removes
    # this helper, fail loudly at engine init rather than silently leaving
    # the compile range narrow (would resurface the Shape:N crash this
    # branch was added to fix).
    sc = vllm_config.scheduler_config
    mc = vllm_config.model_config
    if sc.max_num_batched_tokens < mc.max_model_len:
        if not hasattr(vllm_config, "_set_compile_ranges"):
            raise RuntimeError(
                "pie_driver_vllm: VllmConfig is missing the private "
                "_set_compile_ranges helper that this loader relies on to "
                "re-derive compile_ranges_endpoints after raising "
                "scheduler_config.max_num_batched_tokens. The installed vllm "
                "version is incompatible with pie's compile-range backstop. "
                "Pin a compatible vllm or update pie/pie/src/pie_driver_vllm/loader.py."
            )
        old = sc.max_num_batched_tokens
        sc.max_num_batched_tokens = mc.max_model_len
        if getattr(sc, "max_num_prefill_tokens", None) == old:
            sc.max_num_prefill_tokens = mc.max_model_len
        vllm_config._set_compile_ranges()
        logger.info(
            "backstop raised max_num_batched_tokens %d -> %d "
            "(resolved max_model_len exceeded the pre-construction estimate from HF config).",
            old,
            mc.max_model_len,
        )

    # Decode-side cudagraph (Lever 5, ticket #100). Pie drives the model
    # outside vllm's GPUModelRunner, so vllm's auto-resolution of
    # cudagraph_mode never runs against pie's actual call shapes — vllm
    # leaves it at NONE/PIECEWISE depending on the version. Promote it to
    # FULL_DECODE_ONLY so the FlashInfer decode wrapper publishes its
    # cudagraph capture path (full decode forward as a single graph; mixed
    # / prefill batches stay eager). Only override when not already FULL,
    # so an explicit user choice (e.g. FULL_AND_PIECEWISE) is preserved.
    from vllm.config import CUDAGraphMode

    if not driver_config.enforce_eager:
        cc = vllm_config.compilation_config
        if cc.cudagraph_mode in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE):
            cc.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
            # FULL capture wraps the whole decode forward; the splitting_ops
            # belong to PIECEWISE (used to split inductor partitions at
            # attention boundaries) — clear them so vllm doesn't try to
            # piecewise-split a graph we want captured monolithically.
            cc.splitting_ops = []

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
    compute_pg=None,
) -> LoadedModel:
    """Construct a vllm model on the local rank's device and load its weights."""

    def _log(msg: str, level: str = "INFO"):
        if log_queue is not None:
            log_queue.put({"message": msg, "level": level})

    # Pin the device for this rank — vllm reads current_device() during
    # model construction.
    device_str = str(config.device)
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

    # Resolve HF identity to pie's internal arch name (the dispatch key
    # the Rust runtime expects). Multimodal HF configs (e.g. Qwen3.5)
    # carry `model_type` only on the inner `hf_text_config`; fall through
    # to the outer `hf_config` so text-only loads still resolve.
    from pie_driver.model import resolve as resolve_pie_arch
    try:
        arch_type = resolve_pie_arch(vllm_config.model_config.hf_text_config)
    except Exception:
        arch_type = resolve_pie_arch(vllm_config.model_config.hf_config)

    text_arches = vllm_config.model_config.hf_text_config.architectures
    outer_arches = getattr(vllm_config.model_config.hf_config, "architectures", None)
    arches = list(text_arches or outer_arches or [])

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
