"""Resolve an SGLang model + ModelRunner for the requested HF repo.

Builds an `sglang.srt.server_args.ServerArgs` from the universal
`RuntimeConfig` plus the typed `SGLangDriverConfig` and instantiates
`ModelRunner`. SGLang's `ModelRunner.__init__` brings up its own
`torch.distributed` parallel state — pie's `worker.run_worker` brought
torch.distributed up first; sglang's idempotent init layers TP/PP groups
on top.

This is the analog of `pie_driver_vllm/loader.py`. SGLangDriverConfig
fields mirror `ServerArgs` so values splat verbatim.
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from pie_driver.config import RuntimeConfig

from .config import SGLangDriverConfig


# Map pie's RuntimeConfig.activation_dtype (torch.dtype) to sglang's string form.
_DTYPE_TO_STR = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}


def _resolve_hf_snapshot_dir(hf_repo: str) -> str | None:
    """Find the local cache path for an HF repo's tokenizer/config.

    Pie's Rust runtime needs a local filesystem path containing tokenizer.json.
    """
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
    """Bundle of everything `Engine.load` needs after sglang finishes loading."""

    runner: Any                # sglang.srt.model_executor.model_runner.ModelRunner
    sglang_model_config: Any   # sglang.srt.configs.model_config.ModelConfig
    arch_type: str             # HF architecture string, e.g. "Qwen3ForCausalLM"
    snapshot_dir: str | None
    # If True, the loader skipped `runner.init_device_graphs()` and the
    # engine must call it after installing pie's hooks (see engine.py).
    graph_capture_deferred: bool = False


@contextlib.contextmanager
def _defer_graph_capture(should_defer: bool):
    """Suppress `ModelRunner.init_device_graphs` for the duration of
    `ModelRunner.__init__` so pie's `_HiddenCapture` hook can be installed
    before graph capture binds the original LogitsProcessor."""
    from sglang.srt.model_executor.model_runner import ModelRunner
    if not should_defer:
        yield
        return
    original = ModelRunner.init_device_graphs
    ModelRunner.init_device_graphs = lambda self: None  # type: ignore[assignment]
    try:
        yield
    finally:
        ModelRunner.init_device_graphs = original  # type: ignore[assignment]


def _build_sglang_server_args(config: RuntimeConfig, driver_config: SGLangDriverConfig) -> Any:
    """Build an SGLang ServerArgs from the universal RuntimeConfig + driver knobs.

    Driver-config field names match `ServerArgs` so we splat them in directly.
    """
    from sglang.srt.server_args import ServerArgs

    if config.activation_dtype not in _DTYPE_TO_STR:
        raise ValueError(
            f"Unsupported activation_dtype for sglang driver: {config.activation_dtype}. "
            f"Expected one of {list(_DTYPE_TO_STR)}."
        )

    # sglang's `device` is the device kind (cuda/cpu/...), not the index.
    # The index flows through `gpu_id` to ModelRunner.
    device_kind = "cuda" if str(config.device).startswith("cuda") else str(config.device)

    server_kwargs = dict(
        model_path=config.hf_repo,
        skip_tokenizer_init=True,        # pie owns tokenization
        dtype=_DTYPE_TO_STR[config.activation_dtype],
        device=device_kind,
        tp_size=config.tensor_parallel_size,
        random_seed=config.random_seed,
        download_dir=config.cache_dir,
        # We don't run sglang's HTTP server; these fields are inert here.
        host="127.0.0.1",
        port=30000,
        skip_server_warmup=True,
    )
    # Driver-specific fields splat verbatim — names match ServerArgs. Skip
    # universal pie knobs that don't correspond to sglang ServerArgs:
    #   - cpu_mem_budget_in_gb sizes pie's host KV pool.
    #   - spec_ngram_* drive pie's own NGRAM drafter (engine-side, see
    #     SGLangEngine.spec_step); they don't map to sglang's
    #     speculative_algorithm path.
    _NON_SGLANG_FIELDS = {
        "cpu_mem_budget_in_gb",
        "spec_ngram_enabled",
        "spec_ngram_num_drafts",
        "spec_ngram_max_depth",
        "spec_ngram_capacity",
    }
    server_kwargs.update({
        k: v for k, v in asdict(driver_config).items()
        if v is not None and k not in _NON_SGLANG_FIELDS
    })
    # Pie owns prefix sharing via its own scheduler; sglang's radix cache
    # on top would just be wasted work. Hardcoded rather than configurable
    # because there's no scenario where False is correct under pie.
    server_kwargs["disable_radix_cache"] = True
    return ServerArgs(**server_kwargs)


def load_sglang_model(
    config: RuntimeConfig,
    driver_config: SGLangDriverConfig,
    log_queue: object = None,
) -> LoadedModel:
    """Construct an SGLang ModelRunner on the local rank's device."""

    def _log(msg: str, level: str = "INFO"):
        if log_queue is not None:
            log_queue.put({"message": msg, "level": level})

    # Pin the device for this rank.
    device_str = str(config.device)
    if device_str.startswith("cuda"):
        torch.cuda.set_device(device_str)
        gpu_id = int(device_str.split(":")[1]) if ":" in device_str else 0
    else:
        gpu_id = config.rank

    # ModelRunner.init_torch_distributed reads MASTER_ADDR/MASTER_PORT/RANK/etc.
    # from env on its idempotent / from-scratch path. Pie's worker.run_worker
    # has already brought torch.distributed up via FileStore at the TP-group
    # scale (see `_init_distributed`), so these env vars are only consulted
    # if sglang touches a code path that re-reads them; we set the values
    # consistent with our per-replica torch.dist init.
    import os
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", str(config.rank))
    os.environ.setdefault("WORLD_SIZE", str(config.tensor_parallel_size))
    os.environ.setdefault("LOCAL_RANK", str(gpu_id))

    _log(f"Building SGLang ServerArgs for {config.hf_repo}", "DEBUG")
    server_args = _build_sglang_server_args(config, driver_config)

    from sglang.srt.configs.model_config import ModelConfig as SGLangModelConfig
    sglang_model_config = SGLangModelConfig.from_server_args(server_args)

    _log(f"Loading model weights via SGLang ({config.hf_repo})", "INFO")

    from sglang.srt.model_executor.model_runner import ModelRunner

    # Skip ModelRunner's in-construction graph capture so the engine can
    # install pie's `_HiddenCapture` hook first.
    want_graph = not server_args.disable_cuda_graph
    with _defer_graph_capture(want_graph):
        runner = ModelRunner(
            model_config=sglang_model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=config.rank,
            tp_size=config.tensor_parallel_size,
            moe_ep_rank=0,
            moe_ep_size=1,
            pp_rank=0,
            pp_size=1,
            nccl_port=int(os.environ["MASTER_PORT"]) + 1,
            server_args=server_args,
        )
    _log("Model loaded via SGLang", "INFO")

    arches = list(sglang_model_config.hf_config.architectures)
    arch_type = arches[0] if arches else "Unknown"

    snapshot_dir = _resolve_hf_snapshot_dir(config.hf_repo)
    if snapshot_dir is None:
        _log(f"Could not resolve HF snapshot dir for {config.hf_repo}", "WARN")

    return LoadedModel(
        runner=runner,
        sglang_model_config=sglang_model_config,
        arch_type=arch_type,
        snapshot_dir=snapshot_dir,
        graph_capture_deferred=want_graph,
    )
