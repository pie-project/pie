"""vLLM-backed inference engine.

Mirrors `pie_driver_dev.engine.Engine`'s public surface so that worker.py can use
either driver interchangeably. Internally, the model and kernels come from
vllm; the surrounding RPC, batching, telemetry, and adapter scaffolding are
imported directly from `pie_driver`.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from ._bridge.config import RuntimeConfig
from ._bridge.batching import Batch
from ._bridge import telemetry

from . import _require_vllm
from . import batch_tensors


class VllmEngine:
    """Inference engine that delegates the forward pass to a vllm model.

    Public surface matches `pie_driver_dev.engine.Engine`:
      - `Engine.load(config, ...)` classmethod
      - `engine.fire_batch(inputs, sampling_metadata) -> list`
      - `engine.kv_cache_at_layer`, `engine.kv_cache_at_layer_host`
      - `engine.adapters`, `engine.swap_pool_size`
      - `engine.config`, `engine.model_config`, `engine.arch_type`,
        `engine.snapshot_dir`
      - `engine.query`, `engine.init_adapter`, `engine.update_adapter`,
        `engine.load_adapter`, `engine.save_adapter`
    """

    config: RuntimeConfig
    forward_pass: object
    model_config: object
    kv_cache_at_layer: list[torch.Tensor]
    kv_cache_at_layer_host: list[torch.Tensor]
    swap_pool_size: int
    adapter_at_layer: list
    adapters: dict
    arch_type: str
    info: dict
    snapshot_dir: str | None

    def __init__(
        self,
        config: RuntimeConfig,
        driver_config,
        model_config,
        forward_pass,
        kv_cache_at_layer: list,
        adapter_at_layer: list,
        arch_type: str,
        info: dict,
        snapshot_dir: str | None = None,
        kv_cache_at_layer_host: list | None = None,
        swap_pool_size: int = 0,
    ):
        self.config = config
        self.driver_config = driver_config
        self.model_config = model_config
        self.forward_pass = forward_pass
        self.kv_cache_at_layer = kv_cache_at_layer
        self.kv_cache_at_layer_host = kv_cache_at_layer_host or []
        self.swap_pool_size = swap_pool_size
        self.adapter_at_layer = adapter_at_layer
        self.arch_type = arch_type
        self.info = info
        self.snapshot_dir = snapshot_dir
        self.adapters = {}

        # Speculative decoding: driver-side n-gram drafter. Verification
        # and splice live in the shared `._bridge.batching.Batch`; this
        # engine owns drafting via `spec_step`. Buffers are lazy-init so
        # the numba JIT cost is only paid when spec is actually used.
        self._ngram_buffers = None
        self._ngram_history: dict[int, list[int]] = {}

    @classmethod
    def load(
        cls,
        config: RuntimeConfig,
        driver_config,
        log_queue: object = None,
    ) -> "VllmEngine":
        _require_vllm()

        from .forward_pass import VllmForwardPass
        from .kv_cache import allocate_and_bind_kv_cache, allocate_host_pool
        from .loader import load_vllm_model

        def _log(msg: str, level: str = "INFO"):
            if log_queue is not None:
                log_queue.put({"message": msg, "level": level})

        if config.rank == 0:
            telemetry.init_telemetry(
                enabled=config.telemetry_enabled,
                service_name=config.telemetry_service_name,
                endpoint=config.telemetry_endpoint,
            )

        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.random_seed)

        _log("Loading vllm model", "DEBUG")
        loaded = load_vllm_model(
            config, driver_config, log_queue=log_queue,
        )
        _log("Loaded vllm model", "DEBUG")

        kv_cache_at_layer = allocate_and_bind_kv_cache(loaded, config, driver_config)
        host_kv, pool_size = allocate_host_pool(kv_cache_at_layer, config.swap_budget_bytes)

        # Wire vllm's CUDA-graph dispatcher when capture is enabled (i.e.
        # `enforce_eager=False`). With it disabled, set cg_dispatcher=None
        # and forward_pass falls through to the eager path everywhere.
        cg_dispatcher = _maybe_init_cg_dispatcher(loaded.vllm_config)

        forward_pass = VllmForwardPass(
            model=loaded.model,
            vllm_config=loaded.vllm_config,
            attn_backend=loaded.attn_backend,
            runtime_config=config,
            model_config=loaded.model_config,
            cg_dispatcher=cg_dispatcher,
        )

        if cg_dispatcher is not None:
            max_cg_n = (
                loaded.vllm_config.compilation_config.max_cudagraph_capture_size
                or 0
            )
            forward_pass.setup_cg_buffers(
                max_n=max_cg_n,
                hidden_size=int(loaded.vllm_config.model_config.get_hidden_size()),
            )

            _log("Capturing vllm CUDA graphs", "INFO")
            _capture_vllm_cudagraphs(
                forward_pass=forward_pass,
                cg_dispatcher=cg_dispatcher,
                vllm_config=loaded.vllm_config,
                config=config,
            )
            _log("Capture done", "INFO")

        return cls(
            config=config,
            driver_config=driver_config,
            model_config=loaded.model_config,
            forward_pass=forward_pass,
            kv_cache_at_layer=kv_cache_at_layer,
            adapter_at_layer=[],
            arch_type=loaded.arch_type,
            info=loaded.info,
            snapshot_dir=loaded.snapshot_dir,
            kv_cache_at_layer_host=host_kv,
            swap_pool_size=pool_size,
        )

    # ------------------------------------------------------------------
    # KV swap — page copies. Bridge stays torch-free; the index_copy_
    # calls live here next to the engine's KV tensors.
    # ------------------------------------------------------------------

    def kv_copy_d2h(self, phys_ids: list[int], slots: list[int]) -> None:
        gpu_kv = self.kv_cache_at_layer
        host_kv = self.kv_cache_at_layer_host
        max_gpu = gpu_kv[0].shape[0]
        max_cpu = host_kv[0].shape[0] if host_kv else 0
        for p in phys_ids:
            if p < 0 or p >= max_gpu:
                raise ValueError(f"swap_out: GPU phys_id {p} out of bounds [0, {max_gpu})")
        for s in slots:
            if s < 0 or s >= max_cpu:
                raise ValueError(f"swap_out: CPU slot {s} out of bounds [0, {max_cpu})")
        src = torch.tensor(phys_ids, dtype=torch.long, device=gpu_kv[0].device)
        dst = torch.tensor(slots, dtype=torch.long)
        for layer_idx in range(len(gpu_kv)):
            host_kv[layer_idx].index_copy_(0, dst, gpu_kv[layer_idx][src].cpu())

    def kv_copy_h2d(self, phys_ids: list[int], slots: list[int]) -> None:
        gpu_kv = self.kv_cache_at_layer
        host_kv = self.kv_cache_at_layer_host
        max_gpu = gpu_kv[0].shape[0]
        max_cpu = host_kv[0].shape[0] if host_kv else 0
        for p in phys_ids:
            if p < 0 or p >= max_gpu:
                raise ValueError(f"swap_in: GPU phys_id {p} out of bounds [0, {max_gpu})")
        for s in slots:
            if s < 0 or s >= max_cpu:
                raise ValueError(f"swap_in: CPU slot {s} out of bounds [0, {max_cpu})")
        dst = torch.tensor(phys_ids, dtype=torch.long, device=gpu_kv[0].device)
        src = torch.tensor(slots, dtype=torch.long)
        for layer_idx in range(len(gpu_kv)):
            gpu_kv[layer_idx].index_copy_(0, dst, host_kv[layer_idx][src].to(gpu_kv[layer_idx].device))

    def kv_copy_d2d(self, src_phys_ids: list[int], dst_phys_ids: list[int]) -> None:
        gpu_kv = self.kv_cache_at_layer
        max_gpu = gpu_kv[0].shape[0]
        for p in src_phys_ids:
            if p < 0 or p >= max_gpu:
                raise ValueError(f"copy_d2d: src phys_id {p} out of bounds [0, {max_gpu})")
        for p in dst_phys_ids:
            if p < 0 or p >= max_gpu:
                raise ValueError(f"copy_d2d: dst phys_id {p} out of bounds [0, {max_gpu})")
        src = torch.tensor(src_phys_ids, dtype=torch.long, device=gpu_kv[0].device)
        dst = torch.tensor(dst_phys_ids, dtype=torch.long, device=gpu_kv[0].device)
        for layer_idx in range(len(gpu_kv)):
            gpu_kv[layer_idx].index_copy_(0, dst, gpu_kv[layer_idx][src])

    def kv_copy_h2h(self, src_slots: list[int], dst_slots: list[int]) -> None:
        host_kv = self.kv_cache_at_layer_host
        max_cpu = host_kv[0].shape[0] if host_kv else 0
        for s in src_slots:
            if s < 0 or s >= max_cpu:
                raise ValueError(f"copy_h2h: src slot {s} out of bounds [0, {max_cpu})")
        for s in dst_slots:
            if s < 0 or s >= max_cpu:
                raise ValueError(f"copy_h2h: dst slot {s} out of bounds [0, {max_cpu})")
        src = torch.tensor(src_slots, dtype=torch.long)
        dst = torch.tensor(dst_slots, dtype=torch.long)
        for layer_idx in range(len(host_kv)):
            host_kv[layer_idx].index_copy_(0, dst, host_kv[layer_idx][src])

    def build_model_inputs(self, batch: Batch) -> dict:
        device = torch.device(self.config.device)
        if batch.has_speculative_inputs:
            return batch_tensors.build_spec_expanded_model_inputs(batch, device)
        return batch_tensors.build_model_inputs(batch, device)

    def build_sampling_metadata(self, batch: Batch) -> dict:
        device = torch.device(self.config.device)
        dtype = getattr(torch, self.config.activation_dtype)
        if batch.has_speculative_inputs:
            return batch_tensors.build_spec_expanded_sampling_metadata(
                batch, device, dtype
            )
        return batch_tensors.build_sampling_metadata(batch, device, dtype)

    @torch.inference_mode()
    def fire_batch(self, inputs: dict, sampling_metadata: dict) -> list:
        # This driver runs causal-only attention; user-supplied masks are
        # silently dropped. The capability is advertised via
        # DriverCapabilities so the runtime can route mask-dependent
        # inferlets elsewhere (currently only `native`).
        input_embeds = self.forward_pass.embed_inputs(inputs)

        hidden_states = self.forward_pass.transform(
            input_embeds=input_embeds,
            position_ids=inputs["position_ids"],
            qo_indptr=inputs["qo_indptr"],
            kv_page_indices=inputs["kv_page_indices"],
            kv_page_indptr=inputs["kv_page_indptr"],
            kv_last_page_lens=inputs["kv_last_page_lens"],
        )

        return self.forward_pass.sample(hidden_states, sampling_metadata)

    # ------------------------------------------------------------------
    # Speculative decoding: NGRAM drafter
    # ------------------------------------------------------------------
    #
    # `spec_step` is the contract `._bridge.worker._populate_next_drafts`
    # probes for via `getattr`. Verification + splice are shared (live in
    # `batch_tensors.build_spec_expanded_*` + `Batch.verify_drafts`); this engine
    # only owns the drafter side.

    def _ensure_ngram(self):
        """Lazy-init the numba kernel + scratch buffers on first proposal."""
        if self._ngram_buffers is not None:
            return self._ngram_buffers
        if not getattr(self.driver_config, "spec_ngram_enabled", False):
            return None
        from vllm.v1.spec_decode.ngram_proposer import batch_propose_numba

        max_model_len = int(self.info.get("max_model_len", 0)) or 4096
        max_num_seqs = int(getattr(self.driver_config, "max_num_seqs", 256))
        k = int(self.driver_config.spec_ngram_num_drafts)
        min_n = int(self.driver_config.spec_ngram_min_n)
        max_n = int(self.driver_config.spec_ngram_max_n)
        if min_n < 1 or max_n < min_n:
            raise ValueError(
                "VllmDriverConfig: spec_ngram_min_n must be >= 1 and "
                f"<= spec_ngram_max_n (got min={min_n}, max={max_n})"
            )

        # Trigger numba JIT once with a zeroed batch so the first real
        # call doesn't pay the compile cost on the inference critical path.
        draft_buf = np.zeros((max_num_seqs, k), dtype=np.int32)
        num_drafts_buf = np.zeros(max_num_seqs, dtype=np.int32)
        batch_propose_numba(
            [0],
            np.zeros(1, dtype=np.int32),
            np.zeros((1, max_model_len), dtype=np.int32),
            min_n, max_n, max_model_len, k,
            np.zeros((1, k), dtype=np.int32),
            np.zeros(1, dtype=np.int32),
        )

        self._ngram_buffers = {
            "fn": batch_propose_numba,
            "draft_buf": draft_buf,
            "num_drafts_buf": num_drafts_buf,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "k": k,
            "min_n": min_n,
            "max_n": max_n,
        }
        return self._ngram_buffers

    def spec_step(
        self, sessions: list[tuple[int, list[int]]]
    ) -> list[list[int]]:
        """Per-session NGRAM step: observe accepted, then propose drafts.

        `sessions[i] = (session_id, just_accepted_tokens)` for one request.
        Appends accepted tokens to the per-session history (capped by
        `max_model_len`), runs vllm's longest-suffix-match n-gram kernel
        over the per-batch dense token array, and returns one chain per
        session (possibly empty if no match was found).
        """
        bufs = self._ensure_ngram()
        if bufs is None or not sessions:
            return [[] for _ in sessions]

        max_model_len = bufs["max_model_len"]
        max_num_seqs = bufs["max_num_seqs"]
        k = bufs["k"]
        B = len(sessions)
        if B > max_num_seqs:
            raise RuntimeError(
                f"VllmEngine.spec_step: batch size {B} exceeds "
                f"max_num_seqs {max_num_seqs}"
            )

        # Update histories. The kernel only reads positions < max_model_len,
        # so an oversized history loses its head — n-gram match still works
        # over the retained suffix.
        for sid, accepted in sessions:
            hist = self._ngram_history.get(sid)
            if hist is None:
                hist = []
                self._ngram_history[sid] = hist
            if accepted:
                hist.extend(int(t) for t in accepted)
                if len(hist) > max_model_len:
                    del hist[: len(hist) - max_model_len]

        # Dense [B, max_model_len] tokens + per-session length, the shape
        # the numba kernel expects. Allocated fresh per call — a persistent
        # scratch buffer would have to be cleared anyway.
        token_ids_cpu = np.zeros((B, max_model_len), dtype=np.int32)
        num_tokens = np.zeros(B, dtype=np.int32)
        active_indices: list[int] = []
        for i, (sid, _accepted) in enumerate(sessions):
            hist = self._ngram_history[sid]
            n = len(hist)
            if n == 0:
                continue
            token_ids_cpu[i, :n] = hist
            num_tokens[i] = n
            active_indices.append(i)

        if not active_indices:
            return [[] for _ in sessions]

        draft_buf = bufs["draft_buf"]
        num_drafts_buf = bufs["num_drafts_buf"]
        # Clear only the rows we'll write to; numba reads num_drafts_buf[i]
        # to decide whether row i is valid.
        for i in active_indices:
            num_drafts_buf[i] = 0
        bufs["fn"](
            active_indices,
            num_tokens,
            token_ids_cpu,
            bufs["min_n"], bufs["max_n"], max_model_len, k,
            draft_buf,
            num_drafts_buf,
        )

        out: list[list[int]] = []
        active_set = set(active_indices)
        for i in range(B):
            if i in active_set and num_drafts_buf[i] > 0:
                out.append(draft_buf[i, : num_drafts_buf[i]].tolist())
            else:
                out.append([])
        return out

    def spec_release(self, session_ids: list[int]) -> None:
        """Drop per-session history for finished/evicted contexts."""
        for sid in session_ids:
            self._ngram_history.pop(sid, None)

    # ------------------------------------------------------------------
    # Adapters — deferred. v1 raises clearly so workloads that need them
    # fail fast instead of silently producing wrong tokens.
    # ------------------------------------------------------------------

    def init_adapter(self, *args, **kwargs):
        raise NotImplementedError(
            "Adapters are not yet supported on the vllm driver. "
            "Use the `native` driver for adapter workloads."
        )

    def update_adapter(self, *args, **kwargs):
        raise NotImplementedError("Adapters are not yet supported on the vllm driver.")

    def load_adapter(self, *args, **kwargs):
        raise NotImplementedError("Adapters are not yet supported on the vllm driver.")

    def save_adapter(self, *args, **kwargs):
        return b""

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def query(self, query: str) -> str:
        if query == "ping":
            return "pong"
        return "unknown query"

    def capabilities(self):
        """Report this driver's resolved capacities up to pie's runtime.

        Sources every value from vllm's resolved `VllmConfig` rather than
        echoing the user's `RuntimeConfig`. In particular `kv_page_size`
        comes from the attention backend's chosen block size, which may
        differ from what the user requested.

        Fails loudly if any expected value is missing — the runtime/Rust
        side relies on these being correct, so silent defaulting is unsafe.
        """
        from ._bridge.capabilities import DriverCapabilities

        vc = self.forward_pass.vllm_config
        mc = vc.model_config
        cc = vc.cache_config

        if self.config.max_num_kv_pages is None:
            raise RuntimeError(
                "config.max_num_kv_pages was not set by the loader — KV cache "
                "allocation must run before capabilities() is called."
            )
        if not self.snapshot_dir:
            raise RuntimeError("snapshot_dir is empty; loader did not resolve it.")

        dtype_str = str(mc.dtype).removeprefix("torch.")
        if dtype_str.startswith("torch."):
            raise RuntimeError(
                f"Could not normalize activation dtype {mc.dtype!r}; expected a "
                "torch.dtype with a 'torch.' prefix."
            )

        # vllm expresses batch limits as max_num_seqs / max_num_batched_tokens.
        # Capabilities normalizes them to max_batch_size / max_batch_tokens
        # for pie's runtime side. If max_num_batched_tokens is None (vllm
        # default), use scheduler_config's resolved value.
        max_batch_size = int(self.driver_config.max_num_seqs)
        max_batch_tokens = self.driver_config.max_num_batched_tokens
        if max_batch_tokens is None:
            max_batch_tokens = int(vc.scheduler_config.max_num_batched_tokens)
        else:
            max_batch_tokens = int(max_batch_tokens)

        return DriverCapabilities(
            total_pages=int(self.config.max_num_kv_pages),
            kv_page_size=int(cc.block_size),
            swap_pool_size=int(self.swap_pool_size),
            max_batch_tokens=max_batch_tokens,
            max_batch_size=max_batch_size,
            arch_name=self.arch_type,
            vocab_size=int(mc.get_vocab_size()),
            max_model_len=int(mc.max_model_len),
            activation_dtype=dtype_str,
            # vLLM's V1 attention backends (FLASHINFER, FLEX_ATTENTION, ...)
            # don't have an efficient path for arbitrary user masks under
            # pie's per-batch BRLE layout. The bridge currently runs plain
            # causal attention and silently drops user masks. Inferlets
            # that need non-causal patterns must run on `native`.
            supports_user_attention_mask=False,
            # Adapter operations raise NotImplementedError on this driver.
            supports_adapters=False,
            snapshot_dir=str(self.snapshot_dir),
        )


# ──────────────────────────────────────────────────────────────────────────────
# vllm CUDA-graph piggybacking
# ──────────────────────────────────────────────────────────────────────────────
#
# When `enforce_eager=False`, vllm decorates the model's forward with
# `@support_torch_compile` and stands up a CUDAGraphWrapper for each captured
# (mode, batch_descriptor). At runtime the wrapper consults
# `forward_context.cudagraph_runtime_mode`/`batch_descriptor` to decide
# whether to replay the captured graph or fall through to eager.
#
# Pie's adapter previously bypassed this entirely — calling `model.forward`
# without setting those context fields — which on multi-rank deadlocked
# because the two ranks took different code paths inside the wrapper and
# issued mismatched all-reduces.
#
# We piggyback the standard way: build a `CudagraphDispatcher`, drive a
# capture warmup pass that mirrors `gpu_model_runner._capture_cudagraphs`,
# and let `transform()` ask the dispatcher per fire. No CUDA-specific code
# in pie — vllm's `current_platform.graph_capture` handles platform
# routing (CUDA Graphs on NVIDIA, HIP Graphs on ROCm, no-op elsewhere).


def _maybe_init_cg_dispatcher(vllm_config) -> object | None:
    """Construct + initialize a CudagraphDispatcher, or None if cudagraph
    capture is disabled in vllm_config (i.e. enforce_eager=True)."""
    from ._vllm_compat import CUDAGraphMode, CudagraphDispatcher

    cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
    if cudagraph_mode is None or cudagraph_mode == CUDAGraphMode.NONE:
        return None

    dispatcher = CudagraphDispatcher(vllm_config)
    # uniform_decode_query_len = 1 for plain decode; 1 + num_speculative
    # if speculative decoding is enabled at the vllm level. Pie currently
    # drives spec decoding above vllm (NGRAM via VllmEngine.spec_step), not
    # through vllm's spec config, so 1 is correct here.
    dispatcher.initialize_cudagraph_keys(
        cudagraph_mode, uniform_decode_query_len=1,
    )
    return dispatcher


def _capture_vllm_cudagraphs(
    *,
    forward_pass,
    cg_dispatcher,
    vllm_config,
    config: RuntimeConfig,
) -> None:
    """Drive the model forward at every captured (mode, descriptor) so vllm's
    CUDAGraphWrapper records a graph for each.

    Two phases, mirroring `gpu_worker.compile_or_warm_up_model`:
    1. Warmup OUTSIDE graph_capture — runs each shape with mode=NONE so the
       wrapper falls through to its Dynamo-compiled function, triggering
       Inductor compile. Inductor compile clones CUDA RNG state, which
       fails if it runs inside graph_capture, so it has to happen first.
    2. Capture INSIDE graph_capture, set_cudagraph_capturing_enabled=True —
       runs each shape with the dispatcher-provided mode so the wrapper
       enters its capture branch and records a CUDA graph (or HIP graph
       on ROCm; vllm's `current_platform.graph_capture` handles routing).

    Inputs come from pie's existing metadata builder so the build stays
    backend-agnostic — no per-attention-backend dummy-input plumbing in pie.
    """
    from ._vllm_compat import (
        CUDAGraphMode,
        graph_capture,
        set_cudagraph_capturing_enabled,
        set_forward_context,
    )

    forward_pass._ensure_metadata_builder()
    page_size = forward_pass._kv_spec.block_size
    device = forward_pass.device
    embed_dim = vllm_config.model_config.get_hidden_size()
    dtype = getattr(torch, config.activation_dtype)

    capture_descs = [
        (mode, desc)
        for mode, descs in cg_dispatcher.get_capture_descs()
        if mode != CUDAGraphMode.NONE
        for desc in descs
    ]

    # Phase 1: compile (mode=NONE → wrapper falls through to compiled fn).
    with torch.inference_mode():
        for _mode, desc in capture_descs:
            _run_one_shape(
                forward_pass=forward_pass,
                runtime_mode=CUDAGraphMode.NONE,
                desc=desc,
                page_size=page_size,
                device=device,
                embed_dim=embed_dim,
                dtype=dtype,
                vllm_config=vllm_config,
                set_forward_context=set_forward_context,
            )
    torch.cuda.synchronize()

    # Phase 2: capture.
    set_cudagraph_capturing_enabled(True)
    try:
        with torch.inference_mode(), graph_capture(device=device):
            for mode, desc in capture_descs:
                _run_one_shape(
                    forward_pass=forward_pass,
                    runtime_mode=mode,
                    desc=desc,
                    page_size=page_size,
                    device=device,
                    embed_dim=embed_dim,
                    dtype=dtype,
                    vllm_config=vllm_config,
                    set_forward_context=set_forward_context,
                )
            torch.cuda.synchronize()
    finally:
        set_cudagraph_capturing_enabled(False)


def _run_one_shape(
    *,
    forward_pass,
    runtime_mode,
    desc,
    page_size: int,
    device: torch.device,
    embed_dim: int,
    dtype: torch.dtype,
    vllm_config,
    set_forward_context,
) -> None:
    """Run one dummy forward at (mode, num_tokens) for compile-warmup or
    capture (caller decides via `runtime_mode`). Builds pie-style metadata
    for a uniform-decode batch of `desc.num_tokens` × 1-token requests
    (highest-volume runtime shape — decode phase). PIECEWISE captures
    accept any `num_reqs` per the wrapper's relax_for_mixed_batch logic,
    so this single shape covers both PIECEWISE and FULL captures pie will
    dispatch to at runtime."""
    from .attn_metadata import build_common_metadata

    n = desc.num_tokens
    # Uniform decode: n requests, 1 query token each, 1 KV page of length 1.
    qo_indptr = torch.arange(n + 1, dtype=torch.int32, device=device)
    kv_page_indices = torch.arange(n, dtype=torch.int32, device=device)
    kv_page_indptr = torch.arange(n + 1, dtype=torch.int32, device=device)
    kv_last_page_lens = torch.ones(n, dtype=torch.int32, device=device)

    common = build_common_metadata(
        qo_indptr=qo_indptr,
        kv_page_indices=kv_page_indices,
        kv_page_indptr=kv_page_indptr,
        kv_last_page_lens=kv_last_page_lens,
        page_size=page_size,
        device=device,
    )
    backend_metadata = forward_pass._builder.build(
        common_prefix_len=0, common_attn_metadata=common,
    )
    slot_mapping_dict = {
        name: common.slot_mapping for name in forward_pass._layer_names
    }

    # Use the persistent buffers as inputs so the addresses captured here
    # match what transform() will pass at runtime. The values don't matter
    # for capture (only the shape and address do); zero them defensively.
    forward_pass._buf_input_embeds[:n].zero_()
    forward_pass._buf_positions[:n].zero_()
    input_embeds = forward_pass._buf_input_embeds[:n]
    positions = forward_pass._buf_positions[:n]

    with set_forward_context(
        attn_metadata=backend_metadata,
        vllm_config=vllm_config,
        num_tokens=n,
        slot_mapping=slot_mapping_dict,
        cudagraph_runtime_mode=runtime_mode,
        batch_descriptor=desc,
    ):
        forward_pass.model.forward(
            input_ids=None,
            positions=positions,
            inputs_embeds=input_embeds,
        )
