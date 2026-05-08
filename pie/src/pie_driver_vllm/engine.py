"""vLLM-backed inference engine.

Mirrors `pie_driver.engine.Engine`'s public surface so that worker.py can use
either driver interchangeably. Internally, the model and kernels come from
vllm; the surrounding RPC, batching, telemetry, and adapter scaffolding are
imported directly from `pie_driver`.
"""

from __future__ import annotations

import logging
import random

import numpy as np
import torch

from pie_driver.config import RuntimeConfig
from pie_driver import telemetry

from . import _require_vllm

logger = logging.getLogger(__name__)


# Existing `_log(msg, level: str)` helpers in this module accept string
# level names like "INFO" / "WARN" / "DEBUG" (matching pie's log_queue
# convention). Map them to stdlib logging integer levels so we can route
# through `logger.log(level_int, ...)` without a method lookup per call.
# `WARN` is normalized to `WARNING` (stdlib's canonical name).
_LEVEL_NAME_TO_INT = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def _level_to_int(level: str) -> int:
    """Map a pie log_queue level string to a stdlib logging int level."""
    return _LEVEL_NAME_TO_INT.get(level.upper(), logging.INFO)


class VllmEngine:
    """Inference engine that delegates the forward pass to a vllm model.

    Public surface matches `pie_driver.engine.Engine`:
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
    # `(layer_name, conv_state, ssm_state)` per mamba layer on hybrid models.
    # NOT included in `kv_cache_at_layer` so pie's per-page copy_d2d/h2d/d2h
    # handlers (which assume page-id indexing valid across all listed layers)
    # never see them. Hybrid models advertise `supports_kv_fork=False` via
    # capabilities, so pie's runtime won't issue fork on them; if it does
    # anyway, attention pages are still copied correctly while mamba state
    # stays at the (shared) parent slot — see ticket #107 / #108.
    mamba_layers: list[tuple[str, torch.Tensor, torch.Tensor]]
    is_hybrid: bool
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
        mamba_layers: list | None = None,
        is_hybrid: bool = False,
    ):
        self.config = config
        self.driver_config = driver_config
        self.model_config = model_config
        self.forward_pass = forward_pass
        self.kv_cache_at_layer = kv_cache_at_layer
        self.kv_cache_at_layer_host = kv_cache_at_layer_host or []
        self.mamba_layers = mamba_layers or []
        self.is_hybrid = is_hybrid
        self.swap_pool_size = swap_pool_size
        self.adapter_at_layer = adapter_at_layer
        self.arch_type = arch_type
        self.info = info
        self.snapshot_dir = snapshot_dir
        self.adapters = {}

        # Speculative decoding: driver-side n-gram drafter. Verification
        # and splice live in the shared `pie_driver.batching.Batch`; this
        # engine owns drafting via `spec_step`. Buffers are lazy-init so
        # the numba JIT cost is only paid when spec is actually used.
        self._ngram_buffers = None
        self._ngram_history: dict[int, list[int]] = {}

        # Mamba state-slot allocator (#108 phase 5a). Always set by
        # `Engine.load`; left as None here so unit tests that bypass
        # load() don't have to construct one. `forward_pass.transform`
        # gates use on `is_hybrid` AND `mamba_allocator is not None`.
        self.mamba_allocator = None

    @classmethod
    def load(
        cls,
        config: RuntimeConfig,
        driver_config,
        log_queue: object = None,
        compute_process_group=None,
    ) -> "VllmEngine":
        _require_vllm()

        from .forward_pass import VllmForwardPass
        from .kv_cache import allocate_and_bind_kv_cache, allocate_host_pool
        from .loader import load_vllm_model
        from .mamba_state import MambaSlotAllocator

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
            config, driver_config, log_queue=log_queue, compute_pg=compute_process_group
        )
        _log("Loaded vllm model", "DEBUG")

        layout = allocate_and_bind_kv_cache(loaded, config, driver_config)
        host_kv, pool_size = allocate_host_pool(
            layout.attention_layers_in_order, config.swap_budget_bytes
        )

        # Per-request mamba state-slot allocator (#108 phase 5a).
        # `allocate_and_bind_kv_cache` sized the mamba state pools to
        # `config.max_num_kv_pages = num_blocks`; the allocator hands
        # out indices into that dim-0. Created on every load (allocator
        # is a no-op container on pure-attention since `transform()`
        # gates its use on `is_hybrid`); cheap to construct.
        mamba_allocator = MambaSlotAllocator(int(config.max_num_kv_pages))

        forward_pass = VllmForwardPass(
            model=loaded.model,
            vllm_config=loaded.vllm_config,
            attn_backend=loaded.attn_backend,
            runtime_config=config,
            model_config=loaded.model_config,
            mamba_layers=layout.mamba_layers,
            is_hybrid=layout.is_hybrid,
            mamba_allocator=mamba_allocator,
        )

        engine = cls(
            config=config,
            driver_config=driver_config,
            model_config=loaded.model_config,
            forward_pass=forward_pass,
            kv_cache_at_layer=layout.attention_layers_in_order,
            adapter_at_layer=[],
            arch_type=loaded.arch_type,
            info=loaded.info,
            snapshot_dir=loaded.snapshot_dir,
            kv_cache_at_layer_host=host_kv,
            swap_pool_size=pool_size,
            mamba_layers=layout.mamba_layers,
            is_hybrid=layout.is_hybrid,
        )
        engine.mamba_allocator = mamba_allocator

        if layout.is_hybrid:
            logger.info(
                "vllm driver loaded a hybrid Transformer+Mamba model "
                "(%s) with supports_kv_fork=True (#108 phase 5a/5b: "
                "ContextId-keyed allocator + runtime-side mamba "
                "copy-on-fork wired).",
                loaded.arch_type,
            )

        # Compile + JIT warmup. Drives a synthetic prefill+decode forward
        # so torch.compile (Dynamo + AOT) and FlashInfer JIT (ninja/nvcc)
        # complete during init rather than on the inferlet's first
        # fire_batch. Without this, cold-start cost (15–60+ s on sm_89
        # depending on caches) shows up as visible first-request latency
        # — and on the shmem fast path it can exceed the call deadline
        # and surface downstream as flashinfer "max() iterable empty".
        try:
            engine._warmup_compile(_log)
        except Exception:
            # logger.exception attaches the active traceback automatically.
            logger.exception("vllm warmup FAILED")
            raise

        # Lever 5 (ticket #100): wrap the model with vllm's FULL-mode
        # CUDAGraphWrapper and run synthetic decode passes so each
        # `cudagraph_capture_sizes` shape gets its captured graph. After
        # this returns, transform() can dispatch decode batches through
        # the captured graphs by threading the runtime mode + descriptor
        # through set_forward_context.
        try:
            engine._capture_cudagraphs(_log)
        except Exception:
            # logger.exception attaches the active traceback automatically.
            logger.exception("vllm cudagraph FAILED")
            raise

        return engine

    @torch.inference_mode()
    def _warmup_compile(self, log_fn=None) -> None:
        """Drive synthetic prefill + decode passes so torch.compile and
        FlashInfer JIT complete during engine init.

        Uses physical page 0 of the KV pool as scratch and zeroes the
        layer caches afterwards so no stale activations leak into the
        first real context. Both a multi-token prefill batch and a
        single-token decode batch are exercised because FlashInfer's
        plan builder selects different kernels for the two query-length
        regimes; only pre-planning both leaves the inferlet's first
        fire_batch on a fully-warm path.
        """
        import time as _time
        import traceback as _tb

        device = self.forward_pass.device

        def _log(msg: str, level: str = "INFO") -> None:
            logger.log(_level_to_int(level), "warmup: %s", msg)
            if log_fn is not None:
                log_fn(msg, level)

        if not torch.cuda.is_available() or not str(device).startswith("cuda"):
            # CPU/Metal warmup is unnecessary — the cold-start tail is a
            # CUDA-only phenomenon (torch.compile + FlashInfer JIT).
            _log(f"skipped (device={device})", "INFO")
            return

        _log("starting", "INFO")

        # Page-size from the resolved KV spec. embed_inputs / transform
        # both depend on this, so prime the metadata builder once.
        self.forward_pass._ensure_metadata_builder()
        page_size = int(self.forward_pass._kv_spec.block_size)

        # 8 tokens fits in a single page for any sane page_size; leaves
        # room for the +1 decode position without crossing the boundary.
        # token_id 0 is universally valid (typically <pad> / <unk>).
        n_prefill = max(1, min(8, page_size - 1))

        prefill_inputs = {
            "token_ids": torch.zeros(n_prefill, dtype=torch.long),
            "position_ids": torch.arange(n_prefill, dtype=torch.int32, device=device),
            "qo_indptr": torch.tensor([0, n_prefill], dtype=torch.int32, device=device),
            "kv_page_indices": torch.tensor([0], dtype=torch.int32, device=device),
            "kv_page_indptr": torch.tensor([0, 1], dtype=torch.int32, device=device),
            "kv_last_page_lens": torch.tensor([n_prefill], dtype=torch.int32, device=device),
        }

        t0 = _time.perf_counter()
        embeds = self.forward_pass.embed_inputs(prefill_inputs)
        self.forward_pass.transform(
            input_embeds=embeds,
            position_ids=prefill_inputs["position_ids"],
            qo_indptr=prefill_inputs["qo_indptr"],
            kv_page_indices=prefill_inputs["kv_page_indices"],
            kv_page_indptr=prefill_inputs["kv_page_indptr"],
            kv_last_page_lens=prefill_inputs["kv_last_page_lens"],
            # Synthetic ContextId for warmup; allocator hands out a
            # slot we never read state from (qlen>1 prefill, has_initial_state=False).
            context_ids=[0],
        )
        torch.cuda.synchronize()
        t_prefill = _time.perf_counter() - t0

        # Decode: 1 query token at position n_prefill, attending to the
        # KV just written by prefill. last_page_len advances by 1.
        decode_inputs = {
            "token_ids": torch.zeros(1, dtype=torch.long),
            "position_ids": torch.tensor([n_prefill], dtype=torch.int32, device=device),
            "qo_indptr": torch.tensor([0, 1], dtype=torch.int32, device=device),
            "kv_page_indices": torch.tensor([0], dtype=torch.int32, device=device),
            "kv_page_indptr": torch.tensor([0, 1], dtype=torch.int32, device=device),
            "kv_last_page_lens": torch.tensor([n_prefill + 1], dtype=torch.int32, device=device),
        }

        t0 = _time.perf_counter()
        embeds = self.forward_pass.embed_inputs(decode_inputs)
        decode_hidden = self.forward_pass.transform(
            input_embeds=embeds,
            position_ids=decode_inputs["position_ids"],
            qo_indptr=decode_inputs["qo_indptr"],
            kv_page_indices=decode_inputs["kv_page_indices"],
            kv_page_indptr=decode_inputs["kv_page_indptr"],
            kv_last_page_lens=decode_inputs["kv_last_page_lens"],
            # Same synthetic ContextId as the prefill warmup so the
            # allocator returns the same slot — keeps the warmup decode
            # reading state written by the warmup prefill, matching the
            # production path order.
            context_ids=[0],
        )
        torch.cuda.synchronize()
        t_decode = _time.perf_counter() - t0

        # Sampling path warmup: lm_head (compute_logits) + flashinfer
        # top_k_top_p sampling kernel each carry their own JIT cost on
        # first call (~tens of seconds combined on cold sm_89). Without
        # this, the inferlet's first fire_batch pays them after transform
        # is already warm — pushing total cold-start past the 60s shmem
        # IPC timeout. Synthetic call uses the decode hidden_states slice
        # so dtype/device match production exactly.
        t0 = _time.perf_counter()
        try:
            from flashinfer.sampling import top_k_top_p_sampling_from_probs

            logits = self.forward_pass.model.compute_logits(decode_hidden)
            # compute_logits returns (n, vocab); softmax to probs to match
            # sample_common's scaled_softmax → flashinfer call shape.
            probs = torch.softmax(logits.float(), dim=-1)
            n_rows = probs.shape[0]
            top_k_t = torch.full((n_rows,), 50, dtype=torch.int32, device=device)
            top_p_t = torch.full((n_rows,), 0.9, dtype=torch.float32, device=device)
            _ = top_k_top_p_sampling_from_probs(probs, top_k=top_k_t, top_p=top_p_t)
            torch.cuda.synchronize()
            t_sample = _time.perf_counter() - t0
            _log(f"sample warmup: lm_head + flashinfer top_k_top_p {t_sample:.2f}s", "INFO")
        except Exception as _e:
            _log(f"sample warmup skipped: {_e}\n{_tb.format_exc()}", "WARN")

        # Zero physical page 0 in every KV layer so the first real
        # context that gets allocated this page doesn't read stale
        # warmup activations as committed K/V.
        for layer_kv in self.kv_cache_at_layer:
            layer_kv[0].zero_()
        # Zero slot 0 of every mamba state pool too — warmup wrote
        # recurrent state at the synthetic request's slot (= first page id
        # = 0 by construction), so the first real request that's allocated
        # page 0 must see a clean state.
        for _name, conv_state, ssm_state in self.mamba_layers:
            conv_state[0].zero_()
            ssm_state[0].zero_()

        _log(
            f"vllm engine warmup: prefill ({n_prefill} tok) {t_prefill:.2f}s, "
            f"decode (1 tok) {t_decode:.2f}s",
            "INFO",
        )

    @torch.inference_mode()
    def _capture_cudagraphs(self, log_fn=None) -> None:
        """Capture FULL-mode cudagraphs for decode batches (Lever 5).

        Mirrors vllm's `GPUModelRunner.capture_model` for the slice we need:
        wrap the model in `CUDAGraphWrapper(runtime_mode=FULL)`, build a
        `CudagraphDispatcher` so we know the set of capture sizes, allocate
        persistent input buffers in `forward_pass`, and run one synthetic
        forward per `BatchDescriptor` inside `graph_capture(...)` with the
        runtime mode threaded through forward context. After this returns,
        transform() can dispatch decode batches into the captured graphs.

        Skipped (no-op) when:
          - device is not CUDA;
          - cudagraph_mode is NONE (eager mode requested by user);
          - FlashInfer / attention backend reports no cudagraph support.
        """
        import time as _time

        from ._vllm_compat import (
            CUDAGraphMode,
            CUDAGraphWrapper,
            CudagraphDispatcher,
            graph_capture,
            set_cudagraph_capturing_enabled,
            set_current_vllm_config,
            set_forward_context,
        )

        device = self.forward_pass.device

        def _log(msg: str, level: str = "INFO") -> None:
            logger.log(_level_to_int(level), "cudagraph: %s", msg)
            if log_fn is not None:
                log_fn(msg, level)

        if not torch.cuda.is_available() or not str(device).startswith("cuda"):
            _log(f"skipped (device={device})", "INFO")
            return

        vllm_config = self.forward_pass.vllm_config
        cc = vllm_config.compilation_config
        cg_mode = cc.cudagraph_mode

        if cg_mode == CUDAGraphMode.NONE:
            _log("skipped (cudagraph_mode=NONE)", "INFO")
            return

        if self.is_hybrid:
            # Hybrid Transformer+Mamba: GDN backend has its own
            # AttentionCGSupport.UNIFORM_BATCH cudagraph path that captures
            # the FLA / causal_conv1d kernels itself, with persistent
            # buffers sized to `decode_cudagraph_max_bs`. Mixing it with
            # pie's FULL_DECODE_ONLY wrapper around the whole forward
            # would double-capture and corrupt mamba state. Defer hybrid
            # cudagraph to ticket #107's perf follow-up; eager works.
            _log("skipped (hybrid Transformer+Mamba — eager only)", "INFO")
            return

        # Pie does not own a vllm GPUModelRunner, so the cudagraph_capture_sizes
        # list may not have been populated by vllm's own resolution. Guard
        # for that and bail out cleanly — transform() will then fall back
        # to the eager path on every batch.
        if not cc.cudagraph_capture_sizes or cc.max_cudagraph_capture_size is None:
            _log(
                "skipped (no cudagraph_capture_sizes resolved by vllm)", "WARN"
            )
            return

        # Drain any kernels still queued by warmup before switching streams.
        # `graph_capture()` enters a separate CUDA stream; capture must not
        # observe in-flight default-stream work.
        torch.cuda.synchronize()

        spec_cfg = vllm_config.speculative_config
        query_len = 1 if spec_cfg is None else 1 + spec_cfg.num_speculative_tokens

        # Allocate persistent input buffers sized to the largest captured
        # batch. inputs_embeds dim comes from the first warmup forward
        # (decode_hidden in _warmup_compile would also work, but we read
        # straight off the model config to avoid coupling).
        max_size = int(cc.max_cudagraph_capture_size)
        hidden_size = int(self.model_config.get_hidden_size())
        embed_dtype = self.config.activation_dtype
        self.forward_pass._cg_inputs_embeds_buf = torch.empty(
            (max_size, hidden_size), dtype=embed_dtype, device=device
        )
        self.forward_pass._cg_positions_buf = torch.empty(
            (max_size,), dtype=torch.int64, device=device
        )
        # int64 to match `build_common_metadata`'s slot_mapping dtype.
        self.forward_pass._cg_slot_mapping_buf = torch.empty(
            (max_size,), dtype=torch.int64, device=device
        )
        self.forward_pass._cg_query_len = query_len

        # Wrap the model. vllm's CUDAGraphWrapper.__call__ checks
        # forward_context for cudagraph_runtime_mode/batch_descriptor and
        # captures (first call per descriptor) or replays (subsequent).
        # Pass-through if mode does not match (e.g. prefill batches).
        wrapped = CUDAGraphWrapper(
            self.forward_pass.model,
            vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
        )
        self.forward_pass.model = wrapped

        dispatcher = CudagraphDispatcher(vllm_config)
        dispatcher.initialize_cudagraph_keys(cg_mode, query_len)
        # Stash on forward_pass; transform() dispatches via this.
        self.forward_pass._cg_dispatcher = dispatcher

        capture_descs = dispatcher.get_capture_descs()
        if not capture_descs:
            _log("skipped (dispatcher has no capture descs)", "WARN")
            self.forward_pass._cg_dispatcher = None
            return

        # Drive synthetic decode forwards through the wrapped model. We
        # reuse pie's existing transform()/embed_inputs path so any backend
        # (FlashInfer, FA3, Triton, …) builds its persistent metadata
        # buffers exactly as it would at runtime. The dispatcher is
        # already installed on forward_pass, so transform() will route
        # each call through `set_forward_context(cudagraph_runtime_mode=FULL,
        # batch_descriptor=desc)` once we feed it a uniform-decode batch.
        torch.cuda.empty_cache()
        start_free = torch.cuda.mem_get_info()[0]
        t0 = _time.perf_counter()

        set_cudagraph_capturing_enabled(True)
        try:
            with set_current_vllm_config(vllm_config), graph_capture(device=device):
                for runtime_mode, batch_descs in capture_descs:
                    if runtime_mode != CUDAGraphMode.FULL:
                        # We only install a FULL wrapper; PIECEWISE descs
                        # would need a separate wrapper / inductor partition
                        # path, out of scope for ticket #100.
                        continue
                    for desc in batch_descs:
                        n = desc.num_tokens
                        if n > max_size:
                            continue
                        self._capture_one_decode(n, query_len)
                        torch.cuda.synchronize()
        finally:
            set_cudagraph_capturing_enabled(False)

        end_free = torch.cuda.mem_get_info()[0]
        elapsed = _time.perf_counter() - t0
        graph_bytes = max(0, start_free - end_free)
        _log(
            f"captured FULL graphs for {sum(len(d) for _, d in capture_descs)} "
            f"descriptor(s) in {elapsed:.2f}s "
            f"({graph_bytes / (1 << 20):.1f} MiB graph memory)",
            "INFO",
        )

        # Capture writes K/V into pie's KV layers for pages we used as
        # scratch (page 0). Zero them so the first real context that gets
        # allocated those pages doesn't read stale capture activations.
        for layer_kv in self.kv_cache_at_layer:
            layer_kv[0].zero_()

    @torch.inference_mode()
    def _capture_one_decode(self, n: int, query_len: int) -> None:
        """One synthetic uniform-decode forward at exactly `n` query tokens.

        Triggers FlashInfer to plan the cudagraph-enabled decode wrapper at
        size `n` (under the `enable_cuda_graph and pure_decode and
        num_decode_tokens <= max_bs` branch in FlashInferMetadataBuilder.build),
        and runs the wrapped model so CUDAGraphWrapper captures the forward.

        Layout: `n / query_len` requests, each with `query_len` query tokens
        attending to a single KV page (page 0, pie's reserved warmup
        scratch). Position 0 across all requests — chosen for simplicity;
        FlashInfer's plan() does not validate position values for capture.
        """
        device = self.forward_pass.device
        page_size = int(self.forward_pass._kv_spec.block_size)

        assert n % query_len == 0, (
            f"capture size {n} must be divisible by query_len {query_len}"
        )
        num_reqs = n // query_len

        token_ids = torch.zeros(n, dtype=torch.long)
        position_ids = torch.zeros(n, dtype=torch.int32, device=device)
        # uniform decode → qo_indptr step is `query_len` per request.
        qo_indptr = (
            torch.arange(num_reqs + 1, dtype=torch.int32, device=device)
            * query_len
        )
        # Each req points at page 0; last_page_len=query_len so seq_len = query_len.
        kv_page_indices = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        kv_page_indptr = torch.arange(num_reqs + 1, dtype=torch.int32, device=device)
        kv_last_page_lens = torch.full(
            (num_reqs,), max(query_len, 1), dtype=torch.int32, device=device
        )

        inputs = {
            "token_ids": token_ids,
            "position_ids": position_ids,
            "qo_indptr": qo_indptr,
            "kv_page_indices": kv_page_indices,
            "kv_page_indptr": kv_page_indptr,
            "kv_last_page_lens": kv_last_page_lens,
        }

        embeds = self.forward_pass.embed_inputs(inputs)
        # transform() will see a synthetic uniform-decode batch and the
        # dispatcher will return FULL with a descriptor for size `n`.
        # CUDAGraphWrapper captures on first call per descriptor. We force
        # single_token_mode=True so the dispatch gate fires (the runtime
        # would normally have set it on a real decode batch).
        _ = self.forward_pass.transform(
            input_embeds=embeds,
            position_ids=position_ids,
            qo_indptr=qo_indptr,
            kv_page_indices=kv_page_indices,
            kv_page_indptr=kv_page_indptr,
            kv_last_page_lens=kv_last_page_lens,
            single_token_mode=True,
            # Cudagraph capture is gated off for hybrid (#107), so the GDN
            # branch in transform() never fires here. We still pass synthetic
            # ids so the contract is uniform — defensive against a future
            # hybrid cudagraph path.
            context_ids=[0] * num_reqs,
        )

    def mamba_fork(self, parent_ctx_id: int, child_ctx_id: int) -> None:
        """Copy parent's mamba recurrent state into the child's slot (#108 phase 5b).

        Wired to the runtime's `device::mamba_fork_shmem` op via the
        `pie_driver/worker.py` shmem dispatcher. Always called on fork
        from pie's runtime — no-op on pure-attention (`mamba_layers`
        empty) so the runtime can emit the op unconditionally.

        On hybrid models the actual tensor copy lives in
        `mamba_state.fork_mamba_slot`; we delegate so the per-layer
        loop is testable in isolation without an Engine instance.
        """
        if self.mamba_allocator is None:
            # Engine constructed without `Engine.load` (unit tests bypass
            # the loader). On non-hybrid that's fine — no state to copy.
            # On hybrid this would indicate a wiring bug; surface it.
            if self.is_hybrid:
                raise RuntimeError(
                    "VllmEngine.mamba_fork called on hybrid model with "
                    "no MambaSlotAllocator wired. Engine.load must "
                    "construct one (#108 phase 5a)."
                )
            return
        from .mamba_state import fork_mamba_slot

        fork_mamba_slot(
            parent_ctx_id, child_ctx_id, self.mamba_layers, self.mamba_allocator
        )

    @torch.inference_mode()
    def fire_batch(
        self,
        inputs: dict,
        sampling_metadata: dict,
        gpu_timings: dict | None = None,
    ) -> list:
        # This driver runs causal-only attention; user-supplied masks are
        # silently dropped. The capability is advertised via
        # DriverCapabilities so the runtime can route mask-dependent
        # inferlets elsewhere (currently only `native`).
        #
        # When `gpu_timings` is provided, we record cuda.Event markers
        # around embed/transform/sample to break apart the GPU-side cost
        # of each stage. Reading elapsed_time forces a single sync at the
        # end (already implicit in the inferlet's host-side .tolist() of
        # sampler outputs), so the events themselves do not stall the
        # pipeline. Used by #68085 Phase 44 to localize prefill vs decode
        # overhead at sub-stage granularity.
        # Sync-then-time approach: torch.cuda.synchronize() drains all kernels
        # on every CUDA stream, then perf_counter captures wall — the
        # difference between two adjacent syncs IS the GPU time of the
        # intervening stage. cuda.Events on default stream were unreliable
        # here (vllm/inductor uses dedicated streams; events recorded on
        # default stream don't capture compiled-graph kernels and can also
        # interfere with pie's snapshot.fork() copy_d2d on a separate
        # thread, producing the qo_indptr CUDA assert seen on H100). This
        # is heavier (4× sync per fire_batch) but reliable. Only enabled
        # when gpu_timings is requested, so the hot path is unaffected.
        if gpu_timings is not None:
            torch.cuda.synchronize()
            import time as _time
            t0 = _time.perf_counter()

        input_embeds = self.forward_pass.embed_inputs(inputs)

        if gpu_timings is not None:
            torch.cuda.synchronize()
            t1 = _time.perf_counter()

        hidden_states = self.forward_pass.transform(
            input_embeds=input_embeds,
            position_ids=inputs["position_ids"],
            qo_indptr=inputs["qo_indptr"],
            kv_page_indices=inputs["kv_page_indices"],
            kv_page_indptr=inputs["kv_page_indptr"],
            kv_last_page_lens=inputs["kv_last_page_lens"],
            single_token_mode=inputs.get("single_token_inference_mode", False),
            # Per-row stable ContextIds (#108 phase 5a). On hybrid models
            # `forward_pass` routes these to the mamba state-slot allocator;
            # pure-attention paths ignore the field.
            context_ids=inputs.get("context_ids"),
        )

        if gpu_timings is not None:
            torch.cuda.synchronize()
            t2 = _time.perf_counter()

        out = self.forward_pass.sample(hidden_states, sampling_metadata)

        if gpu_timings is not None:
            torch.cuda.synchronize()
            t3 = _time.perf_counter()
            gpu_timings["embed_ms"] = (t1 - t0) * 1000.0
            gpu_timings["transform_ms"] = (t2 - t1) * 1000.0
            gpu_timings["sample_ms"] = (t3 - t2) * 1000.0

        return out

    # ------------------------------------------------------------------
    # Speculative decoding: NGRAM drafter
    # ------------------------------------------------------------------
    #
    # `spec_step` is the contract `pie_driver.worker._populate_next_drafts`
    # probes for via `getattr`. Verification + splice are shared (live in
    # `Batch.get_spec_expanded_*` and `Batch.verify_drafts`); this engine
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
        from pie.capabilities import DriverCapabilities

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
            # Phase 5a's MambaSlotAllocator gives every fork a fresh
            # slot keyed on its (distinct) ContextId, and phase 5b's
            # `mamba_fork_shmem` op (snapshot.rs → worker.py →
            # `Engine.mamba_fork`) copies parent's per-layer recurrent
            # state into that slot before the runtime resolves
            # `Ok(new_id)`. Hybrid forks now produce a child that
            # inherits the parent's exact mamba state up to the fork
            # point — same correctness guarantee as the attention KV
            # pages that copy_d2d_shmem already handles.
            supports_kv_fork=True,
            snapshot_dir=str(self.snapshot_dir),
        )
