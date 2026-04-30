"""vLLM-backed inference engine.

Mirrors `pie_driver.engine.Engine`'s public surface so that worker.py can use
either driver interchangeably. Internally, the model and kernels come from
vllm; the surrounding RPC, batching, telemetry, and adapter scaffolding are
imported directly from `pie_driver`.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from pie_driver.config import RuntimeConfig
from pie_driver import telemetry

from . import _require_vllm


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
        # and splice live in the shared `pie_driver.batching.Batch`; this
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
        compute_process_group=None,
    ) -> "VllmEngine":
        _require_vllm()

        from .forward_pass import VllmForwardPass
        from .kv_cache import allocate_and_bind_kv_cache, allocate_host_pool
        from .loader import load_vllm_model
        from .mask_strategies import install_mask_strategies

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

        # Wrap each attn layer's impl in a mask-aware proxy. Idempotent;
        # an absent `pie_attn_extras` makes the proxy a single dict.get
        # plus a delegating call. See mask_strategies.py for the per-
        # backend strategies and the refusal policy on unsupported impls.
        install_mask_strategies(loaded.vllm_config)

        kv_cache_at_layer = allocate_and_bind_kv_cache(loaded, config, driver_config)
        host_kv, pool_size = allocate_host_pool(kv_cache_at_layer, config.swap_budget_bytes)

        forward_pass = VllmForwardPass(
            model=loaded.model,
            vllm_config=loaded.vllm_config,
            attn_backend=loaded.attn_backend,
            runtime_config=config,
            model_config=loaded.model_config,
        )

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

    @torch.inference_mode()
    def fire_batch(self, inputs: dict, sampling_metadata: dict) -> list:
        input_embeds = self.forward_pass.embed_inputs(inputs)

        hidden_states = self.forward_pass.transform(
            input_embeds=input_embeds,
            position_ids=inputs["position_ids"],
            qo_indptr=inputs["qo_indptr"],
            kv_cache_at_layer=self.kv_cache_at_layer,
            kv_page_indices=inputs["kv_page_indices"],
            kv_page_indptr=inputs["kv_page_indptr"],
            kv_last_page_lens=inputs["kv_last_page_lens"],
            # Only forward the mask when there's something to apply — the
            # mask-aware impl proxy checks `pie_attn_extras` presence as the
            # zero-overhead signal. `inputs["custom_mask"]` is always
            # populated (BRLE-decoded to all-True when no mask was supplied),
            # so we gate on the explicit `has_custom_mask` flag instead.
            custom_mask=inputs["custom_mask"] if inputs.get("has_custom_mask") else None,
            single_token_inference_mode=inputs["single_token_inference_mode"],
        )

        return self.forward_pass.sample(hidden_states, sampling_metadata)

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
            snapshot_dir=str(self.snapshot_dir),
        )
