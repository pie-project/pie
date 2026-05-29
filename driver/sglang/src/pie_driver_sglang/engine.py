"""SGLang-backed inference engine.

Mirrors the shared `Engine` public surface so pie's worker
(`._bridge.worker.run_worker`) can drive native, vllm, and sglang
interchangeably. The model and kernels come from SGLang; pie owns the
RPC, batching, telemetry, and dispatch around it.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from ._bridge.config import RuntimeConfig
from ._bridge.batching import Batch
from ._bridge import telemetry

from . import _require_sglang
from . import batch_tensors
from .config import SGLangDriverConfig


class SGLangEngine:
    """Inference engine that delegates the forward pass to an SGLang
    ModelRunner. Public surface matches the shared `Engine` contract."""

    config: RuntimeConfig
    driver_config: SGLangDriverConfig
    forward_pass: object
    model_config: object
    kv_cache_at_layer: list[torch.Tensor]
    kv_cache_at_layer_host: list[torch.Tensor]
    swap_pool_size: int
    arch_type: str
    snapshot_dir: str | None

    def __init__(
        self,
        config: RuntimeConfig,
        driver_config: SGLangDriverConfig,
        model_config,
        forward_pass,
        kv_cache_at_layer: list,
        arch_type: str,
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
        self.arch_type = arch_type
        self.snapshot_dir = snapshot_dir
        # Always-empty registry so `._bridge.worker._run_fire_batch_inner`
        # (which passes `engine.adapters` to Batch) finds the attribute.
        # init_adapter / load_adapter raise NotImplementedError on this
        # driver, so nothing ever populates it.
        self.adapters: dict = {}

        # Speculative decoding: driver-side n-gram drafter. Lazy-init on
        # first use so the import cost is paid only when spec is enabled.
        self._ngram_corpus = None
        # Per-session token history. Keyed by an opaque session ID the
        # worker provides (currently the first KV page id, which is stable
        # for an active context). Without this, every iteration's `accepted`
        # is only 1-3 tokens long — too short to either (a) populate the
        # trie usefully or (b) anchor a deep trie match. We append accepted
        # tokens per iteration and feed the recent suffix as the query +
        # the full sequence as the trie insertion.
        self._ngram_history: dict[int, list[int]] = {}

    @classmethod
    def load(
        cls,
        config: RuntimeConfig,
        driver_config: SGLangDriverConfig,
        log_queue: object = None,
    ) -> "SGLangEngine":
        _require_sglang()

        from .loader import load_sglang_model
        from .kv_cache import _rebind_pool_buffers, _create_host_kv_cache
        from .forward_pass import SGLangForwardPass

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

        # SGLang's `parallel_state.initialize_model_parallel` reads world
        # size from torch's default process group. Pie's worker now brings
        # up torch.distributed at the *TP-group* scale (one private nccl
        # init per DP replica — matches sglang's native multi-process DP
        # architecture), so the default PG is exactly sglang's expected
        # `tp_size × pp_size` world. No PG-swap workarounds needed.
        loaded = load_sglang_model(
            config, driver_config, log_queue=log_queue,
        )
        # Rebind sglang's k_buffer/v_buffer to pie-shaped storage so the
        # swap RPC handlers see the canonical (num_blocks, 2, page_size,
        # h, d) layout.
        kv_cache_at_layer, num_blocks = _rebind_pool_buffers(loaded, config)
        config.total_pages = num_blocks

        forward_pass = SGLangForwardPass(
            runner=loaded.runner,
            runtime_config=config,
            page_size=int(loaded.runner.page_size),
        )

        # The loader deferred `init_device_graphs()` so pie's hook could
        # be installed first; capture now that the hook is in place.
        if loaded.graph_capture_deferred:
            loaded.runner.init_device_graphs()

        # Pinned host KV pool — sized from `cpu_mem_budget_in_gb`. Indexing
        # contract matches native: `host_kv[layer][slot]` is one block worth
        # of (K, V) data, ready for `index_copy_` in the swap RPC handlers.
        host_kv, pool_size = _create_host_kv_cache(
            kv_cache_at_layer, config.swap_budget_bytes,
        )

        return cls(
            config=config,
            driver_config=driver_config,
            model_config=loaded.sglang_model_config,
            forward_pass=forward_pass,
            kv_cache_at_layer=kv_cache_at_layer,
            arch_type=loaded.arch_type,
            snapshot_dir=loaded.snapshot_dir,
            kv_cache_at_layer_host=host_kv,
            swap_pool_size=pool_size,
        )

    # ------------------------------------------------------------------
    # Speculative decoding: NGRAM drafter
    # ------------------------------------------------------------------
    #
    # `spec_step` is the contract the worker probes for via `getattr` —
    # sglang-specific and gated by `driver_config.spec_ngram_enabled`. The
    # native pie_driver Engine does not implement it (intentional).

    def _ensure_ngram(self):
        """Lazy-init the NgramCorpus on first proposal/observation."""
        if self._ngram_corpus is not None:
            return self._ngram_corpus
        if not self.driver_config.spec_ngram_enabled:
            return None
        from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus

        self._ngram_corpus = NgramCorpus(
            max_trie_depth=int(self.driver_config.spec_ngram_max_depth),
            min_bfs_breadth=1,
            # Linear (non-tree) drafts only in v1 — the Pie API has no way
            # to convey a tree mask back to the inferlet for next-iteration
            # input_speculative_tokens, so we restrict to a single best path.
            max_bfs_breadth=1,
            draft_token_num=int(self.driver_config.spec_ngram_num_drafts),
            match_type="BFS",
            capacity=int(self.driver_config.spec_ngram_capacity),
        )
        return self._ngram_corpus

    def spec_step(
        self, sessions: list[tuple[int, list[int]]]
    ) -> list[list[int]]:
        """Per-session NGRAM step: observe accepted, then propose drafts.

        `sessions[i] = (session_id, just_accepted_tokens)` for one request.
        The engine appends the accepted tokens to the per-session history,
        re-inserts the (capped) full history into the trie so it grows with
        each iteration, then queries the trie using the recent suffix as
        the anchor. Returns `drafts[i]` — the proposed continuation
        (possibly empty if the trie doesn't have a match yet).

        `session_id` is opaque — any stable per-context value works. The
        worker uses the request's first KV page id, which is stable across
        iterations for an unevicted context.
        """
        corpus = self._ensure_ngram()
        if corpus is None or not sessions:
            return [[] for _ in sessions]

        max_depth = int(self.driver_config.spec_ngram_max_depth)
        # Update per-session histories and gather (history, recent suffix).
        all_histories: list[list[int]] = []
        anchors: list[list[int]] = []
        for sid, accepted in sessions:
            hist = self._ngram_history.get(sid, [])
            if accepted:
                hist = hist + [int(t) for t in accepted]
                # Cap at max_depth × 4 to bound memory; the trie itself
                # only indexes the last max_depth tokens of any insert.
                cap = max_depth * 4
                if len(hist) > cap:
                    hist = hist[-cap:]
                self._ngram_history[sid] = hist
            all_histories.append(hist)
            # Anchor for matching: the recent suffix up to max_depth tokens.
            # Shorter anchors → faster but matches less specifically.
            anchor = hist[-max_depth:] if hist else []
            anchors.append(anchor)

        # Insert full histories into the trie. `batch_put` indexes all
        # n-grams within each sequence, so re-inserting the same history
        # each iteration is idempotent for the slice that was already
        # there and incremental for the new tail.
        useful = [h for h in all_histories if len(h) >= 2]
        if useful:
            corpus.batch_put(useful)

        # Stateless query — `erase_match_state` below clears these IDs
        # before they could collide with a future call, so they only need
        # to be unique within this batch.
        req_ids = [f"q{i}" for i in range(len(anchors))]
        # `batch_get` requires non-empty token contexts; substitute a
        # single zero token for any empty anchor and skip its result.
        sanitized = [a if a else [0] for a in anchors]
        total_lens = [len(s) for s in sanitized]
        try:
            corpus.synchronize()
            decoding_ids, tree_mask = corpus.batch_get(
                req_ids=req_ids,
                batch_tokens=sanitized,
                total_lens=total_lens,
            )
        finally:
            corpus.erase_match_state(req_ids)

        n_drafts = int(self.driver_config.spec_ngram_num_drafts)
        ids_arr = np.asarray(decoding_ids).reshape(-1, n_drafts)
        # tree_mask is shape (num_queries × n_drafts × n_drafts), serialized
        # flat. Row k indicates which prior positions position k attends to;
        # for our linear (max_bfs_breadth=1) configuration, position k is
        # part of the chain iff `mask[k][k-1] == 1`. The leading column is
        # always 1 (everything attends to position 0, the anchor echo).
        mask_arr = np.asarray(tree_mask).reshape(-1, n_drafts, n_drafts)
        out: list[list[int]] = []
        for i, a in enumerate(anchors):
            if not a:
                out.append([])
                continue
            row = ids_arr[i]
            mask = mask_arr[i]
            # Walk forward starting at position 1 (position 0 is the anchor
            # echo — `row[0] == anchor[-1]`, NOT a real prediction). Stop
            # at the first position k where `mask[k][k-1] == 0`, meaning
            # the trie didn't extend the chain.
            chain: list[int] = []
            for k in range(1, n_drafts):
                if int(mask[k][k - 1]) == 0:
                    break
                chain.append(int(row[k]))
            out.append(chain)
        return out

    def spec_release(self, session_ids: list[int]) -> None:
        """Drop per-session history for finished/evicted contexts."""
        for sid in session_ids:
            self._ngram_history.pop(sid, None)

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

    @torch.inference_mode()
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

    def fire_batch(self, inputs: dict, sampling_metadata: dict) -> dict:
        passthrough = self.forward_pass.embed_inputs(inputs)

        gathered_logits = self.forward_pass.transform(
            input_embeds=passthrough,
            position_ids=inputs["position_ids"],
            qo_indptr=inputs["qo_indptr"],
            kv_page_indices=inputs["kv_page_indices"],
            kv_page_indptr=inputs["kv_page_indptr"],
            kv_last_page_lens=inputs["kv_last_page_lens"],
        )
        return self.forward_pass.sample(gathered_logits, sampling_metadata)

    # ------------------------------------------------------------------
    # Adapters — deferred. The bridge raises clearly so workloads that
    # need adapter math fail fast instead of silently producing wrong
    # tokens. CMA-ES lives only on the `native` driver; the capability
    # is advertised via DriverCapabilities so the runtime can route.
    # ------------------------------------------------------------------

    def init_adapter(self, *args, **kwargs):
        raise NotImplementedError(
            "Adapters are not supported on the sglang driver. "
            "Use the `native` driver for adapter workloads."
        )

    def update_adapter(self, *args, **kwargs):
        raise NotImplementedError("Adapters are not supported on the sglang driver.")

    def load_adapter(self, *args, **kwargs):
        raise NotImplementedError("Adapters are not supported on the sglang driver.")

    def save_adapter(self, *args, **kwargs):
        return b""

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def query(self, query: str) -> str:
        return "pong" if query == "ping" else "unknown query"

    def capabilities(self):
        """Report this driver's resolved capacities to pie's runtime.

        Sources values from the loaded SGLang `ModelRunner` so the user's
        preferences are reconciled against what the chosen attention
        kernel actually supports (page_size in particular may differ from
        what was requested via `[model.driver.options]`).
        """
        from ._bridge.capabilities import DriverCapabilities

        runner = self.forward_pass.runner
        sglang_mc = runner.model_config

        if self.config.total_pages is None:
            raise RuntimeError(
                "config.total_pages was not set by the loader — KV cache "
                "rebind must run before capabilities() is called."
            )
        if not self.snapshot_dir:
            raise RuntimeError("snapshot_dir is empty; loader did not resolve it.")

        max_forward_tokens = int(runner.max_total_num_tokens)
        max_forward_requests = int(runner.max_running_requests)
        unconstrained = (1 << 32) - 1

        return DriverCapabilities(
            total_pages=int(self.config.total_pages),
            kv_page_size=int(runner.page_size),
            swap_pool_size=int(self.swap_pool_size),
            # sglang sets these on `ModelRunner` after `init_memory_pool`.
            # They are load-bearing for pie's scheduler, so fail loudly if
            # they are missing rather than defaulting silently.
            max_forward_tokens=max_forward_tokens,
            max_forward_requests=max_forward_requests,
            max_page_refs=int(self.config.total_pages),
            max_logit_rows=unconstrained,
            max_prob_rows=unconstrained,
            max_custom_mask_bytes=unconstrained,
            max_sampler_rows=unconstrained,
            max_logprob_labels=unconstrained,
            arch_name=self.arch_type,
            vocab_size=int(sglang_mc.vocab_size),
            max_model_len=int(sglang_mc.context_len),
            activation_dtype=self.config.activation_dtype,
            # sglang's attention path is causal-only from pie's perspective;
            # user-supplied masks are silently dropped. Inferlets that need
            # non-causal patterns must run on `native`.
            supports_user_attention_mask=False,
            # Adapter operations raise NotImplementedError on this driver.
            supports_adapters=False,
            snapshot_dir=str(self.snapshot_dir),
        )


# Alias so worker code that imports `Engine` from this module works.
Engine = SGLangEngine
