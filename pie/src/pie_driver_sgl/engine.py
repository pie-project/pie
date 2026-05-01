"""SGLang-backed inference engine.

Mirrors `pie_driver.engine.Engine`'s public surface so pie's worker
(`pie_driver.worker.run_worker`) can drive native, vllm, and sglang
interchangeably. The model and kernels come from SGLang; pie owns the
RPC, batching, telemetry, and dispatch around it.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from pie_driver.config import RuntimeConfig
from pie_driver import telemetry

from . import _require_sglang
from .config import SGLangDriverConfig


class SGLangEngine:
    """Inference engine that delegates the forward pass to an SGLang
    ModelRunner. Public surface matches `pie_driver.engine.Engine`."""

    config: RuntimeConfig
    driver_config: SGLangDriverConfig
    forward_pass: object
    model_config: object
    kv_cache_at_layer: list[torch.Tensor]
    kv_cache_at_layer_host: list[torch.Tensor]
    swap_pool_size: int
    adapter_at_layer: list
    adapters: dict
    arch_type: str
    snapshot_dir: str | None

    def __init__(
        self,
        config: RuntimeConfig,
        driver_config: SGLangDriverConfig,
        model_config,
        forward_pass,
        kv_cache_at_layer: list,
        adapter_at_layer: list,
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
        self.adapter_at_layer = adapter_at_layer
        self.arch_type = arch_type
        self.snapshot_dir = snapshot_dir
        self.adapters = {}

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
        compute_process_group=None,
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

        loaded = load_sglang_model(
            config, driver_config,
            log_queue=log_queue, compute_pg=compute_process_group,
        )
        # Rebind sglang's k_buffer/v_buffer to pie-shaped storage so the swap
        # RPC handlers see the canonical (num_blocks, 2, page_size, h, d) layout.
        kv_cache_at_layer, num_blocks = _rebind_pool_buffers(loaded, config)
        config.max_num_kv_pages = num_blocks

        forward_pass = SGLangForwardPass(
            runner=loaded.runner,
            runtime_config=config,
            page_size=int(loaded.runner.page_size),
        )

        # Pinned host KV pool — sized from `cpu_mem_budget_in_gb`. Indexing
        # contract matches native: `host_kv[layer][slot]` is one block worth
        # of (K, V) data, ready for `index_copy_` in the swap RPC handlers.
        host_kv, pool_size = _create_host_kv_cache(
            kv_cache_at_layer, config.swap_budget_bytes,
        )

        # Adapter setup is opt-in (gated by driver_config.enable_adapter).
        # When off, we skip both the per-layer storage allocation and the
        # QKVParallelLinear class-swap, keeping the non-adapter path
        # bit-identical to before.
        if driver_config.enable_adapter:
            adapter_at_layer = cls._create_adapter_cache(
                loaded.sglang_model_config, config, driver_config,
            )
            # Install QKV adapter wrappers + create the per-engine subpass
            # slot the wrappers read from at forward time. Safe to install
            # AFTER load_sglang_model only because adapter mode forces
            # `disable_cuda_graph=True` in the loader (see loader.py); no
            # captured graphs exist to be invalidated by the swap.
            from .adapter_hooks import SubpassSlot, install_adapter_wrappers

            adapter_subpass_slot = SubpassSlot()
            install_adapter_wrappers(loaded.runner, adapter_subpass_slot)
            forward_pass.adapter_subpass_slot = adapter_subpass_slot
        else:
            adapter_at_layer = []
            forward_pass.adapter_subpass_slot = None

        return cls(
            config=config,
            driver_config=driver_config,
            model_config=loaded.sglang_model_config,
            forward_pass=forward_pass,
            kv_cache_at_layer=kv_cache_at_layer,
            adapter_at_layer=adapter_at_layer,
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

    @torch.inference_mode()
    def fire_batch(self, inputs: dict, sampling_metadata: dict) -> dict:
        passthrough = self.forward_pass.embed_inputs(inputs)

        # Build the per-batch CMA-ES adapter subpass when adapter tokens
        # are present in the wire request. The wrappers installed at load
        # time will read it via `adapter_subpass_slot.current`. Mirrors
        # `pie_driver.engine.fire_batch` (engine.py:303-311).
        adapter_subpass = None
        if inputs.get("adapter_indices"):
            from .adapter import AdapterSubpass

            adapter_subpass = AdapterSubpass(
                adapter_at_layer=self.adapter_at_layer,
                adapter_indices=inputs["adapter_indices"],
                adapter_extras=self.adapters,
                rand_seeds=inputs["adapter_seeds"],
                qo_indptr=inputs["qo_indptr"],
            )

        # User-supplied attention masks are silently dropped: this driver
        # runs causal-only attention. The capability is advertised via
        # DriverCapabilities; the runtime is responsible for routing
        # mask-dependent inferlets to a driver that supports them.
        gathered_logits = self.forward_pass.transform(
            input_embeds=passthrough,
            position_ids=inputs["position_ids"],
            qo_indptr=inputs["qo_indptr"],
            kv_page_indices=inputs["kv_page_indices"],
            kv_page_indptr=inputs["kv_page_indptr"],
            kv_last_page_lens=inputs["kv_last_page_lens"],
            adapter_subpass=adapter_subpass,
        )
        return self.forward_pass.sample(gathered_logits, sampling_metadata)

    # ------------------------------------------------------------------
    # CMA-ES adapters
    # ------------------------------------------------------------------
    # Per-layer (down, up) storage is shared via `self.adapter_at_layer`;
    # injection into Q/K/V happens in `adapter_hooks.install_adapter_wrappers`,
    # which class-swaps sglang's `QKVParallelLinear`. TP=1 only in v1.

    @staticmethod
    def _create_adapter_cache(
        sglang_model_config,
        runtime_config: RuntimeConfig,
        driver_config: SGLangDriverConfig,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer (down, up) shared storage.

        Layout (per layer):
          down: (max_num_adapters, hidden_size, max_adapter_rank * 3)
          up  : (max_num_adapters, max_adapter_rank,
                 head_dim * (local_num_q_heads + local_num_kv_heads * 2))
        """
        tp = runtime_config.tensor_parallel_size
        cfg = sglang_model_config
        local_num_q_heads = cfg.num_attention_heads // tp
        local_num_kv_heads = cfg.num_key_value_heads // tp
        local_sum_out = cfg.head_dim * (local_num_q_heads + local_num_kv_heads * 2)
        max_num_adapters = int(driver_config.max_num_adapters)
        max_adapter_rank = int(driver_config.max_adapter_rank)

        return [
            (
                torch.zeros(
                    (
                        max_num_adapters,
                        cfg.hidden_size,
                        max_adapter_rank * 3,
                    ),
                    dtype=runtime_config.activation_dtype,
                    device=runtime_config.device,
                ),
                torch.zeros(
                    (
                        max_num_adapters,
                        max_adapter_rank,
                        local_sum_out,
                    ),
                    dtype=runtime_config.activation_dtype,
                    device=runtime_config.device,
                ),
            )
            for _ in range(cfg.num_hidden_layers)
        ]

    @torch.inference_mode()
    def init_adapter(
        self,
        adapter_ptr: int,
        rank: int,
        alpha: float,
        population_size: int,
        mu_fraction: float,
        initial_sigma: float,
    ):
        """Initialize a CMA-ES adapter slot."""
        from .adapter import CmaesAdapter

        cfg = self.model_config
        max_num_adapters = int(self.driver_config.max_num_adapters)

        if adapter_ptr >= max_num_adapters:
            raise ValueError(
                f"Adapter pointer {adapter_ptr} exceeds max_num_adapters "
                f"{max_num_adapters}"
            )

        tp_size = self.config.tensor_parallel_size
        gpu_rank = self.config.rank % tp_size

        local_num_q_heads = cfg.num_attention_heads // tp_size
        local_num_kv_heads = cfg.num_key_value_heads // tp_size

        local_out_features = [
            cfg.head_dim * local_num_q_heads,
            cfg.head_dim * local_num_kv_heads,
            cfg.head_dim * local_num_kv_heads,
        ]

        self.adapters[adapter_ptr] = CmaesAdapter(
            adapter_id=adapter_ptr,
            adapter_at_layer=self.adapter_at_layer,
            rank=rank,
            alpha=alpha,
            in_features=cfg.hidden_size,
            out_features=local_out_features,
            num_layers=cfg.num_hidden_layers,
            population_size=population_size,
            mu_fraction=mu_fraction,
            initial_sigma=initial_sigma,
            min_sigma=1e-7,
            min_var=1e-8,
            max_var=1e4,
            device=self.config.device,
            dtype=self.config.activation_dtype,
            gpu_rank=gpu_rank,
            world_size=tp_size,
            adapter_path=self.config.adapter_path,
        )

    @torch.inference_mode()
    def update_adapter(
        self,
        adapter_ptr: int,
        scores: list[float],
        seeds: list[int],
        max_sigma: float,
    ):
        """Run one CMA-ES iteration on the named adapter."""
        from .adapter import CmaesAdapter

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.config.device)

        if adapter_ptr in self.adapters:
            adapter = self.adapters[adapter_ptr]
            if isinstance(adapter, CmaesAdapter):
                adapter.update(scores, seeds, max_sigma)

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.config.device)

    def load_adapter(self, adapter_ptr: int, name: str, data: bytes) -> None:
        """Load adapter weights + CMA-ES state from a checkpoint."""
        from .adapter import CmaesAdapter

        if adapter_ptr in self.adapters:
            adapter = self.adapters[adapter_ptr]
            if isinstance(adapter, CmaesAdapter):
                if self.config.world_size > 1:
                    name = f"{name}_rank{self.config.rank}"
                adapter.upload(name, data)

    def save_adapter(self, adapter_ptr: int, name: str) -> bytes:
        """Snapshot adapter weights + CMA-ES state to a checkpoint."""
        from .adapter import CmaesAdapter

        if adapter_ptr in self.adapters:
            adapter = self.adapters[adapter_ptr]
            if isinstance(adapter, CmaesAdapter):
                if self.config.world_size > 1:
                    name = f"{name}_rank{self.config.rank}"
                return adapter.download(name)
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
        what was requested via `[model.X.driver.sglang]`).
        """
        from pie.capabilities import DriverCapabilities

        runner = self.forward_pass.runner
        sglang_mc = runner.model_config

        if self.config.max_num_kv_pages is None:
            raise RuntimeError(
                "config.max_num_kv_pages was not set by the loader — KV cache "
                "rebind must run before capabilities() is called."
            )
        if not self.snapshot_dir:
            raise RuntimeError("snapshot_dir is empty; loader did not resolve it.")

        return DriverCapabilities(
            total_pages=int(self.config.max_num_kv_pages),
            kv_page_size=int(runner.page_size),
            swap_pool_size=int(self.swap_pool_size),
            # sglang sets these on `ModelRunner` after `init_memory_pool` —
            # they're load-bearing for pie's scheduler so fail loudly if
            # they're missing rather than defaulting silently.
            max_batch_tokens=int(runner.max_total_num_tokens),
            max_batch_size=int(runner.max_running_requests),
            arch_name=self.arch_type,
            vocab_size=int(sglang_mc.vocab_size),
            max_model_len=int(sglang_mc.context_len),
            activation_dtype=str(self.config.activation_dtype).removeprefix("torch."),
            # sglang's attention path is causal-only from pie's perspective;
            # user-supplied masks are silently dropped. Inferlets that need
            # non-causal patterns must run on `native`.
            supports_user_attention_mask=False,
            snapshot_dir=str(self.snapshot_dir),
        )


# Alias so worker code that imports `Engine` from this module works.
Engine = SGLangEngine
