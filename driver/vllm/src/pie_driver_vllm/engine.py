"""vLLM-backed inference engine.

Mirrors the shared `Engine` public surface so that worker.py can use
either driver interchangeably. Internally, the model and kernels come from
vllm; the surrounding RPC, batching, telemetry, and adapter scaffolding are
imported directly from `pie_driver`.
"""

from __future__ import annotations

import os
import random
import time
import sys

from dataclasses import dataclass

import numpy as np
import torch

from ._bridge.config import RuntimeConfig
from ._bridge.batching import Batch
from ._bridge import telemetry

from . import _require_vllm
from . import batch_tensors


@dataclass
class _DecodeLookaheadBuffer:
    base_pos: int
    tokens: list[int]
    sampler_key: tuple
    next_idx: int = 1


class VllmEngine:
    """Inference engine that delegates the forward pass to a vllm model.

    Public surface matches the shared `Engine` contract:
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
    vllm_config: object
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
        vllm_config,
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
        self.vllm_config = vllm_config
        self.forward_pass = forward_pass
        self.kv_cache_at_layer = kv_cache_at_layer
        self.kv_cache_at_layer_host = kv_cache_at_layer_host or []
        self.swap_pool_size = swap_pool_size
        self.adapter_at_layer = adapter_at_layer
        self.arch_type = arch_type
        self.info = info
        self.snapshot_dir = snapshot_dir
        self.adapters = {}
        self._profile_enabled = bool(os.environ.get("PIE_VLLM_PROFILE"))
        self._last_fire_profile: dict[str, float] | None = None

        # Speculative decoding: driver-side n-gram drafter. Verification
        # and splice live in the shared `._bridge.batching.Batch`; this
        # engine owns drafting via `spec_step`. Buffers are lazy-init so
        # the numba JIT cost is only paid when spec is actually used.
        self._ngram_buffers = None
        self._ngram_history: dict[int, list[int]] = {}
        self._decode_lookahead: dict[int, _DecodeLookaheadBuffer] = {}

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

        def _debug_stage(msg: str) -> None:
            if os.environ.get("PIE_VLLM_DEBUG_LOAD"):
                print(
                    f"[pie-vllm-load rank={config.rank}] {msg}",
                    file=sys.stderr,
                    flush=True,
                )

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
        _debug_stage("load_vllm_model: begin")
        loaded = load_vllm_model(
            config, driver_config, log_queue=log_queue,
        )
        _debug_stage("load_vllm_model: done")
        _log("Loaded vllm model", "DEBUG")

        _debug_stage("maybe_wrap_full_cudagraph: begin")
        _maybe_wrap_full_cudagraph(loaded)
        _debug_stage("maybe_wrap_full_cudagraph: done")

        _debug_stage("allocate_and_bind_kv_cache: begin")
        kv_cache_at_layer = allocate_and_bind_kv_cache(loaded, config, driver_config)
        _debug_stage("allocate_and_bind_kv_cache: done")
        _debug_stage("allocate_host_pool: begin")
        host_kv, pool_size = allocate_host_pool(kv_cache_at_layer, config.swap_budget_bytes)
        _debug_stage("allocate_host_pool: done")

        # Wire vllm's CUDA-graph dispatcher when capture is enabled (i.e.
        # `enforce_eager=False`). With it disabled, set cg_dispatcher=None
        # and forward_pass falls through to the eager path everywhere.
        _debug_stage("maybe_init_cg_dispatcher: begin")
        cg_dispatcher = _maybe_init_cg_dispatcher(loaded.vllm_config)
        _debug_stage("maybe_init_cg_dispatcher: done")

        _debug_stage("VllmForwardPass: begin")
        forward_pass = VllmForwardPass(
            model=loaded.model,
            vllm_config=loaded.vllm_config,
            attn_backend=loaded.attn_backend,
            runtime_config=config,
            model_config=loaded.model_config,
            cg_dispatcher=cg_dispatcher,
        )
        _debug_stage("VllmForwardPass: done")

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
            _debug_stage("capture_vllm_cudagraphs: begin")
            _capture_vllm_cudagraphs(
                forward_pass=forward_pass,
                cg_dispatcher=cg_dispatcher,
                vllm_config=loaded.vllm_config,
                config=config,
            )
            _debug_stage("capture_vllm_cudagraphs: done")
            _log("Capture done", "INFO")

        return cls(
            config=config,
            driver_config=driver_config,
            model_config=loaded.model_config,
            vllm_config=loaded.vllm_config,
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
    def fire_batch(
        self, inputs: dict, sampling_metadata: dict, batch: Batch | None = None
    ) -> list:
        # This driver runs causal-only attention; user-supplied masks are
        # silently dropped. The capability is advertised via
        # DriverCapabilities so the runtime can route mask-dependent
        # inferlets elsewhere (currently only `native`).
        t0 = time.perf_counter()
        if batch is not None:
            lookahead_hit = self._try_consume_decode_lookahead(batch)
            if lookahead_hit is not None:
                if self._profile_enabled:
                    self._last_fire_profile = {
                        "embed": 0.0,
                        "transform": 0.0,
                        "sample": 0.0,
                        "fire_total": time.perf_counter() - t0,
                        "decode_lookahead_hit_ratio": 1.0,
                    }
                return lookahead_hit

        if self.forward_pass.use_input_ids_forward:
            input_ids = inputs["token_ids"]
            input_embeds = None
        else:
            input_ids = None
            input_embeds = self.forward_pass.embed_inputs(inputs)
        t_embed = time.perf_counter()

        hidden_states = self.forward_pass.transform(
            input_embeds=input_embeds,
            input_ids=input_ids,
            position_ids=inputs["position_ids"],
            qo_indptr=inputs["qo_indptr"],
            kv_page_indices=inputs["kv_page_indices"],
            kv_page_indptr=inputs["kv_page_indptr"],
            kv_last_page_lens=inputs["kv_last_page_lens"],
            qo_indptr_cpu=inputs.get("qo_indptr_cpu"),
            kv_page_indices_cpu=inputs.get("kv_page_indices_cpu"),
            kv_page_indptr_cpu=inputs.get("kv_page_indptr_cpu"),
            kv_last_page_lens_cpu=inputs.get("kv_last_page_lens_cpu"),
        )
        t_transform = time.perf_counter()

        out = None
        if batch is not None and self._decode_lookahead_limit(batch) > 1:
            out = self._try_generate_decode_lookahead(
                hidden_states=hidden_states,
                sampling_metadata=sampling_metadata,
                batch=batch,
                inputs=inputs,
            )
        if out is None:
            out = self.forward_pass.sample(hidden_states, sampling_metadata)
        t_sample = time.perf_counter()
        if self._profile_enabled:
            transform_profile = getattr(self.forward_pass, "_last_transform_profile", {})
            sample_profile = getattr(self.forward_pass, "_last_sample_profile", {})
            self._last_fire_profile = {
                "embed": t_embed - t0,
                "transform": t_transform - t_embed,
                "sample": t_sample - t_transform,
                "fire_total": t_sample - t0,
                "decode_lookahead_hit_ratio": 0.0,
                **{f"transform_{k}": v for k, v in transform_profile.items()},
                **{f"sample_{k}": v for k, v in sample_profile.items()},
            }
        return out

    def _token_result(self, tokens: list[int]) -> dict:
        n = len(tokens)
        return {
            "tokens": tokens,
            "dists": [None] * n,
            "logits": [None] * n,
            "logprobs": [None] * n,
            "entropies": [None] * n,
            "nan_indices": [],
        }

    def _decode_lookahead_limit(self, batch: Batch) -> int:
        limit = int(getattr(self.driver_config, "decode_lookahead_tokens", 1) or 1)
        if limit <= 1:
            return 1
        if not self._decode_lookahead_eligible(batch):
            return 1
        page_size = int(getattr(self.config, "kv_page_size", 0) or 0)
        if page_size <= 0:
            page_size = int(self.vllm_config.cache_config.block_size)
        last_len = int(batch.kv_last_page_lens[0]) if len(batch.kv_last_page_lens) else 0
        # Internal lookahead forwards write KV for every returned token except
        # the last one. Stay within the currently pinned page; the next
        # non-buffered step can allocate/advance normally.
        room = max(1, int(page_size) - last_len)
        return max(1, min(limit, room))

    def _decode_lookahead_eligible(self, batch: Batch) -> bool:
        if batch.has_speculative_inputs:
            return False
        if len(batch.request_output_counts) != 1:
            return False
        if int(batch.request_output_counts[0]) != 1:
            return False
        if not batch.indices_for_logits:
            return False
        if int(batch.indices_for_logits[0]) != int(batch.qo_indptr[1]) - 1:
            return False
        if batch.sampling_masks is not None or batch.logit_masks is not None:
            return False
        if bool(getattr(batch, "adapter_subpass_needed", False)):
            return False
        if not batch.context_ids:
            return False
        stype = int(batch.sampler_types[0])
        if stype not in batch_tensors.TOKEN_SAMPLING_TYPES:
            return False
        if float(batch.temperatures[0]) != 0.0:
            return False
        return True

    def _decode_lookahead_key(self, batch: Batch) -> tuple:
        seed = int(batch.sampler_seeds_arr[0]) if len(batch.sampler_seeds_arr) else 0
        return (
            int(batch.sampler_types[0]),
            float(batch.temperatures[0]),
            int(batch.top_k_values[0]),
            float(batch.top_p_values[0]),
            float(batch.min_p_values[0]),
            seed,
        )

    def _try_consume_decode_lookahead(self, batch: Batch) -> dict | None:
        if not self._decode_lookahead_eligible(batch):
            if batch.context_ids:
                self._decode_lookahead.pop(int(batch.context_ids[0]), None)
            return None
        session_id = int(batch.context_ids[0])
        buf = self._decode_lookahead.get(session_id)
        if buf is None:
            return None
        if buf.sampler_key != self._decode_lookahead_key(batch):
            self._decode_lookahead.pop(session_id, None)
            return None
        sampled_idx = int(batch.indices_for_logits[0])
        prefix_len = int(batch.position_ids[sampled_idx]) + 1
        rel = prefix_len - int(buf.base_pos)
        if rel < 0 or rel >= len(buf.tokens):
            self._decode_lookahead.pop(session_id, None)
            return None
        req_start = int(batch.qo_indptr[0])
        req_end = int(batch.qo_indptr[1])
        if req_end - req_start != 1:
            self._decode_lookahead.pop(session_id, None)
            return None
        if rel > 0 and int(batch.token_ids[req_start]) != int(buf.tokens[rel - 1]):
            self._decode_lookahead.pop(session_id, None)
            return None
        token = int(buf.tokens[rel])
        buf.next_idx = rel + 1
        if buf.next_idx >= len(buf.tokens):
            self._decode_lookahead.pop(session_id, None)
        else:
            self._decode_lookahead[session_id] = buf
        return self._token_result([token])

    def _try_generate_decode_lookahead(
        self,
        *,
        hidden_states: torch.Tensor,
        sampling_metadata: dict,
        batch: Batch,
        inputs: dict,
    ) -> dict | None:
        limit = self._decode_lookahead_limit(batch)
        if limit <= 1:
            return None
        first = self.forward_pass.greedy_top_token_tensor(
            hidden_states, sampling_metadata
        )
        if first is None or int(first.numel()) != 1:
            return None

        sampled_idx = int(batch.indices_for_logits[0])
        prefix_len = int(batch.position_ids[sampled_idx]) + 1
        session_id = int(batch.context_ids[0])
        sampler_key = self._decode_lookahead_key(batch)
        device = torch.device(self.config.device)
        token_tensors = [first.reshape(1)]
        prev_token = token_tensors[0].to(device=device, dtype=torch.long)

        page_size = int(getattr(self.config, "kv_page_size", 0) or 0)
        if page_size <= 0:
            page_size = int(self.vllm_config.cache_config.block_size)
        base_last_len = int(batch.kv_last_page_lens[0])
        qo_cpu = np.asarray([0, 1], dtype=np.int32)
        kv_indptr_cpu = np.asarray(batch.kv_page_indptr, dtype=np.int32)
        kv_indices_cpu = np.asarray(batch.kv_page_indices, dtype=np.int32)
        pos_cpu_all = np.arange(
            prefix_len, prefix_len + max(limit - 1, 0), dtype=np.int32
        )
        last_cpu_all = np.arange(
            base_last_len + 1, base_last_len + limit, dtype=np.int32
        )
        next_sampling = {
            **sampling_metadata,
            "indices_for_logits": [0],
            "all_logits_in_order": True,
        }
        qo_gpu = torch.as_tensor(qo_cpu, device=device, dtype=torch.int32)
        kv_indices_gpu = inputs["kv_page_indices"]
        kv_indptr_gpu = inputs["kv_page_indptr"]
        pos_gpu_all = torch.as_tensor(pos_cpu_all, device=device, dtype=torch.int32)
        last_gpu_all = torch.as_tensor(last_cpu_all, device=device, dtype=torch.int32)

        for i in range(1, limit):
            next_last_len = base_last_len + i
            if next_last_len > page_size:
                break
            last_cpu = last_cpu_all[i - 1 : i]
            pos_gpu = pos_gpu_all[i - 1 : i]
            last_gpu = last_gpu_all[i - 1 : i]
            hidden = self.forward_pass.transform(
                input_embeds=None,
                input_ids=prev_token,
                position_ids=pos_gpu,
                qo_indptr=qo_gpu,
                kv_page_indices=kv_indices_gpu,
                kv_page_indptr=kv_indptr_gpu,
                kv_last_page_lens=last_gpu,
                qo_indptr_cpu=qo_cpu,
                kv_page_indices_cpu=kv_indices_cpu,
                kv_page_indptr_cpu=kv_indptr_cpu,
                kv_last_page_lens_cpu=last_cpu,
            )
            next_token = self.forward_pass.greedy_top_token_tensor(
                hidden, next_sampling
            )
            if next_token is None or int(next_token.numel()) != 1:
                break
            prev_token = next_token.reshape(1).to(device=device, dtype=torch.long)
            token_tensors.append(prev_token)

        tokens_tensor = torch.cat(token_tensors, dim=0)
        tokens = [int(t) for t in tokens_tensor.tolist()]
        if len(tokens) > 1:
            self._decode_lookahead[session_id] = _DecodeLookaheadBuffer(
                base_pos=prefix_len,
                tokens=tokens,
                sampler_key=sampler_key,
            )
        else:
            self._decode_lookahead.pop(session_id, None)
        return self._token_result([tokens[0]])

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
        max_num_seqs = int(self.vllm_config.scheduler_config.max_num_seqs)
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
        echoing input config. In particular `kv_page_size` comes from the
        attention backend's chosen block size.

        Fails loudly if any expected value is missing — the runtime/Rust
        side relies on these being correct, so silent defaulting is unsafe.
        """
        from ._bridge.capabilities import DriverCapabilities

        vc = self.forward_pass.vllm_config
        mc = vc.model_config
        cc = vc.cache_config

        if self.config.total_pages is None:
            raise RuntimeError(
                "config.total_pages was not set by the loader — KV cache "
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

        # vllm resolves scheduler capacity while building VllmConfig. Pie
        # reports those resolved limits rather than accepting user overrides.
        max_forward_requests = int(vc.scheduler_config.max_num_seqs)
        max_forward_tokens = int(vc.scheduler_config.max_num_batched_tokens)
        unconstrained = (1 << 32) - 1

        return DriverCapabilities(
            total_pages=int(self.config.total_pages),
            kv_page_size=int(cc.block_size),
            swap_pool_size=int(self.swap_pool_size),
            max_forward_tokens=max_forward_tokens,
            max_forward_requests=max_forward_requests,
            max_page_refs=int(self.config.total_pages),
            max_logit_rows=unconstrained,
            max_prob_rows=unconstrained,
            max_custom_mask_bytes=unconstrained,
            max_sampler_rows=unconstrained,
            max_logprob_labels=unconstrained,
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


def _maybe_wrap_full_cudagraph(loaded) -> None:
    """Mirror GPUModelRunner's outer FULL CUDA-graph wrapper.

    vLLM's compiled model installs PIECEWISE wrappers inside the AOT graph, but
    FULL decode graphs require an additional wrapper around the whole model.
    Without this, dispatching FULL makes PIECEWISE wrappers fall through and no
    full graph is captured or replayed.
    """
    from ._vllm_compat import CUDAGraphMode

    vllm_config = loaded.vllm_config
    cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
    if cudagraph_mode is None or not cudagraph_mode.has_full_cudagraphs():
        return
    if getattr(vllm_config.parallel_config, "use_ubatching", False):
        return

    from vllm.compilation.cuda_graph import CUDAGraphWrapper

    if isinstance(loaded.model, CUDAGraphWrapper):
        return
    loaded.model = CUDAGraphWrapper(
        loaded.model,
        vllm_config,
        runtime_mode=CUDAGraphMode.FULL,
    )


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
    page_size = forward_pass._page_size
    assert page_size is not None
    device = forward_pass.device
    embed_dim = vllm_config.model_config.get_hidden_size()
    dtype = getattr(torch, config.activation_dtype)
    # vLLM's CUDAGraphWrapper captures/replays on vllm.utils.current_stream().
    # Initialize that stream before entering graph_capture so the capture
    # context restores the same non-default stream for runtime replay.
    from vllm.utils.torch_utils import current_stream as vllm_current_stream

    vllm_current_stream()

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
    capture (caller decides via `runtime_mode`). FULL graphs use uniform
    decode metadata. PIECEWISE captures are keyed by token count and accept
    any `num_reqs` per the wrapper's relax_for_mixed_batch logic, so their
    dummy metadata keeps the token count but caps request count to a valid
    scheduler shape."""
    from ._vllm_compat import CUDAGraphMode

    n = desc.num_tokens
    full_cg = runtime_mode == CUDAGraphMode.FULL
    if full_cg:
        # FULL decode graphs are captured as uniform decode: n requests, 1
        # query token each. The dispatcher only emits FULL descriptors within
        # max_num_seqs.
        num_reqs = n
        query_lens = torch.ones(num_reqs, dtype=torch.int32, device=device)
    else:
        # PIECEWISE graphs are keyed by token count and relax request count at
        # replay time. Keep the dummy shape valid for backends whose metadata
        # buffers are sized by max_num_seqs (FlashInfer), while preserving the
        # captured token count.
        max_reqs = int(vllm_config.scheduler_config.max_num_seqs)
        num_reqs = max(1, min(n, max_reqs))
        query_lens = torch.full(
            (num_reqs,), n // num_reqs, dtype=torch.int32, device=device,
        )
        query_lens[: n % num_reqs] += 1
    qo_indptr = torch.empty(num_reqs + 1, dtype=torch.int32, device=device)
    qo_indptr[0] = 0
    torch.cumsum(query_lens, dim=0, out=qo_indptr[1:])
    kv_page_indices = torch.full(
        (num_reqs,),
        -int(getattr(forward_pass, "_kv_block_id_offset", 0)),
        dtype=torch.int32,
        device=device,
    )
    kv_page_indptr = torch.arange(num_reqs + 1, dtype=torch.int32, device=device)
    kv_last_page_lens = query_lens.clone()

    common_cache = {}
    common = forward_pass._build_common_metadata(
        qo_indptr=qo_indptr,
        kv_page_indices=kv_page_indices,
        kv_page_indptr=kv_page_indptr,
        kv_last_page_lens=kv_last_page_lens,
        page_size=page_size,
        kernel_page_size=page_size,
        full_cg=full_cg,
        batch_desc=desc,
        for_cudagraph_capture=full_cg,
    )
    common_cache[page_size] = common
    backend_metadata = {}
    slot_mapping_dict = {}
    assert forward_pass._metadata_groups is not None
    for group in forward_pass._metadata_groups:
        metadata_kernel_block_size = int(group["metadata_kernel_block_size"])
        group_common = common_cache.get(metadata_kernel_block_size)
        if group_common is None:
            group_common = forward_pass._build_common_metadata(
                qo_indptr=qo_indptr,
                kv_page_indices=kv_page_indices,
                kv_page_indptr=kv_page_indptr,
                kv_last_page_lens=kv_last_page_lens,
                page_size=page_size,
                kernel_page_size=metadata_kernel_block_size,
                full_cg=full_cg,
                batch_desc=desc,
                for_cudagraph_capture=full_cg,
            )
            common_cache[metadata_kernel_block_size] = group_common
        if runtime_mode != CUDAGraphMode.NONE:
            # Match GPUModelRunner dummy capture: dummy runs must not write
            # KV/state cache. Runtime replay updates the same slot_mapping
            # buffer with real slots before launching the captured graph.
            group_common.slot_mapping.fill_(-1)
        if full_cg:
            metadata = group["builder"].build_for_cudagraph_capture(group_common)
        else:
            metadata = group["builder"].build(
                common_prefix_len=0,
                common_attn_metadata=group_common,
            )
        for name in group["layer_names"]:
            backend_metadata[name] = metadata
            slot_mapping_dict[name] = group_common.slot_mapping

    # Use the persistent buffers as inputs so the addresses captured here
    # match what transform() will pass at runtime. The values don't matter
    # for capture (only the shape and address do); zero them defensively.
    if forward_pass.use_input_ids_forward:
        forward_pass._buf_input_ids[:n].zero_()
        input_ids = forward_pass._buf_input_ids[:n]
        input_embeds = None
    else:
        forward_pass._buf_input_embeds[:n].zero_()
        input_ids = None
        input_embeds = forward_pass._buf_input_embeds[:n]
    forward_pass._buf_positions[:n].zero_()
    positions = forward_pass._buf_positions[:n]

    with set_forward_context(
        attn_metadata=backend_metadata,
        vllm_config=vllm_config,
        num_tokens=n,
        slot_mapping=slot_mapping_dict,
        cudagraph_runtime_mode=runtime_mode,
        batch_descriptor=desc,
    ):
        forward_pass._call_model_with_stream_sync(
            input_ids=input_ids,
            positions=positions,
            input_embeds=input_embeds,
            use_vllm_stream=runtime_mode != CUDAGraphMode.NONE,
        )
