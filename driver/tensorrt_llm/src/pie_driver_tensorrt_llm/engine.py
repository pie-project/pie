"""TensorRT-LLM engine adapter for Pie's subprocess worker contract."""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any


TOKEN_SAMPLING_TYPES = frozenset({1, 2, 3, 4, 5})
SPECIAL_SAMPLING_TYPES = frozenset({0, 6, 7, 8, 9, 10})


@dataclass
class _ModelInfo:
    arch_name: str
    vocab_size: int
    max_model_len: int
    eos_token_id: int | None = None
    pad_token_id: int | None = None


@dataclass
class _LookaheadBuffer:
    base_pos: int
    tokens: list[int]
    next_idx: int
    sampler_key: tuple[Any, ...]


@dataclass
class _PyExecutorSession:
    request_id: int
    request: Any
    sampler_key: tuple[Any, ...]


def _resolve_hf_snapshot_dir(hf_repo: str) -> str | None:
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
            snapshots = sorted(
                repo_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
            )
            if snapshots:
                return str(snapshots[0])
    except Exception:
        pass

    return None


def _read_model_info(model_path: str, trust_remote_code: bool) -> _ModelInfo:
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        text_cfg = getattr(cfg, "text_config", None)
        archs = getattr(cfg, "architectures", None) or []
        arch_name = str(archs[0]) if archs else str(getattr(cfg, "model_type", "unknown"))
        vocab_size = int(
            getattr(cfg, "vocab_size", 0)
            or getattr(text_cfg, "vocab_size", 0)
            or getattr(cfg, "padded_vocab_size", 0)
            or getattr(text_cfg, "padded_vocab_size", 0)
            or 128000
        )
        max_model_len = int(
            getattr(cfg, "max_position_embeddings", 0)
            or getattr(text_cfg, "max_position_embeddings", 0)
            or getattr(cfg, "seq_length", 0)
            or getattr(text_cfg, "seq_length", 0)
            or getattr(cfg, "n_positions", 0)
            or getattr(text_cfg, "n_positions", 0)
            or getattr(cfg, "model_max_length", 0)
            or getattr(text_cfg, "model_max_length", 0)
            or 4096
        )
        eos_token_id = _optional_token_id(
            getattr(cfg, "eos_token_id", None)
            if getattr(cfg, "eos_token_id", None) is not None
            else getattr(text_cfg, "eos_token_id", None)
        )
        pad_token_id = _optional_token_id(
            getattr(cfg, "pad_token_id", None)
            if getattr(cfg, "pad_token_id", None) is not None
            else getattr(text_cfg, "pad_token_id", None)
        )
        return _ModelInfo(
            arch_name,
            vocab_size,
            max_model_len,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
    except Exception:
        return _ModelInfo("unknown", 128000, 4096)


class TensorRTLLMEngine:
    """Adapter from Pie's per-step batch contract to TensorRT-LLM LLM.generate.

    The public TensorRT-LLM API is request-oriented rather than Pie-KV-page
    oriented. This engine keeps per-context token histories and asks
    TensorRT-LLM for one generated token at each token-producing Pie sample.
    """

    def __init__(
        self,
        *,
        llm: Any,
        sampling_params_cls: type,
        config,
        driver_config,
        model_info: _ModelInfo,
        snapshot_dir: str,
        pyexecutor: Any | None = None,
        execution_mode: str = "generate",
    ) -> None:
        self.llm = llm
        self.sampling_params_cls = sampling_params_cls
        self.config = config
        self.driver_config = driver_config
        self.model_info = model_info
        self.model_config = model_info
        self.snapshot_dir = snapshot_dir
        self.adapters: dict[int, Any] = {}
        self._histories: OrderedDict[int, list[int]] = OrderedDict()
        self._lookahead: OrderedDict[int, _LookaheadBuffer] = OrderedDict()
        self.execution_mode = _validate_execution_mode(execution_mode)
        self.pyexecutor = pyexecutor
        self._pyexecutor_sessions: dict[int, _PyExecutorSession] = {}
        self._next_pyexecutor_id = 1
        self._emitted_token_counts: dict[int, int] = {}
        self.config.total_pages = int(driver_config.virtual_total_pages)

    @classmethod
    def load(cls, config, driver_config) -> "TensorRTLLMEngine":
        if int(config.tensor_parallel_size) != 1:
            raise NotImplementedError(
                "pie_driver_tensorrt_llm currently supports tensor_parallel_size = 1. "
                "TensorRT-LLM's high-level LLM API does not expose Pie's "
                "per-rank external KV-page contract."
            )

        execution_mode = _validate_execution_mode(
            getattr(driver_config, "execution_mode", "generate")
        )
        if execution_mode == "pyexecutor":
            _validate_pyexecutor_driver_config(driver_config)

        import tensorrt_llm
        from tensorrt_llm import LLM, SamplingParams

        if execution_mode == "pyexecutor":
            version = str(getattr(tensorrt_llm, "__version__", ""))
            if not version.startswith("1.2.1"):
                raise RuntimeError(
                    "TensorRT-LLM pyexecutor mode is pinned to TensorRT-LLM "
                    f"1.2.1 private APIs, but imported {version or 'unknown'}."
                )

        snapshot_dir = str(getattr(config, "snapshot_dir", "") or "")
        if not snapshot_dir:
            snapshot_dir = _resolve_hf_snapshot_dir(config.hf_repo) or ""

        model_path = snapshot_dir or config.hf_repo
        model_info = _read_model_info(model_path, driver_config.trust_remote_code)
        if driver_config.max_seq_len is not None:
            max_seq_len = int(driver_config.max_seq_len)
            if max_seq_len <= 0:
                raise ValueError("max_seq_len must be positive")
            model_info.max_model_len = min(model_info.max_model_len, max_seq_len)

        llm_kwargs = dict(driver_config.llm_kwargs or {})
        if driver_config.backend is not None:
            llm_kwargs.setdefault("backend", driver_config.backend)
        if driver_config.attn_backend is not None:
            llm_kwargs.setdefault("attn_backend", driver_config.attn_backend)
        if driver_config.enable_chunked_prefill is not None:
            llm_kwargs.setdefault(
                "enable_chunked_prefill", driver_config.enable_chunked_prefill
            )
        if driver_config.max_seq_len is not None:
            llm_kwargs.setdefault("max_seq_len", int(driver_config.max_seq_len))
        if driver_config.max_batch_size is not None:
            llm_kwargs.setdefault("max_batch_size", int(driver_config.max_batch_size))
        if driver_config.max_num_tokens is not None:
            llm_kwargs.setdefault("max_num_tokens", int(driver_config.max_num_tokens))
        if driver_config.kv_cache_free_gpu_memory_fraction is not None:
            from tensorrt_llm.llmapi import KvCacheConfig

            llm_kwargs.setdefault(
                "kv_cache_config",
                KvCacheConfig(
                    free_gpu_memory_fraction=float(
                        driver_config.kv_cache_free_gpu_memory_fraction
                    )
                ),
            )
        if execution_mode == "pyexecutor":
            _configure_pyexecutor_llm_kwargs(llm_kwargs, driver_config)

        llm = LLM(
            model=model_path,
            skip_tokenizer_init=bool(driver_config.skip_tokenizer_init),
            trust_remote_code=bool(driver_config.trust_remote_code),
            tensor_parallel_size=int(config.tensor_parallel_size),
            dtype=_normalize_dtype(config.activation_dtype),
            **llm_kwargs,
        )

        if not snapshot_dir:
            raise RuntimeError(
                "Could not resolve a local HF snapshot directory for "
                f"{config.hf_repo!r}. Pie needs a snapshot path for tokenizer.json."
            )

        pyexecutor = None
        if execution_mode == "pyexecutor":
            pyexecutor = _extract_pyexecutor(llm)
            _stop_pyexecutor_worker(
                pyexecutor,
                timeout_s=float(
                    getattr(driver_config, "pyexecutor_worker_stop_timeout_s", 30.0)
                ),
            )

        return cls(
            llm=llm,
            sampling_params_cls=SamplingParams,
            config=config,
            driver_config=driver_config,
            model_info=model_info,
            snapshot_dir=snapshot_dir,
            pyexecutor=pyexecutor,
            execution_mode=execution_mode,
        )

    def query(self, query: str) -> str:
        return "pong" if query == "ping" else "unknown query"

    def prewarm(self) -> None:
        if self.execution_mode != "pyexecutor" or self.pyexecutor is None:
            return

        capacity = self._pyexecutor_session_capacity() or 1
        default_batch_size = min(int(capacity), 8)
        batch_size = max(
            1,
            min(
                int(capacity),
                int(
                    os.environ.get(
                        "PIE_TRTLLM_PREWARM_BATCH_SIZE", default_batch_size
                    )
                    or 1
                ),
            ),
        )
        prompt_tokens = max(
            1, int(os.environ.get("PIE_TRTLLM_PREWARM_PROMPT_TOKENS", "32") or 32)
        )

        from types import SimpleNamespace

        base_context_id = -1_000_000_000
        context_ids = [base_context_id - i for i in range(batch_size)]
        warm_token = int(os.environ.get("PIE_TRTLLM_PREWARM_TOKEN_ID", "100") or 100)
        if warm_token < 0 or warm_token >= int(self.model_info.vocab_size):
            warm_token = 0
        if warm_token in {
            t
            for t in (self.model_info.pad_token_id, self.model_info.eos_token_id)
            if t is not None
        }:
            warm_token = 100 if int(self.model_info.vocab_size) > 100 else 0

        token_ids: list[int] = []
        position_ids: list[int] = []
        qo_indptr = [0]
        for _ in context_ids:
            token_ids.extend([int(warm_token)] * prompt_tokens)
            position_ids.extend(range(prompt_tokens))
            qo_indptr.append(len(token_ids))

        batch = SimpleNamespace(
            has_speculative_inputs=False,
            adapter_subpass_needed=False,
            sampling_masks=None,
            logit_masks=None,
            request_output_counts=[1] * batch_size,
            qo_indptr=qo_indptr,
            token_ids=token_ids,
            position_ids=position_ids,
            context_ids=context_ids,
            kv_page_indptr=[0] * (batch_size + 1),
            kv_page_indices=[],
            sampler_types=[3] * batch_size,
            indices_for_logits=[qo_indptr[i + 1] - 1 for i in range(batch_size)],
            temperatures=[0.0] * batch_size,
            top_k_values=[0] * batch_size,
            top_p_values=[1.0] * batch_size,
            min_p_values=[0.0] * batch_size,
            sampler_seeds_arr=[0] * batch_size,
            output_spec_flags=[True] * batch_size,
        )

        try:
            self.fire_batch({"batch": batch}, {"batch": batch})
        finally:
            for session_id in context_ids:
                session = self._pyexecutor_sessions.get(int(session_id))
                if session is not None:
                    self._terminate_pyexecutor_session(int(session_id), session)
                self._histories.pop(int(session_id), None)
                self._lookahead.pop(int(session_id), None)
                self._emitted_token_counts.pop(int(session_id), None)

    def capabilities(self):
        from ._bridge.capabilities import DriverCapabilities

        unconstrained = (1 << 32) - 1
        max_forward_requests = int(self.driver_config.max_concurrent_requests)
        if self.execution_mode == "pyexecutor":
            capacity = self._pyexecutor_session_capacity()
            if capacity is not None:
                max_forward_requests = min(max_forward_requests, int(capacity))
        return DriverCapabilities(
            total_pages=int(self.config.total_pages),
            kv_page_size=int(self.driver_config.virtual_kv_page_size),
            swap_pool_size=0,
            max_forward_tokens=int(self.driver_config.max_batched_tokens),
            max_forward_requests=max_forward_requests,
            max_page_refs=int(self.config.total_pages),
            max_logit_rows=int(self.driver_config.max_batched_tokens),
            max_prob_rows=int(self.driver_config.max_batched_tokens),
            max_custom_mask_bytes=0,
            max_sampler_rows=unconstrained,
            max_logprob_labels=0,
            arch_name=self.model_info.arch_name,
            vocab_size=int(self.model_info.vocab_size),
            max_model_len=int(self.model_info.max_model_len),
            activation_dtype=str(self.config.activation_dtype),
            supports_user_attention_mask=False,
            supports_adapters=False,
            snapshot_dir=str(self.snapshot_dir),
        )

    def build_model_inputs(self, batch) -> dict:
        return {"batch": batch}

    def build_sampling_metadata(self, batch) -> dict:
        return {"batch": batch}

    @staticmethod
    def _sampling_result(tokens: list[int], **extra: Any) -> dict:
        result = {
            "tokens": tokens,
            "dists": [None] * len(tokens),
            "logits": [None] * len(tokens),
            "logprobs": [None] * len(tokens),
            "entropies": [None] * len(tokens),
        }
        result.update(extra)
        return result

    def fire_batch(self, inputs: dict, sampling_metadata: dict) -> dict:
        batch = inputs["batch"]
        self._reject_unsupported_batch_features(batch)
        self._append_pending_tokens(batch)
        if self.execution_mode == "pyexecutor":
            return self._fire_batch_pyexecutor(batch)

        work = self._prepare_generation_work(batch)
        if not work:
            return self._sampling_result([])

        tokens: list[int | None] = [None] * len(work)
        session_counts = Counter(int(item["session_id"]) for item in work)
        generate_work: list[tuple[int, dict[str, Any]]] = []
        for i, item in enumerate(work):
            buffered = item.get("buffered_token")
            if buffered is None:
                if session_counts[int(item["session_id"])] > 1:
                    token = self._consume_item_or_generate_one(item)
                    tokens[i] = int(token)
                else:
                    generate_work.append((i, item))
            else:
                tokens[i] = int(buffered)

        if generate_work:
            for out_idx, item, token_ids in self._generate_many(generate_work):
                self._append_generated_sequence(item, token_ids)
                tokens[out_idx] = int(token_ids[0])

        final_tokens = [int(t) for t in tokens if t is not None]
        if len(final_tokens) != len(work):
            raise RuntimeError("TensorRT-LLM lookahead path failed to fill all tokens")
        return self._sampling_result(final_tokens)

    def kv_copy_d2h(self, phys_ids: list[int], slots: list[int]) -> None:
        self._validate_copy_args(phys_ids, slots)

    def kv_copy_h2d(self, phys_ids: list[int], slots: list[int]) -> None:
        self._validate_copy_args(phys_ids, slots)

    def kv_copy_d2d(self, src_phys_ids: list[int], dst_phys_ids: list[int]) -> None:
        self._validate_copy_args(src_phys_ids, dst_phys_ids)

    def kv_copy_h2h(self, src_slots: list[int], dst_slots: list[int]) -> None:
        self._validate_copy_args(src_slots, dst_slots)

    def init_adapter(self, *args, **kwargs):
        raise NotImplementedError(
            "Adapters are not supported on the TensorRT-LLM driver. "
            "Use the `native` or `dev` driver for adapter workloads."
        )

    def update_adapter(self, *args, **kwargs):
        raise NotImplementedError("Adapters are not supported on the TensorRT-LLM driver.")

    def load_adapter(self, *args, **kwargs):
        raise NotImplementedError("Adapters are not supported on the TensorRT-LLM driver.")

    def save_adapter(self, *args, **kwargs):
        return b""

    def _reject_unsupported_batch_features(self, batch) -> None:
        if batch.has_speculative_inputs and not (
            self.execution_mode == "pyexecutor"
            and bool(
                getattr(
                    self.driver_config,
                    "pyexecutor_speculative_lookahead",
                    False,
                )
            )
        ):
            raise NotImplementedError(
                "TensorRT-LLM driver does not support Pie speculative verification yet."
            )
        if batch.adapter_subpass_needed:
            raise NotImplementedError(
                "TensorRT-LLM driver does not support Pie adapter subpasses."
            )
        if batch.sampling_masks is not None or batch.logit_masks is not None:
            raise NotImplementedError(
                "TensorRT-LLM driver does not support Pie sampling/logit masks yet."
            )
        if any(int(t) in SPECIAL_SAMPLING_TYPES for t in batch.sampler_types):
            raise NotImplementedError(
                "TensorRT-LLM driver currently supports only token-producing "
                "samplers (Multinomial, TopK, TopP, MinP, TopKTopP)."
            )

    def _append_pending_tokens(self, batch) -> None:
        num_requests = len(batch.request_output_counts)
        for req_idx in range(num_requests):
            start = int(batch.qo_indptr[req_idx])
            end = int(batch.qo_indptr[req_idx + 1])
            if end <= start:
                continue
            session_id = self._session_id(batch, req_idx)
            history = self._history_for(session_id)
            tokens = [int(t) for t in batch.token_ids[start:end]]
            positions = [int(p) for p in batch.position_ids[start:end]]
            if self._merge_tokens(history, tokens, positions, session_id):
                self._drop_lookahead(session_id)
            self._trim_history(history)

    def _prepare_generation_work(
        self, batch, *, use_lookahead: bool = True
    ) -> list[dict[str, Any]]:
        work: list[dict[str, Any]] = []
        sampler_cursor = 0
        num_requests = len(batch.request_output_counts)
        for req_idx in range(num_requests):
            num_outputs = int(batch.request_output_counts[req_idx])
            if num_outputs == 0:
                continue
            if num_outputs != 1:
                raise NotImplementedError(
                    "TensorRT-LLM driver supports one token sampler per Pie request."
                )

            sampler_idx = sampler_cursor
            sampler_cursor += num_outputs
            stype = int(batch.sampler_types[sampler_idx])
            if stype not in TOKEN_SAMPLING_TYPES:
                raise NotImplementedError(
                    f"TensorRT-LLM driver cannot handle sampler type {stype}."
                )

            req_start = int(batch.qo_indptr[req_idx])
            req_end = int(batch.qo_indptr[req_idx + 1])
            sampled_global_idx = int(batch.indices_for_logits[sampler_idx])
            if sampled_global_idx != req_end - 1:
                rel = sampled_global_idx - req_start
                raise NotImplementedError(
                    "TensorRT-LLM LLM.generate can only sample the next token "
                    f"after the request tail; got relative sample index {rel}."
                )

            session_id = self._session_id(batch, req_idx)
            history = self._histories.get(session_id)
            if not history:
                raise RuntimeError(
                    f"Missing token history for Pie context {session_id}; "
                    "TensorRT-LLM driver cannot restore/fork contexts without replay."
                )
            prefix_len = int(batch.position_ids[sampled_global_idx]) + 1
            if prefix_len > len(history):
                tokens = [int(t) for t in batch.token_ids[req_start:req_end]]
                positions = [int(p) for p in batch.position_ids[req_start:req_end]]
                if self._merge_tokens(history, tokens, positions, session_id):
                    self._drop_lookahead(session_id)
            if prefix_len > len(history):
                trimmed_hint = (
                    " This usually means max_history_tokens trimmed an active prefix."
                    if self.driver_config.max_history_tokens is not None
                    else ""
                )
                raise RuntimeError(
                    f"Pie context {session_id} needs prefix length {prefix_len}, "
                    f"but the TensorRT-LLM replay history only contains "
                    f"{len(history)} tokens.{trimmed_hint}"
                )

            sampler_key = self._sampler_key(batch, sampler_idx, stype)
            buffered_token = None
            max_tokens = 1
            if use_lookahead:
                buffered_token = self._consume_lookahead(
                    session_id, prefix_len, sampler_key
                )
                max_tokens = self._lookahead_tokens(batch, sampler_idx, stype)

            work.append(
                {
                    "req_idx": req_idx,
                    "session_id": session_id,
                    "prefix_len": prefix_len,
                    "prompt": list(history[:prefix_len]),
                    "params": self._sampling_params(
                        batch,
                        sampler_idx,
                        stype,
                        max_tokens=max_tokens,
                    ),
                    "sampler_idx": sampler_idx,
                    "stype": stype,
                    "sampler_key": sampler_key,
                    "buffered_token": buffered_token,
                    "max_tokens": max_tokens,
                }
            )
        return work

    def _fire_batch_pyexecutor(self, batch) -> dict:
        debug_t0 = time.perf_counter() if _pyexecutor_debug_enabled() else None
        if self.pyexecutor is None:
            raise RuntimeError("TensorRT-LLM pyexecutor mode was not initialized")
        if batch.has_speculative_inputs:
            return self._fire_batch_pyexecutor_speculative(batch)

        use_lookahead = bool(
            getattr(self.driver_config, "pyexecutor_lookahead", False)
        ) or (
            bool(
                getattr(
                    self.driver_config,
                    "pyexecutor_speculative_lookahead",
                    False,
                )
            )
            and any(bool(v) for v in getattr(batch, "output_spec_flags", []))
        )
        if use_lookahead and not self._pyexecutor_lookahead_batch_is_worthwhile(batch):
            use_lookahead = False
        work = self._prepare_generation_work(batch, use_lookahead=use_lookahead)
        if not work:
            return self._sampling_result([])

        if (
            use_lookahead
            and bool(
                getattr(
                    self.driver_config,
                    "pyexecutor_speculative_lookahead",
                    False,
                )
            )
            and any(bool(v) for v in getattr(batch, "output_spec_flags", []))
        ):
            return self._fire_batch_pyexecutor_direct_accepted(batch, work)

        tokens: list[int | None] = [None] * len(work)
        session_counts = Counter(int(item["session_id"]) for item in work)
        generate_work: list[tuple[int, dict[str, Any]]] = []
        for out_idx, item in enumerate(work):
            buffered = item.get("buffered_token")
            if buffered is not None:
                tokens[out_idx] = int(buffered)
            elif session_counts[int(item["session_id"])] > 1:
                tokens[out_idx] = self._pyexecutor_consume_item_or_generate(item)
            else:
                generate_work.append((out_idx, item))

        if generate_work:
            for out_idx, item, token_ids in self._pyexecutor_generate_many(generate_work):
                self._append_generated_sequence(item, token_ids)
                tokens[out_idx] = int(token_ids[0])

        final_tokens = [int(t) for t in tokens if t is not None]
        if len(final_tokens) != len(work):
            raise RuntimeError("TensorRT-LLM pyexecutor path failed to fill all tokens")
        if any(bool(v) for v in getattr(batch, "output_spec_flags", [])):
            for item, token in zip(work, tokens):
                if token is not None:
                    self._record_emitted_tokens(int(item["session_id"]), 1)
        result = self._sampling_result(final_tokens)
        if debug_t0 is not None:
            elapsed_ms = (time.perf_counter() - debug_t0) * 1000
            sessions = {
                int(session_id): int(session.request_id)
                for session_id, session in self._pyexecutor_sessions.items()
            }
            print(
                "[pie-trtllm-pyexec] fire "
                f"work={len(work)} elapsed_ms={elapsed_ms:.3f} "
                f"sessions={sessions}",
                flush=True,
            )
        return result

    def _fire_batch_pyexecutor_direct_accepted(
        self, batch, work: list[dict[str, Any]]
    ) -> dict:
        tokens: list[int | None] = [None] * len(work)
        accepted_per_req: list[list[int] | None] = [
            None
        ] * len(batch.request_output_counts)
        generate_work: list[tuple[int, dict[str, Any]]] = []

        for out_idx, item in enumerate(work):
            buffered = item.get("buffered_token")
            if buffered is not None:
                token = int(buffered)
                tokens[out_idx] = token
                accepted_per_req[int(item["req_idx"])] = [token]
                self._record_emitted_tokens(int(item["session_id"]), 1)
                continue

            max_tokens = self._direct_accept_max_tokens(item)
            if max_tokens <= 0:
                token = self._pyexecutor_consume_item_or_generate(item)
                tokens[out_idx] = int(token)
                accepted_per_req[int(item["req_idx"])] = [int(token)]
                self._record_emitted_tokens(int(item["session_id"]), 1)
                continue
            if max_tokens != int(item.get("max_tokens", 1)):
                item = dict(item)
                item["max_tokens"] = max_tokens
                item["params"] = self._sampling_params(
                    batch,
                    int(item["sampler_idx"]),
                    int(item["stype"]),
                    max_tokens=max_tokens,
                )
            generate_work.append((out_idx, item))

        if generate_work:
            for out_idx, item, token_ids in self._pyexecutor_generate_many(generate_work):
                emitted = [int(t) for t in token_ids]
                if not emitted:
                    raise RuntimeError(
                        "TensorRT-LLM pyexecutor returned no accepted tokens"
                    )
                self._append_generated_sequence(item, emitted, keep_lookahead=False)
                self._record_emitted_tokens(int(item["session_id"]), len(emitted))
                accepted_per_req[int(item["req_idx"])] = emitted
                tokens[out_idx] = int(emitted[0])

        result_tokens = [int(t) for t in tokens if t is not None]
        if len(result_tokens) != len(work):
            raise RuntimeError(
                "TensorRT-LLM direct-accepted path failed to fill all tokens"
            )
        return self._sampling_result(
            result_tokens,
            spec_accepted_tokens=accepted_per_req,
        )

    def _fire_batch_pyexecutor_speculative(self, batch) -> dict:
        batch._build_spec_plan()
        inferlet_slots = int(sum(int(n) for n in batch.request_output_counts))
        tokens: list[int] = [0] * inferlet_slots
        work = self._prepare_generation_work(batch, use_lookahead=False)
        normal_work = [
            (out_idx, item)
            for out_idx, item in enumerate(work)
            if batch._spec_plan[int(item["req_idx"])] is None
        ]
        for out_idx, item, token_ids in self._pyexecutor_generate_many(normal_work):
            self._append_generated_sequence(item, token_ids)
            tokens[out_idx] = int(token_ids[0])

        verify_tokens: list[int] = []
        verify_slot_starts: list[tuple[int, int] | None] = [
            None
        ] * len(batch.request_output_counts)

        for req_idx, plan in enumerate(batch._spec_plan):
            if plan is None:
                continue
            n_drafts = int(plan["n_drafts"])
            drafts_start = int(plan["drafts_start"])
            drafts = [
                int(t)
                for t in batch.spec_token_ids[drafts_start : drafts_start + n_drafts]
            ]
            session_id = self._session_id(batch, req_idx)
            bonus = self._trusted_lookahead_bonus(session_id, drafts)
            verify_slot_starts[req_idx] = (len(verify_tokens), n_drafts)
            verify_tokens.extend(drafts)
            verify_tokens.append(bonus)
            if _pyexecutor_debug_enabled():
                print(
                    "[pie-trtllm-pyexec] spec_verify "
                    f"session={session_id} drafts={len(drafts)} bonus=1",
                    flush=True,
                )

        batch._verify_slot_starts = verify_slot_starts
        batch._verify_block_offset = inferlet_slots
        tokens.extend(verify_tokens)
        return self._sampling_result(tokens)

    def spec_step(
        self, sessions: list[tuple[int, list[int]]]
    ) -> list[list[int]]:
        if not bool(
            getattr(self.driver_config, "pyexecutor_speculative_lookahead", False)
        ):
            return [[] for _ in sessions]

        drafts: list[list[int]] = []
        for session_id, accepted in sessions:
            session_id = int(session_id)
            buf = self._lookahead.get(session_id)
            if buf is None:
                drafts.append([])
                continue
            accepted = [int(t) for t in accepted]
            if accepted:
                if int(buf.next_idx) > 0 and accepted[-1] != int(
                    buf.tokens[int(buf.next_idx) - 1]
                ):
                    if _pyexecutor_debug_enabled():
                        print(
                            "[pie-trtllm-pyexec] spec_step_drop "
                            f"session={session_id} accepted_last={accepted[-1]} "
                            f"expected={int(buf.tokens[int(buf.next_idx) - 1])}",
                            flush=True,
                        )
                    self._drop_lookahead(session_id)
                    drafts.append([])
                    continue
                start = int(buf.next_idx)
                end = max(start, len(buf.tokens) - 1)
                chain = [int(t) for t in buf.tokens[start:end]]
                self._lookahead.move_to_end(session_id)
                if _pyexecutor_debug_enabled():
                    print(
                        "[pie-trtllm-pyexec] spec_step "
                        f"session={session_id} accepted={len(accepted)} "
                        f"drafts={len(chain)}",
                        flush=True,
                    )
                drafts.append(chain)
            else:
                if _pyexecutor_debug_enabled():
                    print(
                        "[pie-trtllm-pyexec] spec_step "
                        f"session={session_id} accepted=0 drafts=0",
                        flush=True,
                    )
                drafts.append([])
        return drafts

    def _trusted_lookahead_bonus(self, session_id: int, drafts: list[int]) -> int:
        buf = self._lookahead.get(int(session_id))
        if buf is None:
            raise RuntimeError(
                "TensorRT-LLM speculative verification received drafts without "
                f"a lookahead buffer for session {session_id}"
            )
        start = int(buf.next_idx)
        end = start + len(drafts)
        expected = [int(t) for t in buf.tokens[start:end]]
        if expected != [int(t) for t in drafts]:
            self._drop_lookahead(session_id)
            raise RuntimeError(
                "TensorRT-LLM speculative drafts diverged from the trusted "
                f"lookahead buffer for session {session_id}"
            )
        if end >= len(buf.tokens):
            self._drop_lookahead(session_id)
            raise RuntimeError(
                "TensorRT-LLM speculative verification needs a reserved "
                f"bonus token for session {session_id}"
            )
        bonus = int(buf.tokens[end])
        buf.next_idx = end + 1
        if buf.next_idx >= len(buf.tokens):
            self._drop_lookahead(session_id)
        else:
            self._lookahead.move_to_end(session_id)
        return bonus

    def _pyexecutor_generate_many(
        self, generate_work: list[tuple[int, dict[str, Any]]]
    ) -> list[tuple[int, dict[str, Any], list[int]]]:
        out: list[tuple[int, dict[str, Any], list[int]]] = []
        remaining = list(generate_work)
        while remaining:
            seen_sessions: set[int] = set()
            wave: list[tuple[int, dict[str, Any]]] = []
            next_remaining: list[tuple[int, dict[str, Any]]] = []
            for out_idx, item in remaining:
                session_id = int(item["session_id"])
                if session_id in seen_sessions:
                    next_remaining.append((out_idx, item))
                    continue
                seen_sessions.add(session_id)
                wave.append((out_idx, item))

            out.extend(self._pyexecutor_generate_wave(wave))
            remaining = next_remaining
        return out

    def _pyexecutor_generate_wave(
        self, wave: list[tuple[int, dict[str, Any]]]
    ) -> list[tuple[int, dict[str, Any], list[int]]]:
        self._prepare_pyexecutor_capacity_for_wave(wave)
        pending: dict[int, tuple[int, dict[str, Any], _PyExecutorSession]] = {}
        targets: dict[int, int] = {}
        for out_idx, item in wave:
            session = self._pyexecutor_session_for_item(item)
            request_id = int(session.request_id)
            pending[request_id] = (out_idx, item, session)
            targets[request_id] = max(1, int(item.get("max_tokens", 1)))

        generated = self._drive_pyexecutor_until_tokens(targets)
        self._clear_pyexecutor_responses(set(pending))
        return [
            (out_idx, item, [int(t) for t in generated[int(session.request_id)]])
            for out_idx, item, session in pending.values()
        ]

    def _prepare_pyexecutor_capacity_for_wave(
        self, wave: list[tuple[int, dict[str, Any]]]
    ) -> None:
        capacity = self._pyexecutor_session_capacity()
        if capacity is None:
            return
        if capacity <= 0:
            raise RuntimeError("TensorRT-LLM pyexecutor session capacity must be positive")

        wave_items: dict[int, dict[str, Any]] = {}
        for _, item in wave:
            wave_items[int(item["session_id"])] = item
        if len(wave_items) > capacity:
            raise RuntimeError(
                "TensorRT-LLM pyexecutor wave needs "
                f"{len(wave_items)} sessions but max_batch_size is {capacity}"
            )

        for session_id, item in wave_items.items():
            session = self._pyexecutor_sessions.get(session_id)
            if session is not None and not self._pyexecutor_request_matches(session, item):
                self._terminate_pyexecutor_session(session_id, session)

        new_sessions = sum(
            1 for session_id in wave_items if session_id not in self._pyexecutor_sessions
        )
        excess = len(self._pyexecutor_sessions) + new_sessions - capacity
        if excess <= 0:
            return

        wave_session_ids = set(wave_items)
        for session_id, session in list(self._pyexecutor_sessions.items()):
            if excess <= 0:
                break
            if session_id in wave_session_ids:
                continue
            self._terminate_pyexecutor_session(session_id, session)
            excess -= 1

        if excess > 0:
            raise RuntimeError(
                "TensorRT-LLM pyexecutor could not free enough sequence slots "
                f"for wave of {len(wave_items)} sessions"
            )

    def _pyexecutor_session_capacity(self) -> int | None:
        value = getattr(self.driver_config, "max_batch_size", None)
        if value is None:
            llm_kwargs = getattr(self.driver_config, "llm_kwargs", None) or {}
            if isinstance(llm_kwargs, dict):
                value = llm_kwargs.get("max_batch_size")
        if value is not None:
            return int(value)

        if self.pyexecutor is None:
            return None
        try:
            resource_managers = self.pyexecutor.resource_manager.resource_managers
        except Exception:
            return None
        for manager in resource_managers.values():
            if manager.__class__.__name__ == "SeqSlotManager":
                get_count = getattr(manager, "get_max_resource_count", None)
                if get_count is not None:
                    return int(get_count())
        return None

    def _pyexecutor_lookahead_batch_is_worthwhile(self, batch) -> bool:
        min_batch_size = getattr(
            self.driver_config, "pyexecutor_lookahead_min_batch_size", None
        )
        if min_batch_size is None:
            min_batch_size = self._pyexecutor_session_capacity() or 1
        min_batch_size = int(min_batch_size)
        if min_batch_size <= 1:
            return True
        output_count = sum(int(n) for n in batch.request_output_counts)
        return output_count >= min_batch_size

    def _pyexecutor_session_for_item(self, item: dict[str, Any]) -> _PyExecutorSession:
        session_id = int(item["session_id"])
        session = self._pyexecutor_sessions.get(session_id)
        if session is not None and self._pyexecutor_request_matches(session, item):
            return session
        if session is not None:
            self._terminate_pyexecutor_session(session_id, session)

        request_id = self._next_pyexecutor_request_id()
        request = self._create_pyexecutor_request(request_id, item)
        session = _PyExecutorSession(
            request_id=request_id,
            request=request,
            sampler_key=item["sampler_key"],
        )
        self._pyexecutor_sessions[session_id] = session
        return session

    def _pyexecutor_request_matches(
        self, session: _PyExecutorSession, item: dict[str, Any]
    ) -> bool:
        if session.sampler_key != item["sampler_key"]:
            return False
        request = session.request
        if "GENERATION_COMPLETE" in _pyexecutor_state_name(request):
            return False
        request_tokens = _pyexecutor_tokens(request)
        prompt = [int(t) for t in item["prompt"]]
        return request_tokens == prompt

    def _create_pyexecutor_request(self, request_id: int, item: dict[str, Any]) -> Any:
        from tensorrt_llm.bindings import executor as tllm
        from tensorrt_llm._torch.pyexecutor.llm_request import (
            executor_request_to_llm_request,
        )

        prompt = [int(t) for t in item["prompt"]]
        available_tokens = int(self.model_info.max_model_len) - len(prompt)
        if available_tokens <= 0:
            raise RuntimeError(
                "TensorRT-LLM pyexecutor request has no room for generation: "
                f"prompt length {len(prompt)} >= max_model_len "
                f"{self.model_info.max_model_len}"
            )

        params = item["params"]
        max_tokens = max(
            1,
            min(
                int(getattr(self.driver_config, "pyexecutor_max_tokens", 4096)),
                available_tokens,
            ),
        )
        pad_id = getattr(params, "pad_id", None)
        if pad_id is None:
            pad_id = (
                self.model_info.pad_token_id
                if self.model_info.pad_token_id is not None
                else self.model_info.eos_token_id
            )
        if pad_id is None:
            pad_id = 0
        end_id = -1
        if not bool(getattr(params, "ignore_eos", True)):
            end_id = getattr(params, "end_id", None)
            if end_id is None:
                end_id = self.model_info.eos_token_id
            if end_id is None:
                end_id = -1

        request_kwargs: dict[str, Any] = {}
        if getattr(self.driver_config, "enable_cache_salt", True):
            try:
                from tensorrt_llm.inputs import get_cache_salt_id

                request_kwargs["cache_salt_id"] = get_cache_salt_id(
                    str(item["session_id"])
                )
            except Exception:
                pass

        executor_request = tllm.Request(
            client_id=request_id,
            input_token_ids=prompt,
            max_tokens=max_tokens,
            streaming=False,
            sampling_config=params._get_sampling_config(),
            end_id=int(end_id),
            pad_id=int(pad_id),
            output_config=params._get_output_config(is_pytorch_backend=True),
            return_all_generated_tokens=False,
            stop_words=[],
            **request_kwargs,
        )
        return executor_request_to_llm_request(
            request_id, executor_request, [], False
        )

    def _drive_pyexecutor_until_tokens(
        self, pending_targets: dict[int, int]
    ) -> dict[int, list[int]]:
        if self.pyexecutor is None:
            raise RuntimeError("TensorRT-LLM pyexecutor mode was not initialized")

        pending_req_ids = set(pending_targets)
        sessions_by_request_id = {
            int(session.request_id): session
            for session in self._pyexecutor_sessions.values()
        }
        missing_sessions = pending_req_ids - set(sessions_by_request_id)
        if missing_sessions:
            raise KeyError(
                "unknown TensorRT-LLM pyexecutor request ids "
                f"{sorted(missing_sessions)}"
            )
        generated: dict[int, list[int]] = {
            int(req_id): [] for req_id in pending_req_ids
        }
        initial_lengths = {
            int(req_id): len(_pyexecutor_tokens(sessions_by_request_id[req_id].request))
            for req_id in pending_req_ids
        }
        max_iters = max(
            8,
            max(
                (int(target) for target in pending_targets.values()),
                default=0,
            )
            + 8,
        )

        try:
            for _ in range(max_iters):
                active = []
                for req_id in pending_req_ids:
                    if len(generated[req_id]) >= int(pending_targets[req_id]):
                        continue
                    session = sessions_by_request_id.get(req_id)
                    if session is not None:
                        active.append(session.request)
                if not active:
                    break
                before = {
                    int(req.py_request_id): len(_pyexecutor_tokens(req))
                    for req in active
                }

                self.pyexecutor.active_requests = list(active)
                iter_t0 = time.perf_counter() if _pyexecutor_debug_enabled() else None
                scheduled_batch, _, _ = self.pyexecutor._schedule()
                if scheduled_batch.batch_size == 0:
                    continue

                self.pyexecutor.resource_manager.prepare_resources(scheduled_batch)
                batch_outputs = self.pyexecutor._forward_step(scheduled_batch)
                if batch_outputs is None:
                    raise RuntimeError("TensorRT-LLM pyexecutor forward failed")
                sample_state = self.pyexecutor._sample_async(
                    scheduled_batch, batch_outputs
                )
                if sample_state is None:
                    raise RuntimeError("TensorRT-LLM pyexecutor sampling failed")
                sampler_event = getattr(sample_state, "sampler_event", None)
                if sampler_event is not None:
                    sampler_event.synchronize()
                self.pyexecutor._update_request_states(scheduled_batch)
                self.pyexecutor._update_requests(
                    sample_state, self.pyexecutor.resource_manager
                )
                finished_requests = self.pyexecutor._handle_responses()
                attn_metadata = getattr(self.pyexecutor.model_engine, "attn_metadata", None)
                kv_cache_dtype_byte_size = getattr(
                    self.pyexecutor.model_engine, "kv_cache_dtype_byte_size", None
                )
                self.pyexecutor.resource_manager.update_resources(
                    scheduled_batch, attn_metadata, kv_cache_dtype_byte_size
                )
                if iter_t0 is not None:
                    elapsed_ms = (time.perf_counter() - iter_t0) * 1000
                    states = [
                        (
                            int(req.py_request_id),
                            _pyexecutor_state_name(req),
                            len(_pyexecutor_tokens(req)),
                        )
                        for req in scheduled_batch.all_requests()
                    ]
                    print(
                        "[pie-trtllm-pyexec] iter "
                        f"ctx={len(scheduled_batch.context_requests)} "
                        f"gen={len(scheduled_batch.generation_requests)} "
                        f"elapsed_ms={elapsed_ms:.3f} states={states}",
                        flush=True,
                    )

                for req in scheduled_batch.all_requests():
                    req_id = int(req.py_request_id)
                    if req_id not in pending_req_ids:
                        continue
                    after_tokens = _pyexecutor_tokens(req)
                    before_len = max(
                        before.get(req_id, len(after_tokens)),
                        initial_lengths.get(req_id, 0),
                    )
                    target = int(pending_targets[req_id])
                    if len(after_tokens) > before_len and len(generated[req_id]) < target:
                        room = target - len(generated[req_id])
                        generated[req_id].extend(
                            int(t) for t in after_tokens[before_len : before_len + room]
                        )

                self._drop_finished_pyexecutor_sessions(finished_requests)
                for req in finished_requests:
                    sessions_by_request_id.pop(int(req.py_request_id), None)
                if all(
                    len(generated[req_id]) >= int(pending_targets[req_id])
                    for req_id in pending_req_ids
                ):
                    break
            missing = [
                req_id
                for req_id in sorted(pending_req_ids)
                if len(generated[req_id]) < int(pending_targets[req_id])
            ]
            if missing:
                raise RuntimeError(
                    "TensorRT-LLM pyexecutor did not produce tokens for "
                    f"request ids {missing}"
                )
            return generated
        finally:
            self.pyexecutor.active_requests = []

    def _pyexecutor_consume_item_or_generate(self, item: dict[str, Any]) -> int:
        buffered = self._consume_lookahead(
            int(item["session_id"]),
            int(item["prefix_len"]),
            item["sampler_key"],
        )
        if buffered is not None:
            return int(buffered)

        generated = self._pyexecutor_generate_many([(0, item)])
        _, generated_item, token_ids = generated[0]
        self._append_generated_sequence(generated_item, token_ids)
        return int(token_ids[0])

    def _next_pyexecutor_request_id(self) -> int:
        request_id = self._next_pyexecutor_id
        self._next_pyexecutor_id += 1
        return request_id

    def _terminate_pyexecutor_session(
        self, session_id: int, session: _PyExecutorSession
    ) -> None:
        if self.pyexecutor is not None:
            try:
                self.pyexecutor._terminate_request(session.request)
            except Exception:
                try:
                    self.pyexecutor.resource_manager.free_resources(session.request)
                except Exception:
                    pass
        self._pyexecutor_sessions.pop(session_id, None)
        self._clear_pyexecutor_responses({int(session.request_id)})

    def _drop_finished_pyexecutor_sessions(self, requests: list[Any]) -> None:
        if not requests:
            return
        finished_ids = {int(req.py_request_id) for req in requests}
        for session_id, session in list(self._pyexecutor_sessions.items()):
            if int(session.request_id) in finished_ids:
                self._terminate_pyexecutor_session(session_id, session)

    def _clear_pyexecutor_responses(self, request_ids: set[int]) -> None:
        if self.pyexecutor is None or not request_ids:
            return
        with self.pyexecutor.response_cv:
            for request_id in request_ids:
                self.pyexecutor.responses.pop(int(request_id), None)

    def _consume_item_or_generate_one(self, item: dict[str, Any]) -> int:
        buffered = self._consume_lookahead(
            int(item["session_id"]),
            int(item["prefix_len"]),
            item["sampler_key"],
        )
        if buffered is not None:
            return int(buffered)
        generated = self._generate_many([(0, item)])
        _, _, token_ids = generated[0]
        self._append_generated_sequence(item, token_ids)
        return int(token_ids[0])

    def _generate_many(
        self, generate_work: list[tuple[int, dict[str, Any]]]
    ) -> list[tuple[int, dict[str, Any], list[int]]]:
        prompts = [item["prompt"] for _, item in generate_work]
        params = [item["params"] for _, item in generate_work]
        generate_kwargs = {"sampling_params": params, "use_tqdm": False}
        if self.driver_config.enable_cache_salt:
            generate_kwargs["cache_salt"] = [
                str(item["session_id"]) for _, item in generate_work
            ]

        outputs = self.llm.generate(prompts, **generate_kwargs)
        if not isinstance(outputs, list):
            outputs = [outputs]
        if len(outputs) != len(generate_work):
            raise RuntimeError(
                "TensorRT-LLM returned a different number of outputs than "
                f"requested: {len(outputs)} != {len(generate_work)}"
            )

        generated: list[tuple[int, dict[str, Any], list[int]]] = []
        for (out_idx, item), output in zip(generate_work, outputs):
            token_ids = self._extract_token_ids(output)
            if not token_ids:
                raise RuntimeError(
                    f"TensorRT-LLM output did not contain token ids: {output!r}"
                )
            generated.append((out_idx, item, token_ids))
        return generated

    def _sampling_params(
        self, batch, sampler_idx: int, stype: int, *, max_tokens: int = 1
    ):
        temperature = float(batch.temperatures[sampler_idx])
        top_k = int(batch.top_k_values[sampler_idx])
        top_p = float(batch.top_p_values[sampler_idx])
        min_p = float(batch.min_p_values[sampler_idx])
        seed = int(batch.sampler_seeds_arr[sampler_idx])

        kwargs: dict[str, Any] = {
            "max_tokens": int(max_tokens),
            "detokenize": False,
            "add_special_tokens": False,
            "ignore_eos": True,
            "temperature": temperature,
        }
        if self.model_info.eos_token_id is not None:
            kwargs["end_id"] = int(self.model_info.eos_token_id)
            kwargs["pad_id"] = int(
                self.model_info.pad_token_id
                if self.model_info.pad_token_id is not None
                else self.model_info.eos_token_id
            )
        if seed != 0:
            kwargs["seed"] = seed
        if stype == 2:
            kwargs["top_k"] = top_k
        elif stype == 3:
            kwargs["top_p"] = top_p
        elif stype == 4:
            kwargs["min_p"] = min_p
        elif stype == 5:
            kwargs["top_k"] = top_k
            kwargs["top_p"] = top_p

        return self.sampling_params_cls(**kwargs)

    def _sampler_key(self, batch, sampler_idx: int, stype: int) -> tuple[Any, ...]:
        temperature = float(batch.temperatures[sampler_idx])
        seed = int(batch.sampler_seeds_arr[sampler_idx])
        return (
            int(stype),
            temperature,
            int(batch.top_k_values[sampler_idx]),
            float(batch.top_p_values[sampler_idx]),
            float(batch.min_p_values[sampler_idx]),
            0 if temperature == 0.0 else seed,
            self.model_info.eos_token_id,
            self.model_info.pad_token_id,
        )

    def _lookahead_tokens(self, batch, sampler_idx: int, stype: int) -> int:
        limit = int(getattr(self.driver_config, "lookahead_tokens", 1) or 1)
        # Larger chunks can outrun Pie's chain replay bookkeeping with the
        # high-level TensorRT-LLM API and are slower for this PyExecutor path.
        # Keep the public knob, but clamp to the validated stable window for
        # this request-oriented driver.
        limit = min(limit, 16)
        if limit <= 1:
            return 1
        if self.driver_config.max_history_tokens is not None:
            return 1
        if not self._lookahead_eligible(batch, sampler_idx, stype):
            return 1
        return limit

    def _lookahead_eligible(self, batch, sampler_idx: int, stype: int) -> bool:
        # Lookahead must not change sampling semantics. Restrict the default
        # optimization to deterministic decoding; stochastic paths still use
        # one-token LLM.generate calls.
        return int(stype) in TOKEN_SAMPLING_TYPES and float(
            batch.temperatures[sampler_idx]
        ) == 0.0

    def _consume_lookahead(
        self, session_id: int, prefix_len: int, sampler_key: tuple[Any, ...]
    ) -> int | None:
        buf = self._lookahead.get(session_id)
        if buf is None:
            return None
        if buf.sampler_key != sampler_key:
            self._drop_lookahead(session_id)
            return None
        rel = int(prefix_len) - int(buf.base_pos)
        if rel < 0 or rel > len(buf.tokens):
            self._drop_lookahead(session_id)
            return None
        if rel >= len(buf.tokens):
            self._drop_lookahead(session_id)
            return None
        buf.next_idx = rel + 1
        token = int(buf.tokens[rel])
        if buf.next_idx >= len(buf.tokens):
            self._drop_lookahead(session_id)
        else:
            self._lookahead.move_to_end(session_id)
        return token

    def _drop_lookahead(self, session_id: int) -> None:
        self._lookahead.pop(session_id, None)

    def _history_for(self, session_id: int) -> list[int]:
        history = self._histories.get(session_id)
        if history is None:
            max_sessions = int(self.driver_config.max_session_histories)
            if max_sessions <= 0:
                raise ValueError("max_session_histories must be positive")
            while len(self._histories) >= max_sessions:
                evicted_session_id, _ = self._histories.popitem(last=False)
                self._lookahead.pop(evicted_session_id, None)
                self._emitted_token_counts.pop(evicted_session_id, None)
            history = []
            self._histories[session_id] = history
        else:
            self._histories.move_to_end(session_id)
        return history

    def _merge_tokens(
        self,
        history: list[int],
        tokens: list[int],
        positions: list[int],
        session_id: int,
    ) -> bool:
        truncated = False
        for token, pos in zip(tokens, positions):
            if pos < 0:
                raise ValueError(f"negative position {pos} for Pie context {session_id}")
            if pos < len(history):
                if history[pos] != token:
                    del history[pos:]
                    history.append(token)
                    truncated = True
            elif pos == len(history):
                history.append(token)
            else:
                raise RuntimeError(
                    f"Pie context {session_id} has token position {pos}, but the "
                    f"TensorRT-LLM replay history only contains {len(history)} "
                    "tokens. Fork/restore workloads and max_history_tokens values "
                    "that trim active prefixes are not supported by the "
                    "TensorRT-LLM high-level driver yet."
                )
        return truncated

    def _trim_history(self, history: list[int]) -> None:
        limit = self.driver_config.max_history_tokens
        if limit is not None and limit > 0 and len(history) > limit:
            del history[: len(history) - limit]

    def _append_generated_sequence(
        self,
        item: dict[str, Any],
        tokens: list[int],
        *,
        keep_lookahead: bool = True,
    ) -> None:
        session_id = int(item["session_id"])
        prefix_len = int(item["prefix_len"])
        history = self._history_for(session_id)
        if len(history) > prefix_len:
            del history[prefix_len:]
        prompt = [int(t) for t in item["prompt"]]
        if len(history) != prefix_len or history[:prefix_len] != prompt:
            history[:] = prompt
        history.extend(int(t) for t in tokens)
        self._trim_history(history)

        if (
            keep_lookahead
            and len(tokens) > 1
            and self.driver_config.max_history_tokens is None
        ):
            self._lookahead[session_id] = _LookaheadBuffer(
                base_pos=prefix_len,
                tokens=[int(t) for t in tokens],
                next_idx=1,
                sampler_key=item["sampler_key"],
            )
            self._lookahead.move_to_end(session_id)
        else:
            self._drop_lookahead(session_id)

    def _direct_accept_max_tokens(self, item: dict[str, Any]) -> int:
        limit = getattr(self.driver_config, "pyexecutor_direct_token_limit", None)
        max_tokens = int(item.get("max_tokens", 1))
        if limit is None:
            return max_tokens
        emitted = int(self._emitted_token_counts.get(int(item["session_id"]), 0))
        return max(0, min(max_tokens, int(limit) - emitted))

    def _record_emitted_tokens(self, session_id: int, count: int) -> None:
        if count <= 0:
            return
        session_id = int(session_id)
        self._emitted_token_counts[session_id] = (
            int(self._emitted_token_counts.get(session_id, 0)) + int(count)
        )

    def _session_id(self, batch, req_idx: int) -> int:
        if batch.context_ids:
            return int(batch.context_ids[req_idx])
        start = int(batch.kv_page_indptr[req_idx])
        end = int(batch.kv_page_indptr[req_idx + 1])
        if end > start:
            return int(batch.kv_page_indices[start])
        return req_idx

    @staticmethod
    def _extract_one_token(output: Any) -> int:
        token_ids = TensorRTLLMEngine._extract_token_ids(output)
        if token_ids:
            return int(token_ids[-1])
        raise RuntimeError(f"TensorRT-LLM output did not contain token ids: {output!r}")

    @staticmethod
    def _extract_token_ids(output: Any) -> list[int]:
        completions = getattr(output, "outputs", None)
        if completions:
            first = completions[0]
            if isinstance(first, dict):
                token_ids = first.get("token_ids") or first.get("token_ids_diff")
                if token_ids:
                    return [int(t) for t in token_ids]
            token_ids = getattr(first, "token_ids", None)
            if token_ids:
                return [int(t) for t in token_ids]
            token_ids_diff = getattr(first, "token_ids_diff", None)
            if token_ids_diff:
                return [int(t) for t in token_ids_diff]
        token_ids = getattr(output, "token_ids", None)
        if token_ids:
            return [int(t) for t in token_ids]
        if isinstance(output, dict):
            if output.get("outputs"):
                return TensorRTLLMEngine._extract_token_ids(
                    type("OutputProxy", (), {"outputs": output["outputs"]})()
                )
            if output.get("token_ids"):
                return [int(t) for t in output["token_ids"]]
        return []

    @staticmethod
    def _validate_copy_args(srcs: list[int], dsts: list[int]) -> None:
        if len(srcs) != len(dsts):
            raise ValueError(f"copy argument length mismatch: {len(srcs)} != {len(dsts)}")
        if any(int(x) < 0 for x in srcs + dsts):
            raise ValueError("copy indices must be non-negative")


def _normalize_dtype(dtype: str) -> str:
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "fp32": "float32",
        "float32": "float32",
        "auto": "auto",
    }
    if dtype not in aliases:
        raise ValueError(
            f"Unsupported activation dtype for TensorRT-LLM driver: {dtype!r}. "
            f"Expected one of {sorted(aliases)}."
        )
    return aliases[dtype]


def _optional_token_id(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_execution_mode(mode: Any) -> str:
    normalized = str(mode or "generate").strip().lower()
    if normalized not in {"generate", "pyexecutor"}:
        raise ValueError(
            "Unsupported TensorRT-LLM execution_mode "
            f"{mode!r}; expected 'generate' or 'pyexecutor'."
        )
    return normalized


def _validate_pyexecutor_driver_config(driver_config) -> None:
    if getattr(driver_config, "max_history_tokens", None) is not None:
        raise NotImplementedError(
            "TensorRT-LLM pyexecutor mode requires full replay histories; "
            "max_history_tokens is not supported."
        )
    if int(getattr(driver_config, "pyexecutor_max_tokens", 4096)) <= 0:
        raise ValueError("pyexecutor_max_tokens must be positive")


def _configure_pyexecutor_llm_kwargs(
    llm_kwargs: dict[str, Any], driver_config
) -> None:
    backend = llm_kwargs.get("backend", getattr(driver_config, "backend", None))
    if backend not in (None, "pytorch"):
        raise NotImplementedError(
            "TensorRT-LLM pyexecutor mode currently requires backend='pytorch'; "
            f"got {backend!r}."
        )
    llm_kwargs["backend"] = "pytorch"

    env_overrides = llm_kwargs.get("env_overrides")
    if env_overrides is None:
        env_overrides = {}
    elif not isinstance(env_overrides, dict):
        raise TypeError("llm_kwargs.env_overrides must be a mapping in pyexecutor mode")
    else:
        env_overrides = dict(env_overrides)
    env_overrides.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    llm_kwargs["env_overrides"] = env_overrides

    # LLM applies env_overrides during construction; set the process value too
    # so direct private imports see the same mode if TensorRT-LLM checks early.
    os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")


def _extract_pyexecutor(llm: Any) -> Any:
    executor = getattr(llm, "_executor", None)
    pyexecutor = getattr(executor, "engine", None)
    required_attrs = (
        "model_engine",
        "resource_manager",
        "sampler",
        "scheduler",
        "active_requests",
        "worker_thread",
    )
    missing = [attr for attr in required_attrs if not hasattr(pyexecutor, attr)]
    if pyexecutor is None or missing:
        raise RuntimeError(
            "TensorRT-LLM pyexecutor mode could not find the expected private "
            f"PyExecutor object; missing {missing}."
        )
    return pyexecutor


def _stop_pyexecutor_worker(pyexecutor: Any, *, timeout_s: float) -> None:
    thread = getattr(pyexecutor, "worker_thread", None)
    if thread is None or not thread.is_alive():
        return
    pyexecutor.executor_request_queue.enqueue_shutdown_request()
    thread.join(timeout=max(0.0, float(timeout_s)))
    if thread.is_alive():
        raise TimeoutError(
            "Timed out waiting for TensorRT-LLM PyExecutor worker thread to stop"
        )
    pyexecutor.worker_started = False
    pyexecutor.active_requests = []


def _pyexecutor_tokens(request: Any) -> list[int]:
    return [int(t) for t in request.get_tokens(0)]


def _pyexecutor_state_name(request: Any) -> str:
    state = getattr(request, "state", None)
    return str(getattr(state, "name", state))


def _pyexecutor_debug_enabled() -> bool:
    return os.environ.get("PIE_TRTLLM_DEBUG_PYEXEC", "").lower() in {
        "1",
        "true",
        "yes",
    }
