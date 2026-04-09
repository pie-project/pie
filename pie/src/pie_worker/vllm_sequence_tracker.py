"""Sequence tracking state for the vLLM-backed Pie runtime.

Extracted from PieVllmRuntime to isolate request lifecycle management
(new/cached/finished classification, token history, block delta computation)
from the runtime's GPU worker and RPC concerns.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any


@dataclass
class _ActiveRequest:
    """Tracks a request persisted in vLLM's InputBatch across fire_batch calls.

    Attributes:
        req_id: Stable vLLM request ID for this sequence.
        block_ids: Last known block IDs (for computing delta).
        num_output_tokens: Decode tokens generated so far (0 during prefill).
    """
    req_id: str
    block_ids: list[int]
    num_output_tokens: int
    last_block: int = -1


def _expand_block_ids_for_groups(
    pie_block_ids: list[int],
    num_groups: int,
) -> tuple[list[int], ...]:
    """Expand Pie's single block list into per-group block IDs.

    For hybrid models (e.g. Qwen3.5 with DeltaNet + FullAttention), vLLM
    stores different KV cache groups in the same physical memory tensor.
    Each group must use DIFFERENT block IDs to avoid memory aliasing.

    vLLM's own scheduler allocates blocks from a shared pool, giving each
    group unique block IDs.  Since Pie's Rust scheduler provides only one
    block list, the bridge must expand it:

        Pie block b (with +1 offset) -> group g block: (b-1)*G + g + 1

    For single-group models (G=1), this is identity: block b -> block b.

    Args:
        pie_block_ids: Block IDs from Pie (already +1 offset).
        num_groups: Number of KV cache groups.

    Returns:
        Tuple of block ID lists, one per group.
    """
    if num_groups <= 1:
        return (list(pie_block_ids),)
    return tuple(
        [(b - 1) * num_groups + g + 1 for b in pie_block_ids]
        for g in range(num_groups)
    )


class SequenceTracker:
    """Pure state tracker for vLLM request lifecycle management.

    Manages the classification of sequences into new, cached, and finished
    categories across fire_batch calls.  Maintains token history, active
    request metadata, and block delta computation.

    Does NOT hold references to GPU resources (worker, config, queues).
    Those are passed as method parameters when needed.

    Args:
        num_kv_cache_groups: Number of KV cache groups (>1 for hybrid models).
    """

    def __init__(
        self,
        num_kv_cache_groups: int = 1,
        max_token_history_len: int = 131072,
    ) -> None:
        self._max_token_history_len = max_token_history_len

        # Monotonic batch counter for generating unique request IDs.
        self._batch_counter: int = 0

        # Token history per sequence for reconstructing full prompt_token_ids.
        # vLLM's NewRequestData expects ALL tokens (not just new ones), with
        # num_computed_tokens indicating how many are already in KV cache.
        # Pie only sends new tokens per fire_batch, so we accumulate here.
        # Key: request_id string (from Rust-provided explicit identity).
        self._token_history: dict[str, list[int]] = {}

        # Active request tracking for CachedRequestData optimisation.
        # Key: request_id string (from Rust-provided explicit identity)
        # -> _ActiveRequest with stable req_id and state.
        # Sequences present across fire_batch calls use CachedRequestData
        # (lightweight update) instead of NewRequestData (full re-add).
        self._active_requests: dict[str, _ActiveRequest] = {}

        # Ordered req_ids from the last build_scheduler_output call.
        # Used by _package_response to look up tokens in model output.
        self._last_batch_req_ids: list[str] = []

        # All request IDs ever issued to vLLM.  Used as a safety net to
        # ensure stale requests in vLLM's input_batch are always explicitly
        # finished, preventing KeyError in execute_model.
        self._all_issued_req_ids: set[str] = set()

        self._num_kv_cache_groups: int = num_kv_cache_groups

        # Block IDs freed by Rust ResourceManager (explicit finish signal).
        # Accumulated between fire_batch calls, consumed in build_scheduler_output.
        self._freed_block_ids: set[int] = set()

    # -- Public properties -------------------------------------------------

    @property
    def batch_counter(self) -> int:
        return self._batch_counter

    @batch_counter.setter
    def batch_counter(self, value: int) -> None:
        self._batch_counter = value

    @property
    def last_batch_req_ids(self) -> list[str]:
        return self._last_batch_req_ids

    @last_batch_req_ids.setter
    def last_batch_req_ids(self, value: list[str]) -> None:
        self._last_batch_req_ids = value

    @property
    def token_history(self) -> dict[str, list[int]]:
        return self._token_history

    @property
    def active_requests(self) -> dict[str, _ActiveRequest]:
        return self._active_requests

    @property
    def all_issued_req_ids(self) -> set[str]:
        return self._all_issued_req_ids

    # -- Explicit finish signal from Rust ------------------------------------

    def finish_by_block_ids(self, freed_ids: set) -> None:
        """Queue freed block IDs for deterministic sequence cleanup.

        Called when Rust ResourceManager deallocates KV pages (WASM Context
        drops). The actual cleanup happens in build_scheduler_output() so
        that finished_req_ids are included in the SchedulerOutput.
        """
        self._freed_block_ids |= freed_ids

    # -- Core methods ------------------------------------------------------

    def build_scheduler_output(
        self,
        batch_id: int,
        token_ids: "np.ndarray",
        qo_indptr: "np.ndarray",
        tokens_per_req: list[int],
        blocks_per_req: list[list[int]],
        seq_lens: "np.ndarray",
        sampling_params_list: list[dict[str, Any]],
        adapter_indices: list[int | None] | None = None,
        adapter_registry: dict[int, tuple[str, str]] | None = None,
        kv_page_size: int = 0,
        *,
        request_ids: list[str],
        is_new: list[bool],
    ) -> "SchedulerOutput":
        """Build a vLLM SchedulerOutput from translated Pie batch data.

        Uses a two-tier strategy:
          - NEW sequences (is_new=True) -> NewRequestData (adds to
            vLLM's InputBatch).
          - CONTINUING sequences (is_new=False) -> CachedRequestData
            (lightweight state update).
          - FINISHED sequences (freed via finish_by_block_ids or stale
            req_id safety net) -> finished_req_ids (removes from
            InputBatch).

        Sequences are identified by their Rust-provided request_id
        string and is_new flag.  No block-ID-based identity inference.

        Token state consistency is handled by vLLM's pie_manages_tokens flag
        which forces token_ids_cpu updates from CachedRequestData.all_token_ids
        and clears prev_sampled_token_ids. This ensures correctness for flush
        (discarded output), distribution mode (inferlet-chosen tokens), and
        fork (KV prefix sharing) without bridge-side workarounds.

        Args:
            batch_id: Monotonic batch counter for unique request IDs.
            token_ids: Flat array of token IDs across all requests.
            qo_indptr: CSR pointers for query/output tokens per request.
            tokens_per_req: Number of NEW tokens per request.
            blocks_per_req: Per-request block ID lists (already offset +1).
            seq_lens: Per-request total sequence lengths.
            sampling_params_list: Per-request dicts of sampling params.
            adapter_indices: Per-request adapter pointer (None = no adapter).
            adapter_registry: Adapter pointer -> (name, path) mapping.
            request_ids: Per-request identity string from Rust scheduler.
            is_new: Per-request flag indicating NEW (True) vs CONTINUING
                (False).

        Returns:
            SchedulerOutput ready for Worker.execute_model().
        """
        from vllm.v1.core.sched.output import (
            SchedulerOutput,
            NewRequestData,
            CachedRequestData,
        )
        from vllm.sampling_params import SamplingParams

        if adapter_registry is None:
            adapter_registry = {}

        num_requests = len(tokens_per_req)

        new_reqs: list[NewRequestData] = []
        num_scheduled_tokens: dict[str, int] = {}
        total: int = 0

        # CachedRequestData fields (accumulated per continuing request)
        cached_req_ids: list[str] = []
        cached_new_block_ids: list[tuple[list[int], ...] | None] = []
        cached_num_computed: list[int] = []
        cached_num_output: list[int] = []
        cached_all_token_ids: dict[str, list[int]] = {}

        # Ordered req_ids for _package_response lookup
        batch_req_ids: list[str] = []

        # Finished requests (populated during loop for forks, and after
        # loop for sequences that disappeared from the batch).
        finished_req_ids: set[str] = set()

        # Process freed_block_ids BEFORE the per-request loop.
        # Rust sends freed block IDs in the same batch that reuses them.
        # If we process them after, the NEW request's entry gets matched
        # and incorrectly finished.
        if self._freed_block_ids:
            # Build reverse map from last_block to identity key, then
            # finish any active request whose last_block was freed AND
            # whose identity key is NOT in the current batch's request_ids.
            _current_request_ids = set(request_ids)
            _last_block_to_key: dict[int, list[str]] = {}
            for _ik, _ar in self._active_requests.items():
                lb = _ar.last_block
                if lb >= 0:
                    _last_block_to_key.setdefault(lb, []).append(_ik)
            to_remove: list[str] = []
            for freed_block in self._freed_block_ids:
                for _ik in _last_block_to_key.get(freed_block, []):
                    if _ik not in _current_request_ids:
                        active_req = self._active_requests[_ik]
                        if os.environ.get("PIE_VLLM_DEBUG"):
                            print(f"[FREED-CHECK] key={_ik} "
                                  f"req={active_req.req_id} "
                                  f"last_block={freed_block} "
                                  f"freed={self._freed_block_ids}",
                                  file=sys.stderr, flush=True)
                        finished_req_ids.add(active_req.req_id)
                        to_remove.append(_ik)
            for k in to_remove:
                self._active_requests.pop(k, None)
                self._token_history.pop(k, None)
            self._freed_block_ids.clear()

        for i in range(num_requests):
            # Extract this request's NEW token IDs from the flat array
            start = int(qo_indptr[i])
            end = int(qo_indptr[i + 1])
            new_token_ids = token_ids[start:end].tolist()
            num_new_tokens = tokens_per_req[i]

            last_block = blocks_per_req[i][-1] if blocks_per_req[i] else -1

            # Identity is always the Rust-provided request_id string.
            identity_key: str = request_ids[i]
            is_new_req = is_new[i]

            # If this is a continuing request whose identity_key
            # isn't tracked yet (shouldn't happen normally), treat
            # as new to avoid KeyError.
            if not is_new_req and identity_key not in self._active_requests:
                is_new_req = True

            # The scheduler's KV metadata is authoritative.
            # seq_lens[i] = total tokens after this step (from page count
            #   and kv_last_page_lens).
            # num_new_tokens = tokens in this batch (from qo_indptr).
            # Their difference = tokens already cached in KV pages.
            #
            # After fork(), the partial page is dropped and its tokens
            # become pending.  The inferlet then grow()s KV and sends ALL
            # pending tokens.  effective_cached will be smaller than our
            # _token_history because the history still has the parent's
            # full count.  Truncating history to effective_cached fixes
            # this: vLLM then sees the correct num_computed_tokens and
            # processes all new tokens through the model.
            kv_len = int(seq_lens[i])
            effective_cached = kv_len - num_new_tokens

            existing_len = len(self._token_history.get(identity_key, []))

            # Cap at what we've actually tracked. Pre-allocated KV pages
            # (from decode_n's extra_kv_tokens) inflate kv_len beyond the
            # real token count. token_history is the ground truth.
            effective_cached = min(effective_cached, existing_len)

            if existing_len > effective_cached:
                # Distinguish page-overflow from fork/eviction:
                #
                # Page overflow: the Rust scheduler wrapped kv_page_last_len
                # on a page crossing without adding a new page to
                # kv_page_indptr. seq_lens collapsed (e.g., 16 → 1) but
                # the KV data is intact in the original page(s).
                # Signal: effective_cached == 0, sequence is CONTINUING,
                # and kv_len < existing_len (not just slightly less).
                #
                # In this case, trust existing_len as the actual cached
                # count.  The SequenceTracker maintains the authoritative
                # token history; kv_len from the batch metadata is wrong.
                _is_page_overflow = (
                    effective_cached == 0
                    and existing_len > 0
                    and not is_new_req
                    and kv_len < existing_len
                )
                if _is_page_overflow:
                    # Override: use token history as ground truth.
                    effective_cached = existing_len
                else:
                    # Fork or KV eviction: fewer tokens are cached than we
                    # tracked.  Truncate history to match reality and force
                    # NewRequestData so vLLM rebuilds its internal state.
                    self._token_history[identity_key] = (
                        self._token_history[identity_key][:effective_cached]
                    )
                    if identity_key in self._active_requests:
                        finished_req_ids.add(
                            self._active_requests[identity_key].req_id
                        )
                        del self._active_requests[identity_key]
                        is_new_req = True

            # Accumulate full token history for this sequence.
            # For NEW sequences that share prefix with another request
            # in the same batch (fork), we copy the prefix from the
            # parent's history if available.
            if identity_key in self._token_history:
                self._token_history[identity_key].extend(new_token_ids)
            elif effective_cached > 0:
                # Cross-batch case: prefix tokens are cached but this
                # is a new sequence key (fork).  Find the parent by:
                #   1. Block lineage -- walk internal blocks backwards.
                #   2. Active request match -- find parent sharing the
                #      same first block (covers dropped partial pages).
                prefix_tokens: list[int] = []
                # Keys are request_id strings.  Search through
                # _active_requests to find a parent whose block_ids
                # contain the target block.
                for block in reversed(blocks_per_req[i][:-1]):
                    for _ak, _ar in self._active_requests.items():
                        if block in _ar.block_ids and _ak in self._token_history:
                            prefix_tokens = self._token_history[_ak][:effective_cached]
                            break
                    if prefix_tokens:
                        break
                if not prefix_tokens:
                    # Fallback: parent's last block was dropped (fork),
                    # so it's not in our block list.  Find parent via
                    # shared first block in active requests.
                    first_block = blocks_per_req[i][0] if blocks_per_req[i] else -1
                    for akey, active in self._active_requests.items():
                        if (active.block_ids
                                and active.block_ids[0] == first_block
                                and akey in self._token_history):
                            prefix_tokens = self._token_history[akey][:effective_cached]
                            break
                self._token_history[identity_key] = prefix_tokens + new_token_ids
            else:
                self._token_history[identity_key] = list(new_token_ids)

            # Cap token history to prevent unbounded memory growth.
            # Safe because num_computed_tokens ensures vLLM never
            # re-embeds tokens beyond the KV cache window.
            history = self._token_history[identity_key]
            if len(history) > self._max_token_history_len:
                excess = len(history) - self._max_token_history_len
                self._token_history[identity_key] = history[excess:]

            full_prompt_token_ids = list(self._token_history[identity_key])
            num_computed_tokens = len(full_prompt_token_ids) - num_new_tokens

            if not is_new_req:
                # --- CONTINUING sequence: CachedRequestData ---
                active = self._active_requests[identity_key]
                req_id = active.req_id

                # Compute block ID delta (new blocks since last call)
                prev_blocks = active.block_ids
                curr_blocks = blocks_per_req[i]
                if len(curr_blocks) > len(prev_blocks):
                    delta = curr_blocks[len(prev_blocks):]
                    new_block_delta: tuple[list[int], ...] | None = (
                        _expand_block_ids_for_groups(
                            delta, self._num_kv_cache_groups
                        )
                    )
                else:
                    new_block_delta = None

                cached_req_ids.append(req_id)
                cached_new_block_ids.append(new_block_delta)
                cached_num_computed.append(num_computed_tokens)
                active.num_output_tokens += num_new_tokens
                cached_num_output.append(active.num_output_tokens)
                cached_all_token_ids[req_id] = full_prompt_token_ids

                # Update tracking (block_ids and last_block for delta computation)
                active.block_ids = list(curr_blocks)
                active.last_block = last_block
            else:
                # --- NEW sequence: NewRequestData ---
                req_id = f"pie-{batch_id}-{i}"

                block_ids = _expand_block_ids_for_groups(
                    blocks_per_req[i], self._num_kv_cache_groups
                )

                sp = sampling_params_list[i]
                vllm_top_k = sp["top_k"] if sp["top_k"] > 0 else 0
                vllm_sampling_params = SamplingParams(
                    temperature=sp["temperature"],
                    top_k=vllm_top_k,
                    top_p=sp["top_p"],
                    min_p=sp["min_p"],
                    max_tokens=1,  # Pie generates one token at a time
                )

                # Look up LoRA adapter if specified
                lora_request = None
                if adapter_indices and i < len(adapter_indices):
                    adapter_ptr = adapter_indices[i]
                    if adapter_ptr is not None and adapter_ptr in adapter_registry:
                        name, path = adapter_registry[adapter_ptr]
                        from vllm.lora.request import LoRARequest
                        lora_request = LoRARequest(
                            lora_name=name,
                            lora_int_id=adapter_ptr,
                            lora_path=path,
                        )

                use_block_ids = block_ids
                use_num_computed = num_computed_tokens

                new_reqs.append(NewRequestData(
                    req_id=req_id,
                    prompt_token_ids=full_prompt_token_ids,
                    mm_features=[],
                    sampling_params=vllm_sampling_params,
                    pooling_params=None,
                    block_ids=use_block_ids,
                    num_computed_tokens=use_num_computed,
                    lora_request=lora_request,
                ))

                # Start tracking for future CachedRequestData.
                # Token state consistency is handled by vLLM's
                # pie_manages_tokens flag, which forces token_ids_cpu
                # updates from all_token_ids. This works for all request
                # types including flush and distribution mode.
                self._active_requests[identity_key] = _ActiveRequest(
                    req_id=req_id,
                    block_ids=list(blocks_per_req[i]),
                    num_output_tokens=0,
                    last_block=last_block,
                )

            batch_req_ids.append(req_id)
            num_scheduled_tokens[req_id] = num_new_tokens
            total += num_new_tokens

        # freed_block_ids already processed BEFORE the per-request loop.

        # Clean up token history for keys not in _active_requests.
        for fk in list(self._token_history.keys()):
            if fk not in self._active_requests:
                del self._token_history[fk]

        # Safety net: finish stale req_ids in vLLM's input_batch.
        current_req_ids = set(num_scheduled_tokens.keys())
        still_active_req_ids = {ar.req_id for ar in self._active_requests.values()}
        stale_req_ids = self._all_issued_req_ids - current_req_ids - still_active_req_ids
        finished_req_ids |= stale_req_ids
        self._all_issued_req_ids = current_req_ids | still_active_req_ids

        # Build CachedRequestData
        if cached_req_ids:
            cached_reqs = CachedRequestData(
                req_ids=cached_req_ids,
                resumed_req_ids=set(),
                new_token_ids=[],
                all_token_ids=cached_all_token_ids,
                new_block_ids=cached_new_block_ids,
                num_computed_tokens=cached_num_computed,
                num_output_tokens=cached_num_output,
            )
        else:
            cached_reqs = CachedRequestData.make_empty()

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs,
            scheduled_cached_reqs=cached_reqs,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0] * self._num_kv_cache_groups,
            finished_req_ids=finished_req_ids,
            free_encoder_mm_hashes=[],
        )

        # Store for _package_response
        self._last_batch_req_ids = batch_req_ids

        return scheduler_output

    def recover_from_failed_batch(
        self,
        scheduler_output: Any,
        *,
        tp_queue: Any = None,
        vllm_worker: Any = None,
        vllm_config: Any = None,
    ) -> None:
        """Clean up vLLM state after a failed execute_model call.

        When execute_model raises (e.g. shape mismatch during multi-fork),
        vLLM's input_batch may contain partially-added requests.  We send
        a no-op SchedulerOutput that finishes ALL requests from the failed
        batch, preventing stale entries from causing KeyError on the next
        fire_batch call.

        Args:
            scheduler_output: The SchedulerOutput that failed.
            tp_queue: Inter-process queue for TP leader->follower signaling.
            vllm_worker: The vLLM GPU worker instance.
            vllm_config: The vLLM configuration object.
        """
        try:
            from vllm.v1.core.sched.output import (
                SchedulerOutput,
                CachedRequestData,
            )
            from vllm.config import set_current_vllm_config

            # Collect all req_ids that were in the failed batch
            all_req_ids = set(scheduler_output.num_scheduled_tokens.keys())
            # Also include any new request IDs
            for nr in scheduler_output.scheduled_new_reqs:
                all_req_ids.add(nr.req_id)

            if not all_req_ids:
                return

            # Send a no-op batch that finishes everything
            cleanup_output = SchedulerOutput(
                scheduled_new_reqs=[],
                scheduled_cached_reqs=CachedRequestData.make_empty(),
                num_scheduled_tokens={},
                total_num_scheduled_tokens=0,
                scheduled_spec_decode_tokens={},
                scheduled_encoder_inputs={},
                num_common_prefix_blocks=[0] * self._num_kv_cache_groups,
                finished_req_ids=all_req_ids,
                free_encoder_mm_hashes=[],
            )
            # For TP>1, send cleanup to non-leaders so they participate
            if tp_queue is not None:
                tp_queue.put(cleanup_output)
            with set_current_vllm_config(vllm_config):
                vllm_worker.execute_model(cleanup_output)
        except Exception:
            # Recovery itself failed -- log and continue.
            # The next fire_batch may still fail, but we've done our best.
            import traceback
            print("[WARN] recover_from_failed_batch failed:",
                  file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
