"""
Batch management for Pie driver inference.

This module provides the Batch dataclass that holds inference batch state
and handles tensor creation and response packaging.
"""

from __future__ import annotations

from typing import Any

import time
import numpy as np
from numba import njit, prange

from . import message


class Batch:
    """
    Holds the accumulated state for a specific inference step and handles packaging.

    This consolidates the state storage (formerly BatchState) and packaging logic
    (formerly ResponsePackager) into a single unified class.
    """

    def __init__(
        self,
        args: dict[str, Any],
        kv_page_size: int,
        max_dist_size: int,
        adapters: dict[int, Any],
        vocab_size: int | None = None,
    ) -> None:
        """
        Initialize a Batch from BatchedForwardPassRequest dict.

        Args:
            args: Dictionary with batched request fields
            kv_page_size: KV cache page size reported by driver capabilities
            max_dist_size: Max distribution size from model config
            adapters: Dictionary of active adapters
            vocab_size: Model vocabulary size (optional, needed for sampling masks)
        """
        # Initialize timing dict as instance attribute
        self.timing: dict[str, float] = {
            "decode_u32": 0.0,
            "mask_loop": 0.0,
            "brle_decode": 0.0,
            "sampler_loop": 0.0,
        }

        # Initialize mutable containers
        self.adapter_indices: list[int] = []
        self.adapter_seeds: list[int] = []
        self.adapter_subpass_needed: bool = False
        self.indices_for_logits: list[int] = []
        self._kv_page_size = int(kv_page_size)
        self._spec_plan: list[dict | None] | None = None

        # Helper to decode bytes as u32 array or pass through lists.
        # Accepts memoryview (shmem zero-copy view from `shmem_schema`) or list.
        def _decode_u32(data):
            if isinstance(data, (bytes, bytearray, memoryview)):
                return np.frombuffer(data, dtype=np.uint32)
            return np.array(data, dtype=np.uint32)

        # Helper to decode bytes as i64 array
        def _decode_i64(data):
            if isinstance(data, (bytes, bytearray, memoryview)):
                # Wire format stores token_ids as u32; widen to i64 for torch.
                return np.frombuffer(data, dtype=np.uint32).astype(np.int64)
            return np.array(data, dtype=np.int64)

        # Direct assignments - decode bytes as u32 arrays
        t0 = time.perf_counter()
        # token_ids are long (i64)
        self.token_ids = _decode_i64(args["token_ids"])

        self.position_ids = _decode_u32(args["position_ids"]).astype(np.int32)
        self.kv_page_indices = _decode_u32(args["kv_page_indices"]).astype(np.int32)
        self.kv_page_indptr = _decode_u32(args["kv_page_indptr"]).astype(np.int32)
        self.kv_last_page_lens = _decode_u32(args["kv_last_page_lens"]).astype(np.int32)
        self.qo_indptr = _decode_u32(args["qo_indptr"]).astype(np.int32)
        self.single_token_mode = args["single_token_mode"]
        self.total_tokens = int(len(self.token_ids))
        self.timing["decode_u32"] = time.perf_counter() - t0

        # ===== VECTORIZED BATCH METADATA GENERATION =====
        t0 = time.perf_counter()

        # Process per-request data
        # `.view(np.int32)` reinterprets the same bytes as int32 with zero
        # copy. Values are non-negative BRLE run lengths well within int32
        # range so semantics are unchanged. Numba's existing int32
        # specialization for `decode_brle_batch` is faster than its u32 one,
        # which is why we view rather than just drop the conversion.
        flattened_masks_u32 = _decode_u32(args["flattened_masks"]).view(np.int32)
        mask_indptr = _decode_u32(args["mask_indptr"]).view(np.int32)

        num_requests = len(args["adapter_indices"])

        # [OPTIMIZATION] Vectorized computation of per-request token counts
        req_token_counts = np.diff(self.qo_indptr)

        # [OPTIMIZATION] Vectorized sequence length calculation
        # seq_len = (num_pages - 1) * kv_page_size + last_len for num_pages > 0
        num_pages = np.diff(self.kv_page_indptr)
        seq_lens = np.where(
            num_pages > 0,
            (num_pages - 1) * kv_page_size + self.kv_last_page_lens,
            self.kv_last_page_lens,
        )

        # [OPTIMIZATION] Vectorized repeat - No Python loops!
        all_seq_lens = np.repeat(seq_lens, req_token_counts)

        # [OPTIMIZATION] Vectorized position IDs calculation
        # Formula: global_index - request_start + context_len
        context_lens = seq_lens - req_token_counts
        global_indices = np.arange(self.total_tokens, dtype=np.int32)
        request_starts = np.repeat(self.qo_indptr[:-1], req_token_counts)
        request_contexts = np.repeat(context_lens, req_token_counts)
        position_ids_np = global_indices - request_starts + request_contexts

        self.timing["mask_loop"] = time.perf_counter() - t0

        # The flat per-token attention mask is only consumed by the prefill
        # wrapper (`wrapper_append.plan(custom_mask=...)`); the decode
        # wrapper ignores it. When every request has query_len=1 (and no
        # user mask elsewhere — the runtime sets `single_token_mode=False`
        # in that case) we can skip the BRLE decode entirely.
        # Speculative-decoding rebuilds its own mask in
        # `batch_tensors.build_spec_expanded_model_inputs`, so spec batches also don't need
        # this initial decode — but spec adds drafts which makes
        # `single_token_mode` False on the spec path's input dict. Either
        # way, gating on `self.single_token_mode` is sufficient.
        t_brle = time.perf_counter()
        if self.single_token_mode:
            self.attention_masks = None
        else:
            token_acc_seq_lens_np = np.zeros(self.total_tokens + 1, dtype=np.int32)
            np.cumsum(all_seq_lens, out=token_acc_seq_lens_np[1:])
            self.attention_masks = decode_brle_batch(
                flattened_masks_u32, mask_indptr, position_ids_np, token_acc_seq_lens_np
            )
        self.timing["brle_decode"] = time.perf_counter() - t_brle

        # Helper to decode bytes as f32 array.
        # Accepts memoryview (shmem zero-copy view from `shmem_schema`) or list.
        def _decode_f32(data):
            if isinstance(data, (bytes, bytearray, memoryview)):
                return np.frombuffer(data, dtype=np.float32)
            return np.array(data, dtype=np.float32)

        # ===== ZERO-COPY SAMPLER PARAMETER LOADING (SoA from Rust) =====
        t0 = time.perf_counter()

        # Load sampler parameters zero-copy from Rust SoA arrays
        self.temperatures = _decode_f32(args["sampler_temperatures"])
        self.top_k_values = _decode_u32(args["sampler_top_k"]).astype(np.int32)
        self.top_p_values = _decode_f32(args["sampler_top_p"])
        self.min_p_values = _decode_f32(args["sampler_min_p"])
        self.sampler_types = _decode_u32(args["sampler_types"]).tolist()
        self.request_output_counts = _decode_u32(args["request_num_samplers"]).astype(
            np.int32
        )

        # Per-sampler ragged label lists (Logprob/Logprobs).
        # `sampler_label_indptr` has length num_samplers + 1; for sampler i,
        # labels are sampler_label_ids[indptr[i] : indptr[i+1]] (empty for
        # samplers that don't carry labels).
        if "sampler_label_ids" in args:
            self.sampler_label_ids = _decode_u32(args["sampler_label_ids"]).astype(
                np.int32
            )
            self.sampler_label_indptr = _decode_u32(
                args["sampler_label_indptr"]
            ).astype(np.int32)
        else:
            self.sampler_label_ids = np.zeros(0, dtype=np.int32)
            self.sampler_label_indptr = np.zeros(1, dtype=np.int32)

        # Load flattened output token indices
        flat_output_indices = _decode_u32(args["sampling_indices"]).astype(
            np.int32
        )
        output_token_indptr = _decode_u32(args["sampling_indptr"]).astype(np.int32)

        # ===== VECTORIZED OUTPUT INDICES OFFSET CALCULATION =====
        # Each request's indices are relative to its token range, we need global offsets
        num_requests = len(self.request_output_counts)

        # For each index in flat_output_indices, add the corresponding request's token offset
        # Create an array mapping each flat index to its request's token offset
        indices_per_request = np.diff(output_token_indptr)
        self.indices_per_request = indices_per_request
        request_token_offsets = self.qo_indptr[:-1]  # Token offset for each request

        # Expand offsets to match each output index
        flat_offsets = np.repeat(request_token_offsets, indices_per_request)

        # Apply offsets to get global indices
        if len(flat_output_indices) > 0:
            self.indices_for_logits = (flat_output_indices + flat_offsets).tolist()
        else:
            self.indices_for_logits = []

        # Stash the relative sampling indices for spec-expanded recomputation.
        # The spec path can't reuse `self.indices_for_logits` because the
        # global offsets shift once drafts are appended to each request.
        # `self.indices_per_request` is already set above.
        self._sampling_indices_relative = flat_output_indices

        # ===== SAMPLING MASK HANDLING =====
        self.sampling_masks = None
        if vocab_size is not None and "sampling_masks" in args:
            # `.view(np.int32)` is a zero-copy reinterpret; values are
            # non-negative and within int32 range. See note on
            # `flattened_masks_u32` above for why we view rather than skip.
            sampling_masks_u32 = _decode_u32(args["sampling_masks"]).view(np.int32)
            sampling_mask_indptr = _decode_u32(args["sampling_mask_indptr"]).view(np.int32)

            # Check if we have any sampling masks at all
            if len(sampling_masks_u32) > 0:
                # 1. Decode per-request masks (num_requests, vocab_size)
                # Initialize to True (allow all) so empty masks work as expected
                per_request_masks = decode_sampling_masks(
                    sampling_masks_u32, sampling_mask_indptr, num_requests, vocab_size
                )

                # 2. Expand to match logits (tokens per request)
                # If doing single-token decoding, indices_per_request is all 1s
                # Logic matches indices_for_logits expansion
                if len(self.indices_for_logits) > 0:
                    # Use indices_per_request from earlier (diff of output_token_indptr)
                    # We repeat the per-request mask rows
                    self.sampling_masks = np.repeat(
                        per_request_masks, indices_per_request, axis=0
                    )

        # ===== ADAPTER HANDLING (still needs per-request loop for now) =====
        adapter_indices = args["adapter_indices"]
        adapter_seeds = args["adapter_seeds"]

        for i in range(num_requests):
            req_token_count = int(req_token_counts[i])
            adapter_idx = adapter_indices[i]
            if adapter_idx is not None:
                seed = adapter_seeds[i] if adapter_seeds[i] is not None else 0
                self.adapter_seeds.extend([seed] * req_token_count)
                self.adapter_indices.append(adapter_idx)
                self.adapter_subpass_needed = True

        self.timing["sampler_loop"] = time.perf_counter() - t0

        # ===== SPECULATIVE DECODING INPUTS =====
        self.spec_token_ids = _decode_i64(args["spec_token_ids"])
        self.spec_position_ids = _decode_u32(args["spec_position_ids"]).astype(np.int32)
        self.spec_indptr = _decode_u32(args["spec_indptr"]).astype(np.int32)
        self.output_spec_flags = args["output_spec_flags"]
        self.sampler_seeds_arr = _decode_u32(args["sampler_seeds"])
        # Host-side check: if every sampler's seed is 0 (the "user did not
        # ask for determinism" sentinel), we can short-circuit the device-
        # side seed plumbing entirely and skip a per-group GPU sync.
        self.has_user_seeds: bool = bool(self.sampler_seeds_arr.any())

        # ===== CONTEXT IDS (per request) =====
        # Stable per-context identifier. Used by drivers that maintain
        # per-context state (e.g. n-gram drafter token history) as the
        # session key — see worker._populate_next_drafts.
        self.context_ids = list(args.get("context_ids", []))

        # ===== LOGIT MASKS (BRLE per request → bool matrix) =====
        # `.view(np.int32)` zero-copy reinterpret; see flattened_masks_u32 note.
        logit_masks_u32 = _decode_u32(args["logit_masks"]).view(np.int32)
        logit_mask_indptr = _decode_u32(args["logit_mask_indptr"]).view(np.int32)
        if len(logit_masks_u32) > 0 and vocab_size is not None:
            self.logit_masks = decode_sampling_masks(
                logit_masks_u32, logit_mask_indptr, num_requests, vocab_size
            )
        else:
            self.logit_masks = None

    @property
    def has_speculative_inputs(self) -> bool:
        """True if any request in this batch supplied draft tokens to verify."""
        return self.spec_token_ids.size > 0

    # ------------------------------------------------------------------
    # Tensor materialization lives in per-flavor wheels (see each driver's
    # `batch_tensors.py`). Bridge owns the wire model only; torch device
    # types and `torch.as_tensor` calls are deliberately not imported here.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Speculative decoding: spec-expanded views
    # ------------------------------------------------------------------
    #
    # When a request supplies draft tokens (`spec_token_ids` non-empty),
    # the wire format keeps drafts in a separate channel. The driver has
    # to splice them into the forward: per request, append the draft
    # tokens after the pending tokens (matching the layout the model will
    # see) and grow `qo_indptr`, `kv_last_page_lens`, the attention mask,
    # and the sampling indices accordingly. The KV pages already exist —
    # the inferlet reserved enough for `pending + drafts` before pinning,
    # and `pin()` returns the full working-page set in `kv_page_indices`.
    #
    # Linear-draft assumption (v1): draft positions for a request are
    # contiguous and sit immediately after the pending tokens
    # (`draft_pos[k] = last_pending_pos + k + 1`). Tree drafts (repeated
    # / non-monotonic positions) need a custom mask on the input side
    # and a tree-aware verification walk; both are out of scope here.
    # `_build_spec_plan` raises if the wire violates this.

    def _build_spec_plan(self) -> None:
        """Compute per-request spec geometry once and stash on `self`.

        Populates:
          * `self._spec_plan`: list[dict | None], one entry per request with
            `n_pending`, `n_drafts`, `pending_start` (offset into
            `self.token_ids`), `drafts_start` (offset into `self.spec_token_ids`).
            `None` for requests without drafts.
          * `self._spec_pending_counts` / `self._spec_n_drafts_per_req` /
            `self._spec_qo_indptr_ext` / `self._spec_seq_lens_old`: per-request
            int32 arrays used by the downstream
            `batch_tensors.build_spec_expanded_{model_inputs,sampling_metadata}`
            helpers. Cached so the two helpers don't recompute the same
            diffs / cumsums.

        Validates the linear-draft assumption (draft positions contiguous
        after the last pending position) and raises if violated.
        """
        num_requests = len(self.request_output_counts)
        plan: list[dict | None] = [None] * num_requests
        pending_counts = np.diff(self.qo_indptr).astype(np.int32)
        draft_counts = np.diff(self.spec_indptr).astype(np.int32)

        # Per-request seq_len BEFORE drafts (matches the formula in __init__).
        num_pages_per_req = np.diff(self.kv_page_indptr)
        seq_lens_old = np.where(
            num_pages_per_req > 0,
            (num_pages_per_req - 1) * self._kv_page_size + self.kv_last_page_lens,
            self.kv_last_page_lens,
        ).astype(np.int64)

        for i in range(num_requests):
            n_drafts = int(draft_counts[i])
            if n_drafts == 0:
                continue
            n_pending = int(pending_counts[i])
            if n_pending == 0:
                # A draft-only request leaves nothing to anchor verification.
                # The runtime never produces this today; bail early.
                raise ValueError(
                    f"Request {i}: drafts supplied without a pending token; "
                    "verification needs at least one pending token to anchor."
                )

            pending_start = int(self.qo_indptr[i])
            drafts_start = int(self.spec_indptr[i])
            last_pending_pos = int(self.position_ids[pending_start + n_pending - 1])

            # Linearity check: drafts must be contiguous immediately after
            # the last pending position. Tree drafts are not yet supported.
            draft_pos = self.spec_position_ids[drafts_start : drafts_start + n_drafts]
            expected = np.arange(
                last_pending_pos + 1,
                last_pending_pos + 1 + n_drafts,
                dtype=draft_pos.dtype,
            )
            if not np.array_equal(draft_pos, expected):
                raise ValueError(
                    f"Request {i}: spec_position_ids must be contiguous after "
                    f"the last pending position (expected {expected.tolist()}, "
                    f"got {draft_pos.tolist()}). Tree drafts are not supported yet."
                )

            plan[i] = {
                "n_pending": n_pending,
                "n_drafts": n_drafts,
                "pending_start": pending_start,
                "drafts_start": drafts_start,
            }

        # Cache derived arrays on self for downstream methods.
        n_drafts_per_req = draft_counts
        new_token_counts = pending_counts + n_drafts_per_req
        qo_indptr_ext = np.zeros(num_requests + 1, dtype=np.int32)
        np.cumsum(new_token_counts, out=qo_indptr_ext[1:])

        self._spec_plan = plan
        self._spec_pending_counts = pending_counts
        self._spec_n_drafts_per_req = n_drafts_per_req
        self._spec_qo_indptr_ext = qo_indptr_ext
        self._spec_seq_lens_old = seq_lens_old

    # ------------------------------------------------------------------
    # Sampling metadata + verification (verification stays here: pure
    # numpy/python). Tensor-flavored sampling metadata builders live
    # downstream in `batch_tensors.py`.
    # ------------------------------------------------------------------

    def verify_drafts(self, sampling_results: dict[str, Any]) -> None:
        """Walk verification samples and produce per-request accepted tokens.

        For each spec request, the verification block contains samples at
        `[last_pending_pos, draft_1_pos, ..., draft_n_pos]`. Compare the
        (i-1)-th verification sample to the i-th draft to find the longest
        accepted prefix; the first sample is always accepted as the bonus
        token at the last pending position. Mutates `sampling_results` in
        place to add `spec_accepted_tokens: list[list[int] | None]`,
        one entry per request (None when the request had no drafts).
        """
        if self._spec_plan is None:
            return
        plan = self._spec_plan
        num_requests = len(self.request_output_counts)
        tokens = sampling_results.get("tokens", [])
        if not tokens:
            sampling_results["spec_accepted_tokens"] = [None] * num_requests
            return

        verify_offset = self._verify_block_offset
        accepted_per_req: list[list[int] | None] = [None] * num_requests

        for i in range(num_requests):
            if plan[i] is None or self._verify_slot_starts[i] is None:
                continue
            start, n_drafts = self._verify_slot_starts[i]
            block = tokens[verify_offset + start : verify_offset + start + n_drafts + 1]
            # Drafts the inferlet sent for this request.
            ds = plan[i]["drafts_start"]
            drafts = self.spec_token_ids[ds : ds + n_drafts]

            # Match prefix: block[k-1] (= sample at draft_k's parent position)
            # vs drafts[k-1] (= the k-th draft). Stop at first mismatch.
            match = 0
            for k in range(n_drafts):
                if int(block[k]) == int(drafts[k]):
                    match += 1
                else:
                    break
            # Accepted = matched drafts (= block[0..match]) + bonus block[match].
            # Length = match + 1.
            accepted_per_req[i] = [int(t) for t in block[: match + 1]]

        sampling_results["spec_accepted_tokens"] = accepted_per_req

    def create_responses(
        self, sampling_results: dict[str, Any]
    ) -> list[message.ForwardPassResponse]:
        """
        Package the sampling results into responses for each original request.

        Args:
            sampling_results: Dictionary containing 'tokens' and 'dists'.

        Returns:
            List of responses in the order of requests.
        """
        num_requests = len(self.request_output_counts)

        # Early return if no logits needed
        if not self.indices_for_logits:
            return [
                message.ForwardPassResponse(dists=[], tokens=[])
                for _ in range(num_requests)
            ]

        final_dists = sampling_results["dists"]
        final_logits = sampling_results.get("logits") or []
        final_logprobs = sampling_results.get("logprobs") or []
        final_entropies = sampling_results.get("entropies") or []
        final_tokens_list = sampling_results["tokens"]
        spec_tokens_all = sampling_results.get("spec_tokens", None)
        spec_positions_all = sampling_results.get("spec_positions", None)
        # Per-request accepted-token override produced by `verify_drafts`.
        spec_accepted_all = sampling_results.get("spec_accepted_tokens", None)

        responses = []
        cursor = 0
        # Hoist per-iter conversions out of the loop.
        counts_list = self.request_output_counts.tolist() if hasattr(
            self.request_output_counts, "tolist"
        ) else list(self.request_output_counts)
        sampler_types = self.sampler_types  # already a Python list
        output_spec_flags = self.output_spec_flags
        # Bound logits/logprobs/entropies len checks once.
        _len_logits = len(final_logits)
        _len_logprobs = len(final_logprobs)
        _len_entropies = len(final_entropies)

        for req_idx in range(num_requests):
            num_outputs = counts_list[req_idx]
            request_dists = []
            request_tokens = []
            request_logits: list[bytes] = []
            request_logprobs: list[list[float]] = []
            request_entropies: list[float] = []

            spec_accepted = (
                spec_accepted_all[req_idx] if spec_accepted_all is not None else None
            )
            if spec_accepted is not None:
                # Spec-mode request: tokens come from the verification walk,
                # not from the per-sampler aggregation. Distributions in spec
                # mode are not supported (verify_drafts requires a concrete
                # sampler — see batch_tensors.build_spec_expanded_sampling_metadata).
                request_tokens = list(spec_accepted)
            else:
                end = cursor + num_outputs
                # Hot path: 1 token output per request (most common).
                if num_outputs == 1:
                    stype = sampler_types[cursor]
                    if stype == 0:
                        if final_dists[cursor] is not None:
                            request_dists.append(final_dists[cursor])
                    elif stype == 7:
                        if cursor < _len_logits and final_logits[cursor] is not None:
                            request_logits.append(final_logits[cursor])
                    elif stype in (8, 9):
                        if cursor < _len_logprobs and final_logprobs[cursor] is not None:
                            request_logprobs.append(final_logprobs[cursor])
                    elif stype == 10:
                        if cursor < _len_entropies and final_entropies[cursor] is not None:
                            request_entropies.append(final_entropies[cursor])
                    else:
                        request_tokens.append(final_tokens_list[cursor])
                else:
                    for i in range(cursor, end):
                        stype = sampler_types[i]
                        if stype == 0:
                            if final_dists[i] is not None:
                                request_dists.append(final_dists[i])
                        elif stype == 7:
                            if i < _len_logits and final_logits[i] is not None:
                                request_logits.append(final_logits[i])
                        elif stype in (8, 9):
                            if i < _len_logprobs and final_logprobs[i] is not None:
                                request_logprobs.append(final_logprobs[i])
                        elif stype == 10:
                            if i < _len_entropies and final_entropies[i] is not None:
                                request_entropies.append(final_entropies[i])
                        else:
                            request_tokens.append(final_tokens_list[i])

            # Build response with optional speculation
            resp = message.ForwardPassResponse(
                dists=request_dists,
                tokens=request_tokens,
                logits=request_logits,
                logprobs=request_logprobs,
                entropies=request_entropies,
            )
            if (
                spec_tokens_all is not None
                and output_spec_flags[req_idx]
                and spec_tokens_all[req_idx] is not None
            ):
                resp.spec_tokens = spec_tokens_all[req_idx]
                resp.spec_positions = spec_positions_all[req_idx]

            responses.append(resp)
            cursor += num_outputs

        return responses


@njit(cache=True)
def decode_brle_batch(
    flattened_masks: np.ndarray,
    mask_indptr: np.ndarray,
    position_ids: np.ndarray,
    token_acc_seq_lens: np.ndarray,
) -> np.ndarray:
    """
    Decode BRLE masks for an entire batch using Numba JIT.

    Optimized with slice assignment which Numba compiles to SIMD/memset
    block writes - faster than parallel (prange) due to thread pool overhead.

    Args:
        flattened_masks: Concatenated BRLE run lengths (int32)
        mask_indptr: Pointers to BRLE ranges per token (int32)
        position_ids: Position of each token, defines valid_len (int32)
        token_acc_seq_lens: Cumulative bit offsets per token (int32)

    Returns:
        Flat boolean array with all mask values
    """
    num_tokens = len(position_ids)
    total_bits = token_acc_seq_lens[-1]
    result = np.zeros(total_bits, dtype=np.bool_)

    # BRLE format (see runtime/src/inference/brle.rs): the sequence always
    # begins with a `false` run (possibly zero-length), then alternates
    # false, true, false, true, ... The inferlet convention for attention
    # masks (see e.g. inferlets/attention-sink/src/lib.rs) matches:
    #   [count_of_0s, count_of_1s, count_of_0s, ...]
    # where 1 = "attend" (custom_mask True in flashinfer).
    for k in range(num_tokens):
        rle_start = mask_indptr[k]
        rle_end = mask_indptr[k + 1]
        global_bit_start = token_acc_seq_lens[k]
        # Row size is the request's full seq_len (extracted from the
        # cumulative offsets we precomputed). Capping at `position_ids[k] + 1`
        # — as the previous code did — silently dropped any True bits the
        # inferlet emitted past the diagonal, so non-causal patterns
        # (Jacobi, tree decoding, attention sink with explicit forward bits)
        # decoded as effectively causal. For purely-causal masks this is a
        # no-op since the runtime emits zero bits past the diagonal anyway.
        valid_len = token_acc_seq_lens[k + 1] - token_acc_seq_lens[k]

        curr_bit_pos = global_bit_start
        bits_consumed = 0
        is_true_run = False  # BRLE always starts with a (possibly-empty) false run

        for run_idx in range(rle_start, rle_end):
            if bits_consumed >= valid_len:
                break

            run_len = flattened_masks[run_idx]
            remaining = valid_len - bits_consumed
            eff_len = min(run_len, remaining)

            if is_true_run and eff_len > 0:
                # Slice assignment compiles to SIMD/memset
                result[curr_bit_pos : curr_bit_pos + eff_len] = True

            bits_consumed += eff_len
            curr_bit_pos += eff_len
            is_true_run = not is_true_run

    return result


@njit(cache=True)
def decode_sampling_masks(
    flattened_masks: np.ndarray,
    mask_indptr: np.ndarray,
    num_requests: int,
    vocab_size: int,
) -> np.ndarray:
    """
    Decode BRLE sampling masks.

    Args:
        flattened_masks: Concatenated BRLE run lengths (int32)
        mask_indptr: Pointers to BRLE ranges per request (int32)
        num_requests: Number of requests
        vocab_size: Size of vocabulary (dim 1 of output)

    Returns:
        Boolean array of shape (num_requests, vocab_size).

    The BRLE format (see runtime/src/inference/brle.rs) always starts with
    a `false` run (possibly zero-length), then alternates: false, true,
    false, true, ... When a request has an empty BRLE we default to True
    (allow all), matching the legacy "no constraint = no masking" behavior.
    """
    result = np.ones((num_requests, vocab_size), dtype=np.bool_)

    for i in range(num_requests):
        rle_start = mask_indptr[i]
        rle_end = mask_indptr[i + 1]

        # Empty BRLE for this request = no constraint, leave as all-True.
        if rle_end == rle_start:
            continue

        # Non-empty BRLE: start from all-False and fill in the True runs.
        result[i, :] = False

        curr_pos = 0
        is_true_run = False  # BRLE always starts with a (possibly-empty) false run

        for run_idx in range(rle_start, rle_end):
            if curr_pos >= vocab_size:
                break

            run_len = flattened_masks[run_idx]
            remaining = vocab_size - curr_pos
            eff_len = min(run_len, remaining)

            if eff_len > 0 and is_true_run:
                result[i, curr_pos : curr_pos + eff_len] = True

            curr_pos += eff_len
            is_true_run = not is_true_run

    return result
