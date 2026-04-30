"""
Batch management for Pie driver inference.

This module provides the Batch dataclass that holds inference batch state
and handles tensor creation and response packaging.
"""

from __future__ import annotations

from typing import Any

import time
import numpy as np
import torch
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
            kv_page_size: KV cache page size from model config
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
        # Accepts bytes (msgpack), memoryview (shmem zero-copy schema), or list.
        def _decode_u32(data):
            if isinstance(data, (bytes, bytearray, memoryview)):
                return np.frombuffer(data, dtype=np.uint32)
            return np.array(data, dtype=np.uint32)

        # Helper to decode bytes as i64 array
        def _decode_i64(data):
            if isinstance(data, (bytes, bytearray, memoryview)):
                # Wire format stores token_ids as u32 in both msgpack and shmem;
                # widen to i64 for torch.
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

        # The decode kernel is a specialization of the prefill kernel that
        # drops `custom_mask` for efficiency. Rust now sets `single_token_mode`
        # correctly: it forces the prefill path whenever any request supplied
        # a user `attention_mask` (regardless of token count), so we trust the
        # flag here. `has_custom_mask` is kept as a tensor-allocation gate
        # downstream.
        self.has_custom_mask = len(flattened_masks_u32) > 0

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

        # [OPTIMIZATION] Vectorized cumsum for bit offsets
        token_acc_seq_lens_np = np.zeros(self.total_tokens + 1, dtype=np.int32)
        np.cumsum(all_seq_lens, out=token_acc_seq_lens_np[1:])

        self.timing["mask_loop"] = time.perf_counter() - t0

        # Call batch decoder ONCE for all tokens
        t_brle = time.perf_counter()
        self.attention_masks = decode_brle_batch(
            flattened_masks_u32, mask_indptr, position_ids_np, token_acc_seq_lens_np
        )
        self.timing["brle_decode"] = time.perf_counter() - t_brle

        # Helper to decode bytes as f32 array.
        # Accepts bytes (msgpack), memoryview (shmem zero-copy schema), or list.
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

    def get_model_inputs(self, device: torch.device) -> dict[str, Any]:
        """
        Finalize batch preparation and create input tensors for the model.

        Args:
            device: The torch device to create tensors on.

        Returns:
            Dictionary containing input tensors for the model engine.
        """
        # self.adapter_subpass_needed = False  # disable ZO for testing

        # Pre-compute CPU-side bounds for engine.fire_batch's validation —
        # avoids 6 CUDA `.item()` syncs per forward pass.
        if len(self.kv_page_indices) > 0:
            _kv_idx_max = int(self.kv_page_indices.max())
            _kv_idx_min = int(self.kv_page_indices.min())
        else:
            _kv_idx_max = 0
            _kv_idx_min = 0
        _kv_indptr_last = int(self.kv_page_indptr[-1]) if len(self.kv_page_indptr) > 0 else 0
        if len(self.kv_last_page_lens) > 0:
            _kv_last_max = int(self.kv_last_page_lens.max())
            _kv_last_min = int(self.kv_last_page_lens.min())
        else:
            _kv_last_max = 0
            _kv_last_min = 1

        return {
            "kv_page_indices_max": _kv_idx_max,
            "kv_page_indices_min": _kv_idx_min,
            "kv_page_indptr_last": _kv_indptr_last,
            "kv_last_page_lens_max": _kv_last_max,
            "kv_last_page_lens_min": _kv_last_min,
            "token_ids": torch.as_tensor(
                self.token_ids, device=device, dtype=torch.long
            ),
            "position_ids": torch.as_tensor(
                self.position_ids, device=device, dtype=torch.int32
            ),
            "qo_indptr": torch.as_tensor(
                self.qo_indptr, device=device, dtype=torch.int32
            ),
            "kv_page_indices": torch.as_tensor(
                self.kv_page_indices, device=device, dtype=torch.int32
            ).contiguous(),
            "kv_page_indptr": torch.as_tensor(
                self.kv_page_indptr, device=device, dtype=torch.int32
            ).contiguous(),
            "kv_last_page_lens": torch.as_tensor(
                self.kv_last_page_lens, device=device, dtype=torch.int32
            ),
            "custom_mask": torch.as_tensor(
                self.attention_masks, device=device, dtype=torch.bool
            ),
            "has_custom_mask": self.has_custom_mask,
            "single_token_inference_mode": self.single_token_mode,
            "adapter_indices": (
                self.adapter_indices if self.adapter_subpass_needed else []
            ),
            "adapter_seeds": (
                torch.as_tensor(self.adapter_seeds, device=device, dtype=torch.long)
                if self.adapter_subpass_needed
                else None
            ),
            "spec_token_ids": (
                torch.as_tensor(self.spec_token_ids, device=device, dtype=torch.long)
                if len(self.spec_token_ids) > 0
                else None
            ),
            "spec_position_ids": (
                torch.as_tensor(self.spec_position_ids, device=device, dtype=torch.int32)
                if len(self.spec_position_ids) > 0
                else None
            ),
            "spec_indptr": (
                torch.as_tensor(self.spec_indptr, device=device, dtype=torch.int32)
                if len(self.spec_indptr) > 1
                else None
            ),
        }

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
            int32 arrays used by both `get_spec_expanded_model_inputs` and
            `get_spec_expanded_sampling_metadata`. Cached so the two methods
            don't recompute the same diffs / cumsums.

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

    def get_spec_expanded_model_inputs(
        self, device: torch.device
    ) -> dict[str, Any]:
        """Like `get_model_inputs`, but with drafts spliced into the forward.

        Only call when `has_speculative_inputs` is True. Triggers
        `_build_spec_plan` (idempotent) so `get_spec_expanded_sampling_metadata`
        and `verify_drafts` can read the cached `_spec_*` fields.
        """
        self._build_spec_plan()
        plan = self._spec_plan
        pending_counts = self._spec_pending_counts
        n_drafts_per_req = self._spec_n_drafts_per_req
        qo_indptr_ext = self._spec_qo_indptr_ext
        seq_lens_old = self._spec_seq_lens_old
        num_requests = len(self.request_output_counts)
        new_token_counts = pending_counts + n_drafts_per_req

        # ---- token_ids / position_ids: per-request [pending | drafts]. ----
        new_total = int(new_token_counts.sum())
        token_ids_ext = np.empty(new_total, dtype=self.token_ids.dtype)
        position_ids_ext = np.empty(new_total, dtype=self.position_ids.dtype)

        for i in range(num_requests):
            dst = qo_indptr_ext[i]
            n_p = int(pending_counts[i])
            n_d = int(n_drafts_per_req[i])
            ps = int(self.qo_indptr[i])
            token_ids_ext[dst : dst + n_p] = self.token_ids[ps : ps + n_p]
            position_ids_ext[dst : dst + n_p] = self.position_ids[ps : ps + n_p]
            if n_d > 0:
                ds = plan[i]["drafts_start"]
                token_ids_ext[dst + n_p : dst + n_p + n_d] = self.spec_token_ids[
                    ds : ds + n_d
                ]
                position_ids_ext[dst + n_p : dst + n_p + n_d] = self.spec_position_ids[
                    ds : ds + n_d
                ]

        # ---- kv_last_page_lens: bump by n_drafts per request. ----
        # `compute_last_page_len` semantics: r = total_kv % page_size, with
        # 0 → page_size. We mirror that here so spillover into a fresh
        # already-reserved page yields page_size, not 0.
        page_size = self._kv_page_size
        new_total_kv_last = self.kv_last_page_lens.astype(np.int64) + n_drafts_per_req
        kv_last_page_lens_ext = ((new_total_kv_last - 1) % page_size + 1).astype(np.int32)

        # ---- attention_masks: rebuild as a flat causal bool tensor. ----
        # v1 simplification: all rows in the spec-expanded forward use a
        # plain causal mask over [0, position]. This is what the runtime
        # synthesizes for non-spec inferlets (api/inference.rs:297-301).
        # Inferlets that combine custom non-causal masks with drafts are
        # rejected upstream and would need explicit row plumbing here.
        seq_lens_per_req = seq_lens_old + n_drafts_per_req
        per_token_seq_lens = np.repeat(seq_lens_per_req, new_token_counts)
        total_bits = int(per_token_seq_lens.sum())
        attention_masks_ext = np.zeros(total_bits, dtype=np.bool_)
        # Fill: each token at absolute position `pos` attends to indices
        # [0, pos]. The row length equals the request's new seq_len.
        offset = 0
        for k in range(new_total):
            row_len = int(per_token_seq_lens[k])
            valid = min(int(position_ids_ext[k]) + 1, row_len)
            attention_masks_ext[offset : offset + valid] = True
            offset += row_len

        # No longer single-token: we appended drafts, so each request
        # forwards multiple tokens.
        single_token_mode_ext = self.single_token_mode and not self.has_speculative_inputs

        return {
            "token_ids": torch.as_tensor(token_ids_ext, device=device, dtype=torch.long),
            "position_ids": torch.as_tensor(
                position_ids_ext, device=device, dtype=torch.int32
            ),
            "qo_indptr": torch.as_tensor(qo_indptr_ext, device=device, dtype=torch.int32),
            "kv_page_indices": torch.as_tensor(
                self.kv_page_indices, device=device, dtype=torch.int32
            ).contiguous(),
            "kv_page_indptr": torch.as_tensor(
                self.kv_page_indptr, device=device, dtype=torch.int32
            ).contiguous(),
            "kv_last_page_lens": torch.as_tensor(
                kv_last_page_lens_ext, device=device, dtype=torch.int32
            ),
            "custom_mask": torch.as_tensor(
                attention_masks_ext, device=device, dtype=torch.bool
            ),
            "single_token_inference_mode": single_token_mode_ext,
            "adapter_indices": (
                self.adapter_indices if self.adapter_subpass_needed else []
            ),
            "adapter_seeds": (
                torch.as_tensor(self.adapter_seeds, device=device, dtype=torch.long)
                if self.adapter_subpass_needed
                else None
            ),
        }

    def get_sampling_metadata(
        self, device: torch.device, dtype: torch.dtype
    ) -> dict[str, Any]:
        """
        Prepare the metadata required for the SamplingPass.

        Args:
            device: Torch device.
            dtype: Torch dtype for temperatures.

        Returns:
            Dictionary containing sampling metadata.
        """
        # Return empty if no logits needed
        if not self.indices_for_logits:
            return {"indices_for_logits": None}

        indices_for_logits = self.indices_for_logits

        # Vectorized tensor creation from NumPy arrays (no list comprehension)
        temperatures = (
            torch.tensor(self.temperatures, device=device, dtype=dtype)
            .clamp(min=1e-6)
            .unsqueeze(1)
        )

        # Pre-build sampler param tensors (avoid per-group construction)
        top_k_tensor = torch.tensor(self.top_k_values, device=device, dtype=torch.long)
        top_p_tensor = torch.tensor(self.top_p_values, device=device, dtype=dtype)
        min_p_tensor = torch.tensor(self.min_p_values, device=device, dtype=dtype)

        # Group samplers
        sampler_groups: dict[int, list[int]] = {}
        for i, sampler_idx in enumerate(self.sampler_types):
            if sampler_idx not in sampler_groups:
                sampler_groups[sampler_idx] = []
            sampler_groups[sampler_idx].append(i)

        # Per-sampler RNG seeds (u32 values from Rust; 0 means "no seed").
        # Only build the tensor if at least one sampler asked for a seed —
        # the all-zeros case is the common one and we want to skip the
        # device transfer + the kernel-side seed plumbing.
        seeds_tensor = (
            torch.as_tensor(
                self.sampler_seeds_arr.astype(np.int64),
                device=device,
                dtype=torch.long,
            )
            if self.has_user_seeds
            else None
        )

        return {
            "indices_for_logits": indices_for_logits,
            "temperatures": temperatures,
            "sampler_groups": sampler_groups,
            "top_k": top_k_tensor,
            "top_p": top_p_tensor,
            "min_p": min_p_tensor,
            "seeds": seeds_tensor,
            # Per-sampler label lists for Logprob/Logprobs paths. Indexed by
            # the sampler slot (i.e. the i in sampler_groups[type] = [i, ...]).
            "sampler_label_ids": self.sampler_label_ids,
            "sampler_label_indptr": self.sampler_label_indptr,
            # `sampling_masks` is the effective logit mask at each logit
            # position. Prefer whichever source populated it (historically two
            # names coexisted; `logit_masks` came from grammar constraints via
            # forward_pass.logit_mask, `sampling_masks` via a now-unused arg).
            "sampling_masks": (
                torch.as_tensor(self.sampling_masks, device=device, dtype=torch.bool)
                if self.sampling_masks is not None
                else (
                    torch.as_tensor(
                        np.repeat(self.logit_masks, self.indices_per_request, axis=0),
                        device=device,
                        dtype=torch.bool,
                    )
                    if self.logit_masks is not None
                    else None
                )
            ),
        }

    # ------------------------------------------------------------------
    # Speculative decoding: sampling metadata + verification
    # ------------------------------------------------------------------

    def get_spec_expanded_sampling_metadata(
        self, device: torch.device, dtype: torch.dtype
    ) -> dict[str, Any]:
        """Sampling metadata for the spec-expanded forward.

        Recomputes `indices_for_logits` against `qo_indptr_ext` (since the
        global token offsets shift once drafts are appended) and appends a
        verification block of `n_drafts + 1` extra samples per spec
        request — sampled at `[last_pending_pos, draft_1_pos, ..., draft_n_pos]`
        in the expanded layout. The verification samples reuse the
        inferlet's first sampler config for that request, so token-vs-draft
        comparison is consistent with what the inferlet would have sampled.

        Must be called after `get_spec_expanded_model_inputs`, which
        populates `self._spec_plan`.
        """
        if self._spec_plan is None:
            raise RuntimeError(
                "get_spec_expanded_sampling_metadata called before "
                "get_spec_expanded_model_inputs (no _spec_plan cached)."
            )
        plan = self._spec_plan
        qo_indptr_ext = self._spec_qo_indptr_ext
        num_requests = len(self.request_output_counts)

        # ---- Inferlet's samplers, re-offset against qo_indptr_ext. ----
        if len(self._sampling_indices_relative) > 0:
            flat_offsets_new = np.repeat(
                qo_indptr_ext[:-1], self.indices_per_request
            )
            inferlet_indices = (
                self._sampling_indices_relative + flat_offsets_new
            ).tolist()
        else:
            inferlet_indices = []

        # ---- Verification block (per spec request). ----
        verify_indices: list[int] = []
        # Per-request: (start_in_verify_block, n_drafts) so verify_drafts
        # can slice each request's verification samples back out.
        verify_slot_starts: list[tuple[int, int] | None] = [None] * num_requests
        # Per-request: index into the inferlet sampler arrays of the FIRST
        # sampler for that request, used to clone the sampler config across
        # all of this request's verification slots.
        first_sampler_idx_per_req: list[int | None] = [None] * num_requests
        cursor = 0
        for i in range(num_requests):
            if int(self.request_output_counts[i]) > 0:
                first_sampler_idx_per_req[i] = cursor
            cursor += int(self.request_output_counts[i])

        running = 0  # offset within the verification block
        for i in range(num_requests):
            if plan[i] is None:
                continue
            n_p = plan[i]["n_pending"]
            n_d = plan[i]["n_drafts"]
            base = int(qo_indptr_ext[i]) + n_p - 1  # last pending position (global)
            verify_indices.extend(range(base, base + n_d + 1))
            verify_slot_starts[i] = (running, n_d)
            running += n_d + 1

            if first_sampler_idx_per_req[i] is None:
                # Spec mode without an inferlet sampler at the request — no
                # config to copy. Bail with a clear error; the typical inferlet
                # always asks for at least one sample (the bonus token).
                raise ValueError(
                    f"Request {i}: speculative input supplied but no inferlet "
                    "sampler — verification needs a sampler config to clone."
                )

        # Concatenate the inferlet sampling block with the verification block.
        indices_for_logits = inferlet_indices + verify_indices
        if not indices_for_logits:
            return {"indices_for_logits": None}

        # Per-sampler arrays: extend by cloning each spec request's first
        # sampler config across its verification slots.
        def _extend_with_clones(src: np.ndarray) -> np.ndarray:
            extra = []
            for i in range(num_requests):
                if verify_slot_starts[i] is None:
                    continue
                fs = first_sampler_idx_per_req[i]
                count = verify_slot_starts[i][1] + 1
                extra.append(np.repeat(src[fs : fs + 1], count))
            if not extra:
                return src
            return np.concatenate([src] + extra)

        temperatures_arr = _extend_with_clones(self.temperatures)
        top_k_arr = _extend_with_clones(self.top_k_values)
        top_p_arr = _extend_with_clones(self.top_p_values)
        min_p_arr = _extend_with_clones(self.min_p_values)
        seeds_arr = _extend_with_clones(self.sampler_seeds_arr)

        # sampler_types is a Python list, not a numpy array.
        sampler_types_ext = list(self.sampler_types)
        for i in range(num_requests):
            if verify_slot_starts[i] is None:
                continue
            fs = first_sampler_idx_per_req[i]
            count = verify_slot_starts[i][1] + 1
            stype = self.sampler_types[fs]
            if stype == 0:
                # Distribution sampler can't be used for verification — we
                # need a concrete token to compare against. v1 bails; if
                # this turns up in practice we can fall back to greedy.
                raise ValueError(
                    f"Request {i}: first sampler is Distribution-mode, "
                    "which can't be used for spec verification."
                )
            if stype in (7, 8, 9, 10):
                # Same constraint as Distribution: RawLogits/Logprob/
                # Logprobs/Entropy don't yield a sampled token for the
                # verifier to compare against.
                raise ValueError(
                    f"Request {i}: first sampler is type {stype} "
                    "(RawLogits/Logprob/Logprobs/Entropy), which can't be "
                    "used for spec verification."
                )
            sampler_types_ext.extend([stype] * count)

        # Group samplers (now spans inferlet + verification slots).
        sampler_groups: dict[int, list[int]] = {}
        for i, sampler_idx in enumerate(sampler_types_ext):
            sampler_groups.setdefault(sampler_idx, []).append(i)

        # Tensors.
        temperatures_t = (
            torch.tensor(temperatures_arr, device=device, dtype=dtype)
            .clamp(min=1e-6)
            .unsqueeze(1)
        )
        top_k_t = torch.tensor(top_k_arr, device=device, dtype=torch.long)
        top_p_t = torch.tensor(top_p_arr, device=device, dtype=dtype)
        min_p_t = torch.tensor(min_p_arr, device=device, dtype=dtype)
        seeds_t = (
            torch.as_tensor(seeds_arr.astype(np.int64), device=device, dtype=torch.long)
            if self.has_user_seeds
            else None
        )

        # Logit/sampling masks: extend in lock-step with the verification
        # block (same mask as the inferlet's first sampler for that request).
        sampling_masks_t = None
        if self.sampling_masks is not None:
            extras = []
            for i in range(num_requests):
                if verify_slot_starts[i] is None:
                    continue
                fs = first_sampler_idx_per_req[i]
                count = verify_slot_starts[i][1] + 1
                extras.append(np.repeat(self.sampling_masks[fs : fs + 1], count, axis=0))
            mask_np = (
                np.concatenate([self.sampling_masks] + extras, axis=0)
                if extras
                else self.sampling_masks
            )
            sampling_masks_t = torch.as_tensor(mask_np, device=device, dtype=torch.bool)
        elif self.logit_masks is not None:
            base = np.repeat(self.logit_masks, self.indices_per_request, axis=0)
            extras = []
            for i in range(num_requests):
                if verify_slot_starts[i] is None:
                    continue
                count = verify_slot_starts[i][1] + 1
                extras.append(np.repeat(self.logit_masks[i : i + 1], count, axis=0))
            mask_np = np.concatenate([base] + extras, axis=0) if extras else base
            sampling_masks_t = torch.as_tensor(mask_np, device=device, dtype=torch.bool)

        # Stash verify_slot_starts and the offset where the verify block
        # starts in `tokens` for verify_drafts to read.
        self._verify_slot_starts = verify_slot_starts
        self._verify_block_offset = len(inferlet_indices)

        return {
            "indices_for_logits": indices_for_logits,
            "temperatures": temperatures_t,
            "sampler_groups": sampler_groups,
            "top_k": top_k_t,
            "top_p": top_p_t,
            "min_p": min_p_t,
            "seeds": seeds_t,
            "sampling_masks": sampling_masks_t,
        }

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
                # sampler — see get_spec_expanded_sampling_metadata).
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
