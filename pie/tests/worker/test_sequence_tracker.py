"""Tests for SequenceTracker with explicit request identity.

These tests verify that build_scheduler_output correctly classifies
requests as NEW vs CACHED when given explicit request_ids and is_new
parameters, preventing block-reuse collisions.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Mock vLLM types so tests run without the vllm package installed.
# SequenceTracker imports vLLM lazily inside build_scheduler_output,
# so we intercept those imports with lightweight stand-ins.
# ---------------------------------------------------------------------------

class _FakeNewRequestData:
    """Stand-in for vllm.v1.core.sched.output.NewRequestData."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeCachedRequestData:
    """Stand-in for vllm.v1.core.sched.output.CachedRequestData."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def make_empty(cls):
        return cls(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )


class _FakeSchedulerOutput:
    """Stand-in for vllm.v1.core.sched.output.SchedulerOutput."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeSamplingParams:
    """Stand-in for vllm.sampling_params.SamplingParams."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _install_vllm_mocks():
    """Install fake vllm modules so lazy imports inside SequenceTracker succeed."""
    import types

    sched_output = types.ModuleType("vllm.v1.core.sched.output")
    sched_output.NewRequestData = _FakeNewRequestData
    sched_output.CachedRequestData = _FakeCachedRequestData
    sched_output.SchedulerOutput = _FakeSchedulerOutput

    sampling_mod = types.ModuleType("vllm.sampling_params")
    sampling_mod.SamplingParams = _FakeSamplingParams

    # Build the module hierarchy so `from vllm.v1.core.sched.output import ...` works.
    vllm_mod = types.ModuleType("vllm")
    vllm_v1 = types.ModuleType("vllm.v1")
    vllm_v1_core = types.ModuleType("vllm.v1.core")
    vllm_v1_core_sched = types.ModuleType("vllm.v1.core.sched")

    vllm_mod.v1 = vllm_v1
    vllm_v1.core = vllm_v1_core
    vllm_v1_core.sched = vllm_v1_core_sched
    vllm_v1_core_sched.output = sched_output
    vllm_mod.sampling_params = sampling_mod

    sys.modules["vllm"] = vllm_mod
    sys.modules["vllm.v1"] = vllm_v1
    sys.modules["vllm.v1.core"] = vllm_v1_core
    sys.modules["vllm.v1.core.sched"] = vllm_v1_core_sched
    sys.modules["vllm.v1.core.sched.output"] = sched_output
    sys.modules["vllm.sampling_params"] = sampling_mod


# Install mocks before importing SequenceTracker (which may trigger
# top-level vllm references in its module graph).
_install_vllm_mocks()

from pie_worker.vllm_sequence_tracker import SequenceTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch(
    *,
    token_ids: list[int],
    tokens_per_req: list[int],
    blocks_per_req: list[list[int]],
    seq_lens: list[int],
    request_ids: list[str],
    is_new: list[bool],
    batch_id: int = 0,
):
    """Helper to build kwargs for build_scheduler_output."""
    flat = np.array(token_ids, dtype=np.int64)
    cumsum = np.cumsum([0] + tokens_per_req).astype(np.int32)
    sls = np.array(seq_lens, dtype=np.int32)
    sp = [{"temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0}] * len(tokens_per_req)
    return dict(
        batch_id=batch_id,
        token_ids=flat,
        qo_indptr=cumsum,
        tokens_per_req=tokens_per_req,
        blocks_per_req=blocks_per_req,
        seq_lens=sls,
        sampling_params_list=sp,
        request_ids=request_ids,
        is_new=is_new,
    )


def _count_new_reqs(output) -> int:
    """Count NewRequestData entries in a SchedulerOutput."""
    return len(output.scheduled_new_reqs)


def _new_req_ids(output) -> list[str]:
    """Extract req_ids from NewRequestData entries."""
    return [r.req_id for r in output.scheduled_new_reqs]


def _cached_req_ids(output) -> list[str]:
    """Extract req_ids from CachedRequestData."""
    return list(output.scheduled_cached_reqs.req_ids)


# ---------------------------------------------------------------------------
# Test 1: Block-reuse collision
# ---------------------------------------------------------------------------

class TestBlockReuseCollision:
    """Request A uses block 5, finishes. Request B gets block 5.

    The tracker must treat B as a genuinely NEW request (not a
    continuation of A). Token history must NOT be corrupted: B's
    history should contain only B's tokens, not A's.
    """

    def test_new_request_after_block_reuse(self):
        """When block 5 is reused by a new request_id, it must be NEW."""
        tracker = SequenceTracker()

        # --- Batch 0: Request A uses block 5, prefill 3 tokens ---
        out_a = tracker.build_scheduler_output(**_make_batch(
            token_ids=[10, 11, 12],
            tokens_per_req=[3],
            blocks_per_req=[[5]],
            seq_lens=[3],
            request_ids=["req-A"],
            is_new=[True],
            batch_id=0,
        ))
        assert _count_new_reqs(out_a) == 1, "A should be NEW on first appearance"

        # --- Batch 1: Request A continues (1 decode token) ---
        out_a2 = tracker.build_scheduler_output(**_make_batch(
            token_ids=[13],
            tokens_per_req=[1],
            blocks_per_req=[[5]],
            seq_lens=[4],
            request_ids=["req-A"],
            is_new=[False],
            batch_id=1,
        ))
        assert _count_new_reqs(out_a2) == 0, "A should be CACHED on second call"
        assert len(_cached_req_ids(out_a2)) == 1

        # --- Batch 2: A is gone, B arrives with the SAME block 5 ---
        out_b = tracker.build_scheduler_output(**_make_batch(
            token_ids=[50, 51],
            tokens_per_req=[2],
            blocks_per_req=[[5]],
            seq_lens=[2],
            request_ids=["req-B"],
            is_new=[True],
            batch_id=2,
        ))
        assert _count_new_reqs(out_b) == 1, "B must be NEW despite same block"

    def test_token_history_not_corrupted_after_reuse(self):
        """B's token history must contain only B's tokens, not A's."""
        tracker = SequenceTracker()

        # A prefills with tokens [10, 11, 12] on block 5
        tracker.build_scheduler_output(**_make_batch(
            token_ids=[10, 11, 12],
            tokens_per_req=[3],
            blocks_per_req=[[5]],
            seq_lens=[3],
            request_ids=["req-A"],
            is_new=[True],
            batch_id=0,
        ))

        # B arrives on the same block 5 with tokens [50, 51]
        tracker.build_scheduler_output(**_make_batch(
            token_ids=[50, 51],
            tokens_per_req=[2],
            blocks_per_req=[[5]],
            seq_lens=[2],
            request_ids=["req-B"],
            is_new=[True],
            batch_id=1,
        ))

        # Find B's token history. With explicit request_ids, the tracker
        # should key history by request_id (or at least not carry over A's).
        # The key used internally may vary, but the NewRequestData for B
        # must have prompt_token_ids == [50, 51] (B's tokens only).
        # We verify via the last NewRequestData emitted.
        # Re-run B's batch to inspect output.
        out_b = tracker.build_scheduler_output(**_make_batch(
            token_ids=[50, 51],
            tokens_per_req=[2],
            blocks_per_req=[[5]],
            seq_lens=[2],
            request_ids=["req-B"],
            is_new=[True],
            batch_id=2,
        ))
        new_reqs = out_b.scheduled_new_reqs
        assert len(new_reqs) == 1
        # B's prompt must NOT contain A's tokens [10, 11, 12]
        assert 10 not in new_reqs[0].prompt_token_ids
        assert new_reqs[0].prompt_token_ids == [50, 51]


# ---------------------------------------------------------------------------
# Test 2: Two new requests, same block in same batch
# ---------------------------------------------------------------------------

class TestTwoNewRequestsSameBlock:
    """Two new requests in the same batch share the same last block.

    With explicit request_ids, both must be treated as NEW (not one
    new and one cached).
    """

    def test_both_treated_as_new(self):
        tracker = SequenceTracker()

        out = tracker.build_scheduler_output(**_make_batch(
            token_ids=[10, 11, 20, 21],
            tokens_per_req=[2, 2],
            blocks_per_req=[[5], [5]],
            seq_lens=[2, 2],
            request_ids=["req-X", "req-Y"],
            is_new=[True, True],
            batch_id=0,
        ))
        assert _count_new_reqs(out) == 2, "Both X and Y must be NEW"

    def test_distinct_req_ids_assigned(self):
        """Each NewRequestData must have a distinct vLLM req_id."""
        tracker = SequenceTracker()

        out = tracker.build_scheduler_output(**_make_batch(
            token_ids=[10, 11, 20, 21],
            tokens_per_req=[2, 2],
            blocks_per_req=[[5], [5]],
            seq_lens=[2, 2],
            request_ids=["req-X", "req-Y"],
            is_new=[True, True],
            batch_id=0,
        ))
        ids = _new_req_ids(out)
        assert len(set(ids)) == 2, f"req_ids must be unique, got {ids}"

    def test_independent_token_histories(self):
        """Each request must have its own token history."""
        tracker = SequenceTracker()

        out = tracker.build_scheduler_output(**_make_batch(
            token_ids=[10, 11, 20, 21],
            tokens_per_req=[2, 2],
            blocks_per_req=[[5], [5]],
            seq_lens=[2, 2],
            request_ids=["req-X", "req-Y"],
            is_new=[True, True],
            batch_id=0,
        ))
        new_reqs = out.scheduled_new_reqs
        assert new_reqs[0].prompt_token_ids == [10, 11]
        assert new_reqs[1].prompt_token_ids == [20, 21]


# ---------------------------------------------------------------------------
# Test 3: is_new flag controls NEW vs CACHED classification
# ---------------------------------------------------------------------------

class TestIsNewFlag:
    """is_new=True forces NewRequestData, is_new=False forces CachedRequestData."""

    def test_is_new_true_yields_new_request_data(self):
        tracker = SequenceTracker()

        out = tracker.build_scheduler_output(**_make_batch(
            token_ids=[10, 11, 12],
            tokens_per_req=[3],
            blocks_per_req=[[7]],
            seq_lens=[3],
            request_ids=["req-P"],
            is_new=[True],
            batch_id=0,
        ))
        assert _count_new_reqs(out) == 1
        assert _new_req_ids(out) == ["pie-0-0"]
        assert len(_cached_req_ids(out)) == 0

    def test_is_new_false_yields_cached_request_data(self):
        """After the first fire (is_new=True), subsequent calls with
        is_new=False should produce CachedRequestData."""
        tracker = SequenceTracker()

        # First call: NEW
        tracker.build_scheduler_output(**_make_batch(
            token_ids=[10, 11, 12],
            tokens_per_req=[3],
            blocks_per_req=[[7]],
            seq_lens=[3],
            request_ids=["req-P"],
            is_new=[True],
            batch_id=0,
        ))

        # Second call: CACHED
        out2 = tracker.build_scheduler_output(**_make_batch(
            token_ids=[13],
            tokens_per_req=[1],
            blocks_per_req=[[7]],
            seq_lens=[4],
            request_ids=["req-P"],
            is_new=[False],
            batch_id=1,
        ))
        assert _count_new_reqs(out2) == 0, "is_new=False must yield CachedRequestData"
        assert "pie-0-0" in _cached_req_ids(out2)

    def test_mixed_batch_new_and_cached(self):
        """A batch with one new and one continuing request."""
        tracker = SequenceTracker()

        # Prefill request P
        tracker.build_scheduler_output(**_make_batch(
            token_ids=[10, 11],
            tokens_per_req=[2],
            blocks_per_req=[[3]],
            seq_lens=[2],
            request_ids=["req-P"],
            is_new=[True],
            batch_id=0,
        ))

        # Batch with P continuing + Q new
        out = tracker.build_scheduler_output(**_make_batch(
            token_ids=[12, 20, 21],
            tokens_per_req=[1, 2],
            blocks_per_req=[[3], [8]],
            seq_lens=[3, 2],
            request_ids=["req-P", "req-Q"],
            is_new=[False, True],
            batch_id=1,
        ))
        assert _count_new_reqs(out) == 1, "Only Q should be NEW"
        assert _new_req_ids(out) == ["pie-1-1"]
        assert "pie-0-0" in _cached_req_ids(out)


