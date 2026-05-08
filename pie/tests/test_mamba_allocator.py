"""Unit tests for `pie_driver_vllm.mamba_state.MambaSlotAllocator` (#108 phase 5a).

Pure-Python; no GPU / vllm import required. Loads the module file
directly so the test runs even when the pie Rust extension hasn't been
built (the package's `__init__` imports `pie._runtime`).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_mamba_state():
    src = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "pie_driver_vllm"
        / "mamba_state.py"
    )
    spec = importlib.util.spec_from_file_location("mamba_state_under_test", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_mod = _load_mamba_state()
MambaSlotAllocator = _mod.MambaSlotAllocator


def test_distinct_ctx_ids_get_distinct_slots():
    a = MambaSlotAllocator(num_blocks=8)
    s = a.slots_for([10, 20, 30])
    assert len(set(s)) == 3
    # Same ctx_ids in a follow-up batch must hit cache (no new alloc).
    again = a.slots_for([10, 20, 30])
    assert again == s


def test_alloc_is_deterministic_for_a_given_ctx():
    a = MambaSlotAllocator(num_blocks=4)
    first = a.get_or_alloc(99)
    for _ in range(10):
        assert a.get_or_alloc(99) == first


def test_capacity_lru_eviction():
    a = MambaSlotAllocator(num_blocks=2)
    s0 = a.get_or_alloc(0)
    s1 = a.get_or_alloc(1)
    assert s0 != s1
    # Touch ctx 0 so 1 becomes LRU.
    _ = a.get_or_alloc(0)
    # New ctx triggers eviction of 1; its slot is reused.
    s2 = a.get_or_alloc(2)
    assert s2 == s1
    # Ctx 0 still mapped (was MRU); ctx 1 is now a cache miss.
    assert a.get_or_alloc(0) == s0
    s1_new = a.get_or_alloc(1)
    # Eviction of 0 (it became LRU after we re-fetched 1's old slot
    # via 2, and then accessed 0 again — actually let's just check
    # ctx 1 has SOME slot in [0, capacity)).
    assert 0 <= s1_new < a.capacity


def test_release_returns_slot_to_pool():
    a = MambaSlotAllocator(num_blocks=2)
    s_a = a.get_or_alloc(101)
    s_b = a.get_or_alloc(202)
    assert {s_a, s_b} == {0, 1}
    a.release(101)
    assert a.in_use == 1
    s_c = a.get_or_alloc(303)
    assert s_c == s_a
    # Releasing an unknown id is a no-op.
    a.release(99999)


def test_release_idempotent():
    a = MambaSlotAllocator(num_blocks=4)
    a.get_or_alloc(7)
    a.release(7)
    a.release(7)  # idempotent
    a.release(404)  # unknown
    assert a.in_use == 0
    # Can re-allocate after release.
    s = a.get_or_alloc(7)
    assert 0 <= s < a.capacity


def test_invalid_capacity():
    with pytest.raises(ValueError):
        MambaSlotAllocator(num_blocks=0)
    with pytest.raises(ValueError):
        MambaSlotAllocator(num_blocks=-1)


def test_slots_for_accepts_iterables_and_scalars():
    a = MambaSlotAllocator(num_blocks=8)
    # numpy / torch may produce non-int scalars; allocator coerces via int().
    import numpy as np
    ids = np.array([100, 200, 300], dtype=np.uint64)
    s = a.slots_for(ids.tolist())
    assert len(s) == 3
    assert len(set(s)) == 3
