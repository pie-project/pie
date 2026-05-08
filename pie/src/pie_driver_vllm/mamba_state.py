"""Per-request mamba state-slot allocator (ticket #108 phase 5a).

Hybrid Transformer+Mamba models (Qwen3-Next, Qwen3.5-MoE) need one
recurrent-state slot per in-flight request, sized to a fixed pool whose
dim-0 length matches the attention KV pool (`num_blocks`). The slot is
written by GDN's `_forward` via `non_spec_state_indices_tensor`, indexed
into `conv_state` / `ssm_state` tensors held in
`engine.mamba_layers[i][1:]`.

Phase 4 (#107) keyed the slot off the request's first KV page id. That
collides under fork: post-fork siblings share the committed prefix's
first page (refcount), so two siblings in the same fire_batch derive the
same slot id and corrupt each other's recurrent state. The phase 4 driver
ships with `supports_kv_fork=False` on hybrid + an in-batch duplicate-
slot guard to fail loud if a fork-using inferlet leaks through.

This allocator replaces the page-id-keyed scheme with a per-context
allocator, keyed on the stable `ContextId` already supplied per row in
`BatchedForwardPassRequest.context_ids` (runtime/src/inference/request.rs:431,
shmem A_CONTEXT_IDS = 27). pie's runtime mints a fresh `ContextId` on
fork (`runtime/src/context/snapshot.rs:139`), so siblings reach the
driver with **distinct** ids — no alias-by-construction, regardless of
how their committed prefixes overlap.

Phase 5a alone does not enable fork: a fresh child appears with a fresh
ContextId → fresh slot → zero recurrent state, and the GDN forward reads
zeros where the parent's post-prefix state should live. Phase 5b adds an
explicit copy-on-fork hook (`Engine.mamba_fork(parent, child)`) wired to
a sibling shmem op from pie's runtime; once both phases land,
`supports_kv_fork` flips to True on hybrid.

## Eviction policy

ContextIds monotonically increase over the worker's lifetime, so the
naïve `dict[ContextId, slot]` grows unboundedly. We bound it via LRU:
when a request comes in for an unseen ContextId and the free pool is
empty, the least-recently-used (ctx_id, slot) is reclaimed and reused.

In practice the pool is sized to thousands of blocks while the scheduler
caps in-flight requests around `max_num_seqs` (≤ 16), so eviction is
effectively dead code. We still implement it to avoid memory growth in
long-lived workers and to handle the (unlikely) case of an inferlet that
churns short-lived contexts faster than the pool can rotate.

## Eviction safety on hybrid

When a slot is reused, the conv/ssm state tensors at that slot index
still hold the previous owner's recurrent state. The GDN forward reads
them only when `has_initial_state[row] = True`, i.e. when
`num_computed_tokens > 0`. A fresh ContextId entering decode for the
first time has num_computed_tokens=0 (its prefill batch sets the slot
contents on the first forward) → the stale bytes are overwritten before
being read. We therefore do NOT zero the slot on eviction; the kernel's
write-then-read order is sufficient. If a future scheduling change ever
admits a request with `num_computed_tokens > 0` on its very first
appearance to the driver (e.g. resume-from-checkpoint), this contract
breaks and we must zero on alloc.
"""

from __future__ import annotations

from collections import OrderedDict


class MambaSlotAllocator:
    """LRU-backed `ContextId -> slot_idx` map sized to `num_blocks`.

    Single-threaded by construction: pie_driver_vllm runs `fire_batch`
    on one Python thread per worker (the shmem reader). No locking.
    """

    __slots__ = ("_capacity", "_alive", "_free")

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError(
                f"MambaSlotAllocator: num_blocks must be > 0; got {num_blocks}"
            )
        self._capacity = int(num_blocks)
        # OrderedDict: insertion order = recency; move_to_end on hit.
        self._alive: OrderedDict[int, int] = OrderedDict()
        # Free pool, populated lazily. We start with all slots free.
        self._free: list[int] = list(range(self._capacity))

    def get_or_alloc(self, ctx_id: int) -> int:
        """Return the slot for `ctx_id`, allocating if first seen.

        Touching a known ContextId moves it to MRU. Allocating may evict
        the oldest entry if the free pool is empty.
        """
        # `int()` defends against numpy / torch scalars sneaking in.
        key = int(ctx_id)
        slot = self._alive.get(key)
        if slot is not None:
            self._alive.move_to_end(key)
            return slot
        if not self._free:
            # Evict LRU. `popitem(last=False)` returns oldest (FIFO end).
            _victim_id, victim_slot = self._alive.popitem(last=False)
            self._free.append(victim_slot)
        slot = self._free.pop()
        self._alive[key] = slot
        return slot

    def release(self, ctx_id: int) -> None:
        """Free a slot when the runtime tells us a context is gone.

        Phase 5a does not wire a release hook end-to-end — kept here for
        Phase 5b, when the runtime gains a per-context death notification.
        Idempotent: silently no-ops on unknown ids.
        """
        key = int(ctx_id)
        slot = self._alive.pop(key, None)
        if slot is not None:
            self._free.append(slot)

    def slots_for(self, ctx_ids) -> list[int]:
        """Vectorized `get_or_alloc` over a per-batch sequence of ContextIds."""
        return [self.get_or_alloc(c) for c in ctx_ids]

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def in_use(self) -> int:
        return len(self._alive)


def fork_mamba_slot(
    parent_ctx_id: int,
    child_ctx_id: int,
    mamba_layers,
    allocator: "MambaSlotAllocator",
) -> tuple[int, int]:
    """Copy parent's per-layer mamba state into the child's slot (#108 phase 5b).

    Called by the worker's shmem dispatch when the runtime emits a
    `mamba_fork(parent, child)` notification after `ctx.fork()` (see
    `runtime/src/context/snapshot.rs::fork`). Routed on the same Python
    thread as `fire_batch` to preserve CUDA stream FIFO order against
    the child's first forward pass.

    `mamba_layers` is `engine.mamba_layers` — the list
    `[(layer_name, conv_state, ssm_state), ...]` from
    `kv_cache.allocate_and_bind_kv_cache`. Empty on non-hybrid models;
    function returns early.

    Allocator is the engine's `MambaSlotAllocator`. We **must** use the
    allocator's existing slot for the parent (it was set when the
    parent's most recent fire_batch staged the GDN metadata). For the
    child we ALWAYS allocate a fresh slot via `get_or_alloc` rather than
    blindly trusting that the child's first batch hasn't arrived yet —
    in the synchronous shmem path the runtime awaits this op before
    `Ok(new_id)` returns to the inferlet, so the child cannot have
    appeared in any prior fire_batch yet, but that ordering is a
    runtime invariant; we don't depend on it here.

    Returns `(parent_slot, child_slot)` for logging / test assertions.
    """
    if not mamba_layers:
        # Non-hybrid driver / pure-attention model: no recurrent state
        # to migrate. Cheap fast path; called on every fork even on
        # non-hybrid since the runtime emits the op unconditionally.
        return (-1, -1)

    parent_slot = allocator.get_or_alloc(parent_ctx_id)
    child_slot = allocator.get_or_alloc(child_ctx_id)
    if parent_slot == child_slot:
        # Should never happen — distinct ContextIds always map to
        # distinct slots under MambaSlotAllocator's invariants. Fail
        # loud so a future allocator regression doesn't silently
        # alias parent and child here.
        raise RuntimeError(
            f"fork_mamba_slot: parent and child mapped to the same slot "
            f"({parent_slot}); parent_ctx={parent_ctx_id} "
            f"child_ctx={child_ctx_id}. Allocator invariant violation."
        )

    for _layer_name, conv_state, ssm_state in mamba_layers:
        # `conv_state` and `ssm_state` shape: (num_blocks, *spec.shapes[i]).
        # Slot ids index dim-0. Per-layer cost: two contiguous slice
        # copies, no host sync; runs on the calling Python thread
        # (worker shmem dispatcher), which is the same thread that
        # later issues the child's fire_batch.
        conv_state[child_slot].copy_(conv_state[parent_slot])
        ssm_state[child_slot].copy_(ssm_state[parent_slot])
    return (parent_slot, child_slot)
