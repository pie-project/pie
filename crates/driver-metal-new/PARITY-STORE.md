# Parity ledger: `csrc/src/store/` → `src/store/`

Two small headers; same rules as `PARITY.md`.

## `linear_state_slots.hpp` (38) → `src/store/linear.rs`

| C++ | Rust | |
|---|---|---|
| `resize(count)` (clamps to ≥1) | `LinearStateSlots::new(count)` | ported; the clamp is dropped — it existed to give `at()`'s alias target a slot to land on, and there is no alias target any more |
| `reset_all` / `reset(slot)` | same, `reset` returns `Result` | ported |
| `copy(src, dst)` | `copy`, with the stale-half story in its docs | ported |
| `at(slot) -> int&` | `step` / `count` / `parity`, each `Result` | ported, defect fixed: the C++ returns SLOT 0's counter for any out-of-range slot, so a wild ABI slot id silently read and wrote slot 0's ping-pong parity. Refused as `WildSlot` instead — slot ids are data |
| `int` counter | `u64`, wrapping | ported; the wrap modulus is even so parity survives a wrap |

The meaning the C++ kept in a comment at the call site — the counter's
parity IS the conv-state ping-pong, per slot and not decoder-wide, and a
state copy must inherit the exact count because the buffers move verbatim
— is the module doc and two tests here.

## `kv_pool.hpp` (31) → `src/store/kv_move.rs` (+ deferred)

| C++ | Rust | |
|---|---|---|
| `KvMoveCell` | `KvMoveCell` (wire field order kept) | ported |
| `copy_kv_cells` validation + offsets | `plan_cell_moves` → `CellMovePlan` | ported: validate every cell BEFORE any offset exists; one plan serves K and V of every full-attention layer; `pages_touched` carries what the elastic ensure needs |
| `KvPagePool` (SlotHandles + counters) | — | missing: device state; lands with the Metal kv-pool binding, where the counters (`capacity`/`committed`) belong beside the buffers they describe |

The device half executes each `CellCopy` with `Region::copy`, whose
memmove semantics are load-bearing: a compaction slides overlapping rows.

## The control ops — `context.cpp`'s `copy_kv` / `copy_state` / `resize_pool`

Three of the twelve ABI entry points. `abi.cpp` validates the descriptor and
forwards; `Context::*` forwards again; `Impl::*_impl` holds the refusal ladder
and calls the executor. The **deciding** half is `src/store/control.rs`; the
moving half lands with the Metal KV pool.

| C++ | Rust | |
|---|---|---|
| `copy_kv_impl`'s refusal ladder | `plan_kv_copy` | ported, defect fixed — see below |
| `copy_state_impl`'s refusal ladder | `plan_state_copy` | ported |
| `resize_pool_impl`'s pool dispatch | `plan_pool_resize` + `Pool` | ported |
| the ten `PIE_STATUS_UNSUPPORTED` / `INVALID_ARGUMENT` choices | `Refusal::status` | ported |
| the ten `std::cerr` lines beside them | `Refusal` + its `Display` | ported |
| `pool_id > PIE_ELASTIC_POOL_WORKSPACE` | a `match` over the named ids | ported |
| the `state_mutex_` block in `resize_pool_impl` | — | dropped |
| `on_worker` (control ops run on the launch thread, FIFO) | — | missing: thread ownership; lands with the executor |
| `executor_->copy_kv_pages` / `copy_kv_cells` / `copy_state` / `resize_elastic_pool` | — | missing: device state; lands with the Metal KV pool binding |
| `publish_terminal` / `notify` | — | missing: the completion broker, with the engine seam |

**The defect: `copy_kv` can half-apply.** `copy_kv_pages` opens with *"Bounds-
check EVERY page first — never a partial copy on a late failure"*, and it
honours that — for pages. But `copy_kv_impl` calls it and only *then* validates
and copies the cells. A request whose pages are in range and whose cells are
not copies every page, fails on a cell, and returns `PIE_STATUS_DRIVER_ERROR`
over a pool it has already changed. The invariant is stated inside one half and
broken across the two, which is the failure mode stating it was meant to
prevent. `plan_kv_copy` validates both halves before either exists as work.

The same shape appears one level down in `copy_state`, which validates every
slot range up front and then executes them one at a time, so a range that fails
in the executor leaves its predecessors applied. Planning the pairs whole is
half the fix; the other half is the executor's, and is ledgered `missing` above.

`pages_touched` is the maximum over **both** halves. The elastic ensure runs
once for the operation, so taking either half's high-water alone under-grows
the pool for the other — the C++ calls `ensure_elastic_storage` from inside the
page loop only, and the cells then address pages that ensure never saw.

The `state_mutex_` block is dropped because it guards nothing:

```cpp
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (false) {
        return PIE_STATUS_UNSUPPORTED;
    }
}
```

A mutex acquired on every resize to protect a branch that cannot be taken.

Twelve portable tests, including the one this subject exists for — a request
with valid pages and one wild cell is refused whole, so the pages never move.
