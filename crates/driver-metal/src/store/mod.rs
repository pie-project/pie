//! The KV and recurrent-state stores' portable half.
//!
//! What lives here needs no device: the paged pool's move arithmetic
//! ([`plan_cell_moves`]) and the GDN slots' step/parity bookkeeping
//! ([`LinearStateSlots`]). The pool itself — per-layer K/V buffers, the
//! elastic commit — is Metal state and stays under `src/metal/`. The
//! ledger is `.wiki/driver/progress-metal.md`.

mod control;
mod kv_move;
mod linear;

pub use control::{
    Capabilities, KvCopyWork, Pool, Refusal, Resize, plan_kv_copy, plan_pool_resize,
    plan_state_copy,
};
pub use kv_move::{CellCopy, CellMovePlan, CellOutOfRange, KvMoveCell, PoolGrid, plan_cell_moves};
pub use linear::{LinearStateSlots, Parity, WildSlot};
