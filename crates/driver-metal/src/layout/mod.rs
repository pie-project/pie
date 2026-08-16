//! How big, where, how many — and no device to answer it with.
//!
//! Everything here is arithmetic over integers and text. It compiles and
//! tests on a host with no GPU, which is the property `src/lib.rs` calls the
//! crate's reason for being split at all, and it is the half
//! `.wiki/driver/real-metal-north-star.md` §6 puts above the one `#[cfg]`.
//!
//! The cut is not "is this about the GPU" — [`tuning`] is entirely about the
//! GPU and its inputs are two integers. The cut is *does answering this need
//! a device*.
//!
//! * [`bump`] — a bump allocator over an offset range.
//! * [`kv`] — the paged KV pool's SHAPE. `gpu::pools::kv` allocates it.
//! * [`kv_move`] — the page-major move plan a KV copy runs, one plan for
//!   every buffer of every layer.
//! * [`linear`] — the recurrent state slots' step/parity bookkeeping.
//! * [`recurrent`] — the recurrent stack's slab SHAPE. `pools::recurrent`
//!   allocates it.
//! * [`region`] — the trait a device buffer implements so a planner can
//!   name a sub-range of it without naming Metal.
//! * [`shader`] — reading kernel text and stating what a batch of them is.
//! * [`tuning`] — which occupancy a device wants, from its two numbers.

pub mod bump;
pub mod kv;
pub mod kv_move;
pub mod linear;
pub mod recurrent;
pub mod region;
pub mod shader;
pub mod tuning;

pub use kv_move::{CellCopy, CellMovePlan, CellOutOfRange, KvMoveCell, PoolGrid, plan_cell_moves};
pub use linear::{LinearStateSlots, Parity, WildSlot};
pub use region::Region;
pub use shader::{Batch, Request};
