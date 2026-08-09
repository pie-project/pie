//! The one `#[cfg]`: everything that names a CUDA symbol.
//!
//! Above this directory the crate answers questions a card does not
//! change — geometry, budgets, plans, lowering. Below it, vendor
//! vocabulary is allowed and a device is assumed.
//!
//! # Why a directory rather than a per-file gate
//!
//! The gate used to run through `store/`, `model/` and `tensor.rs`
//! file by file. That is discipline, not structure: it holds exactly
//! until someone adds a file, and the cost of it slipping was measured
//! — `memory_planner`, `mla_geometry` and `dsv4_geometry` sat
//! parity-tested with zero callers for months, because the only builds
//! that could have noticed were builds that needed a GPU.
//!
//! # The layers, innermost first
//!
//! - [`device`] — the ONLY place vendor words are correct: stream,
//!   event, heap, allocator, graph. Read alongside CUDA's own
//!   documentation. Above it the crate uses one word per concept,
//!   because that is where a reader crosses between this shell and the
//!   Metal one.
//! - [`weights`] — the checkpoint onto the device. The PLAN half of
//!   that is [`crate::layout`]'s.
//! - [`pools`] — what `layout` planned, allocated: KV, recurrent, swap.
//! - [`bind`] — a lowered launch onto a kernel entry, its arguments and
//!   its grid. Mostly generated from the kernel table.
//! - [`fire`] — one forward pass: its scratch, its tables, its
//!   recordings, its retirement.
//! - [`program`] — user programs: compile, cache, channel, run.
//! - [`serve`] — the door. create / load / launch / transfer / close.

pub mod bind;
pub mod device;
pub mod fire;
pub mod pools;
pub mod program;
#[cfg(feature = "abi")]
pub mod serve;
#[cfg(feature = "abi")]
pub mod weights;
