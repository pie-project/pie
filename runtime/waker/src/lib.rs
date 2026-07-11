//! # X0 — the tensor-waker substrate (Runtime–Driver Boundary, B9–B12)
//!
//! The host-side dual of contract **C2**: the driver waits on words at device
//! cut points; the *host* parks futures on the same ring indices and is woken
//! by the driver (or the mock) through this table. This module is the whole
//! foundation X1–X4 build on: it owns every [`Waker`], hands the other side
//! of the boundary nothing but opaque `u64` slot ids, and closes the
//! register/commit race without a lock shared across the boundary.
//!
//! ## Locked decisions realized here
//!
//! - **B9 — epoch-tagged registration.** A waiter reads the channel's ring
//!   index (head or tail — whichever its condition watches), and registers
//!   `(waker, observed_epoch)`. The committer wakes when the ring index
//!   *passes* the registered epoch ([`WakerTable::wake_past`]). The race
//!   (commit lands between the waiter's observation and its registration) is
//!   closed by the **register-then-recheck protocol**: `register` publishes
//!   the waker *first*, then the caller re-checks its condition — either the
//!   committer sees the published waker, or the re-check sees the committed
//!   index. [`WaitFuture`] encodes the protocol so callers cannot get it
//!   wrong; hand-rolled pollers MUST follow it (documented on
//!   [`WakerTable::register`]).
//! - **B10 — C++ never holds a `Waker`.** The FFI surface is
//!   [`pie_wake`]/[`pie_wake_past`]: opaque `u64` in, `0/1` out, callable
//!   from any thread, never unwinds. All waker memory lives in this table.
//!   Slots are **generation-tagged** (id = `generation << 32 | index`), so a
//!   stale id held by C++ after a channel died is a harmless no-op.
//! - **SPSC ⇒ two fixed slots per host-visible channel** (one reader-waiter,
//!   one writer-waiter — [`ChannelWakers`]): no waiter lists, no thundering
//!   herd, O(1) memory per channel.
//! - **B12 — sweep on poison/close/abort.** [`WakerTable::sweep`] /
//!   [`ChannelWakers::sweep`] wake every registered slot of the touched
//!   channels unconditionally (ignoring epochs), so a blocked
//!   `take().await?` re-polls, observes the poison, and resolves to `Err` —
//!   it never hangs.
//!
//! Spurious wakes are permitted everywhere (the futures contract); the epoch
//! filter exists to keep them *rare* (the `wakes-per-fire` probe), never to
//! guarantee their absence.
//!
//! Mock-first: nothing here touches CUDA. The mock driver calls the same
//! `pie_wake*` exports the C++ driver will.

mod ffi;
mod r#loom;
mod table;
mod wait;

#[cfg(not(loom))]
pub use ffi::{pie_wake, pie_wake_past};
pub use table::{
    ChannelWakers, FIRST_COMPLETION_EPOCH, MetricsSnapshot, WakeOutcome, WakerMetrics, WakerSlotId,
    WakerTable,
};
pub use wait::{Readiness, WaitFuture};

#[cfg(all(test, loom))]
mod loom_tests;
#[cfg(all(test, not(loom)))]
mod tests;
