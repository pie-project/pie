//! # X0 — the tensor-waker substrate (Runtime–Driver Boundary, B9–B12)
//!
//! Host-side dual of contract **C2**: futures park on the same ring indices
//! the driver cuts on, woken by the driver (or mock) through this table. Owns
//! every [`Waker`]; the far side of the boundary sees only opaque `u64` ids.
//!
//! - **B9**: a waiter registers `(waker, observed_epoch)`; the committer wakes
//!   it once the ring index *passes* that epoch ([`WakerTable::wake_past`]).
//!   The register/commit race is closed by register-then-recheck — callers
//!   MUST follow the protocol on [`WakerTable::register`] ([`WaitFuture`] does).
//! - **B10**: C++ only ever holds an opaque, generation-tagged `u64`
//!   ([`pie_wake`]/[`pie_wake_past`]), so a stale id after a channel dies is a no-op.
//! - **B12**: `sweep` wakes every slot of a poisoned/closed/aborted channel
//!   unconditionally, so a blocked `take().await?` resolves to `Err`, not a hang.
//!
//! Spurious wakes are permitted; mock and real driver share the `pie_wake*` FFI.

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
