//! The tensor-waker substrate: futures park on the same ring indices the
//! engine cuts on, woken by the engine (or mock) through this table. Owns
//! every [`Waker`]; the far side sees only opaque, generation-tagged `u64`
//! ids, so a stale id after a channel dies is a no-op. A waiter registers
//! `(waker, observed_epoch)`; the committer wakes it once the ring index
//! passes that epoch ([`WakerTable::wake_past`]) — callers must follow the
//! register-then-recheck protocol on [`WakerTable::register`] (as
//! [`WaitFuture`] does). `sweep` wakes every slot of a poisoned/closed/
//! aborted channel unconditionally, so a blocked `take().await?` resolves to
//! `Err`, not a hang. Spurious wakes are permitted.

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
