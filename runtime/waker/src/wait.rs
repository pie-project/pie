//! The waiting future — register-then-recheck (B9) encoded so callers cannot
//! get the race wrong. Tolerates spurious wakes; resolves after a B12 sweep.

use crate::table::{WakerSlotId, WakerTable};

/// One observation of the waiter's condition.
pub enum Readiness<T> {
    Ready(T),
    /// Not ready; `observed_epoch` is the ring index the check read (what
    /// the eventual commit must pass).
    Pending {
        observed_epoch: u64,
    },
}

/// A future that parks on `slot` until `check` returns [`Readiness::Ready`].
/// Encodes register-then-recheck, tolerates spurious wakes, and resolves
/// (via `check` observing poison and returning `Ready(Err(..))`-shaped
/// values) after a B12 sweep.
pub struct WaitFuture<'t, F> {
    table: &'t WakerTable,
    slot: WakerSlotId,
    check: F,
}

impl<'t, F, T> WaitFuture<'t, F>
where
    F: FnMut() -> Readiness<T> + Unpin,
{
    pub fn new(table: &'t WakerTable, slot: WakerSlotId, check: F) -> Self {
        WaitFuture { table, slot, check }
    }
}

impl<'t, F, T> std::future::Future for WaitFuture<'t, F>
where
    F: FnMut() -> Readiness<T> + Unpin,
{
    type Output = T;

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<T> {
        let this = self.get_mut();
        // Fast path.
        let observed = match (this.check)() {
            Readiness::Ready(v) => return std::task::Poll::Ready(v),
            Readiness::Pending { observed_epoch } => observed_epoch,
        };
        // Publish the waker, then MANDATORY re-check (see `register` docs).
        if !this.table.register(this.slot, cx.waker(), observed) {
            // Stale slot: the channel died between checks — one more check
            // must surface the failure; poll again immediately.
            cx.waker().wake_by_ref();
            return std::task::Poll::Pending;
        }
        match (this.check)() {
            Readiness::Ready(v) => {
                this.table.deregister(this.slot);
                std::task::Poll::Ready(v)
            }
            Readiness::Pending { .. } => std::task::Poll::Pending,
        }
    }
}

impl<F> Drop for WaitFuture<'_, F> {
    fn drop(&mut self) {
        self.table.deregister(self.slot);
    }
}
