use crate::table::{WakerSlotId, WakerTable};

pub enum Readiness<T> {
    Ready(T),
    Pending { observed_epoch: u64 },
}

pub struct WaitFuture<'t, F> {
    table: &'t WakerTable,
    slot: WakerSlotId,
    check: F,
    /// The waker this future last registered on the slot, which it alone
    /// takes back: other tasks may wait on the same slot.
    registered: Option<std::task::Waker>,
}

impl<'t, F, T> WaitFuture<'t, F>
where
    F: FnMut() -> Readiness<T> + Unpin,
{
    pub fn new(table: &'t WakerTable, slot: WakerSlotId, check: F) -> Self {
        WaitFuture {
            table,
            slot,
            check,
            registered: None,
        }
    }
}

impl<'t, F, T> std::future::Future for WaitFuture<'t, F>
where
    F: FnMut() -> Readiness<T> + Unpin,
{
    type Output = T;

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<T> {
        let this = self.get_mut();
        let observed = match (this.check)() {
            Readiness::Ready(v) => {
                this.release();
                return std::task::Poll::Ready(v);
            }
            Readiness::Pending { observed_epoch } => observed_epoch,
        };
        if !this.table.register(this.slot, cx.waker(), observed) {
            cx.waker().wake_by_ref();
            return std::task::Poll::Pending;
        }
        this.registered = Some(cx.waker().clone());
        match (this.check)() {
            Readiness::Ready(v) => {
                this.release();
                std::task::Poll::Ready(v)
            }
            Readiness::Pending { .. } => std::task::Poll::Pending,
        }
    }
}

impl<F> WaitFuture<'_, F> {
    /// Takes back this future's registration. The slot keeps one entry a
    /// task, so another future of the same task may have been relying on
    /// it: the task is woken to poll that one again (at worst a spurious
    /// poll).
    fn release(&mut self) {
        if let Some(waker) = self.registered.take()
            && self.table.deregister_waker(self.slot, &waker)
        {
            waker.wake();
        }
    }
}

impl<F> Drop for WaitFuture<'_, F> {
    fn drop(&mut self) {
        self.release();
    }
}
