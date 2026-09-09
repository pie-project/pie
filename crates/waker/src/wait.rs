use crate::table::{WakerSlotId, WakerTable};

pub enum Readiness<T> {
    Ready(T),
    Pending { observed_epoch: u64 },
}

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
        let observed = match (this.check)() {
            Readiness::Ready(v) => return std::task::Poll::Ready(v),
            Readiness::Pending { observed_epoch } => observed_epoch,
        };
        if !this.table.register(this.slot, cx.waker(), observed) {
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
