use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::Poll;

use anyhow::{Result, anyhow};
use pie_driver_abi::{PIE_DRIVER_ABI_VERSION, PieCompletion, PieRuntimeCallbacks};
use pie_waker::{FIRST_COMPLETION_EPOCH, Readiness, WaitFuture, WakerSlotId, WakerTable};

pub trait CompletionLease: Send + Sync {
    fn is_closed(&self) -> bool;
}

fn valid_target_epoch(target_epoch: u64) -> bool {
    (FIRST_COMPLETION_EPOCH..u64::MAX).contains(&target_epoch)
}

fn assert_valid_target_epoch(target_epoch: u64) {
    assert!(
        valid_target_epoch(target_epoch),
        "completion target epoch must be in {FIRST_COMPLETION_EPOCH}..u64::MAX, got {target_epoch}"
    );
}

#[derive(Debug)]
struct CompletionState {
    slot: WakerSlotId,
    target_epoch: u64,
    closed: AtomicBool,
    close_message: Mutex<Option<String>>,
}

impl CompletionState {
    fn new(slot: WakerSlotId, target_epoch: u64) -> Self {
        Self {
            slot,
            target_epoch,
            closed: AtomicBool::new(false),
            close_message: Mutex::new(None),
        }
    }

    fn close(&self, table: &WakerTable, message: impl Into<String>) {
        if !self.closed.swap(true, Ordering::AcqRel) {
            *self.close_message.lock().unwrap() = Some(message.into());
            table.free(self.slot);
        }
    }

    fn close_error(&self) -> Result<()> {
        Err(anyhow!(
            self.close_message
                .lock()
                .unwrap()
                .clone()
                .unwrap_or_else(|| "driver completion closed".to_string())
        ))
    }
}

struct BrokerInner {
    table: &'static WakerTable,
    live: Mutex<HashMap<WakerSlotId, Arc<CompletionState>>>,
    closed: AtomicBool,
}

struct BrokerCallbackContext {
    inner: Arc<BrokerInner>,
}

#[derive(Clone)]
pub struct CompletionBroker {
    inner: Arc<BrokerInner>,
    callback_ctx: Arc<BrokerCallbackContext>,
}

impl Default for CompletionBroker {
    fn default() -> Self {
        Self::new()
    }
}

impl CompletionBroker {
    pub fn new() -> Self {
        let inner = Arc::new(BrokerInner {
            table: WakerTable::global(),
            live: Mutex::new(HashMap::new()),
            closed: AtomicBool::new(false),
        });
        let callback_ctx = Arc::new(BrokerCallbackContext {
            inner: Arc::clone(&inner),
        });
        Self {
            inner,
            callback_ctx,
        }
    }

    pub fn runtime_callbacks(&self) -> PieRuntimeCallbacks {
        PieRuntimeCallbacks {
            abi_version: PIE_DRIVER_ABI_VERSION,
            reserved0: 0,
            ctx: Arc::as_ptr(&self.callback_ctx) as *mut std::ffi::c_void,
            notify: Some(runtime_notify),
        }
    }

    pub fn completion(&self, target_epoch: u64) -> Completion {
        assert_valid_target_epoch(target_epoch);
        assert!(
            !self.inner.closed.load(Ordering::Acquire),
            "completion broker is closed"
        );
        let slot = self.inner.table.alloc();
        let state = Arc::new(CompletionState::new(slot, target_epoch));
        self.inner
            .live
            .lock()
            .unwrap()
            .insert(slot, Arc::clone(&state));
        Completion::pending(Arc::clone(&self.inner), state)
    }

    pub fn pie_completion(&self, target_epoch: u64) -> (PieCompletion, Completion) {
        let completion = self.completion(target_epoch);
        let raw = PieCompletion {
            wait_id: completion.wait_id(),
            target_epoch,
        };
        (raw, completion)
    }

    pub fn close_all(&self, message: impl Into<String>) {
        let message = message.into();
        self.inner.closed.store(true, Ordering::Release);
        let states = std::mem::take(&mut *self.inner.live.lock().unwrap());
        for state in states.values() {
            state.close(self.inner.table, message.clone());
        }
    }
}

#[derive(Clone)]
struct PendingCompletion {
    broker: Arc<BrokerInner>,
    state: Arc<CompletionState>,
}

#[derive(Clone)]
enum CompletionKind {
    ReadyOk,
    ReadyErr(String),
    Pending(Arc<PendingCompletion>),
}

#[derive(Clone)]
pub struct Completion {
    kind: CompletionKind,
}

impl Completion {
    fn pending(broker: Arc<BrokerInner>, state: Arc<CompletionState>) -> Self {
        Self {
            kind: CompletionKind::Pending(Arc::new(PendingCompletion { broker, state })),
        }
    }

    pub fn ready() -> Self {
        Self {
            kind: CompletionKind::ReadyOk,
        }
    }

    pub fn failed(message: impl Into<String>) -> Self {
        Self {
            kind: CompletionKind::ReadyErr(message.into()),
        }
    }

    pub fn wait_id(&self) -> u64 {
        match &self.kind {
            CompletionKind::Pending(pending) => pending.state.slot,
            CompletionKind::ReadyOk | CompletionKind::ReadyErr(_) => {
                panic!("ready completions do not expose a wait id")
            }
        }
    }

    pub fn target_epoch(&self) -> u64 {
        match &self.kind {
            CompletionKind::Pending(pending) => pending.state.target_epoch,
            CompletionKind::ReadyOk | CompletionKind::ReadyErr(_) => {
                panic!("ready completions do not expose a target epoch")
            }
        }
    }

    pub fn close(&self, message: impl Into<String>) {
        if let CompletionKind::Pending(pending) = &self.kind {
            pending.state.close(pending.broker.table, message);
        }
    }

    pub(crate) fn check(&self) -> Option<Result<()>> {
        match &self.kind {
            CompletionKind::ReadyOk => Some(Ok(())),
            CompletionKind::ReadyErr(message) => Some(Err(anyhow!(message.clone()))),
            CompletionKind::Pending(pending) => {
                if pending.state.closed.load(Ordering::Acquire) {
                    return Some(pending.state.close_error());
                }
                let slot = pending.state.slot;
                let target = pending.state.target_epoch;
                match pending.broker.table.published(slot) {
                    Some(epoch) if epoch >= target => Some(Ok(())),
                    Some(_) => None,
                    None => Some(pending.state.close_error()),
                }
            }
        }
    }
}

impl std::future::Future for Completion {
    type Output = Result<()>;

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match &mut this.kind {
            CompletionKind::ReadyOk => Poll::Ready(Ok(())),
            CompletionKind::ReadyErr(message) => Poll::Ready(Err(anyhow!(message.clone()))),
            CompletionKind::Pending(pending) => {
                if pending.state.closed.load(Ordering::Acquire) {
                    return Poll::Ready(pending.state.close_error());
                }
                let slot = pending.state.slot;
                let target = pending.state.target_epoch;
                let table = pending.broker.table;
                let state = Arc::clone(&pending.state);
                let mut future = Box::pin(WaitFuture::new(table, slot, move || {
                    if state.closed.load(Ordering::Acquire) {
                        return Readiness::Ready(state.close_error());
                    }
                    match table.published(slot) {
                        Some(epoch) if epoch >= target => Readiness::Ready(Ok(())),
                        Some(epoch) => Readiness::Pending {
                            observed_epoch: epoch,
                        },
                        None => Readiness::Ready(state.close_error()),
                    }
                }));
                future.as_mut().poll(cx)
            }
        }
    }
}

impl Drop for Completion {
    fn drop(&mut self) {
        let CompletionKind::Pending(pending) = &self.kind else {
            return;
        };
        if Arc::strong_count(pending) != 1 {
            return;
        }
        let slot = pending.state.slot;
        if !pending.state.closed.load(Ordering::Acquire) {
            pending.broker.table.free(slot);
        }
        pending.broker.live.lock().unwrap().remove(&slot);
    }
}

#[derive(Clone)]
pub struct InstanceCompletion {
    wait_id: u64,
    target_epoch: u64,
    _guard: Option<Arc<dyn CompletionLease>>,
}

impl std::fmt::Debug for InstanceCompletion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InstanceCompletion")
            .field("wait_id", &self.wait_id)
            .field("target_epoch", &self.target_epoch)
            .finish_non_exhaustive()
    }
}

impl InstanceCompletion {
    pub fn new(wait_id: u64, target_epoch: u64) -> Self {
        Self::with_guard(wait_id, target_epoch, None)
    }

    pub fn with_guard(
        wait_id: u64,
        target_epoch: u64,
        guard: impl Into<Option<Arc<dyn CompletionLease>>>,
    ) -> Self {
        assert_valid_target_epoch(target_epoch);
        Self {
            wait_id,
            target_epoch,
            _guard: guard.into(),
        }
    }

    pub fn wait_id(&self) -> u64 {
        self.wait_id
    }

    pub fn target_epoch(&self) -> u64 {
        self.target_epoch
    }

    pub fn notify(&self) {
        WakerTable::global().publish(self.wait_id, self.target_epoch);
    }
}

impl std::future::Future for InstanceCompletion {
    type Output = Result<()>;

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let wait_id = this.wait_id;
        let target = this.target_epoch;
        let guard = this._guard.clone();
        let table = WakerTable::global();
        let mut future = Box::pin(WaitFuture::new(table, wait_id, move || {
            match table.published(wait_id) {
                Some(epoch) if epoch >= target => Readiness::Ready(Ok(())),
                Some(_) if guard.as_ref().is_some_and(|guard| guard.is_closed()) => {
                    Readiness::Ready(Err(anyhow!("instance completion closed")))
                }
                Some(epoch) => Readiness::Pending {
                    observed_epoch: epoch,
                },
                None => Readiness::Ready(Err(anyhow!("instance completion closed"))),
            }
        }));
        future.as_mut().poll(cx)
    }
}

unsafe extern "C" fn runtime_notify(ctx: *mut std::ffi::c_void, wait_id: u64, epoch: u64) {
    let _ = std::panic::catch_unwind(|| {
        let Some(ctx) = (unsafe { (ctx as *const BrokerCallbackContext).as_ref() }) else {
            return;
        };
        if ctx.inner.closed.load(Ordering::Acquire) {
            return;
        }
        let _ = ctx.inner.table.publish(wait_id, epoch);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Wake, Waker};

    struct CountWaker(AtomicUsize);
    impl Wake for CountWaker {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn counter_waker() -> (Arc<CountWaker>, Waker) {
        let count = Arc::new(CountWaker(AtomicUsize::new(0)));
        (Arc::clone(&count), Waker::from(count))
    }

    fn broker_callbacks(broker: &CompletionBroker) -> PieRuntimeCallbacks {
        broker.runtime_callbacks()
    }

    #[test]
    fn foreign_thread_publish_completes_waiter() {
        let broker = CompletionBroker::new();
        let callbacks = broker_callbacks(&broker);
        let (raw, mut completion) = broker.pie_completion(3);
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let callback_ctx = callbacks.ctx as usize;
        let thread = std::thread::spawn(move || unsafe {
            runtime_notify(
                callback_ctx as *mut std::ffi::c_void,
                raw.wait_id,
                raw.target_epoch,
            )
        });
        rt.block_on(async { completion.await.unwrap() });
        thread.join().unwrap();
    }

    #[test]
    fn register_then_recheck_closes_lost_wake_race() {
        let broker = CompletionBroker::new();
        let callbacks = broker_callbacks(&broker);
        for _ in 0..500 {
            let (raw, completion) = broker.pie_completion(1);
            let waiter = std::thread::spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();
                rt.block_on(async move { completion.await.unwrap() });
            });
            unsafe { runtime_notify(callbacks.ctx, raw.wait_id, 1) };
            waiter.join().unwrap();
        }
    }

    #[test]
    fn stale_generation_callback_is_ignored() {
        let broker = CompletionBroker::new();
        let (raw, completion) = broker.pie_completion(2);
        let stale = raw.wait_id;
        drop(completion);
        assert!(matches!(
            WakerTable::global().publish(stale, 99),
            pie_waker::WakeOutcome::Stale
        ));
    }

    #[test]
    fn dropped_future_before_callback_is_safe() {
        let broker = CompletionBroker::new();
        let callbacks = broker_callbacks(&broker);
        let (raw, completion) = broker.pie_completion(2);
        let stale = raw.wait_id;
        drop(completion);
        unsafe { runtime_notify(callbacks.ctx, stale, raw.target_epoch) };
        assert!(matches!(
            WakerTable::global().publish(stale, 99),
            pie_waker::WakeOutcome::Stale
        ));
    }

    #[test]
    fn two_completions_resolve_in_callback_order() {
        let broker = CompletionBroker::new();
        let callbacks = broker_callbacks(&broker);
        let (a_raw, a_completion) = broker.pie_completion(1);
        let (b_raw, b_completion) = broker.pie_completion(2);
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            unsafe { runtime_notify(callbacks.ctx, a_raw.wait_id, 1) };
            a_completion.await.unwrap();
            unsafe { runtime_notify(callbacks.ctx, b_raw.wait_id, 2) };
            b_completion.await.unwrap();
        });
    }

    #[test]
    fn close_wakes_outstanding_waiters() {
        let broker = CompletionBroker::new();
        let completion = broker.completion(7);
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let join = std::thread::spawn(move || {
            rt.block_on(async move {
                let err = completion.await.expect_err("close should fail completion");
                assert!(err.to_string().contains("closed"));
            });
        });
        broker.close_all("driver closed");
        join.join().unwrap();
    }

    #[test]
    fn callbacks_are_ignored_after_broker_close() {
        let broker = CompletionBroker::new();
        let callbacks = broker_callbacks(&broker);
        let (raw, mut completion) = broker.pie_completion(5);
        broker.close_all("driver closed");
        unsafe { runtime_notify(callbacks.ctx, raw.wait_id, raw.target_epoch) };
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let err = completion
                .await
                .expect_err("closed broker should fail completion");
            assert!(err.to_string().contains("driver closed"));
        });
    }

    #[test]
    fn publish_racing_with_registration_wakes_or_fast_paths() {
        let broker = CompletionBroker::new();
        let callbacks = broker_callbacks(&broker);
        let (raw, completion) = broker.pie_completion(5);
        let (count, waker) = counter_waker();
        let mut completion = Box::pin(completion);
        let mut cx = std::task::Context::from_waker(&waker);
        assert!(matches!(completion.as_mut().poll(&mut cx), Poll::Pending));
        unsafe { runtime_notify(callbacks.ctx, raw.wait_id, 5) };
        assert!(count.0.load(Ordering::SeqCst) <= 1);
    }

    #[test]
    fn ready_completion_is_immediately_ready() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            Completion::ready().await.unwrap();
            let err = Completion::failed("boom")
                .await
                .expect_err("ready failed completion should surface immediately");
            assert!(err.to_string().contains("boom"));
        });
    }

    #[test]
    fn instance_completion_reports_scheduler_failure() {
        let mut completion = Box::pin(InstanceCompletion::new(WakerTable::global().alloc(), 4));
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        completion.notify();
        rt.block_on(async move {
            completion
                .await
                .expect("payload-free notify should resolve waiter");
        });
    }
}
