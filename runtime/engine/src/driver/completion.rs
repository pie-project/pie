use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::task::Poll;

use anyhow::{Result, anyhow};
use pie_driver_abi::{
    PIE_DRIVER_ABI_VERSION, PIE_TERMINAL_OUTCOME_FAILED, PIE_TERMINAL_OUTCOME_PENDING,
    PIE_TERMINAL_OUTCOME_SUCCESS, PieCompletion, PieRuntimeCallbacks, PieTerminalCell,
};
use pie_waker::{FIRST_COMPLETION_EPOCH, WakerSlotId, WakerTable};

pub trait CompletionLease: Send + Sync {
    fn is_closed(&self) -> bool;
}

fn valid_target_epoch(target_epoch: u64) -> bool {
    target_epoch == 0 || (FIRST_COMPLETION_EPOCH..u64::MAX).contains(&target_epoch)
}

fn assert_valid_target_epoch(target_epoch: u64) {
    assert!(
        valid_target_epoch(target_epoch),
        "completion target epoch must be 0 or in {FIRST_COMPLETION_EPOCH}..u64::MAX, got {target_epoch}"
    );
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TerminalOutcome {
    Pending,
    Success,
    Failed,
    Invalid(u32),
}

#[derive(Debug)]
struct TerminalCellStorage(UnsafeCell<PieTerminalCell>);

// The driver mutates `outcome` atomically through the ABI pointer. `reserved0`
// is written only before that release publication and is never read by Rust.
unsafe impl Send for TerminalCellStorage {}
unsafe impl Sync for TerminalCellStorage {}

#[derive(Debug)]
struct OwnedTerminalCell {
    raw: Option<Box<TerminalCellStorage>>,
}

fn terminal_cell_pool() -> &'static Mutex<Vec<Box<TerminalCellStorage>>> {
    static POOL: OnceLock<Mutex<Vec<Box<TerminalCellStorage>>>> = OnceLock::new();
    POOL.get_or_init(|| Mutex::new(Vec::new()))
}

impl OwnedTerminalCell {
    fn new() -> Self {
        let raw = terminal_cell_pool()
            .lock()
            .unwrap()
            .pop()
            .unwrap_or_else(|| {
                Box::new(TerminalCellStorage(UnsafeCell::new(PieTerminalCell {
                    outcome: PIE_TERMINAL_OUTCOME_PENDING,
                    reserved0: 0,
                })))
            });
        unsafe {
            *raw.0.get() = PieTerminalCell {
                outcome: PIE_TERMINAL_OUTCOME_PENDING,
                reserved0: 0,
            };
        }
        Self { raw: Some(raw) }
    }

    fn as_mut_ptr(&self) -> *mut PieTerminalCell {
        self.raw
            .as_deref()
            .expect("owned terminal cell is present")
            .0
            .get()
    }

    fn load(&self) -> TerminalOutcome {
        load_terminal_outcome(self.as_mut_ptr())
    }
}

impl Drop for OwnedTerminalCell {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            terminal_cell_pool().lock().unwrap().push(raw);
        }
    }
}

#[cfg(test)]
fn terminal_atomic_ptr(cell: *mut PieTerminalCell) -> *const AtomicU32 {
    cell.cast::<AtomicU32>()
}

fn load_terminal_outcome(cell: *mut PieTerminalCell) -> TerminalOutcome {
    let value = unsafe { AtomicU32::from_ptr(cell.cast::<u32>()).load(Ordering::Acquire) };
    match value {
        PIE_TERMINAL_OUTCOME_PENDING => TerminalOutcome::Pending,
        PIE_TERMINAL_OUTCOME_SUCCESS => TerminalOutcome::Success,
        PIE_TERMINAL_OUTCOME_FAILED => TerminalOutcome::Failed,
        other => TerminalOutcome::Invalid(other),
    }
}

#[derive(Debug)]
enum CompletionMode {
    WakeOnly,
    Terminal { cell: OwnedTerminalCell },
}

#[derive(Debug)]
struct CompletionState {
    slot: WakerSlotId,
    target_epoch: u64,
    mode: CompletionMode,
    closed: AtomicBool,
    close_message: Mutex<Option<String>>,
}

impl CompletionState {
    fn new(slot: WakerSlotId, target_epoch: u64, mode: CompletionMode) -> Self {
        Self {
            slot,
            target_epoch,
            mode,
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

    fn terminal_result(&self) -> Option<Result<()>> {
        match &self.mode {
            CompletionMode::WakeOnly => None,
            CompletionMode::Terminal { cell } => match cell.load() {
                TerminalOutcome::Pending => None,
                TerminalOutcome::Success => Some(Ok(())),
                TerminalOutcome::Failed => Some(Err(anyhow!(
                    "driver operation published Failed terminal outcome"
                ))),
                TerminalOutcome::Invalid(value) => Some(Err(anyhow!(
                    "driver operation published invalid terminal outcome {value}"
                ))),
            },
        }
    }

    fn terminal_cell_ptr(&self) -> Option<*mut PieTerminalCell> {
        match &self.mode {
            CompletionMode::WakeOnly => None,
            CompletionMode::Terminal { cell } => Some(cell.as_mut_ptr()),
        }
    }

    fn expects_terminal_outcome(&self) -> bool {
        matches!(&self.mode, CompletionMode::Terminal { .. })
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

    fn make_completion(&self, target_epoch: u64, mode: CompletionMode) -> Completion {
        assert_valid_target_epoch(target_epoch);
        assert!(
            !self.inner.closed.load(Ordering::Acquire),
            "completion broker is closed"
        );
        let slot = self.inner.table.alloc();
        let state = Arc::new(CompletionState::new(slot, target_epoch, mode));
        self.inner
            .live
            .lock()
            .unwrap()
            .insert(slot, Arc::clone(&state));
        Completion::pending(Arc::clone(&self.inner), state)
    }

    pub fn completion(&self, target_epoch: u64) -> Completion {
        self.make_completion(
            target_epoch,
            CompletionMode::Terminal {
                cell: OwnedTerminalCell::new(),
            },
        )
    }

    pub fn pie_completion(&self, target_epoch: u64) -> (PieCompletion, Completion) {
        let completion = self.completion(target_epoch);
        let raw = PieCompletion {
            wait_id: completion.wait_id(),
            target_epoch,
            terminal_cell: completion
                .terminal_cell_ptr()
                .expect("control completion exposes a terminal cell"),
        };
        (raw, completion)
    }

    pub fn launch_completion(&self, target_epoch: u64) -> (PieCompletion, Completion) {
        let completion = self.make_completion(target_epoch, CompletionMode::WakeOnly);
        let raw = PieCompletion {
            wait_id: completion.wait_id(),
            target_epoch,
            terminal_cell: ptr::null_mut(),
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

    pub(crate) fn terminal_cell_ptr(&self) -> Option<*mut PieTerminalCell> {
        match &self.kind {
            CompletionKind::Pending(pending) => pending.state.terminal_cell_ptr(),
            CompletionKind::ReadyOk | CompletionKind::ReadyErr(_) => None,
        }
    }

    pub fn close(&self, message: impl Into<String>) {
        if let CompletionKind::Pending(pending) = &self.kind {
            pending.state.close(pending.broker.table, message);
        }
    }

    /// Non-blocking probe: whether the completion has settled (successfully
    /// or not) without registering any waker.
    pub(crate) fn is_settled(&self) -> bool {
        self.check().is_some()
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
                    Some(epoch) if epoch >= target => {
                        if pending.state.expects_terminal_outcome() {
                            pending.state.terminal_result().or_else(|| {
                                Some(Err(anyhow!(
                                    "driver callback published before terminal outcome settled"
                                )))
                            })
                        } else {
                            Some(Ok(()))
                        }
                    }
                    Some(_) => None,
                    None => Some(pending.state.close_error()),
                }
            }
        }
    }
}

fn poll_wait_slot<T>(
    table: &WakerTable,
    slot: WakerSlotId,
    cx: &mut std::task::Context<'_>,
    mut check: impl FnMut() -> Option<T>,
) -> Poll<T> {
    if let Some(result) = check() {
        return Poll::Ready(result);
    }
    let observed_epoch = table.published(slot).unwrap_or_default();
    if !table.register(slot, cx.waker(), observed_epoch) {
        cx.waker().wake_by_ref();
        return Poll::Pending;
    }
    match check() {
        Some(result) => {
            table.deregister(slot);
            Poll::Ready(result)
        }
        None => Poll::Pending,
    }
}

impl std::future::Future for Completion {
    type Output = Result<()>;

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match &this.kind {
            CompletionKind::ReadyOk => Poll::Ready(Ok(())),
            CompletionKind::ReadyErr(message) => Poll::Ready(Err(anyhow!(message.clone()))),
            CompletionKind::Pending(pending) => {
                let slot = pending.state.slot;
                let table = pending.broker.table;
                poll_wait_slot(table, slot, cx, || this.check())
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

const INSTANCE_RESOLUTION_PENDING: u32 = 0;
const INSTANCE_RESOLUTION_SUCCESS: u32 = 1;
const INSTANCE_RESOLUTION_FAILED: u32 = 2;

struct InstanceCompletionState {
    slot: WakerSlotId,
    target_epoch: AtomicU64,
    terminal: OwnedTerminalCell,
    resolution: AtomicU32,
    message: Mutex<Option<String>>,
    guard: Option<Arc<dyn CompletionLease>>,
}

impl InstanceCompletionState {
    fn new(slot: WakerSlotId, target_epoch: u64, guard: Option<Arc<dyn CompletionLease>>) -> Self {
        assert_valid_target_epoch(target_epoch);
        Self {
            slot,
            target_epoch: AtomicU64::new(target_epoch),
            terminal: OwnedTerminalCell::new(),
            resolution: AtomicU32::new(INSTANCE_RESOLUTION_PENDING),
            message: Mutex::new(None),
            guard,
        }
    }

    fn target_epoch(&self) -> u64 {
        self.target_epoch.load(Ordering::Acquire)
    }

    fn terminal_cell_ptr(&self) -> *mut PieTerminalCell {
        self.terminal.as_mut_ptr()
    }

    fn commit_target_epoch(&self, target_epoch: u64) {
        assert!(target_epoch >= FIRST_COMPLETION_EPOCH);
        let previous = self
            .target_epoch
            .compare_exchange(0, target_epoch, Ordering::AcqRel, Ordering::Acquire)
            .unwrap_or_else(|current| {
                panic!("instance completion target epoch already committed to {current}")
            });
        debug_assert_eq!(previous, 0);
    }

    fn resolve_success(&self) {
        self.resolution
            .store(INSTANCE_RESOLUTION_SUCCESS, Ordering::Release);
        let _ = WakerTable::global().publish(self.slot, 1);
    }

    fn resolve_failure(&self, message: impl Into<String>) {
        *self.message.lock().unwrap() = Some(message.into());
        self.resolution
            .store(INSTANCE_RESOLUTION_FAILED, Ordering::Release);
        let _ = WakerTable::global().publish(self.slot, 1);
    }

    fn result(&self) -> Option<Result<()>> {
        match self.resolution.load(Ordering::Acquire) {
            INSTANCE_RESOLUTION_PENDING => {
                if self.guard.as_ref().is_some_and(|guard| guard.is_closed()) {
                    return Some(Err(anyhow!("instance completion closed")));
                }
                None
            }
            INSTANCE_RESOLUTION_SUCCESS => Some(Ok(())),
            INSTANCE_RESOLUTION_FAILED => Some(Err(anyhow!(
                self.message
                    .lock()
                    .unwrap()
                    .clone()
                    .unwrap_or_else(|| "instance completion failed".to_string())
            ))),
            other => Some(Err(anyhow!("invalid instance resolution state {other}"))),
        }
    }

    fn resolve_from_terminal(&self) -> Result<()> {
        match self.terminal.load() {
            TerminalOutcome::Pending => {
                self.resolve_failure("instance completion terminal outcome is still Pending");
                Err(anyhow!(
                    "instance completion terminal outcome is still Pending"
                ))
            }
            TerminalOutcome::Success => {
                self.resolve_success();
                Ok(())
            }
            TerminalOutcome::Failed => {
                self.resolve_failure("instance completion published Failed terminal outcome");
                Ok(())
            }
            TerminalOutcome::Invalid(value) => {
                self.resolve_failure(format!(
                    "instance completion published invalid terminal outcome {value}"
                ));
                Err(anyhow!(
                    "instance completion published invalid terminal outcome {value}"
                ))
            }
        }
    }
}

#[derive(Clone)]
pub struct InstanceCompletion {
    state: Arc<InstanceCompletionState>,
}

impl std::fmt::Debug for InstanceCompletion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InstanceCompletion")
            .field("wait_id", &self.wait_id())
            .field("target_epoch", &self.target_epoch())
            .finish_non_exhaustive()
    }
}

impl InstanceCompletion {
    pub fn new(wait_id: u64, target_epoch: u64) -> Self {
        Self {
            state: Arc::new(InstanceCompletionState::new(wait_id, target_epoch, None)),
        }
    }

    pub fn with_guard(
        wait_id: u64,
        target_epoch: u64,
        guard: impl Into<Option<Arc<dyn CompletionLease>>>,
    ) -> Self {
        Self {
            state: Arc::new(InstanceCompletionState::new(
                wait_id,
                target_epoch,
                guard.into(),
            )),
        }
    }

    pub fn deferred_with_guard(guard: impl Into<Option<Arc<dyn CompletionLease>>>) -> Self {
        let slot = WakerTable::global().alloc();
        Self::with_guard(slot, 0, guard)
    }

    pub fn wait_id(&self) -> u64 {
        self.state.slot
    }

    pub fn target_epoch(&self) -> u64 {
        self.state.target_epoch()
    }

    pub(crate) fn terminal_cell_ptr(&self) -> *mut PieTerminalCell {
        self.state.terminal_cell_ptr()
    }

    pub(crate) fn commit_target_epoch(&self, target_epoch: u64) {
        self.state.commit_target_epoch(target_epoch);
    }

    pub(crate) fn reject(&self, message: impl Into<String>) {
        self.state.resolve_failure(message);
    }

    pub(crate) fn resolve_from_terminal(&self) -> Result<()> {
        self.state.resolve_from_terminal()
    }

    /// Non-blocking probe: whether this instance completion has resolved.
    pub(crate) fn is_settled(&self) -> bool {
        self.state.result().is_some()
    }
}

impl std::future::Future for InstanceCompletion {
    type Output = Result<()>;

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        let slot = self.state.slot;
        poll_wait_slot(WakerTable::global(), slot, cx, || self.state.result())
    }
}

impl Drop for InstanceCompletion {
    fn drop(&mut self) {
        if Arc::strong_count(&self.state) != 1 {
            return;
        }
        WakerTable::global().free(self.state.slot);
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

    fn store_terminal(cell: *mut PieTerminalCell, outcome: u32) {
        unsafe { (&*terminal_atomic_ptr(cell)).store(outcome, Ordering::Release) };
    }

    #[test]
    fn foreign_thread_publish_completes_waiter() {
        let broker = CompletionBroker::new();
        let callbacks = broker_callbacks(&broker);
        let (raw, completion) = broker.launch_completion(3);
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
    fn control_completion_checks_terminal_outcome() {
        let broker = CompletionBroker::new();
        let callbacks = broker_callbacks(&broker);
        let (raw, completion) = broker.pie_completion(1);
        store_terminal(raw.terminal_cell, PIE_TERMINAL_OUTCOME_FAILED);
        unsafe { runtime_notify(callbacks.ctx, raw.wait_id, 1) };
        let err = completion
            .check()
            .expect("terminal failure should settle")
            .unwrap_err();
        assert!(err.to_string().contains("Failed terminal outcome"));
    }

    #[test]
    fn control_completion_retains_cell_until_callback() {
        let broker = CompletionBroker::new();
        let callbacks = broker_callbacks(&broker);
        let (raw, completion) = broker.pie_completion(1);
        store_terminal(raw.terminal_cell, PIE_TERMINAL_OUTCOME_SUCCESS);
        assert!(
            completion.check().is_none(),
            "terminal publication alone must not retire callback-owned storage"
        );
        unsafe { runtime_notify(callbacks.ctx, raw.wait_id, 1) };
        completion
            .check()
            .expect("callback should settle completion")
            .unwrap();
    }

    #[test]
    fn control_completion_rejects_callback_before_terminal_publication() {
        let broker = CompletionBroker::new();
        let callbacks = broker_callbacks(&broker);
        let (raw, completion) = broker.pie_completion(1);
        unsafe { runtime_notify(callbacks.ctx, raw.wait_id, 1) };
        let err = completion
            .check()
            .expect("callback should settle completion")
            .unwrap_err();
        assert!(err.to_string().contains("terminal outcome settled"));
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
    fn close_wakes_outstanding_waiters() {
        let broker = CompletionBroker::new();
        let completion = broker.launch_completion(7).1;
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
        let (raw, completion) = broker.launch_completion(5);
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
        let (raw, completion) = broker.launch_completion(5);
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
    fn instance_completion_reports_terminal_failure() {
        let completion = Box::pin(InstanceCompletion::deferred_with_guard(None));
        store_terminal(completion.terminal_cell_ptr(), PIE_TERMINAL_OUTCOME_FAILED);
        let _ = completion.resolve_from_terminal();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let err = completion
                .await
                .expect_err("terminal failure should be observable");
            assert!(err.to_string().contains("Failed terminal outcome"));
        });
    }
}
