use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::task::Poll;

use crate::rt::SegQueue;
use anyhow::{Result, anyhow};
use waker::{FIRST_COMPLETION_EPOCH, WakerSlotId, WakerTable};

pub type TerminalOutcomeCode = u32;

pub const TERMINAL_OUTCOME_PENDING: TerminalOutcomeCode = 0;
pub const TERMINAL_OUTCOME_SUCCESS: TerminalOutcomeCode = 1;
pub const TERMINAL_OUTCOME_FAILED: TerminalOutcomeCode = 2;
pub const TERMINAL_OUTCOME_RETRY: TerminalOutcomeCode = 3;

#[derive(Debug, Default)]
pub struct TerminalCell {
    pub outcome: AtomicU32,
}

impl TerminalCell {
    #[must_use]
    pub const fn pending() -> Self {
        Self {
            outcome: AtomicU32::new(TERMINAL_OUTCOME_PENDING),
        }
    }

    pub fn load(&self) -> TerminalOutcomeCode {
        self.outcome.load(Ordering::Acquire)
    }

    pub fn publish(&self, outcome: TerminalOutcomeCode) {
        self.outcome.store(outcome, Ordering::Release);
    }

    pub fn reset(&self) {
        self.outcome
            .store(TERMINAL_OUTCOME_PENDING, Ordering::Release);
    }
}

/// A terminal cell, alive for as long as any holder of it is.
#[derive(Debug, Clone)]
#[repr(transparent)]
pub struct CellPtr(pub Arc<TerminalCell>);

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
    Retry,
    Invalid(u32),
}

#[derive(Debug)]
struct OwnedTerminalCell {
    raw: Option<Arc<TerminalCell>>,
    recyclable: bool,
}

fn terminal_cell_pool() -> &'static SegQueue<Arc<TerminalCell>> {
    static POOL: OnceLock<SegQueue<Arc<TerminalCell>>> = OnceLock::new();
    POOL.get_or_init(SegQueue::new)
}

fn terminal_cell_quarantine() -> &'static SegQueue<Arc<TerminalCell>> {
    static QUARANTINE: OnceLock<SegQueue<Arc<TerminalCell>>> = OnceLock::new();
    QUARANTINE.get_or_init(SegQueue::new)
}

impl OwnedTerminalCell {
    fn new() -> Self {
        let raw = terminal_cell_pool()
            .pop()
            .unwrap_or_else(|| Arc::new(TerminalCell::pending()));
        let cell = Self {
            raw: Some(raw),
            recyclable: false,
        };
        cell.reset();
        cell
    }

    fn cell(&self) -> &Arc<TerminalCell> {
        self.raw.as_ref().expect("owned terminal cell is present")
    }

    fn as_mut_ptr(&self) -> CellPtr {
        CellPtr(Arc::clone(self.cell()))
    }

    fn load(&self) -> TerminalOutcome {
        classify_terminal_outcome(self.cell().load())
    }

    fn reset(&self) {
        self.cell().reset();
    }

    fn mark_recyclable(&mut self) {
        self.recyclable = true;
    }
}

impl Drop for OwnedTerminalCell {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            if self.recyclable {
                terminal_cell_pool().push(raw);
            } else {
                terminal_cell_quarantine().push(raw);
            }
        }
    }
}

fn classify_terminal_outcome(value: u32) -> TerminalOutcome {
    match value {
        TERMINAL_OUTCOME_PENDING => TerminalOutcome::Pending,
        TERMINAL_OUTCOME_SUCCESS => TerminalOutcome::Success,
        TERMINAL_OUTCOME_FAILED => TerminalOutcome::Failed,
        TERMINAL_OUTCOME_RETRY => TerminalOutcome::Retry,
        other => TerminalOutcome::Invalid(other),
    }
}

#[derive(Debug)]
enum SubmissionCompletionMode {
    Terminal {
        cell: OwnedTerminalCell,
    },
    /// Settled once published: a copy's landing, which has no outcome of
    /// its own to tell.
    Landed,
}

#[derive(Debug)]
struct SubmissionCompletionState {
    slot: WakerSlotId,
    target_epoch: u64,
    mode: SubmissionCompletionMode,
    closed: AtomicBool,
    close_message: Mutex<Option<String>>,
}

impl SubmissionCompletionState {
    fn new(slot: WakerSlotId, target_epoch: u64, mode: SubmissionCompletionMode) -> Self {
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
                .unwrap_or_else(|| "engine submission completion closed".to_string())
        ))
    }

    fn terminal_result(&self) -> Option<Result<()>> {
        match &self.mode {
            SubmissionCompletionMode::Terminal { cell } => match cell.load() {
                TerminalOutcome::Pending => None,
                TerminalOutcome::Success => Some(Ok(())),
                TerminalOutcome::Failed => Some(Err(anyhow!(
                    "engine operation published Failed terminal outcome"
                ))),
                TerminalOutcome::Retry => Some(Err(anyhow!(
                    "engine control operation published unexpected Retry terminal outcome"
                ))),
                TerminalOutcome::Invalid(value) => Some(Err(anyhow!(
                    "engine operation published invalid terminal outcome {value}"
                ))),
            },
            SubmissionCompletionMode::Landed => Some(Ok(())),
        }
    }

    fn terminal_cell_ptr(&self) -> Option<CellPtr> {
        match &self.mode {
            SubmissionCompletionMode::Terminal { cell } => Some(cell.as_mut_ptr()),
            SubmissionCompletionMode::Landed => None,
        }
    }

    fn expects_terminal_outcome(&self) -> bool {
        matches!(&self.mode, SubmissionCompletionMode::Terminal { .. })
    }
}

struct BrokerInner {
    table: &'static WakerTable,
}

#[derive(Clone)]
pub struct CompletionBroker {
    inner: Arc<BrokerInner>,
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
        });
        Self { inner }
    }

    fn make_submission_completion(
        &self,
        target_epoch: u64,
        mode: SubmissionCompletionMode,
    ) -> SubmissionCompletion {
        assert_valid_target_epoch(target_epoch);
        let slot = self.inner.table.alloc();
        let state = Arc::new(SubmissionCompletionState::new(slot, target_epoch, mode));
        SubmissionCompletion::pending(Arc::clone(&self.inner), state)
    }

    pub fn submission_completion(&self, target_epoch: u64) -> SubmissionCompletion {
        self.make_submission_completion(
            target_epoch,
            SubmissionCompletionMode::Terminal {
                cell: OwnedTerminalCell::new(),
            },
        )
    }

    /// A completion that settles once [`Self::notify`] publishes its wait
    /// id at `target_epoch`: a copy the engine answered as still landing
    /// ([`::engine::Tiers::copy_landing`]), its wait id the copy's
    /// token.
    pub fn landing_completion(&self, target_epoch: u64) -> SubmissionCompletion {
        self.make_submission_completion(target_epoch, SubmissionCompletionMode::Landed)
    }

    pub fn notify(&self, wait_id: u64, epoch: u64) {
        let _ = self.inner.table.publish(wait_id, epoch);
    }

    /// A copy answered as still landing ended without landing
    /// ([`::engine::Landing::Failed`]): its completion settles as
    /// [`CopyLandingFailed`], never as landed. Its destination is not to be
    /// trusted, and its sources hold what they held.
    pub fn fail(&self, wait_id: u64, epoch: u64, why: String) {
        failed_landings().lock().unwrap().insert(wait_id, why);
        let _ = self.inner.table.publish(wait_id, epoch);
    }
}

/// A copy that did not land, said by the engine
/// ([`::engine::Landing::Failed`]): its destination is not to be trusted,
/// its sources hold what they held before it.
#[derive(Debug, thiserror::Error)]
#[error("the copy did not land: {0}")]
pub struct CopyLandingFailed(pub String);

/// Forgets the boot's failed landings no one read.
pub(crate) fn reset() {
    failed_landings().lock().unwrap().clear();
}

/// The landing completions told failed, by wait id, until read.
fn failed_landings() -> &'static Mutex<std::collections::HashMap<u64, String>> {
    static FAILED: OnceLock<Mutex<std::collections::HashMap<u64, String>>> = OnceLock::new();
    FAILED.get_or_init(Mutex::default)
}

#[derive(Clone)]
struct PendingSubmissionCompletion {
    broker: Arc<BrokerInner>,
    state: Arc<SubmissionCompletionState>,
}

#[derive(Clone)]
enum SubmissionCompletionKind {
    ReadyOk,
    Pending(Arc<PendingSubmissionCompletion>),
}

#[derive(Clone)]
pub struct SubmissionCompletion {
    kind: SubmissionCompletionKind,
}

impl SubmissionCompletion {
    fn pending(broker: Arc<BrokerInner>, state: Arc<SubmissionCompletionState>) -> Self {
        Self {
            kind: SubmissionCompletionKind::Pending(Arc::new(PendingSubmissionCompletion {
                broker,
                state,
            })),
        }
    }

    pub fn ready() -> Self {
        Self {
            kind: SubmissionCompletionKind::ReadyOk,
        }
    }

    pub fn wait_id(&self) -> u64 {
        match &self.kind {
            SubmissionCompletionKind::Pending(pending) => pending.state.slot,
            SubmissionCompletionKind::ReadyOk => {
                panic!("only pending submission completions expose a wait id")
            }
        }
    }

    pub fn target_epoch(&self) -> u64 {
        match &self.kind {
            SubmissionCompletionKind::Pending(pending) => pending.state.target_epoch,
            SubmissionCompletionKind::ReadyOk => {
                panic!("only pending submission completions expose a target epoch")
            }
        }
    }

    pub fn terminal_cell_ptr(&self) -> Option<CellPtr> {
        match &self.kind {
            SubmissionCompletionKind::Pending(pending) => pending.state.terminal_cell_ptr(),
            SubmissionCompletionKind::ReadyOk => None,
        }
    }

    pub fn close(&self, message: impl Into<String>) {
        match &self.kind {
            SubmissionCompletionKind::Pending(pending) => {
                pending.state.close(pending.broker.table, message);
            }
            SubmissionCompletionKind::ReadyOk => {}
        }
    }

    pub fn is_settled(&self) -> bool {
        self.check().is_some()
    }

    pub fn check(&self) -> Option<Result<()>> {
        match &self.kind {
            SubmissionCompletionKind::ReadyOk => Some(Ok(())),
            SubmissionCompletionKind::Pending(pending) => {
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
                                    "engine callback published before terminal outcome settled"
                                )))
                            })
                        } else if let Some(why) = failed_landings().lock().unwrap().remove(&slot) {
                            Some(Err(anyhow::Error::new(CopyLandingFailed(why))))
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
            // The slot keeps one entry a task: another future of this task
            // may wait on the same completion, so the task polls again.
            if table.deregister_waker(slot, cx.waker()) {
                cx.waker().wake_by_ref();
            }
            Poll::Ready(result)
        }
        None => Poll::Pending,
    }
}

impl std::future::Future for SubmissionCompletion {
    type Output = Result<()>;

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match &this.kind {
            SubmissionCompletionKind::ReadyOk => Poll::Ready(Ok(())),
            SubmissionCompletionKind::Pending(pending) => {
                let slot = pending.state.slot;
                let table = pending.broker.table;
                poll_wait_slot(table, slot, cx, || this.check())
            }
        }
    }
}

impl Drop for SubmissionCompletion {
    fn drop(&mut self) {
        let SubmissionCompletionKind::Pending(pending) = &self.kind else {
            return;
        };
        if Arc::strong_count(pending) != 1 {
            return;
        }
        if !pending.state.closed.load(Ordering::Acquire) {
            pending.broker.table.free(pending.state.slot);
        }
    }
}

const WORK_ITEM_RESOLUTION_PENDING: u32 = 0;
const WORK_ITEM_RESOLUTION_SUCCESS: u32 = 1;
const WORK_ITEM_RESOLUTION_FAILED: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkItemAttemptOutcome {
    Committed,
    Retry,
    Failed,
}

struct WorkItemCompletionState {
    slot: WakerSlotId,
    target_epoch: AtomicU64,
    terminal: OwnedTerminalCell,
    resolution: AtomicU32,
    cancel_requested: AtomicBool,
    native_retired: AtomicBool,
    message: Mutex<Option<String>>,
    guard: Option<Arc<dyn CompletionLease>>,
    /// Run once the completion resolves (or, never resolved, as the last
    /// handle on it goes): see [`WorkItemCompletion::on_settled`].
    on_settled: Mutex<Vec<LandedHook>>,
}

type SettledHook = Box<dyn FnOnce() + Send>;

/// A hook told whether its completion landed, as it resolved.
type LandedHook = Box<dyn FnOnce(bool) + Send>;

/// `hooks`, each told `landed` now: they may run once no handle on the
/// completion is left, past where it could still be asked.
fn told(hooks: Vec<LandedHook>, landed: bool) -> Vec<SettledHook> {
    hooks
        .into_iter()
        .map(|hook| Box::new(move || hook(landed)) as SettledHook)
        .collect()
}

std::thread_local! {
    /// Hooks gathered by [`settle_batched`] on this thread, run together
    /// once it ends.
    static BATCHED: std::cell::RefCell<Option<Vec<SettledHook>>> =
        const { std::cell::RefCell::new(None) };
}

/// Runs `body`, the hooks of every completion it resolves going off the
/// caller's stack together once it ends, as one task: a landing resolves a
/// fire a lane, and a task each (spawned from the lane's own thread) cost
/// that thread ~150 us a step at 64 lanes.
pub(crate) fn settle_batched<T>(body: impl FnOnce() -> T) -> T {
    let outer = BATCHED.with(|batch| batch.borrow().is_some());
    if outer {
        return body();
    }
    BATCHED.with(|batch| *batch.borrow_mut() = Some(Vec::new()));
    struct Flush;
    impl Drop for Flush {
        fn drop(&mut self) {
            let hooks = BATCHED.with(|batch| batch.borrow_mut().take().unwrap_or_default());
            run_settled(hooks);
        }
    }
    let _flush = Flush;
    body()
}

/// Runs `hooks` off the caller's stack where a runtime is up: a resolution
/// may come under a store's lock its hooks take. Within [`settle_batched`]
/// they wait for its end.
fn run_settled(mut hooks: Vec<SettledHook>) {
    if hooks.is_empty() {
        return;
    }
    let deferred = BATCHED.with(|batch| match batch.borrow_mut().as_mut() {
        Some(batch) => {
            batch.append(&mut hooks);
            true
        }
        None => false,
    });
    if deferred {
        return;
    }
    if crate::rt::has_runtime() {
        crate::rt::spawn(async move {
            for hook in hooks {
                hook();
            }
        });
    } else {
        for hook in hooks {
            hook();
        }
    }
}

impl WorkItemCompletionState {
    fn new(slot: WakerSlotId, target_epoch: u64, guard: Option<Arc<dyn CompletionLease>>) -> Self {
        assert_valid_target_epoch(target_epoch);
        Self {
            slot,
            target_epoch: AtomicU64::new(target_epoch),
            terminal: OwnedTerminalCell::new(),
            resolution: AtomicU32::new(WORK_ITEM_RESOLUTION_PENDING),
            cancel_requested: AtomicBool::new(false),

            native_retired: AtomicBool::new(false),
            message: Mutex::new(None),
            guard,
            on_settled: Mutex::new(Vec::new()),
        }
    }

    fn settled(&self) {
        let hooks = std::mem::take(&mut *self.on_settled.lock().unwrap());
        run_settled(told(hooks, matches!(self.result(), Some(Ok(())))));
    }

    fn target_epoch(&self) -> u64 {
        self.target_epoch.load(Ordering::Acquire)
    }

    fn terminal_cell_ptr(&self) -> CellPtr {
        self.terminal.as_mut_ptr()
    }

    fn commit_target_epoch(&self, target_epoch: u64) {
        assert!(target_epoch >= FIRST_COMPLETION_EPOCH);
        self.native_retired.store(false, Ordering::Release);
        let previous = self.target_epoch.swap(target_epoch, Ordering::AcqRel);
        assert!(
            previous == 0 || target_epoch > previous,
            "work item completion attempt epoch must advance (previous {previous}, next {target_epoch})"
        );
    }

    fn resolve_success(&self) {
        self.resolution
            .store(WORK_ITEM_RESOLUTION_SUCCESS, Ordering::Release);

        let outcome = WakerTable::global().wake(self.slot);
        let _ = outcome;
        self.settled();
    }

    fn resolve_failure(&self, message: impl Into<String>) {
        *self.message.lock().unwrap() = Some(message.into());
        self.resolution
            .store(WORK_ITEM_RESOLUTION_FAILED, Ordering::Release);

        let _ = WakerTable::global().wake(self.slot);
        self.settled();
    }

    fn request_cancel(&self) {
        self.cancel_requested.store(true, Ordering::Release);
    }

    fn cancel_requested(&self) -> bool {
        self.cancel_requested.load(Ordering::Acquire)
    }

    fn mark_native_retired(&self) {
        self.native_retired.store(true, Ordering::Release);
    }

    fn result(&self) -> Option<Result<()>> {
        match self.resolution.load(Ordering::Acquire) {
            WORK_ITEM_RESOLUTION_PENDING => {
                if self.guard.as_ref().is_some_and(|guard| guard.is_closed()) {
                    return Some(Err(anyhow!("work item completion closed")));
                }
                None
            }
            WORK_ITEM_RESOLUTION_SUCCESS => Some(Ok(())),
            WORK_ITEM_RESOLUTION_FAILED => Some(Err(anyhow!(
                self.message
                    .lock()
                    .unwrap()
                    .clone()
                    .unwrap_or_else(|| "work item completion failed".to_string())
            ))),
            other => Some(Err(anyhow!("invalid work item resolution state {other}"))),
        }
    }

    fn resolve_from_terminal(&self) -> Result<WorkItemAttemptOutcome> {
        match self.terminal.load() {
            TerminalOutcome::Pending => {
                self.resolve_failure("work item completion terminal outcome is still Pending");
                Err(anyhow!(
                    "work item completion terminal outcome is still Pending"
                ))
            }
            TerminalOutcome::Success => {
                self.resolve_success();
                Ok(WorkItemAttemptOutcome::Committed)
            }
            TerminalOutcome::Failed => {
                self.resolve_failure("work item completion published Failed terminal outcome");
                Ok(WorkItemAttemptOutcome::Failed)
            }
            TerminalOutcome::Retry => {
                self.terminal.reset();
                Ok(WorkItemAttemptOutcome::Retry)
            }
            TerminalOutcome::Invalid(value) => {
                self.resolve_failure(format!(
                    "work item completion published invalid terminal outcome {value}"
                ));
                Err(anyhow!(
                    "work item completion published invalid terminal outcome {value}"
                ))
            }
        }
    }
}

impl Drop for WorkItemCompletionState {
    fn drop(&mut self) {
        if self.native_retired.load(Ordering::Acquire) {
            self.terminal.mark_recyclable();
        }
        // No one holds the work item any more: nothing runs it now.
        run_settled(told(
            std::mem::take(self.on_settled.get_mut().unwrap()),
            false,
        ));
    }
}

#[derive(Clone)]
pub struct WorkItemCompletion {
    state: Arc<WorkItemCompletionState>,
}

impl std::fmt::Debug for WorkItemCompletion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkItemCompletion")
            .field("wait_id", &self.wait_id())
            .field("target_epoch", &self.target_epoch())
            .finish_non_exhaustive()
    }
}

impl WorkItemCompletion {
    pub fn new(wait_id: u64, target_epoch: u64) -> Self {
        Self {
            state: Arc::new(WorkItemCompletionState::new(wait_id, target_epoch, None)),
        }
    }

    pub fn with_guard(
        wait_id: u64,
        target_epoch: u64,
        guard: impl Into<Option<Arc<dyn CompletionLease>>>,
    ) -> Self {
        Self {
            state: Arc::new(WorkItemCompletionState::new(
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

    pub fn terminal_cell_ptr(&self) -> CellPtr {
        self.state.terminal_cell_ptr()
    }

    pub fn commit_target_epoch(&self, target_epoch: u64) {
        self.state.commit_target_epoch(target_epoch);
    }

    pub fn reject(&self, message: impl Into<String>) {
        self.state.resolve_failure(message);
    }

    pub fn reject_unsubmitted(&self, message: impl Into<String>) {
        self.state.mark_native_retired();
        self.state.resolve_failure(message);
    }

    pub fn request_cancel(&self) {
        self.state.request_cancel();
    }

    pub fn cancel_requested(&self) -> bool {
        self.state.cancel_requested()
    }

    pub fn mark_native_retired(&self) {
        self.state.mark_native_retired();
    }

    pub fn resolve_from_terminal(&self) -> Result<WorkItemAttemptOutcome> {
        self.state.resolve_from_terminal()
    }

    /// Resolves a fire its engine settled from its terminal cell, `at` (a
    /// step or a frame) naming where.
    pub fn resolve_landed(&self, at: &str) {
        self.mark_native_retired();
        match self.resolve_from_terminal() {
            Ok(WorkItemAttemptOutcome::Retry) => self.reject(format!(
                "engine published RETRY at {at}; retry is not a v14 outcome (frame admission \
                 bounds every in-frame gate)"
            )),
            Ok(_) => {}
            Err(err) => tracing::warn!(?err, at, "terminal settlement failed"),
        }
    }

    pub fn is_settled(&self) -> bool {
        self.state.result().is_some()
    }

    /// Runs `hook` once this resolves: now if it has, else as it does (or,
    /// never resolved, once the last handle on it goes). Unlike awaiting it,
    /// this takes no waker and no task: a hook runs where the completion
    /// resolves, with nothing left to poll.
    pub fn on_settled(&self, hook: impl FnOnce() + Send + 'static) {
        self.on_settled_landed(move |_| hook());
    }

    /// [`Self::on_settled`], told whether it landed, as it resolved: never
    /// where it resolved otherwise, or no handle on it was left unresolved.
    /// Its hooks run off the resolving stack, often once the last handle
    /// is gone, so what it is told is fixed where it resolves.
    pub fn on_settled_landed(&self, hook: impl FnOnce(bool) + Send + 'static) {
        let mut hooks = self.state.on_settled.lock().unwrap();
        let Some(result) = self.state.result() else {
            hooks.push(Box::new(hook));
            return;
        };
        drop(hooks);
        hook(result.is_ok());
    }

    pub fn same_request(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.state, &other.state)
    }
}

impl std::future::Future for WorkItemCompletion {
    type Output = Result<()>;

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        let slot = self.state.slot;
        poll_wait_slot(WakerTable::global(), slot, cx, || self.state.result())
    }
}

impl Drop for WorkItemCompletion {
    fn drop(&mut self) {
        if Arc::strong_count(&self.state) != 1 {
            return;
        }
        WakerTable::global().free(self.state.slot);
    }
}

pub fn settle(cells: &[CellPtr], outcome: TerminalOutcomeCode) {
    for CellPtr(cell) in cells {
        cell.publish(outcome);
    }
}

struct PendingFrame {
    /// Each step: its fires' terminal cells, the completions they are of,
    /// and the waiters on what it puts; `None` once the step has settled
    /// and published them.
    steps: Vec<Option<LandingStep>>,
    /// The submission's cell, its waiter and epoch, settled once every
    /// step is in: `None` on an engine frame of a submission that took
    /// more than one, all but its last (a refusal of lanes stopped the
    /// one before short, `engine::FrameTicket::refused`).
    frame: Option<(CellPtr, u64, u64)>,
    /// The launch this frame is one of, and its place in it, where the
    /// launch took more than one engine frame.
    split: Option<(Arc<SplitLaunch>, usize)>,
}

/// A launch the runtime submitted as several engine frames (a refusal of
/// lanes stopped one short and the rest went again,
/// `engine::FrameTicket::refused`): its frames land in the order they were
/// submitted, and one landing before an earlier one is in fails every fire
/// of the launch still landing, by name, rather than settle a fire on
/// another frame's tokens.
pub struct SplitLaunch {
    launch: u64,
    frames: usize,
    order: Mutex<SplitOrder>,
}

#[derive(Default)]
struct SplitOrder {
    /// The frame whose steps land next.
    next: usize,
    /// Why the launch's frames are refused, once one landed out of order.
    broken: Option<String>,
}

impl SplitLaunch {
    /// A launch of `frames` engine frames, numbered launch-wide.
    #[must_use]
    pub fn new(frames: usize) -> Arc<SplitLaunch> {
        static LAUNCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        Arc::new(SplitLaunch {
            launch: LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            frames,
            order: Mutex::new(SplitOrder::default()),
        })
    }

    /// Whether a step of the launch's frame `index` may land: every frame
    /// before it in, and none out of order yet.
    fn admit(&self, index: usize) -> Result<(), String> {
        let mut order = self.order.lock().unwrap();
        if let Some(why) = &order.broken {
            return Err(why.clone());
        }
        if index != order.next {
            let why = format!(
                "launch {} settled its engine frame {index} of {} while frame {} was \
                 expected: a launch's frames land in the order they were submitted, so \
                 its fires fail rather than take another frame's tokens",
                self.launch, self.frames, order.next
            );
            order.broken = Some(why.clone());
            return Err(why);
        }
        Ok(())
    }

    /// The launch's frame `index` is in.
    fn finish(&self, index: usize) {
        let mut order = self.order.lock().unwrap();
        if order.next == index {
            order.next = index + 1;
        }
    }
}

/// How a step landed, as far as its fires are concerned: the lanes that
/// faulted, the whole frame lost (the lane went down under it), or the
/// whole frame refused for why.
enum Landing {
    Lanes(Vec<engine::LaneFault>),
    Lost,
    Refused(String),
}

impl PendingFrame {
    /// Takes out what step `step` settles: that step as `landing` says,
    /// or, the frame lost, every step not yet published; and the frame
    /// once none is left. Published with [`Landed::publish`], off the
    /// book's lock.
    fn land(&mut self, step: usize, landing: Landing) -> Landed {
        let landing = match (&self.split, landing) {
            (Some((split, index)), Landing::Lanes(faults)) => match split.admit(*index) {
                Ok(()) => Landing::Lanes(faults),
                Err(why) => Landing::Refused(why),
            },
            (_, landing) => landing,
        };
        let (range, faults, lost, refused) = match landing {
            Landing::Lost => (0..self.steps.len(), Vec::new(), true, None),
            Landing::Refused(why) => (0..self.steps.len(), Vec::new(), true, Some(why)),
            Landing::Lanes(faults) => (
                step.min(self.steps.len())..(step + 1).min(self.steps.len()),
                faults,
                false,
                None,
            ),
        };
        let steps = self.steps[range]
            .iter_mut()
            .filter_map(Option::take)
            .collect();
        let done = lost || self.steps.iter().all(Option::is_none);
        if done
            && !lost
            && let Some((split, index)) = &self.split
        {
            split.finish(*index);
        }
        let frame = self.frame.clone().filter(|_| done);
        Landed {
            steps,
            faults,
            lost,
            refused,
            done,
            frame,
        }
    }
}

/// What a landing settles: its steps' fires and waiters, and the frame's
/// cell and waiter once its last step is in.
struct Landed {
    steps: Vec<LandingStep>,
    /// The lanes of its one step that faulted.
    faults: Vec<engine::LaneFault>,
    lost: bool,
    /// Why the frame's fires are refused, where they are.
    refused: Option<String>,
    /// Every step of the frame is in.
    done: bool,
    frame: Option<(CellPtr, u64, u64)>,
}

impl Landed {
    fn publish(self, broker: &CompletionBroker) {
        let code = if self.lost {
            TERMINAL_OUTCOME_FAILED
        } else {
            TERMINAL_OUTCOME_SUCCESS
        };
        let faults = &self.faults;
        settle_batched(|| {
            for step in self.steps {
                if let Some(why) = &self.refused {
                    settle(&step.cells, TERMINAL_OUTCOME_FAILED);
                    tracing::warn!(%why, "a launch's frame landed out of order; its fires fail");
                    for completion in &step.completions {
                        completion.mark_native_retired();
                        completion.reject(why.clone());
                    }
                } else if faults.is_empty() {
                    settle(&step.cells, code);
                    // Each fire resolves as its step lands, so what waits
                    // on it goes on a step later rather than once the frame
                    // retires.
                    for completion in &step.completions {
                        completion.resolve_landed("step settle");
                    }
                } else {
                    step.settle_lanes(faults);
                }
                for wait_id in &step.wakes {
                    let outcome = waker::WakerTable::global().wake(*wait_id);
                    tracing::debug!(wait_id, ?outcome, "published to a waiter");
                }
            }
        });
        if let Some((cell, wait_id, epoch)) = self.frame {
            settle(&[cell], code);
            broker.notify(wait_id, epoch);
        }
    }
}

/// A step still landing: its fires' terminal cells and completions, and the
/// waiters on what it puts.
#[derive(Default)]
pub struct LandingStep {
    pub cells: Vec<CellPtr>,
    pub completions: Vec<WorkItemCompletion>,
    /// Which of the step's lanes are each fire's: fire `i` (its cell and
    /// its completion) has lanes `member_lanes[i]..member_lanes[i + 1]`.
    pub member_lanes: Vec<u32>,
    pub wakes: Vec<u64>,
}

impl LandingStep {
    /// Settles the step's fires, some of its lanes faulted: a fire with a
    /// faulted lane fails by that lane's reason, and every other fire
    /// commits.
    fn settle_lanes(&self, faults: &[engine::LaneFault]) {
        settle_fires(
            &self.cells,
            &self.completions,
            &self.member_lanes,
            faults,
            "step settle",
        );
    }
}

/// Settles fires `cells` (each the terminal cell of the completion at its
/// index), whose lanes `member_lanes` says, as their lanes landed: a fire
/// with a lane in `faults` fails by the first such lane's reason, which
/// fails that fire's pipeline and nothing else; every other fire commits.
pub(crate) fn settle_fires(
    cells: &[CellPtr],
    completions: &[WorkItemCompletion],
    member_lanes: &[u32],
    faults: &[engine::LaneFault],
    at: &str,
) {
    for (member, cell) in cells.iter().enumerate() {
        let lanes = match (member_lanes.get(member), member_lanes.get(member + 1)) {
            (Some(&start), Some(&end)) => start..end,
            _ => 0..0,
        };
        let fault = faults.iter().find(|fault| lanes.contains(&fault.lane));
        match fault {
            None => {
                settle(std::slice::from_ref(cell), TERMINAL_OUTCOME_SUCCESS);
                if let Some(completion) = completions.get(member) {
                    completion.resolve_landed(at);
                }
            }
            Some(fault) => {
                settle(std::slice::from_ref(cell), TERMINAL_OUTCOME_FAILED);
                let why = match fault.instance {
                    Some(instance) => format!(
                        "engine: lane {} (instance {instance}) faulted: {}",
                        fault.lane, fault.reason
                    ),
                    None => format!("engine: lane {} faulted: {}", fault.lane, fault.reason),
                };
                tracing::warn!(lane = fault.lane, %why, "a lane faulted; its fire fails alone");
                if let Some(completion) = completions.get(member) {
                    completion.mark_native_retired();
                    completion.reject(why);
                }
            }
        }
    }
}

/// The frames on a lane whose steps are still landing: each step publishes
/// what it settled as it lands, so a lane reads its step's values a step
/// later, not a frame.
#[derive(Default)]
pub struct FrameSettlements {
    inner: Mutex<Book>,
}

#[derive(Default)]
struct Book {
    frames: std::collections::HashMap<engine::FrameId, PendingFrame>,
    /// Steps that landed before their frame was booked, and the lanes of
    /// each that faulted.
    early: std::collections::HashMap<engine::FrameId, Vec<(usize, Vec<engine::LaneFault>)>>,
}

impl FrameSettlements {
    #[must_use]
    pub fn new() -> Arc<FrameSettlements> {
        Arc::new(FrameSettlements::default())
    }

    /// Books frame `frame`: each step's fires and the waiters on its puts,
    /// step by step, and `completion`'s cell once they are in, where the
    /// frame is its submission's last (`None` on one before it).
    pub fn expect(
        &self,
        frame: engine::FrameId,
        steps: Vec<LandingStep>,
        completion: Option<&SubmissionCompletion>,
        split: Option<(Arc<SplitLaunch>, usize)>,
        broker: &CompletionBroker,
    ) {
        let frame_completion = match completion {
            Some(completion) => {
                let Some(cell) = completion.terminal_cell_ptr() else {
                    return;
                };
                Some((cell, completion.wait_id(), completion.target_epoch()))
            }
            None => None,
        };
        let mut steps: Vec<_> = steps.into_iter().map(Some).collect();
        if steps.is_empty() {
            steps.push(Some(LandingStep::default()));
        }
        let mut pending = PendingFrame {
            steps,
            frame: frame_completion,
            split,
        };
        let mut landed = Vec::new();
        {
            let mut book = self.inner.lock().unwrap();
            let early = book.early.remove(&frame).unwrap_or_default();
            let mut done = false;
            for (step, faults) in early {
                let landing = pending.land(step, Landing::Lanes(faults));
                done |= landing.done;
                landed.push(landing);
            }
            if !done {
                book.frames.insert(frame, pending);
            }
        }
        for landing in landed {
            landing.publish(broker);
        }
    }

    /// Step `step` of frame `frame` landed, as `outcome` says: the fires
    /// of its faulted lanes fail, each by its lane's reason, and the rest
    /// commit.
    pub fn settled(
        &self,
        frame: engine::FrameId,
        step: usize,
        outcome: &engine::StepOutcome,
        broker: &CompletionBroker,
    ) {
        let faults = outcome.faulted.clone();
        let landed = {
            let mut book = self.inner.lock().unwrap();
            let Some(pending) = book.frames.get_mut(&frame) else {
                tracing::debug!(
                    frame = frame.0,
                    step,
                    faulted = faults.len(),
                    "settled before its frame was booked"
                );
                book.early.entry(frame).or_default().push((step, faults));
                return;
            };
            let landed = pending.land(step, Landing::Lanes(faults));
            if landed.done {
                tracing::debug!(frame = frame.0, "frame settled");
                book.frames.remove(&frame);
            }
            landed
        };
        landed.publish(broker);
    }

    pub fn close_all(&self, broker: &CompletionBroker) {
        let frames: Vec<PendingFrame> = {
            let mut book = self.inner.lock().unwrap();
            book.early.clear();
            book.frames.drain().map(|(_, f)| f).collect()
        };
        for mut pending in frames {
            pending.land(0, Landing::Lost).publish(broker);
        }
    }
}
