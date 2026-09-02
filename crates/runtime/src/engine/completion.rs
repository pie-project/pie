//! The completion broker: run-ahead's bookkeeping. A waker table, a
//! recycling pool of atomic terminal cells, and a per-work-item lease, for
//! how the runtime decides to run ahead of a device.
//!
//! The terminal cell is local (no longer crossing an ABI boundary): nothing
//! crosses now, since [`engine::Engine::fire`] answers a
//! `Result<FireTicket>` and the settle happens on this side of it. [`settle`]
//! is the one place a `Result<FireTicket, Error>` becomes a published
//! outcome, written here rather than by the engine.

use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::task::Poll;

use anyhow::{Result, anyhow};
use crossbeam::queue::SegQueue;
use waker::{FIRST_COMPLETION_EPOCH, WakerSlotId, WakerTable};

/// A terminal outcome, as the cell holds it.
pub type TerminalOutcomeCode = u32;

/// Nothing published yet.
pub const TERMINAL_OUTCOME_PENDING: TerminalOutcomeCode = 0;
/// The work committed.
pub const TERMINAL_OUTCOME_SUCCESS: TerminalOutcomeCode = 1;
/// The work failed and will not be retried.
pub const TERMINAL_OUTCOME_FAILED: TerminalOutcomeCode = 2;
/// The work did not fit; the attempt may be made again.
pub const TERMINAL_OUTCOME_RETRY: TerminalOutcomeCode = 3;

/// One work item's published outcome word.
#[derive(Debug, Default)]
pub struct TerminalCell {
    /// The outcome, as one of the four codes above.
    pub outcome: AtomicU32,
}

impl TerminalCell {
    /// A cell holding nothing yet.
    #[must_use]
    pub const fn pending() -> Self {
        Self {
            outcome: AtomicU32::new(TERMINAL_OUTCOME_PENDING),
        }
    }

    /// What it holds.
    pub fn load(&self) -> TerminalOutcomeCode {
        self.outcome.load(Ordering::Acquire)
    }

    /// Publish an outcome into it.
    pub fn publish(&self, outcome: TerminalOutcomeCode) {
        self.outcome.store(outcome, Ordering::Release);
    }

    /// Return it to the pending state.
    pub fn reset(&self) {
        self.outcome
            .store(TERMINAL_OUTCOME_PENDING, Ordering::Release);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CompletionTarget {

    pub wait_id: u64,

    pub target_epoch: u64,

    pub terminal_cell: *mut TerminalCell,
}

unsafe impl Send for CompletionTarget {}
unsafe impl Sync for CompletionTarget {}

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
struct TerminalCellStorage(TerminalCell);

#[derive(Debug)]
struct OwnedTerminalCell {
    raw: Option<Box<TerminalCellStorage>>,
    recyclable: bool,
}

fn terminal_cell_pool() -> &'static SegQueue<Box<TerminalCellStorage>> {
    static POOL: OnceLock<SegQueue<Box<TerminalCellStorage>>> = OnceLock::new();
    POOL.get_or_init(SegQueue::new)
}

fn terminal_cell_quarantine() -> &'static SegQueue<Box<TerminalCellStorage>> {
    static QUARANTINE: OnceLock<SegQueue<Box<TerminalCellStorage>>> = OnceLock::new();
    QUARANTINE.get_or_init(SegQueue::new)
}

impl OwnedTerminalCell {
    fn new() -> Self {
        let raw = terminal_cell_pool()
            .pop()
            .unwrap_or_else(|| Box::new(TerminalCellStorage(TerminalCell::pending())));
        let cell = Self {
            raw: Some(raw),
            recyclable: false,
        };
        cell.reset();
        cell
    }

    fn cell(&self) -> &TerminalCell {
        &self
            .raw
            .as_deref()
            .expect("owned terminal cell is present")
            .0
    }

    fn as_mut_ptr(&self) -> *mut TerminalCell {
        ptr::from_ref(self.cell()).cast_mut()
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
    WakeOnly,
    Terminal { cell: OwnedTerminalCell },
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
            SubmissionCompletionMode::WakeOnly => None,
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
        }
    }

    fn terminal_cell_ptr(&self) -> Option<*mut TerminalCell> {
        match &self.mode {
            SubmissionCompletionMode::WakeOnly => None,
            SubmissionCompletionMode::Terminal { cell } => Some(cell.as_mut_ptr()),
        }
    }

    fn expects_terminal_outcome(&self) -> bool {
        matches!(&self.mode, SubmissionCompletionMode::Terminal { .. })
    }
}

#[derive(Default)]
struct LiveRegistry {
    entries: Vec<Weak<SubmissionCompletionState>>,
    compact_at: usize,
}

impl LiveRegistry {
    fn insert(&mut self, state: &Arc<SubmissionCompletionState>) {
        if self.entries.len() >= self.compact_at {
            self.entries.retain(|weak| weak.strong_count() > 0);
            self.compact_at = (self.entries.len() * 2).max(64);
        }
        self.entries.push(Arc::downgrade(state));
    }

    fn drain_live(&mut self) -> Vec<Arc<SubmissionCompletionState>> {
        std::mem::take(&mut self.entries)
            .iter()
            .filter_map(Weak::upgrade)
            .collect()
    }
}

struct BrokerInner {
    table: &'static WakerTable,
    live: Mutex<LiveRegistry>,
    closed: AtomicBool,
    close_message: Mutex<Option<String>>,
}

impl BrokerInner {
    fn close_message(&self) -> String {
        self.close_message
            .lock()
            .unwrap()
            .clone()
            .unwrap_or_else(|| "completion broker is closed".to_string())
    }
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
            live: Mutex::new(LiveRegistry::default()),
            closed: AtomicBool::new(false),
            close_message: Mutex::new(None),
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
        self.inner.live.lock().unwrap().insert(&state);

        if self.inner.closed.load(Ordering::Acquire) {
            state.close(self.inner.table, self.inner.close_message());
        }
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

    #[must_use]
    pub fn control_completion(
        &self,
        target_epoch: u64,
    ) -> (CompletionTarget, SubmissionCompletion) {
        let completion = self.submission_completion(target_epoch);
        let target = CompletionTarget {
            wait_id: completion.wait_id(),
            target_epoch,
            terminal_cell: completion
                .terminal_cell_ptr()
                .expect("control completion exposes a terminal cell"),
        };
        (target, completion)
    }

    #[must_use]
    pub fn launch_completion(&self, target_epoch: u64) -> (CompletionTarget, SubmissionCompletion) {
        let completion =
            self.make_submission_completion(target_epoch, SubmissionCompletionMode::WakeOnly);
        let target = CompletionTarget {
            wait_id: completion.wait_id(),
            target_epoch,
            terminal_cell: ptr::null_mut(),
        };
        (target, completion)
    }

    pub fn close_all(&self, message: impl Into<String>) {
        let message = message.into();
        *self.inner.close_message.lock().unwrap() = Some(message.clone());
        self.inner.closed.store(true, Ordering::Release);
        let states = self.inner.live.lock().unwrap().drain_live();
        for state in states {
            state.close(self.inner.table, message.clone());
        }
    }

    pub fn notify(&self, wait_id: u64, epoch: u64) {
        if !self.inner.closed.load(Ordering::Acquire) {
            let _ = self.inner.table.publish(wait_id, epoch);
        }
    }

}

#[derive(Clone)]
struct PendingSubmissionCompletion {
    broker: Arc<BrokerInner>,
    state: Arc<SubmissionCompletionState>,
}

#[derive(Clone)]
enum SubmissionCompletionKind {
    ReadyOk,
    ReadyErr(String),
    Pending(Arc<PendingSubmissionCompletion>),

    All(Arc<Vec<SubmissionCompletion>>),
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

    pub fn failed(message: impl Into<String>) -> Self {
        Self {
            kind: SubmissionCompletionKind::ReadyErr(message.into()),
        }
    }

    pub fn all(parts: Vec<SubmissionCompletion>) -> Self {
        match parts.len() {
            0 => Self::ready(),
            1 => parts.into_iter().next().expect("one part"),
            _ => Self {
                kind: SubmissionCompletionKind::All(Arc::new(parts)),
            },
        }
    }

    pub fn wait_id(&self) -> u64 {
        match &self.kind {
            SubmissionCompletionKind::Pending(pending) => pending.state.slot,
            SubmissionCompletionKind::ReadyOk
            | SubmissionCompletionKind::ReadyErr(_)
            | SubmissionCompletionKind::All(_) => {
                panic!("only pending submission completions expose a wait id")
            }
        }
    }

    pub fn target_epoch(&self) -> u64 {
        match &self.kind {
            SubmissionCompletionKind::Pending(pending) => pending.state.target_epoch,
            SubmissionCompletionKind::ReadyOk
            | SubmissionCompletionKind::ReadyErr(_)
            | SubmissionCompletionKind::All(_) => {
                panic!("only pending submission completions expose a target epoch")
            }
        }
    }

    pub fn terminal_cell_ptr(&self) -> Option<*mut TerminalCell> {
        match &self.kind {
            SubmissionCompletionKind::Pending(pending) => pending.state.terminal_cell_ptr(),
            SubmissionCompletionKind::ReadyOk
            | SubmissionCompletionKind::ReadyErr(_)
            | SubmissionCompletionKind::All(_) => None,
        }
    }

    pub fn close(&self, message: impl Into<String>) {
        match &self.kind {
            SubmissionCompletionKind::Pending(pending) => {
                pending.state.close(pending.broker.table, message);
            }
            SubmissionCompletionKind::All(parts) => {
                let message = message.into();
                for part in parts.iter() {
                    part.close(message.clone());
                }
            }
            SubmissionCompletionKind::ReadyOk | SubmissionCompletionKind::ReadyErr(_) => {}
        }
    }

    pub fn is_settled(&self) -> bool {
        self.check().is_some()
    }

    pub fn check(&self) -> Option<Result<()>> {
        match &self.kind {
            SubmissionCompletionKind::ReadyOk => Some(Ok(())),
            SubmissionCompletionKind::ReadyErr(message) => Some(Err(anyhow!(message.clone()))),
            SubmissionCompletionKind::All(parts) => {
                let mut first_error = None;
                for part in parts.iter() {
                    match part.check() {
                        None => return None,
                        Some(Err(error)) if first_error.is_none() => first_error = Some(error),
                        Some(_) => {}
                    }
                }
                Some(first_error.map_or(Ok(()), Err))
            }
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

impl std::future::Future for SubmissionCompletion {
    type Output = Result<()>;

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match &this.kind {
            SubmissionCompletionKind::ReadyOk => Poll::Ready(Ok(())),
            SubmissionCompletionKind::ReadyErr(message) => {
                Poll::Ready(Err(anyhow!(message.clone())))
            }
            SubmissionCompletionKind::All(parts) => {

                let mut first_error = None;
                let mut pending = false;
                for part in parts.iter() {
                    let mut part = part.clone();
                    match std::pin::Pin::new(&mut part).poll(cx) {
                        Poll::Pending => pending = true,
                        Poll::Ready(Err(error)) if first_error.is_none() => {
                            first_error = Some(error);
                        }
                        Poll::Ready(_) => {}
                    }
                }
                if pending {
                    Poll::Pending
                } else {
                    Poll::Ready(first_error.map_or(Ok(()), Err))
                }
            }
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
        }
    }

    fn target_epoch(&self) -> u64 {
        self.target_epoch.load(Ordering::Acquire)
    }

    fn terminal_cell_ptr(&self) -> *mut TerminalCell {
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
    }

    fn resolve_failure(&self, message: impl Into<String>) {
        *self.message.lock().unwrap() = Some(message.into());
        self.resolution
            .store(WORK_ITEM_RESOLUTION_FAILED, Ordering::Release);

        let _ = WakerTable::global().wake(self.slot);
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

    pub fn terminal_cell_ptr(&self) -> *mut TerminalCell {
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

    pub fn is_settled(&self) -> bool {
        self.state.result().is_some()
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


/// Publish one answer into every terminal cell a submission reserved. The
/// runtime writes them, once, from the answer
/// [`Engine::fire`](engine::Engine::fire) gave.
///
/// Which outcome a caller passes is the whole of the run-ahead policy:
/// [`TERMINAL_OUTCOME_RETRY`] for a [scheduling](engine::Error::is_scheduling)
/// refusal (the work item is still alive and its next attempt resets the
/// cell); [`TERMINAL_OUTCOME_FAILED`] for anything else.
pub fn settle(cells: &[*mut TerminalCell], outcome: TerminalOutcomeCode) {
    for &cell in cells {
        if cell.is_null() {
            continue;
        }
        // SAFETY: the cell belongs to a `WorkItemCompletion` this submission
        // holds a clone of for the whole of the call, and `publish` is a
        // release store into an `AtomicU32`.
        unsafe {
            (*cell).publish(outcome);
        }
    }
}


// Settlement through the broker, for engines that answer before the device
// is done.

/// One admitted frame's settlement, waiting for the device. Registered by
/// the engine lane the instant `submit` answers `Ok`, resolved by the
/// engine's own completion callback (for the CUDA shell, the driver's
/// host-function thread). Everything here is an atomic, a `Vec` read under
/// one briefly-held lock, or a waker publish.
struct PendingFrame {
    /// How many of the frame's steps have not reported yet.
    remaining: usize,
    /// Every work item the frame carries, across all its steps — published
    /// once, when the last step reports.
    cells: Vec<*mut TerminalCell>,
    /// The frame's own completion cell: what the engine lane's
    /// `SubmissionCompletion` is parked on, so run-ahead depth accounting
    /// retires the batch only when the device is actually done with it.
    frame_cell: *mut TerminalCell,
    /// Its waker slot, and the epoch to publish there.
    wait_id: u64,
    epoch: u64,
    /// The adopted channels' reader slots: waking them here (rather than at
    /// submit-return) matters because the guest must not be told to read a
    /// cell the device has not written yet.
    wakes: Vec<u64>,
}

// SAFETY: the pointers are `TerminalCell` addresses owned by
// `WorkItemCompletion`s the scheduler holds for the whole life of the frame
// (the same lifetime argument `settle` makes), and every access through them
// is a release store into an `AtomicU32`. The waker ids are integers.
unsafe impl Send for PendingFrame {}

/// The engine lane's book of frames the device has not finished. One per
/// engine lane, shared with the completion sink that lane installs on its
/// engine: `FrameTicket::id` in, the frame's terminal cells and wakes out.
#[derive(Default)]
pub struct FrameSettlements {
    inner: Mutex<Book>,
}

/// The race this book exists to lose safely: `submit` returns with the
/// device already running, so a step's callback can arrive before the lane
/// has registered the frame it belongs to. Completions for a frame nobody is
/// expecting yet are counted in [`Book::early`] instead of dropped, and
/// [`FrameSettlements::expect`] subtracts what already arrived.
#[derive(Default)]
struct Book {
    frames: std::collections::HashMap<u64, PendingFrame>,
    /// Frame id → (completions seen, whether any of them faulted), for frames
    /// the lane has not registered yet.
    early: std::collections::HashMap<u64, (usize, bool)>,
}

impl FrameSettlements {
    /// A fresh book.
    #[must_use]
    pub fn new() -> Arc<FrameSettlements> {
        Arc::new(FrameSettlements::default())
    }

    /// Register an admitted frame. Called on the engine lane, after `submit`
    /// answered `Ok`. `steps` is how many completions to expect; the cells
    /// are published when the last one arrives.
    pub fn expect(
        &self,
        frame: u64,
        steps: usize,
        cells: Vec<*mut TerminalCell>,
        wakes: Vec<u64>,
        completion: &SubmissionCompletion,
        broker: &CompletionBroker,
    ) {
        let Some(frame_cell) = completion.terminal_cell_ptr() else {
            return;
        };
        let pending = PendingFrame {
            remaining: steps.max(1),
            cells,
            frame_cell,
            wait_id: completion.wait_id(),
            epoch: completion.target_epoch(),
            wakes,
        };
        let resolved = {
            let mut book = self.inner.lock().unwrap();
            // What already arrived while the lane was still on its way here.
            let (seen, failed) = book.early.remove(&frame).unwrap_or((0, false));
            let mut pending = pending;
            pending.remaining = pending.remaining.saturating_sub(seen);
            if pending.remaining > 0 && !failed {
                book.frames.insert(frame, pending);
                None
            } else {
                Some((pending, failed))
            }
        };
        if let Some((pending, failed)) = resolved {
            publish(&pending, failed, broker);
        }
    }

    /// Forget a frame nobody will complete — the poison path. A frame whose
    /// step k faulted has its terminal cells published failed by the lane;
    /// dropping the registration makes the still-coming completions of
    /// steps `0..k` harmless instead of reporting success after failure.
    pub fn forget(&self, frame: u64) {
        let mut book = self.inner.lock().unwrap();
        book.frames.remove(&frame);
        book.early.remove(&frame);
    }

    /// One step of one frame has completed. Called from the engine's
    /// settlement thread. Publishes nothing until the frame's last step
    /// reports, except on a fault, which resolves the frame immediately.
    pub fn settled(&self, frame: u64, outcome: &engine::StepOutcome, broker: &CompletionBroker) {
        let failed = matches!(outcome, engine::StepOutcome::Faulted(_));
        let pending = {
            let mut book = self.inner.lock().unwrap();
            let Some(pending) = book.frames.get_mut(&frame) else {
                // Either already resolved, or ahead of its registration: the
                // completion is banked here and `expect` will subtract it.
                let seen = book.early.entry(frame).or_insert((0, false));
                seen.0 += 1;
                seen.1 |= failed;
                return;
            };
            pending.remaining = pending.remaining.saturating_sub(1);
            if !failed && pending.remaining > 0 {
                return;
            }
            book.frames.remove(&frame).expect("present just above")
        };
        publish(&pending, failed, broker);
    }

    /// Fail every frame still on the books — teardown, and the panic path.
    pub fn close_all(&self, broker: &CompletionBroker) {
        let frames: Vec<PendingFrame> = {
            let mut book = self.inner.lock().unwrap();
            book.early.clear();
            book.frames.drain().map(|(_, f)| f).collect()
        };
        for pending in frames {
            publish(&pending, true, broker);
        }
    }
}

/// Resolve one frame: its work items, then its own cell, then the guests.
/// The order matters: publishing the frame cell before its work items would
/// let the scheduler read a cell nobody had written, and a guest wake
/// promises its cell is readable only after both.
fn publish(pending: &PendingFrame, failed: bool, broker: &CompletionBroker) {
    let code = if failed {
        TERMINAL_OUTCOME_FAILED
    } else {
        TERMINAL_OUTCOME_SUCCESS
    };
    settle(&pending.cells, code);
    settle(&[pending.frame_cell], code);
    broker.notify(pending.wait_id, pending.epoch);
    for wait_id in &pending.wakes {
        waker::WakerTable::global().wake(*wait_id);
    }
}
