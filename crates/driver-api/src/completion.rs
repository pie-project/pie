//! What a driver signals with, and what the engine waits on.
//!
//! # Why this is contract and not scheduler internals
//!
//! It was `engine::driver::completion`, and the seam headers beside it said
//! why it should stay there: *"the shell is the driver's state; the broker is
//! the engine's, because a completion is what a SCHEDULER waits on and a
//! driver has no opinion about it."*
//!
//! Half of that is true and the other half is contradicted by every backend
//! in the tree. A driver does not WAIT on a completion — but it MINTS one:
//! `Driver::launch` answers a [`SubmissionCompletion`], and `copy_kv`,
//! `copy_state`, `resize_pool` and `encode` all answer one too. A type that
//! appears in five of a trait's fourteen return positions is that trait's
//! vocabulary, and a trait cannot be stated in a crate that cannot name its
//! own return types.
//!
//! So the broker moved here with the trait. What stayed in `engine` is what
//! actually is the scheduler's: which completion belongs to which work item,
//! and what to do when one settles.
//!
//! # What went with the C boundary
//!
//! `PieRuntimeCallbacks` — an `abi_version`, a `*mut c_void` context and an
//! `unsafe extern "C" fn(ctx, wait_id, epoch)` — was how a C++ driver told
//! the runtime that something had finished. Handing a Rust driver a function
//! pointer and an erased context to call back into a `CompletionBroker` it
//! could have simply been given is a round trip through C for no reader:
//! [`CompletionBroker`] is `Clone` and `Send`, and a driver holds one.
//!
//! `PieCompletion` went the same way and left [`CompletionTarget`] in its
//! place — the same three fields with the `#[repr(C)]` and the null-pointer
//! contract taken off, because a driver that is handed one is handed it by
//! name.

use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::task::Poll;

use crate::local::{
    PIE_TERMINAL_OUTCOME_FAILED, PIE_TERMINAL_OUTCOME_PENDING, PIE_TERMINAL_OUTCOME_RETRY,
    PIE_TERMINAL_OUTCOME_SUCCESS, TerminalCell,
};
use anyhow::{Result, anyhow};
use crossbeam_queue::SegQueue;
use waker::{FIRST_COMPLETION_EPOCH, WakerSlotId, WakerTable};

/// Where a driver publishes one operation's outcome.
///
/// The owned counterpart of the `PieCompletion` descriptor: the same wait
/// slot, the same epoch, the same terminal cell. What it no longer has is
/// `#[repr(C)]` and a null-pointer convention, because it is handed to a Rust
/// driver by name rather than laid out for a foreign reader.
#[derive(Debug, Clone, Copy)]
pub struct CompletionTarget {
    /// The wait slot to publish on.
    pub wait_id: u64,
    /// The epoch to publish.
    pub target_epoch: u64,
    /// Where to publish the terminal outcome, or null for a frame launch
    /// (whose members carry their own cells).
    pub terminal_cell: *mut TerminalCell,
}

// SAFETY: the cell is an `AtomicU32` the driver publishes into and the engine
// reads; a shared reference to it is all either side ever takes.
unsafe impl Send for CompletionTarget {}
unsafe impl Sync for CompletionTarget {}


/// Keeps an instance's wait slots alive while a completion is in flight.
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

// No `unsafe impl Send`/`Sync` and no `UnsafeCell`: `TerminalCell::outcome`
// is an `AtomicU32`, which is both, and a shared reference to it is all either
// side ever needed. The wrapper survives only because the recycling pools are
// typed on it.

#[derive(Debug)]
struct OwnedTerminalCell {
    raw: Option<Box<TerminalCellStorage>>,
    recyclable: bool,
}

// Only cells whose last accepted native attempt has retired enter this pool.
// Cells dropped while native retirement is uncertain remain quarantined.
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

#[cfg(test)]
fn terminal_atomic_ptr(cell: *mut TerminalCell) -> *const AtomicU32 {
    cell.cast::<AtomicU32>()
}

/// Read a cell the test reached through a raw pointer — including a stale one
/// the ABA test deliberately keeps past its owner.
#[cfg(test)]
fn load_terminal_outcome(cell: *mut TerminalCell) -> TerminalOutcome {
    classify_terminal_outcome(unsafe { (*cell).load() })
}

fn classify_terminal_outcome(value: u32) -> TerminalOutcome {
    match value {
        PIE_TERMINAL_OUTCOME_PENDING => TerminalOutcome::Pending,
        PIE_TERMINAL_OUTCOME_SUCCESS => TerminalOutcome::Success,
        PIE_TERMINAL_OUTCOME_FAILED => TerminalOutcome::Failed,
        PIE_TERMINAL_OUTCOME_RETRY => TerminalOutcome::Retry,
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
                .unwrap_or_else(|| "driver submission completion closed".to_string())
        ))
    }

    fn terminal_result(&self) -> Option<Result<()>> {
        match &self.mode {
            SubmissionCompletionMode::WakeOnly => None,
            SubmissionCompletionMode::Terminal { cell } => match cell.load() {
                TerminalOutcome::Pending => None,
                TerminalOutcome::Success => Some(Ok(())),
                TerminalOutcome::Failed => Some(Err(anyhow!(
                    "driver operation published Failed terminal outcome"
                ))),
                TerminalOutcome::Retry => Some(Err(anyhow!(
                    "driver control operation published unexpected Retry terminal outcome"
                ))),
                TerminalOutcome::Invalid(value) => Some(Err(anyhow!(
                    "driver operation published invalid terminal outcome {value}"
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

/// Insert-only registry of outstanding completions, kept solely so
/// [`CompletionBroker::close_all`] can sweep them. Dropping a completion
/// leaves a dead `Weak` behind instead of taking this lock; dead entries are
/// compacted amortized on insert, bounding the vec at ~2x the live peak.
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

/// Mints the completions a driver answers with.
///
/// `Clone` and `Send`: a driver is handed one at create and keeps it for as
/// long as it can finish work. That is what replaced the `{ctx, notify}` pair
/// a C++ shell was handed — see the module header.
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
        // Insert-then-recheck: a `close_all` whose drain missed this insert
        // must have set `closed` before draining (both under the same lock),
        // so this load observes it and the state is closed here instead.
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

    /// Mint a completion for a control verb, with the terminal cell the
    /// driver publishes its outcome into.
    #[must_use]
    pub fn control_completion(&self, target_epoch: u64) -> (CompletionTarget, SubmissionCompletion) {
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

    /// Mint a completion for a frame launch.
    ///
    /// No terminal cell: a frame's outcome is published per member, through
    /// `StepSubmission::terminal_cells`, and one more cell here would be a
    /// second answer to a question the members already answer.
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

    /// Publish `epoch` on `wait_id`, waking whoever parked on it.
    ///
    /// This IS what the C `notify` callback did; a driver calls it directly.
    pub fn notify(&self, wait_id: u64, epoch: u64) {
        if !self.inner.closed.load(Ordering::Acquire) {
            let _ = self.inner.table.publish(wait_id, epoch);
        }
    }

    #[cfg(test)]
    fn live_len(&self) -> usize {
        self.inner.live.lock().unwrap().entries.len()
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
    /// Settles when EVERY part settles; the first error wins. Used by the
    /// remote edge adapter to fold a frame's per-step RPC completions into
    /// the frame's single completion.
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

    /// A completion that settles when every part settles (first error wins).
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

    /// Non-blocking probe: whether the completion has settled (successfully
    /// or not) without registering any waker.
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
                // Poll every unsettled part (clones share the settled state;
                // polling registers this task's waker on each pending slot).
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
        // The registry entry is a Weak that just went dead; it is compacted
        // lazily on a later insert rather than removed here.
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
            // Unknown until the scheduler either retires an accepted native
            // attempt or explicitly proves the request was never submitted.
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
        // Unconditional doorbell, NOT `publish(slot, 1)`: the resolution
        // lives in the atomic above, outside the epoch machinery, and the
        // driver may already have published a HIGHER epoch to this slot
        // (its per-fire terminal notify). A waiter that re-registered after
        // observing that epoch would filter a `publish(1)` as stale and
        // sleep forever — the lost wakeup behind "settled fire, hung
        // finalize" (found via the Rainer eviction drain, but reachable by
        // any await that straddles the driver-notify → worker-retire gap).
        let outcome = WakerTable::global().wake(self.slot);
        let _ = outcome;
    }

    fn resolve_failure(&self, message: impl Into<String>) {
        *self.message.lock().unwrap() = Some(message.into());
        self.resolution
            .store(WORK_ITEM_RESOLUTION_FAILED, Ordering::Release);
        // See `resolve_success`: unconditional doorbell.
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

    /// Non-blocking probe: whether this work item completion has resolved.
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

    fn pool_contains(
        pool: &SegQueue<Box<TerminalCellStorage>>,
        pointer: *mut TerminalCell,
    ) -> bool {
        let mut drained = Vec::new();
        let mut found = false;
        while let Some(cell) = pool.pop() {
            found |= ptr::from_ref(&cell.0).cast_mut() == pointer;
            drained.push(cell);
        }
        for cell in drained {
            pool.push(cell);
        }
        found
    }

    #[test]
    fn terminal_cells_recycle_only_after_native_attempt_retirement() {
        let retired = WorkItemCompletion::new(WakerTable::global().alloc(), 0);
        let retired_pointer = retired.terminal_cell_ptr();
        retired.commit_target_epoch(FIRST_COMPLETION_EPOCH);
        retired.mark_native_retired();
        drop(retired);
        assert!(pool_contains(terminal_cell_pool(), retired_pointer));

        let uncertain = WorkItemCompletion::new(WakerTable::global().alloc(), 0);
        let uncertain_pointer = uncertain.terminal_cell_ptr();
        uncertain.commit_target_epoch(FIRST_COMPLETION_EPOCH);
        drop(uncertain);
        assert!(!pool_contains(terminal_cell_pool(), uncertain_pointer));
        assert!(pool_contains(terminal_cell_quarantine(), uncertain_pointer));
    }

    fn counter_waker() -> (Arc<CountWaker>, Waker) {
        let count = Arc::new(CountWaker(AtomicUsize::new(0)));
        (Arc::clone(&count), Waker::from(count))
    }

    fn store_terminal(cell: *mut TerminalCell, outcome: u32) {
        unsafe { (&*terminal_atomic_ptr(cell)).store(outcome, Ordering::Release) };
    }

    #[test]
    fn foreign_thread_publish_completes_waiter() {
        let broker = CompletionBroker::new();
        let (raw, completion) = broker.launch_completion(3);
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        // The broker itself crosses the thread, which is what a driver does
        // with it. This used to send a `usize`-laundered `*mut c_void` and
        // call the `extern "C"` notify through it -- the erasure was the C
        // boundary's, and what it was erasing was a `Clone + Send` handle.
        let far_side = broker.clone();
        let thread = std::thread::spawn(move || far_side.notify(raw.wait_id, raw.target_epoch));
        rt.block_on(async { completion.await.unwrap() });
        thread.join().unwrap();
    }

    #[test]
    fn control_completion_checks_terminal_outcome() {
        let broker = CompletionBroker::new();
        let (raw, completion) = broker.control_completion(1);
        store_terminal(raw.terminal_cell, PIE_TERMINAL_OUTCOME_FAILED);
        broker.notify(raw.wait_id, 1);
        let err = completion
            .check()
            .expect("terminal failure should settle")
            .unwrap_err();
        assert!(err.to_string().contains("Failed terminal outcome"));
    }

    #[test]
    fn control_completion_retains_cell_until_callback() {
        let broker = CompletionBroker::new();
        let (raw, completion) = broker.control_completion(1);
        store_terminal(raw.terminal_cell, PIE_TERMINAL_OUTCOME_SUCCESS);
        assert!(
            completion.check().is_none(),
            "terminal publication alone must not retire callback-owned storage"
        );
        broker.notify(raw.wait_id, 1);
        completion
            .check()
            .expect("callback should settle completion")
            .unwrap();
    }

    #[test]
    fn control_completion_rejects_callback_before_terminal_publication() {
        let broker = CompletionBroker::new();
        let (raw, completion) = broker.control_completion(1);
        broker.notify(raw.wait_id, 1);
        let err = completion
            .check()
            .expect("callback should settle completion")
            .unwrap_err();
        assert!(err.to_string().contains("terminal outcome settled"));
    }

    #[test]
    fn stale_generation_callback_is_ignored() {
        let broker = CompletionBroker::new();
        let (raw, completion) = broker.control_completion(2);
        let stale = raw.wait_id;
        drop(completion);
        assert!(matches!(
            WakerTable::global().publish(stale, 99),
            waker::WakeOutcome::Stale
        ));
    }

    #[test]
    fn dropped_future_before_callback_is_safe() {
        let broker = CompletionBroker::new();
        let (raw, completion) = broker.control_completion(2);
        let stale = raw.wait_id;
        drop(completion);
        broker.notify(stale, raw.target_epoch);
        assert!(matches!(
            WakerTable::global().publish(stale, 99),
            waker::WakeOutcome::Stale
        ));
    }

    #[test]
    fn late_terminal_store_cannot_aba_a_new_completion() {
        let first = WorkItemCompletion::deferred_with_guard(None);
        let stale = first.terminal_cell_ptr();
        drop(first);
        let second = WorkItemCompletion::deferred_with_guard(None);
        assert_ne!(stale, second.terminal_cell_ptr());
        store_terminal(stale, PIE_TERMINAL_OUTCOME_SUCCESS);
        assert_eq!(
            load_terminal_outcome(second.terminal_cell_ptr()),
            TerminalOutcome::Pending
        );
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
        let (raw, completion) = broker.launch_completion(5);
        broker.close_all("driver closed");
        broker.notify(raw.wait_id, raw.target_epoch);
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
    fn completion_created_after_close_resolves_with_close_error() {
        let broker = CompletionBroker::new();
        broker.close_all("driver closed");
        let completion = broker.launch_completion(1).1;
        let err = completion
            .check()
            .expect("completion created after close must settle, not hang")
            .unwrap_err();
        assert!(err.to_string().contains("driver closed"));
    }

    #[test]
    fn live_registry_stays_bounded_across_create_drop_cycles() {
        let broker = CompletionBroker::new();
        for _ in 0..10_000 {
            drop(broker.launch_completion(1));
        }
        assert!(
            broker.live_len() <= 64,
            "dead weak entries must be compacted, got {}",
            broker.live_len()
        );
    }

    #[test]
    fn publish_racing_with_registration_wakes_or_fast_paths() {
        let broker = CompletionBroker::new();
        let (raw, completion) = broker.launch_completion(5);
        let (count, waker) = counter_waker();
        let mut completion = Box::pin(completion);
        let mut cx = std::task::Context::from_waker(&waker);
        assert!(matches!(completion.as_mut().poll(&mut cx), Poll::Pending));
        broker.notify(raw.wait_id, 5);
        assert!(count.0.load(Ordering::SeqCst) <= 1);
    }

    #[test]
    fn ready_completion_is_immediately_ready() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            SubmissionCompletion::ready().await.unwrap();
            let err = SubmissionCompletion::failed("boom")
                .await
                .expect_err("ready failed completion should surface immediately");
            assert!(err.to_string().contains("boom"));
        });
    }

    #[test]
    fn work_item_completion_reports_terminal_failure() {
        let completion = Box::pin(WorkItemCompletion::deferred_with_guard(None));
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

    #[test]
    fn retry_terminal_keeps_the_logical_completion_pending_and_resets_the_cell() {
        let completion = WorkItemCompletion::deferred_with_guard(None);
        store_terminal(completion.terminal_cell_ptr(), PIE_TERMINAL_OUTCOME_RETRY);
        assert_eq!(
            completion.resolve_from_terminal().unwrap(),
            WorkItemAttemptOutcome::Retry
        );
        assert!(!completion.is_settled());
        assert_eq!(
            load_terminal_outcome(completion.terminal_cell_ptr()),
            TerminalOutcome::Pending
        );
    }
}
