//! The per-process contention state machine — one transition function, one
//! writer site. Every state change goes through [`ProcState::apply`], which
//! consumes the prior state and either returns the successor or reports the
//! edge as illegal; nothing else assigns the field. The safe point is a
//! STATE, not a place (D2): "pin-free" is what makes a process suspendable,
//! and the states below only track where it stands in the park protocol.

use std::sync::Arc;

use tokio::sync::Notify;

/// Lifecycle of one registered process, as the orchestrator sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ProcState {
    /// Runnable; no park interaction pending.
    Running,
    /// A park request is posted; honored at the process's next safe point
    /// (every fire prologue, every idle-await entry).
    ParkRequested,
    /// The process committed to parking (`begin_quiesce`) and is draining to
    /// its safe point.
    Quiescing,
    /// KV state saved to host swap; a restore entry sits in the queue.
    Suspended,
    /// A restore grant is issued and being applied (H2D + republication).
    Restoring,
}

/// The events that drive [`ProcState::apply`].
#[derive(Clone, Copy, Debug)]
pub(super) enum ProcEvent {
    /// The orchestrator selected this process as a victim.
    RequestPark,
    /// The process accepted the park request at a safe point.
    BeginQuiesce,
    /// The process (or pressure relief) withdrew the park request.
    DeclinePark,
    /// The process saved its state and freed its pages.
    ReportSuspended,
    /// The drain issued this process's restore grant.
    GrantRestore,
    /// H2D + mapping publication completed; the process is runnable.
    ReportRestored,
    /// The restore attempt failed; the process re-queues as suspended.
    RestoreFailed,
}

impl ProcState {
    /// The state machine, in one exhaustive match. `Err` is the unchanged
    /// state: the edge is illegal and the caller decides whether that is a
    /// benign race (most are — e.g. a decline racing a grant) or a bug.
    pub(super) fn apply(self, event: ProcEvent) -> Result<ProcState, ProcState> {
        use ProcEvent::*;
        use ProcState::*;
        match (self, event) {
            (Running, RequestPark) => Ok(ParkRequested),
            (ParkRequested, BeginQuiesce) => Ok(Quiescing),
            (ParkRequested | Quiescing, DeclinePark) => Ok(Running),
            (ParkRequested | Quiescing, ReportSuspended) => Ok(Suspended),
            (Suspended, GrantRestore) => Ok(Restoring),
            (Restoring, ReportRestored) => Ok(Running),
            (Restoring, RestoreFailed) => Ok(Suspended),
            (state, _) => Err(state),
        }
    }
}

/// One registered process.
pub(super) struct Proc {
    /// Spawn-order clock — the single, authoritative FCFS key.
    pub seq: u64,
    pub state: ProcState,
    /// Pages freed when this process suspended — its restore ask, kept here
    /// so a failed restore can re-queue with the same demand.
    pub restore_demand: u32,
    /// Self-reported: parked in a long idle host await (a permanent safe
    /// point). Victim selection prefers idle holders (D6) — suspending one
    /// costs no running lane.
    pub idle: bool,
    /// A progress-deadline kill was dispatched for this process (D7); it is
    /// excluded from further victim and kill selection while its teardown
    /// runs.
    pub killed: bool,
    /// Wakes the process's safe-point watchers when a park request lands or
    /// clears, and its state-transition waiters.
    pub signal: Arc<Notify>,
}

impl Proc {
    pub fn new(seq: u64) -> Self {
        Self {
            seq,
            state: ProcState::Running,
            restore_demand: 0,
            idle: false,
            killed: false,
            signal: Arc::new(Notify::new()),
        }
    }
}
