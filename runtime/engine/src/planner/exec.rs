//! The planner's transfer executors — eviction (D2H) and restore (H2D),
//! run on the planner's own tasks. Nothing here negotiates with the victim:
//! quiescence is ESTABLISHED, not requested.
//!
//! Eviction sequence (each step's failure abandons the attempt; the process
//! stays resident and the next poke re-plans):
//!
//! 1. The residency gate is already closed (state = `Evicting`, flipped by
//!    the planner before spawning), so no new WIT call enters.
//! 2. **Fence** every working set (an atomic on the set's lifecycle, Dekker-
//!    paired with the fire lease): no new fire can lease, prepare, or submit
//!    against the victim's sets from here on.
//! 3. The lane leaves the wait-all quorum (`Close`: the submitted tail
//!    drains untracked; frames seal without the victim immediately).
//! 4. **Drain** the victim's pipeline FIFOs detachably — finalizing the
//!    already-settled host ops releases their fire leases. Unsettled or
//!    non-detachable entries are LEFT for the victim's own yielded task to
//!    settle (see `drain_detachable`); step 5's quiescence is the gate.
//! 5. **Quiesce**: await zero fire leases per set. With the fence up the
//!    count is monotone non-increasing, so this is a finite wait for the
//!    mid-build stragglers that raced step 2 — the exact seal against a
//!    launch writing pages while the copy reads them.
//! 6. prepare → D2H → commit, transactional (`KvSuspendTxn` + guard).
//!
//! Fences stay up for the whole evicted period (the gate parks the guest,
//! but a straggler op already past the gate must still bounce) and clear at
//! restore commit — or at working-set release, which ends the lifecycle the
//! fence lives on.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use super::{ProcessId, ResidencyPlanner};

use crate::store::kv::page_table::WorkingSetId;
use crate::store::kv::working_set::KvSuspendHandle;
use crate::store::kv::{KvRestoreTxn, KvSuspendPrepare, KvSuspendTxn};

/// Spawn one executor plus its join watcher. A panicking executor must
/// never strand its in-flight mark (Evicting gates all further eviction
/// planning; Restoring gates the queue head): the watcher fails the attempt
/// on an abnormal join. Returns false when no runtime exists — the unpolled
/// future drops (releasing anything it captured) and the caller runs its
/// no-runtime fallback.
fn spawn_watched(
    planner: Arc<ResidencyPlanner>,
    pid: ProcessId,
    label: &'static str,
    task: impl std::future::Future<Output = ()> + Send + 'static,
    on_fail: impl FnOnce(&Arc<ResidencyPlanner>, ProcessId) + Send + 'static,
) -> bool {
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        return false;
    };
    let handle = runtime.spawn(task);
    runtime.spawn(async move {
        if let Err(join_error) = handle.await {
            println!("[planner-exec] pid={pid} {label} task DIED: {join_error}");
            on_fail(&planner, pid);
        }
    });
    true
}

pub(super) fn spawn_evict(planner: Arc<ResidencyPlanner>, pid: ProcessId) {
    let task = evict(planner.clone(), pid);
    let spawned = spawn_watched(planner.clone(), pid, "evict", task, |planner, pid| {
        planner.eviction_failed(pid)
    });
    if !spawned {
        planner.eviction_failed(pid);
    }
}

pub(super) fn spawn_restore(
    planner: Arc<ResidencyPlanner>,
    pid: ProcessId,
    pages: super::grant::DevicePageReservation,
) {
    // On the no-runtime path the unpolled future drops `pages` back to the
    // pool; the entry re-queues. Nothing was attempted, so that one is worth
    // a retry — an abnormal join is not: a panicking executor panics again.
    let task = restore(planner.clone(), pid, pages);
    let spawned = spawn_watched(planner.clone(), pid, "restore", task, |planner, pid| {
        planner.restore_failed(pid, "restore executor died")
    });
    if !spawned {
        planner.restore_deferred(pid, "no tokio runtime for the restore executor");
    }
}

/// Unfences on drop — armed for every abandon path, disarmed on the one
/// path where the fences must OUTLIVE the task (a committed eviction).
struct FenceGuard {
    handles: Vec<KvSuspendHandle>,
    armed: bool,
}

impl FenceGuard {
    fn raise(handles: Vec<KvSuspendHandle>) -> Self {
        for handle in &handles {
            handle.fence();
        }
        Self {
            handles,
            armed: true,
        }
    }

    fn keep_raised(&mut self) {
        self.armed = false;
    }
}

impl Drop for FenceGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        for handle in &self.handles {
            handle.unfence();
        }
    }
}

enum ResidencyTxn {
    Suspend(KvSuspendTxn),
    Restore(KvRestoreTxn),
}

/// Abort a residency transaction against its store — the one rollback
/// dispatch, shared by the sync path (guard drop with no copy in flight)
/// and the deferred copy-completion path.
fn abort_residency_txn(model: usize, driver: usize, txn: ResidencyTxn) {
    let stores = crate::store::registry::get(model, driver);
    let tag = match &txn {
        ResidencyTxn::Suspend(_) => "planner-evict",
        ResidencyTxn::Restore(_) => "planner-restore",
    };
    crate::store::registry::with_kv_lock(&stores.kv, tag, |kv| match txn {
        ResidencyTxn::Suspend(txn) => kv.abort_suspend(txn),
        ResidencyTxn::Restore(txn) => kv.abort_restore(txn),
    });
}

struct ResidencyTxnGuard {
    model: usize,
    driver: usize,
    txn: Option<ResidencyTxn>,
    completion: Option<crate::scheduler::ControlCompletion>,
}

impl ResidencyTxnGuard {
    fn suspend(model: usize, driver: usize, txn: KvSuspendTxn) -> Self {
        Self {
            model,
            driver,
            txn: Some(ResidencyTxn::Suspend(txn)),
            completion: None,
        }
    }

    fn restore(model: usize, driver: usize, txn: KvRestoreTxn) -> Self {
        Self {
            model,
            driver,
            txn: Some(ResidencyTxn::Restore(txn)),
            completion: None,
        }
    }

    fn arm(&mut self, completion: crate::scheduler::ControlCompletion) {
        self.completion = Some(completion);
    }

    fn disarm_completion(&mut self) {
        self.completion = None;
    }

    fn take_suspend(&mut self) -> KvSuspendTxn {
        match self.txn.take().expect("suspend transaction present") {
            ResidencyTxn::Suspend(txn) => txn,
            ResidencyTxn::Restore(_) => unreachable!("suspend guard carries suspend txn"),
        }
    }

    fn take_restore(&mut self) -> KvRestoreTxn {
        match self.txn.take().expect("restore transaction present") {
            ResidencyTxn::Restore(txn) => txn,
            ResidencyTxn::Suspend(_) => unreachable!("restore guard carries restore txn"),
        }
    }

    /// The driver copy plan `(gpu_ids, host_slots)` of the held transaction.
    fn copy_plan(&self) -> (Vec<u32>, Vec<u32>) {
        match self.txn.as_ref().expect("transaction present") {
            ResidencyTxn::Suspend(txn) => (txn.gpu_ids(), txn.host_slots()),
            ResidencyTxn::Restore(txn) => (txn.gpu_ids(), txn.host_slots()),
        }
    }

    fn abort_now(&mut self) {
        if let Some(txn) = self.txn.take() {
            abort_residency_txn(self.model, self.driver, txn);
        }
    }
}

impl Drop for ResidencyTxnGuard {
    fn drop(&mut self) {
        let Some(txn) = self.txn.take() else {
            return;
        };
        let Some(completion) = self.completion.take() else {
            abort_residency_txn(self.model, self.driver, txn);
            return;
        };
        let (model, driver) = (self.model, self.driver);
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            tracing::error!(
                model,
                driver,
                "KV residency transaction dropped with a driver copy in flight and no runtime; \
                 preserving its pages and slots to avoid reuse during the copy"
            );
            return;
        };
        runtime.spawn(async move {
            let _ = completion.wait().await;
            abort_residency_txn(model, driver, txn);
            if let Some(planner) = crate::planner::planner_for(model, driver) {
                planner.pages_freed();
            }
        });
    }
}

/// Best-effort detachable drain of `pid`'s pipeline FIFOs — SETTLED host-KV
/// ops only. The planner must NEVER await an unsettled fire completion: the
/// waker table parks ONE waker per slot, and the owning guest may be (or
/// later start) awaiting the same completion from `drain_pipeline_fires`
/// or a channel materialize — the second registration would overwrite the
/// first and strand whichever task lost (the lost-wakeup wedge behind the
/// 2026-07-25 bench freezes). Finalizing a SETTLED op never registers a
/// waker (its poll completes on the first check). Everything unsettled or
/// non-detachable is left to the guest's own settle path, and the lease
/// quiescence wait below is the actual correctness gate.
async fn drain_detachable(pid: ProcessId) {
    let pipelines = crate::inferlet::process::residency::pipelines_of(pid);
    for fires in pipelines {
        let Some(_finalize_guard) = fires.try_finalize_guard() else {
            continue;
        };
        loop {
            let op = {
                let mut queue = fires.lock().unwrap();
                match queue.front() {
                    Some(op) if op.is_preemption_detachable() && op.is_settled() => {
                        queue.pop_front()
                    }
                    _ => None,
                }
            };
            let Some(op) = op else {
                break;
            };
            if let Err(error) = crate::pipeline::fire::finalize_op_detached(op).await {
                tracing::warn!(pid = %pid, %error, "planner: detachable finalize failed");
                break;
            }
        }
    }
}

async fn evict(planner: Arc<ResidencyPlanner>, pid: ProcessId) {
    let (model, driver) = planner.locus();
    let handles = crate::inferlet::process::residency::kv_suspend_handles(pid, model, driver);
    let working_sets: HashSet<WorkingSetId> =
        crate::inferlet::process::residency::kv_working_set_ids(pid, model, driver);
    if working_sets.is_empty() {
        planner.eviction_failed(pid);
        return;
    }
    // Step 2: fences up — no new fire can lease/prepare against these sets.
    let mut fence = FenceGuard::raise(handles);
    // Step 3: EVERY lane the victim owns leaves the wait-all quorum
    // (process-wide — the planner does not know pipeline-scope lane ids),
    // while its submitted tail stays sealable and drains untracked; those
    // finalizations release the leases step 5 waits on. A lane-keyed Close
    // here would match nothing and wedge the fleet's next boundary on the
    // victim's silent lane. Rejoin is implicit on the next accepted fire
    // (post-restore).
    crate::scheduler::worker::notify_process_suspend(pid);
    // Step 4: settle what the planner can (host-KV ops); the victim's own
    // yielded fire task settles the rest. Quiescence below is the gate.
    drain_detachable(pid).await;
    // Step 5: lease quiescence — the seal against mid-build stragglers.
    for handle in fence.handles.iter() {
        handle.quiesce().await;
    }
    // Step 6: prepare → D2H → commit.
    let stores = crate::store::registry::get(model, driver);
    let prepared = crate::store::registry::with_kv_lock(&stores.kv, "planner-evict", |kv| {
        kv.prepare_suspend(&working_sets)
    });
    let txn = match prepared {
        Ok(KvSuspendPrepare::Prepared(txn)) => txn,
        Ok(KvSuspendPrepare::Deferred(_)) => {
            planner.eviction_failed_prepare_deferred(pid);
            return;
        }
        Err(error @ crate::store::kv::KvStoreError::HostSwapFull { .. }) => {
            // No kill rung here: without swap room this victim cannot move,
            // and the head waits for completions instead. Park the victim so
            // the deterministic re-pick cannot spin on it (§20.6).
            tracing::warn!(pid = %pid, %error, "planner: eviction blocked on host swap");
            planner.eviction_failed_host_swap_full(pid);
            return;
        }
        Err(error) => {
            tracing::warn!(pid = %pid, %error, "planner: suspend prepare failed");
            planner.eviction_failed(pid);
            return;
        }
    };
    let mut suspend = ResidencyTxnGuard::suspend(model, driver, txn);
    let (gpu_ids, host_slots) = suspend.copy_plan();
    let copy_started = Instant::now();
    let completion = match crate::scheduler::copy_d2h_tracked(driver, &gpu_ids, &host_slots) {
        Ok(completion) => completion,
        Err(error) => {
            suspend.abort_now();
            tracing::warn!(pid = %pid, %error, "planner: KV D2H eviction copy rejected");
            planner.eviction_failed(pid);
            return;
        }
    };
    suspend.arm(completion.clone());
    let copied = completion.wait().await;
    planner.record_d2h_copy(copy_started.elapsed());
    suspend.disarm_completion();
    if let Err(error) = copied {
        suspend.abort_now();
        tracing::warn!(pid = %pid, %error, "planner: KV D2H eviction copy failed");
        planner.eviction_failed(pid);
        return;
    }
    let txn = suspend.take_suspend();
    let freed = crate::store::registry::with_kv_lock(&stores.kv, "planner-evict", |kv| {
        kv.commit_suspend(txn)
    });
    match freed {
        Ok(freed) => {
            // The fences outlive the task: the evicted period keeps them up,
            // restore (or working-set release) clears them.
            fence.keep_raised();
            planner.report_evicted(pid, freed as u32);
        }
        Err(error) => {
            tracing::warn!(pid = %pid, %error, "planner: suspend commit failed");
            planner.eviction_failed(pid);
        }
    }
}

async fn restore(
    planner: Arc<ResidencyPlanner>,
    pid: ProcessId,
    mut pages: super::grant::DevicePageReservation,
) {
    let (model, driver) = planner.locus();
    // Unfencing on the success paths is owned by `report_restored`
    // ("restored ⇒ unfenced", one site); failure paths keep the fences up.
    let working_sets: HashSet<WorkingSetId> =
        crate::inferlet::process::residency::kv_working_set_ids(pid, model, driver);
    if working_sets.is_empty() {
        // Everything was released while evicted: trivially resident again.
        drop(pages);
        planner.report_restored(pid, 0);
        return;
    }
    let stores = crate::store::registry::get(model, driver);
    let prepared = crate::store::registry::with_kv_lock(&stores.kv, "planner-restore", |kv| {
        kv.prepare_restore(&working_sets, pages.lend())
    });
    // Surplus pages (the swapped count shrank since serve) return via Drop.
    drop(pages);
    let txn = match prepared {
        Ok(txn) => txn,
        Err(error) => {
            // Worth one more round: `Step::ServeRestore` rebuilds the ask
            // from what is swapped now, so a short grant is not short twice.
            planner.restore_deferred(pid, &error.to_string());
            return;
        }
    };
    if txn.page_count() == 0 {
        // Nothing left on swap (discarded/released while evicted): commit
        // the empty transaction and rejoin.
        let committed = crate::store::registry::with_kv_lock(&stores.kv, "planner-restore", |kv| {
            kv.commit_restore(txn)
        });
        match committed {
            Ok(_) => planner.report_restored(pid, 0),
            // Page-table bookkeeping, not transport: it fails identically
            // on a second commit of the same empty transaction.
            Err(error) => planner.restore_failed(pid, &error.to_string()),
        }
        return;
    }
    let mut restore = ResidencyTxnGuard::restore(model, driver, txn);
    let (gpu_ids, host_slots) = restore.copy_plan();
    let copy_started = Instant::now();
    let completion = match crate::scheduler::copy_h2d_tracked(driver, &gpu_ids, &host_slots) {
        Ok(completion) => completion,
        Err(error) => {
            restore.abort_now();
            // Provisionally retryable: a submit that fails once may be a
            // transient driver refusal rather than a dead context. Nobody has
            // measured which, so this keeps the benefit of the doubt — for
            // exactly one attempt.
            planner.restore_deferred(pid, &format!("H2D submit: {error:#}"));
            return;
        }
    };
    restore.arm(completion.clone());
    let copied = completion.wait().await;
    planner.record_h2d_copy(copy_started.elapsed());
    restore.disarm_completion();
    if let Err(error) = copied {
        restore.abort_now();
        // Same benefit of the doubt as the submit above.
        planner.restore_deferred(pid, &format!("H2D copy: {error}"));
        return;
    }
    let txn = restore.take_restore();
    let restored = crate::store::registry::with_kv_lock(&stores.kv, "planner-restore", |kv| {
        kv.commit_restore(txn)
    });
    match restored {
        Ok(restored) => {
            planner.report_restored(pid, restored as u32);
            crate::scheduler::nudge(driver);
        }
        Err(error) => {
            // The copy landed; only the page table refused. Deterministic.
            planner.restore_failed(pid, &error.to_string());
        }
    }
}
