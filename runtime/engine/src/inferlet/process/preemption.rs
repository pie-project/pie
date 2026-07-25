//! Process-owned KV preemption lifecycle — the ONE safe point (D2/D3).
//!
//! WIT bindings delegate here; this module freezes scheduler preparation,
//! drains process fire queues, executes transactional D2H/H2D, and parks the
//! WASM continuation without putting domain orchestration in `inferlet/host`.
//!
//! The contract: eligibility to suspend is a STATE ("holds no pins"), not a
//! place. `yield_point` is the fire-prologue safe point every WIT host
//! method passes through; `wait_yielding` turns every long host await into
//! a STANDING safe point (and `fire::acquire_grant` treats the grant wait
//! the same way); both funnel into one park protocol whose halves are
//! `suspend_at_safe_point` (prepare → D2H → commit → report, with the
//! host-full kill rung) and `restore_from_park` (park → prepare → H2D →
//! commit → report, bounded retries). `ParkGuard` declines the park on
//! every early exit, so refusal is a scope exit, never a hand-written
//! bail-out.

use std::collections::HashSet;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result};

use crate::inferlet::ProcessCtx;
use crate::pipeline::fire::{PendingFires, PendingOp};
use crate::store::kv::page_table::WorkingSetId;
use crate::store::kv::{KvRestoreTxn, KvSuspendPrepare, KvSuspendTxn, SuspendDisposition};

struct TeardownFireContext {
    process_id: uuid::Uuid,
    resources: wasmtime::component::ResourceTable,
    // Dropped after `resources` (field declaration order), so strict
    // admission advances only after pooled resources are released.
    _execution_permit: Option<tokio::sync::OwnedSemaphorePermit>,
    _bind_permit: Option<tokio::sync::OwnedSemaphorePermit>,
}

impl crate::pipeline::fire::FireContext for TeardownFireContext {
    fn resources(&mut self) -> &mut wasmtime::component::ResourceTable {
        &mut self.resources
    }

    fn process_id(&self) -> uuid::Uuid {
        self.process_id
    }

    async fn honor_preemption(&mut self) -> Result<()> {
        Ok(())
    }

    fn preemption_signal(&self) -> Option<Arc<tokio::sync::Notify>> {
        None
    }
}

pub(crate) fn defer_resource_teardown(
    process_id: uuid::Uuid,
    resources: wasmtime::component::ResourceTable,
    residency: Arc<Mutex<crate::inferlet::process::ProcessResidency>>,
    execution_permit: Option<tokio::sync::OwnedSemaphorePermit>,
    bind_permit: Option<tokio::sync::OwnedSemaphorePermit>,
) {
    let capped_execution = execution_permit.is_some();
    let snapshot = residency.lock().unwrap().teardown_snapshot();
    let mut context = TeardownFireContext {
        process_id,
        resources,
        _execution_permit: execution_permit,
        _bind_permit: bind_permit,
    };
    if !capped_execution
        && snapshot.departed_pipeline_ids.is_empty()
        && snapshot
            .pipelines
            .iter()
            .all(|fires| fires.lock().unwrap().is_empty())
    {
        drop(context);
        return;
    }
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        tracing::error!(
            pid = %process_id,
            "process teardown found pending fires without a Tokio runtime; preserving the \
             ResourceTable to avoid recycling pages under native work"
        );
        std::mem::forget(context);
        // The leaked permit shrank the execution pool by one; tell the
        // policy the departure resolves as a FORFEIT so its seal neither
        // waits forever for a release that cannot come nor credits a free
        // slot the semaphore no longer has. (Sync send — no runtime
        // needed.) The terminate tombstone stays: with the pending fires
        // forgotten, late items for this pid remain possible.
        if capped_execution {
            crate::scheduler::worker::notify_execution_slot_forfeited(process_id);
        }
        return;
    };
    runtime.spawn(async move {
        let timing = crate::scheduler::fire_timing_enabled();
        let task_started_us = if timing {
            crate::scheduler::fire_timing_now_us()
        } else {
            0
        };
        if capped_execution {
            // Reference fence, awaited on purpose: after this resolves,
            // every driver's scheduler has purged the pid's queued work and
            // cancelled its protected in-flight control, so the finalize
            // loop below and the resource drop at the end run with no
            // scheduler-side reference to the pages they recycle. (The
            // fence also serializes each retirement behind a scheduler
            // pass; that pacing is a side effect, not the contract.)
            crate::scheduler::worker::notify_process_terminate(process_id).await;
        } else {
            for pipeline_id in snapshot.departed_pipeline_ids {
                crate::scheduler::worker::notify_pipeline_close(pipeline_id).await;
            }
        }
        let terminate_acked_us = if timing {
            crate::scheduler::fire_timing_now_us()
        } else {
            0
        };
        for fires in snapshot.pipelines {
            let _finalize_guard = fires.finalize_guard().await;
            loop {
                let op = fires.lock().unwrap().pop_front();
                let Some(op) = op else {
                    break;
                };
                if let Err(error) = crate::pipeline::fire::finalize_op(&mut context, op).await {
                    tracing::error!(
                        pid = %process_id,
                        %error,
                        "process teardown failed to finalize a pending pipeline operation"
                    );
                }
            }
        }
        if timing {
            let finalized_us = crate::scheduler::fire_timing_now_us();
            crate::scheduler::fire_timing_write(&serde_json::json!({
                "schema": 1,
                "source": "runtime",
                "event": "process_teardown",
                "process_id": process_id,
                "task_started_us": task_started_us,
                "terminate_ack_us": terminate_acked_us - task_started_us,
                "finalize_us": finalized_us - terminate_acked_us,
                "released_us": finalized_us,
            }));
        }
        // Take over the guest channels' close notifications before the
        // table drops, so a departing process posts one batched close per
        // driver instead of one mailbox item per channel (a teardown herd
        // otherwise inflates the epoch a worker pass has to drain). The
        // batch is posted only after `drop(context)` below: the drop's pass
        // teardowns post the instance closes first, and per-producer FIFO
        // then keeps the driver's instance-before-channel close order.
        let channel_close_batches =
            crate::pipeline::channel::detach_channel_close_notifications(&mut context.resources);
        if capped_execution {
            // Announce the slot BEFORE the permit drops (with `context`
            // below): the successor can only acquire after the drop, so its
            // slot-consumed event always enters the scheduler mailbox after
            // this release — the policy's slot balance never sees a consume
            // for a release it hasn't counted. (The terminate notification
            // above already removed this process's own lane: leave first,
            // release second, successor's admission and fire after.)
            crate::scheduler::worker::notify_execution_slot_released(process_id);
        }
        drop(context);
        for (driver_id, ids) in channel_close_batches {
            if let Err(error) = crate::scheduler::close_channels(driver_id, ids) {
                // Same failure mode as the per-endpoint closer (the
                // scheduler is gone at shutdown); driver shutdown closes
                // any channels this batch could not reach.
                tracing::warn!(pid = %process_id, driver_id, %error,
                    "process teardown failed to post its batched channel close");
            }
        }
        // Strictly the process's last event on every mailbox (this task is
        // its final producer, and the drop + batched close above already
        // posted its close controls): the tombstone that deduped its
        // Terminate leaves can now retire.
        crate::scheduler::worker::notify_process_quiesced(process_id);
    });
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
        ResidencyTxn::Suspend(_) => "preemption-suspend",
        ResidencyTxn::Restore(_) => "preemption-restore",
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

    /// The driver copy plan `(gpu_ids, host_slots)` of the held transaction
    /// — both variants carry one; the constructor fixed which.
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
        let model = self.model;
        let driver = self.driver;
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
            if let Some(orchestrator) = crate::store::reclaim::contention() {
                orchestrator.on_blocks_freed();
            }
        });
    }
}

async fn drain_preemption_safe_fires(ctx: &mut ProcessCtx) -> Result<()> {
    let pipelines = ctx.residency_pipelines();
    for fires in pipelines {
        let _finalize_guard = fires.finalize_guard().await;
        loop {
            let op = fires.lock().unwrap().pop_front();
            let Some(op) = op else {
                break;
            };
            crate::pipeline::fire::finalize_op(ctx, op).await?;
        }
    }
    Ok(())
}

// ============================================================================
// The one safe point (D2/D3)
//
// Eligibility to suspend is a STATE — "holds no pins" — not a place. A
// running process passes through it at every fire prologue (`yield_point`);
// a process parked in a long host await sits at it PERMANENTLY
// (`wait_yielding` races the wait against the park signal); and a requester
// waiting for a grant is the same thing again (`fire::acquire_grant` races
// its acquire future the same way). Victim, requester, and idle paths are
// one code path ending in the same suspend/restore halves.
// ============================================================================

/// Where a process stands as it passes the safe point.
pub(crate) enum SafePoint<'a> {
    /// A host-call boundary, with resource-table access: every in-flight
    /// pipeline op can be finalized here.
    Fire(&'a mut ProcessCtx),
    /// A long idle await with no table access: only detachable ops can be
    /// finalized; anything else declines the park.
    Idle {
        pid: uuid::Uuid,
        residency: Arc<Mutex<crate::inferlet::process::ProcessResidency>>,
    },
}

impl SafePoint<'_> {
    fn pid(&self) -> uuid::Uuid {
        match self {
            SafePoint::Fire(ctx) => ctx.id(),
            SafePoint::Idle { pid, .. } => *pid,
        }
    }

    fn snapshot(&self) -> crate::inferlet::process::residency::ResidencySnapshot {
        match self {
            SafePoint::Fire(ctx) => ctx.residency_snapshot(),
            SafePoint::Idle { residency, .. } => residency.lock().unwrap().snapshot(),
        }
    }

    /// Drain in-flight pipeline work down to the safe point. Returns false
    /// when non-detachable work blocks an idle park — the caller declines.
    async fn drain(&mut self) -> Result<bool> {
        match self {
            SafePoint::Fire(ctx) => {
                drain_preemption_safe_fires(ctx).await?;
                Ok(true)
            }
            SafePoint::Idle { residency, .. } => {
                let pipelines = residency.lock().unwrap().pipelines();
                for fires in &pipelines {
                    let _finalize_guard = fires.finalize_guard().await;
                    loop {
                        let op = {
                            let mut queue = fires.lock().unwrap();
                            match queue.front() {
                                Some(op) if op.is_preemption_detachable() => queue.pop_front(),
                                Some(_) => return Ok(false),
                                None => None,
                            }
                        };
                        let Some(op) = op else {
                            break;
                        };
                        crate::pipeline::fire::finalize_op_detached(op).await?;
                    }
                }
                Ok(true)
            }
        }
    }
}

/// Declines the park on every early exit: armed at the `begin_quiesce`
/// commit point, disarmed only after `report_suspended` (the point of no
/// return) or when a kill supersedes the decline. Bail-outs are scope
/// exits, never hand-written.
struct ParkGuard {
    pid: uuid::Uuid,
    driver: usize,
    armed: bool,
}

impl ParkGuard {
    fn new(pid: uuid::Uuid, driver: usize) -> Self {
        Self {
            pid,
            driver,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ParkGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        if let Some(orchestrator) = crate::store::reclaim::contention() {
            orchestrator.decline_park(self.pid);
        }
        crate::scheduler::resume_pipeline(self.pid);
        crate::scheduler::nudge(self.driver);
    }
}

/// THE safe point, fire-prologue form: every WIT host method passes through
/// here. A no-op unless a park request is pending for this process.
pub(crate) async fn yield_point(ctx: &mut ProcessCtx) -> Result<()> {
    yield_point_at(&mut SafePoint::Fire(ctx)).await.map(|_| ())
}

/// The safe point proper. Returns whether the process yielded (suspended
/// and was restored) — callers in wait loops re-issue their waits on true.
async fn yield_point_at(at: &mut SafePoint<'_>) -> Result<bool> {
    let Some(orchestrator) = crate::store::reclaim::contention() else {
        return Ok(false);
    };
    let pid = at.pid();
    if !orchestrator.should_park(pid) || !orchestrator.begin_quiesce(pid) {
        return Ok(false);
    }
    let (model, driver) = orchestrator.locus();
    let mut park = ParkGuard::new(pid, driver);
    if !at.drain().await? {
        return Ok(false); // non-detachable work pending; the guard declines
    }
    orchestrator.leave_for_suspend(pid);
    crate::scheduler::freeze_pipeline(pid)
        .await
        .context("freeze scheduler preparation for suspension")?;
    let snapshot = at.snapshot();
    let working_sets: HashSet<WorkingSetId> = snapshot
        .kv_working_sets
        .iter()
        .filter_map(|&(m, d, ws)| (m == model && d == driver).then_some(ws))
        .collect();
    if working_sets.is_empty() {
        return Ok(true); // honored with nothing to save; the guard declines
    }
    if !suspend_at_safe_point(&mut park, orchestrator, model, driver, pid, &working_sets).await? {
        return Ok(false); // nothing to yield right now; the guard declines
    }
    restore_from_park(orchestrator, model, driver, pid, &working_sets).await?;
    Ok(true)
}

/// A wait that completed, or a park request that pre-empted it.
pub(crate) enum Waited<T> {
    Done(T),
    Yielded,
}

/// Race one long wait against this process's park signal. Every such wait
/// is a STANDING safe point: pin-free by construction, suspendable the
/// moment a request lands. On `Yielded` the caller re-issues its wait.
pub(crate) async fn wait_yielding<T>(
    at: &mut SafePoint<'_>,
    wait: impl Future<Output = T>,
) -> Result<Waited<T>> {
    let Some(orchestrator) = crate::store::reclaim::contention() else {
        return Ok(Waited::Done(wait.await));
    };
    let pid = at.pid();
    if orchestrator.should_park(pid) {
        yield_point_at(at).await?;
        return Ok(Waited::Yielded);
    }
    let Some(signal) = orchestrator.park_signal(pid) else {
        return Ok(Waited::Done(wait.await));
    };
    // A truly idle await (no table access, nothing to drain) is D6's
    // preferred victim class: suspending it costs no running lane. Mark it
    // for the selector; the guard clears the mark on every exit.
    let _idle_mark = matches!(at, SafePoint::Idle { .. }).then(|| {
        orchestrator.note_idle_await(pid, true);
        IdleMark { pid }
    });
    let notified = signal.notified();
    tokio::pin!(notified);
    notified.as_mut().enable();
    // Lost-wakeup guard: a request that landed before the waiter armed.
    if orchestrator.should_park(pid) {
        yield_point_at(at).await?;
        return Ok(Waited::Yielded);
    }
    tokio::pin!(wait);
    tokio::select! {
        output = &mut wait => Ok(Waited::Done(output)),
        _ = &mut notified => {
            yield_point_at(at).await?;
            Ok(Waited::Yielded)
        }
    }
}

/// Clears the process's idle-await mark on every exit from
/// [`wait_yielding`], including cancellation.
struct IdleMark {
    pid: uuid::Uuid,
}

impl Drop for IdleMark {
    fn drop(&mut self) {
        if let Some(orchestrator) = crate::store::reclaim::contention() {
            orchestrator.note_idle_await(self.pid, false);
        }
    }
}

/// Self-serialization for allocation-adjacent prologues: while the pool is
/// contended, a deep-running lane drains its own fires (releasing pins)
/// instead of deepening. Built on the same safe point; kept separate from
/// `yield_point` because paths that FEED in-flight fires (host puts) must
/// never wait on their own settlement.
pub(crate) async fn serialize_under_contention(ctx: &mut ProcessCtx) -> Result<()> {
    let Some(orchestrator) = crate::store::reclaim::contention() else {
        return Ok(());
    };
    loop {
        yield_point(ctx).await?;
        if !orchestrator.contended() {
            return Ok(());
        }
        drain_preemption_safe_fires(ctx).await?;
        let progress = ctx.residency_pipelines().into_iter().find_map(|fires| {
            fires
                .lock()
                .unwrap()
                .front()
                .map(PendingOp::preemption_signal)
        });
        let Some(progress) = progress else {
            return Ok(());
        };
        let mut at = SafePoint::Fire(ctx);
        let _ = wait_yielding(&mut at, progress).await?;
    }
}

// ============================================================================
// The suspend/restore halves
// ============================================================================

/// Suspend half: prepare → D2H → commit → `report_suspended`. Disarms the
/// park guard on success (the point of no return); every failure path
/// declines through it. Returns false when there is nothing to yield now.
async fn suspend_at_safe_point(
    park: &mut ParkGuard,
    orchestrator: &crate::store::reclaim::ContentionOrchestrator,
    model: usize,
    driver: usize,
    pid: uuid::Uuid,
    working_sets: &HashSet<WorkingSetId>,
) -> Result<bool> {
    let stores = crate::store::registry::get(model, driver);
    let prepared =
        match crate::store::registry::with_kv_lock(&stores.kv, "preemption-suspend", |kv| {
            kv.prepare_suspend(working_sets)
        }) {
            Ok(prepared) => prepared,
            Err(error @ crate::store::kv::KvStoreError::HostSwapFull { .. }) => {
                orchestrator.record_host_swap_exhaustion();
                let reason =
                    format!("KV preemption killed process after host swap exhaustion: {error}");
                park.disarm(); // the kill supersedes the decline
                crate::inferlet::process::terminate(pid, Err(reason.clone()));
                return Err(anyhow::anyhow!(reason));
            }
            Err(error) => return Err(error).context("prepare KV suspend"),
        };
    let txn = match prepared {
        KvSuspendPrepare::Prepared(txn) => txn,
        KvSuspendPrepare::Deferred(SuspendDisposition::NothingReclaimable) => return Ok(false),
        KvSuspendPrepare::Deferred(SuspendDisposition::GraceDeferred) => {
            orchestrator.record_grace_deferred();
            return Ok(false);
        }
    };

    let mut suspend = ResidencyTxnGuard::suspend(model, driver, txn);
    let (gpu_ids, host_slots) = suspend.copy_plan();
    let copy_started = Instant::now();
    let completion = match crate::scheduler::copy_d2h_tracked(driver, &gpu_ids, &host_slots) {
        Ok(completion) => completion,
        Err(error) => {
            suspend.abort_now();
            orchestrator.record_suspend_rollback();
            tracing::warn!(pid = %pid, %error, "KV D2H suspend copy rejected");
            return Ok(false);
        }
    };
    suspend.arm(completion.clone());
    if let Err(error) = completion.wait().await {
        orchestrator.record_d2h_copy(copy_started.elapsed());
        suspend.disarm_completion();
        suspend.abort_now();
        orchestrator.record_suspend_rollback();
        tracing::warn!(pid = %pid, %error, "KV D2H suspend copy failed");
        return Ok(false);
    }
    orchestrator.record_d2h_copy(copy_started.elapsed());
    suspend.disarm_completion();
    let txn = suspend.take_suspend();
    let freed = crate::store::registry::with_kv_lock(&stores.kv, "preemption-suspend", |kv| {
        kv.commit_suspend(txn)
    })
    .inspect_err(|_| orchestrator.record_suspend_rollback())
    .context("commit KV suspend")?;
    orchestrator.report_suspended(pid, freed as u32);
    park.disarm();
    Ok(true)
}

/// Restore half: park → prepare → H2D → commit → `report_restored`, with
/// bounded retries from the injected config.
async fn restore_from_park(
    orchestrator: &crate::store::reclaim::ContentionOrchestrator,
    model: usize,
    driver: usize,
    pid: uuid::Uuid,
    working_sets: &HashSet<WorkingSetId>,
) -> Result<()> {
    let stores = crate::store::registry::get(model, driver);
    let max_restore_attempts = orchestrator.config().restore_retries.max(1);
    for attempt in 1..=max_restore_attempts {
        let mut grant = orchestrator
            .park_until_restore_granted(pid)
            .await
            .context("wait for KV restore grant")?;
        // Lend the grant's pages to the store; the grant guard stays armed,
        // so an error below returns every unconsumed page through its Drop.
        let prepared =
            crate::store::registry::with_kv_lock(&stores.kv, "preemption-restore", |kv| {
                kv.prepare_restore(working_sets, grant.lend_kv())
            });
        drop(grant);
        let txn = match prepared {
            Ok(txn) => txn,
            Err(error) => {
                orchestrator.record_restore_rollback();
                orchestrator.report_restore_failed(pid);
                if attempt == max_restore_attempts {
                    return Err(error).context("prepare KV restore");
                }
                continue;
            }
        };
        let mut restore = ResidencyTxnGuard::restore(model, driver, txn);
        orchestrator.record_restore_prepared();
        let (gpu_ids, host_slots) = restore.copy_plan();
        let copy_started = Instant::now();
        let completion = match crate::scheduler::copy_h2d_tracked(driver, &gpu_ids, &host_slots) {
            Ok(completion) => completion,
            Err(error) => {
                restore.abort_now();
                orchestrator.record_restore_rollback();
                orchestrator.report_restore_failed(pid);
                if attempt == max_restore_attempts {
                    return Err(error).context("submit KV H2D restore copy");
                }
                continue;
            }
        };
        orchestrator.record_h2d_submitted();
        restore.arm(completion.clone());
        if let Err(error) = completion.wait().await {
            orchestrator.record_h2d_copy(copy_started.elapsed());
            restore.disarm_completion();
            restore.abort_now();
            orchestrator.record_restore_rollback();
            orchestrator.report_restore_failed(pid);
            if attempt == max_restore_attempts {
                return Err(error).context("KV H2D restore copy");
            }
            continue;
        }
        orchestrator.record_h2d_copy(copy_started.elapsed());
        restore.disarm_completion();
        let txn = restore.take_restore();
        let restored =
            match crate::store::registry::with_kv_lock(&stores.kv, "preemption-restore", |kv| {
                kv.commit_restore(txn)
            }) {
                Ok(restored) => restored,
                Err(error) => {
                    orchestrator.record_restore_rollback();
                    orchestrator.report_restore_failed(pid);
                    if attempt == max_restore_attempts {
                        return Err(error).context("commit KV restore");
                    }
                    continue;
                }
            };
        orchestrator.report_restored(pid, restored as u32);
        crate::scheduler::resume_pipeline(pid);
        crate::scheduler::nudge(driver);
        return Ok(());
    }
    unreachable!("restore loop has at least one attempt")
}

// ============================================================================
// Long host awaits — standing safe points
// ============================================================================

pub(crate) async fn receive_message(
    process_id: uuid::Uuid,
    residency: Arc<Mutex<crate::inferlet::process::ProcessResidency>>,
) -> Result<Option<String>> {
    let mut at = SafePoint::Idle {
        pid: process_id,
        residency,
    };
    loop {
        let wait = crate::server::inbox::receive(process_id.to_string());
        match wait_yielding(&mut at, wait).await? {
            Waited::Done(result) => {
                return result
                    .with_context(|| format!("session.receive failed for process {process_id}"))
                    .map(Some);
            }
            Waited::Yielded => continue,
        }
    }
}

pub(crate) async fn receive_file(
    process_id: uuid::Uuid,
    residency: Arc<Mutex<crate::inferlet::process::ProcessResidency>>,
) -> Result<Option<Vec<u8>>> {
    let Some(client_id) = crate::inferlet::process::get_client_id(process_id)
        .await
        .ok()
        .flatten()
    else {
        return Ok(None);
    };
    let mut at = SafePoint::Idle {
        pid: process_id,
        residency,
    };
    loop {
        let wait = crate::server::receive_file(client_id, process_id);
        match wait_yielding(&mut at, wait).await? {
            Waited::Done(Ok(data)) => return Ok(Some(data.to_vec())),
            Waited::Done(Err(error)) => {
                tracing::warn!(
                    client_id,
                    process_id = %process_id,
                    %error,
                    "session.receive_file delivery failed"
                );
                return Ok(None);
            }
            Waited::Yielded => continue,
        }
    }
}

pub(crate) async fn await_channel_progress_idle(
    process_id: uuid::Uuid,
    residency: Arc<Mutex<crate::inferlet::process::ProcessResidency>>,
    cell: &Arc<Mutex<crate::pipeline::channel::ChannelCell>>,
    fires: Option<&PendingFires>,
) -> Result<(), String> {
    let mut at = SafePoint::Idle {
        pid: process_id,
        residency,
    };
    let wait = crate::pipeline::fire::await_channel_progress(cell, fires);
    match wait_yielding(&mut at, wait).await {
        Ok(Waited::Done(result)) => result,
        // The caller re-evaluates channel state after a yield.
        Ok(Waited::Yielded) => Ok(()),
        Err(error) => Err(error.to_string()),
    }
}

pub(crate) async fn await_writer_progress(
    ctx: &mut ProcessCtx,
    endpoint: &Arc<crate::driver::ChannelEndpoint>,
    observed_head: u64,
) -> Result<(), String> {
    let wait = endpoint.wait_for_writer_change(observed_head);
    let mut at = SafePoint::Fire(ctx);
    match wait_yielding(&mut at, wait).await {
        Ok(Waited::Done(result)) => result.map_err(|error| error.to_string()),
        // The caller re-evaluates writer state after a yield.
        Ok(Waited::Yielded) => Ok(()),
        Err(error) => Err(error.to_string()),
    }
}
