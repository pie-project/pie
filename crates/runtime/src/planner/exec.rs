use crate::rt::Instant;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use super::{ProcessId, ResidencyPlanner};

use crate::scheduler::{TierKind, TierMove};
use crate::store::kv::page_table::WorkingSetId;
use crate::store::kv::working_set::KvSuspendHandle;
use crate::store::kv::{KvResidencyTxn, KvStoreError};
use crate::store::rs::RsResidencyTxn;
use ::engine::transfer::MemoryDomain;

fn spawn_watched(
    planner: Arc<ResidencyPlanner>,
    pid: ProcessId,
    label: &'static str,
    task: impl std::future::Future<Output = ()> + Send + 'static,
    on_fail: impl FnOnce(&Arc<ResidencyPlanner>, ProcessId) + Send + 'static,
) -> bool {
    if !crate::rt::has_runtime() {
        return false;
    };
    let handle = crate::rt::spawn(task);
    crate::rt::spawn(async move {
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

/// No process reads them, so a failure only puts them back.
pub(super) fn spawn_move_out(
    planner: Arc<ResidencyPlanner>,
    txn: KvResidencyTxn,
    label: &'static str,
    done: fn(&Arc<ResidencyPlanner>, bool),
) {
    let (model, engine) = planner.locus();
    // The guard owns the move from here: a task that never runs drops it,
    // and that gives back what the move took.
    let guard = MoveGuard::new(
        model,
        engine,
        Move::Suspend {
            kv: Some(txn),
            rs: None,
        },
    );
    let task = move_out(planner.clone(), guard, label, done);
    let spawned = spawn_watched(
        planner.clone(),
        ProcessId::nil(),
        label,
        task,
        move |planner, _| done(planner, false),
    );
    if !spawned {
        done(&planner, false);
    }
}

pub(super) fn spawn_reseat(planner: Arc<ResidencyPlanner>, pid: ProcessId) {
    let task = {
        let planner = planner.clone();
        async move {
            let seat = crate::inferlet::process::take_seat(pid).await;
            planner.reseated(pid, seat);
        }
    };
    let spawned = spawn_watched(planner.clone(), pid, "reseat", task, |planner, pid| {
        planner.reseat_failed(pid)
    });
    if !spawned {
        planner.reseat_failed(pid);
    }
}

pub(super) fn spawn_restore(
    planner: Arc<ResidencyPlanner>,
    pid: ProcessId,
    pages: super::grant::DevicePageReservation,
    slots: super::grant::RsSlotReservation,
) {
    let task = restore(planner.clone(), pid, pages, slots);
    let spawned = spawn_watched(planner.clone(), pid, "restore", task, |planner, pid| {
        planner.restore_failed(pid, "restore executor died")
    });
    if !spawned {
        planner.requeue_restore(pid, Some("no tokio runtime for the restore executor"));
    }
}

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

/// A process's residency move: its kv and rs transactions, prepared in the
/// stores and committed once the engine copies land, either of which may
/// be absent when that store holds nothing to move.
enum Move {
    Suspend {
        kv: Option<KvResidencyTxn>,
        rs: Option<RsResidencyTxn>,
    },
    Restore {
        kv: Option<KvResidencyTxn>,
        rs: Option<RsResidencyTxn>,
    },
}

impl Move {
    /// The engine copies the move takes: kv pages by the tier they are
    /// backed in, then the rs slots.
    fn copies(&self) -> Vec<TierMove> {
        let (out, kv, rs) = match self {
            Move::Suspend { kv, rs } => (true, kv, rs),
            Move::Restore { kv, rs } => (false, kv, rs),
        };
        let mut copies: Vec<TierMove> = kv
            .iter()
            .flat_map(KvResidencyTxn::copy_plans)
            .map(|(tier, device, slots)| TierMove {
                kind: TierKind::Kv,
                tier,
                out,
                device,
                slots,
            })
            .collect();
        if let Some(rs) = rs.as_ref() {
            let plan = rs.copy_plan();
            let pages = if rs.is_ring() {
                TierKind::Window
            } else {
                TierKind::RsPages
            };
            for (kind, pairs) in [(TierKind::Rs, &plan.slots), (pages, &plan.pages)] {
                if pairs.is_empty() {
                    continue;
                }
                let (device, slots) = pairs.iter().copied().unzip();
                copies.push(TierMove {
                    kind,
                    tier: MemoryDomain::HostPinned,
                    out,
                    device,
                    slots,
                });
            }
        }
        copies
    }
}

fn abort(model: usize, engine: usize, txn: Move) {
    let stores = crate::store::registry::get(model, engine);
    match txn {
        Move::Suspend { kv, rs } => {
            if let Some(txn) = kv {
                crate::store::registry::with_kv_lock(&stores.kv, "planner-evict", |kv| {
                    kv.abort_suspend(txn)
                });
            }
            if let Some(txn) = rs {
                stores.rs.lock().unwrap().abort_suspend(txn);
            }
        }
        Move::Restore { kv, rs } => {
            if let Some(txn) = kv {
                crate::store::registry::with_kv_lock(&stores.kv, "planner-restore", |kv| {
                    kv.abort_restore(txn)
                });
            }
            if let Some(txn) = rs {
                stores.rs.lock().unwrap().abort_restore(txn);
            }
        }
    }
}

/// Aborts the move it still holds when dropped, after any copy in flight.
struct MoveGuard {
    model: usize,
    engine: usize,
    txn: Option<Move>,
    in_flight: Vec<crate::scheduler::ControlCompletion>,
}

impl MoveGuard {
    fn new(model: usize, engine: usize, txn: Move) -> Self {
        Self {
            model,
            engine,
            txn: Some(txn),
            in_flight: Vec::new(),
        }
    }

    /// Submits every copy of the move, then waits for them all.
    async fn copy(&mut self, planner: &ResidencyPlanner, pid: ProcessId) -> Result<(), String> {
        let mut waits = Vec::new();
        for copy in self.txn.as_ref().expect("move present").copies() {
            tracing::debug!(pid = %pid, copy = %copy.label(), count = copy.device.len(), "planner: residency copy");
            let started = Instant::now();
            let completion = crate::scheduler::copy_tier_tracked(self.engine, &copy)
                .map_err(|error| format!("{} submit: {error:#}", copy.label()))?;
            self.in_flight.push(completion.clone());
            waits.push((copy, started, completion));
        }
        for (copy, started, completion) in waits {
            let copied = completion.wait().await;
            planner.record_tier_copy(&copy, started.elapsed());
            copied.map_err(|error| format!("{} copy: {error}", copy.label()))?;
        }
        self.in_flight.clear();
        Ok(())
    }

    fn take(&mut self) -> Move {
        self.txn.take().expect("move present")
    }
}

impl Drop for MoveGuard {
    fn drop(&mut self) {
        let Some(txn) = self.txn.take() else {
            return;
        };
        let in_flight = std::mem::take(&mut self.in_flight);
        if in_flight.is_empty() {
            abort(self.model, self.engine, txn);
            return;
        }
        let (model, engine) = (self.model, self.engine);
        if !crate::rt::has_runtime() {
            tracing::error!(
                model,
                engine,
                "residency move dropped with an engine copy in flight and no runtime; \
                 preserving its pages and slots to avoid reuse during the copy"
            );
            return;
        };
        crate::rt::spawn(async move {
            for completion in in_flight {
                let _ = completion.wait().await;
            }
            abort(model, engine, txn);
            if let Some(planner) = crate::planner::planner_for(model, engine) {
                planner.pages_freed();
            }
        });
    }
}

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
    let (model, engine) = planner.locus();
    let handles = crate::inferlet::process::residency::kv_suspend_handles(pid, model, engine);
    let working_sets: HashSet<WorkingSetId> =
        crate::inferlet::process::residency::kv_working_set_ids(pid, model, engine);
    let rs_working_sets =
        crate::inferlet::process::residency::rs_working_set_ids(pid, model, engine);
    if working_sets.is_empty() && rs_working_sets.is_empty() {
        tracing::debug!(pid = %pid, "planner: eviction found no working set");
        planner.eviction_failed(pid);
        return;
    }
    let mut fence = FenceGuard::raise(handles);
    crate::scheduler::worker::notify_process_suspend(pid);
    drain_detachable(pid).await;
    for handle in fence.handles.iter() {
        handle.quiesce().await;
    }
    let stores = crate::store::registry::get(model, engine);
    let prepared = crate::store::registry::with_kv_lock(&stores.kv, "planner-evict", |kv| {
        kv.prepare_suspend(&working_sets)
    });
    let kv = match prepared {
        Ok(kv) => kv,
        Err(error @ KvStoreError::HostSwapFull { .. }) => {
            tracing::warn!(pid = %pid, %error, "planner: eviction blocked on swap room");
            planner.eviction_failed_host_swap_full(pid);
            return;
        }
        Err(error) => {
            tracing::warn!(pid = %pid, %error, "planner: suspend prepare failed");
            planner.eviction_failed(pid);
            return;
        }
    };
    let rs = stores.rs.lock().unwrap().prepare_suspend(&rs_working_sets);
    if kv.is_none() && rs.is_none() {
        tracing::debug!(pid = %pid, "planner: eviction found nothing reclaimable");
        planner.eviction_found_nothing(pid);
        return;
    }
    let mut guard = MoveGuard::new(model, engine, Move::Suspend { kv, rs });
    if let Err(error) = guard.copy(&planner, pid).await {
        tracing::warn!(pid = %pid, %error, "planner: eviction copy failed");
        planner.eviction_failed(pid);
        return;
    }
    let Move::Suspend { kv, rs } = guard.take() else {
        unreachable!("an eviction carries a suspend")
    };
    let freed = match kv {
        Some(txn) => {
            let freed = crate::store::registry::with_kv_lock(&stores.kv, "planner-evict", |kv| {
                kv.commit_suspend(txn)
            });
            match freed {
                Ok(freed) => freed,
                Err(error) => {
                    abort(model, engine, Move::Suspend { kv: None, rs });
                    tracing::warn!(pid = %pid, %error, "planner: suspend commit failed");
                    planner.eviction_failed(pid);
                    return;
                }
            }
        }
        None => {
            crate::store::registry::with_kv_lock(&stores.kv, "planner-evict", |kv| {
                kv.mark_suspended(&working_sets)
            });
            0
        }
    };
    let rs_freed = rs.map_or(Default::default(), |txn| {
        stores.rs.lock().unwrap().commit_suspend(txn)
    });
    fence.keep_raised();
    planner.report_evicted(pid, freed as u32, rs_freed);
}

const MOVE_WAIT: Duration = Duration::from_secs(10);

async fn move_out(
    planner: Arc<ResidencyPlanner>,
    mut guard: MoveGuard,
    label: &'static str,
    done: fn(&Arc<ResidencyPlanner>, bool),
) {
    let (model, engine) = planner.locus();
    if let Err(error) = guard.copy(&planner, ProcessId::nil()).await {
        tracing::warn!(%error, "planner: {label} copy failed");
        drop(guard);
        done(&planner, false);
        return;
    }
    let Move::Suspend { kv: Some(txn), .. } = guard.take() else {
        unreachable!("a {label} carries a kv suspend")
    };
    let stores = crate::store::registry::get(model, engine);
    let moved = crate::store::registry::with_kv_lock(&stores.kv, "planner-move-out", |kv| {
        kv.commit_suspend(txn)
    });
    if let Err(error) = &moved {
        tracing::warn!(%error, "planner: {label} commit failed");
    }
    done(&planner, moved.is_ok());
}

async fn restore(
    planner: Arc<ResidencyPlanner>,
    pid: ProcessId,
    mut pages: super::grant::DevicePageReservation,
    mut slots: super::grant::RsSlotReservation,
) {
    let (model, engine) = planner.locus();
    let working_sets: HashSet<WorkingSetId> =
        crate::inferlet::process::residency::kv_working_set_ids(pid, model, engine);
    let rs_working_sets =
        crate::inferlet::process::residency::rs_working_set_ids(pid, model, engine);
    let stores = crate::store::registry::get(model, engine);
    let mut kv = None;
    if !working_sets.is_empty() {
        // Another move may hold pages this process shares: it lands soon,
        // and this waits for it rather than for a clock.
        let ended = crate::store::registry::with_kv_lock(&stores.kv, "planner-restore", |kv| {
            kv.moves_ended()
        });
        let deadline = Instant::now() + MOVE_WAIT;
        loop {
            let notified = ended.notified();
            let mut notified = std::pin::pin!(notified);
            notified.as_mut().enable();
            let prepared =
                crate::store::registry::with_kv_lock(&stores.kv, "planner-restore", |kv| {
                    kv.prepare_restore(&working_sets, pages.lend())
                });
            match prepared {
                Ok(txn) => {
                    kv = Some(txn);
                    break;
                }
                Err(KvStoreError::MoveInFlight) if Instant::now() < deadline => {
                    let left = deadline.saturating_duration_since(Instant::now());
                    let _ = crate::rt::time::timeout(left, notified).await;
                }
                // Still moving, or its pages moved meanwhile: queue again,
                // which is no failure of this restore.
                Err(KvStoreError::MoveInFlight | KvStoreError::GrantMismatch { .. }) => {
                    planner.requeue_restore(pid, None);
                    return;
                }
                Err(error) => {
                    planner.requeue_restore(pid, Some(&error.to_string()));
                    return;
                }
            }
        }
    }
    drop(pages);
    let mut rs = None;
    if !rs_working_sets.is_empty() {
        let prepared = stores
            .rs
            .lock()
            .unwrap()
            .prepare_restore(&rs_working_sets, slots.lend());
        match prepared {
            Ok(txn) => rs = Some(txn),
            Err(error) => {
                abort(model, engine, Move::Restore { kv, rs: None });
                planner.requeue_restore(pid, Some(&error.to_string()));
                return;
            }
        }
    }
    drop(slots);
    let mut guard = MoveGuard::new(model, engine, Move::Restore { kv, rs });
    if let Err(error) = guard.copy(&planner, pid).await {
        planner.requeue_restore(pid, Some(&error));
        return;
    }
    let Move::Restore { kv, rs } = guard.take() else {
        unreachable!("a restore carries a restore")
    };
    if let Some(txn) = kv {
        let restored = crate::store::registry::with_kv_lock(&stores.kv, "planner-restore", |kv| {
            kv.commit_restore(txn)
        });
        if let Err(error) = restored {
            abort(model, engine, Move::Restore { kv: None, rs });
            planner.restore_failed(pid, &error.to_string());
            return;
        }
    }
    if let Some(txn) = rs {
        stores.rs.lock().unwrap().commit_restore(txn);
    }
    planner.report_restored(pid);
    crate::scheduler::nudge(engine);
}

/// Persists the sealed run `keys` (from token 0) on the disk tier of
/// `engine`: each resident page not on disk yet is copied to a fresh slot
/// and stays where it is, then the journal names it. Answers the pages
/// persisted; none when the tier is not kept, is full, or the run is not
/// held here.
pub(super) async fn persist(
    model: usize,
    engine: usize,
    keys: Vec<crate::store::kv::hash::Hash256>,
    retention: i64,
) -> Result<usize, String> {
    let stores = crate::store::registry::get(model, engine);
    let Some(txn) = crate::store::registry::with_kv_lock(&stores.kv, "persist", |kv| {
        kv.prepare_persist(&keys, retention)
    }) else {
        return Ok(0);
    };
    let (device, slots) = txn.copy_plan();
    let copy = TierMove {
        kind: TierKind::Kv,
        tier: MemoryDomain::LocalDisk,
        out: true,
        device,
        slots,
    };
    let mut guard = PersistGuard {
        model,
        engine,
        txn: Some(txn),
        in_flight: None,
    };
    let started = Instant::now();
    let completion = crate::scheduler::copy_tier_tracked(engine, &copy)
        .map_err(|error| format!("{} submit: {error:#}", copy.label()))?;
    guard.in_flight = Some(completion.clone());
    let copied = completion.wait().await;
    guard.in_flight = None;
    if let Some(planner) = crate::planner::planner_for(model, engine) {
        planner.record_tier_copy(&copy, started.elapsed());
    }
    copied.map_err(|error| format!("{} copy: {error}", copy.label()))?;
    let txn = guard.txn.take().expect("persist present");
    Ok(crate::store::registry::with_kv_lock(
        &stores.kv,
        "persist",
        |kv| kv.commit_persist(txn),
    ))
}

/// Gives back a persist's slots when it is dropped before it commits,
/// after any copy in flight.
struct PersistGuard {
    model: usize,
    engine: usize,
    txn: Option<crate::store::kv::KvPersistTxn>,
    in_flight: Option<crate::scheduler::ControlCompletion>,
}

impl Drop for PersistGuard {
    fn drop(&mut self) {
        let Some(txn) = self.txn.take() else {
            return;
        };
        let (model, engine) = (self.model, self.engine);
        let abort = move || {
            let stores = crate::store::registry::get(model, engine);
            crate::store::registry::with_kv_lock(&stores.kv, "persist-abort", |kv| {
                kv.abort_persist(txn)
            });
        };
        match self.in_flight.take() {
            None => abort(),
            Some(completion) if crate::rt::has_runtime() => {
                crate::rt::spawn(async move {
                    let _ = completion.wait().await;
                    abort();
                });
            }
            Some(_) => tracing::error!(
                model,
                engine,
                "persist dropped with an engine copy in flight and no runtime; \
                 keeping its slots to avoid reuse during the copy"
            ),
        }
    }
}
