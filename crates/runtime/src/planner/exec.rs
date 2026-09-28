use crate::rt::Instant;
use std::collections::HashSet;
use std::sync::Arc;

use super::{ProcessId, ResidencyPlanner};

use crate::store::kv::page_table::WorkingSetId;
use crate::store::kv::working_set::KvSuspendHandle;
use crate::store::kv::{KvRestoreTxn, KvSuspendPrepare, KvSuspendTxn, SuspendDisposition};
use crate::store::rs::RsResidencyTxn;
use ::engine::transfer::StateDirection;

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
        planner.restore_deferred(pid, "no tokio runtime for the restore executor");
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

enum ResidencyTxn {
    Suspend(KvSuspendTxn),
    Restore(KvRestoreTxn),
    RsSuspend(RsResidencyTxn),
    RsRestore(RsResidencyTxn),
}

fn abort_residency_txn(model: usize, engine: usize, txn: ResidencyTxn) {
    let stores = crate::store::registry::get(model, engine);
    match txn {
        ResidencyTxn::Suspend(txn) => {
            crate::store::registry::with_kv_lock(&stores.kv, "planner-evict", |kv| {
                kv.abort_suspend(txn)
            })
        }
        ResidencyTxn::Restore(txn) => {
            crate::store::registry::with_kv_lock(&stores.kv, "planner-restore", |kv| {
                kv.abort_restore(txn)
            })
        }
        ResidencyTxn::RsSuspend(txn) => stores.rs.lock().unwrap().abort_suspend(txn),
        ResidencyTxn::RsRestore(txn) => stores.rs.lock().unwrap().abort_restore(txn),
    }
}

struct ResidencyTxnGuard {
    model: usize,
    engine: usize,
    txn: Option<ResidencyTxn>,
    completion: Option<crate::scheduler::ControlCompletion>,
}

impl ResidencyTxnGuard {
    fn new(model: usize, engine: usize, txn: ResidencyTxn) -> Self {
        Self {
            model,
            engine,
            txn: Some(txn),
            completion: None,
        }
    }

    /// Submits the transaction's engine copy, if it moves anything.
    fn submit(&mut self) -> anyhow::Result<()> {
        let engine = self.engine;
        let completion = match self.txn.as_ref().expect("transaction present") {
            ResidencyTxn::Suspend(txn) if txn.page_count() > 0 => {
                crate::scheduler::copy_d2h_tracked(engine, &txn.gpu_ids(), &txn.host_slots())?
            }
            ResidencyTxn::Restore(txn) if txn.page_count() > 0 => {
                crate::scheduler::copy_h2d_tracked(engine, &txn.gpu_ids(), &txn.host_slots())?
            }
            ResidencyTxn::RsSuspend(txn) | ResidencyTxn::RsRestore(txn) if txn.slot_count() > 0 => {
                let direction = match self.txn {
                    Some(ResidencyTxn::RsSuspend(_)) => StateDirection::DeviceToHost,
                    _ => StateDirection::HostToDevice,
                };
                let (src, dst) = txn.copy_plan();
                crate::scheduler::copy_rs_tracked(engine, direction, &src, &dst)?
            }
            _ => return Ok(()),
        };
        self.completion = Some(completion);
        Ok(())
    }

    async fn settle(&mut self) -> anyhow::Result<()> {
        let Some(completion) = self.completion.clone() else {
            return Ok(());
        };
        let copied = completion.wait().await;
        self.completion = None;
        copied
    }

    fn take(&mut self) -> ResidencyTxn {
        self.txn.take().expect("transaction present")
    }
}

impl Drop for ResidencyTxnGuard {
    fn drop(&mut self) {
        let Some(txn) = self.txn.take() else {
            return;
        };
        let Some(completion) = self.completion.take() else {
            abort_residency_txn(self.model, self.engine, txn);
            return;
        };
        let (model, engine) = (self.model, self.engine);
        if !crate::rt::has_runtime() {
            tracing::error!(
                model,
                engine,
                "residency transaction dropped with an engine copy in flight and no runtime; \
                 preserving its pages and slots to avoid reuse during the copy"
            );
            return;
        };
        crate::rt::spawn(async move {
            let _ = completion.wait().await;
            abort_residency_txn(model, engine, txn);
            if let Some(planner) = crate::planner::planner_for(model, engine) {
                planner.pages_freed();
            }
        });
    }
}

/// Submits every transaction's copy, then waits them all; a failure leaves
/// the guards to abort their transactions once their copies land.
async fn copy_through(guards: &mut [&mut Option<ResidencyTxnGuard>]) -> Result<(), String> {
    for guard in guards.iter_mut().filter_map(|guard| guard.as_mut()) {
        guard
            .submit()
            .map_err(|error| format!("submit: {error:#}"))?;
    }
    for guard in guards.iter_mut().filter_map(|guard| guard.as_mut()) {
        guard
            .settle()
            .await
            .map_err(|error| format!("copy: {error:#}"))?;
    }
    Ok(())
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
        if working_sets.is_empty() || kv.host_swap_capacity() == 0 {
            return Ok(KvSuspendPrepare::Deferred(
                SuspendDisposition::NothingReclaimable,
            ));
        }
        kv.prepare_suspend(&working_sets)
    });
    let mut kv = match prepared {
        Ok(KvSuspendPrepare::Prepared(txn)) => Some(ResidencyTxnGuard::new(
            model,
            engine,
            ResidencyTxn::Suspend(txn),
        )),
        Ok(KvSuspendPrepare::Deferred(SuspendDisposition::NothingReclaimable)) => None,
        Ok(KvSuspendPrepare::Deferred(_)) => {
            planner.eviction_failed_prepare_deferred(pid);
            return;
        }
        Err(error @ crate::store::kv::KvStoreError::HostSwapFull { .. }) => {
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
    let mut rs = match stores.rs.lock().unwrap().prepare_suspend(&rs_working_sets) {
        Ok(txn) => {
            txn.map(|txn| ResidencyTxnGuard::new(model, engine, ResidencyTxn::RsSuspend(txn)))
        }
        Err(error) => {
            drop(kv);
            tracing::warn!(pid = %pid, %error, "planner: eviction blocked on host rs slots");
            planner.eviction_failed_host_swap_full(pid);
            return;
        }
    };
    if kv.is_none() && rs.is_none() {
        planner.eviction_failed_prepare_deferred(pid);
        return;
    }
    let copy_started = Instant::now();
    if let Err(error) = copy_through(&mut [&mut kv, &mut rs]).await {
        tracing::warn!(pid = %pid, %error, "planner: eviction copy failed");
        planner.eviction_failed(pid);
        return;
    }
    planner.record_d2h_copy(copy_started.elapsed());
    let freed = match kv.as_mut().map(ResidencyTxnGuard::take) {
        Some(ResidencyTxn::Suspend(txn)) => {
            match crate::store::registry::with_kv_lock(&stores.kv, "planner-evict", |kv| {
                kv.commit_suspend(txn)
            }) {
                Ok(freed) => freed,
                Err(error) => {
                    tracing::warn!(pid = %pid, %error, "planner: suspend commit failed");
                    planner.eviction_failed(pid);
                    return;
                }
            }
        }
        _ => 0,
    };
    let rs_freed = match rs.as_mut().map(ResidencyTxnGuard::take) {
        Some(ResidencyTxn::RsSuspend(txn)) => stores.rs.lock().unwrap().commit_suspend(txn),
        _ => 0,
    };
    fence.keep_raised();
    planner.report_evicted(pid, freed as u32, rs_freed as u32);
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
        let prepared = crate::store::registry::with_kv_lock(&stores.kv, "planner-restore", |kv| {
            kv.prepare_restore(&working_sets, pages.lend(super::KvPool::Paged))
        });
        match prepared {
            Ok(txn) => {
                kv = Some(ResidencyTxnGuard::new(
                    model,
                    engine,
                    ResidencyTxn::Restore(txn),
                ))
            }
            Err(error) => {
                planner.restore_deferred(pid, &error.to_string());
                return;
            }
        }
    }
    drop(pages);
    let mut rs = None;
    if !rs_working_sets.is_empty() {
        match stores
            .rs
            .lock()
            .unwrap()
            .prepare_restore(&rs_working_sets, slots.lend())
        {
            Ok(txn) => {
                rs = Some(ResidencyTxnGuard::new(
                    model,
                    engine,
                    ResidencyTxn::RsRestore(txn),
                ))
            }
            Err(error) => {
                planner.restore_deferred(pid, &error.to_string());
                return;
            }
        }
    }
    drop(slots);
    let copy_started = Instant::now();
    if let Err(error) = copy_through(&mut [&mut kv, &mut rs]).await {
        planner.restore_deferred(pid, &format!("H2D {error}"));
        return;
    }
    planner.record_h2d_copy(copy_started.elapsed());
    let restored = match kv.as_mut().map(ResidencyTxnGuard::take) {
        Some(ResidencyTxn::Restore(txn)) => {
            match crate::store::registry::with_kv_lock(&stores.kv, "planner-restore", |kv| {
                kv.commit_restore(txn)
            }) {
                Ok(restored) => restored,
                Err(error) => {
                    planner.restore_failed(pid, &error.to_string());
                    return;
                }
            }
        }
        _ => 0,
    };
    let rs_restored = match rs.as_mut().map(ResidencyTxnGuard::take) {
        Some(ResidencyTxn::RsRestore(txn)) => stores.rs.lock().unwrap().commit_restore(txn),
        _ => 0,
    };
    planner.report_restored(pid, restored as u32, rs_restored as u32);
    crate::scheduler::nudge(engine);
}
