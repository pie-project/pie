use crate::rt::Instant;
use std::collections::HashSet;
use std::sync::Arc;

use super::{ProcessId, ResidencyPlanner};

use crate::store::kv::page_table::WorkingSetId;
use crate::store::kv::working_set::KvSuspendHandle;
use crate::store::kv::{KvRestoreTxn, KvSuspendPrepare, KvSuspendTxn};

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
) {
    let task = restore(planner.clone(), pid, pages);
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
}

fn abort_residency_txn(model: usize, engine: usize, txn: ResidencyTxn) {
    let stores = crate::store::registry::get(model, engine);
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
    engine: usize,
    txn: Option<ResidencyTxn>,
    completion: Option<crate::scheduler::ControlCompletion>,
}

impl ResidencyTxnGuard {
    fn suspend(model: usize, engine: usize, txn: KvSuspendTxn) -> Self {
        Self {
            model,
            engine,
            txn: Some(ResidencyTxn::Suspend(txn)),
            completion: None,
        }
    }

    fn restore(model: usize, engine: usize, txn: KvRestoreTxn) -> Self {
        Self {
            model,
            engine,
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

    fn copy_plan(&self) -> (Vec<u32>, Vec<u32>) {
        match self.txn.as_ref().expect("transaction present") {
            ResidencyTxn::Suspend(txn) => (txn.gpu_ids(), txn.host_slots()),
            ResidencyTxn::Restore(txn) => (txn.gpu_ids(), txn.host_slots()),
        }
    }

    fn abort_now(&mut self) {
        if let Some(txn) = self.txn.take() {
            abort_residency_txn(self.model, self.engine, txn);
        }
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
                "KV residency transaction dropped with an engine copy in flight and no runtime; \
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
    if working_sets.is_empty() {
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
    let txn = match prepared {
        Ok(KvSuspendPrepare::Prepared(txn)) => txn,
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
    let mut suspend = ResidencyTxnGuard::suspend(model, engine, txn);
    let (gpu_ids, host_slots) = suspend.copy_plan();
    let copy_started = Instant::now();
    let completion = match crate::scheduler::copy_d2h_tracked(engine, &gpu_ids, &host_slots) {
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
    let (model, engine) = planner.locus();
    let working_sets: HashSet<WorkingSetId> =
        crate::inferlet::process::residency::kv_working_set_ids(pid, model, engine);
    if working_sets.is_empty() {
        drop(pages);
        planner.report_restored(pid, 0);
        return;
    }
    let stores = crate::store::registry::get(model, engine);
    let prepared = crate::store::registry::with_kv_lock(&stores.kv, "planner-restore", |kv| {
        kv.prepare_restore(&working_sets, pages.lend())
    });
    drop(pages);
    let txn = match prepared {
        Ok(txn) => txn,
        Err(error) => {
            planner.restore_deferred(pid, &error.to_string());
            return;
        }
    };
    if txn.page_count() == 0 {
        let committed = crate::store::registry::with_kv_lock(&stores.kv, "planner-restore", |kv| {
            kv.commit_restore(txn)
        });
        match committed {
            Ok(_) => planner.report_restored(pid, 0),
            Err(error) => planner.restore_failed(pid, &error.to_string()),
        }
        return;
    }
    let mut restore = ResidencyTxnGuard::restore(model, engine, txn);
    let (gpu_ids, host_slots) = restore.copy_plan();
    let copy_started = Instant::now();
    let completion = match crate::scheduler::copy_h2d_tracked(engine, &gpu_ids, &host_slots) {
        Ok(completion) => completion,
        Err(error) => {
            restore.abort_now();
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
            crate::scheduler::nudge(engine);
        }
        Err(error) => {
            planner.restore_failed(pid, &error.to_string());
        }
    }
}
