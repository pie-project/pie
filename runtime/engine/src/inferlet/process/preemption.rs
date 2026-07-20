//! Process-owned KV preemption lifecycle.
//!
//! WIT bindings delegate here; this module freezes scheduler preparation,
//! drains process fire queues, executes transactional D2H/H2D, and parks the
//! WASM continuation without putting domain orchestration in `inferlet/host`.

use std::collections::HashSet;
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
) {
    let snapshot = residency.lock().unwrap().snapshot();
    let mut context = TeardownFireContext {
        process_id,
        resources,
    };
    if snapshot
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
        return;
    };
    runtime.spawn(async move {
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
        drop(context);
    });
}

enum ResidencyTxn {
    Suspend(KvSuspendTxn),
    Restore(KvRestoreTxn),
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

    fn abort_now(&mut self) {
        let Some(txn) = self.txn.take() else {
            return;
        };
        let stores = crate::store::registry::get(self.model, self.driver);
        let tag = match &txn {
            ResidencyTxn::Suspend(_) => "preemption-suspend",
            ResidencyTxn::Restore(_) => "preemption-restore",
        };
        crate::store::registry::with_kv_lock(&stores.kv, tag, |kv| match txn {
            ResidencyTxn::Suspend(txn) => kv.abort_suspend(txn),
            ResidencyTxn::Restore(txn) => kv.abort_restore(txn),
        });
    }
}

impl Drop for ResidencyTxnGuard {
    fn drop(&mut self) {
        let Some(txn) = self.txn.take() else {
            return;
        };
        let Some(completion) = self.completion.take() else {
            self.txn = Some(txn);
            self.abort_now();
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
            let stores = crate::store::registry::get(model, driver);
            let tag = match &txn {
                ResidencyTxn::Suspend(_) => "preemption-suspend",
                ResidencyTxn::Restore(_) => "preemption-restore",
            };
            crate::store::registry::with_kv_lock(&stores.kv, tag, |kv| match txn {
                ResidencyTxn::Suspend(txn) => kv.abort_suspend(txn),
                ResidencyTxn::Restore(txn) => kv.abort_restore(txn),
            });
            if let Some(orchestrator) = crate::store::reclaim::contention() {
                orchestrator.on_blocks_freed();
            }
        });
    }
}

async fn drain_preemption_safe_fires(ctx: &mut ProcessCtx) -> Result<()> {
    let pipelines = ctx.residency_snapshot().pipelines;
    for fires in pipelines {
        let _finalize_guard = fires.finalize_guard().await;
        loop {
            let op = {
                let mut queue = fires.lock().unwrap();
                if queue
                    .front()
                    .is_some_and(PendingOp::is_preemption_safe_unprepared)
                {
                    None
                } else {
                    queue.pop_front()
                }
            };
            let Some(op) = op else {
                break;
            };
            crate::pipeline::fire::finalize_op(ctx, op).await?;
        }
    }
    Ok(())
}

fn decline_park(pid: uuid::Uuid) {
    if let Some(orchestrator) = crate::store::reclaim::contention() {
        orchestrator.decline_park(pid);
    }
    crate::scheduler::resume_pipeline(pid);
    crate::scheduler::nudge(0);
}

pub(crate) async fn contention_gate(ctx: &mut ProcessCtx) -> Result<()> {
    let Some(orchestrator) = crate::store::reclaim::contention() else {
        return Ok(());
    };
    loop {
        honor(ctx).await?;
        if !orchestrator.contended() {
            return Ok(());
        }
        drain_preemption_safe_fires(ctx).await?;
        let progress = ctx
            .residency_snapshot()
            .pipelines
            .into_iter()
            .find_map(|fires| {
                fires
                    .lock()
                    .unwrap()
                    .front()
                    .map(PendingOp::preemption_signal)
            });
        let Some(progress) = progress else {
            return Ok(());
        };
        let Some(park_signal) = orchestrator.park_signal(ctx.id()) else {
            progress.await;
            continue;
        };
        if orchestrator.should_park(ctx.id()) {
            continue;
        }
        let notified = park_signal.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        if orchestrator.should_park(ctx.id()) {
            continue;
        }
        tokio::select! {
            _ = progress => {}
            _ = &mut notified => {}
        }
    }
}

pub(crate) async fn honor_idle(
    pid: uuid::Uuid,
    residency: Arc<Mutex<crate::inferlet::process::ProcessResidency>>,
) -> Result<bool> {
    let Some(orchestrator) = crate::store::reclaim::contention() else {
        return Ok(false);
    };
    if !orchestrator.should_park(pid) {
        return Ok(false);
    }
    if !orchestrator.begin_quiesce(pid) {
        return Ok(false);
    }
    if let Err(error) = crate::scheduler::freeze_pipeline(pid).await {
        decline_park(pid);
        return Err(error).context("freeze scheduler preparation for idle suspension");
    }
    let snapshot = residency.lock().unwrap().snapshot();
    for fires in &snapshot.pipelines {
        let _finalize_guard = fires.finalize_guard().await;
        loop {
            let op = {
                let mut queue = fires.lock().unwrap();
                match queue.front() {
                    Some(op) if op.is_preemption_safe_unprepared() => None,
                    Some(op) if op.is_preemption_detachable() => queue.pop_front(),
                    Some(_) => {
                        drop(queue);
                        decline_park(pid);
                        return Ok(false);
                    }
                    None => None,
                }
            };
            let Some(op) = op else {
                break;
            };
            if let Err(error) = crate::pipeline::fire::finalize_op_detached(op).await {
                decline_park(pid);
                return Err(error);
            }
        }
    }
    let working_sets: HashSet<WorkingSetId> = snapshot
        .kv_working_sets
        .into_iter()
        .filter_map(|(model, driver, ws)| (model == 0 && driver == 0).then_some(ws))
        .collect();
    if working_sets.is_empty() {
        decline_park(pid);
        return Ok(true);
    }
    tracing::debug!(
        pid = %pid,
        rs_working_sets = snapshot.rs_working_sets.len(),
        "idle host await honoring KV-only suspension"
    );
    suspend_restore(pid, working_sets).await?;
    Ok(true)
}

pub(crate) async fn receive_message(
    process_id: uuid::Uuid,
    residency: Arc<Mutex<crate::inferlet::process::ProcessResidency>>,
) -> Result<Option<String>> {
    loop {
        let Some(orchestrator) = crate::store::reclaim::contention() else {
            return crate::server::inbox::receive(process_id.to_string())
                .await
                .with_context(|| format!("session.receive failed for process {process_id}"))
                .map(Some);
        };
        if orchestrator.should_park(process_id) && honor_idle(process_id, residency.clone()).await?
        {
            continue;
        }
        let Some(signal) = orchestrator.park_signal(process_id) else {
            return crate::server::inbox::receive(process_id.to_string())
                .await
                .with_context(|| format!("session.receive failed for process {process_id}"))
                .map(Some);
        };
        let notified = signal.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        if orchestrator.should_park(process_id) {
            tokio::task::yield_now().await;
            continue;
        }
        tokio::select! {
            result = crate::server::inbox::receive(process_id.to_string()) => {
                return result
                    .with_context(|| format!("session.receive failed for process {process_id}"))
                    .map(Some);
            }
            _ = &mut notified => {}
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
    loop {
        let Some(orchestrator) = crate::store::reclaim::contention() else {
            return match crate::server::receive_file(client_id, process_id).await {
                Ok(data) => Ok(Some(data.to_vec())),
                Err(error) => {
                    tracing::warn!(
                        client_id,
                        process_id = %process_id,
                        %error,
                        "session.receive_file delivery failed"
                    );
                    Ok(None)
                }
            };
        };
        if orchestrator.should_park(process_id) && honor_idle(process_id, residency.clone()).await?
        {
            continue;
        }
        let Some(signal) = orchestrator.park_signal(process_id) else {
            continue;
        };
        let notified = signal.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        if orchestrator.should_park(process_id) {
            tokio::task::yield_now().await;
            continue;
        }
        tokio::select! {
            result = crate::server::receive_file(client_id, process_id) => {
                return match result {
                    Ok(data) => Ok(Some(data.to_vec())),
                    Err(error) => {
                        tracing::warn!(
                            client_id,
                            process_id = %process_id,
                            %error,
                            "session.receive_file delivery failed"
                        );
                        Ok(None)
                    }
                };
            }
            _ = &mut notified => {}
        }
    }
}

pub(crate) async fn honor(ctx: &mut ProcessCtx) -> Result<()> {
    let Some(orchestrator) = crate::store::reclaim::contention() else {
        return Ok(());
    };
    let pid = ctx.id();
    if !orchestrator.should_park(pid) || !orchestrator.begin_quiesce(pid) {
        return Ok(());
    }
    if let Err(error) = crate::scheduler::freeze_pipeline(pid).await {
        decline_park(pid);
        return Err(error).context("freeze scheduler preparation for suspension");
    }
    if let Err(error) = drain_preemption_safe_fires(ctx).await {
        decline_park(pid);
        return Err(error);
    }
    let snapshot = ctx.residency_snapshot();
    let rs_slots_remain_resident = snapshot.rs_working_sets.len();
    let working_sets: HashSet<WorkingSetId> = snapshot
        .kv_working_sets
        .into_iter()
        .filter_map(|(model, driver, ws)| (model == 0 && driver == 0).then_some(ws))
        .collect();
    if working_sets.is_empty() {
        decline_park(pid);
        return Ok(());
    }
    tracing::debug!(
        pid = %pid,
        rs_working_sets = rs_slots_remain_resident,
        "quiesced process for KV-only suspension; recurrent state remains resident"
    );
    suspend_restore(pid, working_sets).await
}

async fn suspend_restore(pid: uuid::Uuid, working_sets: HashSet<WorkingSetId>) -> Result<()> {
    let orchestrator = crate::store::reclaim::contention()
        .context("KV contention orchestrator disappeared during suspend")?;
    let stores = crate::store::registry::get(0, 0);
    let prepared =
        match crate::store::registry::with_kv_lock(&stores.kv, "preemption-suspend", |kv| {
            kv.prepare_suspend(&working_sets)
        }) {
            Ok(prepared) => prepared,
            Err(crate::store::kv::KvStoreError::HostSwapFull { .. }) => {
                orchestrator.record_host_swap_exhaustion();
                decline_park(pid);
                return Ok(());
            }
            Err(error) => {
                decline_park(pid);
                return Err(error).context("prepare KV suspend");
            }
        };
    let txn = match prepared {
        KvSuspendPrepare::Prepared(txn) => txn,
        KvSuspendPrepare::Deferred(
            SuspendDisposition::NothingReclaimable | SuspendDisposition::GraceDeferred,
        ) => {
            decline_park(pid);
            return Ok(());
        }
    };

    let mut suspend = ResidencyTxnGuard::suspend(0, 0, txn);
    let gpu_ids = match suspend.txn.as_ref() {
        Some(ResidencyTxn::Suspend(txn)) => txn.gpu_ids(),
        _ => unreachable!(),
    };
    let host_slots = match suspend.txn.as_ref() {
        Some(ResidencyTxn::Suspend(txn)) => txn.host_slots(),
        _ => unreachable!(),
    };
    let copy_started = Instant::now();
    let completion = match crate::scheduler::copy_d2h_tracked(0, &gpu_ids, &host_slots) {
        Ok(completion) => completion,
        Err(error) => {
            suspend.abort_now();
            orchestrator.record_suspend_rollback();
            decline_park(pid);
            tracing::warn!(pid = %pid, %error, "KV D2H suspend copy rejected");
            return Ok(());
        }
    };
    suspend.arm(completion.clone());
    if let Err(error) = completion.wait().await {
        orchestrator.record_d2h_copy(copy_started.elapsed());
        suspend.disarm_completion();
        suspend.abort_now();
        orchestrator.record_suspend_rollback();
        decline_park(pid);
        tracing::warn!(pid = %pid, %error, "KV D2H suspend copy failed");
        return Ok(());
    }
    orchestrator.record_d2h_copy(copy_started.elapsed());
    suspend.disarm_completion();
    let txn = suspend.take_suspend();
    let freed = match crate::store::registry::with_kv_lock(&stores.kv, "preemption-suspend", |kv| {
        kv.commit_suspend(txn)
    }) {
        Ok(freed) => freed,
        Err(error) => {
            orchestrator.record_suspend_rollback();
            decline_park(pid);
            return Err(error).context("commit KV suspend");
        }
    };
    orchestrator.report_suspended(pid, freed as u32);

    let max_restore_attempts = std::env::var("PIE_KV_RESTORE_RETRIES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(3)
        .max(1);
    for attempt in 1..=max_restore_attempts {
        let grant = orchestrator
            .park_until_restore_granted(pid)
            .await
            .context("wait for KV restore grant")?;
        let txn =
            match crate::store::registry::with_kv_lock(&stores.kv, "preemption-restore", |kv| {
                kv.prepare_restore(&working_sets, grant.into_pages())
            }) {
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
        let mut restore = ResidencyTxnGuard::restore(0, 0, txn);
        orchestrator.record_restore_prepared();
        let gpu_ids = match restore.txn.as_ref() {
            Some(ResidencyTxn::Restore(txn)) => txn.gpu_ids(),
            _ => unreachable!(),
        };
        let host_slots = match restore.txn.as_ref() {
            Some(ResidencyTxn::Restore(txn)) => txn.host_slots(),
            _ => unreachable!(),
        };
        let copy_started = Instant::now();
        let completion = match crate::scheduler::copy_h2d_tracked(0, &gpu_ids, &host_slots) {
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
        crate::scheduler::nudge(0);
        return Ok(());
    }
    unreachable!("restore loop has at least one attempt")
}

pub(crate) async fn await_channel_progress_idle(
    process_id: uuid::Uuid,
    residency: Arc<Mutex<crate::inferlet::process::ProcessResidency>>,
    cell: &Arc<Mutex<crate::pipeline::channel::ChannelCell>>,
    fires: Option<&PendingFires>,
) -> Result<(), String> {
    let Some(orchestrator) = crate::store::reclaim::contention() else {
        return crate::pipeline::fire::await_channel_progress(cell, fires).await;
    };
    let Some(signal) = orchestrator.park_signal(process_id) else {
        return crate::pipeline::fire::await_channel_progress(cell, fires).await;
    };
    if orchestrator.should_park(process_id) {
        honor_idle(process_id, residency.clone())
            .await
            .map_err(|error| error.to_string())?;
        return Ok(());
    }
    let notified = signal.notified();
    tokio::pin!(notified);
    notified.as_mut().enable();
    if orchestrator.should_park(process_id) {
        honor_idle(process_id, residency.clone())
            .await
            .map_err(|error| error.to_string())?;
        return Ok(());
    }
    tokio::select! {
        result = crate::pipeline::fire::await_channel_progress(cell, fires) => result,
        _ = &mut notified => {
            honor_idle(process_id, residency)
                .await
                .map_err(|error| error.to_string())?;
            Ok(())
        }
    }
}

pub(crate) async fn await_writer_progress(
    ctx: &mut ProcessCtx,
    endpoint: &Arc<crate::driver::ChannelEndpoint>,
    observed_head: u64,
) -> Result<(), String> {
    let Some(orchestrator) = crate::store::reclaim::contention() else {
        return endpoint
            .wait_for_writer_change(observed_head)
            .await
            .map_err(|error| error.to_string());
    };
    let Some(signal) = orchestrator.park_signal(ctx.id()) else {
        return endpoint
            .wait_for_writer_change(observed_head)
            .await
            .map_err(|error| error.to_string());
    };
    if orchestrator.should_park(ctx.id()) {
        honor(ctx).await.map_err(|error| error.to_string())?;
        return Ok(());
    }
    let notified = signal.notified();
    tokio::pin!(notified);
    notified.as_mut().enable();
    if orchestrator.should_park(ctx.id()) {
        honor(ctx).await.map_err(|error| error.to_string())?;
        return Ok(());
    }
    tokio::select! {
            result = endpoint.wait_for_writer_change(observed_head) => {
                result.map_err(|error| error.to_string())
            }
            _ = &mut notified => {
                honor(ctx).await.map_err(|error| error.to_string())?;
                Ok(())
        }
    }
}
