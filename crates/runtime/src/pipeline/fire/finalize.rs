//! Finalizing fires: draining, settling folds, recording failures.

use super::*;

pub async fn drain_settled<C: FireContext>(
    ctx: &mut C,
    fires: Option<&PendingFires>,
) -> Anyhow<bool> {
    let Some(fires) = fires else {
        return Ok(false);
    };
    let _finalize_guard = fires.finalize_guard().await;
    let mut drained = false;
    loop {
        match pop_settled(Some(fires)) {
            Some(op) => {
                finalize_op(ctx, op).await?;
                drained = true;
            }
            None => return Ok(drained),
        }
    }
}

pub(crate) async fn finalize_all<C: FireContext>(
    ctx: &mut C,
    fires: &PendingFires,
    continue_on_error: bool,
) -> Anyhow<()> {
    let _finalize_guard = fires.finalize_guard().await;
    loop {
        fires.await_front_completion().await;
        let op = fires.lock().unwrap().pop_front();
        let Some(op) = op else {
            return Ok(());
        };
        if let Err(error) = finalize_op(ctx, op).await {
            if !continue_on_error {
                return Err(error);
            }
            tracing::error!(
                pid = %ctx.process_id(),
                %error,
                "failed to finalize a pending pipeline operation"
            );
        }
    }
}

pub(crate) fn pop_settled(fires: Option<&PendingFires>) -> Option<PendingOp> {
    let fires = fires?;
    let mut queue = fires.lock().unwrap();
    if queue.front().is_some_and(PendingOp::is_settled) {
        queue.pop_front()
    } else {
        None
    }
}

pub async fn finalize_op<C: FireContext>(ctx: &mut C, op: PendingOp) -> Anyhow<()> {
    let finalized = finalize_op_await(op).await;
    complete_finalize(ctx, finalized);
    Ok(())
}

/// What a fire off the queue leaves: its fold written, and its failure to
/// latch ([`complete_finalize`]). It has no error to return. A finalizer
/// holds the gate from the fire leaving the queue to that latch, and an
/// exit between the two would lift the gate over a set whose cursor the
/// fire moved and whose pipeline nothing had failed.
pub(crate) async fn finalize_op_await(op: PendingOp) -> FinalizeOutcome {
    match op {
        PendingOp::Fire(fire) => finalize_fire_await(fire).await,
        #[cfg(test)]
        PendingOp::TestStub => FinalizeOutcome {
            action: FinalizeAction::None,
            ws_guard: Vec::new(),
        },
    }
}

pub(crate) fn complete_finalize<C: FireContext>(ctx: &mut C, finalized: FinalizeOutcome) {
    let FinalizeOutcome { action, ws_guard } = finalized;
    match action {
        FinalizeAction::None => {}
        FinalizeAction::Fail {
            fwd_rep,
            cells,
            failure,
            reason,
        } => {
            poison_readers(&cells, &reason);
            fail_pass(ctx, fwd_rep, &reason);
            let mut domain = failure.lock().unwrap();
            if domain.is_none() {
                *domain = Some(reason);
            }
        }
    }
    drop(ws_guard);
}

pub(crate) async fn finalize_op_detached(op: PendingOp) -> Anyhow<()> {
    debug_assert!(op.is_preemption_detachable());
    finalize_unattached(op).await;
    Ok(())
}

/// Settles every fire `pid` has submitted, in each pipeline's order, from
/// its own task where it is about to wait outside any gate that drains its
/// pipelines (a bind waiting for room, or for its restore,
/// [`crate::store::roles::bind_when_resident`]): each fire's leases on its
/// working sets go back as it lands. It holds each pipeline's finalizer
/// gate, so it is that pipeline's one finalizer and the one waiter on its
/// fires' completions. A fire's pass is not reached from here: what failed
/// fails its pipeline and poisons its readers, as an eviction's drain does.
pub(crate) async fn settle_process_fires(pid: uuid::Uuid) {
    for fires in crate::inferlet::process::residency::pipelines_of(pid) {
        let _gate = fires.finalize_guard().await;
        loop {
            fires.await_front_completion().await;
            let op = fires.lock().unwrap().pop_front();
            let Some(op) = op else {
                break;
            };
            finalize_unattached(op).await;
        }
    }
}

/// Finalizes `op` with no context: its failure latched on its pipeline and
/// its readers poisoned, its leases dropped after.
pub(super) async fn finalize_unattached(op: PendingOp) {
    let FinalizeOutcome { action, ws_guard } = finalize_op_await(op).await;
    match action {
        FinalizeAction::None => {}
        FinalizeAction::Fail {
            cells,
            failure,
            reason,
            ..
        } => {
            poison_readers(&cells, &reason);
            let mut domain = failure.lock().unwrap();
            if domain.is_none() {
                *domain = Some(reason);
            }
        }
    }
    drop(ws_guard);
}

/// What the fire froze seals; a fold placed only now is frozen first, the
/// working set's one folded length moving by the count the engine reports
/// or the host stated, and its recurrent state's by the same count. The
/// last such fold of the set to settle makes that length exact again.
/// `written_end` is where the fire's rows ended, where they are known.
/// `failure` is why the fire did not run to its end, when it did not: what
/// it was to freeze only now stays unfrozen, and a recurrent state it
/// folded by a count the device computes is left refused by that name. One
/// whose fire ran to its end and brought no count is left refused too
/// ([`RsDeviceFold::UNREPORTED`]): the engine's record went on by a count
/// the frozen length never takes, and what copies or moves the set would
/// state a length behind it.
pub(super) fn settle_fold(
    stores: &crate::store::registry::Stores,
    ws: crate::store::kv::page_table::WorkingSetId,
    fold: FoldAt,
    reported: &std::collections::HashMap<u64, ::engine::KvWrites>,
    written_end: Option<u64>,
    failure: Option<&str>,
) {
    let success = failure.is_none();
    let FoldAt { mut by, decoded } = fold;
    // Taken before the kv lock: ending the one takes the rs store's, and
    // the other, dropped unsettled, the kv store's. The count is the
    // host's, or what the engine reported under the lanes' tags; a fire
    // that ran and brought neither went unreported.
    let (device_fold, unreported, mut unplaced) = match &mut by {
        FoldBy::Settled {
            count,
            tags,
            rs,
            unplaced,
            ..
        } => {
            *count = count.or_else(|| {
                tags.iter()
                    .map(|tag| reported.get(tag).and_then(|written| written.fold))
                    .collect::<Option<Vec<u32>>>()
                    .and_then(|counts| counts.into_iter().max())
            });
            (rs.take(), success && count.is_none(), unplaced.take())
        }
        FoldBy::To(_) => (None, false, None),
    };
    let mut device_slots = None;
    let folded = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
        let frozen = kv.folded(ws)?;
        let to = match by {
            FoldBy::To(to) => Some(to),
            FoldBy::Settled {
                count,
                tags: _,
                reach,
                written,
                rs: _,
                unplaced: _,
            } => {
                let reach = match (written, written_end) {
                    (false, _) => Some(reach),
                    (true, end) => end.map(|end| end.min(reach)),
                };
                let to = count
                    .zip(reach)
                    .map(|(count, reach)| frozen.saturating_add(u64::from(count)).min(reach));
                // The recurrent state froze the same slots on the device.
                device_slots = to
                    .filter(|_| success)
                    .map(|to| u32::try_from(to.saturating_sub(frozen)).unwrap_or(u32::MAX));
                to
            }
        };
        let placed = to.filter(|_| success);
        let pages = match placed {
            Some(to) => kv.fold(ws, to, decoded),
            None => Ok(0),
        };
        // The fold is settled whether or not it was placed: one never
        // placed leaves the folded length a lower bound.
        if let Some(unplaced) = &mut unplaced {
            unplaced.settle(kv, placed.is_some() && pages.is_ok());
        }
        pages
    });
    // One whose set was gone lifts nothing, outside the store's lock.
    drop(unplaced);
    match folded {
        Ok(pages) => {
            SEALED_PAGES.fetch_add(pages, std::sync::atomic::Ordering::Relaxed);
        }
        Err(error) => {
            tracing::debug!(%error, "pipeline: sealing what a fire froze");
            device_slots = None;
        }
    }
    // The count lands and the mark lifts together, whichever way the fire
    // ended: what waited on the set reads the frozen length this leaves.
    if let Some(device_fold) = device_fold {
        let failure = failure.or(unreported.then_some(RsDeviceFold::UNREPORTED));
        device_fold.settle(device_slots, failure);
    }
}

pub(super) async fn finalize_fire_await(fire: PendingFire) -> FinalizeOutcome {
    let PendingFire {
        completion,
        checkpoints,
        kv,
        device_geometry: _,
        rstxn,
        model,
        engine,
        fwd_rep,
        cells,
        failure,
        retired,
        lane: _,
    } = fire;
    let prior_failure = failure.lock().unwrap().clone();
    let result = completion.await;
    let failure_reason = prior_failure.or_else(|| {
        result
            .err()
            .map(|error| format!("pipeline: forward failed: {error:#}"))
    });
    let success = failure_reason.is_none();
    let mut leases = Vec::with_capacity(kv.len());
    // A checkpoint the pass asked and could not keep fails it by name.
    let mut refused = None;

    {
        let stores = crate::store::registry::get(model, engine);
        let txn = rstxn.into_inner();
        if txn.is_some() {
            let mut rs_store = stores.rs.lock().unwrap();
            rs::settle(&mut rs_store, txn);
        }
        let mut logs = Vec::with_capacity(kv.len());
        let mut folds = Vec::with_capacity(kv.len());
        let mut staged = Vec::new();
        let mut reported = std::collections::HashMap::new();
        for part in kv {
            if let Some(kvtxn) = part.kvtxn.into_inner() {
                crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv_store| {
                    kv::settle(kv_store, kvtxn);
                });
            }
            for tag in &part.reports {
                if let Some(written) = tag.take() {
                    reported.insert(tag.0, written);
                }
            }
            if let Some(log) = part.log {
                logs.push((part.ws, log));
            }
            if let Some(fold) = part.fold {
                folds.push((part.ws, fold));
            }
            staged.extend(part.staged);
            leases.push(part.lease);
        }
        // Where its rows ended, before what it froze: a set never counts
        // fewer rows unfolded than stand in its buffer.
        for fire in &staged {
            fire.settle(&reported, success);
        }
        // Where each working set's rows ended, where the fire's rows are known.
        let mut written_ends = std::collections::HashMap::new();
        for (ws, LogFire { seq, rows }) in logs {
            let known = match rows {
                _ if !success => None,
                LogRows::Known { start, tokens } => Some((start, tokens)),
                LogRows::Reported {
                    tag,
                    pages,
                    page_size,
                } => reported
                    .get(&tag)
                    .and_then(|written| reported_rows(written, &pages, page_size)),
                LogRows::Unknown => None,
            };
            let logged = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
                kv.log_end(
                    ws,
                    seq,
                    known
                        .as_ref()
                        .map(|(start, tokens)| (*start, tokens.as_slice())),
                )
            });
            if let Err(error) = logged {
                tracing::debug!(%error, "pipeline: logging what a fire wrote");
            }
            if let Some((start, tokens)) = &known {
                written_ends.insert(ws, start + tokens.len() as u64);
            }
        }
        // What the fire froze seals now that its rows are logged.
        for (ws, fold) in folds {
            let written_end = written_ends.get(&ws).copied();
            settle_fold(
                &stores,
                ws,
                fold,
                &reported,
                written_end,
                failure_reason.as_deref(),
            );
        }
        for checkpoint in &checkpoints {
            if let Err(refusal) = take_checkpoint(&stores, checkpoint, success) {
                refused.get_or_insert(refusal);
            }
        }
    }
    let failure_reason = failure_reason.or(refused);

    for planner in crate::planner::planners() {
        planner.pages_freed();
    }

    if let Some(mut pass) = retired {
        pass.close_native();
    }

    let action = if let Some(reason) = failure_reason {
        FinalizeAction::Fail {
            fwd_rep,
            cells,
            failure,
            reason,
        }
    } else {
        FinalizeAction::None
    };
    FinalizeOutcome {
        action,
        ws_guard: leases,
    }
}

pub(super) fn fail_pass<C: FireContext>(ctx: &mut C, fwd_rep: u32, reason: &str) {
    let res: Resource<ForwardPass> = Resource::new_borrow(fwd_rep);
    if let Ok(p) = ctx.resources().get_mut(&res)
        && p.failed.is_none()
    {
        p.failed = Some(reason.to_string());
    }
}

pub(super) fn record_submit_failure<C: FireContext>(
    ctx: &mut C,
    fwd: &Resource<ForwardPass>,
    failure: &PipelineFailure,
    reason: &str,
) {
    if let Ok(pass) = ctx.resources().get_mut(fwd)
        && pass.failed.is_none()
    {
        pass.failed = Some(reason.to_string());
    }
    let mut pipeline = failure.lock().unwrap();
    if pipeline.is_none() {
        *pipeline = Some(reason.to_string());
    }
}

pub(super) fn fold_of_pass(pass: &crate::pipeline::instance::BoundForwardPass) -> geometry::Fold {
    let bound = &pass.instance.program.bound;
    if !bound
        .container
        .ports
        .iter()
        .any(|binding| binding.port == Port::Fold)
    {
        return geometry::Fold::None;
    }
    let mut known = |chan: u32| pass.host_shadow.fire_value(bound, &pass.cells, chan);
    match eta_compiler::eval::pareval::eval_descriptor_ports(bound, &mut known) {
        Ok(ports) => ports
            .into_iter()
            .find(|(port, _)| *port == Port::Fold)
            .and_then(|(_, value)| value.ok())
            .map_or(geometry::Fold::Device, |value| {
                geometry::Fold::Host(geometry::value_as_u32(&value))
            }),
        Err(_) => geometry::Fold::Device,
    }
}
