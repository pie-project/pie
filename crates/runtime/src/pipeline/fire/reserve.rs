//! A fire's reservation: its kv and rs demand, held until it is submitted and queued to settle.

use super::*;

pub(super) fn rs_slot_demand(
    stores: &crate::store::registry::Stores,
    ids: &[crate::store::rs::RsWorkingSetId],
    plan: &rs::RsPlan,
) -> Result<crate::store::rs::RsUnits, String> {
    if ids.is_empty() {
        return Ok(Default::default());
    }
    rs::demand(&stores.rs.lock().unwrap(), ids, plan)
}

pub(super) fn prepare_host_kv_reserved(
    stores: &crate::store::registry::Stores,
    ws: &KvWorkingSet,
    written: std::ops::Range<u64>,
    grant: &mut crate::planner::AllocationGrant,
) -> Result<kv::CopiedBeforeWrite, ReservedError> {
    if stores.context_pages != 0 && written.end > stores.context_pages {
        let page = u64::from(stores.kv_page_size);
        return Err(ReservedError::Fatal(format!(
            "pipeline: this fire would grow its sequence to {} KV page(s) (up to {} tokens), \
             past the {} tokens ({} page(s)) one sequence holds on this engine; raise \
             `[engine] max_context` or send fewer tokens",
            written.end,
            written.end.saturating_mul(page),
            stores.context_pages.saturating_mul(page),
            stores.context_pages,
        )));
    }
    crate::store::registry::with_kv_lock(&stores.kv, "host-other", |store| {
        let required = host_kv_demand_locked(store, ws, written.clone())
            .map_err(|error| ReservedError::Fatal(format!("pipeline: KV demand: {error}")))?;
        if required > grant.remaining_kv() {
            return Err(ReservedError::Stale);
        }
        let (copies, txn) =
            kv::realize_written_reserved(store, ws.id, written.clone(), grant.lend_kv()).map_err(
                |error| ReservedError::Fatal(format!("pipeline: KV copy before write: {error}")),
            )?;
        if let Err(error) = store.ensure_backed_reserved(ws.id, written.end, grant.lend_kv()) {
            if let Some(txn) = txn {
                kv::settle(store, txn);
            }
            return Err(ReservedError::Fatal(format!(
                "pipeline: KV backing frontier: {error}"
            )));
        }
        Ok((copies, txn))
    })
}

pub(super) fn prepare_explicit_kv_reserved(
    stores: &crate::store::registry::Stores,
    ws: &KvWorkingSet,
    write_indexes: &[u64],
    grant: &mut crate::planner::AllocationGrant,
) -> Result<kv::CopiedBeforeWrite, ReservedError> {
    crate::store::registry::with_kv_lock(&stores.kv, "host-other", |store| {
        let required = store.write_demand(ws.id, write_indexes).map_err(|error| {
            ReservedError::Fatal(format!("pipeline: device-geometry demand: {error}"))
        })?;
        if required > grant.remaining_kv() {
            return Err(ReservedError::Stale);
        }
        kv::prepare_explicit_reserved(store, ws.id, write_indexes, grant.lend_kv())
            .map(|(copies, txn)| (copies, Some(txn)))
            .map_err(|error| {
                ReservedError::Fatal(format!("pipeline: device-geometry grant: {error}"))
            })
    })
}

pub(super) struct Reserved {
    pub(super) leases: Vec<KvFireLease>,
    pub(super) copies: kv::PageCopies,
    pub(super) kvtxns: Vec<KvTxnGuard>,
    pub(super) rs: rs::PreparedRs,
    pub(super) epoch: FireEpoch,
    pub(super) unfired: Unfired,
}

/// What a pipeline says of a fire prepared that did not follow, where no
/// refusal said more.
pub(super) const UNFIRED: &str = "pipeline: a fire was prepared and never submitted";

/// A fire prepared and still to be submitted. [`reserve`] ended by
/// committing what it folds into its recurrent states, and each piece's
/// prepare freezes what it folds by a count the host knows
/// ([`begin_piece`]); neither is taken back. So a fire that does not follow
/// fails its pipeline, and its sets with it, however the call that prepared
/// it ends: by a refusal, by a trap, or dropped where it waits. Held from
/// [`reserve`] to the submit, this leaves the failure when it is dropped
/// unless the fire was submitted ([`Unfired::submitted`]); a refusal that
/// left its own reason first keeps it.
pub(super) struct Unfired {
    pub(super) failure: Option<PipelineFailure>,
}

impl Unfired {
    pub(super) fn new(failure: &PipelineFailure) -> Self {
        Self {
            failure: Some(failure.clone()),
        }
    }

    /// The engine took the fire: its pipeline is still to queue it.
    pub(super) fn submitted(mut self) -> Unqueued {
        Unqueued {
            failure: self.failure.take(),
        }
    }
}

impl Drop for Unfired {
    fn drop(&mut self) {
        if let Some(failure) = self.failure.take() {
            fail_unless_failed(&failure, UNFIRED);
        }
    }
}

/// What a pipeline says of a fire the engine took that it never queued to
/// settle.
pub(super) const UNQUEUED: &str = "pipeline: a fire was submitted and never queued to settle";

/// A fire the engine took, still to be queued on its pipeline. A call that
/// ends before it queues the fire (by a trap: nothing between the submit
/// and the queueing waits) leaves a fire nothing finalizes: the fold it
/// places as it settles is never placed, and its sets' cursors stand short
/// of what the engine writes. So it fails its pipeline, and its sets with
/// it. Held from the submit ([`Unfired::submitted`]) to the queueing
/// ([`Unqueued::queue`]), this leaves the failure when it is dropped.
///
/// The marks the submit left on the fire's sets wait on no pipeline. The
/// fire's parts go with the call, and each mark dropped unsettled hangs on
/// the fire's completion ([`Unplaced`], [`RsDeviceFold`]): it lifts where
/// the engine resolves that, or lets its last handle go, as it does for a
/// fire dropped off its pipeline's queue.
#[must_use]
pub(super) struct Unqueued {
    pub(super) failure: Option<PipelineFailure>,
}

impl Unqueued {
    /// `fire` is its pipeline's to finalize from here.
    pub(super) fn queue(mut self, fires: &PendingFireQueue, fire: PendingFire) {
        fires.lock().unwrap().push_back(PendingOp::Fire(fire));
        self.failure = None;
    }
}

impl Drop for Unqueued {
    fn drop(&mut self) {
        if let Some(failure) = self.failure.take() {
            fail_unless_failed(&failure, UNQUEUED);
        }
    }
}

/// `reason` fails the pipeline unless something said why first.
pub(super) fn fail_unless_failed(failure: &PipelineFailure, reason: &str) {
    let mut pipeline = failure.lock().expect("the pipeline failure poisoned");
    if pipeline.is_none() {
        *pipeline = Some(reason.to_string());
    }
}

/// A demand that drifted from its grant is quoted again. What it binds
/// opens the fire's epoch on engine `engine` of model `model`. What it
/// returns holds the pipeline's `failure` to the fire's submit
/// ([`Unfired`]); a reserve refused committed nothing, and leaves none.
#[allow(clippy::too_many_arguments)]
pub(super) async fn reserve<C: FireContext>(
    ctx: &mut C,
    (model, engine): (usize, usize),
    stores: &crate::store::registry::Stores,
    pipeline_id: uuid::Uuid,
    failure: &PipelineFailure,
    sets: &[(&KvWorkingSet, Written)],
    set_lanes: &[Vec<usize>],
    rs_ids: &[crate::store::rs::RsWorkingSetId],
    rs_plan: impl Fn() -> Result<rs::RsPlan, String>,
) -> Result<Reserved, String> {
    let mut stale_at = None;
    'grant: loop {
        let mut kv_pages = 0usize;
        for (set, written) in sets {
            kv_pages = kv_pages.saturating_add(kv_demand(stores, set, written)?);
        }
        let plan = rs_plan().map_err(|error| format!("pipeline: recurrent-state mode: {error}"))?;
        let demand = crate::planner::Demand {
            kv_pages: u32::try_from(kv_pages)
                .map_err(|_| "pipeline: KV demand exceeds the planner ABI".to_string())?,
            rs: rs_slot_demand(stores, rs_ids, &plan)?,
        };
        let alone = || alone_demands(stores, sets, set_lanes, rs_ids, &plan);
        let Some(mut grant) = acquire_grant(ctx, pipeline_id, demand, alone).await? else {
            continue;
        };
        let mut leases = Vec::with_capacity(sets.len());
        for (set, _) in sets {
            match set.fire_lease() {
                Ok(lease) => leases.push(lease),
                Err(crate::store::kv::working_set::FireLeaseError::Fenced) => {
                    drop(leases);
                    drop(grant); // the pages fund the eviction's head
                    settle_and_wait_resident(ctx).await?;
                    continue 'grant;
                }
                Err(error) => return Err(format!("pipeline: KV working set: {error}")),
            }
        }
        match prepare_reserved(stores, sets, rs_ids, &plan, &mut grant) {
            Ok((copies, kvtxns, rs)) => {
                return Ok(Reserved {
                    leases,
                    copies,
                    kvtxns,
                    rs,
                    epoch: FireEpoch::begin(model, engine, stores, ctx.process_id()),
                    unfired: Unfired::new(failure),
                });
            }
            Err(ReservedError::Stale) if requote_on_drift(&mut stale_at, demand) => {}
            Err(ReservedError::Stale) => return Err(stale_demand_error()),
            Err(ReservedError::Fatal(error)) => return Err(error),
        }
    }
}

pub(super) fn prepare_reserved(
    stores: &crate::store::registry::Stores,
    sets: &[(&KvWorkingSet, Written)],
    rs_ids: &[crate::store::rs::RsWorkingSetId],
    plan: &rs::RsPlan,
    grant: &mut crate::planner::AllocationGrant,
) -> Result<(kv::PageCopies, Vec<KvTxnGuard>, rs::PreparedRs), ReservedError> {
    let mut copies: kv::PageCopies = (Vec::new(), Vec::new());
    let mut kvtxns = Vec::with_capacity(sets.len());
    for (set, written) in sets {
        let ((src, dst), txn) = match written {
            Written::Span(span) => prepare_host_kv_reserved(stores, set, span.clone(), grant),
            Written::Pages(pages) => prepare_explicit_kv_reserved(stores, set, pages, grant),
        }?;
        copies.0.extend(src);
        copies.1.extend(dst);
        kvtxns.push(KvTxnGuard::new(set.model, set.engine, txn));
    }
    let rs = prepare_rs_reserved(stores, rs_ids, plan, grant)?;
    Ok((copies, kvtxns, rs))
}
