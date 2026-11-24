//! A beam group's relist: the grant a fire acquires, and the wait on the tails it lands behind.

use super::*;

/// `windowed`: the fire reads window slots, which come back only once every
/// fire that read them settles (its process finalizes it). A fire that would
/// park finalizes its own tail as it lands while it waits, rather than
/// settle it whole first: the tail may hold this pass's earlier pieces,
/// which make up one frame with the piece now asking, and settling them
/// before this ask returns waited on a frame only this ask completes (Mac's
/// gemma-4-e4b c64 stall). The lane is parked first, so the scheduler waits
/// on none of its fires.
///
/// `alone`: what each unit of the fire takes on the device by itself
/// ([`alone_demands`]), read only where the planner cannot grant the whole
/// at once: a unit past the whole device is refused by name, never waited
/// on (over-demand is served slower; only a need that could never fit
/// alone is refused).
pub(super) async fn acquire_grant<C: FireContext>(
    ctx: &mut C,
    pipeline_id: uuid::Uuid,
    demand: crate::planner::Demand,
    windowed: bool,
    alone: impl FnOnce() -> Result<Vec<crate::planner::Demand>, String>,
) -> Result<Option<crate::planner::AllocationGrant>, String> {
    let pid = ctx.process_id();
    let Some(planner) = crate::planner::planner_of(pid) else {
        if demand.is_zero() {
            return Ok(Some(crate::planner::AllocationGrant::empty()));
        }
        return Err("pipeline: KV residency planner is not installed".to_string());
    };
    // Even an empty grant is the planner's: a process that is not resident
    // gets none, and its fire waits for its restore.
    if demand.is_zero()
        && let Some(grant) = planner.grant_now(pid, demand)
    {
        return Ok(Some(grant));
    }
    if windowed {
        if let Some(grant) = planner.grant_now(pid, demand) {
            planner.note_peak(pid, demand);
            return Ok(Some(grant));
        }
        crate::scheduler::worker::notify_lane_close(pipeline_id, Some(pid));
    } else if !demand.is_zero()
        && let Some(grant) = planner.grant_now(pid, demand)
    {
        planner.note_peak(pid, demand);
        return Ok(Some(grant));
    }
    // Granted whole, every unit fit beside the others; short, one past the
    // device alone is refused here rather than parked.
    if !demand.is_zero() {
        for unit in alone()? {
            if let Some(error) = planner.past_device(unit) {
                return Err(format!("pipeline: KV capacity: {error}"));
            }
        }
    }
    // A yield may have dropped snapshots the demand was quoted against, so
    // the caller quotes it again rather than asking for the same.
    let acquire = crate::inferlet::process::said(
        "plan: the planner's grant",
        pid,
        planner.acquire(pid, pipeline_id, demand),
    );
    let acquired = if windowed {
        ask_while_tail_lands(ctx, acquire, || planner.pages_freed()).await?
    } else {
        acquire.await
    };
    // A request the planner dropped while its process is still its own
    // (its entry gone under it) is asked again, quoted anew: a wait, never
    // the client's failure (a beam must not fail with "planner request
    // cancelled"). One whose
    // process left the planner is its departure's.
    if matches!(acquired, Err(crate::planner::PlannerError::Cancelled)) && planner.registered(pid) {
        tracing::debug!(%pid, "pipeline: a KV request the planner dropped is asked again");
        return Ok(None);
    }
    match acquired.map_err(|error| format!("pipeline: KV capacity: {error}"))? {
        crate::planner::Acquired::Granted(grant) => Ok(Some(grant)),
        crate::planner::Acquired::Yield => {
            settle_and_wait_resident(ctx).await?;
            Ok(None)
        }
        crate::planner::Acquired::Requote => Ok(None),
    }
}

/// A beam search's finish of kv set `ws` from page `f0` by `order`
/// (`KvStore::relist_beam_with`): once the set is out of any move (a
/// suspend or a sweep in flight pins its old tail: waited on, never a
/// refusal), and with each chosen page something else still reads copied
/// first onto a page the planner grants (copy-on-write, the rule every
/// shared page a set changes keeps), so no page ends up named twice.
pub(crate) async fn relist_beam<C: FireContext>(
    ctx: &mut C,
    model: usize,
    engine: usize,
    ws: crate::store::kv::page_table::WorkingSetId,
    leases: Option<crate::store::kv::working_set::KvSuspendHandle>,
    f0: u64,
    order: &[u32],
) -> Result<(), String> {
    let stores = crate::store::registry::get(model, engine);
    // Every fire the process submitted lands first, a frame the scheduler
    // holds out under the rung's cap among them: built before the finish,
    // its lanes name the set's tail as it stood (each lane's reserve is the
    // set's whole table at its submit), and fired after it, it read the
    // pages the finish let go (a freed page of the old tail, which a swap
    // may since have put on a released segment).
    ctx.settle_pipeline_tail()
        .await
        .map_err(|error| format!("a beam's finish waiting for its fires to land: {error:#}"))?;
    // And every fire the set itself has submitted, on any pipeline (a closed
    // one's run-ahead fires included, which the process's tail no longer
    // lists): each holds a lease on the set until it lands (a fire landing
    // after the finish carries the set's pre-relist reserve).
    if let Some(leases) = &leases {
        leases.quiesce().await;
    }
    loop {
        let needs = match crate::store::registry::with_kv_lock(&stores.kv, "host-working-set", |kv| {
            kv.relist_beam_needs(ws, f0, order)
        }) {
            Ok(needs) => needs,
            // Its pages went below with its process (evicted since its
            // last fire): the finish waits for its restore, as a fire does,
            // and goes then.
            Err(crate::store::kv::KvStoreError::Table(
                crate::store::kv::page_table::KvTableError::NonResident { .. },
            )) => {
                settle_and_wait_resident(ctx).await?;
                continue;
            }
            Err(error) => return Err(error.to_string()),
        };
        let shared = match needs {
            crate::store::kv::RelistNeeds::MoveEnd(ended) => {
                let notified = ended.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                if crate::store::registry::with_kv_lock(&stores.kv, "host-working-set", |kv| {
                    kv.set_in_move(ws)
                }) {
                    notified.await;
                }
                continue;
            }
            crate::store::kv::RelistNeeds::Copies(shared) => shared,
        };
        let mut copies = Vec::new();
        let mut grant = None;
        if !shared.is_empty() {
            let demand = crate::planner::Demand {
                kv_pages: u32::try_from(shared.len()).unwrap_or(u32::MAX),
                rs: crate::store::rs::RsUnits::default(),
            };
            let pid = ctx.process_id();
            let Some(mut granted) = acquire_grant(ctx, pid, demand, false, || Ok(vec![demand])).await? else {
                continue;
            };
            let pages: Vec<crate::store::kv::page_table::PhysicalKvPageId> =
                granted.lend_kv().drain(..shared.len()).collect();
            grant = Some(granted);
            let shape = crate::store::registry::with_kv_lock(&stores.kv, "host-working-set", |kv| kv.list_shape());
            let copy = crate::scheduler::TierMove {
                kind: crate::scheduler::TierKind::kv(shape),
                tier: crate::scheduler::device_domain(engine),
                out: true,
                device: shared.iter().map(|page| page.0).collect(),
                slots: pages.iter().map(|page| page.0).collect(),
                source: None,
            };
            let copied = match crate::scheduler::copy_tier_tracked(engine, &copy) {
                Ok(completion) => completion.wait().await.map_err(|error| error.to_string()),
                Err(error) => Err(error.to_string()),
            };
            if let Err(error) = copied {
                crate::store::registry::with_kv_lock(&stores.kv, "host-working-set", |kv| {
                    kv.release_device_reservation(pages);
                });
                return Err(format!("a beam's finish copying the chosen pages another reads: {error}"));
            }
            copies = shared.into_iter().zip(pages).collect::<Vec<_>>();
        }
        let done = crate::store::registry::with_kv_lock(&stores.kv, "host-working-set", |kv| {
            kv.relist_beam_with(ws, f0, order, &copies)
        });
        drop(grant);
        match done {
            Ok(true) => return Ok(()),
            // Evicted while the copies ran: they go back, and the finish
            // waits for its restore.
            Err(crate::store::kv::KvStoreError::Table(
                crate::store::kv::page_table::KvTableError::NonResident { .. },
            )) => {
                crate::store::registry::with_kv_lock(&stores.kv, "host-working-set", |kv| {
                    kv.release_device_reservation(copies.into_iter().map(|(_, copy)| copy).collect());
                });
                settle_and_wait_resident(ctx).await?;
            }
            // A move began while the copies ran: they go back, and the
            // finish waits for its end.
            Ok(false) => crate::store::registry::with_kv_lock(&stores.kv, "host-working-set", |kv| {
                kv.release_device_reservation(copies.into_iter().map(|(_, copy)| copy).collect());
            }),
            Err(error) => {
                crate::store::registry::with_kv_lock(&stores.kv, "host-working-set", |kv| {
                    kv.release_device_reservation(copies.into_iter().map(|(_, copy)| copy).collect());
                });
                return Err(error.to_string());
            }
        }
    }
}

/// Awaits `ask` (a windowed planner ask) while finalizing the process's
/// pending fires as each lands, `drained` told when any did (their window
/// slots are back): never waits on the tail whole before the ask, since the
/// tail may be this pass's earlier pieces, whose frame only the piece now
/// asking completes.
pub(super) async fn ask_while_tail_lands<C: FireContext, A: std::future::Future>(
    ctx: &mut C,
    ask: A,
    drained: impl Fn(),
) -> Result<A::Output, String> {
    tokio::pin!(ask);
    loop {
        let tails = ctx.pipeline_tails();
        let landed = tail_landed(&tails);
        tokio::select! {
            asked = &mut ask => return Ok(asked),
            () = landed => {
                // What landed is finalized: the window slots its fires read
                // come back, and the planner looks again.
                let mut any = false;
                for fires in &tails {
                    any |= drain_settled(ctx, Some(fires))
                        .await
                        .map_err(|error| format!("pipeline: settle for windowed kv: {error:#}"))?;
                }
                // A front signals exactly when it reads as settled: none
                // drained means another finalizer took it, and the next
                // front is waited on.
                if any {
                    drained();
                }
            }
        }
    }
}

/// Resolves once the front of any of `tails` has landed; never where all
/// are empty.
pub(super) async fn tail_landed(tails: &[PendingFires]) {
    let fronts: Vec<_> = tails
        .iter()
        .filter_map(|fires| {
            fires
                .lock()
                .unwrap()
                .front()
                .map(PendingOp::completion_signal)
        })
        .map(Box::pin)
        .collect();
    if fronts.is_empty() {
        return std::future::pending().await;
    }
    futures::future::select_all(fronts).await;
}

pub(super) async fn settle_and_wait_resident<C: FireContext>(ctx: &mut C) -> Result<(), String> {
    let Some(planner) = crate::planner::planner_of(ctx.process_id()) else {
        return Err("pipeline: KV working set is suspend-fenced".to_string());
    };
    ctx.settle_pipeline_tail()
        .await
        .map_err(|error| format!("pipeline: settle for eviction: {error:#}"))?;
    crate::inferlet::process::said(
        "plan: resident again",
        ctx.process_id(),
        planner.wait_resident(ctx.process_id()),
    )
    .await
    .map_err(|error| format!("pipeline: KV residency: {error}"))
}
