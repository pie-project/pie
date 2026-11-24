#![allow(unsafe_code)]

/// A prologue that takes or puts a channel commits before its fire's forward.
fn container_prologue_moves(container: &eta_ir::container::TraceContainer) -> bool {
    use eta_ir::op::ChannelUse;
    use eta_ir::registry::Stage;
    container
        .stages
        .iter()
        .filter(|s| s.stage == Stage::Prologue)
        .flat_map(|s| s.ops.iter())
        .any(|op| {
            matches!(
                op.channel_use(),
                Some((ChannelUse::Take | ChannelUse::Put, _))
            )
        })
}

fn container_has_lora_sink(container: &eta_ir::container::TraceContainer) -> bool {
    container
        .stages
        .iter()
        .flat_map(|s| s.ops.iter())
        .any(|op| {
            matches!(
                op,
                eta_ir::op::Op::SinkCall { name, .. }
                    if container.names.get(*name as usize).map(String::as_str) == Some("lora")
            )
        })
}

pub mod context;
pub mod float;
pub mod geometry;
pub mod host_geometry;
pub mod kv;
pub mod rs;
pub mod shadow;

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use wasmtime::component::Resource;

pub use context::FireContext;

use eta_ir::registry::{GeometryClass, Port, PortMask};

use crate::pipeline::Pipeline;
use crate::pipeline::channel::{BoundCells, Channel, ChannelError};
use crate::pipeline::instance::ForwardPass;
use crate::store::kv::working_set::{KvFireLease, KvWorkingSet};
use crate::store::rs::working_set::RsWorkingSet;
use eta_ir::container::HostRole;

pub struct PendingFireQueue {
    queue: Mutex<VecDeque<PendingOp>>,
    finalizer: Arc<tokio::sync::Mutex<()>>,
}

impl PendingFireQueue {
    pub fn new() -> Self {
        Self::from_queue(VecDeque::new())
    }

    pub(crate) fn from_queue(queue: VecDeque<PendingOp>) -> Self {
        Self {
            queue: Mutex::new(queue),
            finalizer: Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    pub fn lock(&self) -> std::sync::LockResult<std::sync::MutexGuard<'_, VecDeque<PendingOp>>> {
        self.queue.lock()
    }

    pub(crate) async fn finalize_guard(&self) -> tokio::sync::OwnedMutexGuard<()> {
        Arc::clone(&self.finalizer).lock_owned().await
    }

    /// Keep the transaction in the shared FIFO while waiting, so an aborted
    /// guest leaves teardown its owner. Caller holds the finalizer guard;
    /// errors are read by finalization too, rather than skipping cleanup.
    pub(crate) async fn await_front_completion(&self) {
        let completion = self
            .lock()
            .unwrap()
            .front()
            .map(PendingOp::completion_signal);
        if let Some(completion) = completion {
            completion.await;
        }
    }

    /// How many of `cell`'s unread cells can still be a pending fire's. A
    /// fire writes at most one cell of a channel (its one ring ticket) and
    /// the pending fires are the newest, so unread cells beyond this count
    /// are the oldest ones, and their fires are finalized.
    pub(crate) fn writers_of(
        &self,
        cell: &Arc<Mutex<crate::pipeline::channel::ChannelCell>>,
    ) -> usize {
        self.lock()
            .unwrap()
            .iter()
            .filter(|op| op.binds(cell))
            .count()
    }

    pub(crate) fn try_finalize_guard(&self) -> Option<tokio::sync::OwnedMutexGuard<()>> {
        Arc::clone(&self.finalizer).try_lock_owned().ok()
    }

    /// No fire is in it and none is being finalized. A fire leaves the
    /// queue under the finalizer gate, ahead of its fold and its failure
    /// being written, and its finalizer holds the gate until they are: an
    /// empty queue whose gate is held still has a fire to settle.
    pub(crate) fn drained(&self) -> bool {
        self.try_finalize_guard()
            .is_some_and(|_gate| self.lock().unwrap().is_empty())
    }
}

pub type PendingFires = Arc<PendingFireQueue>;
pub type PipelineFailure = Arc<Mutex<Option<String>>>;

fn pipeline_failed(failure: &PipelineFailure) -> Option<String> {
    failure
        .lock()
        .unwrap()
        .as_ref()
        .map(|reason| format!("pipeline: pipeline failed: {reason}"))
}

enum ReservedError {
    Stale,
    Fatal(String),
}

fn stale_demand_error() -> String {
    "pipeline: resource demand reads stale against a grant quoted for the same demand".to_string()
}

/// A grant found stale is quoted again only on the drift event: the demand
/// read now differs from the one it was quoted for. Stale against the very
/// demand it was quoted for, no new quote can cover it, and it is refused.
fn requote_on_drift(
    stale_at: &mut Option<crate::planner::Demand>,
    quoted: crate::planner::Demand,
) -> bool {
    if *stale_at == Some(quoted) {
        return false;
    }
    *stale_at = Some(quoted);
    true
}

/// A kv epoch a fire holds: settled by whoever finishes the fire, or, let go
/// unsettled, when dropped — at once while the fire was never submitted,
/// else once the engine has run it ([`Self::retire_on`]). A pending fire
/// dropped unfinalized (its process gone) would otherwise hold the epoch
/// open for good, and every id freed after it would wait on it.
struct KvTxnGuard {
    model: usize,
    engine: usize,
    txn: Option<kv::KvTxn>,
    submitted: Option<crate::engine::WorkItemCompletion>,
}

impl KvTxnGuard {
    fn new(model: usize, engine: usize, txn: Option<kv::KvTxn>) -> Self {
        Self {
            model,
            engine,
            txn,
            submitted: None,
        }
    }

    fn into_inner(mut self) -> Option<kv::KvTxn> {
        self.txn.take()
    }

    /// The fire is submitted: dropped unsettled, it settles once
    /// `completion` resolves, not sooner.
    fn retire_on(&mut self, completion: &crate::engine::WorkItemCompletion) {
        self.submitted = Some(completion.clone());
    }

    fn settle(model: usize, engine: usize, txn: kv::KvTxn) {
        let stores = crate::store::registry::get(model, engine);
        crate::store::registry::with_kv_lock(&stores.kv, "host-other", |store| {
            kv::settle(store, txn);
        });
        if let Some(planner) = crate::planner::planner_for(model, engine) {
            planner.pages_freed();
        }
    }
}

impl Drop for KvTxnGuard {
    fn drop(&mut self) {
        let Some(txn) = self.txn.take() else {
            return;
        };
        let (model, engine) = (self.model, self.engine);
        match self.submitted.take() {
            Some(completion) => completion.on_settled(move || Self::settle(model, engine, txn)),
            None => Self::settle(model, engine, txn),
        }
    }
}

/// A fire's own epoch in its engine's kv store (CONTRACT-BC 17): opened
/// once its slots are bound, closed when the engine retires it, or, never
/// submitted, when it is dropped. Every fire holds one, so the kv epoch
/// that fences freed pages, the shared pool's freed slots and an evacuated
/// segment's drain moves with every fire, not only with a copy-on-write.
/// It counts the fire in flight for its process, too, until it closes: the
/// count drops after the epoch's settling, so the planner never reads a
/// fire in flight that has settled.
struct FireEpoch {
    _txn: KvTxnGuard,
    _in_flight: Option<crate::planner::FireInFlight>,
}

impl FireEpoch {
    fn begin(
        model: usize,
        engine: usize,
        stores: &crate::store::registry::Stores,
        pid: crate::planner::ProcessId,
    ) -> Self {
        let txn = crate::store::registry::with_kv_lock(&stores.kv, "fire-epoch", kv::begin_fire);
        Self {
            _txn: KvTxnGuard::new(model, engine, Some(txn)),
            _in_flight: crate::planner::planner_for(model, engine)
                .map(|planner| planner.fire_in_flight(pid)),
        }
    }

    /// Closes the epoch once the engine has run the fire `completion`
    /// waits on: no sooner, since what it frees may still be read, and no
    /// later than that, since a pass may not settle its fires for long.
    /// By a hook, not a task awaiting it: nothing is spawned a fire, and the
    /// epoch closes where the completion resolves. (Awaiting it once lost
    /// the wakeup to the pass that finalizes the fire, when a completion
    /// woke one waiter, and held the epoch open until shutdown.)
    fn retire_on(self, completion: &crate::engine::WorkItemCompletion) {
        completion.on_settled(move || drop(self));
    }
}

struct RsTxnsGuard {
    model: usize,
    engine: usize,
    txn: Option<rs::RsTxn>,
}

impl RsTxnsGuard {
    fn new(model: usize, engine: usize, txn: Option<rs::RsTxn>) -> Self {
        Self { model, engine, txn }
    }

    fn into_inner(mut self) -> Option<rs::RsTxn> {
        self.txn.take()
    }
}

impl Drop for RsTxnsGuard {
    fn drop(&mut self) {
        let Some(txn) = self.txn.take() else {
            return;
        };
        let stores = crate::store::registry::get(self.model, self.engine);
        {
            let mut store = stores.rs.lock().unwrap();
            txn.abandon(&mut store);
            rs::settle(&mut store, Some(txn));
        }
        if let Some(planner) = crate::planner::planner_for(self.model, self.engine) {
            planner.pages_freed();
        }
    }
}

fn host_kv_demand_locked(
    store: &mut crate::store::kv::KvStore,
    ws: &KvWorkingSet,
    written: std::ops::Range<u64>,
) -> Result<usize, String> {
    let realization =
        kv::written_demand(store, ws.id, written.clone()).map_err(|error| error.to_string())?;
    let backing = store
        .backing_demand(ws.id, written.end)
        .map_err(|error| error.to_string())?;
    realization
        .checked_add(backing)
        .ok_or_else(|| "KV demand exceeds usize".to_string())
}

enum Written {
    Span(std::ops::Range<u64>),
    Pages(Vec<u64>),
}

fn kv_demand(
    stores: &crate::store::registry::Stores,
    ws: &KvWorkingSet,
    written: &Written,
) -> Result<usize, String> {
    crate::store::registry::with_kv_lock(&stores.kv, "host-other", |store| match written {
        Written::Span(span) => host_kv_demand_locked(store, ws, span.clone()),
        Written::Pages(pages) => store
            .write_demand(ws.id, pages)
            .map_err(|error| error.to_string()),
    })
    .map_err(|error| format!("pipeline: KV demand: {error}"))
}

/// The pages `ws` holds on the device at once to take `written` alone: what
/// it maps up to the fire's end (a prefix it shares counted), the pages it
/// backs past that, and the copies of shared pages it writes.
fn alone_kv_pages(
    stores: &crate::store::registry::Stores,
    ws: &KvWorkingSet,
    written: &Written,
) -> Result<usize, String> {
    crate::store::registry::with_kv_lock(&stores.kv, "host-other", |store| {
        let mapped = store.mapped_len(ws.id).map_err(|error| error.to_string())?;
        let (held, asked) = match written {
            Written::Span(span) => (
                mapped.min(span.end),
                host_kv_demand_locked(store, ws, span.clone())?,
            ),
            Written::Pages(pages) => (
                mapped,
                store
                    .write_demand(ws.id, pages)
                    .map_err(|error| error.to_string())?,
            ),
        };
        usize::try_from(held)
            .ok()
            .and_then(|held| held.checked_add(asked))
            .ok_or_else(|| "KV demand exceeds usize".to_string())
    })
    .map_err(|error| format!("pipeline: KV demand: {error}"))
}

/// What each unit of a fire takes on the device alone: one per KV working
/// set of `sets` ([`alone_kv_pages`]), with the state the lanes it carries
/// (`lanes`, beside it) ask of `plan`.
fn alone_demands(
    stores: &crate::store::registry::Stores,
    sets: &[(&KvWorkingSet, Written)],
    lanes: &[Vec<usize>],
    rs_ids: &[crate::store::rs::RsWorkingSetId],
    plan: &rs::RsPlan,
) -> Result<Vec<crate::planner::Demand>, String> {
    sets.iter()
        .enumerate()
        .map(|(at, (set, written))| {
            let kv_pages = u32::try_from(alone_kv_pages(stores, set, written)?)
                .map_err(|_| "pipeline: KV demand exceeds the planner ABI".to_string())?;
            let rs = match lanes.get(at) {
                Some(lanes) if !rs_ids.is_empty() => {
                    rs::demand_of_lanes(&stores.rs.lock().unwrap(), rs_ids, plan, lanes)?
                }
                _ => crate::store::rs::RsUnits::default(),
            };
            Ok(crate::planner::Demand { kv_pages, rs })
        })
        .collect()
}

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
async fn acquire_grant<C: FireContext>(
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
async fn ask_while_tail_lands<C: FireContext, A: std::future::Future>(
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
async fn tail_landed(tails: &[PendingFires]) {
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

async fn settle_and_wait_resident<C: FireContext>(ctx: &mut C) -> Result<(), String> {
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

fn bound_rs_working_sets<C: FireContext>(
    ctx: &mut C,
    ws: &KvWorkingSet,
    rs_reps: &[u32],
    qo_indptr: &[u32],
    scope: &crate::store::PipelineScope,
) -> Anyhow<Result<Vec<crate::store::rs::RsWorkingSetId>, String>> {
    let has_state = crate::model::model().rs_caps().has_state();
    if let Err(error) = rs::validate_count(rs_reps.len(), qo_indptr, has_state) {
        return Ok(Err(format!("pipeline: recurrent-state binding: {error}")));
    }
    let mut ids = Vec::with_capacity(rs_reps.len());
    for (row, &rep) in rs_reps.iter().enumerate() {
        let resource: Resource<RsWorkingSet> = Resource::new_borrow(rep);
        let rs = ctx.resources().get(&resource)?.clone();
        if rs.model != ws.model || rs.engine != ws.engine {
            return Ok(Err(format!(
                "pipeline: lane {row}'s recurrent state belongs to model/engine \
                 ({}, {}), expected ({}, {})",
                rs.model, rs.engine, ws.model, ws.engine
            )));
        }
        if let Err(refusal) = rs.claim_pipeline_scope(scope) {
            return Ok(Err(format!("pipeline: lane {row}'s working set {refusal}")));
        }
        ids.push(rs.id);
    }
    Ok(Ok(ids))
}

/// The window plan of a host fire on a model whose state is window lists:
/// each lane lands its rows from `held`, which every earlier token of its
/// sequence precedes, and reads its tables through them.
/// `keep`: the position a checkpoint inside the fire lands at, whose slots
/// stay past it.
///
/// Window-list path: what an engine that does not state
/// `Capabilities::windowed_state` is served by.
fn host_window_plan(
    req: &crate::engine::FireRequest,
    fold: &geometry::Fold,
    ids: &[crate::store::rs::RsWorkingSetId],
    keep: Option<u32>,
) -> Result<rs::RsPlan, String> {
    if ids.is_empty() {
        return Ok(rs::RsPlan::Groups(Vec::new()));
    }
    Ok(rs::RsPlan::Window(
        req.lanes
            .iter()
            .enumerate()
            .map(|(row, lane)| {
                let rows = u32::try_from(lane.tokens.len()).unwrap_or(u32::MAX);
                let end = lane.kv.held.saturating_add(rows);
                rs::WindowRow {
                    writes: lane.kv.held..end,
                    reads: end,
                    advance: crate::store::rs::WindowAdvance {
                        positioned: true,
                        settled: lane.kv.held,
                        reach: lane.kv.held,
                        fold: fold.of(row),
                        commit_end: end,
                        head: (rows > 0).then_some(end),
                        keep,
                    },
                }
            })
            .collect(),
    ))
}

/// A working set one lane writes from where its state ends, folding all it
/// writes, buffers nothing; that lane lands its state as of `snapshots` too.
fn host_rs_plan(
    stores: &crate::store::registry::Stores,
    ids: &[crate::store::rs::RsWorkingSetId],
    groups: &LaneGroups,
    req: &crate::engine::FireRequest,
    fold: &geometry::Fold,
    snapshots: &[u32],
    beam_ends: Option<&[u32]>,
    beam_commons: Option<&[u32]>,
) -> Result<rs::RsPlan, String> {
    if ids.is_empty() {
        return Ok(rs::RsPlan::Groups(Vec::new()));
    }
    let store = stores.rs.lock().unwrap();
    let span = |lane: usize| {
        let lane = &req.lanes[lane];
        let rows = u32::try_from(lane.tokens.len()).unwrap_or(u32::MAX);
        (lane.kv.held, lane.kv.held.saturating_add(rows))
    };
    let mut plans = Vec::with_capacity(groups.len());
    for (at, (_, lanes)) in groups.iter().enumerate() {
        if let Some(&empty) = lanes
            .iter()
            .find(|&&lane| req.lanes[lane].tokens.is_empty())
        {
            return Err(format!(
                "lane {empty} carries no rows; on a model with recurrent layers a lane folds \
                 what it writes, so fold with the pass that writes it"
            ));
        }
        let first = lanes[0];
        let frozen = store
            .frozen(ids[first])
            .map_err(|error| error.to_string())?;
        let (start, end) = span(first);
        let fold = group_fold(fold, first, frozen, end);
        let writes =
            if lanes.len() == 1 && start == frozen && fold == rs::GroupFold::Host(end - start) {
                None
            } else {
                let low = lanes
                    .iter()
                    .map(|&lane| span(lane).0)
                    .min()
                    .unwrap_or(start);
                let high = lanes.iter().map(|&lane| span(lane).1).max().unwrap_or(end);
                Some(low..high)
            };
        // A beam group's beams name any page of its pool, and their rows lie
        // slot by slot on its pages (§11): its buffer is backed from the
        // state's end to the pool's.
        let beam_end = beam_ends.and_then(|ends| ends.get(at).copied());
        // A beam group folds the rows every live beam shares into its set
        // once (§11.3): to its common slot, never past the rows it has.
        let fold = match (beam_end, beam_commons.and_then(|commons| commons.get(at).copied())) {
            (Some(_), Some(common)) if common > frozen => {
                rs::GroupFold::Host(common.min(end).saturating_sub(frozen))
            }
            _ => fold,
        };
        let writes = match beam_end {
            Some(end) => {
                let high = writes.as_ref().map_or(end, |writes| writes.end);
                Some(frozen..high.max(end))
            }
            None => writes,
        };
        let snapshots = match (&writes, groups.len()) {
            (None, 1) => snapshots.to_vec(),
            _ => Vec::new(),
        };
        plans.push(rs::GroupPlan {
            lanes: lanes.clone(),
            writes,
            fold,
            snapshots,
            beams: beam_end.is_some(),
        });
    }
    Ok(rs::RsPlan::Groups(plans))
}

/// Each working set buffers every slot from where its state ends up to what it
/// reserved, and folds what [`device_fold`] says of its first lane's rows,
/// `split` apart: a recurrent state's rows are never reported, so a fold the
/// engine alone could place is refused.
fn device_rs_plan(
    stores: &crate::store::registry::Stores,
    ids: &[crate::store::rs::RsWorkingSetId],
    groups: &LaneGroups,
    lanes: &PassLanes,
    split: &[u32],
    fold: &geometry::Fold,
    beams: bool,
) -> Result<rs::RsPlan, String> {
    if ids.is_empty() {
        return Ok(rs::RsPlan::Groups(Vec::new()));
    }
    // A beam group's beams name any page of their set's kv reservation (its
    // pool, grown a column at a time), and their rows lie slot by slot on its
    // pages (`BEAM-PAGE-LISTS.md` §11): the rs buffer is backed to the same
    // end. Read before the rs store's lock is taken.
    let mut beam_ends = std::collections::HashMap::new();
    if beams {
        let page = u64::from(stores.kv_page_size.max(1));
        for (_, group) in groups {
            let first = group[0];
            let pages = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
                kv.page_len(lanes.kv[first].id)
            })
            .map_err(|error| format!("pipeline: a beam group's kv reservation: {error}"))?;
            beam_ends.insert(first, u32::try_from(pages * page).unwrap_or(u32::MAX));
        }
    }
    let store = stores.rs.lock().unwrap();
    let mut plans = Vec::with_capacity(groups.len());
    for (_, group) in groups {
        let first = group[0];
        let frozen = store
            .frozen(ids[first])
            .map_err(|error| error.to_string())?;
        let mut reserved = lanes.cursors[first].lock().unwrap().reserved;
        if let Some(&end) = beam_ends.get(&first) {
            reserved = reserved.max(end);
        }
        let rows = split[first + 1] - split[first];
        let fold = match device_fold(
            fold.of(first),
            rows,
            reserved.saturating_sub(frozen),
            appends_folded(lanes, group),
        ) {
            DeviceFold::None => rs::GroupFold::None,
            DeviceFold::Device => rs::GroupFold::Device,
            DeviceFold::Count(count) => rs::GroupFold::Host(count),
            DeviceFold::PastRows(_) => return Err(past_rows_refusal(first, rows)),
        };
        // The device picks where a lane's rows start: what it carries in is
        // bounded by what stands unfolded in its set ([`device_lane_rows`]).
        plans.push(rs::GroupPlan {
            lanes: group.clone(),
            writes: Some(frozen..reserved),
            fold,
            snapshots: Vec::new(),
            beams,
        });
    }
    Ok(rs::RsPlan::Groups(plans))
}

/// A lane's rows in a fire, or a fire's (its lanes' summed): its own, and
/// the recurrent rows it carries in unfolded, which earlier fires wrote and
/// nothing has folded, and which it replays ahead of its own
/// ([`::engine::fire::rung_holds`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct FireRows {
    own: u32,
    carried: u32,
}

impl FireRows {
    /// The rows a rung must hold for these: its own and its carried rows
    /// each.
    #[cfg(test)]
    fn ask(self) -> u32 {
        self.own.max(self.carried)
    }

    fn of(lanes: impl IntoIterator<Item = FireRows>) -> FireRows {
        lanes
            .into_iter()
            .fold(FireRows::default(), |sum, lane| FireRows {
                own: sum.own.saturating_add(lane.own),
                carried: sum.carried.saturating_add(lane.carried),
            })
    }
}

/// The own and carried rows of a lane the host places at `start..end`, its
/// recurrent state folded through `frozen` (`None`: it has none, and
/// carries nothing): it carries the rows between its state and its first
/// row. A lane of one row asks the rung for none of them: the engines'
/// decode step replays them as data, inside its own rows' body.
fn lane_rows(start: u32, end: u32, frozen: Option<u32>) -> FireRows {
    let own = end.saturating_sub(start);
    FireRows {
        own,
        carried: ::engine::fire::lane_carried(
            own,
            frozen.map_or(0, |frozen| start.saturating_sub(frozen)),
        ),
    }
}

/// The `own` rows of a lane the device places, and a bound on what it
/// carries, its recurrent state folded through `frozen` (`None`: it has
/// none, and carries nothing). Its first row is its program's, resolved on
/// the device after its set's earlier fires have run, so no count exists
/// here; it replays no more than the rows standing unfolded in its set
/// then: those written past `frozen`, up to the set's written end, by
/// fires that settled, and every row of its fires staged and still to
/// settle ([`::engine::fire::rung_holds`], device geometry).
fn device_lane_rows(
    own: u32,
    cursor: &crate::inferlet::host::working_set::Cursor,
    frozen: Option<u32>,
) -> FireRows {
    FireRows {
        own,
        carried: ::engine::fire::lane_carried(
            own,
            frozen.map_or(0, |frozen| cursor.unfolded(frozen)),
        ),
    }
}

/// The frozen length of each lane's recurrent state, by the lane's group
/// (`None` a lane where the pass binds none, or on a model with window
/// lists, whose rows trim behind their windows and are never replayed, or
/// whose windowed layers are ring + buffer, `replays_none`: their buffered
/// rows' K/V are read from the window buffer, never replayed).
fn lanes_frozen(
    stores: &crate::store::registry::Stores,
    ids: &[crate::store::rs::RsWorkingSetId],
    groups: &LaneGroups,
    lanes: usize,
    replays_none: bool,
) -> Result<Vec<Option<u32>>, String> {
    let mut frozen = vec![None; lanes];
    if ids.is_empty() || replays_none {
        return Ok(frozen);
    }
    let store = stores.rs.lock().unwrap();
    for (_, group) in groups {
        let at = store
            .frozen(ids[group[0]])
            .map_err(|error| format!("pipeline: recurrent-state mode: {error}"))?;
        for &lane in group {
            frozen[lane] = Some(at);
        }
    }
    Ok(frozen)
}

/// Whether the load's only state is windowed layers served as ring +
/// buffer (`window_state`, its record the rings alone: no recurrent layer,
/// whatever buffer lists the engine states): its lanes, a beam group's
/// sharing one record among them, replay nothing, their unfolded rows going
/// to window-buffer slots on demand, so a pass of several lanes is cut as
/// one of kv lanes is (CONTRACT-BC 19: a fire too wide is cut like chunked
/// prefill, never refused). A hybrid's recurrent layers replay the rows its
/// lanes carry: its lanes are cut as an rs load's.
pub(crate) fn window_only(stores: &crate::store::registry::Stores, window_state: bool) -> bool {
    window_state
        && stores
            .shared
            .as_ref()
            .is_some_and(|shared| !shared.lock().unwrap().recurrent())
}

/// The rows of one piece of a host pass: each lane's cut of `full`, over a
/// state folded through `frozen` when the pass began. A pass that folds
/// all it writes folds piece by piece (a fold lands as its fire is
/// prepared), so its pieces past the first carry nothing (`folded_behind`);
/// one that folds nothing carries every row its earlier pieces wrote.
fn piece_rows(
    full: &crate::engine::FireRequest,
    piece: &[Option<(u32, u32)>],
    frozen: &[Option<u32>],
    folded_behind: bool,
) -> FireRows {
    FireRows::of(
        full.lanes
            .iter()
            .zip(piece)
            .enumerate()
            .filter_map(|(lane, (rows, cut))| {
                let (from, to) = (*cut)?;
                let start = rows.kv.held.saturating_add(from);
                let end = rows.kv.held.saturating_add(to);
                let frozen = frozen
                    .get(lane)
                    .copied()
                    .flatten()
                    .map(|frozen| if folded_behind { start } else { frozen });
                Some(lane_rows(start, end, frozen))
            }),
    )
}

/// The rung a fire of `rows` asks, `replayed` every lane's carried rows (a
/// decode lane's included): the rows of the smallest activation rung whose
/// stated replay scratch holds it ([`::engine::fire::rung_holds`] against
/// each rung's [`::engine::RungReplay`], its replay scratch rows counted as
/// the load lays them, [`::engine::ReplayRows::of_sums`]), or its own and
/// carried rows where the first rung that wide holds it already. Where the
/// pool states no rung, one fire's tokens at the contract's floor. Refused
/// by name, naming the bound that binds, where no rung holds it: before any
/// step. Own rows past every rung are not this check's (a fire's capacity,
/// refused where the pass is cut).
fn replay_rung(
    stores: &crate::store::registry::Stores,
    engine: usize,
    rows: FireRows,
    replayed: u32,
    lanes: u32,
    compact: bool,
) -> Result<u32, String> {
    let holds = crate::engine::get_spec(engine).map_or(u32::MAX, |spec| {
        u32::try_from(spec.limits.max_forward_tokens).unwrap_or(u32::MAX)
    });
    let (stated, widest) = stores.shared.as_ref().map_or_else(
        || (std::sync::Arc::new(Vec::new()), u32::MAX),
        |shared| shared.lock().unwrap().rung_replays(),
    );
    let reach = holds.min(widest);
    let end = stated
        .iter()
        .position(|&(tokens, _)| tokens > reach)
        .unwrap_or(stated.len());
    let pinned;
    let rungs: &[(u32, ::engine::RungReplay)] = if end == 0 {
        // No rung stated (pinned pools) or none a fire may take: one fire's
        // tokens, its own and carried rows each, the engine's scratch its own.
        pinned = [(holds, crate::store::shared::rows_only(holds))];
        &pinned
    } else {
        &stated[..end]
    };
    let count = ::engine::ReplayRows::of_sums(rows.own, rows.carried, replayed, lanes, compact);
    let floor = rows.own.max(rows.carried);
    let mut first = true;
    let mut past = None;
    for (tokens, bound) in rungs {
        let tokens = *tokens;
        if tokens < floor {
            continue;
        }
        match ::engine::fire::rung_holds(bound, count) {
            Ok(()) => return Ok(if first { floor } else { tokens }),
            Err(why) => past = Some((tokens, why)),
        }
        first = false;
    }
    let (tokens, why) = match past {
        Some(past) => past,
        None => {
            let (tokens, bound) = rungs.last().expect("one rung at least");
            let tokens = *tokens;
            match ::engine::fire::rung_holds(bound, count) {
                Ok(()) => return Ok(tokens),
                Err(why) => (tokens, why),
            }
        }
    };
    if why.bound == ::engine::ReplayBound::Own {
        return Ok(tokens);
    }
    Err(format!(
        "pipeline: no activation rung holds this fire's replay: at the largest, of {tokens} \
         tokens, it has {why}; fold what its lanes have written before they write more"
    ))
}

/// The load's stated rungs a fire carrying recurrent state hands the
/// scheduler, which holds a step of several such fires to the rung its
/// width picks ([`crate::engine::FireRequest::replay_rungs`]); `None` for a
/// fire carrying none.
fn fire_rungs(
    stores: &crate::store::registry::Stores,
    carries: bool,
) -> Option<crate::store::shared::Replays> {
    let shared = stores.shared.as_ref().filter(|_| carries)?;
    Some(shared.lock().unwrap().rung_replays().0)
}

/// The widest rung a host pass's pieces ask, each piece as it is cut
/// ([`replay_rung`]: its own rows and the rows it carries in, against every
/// rung's stated bounds); refused by name where some piece no rung holds.
/// The uncut pass's own rows are no piece's.
fn pieces_rung(
    stores: &crate::store::registry::Stores,
    engine: usize,
    full: &crate::engine::FireRequest,
    cuts: &[Vec<Option<(u32, u32)>>],
    frozen: &[Option<u32>],
    folds_each: bool,
    scratch: &PassScratch<'_>,
) -> Result<u32, String> {
    let mut widest = 0u32;
    for (at, piece) in cuts.iter().enumerate() {
        let behind = folds_each && at > 0;
        let rows = replay_rung(
            stores,
            engine,
            piece_rows(full, piece, frozen, behind),
            piece_replayed(full, piece, frozen, behind),
            piece_lanes(piece),
            scratch.compact(stores, full, piece, frozen, behind),
        )?;
        widest = widest.max(rows);
    }
    Ok(widest)
}

/// The lanes with rows in `piece`.
fn piece_lanes(piece: &[Option<(u32, u32)>]) -> u32 {
    u32::try_from(piece.iter().filter(|cut| cut.is_some()).count()).unwrap_or(u32::MAX)
}

/// What a host pass's pieces' rs scratch is laid by
/// ([`::engine::fire::scratch_compact`]), from what the runtime puts in
/// each piece's fire: which lanes share their set with another (their
/// buffered reads are their masks' lineage, `Lineage::Admitted`, else
/// every buffered slot, `Lineage::Set`, as `RsVerb::Buffer` states them),
/// and the rows a piece lands its state at (`checkpoints`, as many of them
/// inside a fire as the engine lands, `snapshots`; a piece of several
/// lanes lands none), each an rs snapshot verb.
pub(crate) struct PassScratch<'a> {
    pub(crate) shared: Vec<bool>,
    pub(crate) checkpoints: &'a [u64],
    pub(crate) snapshots: usize,
}

impl PassScratch<'_> {
    /// Whether `piece`'s fire's rs scratch is compact.
    fn compact(
        &self,
        stores: &crate::store::registry::Stores,
        full: &crate::engine::FireRequest,
        piece: &[Option<(u32, u32)>],
        frozen: &[Option<u32>],
        folded_behind: bool,
    ) -> bool {
        let mut kept: Vec<(u64, u64)> = Vec::new();
        let mut one_run = true;
        for (lane, (rows, cut)) in full.lanes.iter().zip(piece).enumerate() {
            let Some((from, to)) = *cut else {
                continue;
            };
            let start = rows.kv.held.saturating_add(from);
            let end = rows.kv.held.saturating_add(to);
            kept.push((u64::from(start), u64::from(end)));
            let Some(state) = frozen.get(lane).copied().flatten() else {
                continue;
            };
            let folded = if folded_behind { start } else { state };
            let shared = self.shared.get(lane).copied().unwrap_or(false);
            let lineage = if shared {
                ::engine::fire::Lineage::Admitted
            } else {
                ::engine::fire::Lineage::Set
            };
            // A lane sharing its set whose mask the host does not state (the
            // device resolves it) reads slots the host cannot name: not one
            // run as far as it can say.
            one_run &= !(shared && rows.mask.is_none())
                && ::engine::fire::lane_one_run(rows.mask.as_ref(), folded, start, lineage);
        }
        let snapshots = match kept.as_slice() {
            [(start, end)] => {
                self.snapshots > 0
                    && self.checkpoints.iter().any(|&row| *start < row && row < *end)
            }
            _ => false,
        };
        fire_compact(stores, one_run, snapshots)
    }
}

/// Whether a fire whose lanes' buffered rows are each one stretch (or not)
/// and that lands rs snapshots (or not) has a compact rs scratch on these
/// stores' load ([`::engine::fire::scratch_compact`]).
fn fire_compact(
    stores: &crate::store::registry::Stores,
    every_lane_one_run: bool,
    any_lane_snapshots: bool,
) -> bool {
    stores.shared.as_ref().is_some_and(|shared| {
        let pool = shared.lock().unwrap();
        ::engine::fire::scratch_compact(
            pool.rs_compact(),
            pool.rs_scratch_per_fire(),
            every_lane_one_run,
            any_lane_snapshots,
        )
    })
}

/// The buffered recurrent rows one piece of a host pass replays, summed
/// over every lane, a decode lane's included (what
/// [`::engine::FireLimits::replayed_rows`] bounds a step by): each lane's
/// rows between its state and its first, as [`piece_rows`] counts them
/// before [`::engine::fire::lane_carried`] drops a decode lane's.
fn piece_replayed(
    full: &crate::engine::FireRequest,
    piece: &[Option<(u32, u32)>],
    frozen: &[Option<u32>],
    folded_behind: bool,
) -> u32 {
    full.lanes
        .iter()
        .zip(piece)
        .enumerate()
        .filter_map(|(lane, (rows, cut))| {
            let (from, _) = (*cut)?;
            let start = rows.kv.held.saturating_add(from);
            let frozen = frozen.get(lane).copied().flatten()?;
            Some(if folded_behind { 0 } else { start.saturating_sub(frozen) })
        })
        .fold(0u32, u32::saturating_add)
}


fn group_fold(fold: &geometry::Fold, first: usize, frozen: u32, end: u32) -> rs::GroupFold {
    match fold.of(first) {
        None => rs::GroupFold::Device,
        Some(count) => match count.min(end.saturating_sub(frozen)) {
            0 => rs::GroupFold::None,
            count => rs::GroupFold::Host(count),
        },
    }
}

/// Whether the pass's program adapts the KV it writes beyond its tokens.
fn lora_program_of<C: FireContext>(ctx: &mut C, fwd: &Resource<ForwardPass>) -> Anyhow<bool> {
    let p = ctx.resources().get(fwd)?;
    Ok(container_has_lora_sink(&p.instance.program.bound.container))
}

/// The rows a piece of a lone masked lane takes so its mask plane fits the
/// room of the rung it runs at: a rung of `t` tokens holds `max(t * t,
/// max_context)` mask bits ([`::engine::Lane::mask`]), and a piece of `r`
/// rows over a buffer of `b` stages `r * (b + r)`. Bounded by the buffer the
/// pass's last piece finds (`b0 + rows - r`), every piece fits; nothing is
/// refused for a long buffer, the pass takes more fires.
fn mask_share(
    req: &crate::engine::FireRequest,
    share: u32,
    stores: &crate::store::registry::Stores,
    engine: usize,
) -> u32 {
    let [lane] = req.lanes.as_slice() else {
        return share;
    };
    if lane.mask.is_none() || lane.kv_less {
        return share;
    }
    let context = crate::engine::get_spec(engine).map_or(0, |spec| spec.limits.max_context) as u64;
    let rows = lane.tokens.len() as u64;
    let span = u64::from(lane.kv.held.saturating_sub(lane.kv.folded)) + rows;
    let rung = |r: u32| -> u64 {
        stores
            .shared
            .as_ref()
            .map_or(r, |shared| shared.lock().unwrap().rung_for(r).max(r))
            .into()
    };
    mask_rows(share, span, context, rung)
}

/// The largest `r <= share` with `r * span <= max(rung(r)^2, context)`, at
/// least one row: [`mask_share`] apart from the stores.
fn mask_rows(share: u32, span: u64, context: u64, rung: impl Fn(u32) -> u64) -> u32 {
    let room = |r: u32| {
        let t = rung(r);
        t.saturating_mul(t).max(context)
    };
    let mut r = share.max(1);
    while r > 1 && u64::from(r).saturating_mul(span) > room(r) {
        let fits = u32::try_from(room(r) / span.max(1)).unwrap_or(u32::MAX);
        r = fits.min(r - 1).max(1);
    }
    r
}

/// The prompt rows one fire on `engine` carries: the prefill budget shared
/// among the live lanes.
fn prefill_share(page: u32, engine: usize) -> u32 {
    crate::engine::get_spec(engine).map_or(u32::MAX, |spec| {
        crate::inferlet::host::model::prefill_share(
            // Within the pool's activation rung (SPEC decision 6).
            crate::inferlet::host::model::prefill_budget(spec.limits)
                .min(crate::store::shared::fire_cap(engine) as usize)
                .max(1),
            crate::inferlet::process::live_count(),
            page as usize,
        )
    })
}

/// The rows a fire on `engine` takes to carry `rows` of a prompt whole
/// beside the decode row each other live lane spends in it
/// ([`crate::inferlet::host::model::prefill_share`]), up to what the
/// largest activation rung holds: what a pass asks the rung for
/// ([`crate::store::shared::want_rows`]).
fn prefill_want(rows: usize, engine: usize) -> u32 {
    let live = crate::inferlet::process::live_count();
    let wide = rows.saturating_add(live.saturating_sub(1));
    let wide = u32::try_from(wide).unwrap_or(u32::MAX);
    crate::engine::get_spec(engine).map_or(wide, |spec| {
        let budget = crate::inferlet::host::model::prefill_budget(spec.limits).max(1);
        wide.min(u32::try_from(budget).unwrap_or(u32::MAX))
    })
}

/// The rows past which a pass's ask is its process's between passes
/// ([`crate::store::shared::hold_for_process`]): a decode's few rows ask
/// only while they are queued.
const RUNG_WANT_FROM: usize = 16;

/// How many of a pass's pieces run ahead of the one being cut while its
/// rung's raise is pending: the device stays fed, and the rest is cut wider
/// once the raise lands.
const PIECES_AHEAD: usize = 2;

/// A pass's ask for the activation rung, raised from free room: where the
/// raise lacks chunks that kv holds (the span a lowered rung lent among
/// them), it asks no planner, since a rise funded as a kv demand would wait
/// among the waiters it starves, or ahead of them; it says it waits for
/// room ([`RaiseFunding::waits_for_room`]), the pass fires at the rung held,
/// and the rise is taken once the pool gives room back (its room signal,
/// [`crate::store::slots::SlotPool::room_watch`]). Runs beside the pass
/// until the raise lands or the pass is through (dropped). A piece that
/// cannot fire narrower asks for its rise itself ([`RungAsk::wait`]).
struct RaiseFunding {
    stop: Arc<tokio::sync::Notify>,
    stopped: Arc<std::sync::atomic::AtomicBool>,
    /// The rise is short of free room: the pass fires at the rung held.
    short: Arc<std::sync::atomic::AtomicBool>,
}

impl RaiseFunding {
    fn start(shared: crate::store::shared::Shared, engine: usize, rows: u32) -> Self {
        let stop = Arc::new(tokio::sync::Notify::new());
        let stopped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let short = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let funding = Self {
            stop: stop.clone(),
            stopped: stopped.clone(),
            short: short.clone(),
        };
        crate::rt::spawn(async move {
            let target = crate::store::shared::cap_for(&shared, rows);
            let says = |waits: bool| {
                if short.swap(waits, std::sync::atomic::Ordering::AcqRel) != waits {
                    // The pass reads it at its next cut.
                    crate::store::shared::rung_changed(engine);
                }
            };
            // A step with nothing to do waits on the rung's next change (a
            // raise, a copy, the cap), on room given back where the rise is
            // short of it, or on the stop; each armed before it looks.
            while !stopped.load(std::sync::atomic::Ordering::Acquire) {
                let watch = crate::store::shared::rung_watch(engine);
                tokio::pin!(watch);
                watch.as_mut().enable();
                let room = shared.lock().unwrap().pool().room_watch();
                crate::store::shared::raise_now(&shared, engine, rows);
                if crate::store::shared::fire_cap(engine) >= target {
                    says(false);
                    return;
                }
                // Quoted at the rung the raise is taken at: the one the
                // engine holds where that is wider
                // (`SharedPool::raise_at`).
                let quote = {
                    let pool = shared.lock().unwrap();
                    let goal = pool.rung_for(rows);
                    pool.raise_at(goal).map(|at| {
                        let (chunks, _) = pool.raise_short(at);
                        let spc = pool.list().map_or(1, |list| list.slots_per_chunk.max(1));
                        (chunks, pool.pool().held_in_span(at).saturating_mul(spc))
                    })
                };
                let Some((chunks, held)) = quote else {
                    // The engine holds a rung the guest terms no longer
                    // fit: the walk brings it down first, and tells the
                    // rung.
                    crate::store::shared::ask_walk(engine);
                    tokio::select! {
                        () = &mut watch => {}
                        () = stop.notified() => return,
                    }
                    continue;
                };
                if chunks == 0 {
                    // The room is there and state lies in the span: the
                    // walker moves it out, the engine copying it, and
                    // tells the rung.
                    says(false);
                    tokio::select! {
                        () = &mut watch => {}
                        () = stop.notified() => return,
                    }
                    continue;
                }
                // Short of free room: the chunks the rise lacks are kv's.
                // The pass fires at the rung held, and the rise is asked
                // again only once the pool gives room back.
                if !short.load(std::sync::atomic::Ordering::Acquire) {
                    tracing::debug!(
                        engine,
                        rows,
                        chunks,
                        held,
                        "a rise waits for room: {held} pages held by kv"
                    );
                }
                says(true);
                crate::store::shared::ask_walk(engine);
                tokio::select! {
                    () = room.given() => {}
                    () = &mut watch => {}
                    () = stop.notified() => return,
                }
            }
        });
        funding
    }

    /// Whether the rise is short of free room: the pass fires at the rung
    /// held until room is given back.
    fn waits_for_room(&self) -> bool {
        self.short.load(std::sync::atomic::Ordering::Acquire)
    }
}

/// A pass's ask of the activation rung, the one rule for every fire the
/// host stages (a prompt's cut pieces, a decode's few rows, a
/// device-carried decode staged whole): a pass asks the rung for the rows
/// its fire takes before its first fire is staged and holds the ask until
/// its last has settled, whatever the cap is when it is cut; only a lone
/// row the cap covers asks nothing, no cap being narrower than it.
///
/// The scheduler keeps a fire past the cap queued until the rung is back,
/// and the walker, bringing the rung down, cuts fires to what is asked, the
/// live lanes' one row each otherwise: a fire wider than a row staged under
/// a cap that covered it, asking nothing, sat queued for good once the
/// walker took the cap down under it.
struct RungAsk {
    engine: usize,
    pid: uuid::Uuid,
    quorum: uuid::Uuid,
    /// The pool whose rung it asks, and the stores' kv, whose fire epoch a
    /// rise's copy is carried at.
    shared: Option<crate::store::shared::Shared>,
    kv: Arc<parking_lot::Mutex<crate::store::kv::KvStore>>,
    /// Held (the walker cuts nothing under it) until dropped.
    want: crate::store::shared::RungWant,
    /// The raise the rung lacks for it, taken beside the pass from free
    /// room; `None` where fires may take the rows now.
    funding: Option<RaiseFunding>,
}

/// Asks the rung on `engine` for the `rows` a fire of a pass of `own` rows
/// takes ([`RungAsk`]; `None`: a lone row the cap covers): the rung is
/// raised at once where nothing moves, a cap the walker lowered behind a
/// rung still up is lifted back, and what the raise lacks is waited for
/// beside the pass ([`RaiseFunding`]).
fn ask_rung(
    stores: &crate::store::registry::Stores,
    engine: usize,
    pid: uuid::Uuid,
    quorum: uuid::Uuid,
    own: usize,
    rows: u32,
) -> Option<RungAsk> {
    if own <= 1 && rows <= crate::store::shared::fire_cap(engine) {
        return None;
    }
    // Held before the cap is read: the walker takes no cap down past an
    // ask, and `raise_now` lifts one it took before, both in the pool's
    // lock.
    let want = crate::store::shared::want_rows(engine, rows);
    let funding = stores.shared.as_ref().and_then(|shared| {
        crate::store::shared::raise_now(shared, engine, rows);
        (crate::store::shared::fire_cap(engine) < crate::store::shared::cap_for(shared, rows))
            .then(|| RaiseFunding::start(shared.clone(), engine, rows))
    });
    Some(RungAsk {
        engine,
        pid,
        quorum,
        shared: stores.shared.clone(),
        kv: stores.kv.clone(),
        want,
        funding,
    })
}

impl RungAsk {
    /// Returns once fires may take `width` rows, or with what the planner
    /// refused the rise. A piece of the held width returns at once; one
    /// wider cannot fire narrower (no cut brings it under the rung held),
    /// so its rise is asked as a keyed demand of its process, as a
    /// program's fire term is ([`crate::store::shared::programs::fund_span`]),
    /// served in key order; never on a rung that does not come.
    async fn wait(&self, width: u32) -> Result<(), String> {
        let engine = self.engine;
        let (Some(shared), Some(_)) = (self.shared.as_ref(), self.funding.as_ref()) else {
            // Nothing funds a raise (no shared pool): the cap never moves.
            return Ok(());
        };
        if crate::store::shared::fire_cap(engine) >= width {
            return Ok(());
        }
        // Until the raise, the process submits nothing: no frame waits on
        // its lanes (the frame's firing frees what the raise is funded
        // with; the frame is not what funds it). Its owner is told blocked
        // from its first look short of the rung (`parks_here`) until the
        // rung is there, and unblocked as that park ends: no owner is left
        // blocked with no unblock to come.
        let funder = crate::store::shared::programs::Planner {
            pid: self.pid,
            quorum: self.quorum,
        };
        let kv = self.kv.clone();
        let epoch_now =
            move || crate::store::registry::with_kv_lock(&kv, "rung-rise", |kv| kv.current_epoch());
        let mover = crate::store::shared::programs::LaneMover {
            engine,
            epoch_now: &epoch_now,
        };
        crate::inferlet::process::said("enqueue: the activation rung", self.pid, async {
            // Parked once it waits, it stays parked until the rung is
            // there: between two waits (the funding's, the watch's) its
            // owner was unblocked, its lanes expected again, and a frame
            // holding three complete fronts waited on a lane whose process
            // waited on a raise that frame held up (the rung lowered under a
            // wider ask).
            let mut parked = None;
            loop {
                let watch = crate::store::shared::rung_watch(engine);
                tokio::pin!(watch);
                watch.as_mut().enable();
                if crate::store::shared::fire_cap(engine) >= width {
                    return Ok(());
                }
                if parked.is_none() {
                    parked = crate::inferlet::process::parks_here();
                }
                crate::store::shared::programs::fund_span(shared, || width, &funder, &mover)
                    .await?;
                crate::store::shared::raise_now(shared, engine, width);
                if crate::store::shared::fire_cap(engine) >= width {
                    return Ok(());
                }
                // Waiting on a raise of the rung: its owner is parked here.
                watch.await;
            }
        })
        .await
    }

    /// Whether the rise is short of free room, so the pass fires at the rung
    /// held: read before each piece is cut and submitted, never learnt by
    /// a wide fire refused.
    fn fires_at_held_rung(&self) -> bool {
        self.funding
            .as_ref()
            .is_some_and(RaiseFunding::waits_for_room)
    }

    /// The ask stands until `last`, the pass's last fire, settles; its
    /// funding ends here, every fire staged.
    fn until_settled(self, last: &crate::engine::completion::WorkItemCompletion) {
        let want = self.want;
        last.on_settled(move || drop(want));
    }
}

impl Drop for RaiseFunding {
    fn drop(&mut self) {
        self.stopped
            .store(true, std::sync::atomic::Ordering::Release);
        self.stop.notify_one();
    }
}

/// Each piece but the last ends on a page boundary; the last holds every row
/// the program reads out. The pages before the first media row are a piece
/// of their own, which seals.
/// `tail`: the last rows of the lane an output it reads is computed over.
fn host_cuts(
    req: &crate::engine::FireRequest,
    cuttable: bool,
    share: u32,
    page: u32,
    forced: &[u32],
    tail: u32,
) -> Vec<(u32, u32)> {
    host_cuts_from(req, cuttable, share, page, forced, tail, 0)
}

/// [`host_cuts`] of the rows from `start` on: the rest of a pass cut again
/// once its rung was raised.
fn host_cuts_from(
    req: &crate::engine::FireRequest,
    cuttable: bool,
    share: u32,
    page: u32,
    forced: &[u32],
    tail: u32,
    start: u32,
) -> Vec<(u32, u32)> {
    let rows =
        |lane: &crate::engine::FireLane| u32::try_from(lane.tokens.len()).unwrap_or(u32::MAX);
    let [lane] = req.lanes.as_slice() else {
        return vec![(0, req.lanes.first().map_or(0, rows))];
    };
    let rows = rows(lane);
    if !cuttable || !separable_rows(req) {
        return vec![(0, rows)];
    }
    let (held, page) = (lane.kv.held, page.max(1));
    let media = first_media_row(req);
    let tail = tail_row(lane, rows, tail).min(media.unwrap_or(rows));
    let mut ends: Vec<u32> = forced
        .iter()
        .copied()
        .chain(media.map(|row| ((held + row) / page * page).saturating_sub(held)))
        .filter(|&end| start < end && end < rows)
        .collect();
    if share > 0 {
        let mut from = start;
        while rows - from > share {
            let mut to = ((held + from + share) / page * page).saturating_sub(held);
            if to <= from {
                to = from + share;
            }
            ends.push(to);
            from = to;
        }
    }
    ends.retain(|&end| end <= tail);
    ends.push(rows);
    ends.sort_unstable();
    ends.dedup();
    let mut from = start;
    ends.into_iter()
        .map(|to| {
            let cut = (from, to);
            from = to;
            cut
        })
        .collect()
}

fn plain_rows(req: &crate::engine::FireRequest) -> bool {
    separable_rows(req)
        && req.lanes[0].mask.is_none()
        && matches!(req.lanes[0].readout, ::engine::Readout::Last)
}

/// Cut pieces end before it: the spans move the positions of the rows after
/// them (M-RoPE).
fn first_media_row(req: &crate::engine::FireRequest) -> Option<u32> {
    req.media
        .iter()
        .filter(|media| media.lane == 0)
        .flat_map(|media| media.routes.iter())
        .filter_map(|&row| u32::try_from(row).ok())
        .min()
}

/// A mask is a row's own, so a masked lane qualifies.
fn separable_rows(req: &crate::engine::FireRequest) -> bool {
    matches!(req.lanes.as_slice(), [lane] if plain_lane(lane)) && req.voxels.is_empty()
}

fn plain_lane(lane: &::engine::Lane) -> bool {
    lane.adapter.is_none() && lane.inputs.is_empty() && !lane.kv_less
}

/// The first row the last piece must hold: what the readout names, and the
/// last `tail` rows an output the program reads is computed over.
fn tail_row(lane: &::engine::Lane, rows: u32, tail: u32) -> u32 {
    let read = match &lane.readout {
        ::engine::Readout::Last => rows.saturating_sub(1),
        ::engine::Readout::Rows(read) => read.iter().copied().min().unwrap_or(rows),
        ::engine::Readout::None => rows,
    };
    if tail > 0 {
        read.min(rows.saturating_sub(tail))
    } else {
        read
    }
}

/// Every fire but the last runs the model's forward alone: no program stages,
/// channel values or readout.
pub(crate) fn rows_of_lanes(
    full: &crate::engine::FireRequest,
    piece: &[Option<(u32, u32)>],
    last: bool,
    page: u32,
) -> crate::engine::FireRequest {
    let mut req = full.clone();
    req.lanes = full
        .lanes
        .iter()
        .zip(piece)
        .filter_map(|(lane, &rows)| {
            let (from, to) = rows?;
            let mut lane = lane.clone();
            let (a, b) = (from as usize, to as usize);
            lane.tokens = lane.tokens[a..b].to_vec();
            if !lane.positions.is_empty() {
                lane.positions = lane.positions[a..b].to_vec();
            }
            // A row's mask counts its columns from the boundary its piece
            // states ([`restate_boundary`]): its rows are cut, its columns not.
            if let Some(::engine::Masking::Rows(masks)) = &mut lane.mask {
                *masks = masks[a..b].to_vec();
            }
            lane.kv.held += from;
            let len = lane.kv.held + (to - from);
            lane.pages.truncate(len.div_ceil(page.max(1)) as usize);
            if last {
                if let ::engine::Readout::Rows(read) = &mut lane.readout {
                    for row in read.iter_mut() {
                        *row -= from;
                    }
                }
            } else {
                lane.readout = ::engine::Readout::None;
            }
            Some(lane)
        })
        .collect();
    if !last {
        req.boundary_program = false;
        req.ring_moves.clear();
    }
    // A cut ends before the first media span ([`first_media_row`]): only
    // the last piece carries media, at its own rows.
    if last {
        let from = piece.first().copied().flatten().map_or(0, |(from, _)| from);
        let to = piece.first().copied().flatten().map_or(0, |(_, to)| to);
        for media in &mut req.media {
            for row in media.routes.iter_mut().filter(|row| **row >= 0) {
                *row -= from as i32;
            }
            if !media.token_positions.is_empty() {
                media.token_positions =
                    media.token_positions[3 * from as usize..3 * to as usize].to_vec();
            }
        }
    } else {
        req.media.clear();
    }
    req.single_token_mode = req.lanes.iter().all(|lane| lane.tokens.len() == 1);
    req
}

/// A windowed pass's piece over slots `slots`: the window checkpoint it
/// takes (the furthest of the pass's `checkpoints` inside it, its own end
/// included), and the row its advance retires no slot behind (`keep`): the
/// pass's earliest checkpoint past `held` (where the pass starts) at or
/// before the piece's end, its own end included and one an earlier piece
/// takes (its next pieces may be on their way before it settles and takes
/// it). Its windows reach back from there.
fn piece_window_rows(
    checkpoints: &[u64],
    windowed: bool,
    held: u64,
    slots: &std::ops::Range<u64>,
) -> (Option<u64>, Option<u32>) {
    let window_at = checkpoints
        .iter()
        .copied()
        .filter(|&row| windowed && slots.start < row && row <= slots.end)
        .max();
    let keep = checkpoints
        .iter()
        .copied()
        .filter(|&row| windowed && held < row && row <= slots.end)
        .min()
        .and_then(|row| u32::try_from(row).ok());
    (window_at, keep)
}

/// Several lanes are cut alike and aligned at their ends, so the last piece
/// holds every lane's read-out rows.
pub(crate) fn pass_pieces(
    req: &crate::engine::FireRequest,
    cuttable: bool,
    lanes_cuttable: bool,
    share: u32,
    page: u32,
    forced: &[u32],
    tails: &[u32],
) -> Vec<Vec<Option<(u32, u32)>>> {
    let tail = |lane: usize| tails.get(lane).copied().unwrap_or(0);
    let whole = || {
        vec![
            req.lanes
                .iter()
                .map(|lane| Some((0, u32::try_from(lane.tokens.len()).unwrap_or(u32::MAX))))
                .collect(),
        ]
    };
    if req.lanes.len() <= 1 {
        return host_cuts(req, cuttable, share, page, forced, tail(0))
            .into_iter()
            .map(|cut| vec![Some(cut)])
            .collect();
    }
    if !cuttable || !lanes_cuttable || !separable_lanes(req) {
        return whole();
    }
    let page = page.max(1);
    let lanes = u32::try_from(req.lanes.len()).unwrap_or(u32::MAX);
    let share = (share / lanes / page * page).max(page);
    let per_lane: Vec<Vec<(u32, u32)>> = req
        .lanes
        .iter()
        .enumerate()
        .map(|(at, lane)| {
            let one = crate::engine::FireRequest::one(lane.clone());
            host_cuts(&one, true, share, page, &[], tail(at))
        })
        .collect();
    let count = per_lane.iter().map(Vec::len).max().unwrap_or(1);
    (0..count)
        .map(|at| {
            per_lane
                .iter()
                .map(|cuts| {
                    (at + cuts.len())
                        .checked_sub(count)
                        .map(|index| cuts[index])
                })
                .collect()
        })
        .collect()
}

fn separable_lanes(req: &crate::engine::FireRequest) -> bool {
    req.media.is_empty()
        && req.voxels.is_empty()
        && req.lanes.iter().all(|lane| {
            lane.group.is_none() && separable_rows(&crate::engine::FireRequest::one(lane.clone()))
        })
}

/// The leading rows whose KV is their tokens' alone (before any media row), so their pages seal.
fn keyed_rows(req: &crate::engine::FireRequest) -> u32 {
    let [lane] = req.lanes.as_slice() else {
        return 0;
    };
    let held = lane.kv.held;
    let keyed = plain_lane(lane)
        && lane.mask.is_none()
        && !lane.tokens.contains(&u32::MAX)
        && (lane.positions.is_empty()
            || lane
                .positions
                .iter()
                .enumerate()
                .all(|(i, &at)| at == held + i as u32));
    if !keyed {
        return 0;
    }
    let rows = u32::try_from(lane.tokens.len()).unwrap_or(u32::MAX);
    first_media_row(req).unwrap_or(rows).min(rows)
}

/// The rows a pass over `rows` keeps its state at: those the inferlet asks
/// for (`asked`), each floored to whole pages and within the `keyed` pages
/// the pass keys. The runtime keeps none unasked: where a state is kept is
/// the inferlet's (and its policy's) to say.
pub(crate) fn checkpoint_rows(
    asked: &[u32],
    rows: std::ops::Range<u64>,
    page_size: u32,
    keyed: usize,
) -> Vec<u64> {
    let page = u64::from(page_size.max(1));
    let mut at: Vec<u64> = asked
        .iter()
        .map(|&row| u64::from(row) / page * page)
        .filter(|&row| rows.start < row && row <= rows.end && row / page <= keyed as u64)
        .collect();
    at.sort_unstable();
    at.dedup();
    at
}

static SEALED_PAGES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// What a keyed pass claimed to compute, and the rows it computes: passes
/// whose first pages it claimed wait for its fires to land. Landed whole,
/// the rows it froze seal there and then, and the waiters map them in;
/// otherwise (failed, refused, dropped) the claims go unsealed, and a waiter
/// computes the pages itself. Never waits on anything: it runs where the
/// pass's last fire resolves, or goes as the pass does.
struct SealOnLanding {
    claims: crate::pipeline::coalesce::ClaimGuard,
    stores: crate::store::registry::Stores,
    ws: crate::store::kv::page_table::WorkingSetId,
    start: u64,
    slots: Vec<crate::store::kv::hash::Hash256>,
    page_size: u32,
    failure: PipelineFailure,
}

impl SealOnLanding {
    /// Seals once every one of the pass's `fires` has settled, whatever
    /// order they land in, if each landed; a hook never run (its fire gone
    /// unresolved, or the runtime with it) counts as not landed, or drops
    /// the claims as the last hook goes.
    fn on_landing(self, fires: &[crate::engine::WorkItemCompletion]) {
        struct Countdown {
            left: std::sync::atomic::AtomicUsize,
            whole: std::sync::atomic::AtomicBool,
            sealing: Mutex<Option<SealOnLanding>>,
        }
        use std::sync::atomic::Ordering::{AcqRel, Acquire, Release};
        let countdown = Arc::new(Countdown {
            left: std::sync::atomic::AtomicUsize::new(fires.len()),
            whole: std::sync::atomic::AtomicBool::new(true),
            sealing: Mutex::new(Some(self)),
        });
        for fire in fires {
            let countdown = Arc::clone(&countdown);
            fire.on_settled_landed(move |landed| {
                if !landed {
                    countdown.whole.store(false, Release);
                }
                if countdown.left.fetch_sub(1, AcqRel) == 1
                    && let Some(sealing) = countdown.sealing.lock().unwrap().take()
                {
                    sealing.landed(countdown.whole.load(Acquire));
                }
            });
        }
    }

    fn landed(self, landed: bool) {
        if landed && pipeline_failed(&self.failure).is_none() {
            let sealed =
                crate::store::registry::with_kv_lock(&self.stores.kv, "host-other", |kv| {
                    kv.seal_landed(self.ws, self.start, &self.slots, self.page_size)
                });
            match sealed {
                Ok(pages) => {
                    SEALED_PAGES.fetch_add(u64::from(pages), std::sync::atomic::Ordering::Relaxed);
                }
                Err(error) => tracing::debug!(%error, "kv supply: sealing what a pass landed"),
            }
        }
        drop(self.claims);
    }
}

struct SupplyCounts {
    passes: std::sync::atomic::AtomicU64,
    tokens: std::sync::atomic::AtomicU64,
    held: std::sync::atomic::AtomicU64,
    found: std::sync::atomic::AtomicU64,
    mapped: std::sync::atomic::AtomicU64,
}

static SUPPLIED: SupplyCounts = SupplyCounts {
    passes: std::sync::atomic::AtomicU64::new(0),
    tokens: std::sync::atomic::AtomicU64::new(0),
    held: std::sync::atomic::AtomicU64::new(0),
    found: std::sync::atomic::AtomicU64::new(0),
    mapped: std::sync::atomic::AtomicU64::new(0),
};

#[derive(Clone, Copy)]
enum Attach {
    No,
    Kv,
    /// KV pages on a fresh state would read as another sequence.
    Checkpoint(crate::store::rs::RsWorkingSetId),
}

#[derive(Default)]
struct Supplied {
    rows: u32,
    keys: Vec<crate::store::kv::hash::Hash256>,
}

/// Maps in the sealed pages holding a keyed pass's first rows, short of its
/// last row's page, which is read out so always computed. Mapping freezes
/// nothing: what the set has folded is what its passes state. A recurrent
/// state is folded through the checkpoint it resumes from, so it resumes no
/// further than the `folds` rows the pass folds.
fn supply(
    stores: &crate::store::registry::Stores,
    ws: crate::store::kv::page_table::WorkingSetId,
    full: &crate::engine::FireRequest,
    slots: &[crate::store::kv::hash::Hash256],
    attach: Attach,
    folds: u64,
    page_size: u32,
) -> Supplied {
    use std::sync::atomic::Ordering::Relaxed;
    let held = u64::from(full.lanes[0].kv.held);
    let rows = slots.len() as u64;
    let page = u64::from(page_size.max(1));
    let limit = (held + rows - 1) / page;
    let tokens_of = |pages: u64| &full.lanes[0].tokens[..((pages * page) - held) as usize];
    let fresh = (held == 0).then(|| crate::store::kv::KvStore::fresh_page_keys(slots, page_size));
    let looked = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
        let keys = match fresh {
            Some(keys) => keys,
            None => {
                let Some(keys) = kv.page_keys(ws, held, slots, page_size)? else {
                    return Ok(None);
                };
                keys
            }
        };
        // A pass that may map nothing in looks nothing up.
        let run = match attach {
            Attach::No => Vec::new(),
            _ => kv.cached_run(&keys),
        };
        let clean = held.is_multiple_of(page) && kv.mapped_len(ws)? == held / page;
        let mapped = match attach {
            Attach::Kv if clean && kv.keeps() => {
                let mapped = kv.attach_run(ws, &run, limit)?;
                kv.log_rows(ws, held, tokens_of(held / page + mapped))?;
                mapped
            }
            Attach::Kv if !run.is_empty() => {
                tracing::debug!(
                    target: "pie::kv",
                    found = run.len(),
                    held,
                    why = if kv.keeps() {
                        "the set's rows end inside a page or past what it maps"
                    } else {
                        "the store keeps nothing"
                    },
                    "attach: found pages not mapped"
                );
                0
            }
            _ => 0,
        };
        Ok::<_, crate::store::kv::KvStoreError>(Some((run, keys, mapped, clean)))
    });
    let (run, keys, mut mapped, clean) = match looked {
        Ok(Some(found)) => found,
        Ok(None) => return Supplied::default(),
        Err(error) => {
            tracing::warn!(%error, "kv supply: the lookup failed; the pass computes all it writes");
            return Supplied::default();
        }
    };
    if let Attach::Checkpoint(rs) = attach
        && clean
        && held == 0
    {
        mapped = resume(
            stores,
            ws,
            rs,
            &keys,
            run.len().min((limit.min(folds / page)) as usize),
            page_size,
        );
        if mapped > 0 {
            let logged = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
                kv.log_rows(ws, 0, tokens_of(mapped))
            });
            if let Err(error) = logged {
                tracing::debug!(%error, "kv supply: logging what was mapped in");
            }
        }
    }
    let total = held + rows;
    // As far as it could map: whole pages short of the last row's.
    let found = (run.len() as u64).min(limit) * page;
    let mapped_rows = mapped * page;
    let passes = SUPPLIED.passes.fetch_add(1, Relaxed) + 1;
    let tokens = SUPPLIED.tokens.fetch_add(total, Relaxed) + total;
    let held = SUPPLIED.held.fetch_add(held, Relaxed) + held;
    let found = SUPPLIED.found.fetch_add(found, Relaxed) + found;
    let mapped_total = SUPPLIED.mapped.fetch_add(mapped_rows, Relaxed) + mapped_rows;
    if passes.is_multiple_of(256) {
        tracing::info!(
            passes,
            tokens,
            held,
            found,
            mapped = mapped_total,
            sealed_pages = SEALED_PAGES.load(Relaxed),
            "kv supply: over keyed passes, what sealed pages held and what was mapped in"
        );
    }
    Supplied {
        rows: u32::try_from(mapped_rows).unwrap_or(0),
        keys,
    }
}

/// Runs under a fire lease, so its own process is not suspended under it; the
/// run stays held until mapped in.
async fn promote<C: FireContext>(
    ctx: &mut C,
    stores: &crate::store::registry::Stores,
    ws: &KvWorkingSet,
    full: &crate::engine::FireRequest,
    slots: &[crate::store::kv::hash::Hash256],
    pipeline_id: uuid::Uuid,
) -> Result<Option<Promoted>, String> {
    let (engine, page_size) = (ws.engine, stores.kv_page_size);
    let held = u64::from(full.lanes[0].kv.held);
    let page = u64::from(page_size.max(1));
    let limit = ((held + slots.len() as u64 - 1) / page) as usize;
    let fresh = (held == 0).then(|| crate::store::kv::KvStore::fresh_page_keys(slots, page_size));
    let found = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
        let keys = match fresh {
            Some(keys) => keys,
            None => {
                let Some(keys) = kv.page_keys(ws.id, held, slots, page_size)? else {
                    return Ok::<_, crate::store::kv::KvStoreError>(None);
                };
                keys
            }
        };
        let mut run = kv.cached_run_any_tier(&keys);
        run.truncate(limit);
        let below = kv.below_device(&run);
        Ok((below > 0).then_some((run, below)))
    })
    .map_err(|error| error.to_string())?;
    let Some((run, below)) = found else {
        return Ok(None);
    };
    let demand = crate::planner::Demand {
        kv_pages: u32::try_from(below).unwrap_or(u32::MAX),
        rs: Default::default(),
    };
    let Some(mut grant) = acquire_grant(ctx, pipeline_id, demand, false, || Ok(Vec::new())).await? else {
        return Ok(None);
    };
    let Ok(lease) = ws.fire_lease() else {
        return Ok(None);
    };
    let prepared = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
        kv.prepare_promote(&run, grant.lend_kv())
    })
    .map_err(|error| error.to_string())?;
    let Some(txn) = prepared else {
        return Ok(None);
    };
    let mut guard = PromoteGuard {
        stores: stores.clone(),
        engine,
        txn: Some(txn),
        in_flight: Vec::new(),
        submitted: false,
    };
    let (kind, plans) = guard.txn.as_ref().map_or_else(
        || (crate::scheduler::TierKind::Kv, Vec::new()),
        |txn| {
            (
                crate::scheduler::TierKind::kv(txn.shape()),
                txn.copy_plans(),
            )
        },
    );
    for (tier, device, slots) in plans {
        let copy = crate::scheduler::TierMove {
            kind,
            tier,
            out: false,
            device,
            slots,
            source: None,
        };
        let completion = crate::scheduler::copy_tier_tracked(engine, &copy)
            .map_err(|error| format!("{error:#}"))?;
        guard.submitted = true;
        guard.in_flight.push(completion.clone());
        completion.wait().await.map_err(|error| error.to_string())?;
    }
    guard.in_flight.clear();
    let txn = guard.txn.take().expect("taken once");
    let pin =
        crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| kv.commit_promote(txn))
            .map_err(|error| error.to_string())?;
    Ok(Some(Promoted {
        pages: below as u64,
        stores: stores.clone(),
        pin: Some(pin),
        _lease: lease,
    }))
}

/// Dropped before it commits, it gives back what it took once any copy it
/// started has landed: written its destinations, not only settled.
struct PromoteGuard {
    stores: crate::store::registry::Stores,
    engine: usize,
    txn: Option<crate::store::kv::KvResidencyTxn>,
    in_flight: Vec<crate::scheduler::ControlCompletion>,
    /// A copy of the promote was handed to the engine.
    submitted: bool,
}

impl Drop for PromoteGuard {
    fn drop(&mut self) {
        let Some(txn) = self.txn.take() else {
            return;
        };
        let stores = self.stores.clone();
        let in_flight = std::mem::take(&mut self.in_flight);
        let abort = move || {
            crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
                kv.abort_restore(txn);
            });
            for planner in crate::planner::planners() {
                planner.pages_freed();
            }
        };
        if !self.submitted {
            abort();
            return;
        }
        if !crate::rt::has_runtime() {
            tracing::error!(
                "a promote dropped with a copy handed to the engine and no runtime; \
                 preserving its pages to avoid reuse under the copy"
            );
            return;
        }
        let engine = self.engine;
        crate::rt::spawn(async move {
            let settled = crate::scheduler::settle_copies(engine);
            if let Err(error) =
                crate::scheduler::abort_once_landed(in_flight, settled, abort).await
            {
                tracing::error!(
                    engine,
                    %error,
                    "a promote's copies did not settle; preserving its pages to avoid reuse \
                     under a copy"
                );
            }
        });
    }
}

struct Promoted {
    /// Pages of the run that came up from below the device.
    pages: u64,
    stores: crate::store::registry::Stores,
    pin: Option<crate::store::kv::ReaderPin>,
    _lease: crate::store::kv::working_set::KvFireLease,
}

impl Drop for Promoted {
    fn drop(&mut self) {
        if let Some(pin) = self.pin.take() {
            crate::store::registry::with_kv_lock(&self.stores.kv, "host-other", |kv| {
                kv.release_reader(pin);
            });
        }
    }
}

/// Resumes a hybrid pass from its furthest checkpoint within `reach` pages,
/// which both halves are then folded through; none when either half is gone
/// by then, and `rs` holds no state again, the set having folded nothing.
pub(crate) fn resume(
    stores: &crate::store::registry::Stores,
    ws: crate::store::kv::page_table::WorkingSetId,
    rs: crate::store::rs::RsWorkingSetId,
    keys: &[crate::store::kv::hash::Hash256],
    reach: usize,
    page_size: u32,
) -> u64 {
    let adopted = {
        let mut store = stores.rs.lock().unwrap();
        (1..=reach).rev().find(|&pages| {
            let key = &keys[pages - 1];
            store.has_checkpoint(key) && store.adopt_checkpoint(rs, key).unwrap_or(false)
        })
    };
    let Some(pages) = adopted else {
        return 0;
    };
    let pages = pages as u64;
    let mapped = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
        let run = kv.cached_run(&keys[..pages as usize]);
        if run.len() as u64 != pages {
            return Ok(0);
        }
        let mapped = kv.attach_run(ws, &run, pages)?;
        if mapped == pages {
            kv.freeze(ws, pages * u64::from(page_size.max(1)))?;
        }
        Ok::<_, crate::store::kv::KvStoreError>(mapped)
    });
    match mapped {
        Ok(mapped) if mapped == pages => pages,
        outcome => {
            if let Err(error) = outcome {
                tracing::debug!(%error, "kv supply: mapping a checkpoint's pages in");
            }
            if let Err(error) = stores.rs.lock().unwrap().forget_state(rs) {
                tracing::debug!(%error, "kv supply: dropping a state not resumed");
            }
            0
        }
    }
}

fn rs_slot_demand(
    stores: &crate::store::registry::Stores,
    ids: &[crate::store::rs::RsWorkingSetId],
    plan: &rs::RsPlan,
) -> Result<crate::store::rs::RsUnits, String> {
    if ids.is_empty() {
        return Ok(Default::default());
    }
    rs::demand(&stores.rs.lock().unwrap(), ids, plan)
}

fn prepare_host_kv_reserved(
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

fn prepare_explicit_kv_reserved(
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

struct Reserved {
    leases: Vec<KvFireLease>,
    copies: kv::PageCopies,
    kvtxns: Vec<KvTxnGuard>,
    rs: rs::PreparedRs,
    epoch: FireEpoch,
    unfired: Unfired,
}

/// What a pipeline says of a fire prepared that did not follow, where no
/// refusal said more.
const UNFIRED: &str = "pipeline: a fire was prepared and never submitted";

/// A fire prepared and still to be submitted. [`reserve`] ended by
/// committing what it folds into its recurrent states, and each piece's
/// prepare freezes what it folds by a count the host knows
/// ([`begin_piece`]); neither is taken back. So a fire that does not follow
/// fails its pipeline, and its sets with it, however the call that prepared
/// it ends: by a refusal, by a trap, or dropped where it waits. Held from
/// [`reserve`] to the submit, this leaves the failure when it is dropped
/// unless the fire was submitted ([`Unfired::submitted`]); a refusal that
/// left its own reason first keeps it.
struct Unfired {
    failure: Option<PipelineFailure>,
}

impl Unfired {
    fn new(failure: &PipelineFailure) -> Self {
        Self {
            failure: Some(failure.clone()),
        }
    }

    /// The engine took the fire: its pipeline is still to queue it.
    fn submitted(mut self) -> Unqueued {
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
const UNQUEUED: &str = "pipeline: a fire was submitted and never queued to settle";

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
struct Unqueued {
    failure: Option<PipelineFailure>,
}

impl Unqueued {
    /// `fire` is its pipeline's to finalize from here.
    fn queue(mut self, fires: &PendingFireQueue, fire: PendingFire) {
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
fn fail_unless_failed(failure: &PipelineFailure, reason: &str) {
    let mut pipeline = failure
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if pipeline.is_none() {
        *pipeline = Some(reason.to_string());
    }
}

/// A demand that drifted from its grant is quoted again. What it binds
/// opens the fire's epoch on engine `engine` of model `model`. What it
/// returns holds the pipeline's `failure` to the fire's submit
/// ([`Unfired`]); a reserve refused committed nothing, and leaves none.
#[allow(clippy::too_many_arguments)]
async fn reserve<C: FireContext>(
    ctx: &mut C,
    (model, engine): (usize, usize),
    stores: &crate::store::registry::Stores,
    pipeline_id: uuid::Uuid,
    failure: &PipelineFailure,
    sets: &[(&KvWorkingSet, Written)],
    set_lanes: &[Vec<usize>],
    rs_ids: &[crate::store::rs::RsWorkingSetId],
    rs_plan: impl Fn() -> Result<rs::RsPlan, String>,
    windowed: bool,
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
        let Some(mut grant) = acquire_grant(ctx, pipeline_id, demand, windowed, alone).await?
        else {
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

fn prepare_reserved(
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

pub enum PendingOp {
    Fire(PendingFire),
    #[cfg(test)]
    TestStub,
}

impl PendingOp {
    pub(crate) fn is_settled(&self) -> bool {
        match self {
            PendingOp::Fire(fire) => fire.completion.is_settled(),
            #[cfg(test)]
            PendingOp::TestStub => true,
        }
    }

    fn binds(&self, cell: &Arc<Mutex<crate::pipeline::channel::ChannelCell>>) -> bool {
        match self {
            PendingOp::Fire(fire) => fire.cells.iter().any(|bound| Arc::ptr_eq(bound, cell)),
            #[cfg(test)]
            PendingOp::TestStub => true,
        }
    }

    /// Tells the scheduler that the process waits on this fire's output:
    /// until it settles, its lane submits nothing, and no frame waits on it.
    pub(crate) fn tell_awaited(&self) {
        match self {
            PendingOp::Fire(PendingFire {
                completion,
                engine,
                lane: Some(lane),
                ..
            }) => crate::scheduler::worker::notify_lane_waits(*engine, *lane, completion.clone()),
            PendingOp::Fire(_) => {}
            #[cfg(test)]
            PendingOp::TestStub => {}
        }
    }

    fn completion_signal(&self) -> OpSignal {
        match self {
            PendingOp::Fire(fire) => OpSignal::Fire(fire.completion.clone()),
            #[cfg(test)]
            PendingOp::TestStub => OpSignal::TestReady,
        }
    }

    pub(crate) fn is_preemption_detachable(&self) -> bool {
        matches!(
            self,
            PendingOp::Fire(PendingFire {
                device_geometry: false,
                ..
            })
        )
    }
}

enum FinalizeAction {
    None,
    Fail {
        fwd_rep: u32,
        cells: BoundCells,
        failure: PipelineFailure,
        reason: String,
    },
}

pub(crate) struct FinalizeOutcome {
    action: FinalizeAction,
    ws_guard: Vec<KvFireLease>,
}

enum OpSignal {
    Fire(crate::engine::WorkItemCompletion),
    #[cfg(test)]
    TestReady,
}

impl std::future::Future for OpSignal {
    type Output = ();

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<()> {
        match self.get_mut() {
            OpSignal::Fire(completion) => std::pin::Pin::new(completion).poll(cx).map(|_| ()),
            #[cfg(test)]
            OpSignal::TestReady => std::task::Poll::Ready(()),
        }
    }
}

#[cfg(test)]
pub(crate) fn test_pending_op_stub() -> PendingOp {
    PendingOp::TestStub
}

struct FireWs {
    ws: crate::store::kv::page_table::WorkingSetId,
    kvtxn: KvTxnGuard,
    log: Option<LogFire>,
    lease: KvFireLease,
    fold: Option<FoldAt>,
    reports: Vec<ReportTag>,
    staged: Option<StagedFire>,
}

/// A fire's rows on one working set that land where the device says: the
/// set counts them as unfolded until the fire settles, and learns from the
/// reports under `tags` where they ended.
struct StagedFire {
    rows: std::sync::Arc<crate::inferlet::host::working_set::StagedRows>,
    tags: Vec<u64>,
}

impl StagedFire {
    /// The end its lanes reported, the furthest of them; none unless each
    /// reported one.
    fn settle(&self, reported: &std::collections::HashMap<u64, ::engine::KvWrites>, success: bool) {
        let end = self
            .tags
            .iter()
            .map(|tag| reported.get(tag).and_then(|written| written.end))
            .collect::<Option<Vec<u32>>>()
            .and_then(|ends| ends.into_iter().max())
            .filter(|_| success);
        self.rows.settle(end);
    }
}

/// A fire dropped unsettled reported no end.
impl Drop for StagedFire {
    fn drop(&mut self) {
        self.rows.settle(None);
    }
}

struct FoldAt {
    by: FoldBy,
    decoded: bool,
}

enum FoldBy {
    /// Frozen as the fire was prepared; its pages seal when it settles.
    To(u64),
    /// Frozen when the fire settles: the folded length is exact again once
    /// the last fire of its working set placed this way settles. `written`
    /// when it reaches no further than where the engine reports the fire's
    /// rows ended, and is not placed without that report.
    Settled {
        count: Option<u32>,
        tags: Vec<u64>,
        reach: u64,
        written: bool,
        rs: Option<RsDeviceFold>,
        /// Held from the fire's submit ([`unsettle`]).
        unplaced: Option<Unplaced>,
    },
}

/// A working set's folded length is a lower bound while a fire on its way
/// folds it by a count placed when the fire settles
/// ([`crate::store::kv::KvStore::unsettle`]): the set is not forked or
/// exported until then. Every way the fire ends lifts it: its fold
/// settling, placed or not ([`Self::settle`]), and, dropped before that,
/// the engine being done with it, the fold then never placed.
struct Unplaced {
    ws: crate::store::kv::page_table::WorkingSetId,
    submitted: Option<(
        Arc<parking_lot::Mutex<crate::store::kv::KvStore>>,
        crate::engine::WorkItemCompletion,
    )>,
}

impl Unplaced {
    /// The fold settled, `placed` when its count was.
    fn settle(&mut self, kv: &mut crate::store::kv::KvStore, placed: bool) {
        if self.submitted.take().is_some()
            && let Err(error) = kv.fold_settled(self.ws, placed)
        {
            tracing::debug!(%error, "pipeline: a fold settled on a set gone");
        }
    }
}

impl Drop for Unplaced {
    fn drop(&mut self) {
        let Some((store, completion)) = self.submitted.take() else {
            return;
        };
        let ws = self.ws;
        completion.on_settled(move || {
            crate::store::registry::with_kv_lock(&store, "host-other", |kv| {
                let _ = kv.fold_settled(ws, false);
            });
        });
    }
}

/// The recurrent state a fold the device counts freezes with the KV, by the
/// same count, when the fire settles. From the fire's submit until then its
/// working set is marked in the store
/// ([`crate::store::rs::RsStore::begin_device_fold`]): the engine's record
/// stands ahead of the frozen length the runtime states, so the set is not
/// forked, checkpointed, exported or moved. Every way the fire ends lifts
/// the mark: its fold settling, counted or not ([`Self::settle`]), and,
/// dropped before that, the engine being done with it.
struct RsDeviceFold {
    ws: crate::store::rs::RsWorkingSetId,
    /// Checkpoints of `ws` taken ahead of the fire's later fires
    /// ([`checkpoints_ahead`]), forked before the fold: each folds by the
    /// same count.
    followers: Vec<crate::store::rs::RsWorkingSetId>,
    submitted: Option<(
        std::sync::Arc<std::sync::Mutex<crate::store::rs::RsStore>>,
        crate::engine::WorkItemCompletion,
    )>,
}

impl RsDeviceFold {
    /// What a fire dropped before it settled leaves its set refused with:
    /// it may have folded, and nothing read by how much.
    const DROPPED: &str = "pipeline: the fire was dropped before it settled";

    /// What a fire that ran to its end and brought no count leaves its set
    /// refused with: the engine folded by a count of its own and its
    /// record stands where that left it, which the frozen length the
    /// runtime states never reaches.
    const UNREPORTED: &str = "pipeline: the fire ran and its engine did not report what it folded";

    fn planned(ws: crate::store::rs::RsWorkingSetId) -> Self {
        Self {
            ws,
            followers: Vec::new(),
            submitted: None,
        }
    }

    fn sets(&self) -> impl Iterator<Item = crate::store::rs::RsWorkingSetId> + '_ {
        std::iter::once(self.ws).chain(self.followers.iter().copied())
    }

    fn submitted(
        &mut self,
        stores: &crate::store::registry::Stores,
        completion: &crate::engine::WorkItemCompletion,
    ) {
        let mut store = stores.rs.lock().unwrap();
        for ws in self.sets() {
            store.begin_device_fold(ws);
        }
        drop(store);
        self.submitted = Some((stores.rs.clone(), completion.clone()));
    }

    /// `slots` the fold froze, where the fire ran and reported them;
    /// `failure` why it did not run.
    fn settle(mut self, slots: Option<u32>, failure: Option<&str>) {
        if let Some((store, _)) = self.submitted.take() {
            let mut store = store.lock().unwrap();
            for ws in self.sets() {
                store.end_device_fold(ws, slots, failure);
            }
        }
    }
}

impl Drop for RsDeviceFold {
    fn drop(&mut self) {
        let Some((store, completion)) = self.submitted.take() else {
            return;
        };
        let sets: Vec<_> = self.sets().collect();
        completion.on_settled(move || {
            let mut store = store.lock().unwrap();
            for ws in sets {
                store.end_device_fold(ws, None, Some(Self::DROPPED));
            }
        });
    }
}

/// The checkpoints of a recurrent state (a record: a folded state, or a
/// ring-only window) at the end of the fire `parts` describes, just
/// submitted, taken now ([`crate::store::rs::RsStore::checkpoint_ahead`]):
/// before any later fire of the set is laid, a cut pass's next piece or a
/// decode run ahead, which would otherwise write the record in place
/// before the take at this fire's settle could fork it. The later fire
/// finds the record shared and copies it on the device, behind this one;
/// nothing waits. Called before the fire's device folds are marked, so a
/// fold the device counts is followed by the fork. One refused stays to be
/// taken at the settle, which names why.
fn checkpoints_ahead(
    stores: &crate::store::registry::Stores,
    checkpoints: &mut [TakeCheckpoint],
    parts: &mut [FireWs],
) {
    for checkpoint in checkpoints
        .iter_mut()
        .filter(|checkpoint| !checkpoint.snapshot && checkpoint.window_at.is_none())
    {
        let forked = stores.rs.lock().unwrap().checkpoint_ahead(checkpoint.rs);
        let snapshot = match forked {
            Ok(Some(snapshot)) => snapshot,
            Ok(None) => continue,
            Err(error) => {
                tracing::debug!(%error, "pipeline: a checkpoint taken ahead of the next fire");
                continue;
            }
        };
        for part in parts.iter_mut() {
            if let Some(FoldAt {
                by: FoldBy::Settled { rs: Some(fold), .. },
                ..
            }) = &mut part.fold
                && fold.ws == checkpoint.rs
            {
                fold.followers.push(snapshot);
            }
        }
        checkpoint.rs = snapshot;
        checkpoint.snapshot = true;
    }
}

/// The fire is submitted: each working set whose recurrent state it folds
/// by a count the device computes is marked until that fold settles.
fn mark_device_folds(
    stores: &crate::store::registry::Stores,
    completion: &crate::engine::WorkItemCompletion,
    parts: &mut [FireWs],
) {
    for part in parts {
        if let Some(FoldAt {
            by: FoldBy::Settled { rs: Some(rs), .. },
            ..
        }) = &mut part.fold
        {
            rs.submitted(stores, completion);
        }
    }
}

fn plain_decode(req: &crate::engine::FireRequest, adapted: bool, fire_wide_mask: bool) -> bool {
    !adapted
        && !fire_wide_mask
        && matches!(req.lanes.as_slice(), [lane] if plain_lane(lane) && lane.mask.is_none())
}

struct LogFire {
    seq: u64,
    rows: LogRows,
}

enum LogRows {
    Known {
        start: u64,
        tokens: Vec<u32>,
    },
    Reported {
        tag: u64,
        pages: Vec<(u32, u64)>,
        page_size: u32,
    },
    /// Not known: the log is cut where the fire may start writing.
    Unknown,
}

static KV_WRITES: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<u64, ::engine::KvWrites>>,
> = std::sync::LazyLock::new(Default::default);

/// Each beam group's common slot, by its instance (`BEAM-PAGE-LISTS.md`
/// §11.3): the largest any landed step reported. Monotone, so a later report
/// never lowers it.
static BEAM_COMMON: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<::engine::InstanceId, u32>>,
> = std::sync::LazyLock::new(Default::default);

pub(crate) fn record_beam_common(commons: Vec<(::engine::InstanceId, u32)>) {
    if commons.is_empty() {
        return;
    }
    let mut held = BEAM_COMMON.lock().unwrap();
    for (instance, common) in commons {
        let at = held.entry(instance).or_insert(0);
        *at = (*at).max(common);
    }
}

/// The largest common slot instance `instance`'s beam group has reported.
fn beam_common_of(instance: ::engine::InstanceId) -> u32 {
    BEAM_COMMON.lock().unwrap().get(&instance).copied().unwrap_or(0)
}

pub(crate) fn record_kv_writes(writes: Vec<::engine::KvWrites>) {
    let mut held = KV_WRITES.lock().unwrap();
    for written in writes {
        held.insert(written.tag, written);
    }
}

/// What is kept under `tag`, taken.
#[cfg(test)]
pub(crate) fn kept_kv_writes(tag: u64) -> Option<::engine::KvWrites> {
    ReportTag(tag).take()
}

struct ReportTag(u64);

impl ReportTag {
    fn new() -> ReportTag {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        ReportTag(NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }

    fn take(&self) -> Option<::engine::KvWrites> {
        KV_WRITES.lock().unwrap().remove(&self.0)
    }
}

/// The lane of `req` whose fold length is the device's
/// ([`::engine::fire::FoldLen::Device`]) and which asks no report: none in
/// a request staged whole. The engine answers such a lane's fold under its
/// tag, and a fire that brings none is taken to have gone unreported
/// ([`RsDeviceFold::UNREPORTED`]), so a lane staged without one would fail
/// its set for what the runtime never asked.
fn untagged_device_fold(req: &crate::engine::FireRequest) -> Option<usize> {
    req.lanes.iter().position(|lane| {
        lane.reports == 0
            && matches!(
                lane.rs,
                ::engine::fire::RsVerb::Buffer {
                    fold: ::engine::fire::FoldLen::Device,
                    ..
                }
            )
    })
}

fn report_on(req: &mut crate::engine::FireRequest, lane: usize, tags: &mut Vec<ReportTag>) -> u64 {
    let lane = &mut req.lanes[lane];
    if lane.reports == 0 {
        let tag = ReportTag::new();
        lane.reports = tag.0;
        tags.push(tag);
    }
    lane.reports
}

impl Drop for ReportTag {
    fn drop(&mut self) {
        KV_WRITES.lock().unwrap().remove(&self.0);
    }
}

/// A snapshot no checkpoint takes is released. A checkpoint the pass asked
/// and its fire ran to is kept, or refused by name: never let go unsaid.
pub(crate) fn take_checkpoint(
    stores: &crate::store::registry::Stores,
    checkpoint: &TakeCheckpoint,
    success: bool,
) -> Result<(), String> {
    let release = || {
        if checkpoint.snapshot {
            stores.rs.lock().unwrap().release_working_set(checkpoint.rs);
        }
    };
    if !success {
        release();
        return Ok(());
    }
    let owner = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
        kv.owner_of(checkpoint.kv)
    });
    let taken = if checkpoint.snapshot {
        let mut rs = stores.rs.lock().unwrap();
        // A fork that followed a fold the device counted, which went
        // unreported, states no length its record stands at.
        match rs.device_fold_gate(checkpoint.rs) {
            Ok(()) => {
                rs.keep(checkpoint.rs, checkpoint.key, owner);
                Ok(true)
            }
            Err(error) => {
                rs.release_working_set(checkpoint.rs);
                Err(error)
            }
        }
    } else {
        let mut rs = stores.rs.lock().unwrap();
        match checkpoint.window_at {
            Some(at) => rs.checkpoint_at(checkpoint.rs, checkpoint.key, owner, at),
            None => rs.checkpoint(checkpoint.rs, checkpoint.key, owner),
        }
    };
    match taken {
        Ok(true) => {
            crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
                kv.keep_checkpoint(checkpoint.kv, checkpoint.pages, owner);
            });
            Ok(())
        }
        // A state that holds nothing yet: nothing to keep.
        Ok(false) => Ok(()),
        Err(error) => Err(format!(
            "pipeline: the checkpoint asked at page {}: {error}",
            checkpoint.pages
        )),
    }
}

fn reported_rows(
    written: &::engine::KvWrites,
    pages: &[(u32, u64)],
    page_size: u32,
) -> Option<(u64, Vec<u32>)> {
    let rows = written.tokens.len();
    if rows == 0
        || written.positions.len() != rows
        || written.pages.len() != rows
        || written.offsets.len() != rows
    {
        return None;
    }
    let page = u64::from(page_size);
    let slot = |row: usize| {
        let (_, index) = pages.iter().find(|(id, _)| *id == written.pages[row])?;
        let offset = u64::from(written.offsets[row]);
        (offset < page).then_some(index * page + offset)
    };
    let first = slot(0)?;
    for row in 0..rows {
        let at = first + row as u64;
        if slot(row)? != at || u64::from(written.positions[row]) != at {
            return None;
        }
    }
    Some((first, written.tokens.clone()))
}

/// `rs` is the pass's working set, forked once the fire settles, or with
/// `snapshot` the hidden one the fire landed its state as of `pages` in.
pub(crate) struct TakeCheckpoint {
    pub(crate) kv: crate::store::kv::page_table::WorkingSetId,
    pub(crate) rs: crate::store::rs::RsWorkingSetId,
    pub(crate) snapshot: bool,
    pub(crate) pages: u64,
    pub(crate) key: crate::store::kv::hash::Hash256,
    /// A window's, as of this position, at or behind the fire's end.
    pub(crate) window_at: Option<u32>,
}

pub struct PendingFire {
    completion: crate::engine::WorkItemCompletion,
    checkpoints: Vec<TakeCheckpoint>,
    kv: Vec<FireWs>,
    device_geometry: bool,
    rstxn: RsTxnsGuard,
    model: usize,
    engine: usize,
    fwd_rep: u32,
    cells: BoundCells,
    failure: PipelineFailure,
    /// A dropped pass whose last fire this is; it closes when the fire settles.
    retired: Option<Box<crate::pipeline::instance::BoundForwardPass>>,
    /// The scheduler lane it fired on: a process waiting on its output
    /// tells that lane's frames not to wait for it.
    lane: Option<crate::scheduler::ProcessId>,
}

/// Hands a dropped pass to its last pending fire, or back if none is pending.
pub(crate) fn retire_pass(
    rep: u32,
    bound: Box<crate::pipeline::instance::BoundForwardPass>,
) -> Option<Box<crate::pipeline::instance::BoundForwardPass>> {
    let Some(fires) = bound.fires.clone() else {
        return Some(bound);
    };
    let mut queue = fires.lock().unwrap();
    let mut last = None;
    for op in queue.iter_mut() {
        if let PendingOp::Fire(fire) = op
            && fire.fwd_rep == rep
        {
            fire.fwd_rep = u32::MAX;
            last = Some(fire);
        }
    }
    match last {
        Some(fire) => {
            fire.retired = Some(bound);
            None
        }
        None => Some(bound),
    }
}

fn prepare_rs_reserved(
    stores: &crate::store::registry::Stores,
    ids: &[crate::store::rs::RsWorkingSetId],
    plan: &rs::RsPlan,
    grant: &mut crate::planner::AllocationGrant,
) -> Result<rs::PreparedRs, ReservedError> {
    let mut store = stores.rs.lock().unwrap();
    let required = rs::demand(&store, ids, plan)
        .map_err(|error| ReservedError::Fatal(format!("pipeline: rs demand: {error}")))?;
    if !grant.remaining_rs().covers(required) {
        return Err(ReservedError::Stale);
    }
    rs::prepare_many(&mut store, ids, plan, Some(grant.lend_rs()))
        .map_err(|error| ReservedError::Fatal(format!("pipeline: rs prepare: {error}")))
}

pub(crate) async fn drain_pipeline_fires<C: FireContext>(
    ctx: &mut C,
    fires: &PendingFires,
) -> Anyhow<()> {
    loop {
        let completion = fires
            .lock()
            .unwrap()
            .front()
            .map(PendingOp::completion_signal);
        let Some(completion) = completion else {
            return Ok(());
        };
        completion.await;

        let _finalize_guard = fires.finalize_guard().await;
        let op = {
            let mut queue = fires.lock().unwrap();
            queue
                .front()
                .is_some_and(PendingOp::is_settled)
                .then(|| queue.pop_front())
                .flatten()
        };
        if let Some(op) = op {
            finalize_op(ctx, op).await?;
        }
    }
}

/// The cursor a copy of `kv`'s set starts from: read once every fire on its
/// way has placed its fold. Until then the frozen length the KV working set
/// states is a bound, and a child that took it would state it for good: no
/// fire settles a copy.
///
/// The KV working set is the mark, on every set (its fires on their way that
/// fold by a count placed when they settle); the recurrent store's is only on a
/// set that keeps recurrent state. The fires are `fires`', the pipeline the
/// set is scoped to, and only this task settles them: they run to their end
/// first, woken as each completes. One dropped unsettled lifts the
/// recurrent store's mark when the engine is done with it, which wakes the
/// wait. One another finalizer took off the queue is under the finalizer
/// gate until its fold is placed and its failure latched: the frozen
/// length and the failure are read with the gate in hand.
///
/// A fire that failed refuses the copy as the pipeline's failure. A fold no
/// fire is left to place leaves the length unknown for good, and the copy
/// is refused by that name ([`UNPLACED`]) rather than left waiting.
///
/// [`UNPLACED`]: crate::store::kv::page_table::UNPLACED
pub(crate) async fn cursor_at_rest<C: FireContext>(
    ctx: &mut C,
    fires: &PendingFires,
    failure: &PipelineFailure,
    kv: &crate::store::kv::working_set::KvWorkingSet,
    rs: Option<crate::store::rs::RsWorkingSetId>,
    cursor: &CursorCell,
) -> Anyhow<Result<crate::inferlet::host::working_set::Cursor, String>> {
    let stores = crate::store::registry::get(kv.model, kv.engine);
    let marked = rs.is_some_and(|rs| {
        stores.rs.lock().unwrap().device_fold_gate(rs)
            == Err(crate::store::rs::RsError::DeviceFoldUnsettled)
    });
    let unsettled =
        crate::store::registry::with_kv_lock(&stores.kv, "host-working-set", |store| {
            store.unsettled(kv.id)
        })?;
    if unsettled > 0 || marked {
        drain_pipeline_fires(ctx, fires).await?;
        if let Some(rs) = rs {
            crate::store::rs::device_folds_settled(&stores.rs, &[rs]).await;
        }
    }
    let _gate = fires.finalize_guard().await;
    if let Some(reason) = failure.lock().unwrap().clone() {
        return Ok(Err(format!("pipeline failed: {reason}")));
    }
    let exact = crate::store::registry::with_kv_lock(&stores.kv, "host-working-set", |store| {
        store.folded_exact(kv.id)
    })?;
    let cursor = cursor.lock().unwrap().clone();
    Ok(if exact {
        Ok(cursor)
    } else {
        // The recurrent store says how the fire ended, where it keeps one.
        Err(rs
            .and_then(|rs| stores.rs.lock().unwrap().device_fold_gate(rs).err())
            .map_or_else(
                || format!("working set: {}", crate::store::kv::page_table::UNPLACED),
                |named| named.to_string(),
            ))
    })
}

/// Each lane's guest switches, and what the outputs the program reads ask of
/// it: the runtime switches their writers run under, and the last rows they
/// are computed over.
#[derive(Clone, Debug, Default)]
pub(crate) struct Switched {
    pub(crate) lanes: Vec<Vec<(String, String)>>,
    pub(crate) on: Vec<Vec<poem_ir::Builtin>>,
    pub(crate) tails: Vec<u32>,
    /// It reads a row output a cut fire would not hold whole.
    pub(crate) rows_read: bool,
}

impl Switched {
    pub(crate) fn of(pass: &ForwardPass) -> Switched {
        let model = crate::model::model();
        let lanes: Vec<Vec<(String, String)>> = pass
            .facts
            .iter()
            .map(|lane| lane.switches.clone())
            .collect();
        let mut on = vec![Vec::new(); lanes.len()];
        let mut tails = vec![0; lanes.len()];
        let mut rows_read = false;
        for (name, lane) in &pass.instance.program.reads {
            let switches = model.output_switches(name);
            if let Some(decl) = model.output(name) {
                rows_read |= decl.axis != poem_ir::InputAxis::Lane
                    && (!decl.readout || !switches.is_empty());
                if let Some(tail) = tails.get_mut(*lane as usize) {
                    *tail = (*tail).max(decl.tail);
                }
            }
            if let Some(lane_on) = on.get_mut(*lane as usize) {
                for builtin in switches {
                    if !lane_on.contains(&builtin) {
                        lane_on.push(builtin);
                    }
                }
            }
        }
        Switched {
            lanes,
            on,
            tails,
            rows_read,
        }
    }

    /// Whether the pass's rows may be cut, keyed and logged by their tokens.
    pub(crate) fn plain(&self) -> bool {
        let facts = crate::model::model().facts();
        !self.rows_read
            && self.lanes.iter().flatten().all(|(name, value)| {
                facts
                    .declared(name)
                    .is_some_and(|decl| decl.default_value() == value)
            })
    }
}

/// Stamps each lane's word: lane `at` of `req` is the pass's lane `of[at]`.
/// `device_rows`, when given, are the rows each lane carries on the device:
/// a pooled pass stages no host tokens, and its words still state its rows,
/// so a single-token lane takes the decode branch.
fn stamp_lane_words(
    req: &mut crate::engine::FireRequest,
    switched: &Switched,
    of: &[usize],
    fire_wide_mask: bool,
    device_rows: Option<&[u32]>,
) {
    let model = crate::model::model();
    let requests = lane_requests(req, switched, of, fire_wide_mask, device_rows);
    for (lane, request) in req.lanes.iter_mut().zip(&requests) {
        lane.word = model.word(request);
    }
}

/// What each lane of `req` asks of the model ([`stamp_lane_words`]). A lane
/// captures its attention scores only on the fire that runs the pass's
/// program, the one reading them: an earlier cut of a cut pass carries the
/// adapter alone, and scores it captured would have no reader.
fn lane_requests(
    req: &crate::engine::FireRequest,
    switched: &Switched,
    of: &[usize],
    fire_wide_mask: bool,
    device_rows: Option<&[u32]>,
) -> Vec<poem::Request> {
    let media: Vec<u32> = req.media.iter().map(|media| media.lane).collect();
    req.lanes
        .iter()
        .enumerate()
        .map(|(at, lane)| {
            let rows = device_rows
                .and_then(|rows| rows.get(at).copied())
                .unwrap_or_else(|| u32::try_from(lane.tokens.len()).unwrap_or(u32::MAX));
            let on = of
                .get(at)
                .and_then(|&pass_lane| switched.on.get(pass_lane))
                .map_or(&[][..], Vec::as_slice);
            let request = poem::Request::new(rows, lane.mask.is_some() || fire_wide_mask)
                .adapted(lane.adapter.is_some())
                .drafting(on.contains(&poem_ir::Builtin::Drafts))
                .capturing_scores(req.boundary_program && on.contains(&poem_ir::Builtin::Scores))
                .with_media(media.contains(&(at as u32)))
                .starting(starts(lane));
            let switches = of
                .get(at)
                .and_then(|&pass_lane| switched.lanes.get(pass_lane))
                .map_or(&[][..], Vec::as_slice);
            switches.iter().fold(request, |request, (name, value)| {
                request.switched(name, value)
            })
        })
        .collect()
}

/// Whether the lane's recurrent state is empty, as the engine reads its reset.
fn starts(lane: &::engine::Lane) -> bool {
    match lane.rs_reset {
        ::engine::RsReset::Fresh => true,
        ::engine::RsReset::Held => false,
        ::engine::RsReset::Inferred => !lane.kv_less && lane.kv.held == 0,
    }
}

fn every_lane(req: &crate::engine::FireRequest) -> Vec<usize> {
    (0..req.lanes.len()).collect()
}

/// A lane in a both-ways class is never cut, keyed or logged by its tokens.
fn reads_each_other(req: &crate::engine::FireRequest) -> bool {
    let model = crate::model::model();
    req.lanes.iter().any(|lane| model.wide(lane.word))
}

pub(crate) type LaneGroups = Vec<(crate::store::kv::page_table::WorkingSetId, Vec<usize>)>;

pub(crate) fn lane_groups(ids: &[crate::store::kv::page_table::WorkingSetId]) -> LaneGroups {
    let mut groups: LaneGroups = Vec::new();
    for (lane, &id) in ids.iter().enumerate() {
        match groups.iter_mut().find(|(ws, _)| *ws == id) {
            Some((_, lanes)) => lanes.push(lane),
            None => groups.push((id, vec![lane])),
        }
    }
    groups
}

/// Each lane rides a seat its working set holds: lane `i` of a group its
/// working set's `i`th.
pub(crate) fn stamp_lane_slots(
    req: &mut crate::engine::FireRequest,
    stores: &crate::store::registry::Stores,
    groups: &LaneGroups,
) -> Result<(), crate::store::chunks::SeatsShort> {
    for (ws, lanes) in groups {
        let seats = stores.seats.lock().unwrap().seats(*ws, lanes.len())?;
        for (&lane, &seat) in lanes.iter().zip(&seats) {
            if let Some(lane) = req.lanes.get_mut(lane) {
                lane.slot = seat;
            }
        }
    }
    Ok(())
}

/// How long a process's fire may go without progress before the planner
/// counts it lapsed and plans again (its `FIRE_PROGRESS`); no waiter fails
/// on it.
pub(crate) const SEAT_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

pub(crate) async fn seat_lane_slots(
    req: &mut crate::engine::FireRequest,
    stores: &crate::store::registry::Stores,
    groups: &LaneGroups,
) -> Result<(), String> {
    let freed = stores.seats.lock().unwrap().freed();
    // A working set letting its seats go is the event: it waits for it,
    // however long, and is refused at once only where no release could
    // ever seat it.
    loop {
        let mut freed = Box::pin(freed.notified());
        freed.as_mut().enable();
        let refusal = match stamp_lane_slots(req, stores, groups) {
            Ok(()) => return Ok(()),
            Err(error) => error,
        };
        if !stores.seats.lock().unwrap().could_seat(refusal.need) {
            return Err(format!(
                "pipeline: seating this fire's lanes: {refusal}; no release can seat it"
            ));
        }
        // Waiting for seats let go: its owner is parked here.
        let _parked = crate::inferlet::process::parks_here();
        freed.await;
    }
}

pub(crate) fn map_lane_pages(
    req: &mut crate::engine::FireRequest,
    stores: &crate::store::registry::Stores,
    groups: &LaneGroups,
) -> Result<(), String> {
    for (ws, lanes) in groups {
        if lanes
            .iter()
            .all(|&lane| req.lanes.get(lane).is_none_or(|lane| lane.pages.is_empty()))
        {
            continue;
        }
        let table = working_set_flat_table(stores, *ws)?;
        for &lane in lanes {
            let Some(lane) = req.lanes.get_mut(lane) else {
                continue;
            };
            map_pages(&mut lane.pages, &table)?;
            lane.ws = Some(*ws);
        }
    }
    Ok(())
}

fn map_pages(pages: &mut [u32], table: &[u32]) -> Result<(), String> {
    for page in pages.iter_mut() {
        let Some(&pool) = table.get(*page as usize) else {
            return Err(format!(
                "pipeline: KV page {page} escapes the working set's {} mapped page(s)",
                table.len()
            ));
        };
        *page = pool;
    }
    Ok(())
}

fn working_set_flat_table(
    stores: &crate::store::registry::Stores,
    ws: crate::store::kv::page_table::WorkingSetId,
) -> Result<Vec<u32>, String> {
    crate::store::registry::with_kv_lock(&stores.kv, "host-pages", |kv| {
        Ok::<Vec<u32>, crate::store::kv::KvStoreError>(
            kv.flat_table(ws)?.iter().map(|page| page.0).collect(),
        )
    })
    .map_err(|error| format!("pipeline: KV page translation: {error}"))
}

/// Hands each lane that states its own page list its working set's whole
/// table: the pages its list names lie in it (`engine::fire::Lane::reserve`).
/// Its own list stays its rows' pages, which the engine stages and the
/// device then resolves over.
fn stamp_lane_reserve(
    req: &mut crate::engine::FireRequest,
    stores: &crate::store::registry::Stores,
    groups: &LaneGroups,
) -> Result<(), String> {
    for (ws, lanes) in groups {
        let table = working_set_flat_table(stores, *ws)?;
        for &lane in lanes {
            if let Some(lane) = req.lanes.get_mut(lane) {
                lane.lane.reserve.clone_from(&table);
                lane.ws = Some(*ws);
            }
        }
    }
    Ok(())
}

/// The named refusal of a beam group's growth that could never fit alone:
/// its set would need `end` pages (its frozen ones and its pool's `named`
/// for this fire's column) and the kv pool holds `capacity` with nothing
/// else resident. Anything smaller waits like any reservation. The search's
/// earlier columns stay valid output (`BEAM-PAGE-LISTS.md` §3).
pub(crate) const BEAM_POOL_OUTGROWN: &str = "outgrows the kv pool";

fn beam_pool_fits(lane: usize, end: u64, named: u64, capacity: u32) -> Result<(), String> {
    if end > u64::from(capacity) {
        return Err(format!(
            "pipeline: lane {lane}'s beams' pool {BEAM_POOL_OUTGROWN}: its next column needs \
             {named} pool page(s), {end} with its frozen ones, and the kv pool holds \
             {capacity} with nothing else resident; the search's earlier columns are its output"
        ));
    }
    Ok(())
}

/// Whether `fwd`'s program states each beam's parent (its lanes hold their
/// own page lists of their set's reserve): decided when it registered.
fn binds_pages<C: FireContext>(ctx: &mut C, fwd: &Resource<ForwardPass>) -> Anyhow<bool> {
    Ok(ctx.resources().get(fwd)?.instance.program.binds_pages)
}

/// The slots lane `lane`'s rows may land at where the device places them,
/// on a set frozen through `folded` that reserved through `reserved`: past
/// its frozen prefix, inside what `reserve` granted. The engine is handed
/// the set's whole page table to read through, and refuses a row its
/// program places outside these ([`::engine::KvDelta::writable`]).
fn device_writable(
    lane: usize,
    folded: u32,
    reserved: u32,
) -> Result<std::ops::Range<u32>, String> {
    if reserved <= folded {
        return Err(format!(
            "pipeline: lane {lane}'s rows land where the device says, inside what \
             `reserve` granted; its working set reserved {reserved} slot(s) and froze \
             {folded}. Reserve the slots it may write"
        ));
    }
    Ok(folded..reserved)
}

/// The pages under `writable`, which the fire makes its set's own (copied
/// where shared) before it runs.
fn writable_pages(writable: &std::ops::Range<u32>, page: u64) -> std::ops::Range<u64> {
    u64::from(writable.start) / page..u64::from(writable.end).div_ceil(page)
}

fn stamp_lane_translation(
    req: &mut crate::engine::FireRequest,
    stores: &crate::store::registry::Stores,
    groups: &LaneGroups,
) -> Result<(), String> {
    for (ws, lanes) in groups {
        let table = working_set_flat_table(stores, *ws)?;
        for &lane in lanes {
            if let Some(lane) = req.lanes.get_mut(lane) {
                lane.translation.clone_from(&table);
                lane.ws = Some(*ws);
            }
        }
    }
    Ok(())
}

/// The window plan of a device-geometry fire: every row lands its rows in
/// the kv pages the fire's write indexes name (each window list's slots
/// over them) and reads its tables through the working set's mapped pages.
/// The device decides where its tokens land, so the windows keep every slot
/// it writes and commit nothing; what each list retires is behind the first
/// page it writes, where every row lands past, so a decode loop that never
/// states its positions still holds its windows and not its whole sequence.
///
/// Window-list path: what an engine that does not state
/// `Capabilities::windowed_state` is served by.
fn device_window_plan(
    stores: &crate::store::registry::Stores,
    ws: crate::store::kv::page_table::WorkingSetId,
    write_indexes: &[u64],
    ids: &[crate::store::rs::RsWorkingSetId],
) -> Result<rs::RsPlan, String> {
    if ids.is_empty() {
        return Ok(rs::RsPlan::Groups(Vec::new()));
    }
    let mapped =
        crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| kv.mapped_len(ws))
            .map_err(|error| error.to_string())?;
    let page = stores.kv_page_size.max(1);
    let index = |at: u64| u32::try_from(at).unwrap_or(u32::MAX);
    let first = write_indexes.iter().copied().min().map(index);
    let last = write_indexes.iter().copied().max().map(index);
    let reach = first.map_or(0, |first| first.saturating_mul(page));
    let writes = match (first, last) {
        (Some(first), Some(last)) => {
            first.saturating_mul(page)..last.saturating_add(1).saturating_mul(page)
        }
        _ => 0..0,
    };
    let extent = u32::try_from(mapped)
        .unwrap_or(u32::MAX)
        .max(last.map_or(0, |last| last.saturating_add(1)));
    Ok(rs::RsPlan::Window(
        (0..ids.len())
            .map(|_| rs::WindowRow {
                writes: writes.clone(),
                reads: extent.saturating_mul(page),
                advance: crate::store::rs::WindowAdvance {
                    positioned: false,
                    settled: 0,
                    reach,
                    fold: None,
                    commit_end: 0,
                    head: None,
                    keep: None,
                },
            })
            .collect(),
    ))
}

fn poison_readers(cells: &BoundCells, reason: &str) {
    for cell in cells {
        let mut c = cell.lock().unwrap();
        if c.role == Some(HostRole::Reader) {
            c.poison(reason);
            if let Some(endpoint) = c.endpoint() {
                waker::WakerTable::global().wake(endpoint.registered().reader_wait_id);
            }
        }
    }
}

/// The moves a fire counted on its rings, given back if the fire never
/// reaches the engine.
struct TicketReservation {
    cells: BoundCells,
    accesses: Vec<(bool, bool)>,
    committed: bool,
}

impl TicketReservation {
    fn new(cells: &BoundCells, accesses: &[(bool, bool)]) -> Self {
        for (cell, &(consume, publish)) in cells.iter().zip(accesses) {
            cell.lock().unwrap().reserve_device_moves(consume, publish);
        }
        Self {
            cells: cells.clone(),
            accesses: accesses[..cells.len().min(accesses.len())].to_vec(),
            committed: false,
        }
    }

    /// The fire carries which ends of which rings it moves, so the step
    /// builder keeps two movers of one end in different steps.
    fn apply_to(&self, request: &mut crate::engine::FireRequest) {
        request.ring_moves = self
            .cells
            .iter()
            .zip(&self.accesses)
            .filter(|(_, (consume, publish))| *consume || *publish)
            .map(|(cell, &(takes, puts))| crate::engine::fire::RingMove {
                channel: cell.lock().unwrap().global_id,
                takes,
                puts,
            })
            .collect();
    }

    fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for TicketReservation {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        for (cell, &(consume, publish)) in self.cells.iter().zip(&self.accesses) {
            cell.lock().unwrap().release_device_moves(consume, publish);
        }
    }
}

fn trace_channel_wait(waiting_on_reader: bool, waiting_on_fire: bool) {
    tracing::debug!(
        waiting_on_reader,
        waiting_on_fire,
        "guest parks on a channel"
    );
}

/// The process is about to wait on its pipeline's oldest fire: its lane
/// tells the scheduler so ([`PendingOp::tell_awaited`]).
pub(crate) fn tell_front_awaited(fires: Option<&PendingFires>) {
    if let Some(fires) = fires
        && let Some(front) = fires.lock().unwrap().front()
    {
        front.tell_awaited();
    }
}

pub(crate) async fn await_channel_progress(
    cell: &Arc<Mutex<crate::pipeline::channel::ChannelCell>>,
    fires: Option<&PendingFires>,
) -> Result<(), String> {
    let wait = cell.lock().unwrap().reader_wait_state();
    let oldest = fires.and_then(|f| f.lock().unwrap().front().map(|op| op.completion_signal()));
    trace_channel_wait(wait.is_some(), oldest.is_some());
    match (wait, oldest) {
        (Some((endpoint, observed_tail)), Some(signal)) => {
            tokio::select! {
                _ = endpoint.wait_for_reader_change(observed_tail) => {}
                _ = signal => {}
            }
            Ok(())
        }
        (Some((endpoint, observed_tail)), None) => endpoint
            .wait_for_reader_change(observed_tail)
            .await
            .map_err(|error| error.to_string()),
        (None, Some(signal)) => {
            signal.await;
            Ok(())
        }
        (None, None) => Err(ChannelError::Empty.to_string()),
    }
}

type Anyhow<T> = anyhow::Result<T>;

struct PassLanes {
    places: Vec<geometry::Place>,
    cursors: Vec<std::sync::Arc<std::sync::Mutex<crate::inferlet::host::working_set::Cursor>>>,
    kv: Vec<KvWorkingSet>,
}

fn pass_lanes<C: FireContext>(ctx: &mut C, fwd: &Resource<ForwardPass>) -> Anyhow<PassLanes> {
    use crate::inferlet::host::working_set::WorkingSet;
    let (lanes, kv_lanes) = {
        let pass = ctx.resources().get(fwd)?;
        (pass.lanes.clone(), pass.kv_lanes.clone())
    };
    let mut kv = Vec::with_capacity(kv_lanes.len());
    for &rep in &kv_lanes {
        let res: Resource<KvWorkingSet> = Resource::new_borrow(rep);
        kv.push(ctx.resources().get(&res)?.clone());
    }
    // What each lane's working set has folded, and whether that is all of
    // it, are its KV working set's.
    let folded: Vec<(u32, bool)> = match kv.first() {
        Some(ws) => {
            let stores = crate::store::registry::get(ws.model, ws.engine);
            crate::store::registry::with_kv_lock(&stores.kv, "host-other", |store| {
                kv.iter()
                    .map(|ws| {
                        let folded = u32::try_from(store.folded(ws.id)?).unwrap_or(u32::MAX);
                        Ok((folded, store.folded_exact(ws.id)?))
                    })
                    .collect::<Result<_, crate::store::kv::KvStoreError>>()
            })?
        }
        None => Vec::new(),
    };
    let mut places = Vec::with_capacity(lanes.len());
    let mut cursors = Vec::with_capacity(lanes.len());
    for (lane, &rep) in lanes.iter().enumerate() {
        let res: Resource<WorkingSet> = Resource::new_borrow(rep);
        let cursor = ctx.resources().get(&res)?.cursor.clone();
        let now = cursor.lock().unwrap().clone();
        let (folded, folded_exact) = folded.get(lane).copied().unwrap_or((0, true));
        places.push(geometry::Place {
            folded,
            folded_exact,
            tail: now.tail,
            shared: lanes.iter().filter(|&&other| other == rep).count() > 1,
            readable: now.readable(),
            readable_known: now.readable_known(),
            set: kv.get(lane).map_or(0, |ws| set_token(ws.id)),
            spans: now.spans,
        });
        cursors.push(cursor);
    }
    Ok(PassLanes {
        places,
        cursors,
        kv,
    })
}

type CursorCell = std::sync::Arc<std::sync::Mutex<crate::inferlet::host::working_set::Cursor>>;

/// The token a working set's lanes name it by to the engine
/// ([`::engine::KvDelta::set`]): never 0, which names no set.
fn set_token(ws: crate::store::kv::page_table::WorkingSetId) -> u64 {
    ws.to_bits().wrapping_add(1)
}

/// A fire on its way moves its working set's cursor, as soon as it is
/// submitted: a pass that stops short leaves the cursor at the fires that
/// went. `tail` is where its rows end, `None` where the device places them
/// past rows the host cannot place; `spans` the first slot it writes and
/// the lags of the media runs it carries. What it folds is not the
/// cursor's: that is its KV working set's.
fn advance_cursor(
    cell: &CursorCell,
    tail: Option<u32>,
    spans: Option<(u32, Vec<(u32, u32, u32)>)>,
) {
    let mut cursor = cell.lock().unwrap();
    cursor.tail = tail;
    if let Some(end) = tail {
        cursor.reserved = cursor.reserved.max(end);
        cursor.written = cursor.written.max(end);
    }
    if let Some((start, spans)) = spans {
        cursor.rewrite(start, spans);
    }
}

/// A fire just submitted lands host-placed rows `start..end` on the set:
/// a fire staged after it reads them. Should it fail, they may never have
/// landed, and nothing from `start` on counts as written.
fn host_wrote(
    cell: &CursorCell,
    fire: &crate::engine::completion::WorkItemCompletion,
    start: u32,
    end: u32,
) {
    cell.lock().unwrap().wrote(end);
    let cell = std::sync::Arc::downgrade(cell);
    fire.on_settled_landed(move |ok| {
        if let (false, Some(cell)) = (ok, cell.upgrade()) {
            cell.lock().unwrap().unwrote(start);
        }
    });
}

/// The lags of the media runs `lanes` carry that start in `slots`.
fn lagging_spans(
    landings: &[geometry::Landing],
    matched: &[crate::pipeline::media::MatchedRun],
    lanes: &[usize],
    slots: &std::ops::Range<u64>,
) -> Vec<(u32, u32, u32)> {
    matched
        .iter()
        .filter(|run| lanes.contains(&(run.lane as usize)))
        .map(|run| {
            let at = landings[run.lane as usize].start + run.anchor;
            let lag = run.span.token_count.saturating_sub(run.span.position_span);
            (at, at + run.rows, lag)
        })
        .filter(|&(at, _, lag)| lag > 0 && slots.contains(&u64::from(at)))
        .collect()
}

/// The fire is submitted: each working set it folds by a count placed when
/// it settles says so, its folded length a lower bound until the fire
/// settles ([`settle_fold`]) or, dropped, the engine is done with it. It is
/// not forked or exported until then.
fn unsettle(
    stores: &crate::store::registry::Stores,
    completion: &crate::engine::WorkItemCompletion,
    parts: &mut [FireWs],
) {
    let mut settling = parts
        .iter_mut()
        .filter_map(|part| match &mut part.fold {
            Some(FoldAt {
                by: FoldBy::Settled { unplaced, .. },
                ..
            }) => Some((part.ws, unplaced)),
            _ => None,
        })
        .peekable();
    if settling.peek().is_none() {
        return;
    }
    crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
        for (ws, unplaced) in settling {
            match kv.unsettle(ws) {
                Ok(()) => {
                    *unplaced = Some(Unplaced {
                        ws,
                        submitted: Some((stores.kv.clone(), completion.clone())),
                    });
                }
                Err(error) => {
                    tracing::debug!(%error, "pipeline: a fold placed at settle on a set gone");
                }
            }
        }
    });
}

/// The fold of a run-ahead fire (its tokens the device's) on sets with a
/// recurrent state or windows: each group's no further than where the rows
/// its program has accepted end ([`Cursor::accepted`], bounded by the
/// run-ahead window), so the rows past them stay buffered and a stop takes
/// them back. (A decode's buffered row costs it about what one fold's
/// state write does, CUDA's gated delta on Qwen3.5-0.8B at 64 lanes: a fold
/// takes every accepted row at once.) A fold the device counts, or over a
/// folded length the host does not know, stays as stated.
///
/// [`Cursor::accepted`]: crate::inferlet::host::working_set::Cursor::accepted
fn lagged_fold(
    fold: &geometry::Fold,
    lanes: &PassLanes,
    groups: &LaneGroups,
    landings: &[geometry::Landing],
) -> geometry::Fold {
    let count = landings.len();
    if (0..count).any(|lane| fold.of(lane).is_none()) {
        return fold.clone();
    }
    let bound = crate::scheduler::run_ahead_window();
    let mut counts: Vec<u32> = (0..count).map(|lane| fold.of(lane).unwrap_or(0)).collect();
    for (_, group) in groups {
        let first = group[0];
        let place = &lanes.places[first];
        if !place.folded_exact {
            continue;
        }
        let accepted = lanes.cursors[first]
            .lock()
            .unwrap()
            .accepted(landings[first].start, bound);
        let room = accepted.saturating_sub(place.folded);
        for &lane in group {
            counts[lane] = counts[lane].min(room);
        }
    }
    geometry::Fold::Host(counts)
}

/// What a host-placed fire folds, `fold` as its pass states it, on sets
/// that hold unfolded rows (`lags`: a recurrent state or windows). A
/// run-ahead fire (`run_ahead`, its tokens the device's) folds no further
/// than its program has accepted ([`lagged_fold`]), unless its pass stated
/// its rows final (CONTRACT-BC 25): then, as a fire whose tokens the host
/// places, it folds what it states at fire and accepts every row its sets
/// hold, so its next fire carries nothing in.
fn pass_fold(
    fold: geometry::Fold,
    lags: bool,
    run_ahead: bool,
    final_rows: bool,
    lanes: &PassLanes,
    groups: &LaneGroups,
    landings: &[geometry::Landing],
) -> geometry::Fold {
    if !lags {
        return fold;
    }
    if run_ahead && !final_rows {
        return lagged_fold(&fold, lanes, groups, landings);
    }
    for cursor in &lanes.cursors {
        cursor.lock().unwrap().accept_all();
    }
    fold
}

/// The outputs a fire just reserved its moves for publishes to the host:
/// the program accepts the fire's rows as it takes them.
fn host_reads(
    cells: &BoundCells,
    accesses: &[(bool, bool)],
) -> Vec<crate::inferlet::host::working_set::HostRead> {
    cells
        .iter()
        .zip(accesses)
        .filter(|(_, (_, publish))| *publish)
        .filter_map(|(cell, _)| {
            let c = cell.lock().unwrap();
            if c.role != Some(HostRole::Reader) {
                return None;
            }
            Some(crate::inferlet::host::working_set::HostRead {
                cell: Arc::downgrade(cell),
                index: c.last_published()?,
            })
        })
        .collect()
}

/// What a fire whose rows the host places folds on `group`'s working set.
/// A count the host knows over a folded length it knows is a slot, frozen
/// as the fire is prepared; anything else is placed when the pass's last
/// fire settles. A fold reaches no further than where the group's first
/// lane lands its rows (`landings`), as its recurrent state's does
/// ([`host_rs_plan`]), nor than `end`, where this fire's rows end.
#[allow(
    clippy::too_many_arguments,
    reason = "the fold, where it ends, and the request whose lanes report it"
)]
fn fold_at(
    lanes: &PassLanes,
    group: &[usize],
    landings: &[geometry::Landing],
    fold: &geometry::Fold,
    end: u64,
    last: bool,
    rs: Option<crate::store::rs::RsWorkingSetId>,
    req: &mut crate::engine::FireRequest,
    tags: &mut Vec<ReportTag>,
) -> Option<FoldBy> {
    if matches!(fold, geometry::Fold::None) {
        return None;
    }
    let first = *group.first()?;
    let place = &lanes.places[first];
    let reach = u64::from(landings[first].end());
    let count = fold.of(first);
    if count == Some(0) {
        return None;
    }
    if place.folded_exact
        && let Some(count) = count
    {
        return Some(FoldBy::To(
            u64::from(place.folded)
                .saturating_add(u64::from(count))
                .min(reach)
                .min(end),
        ));
    }
    if !last {
        return None;
    }
    let (tags, rs) = match count {
        Some(_) => (Vec::new(), None),
        None => (
            vec![report_on(req, first, tags)],
            rs.map(RsDeviceFold::planned),
        ),
    };
    Some(FoldBy::Settled {
        count,
        tags,
        reach: reach.min(end),
        written: false,
        rs,
        unplaced: None,
    })
}

/// A piece states the boundary its prepare found, which a cut pass that
/// folds as it goes moves on from the one the pass was staged on: the
/// columns of the lane's mask between the two are now committed slots, read
/// as every committed slot is, and the mask starts past them.
fn restate_boundary(lane: &mut ::engine::Lane, folded: u64) {
    let folded = u32::try_from(folded).unwrap_or(u32::MAX).min(lane.kv.held);
    let moved = u64::from(folded.saturating_sub(lane.kv.folded));
    if moved > 0 {
        match &mut lane.mask {
            Some(::engine::Masking::Extent(mask)) => *mask = mask.after(moved),
            Some(::engine::Masking::Rows(masks)) => {
                for mask in masks.iter_mut() {
                    *mask = mask.after(moved);
                }
            }
            None => {}
        }
    }
    lane.kv.folded = lane.kv.folded.max(folded);
}

/// The last step of a piece's prepare on a working set it writes from
/// `start`: its log begins there, and what it folds by a count the host
/// knows is frozen through `freeze`. A piece that starts inside what the
/// set has frozen begins no log, and freezes nothing.
fn begin_piece(
    kv: &mut crate::store::kv::KvStore,
    ws: crate::store::kv::page_table::WorkingSetId,
    start: u64,
    freeze: Option<u64>,
) -> Result<Option<u64>, crate::store::kv::KvStoreError> {
    if start < kv.folded(ws)? {
        return Ok(None);
    }
    let seq = kv.log_begin(ws, start)?;
    if let Some(to) = freeze {
        kv.freeze(ws, to)?;
    }
    Ok(Some(seq))
}

/// What a pass whose rows the device places folds on a working set, as far
/// as the host can say before the fire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeviceFold {
    None,
    /// By a count the device computes.
    Device,
    /// This many slots past the folded length.
    Count(u32),
    /// By a count that runs past the lane's rows, from rows the host cannot
    /// place: how far it reaches is where the engine says the rows ended.
    PastRows(u32),
}

/// Whether the rows the device places on a working set start at its folded
/// length: one lane writes it, it holds no unfolded row, and the host knows
/// it.
fn appends_folded(lanes: &PassLanes, group: &[usize]) -> bool {
    match group {
        [lane] => {
            let place = &lanes.places[*lane];
            place.folded_exact && place.tail == Some(place.folded)
        }
        _ => false,
    }
}

/// The engine folds through `min(folded + count, first + rows)`, `first`
/// being where the device lands the lane's `rows` rows, at or past the
/// folded length. A count within the rows stops short of them wherever
/// they land; one past them stops at their end, which the host knows only
/// when they start at the folded length. `room` is what the set reserved
/// past it.
fn device_fold(count: Option<u32>, rows: u32, room: u32, appends: bool) -> DeviceFold {
    match count {
        None => DeviceFold::Device,
        Some(0) => DeviceFold::None,
        Some(count) if count <= rows || appends => match count.min(rows).min(room) {
            0 => DeviceFold::None,
            count => DeviceFold::Count(count),
        },
        Some(count) => DeviceFold::PastRows(count),
    }
}

/// The refusal of [`DeviceFold::PastRows`] where nothing tells the host
/// where the rows ended: a fire that reports no rows, or a recurrent state,
/// whose fold the host states is final as stated.
fn past_rows_refusal(lane: usize, rows: u32) -> String {
    format!(
        "pipeline: lane {lane} folds past the {rows} row(s) it writes, which land where the \
         device says on a working set that may hold unfolded rows before them; that fold \
         ends where the rows do, which the device alone knows here. Fold no more than the \
         lane writes, or fold what the set holds before this pass"
    )
}

/// [`fold_at`] for rows the device places: `stated` is [`device_fold`] for
/// the working set's first lane, `reported` whether the fire reports the
/// rows it wrote.
fn device_fold_at(
    lanes: &PassLanes,
    first: usize,
    stated: DeviceFold,
    reported: bool,
    rs: Option<crate::store::rs::RsWorkingSetId>,
    req: &mut crate::engine::FireRequest,
    tags: &mut Vec<ReportTag>,
) -> Option<FoldBy> {
    let place = &lanes.places[first];
    let reach = u64::from(lanes.cursors[first].lock().unwrap().reserved);
    let settled = |count, tags, written, rs| FoldBy::Settled {
        count,
        tags,
        reach,
        written,
        rs,
        unplaced: None,
    };
    match stated {
        DeviceFold::None => None,
        DeviceFold::Device => {
            let tag = report_on(req, first, tags);
            Some(settled(
                None,
                vec![tag],
                false,
                rs.map(RsDeviceFold::planned),
            ))
        }
        DeviceFold::Count(count) if place.folded_exact => Some(FoldBy::To(
            u64::from(place.folded).saturating_add(u64::from(count)),
        )),
        DeviceFold::Count(count) => Some(settled(Some(count), Vec::new(), false, None)),
        DeviceFold::PastRows(count) => {
            reported.then(|| settled(Some(count), Vec::new(), true, None))
        }
    }
}

/// The working sets a piece writes: those with a lane that has rows in it. A set whose lanes
/// all ride only in a later piece (a decode lane's one row beside a prompt cut into pieces) is
/// not written here; its span would be empty, which reads as a write at slot 0, inside what the
/// set already froze, so it takes no lease, transaction or log in this piece.
fn written_groups<W>(piece_groups: &[(W, Vec<usize>)]) -> Vec<usize> {
    piece_groups
        .iter()
        .enumerate()
        .filter(|(_, (_, lanes))| !lanes.is_empty())
        .map(|(group, _)| group)
        .collect()
}

fn written_span_of(
    req: &crate::engine::FireRequest,
    lanes: &[usize],
    page: u32,
) -> (std::ops::Range<u64>, std::ops::Range<u64>) {
    let page = u64::from(page.max(1));
    let rows = lanes
        .iter()
        .filter_map(|&lane| req.lanes.get(lane))
        .filter(|lane| !lane.tokens.is_empty());
    let start = rows
        .clone()
        .map(|lane| u64::from(lane.kv.held))
        .min()
        .unwrap_or(0);
    let end = rows
        .map(|lane| u64::from(lane.kv.held) + lane.tokens.len() as u64)
        .max()
        .unwrap_or(start);
    (start..end, start / page..end.div_ceil(page))
}

async fn open_pipeline<C: FireContext>(
    ctx: &mut C,
    this: &Resource<Pipeline>,
    fwd: &Resource<ForwardPass>,
) -> Anyhow<Result<(PendingFires, PipelineFailure, crate::store::PipelineScope), String>> {
    let (fires, failure, scope) = {
        let pipeline = ctx.resources().get(this)?;
        if pipeline.scope.is_closed() {
            return Ok(Err("pipeline: pipeline is closed".to_string()));
        }
        (
            pipeline.fires.clone(),
            pipeline.failure.clone(),
            pipeline.scope.clone(),
        )
    };
    drain_settled(ctx, Some(&fires)).await?;
    if let Some(error) = pipeline_failed(&failure) {
        return Ok(Err(error));
    }
    if let Err(error) = wire_channels_to_pipeline(ctx, fwd, &fires)? {
        return Ok(Err(error));
    }
    Ok(Ok((fires, failure, scope)))
}

fn stamp_lane_facts<C: FireContext>(
    ctx: &mut C,
    fwd: &Resource<ForwardPass>,
    req: &mut crate::engine::FireRequest,
) -> Anyhow<()> {
    let pass = ctx.resources().get(fwd)?;
    crate::pipeline::instance::LaneFacts::stamp(&pass.facts, req);
    Ok(())
}

/// A lane binding no span carries no run of the model's placeholder.
fn scan_lane_media(
    lane_tokens: &[&[u32]],
    spans: &[Vec<Arc<media::front::EncodedSpan>>],
) -> Result<Vec<crate::pipeline::media::MatchedRun>, crate::pipeline::media::Refusal> {
    let mut matched = Vec::new();
    for (lane, tokens) in lane_tokens.iter().enumerate() {
        match spans.get(lane).filter(|spans| !spans.is_empty()) {
            Some(spans) => {
                for mut run in crate::pipeline::media::scan(&[tokens], spans)? {
                    run.lane = lane as u32;
                    matched.push(run);
                }
            }
            None => {
                crate::pipeline::media::refuse_orphan_runs(&[tokens], crate::model::media_pad())?
            }
        }
    }
    Ok(matched)
}

fn advance_shadow<C: FireContext>(ctx: &mut C, fwd: &Resource<ForwardPass>) -> Anyhow<()> {
    let pass = ctx
        .resources()
        .get_mut(fwd)?
        .bound_mut()
        .map_err(anyhow::Error::msg)?;
    pass.host_shadow
        .advance(&pass.instance.program.bound, &pass.cells);
    Ok(())
}

pub async fn submit_pass_stamped<C: FireContext>(
    ctx: &mut C,
    this: Resource<Pipeline>,
    fwd: Resource<ForwardPass>,
    frame: Option<crate::scheduler::FrameStamp>,
    ahead: u32,
) -> Anyhow<Result<(), String>> {
    let submit_probe = crate::scheduler::probe::host_submit();
    let submit_clock = crate::scheduler::probe::ProbeClock::start();
    if let Some(refusal) = crate::policy::stopped(ctx.process_id()) {
        return Ok(Err(refusal));
    }
    {
        let placed_by_device = {
            let pass = ctx.resources().get(&fwd)?;
            pass.devgeo.is_some() || pass.float.is_some()
        };
        if placed_by_device {
            let pipeline = this.rep();
            let final_rows = ctx.resources().get(&fwd)?.bindings.final_rows;
            let submitted = if ctx.resources().get(&fwd)?.devgeo.is_some() {
                fire_device_geometry(ctx, this, fwd, frame).await?
            } else {
                float::fire_float_lane(ctx, this, fwd, frame).await?
            };
            if submitted.is_ok() {
                record_writes(ctx, pipeline, None, final_rows)?;
            }
            return Ok(submitted);
        }
        let began = crate::scheduler::probe::ProbeClock::start();
        let opened = open_pipeline(ctx, &this, &fwd).await?;
        crate::probe_fire_record!(submit_probe.drain_settled_us, began.elapsed());
        let (pipe_fires, pipeline_failure, pipeline_scope) = match opened {
            Ok(opened) => opened,
            Err(error) => return Ok(Err(error)),
        };
        let (
            geometry,
            cells,
            rs_reps,
            fwd_rep,
            instance_id,
            scheduler,
            attn_mask,
            accesses,
            decode_envelope,
        ) = {
            let p = ctx.resources().get_mut(&fwd)?;
            if let Some(e) = &p.failed {
                return Ok(Err(format!(
                    "pipeline: forward-pass failed by an earlier fire: {e}"
                )));
            }
            let device_resolved = p.decode_envelope.unwrap_or(PortMask::NONE);
            let geometry_clock = crate::scheduler::probe::ProbeClock::start();
            let (geometry, attn_mask) = {
                let bound = &p.instance.program.bound;
                let (shadow, shadow_cells) = (&p.host_shadow, &p.cells);
                let mut known = |chan: u32| shadow.fire_value(bound, shadow_cells, chan);
                match geometry::map_geometry_evaluated_with(bound, &mut known, device_resolved) {
                    Ok((geometry, evaluated)) => {
                        if device_resolved.is_empty() && geometry.token_ids.contains(&u32::MAX) {
                            return Ok(Err(
                                "pipeline: fire geometry: in-band -1 skip tokens require a \
                                 device-resolved geometry class; this fire resolved on the \
                                 host wire"
                                    .to_string(),
                            ));
                        }
                        // A mask the program computes on the device is the
                        // engine's to resolve, even on a fire whose seeds let
                        // the host evaluate it: one source a port.
                        let attn_mask = match if device_resolved.contains(Port::AttnMask) {
                            Ok(geometry::FireAttnMask::Device)
                        } else {
                            geometry::lower_attn_mask_evaluated(
                                &bound.container,
                                &geometry.qo_indptr,
                                &evaluated,
                            )
                        } {
                            Ok(mask) => mask,
                            Err(error) => {
                                return Ok(Err(format!("pipeline: fire attention mask: {error}")));
                            }
                        };
                        (geometry, attn_mask)
                    }
                    Err(error) => {
                        return Ok(Err(format!("pipeline: fire geometry: {error}")));
                    }
                }
            };
            crate::probe_fire_record!(submit_probe.geometry_us, geometry_clock.elapsed());
            if let Some(fires) = &p.host_geometry {
                fires.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            let accesses = p.instance.program.channel_accesses.clone();
            (
                geometry,
                p.cells.clone(),
                p.rs_ws.clone(),
                fwd.rep(),
                p.bound_instance.instance_id,
                p.scheduler.clone(),
                attn_mask,
                accesses,
                p.decode_envelope,
            )
        };
        // A recurrent state a device-counted fold left unsettled is staged
        // on the frozen length as a lower bound, and the engine replays
        // from where its record stands: the fire goes ahead of the settle.
        // It could not wait for it: a frame's fires are dispatched once all
        // of them have arrived, this one among them.
        let pass_lanes = pass_lanes(ctx, &fwd)?;
        let ws = pass_lanes.kv[0].clone();
        let stores = crate::store::registry::get(ws.model, ws.engine);
        let page_size = stores.kv_page_size;
        let (lanes, landings) = match geometry.lanes(&pass_lanes.places, page_size) {
            Ok(placed) => placed,
            Err(error) => return Ok(Err(format!("pipeline: {error}"))),
        };
        let mut req = crate::engine::FireRequest {
            lanes,
            ..crate::engine::FireRequest::default()
        };
        let fold = geometry.fold.clone();
        stamp_lane_facts(ctx, &fwd, &mut req)?;
        req.boundary_program = true;
        let switched = Switched::of(ctx.resources().get(&fwd)?);
        let lanes_of = every_lane(&req);
        req.geometry = if decode_envelope.is_some() {
            GeometryClass::DecodeEnvelope
        } else {
            GeometryClass::Host
        };
        tracing::debug!(
            geometry = ?req.geometry,
            lanes = req.lanes.len(),
            qo_indptr = ?geometry.qo_indptr,
            "fire geometry resolved on host path"
        );
        req.single_token_mode = req.lanes.iter().all(|lane| lane.tokens.len() == 1);
        let media_spans: Vec<Vec<Arc<media::front::EncodedSpan>>> = ctx
            .resources()
            .get(&fwd)?
            .bindings
            .lanes
            .iter()
            .map(|lane| lane.media.clone())
            .collect();
        let carries_media = media_spans.iter().any(|spans| !spans.is_empty());
        let matched = {
            let lane_tokens: Vec<&[u32]> = req
                .lanes
                .iter()
                .map(|lane| lane.tokens.as_slice())
                .collect();
            match scan_lane_media(&lane_tokens, &media_spans) {
                Ok(matched) => matched,
                Err(refusal) => return Ok(Err(format!("pipeline: {refusal}"))),
            }
        };
        let fire_wide_mask = matches!(attn_mask, geometry::FireAttnMask::Device);
        if let Err(error) = attn_mask.apply_to(&mut req) {
            return Ok(Err(format!("pipeline: fire attention mask: {error}")));
        }
        if !matched.is_empty() {
            let lane_rows: Vec<u32> = req
                .lanes
                .iter()
                .map(|lane| u32::try_from(lane.tokens.len()).unwrap_or(u32::MAX))
                .collect();
            let lane_base: Vec<u32> = req
                .lanes
                .iter()
                .map(|lane| lane.positions.first().copied().unwrap_or(lane.kv.held))
                .collect();
            req.media = crate::pipeline::media::lane_media(&matched, &lane_rows, &lane_base);
        }
        stamp_lane_words(&mut req, &switched, &lanes_of, fire_wide_mask, None);
        let wide = reads_each_other(&req);
        let kv_clock = crate::scheduler::probe::ProbeClock::start();
        let groups = lane_groups(&pass_lanes.kv.iter().map(|ws| ws.id).collect::<Vec<_>>());
        let group_sets: Vec<KvWorkingSet> = groups
            .iter()
            .map(|(_, lanes)| pass_lanes.kv[lanes[0]].clone())
            .collect();
        // Lanes that state their own page lists (beam search) write pages of
        // their sets' reserve the device names: every page from the frozen
        // full pages to what `reserve` granted is backed, and the engine is
        // handed each set's whole table to resolve them in.
        // Each beam group's pool end (a slot), which its rs buffer is backed
        // to as well (`BEAM-PAGE-LISTS.md` §11).
        let mut beam_ends: Vec<u32> = Vec::new();
        // Each beam group's common slot, as its last harvested steps left it
        // (§11.3): one pass, one group, one instance.
        let beam_common = beam_common_of(instance_id);
        let mut beam_commons: Vec<u32> = Vec::new();
        let paged_reserve: Option<Vec<Vec<u64>>> = if binds_pages(ctx, &fwd)? {
            let page = u64::from(page_size.max(1));
            let mut reserves = Vec::with_capacity(groups.len());
            for (_, lanes) in &groups {
                let lane = lanes[0];
                let folded = pass_lanes.places[lane].folded;
                // The beams' pool grows a column at a time (BEAM-PAGE-LISTS.md
                // §3): before the fire that opens a column the set reserves
                // what its column needs, waiting like any reservation. The
                // step hands out the lowest free pages, so every page named or
                // pending lies below `B (c + 2) + B F + 1`, `c` this fire's
                // column and `F` the engine's frames in flight.
                let f0 = u64::from(folded) / page;
                let last = lanes
                    .iter()
                    .map(|&lane| u64::from(landings[lane].start) + u64::from(landings[lane].rows.max(1)) - 1)
                    .max()
                    .unwrap_or(u64::from(folded));
                let beams = lanes.len() as u64;
                let frames = crate::scheduler::configured_dispatch_depth().max(1) as u64;
                let column = (last / page).saturating_sub(f0);
                let named = beams * (column + 2) + beams * frames + 1;
                let end = f0 + named;
                let end_slot = u32::try_from(end * page).unwrap_or(u32::MAX);
                let capacity = crate::store::registry::with_kv_lock(
                    &stores.kv,
                    "host-other",
                    |kv| kv.capacity_pages(),
                );
                if let Err(refusal) = beam_pool_fits(lane, end, named, capacity) {
                    return Ok(Err(refusal));
                }
                if let Err(refusal) = crate::inferlet::host::working_set::grow_pages(
                    &pass_lanes.kv[lane],
                    end_slot,
                ) {
                    return Ok(Err(format!(
                        "pipeline: lane {lane}'s beams' pool cannot grow to its column's {named} \
                         page(s): {refusal}"
                    )));
                }
                {
                    let mut cursor = pass_lanes.cursors[lane].lock().unwrap();
                    cursor.reserved = cursor.reserved.max(end_slot);
                }
                beam_ends.push(end_slot);
                beam_commons.push(beam_common);
                // Only the pages the column adds are backed here: the pool's
                // pages are the group's alone and the device keeps what they
                // hold, so a page an earlier fire still in flight writes is
                // not copied before this one writes it.
                let mapped = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
                    kv.mapped_len(pass_lanes.kv[lane].id)
                })
                .map_err(|error| anyhow::anyhow!("pipeline: KV rows: {error}"))?;
                reserves.push((mapped.max(f0)..end).collect());
            }
            Some(reserves)
        } else {
            None
        };
        if let Err(refusal) = crate::inferlet::process::said(
            "seat the lanes",
            ctx.process_id(),
            seat_lane_slots(&mut req, &stores, &groups),
        )
        .await
        {
            return Ok(Err(refusal));
        }
        for (group, (_, lanes)) in groups.iter().enumerate() {
            let end = lanes
                .iter()
                .map(|&lane| landings[lane].end())
                .max()
                .unwrap_or(0);
            if let Err(error) =
                crate::inferlet::host::working_set::grow_pages(&group_sets[group], end)
            {
                return Ok(Err(format!("pipeline: KV working set: {error}")));
            }
        }
        let model = ws.model;
        let engine = ws.engine;
        let pid = ctx.process_id();
        let quorum_pipeline_id = pipeline_scope.scheduler_id();
        for set in &group_sets {
            if let Err(refusal) = set.claim_pipeline_scope(&pipeline_scope) {
                return Ok(Err(format!("pipeline: KV working set {refusal}")));
            }
        }
        let rs_ws_ids =
            match bound_rs_working_sets(ctx, &ws, &rs_reps, &req.qo_indptr(), &pipeline_scope)? {
                Ok(ids) => ids,
                Err(error) => return Ok(Err(error)),
            };
        let windowed = crate::model::model().rs_caps().window_tokens > 0;
        // On a set with a recurrent state or windows a run-ahead fire folds
        // no further than the rows the program has accepted, so a stop can
        // take back the rest; a fire whose tokens the host places is
        // accepted as it is submitted, and everything before it with it.
        // A state that buffers no row folds every row as before.
        // A pass whose program stated its rows final (CONTRACT-BC 25) takes
        // no lag: its rows are never taken back, so it folds at fire what
        // it states, as a host-placed fire does, and its next fire carries
        // nothing in.
        let lags = !rs_ws_ids.is_empty() && stores.rs.lock().unwrap().holds_unfolded();
        let final_rows = ctx.resources().get(&fwd)?.bindings.final_rows;
        let fold = pass_fold(
            fold,
            lags,
            decode_envelope.is_some(),
            final_rows,
            &pass_lanes,
            &groups,
            &landings,
        );
        let window_state = crate::model::model().rs_caps().window_state;
        let window_only = window_only(&stores, window_state);
        // A long host pass runs as several fires; all but the last run the
        // model's forward alone. A recurrent state folding all its rows, or
        // none, folds each fire's rows in turn; a windowed-only load's
        // other folds (a beam group's lagged common-ancestor fold) land with
        // the pass's last piece, its earlier pieces' rows buffered
        // (`piece_fold`).
        let folds_all = rs_ws_ids.is_empty()
            || [0, u32::MAX]
                .into_iter()
                .any(|count| (0..req.lanes.len()).all(|lane| fold.of(lane) == Some(count)));
        let adapted = lora_program_of(ctx, &fwd)?;
        // The program runs on a cut pass's last fire, after the first fires' forwards.
        let prologue_moves =
            container_prologue_moves(&ctx.resources().get(&fwd)?.instance.program.bound.container);
        let plain = switched.plain();
        let keyable = decode_envelope.is_none()
            && !wide
            && plain
            && groups.len() == 1
            && !adapted
            && !fire_wide_mask;
        // A pass that may start from a prefix the cache holds may be cut, a
        // window's too: each piece lands its rows in the window lists and
        // trims behind them as a fire of its own.
        let cuttable = decode_envelope.is_none()
            && !wide
            && plain
            && !prologue_moves
            && (!adapted || crate::model::model().eta_caps().adapter_attachments)
            && !fire_wide_mask
            && (folds_all || window_only);
        let mut full = req;
        let keyed = if keyable { keyed_rows(&full) } else { 0 };
        // Keyed under the set's domain: its scope's, resolved again now,
        // where it asked to share (`working-set.share`); else its own.
        let mut shares = false;
        let mut key_domain = None;
        let mut pass_slots = (keyed > 0)
            .then(|| {
                let lane = &full.lanes[0];
                let start = u64::from(lane.kv.held);
                let root = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
                    let (domain, shared) = kv.pass_domain(ws.id)?;
                    kv.chain_root(ws.id, start, page_size)
                        .map(|root| root.map(|prev| (domain, shared, prev)))
                });
                let (domain, shared, prev) = root.ok().flatten()?;
                shares = shared;
                key_domain = Some(domain);
                Some(crate::store::kv::hash::chain_slots(
                    &domain,
                    prev,
                    start,
                    &lane.tokens[..keyed as usize],
                ))
            })
            .flatten();
        // A window's pages are checkpointed like a folded state: a fork
        // the prefix cache keeps, sharing them by refcount (CONTRACT-BC 19).
        let state = match rs_ws_ids.as_slice() {
            [rs] => Some(*rs),
            _ => None,
        };
        // Only a set that shares maps, waits on, fetches or resumes from
        // what others computed.
        let attach = if !shares || !cuttable || !keyable || !plain_rows(&full) {
            Attach::No
        } else if rs_ws_ids.is_empty() {
            Attach::Kv
        } else if let Some(rs) = state {
            Attach::Checkpoint(rs)
        } else {
            Attach::No
        };
        // What this pass claims to compute for others to wait on, held until
        // its last piece settles.
        let mut coalesced: Option<crate::pipeline::coalesce::ClaimGuard> = None;
        // Rows no worker here holds may be prefilled on a peer, whose pages
        // land in the host tier for the promote below to find. Media stays
        // here.
        if matches!(attach, Attach::Kv)
            && !carries_media
            && let Some(slots) = &pass_slots
            && let Some(domain) = key_domain
        {
            let lane = &full.lanes[0];
            // Pages another pass is computing now are waited for, not
            // computed again. Waiting on that pass's fire, the process
            // submits nothing: it is parked while it waits, unblocked as the
            // wait ends (greedy decodes of one prompt arriving together held
            // the claimant's first frame for the waiters' lanes for good).
            coalesced = crate::inferlet::process::said(
                "plan: pages another pass computes",
                pid,
                crate::pipeline::coalesce::before_pass(
                    &stores,
                    ws.id,
                    u64::from(lane.kv.held),
                    slots,
                    crate::inferlet::process::parks_here,
                ),
            )
            .await;
            crate::offload::prefill::before_pass(
                pid,
                &stores,
                engine,
                domain,
                ws.id,
                u64::from(lane.kv.held),
                &lane.tokens,
                slots,
            )
            .await;
            // A pass that computes a run a peer streams taps its fires.
            crate::offload::stream::bind(pid, &stores, ws.id, u64::from(lane.kv.held), slots);
        }
        // Pages below the device come back up first; the run stays held
        // until it is mapped in just below.
        let promoted = match &pass_slots {
            Some(slots) if !matches!(attach, Attach::No) => promote(
                ctx,
                &stores,
                &ws,
                &full,
                slots,
                quorum_pipeline_id,
            )
            .await
            .unwrap_or_else(|error| {
                tracing::debug!(%error, "kv supply: bringing sealed pages back to the device");
                None
            }),
            _ => None,
        };
        let Supplied {
            rows: supplied,
            keys,
        } = match &pass_slots {
            Some(slots) => {
                let folds = fold.of(0).map_or(0, |count| {
                    u64::from(count).min(full.lanes[0].tokens.len() as u64)
                });
                supply(&stores, ws.id, &full, slots, attach, folds, page_size)
            }
            None => Supplied::default(),
        };
        // A state resumed is folded through what was mapped in, rows the
        // set holds from here on whatever becomes of the rest of the pass.
        if supplied > 0 && matches!(attach, Attach::Checkpoint(_)) {
            let mut cursor = pass_lanes.cursors[0].lock().unwrap();
            let end = full.lanes[0].kv.held.saturating_add(supplied);
            cursor.tail = Some(end);
            cursor.reserved = cursor.reserved.max(end);
        }
        // Of the rows mapped in, those on pages that came up from below.
        if let Some(promoted) = &promoted {
            let below = (promoted.pages * u64::from(page_size)).min(u64::from(supplied));
            crate::store::CACHED_FROM_BELOW_TOKENS
                .fetch_add(below, std::sync::atomic::Ordering::Relaxed);
        }
        drop(promoted);
        // What the fire writes and reads out, for the policies' prompt and
        // generated counts: rows mapped in from the cache are rows of the
        // prompt too.
        crate::policy::report(crate::policy::Event::Fired(
            pid,
            full.lanes.iter().map(|lane| lane.tokens.len() as u64).sum(),
            full.lanes
                .iter()
                .map(|lane| match &lane.readout {
                    ::engine::Readout::None => 0,
                    ::engine::Readout::Last => 1,
                    ::engine::Readout::Rows(rows) => rows.len() as u64,
                })
                .sum(),
        ));
        if pass_slots.is_some() && shares {
            // It looked, so what it found is measured, nothing included.
            ctx.note_cached(u64::from(supplied));
            crate::policy::report(crate::policy::Event::Cached(pid, u64::from(supplied)));
        }
        if supplied > 0 {
            // What was mapped in is skipped: the pass runs from there. The
            // rows mapped in are the set's to read: a prefix it shares.
            let mapped = full.lanes[0].kv.held.saturating_add(supplied);
            pass_lanes.cursors[0].lock().unwrap().wrote(mapped);
            let rows = u32::try_from(full.lanes[0].tokens.len()).unwrap_or(u32::MAX);
            full = rows_of_lanes(&full, &[Some((supplied, rows))], true, page_size);
            for lane in &mut full.lanes {
                lane.kv.readable = lane.kv.readable.max(mapped);
            }
            stamp_lane_words(&mut full, &switched, &lanes_of, false, None);
            pass_slots = pass_slots.map(|slots| slots[supplied as usize..].to_vec());
        }
        // Where the pass keeps its state: the checkpoint rows it reaches,
        // floored to whole pages it keys, each the end of a fire of its own.
        let held = u64::from(full.lanes[0].kv.held);
        // The rows it computes from here, sealed for its claims' waiters
        // once its fires land.
        let mut sealing = coalesced.take().map(|claims| SealOnLanding {
            claims,
            stores: stores.clone(),
            ws: ws.id,
            start: held,
            slots: pass_slots.clone().unwrap_or_default(),
            page_size,
            failure: pipeline_failure.clone(),
        });
        let reach = held + full.lanes[0].tokens.len() as u64;
        // Only on a set that shares: a later pass resumes from one only
        // through its scope. Asking on one that never did is refused at the
        // ask; one whose scope moved past its rows keeps none.
        let checkpoints: Vec<u64> = match state {
            Some(_) if shares => checkpoint_rows(
                &ctx.resources().get(&fwd)?.bindings.checkpoints,
                held..reach,
                page_size,
                keys.len(),
            ),
            _ => Vec::new(),
        };
        // An engine that lands a lane's state at rows inside a fire is cut only
        // past as many as it lands at once. A window's piece keeps the
        // furthest row inside it as of that row (checkpoint_at), so it forces
        // no cut.
        let snapshots = if windowed {
            0
        } else {
            crate::model::model().eta_caps().rs_snapshots as usize
        };
        let forced: Vec<u32> = checkpoints
            .iter()
            .filter(|_| !windowed)
            .skip(snapshots)
            .step_by(snapshots + 1)
            .map(|&row| (row - held) as u32)
            .collect();
        // Several lanes run in pieces only where none carries recurrent
        // state, which is bound lane by lane.
        let rows_total = full
            .lanes
            .iter()
            .map(|lane| lane.tokens.len())
            .sum::<usize>();
        // A prompt asks the activation rung for the rows a fire takes to
        // carry it whole beside the other live lanes' decodes (SPEC decision
        // 6, by demand), from before it is cut until its last piece settles:
        // the pool raises toward it where room allows, and lowers once no
        // pass asks past its smallest rung, so no piece cut wider waits on a
        // lowered rung. A few rows ask as a prompt does, whatever the cap
        // is now (the smallest rung that covers them, held while they are
        // queued); only a lone row the rung carries beside the live lanes'
        // asks nothing, no cap being narrower than it.
        // A fire asks for the recurrent rows it carries in unfolded too,
        // where they are more than its own: a rung's replay scratch holds
        // its tokens of them. The pass carries these into its first fire.
        let frozen = match lanes_frozen(
            &stores,
            &rs_ws_ids,
            &groups,
            full.lanes.len(),
            windowed || window_state,
        ) {
            Ok(frozen) => frozen,
            Err(error) => return Ok(Err(error)),
        };
        // A pass folding all it writes folds piece by piece, so only its
        // first piece carries rows in; one folding nothing carries its
        // earlier pieces' rows into each later one.
        let folds_each = (0..full.lanes.len()).all(|lane| fold.of(lane) == Some(u32::MAX));
        let whole: Vec<Option<(u32, u32)>> = full
            .lanes
            .iter()
            .map(|lane| Some((0, u32::try_from(lane.tokens.len()).unwrap_or(u32::MAX))))
            .collect();
        // What no cut removes: the rows the pass carries in, checked with no
        // own row (a piece as small as the rules allow); its own rows and
        // carried rows together are each piece's, checked as it is cut.
        let carried = piece_rows(&full, &whole, &frozen, false).carried;
        let replayed = piece_replayed(&full, &whole, &frozen, false);
        let scratch = PassScratch {
            shared: (0..full.lanes.len())
                .map(|lane| {
                    groups
                        .iter()
                        .any(|(_, lanes)| lanes.len() > 1 && lanes.contains(&lane))
                })
                .collect(),
            checkpoints: &checkpoints,
            snapshots,
        };
        let whole_compact = scratch.compact(&stores, &full, &whole, &frozen, false);
        if let Err(refusal) = replay_rung(
            &stores,
            engine,
            FireRows { own: 0, carried },
            replayed,
            piece_lanes(&whole),
            whole_compact,
        ) {
            return Ok(Err(refusal));
        }
        let wanted = prefill_want(rows_total, engine).max(carried);
        // What its fires take of the rung: a lone row carrying more than
        // itself asks as a wider pass does.
        let taken = rows_total.max(carried as usize);
        let rung_ask = ask_rung(&stores, engine, pid, quorum_pipeline_id, taken, wanted);
        let mut cut_at = crate::store::shared::fire_cap(engine);
        let mut cuts = pass_pieces(
            &full,
            cuttable,
            rs_ws_ids.is_empty() || window_only,
            mask_share(&full, prefill_share(page_size, engine), &stores, engine),
            page_size,
            &forced,
            &switched.tails,
        );
        if let Err(refusal) = every_lane_in_last_piece(&cuts, full.lanes.len()) {
            return Ok(Err(refusal));
        }
        // Only a lone lane's rest is cut again once a raise lands.
        let recuttable = full.lanes.len() == 1 && cuts.len() > 1;
        // A piece that would not cut to the rung asks for its own width,
        // and one that carries more rows than its own for those.
        let pieces: Vec<FireRows> = cuts
            .iter()
            .enumerate()
            .map(|(at, piece)| piece_rows(&full, piece, &frozen, folds_each && at > 0))
            .collect();
        let widest = match pieces_rung(&stores, engine, &full, &cuts, &frozen, folds_each, &scratch)
        {
            Ok(widest) => widest,
            Err(refusal) => return Ok(Err(refusal)),
        };
        let (wanted, rung_ask) = if widest > wanted {
            // The wider ask stands before the narrower goes.
            let taken = taken.max(widest as usize);
            let wider = ask_rung(&stores, engine, pid, quorum_pipeline_id, taken, widest);
            drop(rung_ask);
            (widest, wider)
        } else {
            (wanted, rung_ask)
        };
        // The process holds the ask past this pass (an inferlet feeding a
        // prompt in several passes lowers nothing between them); a pass of a
        // few rows says its process asks nothing, unless it carries more.
        let pass_rows = pieces
            .iter()
            .map(|rows| rows.carried as usize)
            .max()
            .unwrap_or(0)
            .max(rows_total);
        crate::store::shared::hold_for_process(
            pid,
            engine,
            (pass_rows > RUNG_WANT_FROM).then_some(wanted),
        );
        // What one fire would run past what a fire holds is refused here, by
        // name, rather than by the scheduler's row count.
        let capacity = crate::engine::get_spec(engine)
            .map_or(usize::MAX, |spec| spec.limits.max_forward_tokens);
        let last_rows: usize = cuts.last().map_or(0, |piece| {
            piece
                .iter()
                .flatten()
                .map(|&(from, to)| (to - from) as usize)
                .sum()
        });
        if last_rows > capacity {
            let mut why: Vec<&str> = Vec::new();
            if cuts.len() > 1 || (cuttable && separable_rows(&full)) {
                why.push("the rows it reads out, or a media span and what follows it, ride its last fire");
            }
            if decode_envelope.is_some() || fire_wide_mask {
                why.push("its geometry or mask resolves on the device");
            }
            if prologue_moves {
                why.push("its prologue takes or puts a channel before the forward");
            }
            if adapted && !crate::model::model().eta_caps().adapter_attachments {
                why.push("this engine cannot carry its adapter on a fire of its own");
            }
            if switched.rows_read {
                why.push("it reads out an output of rows a cut fire would not hold");
            }
            if !folds_all && !window_only {
                why.push("it states where its recurrent state folds");
            }
            if full.lanes.len() > 1 && !rs_ws_ids.is_empty() && !window_only {
                why.push(
                    "its lanes carry rows its recurrent layers replay, which several lanes do \
                     not take in pieces",
                );
            }
            if why.is_empty() {
                why.push("its rows read one another (attention both ways, a block draft, an attention group, inputs or voxels)");
            }
            return Ok(Err(format!(
                "pipeline: a pass of {rows_total} rows would run {last_rows} in one fire, \
                 past the {capacity} a fire holds, and cannot be cut further: {}",
                why.join("; ")
            )));
        }
        // The pieces in frames of their own, in the order they are cut, so
        // the pieces of concurrent prompts seal together.
        let frame_size = crate::scheduler::configured_frame_size().max(1);
        // A cut pass behind fires of its own frame would run its leading
        // pieces (unframed) between them and its last piece (framed with
        // them): no order runs both before it. The frame goes with the fires
        // ahead of it, and the pass and the frame's later slots in frames of
        // their own, in order.
        let frame = match frame {
            Some(stamp) if reframes(stamp, ahead, cuts.len()) => {
                let _ = scheduler.frame_truncate(stamp.lane, stamp.seq, ahead);
                let pipeline = ctx.resources().get(&this)?;
                pipeline.mark_reframed(stamp.seq);
                Some(crate::scheduler::FrameStamp {
                    seq: pipeline.next_frame_seq(),
                    slot: 0,
                    fires: 1,
                    ..stamp
                })
            }
            frame => frame,
        };
        let mut frames = {
            let pipeline = ctx.resources().get(&this)?;
            piece_frames(frame, cuts.len(), frame_size, || pipeline.next_frame_seq())
        };
        let framed = matches!(frame, Some(stamp) if stamp.fires == 1);
        let mut submitted: Vec<crate::engine::completion::WorkItemCompletion> = Vec::new();
        // The working sets a piece of this pass has moved the cursor of.
        let mut advanced = vec![false; groups.len()];
        let mut writes = Vec::with_capacity(groups.len());
        let set_start = |group: usize, start: u64| {
            let (set, lanes) = (&group_sets[group], &groups[group].1);
            SetStart {
                model: set.model,
                engine: set.engine,
                kv: set.id,
                cursor: pass_lanes.cursors[lanes[0]].clone(),
                start,
                rs: rs_ws_ids.get(lanes[0]).copied(),
            }
        };
        let mut at = 0;
        while at < cuts.len() {
            if recuttable && at > 0 && at + 1 < cuts.len() {
                // While the raise is pending the pass runs no further ahead
                // of the device than a few pieces, so what is left is cut at
                // the raised rung once it lands.
                // A framed piece waits only on a frame already whole.
                let ahead = PIECES_AHEAD.max(frame_size);
                // A rise short of free room lands only once kv gives room
                // back: the pass fires at the rung held, and holds back no
                // piece for it.
                let held = rung_ask.as_ref().is_some_and(RungAsk::fires_at_held_rung);
                if at >= ahead && wanted > cut_at && !held {
                    let behind = submitted[at - ahead].clone();
                    // The piece behind settling or the cap rising, each its
                    // own event.
                    loop {
                        let watch = crate::store::shared::rung_watch(engine);
                        tokio::pin!(watch);
                        watch.as_mut().enable();
                        if behind.is_settled() || crate::store::shared::fire_cap(engine) > cut_at {
                            break;
                        }
                        tokio::select! {
                            () = &mut watch => {}
                            _ = behind.clone() => {}
                        }
                    }
                }
                let cap = crate::store::shared::fire_cap(engine);
                if cap > cut_at && (!framed || at % frame_size == 0) {
                    cut_at = cap;
                    let start = cuts[at - 1][0].map_or(0, |(_, to)| to);
                    let rest = host_cuts_from(
                        &full,
                        cuttable,
                        mask_share(&full, prefill_share(page_size, engine), &stores, engine),
                        page_size,
                        &forced,
                        switched.tails.first().copied().unwrap_or(0),
                        start,
                    );
                    cuts.truncate(at);
                    cuts.extend(rest.into_iter().map(|cut| vec![Some(cut)]));
                    let stamp = if framed {
                        let pipeline = ctx.resources().get(&this)?;
                        frame.map(|stamp| crate::scheduler::FrameStamp {
                            seq: pipeline.next_frame_seq(),
                            ..stamp
                        })
                    } else {
                        frame
                    };
                    let pipeline = ctx.resources().get(&this)?;
                    frames.truncate(at);
                    frames.extend(piece_frames(stamp, cuts.len() - at, frame_size, || {
                        pipeline.next_frame_seq()
                    }));
                }
            }
            let piece = cuts[at].clone();
            let piece = &piece;
            // A piece wider than the rung (one no cut brings under it) waits
            // for the raise its pass funds, or fails with what the planner
            // said: it never waits on a rung that does not come. A piece
            // cut again once a raise landed is counted as it is cut now.
            let counted = piece_rows(&full, piece, &frozen, folds_each && at > 0);
            let width = match replay_rung(
                &stores,
                engine,
                counted,
                piece_replayed(&full, piece, &frozen, folds_each && at > 0),
                piece_lanes(piece),
                scratch.compact(&stores, &full, piece, &frozen, folds_each && at > 0),
            ) {
                Ok(width) => width,
                Err(refusal) => return Ok(Err(refusal)),
            };
            let piece_compact =
                scratch.compact(&stores, &full, piece, &frozen, folds_each && at > 0);
            if let Some(ask) = rung_ask.as_ref()
                && let Err(refusal) = ask.wait(width).await
            {
                return Ok(Err(refusal));
            }
            let last = at + 1 == cuts.len();
            let mut req = if cuts.len() == 1 {
                full.clone()
            } else {
                let mut rows = rows_of_lanes(&full, piece, last, page_size);
                let kept: Vec<usize> = piece
                    .iter()
                    .enumerate()
                    .filter(|(_, cut)| cut.is_some())
                    .map(|(lane, _)| lane)
                    .collect();
                stamp_lane_words(&mut rows, &switched, &kept, false, None);
                rows
            };
            // What the scheduler steps and holds the fire by.
            req.carried = counted.carried;
            req.replayed = piece_replayed(&full, piece, &frozen, folds_each && at > 0);
            req.scratch_compact = piece_compact;
            req.replay_rungs = fire_rungs(&stores, !rs_ws_ids.is_empty());
            // A piece's request keeps only the lanes with rows in it,
            // renumbered; `groups` holds the pass's lane indices. What reads
            // the piece's request (its written spans, its page tables) needs
            // the piece's.
            let kept_lanes: Vec<usize> = if cuts.len() == 1 {
                (0..full.lanes.len()).collect()
            } else {
                piece
                    .iter()
                    .enumerate()
                    .filter(|(_, cut)| cut.is_some())
                    .map(|(lane, _)| lane)
                    .collect()
            };
            let piece_groups: LaneGroups = groups
                .iter()
                .map(|(ws, lanes)| {
                    (
                        *ws,
                        lanes
                            .iter()
                            .filter_map(|lane| kept_lanes.iter().position(|kept| kept == lane))
                            .collect(),
                    )
                })
                .collect();
            let spans: Vec<(std::ops::Range<u64>, std::ops::Range<u64>)> = piece_groups
                .iter()
                .map(|(_, lanes)| written_span_of(&req, lanes, page_size))
                .collect();
            let (slots, _) = spans[0].clone();
            let at_row = |rs, row: u64, snapshot| {
                let pages = row / u64::from(page_size.max(1));
                TakeCheckpoint {
                    kv: ws.id,
                    rs,
                    snapshot,
                    pages,
                    key: keys[pages as usize - 1],
                    window_at: None,
                }
            };
            let (window_at, keep) = piece_window_rows(&checkpoints, windowed, held, &slots);
            let mut checkpoints_here: Vec<TakeCheckpoint> = match (state, window_at) {
                (Some(rs), Some(row)) => vec![TakeCheckpoint {
                    window_at: u32::try_from(row).ok(),
                    ..at_row(rs, row, false)
                }],
                (Some(rs), None) if checkpoints.contains(&slots.end) => {
                    vec![at_row(rs, slots.end, false)]
                }
                _ => Vec::new(),
            };
            // The furthest the engine lands of those inside the fire.
            let mut inside: Vec<u32> = checkpoints
                .iter()
                .filter(|&&row| slots.start < row && row < slots.end)
                .map(|&row| (row - slots.start) as u32)
                .collect();
            inside.drain(..inside.len().saturating_sub(snapshots));
            let present = written_groups(&piece_groups);
            let sets: Vec<(&KvWorkingSet, Written)> = present
                .iter()
                .map(|&group| {
                    let written = match &paged_reserve {
                        Some(reserves) => Written::Pages(reserves[group].clone()),
                        None => Written::Span(spans[group].1.clone()),
                    };
                    (&group_sets[group], written)
                })
                .collect();
            // The piece's lanes' recurrent states, in the piece's lane
            // order: a cut of several lanes (a windowed-only load's) keeps
            // only those with rows in it.
            // What the piece folds: the pass's fold where it folds all it
            // writes or nothing, else (a windowed-only load's cut of other
            // folds) nothing before its last piece, which folds as the pass
            // states, every lane in it.
            let piece_fold = if folds_all || last {
                fold.clone()
            } else {
                geometry::Fold::None
            };
            let piece_rs_ids: Vec<crate::store::rs::RsWorkingSetId> = if rs_ws_ids.is_empty() {
                Vec::new()
            } else {
                kept_lanes.iter().map(|&lane| rs_ws_ids[lane]).collect()
            };
            let piece_rs_groups: LaneGroups = piece_groups
                .iter()
                .filter(|(_, lanes)| !lanes.is_empty())
                .cloned()
                .collect();
            // Each beam group's pool end and common slot, for the groups the
            // piece keeps (in `piece_rs_groups`' order).
            let (piece_beam_ends, piece_beam_commons): (Vec<u32>, Vec<u32>) = piece_groups
                .iter()
                .zip(beam_ends.iter().zip(&beam_commons))
                .filter(|((_, lanes), _)| !lanes.is_empty())
                .map(|(_, (&end, &common))| (end, common))
                .unzip();
            // The lanes each set carries, for what their state asks alone.
            let set_lanes: Vec<Vec<usize>> = if rs_ws_ids.is_empty() {
                Vec::new()
            } else {
                present.iter().map(|&group| piece_groups[group].1.clone()).collect()
            };
            let rs_plan = || {
                if windowed {
                    host_window_plan(&req, &piece_fold, &piece_rs_ids, keep)
                } else {
                    host_rs_plan(
                        &stores,
                        &piece_rs_ids,
                        &piece_rs_groups,
                        &req,
                        &piece_fold,
                        &inside,
                        (!piece_beam_ends.is_empty()).then_some(piece_beam_ends.as_slice()),
                        (!piece_beam_ends.is_empty()).then_some(piece_beam_commons.as_slice()),
                    )
                }
            };
            let reserved = reserve(
                ctx,
                (model, engine),
                &stores,
                quorum_pipeline_id,
                &pipeline_failure,
                &sets,
                &set_lanes,
                &piece_rs_ids,
                rs_plan,
                windowed,
            )
            .await;
            let Reserved {
                leases,
                copies: (copy_src, copy_dst),
                kvtxns,
                rs: rs_prepared,
                epoch,
                unfired,
            } = match reserved {
                Ok(reserved) => reserved,
                Err(error) => return Ok(Err(error)),
            };
            crate::probe_fire_record!(submit_probe.kv_prepare_us, kv_clock.elapsed());
            // Every fire logs what it writes where its tokens are known; what
            // it folds seals once it settles.
            let reports = pass_slots.is_none()
                && paged_reserve.is_none()
                && decode_envelope.is_some()
                && !wide
                && plain
                && plain_decode(&req, adapted, fire_wide_mask);
            let mut parts = Vec::with_capacity(group_sets.len());
            for (&group, (lease, kvtxn)) in present.iter().zip(leases.into_iter().zip(kvtxns)) {
                let (set, (slots, written)) = (&group_sets[group], &spans[group]);
                let lanes = &groups[group].1;
                let mut tags = Vec::new();
                let rs = rs_ws_ids.get(lanes[0]).copied().filter(|_| !windowed);
                let by = fold_at(
                    &pass_lanes,
                    lanes,
                    &landings,
                    &piece_fold,
                    slots.end,
                    last,
                    rs,
                    &mut req,
                    &mut tags,
                );
                // What the piece folds is frozen with its log begun, in the
                // step `reserve` ended by committing the recurrent state's
                // fold: nothing is awaited between them. A fire that does
                // not follow fails the pipeline, and its sets with it,
                // however this call ends (`unfired`).
                let freeze = match &by {
                    Some(FoldBy::To(to)) => Some(*to),
                    _ => None,
                };
                let logged = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
                    let folded = kv.folded(set.id)?;
                    let pages = if reports && kv.keeps() {
                        let flat = kv.flat_table(set.id)?;
                        Some(
                            written
                                .clone()
                                .filter_map(|index| {
                                    flat.get(index as usize).map(|page| (page.0, index))
                                })
                                .collect::<Vec<_>>(),
                        )
                    } else {
                        None
                    };
                    let seq = begin_piece(kv, set.id, slots.start, freeze)?;
                    Ok::<_, crate::store::kv::KvStoreError>((folded, seq, pages))
                });
                let (folded, seq, pages) = match logged {
                    Ok(logged) => logged,
                    Err(error) => {
                        let reason = format!("pipeline: KV rows: {error}");
                        record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
                        return Ok(Err(reason));
                    }
                };
                for &lane in lanes {
                    if let Some(lane) = req.lanes.get_mut(lane) {
                        restate_boundary(lane, folded);
                    }
                }
                let Some(seq) = seq else {
                    let reason = format!(
                        "pipeline: a fire writes slot {}, inside its working set's frozen \
                         prefix (0..{folded})",
                        slots.start
                    );
                    record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
                    return Ok(Err(reason));
                };
                // A piece carrying media is not its tokens' alone.
                let rows = match (&pass_slots, pages) {
                    // A lane's own page list holds its slots apart from the
                    // set's: the set's rows there are no one token's.
                    _ if paged_reserve.is_some() => LogRows::Unknown,
                    (Some(_), _) if !req.media.is_empty() => LogRows::Unknown,
                    (Some(_), _) => LogRows::Known {
                        start: slots.start,
                        tokens: req.lanes[0].tokens.clone(),
                    },
                    (None, Some(pages)) => LogRows::Reported {
                        tag: report_on(&mut req, 0, &mut tags),
                        pages,
                        page_size,
                    },
                    (None, None) => LogRows::Unknown,
                };
                parts.push(FireWs {
                    ws: set.id,
                    kvtxn,
                    log: Some(LogFire { seq, rows }),
                    lease,
                    fold: by.map(|by| FoldAt {
                        by,
                        decoded: reports,
                    }),
                    reports: tags,
                    staged: None,
                });
            }
            if let Err(refusal) = map_lane_pages(&mut req, &stores, &piece_groups) {
                record_submit_failure(ctx, &fwd, &pipeline_failure, &refusal);
                return Ok(Err(refusal));
            }
            if paged_reserve.is_some()
                && let Err(refusal) = stamp_lane_reserve(&mut req, &stores, &piece_groups)
            {
                record_submit_failure(ctx, &fwd, &pipeline_failure, &refusal);
                return Ok(Err(refusal));
            }
            crate::offload::stream::tap(pid, &stores, &mut req);
            rs_prepared.apply_to(&mut req);
            debug_assert_eq!(
                untagged_device_fold(&req),
                None,
                "a lane whose fold the device counts asks for its report"
            );
            rs::apply_shared(&mut req, stores.shared.as_ref(), Some(&rs_prepared));
            checkpoints_here.extend(
                rs_prepared
                    .snapshots
                    .iter()
                    .map(|snap| at_row(snap.snapshot, slots.start + u64::from(snap.row), true)),
            );
            let rs_copy = rs_prepared.state_copy();
            let window_copies = rs_prepared.windowed_copies();
            let rstxns = RsTxnsGuard::new(model, engine, rs_prepared.txn);
            let completion = ctx
                .resources()
                .get_mut(&fwd)?
                .bound_instance
                .reserve_completion();

            // Only the last fire runs the program, so only it takes channel values.
            let ticket_reservation = last.then(|| TicketReservation::new(&cells, &accesses));
            if let Some(reservation) = &ticket_reservation {
                reservation.apply_to(&mut req);
            }
            // A run-ahead fire's rows on a set with a recurrent state or
            // windows wait for the program to take what it publishes.
            let run_ahead_reads = (last && decode_envelope.is_some() && lags && !final_rows)
                .then(|| host_reads(&cells, &accesses));

            let scheduler_clock = crate::scheduler::probe::ProbeClock::start();
            let submit_error =
                crate::scheduler::submit_prebuilt_tracked_async_with_kv_and_rs_copy_on(
                    &scheduler,
                    req,
                    instance_id,
                    pid,
                    quorum_pipeline_id,
                    completion.clone(),
                    copy_src,
                    copy_dst,
                    window_copies,
                    rs_copy,
                    frames[at],
                    adapted,
                )
                .err()
                .map(|error| format!("{error:#}"));
            crate::probe_fire_record!(submit_probe.scheduler_submit_us, scheduler_clock.elapsed());
            if let Some(error) = submit_error {
                let reason = format!("pipeline: submit failed: {error}");
                record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
                return Ok(Err(reason));
            }
            let unqueued = unfired.submitted();
            epoch.retire_on(&completion);
            for part in &mut parts {
                part.kvtxn.retire_on(&completion);
            }
            // Before the next piece is laid, which would write the record
            // the checkpoint keeps.
            checkpoints_ahead(&stores, &mut checkpoints_here, &mut parts);
            mark_device_folds(&stores, &completion, &mut parts);
            unsettle(&stores, &completion, &mut parts);
            submitted.push(completion.clone());
            if let Some(reservation) = ticket_reservation {
                reservation.commit();
            }

            if last {
                let began = crate::scheduler::probe::ProbeClock::start();
                advance_shadow(ctx, &fwd)?;
                crate::probe_fire_record!(submit_probe.shadow_advance_us, began.elapsed());
            }

            // Each set the piece writes ends where the piece does: a pass
            // that stops short of its last piece leaves the cursors at the
            // pieces that went.
            for &group in &present {
                let (lanes, slots) = (&groups[group].1, &spans[group].0);
                let cell = &pass_lanes.cursors[lanes[0]];
                let start = if advanced[group] {
                    slots.start
                } else {
                    let start = lanes
                        .iter()
                        .map(|&lane| u64::from(landings[lane].start))
                        .min()
                        .unwrap_or(slots.start);
                    writes.push(set_start(group, start));
                    start
                };
                advanced[group] = true;
                advance_cursor(
                    cell,
                    Some(slots.end as u32),
                    Some((
                        start as u32,
                        lagging_spans(&landings, &matched, lanes, &(start..slots.end)),
                    )),
                );
                host_wrote(cell, &completion, start as u32, slots.end as u32);
                if let Some(reads) = &run_ahead_reads {
                    cell.lock()
                        .unwrap()
                        .run_ahead(start as u32, slots.end as u32, reads.clone());
                }
            }
            unqueued.queue(
                &pipe_fires,
                PendingFire {
                    completion,
                    checkpoints: checkpoints_here,
                    kv: parts,
                    device_geometry: false,
                    rstxn: rstxns,
                    model,
                    engine,
                    fwd_rep,
                    cells: cells.clone(),
                    failure: pipeline_failure.clone(),
                    retired: None,
                    lane: Some(quorum_pipeline_id),
                },
            );
            // The next piece reads the KV this one writes: they run in
            // order on the device, so it is submitted now, as a decode's
            // next pass is, rather than once this one retires.
            at += 1;
        }
        if let (Some(ask), Some(last)) = (rung_ask, submitted.last()) {
            ask.until_settled(last);
        }
        // The claims on what this pass computes stand until its last piece
        // settles, its pages sealed if it landed; a pass that ends early
        // drops them.
        if let Some(sealing) = sealing.take()
            && !submitted.is_empty()
        {
            sealing.on_landing(&submitted);
        }
        // A set no piece wrote (its lanes carry no rows) ends where they land.
        for (group, (_, lanes)) in groups.iter().enumerate() {
            if advanced[group] {
                continue;
            }
            let end = lanes.iter().map(|&lane| landings[lane].end()).max();
            let start = lanes.iter().map(|&lane| landings[lane].start).min();
            if let (Some(end), Some(start)) = (end, start) {
                let cell = &pass_lanes.cursors[lanes[0]];
                advance_cursor(cell, Some(end), Some((start, Vec::new())));
                writes.push(set_start(group, u64::from(start)));
                if let Some(last) = submitted.last() {
                    host_wrote(cell, last, start, end);
                }
            }
        }
        record_writes(ctx, this.rep(), Some(writes), final_rows)?;
        crate::probe_fire_record!(submit_probe.total_us, submit_clock.elapsed());
        crate::probe_fire_count!(submit_probe.submits);
        Ok(Ok(()))
    }
}

/// The frame each of a pass's `pieces` fires carries, the pass's frame
/// `frame`, in frames of `k` steps. A piece reads the KV and state the one
/// before it writes, so a pass alone in its frame runs its pieces in order,
/// one a step, `k` to a frame: the first frame takes `frame`'s sequence and
/// each later one the next the pipeline hands out (`next_seq`). A lane's
/// pieces queue in their order and the pieces of concurrent lanes seal
/// together rather than one launch at a time. A pass sharing its frame keeps
/// that frame for its last piece, its earlier pieces unframed.
/// Whether a pass cut into `pieces`, stamped `stamp` behind `ahead` fires
/// of its own frame, goes in frames of its own: its leading pieces go
/// unframed and its last with the frame, so with fires of its frame ahead
/// no order would run them all first.
fn reframes(stamp: crate::scheduler::FrameStamp, ahead: u32, pieces: usize) -> bool {
    stamp.fires > 1 && ahead > 0 && pieces > 1
}

/// Every lane of a pass rides its last fire, the one that runs the program:
/// a pass whose last fire left a lane out would read it nowhere.
fn every_lane_in_last_piece(cuts: &[Vec<Option<(u32, u32)>>], lanes: usize) -> Result<(), String> {
    let held = cuts
        .last()
        .map_or(0, |piece| piece.iter().filter(|cut| cut.is_some()).count());
    if held == lanes {
        Ok(())
    } else {
        Err(format!(
            "pipeline: the last fire of a pass of {lanes} lanes carries {held}: every lane of \
             a pass rides its last fire, the one that runs its program"
        ))
    }
}

pub(crate) fn piece_frames(
    frame: Option<crate::scheduler::FrameStamp>,
    pieces: usize,
    k: usize,
    mut next_seq: impl FnMut() -> u64,
) -> Vec<Option<crate::scheduler::FrameStamp>> {
    let k = k.max(1);
    match frame {
        Some(stamp) if stamp.fires == 1 => {
            let mut seq = stamp.seq;
            (0..pieces)
                .map(|at| {
                    if at > 0 && at % k == 0 {
                        seq = next_seq();
                    }
                    let first = at - at % k;
                    Some(crate::scheduler::FrameStamp {
                        seq,
                        slot: (at % k) as u32,
                        fires: (pieces - first).min(k) as u32,
                        ..stamp
                    })
                })
                .collect()
        }
        frame => (0..pieces)
            .map(|at| if at + 1 == pieces { frame } else { None })
            .collect(),
    }
}

pub async fn submit_frame<C: FireContext>(
    ctx: &mut C,
    this: Resource<Pipeline>,
    slot_reps: Vec<Option<u32>>,
) -> Anyhow<Result<(), String>> {
    tracing::debug!("guest submits a frame");
    let k = crate::scheduler::configured_frame_size();
    if slot_reps.len() != k {
        return Ok(Err(format!(
            "pipeline: frame holds {} slot(s); model.frame-size() is {k} — \
             supply exactly k ordered slots (none = no-op)",
            slot_reps.len()
        )));
    }
    let fired: Vec<(u32, u32)> = slot_reps
        .iter()
        .enumerate()
        .filter_map(|(slot, rep)| rep.map(|rep| (slot as u32, rep)))
        .collect();
    if fired.is_empty() {
        return Ok(Err(
            "pipeline: a frame needs at least one non-no-op slot".to_string()
        ));
    }
    for &(slot, rep) in &fired {
        let fwd: Resource<ForwardPass> = Resource::new_borrow(rep);
        if !ctx.resources().get(&fwd)?.is_bound() {
            return Ok(Err(format!(
                "pipeline: frame slot {slot}: forward pass program is not attached"
            )));
        }
    }
    if k == 1 {
        let (_, rep) = fired[0];
        return submit_pass_stamped(ctx, this, Resource::new_borrow(rep), None, 0).await;
    }
    {
        let probe = crate::scheduler::probe::host_submit();
        let began = crate::scheduler::probe::ProbeClock::start();
        let verdict = validate_frame(ctx, k, &fired)?;
        crate::probe_fire_record!(probe.validate_frame_us, began.elapsed());
        crate::probe_fire_count!(probe.validate_frame_calls);
        if let Err(error) = verdict {
            return Ok(Err(error));
        }
    }
    let (lane, seq) = {
        let pipeline = ctx.resources().get(&this)?;
        if pipeline.scope.is_closed() {
            return Ok(Err("pipeline: pipeline is closed".to_string()));
        }
        (pipeline.scope.scheduler_id(), pipeline.next_frame_seq())
    };
    let fires = fired.len() as u32;
    for (index, &(slot, rep)) in fired.iter().enumerate() {
        // Once a cut pass truncated the frame, each slot after it goes in a
        // frame of its own.
        let reframed = ctx.resources().get(&this)?.reframed(seq);
        let (stamp, ahead) = if reframed {
            let seq = ctx.resources().get(&this)?.next_frame_seq();
            let own = crate::scheduler::FrameStamp {
                lane,
                seq,
                slot: 0,
                fires: 1,
            };
            (own, 0)
        } else {
            let stamp = crate::scheduler::FrameStamp {
                lane,
                seq,
                slot,
                fires,
            };
            (stamp, index as u32)
        };
        let pipeline: Resource<Pipeline> = Resource::new_borrow(this.rep());
        let fwd: Resource<ForwardPass> = Resource::new_borrow(rep);
        let outcome = submit_pass_stamped(ctx, pipeline, fwd, Some(stamp), ahead).await;
        if !matches!(outcome, Ok(Ok(()))) && index > 0 && !reframed {
            let first: Resource<ForwardPass> = Resource::new_borrow(fired[0].1);
            if let Ok(pass) = ctx.resources().get(&first)
                && let Ok(bound) = pass.bound()
            {
                let _ = bound.scheduler.frame_truncate(lane, seq, index as u32);
            }
        }
        match outcome {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                return Ok(Err(format!("pipeline: frame slot {slot}: {error}")));
            }
            Err(error) => return Err(error),
        }
    }
    Ok(Ok(()))
}

pub fn park_frame<C: FireContext>(ctx: &mut C, this: Resource<Pipeline>) -> Anyhow<()> {
    let pipeline = ctx.resources().get(&this)?;
    // Parked, it holds no batch, whatever the frame size.
    pipeline.set_active(false);
    if crate::scheduler::configured_frame_size() == 1 {
        return Ok(());
    }
    if pipeline.scope.is_closed() {
        return Ok(());
    }
    let (lane, seq) = (pipeline.scope.scheduler_id(), pipeline.next_frame_seq());
    crate::scheduler::worker::notify_lane_park(lane, seq);
    Ok(())
}

fn validate_frame<C: FireContext>(
    ctx: &mut C,
    k: usize,
    fired: &[(u32, u32)],
) -> Anyhow<Result<(), String>> {
    let mut slots: Vec<SlotAccess> = Vec::with_capacity(fired.len());
    for &(_, rep) in fired {
        let fwd: Resource<ForwardPass> = Resource::new_borrow(rep);
        let pass = ctx.resources().get(&fwd)?;
        let bound = match pass.bound() {
            Ok(bound) => bound,
            Err(error) => return Ok(Err(format!("pipeline: {error}"))),
        };
        slots.push(SlotAccess {
            cells: bound.cells.clone(),
            accesses: bound.instance.program.channel_accesses.clone(),
        });
    }
    Ok(prove_frame_admissible(k, &slots))
}

struct SlotAccess {
    cells: BoundCells,
    accesses: Vec<(bool, bool)>,
}

fn prove_frame_admissible(k: usize, slots: &[SlotAccess]) -> Result<(), String> {
    struct ChannelUse {
        cell: Arc<Mutex<crate::pipeline::channel::ChannelCell>>,
        consumes: usize,
        publishes: usize,
    }
    struct DeviceRingUse {
        global_id: ::engine::ChannelId,
        capacity: u64,
        pressure: u64,
    }
    let mut uses: std::collections::HashMap<usize, ChannelUse> = std::collections::HashMap::new();
    let mut device_rings: std::collections::HashMap<usize, DeviceRingUse> =
        std::collections::HashMap::new();

    for (slot, step) in slots.iter().enumerate() {
        for (cell, &(consume, publish)) in step.cells.iter().zip(&step.accesses) {
            let key = Arc::as_ptr(cell) as usize;
            let entry = uses.entry(key).or_insert_with(|| ChannelUse {
                cell: cell.clone(),
                consumes: 0,
                publishes: 0,
            });
            entry.consumes += usize::from(consume);
            entry.publishes += usize::from(publish);

            if consume || publish {
                let guard = cell.lock().unwrap();
                if guard.role == Some(HostRole::None) && !guard.seeded {
                    let ring = device_rings.entry(key).or_insert_with(|| DeviceRingUse {
                        global_id: guard.global_id,
                        capacity: u64::from(guard.capacity),
                        pressure: guard.device_ring_backlog(),
                    });
                    if publish && ring.pressure >= ring.capacity + u64::from(consume) {
                        return Err(format!(
                            "pipeline: channel {}: frame slot {slot} would raise \
                             device-ring occupancy past capacity {} (reserved backlog \
                             plus in-frame publishes) — size device-only rings so \
                             every frame's publish backlog fits",
                            ring.global_id, ring.capacity,
                        ));
                    }
                    ring.pressure += u64::from(publish);
                    ring.pressure = ring.pressure.saturating_sub(u64::from(consume));
                }
            }
        }
    }

    for entry in uses.values() {
        let cell = entry.cell.lock().unwrap();
        match cell.role {
            Some(HostRole::Writer) => {
                if entry.publishes == 0 && entry.consumes > 0 {
                    let available = cell.writer_available_cells();
                    if available < entry.consumes as u64 {
                        return Err(format!(
                            "pipeline: channel {}: frame consumes {} host-writer \
                             cell(s) but only {available} are staged — stage every \
                             per-fire input before submitting the frame",
                            cell.global_id, entry.consumes
                        ));
                    }
                } else if entry.publishes == 0 && entry.consumes == 0 && !cell.has_committed_front()
                {
                    return Err(format!(
                        "pipeline: channel {}: latest-value control word has \
                         no committed cell at frame submit",
                        cell.global_id
                    ));
                }
            }
            Some(HostRole::Reader) if entry.publishes > 0 => {
                let (reserved_tail, consumed) = cell.reader_ring_pressure();
                let needed = reserved_tail
                    .saturating_sub(consumed)
                    .saturating_add(entry.publishes as u64);
                if needed > u64::from(cell.capacity) {
                    return Err(format!(
                        "pipeline: channel {}: frame would need {needed} reader \
                             cell(s) (capacity {}) — size take-side channels to at \
                             least 2k-1 = {} for frame-size k = {k}",
                        cell.global_id,
                        cell.capacity,
                        2 * k - 1,
                    ));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

async fn pipeline_close_inner<C: FireContext>(
    ctx: &mut C,
    this: &Resource<Pipeline>,
) -> Anyhow<()> {
    let state = ctx.resources().get(this).ok().map(|pipeline| {
        pipeline.set_active(false);
        let first_close = pipeline.scope.close();
        (
            first_close,
            pipeline.scope.scheduler_id(),
            pipeline.fires.clone(),
        )
    });
    if let Some((first_close, pipeline_id, fires)) = state {
        if first_close {
            crate::scheduler::worker::notify_lane_close(pipeline_id, None);
        }
        drain_settled(ctx, Some(&fires)).await?;
    }
    Ok(())
}

pub async fn pipeline_close<C: FireContext>(ctx: &mut C, this: Resource<Pipeline>) -> Anyhow<()> {
    pipeline_close_inner(ctx, &this).await
}

/// Where one fire began writing one working set.
#[derive(Clone)]
pub(crate) struct SetStart {
    model: usize,
    engine: crate::engine::EngineId,
    kv: crate::store::kv::page_table::WorkingSetId,
    cursor: CursorCell,
    start: u64,
    /// The set's recurrent state (or windows) the fire's rows land in: its
    /// buffer or its windows' head, taken back with the rows.
    rs: Option<crate::store::rs::RsWorkingSetId>,
}

/// What one fire wrote, set by set: `None` where the device placed its rows.
pub(crate) type FireWrites = Option<Vec<SetStart>>;

/// One fire a pipeline keeps for `retract`: what it wrote, and whether its
/// pass stated its rows final (CONTRACT-BC 25), which no retract reaches.
#[derive(Clone)]
pub(crate) struct WrittenFire {
    pub(crate) sets: FireWrites,
    pub(crate) final_rows: bool,
}

fn record_writes<C: FireContext>(
    ctx: &mut C,
    pipeline: u32,
    writes: FireWrites,
    final_rows: bool,
) -> Anyhow<()> {
    let pipeline: Resource<Pipeline> = Resource::new_borrow(pipeline);
    let pipeline = ctx.resources().get(&pipeline)?;
    let mut written = pipeline.written.lock().unwrap();
    written.push_back(WrittenFire {
        sets: writes,
        final_rows,
    });
    // A fire further back than the run-ahead window is one the program
    // read past before submitting the newest: never taken back.
    let window = crate::scheduler::run_ahead_window();
    while written.len() > window {
        written.pop_front();
    }
    Ok(())
}

/// The row each working set the last `fires` fires wrote ends at once they
/// are taken back: where the first of them began writing it. `Err` names
/// what cannot be taken back.
fn retracted_ends(
    written: &std::collections::VecDeque<WrittenFire>,
    fires: usize,
) -> Result<Vec<SetStart>, String> {
    let mut ends: Vec<SetStart> = Vec::new();
    for (at, fire) in written.iter().skip(written.len() - fires).enumerate() {
        let Some(sets) = &fire.sets else {
            return Err(format!(
                "the {} fire from the end placed its rows where the device says, which the \
                 host cannot take back",
                fires - at
            ));
        };
        for set in sets {
            match ends.iter_mut().find(|end| end.kv == set.kv) {
                Some(end) => end.start = end.start.min(set.start),
                None => ends.push(set.clone()),
            }
        }
    }
    Ok(ends)
}

/// Which of the last `fires` fires (counted from the end, 1 the last) is
/// the latest whose pass stated its rows final, if any.
fn final_fire(written: &std::collections::VecDeque<WrittenFire>, fires: usize) -> Option<usize> {
    written
        .iter()
        .rev()
        .take(fires)
        .position(|fire| fire.final_rows)
        .map(|at| at + 1)
}

/// Takes back what the last `fires` fires submitted on the closed pipeline
/// `this` wrote, once every fire on it has landed: each working set they
/// wrote ends where the first of them began writing it. A set whose rows
/// cannot be taken back leaves the pipeline failed by that name, so the set
/// is refused wherever it is used next rather than read past its stop.
pub async fn pipeline_retract<C: FireContext>(
    ctx: &mut C,
    this: Resource<Pipeline>,
    fires: u32,
) -> Anyhow<Result<(), String>> {
    let fires = fires as usize;
    let (pending, failure, written, ends) = {
        let pipeline = ctx.resources().get(&this)?;
        if !pipeline.scope.is_closed() {
            return Ok(Err(
                "pipeline: retract on an open pipeline: close it first, so no fire follows the \
                 ones taken back"
                    .to_string(),
            ));
        }
        let written = pipeline.written.lock().unwrap();
        if fires > written.len() {
            return Ok(Err(format!(
                "pipeline: retract of {fires} fire(s), past the {} it can take back: the \
                 fires submitted, up to its run-ahead window of {}",
                written.len(),
                crate::scheduler::run_ahead_window()
            )));
        }
        // A fire whose pass stated its rows final is never taken back:
        // refused by name before anything is cut (CONTRACT-BC 25).
        if let Some(at) = final_fire(&written, fires) {
            return Ok(Err(format!(
                "final rows: pipeline: retract of {fires} fire(s) reaches the {at} fire from the \
                 end, whose pass stated its sampled rows final; they are never taken back"
            )));
        }
        (
            pipeline.fires.clone(),
            pipeline.failure.clone(),
            written.len() - fires,
            retracted_ends(&written, fires),
        )
    };
    if fires == 0 {
        return Ok(Ok(()));
    }
    // A fire that may still write is never cut under: every fire on the
    // pipeline lands, and its fold and log settle, first.
    finalize_all(ctx, &pending, false).await?;
    ctx.resources()
        .get(&this)?
        .written
        .lock()
        .unwrap()
        .truncate(written);
    if pipeline_failed(&failure).is_some() {
        return Ok(Ok(()));
    }
    let refused = match ends {
        Err(why) => Some(why),
        Ok(ends) => {
            let mut refused = None;
            for end in &ends {
                let stores = crate::store::registry::get(end.model, end.engine);
                match take_back(&stores, end).await {
                    Ok(()) => retract_cursor(&end.cursor, end.start),
                    Err(why) => {
                        refused = Some(why);
                        break;
                    }
                }
            }
            refused
        }
    };
    if let Some(why) = refused {
        let mut failure = failure.lock().unwrap();
        if failure.is_none() {
            *failure = Some(format!(
                "its rows past where the program stopped could not be taken back: {why}"
            ));
        }
    }
    Ok(Ok(()))
}

/// Returns once kv set `ws` is in no move, waiting on each move's end.
async fn kv_out_of_move(stores: &crate::store::registry::Stores, ws: crate::store::kv::page_table::WorkingSetId) {
    loop {
        let ended = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| kv.moves_ended());
        let notified = ended.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        if !crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| kv.set_in_move(ws)) {
            return;
        }
        notified.await;
    }
}

/// Takes `end`'s set back to where the retracted fires began writing it:
/// its unaccepted run-ahead rows first (refused where the run-ahead bound
/// accepted rows past the cut), then its recurrent state's buffer or its
/// windows' head (refused where a fold reached past the cut), then its KV.
/// Under the lagged fold none of these touches a folded row.
async fn take_back(stores: &crate::store::registry::Stores, end: &SetStart) -> Result<(), String> {
    let cut = u32::try_from(end.start).unwrap_or(u32::MAX);
    end.cursor.lock().unwrap().take_back(cut)?;
    if let Some(rs) = end.rs {
        match stores.rs.lock().unwrap().truncate(rs, cut) {
            // A set released already holds nothing to take back.
            Ok(()) | Err(crate::store::rs::RsError::UnknownWorkingSet) => {}
            Err(error) => return Err(error.to_string()),
        }
    }
    // A set in a move (a suspend or a sweep pins its pages) is cut once
    // the move ends: cut under it, the pages it freed went back again as
    // the move committed (a kv page released twice).
    loop {
        kv_out_of_move(stores, end.kv).await;
        match crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
            kv.truncate(end.kv, end.start)
        }) {
            Ok(())
            | Err(crate::store::kv::KvStoreError::Table(
                crate::store::kv::page_table::KvTableError::UnknownWorkingSet,
            )) => return Ok(()),
            Err(crate::store::kv::KvStoreError::MoveInFlight) => {}
            Err(error) => return Err(error.to_string()),
        }
    }
}

/// The host's record of a set's rows ends at `end`.
fn retract_cursor(cell: &CursorCell, end: u64) {
    let end = u32::try_from(end).unwrap_or(u32::MAX);
    let mut cursor = cell.lock().unwrap();
    cursor.tail = Some(end);
    cursor.written = cursor.written.min(end);
    cursor.spans.retain(|&(from, _, _)| from < end);
}

pub async fn pipeline_drop<C: FireContext>(ctx: &mut C, this: Resource<Pipeline>) -> Anyhow<()> {
    pipeline_close_inner(ctx, &this).await?;
    ctx.resources().delete(this)?;
    Ok(())
}

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
async fn finalize_unattached(op: PendingOp) {
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
fn settle_fold(
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

async fn finalize_fire_await(fire: PendingFire) -> FinalizeOutcome {
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

fn fail_pass<C: FireContext>(ctx: &mut C, fwd_rep: u32, reason: &str) {
    let res: Resource<ForwardPass> = Resource::new_borrow(fwd_rep);
    if let Ok(p) = ctx.resources().get_mut(&res)
        && p.failed.is_none()
    {
        p.failed = Some(reason.to_string());
    }
}

fn record_submit_failure<C: FireContext>(
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

fn fold_of_pass(pass: &crate::pipeline::instance::BoundForwardPass) -> geometry::Fold {
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

async fn fire_device_geometry<C: FireContext>(
    ctx: &mut C,
    this: Resource<Pipeline>,
    fwd: Resource<ForwardPass>,
    frame: Option<crate::scheduler::FrameStamp>,
) -> Anyhow<Result<(), String>> {
    if ctx
        .resources()
        .get(&fwd)?
        .bindings
        .lanes
        .iter()
        .any(|lane| !lane.media.is_empty())
    {
        return Ok(Err(
            "pipeline: MediaDeviceGeometry: this pass attached media spans \
             and resolves its token ids on the device, so the host has no \
             token list to scan for their placeholder runs — media rides a \
             host-resolved geometry, where the runs can be checked"
                .to_string(),
        ));
    }
    let (pipe_fires, pipeline_failure, pipeline_scope) =
        match open_pipeline(ctx, &this, &fwd).await? {
            Ok(opened) => opened,
            Err(error) => return Ok(Err(error)),
        };

    let (rs_reps, fold) = {
        let pass = ctx.resources().get(&fwd)?;
        (pass.rs_ws.clone(), fold_of_pass(pass))
    };
    // A recurrent state a device-counted fold left unsettled is read where
    // the engine left it: the fire goes ahead of the settle.
    let pass_lanes = pass_lanes(ctx, &fwd)?;
    // A fold list the host states leaves no lane without a count.
    if let Err(error) = fold.states(pass_lanes.kv.len()) {
        return Ok(Err(format!("pipeline: fire geometry: {error}")));
    }
    let ws = pass_lanes.kv[0].clone();
    let groups = lane_groups(&pass_lanes.kv.iter().map(|ws| ws.id).collect::<Vec<_>>());
    let group_sets: Vec<KvWorkingSet> = groups
        .iter()
        .map(|(_, lanes)| pass_lanes.kv[lanes[0]].clone())
        .collect();
    let stores = crate::store::registry::get(ws.model, ws.engine);
    let pid = ctx.process_id();
    let quorum_pipeline_id = pipeline_scope.scheduler_id();
    for set in &group_sets {
        if let Err(refusal) = set.claim_pipeline_scope(&pipeline_scope) {
            return Ok(Err(format!("pipeline: KV working set {refusal}")));
        }
    }
    {
        let p = ctx.resources().get(&fwd)?;
        if let Some(e) = &p.failed {
            return Ok(Err(format!(
                "pipeline: forward-pass failed by an earlier fire: {e}"
            )));
        }
    }

    // Every page from the frozen prefix up to what `reserve` granted is backed
    // before the fire.
    let page = u64::from(stores.kv_page_size.max(1));
    let states_positions = ctx
        .resources()
        .get(&fwd)?
        .instance
        .program
        .bound
        .container
        .ports
        .iter()
        .any(|binding| binding.port == Port::Positions);
    // A beam group's pass (`parents`): its beams' recurrent rows are each
    // their own list's (`BEAM-PAGE-LISTS.md` §11).
    let states_parents = ctx
        .resources()
        .get(&fwd)?
        .instance
        .program
        .bound
        .container
        .ports
        .iter()
        .any(|binding| binding.port == Port::Parents);
    let mut write_indexes: Vec<Vec<u64>> = Vec::with_capacity(groups.len());
    let mut write_ends: Vec<u32> = Vec::with_capacity(groups.len());
    for (_, lanes) in &groups {
        let lane = lanes[0];
        let place = &pass_lanes.places[lane];
        // Past an M-RoPE span a row rotates behind its slot, which the device picks.
        if !states_positions && !place.spans.is_empty() {
            return Ok(Err(format!(
                "pipeline: lane {lane}'s rows land where the device says, past a media span \
                 whose rotation runs behind its slots; state its positions"
            )));
        }
        let reserved = pass_lanes.cursors[lane].lock().unwrap().reserved;
        let writable = match device_writable(lane, place.folded, reserved) {
            Ok(writable) => writable,
            Err(refusal) => return Ok(Err(refusal)),
        };
        write_indexes.push(writable_pages(&writable, page).collect());
        write_ends.push(writable.end);
    }

    let split = ctx
        .resources()
        .get(&fwd)?
        .devgeo
        .clone()
        .expect("fire_device_geometry on a non-device-geometry pass");
    let wide = split.windows(2).any(|lane| lane[1] > lane[0] + 1);
    let resolved_qo_indptr = if wide {
        split.clone()
    } else {
        vec![0; pass_lanes.places.len() + 1]
    };
    let rs_ws_ids = match bound_rs_working_sets(ctx, &ws, &rs_reps, &split, &pipeline_scope)? {
        Ok(ids) => ids,
        Err(error) => return Ok(Err(error)),
    };
    // What each working set folds, as far as the host can say before the fire.
    let stated: Vec<DeviceFold> = groups
        .iter()
        .map(|(_, lanes)| {
            let first = lanes[0];
            let place = &pass_lanes.places[first];
            let reserved = pass_lanes.cursors[first].lock().unwrap().reserved;
            device_fold(
                fold.of(first),
                split[first + 1] - split[first],
                reserved.saturating_sub(place.folded),
                appends_folded(&pass_lanes, lanes),
            )
        })
        .collect();
    let windowed = crate::model::model().rs_caps().window_tokens > 0;
    // The fire stages its rows whole (the device places them, so it is never
    // cut): it asks the activation rung for them as every pass does
    // ([`RungAsk`]), and is staged once the rung takes them. It asks for
    // the recurrent rows it may carry in as well: each lane's are bounded
    // by what stands unfolded in its set ([`device_lane_rows`]).
    let frozen = match lanes_frozen(
        &stores,
        &rs_ws_ids,
        &groups,
        pass_lanes.places.len(),
        windowed || crate::model::model().rs_caps().window_state,
    ) {
        Ok(frozen) => frozen,
        Err(error) => return Ok(Err(error)),
    };
    // Each group's bound on what its lanes replay, every lane's, a decode
    // lane's included (what a step is held to by the engine's stated
    // replayed rows).
    let replayed = groups
        .iter()
        .map(|(_, lanes)| {
            let cursor = pass_lanes.cursors[lanes[0]].lock().unwrap();
            let carried = device_lane_rows(0, &cursor, frozen[lanes[0]]).carried;
            carried.saturating_mul(u32::try_from(lanes.len()).unwrap_or(u32::MAX))
        })
        .fold(0u32, u32::saturating_add);
    let counted = FireRows::of(groups.iter().flat_map(|(_, lanes)| {
        let carried = {
            let cursor = pass_lanes.cursors[lanes[0]].lock().unwrap();
            device_lane_rows(0, &cursor, frozen[lanes[0]]).carried
        };
        let split = &resolved_qo_indptr;
        lanes.iter().map(move |&lane| {
            let own = split[lane + 1] - split[lane];
            FireRows {
                own,
                carried: ::engine::fire::lane_carried(own, carried),
            }
        })
    }));
    // A device-placed fire lands no rs snapshot. A lane alone on its set
    // replays every buffered slot of it, one run ending at its first row
    // (`Lineage::Set`); a lane sharing its set reads the slots its mask
    // admits, resolved on the device: not one run as far as the host can
    // say.
    let every_lane_one_run = groups
        .iter()
        .all(|(_, lanes)| lanes.len() == 1 || frozen[lanes[0]].is_none());
    let compact = fire_compact(&stores, every_lane_one_run, false);
    let fire_lanes = u32::try_from(groups.iter().map(|(_, lanes)| lanes.len()).sum::<usize>())
        .unwrap_or(u32::MAX);
    let rows = match replay_rung(&stores, ws.engine, counted, replayed, fire_lanes, compact) {
        Ok(rows) => rows,
        Err(refusal) => return Ok(Err(refusal)),
    };
    let rung_ask = ask_rung(
        &stores,
        ws.engine,
        pid,
        quorum_pipeline_id,
        rows as usize,
        rows,
    );
    if let Some(ask) = rung_ask.as_ref()
        && let Err(refusal) = ask.wait(rows).await
    {
        return Ok(Err(refusal));
    }
    if windowed && groups.len() > 1 {
        return Ok(Err(
            "pipeline: on a model with window lists, a pass whose rows land where the device \
             says writes one working set"
                .to_string(),
        ));
    }
    let sets: Vec<(&KvWorkingSet, Written)> = group_sets
        .iter()
        .zip(&write_indexes)
        .map(|(set, indexes)| (set, Written::Pages(indexes.clone())))
        .collect();
    let set_lanes: Vec<Vec<usize>> = if rs_ws_ids.is_empty() {
        Vec::new()
    } else {
        groups.iter().map(|(_, lanes)| lanes.clone()).collect()
    };
    let rs_plan = || {
        if windowed {
            device_window_plan(&stores, ws.id, &write_indexes[0], &rs_ws_ids)
        } else {
            device_rs_plan(&stores, &rs_ws_ids, &groups, &pass_lanes, &split, &fold, states_parents)
        }
    };
    let reserved = reserve(
        ctx,
        (ws.model, ws.engine),
        &stores,
        quorum_pipeline_id,
        &pipeline_failure,
        &sets,
        &set_lanes,
        &rs_ws_ids,
        rs_plan,
        windowed,
    )
    .await;
    let Reserved {
        leases,
        copies: (copy_src, copy_dst),
        kvtxns,
        rs: mut rs_prepared,
        epoch,
        unfired,
    } = match reserved {
        Ok(reserved) => reserved,
        Err(error) => return Ok(Err(error)),
    };
    let rs_copy = rs_prepared.state_copy();
    let rstxns = RsTxnsGuard::new(ws.model, ws.engine, rs_prepared.txn.take());
    // The device picks the slots a decode writes: each working set's log is
    // cut where it may start writing, its frozen length, unless the engine
    // reports what it wrote. A fold the host can place is frozen with it, in
    // the step `reserve` ended by committing the recurrent state's fold; a
    // fire that does not follow fails the pipeline, and its sets with it,
    // however this call ends (`unfired`).
    let mut logged = Vec::with_capacity(group_sets.len());
    for (((set, indexes), (_, lanes)), stated) in group_sets
        .iter()
        .zip(&write_indexes)
        .zip(&groups)
        .zip(&stated)
    {
        let place = &pass_lanes.places[lanes[0]];
        let known = u64::from(place.folded);
        let freeze = match stated {
            DeviceFold::Count(count) if place.folded_exact => Some(known + u64::from(*count)),
            _ => None,
        };
        let entry = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
            let cut = kv.folded(set.id)?.max(known);
            let seq = begin_piece(kv, set.id, cut, freeze)?.ok_or(
                crate::store::kv::KvStoreError::BadWriteSet {
                    reason: "a fire writes inside its working set's frozen prefix",
                },
            )?;
            let pages = if kv.keeps() && rs_ws_ids.is_empty() && group_sets.len() == 1 {
                let flat = kv.flat_table(set.id)?;
                Some(
                    indexes
                        .iter()
                        .filter_map(|&index| flat.get(index as usize).map(|page| (page.0, index)))
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            };
            Ok::<_, crate::store::kv::KvStoreError>((seq, pages, cut))
        });
        match entry {
            Ok(entry) => logged.push(entry),
            Err(error) => {
                let reason = format!("pipeline: KV rows: {error}");
                record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
                return Ok(Err(reason));
            }
        }
    }

    let (completion, instance_id, scheduler, cells, fwd_rep, accesses) = {
        let p = ctx.resources().get_mut(&fwd)?;
        let completion = p.bound_instance.reserve_completion();
        let accesses = p.instance.program.channel_accesses.clone();
        (
            completion,
            p.bound_instance.instance_id,
            p.scheduler.clone(),
            p.cells.clone(),
            fwd.rep(),
            accesses,
        )
    };

    let mask_qo_indptr: Vec<u32> = (0..resolved_qo_indptr.len() as u32).collect();
    let attn_mask = {
        let p = ctx.resources().get(&fwd)?;
        let bound = &p.instance.program.bound;
        let channel_bound_mask = bound.container.ports.iter().any(|binding| {
            binding.port == eta_ir::registry::Port::AttnMask
                && matches!(binding.source, eta_ir::container::PortSource::Channel(_))
        });
        if channel_bound_mask {
            Ok(geometry::FireAttnMask::Device)
        } else {
            let (shadow, shadow_cells) = (&p.host_shadow, &p.cells);
            let mut known = |chan: u32| shadow.fire_value(bound, shadow_cells, chan);
            geometry::evaluate_attn_mask(bound, &mut known, &mask_qo_indptr)
        }
    };
    let attn_mask = match attn_mask {
        Ok(mask) => mask,
        Err(error) => {
            let reason = format!("pipeline: device-geometry attention mask: {error}");
            record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
            return Ok(Err(reason));
        }
    };

    let mut req = crate::engine::FireRequest {
        boundary_program: true,
        lanes: resolved_qo_indptr
            .windows(2)
            .map(|span| {
                ::engine::Lane {
                    tokens: vec![0; (span[1] - span[0]) as usize],
                    ..::engine::Lane::default()
                }
                .into()
            })
            .collect(),
        carried: counted.carried,
        replayed,
        scratch_compact: compact,
        replay_rungs: fire_rungs(&stores, !rs_ws_ids.is_empty()),
        ..crate::engine::FireRequest::default()
    };
    rs_prepared.apply_to(&mut req);
    stamp_lane_facts(ctx, &fwd, &mut req)?;
    if wide {
        let p = ctx.resources().get(&fwd)?;
        let bound = &p.instance.program.bound;
        let readout = bound.container.ports.iter().find_map(|binding| {
            match (&binding.port, &binding.source) {
                (eta_ir::registry::Port::Readout, eta_ir::container::PortSource::Channel(chan)) => {
                    p.host_shadow.fire_value(bound, &p.cells, *chan)
                }
                _ => None,
            }
        });
        if let Some(rows) = readout {
            let rows = geometry::value_as_u32(&rows);
            for (lane, span) in req.lanes.iter_mut().zip(resolved_qo_indptr.windows(2)) {
                let local: Vec<u32> = rows
                    .iter()
                    .filter(|&&row| row >= span[0] && row < span[1])
                    .map(|&row| row - span[0])
                    .collect();
                lane.readout = ::engine::Readout::Rows(local);
            }
        }
    }
    let fire_wide_mask = matches!(attn_mask, geometry::FireAttnMask::Device);
    if let Err(error) = attn_mask.apply_to(&mut req) {
        let reason = format!("pipeline: device-geometry attention mask: {error}");
        record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
        return Ok(Err(reason));
    }
    let switched = Switched::of(ctx.resources().get(&fwd)?);
    // A split narrow enough to stage no host rows still knows each lane's
    // rows; its words carry them.
    let device_rows: Option<Vec<u32>> =
        (!wide).then(|| split.windows(2).map(|lane| lane[1] - lane[0]).collect());
    let lanes_of = every_lane(&req);
    stamp_lane_words(
        &mut req,
        &switched,
        &lanes_of,
        fire_wide_mask,
        device_rows.as_deref(),
    );
    let reads_each_other = reads_each_other(&req);
    if let Err(refusal) = crate::inferlet::process::said(
        "seat the lanes",
        ctx.process_id(),
        seat_lane_slots(&mut req, &stores, &groups),
    )
    .await
    {
        record_submit_failure(ctx, &fwd, &pipeline_failure, &refusal);
        return Ok(Err(refusal));
    }
    if let Err(refusal) = stamp_lane_translation(&mut req, &stores, &groups) {
        record_submit_failure(ctx, &fwd, &pipeline_failure, &refusal);
        return Ok(Err(refusal));
    }
    // The device places the rows, so their engine holds each lane to what
    // its set wrote ([`::engine::extent`]), from where the host knows it.
    for (lane, place) in req.lanes.iter_mut().zip(&pass_lanes.places) {
        lane.kv.readable = place.readable;
        lane.kv.set = place.set;
    }
    req.geometry = GeometryClass::DeviceGeometry;
    tracing::debug!(
        geometry = ?req.geometry,
        lanes = req.lanes.len(),
        qo_indptr = ?resolved_qo_indptr,
        reads_rows = switched.rows_read,
        "fire geometry resolved on device path"
    );
    let ticket_reservation = TicketReservation::new(&cells, &accesses);
    ticket_reservation.apply_to(&mut req);

    let lora_program = lora_program_of(ctx, &fwd)?;
    // A decode lane whose KV is its tokens' alone asks the engine what it
    // wrote, so its slots are logged.
    let plain_decode =
        !reads_each_other && switched.plain() && plain_decode(&req, lora_program, fire_wide_mask);
    let mut parts = Vec::with_capacity(group_sets.len());
    for ((((((set, (log_seq, log_pages, boundary)), lease), kvtxn), (_, lanes)), &stated), &end) in
        group_sets
            .iter()
            .zip(logged)
            .zip(leases)
            .zip(kvtxns)
            .zip(&groups)
            .zip(&stated)
            .zip(&write_ends)
    {
        // Each lane's mask speaks of the slots from the set's boundary on,
        // and its rows land between it and what `reserve` granted: the
        // slots of the pages the fire made the set's own.
        for &lane in lanes {
            if let Some(lane) = req.lanes.get_mut(lane) {
                lane.kv.folded = u32::try_from(boundary).unwrap_or(u32::MAX);
                lane.kv.write_end = end;
            }
        }
        let mut tags = Vec::new();
        let rows = match log_pages.filter(|_| plain_decode) {
            Some(pages) => LogRows::Reported {
                tag: report_on(&mut req, 0, &mut tags),
                pages,
                page_size: stores.kv_page_size,
            },
            None => LogRows::Unknown,
        };
        let reported = matches!(rows, LogRows::Reported { .. });
        if let (DeviceFold::PastRows(_), false) = (stated, reported) {
            let first = lanes[0];
            let reason = past_rows_refusal(first, split[first + 1] - split[first]);
            record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
            return Ok(Err(reason));
        }
        // What the host could place was frozen as the fire was prepared; the
        // rest is placed when the fire settles.
        let rs = rs_ws_ids.get(lanes[0]).copied().filter(|_| !windowed);
        let by = device_fold_at(
            &pass_lanes,
            lanes[0],
            stated,
            reported,
            rs,
            &mut req,
            &mut tags,
        );
        // A set with a recurrent state counts the fire's rows unfolded
        // until it settles, and each of its lanes reports where its rows
        // ended.
        let staged = frozen[lanes[0]].map(|_| StagedFire {
            rows: crate::inferlet::host::working_set::StagedRows::new(
                lanes
                    .iter()
                    .map(|&lane| split[lane + 1] - split[lane])
                    .sum(),
            ),
            tags: lanes
                .iter()
                .map(|&lane| report_on(&mut req, lane, &mut tags))
                .collect(),
        });
        parts.push(FireWs {
            ws: set.id,
            kvtxn,
            log: Some(LogFire { seq: log_seq, rows }),
            lease,
            fold: by.map(|by| FoldAt {
                by,
                decoded: plain_decode,
            }),
            reports: tags,
            staged,
        });
    }
    debug_assert_eq!(
        untagged_device_fold(&req),
        None,
        "a lane whose fold the device counts asks for its report"
    );
    rs::apply_shared(&mut req, stores.shared.as_ref(), Some(&rs_prepared));
    let submit_error = crate::scheduler::submit_prebuilt_tracked_async_with_kv_and_rs_copy_on(
        &scheduler,
        req,
        instance_id,
        pid,
        quorum_pipeline_id,
        completion.clone(),
        copy_src,
        copy_dst,
        rs_prepared.windowed_copies(),
        rs_copy,
        frame,
        lora_program,
    )
    .err()
    .map(|error| format!("{error:#}"));
    if let Some(error) = submit_error {
        let reason = format!("pipeline: device-geometry submit failed: {error}");
        record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
        return Ok(Err(reason));
    }
    let unqueued = unfired.submitted();
    epoch.retire_on(&completion);
    for part in &mut parts {
        part.kvtxn.retire_on(&completion);
    }
    mark_device_folds(&stores, &completion, &mut parts);
    unsettle(&stores, &completion, &mut parts);
    if let Some(ask) = rung_ask {
        ask.until_settled(&completion);
    }
    ticket_reservation.commit();
    advance_shadow(ctx, &fwd)?;
    // Rows that start at the folded length and are folded whole end there;
    // any others end where the device put them.
    for (part, (_, lanes)) in parts.iter().zip(&groups) {
        let first = lanes[0];
        let (place, cell) = (&pass_lanes.places[first], &pass_lanes.cursors[first]);
        let rows = split[first + 1] - split[first];
        let end = place.folded.saturating_add(rows);
        let folded_whole = appends_folded(&pass_lanes, lanes)
            && matches!(
                &part.fold,
                Some(FoldAt { by: FoldBy::To(to), .. }) if *to == u64::from(end)
            );
        advance_cursor(cell, folded_whole.then_some(end), None);
        if let Some(staged) = &part.staged {
            cell.lock().unwrap().stage(staged.rows.clone());
        }
    }

    unqueued.queue(
        &pipe_fires,
        PendingFire {
            completion,
            checkpoints: Vec::new(),
            kv: parts,
            device_geometry: true,
            rstxn: rstxns,
            model: ws.model,
            engine: ws.engine,
            fwd_rep,
            cells,
            failure: pipeline_failure,
            retired: None,
            lane: Some(quorum_pipeline_id),
        },
    );
    Ok(Ok(()))
}

fn wire_channels_to_pipeline<C: FireContext>(
    ctx: &mut C,
    fwd: &Resource<ForwardPass>,
    pipe_fires: &PendingFires,
) -> Anyhow<Result<(), String>> {
    if let Some(existing) = &ctx.resources().get(fwd)?.fires
        && !Arc::ptr_eq(existing, pipe_fires)
    {
        return Ok(Err(
            "pipeline: a pass cannot submit across different pipelines".into(),
        ));
    }
    let reps = ctx.resources().get(fwd)?.channel_reps.clone();
    for rep in reps {
        let cres: Resource<Channel> = Resource::new_borrow(rep);
        if let Ok(ch) = ctx.resources().get_mut(&cres) {
            match &ch.fires {
                Some(existing) if !Arc::ptr_eq(existing, pipe_fires) => {
                    return Ok(Err("pipeline: a channel is shared across pipelines \
                         (all passes binding a channel must submit on the same \
                         pipeline)"
                        .into()));
                }
                _ => ch.fires = Some(pipe_fires.clone()),
            }
        }
    }
    ctx.resources().get_mut(fwd)?.fires = Some(pipe_fires.clone());
    Ok(Ok(()))
}

#[cfg(test)]
#[path = "fire/tests/lifecycle_tests.rs"]
pub(crate) mod lifecycle_tests;

#[cfg(test)]
#[path = "fire/tests/static_admission_tests.rs"]
mod static_admission_tests;

/// A piece wider than the rung never waits on a rung that does not come:
/// one of the held width fires at once, one wider asks its rise as a keyed
/// demand, and a pass whose rise is short of free room fires at the rung
/// held until room is given back.
#[cfg(test)]
#[path = "fire/tests/rung_wait_tests.rs"]
mod rung_wait_tests;

#[cfg(test)]
#[path = "fire/tests/cut_tests.rs"]
mod cut_tests;

#[cfg(test)]
#[path = "fire/tests/fold_tests.rs"]
pub(crate) mod fold_tests;

#[cfg(test)]
#[path = "fire/tests/unfolded_tests.rs"]
mod unfolded_tests;

#[cfg(test)]
#[path = "fire/tests/mask_cut_tests.rs"]
mod mask_cut_tests;

#[cfg(test)]
#[path = "fire/tests/writable_tests.rs"]
mod writable_tests;

#[cfg(test)]
#[path = "fire/tests/alone_tests.rs"]
mod alone_tests;
