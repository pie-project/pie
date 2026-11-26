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
mod beam;
mod cuts;
mod device;
mod finalize;
mod lanes;
mod promote;
mod reserve;
mod retract;
mod rows;
mod rung;
mod submit;

pub(crate) use beam::*;
pub(crate) use cuts::*;
use device::*;
pub use finalize::*;
pub(crate) use lanes::*;
pub(crate) use promote::*;
use reserve::*;
pub use retract::*;
pub(crate) use rows::*;
use rung::*;
pub use submit::*;

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

/// Runs `settle`, a landed fire's kv settle, off an engine lane: a lane
/// resolves its fires' completions as they land, and launches its next step
/// after; settled there, it waited on the kv lock every guest's fire takes,
/// with the device idle. Nothing the next launch reads comes from a settle
/// (what it frees is reused only by grants after it), so settles go in the
/// order they are asked, on one thread of their own. Asked elsewhere, a
/// settle runs where it is asked.
fn settled_off_lane(settle: impl FnOnce() + Send + 'static) {
    type Settle = Box<dyn FnOnce() + Send>;
    static SETTLER: std::sync::LazyLock<std::sync::Mutex<std::sync::mpsc::Sender<Settle>>> =
        std::sync::LazyLock::new(|| {
            let (tx, rx) = std::sync::mpsc::channel::<Settle>();
            crate::rt::thread::Builder::new()
                .name("pie-kv-settle".to_string())
                .spawn(move || {
                    while let Ok(settle) = rx.recv() {
                        settle();
                    }
                })
                .expect("the settle thread starts");
            std::sync::Mutex::new(tx)
        });
    let on_lane = std::thread::current()
        .name()
        .is_some_and(|name| name.starts_with("pie-engine-"));
    if !on_lane {
        return settle();
    }
    if let Err(std::sync::mpsc::SendError(settle)) = SETTLER.lock().unwrap().send(Box::new(settle)) {
        settle();
    }
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
            Some(completion) => {
                completion.on_settled(move || settled_off_lane(move || Self::settle(model, engine, txn)));
            }
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
        completion.on_settled(move || settled_off_lane(move || drop(self)));
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
        .filter(|checkpoint| !checkpoint.snapshot)
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

/// Forgets the boot's landed records, for [`crate::bootstrap::reset`].
pub(crate) fn reset() {
    KV_WRITES.lock().unwrap().clear();
    BEAM_COMMON.lock().unwrap().clear();
    host_geometry::reset();
    promote::reset();
}

pub(crate) fn record_kv_writes(writes: Vec<::engine::KvWrites>) {
    let mut held = KV_WRITES.lock().unwrap();
    for written in writes {
        held.insert(written.tag, written);
    }
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
        stores
            .rs
            .lock()
            .unwrap()
            .checkpoint(checkpoint.rs, checkpoint.key, owner)
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
    for (row, &position) in written.positions.iter().enumerate() {
        let at = first + row as u64;
        if slot(row)? != at || u64::from(position) != at {
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
        let model = crate::model::model();
        let facts = model.facts();
        !self.rows_read
            && self.lanes.iter().flatten().all(|(name, value)| {
                facts
                    .declared(name)
                    .is_some_and(|decl| decl.default_value() == value)
            })
    }
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
