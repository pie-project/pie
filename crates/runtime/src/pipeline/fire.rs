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

/// `windowed`: the fire reads window slots, which come back only once every
/// fire that read them settles (its process finalizes it). A fire that would
/// park finalizes its own tail as it lands while it waits, rather than
/// settle it whole first: the tail may hold this pass's earlier pieces,
/// which make up one frame with the piece now asking, and settling them
/// before this ask returns waited on a frame only this ask completes (Mac's
/// gemma-4-e4b c64 stall). The lane is parked first, so the scheduler waits
/// on none of its fires.
async fn acquire_grant<C: FireContext>(
    ctx: &mut C,
    pipeline_id: uuid::Uuid,
    demand: crate::planner::Demand,
    windowed: bool,
) -> Result<Option<crate::planner::AllocationGrant>, String> {
    if demand.is_zero() {
        return Ok(Some(crate::planner::AllocationGrant::empty()));
    }
    let pid = ctx.process_id();
    let Some(planner) = crate::planner::planner_of(pid) else {
        return Err("pipeline: KV residency planner is not installed".to_string());
    };
    if windowed {
        if let Some(grant) = planner.grant_now(pid, demand) {
            planner.note_peak(pid, demand);
            return Ok(Some(grant));
        }
        crate::scheduler::worker::notify_lane_close(pipeline_id, Some(pid));
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
    match acquired.map_err(|error| format!("pipeline: KV capacity: {error}"))? {
        crate::planner::Acquired::Granted(grant) => Ok(Some(grant)),
        crate::planner::Acquired::Yield => {
            settle_and_wait_resident(ctx).await?;
            Ok(None)
        }
        crate::planner::Acquired::Requote => Ok(None),
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
    for (_, lanes) in groups {
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
        let snapshots = match (&writes, groups.len()) {
            (None, 1) => snapshots.to_vec(),
            _ => Vec::new(),
        };
        plans.push(rs::GroupPlan {
            lanes: lanes.clone(),
            writes,
            fold,
            snapshots,
        });
    }
    Ok(rs::RsPlan::Groups(plans))
}

/// Each working set buffers every slot from where its state ends up to what it
/// reserved.
fn device_rs_plan(
    stores: &crate::store::registry::Stores,
    ids: &[crate::store::rs::RsWorkingSetId],
    groups: &LaneGroups,
    lanes: &PassLanes,
    fold: &geometry::Fold,
) -> Result<rs::RsPlan, String> {
    if ids.is_empty() {
        return Ok(rs::RsPlan::Groups(Vec::new()));
    }
    let store = stores.rs.lock().unwrap();
    let mut plans = Vec::with_capacity(groups.len());
    for (_, group) in groups {
        let first = group[0];
        let frozen = store
            .frozen(ids[first])
            .map_err(|error| error.to_string())?;
        let reserved = lanes.cursors[first].lock().unwrap().reserved;
        let fold = group_fold(fold, first, frozen, reserved);
        // The device picks where a lane's rows start: what it carries in is
        // bounded by what stands unfolded in its set ([`device_lane_rows`]).
        plans.push(rs::GroupPlan {
            lanes: group.clone(),
            writes: Some(frozen..reserved),
            fold,
            snapshots: Vec::new(),
        });
    }
    Ok(rs::RsPlan::Groups(plans))
}

/// A lane's rows in a fire, or a fire's (its lanes' summed): its own, and
/// the recurrent rows it carries in unfolded, which earlier fires wrote and
/// nothing has folded, and which it replays ahead of its own
/// ([`::engine::fire::rung_rows`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct FireRows {
    own: u32,
    carried: u32,
}

impl FireRows {
    /// What a fire of these rows asks the activation rung for: a rung
    /// holds its tokens of own rows and as many carried ones.
    fn ask(self) -> u32 {
        ::engine::fire::rung_rows(self.own, self.carried)
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
/// row.
fn lane_rows(start: u32, end: u32, frozen: Option<u32>) -> FireRows {
    FireRows {
        own: end.saturating_sub(start),
        carried: frozen.map_or(0, |frozen| start.saturating_sub(frozen)),
    }
}

/// The `own` rows of a lane the device places, and a bound on what it
/// carries, its recurrent state folded through `frozen` (`None`: it has
/// none, and carries nothing). Its first row is its program's, resolved on
/// the device after its set's earlier fires have run, so no count exists
/// here; it replays no more than the rows standing unfolded in its set
/// then: those written past `frozen`, up to the set's written end, by
/// fires that settled, and every row of its fires staged and still to
/// settle ([`::engine::fire::rung_rows`], device geometry).
fn device_lane_rows(
    own: u32,
    cursor: &crate::inferlet::host::working_set::Cursor,
    frozen: Option<u32>,
) -> FireRows {
    FireRows {
        own,
        carried: frozen.map_or(0, |frozen| cursor.unfolded(frozen)),
    }
}

/// The frozen length of each lane's recurrent state, by the lane's group
/// (`None` a lane where the pass binds none, or on a model with window
/// lists, whose rows trim behind their windows and are never replayed).
fn lanes_frozen(
    stores: &crate::store::registry::Stores,
    ids: &[crate::store::rs::RsWorkingSetId],
    groups: &LaneGroups,
    lanes: usize,
    windowed: bool,
) -> Result<Vec<Option<u32>>, String> {
    let mut frozen = vec![None; lanes];
    if ids.is_empty() || windowed {
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

/// The most carried rows a fire holds on `engine`: the tokens of the
/// largest activation rung fires may take, whose replay scratch holds as
/// many; what one fire holds where the pool states no rung.
fn largest_rung(stores: &crate::store::registry::Stores, engine: usize) -> u32 {
    let holds = crate::engine::get_spec(engine).map_or(u32::MAX, |spec| {
        u32::try_from(spec.limits.max_forward_tokens).unwrap_or(u32::MAX)
    });
    stores.shared.as_ref().map_or(holds, |shared| {
        shared.lock().unwrap().rung_for(u32::MAX).min(holds)
    })
}

/// A fire carrying more recurrent rows than the largest rung's tokens is
/// refused by name, before anything is submitted: no rung's replay scratch
/// holds them. Up to there it raises the rung as a wider fire does.
fn check_largest_rung(rows: FireRows, largest: u32) -> Result<(), String> {
    if rows.carried <= largest {
        return Ok(());
    }
    Err(format!(
        "pipeline: a fire would carry {} recurrent rows unfolded into it, past the {largest} \
         tokens of the largest activation rung, whose replay scratch holds as many; fold \
         what its lanes have written before they write more",
        rows.carried
    ))
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

/// `PIE_FIRE_ROWS` overrides it, so a test compares outputs cut at the same
/// rows.
fn prefill_share(page: u32, engine: usize) -> u32 {
    if let Some(rows) = fire_rows() {
        return rows;
    }
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
    if let Some(fire) = fire_rows() {
        return fire.min(u32::try_from(rows).unwrap_or(u32::MAX));
    }
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

/// A pass's ask for the activation rung funded as a planner waiter: the
/// room the raise lacks is asked of the planner under the pass's process,
/// so it is freed as any waiter's pages are (idle cache, processes moved to
/// the host, the policy's victims); once granted the room is given back and
/// the rung raised. Runs beside the pass, which keeps firing at the rung it
/// has, until the raise lands, the pass is through (dropped) or the planner
/// refuses (`failed`).
struct RaiseFunding {
    stop: Arc<tokio::sync::Notify>,
    stopped: Arc<std::sync::atomic::AtomicBool>,
    failed: Arc<Mutex<Option<String>>>,
}

impl RaiseFunding {
    fn start(
        shared: crate::store::shared::Shared,
        engine: usize,
        pid: uuid::Uuid,
        quorum: uuid::Uuid,
        rows: u32,
    ) -> Self {
        let stop = Arc::new(tokio::sync::Notify::new());
        let stopped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let failed = Arc::new(Mutex::new(None));
        let funding = Self {
            stop: stop.clone(),
            stopped: stopped.clone(),
            failed: failed.clone(),
        };
        crate::rt::spawn(async move {
            let Some(planner) = crate::planner::planner_of(pid) else {
                tracing::debug!(%pid, "a raise's funding finds no planner");
                return;
            };
            let target = crate::store::shared::cap_for(&shared, rows);
            // A step with nothing to do waits on the rung's next change (a
            // raise, a copy, the cap, room freed), armed at its top, or the
            // stop.
            while !stopped.load(std::sync::atomic::Ordering::Acquire) {
                let watch = crate::store::shared::rung_watch(engine);
                tokio::pin!(watch);
                watch.as_mut().enable();
                crate::store::shared::raise_now(&shared, engine, rows);
                if crate::store::shared::fire_cap(engine) >= target {
                    return;
                }
                // Quoted at the rung the raise is taken at: the one the
                // engine holds where that is wider
                // (`SharedPool::raise_at`).
                let pages = {
                    let pool = shared.lock().unwrap();
                    let goal = pool.rung_for(rows);
                    pool.raise_at(goal).map(|at| pool.raise_short(at).1)
                };
                let Some(pages) = pages else {
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
                if pages == 0 {
                    // The room is there and idle cache lies in the span:
                    // the walker has it cleared and tells the rung.
                    tokio::select! {
                        () = &mut watch => {}
                        () = stop.notified() => return,
                    }
                    continue;
                }
                let demand = crate::planner::Demand {
                    kv_pages: pages,
                    rs: crate::store::rs::RsUnits::default(),
                };
                tracing::debug!(engine, rows, pages, "a raise asks the planner for room");
                let asked = tokio::select! {
                    asked = planner.acquire(pid, quorum, demand) => asked,
                    () = stop.notified() => return,
                };
                match asked {
                    Ok(crate::planner::Acquired::Granted(grant)) => {
                        // The room is given back and the rung takes it at
                        // once, before the pass's own pieces open it again.
                        // Given back first: its pages go back through the
                        // pool's own lock.
                        drop(grant);
                        let mover = crate::store::shared::programs::LaneMover { engine };
                        let taken = crate::store::shared::programs::take_raise(
                            &shared,
                            |pool| pool.rung_for(rows),
                            &mover,
                        )
                        .await;
                        if let Err(error) = taken {
                            tracing::error!(engine, %error, "a raise's carve was refused");
                            *failed.lock().unwrap() = Some(error);
                            crate::store::shared::rung_changed(engine);
                            return;
                        }
                        crate::store::shared::raise_now(&shared, engine, rows);
                    }
                    Ok(crate::planner::Acquired::Requote) => {
                        tracing::debug!(engine, "a raise's room is quoted again");
                    }
                    Ok(crate::planner::Acquired::Yield) => {
                        tracing::debug!(engine, "a raise's process yields");
                        tokio::select! {
                            _ = planner.wait_resident(pid) => {}
                            () = stop.notified() => return,
                        }
                    }
                    Err(error) => {
                        tracing::debug!(engine, %error, "a raise's room is refused");
                        *failed.lock().unwrap() = Some(error.to_string());
                        crate::store::shared::rung_changed(engine);
                        return;
                    }
                }
            }
        });
        funding
    }

    fn failed(&self) -> Option<String> {
        self.failed.lock().unwrap().clone()
    }
}

/// Waits until fires on `engine` may take `width` rows, while `failed`
/// (the pass's raise funding) has not said the room was refused; at once
/// where nothing funds a raise (no shared pool: the cap never moves).
async fn wait_for_rung(
    engine: usize,
    width: u32,
    failed: Option<impl Fn() -> Option<String>>,
) -> Result<(), String> {
    let Some(failed) = failed else {
        return Ok(());
    };
    // The cap moving and the funding failing both tell the rung.
    let mut refused = None;
    crate::store::shared::rung_wait_until(engine, || {
        refused = failed();
        refused.is_some() || crate::store::shared::fire_cap(engine) >= width
    })
    .await;
    if crate::store::shared::fire_cap(engine) >= width {
        return Ok(());
    }
    Err(format!(
        "pipeline: a fire of {width} rows needs a wider activation rung, and its room was \
         refused: {}",
        refused.unwrap_or_default()
    ))
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
    /// Held (the walker cuts nothing under it) until dropped.
    want: crate::store::shared::RungWant,
    /// The room the rung lacks for it, asked of the planner beside the
    /// pass; `None` where fires may take the rows now.
    funding: Option<RaiseFunding>,
}

/// Asks the rung on `engine` for the `rows` a fire of a pass of `own` rows
/// takes ([`RungAsk`]; `None`: a lone row the cap covers): the rung is
/// raised at once where nothing moves, a cap the walker lowered behind a
/// rung still up is lifted back, and what the raise lacks is funded under
/// the pass's process.
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
            .then(|| RaiseFunding::start(shared.clone(), engine, pid, quorum, rows))
    });
    Some(RungAsk {
        engine,
        pid,
        want,
        funding,
    })
}

impl RungAsk {
    /// Returns once fires may take `width` rows, or with what the planner
    /// refused the raise: a fire wider than the rung waits for the raise
    /// its pass funds, never on a rung that does not come.
    async fn wait(&self, width: u32) -> Result<(), String> {
        crate::inferlet::process::said(
            "enqueue: the activation rung",
            self.pid,
            wait_for_rung(
                self.engine,
                width,
                self.funding
                    .as_ref()
                    .map(|funding| move || funding.failed()),
            ),
        )
        .await
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

fn fire_rows() -> Option<u32> {
    static ROWS: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| {
        std::env::var("PIE_FIRE_ROWS")
            .ok()
            .and_then(|rows| rows.parse().ok())
            .filter(|&rows| rows > 0)
    })
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
    let rows = |lane: &::engine::Lane| u32::try_from(lane.tokens.len()).unwrap_or(u32::MAX);
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
fn rows_of_lanes(
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
            // A row's mask names absolute keys: it is the same in any piece.
            if let Some(::engine::Masking::Rows(masks)) = &mut lane.mask {
                *masks = masks[a..b].to_vec();
            }
            lane.kv.held += from;
            let len = lane.kv.held + (to - from);
            lane.kv.pages.truncate(len.div_ceil(page.max(1)) as usize);
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

/// Several lanes are cut alike and aligned at their ends, so the last piece
/// holds every lane's read-out rows.
fn pass_pieces(
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
/// for (`asked`), and with `auto` the pass's end, each floored to whole pages
/// and within the `keyed` pages the pass keys.
pub(crate) fn checkpoint_rows(
    asked: &[u32],
    rows: std::ops::Range<u64>,
    page_size: u32,
    keyed: usize,
    auto: bool,
) -> Vec<u64> {
    let page = u64::from(page_size.max(1));
    let mut at: Vec<u64> = asked
        .iter()
        .map(|&row| u64::from(row) / page * page)
        .chain(auto.then_some(rows.end / page * page))
        .filter(|&row| rows.start < row && row <= rows.end && row / page <= keyed as u64)
        .collect();
    at.sort_unstable();
    at.dedup();
    at
}

static SEALED_PAGES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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
/// last row's page, which is read out so always computed.
fn supply(
    stores: &crate::store::registry::Stores,
    ws: crate::store::kv::page_table::WorkingSetId,
    full: &crate::engine::FireRequest,
    slots: &[crate::store::kv::hash::Hash256],
    attach: Attach,
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
        let run = kv.cached_run(&keys);
        let clean = held.is_multiple_of(page) && kv.mapped_len(ws)? == held / page;
        let mapped = match attach {
            Attach::Kv if clean && kv.keeps() => {
                let mapped = kv.attach_run(ws, &run, limit, page_size)?;
                kv.log_rows(ws, held, tokens_of(held / page + mapped))?;
                mapped
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
            run.len().min(limit as usize),
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
    let found = (run.len() as u64 * page).min(total - 1);
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
    let Some(mut grant) = acquire_grant(ctx, pipeline_id, demand, false).await? else {
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
        txn: Some(txn),
        in_flight: Vec::new(),
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
        guard.in_flight.push(completion.clone());
        completion.wait().await.map_err(|error| error.to_string())?;
    }
    guard.in_flight.clear();
    let txn = guard.txn.take().expect("taken once");
    let pin =
        crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| kv.commit_promote(txn))
            .map_err(|error| error.to_string())?;
    Ok(Some(Promoted {
        stores: stores.clone(),
        pin: Some(pin),
        _lease: lease,
    }))
}

/// Dropped before it commits, it gives back what it took once any copy it
/// started has landed.
struct PromoteGuard {
    stores: crate::store::registry::Stores,
    txn: Option<crate::store::kv::KvResidencyTxn>,
    in_flight: Vec<crate::scheduler::ControlCompletion>,
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
        if in_flight.is_empty() || !crate::rt::has_runtime() {
            abort();
            return;
        }
        crate::rt::spawn(async move {
            for completion in in_flight {
                let _ = completion.wait().await;
            }
            abort();
        });
    }
}

struct Promoted {
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

/// Resumes a hybrid pass from its furthest checkpoint; none when either half is
/// gone by then, and `rs` holds no state again.
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
        kv.attach_run(ws, &run, pages, page_size)
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
}

/// A demand that drifted from its grant is quoted again. What it binds
/// opens the fire's epoch on engine `engine` of model `model`.
#[allow(clippy::too_many_arguments)]
async fn reserve<C: FireContext>(
    ctx: &mut C,
    (model, engine): (usize, usize),
    stores: &crate::store::registry::Stores,
    pipeline_id: uuid::Uuid,
    sets: &[(&KvWorkingSet, Written)],
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
        let Some(mut grant) = acquire_grant(ctx, pipeline_id, demand, windowed).await? else {
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
    To(u64),
    /// The cursor is exact again once the last fire of its working set placed
    /// this way settles.
    Settled {
        count: Option<u32>,
        tags: Vec<u64>,
        reach: u64,
        cursor: std::sync::Arc<std::sync::Mutex<crate::inferlet::host::working_set::Cursor>>,
        rs: Option<RsDeviceFold>,
    },
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
            submitted: None,
        }
    }

    fn submitted(
        &mut self,
        stores: &crate::store::registry::Stores,
        completion: &crate::engine::WorkItemCompletion,
    ) {
        stores.rs.lock().unwrap().begin_device_fold(self.ws);
        self.submitted = Some((stores.rs.clone(), completion.clone()));
    }

    /// `slots` the fold froze, where the fire ran and reported them;
    /// `failure` why it did not run.
    fn settle(mut self, slots: Option<u32>, failure: Option<&str>) {
        if let Some((store, _)) = self.submitted.take() {
            store
                .lock()
                .unwrap()
                .end_device_fold(self.ws, slots, failure);
        }
    }
}

impl Drop for RsDeviceFold {
    fn drop(&mut self) {
        let Some((store, completion)) = self.submitted.take() else {
            return;
        };
        let ws = self.ws;
        completion.on_settled(move || {
            store
                .lock()
                .unwrap()
                .end_device_fold(ws, None, Some(Self::DROPPED));
        });
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

/// A snapshot no checkpoint takes is released.
pub(crate) fn take_checkpoint(
    stores: &crate::store::registry::Stores,
    checkpoint: &TakeCheckpoint,
    success: bool,
) {
    let release = || {
        if checkpoint.snapshot {
            stores.rs.lock().unwrap().release_working_set(checkpoint.rs);
        }
    };
    if !success {
        return release();
    }
    let owner = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
        kv.owner_of(checkpoint.kv)
    });
    let taken = if checkpoint.snapshot {
        stores
            .rs
            .lock()
            .unwrap()
            .keep(checkpoint.rs, checkpoint.key, owner);
        Ok(true)
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
        }
        Ok(false) => {}
        Err(error) => tracing::debug!(%error, "pipeline: keeping a checkpoint"),
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
/// way has placed its fold. Until then the frozen length the cursor states
/// is a bound, and a child that took it would state it for good: no fire
/// settles a copy.
///
/// The cursor is the mark, on every set; the recurrent store's is only on a
/// set that keeps recurrent state. The fires are `fires`', the pipeline the
/// set is scoped to, and only this task settles them: they run to their end
/// first, woken as each completes. One dropped unsettled lifts the
/// recurrent store's mark when the engine is done with it, which wakes the
/// wait. One another finalizer took off the queue is under the finalizer
/// gate until its fold is placed and its failure latched: the cursor and
/// the failure are read with the gate in hand.
///
/// A fire that failed refuses the copy as the pipeline's failure. A fold no
/// fire is left to place leaves the length unknown for good, and the copy
/// is refused by that name ([`Cursor::UNPLACED`]) rather than left waiting.
///
/// [`Cursor::UNPLACED`]: crate::inferlet::host::working_set::Cursor::UNPLACED
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
    let at_rest = cursor.lock().unwrap().at_rest().is_ok();
    if !at_rest || marked {
        drain_pipeline_fires(ctx, fires).await?;
        if let Some(rs) = rs {
            crate::store::rs::device_folds_settled(&stores.rs, &[rs]).await;
        }
    }
    let _gate = fires.finalize_guard().await;
    if let Some(reason) = failure.lock().unwrap().clone() {
        return Ok(Err(format!("pipeline failed: {reason}")));
    }
    let cursor = cursor.lock().unwrap().clone();
    Ok(match cursor.at_rest() {
        Ok(()) => Ok(cursor),
        // The recurrent store says how the fire ended, where it keeps one.
        Err(why) => Err(rs
            .and_then(|rs| stores.rs.lock().unwrap().device_fold_gate(rs).err())
            .map_or_else(|| format!("working set: {why}"), |named| named.to_string())),
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
    let media: Vec<u32> = req.media.iter().map(|media| media.lane).collect();
    for (at, lane) in req.lanes.iter_mut().enumerate() {
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
            .capturing_scores(on.contains(&poem_ir::Builtin::Scores))
            .with_media(media.contains(&(at as u32)))
            .starting(starts(lane));
        let switches = of
            .get(at)
            .and_then(|&pass_lane| switched.lanes.get(pass_lane))
            .map_or(&[][..], Vec::as_slice);
        let request = switches.iter().fold(request, |request, (name, value)| {
            request.switched(name, value)
        });
        lane.word = model.word(&request);
    }
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

/// Also how long a planner waiter stands short of room only the policy can
/// make before it fails.
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
        freed.await;
    }
}

pub(crate) fn map_lane_pages(
    req: &mut crate::engine::FireRequest,
    stores: &crate::store::registry::Stores,
    groups: &LaneGroups,
) -> Result<(), String> {
    for (ws, lanes) in groups {
        if lanes.iter().all(|&lane| {
            req.lanes
                .get(lane)
                .is_none_or(|lane| lane.kv.pages.is_empty())
        }) {
            continue;
        }
        let table = working_set_flat_table(stores, *ws)?;
        for &lane in lanes {
            let Some(lane) = req.lanes.get_mut(lane) else {
                continue;
            };
            map_pages(&mut lane.kv.pages, &table)?;
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

fn stamp_lane_translation(
    req: &mut crate::engine::FireRequest,
    stores: &crate::store::registry::Stores,
    groups: &LaneGroups,
) -> Result<(), String> {
    for (ws, lanes) in groups {
        let table = working_set_flat_table(stores, *ws)?;
        for &lane in lanes {
            if let Some(lane) = req.lanes.get_mut(lane) {
                lane.kv.translation.clone_from(&table);
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
    let mut places = Vec::with_capacity(lanes.len());
    let mut cursors = Vec::with_capacity(lanes.len());
    for &rep in &lanes {
        let res: Resource<WorkingSet> = Resource::new_borrow(rep);
        let cursor = ctx.resources().get(&res)?.cursor.clone();
        let now = cursor.lock().unwrap().clone();
        places.push(geometry::Place {
            folded: now.folded,
            folded_exact: now.folded_exact,
            tail: now.tail,
            shared: lanes.iter().filter(|&&other| other == rep).count() > 1,
            spans: now.spans,
        });
        cursors.push(cursor);
    }
    let mut kv = Vec::with_capacity(kv_lanes.len());
    for &rep in &kv_lanes {
        let res: Resource<KvWorkingSet> = Resource::new_borrow(rep);
        kv.push(ctx.resources().get(&res)?.clone());
    }
    Ok(PassLanes {
        places,
        cursors,
        kv,
    })
}

type CursorCell = std::sync::Arc<std::sync::Mutex<crate::inferlet::host::working_set::Cursor>>;

/// Working sets in `settles` freeze where the fire settles.
fn advance_cursors(
    lanes: &PassLanes,
    landings: Option<&[geometry::Landing]>,
    matched: &[crate::pipeline::media::MatchedRun],
    fold: &geometry::Fold,
    settles: &[CursorCell],
) {
    let mut done: Vec<&CursorCell> = Vec::new();
    for (lane, cell) in lanes.cursors.iter().enumerate() {
        if done.iter().any(|seen| std::sync::Arc::ptr_eq(seen, cell)) {
            continue;
        }
        done.push(cell);
        let mine: Vec<usize> = (lane..lanes.cursors.len())
            .filter(|&other| std::sync::Arc::ptr_eq(&lanes.cursors[other], cell))
            .collect();
        let mut cursor = cell.lock().unwrap();
        if settles
            .iter()
            .any(|later| std::sync::Arc::ptr_eq(later, cell))
        {
            cursor.unsettled += 1;
            cursor.folded_exact = false;
        }
        let Some(landings) = landings else {
            cursor.tail = None;
            continue;
        };
        let end = mine
            .iter()
            .filter_map(|&lane| landings.get(lane))
            .map(|landing| landing.end())
            .max()
            .unwrap_or(0);
        cursor.tail = Some(end);
        cursor.reserved = cursor.reserved.max(end);
        cursor.written = cursor.written.max(end);
        let start = mine
            .iter()
            .filter_map(|&lane| landings.get(lane))
            .map(|landing| landing.start)
            .min()
            .unwrap_or(end);
        let spans = matched
            .iter()
            .filter(|run| mine.contains(&(run.lane as usize)))
            .map(|run| {
                let at = landings[run.lane as usize].start + run.anchor;
                let lag = run.span.token_count.saturating_sub(run.span.position_span);
                (at, at + run.rows, lag)
            })
            .filter(|&(_, _, lag)| lag > 0);
        cursor.rewrite(start, spans);
        // Exact, or a bound below the fold still to be placed.
        if let Some(count) = fold.of(mine[0]) {
            let folded = cursor.folded.saturating_add(count).min(end);
            cursor.folded = cursor.folded.max(folded);
        }
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "the fold, where it ends, and the request whose lanes report it"
)]
fn fold_at(
    lanes: &PassLanes,
    group: &[usize],
    reach: Option<u64>,
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
    let count = fold.of(first);
    if count == Some(0) {
        return None;
    }
    if place.folded_exact
        && let (Some(count), Some(reach)) = (count, reach)
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
    let cursor = lanes.cursors[first].clone();
    let reach = reach
        .unwrap_or_else(|| u64::from(cursor.lock().unwrap().reserved))
        .min(end);
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
        reach,
        cursor,
        rs,
    })
}

fn settling(parts: &[FireWs]) -> Vec<CursorCell> {
    parts
        .iter()
        .filter_map(|part| match &part.fold {
            Some(FoldAt {
                by: FoldBy::Settled { cursor, .. },
                ..
            }) => Some(cursor.clone()),
            _ => None,
        })
        .collect()
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
) -> Anyhow<Result<(), String>> {
    let submit_probe = crate::scheduler::probe::host_submit();
    let submit_clock = crate::scheduler::probe::ProbeClock::start();
    {
        if ctx.resources().get(&fwd)?.devgeo.is_some() {
            return fire_device_geometry(ctx, this, fwd, frame).await;
        }
        if ctx.resources().get(&fwd)?.float.is_some() {
            return float::fire_float_lane(ctx, this, fwd, frame).await;
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
        // A long host pass runs as several fires; all but the last run the
        // model's forward alone. A recurrent state folding all its rows, or
        // none, folds each fire's rows in turn.
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
            && folds_all;
        let mut full = req;
        let keyed = if keyable { keyed_rows(&full) } else { 0 };
        let mut pass_slots = (keyed > 0)
            .then(|| {
                let lane = &full.lanes[0];
                let start = u64::from(lane.kv.held);
                let root = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
                    kv.chain_root(ws.id, start, page_size)
                        .map(|root| root.map(|prev| (kv.domain(), prev)))
                });
                let (domain, prev) = root.ok().flatten()?;
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
        let attach = if !cuttable || !keyable || !plain_rows(&full) {
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
        {
            let lane = &full.lanes[0];
            // Pages another pass is computing now are waited for, not
            // computed again.
            coalesced = crate::pipeline::coalesce::before_pass(
                &stores,
                ws.id,
                u64::from(lane.kv.held),
                slots,
            )
            .await;
            crate::offload::prefill::before_pass(
                pid,
                &stores,
                engine,
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
            Some(slots) => supply(&stores, ws.id, &full, slots, attach, page_size),
            None => Supplied::default(),
        };
        drop(promoted);
        if pass_slots.is_some() {
            // It looked, so what it found is measured, nothing included.
            ctx.note_cached(u64::from(supplied));
            crate::policy::report(crate::policy::Event::Cached(pid, u64::from(supplied)));
        }
        if supplied > 0 {
            // What was mapped in is skipped: the pass runs from there.
            let rows = u32::try_from(full.lanes[0].tokens.len()).unwrap_or(u32::MAX);
            full = rows_of_lanes(&full, &[Some((supplied, rows))], true, page_size);
            stamp_lane_words(&mut full, &switched, &lanes_of, false, None);
            pass_slots = pass_slots.map(|slots| slots[supplied as usize..].to_vec());
        }
        // Where the pass keeps its state: the checkpoint rows it reaches,
        // floored to whole pages it keys, each the end of a fire of its own.
        let held = u64::from(full.lanes[0].kv.held);
        let reach = held + full.lanes[0].tokens.len() as u64;
        // A model whose only state beside its kv is window lists keeps one at
        // the pass's last page boundary unasked: the refs to its lists' last slots are all a
        // later prompt sharing the prefix needs, where kv alone resumed it
        // before windows were trimmed. A recurrent state's would hold its
        // record, which the pass's next write copies whole: it keeps one
        // only where the inferlet asks.
        let auto = windowed && state.is_some() && stores.rs.lock().unwrap().is_window();
        let checkpoints: Vec<u64> = match state {
            Some(_) => checkpoint_rows(
                &ctx.resources().get(&fwd)?.bindings.checkpoints,
                held..reach,
                page_size,
                keys.len(),
                auto,
            ),
            None => Vec::new(),
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
        let frozen = match lanes_frozen(&stores, &rs_ws_ids, &groups, full.lanes.len(), windowed) {
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
        let carried = piece_rows(&full, &whole, &frozen, false).carried;
        let largest = largest_rung(&stores, engine);
        if let Err(refusal) = check_largest_rung(FireRows { own: 0, carried }, largest) {
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
            rs_ws_ids.is_empty(),
            prefill_share(page_size, engine),
            page_size,
            &forced,
            &switched.tails,
        );
        // Only a lone lane's rest is cut again once a raise lands.
        let recuttable = full.lanes.len() == 1 && cuts.len() > 1;
        // A piece that would not cut to the rung asks for its own width,
        // and one that carries more rows than its own for those.
        let pieces: Vec<FireRows> = cuts
            .iter()
            .enumerate()
            .map(|(at, piece)| piece_rows(&full, piece, &frozen, folds_each && at > 0))
            .collect();
        for rows in &pieces {
            if let Err(refusal) = check_largest_rung(*rows, largest) {
                return Ok(Err(refusal));
            }
        }
        let widest = pieces.iter().map(|rows| rows.ask()).max().unwrap_or(0);
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
            if !folds_all {
                why.push("it states where its recurrent state folds");
            }
            if full.lanes.len() > 1 && !rs_ws_ids.is_empty() {
                why.push("its lanes carry recurrent state");
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
        let mut frames = {
            let pipeline = ctx.resources().get(&this)?;
            piece_frames(frame, cuts.len(), frame_size, || pipeline.next_frame_seq())
        };
        let framed = matches!(frame, Some(stamp) if stamp.fires == 1);
        let mut submitted: Vec<crate::engine::completion::WorkItemCompletion> = Vec::new();
        let mut settles = Vec::new();
        let mut at = 0;
        while at < cuts.len() {
            if recuttable && at > 0 && at + 1 < cuts.len() {
                // While the raise is pending the pass runs no further ahead
                // of the device than a few pieces, so what is left is cut at
                // the raised rung once it lands.
                // A framed piece waits only on a frame already whole.
                let ahead = PIECES_AHEAD.max(frame_size);
                if at >= ahead && wanted > cut_at {
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
                        prefill_share(page_size, engine),
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
            if let Err(refusal) = check_largest_rung(counted, largest) {
                return Ok(Err(refusal));
            }
            let width = counted.ask();
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
            // A window's piece keeps the furthest of its rows, which its
            // windows stand at once it lands, or reach back to from its end
            // (the slots that window touches retire no earlier).
            let window_at = checkpoints
                .iter()
                .copied()
                .filter(|&row| windowed && slots.start < row && row <= slots.end)
                .max();
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
            let keep = window_at
                .filter(|&row| row < slots.end)
                .and_then(|row| u32::try_from(row).ok());
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
                .map(|&group| (&group_sets[group], Written::Span(spans[group].1.clone())))
                .collect();
            let rs_plan = || {
                if windowed {
                    host_window_plan(&req, &fold, &rs_ws_ids, keep)
                } else {
                    host_rs_plan(&stores, &rs_ws_ids, &groups, &req, &fold, &inside)
                }
            };
            let reserved = reserve(
                ctx,
                (model, engine),
                &stores,
                quorum_pipeline_id,
                &sets,
                &rs_ws_ids,
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
            } = match reserved {
                Ok(reserved) => reserved,
                Err(error) => return Ok(Err(error)),
            };
            crate::probe_fire_record!(submit_probe.kv_prepare_us, kv_clock.elapsed());
            // Every fire logs what it writes where its tokens are known; what
            // it folds seals once it settles.
            let reports = pass_slots.is_none()
                && decode_envelope.is_some()
                && !wide
                && plain
                && plain_decode(&req, adapted, fire_wide_mask);
            let mut parts = Vec::with_capacity(group_sets.len());
            for (&group, (lease, kvtxn)) in present.iter().zip(leases.into_iter().zip(kvtxns)) {
                let (set, (slots, written)) = (&group_sets[group], &spans[group]);
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
                    let seq = (slots.start >= folded)
                        .then(|| kv.log_begin(set.id, slots.start))
                        .transpose()?;
                    Ok::<_, crate::store::kv::KvStoreError>((folded, seq, pages))
                });
                let (folded, seq, pages) = match logged {
                    Ok(logged) => logged,
                    Err(error) => return Ok(Err(format!("pipeline: KV rows: {error}"))),
                };
                let Some(seq) = seq else {
                    return Ok(Err(format!(
                        "pipeline: a fire writes slot {}, inside its working set's frozen \
                         prefix (0..{folded})",
                        slots.start
                    )));
                };
                let mut tags = Vec::new();
                // A piece carrying media is not its tokens' alone.
                let rows = match (&pass_slots, pages) {
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
                let lanes = &groups[group].1;
                let reach = Some(u64::from(landings[lanes[0]].end()));
                let rs = rs_ws_ids.get(lanes[0]).copied().filter(|_| !windowed);
                let by = fold_at(
                    &pass_lanes,
                    lanes,
                    reach,
                    &fold,
                    slots.end,
                    last,
                    rs,
                    &mut req,
                    &mut tags,
                );
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
            settles.extend(settling(&parts));
            if let Err(refusal) = map_lane_pages(&mut req, &stores, &piece_groups) {
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
            epoch.retire_on(&completion);
            for part in &mut parts {
                part.kvtxn.retire_on(&completion);
            }
            mark_device_folds(&stores, &completion, &mut parts);
            submitted.push(completion.clone());
            if let Some(reservation) = ticket_reservation {
                reservation.commit();
            }

            if last {
                let began = crate::scheduler::probe::ProbeClock::start();
                advance_shadow(ctx, &fwd)?;
                crate::probe_fire_record!(submit_probe.shadow_advance_us, began.elapsed());
            }

            pipe_fires
                .lock()
                .unwrap()
                .push_back(PendingOp::Fire(PendingFire {
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
                }));
            // The next piece reads the KV this one writes: they run in
            // order on the device, so it is submitted now, as a decode's
            // next pass is, rather than once this one retires.
            at += 1;
        }
        if let (Some(ask), Some(last)) = (rung_ask, submitted.last()) {
            ask.until_settled(last);
        }
        // The claims on what this pass computes stand until its last piece
        // settles (its pages sealed); a pass that ends early drops them.
        if let (Some(guard), Some(last)) = (coalesced.take(), submitted.last()) {
            last.on_settled(move || drop(guard));
        }
        advance_cursors(&pass_lanes, Some(&landings), &matched, &fold, &settles);
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
        return submit_pass_stamped(ctx, this, Resource::new_borrow(rep), None).await;
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
        let stamp = crate::scheduler::FrameStamp {
            lane,
            seq,
            slot,
            fires,
        };
        let pipeline: Resource<Pipeline> = Resource::new_borrow(this.rep());
        let fwd: Resource<ForwardPass> = Resource::new_borrow(rep);
        let outcome = submit_pass_stamped(ctx, pipeline, fwd, Some(stamp)).await;
        if !matches!(outcome, Ok(Ok(()))) && index > 0 {
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
    Ok(())
}

/// The last fold of the working set to settle makes its cursor exact.
/// `failure` is why the fire did not run to its end, when it did not: what
/// it was to freeze stays unfrozen, and a recurrent state it folded by a
/// count the device computes is left refused by that name. One whose fire
/// ran to its end and brought no count is left refused too
/// ([`RsDeviceFold::UNREPORTED`]): the engine's record went on by a count
/// the frozen length never takes, and what copies or moves the set would
/// state a length behind it.
fn settle_fold(
    stores: &crate::store::registry::Stores,
    ws: crate::store::kv::page_table::WorkingSetId,
    fold: FoldAt,
    reported: &std::collections::HashMap<u64, ::engine::KvWrites>,
    failure: Option<&str>,
) {
    let success = failure.is_none();
    let FoldAt { mut by, decoded } = fold;
    // Taken before the kv lock: ending it takes the rs store's. The count
    // is the host's, or what the engine reported under the lanes' tags; a
    // fire that ran and brought neither went unreported.
    let (device_fold, unreported) = match &mut by {
        FoldBy::Settled {
            count, tags, rs, ..
        } => {
            *count = count.or_else(|| {
                tags.iter()
                    .map(|tag| reported.get(tag).and_then(|written| written.fold))
                    .collect::<Option<Vec<u32>>>()
                    .and_then(|counts| counts.into_iter().max())
            });
            (rs.take(), success && count.is_none())
        }
        FoldBy::To(_) => (None, false),
    };
    let mut device_slots = None;
    let folded = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
        let frozen = kv.folded(ws)?;
        let (to, cursor) = match by {
            FoldBy::To(to) => (Some(to), None),
            FoldBy::Settled {
                count,
                tags: _,
                reach,
                cursor,
                rs: _,
            } => {
                let to = count.map(|count| frozen.saturating_add(u64::from(count)).min(reach));
                // The recurrent state froze the same slots on the device.
                device_slots = to
                    .filter(|_| success)
                    .map(|to| u32::try_from(to.saturating_sub(frozen)).unwrap_or(u32::MAX));
                (to, Some((cursor, to.is_some())))
            }
        };
        let pages = match to.filter(|_| success) {
            Some(to) => kv.fold(ws, to, decoded)?,
            None => 0,
        };
        Ok::<_, crate::store::kv::KvStoreError>((pages, kv.folded(ws)?, cursor))
    });
    match folded {
        Ok((pages, frozen, cursor)) => {
            SEALED_PAGES.fetch_add(pages, std::sync::atomic::Ordering::Relaxed);
            if let Some((cursor, placed)) = cursor
                && placed
            {
                let mut cursor = cursor.lock().unwrap();
                cursor.unsettled = cursor.unsettled.saturating_sub(1);
                if cursor.unsettled == 0 {
                    cursor.folded = u32::try_from(frozen).unwrap_or(u32::MAX);
                    cursor.folded_exact = true;
                }
            }
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
        }
        // What the fire froze seals now that its rows are logged.
        for (ws, fold) in folds {
            settle_fold(&stores, ws, fold, &reported, failure_reason.as_deref());
        }
        for checkpoint in &checkpoints {
            take_checkpoint(&stores, checkpoint, success);
        }
    }

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
    let mut write_indexes: Vec<Vec<u64>> = Vec::with_capacity(groups.len());
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
        if reserved <= place.folded {
            return Ok(Err(format!(
                "pipeline: lane {lane}'s rows land where the device says, inside what \
                 `reserve` granted; its working set reserved {reserved} slot(s) and froze \
                 {}. Reserve the slots it may write",
                place.folded
            )));
        }
        let first = u64::from(place.folded) / page;
        let end = u64::from(reserved).div_ceil(page);
        write_indexes.push((first..end).collect());
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
        windowed,
    ) {
        Ok(frozen) => frozen,
        Err(error) => return Ok(Err(error)),
    };
    let counted = FireRows::of(groups.iter().flat_map(|(_, lanes)| {
        let carried = {
            let cursor = pass_lanes.cursors[lanes[0]].lock().unwrap();
            device_lane_rows(0, &cursor, frozen[lanes[0]]).carried
        };
        let split = &resolved_qo_indptr;
        lanes.iter().map(move |&lane| FireRows {
            own: split[lane + 1] - split[lane],
            carried,
        })
    }));
    if let Err(refusal) = check_largest_rung(counted, largest_rung(&stores, ws.engine)) {
        return Ok(Err(refusal));
    }
    let rows = counted.ask();
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
    let rs_plan = || {
        if windowed {
            device_window_plan(&stores, ws.id, &write_indexes[0], &rs_ws_ids)
        } else {
            device_rs_plan(&stores, &rs_ws_ids, &groups, &pass_lanes, &fold)
        }
    };
    let reserved = reserve(
        ctx,
        (ws.model, ws.engine),
        &stores,
        quorum_pipeline_id,
        &sets,
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
    } = match reserved {
        Ok(reserved) => reserved,
        Err(error) => return Ok(Err(error)),
    };
    let rs_copy = rs_prepared.state_copy();
    let rstxns = RsTxnsGuard::new(ws.model, ws.engine, rs_prepared.txn.take());
    // The device picks the slots a decode writes: each working set's log is
    // cut where it may start writing, its frozen length, unless the engine
    // reports what it wrote.
    let mut logged = Vec::with_capacity(group_sets.len());
    for ((set, indexes), (_, lanes)) in group_sets.iter().zip(&write_indexes).zip(&groups) {
        let known = u64::from(pass_lanes.places[lanes[0]].folded);
        let entry = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
            let cut = kv.folded(set.id)?.max(known);
            let seq = kv.log_begin(set.id, cut)?;
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
            Ok::<_, crate::store::kv::KvStoreError>((seq, pages))
        });
        match entry {
            Ok(entry) => logged.push(entry),
            Err(error) => return Ok(Err(format!("pipeline: KV rows: {error}"))),
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
            .map(|span| ::engine::Lane {
                tokens: vec![0; (span[1] - span[0]) as usize],
                ..::engine::Lane::default()
            })
            .collect(),
        carried: counted.carried,
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
    for ((((set, (log_seq, log_pages)), lease), kvtxn), (_, lanes)) in group_sets
        .iter()
        .zip(logged)
        .zip(leases)
        .zip(kvtxns)
        .zip(&groups)
    {
        let mut tags = Vec::new();
        let rows = match log_pages.filter(|_| plain_decode) {
            Some(pages) => LogRows::Reported {
                tag: report_on(&mut req, 0, &mut tags),
                pages,
                page_size: stores.kv_page_size,
            },
            None => LogRows::Unknown,
        };
        // Where the rows land is the device's: the fold is placed when the
        // fire settles.
        let rs = rs_ws_ids.get(lanes[0]).copied().filter(|_| !windowed);
        let by = fold_at(
            &pass_lanes,
            lanes,
            None,
            &fold,
            u64::MAX,
            true,
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
    let settles = settling(&parts);
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
    epoch.retire_on(&completion);
    for part in &mut parts {
        part.kvtxn.retire_on(&completion);
    }
    mark_device_folds(&stores, &completion, &mut parts);
    if let Some(ask) = rung_ask {
        ask.until_settled(&completion);
    }
    ticket_reservation.commit();
    advance_shadow(ctx, &fwd)?;
    advance_cursors(&pass_lanes, None, &[], &fold, &settles);
    for (part, (_, lanes)) in parts.iter().zip(&groups) {
        if let Some(staged) = &part.staged {
            pass_lanes.cursors[lanes[0]]
                .lock()
                .unwrap()
                .stage(staged.rows.clone());
        }
    }

    pipe_fires
        .lock()
        .unwrap()
        .push_back(PendingOp::Fire(PendingFire {
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
        }));
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
pub(crate) mod lifecycle_tests {
    use super::*;
    use wasmtime::component::ResourceTable;

    pub(super) struct TestContext {
        pub(super) id: uuid::Uuid,
        pub(super) resources: ResourceTable,
        pub(super) tails: Vec<PendingFires>,
    }

    impl FireContext for TestContext {
        fn resources(&mut self) -> &mut ResourceTable {
            &mut self.resources
        }

        fn process_id(&self) -> uuid::Uuid {
            self.id
        }

        fn pipeline_tails(&self) -> Vec<PendingFires> {
            self.tails.clone()
        }
    }

    /// Mac's gemma-4-e4b c64 stall: a windowed ask that settled its tail
    /// whole first waited on its pass's earlier piece, whose frame only the
    /// piece asking completes. The ask now returns once the planner grants
    /// (growth), the tail still in flight and untouched.
    #[tokio::test(flavor = "current_thread")]
    async fn a_windowed_ask_never_waits_on_its_own_unlanded_tail() -> anyhow::Result<()> {
        let (fires, completion, ws, _stores, _failure) = pending_fire_fixture(Vec::new());
        let mut context = TestContext {
            id: uuid::Uuid::new_v4(),
            resources: ResourceTable::new(),
            tails: vec![fires.clone()],
        };
        context.resources.push(ws)?;
        let (grow, grown) = tokio::sync::oneshot::channel::<()>();
        let ask = async move {
            grown.await.expect("the pool grows");
            "granted"
        };
        let task = tokio::spawn(async move {
            // The pool grows a moment later; the tail never lands meanwhile.
            for _ in 0..10 {
                tokio::task::yield_now().await;
            }
            let _ = grow.send(());
        });
        let asked = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            ask_while_tail_lands(&mut context, ask, || {}),
        )
        .await
        .expect("the ask returns without its tail landing");
        assert_eq!(asked.map_err(anyhow::Error::msg)?, "granted");
        task.await?;
        assert_eq!(fires.lock().expect("test FIFO").len(), 1, "the tail stays");
        resolve_test_completion(&completion, false);
        finalize_all(&mut context, &fires, true).await?;
        Ok(())
    }

    /// Two windowed processes, a pool that cannot grant at once: one ask is
    /// served only by the window slots the other's tail gives back, and the
    /// other's tail lands while it asks too. Each finalizes its tail as it
    /// lands, so the slots come back and both asks are served.
    #[tokio::test(flavor = "current_thread")]
    async fn two_windowed_asks_are_served_by_the_tails_they_finalize_as_they_land()
    -> anyhow::Result<()> {
        let mut processes = Vec::new();
        for _ in 0..2 {
            let (fires, completion, ws, _stores, _failure) = pending_fire_fixture(Vec::new());
            let mut context = TestContext {
                id: uuid::Uuid::new_v4(),
                resources: ResourceTable::new(),
                tails: vec![fires.clone()],
            };
            context.resources.push(ws)?;
            processes.push((context, fires, completion));
        }
        // The planner's room: slots come back as a tail is finalized.
        let freed = Arc::new(tokio::sync::Semaphore::new(0));
        let mut tasks = Vec::new();
        for (mut context, fires, completion) in processes {
            let freed = freed.clone();
            let released = freed.clone();
            tasks.push(tokio::spawn(async move {
                let ask = async move {
                    // Served once the other process's tail gave its slots back.
                    freed.acquire().await.expect("room").forget();
                };
                let lands = tokio::spawn(async move {
                    for _ in 0..5 {
                        tokio::task::yield_now().await;
                    }
                    resolve_test_completion(&completion, false);
                });
                ask_while_tail_lands(&mut context, ask, || released.add_permits(1))
                    .await
                    .map_err(anyhow::Error::msg)?;
                lands.await?;
                // Whatever landed after the grant is finalized on the fire's
                // own path, and its slots go back to the planner as well.
                finalize_all(&mut context, &fires, true).await?;
                released.add_permits(1);
                anyhow::Ok(fires.lock().expect("test FIFO").len())
            }));
        }
        for task in tasks {
            let left = tokio::time::timeout(std::time::Duration::from_secs(5), task)
                .await
                .expect("both asks are served")??;
            assert_eq!(left, 0);
        }
        Ok(())
    }

    /// Yields until `task` has run as far as it can without an event.
    async fn settle_tasks() {
        for _ in 0..20 {
            tokio::task::yield_now().await;
        }
    }

    /// A stale grant is quoted again on drift (the demand changed), not by
    /// count, and refused when stale against the demand it was quoted for.
    #[test]
    fn a_stale_grant_is_quoted_again_only_on_drift() {
        let demand = |kv_pages| crate::planner::Demand {
            kv_pages,
            rs: crate::store::rs::RsUnits::default(),
        };
        let mut stale_at = None;
        assert!(
            requote_on_drift(&mut stale_at, demand(4)),
            "first stale: quoted again"
        );
        assert!(
            requote_on_drift(&mut stale_at, demand(6)),
            "drifted: quoted again"
        );
        assert!(
            requote_on_drift(&mut stale_at, demand(7)),
            "drifted again: however often"
        );
        assert!(
            !requote_on_drift(&mut stale_at, demand(7)),
            "stale against the demand it was quoted for: refused"
        );
    }

    /// A piece waiting for a wider rung wakes when the cap is lifted, not
    /// on a clock.
    #[tokio::test(flavor = "current_thread")]
    async fn a_piece_waiting_on_the_rung_wakes_when_the_cap_lifts() {
        const ENGINE: usize = 40;
        crate::store::shared::set_fire_cap_for_test(ENGINE, 16);
        let waiting = tokio::spawn(wait_for_rung(ENGINE, 256, Some(|| None)));
        settle_tasks().await;
        assert!(!waiting.is_finished(), "cut to 16, it waits");
        crate::store::shared::set_fire_cap_for_test(ENGINE, 256);
        settle_tasks().await;
        assert!(waiting.is_finished(), "the cap lifted: it runs");
        assert!(waiting.await.unwrap().is_ok());
        crate::store::shared::set_fire_cap_for_test(ENGINE, u32::MAX);
    }

    /// A piece waiting for a wider rung wakes when its raise's funding is
    /// refused, and says so.
    #[tokio::test(flavor = "current_thread")]
    async fn a_piece_waiting_on_the_rung_wakes_when_its_funding_is_refused() {
        const ENGINE: usize = 41;
        crate::store::shared::set_fire_cap_for_test(ENGINE, 16);
        let refused = Arc::new(Mutex::new(None::<String>));
        let failed = {
            let refused = refused.clone();
            move || refused.lock().unwrap().clone()
        };
        let waiting = tokio::spawn(wait_for_rung(ENGINE, 256, Some(failed)));
        settle_tasks().await;
        assert!(!waiting.is_finished());
        *refused.lock().unwrap() = Some("no room".into());
        crate::store::shared::rung_changed(ENGINE);
        settle_tasks().await;
        assert!(waiting.is_finished(), "the refusal told the rung: it ends");
        assert!(waiting.await.unwrap().unwrap_err().contains("no room"));
        crate::store::shared::set_fire_cap_for_test(ENGINE, u32::MAX);
    }

    pub(crate) fn pending_fire_fixture(
        cells: BoundCells,
    ) -> (
        PendingFires,
        crate::engine::WorkItemCompletion,
        KvWorkingSet,
        crate::store::registry::Stores,
        PipelineFailure,
    ) {
        let model = crate::store::registry::register_model(32, &[1], &[1]);
        let stores = crate::store::registry::get(model, 0);
        let (id, txn) = {
            let mut store = stores.kv.lock();
            let id = store.create_working_set();
            store.reserve(id, 1).expect("reserve the test sequence");
            let mut granted = store
                .reserve_device_pages(1)
                .expect("one free physical page");
            let (_, txn) = kv::prepare_explicit_reserved(&mut store, id, &[0], &mut granted)
                .expect("publish a real pending KV transaction");
            (id, txn)
        };
        let ws = KvWorkingSet::new(model, 0, id, 32);
        let completion = crate::engine::WorkItemCompletion::deferred_with_guard(None);
        let failure = Arc::new(Mutex::new(None));
        let fires = Arc::new(PendingFireQueue::new());
        fires
            .lock()
            .expect("test FIFO")
            .push_back(PendingOp::Fire(PendingFire {
                completion: completion.clone(),
                checkpoints: Vec::new(),
                kv: vec![FireWs {
                    ws: id,
                    kvtxn: KvTxnGuard::new(model, 0, Some(txn)),
                    log: None,
                    lease: ws.fire_lease().expect("lease the submitted working set"),
                    fold: None,
                    reports: Vec::new(),
                    staged: None,
                }],
                device_geometry: false,
                rstxn: RsTxnsGuard::new(model, 0, None),
                model,
                engine: 0,
                fwd_rep: u32::MAX,
                cells,
                failure: failure.clone(),
                retired: None,
            }));
        (fires, completion, ws, stores, failure)
    }

    pub(crate) fn resolve_test_completion(
        completion: &crate::engine::WorkItemCompletion,
        failed: bool,
    ) {
        completion.mark_native_retired();
        if failed {
            completion.reject("test callback failure");
        } else {
            // SAFETY: `completion` is borrowed for the whole call and owns this
            // terminal cell, so the pointer is live; `publish` is an atomic store.
            unsafe {
                (*completion.terminal_cell_ptr().0)
                    .publish(crate::engine::completion::TERMINAL_OUTCOME_SUCCESS);
            }
            completion
                .resolve_from_terminal()
                .expect("publish successful completion");
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn cancelled_full_drain_keeps_pending_kv_for_teardown() -> anyhow::Result<()> {
        use std::future::Future;
        for failed in [false, true] {
            let (fires, completion, ws, stores, failure) = pending_fire_fixture(Vec::new());
            let mut context = TestContext {
                id: uuid::Uuid::new_v4(),
                resources: ResourceTable::new(),
                tails: Vec::new(),
            };
            context.resources.push(ws)?;
            {
                let mut drain = Box::pin(finalize_all(&mut context, &fires, false));
                let mut task = std::task::Context::from_waker(std::task::Waker::noop());
                assert!(drain.as_mut().poll(&mut task).is_pending());
                // Dropping the real host future models guest-task cancellation.
            }
            assert_eq!(
                fires.lock().expect("test FIFO").len(),
                1,
                "cancelled drain must leave its unresolved operation in the FIFO"
            );
            context.resources = ResourceTable::new();
            assert_eq!(
                stores.kv.lock().available_pages(),
                0,
                "resource destruction must not recycle pages before completion"
            );
            resolve_test_completion(&completion, failed);
            finalize_all(&mut context, &fires, true).await?;
            assert!(fires.lock().expect("test FIFO").is_empty());
            assert_eq!(
                stores.kv.lock().available_pages(),
                1,
                "teardown must settle the original transaction and release its page"
            );
            assert_eq!(failure.lock().expect("test failure").is_some(), failed);
        }
        Ok(())
    }

    struct Flag(std::sync::atomic::AtomicBool);

    impl std::task::Wake for Flag {
        fn wake(self: Arc<Self>) {
            self.0.store(true, std::sync::atomic::Ordering::Release);
        }
    }

    /// Lanes that all wait on one front fire register their wakers on its
    /// completion before it settles: settling it wakes every one of them,
    /// not only the last to register (MI300X: 5-7 of 8 lanes slept for good
    /// on a settled fire while the eighth decoded alone).
    #[tokio::test(flavor = "current_thread")]
    async fn every_lane_waiting_on_the_front_fire_wakes_when_it_settles() -> anyhow::Result<()> {
        use std::future::Future;
        use std::sync::atomic::{AtomicBool, Ordering};
        let (fires, completion, ws, _stores, _failure) = pending_fire_fixture(Vec::new());
        let mut context = TestContext {
            id: uuid::Uuid::new_v4(),
            resources: ResourceTable::new(),
            tails: Vec::new(),
        };
        context.resources.push(ws)?;
        let flag = || Arc::new(Flag(AtomicBool::new(false)));
        // Three lanes at the front fire, and a drain awaiting its completion.
        let flags: Vec<Arc<Flag>> = (0..4).map(|_| flag()).collect();
        let mut woken = Vec::new();
        {
            let mut lanes: Vec<_> = flags[..3]
                .iter()
                .map(|flag| {
                    (
                        Box::pin(fires.await_front_completion()),
                        std::task::Waker::from(flag.clone()),
                    )
                })
                .collect();
            let mut drain = completion.clone();
            let drain_waker = std::task::Waker::from(flags[3].clone());
            for (wait, waker) in &mut lanes {
                let mut cx = std::task::Context::from_waker(waker);
                assert!(
                    wait.as_mut().poll(&mut cx).is_pending(),
                    "nothing settled yet"
                );
            }
            let mut cx = std::task::Context::from_waker(&drain_waker);
            assert!(std::pin::Pin::new(&mut drain).poll(&mut cx).is_pending());

            resolve_test_completion(&completion, false);
            woken.extend(flags.iter().map(|flag| flag.0.load(Ordering::Acquire)));
            for (wait, waker) in &mut lanes {
                let mut cx = std::task::Context::from_waker(waker);
                assert!(wait.as_mut().poll(&mut cx).is_ready(), "the fire settled");
            }
        }
        finalize_all(&mut context, &fires, true).await?;
        assert_eq!(
            woken,
            vec![true; 4],
            "every waiter registered before the settle is woken by it (three lanes, then the \
             drain)"
        );
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn close_and_drop_share_graceful_fifo_drain_semantics() -> anyhow::Result<()> {
        let mut context = TestContext {
            id: uuid::Uuid::new_v4(),
            resources: ResourceTable::new(),
            tails: Vec::new(),
        };
        let pipeline = context.resources.push(Pipeline::new())?;
        let rep = pipeline.rep();
        let borrowed: Resource<Pipeline> = Resource::new_borrow(rep);
        let fires = context.resources.get(&borrowed)?.fires.clone();
        fires
            .lock()
            .unwrap()
            .extend([test_pending_op_stub(), test_pending_op_stub()]);

        pipeline_close(&mut context, Resource::new_borrow(rep)).await?;
        assert!(context.resources.get(&borrowed)?.scope.is_closed());
        assert!(fires.lock().unwrap().is_empty());
        pipeline_close(&mut context, Resource::new_borrow(rep)).await?;
        assert!(fires.lock().unwrap().is_empty());

        pipeline_drop(&mut context, pipeline).await?;
        assert!(context.resources.get(&borrowed).is_err());
        Ok(())
    }
}

#[cfg(test)]
mod static_admission_tests {
    use super::*;
    use crate::pipeline::channel::ChannelCell;
    use eta_ir::container::ChannelDecl;
    use eta_ir::types::{Dtype, Shape};

    fn channel(role: HostRole, capacity: u32, seeded: bool) -> Arc<Mutex<ChannelCell>> {
        let mut cell = ChannelCell::new(vec![1], Dtype::U32, capacity);
        cell.bind(&ChannelDecl {
            shape: Shape::new(&[1]).expect("a one-element cell"),
            dtype: eta_ir::container::ChanDType::Concrete(Dtype::U32),
            capacity,
            host_role: role,
            seeded,
        });
        Arc::new(Mutex::new(cell))
    }

    fn slot(cells: &[Arc<Mutex<ChannelCell>>], accesses: &[(bool, bool)]) -> SlotAccess {
        SlotAccess {
            cells: cells.to_vec(),
            accesses: accesses.to_vec(),
        }
    }

    #[test]
    fn admission_every_case() {
        a_device_ring_frame_that_overflows_is_refused_by_name();
        a_seeded_descriptor_ring_is_not_walked();
        a_frame_that_drains_more_writer_cells_than_are_staged_is_refused();
        a_latest_value_word_with_no_committed_cell_is_refused();
        a_reader_ring_too_small_for_the_frames_writes_is_refused();
        a_slot_that_consumes_what_an_earlier_slot_published_is_admitted();
        a_dropped_reservation_gives_its_count_back_while_a_later_one_stands();
        two_passes_putting_one_ring_get_distinct_cells_in_dispatch_order();
    }

    fn a_dropped_reservation_gives_its_count_back_while_a_later_one_stands() {
        let ring = channel(HostRole::None, 4, false);
        let cells: BoundCells = vec![Arc::clone(&ring)];
        let earlier = TicketReservation::new(&cells, &[(false, true)]);
        let later = TicketReservation::new(&cells, &[(false, true)]);
        assert_eq!(ring.lock().unwrap().device_ring_backlog(), 2);
        drop(earlier);
        assert_eq!(
            ring.lock().unwrap().device_ring_backlog(),
            1,
            "the earlier fire never reached the engine"
        );
        later.commit();
        assert_eq!(ring.lock().unwrap().device_ring_backlog(), 1);
    }

    /// A channel is its process's: two passes of the process put one device
    /// ring, each a fire of its own lane, and the engine picks each put's
    /// cell from the ring's cursor as it builds the step. In each way the
    /// fire that reserved later can reach the engine first, the two get
    /// distinct cells, in the order they are dispatched, with no rule about
    /// the order they reserved in.
    fn two_passes_putting_one_ring_get_distinct_cells_in_dispatch_order() {
        use crate::scheduler::worker::PendingRequest;
        type Submissions = Vec<Vec<Vec<u64>>>;
        // Each case as the submissions it dispatches, each a list of waves of
        // the fires' instances: 1 reserved first, 2 second.
        let cases: [(&str, Submissions); 7] = [
            ("in reservation order", vec![vec![vec![1, 2]]]),
            ("the seal leaves the earlier frame for later", vec![vec![vec![2]], vec![vec![1]]]),
            ("a held prompt waits a wave", vec![vec![vec![2], vec![1]]]),
            ("the earlier lane stands later in the wave", vec![vec![vec![2, 1]]]),
            ("a boundary falls between them", vec![vec![vec![2]], vec![vec![1]]]),
            ("the later is a rider ahead of the framed", vec![vec![vec![2]], vec![vec![1]]]),
            ("the later is framed ahead of a rider", vec![vec![vec![2]], vec![vec![1]]]),
        ];
        let limits = crate::engine::SchedulerLimits {
            max_forward_requests: 8,
            max_forward_tokens: 64,
            max_page_refs: 64,
            max_context: 0,
            prefill_rows: 0,
            prefill_lanes: 0,
            max_readouts: 0,
        };
        for (case, submissions) in cases {
            let ring = channel(HostRole::None, 4, false);
            let cells: BoundCells = vec![Arc::clone(&ring)];
            let id = ring.lock().unwrap().global_id;
            let reservations = [
                TicketReservation::new(&cells, &[(false, true)]),
                TicketReservation::new(&cells, &[(false, true)]),
            ];
            let fire = |instance: u64| {
                let mut request = crate::engine::FireRequest::one(crate::engine::fire::lane_of(
                    0,
                    vec![instance as u32],
                    0,
                    vec![instance as u32],
                ));
                request.single_token_mode = true;
                reservations[instance as usize - 1].apply_to(&mut request);
                assert_eq!(
                    request.ring_moves,
                    [crate::engine::fire::RingMove {
                        channel: id,
                        takes: false,
                        puts: true,
                    }],
                    "the fire carries the end it moves and no place"
                );
                Box::new(PendingRequest {
                    lora_program: false,
                    logical_fire_id: instance,
                    request,
                    instance_id: ::engine::InstanceId(instance),
                    completion: crate::engine::WorkItemCompletion::new(instance, 0),
                    process_id: Some(uuid::Uuid::from_u128(1)),
                    pipeline_id: Some(uuid::Uuid::from_u128(u128::from(instance))),
                    prelaunch_copy: None,
                    prelaunch_state_copy: None,
                    frame: None,
                })
            };
            // The engine's ring cursor, moved by each put as it builds a step.
            let mut cursor = 0u64;
            let mut cells_of = std::collections::BTreeMap::new();
            let mut dispatched = Vec::new();
            for waves in submissions {
                let waves = waves
                    .into_iter()
                    .map(|wave| wave.into_iter().map(fire).collect())
                    .collect();
                let (frame, _) = crate::scheduler::batch::build_frame_submission(
                    waves,
                    limits,
                    16,
                    &crate::scheduler::stats::SchedulerStats::default(),
                );
                for step in &frame.steps {
                    let before = cursor;
                    for instance in &step.instances {
                        cells_of.insert(instance.0, cursor);
                        cursor += 1;
                        dispatched.push(instance.0);
                    }
                    assert!(cursor - before <= 1, "{case}: one put of the ring a step");
                }
            }
            let cells: Vec<u64> = dispatched.iter().map(|fire| cells_of[fire]).collect();
            assert_eq!(cells, [0, 1], "{case}: distinct cells in dispatch order");
            assert_eq!(ring.lock().unwrap().device_ring_backlog(), 2, "{case}");
            for reservation in reservations {
                reservation.commit();
            }
        }
    }

    fn a_device_ring_frame_that_overflows_is_refused_by_name() {
        let ring = channel(HostRole::None, 1, false);
        let slots = [
            slot(std::slice::from_ref(&ring), &[(false, true)]),
            slot(std::slice::from_ref(&ring), &[(false, true)]),
        ];
        let refusal = prove_frame_admissible(2, &slots).expect_err("two publishes, capacity one");
        assert!(
            refusal.contains("device-ring occupancy past capacity 1"),
            "{refusal}"
        );
        assert!(refusal.contains("frame slot 1"), "{refusal}");
    }

    fn a_seeded_descriptor_ring_is_not_walked() {
        let seeded = channel(HostRole::None, 1, true);
        let slots = [
            slot(std::slice::from_ref(&seeded), &[(false, true)]),
            slot(std::slice::from_ref(&seeded), &[(false, true)]),
        ];
        assert_eq!(prove_frame_admissible(2, &slots), Ok(()));
    }

    fn a_frame_that_drains_more_writer_cells_than_are_staged_is_refused() {
        let writer = channel(HostRole::Writer, 4, false);
        writer
            .lock()
            .unwrap()
            .put_ref(&[0u8; 4])
            .expect("one staged cell");
        let one = [slot(std::slice::from_ref(&writer), &[(true, false)])];
        assert_eq!(prove_frame_admissible(2, &one), Ok(()));

        let two = [
            slot(std::slice::from_ref(&writer), &[(true, false)]),
            slot(std::slice::from_ref(&writer), &[(true, false)]),
        ];
        let refusal = prove_frame_admissible(2, &two).expect_err("two consumes, one staged");
        assert!(
            refusal.contains("consumes 2 host-writer cell(s)"),
            "{refusal}"
        );
        assert!(refusal.contains("only 1 are staged"), "{refusal}");
    }

    fn a_latest_value_word_with_no_committed_cell_is_refused() {
        let word = channel(HostRole::Writer, 1, false);
        let slots = [
            slot(std::slice::from_ref(&word), &[(false, false)]),
            slot(std::slice::from_ref(&word), &[(false, false)]),
        ];
        let refusal = prove_frame_admissible(2, &slots).expect_err("never set");
        assert!(refusal.contains("latest-value control word"), "{refusal}");

        word.lock()
            .unwrap()
            .put_ref(&[0u8; 4])
            .expect("the host writes the word");
        assert_eq!(prove_frame_admissible(2, &slots), Ok(()));
    }

    fn a_reader_ring_too_small_for_the_frames_writes_is_refused() {
        let reader = channel(HostRole::Reader, 1, false);
        let slots = [
            slot(std::slice::from_ref(&reader), &[(false, true)]),
            slot(std::slice::from_ref(&reader), &[(false, true)]),
        ];
        let refusal = prove_frame_admissible(2, &slots).expect_err("two writes, capacity one");
        assert!(
            refusal.contains("frame would need 2 reader cell(s)"),
            "{refusal}"
        );
        assert!(refusal.contains("2k-1 = 3"), "{refusal}");

        let roomy = channel(HostRole::Reader, 3, false);
        let ok = [
            slot(std::slice::from_ref(&roomy), &[(false, true)]),
            slot(std::slice::from_ref(&roomy), &[(false, true)]),
        ];
        assert_eq!(prove_frame_admissible(2, &ok), Ok(()));
    }

    fn a_slot_that_consumes_what_an_earlier_slot_published_is_admitted() {
        let chained = channel(HostRole::None, 2, false);
        let slots = [
            slot(std::slice::from_ref(&chained), &[(false, true)]),
            slot(std::slice::from_ref(&chained), &[(true, false)]),
        ];
        assert_eq!(prove_frame_admissible(2, &slots), Ok(()));
    }
}

/// A piece wider than the rung never waits on a rung that does not come:
/// it fires once the raise lands, and fails with what the planner said once
/// the raise's room is refused.
#[cfg(test)]
mod rung_wait_tests {
    use super::wait_for_rung;
    use std::sync::{Arc, Mutex};

    /// Engines no other test drives, so the caps set here are theirs alone.
    const RAISED: usize = 61;
    const REFUSED: usize = 62;
    const UNFUNDED: usize = 63;

    #[tokio::test(flavor = "current_thread")]
    async fn a_wide_piece_fires_once_the_raise_lands() {
        crate::store::shared::set_fire_cap_for_test(RAISED, 256);
        let raise = tokio::spawn(async {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            crate::store::shared::set_fire_cap_for_test(RAISED, u32::MAX);
        });
        let waited = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            wait_for_rung(RAISED, 1600, Some(|| None)),
        )
        .await;
        assert_eq!(waited, Ok(Ok(())));
        raise.await.unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_wide_piece_fails_once_its_room_is_refused() {
        crate::store::shared::set_fire_cap_for_test(REFUSED, 256);
        let refused: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let says = refused.clone();
        let refuse = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            *says.lock().unwrap() = Some("the device is short".into());
            // As a raise's funding tells the rung when it fails.
            crate::store::shared::rung_changed(REFUSED);
        });
        let waited = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            wait_for_rung(REFUSED, 1600, Some(|| refused.lock().unwrap().clone())),
        )
        .await
        .expect("no indefinite wait");
        assert!(waited.unwrap_err().contains("the device is short"));
        refuse.await.unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_piece_nothing_funds_does_not_wait() {
        crate::store::shared::set_fire_cap_for_test(UNFUNDED, 256);
        let unfunded: Option<fn() -> Option<String>> = None;
        assert_eq!(wait_for_rung(UNFUNDED, 1600, unfunded).await, Ok(()));
        assert_eq!(wait_for_rung(UNFUNDED, 200, Some(|| None)).await, Ok(()));
    }

    /// The MTP k=4 stall: a device-carried decode's fires of two rows sat
    /// queued behind a rung the walker had lowered to one row, since the
    /// scheduler's lane count asks one row a lane and nothing asked for
    /// theirs; the scheduler keeps a fire past the rung queued, so its frame
    /// never ran. Such a fire now asks the rung for its rows, enqueues once
    /// the rung takes them, and holds the ask while it is queued.
    #[tokio::test(flavor = "current_thread")]
    async fn a_device_fire_asks_the_rung_for_its_rows_and_holds_it() {
        const ENGINE: usize = 58;
        let stores = super::unfolded_tests::pool_of_one_row_rs_pages();
        let pid = uuid::Uuid::new_v4();
        crate::store::shared::set_fire_cap_for_test(ENGINE, 1);
        // One row a lane: the lane count holds it; it asks nothing.
        assert!(super::ask_rung(&stores, ENGINE, pid, pid, 1, 1).is_none());
        assert_eq!(crate::store::shared::wanted_rows(ENGINE), 0);

        let raise = tokio::spawn(async {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            // As the walker lifts the cap toward what is asked.
            assert_eq!(crate::store::shared::wanted_rows(ENGINE), 2);
            crate::store::shared::set_fire_cap_for_test(ENGINE, 2);
        });
        let ask = super::ask_rung(&stores, ENGINE, pid, pid, 2, 2).expect("two rows ask");
        tokio::time::timeout(std::time::Duration::from_secs(5), ask.wait(2))
            .await
            .expect("no indefinite wait")
            .expect("the rung takes it");
        raise.await.unwrap();
        assert!(
            crate::store::shared::fire_cap(ENGINE) >= 2,
            "enqueued once the rung takes it"
        );
        assert_eq!(
            crate::store::shared::wanted_rows(ENGINE),
            2,
            "the ask stands while the fire is queued"
        );
        drop(ask);
        assert_eq!(crate::store::shared::wanted_rows(ENGINE), 0);
    }

    /// The MTP `--k 0 --pad 1` stall: a decode's pass of two rows was
    /// staged while the cap stood wide (a bind had raised the rung), so it
    /// asked nothing; the walker, seeing one live lane and no ask, then cut
    /// fires to one row, and the scheduler kept the pass's two-row fires
    /// queued behind a rung nothing raised. The pass now asks whatever the
    /// cap is, and the walker cuts nothing under it until its last fire has
    /// settled.
    #[tokio::test(flavor = "current_thread")]
    async fn a_few_rows_staged_under_a_wide_cap_hold_the_rung_until_they_settle() {
        const ENGINE: usize = 49;
        let stores = super::unfolded_tests::pool_with_rungs(vec![(1, 1), (2, 2), (4, 3)]);
        let shared = stores.shared.clone().expect("a shared pool");
        // The walker of a pool with one live lane, as the governor ticks it.
        let grow =
            crate::store::shared::SharedGrow::new(0, ENGINE, shared.clone()).with_live_lanes(|| 1);
        let pid = uuid::Uuid::new_v4();
        // The rung is up and fires may take any width: a cap that covers
        // the pass.
        assert_eq!(shared.lock().unwrap().rung(), 4);
        assert_eq!(crate::store::shared::fire_cap(ENGINE), u32::MAX);

        // A two-row pass is cut and its fire staged under that cap.
        let ask = super::ask_rung(&stores, ENGINE, pid, pid, 2, 2)
            .expect("two rows ask whatever the cap");
        assert_eq!(ask.wait(2).await, Ok(()));
        let fire = crate::engine::completion::WorkItemCompletion::new(1, 1);
        ask.until_settled(&fire);

        // The pool idles past the walker's patience with the fire queued:
        // fires are cut to what is asked, never under the staged fire.
        for _ in 0..2 * crate::governor::PATIENCE + 2 {
            grow.ticked_for_test();
            grow.walk_rung().await;
            assert!(
                crate::store::shared::fire_cap(ENGINE) >= 2,
                "never cut under the staged fire"
            );
        }
        assert_eq!(
            crate::store::shared::fire_cap(ENGINE),
            2,
            "cut to the rung the staged fire asks, not to the live lane's row"
        );
        assert!(!fire.is_settled());

        // Settled, the ask goes, and the walker takes the cap down to the
        // live lane's row.
        fire.reject("settled");
        // A settlement runs its hooks off the caller's stack.
        tokio::task::yield_now().await;
        assert!(fire.is_settled());
        assert_eq!(crate::store::shared::wanted_rows(ENGINE), 0);
        for _ in 0..2 * crate::governor::PATIENCE + 2 {
            grow.ticked_for_test();
            grow.walk_rung().await;
        }
        assert_eq!(crate::store::shared::fire_cap(ENGINE), 1);
    }
}

#[cfg(test)]
mod cut_tests {
    use super::*;

    #[test]
    fn cut_every_case() {
        a_long_plain_pass_is_cut_on_page_boundaries();
        only_the_last_cut_runs_the_program();
        a_masked_pass_is_cut_with_its_rows_masks();
        the_last_piece_holds_what_the_program_reads();
        several_lanes_are_cut_and_end_together();
        a_lane_only_in_the_last_piece_is_not_written_before_it();
        a_pass_with_media_is_cut_before_its_first_span();
        a_pass_with_media_keys_the_rows_before_it();
        the_rest_of_a_pass_is_cut_again_wider();
    }

    /// A raise landing mid-pass: what is left from the last cut is cut at
    /// the wider share, on the same page boundaries and checkpoint rows.
    fn the_rest_of_a_pass_is_cut_again_wider() {
        let req = prefill_lane(5, 200, 16);
        let narrow = host_cuts(&req, true, 32, 16, &[123], 0);
        assert_eq!(narrow[..2], [(0, 27), (27, 59)]);
        assert_eq!(
            host_cuts_from(&req, true, 160, 16, &[123], 0, 59),
            [(59, 123), (123, 200)],
            "the checkpoint row still cuts"
        );
        assert_eq!(
            host_cuts_from(&req, true, 64, 16, &[], 0, 59),
            [(59, 123), (123, 187), (187, 200)]
        );
    }

    fn prefill_lane(held: u32, rows: u32, page: u32) -> crate::engine::FireRequest {
        let mut lane = ::engine::Lane::decode(0, 0, 1, 0);
        lane.tokens = (0..rows).map(|t| t + 100).collect();
        lane.kv.held = held;
        lane.kv.pages = (0..(held + rows).div_ceil(page)).collect();
        lane.readout = ::engine::Readout::Last;
        let mut req = crate::engine::FireRequest::one(lane);
        req.boundary_program = true;
        req
    }

    fn a_long_plain_pass_is_cut_on_page_boundaries() {
        let req = prefill_lane(5, 100, 16);
        assert_eq!(
            host_cuts(&req, true, 128, 16, &[], 0),
            [(0, 100)],
            "short enough"
        );
        assert_eq!(
            host_cuts(&req, false, 32, 16, &[], 0),
            [(0, 100)],
            "not cuttable"
        );
        // Held 5: the first cut ends where page 2 does (32 - 5).
        assert_eq!(
            host_cuts(&req, true, 32, 16, &[], 0),
            [(0, 27), (27, 59), (59, 91), (91, 100)]
        );
        let mut masked = req.clone();
        masked.lanes[0].readout = ::engine::Readout::Rows(vec![3]);
        assert_eq!(
            host_cuts(&masked, true, 32, 16, &[], 0),
            [(0, 100)],
            "reads a middle row"
        );
        // Checkpoint rows cut it too, short as it is.
        assert_eq!(
            host_cuts(&req, true, 128, 16, &[43, 100, 0], 0),
            [(0, 43), (43, 100)]
        );
        assert_eq!(
            host_cuts(&req, true, 32, 16, &[43], 0),
            [(0, 27), (27, 43), (43, 59), (59, 91), (91, 100)]
        );
    }

    fn only_the_last_cut_runs_the_program() {
        let req = prefill_lane(5, 100, 16);
        let first = rows_of_lanes(&req, &[Some((0, 27))], false, 16);
        let lane = &first.lanes[0];
        assert_eq!(
            (lane.kv.held, lane.tokens.len(), lane.kv.pages.len()),
            (5, 27, 2)
        );
        assert_eq!(lane.tokens[0], 100);
        assert!(matches!(lane.readout, ::engine::Readout::None));
        assert!(
            !first.boundary_program,
            "no program stages before the last fire"
        );
        let tail = rows_of_lanes(&req, &[Some((91, 100))], true, 16);
        let lane = &tail.lanes[0];
        assert_eq!(
            (lane.kv.held, lane.tokens.len(), lane.kv.pages.len()),
            (96, 9, 7)
        );
        assert_eq!(lane.tokens[0], 191);
        assert!(matches!(lane.readout, ::engine::Readout::Last));
        assert!(tail.boundary_program);
    }

    fn a_pass_with_media_is_cut_before_its_first_span() {
        let mut req = prefill_lane(5, 100, 16);
        req.media.push(::engine::fire::StepMedia {
            lane: 0,
            rows: vec![10],
            routes: (60..70).chain([-1, -1]).collect(),
            token_positions: (0..300).collect(),
            ..Default::default()
        });
        let cuts = host_cuts(&req, true, 32, 16, &[], 0);
        assert_eq!(cuts, [(0, 27), (27, 59), (59, 100)], "no cut past row 60");
        let first = rows_of_lanes(&req, &[Some((0, 27))], false, 16);
        assert!(first.media.is_empty(), "the text before it carries none");
        let tail = rows_of_lanes(&req, &[Some((59, 100))], true, 16);
        assert_eq!(tail.media[0].routes[..2], [1, 2]);
        assert_eq!(tail.media[0].routes[10], -1, "a dropped row stays dropped");
        assert_eq!(tail.media[0].token_positions.len(), 3 * 41);
        assert_eq!(tail.media[0].token_positions[0], 3 * 59);
    }

    fn a_pass_with_media_keys_the_rows_before_it() {
        let mut req = prefill_lane(5, 100, 16);
        assert_eq!(keyed_rows(&req), 100);
        req.media.push(::engine::fire::StepMedia {
            lane: 0,
            rows: vec![10],
            routes: (60..70).collect(),
            ..Default::default()
        });
        assert_eq!(keyed_rows(&req), 60, "up to its first media row");
        let cuts = host_cuts(&req, true, 128, 16, &[], 0);
        assert_eq!(cuts, [(0, 59), (59, 100)]);
        let first = rows_of_lanes(&req, &[Some(cuts[0])], false, 16);
        assert!(first.media.is_empty(), "it logs its rows and seals");
        assert_eq!(keyed_rows(&first), 59);
        let tail = rows_of_lanes(&req, &[Some(cuts[1])], true, 16);
        assert!(!tail.media.is_empty(), "it cuts the log");
        req.lanes[0].positions = (0..100).collect();
        assert_eq!(keyed_rows(&req), 0, "rows rotated off their slots");
    }

    fn several_lanes_are_cut_and_end_together() {
        let mut req = prefill_lane(5, 100, 16);
        let other = prefill_lane(0, 40, 16).lanes.remove(0);
        req.lanes.push(other);
        let pieces = pass_pieces(&req, true, true, 32, 16, &[], &[]);
        let last = pieces.last().unwrap();
        assert_eq!(
            last,
            &[Some((91, 100)), Some((32, 40))],
            "both lanes' tails"
        );
        assert_eq!(
            pieces[0],
            [Some((0, 11)), None],
            "the short lane starts later"
        );
        for lane in 0..2 {
            let rows: u32 = pieces
                .iter()
                .filter_map(|p| p[lane])
                .map(|(a, b)| b - a)
                .sum();
            assert_eq!(rows, [100, 40][lane], "every row once");
        }
        let first = rows_of_lanes(&req, &pieces[0], false, 16);
        assert_eq!(first.lanes.len(), 1);
        assert!(!first.boundary_program);
        let tail = rows_of_lanes(&req, last, true, 16);
        assert_eq!(tail.lanes.len(), 2);
        assert!(tail.boundary_program);
        assert_eq!(
            pass_pieces(&req, true, false, 32, 16, &[], &[]).len(),
            1,
            "not where the lanes carry recurrent state"
        );
    }

    fn a_lane_only_in_the_last_piece_is_not_written_before_it() {
        // A prompt cut into pieces beside a decode lane whose one row rides only in the last.
        let mut req = prefill_lane(0, 100, 16);
        let decode = prefill_lane(17, 1, 16).lanes.remove(0);
        req.lanes.push(decode);
        let pieces = pass_pieces(&req, true, true, 32, 16, &[], &[]);
        assert!(pieces.len() > 1, "the prompt is cut");
        let groups = [(0u32, vec![0]), (1u32, vec![1])];
        for (at, piece) in pieces.iter().enumerate() {
            let kept: Vec<usize> = (0..req.lanes.len())
                .filter(|&lane| piece[lane].is_some())
                .collect();
            let piece_groups: Vec<(u32, Vec<usize>)> = groups
                .iter()
                .map(|(ws, lanes)| {
                    (
                        *ws,
                        lanes
                            .iter()
                            .filter_map(|lane| kept.iter().position(|k| k == lane))
                            .collect(),
                    )
                })
                .collect();
            let last = at + 1 == pieces.len();
            let written = written_groups(&piece_groups);
            assert_eq!(
                written,
                if last { vec![0, 1] } else { vec![0] },
                "piece {at}"
            );
            if !last {
                // what the decode lane's set would have logged here: a write at slot 0, inside
                // the 17 slots it already froze
                let rows = rows_of_lanes(&req, piece, false, 16);
                assert_eq!(written_span_of(&rows, &piece_groups[1].1, 16), (0..0, 0..0));
            }
        }
    }

    fn the_last_piece_holds_what_the_program_reads() {
        let mut req = prefill_lane(5, 100, 16);
        req.lanes[0].readout = ::engine::Readout::Rows(vec![96, 95]);
        let tail = rows_of_lanes(&req, &[Some((91, 100))], true, 16);
        assert_eq!(tail.lanes[0].readout, ::engine::Readout::Rows(vec![5, 4]));
        req.lanes[0].readout = ::engine::Readout::Rows(vec![60, 99]);
        assert_eq!(
            host_cuts(&req, true, 32, 16, &[], 0),
            [(0, 27), (27, 59), (59, 100)],
            "no cut past the first row read"
        );
        // An output over the lane's last rows: the last piece holds them all.
        let req = prefill_lane(5, 100, 16);
        assert_eq!(
            host_cuts(&req, true, 32, 16, &[], 32),
            [(0, 27), (27, 59), (59, 100)]
        );
    }

    fn a_masked_pass_is_cut_with_its_rows_masks() {
        let mut req = prefill_lane(5, 100, 16);
        let masks = (0..100)
            .map(|row| ::engine::Mask::new(vec![row, 1], 105))
            .collect();
        req.lanes[0].mask = Some(::engine::Masking::Rows(masks));
        let piece = rows_of_lanes(&req, &[Some((27, 59))], false, 16);
        let Some(::engine::Masking::Rows(masks)) = &piece.lanes[0].mask else {
            panic!("the piece keeps its rows' masks");
        };
        assert_eq!(masks.len(), 32);
        assert_eq!(masks[0].runs, [27, 1], "row 27's own mask");
    }
}

#[cfg(test)]
pub(crate) mod fold_tests {
    use super::*;
    use crate::inferlet::host::working_set::Cursor;

    fn cursor(folded: u32, reserved: u32) -> CursorCell {
        Arc::new(Mutex::new(Cursor {
            folded,
            reserved,
            ..Cursor::default()
        }))
    }

    fn lanes(cursors: &[&CursorCell]) -> PassLanes {
        PassLanes {
            places: cursors
                .iter()
                .map(|cell| {
                    let now = cell.lock().unwrap().clone();
                    geometry::Place {
                        folded: now.folded,
                        folded_exact: now.folded_exact,
                        tail: now.tail,
                        shared: false,
                        spans: now.spans,
                    }
                })
                .collect(),
            cursors: cursors.iter().map(|&cell| cell.clone()).collect(),
            kv: Vec::new(),
        }
    }

    fn request(lanes: usize) -> crate::engine::FireRequest {
        crate::engine::FireRequest {
            lanes: (0..lanes).map(|_| ::engine::Lane::default()).collect(),
            ..crate::engine::FireRequest::default()
        }
    }

    fn frozen_at(
        folded: u64,
    ) -> (
        crate::store::registry::Stores,
        crate::store::kv::page_table::WorkingSetId,
    ) {
        let model = crate::store::registry::register_model(4, &[8], &[1]);
        let stores = crate::store::registry::get(model, 0);
        let ws = {
            let mut kv = stores.kv.lock();
            let ws = kv.create_working_set();
            kv.fold(ws, folded, false).unwrap();
            ws
        };
        (stores, ws)
    }

    fn taken(tags: &[ReportTag]) -> std::collections::HashMap<u64, ::engine::KvWrites> {
        tags.iter()
            .filter_map(|tag| tag.take().map(|written| (tag.0, written)))
            .collect()
    }

    fn report(tag: u64, fold: u32) -> ::engine::KvWrites {
        ::engine::KvWrites {
            tag,
            fold: Some(fold),
            ..::engine::KvWrites::default()
        }
    }

    #[test]
    fn fold_every_case() {
        a_fold_the_host_knows_is_placed_at_submit();
        a_device_fold_is_reported_and_placed_when_it_settles();
        the_last_fold_to_settle_makes_the_cursor_exact();
        a_fold_never_reported_leaves_the_cursor_a_bound();
        a_fire_carrying_a_span_leaves_its_lag();
    }

    fn a_fire_carrying_a_span_leaves_its_lag() {
        let cell = cursor(0, 0);
        cell.lock().unwrap().spans = vec![(10, 14, 2), (50, 60, 4)];
        let pass = lanes(&[&cell]);
        let span = Arc::new(media::front::EncodedSpan {
            token_count: 20,
            position_span: 5,
            grid: media::front::Grid::still(4, 5),
            patch_grid: media::front::Grid::still(8, 10),
            uses_mrope: true,
            payload: Vec::new(),
            rows: 80,
            positions: Vec::new(),
            prefix: Vec::new(),
            placeholder: 0,
            suffix: Vec::new(),
        });
        let run = crate::pipeline::media::MatchedRun {
            lane: 0,
            anchor: 3,
            rows: 20,
            span,
        };
        let landings = [geometry::Landing {
            start: 40,
            rows: 30,
        }];
        advance_cursors(&pass, Some(&landings), &[run], &geometry::Fold::None, &[]);
        let now = cell.lock().unwrap().clone();
        assert_eq!(now.tail, Some(70));
        assert_eq!(
            now.spans,
            [(10, 14, 2), (43, 63, 15)],
            "the write from slot 40 replaces what lay past it"
        );
        assert_eq!(crate::inferlet::host::working_set::lag(&now.spans, 70), 17);
    }

    fn a_fold_the_host_knows_is_placed_at_submit() {
        let cell = cursor(4, 0);
        let pass = lanes(&[&cell]);
        let mut req = request(1);
        let mut tags = Vec::new();
        let mut at = |fold: &geometry::Fold, end: u64, last: bool| {
            fold_at(
                &pass,
                &[0],
                Some(10),
                fold,
                end,
                last,
                None,
                &mut req,
                &mut tags,
            )
        };
        let three = geometry::Fold::Host(vec![3]);
        assert!(matches!(at(&three, 10, false), Some(FoldBy::To(7))));
        assert!(
            matches!(at(&three, 6, true), Some(FoldBy::To(6))),
            "short of what this fire writes"
        );
        assert!(at(&geometry::Fold::Host(vec![0]), 10, true).is_none());
        assert!(tags.is_empty() && req.lanes[0].reports == 0);
    }

    fn a_device_fold_is_reported_and_placed_when_it_settles() {
        let (stores, ws) = frozen_at(4);
        let cell = cursor(4, 12);
        let pass = lanes(&[&cell, &cell]);
        let mut req = request(2);
        let mut tags = Vec::new();
        let by = fold_at(
            &pass,
            &[0, 1],
            None,
            &geometry::Fold::Device,
            u64::MAX,
            true,
            None,
            &mut req,
            &mut tags,
        )
        .expect("a device fold is placed when it settles");
        assert!(
            req.lanes[0].reports != 0 && req.lanes[1].reports == 0,
            "the first lane carries the fold"
        );
        assert!(matches!(
            &by,
            FoldBy::Settled {
                count: None,
                reach: 12,
                ..
            }
        ));
        advance_cursors(
            &pass,
            None,
            &[],
            &geometry::Fold::Device,
            std::slice::from_ref(&cell),
        );
        let now = cell.lock().unwrap().clone();
        assert!(!now.folded_exact && now.unsettled == 1 && now.tail.is_none());

        record_kv_writes(vec![report(req.lanes[0].reports, 3)]);
        let reported = taken(&tags);
        let by = FoldBy::Settled {
            count: None,
            tags: tags.iter().map(|tag| tag.0).collect(),
            reach: 12,
            cursor: cell.clone(),
            rs: None,
        };
        settle_fold(&stores, ws, FoldAt { by, decoded: true }, &reported, None);
        assert_eq!(stores.kv.lock().folded(ws).unwrap(), 7);
        let now = cell.lock().unwrap().clone();
        assert!(now.folded_exact && now.unsettled == 0 && now.folded == 7);
    }

    fn the_last_fold_to_settle_makes_the_cursor_exact() {
        let (stores, ws) = frozen_at(4);
        let cell = cursor(4, 0);
        {
            let mut now = cell.lock().unwrap();
            now.unsettled = 2;
            now.folded_exact = false;
        }
        let settled = |count: u32, reach: u64| FoldAt {
            by: FoldBy::Settled {
                count: Some(count),
                tags: Vec::new(),
                reach,
                cursor: cell.clone(),
                rs: None,
            },
            decoded: false,
        };
        let none = std::collections::HashMap::new();
        settle_fold(&stores, ws, settled(2, 20), &none, None);
        assert!(!cell.lock().unwrap().folded_exact, "another is on its way");
        // Counted from where the first left it, short of what the fire wrote.
        settle_fold(&stores, ws, settled(9, 10), &none, None);
        assert_eq!(stores.kv.lock().folded(ws).unwrap(), 10);
        let now = cell.lock().unwrap().clone();
        assert!(now.folded_exact && now.folded == 10);
    }

    fn a_fold_never_reported_leaves_the_cursor_a_bound() {
        let (stores, ws) = frozen_at(4);
        let cell = cursor(4, 0);
        {
            let mut now = cell.lock().unwrap();
            now.unsettled = 1;
            now.folded_exact = false;
        }
        let by = FoldBy::Settled {
            count: None,
            tags: vec![u64::MAX],
            reach: 20,
            cursor: cell.clone(),
            rs: None,
        };
        settle_fold(
            &stores,
            ws,
            FoldAt { by, decoded: false },
            &std::collections::HashMap::new(),
            None,
        );
        assert_eq!(stores.kv.lock().folded(ws).unwrap(), 4);
        let now = cell.lock().unwrap().clone();
        assert!(!now.folded_exact && now.unsettled == 1);
    }

    // A fold the device counts, between its fire's submit and its settle:
    // the set's recurrent state is not forked, kept or moved, and what
    // waits for it is woken by the fire's end, however it ends.

    use super::lifecycle_tests::{TestContext, resolve_test_completion};
    use crate::store::rs::{RsError, RsWorkingSetId, device_folds_settled};
    use futures::FutureExt;

    /// One fire in flight on a pipeline, on a working set frozen at 4 in
    /// both stores with 12 slots reserved.
    struct Flight {
        stores: crate::store::registry::Stores,
        fires: PendingFires,
        completion: crate::engine::WorkItemCompletion,
        failure: PipelineFailure,
        kv: KvWorkingSet,
        rs: RsWorkingSetId,
        /// Where the pass's fires are staged from.
        cursor: CursorCell,
        /// What its first lane reports under, where the device counts.
        tag: u64,
    }

    impl Flight {
        /// The pass's next fire, submitted behind the one in flight: its
        /// completion, and what its first lane reports under.
        fn next(&self, fold: &geometry::Fold) -> (crate::engine::WorkItemCompletion, u64) {
            submit(
                &self.kv,
                Some(self.rs),
                &self.cursor,
                &self.fires,
                &self.failure,
                fold,
                None,
            )
        }

        fn fork(&self) -> Result<u32, RsError> {
            let mut rs = self.stores.rs.lock().unwrap();
            let child = rs.fork(self.rs)?;
            rs.frozen(child)
        }

        fn exported(&self) -> Result<u32, RsError> {
            let rs = self.stores.rs.lock().unwrap();
            rs.export(self.rs).map(|_| rs.frozen(self.rs).unwrap())
        }

        fn frozen(&self) -> (u64, u32) {
            (
                self.stores.kv.lock().folded(self.kv.id).unwrap(),
                self.stores.rs.lock().unwrap().frozen(self.rs).unwrap(),
            )
        }

        fn context() -> TestContext {
            TestContext {
                id: uuid::Uuid::new_v4(),
                resources: wasmtime::component::ResourceTable::new(),
                tails: Vec::new(),
            }
        }
    }

    /// `exact` is whether the host knew the frozen length when it planned
    /// the fold; `reach` where the fire's own rows end, where it knew.
    fn in_flight(fold: &geometry::Fold, exact: bool, reach: Option<u64>) -> Flight {
        in_flight_on(fold, exact, reach, true)
    }

    /// `recurrent` is whether the pass keeps recurrent state: a pass that
    /// keeps none folds its set with the recurrent store left out.
    fn in_flight_on(
        fold: &geometry::Fold,
        exact: bool,
        reach: Option<u64>,
        recurrent: bool,
    ) -> Flight {
        let model = crate::store::registry::register_model(4, &[8], &[1]);
        let stores = crate::store::registry::get(model, 0);
        let id = {
            let mut kv = stores.kv.lock();
            let id = kv.create_working_set();
            kv.fold(id, 4, false).unwrap();
            id
        };
        let rs = {
            let mut rs = stores.rs.lock().unwrap();
            let ws = rs.create_working_set(crate::store::rs::RsGeometry {
                state_size: 8,
                buffer_page_tokens: 4,
            });
            rs.fold(ws, 4);
            ws
        };
        let cell = cursor(4, 12);
        cell.lock().unwrap().folded_exact = exact;
        let kv = KvWorkingSet::new(model, 0, id, 8);
        let failure: PipelineFailure = Arc::new(Mutex::new(None));
        let fires = Arc::new(PendingFireQueue::new());
        let part = recurrent.then_some(rs);
        let (completion, tag) = submit(&kv, part, &cell, &fires, &failure, fold, reach);
        Flight {
            stores,
            fires,
            completion,
            failure,
            kv,
            rs,
            cursor: cell,
            tag,
        }
    }

    /// A fire of a pass on `kv` and `rs`, submitted: staged on what
    /// `cursor` states now, its set marked, queued on `fires` behind the
    /// pass's fires in flight, the cursor moved on. Its completion, and
    /// what its first lane reports under, where the device counts.
    fn submit(
        kv: &KvWorkingSet,
        rs: Option<RsWorkingSetId>,
        cursor: &CursorCell,
        fires: &PendingFires,
        failure: &PipelineFailure,
        fold: &geometry::Fold,
        reach: Option<u64>,
    ) -> (crate::engine::WorkItemCompletion, u64) {
        let stores = crate::store::registry::get(kv.model, kv.engine);
        let pass = lanes(&[cursor]);
        let mut req = request(1);
        let mut tags = Vec::new();
        let by = fold_at(
            &pass,
            &[0],
            reach,
            fold,
            u64::MAX,
            true,
            rs,
            &mut req,
            &mut tags,
        )
        .expect("the fire folds");
        let completion = crate::engine::WorkItemCompletion::deferred_with_guard(None);
        let mut parts = vec![FireWs {
            ws: kv.id,
            kvtxn: KvTxnGuard::new(kv.model, kv.engine, None),
            log: None,
            lease: kv.fire_lease().expect("lease the submitted working set"),
            fold: Some(FoldAt { by, decoded: false }),
            reports: tags,
            staged: None,
        }];
        mark_device_folds(&stores, &completion, &mut parts);
        let settles = settling(&parts);
        fires
            .lock()
            .unwrap()
            .push_back(PendingOp::Fire(PendingFire {
                completion: completion.clone(),
                checkpoints: Vec::new(),
                kv: parts,
                device_geometry: true,
                rstxn: RsTxnsGuard::new(kv.model, kv.engine, None),
                model: kv.model,
                engine: kv.engine,
                fwd_rep: u32::MAX,
                cells: Vec::new(),
                failure: failure.clone(),
                retired: None,
            }));
        advance_cursors(&pass, None, &[], fold, &settles);
        (completion, req.lanes[0].reports)
    }

    /// A process at its end, as the SDK's `run_ahead` leaves it: its
    /// pipeline closed with its last fire still in flight, a fire that folds
    /// by a count the device computes a set that keeps no recurrent state,
    /// frozen at 4 and still in the process's table.
    pub(crate) struct Ending {
        resources: wasmtime::component::ResourceTable,
        pub(crate) residency: Arc<Mutex<crate::inferlet::process::ProcessResidency>>,
        /// The set's rep in the table.
        pub(crate) ws: u32,
        pub(crate) cursor: CursorCell,
        pub(crate) stores: crate::store::registry::Stores,
        completion: crate::engine::WorkItemCompletion,
        tag: u64,
        /// Stands where the pass does. A pass dropped with a fire on its way
        /// is handed to its last pending fire ([`retire_pass`]) and keeps
        /// its pipeline's queue until that fire is finalized; no pass can
        /// be bound without an engine.
        _pass: PendingFires,
    }

    impl Ending {
        /// The process's table, as its end hands it to the teardown.
        pub(crate) fn table(&mut self) -> wasmtime::component::ResourceTable {
            std::mem::replace(
                &mut self.resources,
                wasmtime::component::ResourceTable::new(),
            )
        }

        /// The device ran the fire to its end and counted `fold` more
        /// slots frozen.
        pub(crate) fn settle(&self, fold: u32) {
            record_kv_writes(vec![report(self.tag, fold)]);
            resolve_test_completion(&self.completion, false);
        }
    }

    /// `dropped` is whether the guest dropped its pipeline too, as the end
    /// of its `main` does, or only closed it.
    pub(crate) async fn ending(dropped: bool) -> Anyhow<Ending> {
        use crate::inferlet::process::residency::ResidentPipeline;
        let model = crate::store::registry::register_model(4, &[8], &[1]);
        let stores = crate::store::registry::get(model, 0);
        let id = {
            let mut kv = stores.kv.lock();
            let id = kv.create_working_set();
            kv.fold(id, 4, false).unwrap();
            id
        };
        let kv = KvWorkingSet::new(model, 0, id, 8);
        let cell = cursor(4, 12);
        let mut ctx = Flight::context();
        let pipeline = ctx.resources.push(Pipeline::new())?;
        let rep = pipeline.rep();
        let (fires, failure, scope) = {
            let pipeline = ctx.resources.get(&pipeline)?;
            (
                pipeline.fires.clone(),
                pipeline.failure.clone(),
                pipeline.scope.clone(),
            )
        };
        // As the host registers a pipeline its guest makes.
        let residency = Arc::new(Mutex::new(
            crate::inferlet::process::ProcessResidency::default(),
        ));
        residency.lock().unwrap().pipelines.push(ResidentPipeline {
            scope: scope.clone(),
            fires: Arc::downgrade(&fires),
        });
        assert_eq!(kv.claim_pipeline_scope(&scope), Ok(()));
        let fold = geometry::Fold::Device;
        let (completion, tag) = submit(&kv, None, &cell, &fires, &failure, &fold, None);
        pipeline_close(&mut ctx, Resource::new_borrow(rep)).await?;
        if dropped {
            pipeline_drop(&mut ctx, pipeline).await?;
        }
        assert!(
            !fires.lock().unwrap().is_empty() && cell.lock().unwrap().at_rest().is_err(),
            "a close does not wait for a fire in flight"
        );
        let kv = ctx.resources.push(kv)?.rep();
        let ws = crate::inferlet::host::working_set::WorkingSet {
            kv,
            rs: None,
            cursor: cell.clone(),
        };
        let ws = ctx.resources.push(ws)?.rep();
        Ok(Ending {
            resources: ctx.resources,
            residency,
            ws,
            cursor: cell,
            stores,
            completion,
            tag,
            _pass: fires,
        })
    }

    /// The in-pass fork: the guest forks a set between its own pipelined
    /// fires. The host fork runs the pipeline's fires to their end
    /// ([`drain_pipeline_fires`], which only its own task can); a handle's
    /// fork, a save and the planner's suspend wait on the store and on the
    /// fire's lease. None ends before the fire settles, and each then reads
    /// the frozen length the device's count left.
    #[tokio::test(flavor = "current_thread")]
    async fn what_copies_or_moves_a_set_waits_for_the_fold_the_device_counts() -> Anyhow<()> {
        let flight = in_flight(&geometry::Fold::Device, true, None);
        let mut ctx = Flight::context();
        assert_eq!(flight.fork(), Err(RsError::DeviceFoldUnsettled));
        assert_eq!(flight.exported(), Err(RsError::DeviceFoldUnsettled));
        let sets = [flight.rs];
        let mut waiting = Box::pin(device_folds_settled(&flight.stores.rs, &sets));
        assert!(waiting.as_mut().now_or_never().is_none());
        let suspend = flight.kv.suspend_handle();
        let mut quiesce = Box::pin(suspend.quiesce());
        assert!(quiesce.as_mut().now_or_never().is_none());
        {
            let mut drain = Box::pin(drain_pipeline_fires(&mut ctx, &flight.fires));
            assert!(
                drain.as_mut().now_or_never().is_none(),
                "the fork goes no further than the fire in flight"
            );
            assert_eq!(flight.frozen(), (4, 4), "what a copy made now would state");
            record_kv_writes(vec![report(flight.tag, 3)]);
            resolve_test_completion(&flight.completion, false);
            drain.await?;
        }
        assert!(waiting.as_mut().now_or_never().is_some());
        assert!(quiesce.as_mut().now_or_never().is_some());
        assert_eq!(flight.frozen(), (7, 7));
        assert_eq!(flight.fork(), Ok(7), "the child takes what the fold left");
        assert_eq!(flight.exported(), Ok(7));
        assert!(flight.failure.lock().unwrap().is_none());
        Ok(())
    }

    /// A set that keeps no recurrent state is folded by the device's count
    /// too, and nothing in the recurrent store marks it: its cursor does.
    /// The fork reads the cursor only at rest ([`cursor_at_rest`]): it goes
    /// no further than the fire in flight, nor than one another finalizer
    /// holds off the queue, and the child then starts from the frozen
    /// length the count left, the one its pages are frozen to.
    #[tokio::test(flavor = "current_thread")]
    async fn a_set_with_no_recurrent_state_forks_at_the_length_its_fold_leaves() -> Anyhow<()> {
        // Who settles the fire: the fork's own drain, or another finalizer
        // that took it off the queue under the gate before the fork began.
        for other_finalizer in [false, true] {
            let flight = in_flight_on(&geometry::Fold::Device, true, None, false);
            let (mut ctx, mut other) = (Flight::context(), Flight::context());
            assert_eq!(
                flight.stores.rs.lock().unwrap().device_fold_gate(flight.rs),
                Ok(()),
                "the recurrent store has nothing of this pass to mark"
            );
            let staged = flight.cursor.lock().unwrap().clone();
            assert!(
                staged.at_rest().is_err() && staged.unsettled == 1 && staged.folded == 4,
                "what a child made now would state for good: {staged:?}"
            );
            let window = if other_finalizer {
                record_kv_writes(vec![report(flight.tag, 3)]);
                resolve_test_completion(&flight.completion, false);
                let gate = flight.fires.finalize_guard().await;
                let op = pop_settled(Some(&flight.fires)).expect("the fire settled");
                let finalized = finalize_op_await(op).await;
                assert!(flight.fires.lock().unwrap().is_empty());
                Some((gate, finalized))
            } else {
                None
            };
            let child = {
                let mut fork = Box::pin(cursor_at_rest(
                    &mut ctx,
                    &flight.fires,
                    &flight.failure,
                    &flight.kv,
                    None,
                    &flight.cursor,
                ));
                assert!(
                    fork.as_mut().now_or_never().is_none(),
                    "the fork goes no further than the fire: {other_finalizer}"
                );
                match window {
                    Some((gate, finalized)) => {
                        complete_finalize(&mut other, finalized);
                        drop(gate);
                    }
                    None => {
                        record_kv_writes(vec![report(flight.tag, 3)]);
                        resolve_test_completion(&flight.completion, false);
                    }
                }
                fork.await?.expect("the fold was placed")
            };
            assert!(
                child.at_rest().is_ok() && child.unsettled == 0 && child.folded == 7,
                "{child:?}"
            );
            let pages = {
                let mut kv = flight.stores.kv.lock();
                let forked = kv.fork(flight.kv.id).unwrap();
                kv.folded(forked).unwrap()
            };
            assert_eq!(
                u64::from(child.folded),
                pages,
                "the child states the length its pages are frozen to"
            );
            // At rest there is nothing to wait for.
            let again = cursor_at_rest(
                &mut ctx,
                &flight.fires,
                &flight.failure,
                &flight.kv,
                None,
                &flight.cursor,
            );
            let again = again.now_or_never().expect("no fire is on its way")?;
            assert_eq!(again.map(|cursor| cursor.folded), Ok(7));
        }
        Ok(())
    }

    /// A fold no fire is left to place never lets the cursor come to rest:
    /// the fork does not hang on it, and does not hand out the bound. It is
    /// refused by name where the fire was dropped unsettled or ran and
    /// brought no count (the recurrent store's name for it, where the set
    /// keeps recurrent state), and by the pipeline's failure where the fire
    /// failed: also one another finalizer holds off the queue, its failure
    /// not latched yet.
    #[tokio::test(flavor = "current_thread")]
    async fn a_fold_that_is_never_placed_refuses_the_fork_by_name() -> Anyhow<()> {
        use crate::inferlet::host::working_set::Cursor;
        for ended in ["dropped", "no count", "failed", "failed, off the queue"] {
            let flight = in_flight_on(&geometry::Fold::Device, true, None, false);
            let (mut ctx, mut other) = (Flight::context(), Flight::context());
            let mut window = None;
            match ended {
                "dropped" => flight.fires.lock().unwrap().clear(),
                "no count" => resolve_test_completion(&flight.completion, false),
                "failed" => resolve_test_completion(&flight.completion, true),
                _ => {
                    resolve_test_completion(&flight.completion, true);
                    let gate = flight.fires.finalize_guard().await;
                    let op = pop_settled(Some(&flight.fires)).expect("the fire settled");
                    let finalized = finalize_op_await(op).await;
                    assert_eq!(flight.failure.lock().unwrap().as_ref(), None);
                    window = Some((gate, finalized));
                }
            }
            let mut fork = Box::pin(cursor_at_rest(
                &mut ctx,
                &flight.fires,
                &flight.failure,
                &flight.kv,
                None,
                &flight.cursor,
            ));
            if let Some((gate, finalized)) = window {
                assert!(
                    fork.as_mut().now_or_never().is_none(),
                    "its failure is not written yet"
                );
                complete_finalize(&mut other, finalized);
                drop(gate);
            }
            let refused = fork
                .as_mut()
                .now_or_never()
                .unwrap_or_else(|| panic!("{ended}: it never hangs"))?
                .expect_err("the frozen length is not known");
            let staged = flight.cursor.lock().unwrap().clone();
            assert!(
                staged.at_rest().is_err() && staged.unsettled == 1 && staged.folded == 4,
                "{ended}: {staged:?}"
            );
            match ended {
                "dropped" | "no count" => {
                    assert_eq!(refused, format!("working set: {}", Cursor::UNPLACED));
                    assert!(flight.failure.lock().unwrap().is_none(), "{ended}");
                }
                _ => assert!(
                    refused.starts_with("pipeline failed: pipeline: forward failed"),
                    "{ended}: {refused}"
                ),
            }
        }
        // A set that keeps recurrent state is refused as its store names
        // how the fire ended.
        let flight = in_flight(&geometry::Fold::Device, true, None);
        let mut ctx = Flight::context();
        resolve_test_completion(&flight.completion, false);
        let fork = cursor_at_rest(
            &mut ctx,
            &flight.fires,
            &flight.failure,
            &flight.kv,
            Some(flight.rs),
            &flight.cursor,
        );
        let refused = fork.now_or_never().expect("it never hangs")?;
        let unreported = RsError::DeviceFoldFailed(RsDeviceFold::UNREPORTED.to_string());
        assert_eq!(
            refused.map(|cursor| cursor.folded),
            Err(unreported.to_string())
        );
        Ok(())
    }

    /// A pass that fails releases whoever waited, and what they asked for
    /// is refused by the pass's own failure: the engine's, or that of the
    /// pipeline's earlier fire.
    #[tokio::test(flavor = "current_thread")]
    async fn a_failed_pass_ends_the_wait_with_its_failure() -> Anyhow<()> {
        for earlier in [None, Some("pipeline: an earlier fire failed")] {
            let flight = in_flight(&geometry::Fold::Device, true, None);
            let mut ctx = Flight::context();
            let sets = [flight.rs];
            let mut waiting = Box::pin(device_folds_settled(&flight.stores.rs, &sets));
            assert!(waiting.as_mut().now_or_never().is_none());
            *flight.failure.lock().unwrap() = earlier.map(str::to_string);
            record_kv_writes(vec![report(flight.tag, 3)]);
            resolve_test_completion(&flight.completion, earlier.is_none());
            drain_pipeline_fires(&mut ctx, &flight.fires).await?;
            assert!(waiting.as_mut().now_or_never().is_some(), "it never hangs");
            let reason = flight
                .failure
                .lock()
                .unwrap()
                .clone()
                .expect("the pipeline failed: the host fork refuses by this");
            match earlier {
                Some(earlier) => assert_eq!(reason, earlier),
                None => assert!(
                    reason.starts_with("pipeline: forward failed") && reason.contains("test"),
                    "{reason}"
                ),
            }
            assert_eq!(
                flight.fork(),
                Err(RsError::DeviceFoldFailed(reason.clone()))
            );
            assert_eq!(flight.exported(), Err(RsError::DeviceFoldFailed(reason)));
            assert_eq!(
                flight.frozen(),
                (4, 4),
                "a count from a failed pass is not taken"
            );
        }
        Ok(())
    }

    /// A fire that fails ends its pipeline and the sets the pipeline owns:
    /// closed and drained, it passes none of them to another pipeline, and
    /// the refusal names it and its first failure. The cursor the failed
    /// fire moved when it was staged is not moved back, which is why: here
    /// its fold stays counted as on its way and its tail unknown. A set
    /// whose pipeline's fires all served passes on as it did. Neither
    /// passes on while its fire, out of the queue, is still being
    /// finalized.
    #[tokio::test(flavor = "current_thread")]
    async fn a_set_of_a_failed_pipeline_fires_on_no_other() -> Anyhow<()> {
        use crate::store::{PipelineScope, ScopeRefusal};
        for failed in [true, false] {
            let flight = in_flight(&geometry::Fold::Device, true, None);
            let mut ctx = Flight::context();
            let fires = Arc::downgrade(&flight.fires);
            let own = PipelineScope::failing_as(flight.failure.clone(), move || {
                fires.upgrade().is_none_or(|fires| fires.drained())
            });
            let rs = RsWorkingSet::new(
                flight.kv.model,
                0,
                flight.rs,
                crate::store::rs::RsGeometry {
                    state_size: 8,
                    buffer_page_tokens: 4,
                },
            );
            assert_eq!(flight.kv.claim_pipeline_scope(&own), Ok(()));
            assert_eq!(rs.claim_pipeline_scope(&own), Ok(()));

            // A fire that failed before it ran reports nothing.
            if !failed {
                record_kv_writes(vec![report(flight.tag, 3)]);
            }
            resolve_test_completion(&flight.completion, failed);
            own.close();
            let next = PipelineScope::new(|| true);

            // The fire leaves its pipeline's queue ahead of its failure
            // being written, as every finalizer takes it: under the gate.
            // Until the gate lifts the set is still its pipeline's, however
            // the fire ended.
            {
                let _gate = flight.fires.finalize_guard().await;
                let op = pop_settled(Some(&flight.fires)).expect("the fire settled");
                let finalized = finalize_op_await(op).await;
                assert!(flight.fires.lock().unwrap().is_empty());
                assert_eq!(flight.failure.lock().unwrap().as_ref(), None);
                let held = Err(ScopeRefusal::Held(own.id()));
                assert_eq!(flight.kv.claim_pipeline_scope(&next), held);
                assert_eq!(rs.claim_pipeline_scope(&next), held);
                complete_finalize(&mut ctx, finalized);
            }

            let staged = flight.cursor.lock().unwrap().clone();
            if !failed {
                assert!(staged.folded_exact && staged.unsettled == 0 && staged.folded == 7);
                assert_eq!(flight.kv.claim_pipeline_scope(&next), Ok(()));
                assert_eq!(rs.claim_pipeline_scope(&next), Ok(()));
                continue;
            }
            assert!(
                !staged.folded_exact && staged.unsettled == 1 && staged.tail.is_none(),
                "the cursor stands as the failed fire staged it: {staged:?}"
            );
            let reason = flight.failure.lock().unwrap().clone().expect("it failed");
            let refused = ScopeRefusal::Failed {
                owner: own.id(),
                reason,
            };
            assert_eq!(flight.kv.claim_pipeline_scope(&next), Err(refused.clone()));
            assert_eq!(rs.claim_pipeline_scope(&next), Err(refused.clone()));
            let text = refused.to_string();
            assert!(
                text.contains("which failed (pipeline: forward failed") && text.contains("test"),
                "{text}"
            );
        }
        Ok(())
    }

    /// A fire that ran to its end on a set whose fold the device counts
    /// owes the count: the engine's record went on by it. One that brings
    /// none, or its rows' end without it, lifts the mark and leaves the set
    /// refused by name, where a count of nothing is a count. A fire that
    /// failed keeps its own failure.
    #[tokio::test(flavor = "current_thread")]
    async fn a_fire_that_ran_and_reported_no_fold_fails_its_set_by_name() -> Anyhow<()> {
        type Brought = fn(u64) -> Option<::engine::KvWrites>;
        let unreported = || RsError::DeviceFoldFailed(RsDeviceFold::UNREPORTED.to_string());
        // What the engine brought under the lane's tag, whether its fire
        // ran, and where the fold left the set frozen, where it is known.
        let cases: [(&str, Brought, bool, Option<u32>); 5] = [
            ("no report", |_| None, true, None),
            (
                "the rows' end alone",
                |tag| {
                    Some(::engine::KvWrites {
                        tag,
                        end: Some(9),
                        ..::engine::KvWrites::default()
                    })
                },
                true,
                None,
            ),
            (
                "a fold of nothing",
                |tag| Some(report(tag, 0)),
                true,
                Some(4),
            ),
            ("a fold of three", |tag| Some(report(tag, 3)), true, Some(7)),
            ("no report from a fire that failed", |_| None, false, None),
        ];
        for (what, brought, ran, left) in cases {
            let flight = in_flight(&geometry::Fold::Device, true, None);
            let mut ctx = Flight::context();
            let sets = [flight.rs];
            let mut waiting = Box::pin(device_folds_settled(&flight.stores.rs, &sets));
            assert!(waiting.as_mut().now_or_never().is_none(), "{what}");
            if let Some(written) = brought(flight.tag) {
                record_kv_writes(vec![written]);
            }
            resolve_test_completion(&flight.completion, !ran);
            drain_pipeline_fires(&mut ctx, &flight.fires).await?;
            assert!(
                waiting.as_mut().now_or_never().is_some(),
                "{what}: the mark lifts however the fire ended"
            );
            let failed = flight.failure.lock().unwrap().clone();
            assert_eq!(
                failed.is_none(),
                ran,
                "{what}: the pass fails only by its fire"
            );
            match (left, failed) {
                (Some(left), _) => {
                    assert_eq!(flight.frozen(), (u64::from(left), left), "{what}");
                    assert_eq!(flight.fork(), Ok(left), "{what}");
                    assert_eq!(flight.exported(), Ok(left), "{what}");
                    assert!(flight.cursor.lock().unwrap().folded_exact, "{what}");
                }
                (None, None) => {
                    assert_eq!(flight.frozen(), (4, 4), "{what}: nothing is frozen");
                    assert_eq!(flight.fork(), Err(unreported()), "{what}");
                    assert_eq!(flight.exported(), Err(unreported()), "{what}");
                    let now = flight.cursor.lock().unwrap().clone();
                    assert!(
                        !now.folded_exact && now.unsettled == 1,
                        "{what}: its fires go on from the bound"
                    );
                }
                (None, Some(reason)) => {
                    assert_ne!(reason, RsDeviceFold::UNREPORTED, "{what}");
                    assert_eq!(
                        flight.fork(),
                        Err(RsError::DeviceFoldFailed(reason)),
                        "{what}: the fire's own failure names it"
                    );
                }
            }
        }
        Ok(())
    }

    /// A fold the host counts states the frozen length at submit: nothing
    /// waits on its fire.
    #[tokio::test(flavor = "current_thread")]
    async fn a_fold_the_host_counts_delays_nothing() -> Anyhow<()> {
        let host = geometry::Fold::Host(vec![2]);
        for (exact, reach) in [(true, Some(10)), (false, None)] {
            let flight = in_flight(&host, exact, reach);
            let mut ctx = Flight::context();
            let sets = [flight.rs];
            assert_eq!(flight.tag, 0, "the host needs no report");
            assert!(
                device_folds_settled(&flight.stores.rs, &sets)
                    .now_or_never()
                    .is_some()
            );
            assert_eq!(flight.fork(), Ok(4));
            assert_eq!(flight.exported(), Ok(4));
            let mut drain = Box::pin(drain_pipeline_fires(&mut ctx, &flight.fires));
            assert!(
                drain.as_mut().now_or_never().is_none(),
                "while its fire is still in flight"
            );
            resolve_test_completion(&flight.completion, false);
            drain.await?;
            assert_eq!(flight.frozen().0, 6);
        }
        Ok(())
    }

    /// A fire dropped before anything settled it (its process torn down)
    /// still lifts the mark: when the engine is done with it, or when the
    /// engine itself is gone and its completion with it. What it folded is
    /// then not known, and the set says so.
    #[test]
    fn a_fire_dropped_unsettled_lifts_the_mark_when_the_engine_is_done_with_it() {
        let dropped = || RsError::DeviceFoldFailed(RsDeviceFold::DROPPED.to_string());
        for lost in [false, true] {
            let Flight {
                stores,
                fires,
                completion,
                rs,
                ..
            } = in_flight(&geometry::Fold::Device, true, None);
            let sets = [rs];
            let mut waiting = Box::pin(device_folds_settled(&stores.rs, &sets));
            drop(fires);
            assert_eq!(
                stores.rs.lock().unwrap().device_fold_gate(rs),
                Err(RsError::DeviceFoldUnsettled),
                "the engine may still run it"
            );
            assert!(waiting.as_mut().now_or_never().is_none());
            if lost {
                drop(completion);
            } else {
                resolve_test_completion(&completion, false);
            }
            assert!(waiting.as_mut().now_or_never().is_some());
            assert_eq!(
                stores.rs.lock().unwrap().device_fold_gate(rs),
                Err(dropped())
            );
        }
    }

    // A pass that runs ahead submits a frame's fires one after another, and
    // the scheduler dispatches none of them before the frame's last has
    // arrived. A fire behind one that folds by a count the device computes
    // is therefore staged before that count exists.

    use crate::scheduler::frame::{FramePlan, FramePolicy, FrameStamp, QueuedFireIds};

    /// The scheduler's frames, two fires each, as one lane fills them.
    struct Frames {
        policy: FramePolicy,
        lane: uuid::Uuid,
        arrived: Vec<u64>,
    }

    impl Frames {
        fn of_two(lane: uuid::Uuid) -> Self {
            Self {
                policy: FramePolicy::new(2, 64, 4096, None),
                lane,
                arrived: Vec::new(),
            }
        }

        /// The fire in `slot` of the lane's first frame reaches the
        /// scheduler, under its slot as its id.
        fn arrive(&mut self, slot: u32) {
            let stamp = FrameStamp {
                lane: self.lane,
                seq: 0,
                slot,
                fires: 2,
            };
            let fire = u64::from(slot);
            self.policy
                .on_fire_enqueued(stamp, Some(self.lane), fire, 1, 1);
            self.arrived.push(fire);
        }

        /// The fires the scheduler dispatches now, wave by wave: none while
        /// it holds the frame for a fire still to arrive.
        fn dispatched(&mut self) -> Vec<Vec<u64>> {
            let queued: QueuedFireIds = self.arrived.iter().copied().collect();
            let none_blocked = std::collections::HashSet::new();
            match self.policy.plan_dispatch(
                &queued,
                &none_blocked,
                false,
                crate::rt::Instant::now(),
            ) {
                FramePlan::Dispatch(waves) => waves,
                FramePlan::Hold(_) | FramePlan::Park => Vec::new(),
                FramePlan::Terminate(lanes) => panic!("the scheduler gave up on {lanes:?}"),
            }
        }
    }

    /// The first fire of a frame of two is submitted, folding by a count
    /// the device computes: the scheduler holds it for the second, and the
    /// host no longer knows where the set is frozen.
    fn first_of_a_frame() -> (Flight, Frames, TestContext) {
        let ctx = Flight::context();
        let mut frames = Frames::of_two(ctx.id);
        let flight = in_flight(&geometry::Fold::Device, true, None);
        frames.arrive(0);
        assert!(
            frames.dispatched().is_empty(),
            "a frame is dispatched whole"
        );
        let cursor = flight.cursor.lock().unwrap().clone();
        assert!(!cursor.folded_exact && cursor.unsettled == 1 && cursor.folded == 4);
        (flight, frames, ctx)
    }

    /// The second fire is staged on the frozen length as a lower bound,
    /// behind the first: the frame seals and dispatches both, and the
    /// counts they report, taken in order, leave the set frozen where the
    /// device left it, in both stores and on the cursor.
    #[tokio::test(flavor = "current_thread")]
    async fn a_frame_of_fires_that_fold_by_a_device_count_seals_and_dispatches() -> Anyhow<()> {
        let device = geometry::Fold::Device;
        let (flight, mut frames, mut ctx) = first_of_a_frame();
        let (second, tag) = flight.next(&device);
        assert!(tag != 0 && tag != flight.tag, "each fire reports its own");
        let cursor = flight.cursor.lock().unwrap().clone();
        assert!(!cursor.folded_exact && cursor.unsettled == 2 && cursor.folded == 4);
        frames.arrive(1);
        assert_eq!(
            frames.dispatched(),
            [[0], [1]],
            "the frame is whole: both fires go, in slot order"
        );
        assert_eq!(flight.fork(), Err(RsError::DeviceFoldUnsettled));
        for (completion, tag) in [(&flight.completion, flight.tag), (&second, tag)] {
            record_kv_writes(vec![report(tag, 3)]);
            resolve_test_completion(completion, false);
        }
        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            drain_pipeline_fires(&mut ctx, &flight.fires),
        )
        .await
        .expect("both fires settle")?;
        assert_eq!(flight.frozen(), (10, 10), "the first fold, then the second");
        let cursor = flight.cursor.lock().unwrap().clone();
        assert!(cursor.folded_exact && cursor.unsettled == 0 && cursor.folded == 10);
        assert_eq!(flight.fork(), Ok(10));
        assert!(flight.failure.lock().unwrap().is_none());
        Ok(())
    }

    /// The control. Until the second fire was staged on a lower bound, the
    /// submit of a recurrent fire first ran the pass's fires in flight to
    /// their end wherever a frozen length was not exact. In a frame that
    /// wait is for a fire the scheduler dispatches only once the fire that
    /// waits has arrived: it does not end, and the frame never fills.
    #[tokio::test(flavor = "current_thread")]
    async fn a_fire_that_waited_for_the_fold_before_it_would_hold_its_own_frame() {
        let (flight, mut frames, mut ctx) = first_of_a_frame();
        let waited = tokio::time::timeout(
            std::time::Duration::from_millis(250),
            drain_pipeline_fires(&mut ctx, &flight.fires),
        )
        .await;
        assert!(waited.is_err(), "nothing ends the wait");
        assert!(
            frames.dispatched().is_empty(),
            "the fire it waits for is held for the fire that waits"
        );
        assert_eq!(flight.frozen(), (4, 4));
    }
}

#[cfg(test)]
mod unfolded_tests {
    use super::*;
    use crate::inferlet::host::working_set::Cursor;

    /// A lane's rows in a fire: its own, and what it carries in unfolded.
    /// A host-placed lane carries the rows between its state and its first
    /// row; a lane with no recurrent state, nothing.
    #[test]
    fn a_lane_counts_its_own_rows_and_the_rows_it_carries() {
        let host = lane_rows;
        assert_eq!(host(105, 113, Some(100)), FireRows { own: 8, carried: 5 });
        assert_eq!(host(100, 101, Some(100)), FireRows { own: 1, carried: 0 });
        assert_eq!(host(105, 113, None), FireRows { own: 8, carried: 0 });
        // A fire's are its lanes' summed.
        assert_eq!(
            FireRows::of([host(105, 113, Some(100)), host(102, 107, Some(100))]),
            FireRows {
                own: 13,
                carried: 7
            }
        );
    }

    /// A set whose prompt the host placed through slot 105, its state
    /// folded through 100, with `reserved` slots granted.
    fn decoding(reserved: u32) -> Cursor {
        Cursor {
            folded: 100,
            tail: Some(105),
            reserved,
            written: 105,
            ..Cursor::default()
        }
    }

    /// A lane the device places carries no more than stands unfolded in
    /// its set: the rows between its state and the set's written end, and
    /// every row of the fires staged and not yet settled. With `n` fires
    /// of two rows staged since the last settle it asks for `5 + 2n`.
    #[test]
    fn a_device_lane_asks_for_what_stands_unfolded_in_its_set() {
        use crate::inferlet::host::working_set::StagedRows;
        let mut set = decoding(184);
        assert_eq!(
            device_lane_rows(2, &set, Some(100)),
            FireRows { own: 2, carried: 5 }
        );
        assert_eq!(device_lane_rows(2, &set, None).carried, 0);
        let mut fires = Vec::new();
        for n in 1..=10u32 {
            let fire = StagedRows::new(2);
            set.stage(fire.clone());
            fires.push(fire);
            let rows = device_lane_rows(2, &set, Some(100));
            assert_eq!(rows.carried, 5 + 2 * n);
            assert_eq!(rows.ask(), 5 + 2 * n);
        }
        // The first settles, its rows ended at 107 and one slot frozen:
        // its rows are counted where they landed, not as staged.
        fires[0].settle(Some(107));
        assert_eq!(set.standing(), (107, 18));
        assert_eq!(device_lane_rows(2, &set, Some(100)).carried, 25);
        assert_eq!(device_lane_rows(2, &set, Some(101)).carried, 24);
        // The second rewound and ended at 106: the rows through 107 are
        // still buffered, so the written end stays.
        fires[1].settle(Some(106));
        assert_eq!(set.standing(), (107, 16));
        // The rest settle; the set froze through 107: nothing is carried,
        // and the fire asks for its own rows.
        for fire in &fires[2..] {
            fire.settle(Some(107));
        }
        assert_eq!(
            device_lane_rows(2, &set, Some(107)),
            FireRows { own: 2, carried: 0 }
        );
        assert_eq!(device_lane_rows(2, &set, Some(107)).ask(), 2);
        // Staging takes in what settled.
        set.stage(StagedRows::new(2));
        assert_eq!((set.written, set.staged.len()), (107, 1));
        // A fire's rows are its lanes' summed, each lane of a set carrying
        // what stands unfolded in it.
        let lanes = [
            device_lane_rows(2, &set, Some(107)),
            device_lane_rows(3, &set, Some(107)),
        ];
        assert_eq!(FireRows::of(lanes), FireRows { own: 5, carried: 4 });
    }

    /// A fire that settles with no end reported (it failed, or its engine
    /// tells none) wrote inside what its set reserved; a fork made while a
    /// fire is in flight shares its rows, and counts them until it settles.
    #[test]
    fn a_fire_with_no_end_wrote_inside_what_was_reserved_and_a_fork_shares_it() {
        use crate::inferlet::host::working_set::StagedRows;
        let mut set = decoding(184);
        let fire = StagedRows::new(2);
        set.stage(fire.clone());
        let fork = set.clone();
        assert_eq!(fork.unfolded(100), 7);
        assert_eq!(fork.written_end(), 184, "a snapshot states no end it lacks");
        fire.settle(Some(107));
        assert_eq!((set.unfolded(100), fork.unfolded(100)), (7, 7));
        assert_eq!((set.standing(), fork.standing()), ((107, 0), (107, 0)));
        assert_eq!(fork.written_end(), 107);

        let fire = StagedFire {
            rows: StagedRows::new(2),
            tags: vec![7],
        };
        set.stage(fire.rows.clone());
        // Its lane reported, but the fire failed.
        let reported = std::collections::HashMap::from([(
            7,
            ::engine::KvWrites {
                tag: 7,
                end: Some(109),
                ..::engine::KvWrites::default()
            },
        )]);
        fire.settle(&reported, false);
        assert_eq!(set.standing(), (184, 0));

        let mut set = decoding(184);
        let fire = StagedFire {
            rows: StagedRows::new(2),
            tags: vec![7, 8],
        };
        set.stage(fire.rows.clone());
        // One of its two lanes reported no end.
        fire.settle(&reported, true);
        assert_eq!(set.standing(), (184, 0));

        let mut set = decoding(184);
        let fire = StagedFire {
            rows: StagedRows::new(2),
            tags: vec![7],
        };
        set.stage(fire.rows.clone());
        fire.settle(&reported, true);
        assert_eq!(set.standing(), (109, 0));

        // Dropped unsettled.
        let mut set = decoding(184);
        let fire = StagedFire {
            rows: StagedRows::new(2),
            tags: vec![7],
        };
        set.stage(fire.rows.clone());
        drop(fire);
        assert_eq!(set.standing(), (184, 0));
    }

    /// The bound is never below what the engine replays: over runs of
    /// fires staged ahead, each landing wherever a lane may (at or past
    /// the engine's folded length, no further than rows were written),
    /// folding what its program says, and settling later, the rows
    /// `buffered_reads` returns for the fire are no more than it stated
    /// when it was staged.
    #[test]
    fn the_bound_is_never_below_what_the_engine_replays() {
        use crate::inferlet::host::working_set::StagedRows;
        struct Fire {
            own: u32,
            stated: u32,
            rows: Arc<StagedRows>,
            /// Once run: where its rows ended, and what it folded.
            ran: Option<(u32, u32)>,
        }
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut pick = |below: u32| {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((seed >> 33) % u64::from(below.max(1))) as u32
        };
        let mut replayed = 0u32;
        for depth in [1usize, 2, 10] {
            let mut set = decoding(4096);
            // The runtime's frozen length, the engine's folded length (past
            // it while a device-counted fold has not settled), and the
            // furthest slot written.
            let (mut frozen, mut folded, mut end) = (100u32, 100u32, 105u32);
            let mut fires: std::collections::VecDeque<Fire> = Default::default();
            for _ in 0..4000 {
                let running = fires.iter().position(|fire| fire.ran.is_none());
                match pick(3) {
                    0 if fires.len() < depth => {
                        let own = 1 + pick(5);
                        let stated = device_lane_rows(own, &set, Some(frozen)).carried;
                        let rows = StagedRows::new(own);
                        set.stage(rows.clone());
                        fires.push_back(Fire {
                            own,
                            stated,
                            rows,
                            ran: None,
                        });
                    }
                    1 if running.is_some() => {
                        let fire = &mut fires[running.unwrap()];
                        // Anywhere from its state to the furthest row written:
                        // on, or rewound over rows it rejects.
                        let first = folded + pick(end - folded + 1);
                        let reads =
                            ::engine::fire::buffered_reads(None, folded, first, fire.own).unwrap();
                        let carried: u32 = reads.iter().map(|span| span.end - span.start).sum();
                        assert!(
                            carried <= fire.stated,
                            "replays {carried} rows, stated {}",
                            fire.stated
                        );
                        replayed = replayed.max(carried);
                        // Most fires fold nothing (their rows stand buffered);
                        // one in four folds any part of what stands.
                        let fold = match pick(4) {
                            0 => pick(first + fire.own - folded + 1),
                            _ => 0,
                        };
                        folded += fold;
                        end = end.max(first + fire.own);
                        fire.ran = Some((first + fire.own, fold));
                    }
                    _ => {
                        if fires.front().is_some_and(|fire| fire.ran.is_some()) {
                            let fire = fires.pop_front().unwrap();
                            let (ended, fold) = fire.ran.unwrap();
                            fire.rows.settle(Some(ended));
                            frozen += fold;
                        }
                    }
                }
            }
        }
        assert!(replayed > 16, "the runs replay past what a rung of 16 held");
    }

    /// The rows a device-placed fire may carry raise the rung through the
    /// ask every pass makes: the fire enqueues once the rung takes them.
    #[tokio::test(flavor = "current_thread")]
    async fn a_device_fire_asks_the_rung_for_the_rows_it_may_carry() {
        use crate::inferlet::host::working_set::StagedRows;
        const ENGINE: usize = 48;
        let stores = pool_of_one_row_rs_pages();
        let pid = uuid::Uuid::new_v4();
        crate::store::shared::set_fire_cap_for_test(ENGINE, 8);
        // Two rows of its own, behind ten fires of two rows still to
        // settle and five rows unfolded.
        let mut set = decoding(184);
        for _ in 0..10 {
            set.stage(StagedRows::new(2));
        }
        let rows = device_lane_rows(2, &set, Some(100)).ask();
        assert_eq!(rows, 25);
        let raise = tokio::spawn(async {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            assert_eq!(crate::store::shared::wanted_rows(ENGINE), 25);
            crate::store::shared::set_fire_cap_for_test(ENGINE, 32);
        });
        let ask = ask_rung(&stores, ENGINE, pid, pid, rows as usize, rows).expect("it asks");
        tokio::time::timeout(std::time::Duration::from_secs(5), ask.wait(rows))
            .await
            .expect("no indefinite wait")
            .expect("the rung takes it");
        raise.await.unwrap();
        assert!(crate::store::shared::fire_cap(ENGINE) >= 25);
        drop(ask);
        assert_eq!(crate::store::shared::wanted_rows(ENGINE), 0);
    }

    /// A fire carrying no more rows than its own asks its rung for what it
    /// asked before: its own rows, in the pass's ask and in what the
    /// scheduler steps and holds it by.
    #[test]
    fn a_fire_carrying_no_more_than_its_own_rows_asks_as_before() {
        const ENGINE: usize = 47;
        for carried in [0, 1, 7, 8] {
            assert_eq!(FireRows { own: 8, carried }.ask(), 8);
            let mut req = lane(100 + carried, 8);
            let plain = req.rung_rows();
            req.carried = carried;
            assert_eq!(req.rung_rows(), plain);
            assert_eq!(plain, 8);
        }
        // A decode of one row carrying none, or the one row before it: a
        // lone row the cap covers asks nothing.
        let stores = pool_of_one_row_rs_pages();
        let pid = uuid::Uuid::new_v4();
        crate::store::shared::set_fire_cap_for_test(ENGINE, 1);
        for carried in [0, 1] {
            let decode = FireRows { own: 1, carried };
            let rows = decode.ask();
            assert!(ask_rung(&stores, ENGINE, pid, pid, rows as usize, rows).is_none());
        }
        assert_eq!(crate::store::shared::wanted_rows(ENGINE), 0);
    }

    /// A fire carrying more rows than its own asks for those, as a fire
    /// that wide does: its pass asks whatever the cap, and what the
    /// scheduler holds it by is what it carries.
    #[tokio::test(flavor = "current_thread")]
    async fn a_fire_carrying_more_than_its_own_rows_asks_as_one_that_wide() {
        const ENGINE: usize = 46;
        let draft = FireRows {
            own: 5,
            carried: 40,
        };
        assert_eq!(draft.ask(), 40);
        let mut req = lane(140, 5);
        req.carried = 40;
        assert_eq!(req.rung_rows(), 40);
        // A lone row carrying a handful asks for them under a cap that
        // covers them too: the walker cuts nothing under the ask.
        let few = FireRows {
            own: 1,
            carried: 12,
        };
        let stores = pool_of_one_row_rs_pages();
        let pid = uuid::Uuid::new_v4();
        crate::store::shared::set_fire_cap_for_test(ENGINE, 16);
        let rows = few.ask();
        let ask = ask_rung(&stores, ENGINE, pid, pid, rows as usize, rows)
            .expect("a row carrying twelve asks as twelve rows do");
        assert_eq!(crate::store::shared::wanted_rows(ENGINE), 12);
        drop(ask);
        assert_eq!(crate::store::shared::wanted_rows(ENGINE), 0);
    }

    /// No rung's replay scratch holds more carried rows than the largest
    /// rung's tokens: a fire carrying more is refused by name, before
    /// anything is submitted, whatever its own rows.
    #[test]
    fn a_fire_carrying_past_the_largest_rung_is_refused_by_name() {
        let carrying = |carried| FireRows { own: 5, carried };
        assert!(check_largest_rung(carrying(0), 2048).is_ok());
        assert!(check_largest_rung(carrying(2048), 2048).is_ok());
        let host = check_largest_rung(carrying(2049), 2048).unwrap_err();
        assert!(
            host.contains("carry 2049 recurrent rows unfolded")
                && host.contains("past the 2048 tokens of the largest activation rung")
                && host.contains("fold what its lanes have written before they write more"),
            "{host}"
        );
        // Own rows past it are another refusal's (the capacity of a fire).
        assert!(
            check_largest_rung(
                FireRows {
                    own: 4096,
                    carried: 0
                },
                2048
            )
            .is_ok()
        );
        // The pool's largest rung is what is asked against.
        let stores = pool_of_one_row_rs_pages();
        let largest = largest_rung(&stores, 56);
        let pool = stores.shared.as_ref().unwrap().lock().unwrap();
        assert_eq!(largest, pool.rung_for(u32::MAX));
        assert!(pool.rungs().last().is_none_or(|&last| largest <= last));
    }

    /// A pass cut into pieces: one folding all it writes folds piece by
    /// piece, so only its first piece carries rows in; one folding nothing
    /// carries its earlier pieces' rows into each later one.
    #[test]
    fn a_cut_pass_counts_what_each_piece_carries() {
        // Folded through 12, 8 rows written since, 30 more in three pieces.
        let full = lane(20, 30);
        let frozen = [Some(12)];
        let piece = |from, to| vec![Some((from, to))];
        let cuts = [piece(0, 10), piece(10, 20), piece(20, 30)];
        let rows = |folds_each: bool| -> Vec<FireRows> {
            cuts.iter()
                .enumerate()
                .map(|(at, cut)| piece_rows(&full, cut, &frozen, folds_each && at > 0))
                .collect()
        };
        let own = |carried| FireRows { own: 10, carried };
        assert_eq!(rows(true), [own(8), own(0), own(0)]);
        assert_eq!(rows(false), [own(8), own(18), own(28)]);
        assert_eq!(
            rows(false).iter().map(|rows| rows.ask()).max(),
            Some(28),
            "the widest ask of a pass folding nothing is what its last piece carries"
        );
        // The whole pass in one fire carries what its first piece does.
        assert_eq!(
            piece_rows(&full, &piece(0, 30), &frozen, false),
            FireRows {
                own: 30,
                carried: 8
            }
        );
        // A lane with no recurrent state carries nothing; one with no rows
        // in a piece counts nothing.
        assert_eq!(piece_rows(&full, &piece(10, 20), &[None], false), own(0));
        assert_eq!(
            piece_rows(&full, &[None], &frozen, false),
            FireRows::default()
        );
    }

    pub(super) fn pool_of_one_row_rs_pages() -> crate::store::registry::Stores {
        pool_with_rungs(Vec::new())
    }

    /// That pool with activation rungs `act_chunks` (tokens, chunks), its
    /// span backed from the load: it starts at the largest.
    pub(super) fn pool_with_rungs(act_chunks: Vec<(u32, u32)>) -> crate::store::registry::Stores {
        use ::engine::chunk::{
            ChunkFacts, Growth, Reserved, ReservedRun, SegmentFacts, ShapeId, SlotKind, SlotRole,
            SlotShape,
        };
        let shape = |id, kind: SlotKind, role| SlotShape {
            id: ShapeId(id),
            kind,
            slot_bytes: 1 << 20,
            slots_per_chunk: 1,
            sets_per_working_set: kind.sets_per_working_set(),
            rows: Vec::new(),
            pinned: None,
            role,
        };
        let list = |tokens| SlotKind::List {
            tokens,
            rate: 1,
            window: None,
        };
        let facts = ChunkFacts {
            chunk_bytes: 1 << 20,
            total_chunks: 16,
            segment_chunks: 16,
            segments: SegmentFacts {
                committed: 1,
                max: 1,
                growth: Growth::Fixed,
            },
            reserved: vec![ReservedRun {
                what: Reserved::Null,
                first: 0,
                chunks: 1,
            }],
            shapes: vec![
                shape(0, list(16), SlotRole::State),
                // One recurrent row an rs page, as Qwen3.5's.
                shape(1, list(1), SlotRole::RsBuffer),
                shape(2, SlotKind::Fixed { pieces: 1 }, SlotRole::State),
            ],
            act_chunks,
            ..ChunkFacts::default()
        };
        let pool = crate::store::shared::SharedPool::new(&facts, 1024).unwrap();
        let model = crate::store::registry::register_model_with_chunks(
            16,
            &[crate::store::registry::EngineStores {
                pools: crate::store::chunks::StorePools::of(&facts).unwrap(),
                max_context: 1024,
                shared: Some(Arc::new(Mutex::new(pool))),
                tiers: crate::store::shared::Tiers::default(),
            }],
        );
        crate::store::registry::get(model, 0)
    }

    fn lane(held: u32, rows: u32) -> crate::engine::FireRequest {
        crate::engine::FireRequest {
            lanes: vec![::engine::Lane {
                tokens: vec![0; rows as usize],
                kv: ::engine::KvDelta {
                    held,
                    ..Default::default()
                },
                ..::engine::Lane::default()
            }],
            ..crate::engine::FireRequest::default()
        }
    }

    fn groups(plan: rs::RsPlan) -> Vec<rs::GroupPlan> {
        match plan {
            rs::RsPlan::Groups(groups) => groups,
            rs::RsPlan::Window(_) => panic!("a plain recurrent model plans groups"),
        }
    }

    /// A lane staged to fold by the device's count asks for its report,
    /// whichever pass stages it: the one whose rows the host places and the
    /// one whose rows the device places. The fold's verb is the store's and
    /// the tag is the fold's placing, so the verb alone carries none; a
    /// fold the host counts asks for nothing.
    #[test]
    fn a_lane_the_device_folds_is_staged_asking_for_its_report() {
        use ::engine::fire::{FoldLen, RsVerb};
        let device = geometry::Fold::Device;
        let host = geometry::Fold::Host(vec![2]);
        for (device_places, fold) in [(false, &device), (true, &device), (true, &host)] {
            let what = format!("device_places {device_places}, {fold:?}");
            let stores = pool_of_one_row_rs_pages();
            let kv = stores.kv.lock().create_working_set();
            let ws = stores
                .rs
                .lock()
                .unwrap()
                .create_working_set(crate::store::rs::RsGeometry {
                    state_size: 1 << 10,
                    buffer_page_tokens: 1,
                });
            stores.rs.lock().unwrap().fold(ws, 10);
            let ids = [ws];
            let groups_of = lane_groups(&[kv]);
            let pass = PassLanes {
                places: vec![geometry::Place {
                    folded: 10,
                    folded_exact: true,
                    tail: None,
                    shared: false,
                    spans: Vec::new(),
                }],
                cursors: vec![Arc::new(Mutex::new(Cursor {
                    folded: 10,
                    reserved: 14,
                    tail: None,
                    ..Cursor::default()
                }))],
                kv: Vec::new(),
            };
            let mut req = lane(10, 4);
            let plan = if device_places {
                device_rs_plan(&stores, &ids, &groups_of, &pass, fold)
            } else {
                host_rs_plan(&stores, &ids, &groups_of, &req, fold, &[])
            }
            .expect(&what);
            let prepared =
                rs::prepare_many(&mut stores.rs.lock().unwrap(), &ids, &plan, None).expect(&what);
            prepared.apply_to(&mut req);
            let counts = matches!(fold, geometry::Fold::Device);
            assert_eq!(
                matches!(
                    req.lanes[0].rs,
                    RsVerb::Buffer {
                        fold: FoldLen::Device,
                        ..
                    }
                ),
                counts,
                "{what}"
            );
            assert_eq!(
                untagged_device_fold(&req),
                counts.then_some(0),
                "{what}: the verb alone asks for nothing"
            );
            let mut tags = Vec::new();
            let by = fold_at(
                &pass,
                &groups_of[0].1,
                None,
                fold,
                u64::MAX,
                true,
                Some(ws),
                &mut req,
                &mut tags,
            );
            assert!(matches!(by, Some(FoldBy::Settled { .. })), "{what}");
            assert_eq!(req.lanes[0].reports != 0, counts, "{what}");
            assert_eq!(untagged_device_fold(&req), None, "{what}");
        }
    }

    /// MTP with k = 4 on a pool whose rs pages hold one row: the draft
    /// writes a window of five rows unfolded, the next window carries those
    /// five in and folds the accepted rows. Not one rs page: a lane's buffer
    /// names several. No count of rows stops a lane that never folds: its
    /// fire asks the rung for what it carries.
    #[test]
    fn mtp_k4_rides_one_row_rs_pages() {
        let stores = pool_of_one_row_rs_pages();
        let kv = stores.kv.lock().create_working_set();
        let ws = stores
            .rs
            .lock()
            .unwrap()
            .create_working_set(crate::store::rs::RsGeometry {
                state_size: 1 << 10,
                buffer_page_tokens: 1,
            });
        // The prompt is folded through 10.
        stores.rs.lock().unwrap().fold(ws, 10);
        let ids = [ws];
        let groups_of = lane_groups(&[kv]);
        let host = |held, rows, fold: Vec<u32>| {
            host_rs_plan(
                &stores,
                &ids,
                &groups_of,
                &lane(held, rows),
                &geometry::Fold::Host(fold),
                &[],
            )
        };

        // Draft: five rows (the truth and k = 4 drafts), nothing folded.
        let draft = groups(host(10, 5, vec![0]).expect("a draft of five rows"));
        assert_eq!(draft[0].writes, Some(10..15));
        assert_eq!(draft[0].fold, rs::GroupFold::None);

        // Verify: carries the draft's five in and folds the accepted three.
        let verify = groups(host(15, 5, vec![3]).expect("carries five in"));
        assert_eq!(verify[0].writes, Some(15..20));
        assert_eq!(verify[0].fold, rs::GroupFold::Host(3));
        stores.rs.lock().unwrap().fold(ws, 3);
        assert_eq!(stores.rs.lock().unwrap().frozen(ws).unwrap(), 13);

        // A lane that never folds piles up: past 13, carrying 100 in. Its
        // plan stands; the fire asks the rung for the hundred.
        let piled = groups(host(113, 5, vec![0]).expect("planned whatever it carries"));
        assert_eq!(piled[0].writes, Some(113..118));
        let frozen = lanes_frozen(&stores, &ids, &groups_of, 1, false).unwrap();
        assert_eq!(frozen, [Some(13)]);
        let counted = piece_rows(&lane(113, 5), &[Some((0, 5))], &frozen, false);
        assert_eq!(
            counted,
            FireRows {
                own: 5,
                carried: 100
            }
        );
        assert_eq!(counted.ask(), 100);
        // A host fold may leave any count behind it: the next fire asks.
        let left = groups(host(13, 100, vec![4]).expect("a fold of four of a hundred"));
        assert_eq!(left[0].fold, rs::GroupFold::Host(4));
        // A model with window lists, or a pass binding no state, carries none.
        assert_eq!(
            lanes_frozen(&stores, &ids, &groups_of, 1, true).unwrap(),
            [None]
        );
        assert_eq!(
            lanes_frozen(&stores, &[], &groups_of, 1, false).unwrap(),
            [None]
        );

        // The device plan: the device picks where the rows start, inside
        // what the lane reserved.
        let device = |reserved: u32| {
            let cell = Arc::new(Mutex::new(Cursor {
                folded: 13,
                reserved,
                tail: None,
                ..Cursor::default()
            }));
            let lanes = PassLanes {
                places: vec![geometry::Place {
                    folded: 13,
                    folded_exact: true,
                    tail: None,
                    shared: false,
                    spans: Vec::new(),
                }],
                cursors: vec![cell],
                kv: Vec::new(),
            };
            device_rs_plan(&stores, &ids, &groups_of, &lanes, &geometry::Fold::Device)
        };
        let planned = groups(device(13 + 5 + 5).unwrap());
        assert_eq!(planned[0].writes, Some(13..23));
        assert_eq!(planned[0].fold, rs::GroupFold::Device);
        // The decode reserved its whole extent up front, as the MTP example:
        // what it reserved is buffered, and is no count of what it carries.
        let ahead = groups(device(13 + 64).expect("planned whatever it reserved"));
        assert_eq!(ahead[0].writes, Some(13..13 + 64));
    }
}
