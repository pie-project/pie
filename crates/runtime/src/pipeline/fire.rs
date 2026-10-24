#![allow(unsafe_code)]
fn container_has_attention_stages(container: &eta_ir::container::TraceContainer) -> bool {
    use eta_ir::registry::Stage;
    container
        .stages
        .iter()
        .any(|s| matches!(s.stage, Stage::OnAttnProj | Stage::OnAttn))
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
    format!(
        "pipeline: resource demand kept drifting under contention \
         (still stale after {STALE_DEMAND_ATTEMPTS} re-acquisitions)"
    )
}

const STALE_DEMAND_ATTEMPTS: usize = 3;

struct KvTxnGuard {
    model: usize,
    engine: usize,
    txn: Option<kv::KvTxn>,
}

impl KvTxnGuard {
    fn new(model: usize, engine: usize, txn: Option<kv::KvTxn>) -> Self {
        Self { model, engine, txn }
    }

    fn into_inner(mut self) -> Option<kv::KvTxn> {
        self.txn.take()
    }
}

impl Drop for KvTxnGuard {
    fn drop(&mut self) {
        let Some(txn) = self.txn.take() else {
            return;
        };
        let stores = crate::store::registry::get(self.model, self.engine);
        crate::store::registry::with_kv_lock(&stores.kv, "host-other", |store| {
            kv::settle(store, txn);
        });
        if let Some(planner) = crate::planner::planner_for(self.model, self.engine) {
            planner.pages_freed();
        }
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

/// `windowed`: the fire reads ring pages, which come back only once every
/// fire that read them settles, so a fire that would park settles its own
/// tail first rather than hold back the pages parked fires wait on.
async fn acquire_grant<C: FireContext>(
    ctx: &mut C,
    pipeline_id: uuid::Uuid,
    demand: crate::planner::Demand,
    windowed: bool,
) -> Result<Option<crate::planner::AllocationGrant>, String> {
    if demand.is_zero() {
        return Ok(Some(crate::planner::AllocationGrant::empty()));
    }
    let Some(planner) = crate::planner::planner() else {
        return Err("pipeline: KV residency planner is not installed".to_string());
    };
    let pid = ctx.process_id();
    if windowed {
        if let Some(grant) = planner.grant_now(pid, demand) {
            return Ok(Some(grant));
        }
        // The tail may hold this frame's earlier slots, which fire only once
        // the scheduler stops waiting on this one: park the lane first.
        crate::scheduler::worker::notify_lane_close(pipeline_id, Some(pid));
        ctx.settle_pipeline_tail()
            .await
            .map_err(|error| format!("pipeline: settle for windowed kv: {error:#}"))?;
    }
    // A yield may have dropped snapshots the demand was quoted against, so
    // the caller quotes it again rather than asking for the same.
    match planner
        .acquire(pid, pipeline_id, demand)
        .await
        .map_err(|error| format!("pipeline: KV capacity: {error}"))?
    {
        crate::planner::Acquired::Granted(grant) => Ok(Some(grant)),
        crate::planner::Acquired::Yield => {
            settle_and_wait_resident(ctx).await?;
            Ok(None)
        }
        crate::planner::Acquired::Requote => Ok(None),
    }
}

async fn settle_and_wait_resident<C: FireContext>(ctx: &mut C) -> Result<(), String> {
    let Some(planner) = crate::planner::planner() else {
        return Err("pipeline: KV working set is suspend-fenced".to_string());
    };
    ctx.settle_pipeline_tail()
        .await
        .map_err(|error| format!("pipeline: settle for eviction: {error:#}"))?;
    planner
        .wait_resident(ctx.process_id())
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
        if let Err(owner) = rs.claim_pipeline_scope(scope) {
            return Ok(Err(format!(
                "pipeline: lane {row}'s working set is already scoped to pipeline \
                 {owner:#x}"
            )));
        }
        ids.push(rs.id);
    }
    Ok(Ok(ids))
}

/// The ring plan of a host fire on a model whose state is a window: each
/// lane writes the pages its tokens land on from `held`, which every
/// earlier token of its sequence precedes, and reads through its own pages.
fn host_ring_plan(
    req: &crate::engine::FireRequest,
    page_size: u32,
    fold: &geometry::Fold,
    ids: &[crate::store::rs::RsWorkingSetId],
) -> Result<rs::RsPlan, String> {
    if ids.is_empty() {
        return Ok(rs::RsPlan::Groups(Vec::new()));
    }
    let page_size = page_size.max(1);
    Ok(rs::RsPlan::Ring(
        req.lanes
            .iter()
            .enumerate()
            .map(|(row, lane)| {
                let rows = u32::try_from(lane.tokens.len()).unwrap_or(u32::MAX);
                let end = lane.kv.held.saturating_add(rows);
                let pages = if rows == 0 {
                    Vec::new()
                } else {
                    let first = (lane.kv.held / page_size) as usize;
                    let last = ((end - 1) / page_size) as usize;
                    let held = lane.kv.pages.len();
                    lane.kv.pages[first.min(held)..(last + 1).min(held)].to_vec()
                };
                rs::RingRow {
                    pages,
                    translate: lane.kv.pages.clone(),
                    advance: crate::store::rs::RingAdvance {
                        positioned: true,
                        settled: lane.kv.held,
                        reach: lane.kv.held,
                        fold: fold.of(row),
                        commit_end: end,
                        head: (rows > 0).then_some(end),
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
        plans.push(rs::GroupPlan {
            lanes: group.clone(),
            writes: Some(frozen..reserved),
            fold: group_fold(fold, first, frozen, reserved),
            snapshots: Vec::new(),
        });
    }
    Ok(rs::RsPlan::Groups(plans))
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

/// Before a recurrent fire: its working sets' frozen lengths are known, so
/// a fire that folded one by a count the device computed has settled.
async fn settle_device_folds<C: FireContext>(
    ctx: &mut C,
    fwd: &Resource<ForwardPass>,
    fires: &PendingFires,
    lanes: &mut PassLanes,
) -> Anyhow<Result<(), String>> {
    if lanes.places.iter().all(|place| place.folded_exact) {
        return Ok(Ok(()));
    }
    drain_pipeline_fires(ctx, fires).await?;
    *lanes = pass_lanes(ctx, fwd)?;
    Ok(
        match lanes.places.iter().position(|place| !place.folded_exact) {
            Some(lane) => Err(format!(
                "pipeline: lane {lane}'s working set was folded by a count the device computed \
                 and the engine did not report; its recurrent state cannot go on from a frozen \
                 length the host does not know"
            )),
            None => Ok(()),
        },
    )
}

/// What changes the KV a pass writes beyond its tokens.
fn hook_program_of<C: FireContext>(
    ctx: &mut C,
    fwd: &Resource<ForwardPass>,
) -> Anyhow<(bool, bool)> {
    let p = ctx.resources().get(fwd)?;
    let container = &p.instance.program.bound.container;
    Ok((
        container_has_attention_stages(container),
        container_has_lora_sink(container),
    ))
}

/// `PIE_FIRE_ROWS` overrides it, so a test compares outputs cut at the same
/// rows.
fn prefill_share(page: u32, engine: usize) -> u32 {
    if let Some(rows) = fire_rows() {
        return rows;
    }
    crate::engine::get_spec(engine).map_or(u32::MAX, |spec| {
        crate::inferlet::host::model::prefill_share(
            crate::inferlet::host::model::prefill_budget(spec.limits),
            crate::inferlet::process::live_count(),
            page as usize,
        )
    })
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
/// the program reads out.
fn host_cuts(
    req: &crate::engine::FireRequest,
    cuttable: bool,
    share: u32,
    page: u32,
    forced: &[u32],
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
    let tail = tail_row(lane, rows).min(first_media_row(req).unwrap_or(rows));
    let mut ends: Vec<u32> = forced
        .iter()
        .copied()
        .filter(|&end| 0 < end && end < rows)
        .collect();
    if share > 0 {
        let mut from = 0;
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
    let mut from = 0;
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
        && !req.lanes[0].captures_scores
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
    lane.attn_classes.is_none()
        && lane.adapter.is_none()
        && !lane.drafts
        && !lane.block_draft
        && lane.self_cond.is_none()
        && lane.ports.is_empty()
        && !lane.kv_less
}

/// The first row the last piece must hold: what the readout names and what a
/// score capture observes.
fn tail_row(lane: &::engine::Lane, rows: u32) -> u32 {
    let read = match &lane.readout {
        ::engine::Readout::Last => rows.saturating_sub(1),
        ::engine::Readout::Rows(read) => read.iter().copied().min().unwrap_or(rows),
        ::engine::Readout::None => rows,
    };
    if lane.captures_scores {
        read.min(rows.saturating_sub(eta_ir::registry::ATTN_SCORE_OBSERVE))
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
                lane.captures_scores = false;
                lane.channels.clear();
            }
            Some(lane)
        })
        .collect();
    if !last {
        req.boundary_program = false;
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
) -> Vec<Vec<Option<(u32, u32)>>> {
    let whole = || {
        vec![
            req.lanes
                .iter()
                .map(|lane| Some((0, u32::try_from(lane.tokens.len()).unwrap_or(u32::MAX))))
                .collect(),
        ]
    };
    if req.lanes.len() <= 1 {
        return host_cuts(req, cuttable, share, page, forced)
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
        .map(|lane| {
            let one = crate::engine::FireRequest::one(lane.clone());
            host_cuts(&one, true, share, page, &[])
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
    req.cohort.is_none()
        && req.media.is_empty()
        && req.voxels.is_empty()
        && req.lanes.iter().all(|lane| {
            lane.group.is_none()
                && lane.peer.is_none()
                && separable_rows(&crate::engine::FireRequest::one(lane.clone()))
        })
}

/// Whether a fire's KV is its tokens' alone, so its pages seal under their keys.
fn keyed_rows(req: &crate::engine::FireRequest) -> bool {
    let [lane] = req.lanes.as_slice() else {
        return false;
    };
    let held = lane.kv.held;
    plain_lane(lane)
        && lane.mask.is_none()
        && !lane.tokens.is_empty()
        && !lane.tokens.contains(&u32::MAX)
        && (lane.positions.is_empty()
            || lane
                .positions
                .iter()
                .enumerate()
                .all(|(i, &at)| at == held + i as u32))
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
    let looked = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
        let Some(keys) = kv.page_keys(ws, held, slots, page_size)? else {
            return Ok(None);
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
    let found = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
        let Some(keys) = kv.page_keys(ws.id, held, slots, page_size)? else {
            return Ok::<_, crate::store::kv::KvStoreError>(None);
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
    let plans = guard
        .txn
        .as_ref()
        .map_or_else(Vec::new, |txn| txn.copy_plans());
    for (tier, device, slots) in plans {
        let copy = crate::scheduler::TierMove {
            kind: crate::scheduler::TierKind::Kv,
            tier,
            out: false,
            device,
            slots,
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
            if let Some(planner) = crate::planner::planner() {
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
fn resume(
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
             `[engine] max_model_len` or send fewer tokens",
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
}

/// A demand that drifted from its grant is quoted again.
async fn reserve<C: FireContext>(
    ctx: &mut C,
    stores: &crate::store::registry::Stores,
    pipeline_id: uuid::Uuid,
    sets: &[(&KvWorkingSet, Written)],
    rs_ids: &[crate::store::rs::RsWorkingSetId],
    rs_plan: impl Fn() -> Result<rs::RsPlan, String>,
    windowed: bool,
) -> Result<Reserved, String> {
    let mut attempts = 0;
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
                });
            }
            Err(ReservedError::Stale) if attempts < STALE_DEMAND_ATTEMPTS => attempts += 1,
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
    kvtxn: Option<kv::KvTxn>,
    log: Option<LogFire>,
    lease: KvFireLease,
    fold: Option<FoldAt>,
    reports: Vec<ReportTag>,
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
        rs: Option<crate::store::rs::RsWorkingSetId>,
    },
}

fn plain_decode(
    req: &crate::engine::FireRequest,
    hooked: bool,
    fire_wide_mask: bool,
    max_layers: Option<u32>,
) -> bool {
    !hooked
        && !fire_wide_mask
        && max_layers.is_none()
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
fn take_checkpoint(
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
struct TakeCheckpoint {
    kv: crate::store::kv::page_table::WorkingSetId,
    rs: crate::store::rs::RsWorkingSetId,
    snapshot: bool,
    pages: u64,
    key: crate::store::kv::hash::Hash256,
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

fn stamp_self_cond(
    req: &mut crate::engine::FireRequest,
    payload: Option<crate::pipeline::instance::SelfCondPayload>,
) -> Result<(), String> {
    let wanted: usize = req.lanes.iter().map(|lane| lane.tokens.len()).sum();
    let mut cursor = 0usize;
    for lane in &mut req.lanes {
        if let Some(payload) = &payload
            && let Some((rows_channel, weights_channel)) = payload.channels
        {
            lane.self_cond = Some(::engine::fire::SelfCondInput::from_channels(
                payload.taps,
                rows_channel,
                weights_channel,
            ));
            continue;
        }
        if let Some(payload) = &payload {
            let cells = lane.tokens.len() * payload.taps as usize;
            let end = cursor + cells;
            if end > payload.rows.len() {
                return Err(format!(
                    "the staged payload holds {} taps and this fire's {wanted} rows want {}",
                    payload.rows.len(),
                    wanted * payload.taps as usize
                ));
            }
            lane.self_cond = Some(::engine::fire::SelfCondInput::new(
                payload.taps,
                payload.rows[cursor..end].to_vec(),
                &payload.weights[cursor..end],
            ));
            cursor = end;
        }
    }
    if let Some(payload) = &payload
        && cursor != payload.rows.len()
    {
        return Err(format!(
            "the staged payload holds {} taps and this fire's {wanted} rows want {cursor}",
            payload.rows.len()
        ));
    }
    Ok(())
}

/// `device_rows`, when given, are the rows each lane carries on the device:
/// a pooled pass stages no host tokens, and its words still state its rows,
/// so a single-token lane takes the decode branch.
fn stamp_lane_words(
    req: &mut crate::engine::FireRequest,
    fire_wide_mask: bool,
    carries_media: bool,
    denoise: bool,
    device_rows: Option<&[u32]>,
) {
    let model = crate::model::model();
    for (at, lane) in req.lanes.iter_mut().enumerate() {
        let rows = device_rows
            .and_then(|rows| rows.get(at).copied())
            .unwrap_or_else(|| u32::try_from(lane.tokens.len()).unwrap_or(u32::MAX));
        lane.word = model.word(
            rows,
            lane.mask.is_some() || fire_wide_mask,
            lane.adapter.is_some(),
            lane.drafts,
            lane.captures_scores,
            carries_media,
            lane.block_draft,
            denoise,
            crate::pipeline::instance::stream_of_lane(lane.stream),
            lane.reading,
        );
    }
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

pub(crate) fn stamp_lane_slots(
    req: &mut crate::engine::FireRequest,
    stores: &crate::store::registry::Stores,
    groups: &LaneGroups,
) -> Result<(), crate::store::seat::SeatError> {
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

const SEAT_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

pub(crate) async fn seat_lane_slots(
    req: &mut crate::engine::FireRequest,
    stores: &crate::store::registry::Stores,
    groups: &LaneGroups,
) -> Result<(), String> {
    use crate::store::seat::SeatError;
    let deadline = crate::rt::time::Instant::now() + SEAT_WAIT;
    loop {
        let mut freed = Box::pin(stores.seats_freed.notified());
        freed.as_mut().enable();
        let refusal = match stamp_lane_slots(req, stores, groups) {
            Ok(()) => return Ok(()),
            Err(error) => error,
        };
        let fits = match refusal {
            SeatError::Exhausted { need, capacity, .. } => capacity != 0 && need <= capacity,
        };
        if !fits {
            return Err(format!(
                "pipeline: seating this fire's lanes: {refusal}; no release can seat it"
            ));
        }
        let now = crate::rt::time::Instant::now();
        if now >= deadline {
            return Err(format!(
                "pipeline: seating this fire's lanes: {refusal}; waited {:?} for a peer to \
                 release seats and none did",
                SEAT_WAIT
            ));
        }
        let _ = crate::rt::time::timeout(deadline - now, freed).await;
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

/// The ring plan of a device-geometry fire: every row writes the pages
/// the fire's write indexes name and reads through the working set's mapped
/// pages. The device decides where its tokens land, so the ring keeps every
/// page it writes and commits nothing; what it retires is behind the first
/// page it writes, where every row lands past, so a decode loop that never
/// states its positions still holds a window and not its whole sequence.
fn device_ring_plan(
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
    let written: Vec<u32> = write_indexes
        .iter()
        .map(|&index| u32::try_from(index).unwrap_or(u32::MAX))
        .collect();
    let last = written.iter().copied().max();
    let reach = written
        .iter()
        .copied()
        .min()
        .map_or(0, |first| first.saturating_mul(stores.kv_page_size));
    let extent = u32::try_from(mapped)
        .unwrap_or(u32::MAX)
        .max(last.map_or(0, |last| last.saturating_add(1)));
    let translate: Vec<u32> = (0..extent).collect();
    Ok(rs::RsPlan::Ring(
        (0..ids.len())
            .map(|_| rs::RingRow {
                pages: written.clone(),
                translate: translate.clone(),
                advance: crate::store::rs::RingAdvance {
                    positioned: false,
                    settled: 0,
                    reach,
                    fold: None,
                    commit_end: 0,
                    head: None,
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

struct TicketReservation {
    cells: BoundCells,
    heads: Vec<u64>,
    tails: Vec<u64>,
    committed: bool,
}

impl TicketReservation {
    fn new(cells: &BoundCells, accesses: &[(bool, bool)]) -> Self {
        let (heads, tails) = cells
            .iter()
            .zip(accesses)
            .map(|(cell, &(consume, publish))| {
                cell.lock().unwrap().reserve_device_ticket(consume, publish)
            })
            .unzip();
        Self {
            cells: cells.clone(),
            heads,
            tails,
            committed: false,
        }
    }

    fn apply_to(&self, request: &mut crate::engine::FireRequest) {
        if self.cells.is_empty() {
            return;
        }
        let mut adopted = true;
        let tickets: Vec<engine::Ticket> = self
            .cells
            .iter()
            .zip(&self.heads)
            .zip(&self.tails)
            .map(|((cell, &head), &tail)| {
                let cell = cell.lock().unwrap();
                adopted &= cell
                    .endpoint()
                    .is_some_and(|endpoint| endpoint.registered().adopted());
                engine::Ticket {
                    channel: cell.global_id,
                    expected_head: head,
                    expected_tail: tail,
                }
            })
            .collect();
        if let Some(lane) = request.lanes.first_mut()
            && adopted
        {
            lane.channels = tickets;
        }
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
        for ((cell, &head), &tail) in self.cells.iter().zip(&self.heads).zip(&self.tails).rev() {
            if !cell.lock().unwrap().rollback_device_ticket(head, tail) {
                tracing::error!(
                    "channel ticket rollback lost LIFO ownership; preserving newer reservations"
                );
            }
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
        let now = *cursor.lock().unwrap();
        places.push(geometry::Place {
            folded: now.folded,
            folded_exact: now.folded_exact,
            tail: now.tail,
            shared: lanes.iter().filter(|&&other| other == rep).count() > 1,
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
    let place = lanes.places[first];
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
        None => (vec![report_on(req, first, tags)], rs),
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
    let (group, peer) = {
        let lane = &ctx.resources().get(fwd)?.lane;
        lane.stamp(req);
        (lane.group, lane.peer)
    };
    req.cohort = crate::pipeline::instance::cohort_of(ctx.resources(), group, peer);
    Ok(())
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
            p_reads_attn_score,
            p_reads_mtp_logits,
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
                        let attn_mask = match geometry::lower_attn_mask_evaluated(
                            &bound.container,
                            &geometry.qo_indptr,
                            &evaluated,
                        ) {
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
            let reads_attn_score = p.instance.program.reads_attn_score;
            let reads_mtp_logits = p.instance.program.reads_mtp_logits;
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
                reads_attn_score,
                reads_mtp_logits,
            )
        };
        let mut pass_lanes = pass_lanes(ctx, &fwd)?;
        if !rs_reps.is_empty()
            && let Err(error) = settle_device_folds(ctx, &fwd, &pipe_fires, &mut pass_lanes).await?
        {
            return Ok(Err(error));
        }
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
        if p_reads_attn_score {
            for lane in &mut req.lanes {
                lane.captures_scores = true;
            }
        }
        if p_reads_mtp_logits {
            for lane in &mut req.lanes {
                lane.drafts = true;
            }
        }
        if ctx.resources().get(&fwd)?.block_draft {
            for lane in &mut req.lanes {
                lane.block_draft = true;
            }
        }
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
        req.max_layers = {
            let p = ctx.resources().get(&fwd)?;
            p.max_layers
        };
        let media_spans = ctx.resources().get(&fwd)?.bindings.media.clone();
        let carries_media = !media_spans.is_empty();
        let matched = {
            let lane_tokens: Vec<&[u32]> = req
                .lanes
                .iter()
                .map(|lane| lane.tokens.as_slice())
                .collect();
            let scanned = if carries_media {
                crate::pipeline::media::scan(&lane_tokens, &media_spans)
            } else {
                crate::pipeline::media::refuse_orphan_runs(&lane_tokens, crate::model::media_pad())
                    .map(|()| Vec::new())
            };
            match scanned {
                Ok(matched) => matched,
                Err(refusal) => return Ok(Err(format!("pipeline: {refusal}"))),
            }
        };
        let fire_wide_mask = matches!(attn_mask, geometry::FireAttnMask::Device);
        if let Err(error) = attn_mask.apply_to(&mut req) {
            return Ok(Err(format!("pipeline: fire attention mask: {error}")));
        }
        // A denoise pass's rows read one another: it is never cut, keyed or
        // logged by its tokens.
        let denoise = {
            let pass = ctx.resources().get_mut(&fwd)?;
            let denoise =
                pass.bindings.canvas == Some(crate::pipeline::instance::CanvasMode::Denoise);
            if denoise {
                let payload = if pass
                    .bindings
                    .self_cond
                    .as_ref()
                    .is_some_and(|p| p.channels.is_some())
                {
                    pass.bindings.self_cond.clone()
                } else {
                    pass.bindings.self_cond.take()
                };
                if let Err(error) = stamp_self_cond(&mut req, payload) {
                    return Ok(Err(format!("pipeline: self-conditioning: {error}")));
                }
            }
            denoise
        };
        stamp_lane_words(&mut req, fire_wide_mask, carries_media, denoise, None);
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
        let kv_clock = crate::scheduler::probe::ProbeClock::start();
        let groups = lane_groups(&pass_lanes.kv.iter().map(|ws| ws.id).collect::<Vec<_>>());
        let group_sets: Vec<KvWorkingSet> = groups
            .iter()
            .map(|(_, lanes)| pass_lanes.kv[lanes[0]].clone())
            .collect();
        if let Err(refusal) = seat_lane_slots(&mut req, &stores, &groups).await {
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
            if let Err(owner) = set.claim_pipeline_scope(&pipeline_scope) {
                return Ok(Err(format!(
                    "pipeline: KV working set is already scoped to pipeline {owner:032x}"
                )));
            }
        }
        let rs_ws_ids =
            match bound_rs_working_sets(ctx, &ws, &rs_reps, &req.qo_indptr(), &pipeline_scope)? {
                Ok(ids) => ids,
                Err(error) => return Ok(Err(error)),
            };
        let ring = crate::model::model().rs_caps().window_tokens > 0;
        // A long host pass runs as several fires; all but the last run the
        // model's forward alone. A recurrent state folding all its rows, or
        // none, folds each fire's rows in turn.
        let folds_all = rs_ws_ids.is_empty()
            || [0, u32::MAX]
                .into_iter()
                .any(|count| (0..req.lanes.len()).all(|lane| fold.of(lane) == Some(count)));
        let (attention_stages, adapted) = hook_program_of(ctx, &fwd)?;
        let hooked = attention_stages || adapted;
        let keyable = decode_envelope.is_none()
            && !denoise
            && groups.len() == 1
            && !hooked
            && !carries_media
            && !fire_wide_mask
            && !p_reads_mtp_logits
            && req.max_layers.is_none();
        let cuttable = decode_envelope.is_none()
            && !denoise
            && !attention_stages
            && (!adapted || crate::model::model().eta_caps().adapter_attachments)
            && !fire_wide_mask
            && !p_reads_mtp_logits
            && !ring
            && folds_all;
        let mut full = req;
        // Its fires seal their share of the slot hashes; sealed pages holding
        // its rows already are mapped in.
        let mut pass_slots = (keyable && keyed_rows(&full))
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
                    &lane.tokens,
                ))
            })
            .flatten();
        let state = match rs_ws_ids.as_slice() {
            [rs] if !ring => Some(*rs),
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
        // Rows no worker here holds may be prefilled on a peer, whose pages
        // land in the host tier for the promote below to find.
        if matches!(attach, Attach::Kv)
            && let Some(slots) = &pass_slots
        {
            let lane = &full.lanes[0];
            // Pages another pass is computing now are waited for, not
            // computed again.
            crate::pipeline::coalesce::before_pass(&stores, ws.id, u64::from(lane.kv.held), slots)
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
        if supplied > 0 {
            // What was mapped in is skipped: the pass runs from there.
            let rows = u32::try_from(full.lanes[0].tokens.len()).unwrap_or(u32::MAX);
            full = rows_of_lanes(&full, &[Some((supplied, rows))], true, page_size);
            stamp_lane_words(&mut full, false, false, false, None);
            pass_slots = pass_slots.map(|slots| slots[supplied as usize..].to_vec());
            let total = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
                kv.supplied_tokens(ws.id)
            })
            .unwrap_or(0);
            crate::policy::report(crate::policy::Event::Cached(pid, total));
        }
        // Where the pass keeps its state: the checkpoint rows it reaches,
        // floored to whole pages it keys, each the end of a fire of its own.
        let held = u64::from(full.lanes[0].kv.held);
        let reach = held + full.lanes[0].tokens.len() as u64;
        let checkpoints: Vec<u64> = match state {
            Some(_) if !keys.is_empty() => {
                let page = u64::from(page_size.max(1));
                let mut rows: Vec<u64> = ctx
                    .resources()
                    .get(&fwd)?
                    .bindings
                    .checkpoints
                    .iter()
                    .map(|&row| u64::from(row) / page * page)
                    .filter(|&row| held < row && row <= reach && row / page <= keys.len() as u64)
                    .collect();
                rows.dedup();
                rows
            }
            _ => Vec::new(),
        };
        // An engine that lands a lane's state at rows inside a fire is cut only
        // past as many as it lands at once.
        let snapshots = crate::model::model().eta_caps().rs_snapshots as usize;
        let forced: Vec<u32> = checkpoints
            .iter()
            .skip(snapshots)
            .step_by(snapshots + 1)
            .map(|&row| (row - held) as u32)
            .collect();
        // Several lanes run in pieces only where none carries recurrent
        // state, which is bound lane by lane.
        let cuts = pass_pieces(
            &full,
            cuttable,
            rs_ws_ids.is_empty(),
            prefill_share(page_size, engine),
            page_size,
            &forced,
        );
        let rows_total = full
            .lanes
            .iter()
            .map(|lane| lane.tokens.len())
            .sum::<usize>();
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
            if attention_stages {
                why.push("its program has stages at attention");
            }
            if adapted && !crate::model::model().eta_caps().adapter_attachments {
                why.push("this engine cannot carry its adapter on a fire of its own");
            }
            if p_reads_mtp_logits {
                why.push("its draft arm reads every row");
            }
            if ring {
                why.push("its model keeps a window ring");
            }
            if !folds_all {
                why.push("it states where its recurrent state folds");
            }
            if full.lanes.len() > 1 && !rs_ws_ids.is_empty() {
                why.push("its lanes carry recurrent state");
            }
            if why.is_empty() {
                why.push("its rows read one another (attention both ways, a block draft, an attention group, ports or voxels)");
            }
            return Ok(Err(format!(
                "pipeline: a pass of {rows_total} rows would run {last_rows} in one fire, \
                 past the {capacity} a fire holds, and cannot be cut further: {}",
                why.join("; ")
            )));
        }
        // The pieces in frames of their own, in the order they are cut, so
        // the pieces of concurrent prompts seal together.
        let piece_frames = {
            let pipeline = ctx.resources().get(&this)?;
            piece_frames(
                frame,
                cuts.len(),
                crate::scheduler::configured_frame_size(),
                || pipeline.next_frame_seq(),
            )
        };
        let mut settles = Vec::new();
        for (at, piece) in cuts.iter().enumerate() {
            let last = at + 1 == cuts.len();
            let mut req = if cuts.len() == 1 {
                full.clone()
            } else {
                let mut rows = rows_of_lanes(&full, piece, last, page_size);
                let media = !rows.media.is_empty();
                stamp_lane_words(&mut rows, false, media, false, None);
                rows
            };
            let spans: Vec<(std::ops::Range<u64>, std::ops::Range<u64>)> = groups
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
                }
            };
            let mut checkpoints_here: Vec<TakeCheckpoint> = state
                .filter(|_| checkpoints.contains(&slots.end))
                .map(|rs| at_row(rs, slots.end, false))
                .into_iter()
                .collect();
            // The furthest the engine lands of those inside the fire.
            let mut inside: Vec<u32> = checkpoints
                .iter()
                .filter(|&&row| slots.start < row && row < slots.end)
                .map(|&row| (row - slots.start) as u32)
                .collect();
            inside.drain(..inside.len().saturating_sub(snapshots));
            let sets: Vec<(&KvWorkingSet, Written)> = group_sets
                .iter()
                .zip(&spans)
                .map(|(set, (_, written))| (set, Written::Span(written.clone())))
                .collect();
            let rs_plan = || {
                if ring {
                    host_ring_plan(&req, page_size, &fold, &rs_ws_ids)
                } else {
                    host_rs_plan(&stores, &rs_ws_ids, &groups, &req, &fold, &inside)
                }
            };
            let reserved = reserve(
                ctx,
                &stores,
                quorum_pipeline_id,
                &sets,
                &rs_ws_ids,
                rs_plan,
                ring,
            )
            .await;
            let Reserved {
                leases,
                copies: (copy_src, copy_dst),
                kvtxns,
                rs: rs_prepared,
            } = match reserved {
                Ok(reserved) => reserved,
                Err(error) => return Ok(Err(error)),
            };
            crate::probe_fire_record!(submit_probe.kv_prepare_us, kv_clock.elapsed());
            // Every fire logs what it writes where its tokens are known; what
            // it folds seals once it settles.
            let reports = pass_slots.is_none()
                && decode_envelope.is_some()
                && !denoise
                && plain_decode(&req, hooked, fire_wide_mask, req.max_layers);
            let mut parts = Vec::with_capacity(group_sets.len());
            for (group, ((set, (slots, written)), (lease, kvtxn))) in group_sets
                .iter()
                .zip(&spans)
                .zip(leases.into_iter().zip(kvtxns))
                .enumerate()
            {
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
                let rows = match (&pass_slots, pages) {
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
                let rs = rs_ws_ids.get(lanes[0]).copied().filter(|_| !ring);
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
                    kvtxn: kvtxn.into_inner(),
                    log: Some(LogFire { seq, rows }),
                    lease,
                    fold: by.map(|by| FoldAt {
                        by,
                        decoded: reports,
                    }),
                    reports: tags,
                });
            }
            settles.extend(settling(&parts));
            if let Err(refusal) = map_lane_pages(&mut req, &stores, &groups) {
                return Ok(Err(refusal));
            }
            crate::offload::stream::tap(pid, &stores, &mut req);
            rs_prepared.apply_to(&mut req);
            checkpoints_here.extend(
                rs_prepared
                    .snapshots
                    .iter()
                    .map(|snap| at_row(snap.snapshot, slots.start + u64::from(snap.row), true)),
            );
            let rs_copy = rs_prepared.state_copy();
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
                    rs_copy,
                    piece_frames[at],
                    attention_stages,
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
                }));
            // The next piece reads the KV this one writes: they run in
            // order on the device, so it is submitted now, as a decode's
            // next pass is, rather than once this one retires.
        }
        advance_cursors(&pass_lanes, Some(&landings), &fold, &settles);
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
    if crate::scheduler::configured_frame_size() == 1 {
        return Ok(());
    }
    let pipeline = ctx.resources().get(&this)?;
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
    let finalized = finalize_op_await(op).await?;
    complete_finalize(ctx, finalized);
    Ok(())
}

pub(crate) async fn finalize_op_await(op: PendingOp) -> Anyhow<FinalizeOutcome> {
    match op {
        PendingOp::Fire(fire) => finalize_fire_await(fire).await,
        #[cfg(test)]
        PendingOp::TestStub => Ok(FinalizeOutcome {
            action: FinalizeAction::None,
            ws_guard: Vec::new(),
        }),
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
    let FinalizeOutcome { action, ws_guard } = finalize_op_await(op).await?;
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
fn settle_fold(
    stores: &crate::store::registry::Stores,
    ws: crate::store::kv::page_table::WorkingSetId,
    fold: FoldAt,
    reported: &std::collections::HashMap<u64, ::engine::KvWrites>,
    success: bool,
) {
    let FoldAt { by, decoded } = fold;
    let folded = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
        let frozen = kv.folded(ws)?;
        let (to, cursor) = match by {
            FoldBy::To(to) => (Some(to), None),
            FoldBy::Settled {
                count,
                tags,
                reach,
                cursor,
                rs,
            } => {
                let count = count.or_else(|| {
                    tags.iter()
                        .map(|tag| reported.get(tag).and_then(|written| written.fold))
                        .collect::<Option<Vec<u32>>>()
                        .and_then(|counts| counts.into_iter().max())
                });
                let to = count.map(|count| frozen.saturating_add(u64::from(count)).min(reach));
                // The recurrent state froze the same slots on the device.
                let rs = rs.zip(to).map(|(rs, to)| (rs, to.saturating_sub(frozen)));
                (to, Some((cursor, to.is_some(), rs)))
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
            if let Some((_, _, Some((rs, slots)))) = &cursor
                && success
            {
                let slots = u32::try_from(*slots).unwrap_or(u32::MAX);
                stores.rs.lock().unwrap().fold(*rs, slots);
            }
            let cursor = cursor.map(|(cursor, placed, _)| (cursor, placed));
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
        Err(error) => tracing::debug!(%error, "pipeline: sealing what a fire froze"),
    }
}

async fn finalize_fire_await(fire: PendingFire) -> Anyhow<FinalizeOutcome> {
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
    } = fire;
    let prior_failure = failure.lock().unwrap().clone();
    let result = completion.await;
    let success = result.is_ok() && prior_failure.is_none();
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
        let mut reported = std::collections::HashMap::new();
        for part in kv {
            if let Some(kvtxn) = part.kvtxn {
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
            leases.push(part.lease);
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
            settle_fold(&stores, ws, fold, &reported, success);
        }
        for checkpoint in &checkpoints {
            take_checkpoint(&stores, checkpoint, success);
        }
    }

    if let Some(planner) = crate::planner::planner() {
        planner.pages_freed();
    }

    let failure_reason = prior_failure.or_else(|| {
        result
            .err()
            .map(|error| format!("pipeline: forward failed: {error:#}"))
    });
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
    Ok(FinalizeOutcome {
        action,
        ws_guard: leases,
    })
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
    if !ctx.resources().get(&fwd)?.bindings.media.is_empty() {
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
        if let Err(owner) = set.claim_pipeline_scope(&pipeline_scope) {
            return Ok(Err(format!(
                "pipeline: KV working set is already scoped to pipeline {owner:032x}"
            )));
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
    let mut write_indexes: Vec<Vec<u64>> = Vec::with_capacity(groups.len());
    for (_, lanes) in &groups {
        let lane = lanes[0];
        let place = pass_lanes.places[lane];
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
    let ring = crate::model::model().rs_caps().window_tokens > 0;
    if ring && groups.len() > 1 {
        return Ok(Err(
            "pipeline: on a model with a window ring, a pass whose rows land where the device \
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
        if ring {
            device_ring_plan(&stores, ws.id, &write_indexes[0], &rs_ws_ids)
        } else {
            device_rs_plan(&stores, &rs_ws_ids, &groups, &pass_lanes, &fold)
        }
    };
    let reserved = reserve(
        ctx,
        &stores,
        quorum_pipeline_id,
        &sets,
        &rs_ws_ids,
        rs_plan,
        ring,
    )
    .await;
    let Reserved {
        leases,
        copies: (copy_src, copy_dst),
        kvtxns,
        rs: mut rs_prepared,
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
    let denoise = {
        let pass = ctx.resources().get_mut(&fwd)?;
        let denoise = pass.bindings.canvas == Some(crate::pipeline::instance::CanvasMode::Denoise);
        if denoise {
            let payload = if pass
                .bindings
                .self_cond
                .as_ref()
                .is_some_and(|p| p.channels.is_some())
            {
                pass.bindings.self_cond.clone()
            } else {
                pass.bindings.self_cond.take()
            };
            if let Err(error) = stamp_self_cond(&mut req, payload) {
                let reason = format!("pipeline: self-conditioning: {error}");
                record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
                return Ok(Err(reason));
            }
        }
        denoise
    };
    {
        let program = &ctx.resources().get(&fwd)?.instance.program;
        let (drafts, captures) = (program.reads_mtp_logits, program.reads_attn_score);
        for lane in &mut req.lanes {
            lane.drafts = drafts;
            lane.captures_scores = captures;
        }
    }
    // A drafted block whose layout the device carries (a block drafter's
    // loop on the device): its lanes are the block's, as on the host path.
    if ctx.resources().get(&fwd)?.block_draft {
        for lane in &mut req.lanes {
            lane.block_draft = true;
        }
    }
    // A split narrow enough to stage no host rows still knows each lane's
    // rows; its words carry them.
    let device_rows: Option<Vec<u32>> =
        (!wide).then(|| split.windows(2).map(|lane| lane[1] - lane[0]).collect());
    stamp_lane_words(
        &mut req,
        fire_wide_mask,
        false,
        denoise,
        device_rows.as_deref(),
    );
    if let Err(refusal) = seat_lane_slots(&mut req, &stores, &groups).await {
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
        reads_mtp_logits = req.lanes.first().is_some_and(|lane| lane.drafts),
        "fire geometry resolved on device path"
    );
    let ticket_reservation = TicketReservation::new(&cells, &accesses);
    ticket_reservation.apply_to(&mut req);

    let (hook_program, lora_program) = hook_program_of(ctx, &fwd)?;
    // A decode lane whose KV is its tokens' alone asks the engine what it
    // wrote, so its slots are logged.
    let max_layers = ctx.resources().get(&fwd)?.max_layers;
    let plain_decode = !denoise
        && plain_decode(
            &req,
            hook_program || lora_program,
            fire_wide_mask,
            max_layers,
        );
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
        let rs = rs_ws_ids.get(lanes[0]).copied().filter(|_| !ring);
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
        parts.push(FireWs {
            ws: set.id,
            kvtxn: kvtxn.into_inner(),
            log: Some(LogFire { seq: log_seq, rows }),
            lease,
            fold: by.map(|by| FoldAt {
                by,
                decoded: plain_decode,
            }),
            reports: tags,
        });
    }
    let settles = settling(&parts);
    let submit_error = crate::scheduler::submit_prebuilt_tracked_async_with_kv_and_rs_copy_on(
        &scheduler,
        req,
        instance_id,
        pid,
        quorum_pipeline_id,
        completion.clone(),
        copy_src,
        copy_dst,
        rs_copy,
        frame,
        hook_program,
        lora_program,
    )
    .err()
    .map(|error| format!("{error:#}"));
    if let Some(error) = submit_error {
        let reason = format!("pipeline: device-geometry submit failed: {error}");
        record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
        return Ok(Err(reason));
    }
    ticket_reservation.commit();
    advance_shadow(ctx, &fwd)?;
    advance_cursors(&pass_lanes, None, &fold, &settles);

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

    struct TestContext {
        id: uuid::Uuid,
        resources: ResourceTable,
    }

    impl FireContext for TestContext {
        fn resources(&mut self) -> &mut ResourceTable {
            &mut self.resources
        }

        fn process_id(&self) -> uuid::Uuid {
            self.id
        }
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
                    kvtxn: Some(txn),
                    log: None,
                    lease: ws.fire_lease().expect("lease the submitted working set"),
                    fold: None,
                    reports: Vec::new(),
                }],
                device_geometry: false,
                rstxn: RsTxnsGuard::new(model, 0, None),
                model,
                engine: 0,
                fwd_rep: u32::MAX,
                cells,
                failure: failure.clone(),
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

    #[tokio::test(flavor = "current_thread")]
    async fn close_and_drop_share_graceful_fifo_drain_semantics() -> anyhow::Result<()> {
        let mut context = TestContext {
            id: uuid::Uuid::new_v4(),
            resources: ResourceTable::new(),
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
        a_pass_with_media_is_cut_before_its_first_span();
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
            host_cuts(&req, true, 128, 16, &[]),
            [(0, 100)],
            "short enough"
        );
        assert_eq!(
            host_cuts(&req, false, 32, 16, &[]),
            [(0, 100)],
            "not cuttable"
        );
        // Held 5: the first cut ends where page 2 does (32 - 5).
        assert_eq!(
            host_cuts(&req, true, 32, 16, &[]),
            [(0, 27), (27, 59), (59, 91), (91, 100)]
        );
        let mut masked = req.clone();
        masked.lanes[0].readout = ::engine::Readout::Rows(vec![3]);
        assert_eq!(
            host_cuts(&masked, true, 32, 16, &[]),
            [(0, 100)],
            "reads a middle row"
        );
        // Checkpoint rows cut it too, short as it is.
        assert_eq!(
            host_cuts(&req, true, 128, 16, &[43, 100, 0]),
            [(0, 43), (43, 100)]
        );
        assert_eq!(
            host_cuts(&req, true, 32, 16, &[43]),
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
        let cuts = host_cuts(&req, true, 32, 16, &[]);
        assert_eq!(cuts, [(0, 27), (27, 59), (59, 100)], "no cut past row 60");
        let first = rows_of_lanes(&req, &[Some((0, 27))], false, 16);
        assert!(first.media.is_empty(), "the text before it carries none");
        let tail = rows_of_lanes(&req, &[Some((59, 100))], true, 16);
        assert_eq!(tail.media[0].routes[..2], [1, 2]);
        assert_eq!(tail.media[0].routes[10], -1, "a dropped row stays dropped");
        assert_eq!(tail.media[0].token_positions.len(), 3 * 41);
        assert_eq!(tail.media[0].token_positions[0], 3 * 59);
    }

    fn several_lanes_are_cut_and_end_together() {
        let mut req = prefill_lane(5, 100, 16);
        let other = prefill_lane(0, 40, 16).lanes.remove(0);
        req.lanes.push(other);
        let pieces = pass_pieces(&req, true, true, 32, 16, &[]);
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
            pass_pieces(&req, true, false, 32, 16, &[]).len(),
            1,
            "not where the lanes carry recurrent state"
        );
    }

    fn the_last_piece_holds_what_the_program_reads() {
        let mut req = prefill_lane(5, 100, 16);
        req.lanes[0].readout = ::engine::Readout::Rows(vec![96, 95]);
        let tail = rows_of_lanes(&req, &[Some((91, 100))], true, 16);
        assert_eq!(tail.lanes[0].readout, ::engine::Readout::Rows(vec![5, 4]));
        req.lanes[0].readout = ::engine::Readout::Rows(vec![60, 99]);
        assert_eq!(
            host_cuts(&req, true, 32, 16, &[]),
            [(0, 27), (27, 59), (59, 100)],
            "no cut past the first row read"
        );
        // A score capture observes its fire's last rows: the last piece
        // holds them all.
        let mut req = prefill_lane(5, 100, 16);
        req.lanes[0].captures_scores = true;
        assert_eq!(
            host_cuts(&req, true, 32, 16, &[]),
            [(0, 27), (27, 59), (59, 100)]
        );
        let first = rows_of_lanes(&req, &[Some((0, 27))], false, 16);
        assert!(!first.lanes[0].captures_scores);
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
mod fold_tests {
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
                    let now = *cell.lock().unwrap();
                    geometry::Place {
                        folded: now.folded,
                        folded_exact: now.folded_exact,
                        tail: now.tail,
                        shared: false,
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
            &geometry::Fold::Device,
            std::slice::from_ref(&cell),
        );
        let now = *cell.lock().unwrap();
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
        settle_fold(&stores, ws, FoldAt { by, decoded: true }, &reported, true);
        assert_eq!(stores.kv.lock().folded(ws).unwrap(), 7);
        let now = *cell.lock().unwrap();
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
        settle_fold(&stores, ws, settled(2, 20), &none, true);
        assert!(!cell.lock().unwrap().folded_exact, "another is on its way");
        // Counted from where the first left it, short of what the fire wrote.
        settle_fold(&stores, ws, settled(9, 10), &none, true);
        assert_eq!(stores.kv.lock().folded(ws).unwrap(), 10);
        let now = *cell.lock().unwrap();
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
            true,
        );
        assert_eq!(stores.kv.lock().folded(ws).unwrap(), 4);
        let now = *cell.lock().unwrap();
        assert!(!now.folded_exact && now.unsettled == 1);
    }
}
