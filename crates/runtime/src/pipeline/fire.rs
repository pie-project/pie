//! Fire engine: prepare, run-ahead submit, finalize/poison a pass's fires.

/// Whether the bound container carries attention-stage programs.
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
pub mod geometry;
pub mod kv;
pub mod lease;
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

/// A pass's in-flight fires, submit order. The queue mutex is never held across
/// an await; the async finalizer gate serializes pop-through-finalize instead.
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

    /// Non-blocking [`Self::finalize_guard`], for the planner's eviction
    /// drain: the guard may be held indefinitely by the owner's guest task
    /// (e.g. a channel materialize awaiting a peer), and an eviction must
    /// abort rather than wait out another task's await.
    pub(crate) fn try_finalize_guard(&self) -> Option<tokio::sync::OwnedMutexGuard<()>> {
        Arc::clone(&self.finalizer).try_lock_owned().ok()
    }
}

pub type PendingFires = Arc<PendingFireQueue>;
pub type PipelineFailure = Arc<Mutex<Option<String>>>;

/// The pipeline's sticky failure, formatted as the guest-visible submit
/// error — one helper so the message cannot drift between the gates.
fn pipeline_failed(failure: &PipelineFailure) -> Option<String> {
    failure
        .lock()
        .unwrap()
        .as_ref()
        .map(|reason| format!("pipeline: pipeline failed: {reason}"))
}

struct CopyCompletionGuard {
    completion: Option<crate::engine::SubmissionCompletion>,
    lease: Option<KvFireLease>,
    model: usize,
    engine: usize,
    ws: crate::store::kv::page_table::WorkingSetId,
    indexes: Vec<u32>,
}

impl CopyCompletionGuard {
    fn invalidate(
        model: usize,
        engine: usize,
        ws: crate::store::kv::page_table::WorkingSetId,
        indexes: &[u32],
    ) {
        let stores = crate::store::registry::get(model, engine);
        if let Err(error) =
            crate::store::registry::with_kv_lock(&stores.kv, "host-working-set", |kv| {
                kv.invalidate_copied_pages(ws, indexes)
            })
        {
            tracing::error!(%error, "failed to invalidate copied KV page metadata");
        }
    }

    async fn finish(mut self) -> anyhow::Result<()> {
        let completion = self.completion.take().expect("copy completion present");
        let result = completion.await;
        Self::invalidate(self.model, self.engine, self.ws, &self.indexes);
        drop(self.lease.take());
        result
    }
}

impl Drop for CopyCompletionGuard {
    fn drop(&mut self) {
        let Some(completion) = self.completion.take() else {
            return;
        };
        let lease = self.lease.take();
        let model = self.model;
        let engine = self.engine;
        let ws = self.ws;
        let indexes = std::mem::take(&mut self.indexes);
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            Self::invalidate(model, engine, ws, &indexes);
            if let Some(lease) = lease {
                std::mem::forget(lease);
            }
            tracing::error!(
                "KV copy dropped without a Tokio runtime; invalidated metadata and preserved its lease"
            );
            return;
        };
        runtime.spawn(async move {
            let _ = completion.await;
            Self::invalidate(model, engine, ws, &indexes);
            drop(lease);
        });
    }
}

type PreparedExplicitKv = kv::PreparedExplicit;

/// Host-path prepared KV: the pre-launch D2D copy plan plus the open
/// transaction (`None` when nothing rebased).
type PreparedHostKv = kv::RealizedDeclaration;

/// `Stale`: demand grew between phase-A computation and prepare, nothing was
/// consumed, caller recomputes and re-acquires (bounded). `Fatal`: a real
/// preparation failure.
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

/// Bounded stale-demand recovery: how many times a fire re-acquires its
/// grant when the demand drifts while it awaits the previous one.
const STALE_DEMAND_ATTEMPTS: usize = 3;

/// Abort-on-drop guard for a fire's prepared KV transaction, armed until the
/// fire is enqueued (from there [`PendingFire`] owns commit/abort).
struct KvTxnGuard {
    model: usize,
    engine: usize,
    txn: Option<kv::KvTxn>,
}

impl KvTxnGuard {
    fn new(model: usize, engine: usize, txn: Option<kv::KvTxn>) -> Self {
        Self { model, engine, txn }
    }

    fn mapping_version(&self) -> Option<u64> {
        self.txn.as_ref().map(kv::KvTxn::mapping_version)
    }

    /// Hand the transaction to its post-submit owner ([`PendingFire`]).
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
            kv::abandon(store, txn);
        });
        // The abort recycled pages; parked asks may fit now.
        if let Some(planner) = crate::planner::planner_for(self.model, self.engine) {
            planner.pages_freed();
        }
    }
}

/// Settle-on-drop guard for a fire's published RS transaction; same protocol
/// as [`KvTxnGuard`], but the mapping is already authoritative so dropping
/// only releases the in-flight hold.
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
            rs::settle(&mut store, Some(txn));
        }
        // Settlement retired recycled slots; parked asks may fit now.
        if let Some(planner) = crate::planner::planner_for(self.model, self.engine) {
            planner.pages_freed();
        }
    }
}

fn host_kv_demand_locked(
    store: &mut crate::store::kv::KvStore,
    ws: &KvWorkingSet,
    writable: std::ops::Range<u64>,
    declaration_realized: bool,
) -> Result<usize, String> {
    let realization = if declaration_realized {
        0
    } else {
        kv::realize_declaration_demand(store, ws.id, writable.clone())
            .map_err(|error| error.to_string())?
    };
    let backing = store
        .backing_demand(ws.id, writable.end)
        .map_err(|error| error.to_string())?;
    realization
        .checked_add(backing)
        .ok_or_else(|| "KV demand exceeds usize".to_string())
}

/// Phase-A demand: pure computation, holding no grant, no pins, no txn.
fn host_kv_demand(
    stores: &crate::store::registry::Stores,
    ws: &KvWorkingSet,
    writable: std::ops::Range<u64>,
    declaration_realized: bool,
) -> Result<usize, String> {
    crate::store::registry::with_kv_lock(&stores.kv, "host-other", |store| {
        host_kv_demand_locked(store, ws, writable.clone(), declaration_realized)
    })
    .map_err(|error| format!("pipeline: KV demand: {error}"))
}

/// Acquire this fire's grant (KV pages and RS slots together; never
/// half-succeeds) from the residency planner. Zero demand yields an empty
/// grant. On `Yield` the process was chosen for eviction: settle its tail
/// (holding no lease/pins/txn at that point), wait out the eviction, re-ask.
async fn acquire_grant<C: FireContext>(
    ctx: &mut C,
    pipeline_id: uuid::Uuid,
    demand: crate::planner::Demand,
) -> Result<crate::planner::AllocationGrant, String> {
    if demand.is_zero() {
        return Ok(crate::planner::AllocationGrant::empty());
    }
    let Some(planner) = crate::planner::planner() else {
        return Err("pipeline: KV residency planner is not installed".to_string());
    };
    let pid = ctx.process_id();
    loop {
        match planner
            .acquire(pid, pipeline_id, demand)
            .await
            .map_err(|error| format!("pipeline: KV capacity: {error}"))?
        {
            crate::planner::Acquired::Granted(grant) => return Ok(grant),
            crate::planner::Acquired::Yield => settle_and_wait_resident(ctx).await?,
        }
    }
}

/// Back off for this process's eviction: settle its pipeline tails (releases
/// the fire leases the eviction's quiescence waits on), then wait it out.
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

/// Resolve the bound RS working sets for demand sizing (phase A: validation
/// only, no scope claim, no allocation).
fn bound_rs_working_set_ids<C: FireContext>(
    ctx: &mut C,
    model: usize,
    engine: usize,
    rs_reps: &[u32],
) -> Anyhow<Result<Vec<crate::store::rs::RsWorkingSetId>, String>> {
    let mut ids = Vec::with_capacity(rs_reps.len());
    for (row, &rep) in rs_reps.iter().enumerate() {
        let resource: Resource<RsWorkingSet> = Resource::new_borrow(rep);
        let rs = ctx.resources().get(&resource)?.clone();
        if rs.model != model || rs.engine != engine {
            return Ok(Err(format!(
                "pipeline: rs-working-set at request row {row} belongs to model/engine \
                 ({}, {}), expected ({model}, {engine})",
                rs.model, rs.engine
            )));
        }
        ids.push(rs.id);
    }
    Ok(Ok(ids))
}

/// Resolve the fire's `rs-geometry.fold-len` into the per-row plan the
/// lowering needs. `fold-len` counts the folded boundary from the current
/// boundary over `[buffer | this fire's tokens]`, clamped to `B + T`
/// (`u32::MAX` means "fold everything").
///
/// | `fold_len`   | buffer    | plan                       |
/// |--------------|-----------|----------------------------|
/// | `> 0`        | empty     | `Fold`                     |
/// | `0`          | empty     | `Buffer`                   |
/// | `0 < n <= B` | non-empty | `FoldBuffered`             |
///
/// A boundary strictly inside this fire's own new tokens (`B < n < B + T`)
/// is refused, not approximated: the tokens past it would get no outputs.
fn rs_plan_for(
    fold_len: &Option<Vec<u32>>,
    stores: &crate::store::registry::Stores,
    ids: &[crate::store::rs::RsWorkingSetId],
    qo_indptr: &[u32],
) -> Result<rs::RsPlan, String> {
    let rows = ids.len();
    if rows == 0 {
        return Ok(rs::RsPlan::Fold);
    }
    if let Some(lens) = fold_len
        && lens.len() != rows
    {
        return Err(format!(
            "rs-geometry.fold-len supplied {} length(s) for {rows} request row(s)",
            lens.len()
        ));
    }
    let mut row_tokens: Vec<u32> = (0..rows)
        .map(|row| {
            qo_indptr
                .get(row + 1)
                .zip(qo_indptr.get(row))
                .map(|(end, start)| end.saturating_sub(*start))
                .unwrap_or(0)
        })
        .collect();
    // **THE DEVICE-RESIDENT ROUND.** A device-decided fold length on a fire
    // that carries rows is the speculative window verb: the working set's
    // buffer is two runs of equal length, this fire writes one and replays
    // the port's count of the other. No occupancy is read or kept — the
    // host never learns how many tokens were accepted, which is the point.
    if fold_len.is_none() && row_tokens.iter().all(|&t| t > 0) {
        let store = stores.rs.lock().unwrap();
        let mut pages = Vec::with_capacity(rows);
        let mut phase = Vec::with_capacity(rows);
        let mut page_tokens = Vec::with_capacity(rows);
        for (row, id) in ids.iter().enumerate() {
            let slots = store.buffer_size(*id).map_err(|e| e.to_string())?;
            if slots < 2 || slots % 2 != 0 {
                return Err(format!(
                    "request row {row} folds a device-decided count over rows it buffers, which \
                     needs its rs-working-set buffer to be TWO equal runs (alloc-buffer an even \
                     count, twice the pages one window needs); it holds {slots} slot(s)"
                ));
            }
            let page = store.geometry(*id).map_err(|e| e.to_string())?.buffer_page_tokens.max(1);
            let run = slots / 2;
            if row_tokens[row] > run * page {
                return Err(format!(
                    "request row {row} fires {} rows into a window run of {run} page(s) x {page} \
                     tokens; grant a larger buffer",
                    row_tokens[row]
                ));
            }
            pages.push(run);
            phase.push(store.window_phase(*id).map_err(|e| e.to_string())?);
            page_tokens.push(page);
        }
        return Ok(rs::RsPlan::Window {
            pages,
            phase,
            page_tokens,
        });
    }
    // Buffer occupancy, in tokens, exact (not page-granular).
    let mut buffered: Vec<u32> = {
        let store = stores.rs.lock().unwrap();
        let mut out = Vec::with_capacity(rows);
        for (row, id) in ids.iter().enumerate() {
            match store.buffer_tokens(*id) {
                Ok(tokens) => out.push(tokens),
                // No exact occupancy after a device-resident fold length.
                Err(crate::store::rs::RsError::BufferOccupancyIndeterminate { bound }) => {
                    return Err(format!(
                        "request row {row} needs its exact buffer occupancy, but the working \
                         set's last fold had a device-resident length: at most {bound} token(s) \
                         remain and the true count reached only the engine. Free the buffer to \
                         settle the boundary before a fire that must replay it"
                    ));
                }
                Err(_) => out.push(0),
            }
        }
        out
    };

    // fold_len is device-resident; the host can still plan because the
    // engine clamps the resolved count to [1, b] (the row's live buffer),
    // always the same FoldBuffered replay regardless of this fire's tokens.
    if fold_len.is_none() {
        if let Some(row) = (0..rows).find(|row| buffered[*row] == 0) {
            return Err(format!(
                "rs-geometry.fold-len is device-resident, but request row {row} has an empty \
                 buffer: there is nothing for the device's count to name"
            ));
        }
        return Ok(rs::RsPlan::FoldBuffered {
            tokens: buffered,
            fold_len_is_device: true,
        });
    }
    let fold_len = fold_len.as_deref().expect("checked above");

    // Classify each row, then decide what the pass can be. Fold and Buffer
    // mix freely (both compute over the extended layout, differing only in
    // whether the recurrence persists); Commit does not mix with anything,
    // since it replays slabs instead of computing.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Position {
        Fold,
        Buffer,
        Commit,
    }
    let mut kinds: Vec<Position> = Vec::with_capacity(rows);
    let mut fold_tokens: Vec<u32> = vec![0; rows];
    for row in 0..rows {
        let (b, t) = (buffered[row], row_tokens[row]);
        let n = fold_len[row].min(b + t);
        fold_tokens[row] = n;
        let here = if t == 0 {
            // A row spanning no tokens is a pure replay: gather the buffered
            // prefix [0, n) and return before the output projection.
            if n == 0 {
                return Err(format!(
                    "request row {row} carries no tokens and folds nothing: it neither \
                     computes nor moves the boundary, so the fire has no effect. \
                     A row spanning no tokens means \"replay the buffered prefix \
                     and stop\", which needs a fold length"
                ));
            }
            Position::Commit
        } else if n == 0 {
            // Pure append: extended layout when b > 0, boundary unmoved.
            Position::Buffer
        } else if n == b + t {
            // Fold takes the whole row: plain in-forward advance if b == 0,
            // else the same over the extended [b|t] layout (boundary is the
            // last token, so ordinary EOS writeback covers it).
            if b == 0 {
                Position::Fold
            } else {
                Position::Buffer
            }
        } else {
            // Otherwise runs over the extended [b|t] layout: replays the b
            // buffered tokens ahead of the t new ones and snapshots recurrent
            // state at extended token n. Not `commit_len` (which truncates):
            // an interior boundary still owes logits past it.
            Position::Buffer
        };
        kinds.push(here);
    }

    // A pure commit cannot share a fire: its rows are not computed at all, so
    // there is no per-row switch that would let a computing row ride along.
    if kinds.contains(&Position::Commit) && !kinds.iter().all(|k| *k == Position::Commit) {
        let row = kinds
            .iter()
            .position(|k| *k == Position::Commit)
            .expect("checked above");
        return Err(format!(
            "request row {row} only replays buffered tokens while another row of the same \
             fire computes new ones; a buffered commit gathers its activations instead of \
             producing them, so it cannot share a pass. Split the fire"
        ));
    }

    let pass = if kinds.iter().all(|k| *k == Position::Commit) {
        Position::Commit
    } else if kinds.iter().all(|k| *k == Position::Fold) {
        Position::Fold
    } else {
        // Mixed or uniformly buffered: pass runs the buffered shape; an
        // `in_forward` row simply owns no buffer inside it.
        Position::Buffer
    };
    let in_forward: Vec<bool> = kinds.iter().map(|k| *k == Position::Fold).collect();
    for (row, forward) in in_forward.iter().enumerate() {
        if *forward {
            // Boundary moves through its own tokens, not a buffer: no fold
            // length (validate_fold measures against buffered capacity).
            fold_tokens[row] = 0;
            buffered[row] = 0;
            row_tokens[row] = 0;
        }
    }

    Ok(match pass {
        Position::Fold => rs::RsPlan::Fold,
        // Each row opens at its own exact occupancy.
        Position::Buffer => rs::RsPlan::Buffer {
            start_tokens: buffered,
            row_tokens,
            fold_tokens,
            in_forward,
        },
        Position::Commit => rs::RsPlan::FoldBuffered {
            fold_len_is_device: false,
            // fold_tokens, not raw fold_len: WIT clamps the length to the
            // tail, so "fold everything" is the fire-invariant u32::MAX.
            tokens: fold_tokens,
        },
    })
}

/// Drop the synthesized readout for a `FoldBuffered` fire: it produces no
/// logits, and a defaulted readout has no expressible shape here. An
/// explicit readout on a buffered fold still reaches the engine (and is
/// refused there).
fn suppress_defaulted_readout_for_fold(
    req: &mut crate::engine::FireRequest,
    readout_defaulted: bool,
    plan: &rs::RsPlan,
) {
    let replays_buffer = matches!(plan, rs::RsPlan::FoldBuffered { .. });
    if !replays_buffer || !readout_defaulted {
        return;
    }
    // Readout::None: this lane runs for its cache writes alone.
    for lane in &mut req.lanes {
        lane.readout = ::engine::Readout::None;
    }
}

fn rs_slot_demand(
    stores: &crate::store::registry::Stores,
    ids: &[crate::store::rs::RsWorkingSetId],
    plan: &rs::RsPlan,
) -> Result<u32, String> {
    if ids.is_empty() {
        return Ok(0);
    }
    let store = stores.rs.lock().unwrap();
    let demand = rs::demand(&store, ids, plan)?;
    u32::try_from(demand).map_err(|_| "pipeline: RS demand exceeds the contention ABI".to_string())
}

/// Prepare the host-projected KV write from the fire's grant, under one
/// store lock (staleness gate re-checked). On `Stale` nothing was consumed.
fn prepare_host_kv_reserved(
    stores: &crate::store::registry::Stores,
    ws: &KvWorkingSet,
    writable: std::ops::Range<u64>,
    declaration_realized: bool,
    grant: &mut crate::planner::AllocationGrant,
) -> Result<PreparedHostKv, ReservedError> {
    crate::store::registry::with_kv_lock(&stores.kv, "host-other", |store| {
        let required = host_kv_demand_locked(store, ws, writable.clone(), declaration_realized)
            .map_err(|error| ReservedError::Fatal(format!("pipeline: KV demand: {error}")))?;
        if required > grant.remaining_kv() {
            return Err(ReservedError::Stale);
        }
        let (copies, txn) = if declaration_realized {
            ((Vec::new(), Vec::new()), None)
        } else {
            kv::realize_declaration_reserved(store, ws.id, writable.clone(), grant.lend_kv())
                .map_err(|error| {
                    ReservedError::Fatal(format!("pipeline: KV declaration realization: {error}"))
                })?
        };
        if let Err(error) = store.ensure_backed_reserved(ws.id, writable.end, grant.lend_kv()) {
            if let Some(txn) = txn {
                kv::abandon(store, txn);
            }
            return Err(ReservedError::Fatal(format!(
                "pipeline: KV backing frontier: {error}"
            )));
        }
        Ok((copies, txn))
    })
}

/// Prepare a device-geometry explicit-KV write from the fire's grant, under
/// one store lock with the same staleness gate as the host path.
fn prepare_explicit_kv_reserved(
    stores: &crate::store::registry::Stores,
    ws: &KvWorkingSet,
    write_indexes: &[u64],
    grant: &mut crate::planner::AllocationGrant,
) -> Result<PreparedExplicitKv, ReservedError> {
    crate::store::registry::with_kv_lock(&stores.kv, "host-other", |store| {
        let required =
            kv::prepare_explicit_demand(store, ws.id, write_indexes).map_err(|error| {
                ReservedError::Fatal(format!("pipeline: device-geometry demand: {error}"))
            })?;
        if required > grant.remaining_kv() {
            return Err(ReservedError::Stale);
        }
        kv::prepare_explicit_reserved(store, ws.id, write_indexes, grant.lend_kv()).map_err(
            |error| ReservedError::Fatal(format!("pipeline: device-geometry grant: {error}")),
        )
    })
}

/// A pipeline FIFO entry: a forward fire holding its ordered slot on the
/// stream; `take`/`read` drain entries in submit order. KV cell moves are
/// awaited inline by `copy_into_inner` and never enter the FIFO.
pub enum PendingOp {
    Fire(PendingFire),
    #[cfg(test)]
    TestStub,
}

impl PendingOp {
    /// Non-blocking probe: whether the op's completion has settled.
    pub(crate) fn is_settled(&self) -> bool {
        match self {
            PendingOp::Fire(fire) => fire.completion.is_settled(),
            #[cfg(test)]
            PendingOp::TestStub => true,
        }
    }

    /// An owned, payload-free await on this op's completion (cloned so the
    /// pipeline queue lock is not held across the await). The outcome is
    /// ignored; the FIFO drain reads the real result.
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
                kv: FireKv::Host(_),
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
    ReclaimDeviceGeometry {
        fwd_rep: u32,
        instance_id: u64,
    },
}

pub(crate) struct FinalizeOutcome {
    action: FinalizeAction,
    ws_guard: Option<KvFireLease>,
}

/// See [`PendingOp::completion_signal`].
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

/// Test-only inert FIFO entry: makes a pass's shared fires queue non-empty
/// for `ForwardPass` drop tests without needing a live completion.
#[cfg(test)]
pub(crate) fn test_pending_op_stub() -> PendingOp {
    PendingOp::TestStub
}

/// The open KV/arena transaction(s) one in-flight fire holds until it
/// resolves. Two shapes: the ordinary single-seq / MTP projection ([`kv`]),
/// or a device-geometry fire whose KV the engine resolves+writes itself.
enum FireKv {
    Host(Option<kv::KvTxn>),
    /// A device-geometry fire's prepared write over the lease-granted
    /// slots: same commit/abort protocol, no host projection.
    DeviceGeom {
        kvtxn: kv::KvTxn,
    },
}

/// One in-flight fire: the work item completion plus everything needed to
/// finalize when it resolves — the open KV/RS txns (pins held until
/// settlement) and the bound cells whose mirror epochs become visible.
pub struct PendingFire {
    completion: crate::engine::WorkItemCompletion,
    kv: FireKv,
    /// The published RS write, still holding the store's retirement pin.
    ///
    /// A GUARD, not a bare txn: a fire that is never finalized — its process
    /// torn down, its completion cancelled — must still release the pin. An
    /// unsettled sequence stays in `RsStore::outstanding` forever, and
    /// `retire_idle` retires only through `oldest_outstanding - 1`, so ONE
    /// leaked fire stops every recycled state slot in the pool from ever being
    /// handed out again. That is a permanent, whole-server outage, so the
    /// release cannot depend on reaching any particular line.
    rstxn: RsTxnsGuard,
    ws_guard: KvFireLease,
    model: usize,
    engine: usize,
    /// The owning pass, to fail it on a fire error.
    fwd_rep: u32,
    instance_id: u64,
    cells: BoundCells,
    failure: PipelineFailure,
}

#[allow(
    clippy::too_many_arguments,
    reason = "one fire's recurrent-state binding context. `ctx`, `stores` and `grant` \
              are three separate `&mut` borrows of three different owners — bundling \
              them into one struct would force a single borrow and make the \
              disjointness the borrow checker currently proves impossible to express"
)]
fn prepare_bound_rs<C: FireContext>(
    ctx: &mut C,
    stores: &crate::store::registry::Stores,
    model: usize,
    engine: usize,
    rs_reps: &[u32],
    qo_indptr: &[u32],
    pipeline_scope: &crate::store::PipelineScope,
    plan: &rs::RsPlan,
    grant: &mut crate::planner::AllocationGrant,
) -> Anyhow<Result<rs::PreparedRs, ReservedError>> {
    let has_recurrent_state = crate::model::model().rs_caps().state_size > 0;
    if let Err(error) = rs::validate_count(rs_reps.len(), qo_indptr, has_recurrent_state) {
        return Ok(Err(ReservedError::Fatal(format!(
            "pipeline: recurrent-state binding: {error}"
        ))));
    }
    if rs_reps.is_empty() {
        return Ok(Ok(rs::PreparedRs::empty()));
    }

    // Resolution + model/engine validation live in the phase-A resolver.
    let working_sets = match bound_rs_working_set_ids(ctx, model, engine, rs_reps)? {
        Ok(ids) => ids,
        Err(error) => return Ok(Err(ReservedError::Fatal(error))),
    };
    for (row, &rep) in rs_reps.iter().enumerate() {
        let resource: Resource<RsWorkingSet> = Resource::new_borrow(rep);
        let rs = ctx.resources().get(&resource)?.clone();
        if let Err(owner) = rs.claim_pipeline_scope(pipeline_scope) {
            return Ok(Err(ReservedError::Fatal(format!(
                "pipeline: rs-working-set at request row {row} is already scoped to pipeline \
                 {owner:#x}"
            ))));
        }
    }

    let prepared = {
        let mut store = stores.rs.lock().unwrap();
        // Staleness gate under the same lock as the prepare: if demand grew
        // while awaiting the grant, nothing is consumed and the caller
        // re-acquires.
        let required = match rs::demand(&store, &working_sets, plan) {
            Ok(required) => required,
            Err(error) => {
                return Ok(Err(ReservedError::Fatal(format!(
                    "pipeline: rs demand: {error}"
                ))));
            }
        };
        if required > grant.remaining_rs() {
            return Ok(Err(ReservedError::Stale));
        }
        rs::prepare_many_reserved(&mut store, &working_sets, plan, grant.lend_rs())
    };
    Ok(prepared.map_err(|error| ReservedError::Fatal(format!("pipeline: rs prepare: {error}"))))
}

/// Drain every in-flight fire on this pipeline to settlement. This is the
/// host-side ordering seam for out-of-band ops (`copy_into`) that read
/// committed physical ids and act on them off the fire path — not an RS
/// ordering rule, since RS mappings publish at prepare in submission order.
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
        // Predecessor fires retire regardless of this process's residency,
        // so a plain await cannot deadlock the planner.
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

/// Stamp each lane's fact word, which `engine::fire::compose` turns into a
/// class and therefore the row window every guarded node runs over. Called
/// here because a lane's mask is only known once the fire's mask has
/// lowered. `fire_wide_mask` covers the device-resident case, where the mask
/// lives on the fire's channel cell rather than any `Lane::mask`.
/// `carries_media` is the submission's fact, not the lane's, and must be set
/// wherever a model guards its embed merge on `Facts::media`.
fn stamp_lane_words(
    req: &mut crate::engine::FireRequest,
    fire_wide_mask: bool,
    carries_media: bool,
) {
    let model = crate::model::model();
    for lane in &mut req.lanes {
        let rows = u32::try_from(lane.tokens.len()).unwrap_or(u32::MAX);
        // Word and the intents it names must be one reading of one lane, or
        // the shell refuses the fire (Fault::AdapterWord/DraftWord/ScoreWord).
        lane.word = model.word(
            rows,
            lane.mask.is_some() || fire_wide_mask,
            lane.adapter.is_some(),
            lane.drafts,
            lane.captures_scores,
            carries_media,
            lane.block_draft,
        );
    }
}

/// Seat this fire's lanes (`Lane::slot`) from the KV working set that owns
/// them. A slot is a sequence's seat in the shell's pools (kv block,
/// recurrent bank row, `slot_ids[lane]`). Lane `i` of every fire of a
/// working set sits in the same seat, which a recurrent bank row depends on.
///
/// # Errors
///
/// When this fire would seat more sequences than the deployment's pools
/// hold (`Budgets::slots`).
pub(crate) fn stamp_lane_slots(
    req: &mut crate::engine::FireRequest,
    stores: &crate::store::registry::Stores,
    ws: crate::store::kv::page_table::WorkingSetId,
) -> Result<(), String> {
    let seats = stores
        .seats
        .lock()
        .unwrap()
        .seats(ws, req.lanes.len())
        .map_err(|error| format!("pipeline: seating this fire's lanes: {error}"))?;
    for (lane, &seat) in req.lanes.iter_mut().zip(&seats) {
        lane.slot = seat;
    }
    Ok(())
}

/// Rewrite each lane's page list from working-set-relative indexes to pool
/// page ids: [`KvDelta::pages`] takes ids directly, so the guest's relative
/// indexing must be translated here. Runs after the fire's KV prepare, so a
/// first prefill's own pages are already in the mapping. A page the working
/// set hasn't mapped is refused, rather than aliasing another sequence's page.
///
/// [`KvDelta::pages`]: ::engine::KvDelta::pages
pub(crate) fn map_lane_pages(
    req: &mut crate::engine::FireRequest,
    stores: &crate::store::registry::Stores,
    ws: crate::store::kv::page_table::WorkingSetId,
) -> Result<(), String> {
    if req.lanes.iter().all(|lane| lane.kv.pages.is_empty()) {
        return Ok(());
    }
    let table = working_set_flat_table(stores, ws)?;
    for lane in &mut req.lanes {
        for page in &mut lane.kv.pages {
            let Some(&pool) = table.get(*page as usize) else {
                return Err(format!(
                    "pipeline: KV page {page} escapes the working set's {} mapped page(s)",
                    table.len()
                ));
            };
            *page = pool;
        }
    }
    Ok(())
}

/// The working set's flattened logical-to-physical table: entry `i` is the
/// pool page backing working-set-relative index `i`.
fn working_set_flat_table(
    stores: &crate::store::registry::Stores,
    ws: crate::store::kv::page_table::WorkingSetId,
) -> Result<Vec<u32>, String> {
    crate::store::registry::with_kv_lock(&stores.kv, "host-pages", |kv| {
        Ok::<Vec<u32>, crate::store::kv::KvStoreError>(
            kv.flat_table(ws)?.1.iter().map(|page| page.0).collect(),
        )
    })
    .map_err(|error| format!("pipeline: KV page translation: {error}"))
}

/// Hand the engine the page table directly, for the device-geometry class
/// whose page references the host never reads (nothing to rewrite as in
/// [`map_lane_pages`]).
fn stamp_lane_translation(
    req: &mut crate::engine::FireRequest,
    stores: &crate::store::registry::Stores,
    ws: crate::store::kv::page_table::WorkingSetId,
) -> Result<(), String> {
    let table = working_set_flat_table(stores, ws)?;
    for lane in &mut req.lanes {
        lane.kv.translation.clone_from(&table);
    }
    Ok(())
}

/// Poison every host-reader cell of a pass with the failed fire's error —
/// under run-ahead this IS the error channel (`take`/`read` surface it).
fn poison_readers(cells: &BoundCells, reason: &str) {
    for cell in cells {
        let mut c = cell.lock().unwrap();
        if c.role == Some(HostRole::Reader) {
            c.poison(reason);
            // A waiter parked on the reader wait slot must observe the poison.
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

    /// The host predicts these head/tail reservations; the device checks
    /// them where the data is. Lands on the request's first lane, and only
    /// when every channel was adopted.
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

/// Park until the channel can make progress: its endpoint's reader word
/// advances, or the oldest in-flight pipeline op settles so the caller can
/// drain it. Errors surface poison/closure or a definitively empty channel.
pub(crate) async fn await_channel_progress(
    cell: &Arc<Mutex<crate::pipeline::channel::ChannelCell>>,
    fires: Option<&PendingFires>,
) -> Result<(), String> {
    let wait = cell.lock().unwrap().reader_wait_state();
    let oldest = fires.and_then(|f| f.lock().unwrap().front().map(|op| op.completion_signal()));
    match (wait, oldest) {
        (Some((endpoint, observed_tail)), Some(signal)) => {
            // Race the direct channel wake against the oldest op so a fire
            // that resolves without producing here still unblocks the loop.
            // Poison/closure re-classify on the caller's next take/read.
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

/// The body behind one non-no-op slot of `forward.submit(on, slots)`.
pub async fn submit_pass_stamped<C: FireContext>(
    ctx: &mut C,
    this: Resource<Pipeline>,
    fwd: Resource<ForwardPass>,
    frame: Option<crate::scheduler::FrameStamp>,
) -> Anyhow<Result<(), String>> {
    // Compiles away without `profile-fire`.
    let submit_probe = crate::scheduler::probe::host_submit();
    let submit_clock = crate::scheduler::probe::ProbeClock::start();
    {
        // Device-geometry pass: geometry is device-produced, so this pass
        // leases physical pages and fires prebuilt, still running ahead via
        // the FIFO like any pass.
        if ctx.resources().get(&fwd)?.devgeo.is_some() {
            return fire_device_geometry(ctx, this, fwd, frame).await;
        }
        // Point this pass's channels at this pipeline's FIFO so take/read
        // await the right queue; every pass binding a channel must submit
        // on one pipeline.
        let (pipe_fires, pipeline_failure, pipeline_scope) = {
            let pipeline = ctx.resources().get(&this)?;
            if pipeline.scope.is_closed() {
                return Ok(Err("pipeline: pipeline is closed".to_string()));
            }
            (
                pipeline.fires.clone(),
                pipeline.failure.clone(),
                pipeline.scope.clone(),
            )
        };
        // Non-blocking settlement drain: resolved fires' KV/RS txns finalize
        // here so arena pins stay bounded even when the guest never takes.
        {
            let began = crate::scheduler::probe::ProbeClock::start();
            drain_settled(ctx, Some(&pipe_fires)).await?;
            crate::probe_fire_record!(submit_probe.drain_settled_us, began.elapsed());
        }
        if let Some(error) = pipeline_failed(&pipeline_failure) {
            return Ok(Err(error));
        }
        if let Err(error) = wire_channels_to_pipeline(ctx, &fwd, &pipe_fires)? {
            return Ok(Err(error));
        }
        // An RS-binding pass needs no extra serialization: its mapping
        // publishes at prepare, in submission order.
        let (
            geometry,
            cells,
            ws_rep,
            rs_reps,
            rs_fold_len,
            kv_declaration,
            kv_declaration_realized,
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
            // A decode-envelope pass carries exactly one value the host
            // cannot fold: the sampled token, read off the ring the
            // previous epilogue wrote.
            let device_resolved = match &p.decode_envelope {
                Some(envelope) if envelope.device_fold_len => {
                    PortMask::of(&[Port::EmbedTokens, Port::RsFoldLen])
                }
                Some(_) => PortMask::of(&[Port::EmbedTokens]),
                None => PortMask::NONE,
            };
            let geometry_clock = crate::scheduler::probe::ProbeClock::start();
            let (geometry, attn_mask) = {
                let bound = &p.instance.program.bound;
                let (shadow, shadow_cells) = (&p.host_shadow, &p.cells);
                let mut known = |chan: u32| shadow.fire_value(bound, shadow_cells, chan);
                match geometry::map_geometry_evaluated_with(bound, &mut known, device_resolved) {
                    Ok((geometry, evaluated)) => {
                        // In-band -1 skips are the device-resolved contract;
                        // a host-wire fire would embed the sentinel as a
                        // real token.
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
            // A fold length the envelope resolves on the device is the
            // device's from the first fire: the host's seed is only the first
            // cell the port reads, so the plan is the window verb even when
            // the host could have peeked that cell.
            let rs_fold_len = if p
                .decode_envelope
                .as_ref()
                .is_some_and(|envelope| envelope.device_fold_len)
            {
                None
            } else {
                p.rs_fold_len.clone()
            };
            (
                geometry,
                p.cells.clone(),
                p.kv_ws,
                p.rs_ws.clone(),
                rs_fold_len,
                p.kv_declaration,
                p.kv_declaration_realized,
                fwd.rep(),
                p.bound_instance.instance_id,
                p.scheduler.clone(),
                attn_mask,
                accesses,
                p.decode_envelope.clone(),
                reads_attn_score,
                reads_mtp_logits,
            )
        };
        let mut req = crate::engine::FireRequest::default();
        let readout_defaulted = geometry.readout_defaulted;
        geometry.apply_to(&mut req);
        // Every fire through here fires a `BoundForwardPass`: the engine
        // runs its pass after the forward with this lane's logits row bound
        // as the `logits` intrinsic.
        req.boundary_program = true;
        // A program that materializes `IntrinsicId::AttnScore` is a
        // capturing lane, stamped on every lane of the member.
        if p_reads_attn_score {
            for lane in &mut req.lanes {
                lane.captures_scores = true;
            }
        }
        // A program that materializes `IntrinsicId::MtpLogits` is a drafting
        // lane: the fact selects the model text's draft arm for its rows, and
        // the engine binds the `mtp` seam's rectangle to the intrinsic.
        if p_reads_mtp_logits {
            for lane in &mut req.lanes {
                lane.drafts = true;
            }
        }
        // A pass the guest marked `set-drafting-block` carries a block
        // drafter's proposal rows, which a plan guards its trunk against.
        // Stated, not inferred: what makes a fire a draft is the anchor the
        // guest chose from the accepted prefix, and no intrinsic shows it.
        if ctx.resources().get(&fwd)?.block_draft {
            for lane in &mut req.lanes {
                lane.block_draft = true;
            }
        }
        // A class, not a bool: `fire_device_geometry` stamps the third class
        // this path cannot reach.
        req.geometry = if decode_envelope.is_some() {
            GeometryClass::DecodeEnvelope
        } else {
            GeometryClass::Host
        };
        req.single_token_mode = req.lanes.iter().all(|lane| lane.tokens.len() == 1);
        // The pass's layer truncation rides every fire; the scheduler's
        // region table carries it to the engine as per-region k.
        req.max_layers = {
            let p = ctx.resources().get(&fwd)?;
            p.max_layers
        };
        // The run scan finds the model's reserved placeholder runs in the
        // submitted tokens and matches them to the attached media spans in
        // order, refusing any disagreement.
        let media_spans = ctx.resources().get(&fwd)?.bindings.media.clone();
        let carries_media = !media_spans.is_empty();
        let matched = {
            let lane_tokens: Vec<&[u32]> =
                req.lanes.iter().map(|lane| lane.tokens.as_slice()).collect();
            let scanned = if carries_media {
                crate::pipeline::media::scan(&lane_tokens, &media_spans)
            } else {
                // Text-only fire: a run with no span behind it would embed
                // the pad id as an ordinary token.
                crate::pipeline::media::refuse_orphan_runs(
                    &lane_tokens,
                    crate::model::media_pad(),
                )
                .map(|()| Vec::new())
            };
            match scanned {
                Ok(matched) => matched,
                Err(refusal) => return Ok(Err(format!("pipeline: {refusal}"))),
            }
        };
        // The mask is on the fire when it stays device-resident, with no
        // `Lane::mask` to read it back off — so the class is read here,
        // before `apply_to` consumes it.
        let fire_wide_mask = matches!(attn_mask, geometry::FireAttnMask::Device);
        if let Err(error) = attn_mask.apply_to(&mut req) {
            return Ok(Err(format!("pipeline: fire attention mask: {error}")));
        }
        stamp_lane_words(&mut req, fire_wide_mask, carries_media);
        // `engine::fire::StepMedia` is the parallel slice keyed by lane;
        // `scheduler::batch` rebases each row's lane onto its co-batched
        // step.
        if !matched.is_empty() {
            let lane_rows: Vec<u32> = req
                .lanes
                .iter()
                .map(|lane| u32::try_from(lane.tokens.len()).unwrap_or(u32::MAX))
                .collect();
            let lane_base: Vec<u32> = req
                .lanes
                .iter()
                .map(|lane| lane.positions.first().copied().unwrap_or(0))
                .collect();
            req.media = crate::pipeline::media::lane_media(&matched, &lane_rows, &lane_base);
        }
        crate::offload::try_encode(&mut req).await;
        // Resource preparation is independent of token position: realize the
        // declaration once, back only its missing frontier.
        let kv_clock = crate::scheduler::probe::ProbeClock::start();
        let ws_res: Resource<KvWorkingSet> = Resource::new_borrow(ws_rep);
        let ws = ctx.resources().get(&ws_res)?.clone();
        let stores = crate::store::registry::get(ws.model, ws.engine);
        if let Err(refusal) = stamp_lane_slots(&mut req, &stores, ws.id) {
            return Ok(Err(refusal));
        }
        let (readable_pages, writable_pages) =
            match crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv_store| {
                let page_len = kv_store.page_len(ws.id)?;
                Ok::<_, crate::store::kv::KvStoreError>((
                    kv_declaration.readable.resolve(page_len).map_err(|_| {
                        crate::store::kv::KvStoreError::BadWriteSet {
                            reason: "invalid readable page declaration",
                        }
                    })?,
                    kv_declaration.writable.resolve(page_len).map_err(|_| {
                        crate::store::kv::KvStoreError::BadWriteSet {
                            reason: "invalid writable page declaration",
                        }
                    })?,
                ))
            }) {
                Ok(ranges) => ranges,
                Err(error) => {
                    return Ok(Err(format!(
                        "pipeline: KV working-set declaration: {error}"
                    )));
                }
            };
        // Structural declaration checks — fail fast, before anything is
        // claimed, acquired, or prepared.
        if writable_pages.is_empty() {
            return Ok(Err(
                "pipeline: writable KV page declaration is empty".to_string()
            ));
        }
        if decode_envelope.is_none()
            && let Some(page) = req
                .pages()
                .find(|&page| !readable_pages.contains(&u64::from(page)))
        {
            return Ok(Err(format!(
                "pipeline: KV read page {page} escapes the readable declaration"
            )));
        }
        let model = ws.model;
        let engine = ws.engine;
        let pid = ctx.process_id();
        let quorum_pipeline_id = pipeline_scope.scheduler_id();
        if let Err(owner) = ws.claim_pipeline_scope(&pipeline_scope) {
            return Ok(Err(format!(
                "pipeline: KV working set is already scoped to pipeline {owner:032x}"
            )));
        }
        // Phase A: pure demand, holding nothing. Phase B: acquire the one
        // grant (KV pages + RS slots; never half-succeeds). Phase C: prepare
        // from the grant, re-acquiring (bounded) if demand drifted while
        // waiting.
        let rs_ws_ids = match bound_rs_working_set_ids(ctx, model, engine, &rs_reps)? {
            Ok(ids) => ids,
            Err(error) => return Ok(Err(error)),
        };
        let rs_plan = match rs_plan_for(&rs_fold_len, &stores, &rs_ws_ids, &req.qo_indptr()) {
            Ok(plan) => plan,
            Err(error) => {
                return Ok(Err(format!("pipeline: recurrent-state mode: {error}")));
            }
        };
        suppress_defaulted_readout_for_fold(&mut req, readout_defaulted, &rs_plan);
        let mut attempts = 0;
        let (ws_guard, (copy_src, copy_dst), kvtxn, rs_prepared) = loop {
            let kv_demand = match host_kv_demand(
                &stores,
                &ws,
                writable_pages.clone(),
                kv_declaration_realized,
            ) {
                Ok(demand) => demand,
                Err(error) => return Ok(Err(error)),
            };
            let Ok(kv_demand) = u32::try_from(kv_demand) else {
                return Ok(Err(
                    "pipeline: KV demand exceeds the planner ABI".to_string()
                ));
            };
            let rs_demand = match rs_slot_demand(&stores, &rs_ws_ids, &rs_plan) {
                Ok(demand) => demand,
                Err(error) => return Ok(Err(error)),
            };
            let demand = crate::planner::Demand {
                kv_pages: kv_demand,
                rs_slots: rs_demand,
            };
            let mut grant = match acquire_grant(ctx, quorum_pipeline_id, demand).await {
                Ok(grant) => grant,
                Err(error) => return Ok(Err(error)),
            };
            // The lease is the suspend seal: acquired after any park (a
            // parked ask must hold no lease) and before the prepare (so an
            // eviction either sees this lease and waits it out, or fences
            // first and this fire backs off to wait it out).
            let ws_guard = match ws.fire_lease() {
                Ok(lease) => lease,
                Err(crate::store::kv::working_set::FireLeaseError::Fenced) => {
                    drop(grant); // the pages fund the eviction's head
                    if let Err(error) = settle_and_wait_resident(ctx).await {
                        return Ok(Err(error));
                    }
                    continue;
                }
                Err(error) => return Ok(Err(format!("pipeline: KV working set: {error}"))),
            };
            let (copies, kvtxn) = match prepare_host_kv_reserved(
                &stores,
                &ws,
                writable_pages.clone(),
                kv_declaration_realized,
                &mut grant,
            ) {
                Ok(prepared) => prepared,
                Err(ReservedError::Stale) if attempts < STALE_DEMAND_ATTEMPTS => {
                    attempts += 1;
                    continue; // the grant drops here; its pages serve the queue
                }
                Err(ReservedError::Stale) => return Ok(Err(stale_demand_error())),
                Err(ReservedError::Fatal(error)) => return Ok(Err(error)),
            };
            let kvtxn = KvTxnGuard::new(model, engine, kvtxn);
            // Recurrent-state rows are lowered independently, in resolved
            // request order; their CoW copies ride the scheduler's typed
            // pre-launch state copy so a copy failure rejects the fire
            // before model execution.
            match prepare_bound_rs(
                ctx,
                &stores,
                model,
                engine,
                &rs_reps,
                &req.qo_indptr(),
                &pipeline_scope,
                &rs_plan,
                &mut grant,
            )? {
                Ok(prepared) => break (ws_guard, copies, kvtxn, prepared),
                Err(ReservedError::Stale) if attempts < STALE_DEMAND_ATTEMPTS => {
                    attempts += 1;
                    // kvtxn's guard aborts the prepared KV write on drop.
                    continue;
                }
                Err(ReservedError::Stale) => return Ok(Err(stale_demand_error())),
                Err(ReservedError::Fatal(error)) => return Ok(Err(error)),
            }
        };
        crate::probe_fire_record!(submit_probe.kv_prepare_us, kv_clock.elapsed());
        // Earliest instant the mapping is complete: this fire's own write
        // targets are reserved only now.
        if let Err(refusal) = map_lane_pages(&mut req, &stores, ws.id) {
            return Ok(Err(refusal));
        }
        rs_prepared.apply_to(&mut req);
        let (rs_copy_src, rs_copy_dst) = rs_prepared.copies.clone();
        let rstxns = RsTxnsGuard::new(model, engine, rs_prepared.txn);
        let completion = ctx
            .resources()
            .get_mut(&fwd)?
            .bound_instance
            .reserve_completion();

        // Preparation is complete in guest order; the scheduler sees only
        // launch-ready work.
        let ticket_reservation = TicketReservation::new(&cells, &accesses);
        ticket_reservation.apply_to(&mut req);

        let (hook_program, lora_program) = {
            let p = ctx.resources().get(&fwd)?;
            let container = &p.instance.program.bound.container;
            (
                container_has_attention_stages(container),
                container_has_lora_sink(container),
            )
        };
        let scheduler_clock = crate::scheduler::probe::ProbeClock::start();
        let submit_error = crate::scheduler::submit_prebuilt_tracked_async_with_kv_and_rs_copy_on(
            &scheduler,
            req,
            instance_id,
            pid,
            quorum_pipeline_id,
            completion.clone(),
            copy_src,
            copy_dst,
            rs_copy_src,
            rs_copy_dst,
            frame,
            hook_program,
            lora_program,
        )
        .err()
        .map(|error| format!("{error:#}"));
        crate::probe_fire_record!(submit_probe.scheduler_submit_us, scheduler_clock.elapsed());
        if let Some(error) = submit_error {
            // The KV/RS transaction guards roll everything back on return.
            let reason = format!("pipeline: submit failed: {error}");
            record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
            return Ok(Err(reason));
        }
        ticket_reservation.commit();

        {
            let began = crate::scheduler::probe::ProbeClock::start();
            let p = ctx.resources().get_mut(&fwd)?;
            let p = p.bound_mut().map_err(anyhow::Error::msg)?;
            p.kv_declaration_realized = true;
            let (shadow, bound, shadow_cells) =
                (&mut p.host_shadow, &p.instance.program.bound, &p.cells);
            shadow.advance(bound, shadow_cells);
            crate::probe_fire_record!(submit_probe.shadow_advance_us, began.elapsed());
        }

        pipe_fires
            .lock()
            .unwrap()
            .push_back(PendingOp::Fire(PendingFire {
                completion,
                kv: FireKv::Host(kvtxn.into_inner()),
                rstxn: rstxns,
                ws_guard,
                model,
                engine,
                fwd_rep,
                instance_id,
                cells,
                failure: pipeline_failure,
            }));
        crate::probe_fire_record!(submit_probe.total_us, submit_clock.elapsed());
        crate::probe_fire_count!(submit_probe.submits);
        Ok(Ok(()))
    }
}

/// Frame submission: exactly `model.frame-size()` ordered slots; slot i
/// executes in wave i; `none` is a no-op. At k = 1 this is identical to a
/// per-pass submit; at k > 1 the frame validates structurally, then prepares
/// and enqueues each slot in order under one frame stamp.
pub async fn submit_frame<C: FireContext>(
    ctx: &mut C,
    this: Resource<Pipeline>,
    slot_reps: Vec<Option<u32>>,
) -> Anyhow<Result<(), String>> {
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
            // Mid-frame failure: fires already submitted stand and execute
            // as a truncated frame; tell the scheduler how many exist so it
            // can still seal (else the wait-all gate holds the fleet on a
            // frame that can never complete).
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

/// `forward.park()`: leave the frame wait-set on `this` until it submits
/// again. Consumes a frame seq so the exit orders against this pipeline's own
/// submits, which is what makes it legal to park with fires still
/// outstanding. Silent no-op when there is no wait-set to leave.
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

/// Static admission for a frame: walks the frame's steps in slot order and
/// proves, against the channels' declared capacities, that no step can meet
/// a gate the device would refuse. Three classes, three proofs:
///
/// ```text
/// device-only ring   occupancy in SLOT ORDER: reserved backlog plus this
///                    frame's in-order net growth fits the capacity
/// host-writer        every cell a consuming step drains is already staged
///                    (frames execute uninterrupted; no mid-frame put lands)
/// reader ring        worst-case pressure — reserved by accepted unsettled
///                    fires, minus host-consumed, plus this frame's writes —
///                    fits the capacity
/// ```
fn validate_frame<C: FireContext>(
    ctx: &mut C,
    k: usize,
    fired: &[(u32, u32)],
) -> Anyhow<Result<(), String>> {
    // Separated from the resource lookup so it can be tested against
    // channels rather than a wasm store.
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

/// One frame slot's channels and what its pass does to each: `(consume, publish)`.
struct SlotAccess {
    cells: BoundCells,
    accesses: Vec<(bool, bool)>,
}

/// The proof [`validate_frame`] gathers for; see its doc for the three
/// classes.
fn prove_frame_admissible(k: usize, slots: &[SlotAccess]) -> Result<(), String> {
    /// One channel's part in this frame: how many steps take from it and how
    /// many put into it.
    struct ChannelUse {
        cell: Arc<Mutex<crate::pipeline::channel::ChannelCell>>,
        consumes: usize,
        publishes: usize,
    }
    /// A device-only ring's running occupancy as the walk crosses the frame.
    struct DeviceRingUse {
        global_id: u64,
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

            // Device-only ring occupancy, walked in slot order: the device
            // publish gate admits a publish only while occupancy stays below
            // capacity (+1 on same-fire consume credit). Seeded descriptor
            // channels are exempt.
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
                    // Host-writer, staged class: each consuming fire drains
                    // one host cell, and every one must already exist (a
                    // frame executes uninterrupted).
                    let available = cell.writer_available_cells();
                    if available < entry.consumes as u64 {
                        return Err(format!(
                            "pipeline: channel {}: frame consumes {} host-writer \
                             cell(s) but only {available} are staged — stage every \
                             per-fire input before submitting the frame",
                            cell.global_id, entry.consumes
                        ));
                    }
                } else if entry.publishes == 0 && entry.consumes == 0 {
                    // Host-writer, latest-value class: a control word the
                    // program only reads. One committed cell suffices.
                    if !cell.has_committed_front() {
                        return Err(format!(
                            "pipeline: channel {}: latest-value control word has \
                             no committed cell at frame submit",
                            cell.global_id
                        ));
                    }
                }
                // publishes > 0: device-advanced, the program carries an
                // advance rule for it, so the host stages nothing.
            }
            Some(HostRole::Reader) if entry.publishes > 0 => {
                // Reader ring, worst case: cells reserved by accepted
                // unsettled fires, minus host-consumed, plus this frame's
                // writes, must fit the channel's declared capacity.
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
            _ => {
                // Device-only rings are proved in the slot-order occupancy
                // walk above; seeded descriptor channels are governed by the
                // seed protocol (`SeedAlreadyStaged` guards mutation).
            }
        }
    }
    Ok(())
}

/// Compaction (lazy KV GC): move `n` token KV cells within `ws`, all layers,
/// from (`src_page_ids[i]`, `src_tok_idx[i]`) -> (`dst_page_ids[i]`,
/// `dst_tok_idx[i]`). Submitted in pipeline order and awaited here, so no
/// separate pending-move lifetime or recycle epoch exists.
pub async fn copy_into_inner<C: FireContext>(
    ctx: &mut C,
    this: Resource<Pipeline>,
    ws: Resource<KvWorkingSet>,
    dst_page_ids: Vec<u32>,
    dst_tok_idx: Vec<u32>,
    src_page_ids: Vec<u32>,
    src_tok_idx: Vec<u32>,
) -> Anyhow<Result<(), String>> {
    if ctx.resources().get(&this)?.scope.is_closed() {
        return Ok(Err("pipeline copy_into: pipeline is closed".to_string()));
    }
    let n = dst_page_ids.len();
    if dst_tok_idx.len() != n || src_page_ids.len() != n || src_tok_idx.len() != n {
        return Ok(Err(format!(
            "pipeline copy_into: the four (dst_page,dst_tok,src_page,src_tok) lists \
                 must be equal length (got {}, {}, {}, {})",
            dst_page_ids.len(),
            dst_tok_idx.len(),
            src_page_ids.len(),
            src_tok_idx.len()
        )));
    }
    if n == 0 {
        return Ok(Ok(()));
    }
    let (pipeline_scope, pipe_fires, pipeline_failure) = {
        let pipeline = ctx.resources().get(&this)?;
        (
            pipeline.scope.clone(),
            pipeline.fires.clone(),
            pipeline.failure.clone(),
        )
    };
    drain_pipeline_fires(ctx, &pipe_fires).await?;
    if let Some(error) = pipeline_failed(&pipeline_failure) {
        return Ok(Err(error));
    }
    let ws_handle = ctx.resources().get(&ws)?.clone();

    // The WIT contract passes working-set-relative page indexes (guests
    // never hold physical ids); translate through the flattened table.
    // Translated at enqueue against the committed mapping — a same-WS
    // in-flight fire that remaps these pages is the guest's ordering hazard.
    let (kv_move_dst_pages, kv_move_src_pages): (Vec<u32>, Vec<u32>) = {
        let stores = crate::store::registry::get(ws_handle.model, ws_handle.engine);
        if let Err(owner) = ws_handle.claim_pipeline_scope(&pipeline_scope) {
            return Ok(Err(format!(
                "pipeline: KV working set is already scoped to pipeline {owner:032x}"
            )));
        }
        let translated = crate::store::registry::with_kv_lock(
            &stores.kv,
            "host-other",
            |kv_store| -> anyhow::Result<Result<kv::PageCopies, String>> {
                let (_, flat) = kv_store
                    .flat_table(ws_handle.id)
                    .map_err(|e| anyhow::anyhow!("copy_into flat table: {e}"))?;
                let translate = |ids: &[u32]| -> Result<Vec<u32>, String> {
                    ids.iter()
                        .map(|&i| {
                            flat.get(i as usize).map(|p| p.0).ok_or_else(|| {
                                format!("copy_into: page index {i} beyond the mapped extent")
                            })
                        })
                        .collect()
                };
                match (translate(&dst_page_ids), translate(&src_page_ids)) {
                    (Ok(dst), Ok(src)) => Ok(Ok((dst, src))),
                    (Err(error), _) | (_, Err(error)) => Ok(Err(error)),
                }
            },
        )?;
        match translated {
            Ok(pages) => pages,
            Err(error) => return Ok(Err(error)),
        }
    };

    let cells = kv_move_dst_pages
        .into_iter()
        .zip(dst_tok_idx)
        .zip(kv_move_src_pages.into_iter().zip(src_tok_idx))
        .map(
            |((dst_page_id, dst_token_offset), (src_page_id, src_token_offset))| {
                ::engine::KvMove {
                    dst_page_id,
                    dst_token_offset,
                    src_page_id,
                    src_token_offset,
                }
            },
        )
        .collect::<Vec<_>>();
    let lease = loop {
        match ws_handle.fire_lease() {
            Ok(lease) => break lease,
            Err(crate::store::kv::working_set::FireLeaseError::Fenced) => {
                // The planner is suspending this working set; wait out the
                // eviction and retry. Unlike `settle_and_wait_resident`,
                // this does not settle the process's pipeline tails first.
                let pid = ctx.process_id();
                let Some(planner) = crate::planner::planner() else {
                    return Ok(Err(
                        "pipeline copy_into: working set is suspend-fenced".into()
                    ));
                };
                if let Err(error) = planner.wait_resident(pid).await {
                    return Ok(Err(format!("pipeline copy_into: {error}")));
                }
            }
            Err(error) => return Ok(Err(format!("pipeline copy_into: {error}"))),
        }
    };
    let completion = match crate::scheduler::copy_kv_cells(0, cells).await {
        Ok(completion) => completion,
        Err(e) => return Ok(Err(format!("pipeline copy_into: submit failed: {e:#}"))),
    };
    let result = CopyCompletionGuard {
        completion: Some(completion),
        lease: Some(lease),
        model: ws_handle.model,
        engine: ws_handle.engine,
        ws: ws_handle.id,
        indexes: dst_page_ids,
    }
    .finish()
    .await;
    if let Err(error) = result {
        let reason = format!("pipeline kv-move (copy_into) failed: {error:#}");
        let mut failure = pipeline_failure.lock().unwrap();
        if failure.is_none() {
            *failure = Some(reason.clone());
        }
        return Ok(Err(reason));
    }
    Ok(Ok(()))
}

/// Shared close/drop body. Close is the sole end-of-stream verb: it rejects
/// later submissions and releases the scheduler wait-set immediately. Already
/// settled FIFO entries are finalized opportunistically, but close never
/// waits for an unsettled fire — a full reader ring may require a post-close
/// `take` before the next submitted fire can settle.
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

/// The body behind `kv-working-set.copy-into(on, ...)` (called from
/// `inferlet::host::kv_working_set`): an ordered KV cell move on the pipeline
/// FIFO.
pub async fn working_set_copy_into<C: FireContext>(
    ctx: &mut C,
    ws: Resource<KvWorkingSet>,
    on: Resource<Pipeline>,
    dst_page_ids: Vec<u32>,
    dst_tok_idx: Vec<u32>,
    src_page_ids: Vec<u32>,
    src_tok_idx: Vec<u32>,
) -> Anyhow<Result<(), String>> {
    copy_into_inner(
        ctx,
        on,
        ws,
        dst_page_ids,
        dst_tok_idx,
        src_page_ids,
        src_tok_idx,
    )
    .await
}

/// Pop and finalize pipeline ops whose completions have already settled,
/// without blocking: submit and take/read entry call this so KV/RS
/// transaction pins stay bounded by run-ahead depth while value waiting
/// rides the channel wait slots. Returns whether anything drained.
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

/// Pop and finalize every pending op of one pipeline FIFO, in submit order,
/// under the queue's finalize guard — the full-drain sibling of
/// [`drain_settled`]. `continue_on_error` is the teardown policy (log and
/// keep draining); the strict form propagates the first failure.
pub(crate) async fn finalize_all<C: FireContext>(
    ctx: &mut C,
    fires: &PendingFires,
    continue_on_error: bool,
) -> Anyhow<()> {
    let _finalize_guard = fires.finalize_guard().await;
    loop {
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
            ws_guard: None,
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
        FinalizeAction::ReclaimDeviceGeometry {
            fwd_rep,
            instance_id,
        } => reclaim_device_geometry_grants(ctx, fwd_rep, instance_id),
    }
    drop(ws_guard);
}

/// Finalize an ordinary host-geometry op without a `ResourceTable` borrow.
/// Used by the idle-park drain, before the scheduler freeze barrier.
/// Device-geometry fires are excluded: their lease reclamation lives on the
/// `ForwardPass` resource and still requires `FireContext`.
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
        FinalizeAction::ReclaimDeviceGeometry { .. } => {
            unreachable!("device-geometry fires require FireContext finalization")
        }
    }
    drop(ws_guard);
    Ok(())
}

/// Resolve one in-flight fire: await the payload-free callback, finalize the
/// KV/RS txns, and expose the release-published mirror tails. Values remain
/// in engine-owned channel memory until `channel.take` or `channel.read`.
async fn finalize_fire_await(fire: PendingFire) -> Anyhow<FinalizeOutcome> {
    let PendingFire {
        completion,
        kv,
        rstxn,
        ws_guard,
        model,
        engine,
        fwd_rep,
        instance_id,
        cells,
        failure,
    } = fire;
    let device_geometry = matches!(&kv, FireKv::DeviceGeom { .. });
    let prior_failure = failure.lock().unwrap().clone();
    let result = completion.await;
    let success = result.is_ok() && prior_failure.is_none();

    let (kv_failure, rs_failure) = {
        let stores = crate::store::registry::get(model, engine);
        // RS transactions have no Drop rollback: settle them before the only
        // await below so process cancellation cannot leak their slots.
        // Settlement doesn't depend on `success` — the mapping is already
        // published.
        let rs_failure: Option<String> = {
            // `into_inner` disarms the guard, so its Drop below is a no-op and
            // cannot re-enter the lock this scope holds. Had `completion.await`
            // above been cancelled instead, the guard would have settled on the
            // way out — which is the whole reason it is carried this far.
            let txn = rstxn.into_inner();
            if txn.is_some() {
                let mut rs_store = stores.rs.lock().unwrap();
                rs::settle(&mut rs_store, txn);
            }
            None
        };
        let kvtxn = match kv {
            FireKv::DeviceGeom { kvtxn } => Some(kvtxn),
            FireKv::Host(kvtxn) => kvtxn,
        };
        let kv_failure = kvtxn.and_then(|kvtxn| {
            crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv_store| {
                kv::finalize(kv_store, kvtxn, success)
                    .err()
                    .map(|error| format!("pipeline: KV finalize failed: {error}"))
            })
        });
        (kv_failure, rs_failure)
    }; // store locks released before the contention drain re-locks pools

    // The fire's sequence retired: recycled slots are allocatable now, and
    // this is the planner's per-fire quiescence event (the fire's lease has
    // just released).
    if let Some(planner) = crate::planner::planner() {
        planner.pages_freed();
    }

    // Values are already visible through the release-published tail words;
    // resolving the fire only classifies success and settles the
    // transactions above.
    let failure_reason = prior_failure
        .or_else(|| {
            result
                .err()
                .map(|error| format!("pipeline: forward failed: {error:#}"))
        })
        .or(kv_failure)
        .or(rs_failure);
    let action = if let Some(reason) = failure_reason {
        FinalizeAction::Fail {
            fwd_rep,
            cells,
            failure,
            reason,
        }
    } else if device_geometry {
        FinalizeAction::ReclaimDeviceGeometry {
            fwd_rep,
            instance_id,
        }
    } else {
        FinalizeAction::None
    };
    Ok(FinalizeOutcome {
        action,
        ws_guard: Some(ws_guard),
    })
}

/// Mark a pass failed (first failure wins). The guest may have dropped
/// the pass handle already — then there is nothing to mark.
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

fn reclaim_pending_device_grant<C: FireContext>(ctx: &mut C, fwd: &Resource<ForwardPass>) {
    if let Ok(pass) = ctx.resources().get_mut(fwd)
        && let Some(devgeo) = pass.devgeo.as_mut()
    {
        devgeo.lease.reclaim_after_fire(&vec![true; devgeo.b]);
    }
}

/// Device-geometry fire: the pass's [B,P] geometry is device-produced and the
/// engine resolves it pre-forward, so the host neither replays the epilogue
/// arithmetic nor projects per-lane KV. The runtime leases `B` fresh physical
/// pages, delivers them via a host-put on the `fresh` channel, submits the
/// fire prebuilt, and runs it ahead onto the pipeline FIFO. Per-fire
/// arena/write txns ride the `PendingFire`; `finalize_op` commits/aborts
/// them and reclaims continuing heirs' unused grants.
async fn fire_device_geometry<C: FireContext>(
    ctx: &mut C,
    this: Resource<Pipeline>,
    fwd: Resource<ForwardPass>,
    frame: Option<crate::scheduler::FrameStamp>,
) -> Anyhow<Result<(), String>> {
    // A device-resolved geometry cannot carry media: its token ids never
    // reach the host, so there is no submitted token list to scan for
    // placeholder runs.
    if !ctx.resources().get(&fwd)?.bindings.media.is_empty() {
        return Ok(Err(
            "pipeline: MediaDeviceGeometry: this pass attached media spans \
             and resolves its token ids on the device, so the host has no \
             token list to scan for their placeholder runs — media rides a \
             host-resolved geometry, where the runs can be checked"
                .to_string(),
        ));
    }
    // Wire each of this pass's channels at this pipeline's FIFO: all passes
    // binding a channel must submit on one pipeline.
    let (pipe_fires, pipeline_failure, pipeline_scope) = {
        let pipeline = ctx.resources().get(&this)?;
        if pipeline.scope.is_closed() {
            return Ok(Err("pipeline: pipeline is closed".to_string()));
        }
        (
            pipeline.fires.clone(),
            pipeline.failure.clone(),
            pipeline.scope.clone(),
        )
    };
    // Non-blocking settlement drain, as in the ordinary submit.
    drain_settled(ctx, Some(&pipe_fires)).await?;
    if let Some(error) = pipeline_failed(&pipeline_failure) {
        return Ok(Err(error));
    }
    if let Err(e) = wire_channels_to_pipeline(ctx, &fwd, &pipe_fires)? {
        return Ok(Err(e));
    }
    // No RS serialization: the mapping publishes at prepare (submission
    // order), so a recurrent-state device-geometry pass runs ahead too.

    let (ws_rep, rs_reps, rs_fold_len) = {
        let pass = ctx.resources().get(&fwd)?;
        // A fold length bound to a channel is the device's from the first
        // fire (a speculative window's accepted count): the host's seed is
        // only the first cell the port reads, so the plan is the window verb
        // and the host never folds the number — as the envelope path states.
        let device_fold_len = pass.instance.program.bound.container.ports.iter().any(|binding| {
            binding.port == eta_ir::registry::Port::RsFoldLen
                && matches!(binding.source, eta_ir::container::PortSource::Channel(_))
        });
        let rs_fold_len = if device_fold_len {
            None
        } else {
            pass.rs_fold_len.clone()
        };
        (pass.kv_ws, pass.rs_ws.clone(), rs_fold_len)
    };
    let ws_res: Resource<KvWorkingSet> = Resource::new_borrow(ws_rep);
    let ws = ctx.resources().get(&ws_res)?.clone();
    let stores = crate::store::registry::get(ws.model, ws.engine);
    let pid = ctx.process_id();
    let quorum_pipeline_id = pipeline_scope.scheduler_id();
    if let Err(owner) = ws.claim_pipeline_scope(&pipeline_scope) {
        return Ok(Err(format!(
            "pipeline: KV working set is already scoped to pipeline {owner:032x}"
        )));
    }
    // Fail-fast on an already-failed pass.
    {
        let p = ctx.resources().get(&fwd)?;
        if let Some(e) = &p.failed {
            return Ok(Err(format!(
                "pipeline: forward-pass failed by an earlier fire: {e}"
            )));
        }
    }

    // Grant B logical slots and size the physical demand; every error path
    // reverts the lease grant through `reclaim_pending_device_grant`.
    // A pool-owned pass resolves its own write targets in-graph, so the host
    // materializes the program's declared writable span instead, to cover
    // every slot the device might pick.
    let pooled = ctx
        .resources()
        .get(&fwd)?
        .devgeo
        .as_ref()
        .expect("fire_device_geometry on a non-device-geometry pass")
        .pooled;
    let writable_span: Option<std::ops::Range<u64>> = {
        let writable = ctx.resources().get(&fwd)?.kv_declaration.writable;
        let reserved = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv_store| {
            kv_store.page_len(ws.id)
        });
        match reserved
            .map_err(|error| error.to_string())
            .and_then(|page_len| writable.resolve(page_len))
        {
            Ok(span) => Some(span),
            Err(error) if pooled => {
                return Ok(Err(format!(
                    "pipeline: pool-owned device geometry: {error}"
                )));
            }
            // Not pooled: a span that won't resolve isn't fatal here — it
            // only costs the containment bound below.
            Err(_) => None,
        }
    };
    let pooled_write_indexes: Vec<u64> = if pooled {
        writable_span
            .clone()
            .expect("a pooled pass returns above when its span will not resolve")
            .collect()
    } else {
        Vec::new()
    };
    let (grant_slots, write_indexes, fresh_dense, devgeo_b, devgeo_split) = {
        let p = ctx.resources().get_mut(&fwd)?;
        let devgeo = p
            .devgeo
            .as_mut()
            .expect("fire_device_geometry on a non-device-geometry pass");
        let fresh_dense = devgeo.fresh_dense;
        let devgeo_b = devgeo.b;
        let devgeo_split = devgeo.qo_indptr.clone();

        // Grant B slots: lease free-list first, then fresh logical reserves.
        // Purely logical, so it runs once (only the physical prepare below
        // retries). A pool-owned pass grants nothing.
        let grant_slots = if devgeo.pooled {
            Vec::new()
        } else {
            crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv_store| {
                devgeo.lease.grant(|| {
                    kv_store
                        .reserve(ws.id, 1)
                        .map(|r| r.start as u32)
                        .unwrap_or(0)
                })
            })
        };
        let mut write_indexes: Vec<u64> = if devgeo.pooled {
            pooled_write_indexes
        } else {
            grant_slots.iter().map(|&slot| u64::from(slot)).collect()
        };
        write_indexes.sort_unstable();
        write_indexes.dedup();
        (grant_slots, write_indexes, fresh_dense, devgeo_b, devgeo_split)
    };

    // Device geometry resolves B request rows in-graph; the zero-valued
    // `qo_indptr` carries only the known row count, since the engine still
    // resolves its values in-graph. A pass whose lanes carry several rows
    // each (a speculative window on a recurrent state) states its seeded
    // split instead: the rows are the host's to plan — the lane's facts word
    // and the recurrent window verb both read them — and the device's to
    // resolve.
    let wide = devgeo_split
        .as_ref()
        .is_some_and(|split| split.windows(2).any(|lane| lane[1] > lane[0] + 1));
    let resolved_qo_indptr = match &devgeo_split {
        Some(split) if wide => split.clone(),
        _ => vec![0; devgeo_b + 1],
    };
    // The recurrent plan reads a row's token count whatever the lane shape:
    // a one-token lane buffers or folds its one row, and a zero-row lane is
    // a fire with no effect, refused by name.
    let rs_qo_indptr = devgeo_split.clone().unwrap_or_else(|| resolved_qo_indptr.clone());
    let rs_ws_ids = match bound_rs_working_set_ids(ctx, ws.model, ws.engine, &rs_reps)? {
        Ok(ids) => ids,
        Err(error) => {
            reclaim_pending_device_grant(ctx, &fwd);
            return Ok(Err(error));
        }
    };
    // Phase B: acquire the one grant (KV pages + RS slots) — the only
    // awaits in this build; nothing physical is held. Phase C: prepare from
    // it; on stale demand recompute both figures and re-acquire, bounded.
    let mut attempts = 0;
    // `_pages` is unused: the device picks which leased page a token lands
    // in, so the host cannot state a last-page-length for it.
    let (ws_guard, _pages, (copy_src, copy_dst), kvtxn, rs_prepared) = loop {
        let kv_demand =
            match crate::store::registry::with_kv_lock(&stores.kv, "host-other", |store| {
                kv::prepare_explicit_demand(store, ws.id, &write_indexes)
            }) {
                Ok(demand) => demand,
                Err(error) => {
                    reclaim_pending_device_grant(ctx, &fwd);
                    return Ok(Err(format!("pipeline: device-geometry demand: {error}")));
                }
            };
        let Ok(kv_demand) = u32::try_from(kv_demand) else {
            reclaim_pending_device_grant(ctx, &fwd);
            return Ok(Err(
                "pipeline: KV demand exceeds the planner ABI".to_string()
            ));
        };
        let rs_plan = match rs_plan_for(&rs_fold_len, &stores, &rs_ws_ids, &rs_qo_indptr) {
            Ok(plan) => plan,
            Err(error) => {
                reclaim_pending_device_grant(ctx, &fwd);
                return Ok(Err(format!("pipeline: recurrent-state mode: {error}")));
            }
        };
        let rs_demand = match rs_slot_demand(&stores, &rs_ws_ids, &rs_plan) {
            Ok(demand) => demand,
            Err(error) => {
                reclaim_pending_device_grant(ctx, &fwd);
                return Ok(Err(error));
            }
        };
        let demand = crate::planner::Demand {
            kv_pages: kv_demand,
            rs_slots: rs_demand,
        };
        let mut grant = match acquire_grant(ctx, quorum_pipeline_id, demand).await {
            Ok(grant) => grant,
            Err(error) => {
                reclaim_pending_device_grant(ctx, &fwd);
                return Ok(Err(error));
            }
        };
        // The suspend seal — see the host path: lease after any park,
        // before the prepare.
        let ws_guard = match ws.fire_lease() {
            Ok(lease) => lease,
            Err(crate::store::kv::working_set::FireLeaseError::Fenced) => {
                drop(grant);
                if let Err(error) = settle_and_wait_resident(ctx).await {
                    reclaim_pending_device_grant(ctx, &fwd);
                    return Ok(Err(error));
                }
                continue;
            }
            Err(error) => {
                reclaim_pending_device_grant(ctx, &fwd);
                return Ok(Err(format!("pipeline: KV working set: {error}")));
            }
        };
        let (pages, copies, _kv_translation, kvtxn) =
            match prepare_explicit_kv_reserved(&stores, &ws, &write_indexes, &mut grant) {
                Ok(prepared) => prepared,
                Err(ReservedError::Stale) if attempts < STALE_DEMAND_ATTEMPTS => {
                    attempts += 1;
                    continue;
                }
                Err(ReservedError::Stale) => {
                    reclaim_pending_device_grant(ctx, &fwd);
                    return Ok(Err(stale_demand_error()));
                }
                Err(ReservedError::Fatal(error)) => {
                    reclaim_pending_device_grant(ctx, &fwd);
                    return Ok(Err(error));
                }
            };
        let kvtxn = KvTxnGuard::new(ws.model, ws.engine, Some(kvtxn));
        match prepare_bound_rs(
            ctx,
            &stores,
            ws.model,
            ws.engine,
            &rs_reps,
            &rs_qo_indptr,
            &pipeline_scope,
            &rs_plan,
            &mut grant,
        ) {
            Ok(Ok(prepared)) => break (ws_guard, pages, copies, kvtxn, prepared),
            Ok(Err(ReservedError::Stale)) if attempts < STALE_DEMAND_ATTEMPTS => {
                attempts += 1;
                // kvtxn's guard aborts the prepared KV write on drop.
                continue;
            }
            Ok(Err(ReservedError::Stale)) => {
                reclaim_pending_device_grant(ctx, &fwd);
                return Ok(Err(stale_demand_error()));
            }
            Ok(Err(ReservedError::Fatal(error))) => {
                reclaim_pending_device_grant(ctx, &fwd);
                record_submit_failure(ctx, &fwd, &pipeline_failure, &error);
                return Ok(Err(error));
            }
            Err(error) => {
                reclaim_pending_device_grant(ctx, &fwd);
                let reason = format!("pipeline: device-geometry RS prepare failed: {error:#}");
                record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
                return Err(error);
            }
        }
    };
    let (rs_copy_src, rs_copy_dst) = rs_prepared.copies.clone();
    let mut rs_prepared = rs_prepared;
    let rstxns = RsTxnsGuard::new(ws.model, ws.engine, rs_prepared.txn.take());

    // Deliver the fresh grant to the program as a direct put on its `fresh`
    // channel. The grants are working-set-relative indexes: the program's
    // in-graph geometry stays logical end-to-end, backed by the prepared
    // write above.
    let (completion, instance_id, scheduler, cells, fwd_rep, accesses) = {
        let p = ctx.resources().get_mut(&fwd)?;
        let bytes: Vec<u8> = grant_slots.iter().flat_map(|s| s.to_le_bytes()).collect();
        // A pool-owned pass resolves its write targets in-graph, so there is no
        // host grant to deliver and its `fresh` channel is never read.
        let fresh_error = if pooled {
            None
        } else {
            match p.cells.get(fresh_dense) {
                Some(cell) => cell.lock().unwrap().put(bytes).err(),
                None => Some(ChannelError::Empty),
            }
        };
        if let Some(error) = fresh_error {
            reclaim_pending_device_grant(ctx, &fwd);
            let reason = format!("pipeline: device-geometry fresh grant put: {error}");
            record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
            return Ok(Err(reason));
        }
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
    debug_assert!(
        kvtxn.mapping_version().is_some(),
        "device-geometry fire always holds a KV transaction"
    );

    // For a device-geometry instance, the engine's descriptor resolver reads
    // the `AttnMask` channel cell on every fire — that is what the class
    // means — so lowering a host-known mask to wire BRLE here would report
    // the same mask twice and the engine refuses the step. Bind the whole
    // pass to the channel instead: one mask, one reader.
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
            reclaim_pending_device_grant(ctx, &fwd);
            let reason = format!("pipeline: device-geometry attention mask: {error}");
            record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
            return Ok(Err(reason));
        }
    };

    // A device-geometry fire states its row split and nothing else: one lane
    // per entry, tokens resolved by the device.
    let mut req = crate::engine::FireRequest {
        boundary_program: true,
        // `Lane::word` and `Lane::slot` are left at their defaults here and
        // stamped below by `stamp_lane_words`/`stamp_lane_slots`, once the
        // mask has lowered and the working set is known.
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
    // A lane of several rows reads out the rows its `readout` port names
    // (all of a verify window's, for the verifier), lane-relative; the port
    // is a seeded, never-put channel, so the host shadow knows it. One token
    // a lane keeps the engine's default, its last (only) row.
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
        reclaim_pending_device_grant(ctx, &fwd);
        let reason = format!("pipeline: device-geometry attention mask: {error}");
        record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
        return Ok(Err(reason));
    }
    // The program's facts ride every lane here as they do on the host path:
    // a program that materializes a draft-head rectangle is a drafting lane
    // (the draft arm runs and rewrites its plane every fire — a window that
    // read a stale plane would replay the prefill's drafts for the whole
    // generation), and one that materializes the score rectangle captures.
    {
        let program = &ctx.resources().get(&fwd)?.instance.program;
        let (drafts, captures) = (program.reads_mtp_logits, program.reads_attn_score);
        for lane in &mut req.lanes {
            lane.drafts = drafts;
            lane.captures_scores = captures;
        }
    }
    // `false` is the only word this path can state: token ids are resolved
    // on the device, so there is no host-side token list to scan (this entry
    // point already refused a media binding).
    stamp_lane_words(&mut req, fire_wide_mask, false);
    // A device-geometry fire resolves its row split on the device but its
    // seats here: a seat is which sequence each row group is.
    if let Err(refusal) = stamp_lane_slots(&mut req, &stores, ws.id) {
        reclaim_pending_device_grant(ctx, &fwd);
        record_submit_failure(ctx, &fwd, &pipeline_failure, &refusal);
        return Ok(Err(refusal));
    }
    // A device-geometry lane's `KvDelta::pages` is empty (its pages live in
    // a channel), so the page table crosses via `stamp_lane_translation`.
    if let Err(refusal) = stamp_lane_translation(&mut req, &stores, ws.id) {
        reclaim_pending_device_grant(ctx, &fwd);
        record_submit_failure(ctx, &fwd, &pipeline_failure, &refusal);
        return Ok(Err(refusal));
    }
    // A pooled device-geometry fire states its pages in a channel and picks
    // a subset in-graph, so the engine needs the class stamped explicitly.
    // The non-pooled case keeps class 0 (`Host`): its engine resolves
    // geometry from the device-composed template regardless.
    if ctx
        .resources()
        .get(&fwd)?
        .devgeo
        .as_ref()
        .is_some_and(|devgeo| devgeo.pooled)
    {
        req.geometry = GeometryClass::DeviceGeometry;
    }
    let ticket_reservation = TicketReservation::new(&cells, &accesses);
    ticket_reservation.apply_to(&mut req);

    let (hook_program, lora_program) = {
        let p = ctx.resources().get(&fwd)?;
        let container = &p.instance.program.bound.container;
        (
            container_has_attention_stages(container),
            container_has_lora_sink(container),
        )
    };
    let submit_error = crate::scheduler::submit_prebuilt_tracked_async_with_kv_and_rs_copy_on(
        &scheduler,
        req,
        instance_id,
        pid,
        quorum_pipeline_id,
        completion.clone(),
        copy_src,
        copy_dst,
        rs_copy_src,
        rs_copy_dst,
        frame,
        hook_program,
        lora_program,
    )
    .err()
    .map(|error| format!("{error:#}"));
    if let Some(error) = submit_error {
        // The KV/RS transaction guards roll everything back on return.
        let reason = format!("pipeline: device-geometry submit failed: {error}");
        reclaim_pending_device_grant(ctx, &fwd);
        record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
        return Ok(Err(reason));
    }
    ticket_reservation.commit();
    {
        let p = ctx.resources().get_mut(&fwd)?;
        let p = p.bound_mut().map_err(anyhow::Error::msg)?;
        let crate::pipeline::instance::BoundForwardPass {
            host_shadow,
            instance,
            cells,
            ..
        } = p;
        host_shadow.advance(&instance.program.bound, cells);
    }

    pipe_fires
        .lock()
        .unwrap()
        .push_back(PendingOp::Fire(PendingFire {
            completion,
            kv: FireKv::DeviceGeom {
                kvtxn: kvtxn
                    .into_inner()
                    .expect("device-geometry fire always holds a KV transaction"),
            },
            rstxn: rstxns,
            ws_guard,
            model: ws.model,
            engine: ws.engine,
            fwd_rep,
            instance_id,
            cells,
            failure: pipeline_failure,
        }));
    Ok(Ok(()))
}

/// Point each of a pass's channels at `pipe_fires` (the feeding pipeline's
/// FIFO), enforcing the same-pipeline invariant. Returns `Ok(Err(..))` if a
/// channel is already bound to a different pipeline.
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

/// Device-geometry per-fire page reclaim: read the harvested `w_cont` (`[B]`
/// bool: heir/fork) from its bound mirror, reclaim the continuing heirs'
/// unused fresh page grants, and free those ws slots. No-op otherwise.
fn reclaim_device_geometry_grants<C: FireContext>(ctx: &mut C, fwd_rep: u32, instance_id: u64) {
    let res: Resource<ForwardPass> = Resource::new_borrow(fwd_rep);
    let Ok(p) = ctx.resources().get_mut(&res) else {
        return;
    };
    let Ok(p) = p.bound_mut() else {
        return;
    };
    let Some(devgeo) = p.devgeo.as_mut() else {
        return;
    };
    let Some(cell) = p.cells.get(devgeo.w_cont_dense) else {
        return;
    };
    let w_cont = cell
        .lock()
        .unwrap()
        .latest_reader_value(instance_id)
        .ok()
        .flatten()
        .unwrap_or_default();
    let w_cont: Vec<bool> = w_cont.iter().map(|&byte| byte != 0).collect();
    // Reclaimed grants return to the lease free-list and are re-granted to
    // later fires; the store mapping keeps their committed pages until the
    // working set discards or drops them.
    devgeo.lease.reclaim_after_fire(&w_cont);
}

#[cfg(test)]
mod lifecycle_tests {
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
        let missing_ws: Resource<KvWorkingSet> = Resource::new_borrow(u32::MAX);
        assert_eq!(
            copy_into_inner(
                &mut context,
                Resource::new_borrow(rep),
                missing_ws,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
            .await?,
            Err("pipeline copy_into: pipeline is closed".to_string()),
            "later submissions fail before touching their work resources"
        );
        // Repeated close is idempotent and does not manufacture work.
        pipeline_close(&mut context, Resource::new_borrow(rep)).await?;
        assert!(fires.lock().unwrap().is_empty());

        pipeline_drop(&mut context, pipeline).await?;
        assert!(context.resources.get(&borrowed).is_err());
        Ok(())
    }
}

/// Tests [`prove_frame_admissible`] directly against channels rather than a
/// wasm store.
#[cfg(test)]
mod static_admission_tests {
    use super::*;
    use crate::pipeline::channel::ChannelCell;
    use eta_ir::container::ChannelDecl;
    use eta_ir::types::{Dtype, Shape};

    /// One bound channel: a cell with a role, a capacity and a seeded flag.
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

    /// A slot that does `accesses[i]` to `cells[i]`.
    fn slot(cells: &[Arc<Mutex<ChannelCell>>], accesses: &[(bool, bool)]) -> SlotAccess {
        SlotAccess {
            cells: cells.to_vec(),
            accesses: accesses.to_vec(),
        }
    }

    /// The same shape past the capacity is refused, and the refusal names
    /// the channel, the slot, and the capacity.
    #[test]
    fn a_device_ring_frame_that_overflows_is_refused_by_name() {
        let ring = channel(HostRole::None, 1, false);
        let slots = [
            slot(std::slice::from_ref(&ring), &[(false, true)]),
            slot(std::slice::from_ref(&ring), &[(false, true)]),
        ];
        let refusal = prove_frame_admissible(2, &slots).expect_err("two publishes, capacity one");
        assert!(refusal.contains("device-ring occupancy past capacity 1"), "{refusal}");
        assert!(refusal.contains("frame slot 1"), "{refusal}");
    }

    /// A seeded descriptor channel is exempt: its occupancy belongs to the
    /// seed protocol, not to the reserved-ticket ledger.
    #[test]
    fn a_seeded_descriptor_ring_is_not_walked() {
        let seeded = channel(HostRole::None, 1, true);
        let slots = [
            slot(std::slice::from_ref(&seeded), &[(false, true)]),
            slot(std::slice::from_ref(&seeded), &[(false, true)]),
        ];
        assert_eq!(prove_frame_admissible(2, &slots), Ok(()));
    }

    /// Every host-writer cell a frame drains must already be staged: two
    /// consuming slots against one staged cell is refused at submit.
    #[test]
    fn a_frame_that_drains_more_writer_cells_than_are_staged_is_refused() {
        let writer = channel(HostRole::Writer, 4, false);
        writer
            .lock()
            .unwrap()
            .put(vec![0u8; 4])
            .expect("one staged cell");
        let one = [slot(std::slice::from_ref(&writer), &[(true, false)])];
        assert_eq!(prove_frame_admissible(2, &one), Ok(()));

        let two = [
            slot(std::slice::from_ref(&writer), &[(true, false)]),
            slot(std::slice::from_ref(&writer), &[(true, false)]),
        ];
        let refusal = prove_frame_admissible(2, &two).expect_err("two consumes, one staged");
        assert!(refusal.contains("consumes 2 host-writer cell(s)"), "{refusal}");
        assert!(refusal.contains("only 1 are staged"), "{refusal}");
    }

    /// A latest-value control word — a Writer nobody consumes and nobody
    /// publishes to — needs one committed cell, and says so when it has none.
    #[test]
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
            .put(vec![0u8; 4])
            .expect("the host writes the word");
        assert_eq!(prove_frame_admissible(2, &slots), Ok(()));
    }

    /// The reader ring is sized for the worst case: the guest draining
    /// nothing before the frame executes.
    #[test]
    fn a_reader_ring_too_small_for_the_frames_writes_is_refused() {
        let reader = channel(HostRole::Reader, 1, false);
        let slots = [
            slot(std::slice::from_ref(&reader), &[(false, true)]),
            slot(std::slice::from_ref(&reader), &[(false, true)]),
        ];
        let refusal = prove_frame_admissible(2, &slots).expect_err("two writes, capacity one");
        assert!(refusal.contains("frame would need 2 reader cell(s)"), "{refusal}");
        assert!(refusal.contains("2k-1 = 3"), "{refusal}");

        let roomy = channel(HostRole::Reader, 3, false);
        let ok = [
            slot(std::slice::from_ref(&roomy), &[(false, true)]),
            slot(std::slice::from_ref(&roomy), &[(false, true)]),
        ];
        assert_eq!(prove_frame_admissible(2, &ok), Ok(()));
    }

    /// A chained slot is admitted: slot 0 publishes a device-only descriptor
    /// ring and slot 1 consumes it.
    #[test]
    fn a_slot_that_consumes_what_an_earlier_slot_published_is_admitted() {
        let chained = channel(HostRole::None, 2, false);
        let slots = [
            slot(std::slice::from_ref(&chained), &[(false, true)]),
            slot(std::slice::from_ref(&chained), &[(true, false)]),
        ];
        assert_eq!(prove_frame_admissible(2, &slots), Ok(()));
    }
}
