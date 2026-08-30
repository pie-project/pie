//! One fire: prepare → run-ahead submit → finalize/poison — the non-glue
//! run-ahead engine (the WIT host glue lives one layer up, in the
//! `inferlet::host` forward/pipeline modules).
//!
//! **Run-ahead** (overview §3): pure-attention `pipeline.submit` does not block.
//! RS-bound submission first finalizes prior FIFO operations so folded-state
//! preparation observes the preceding commit. It then prepares the
//! fire (seeds, host puts, KV/RS projection), hands the request to the
//! scheduler, and enqueues a [`PendingFire`] (the payload-free completion + the
//! open KV/RS txns) on the pass — the classic `execute()`/`output()` split
//! (`PendingForward`, Option A) applied to this runtime. Pure-attention fires
//! read a standing WorkingSet translation; RS step t+1 waits for t's committed
//! folded mapping.
//! `channel.take`/`read` also finalize in-flight fires FIFO until the cell
//! fills. A failed fire **poisons** the pass's host-reader channels and fails
//! the pass for further submits.
//!
//! **Layering.** The orchestration functions below (`submit_pass_stamped`,
//! `finalize_op`, `drain_settled`, `wire_channels_to_pipeline`,
//! `fire_device_geometry`, `pipeline_close`/`pipeline_drop`, `copy_into_inner`)
//! need to get/get_mut/delete/push `Resource<Channel>`/`Resource<ForwardPass>`/
//! `Resource<Pipeline>` handles in the WASM component resource table, but they
//! are plain functions generic over [`FireContext`] — a narrow trait this
//! module defines (naming only the external `wasmtime::component::
//! ResourceTable` leaf type, never `ProcessCtx`/`inferlet`). `inferlet::host`
//! (L4) implements `FireContext` for `ProcessCtx` and its `Host*` impls call
//! these functions with `self`. This keeps `pipeline/` strictly below
//! `inferlet/` in the layering while the fire engine still owns every bit of
//! the algorithm — see [`FireContext`]'s doc for the design rationale.

/// tart (0.3 re-port step 1): whether the bound container carries
/// attention-stage programs — the fire planner's hook divergence fact.
fn container_has_attention_stages(container: &eta_ir::container::TraceContainer) -> bool {
    use eta_ir::registry::Stage;
    container
        .stages
        .iter()
        .any(|s| matches!(s.stage, Stage::OnAttnProj | Stage::OnAttn))
}

// `container_writes_page_mask` stood here: does the pass write the
// `attn_page_mask` sink? Its one consumer was `LaunchGrouping`, which kept
// such a fire out of a group that carried a multi-token row because the
// deleted driver's page-list substitution needed the paged decode path. The
// palo shells have no such path and no such refusal (grep: `attn_page_mask`
// appears in engine-metal only as a capability bit it answers `false` to), so
// the question has no asker. Deleted with the rule (alto E).

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

/// A reserved-path preparation error. `Stale` means the demand grew between
/// its phase-A computation and this prepare (a peer lane touched the same
/// working set while the requester awaited its grant): nothing was consumed;
/// the caller recomputes both demands and re-acquires, bounded. `Fatal` is a
/// real preparation failure.
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

/// Abort-on-drop guard for a fire's prepared KV transaction: armed from
/// preparation until the fire is enqueued on the pipeline FIFO (from there
/// [`PendingFire`] owns commit/abort). Drop settles the write as failed, so
/// error returns between prepare and enqueue need no hand-written rollback.
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

/// Settle-on-drop guard for a fire's published RS transaction (which has no
/// store-side Drop of its own). Same protocol as [`KvTxnGuard`]: the mapping
/// is already authoritative, so dropping only releases the in-flight hold.
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

/// Acquire this fire's grant (KV pages and RS slots together — a fire never
/// half-succeeds) from the residency planner. Zero demand yields an empty
/// grant so the build path stays uniform — every fire shape goes through
/// prefix lending against one owned grant.
///
/// Uncontended (no waiters, everyone resident) this is two free-list pops
/// with no planner lock. Under pressure the ask parks FCFS at the process's
/// spawn position; the parked task holds no lease, no pins, and no open
/// transaction, so the planner is free to evict around (or through) it.
/// On [`crate::planner::Acquired::Yield`] the process was chosen for
/// eviction (or is already out of the set): settle its tail HERE — the ask
/// holds no lease, no pins, and no open transaction, so this is the one
/// safe point — then wait out the eviction and re-ask.
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

/// Back off for THIS process's eviction: settle its pipeline tails (the
/// parked task must hold no pins, and those finalizations release the fire
/// leases the eviction's quiescence waits on), then wait out the eviction.
/// `wait_resident` re-posts the process-wide leave before parking (the
/// lane-resurrection seal wedge, CONTENTION_FOLLOWUP.md §15.2). One owner
/// for the back-off protocol — every fire-submit park site calls this.
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
/// lowering needs.
///
/// `fold-len` says where each request's folded boundary lands, counted from
/// the current boundary over `[buffer | this fire's tokens]`. That single
/// scalar subsumes the three fold-mode methods this replaced:
///
/// | `fold_len`   | buffer    | plan                       |
/// |--------------|-----------|----------------------------|
/// | `> 0`        | empty     | `Fold` — the fast path     |
/// | `0`          | empty     | `Buffer` — open a chunk    |
/// | `0 < n <= B` | non-empty | `FoldBuffered` — commit    |
///
/// `fold-len` is CLAMPED to `B + T`, so "fold everything" is the fire-invariant
/// constant `u32::MAX` — the SDK's default — rather than a value the guest
/// would have to recompute from each fire's token count.
///
/// Every remaining position needs the SAME missing primitive: a recurrence
/// that starts from `folded ⊕ replay(buffer)` rather than from `folded`. The
/// engine has no such read path (see
/// `.wiki/designs/linear-state-programming-model.md` §2), so all of them are
/// refused:
///
/// - `n == B + T` with `B > 0` — the fast path, "fold everything I have".
/// - `B < n < B + T` — a boundary inside this fire's own new tokens. The
///   extended layout describes it fine; the KERNELS do not, because
///   `commit_len` truncates the sequence rather than merely moving the state
///   snapshot, so the tokens past the boundary would get no outputs.
/// - `n == 0` with `B > 0` — appending a second chunk. This one is quiet: the
///   fire happily emits logits computed as though the buffer were empty.
///   (Both of the above now RUN; only the interior boundary is still refused.)
///
/// They are errors rather than silent approximations because approximating a
/// fold in either direction is unrecoverable: folding early destroys tokens
/// the guest still wanted, and folding late silently drops them from the
/// context.
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
    // Buffer occupancy in TOKENS, exactly. This used to be a page-granular
    // upper bound (`buffer_size() * page_tokens`), which is wrong in the one
    // way that matters: you must reserve a page BEFORE you can buffer into it,
    // so a freshly allocated, genuinely empty buffer reported a full page of
    // occupancy and the legal "append onto an empty buffer" fire was refused
    // as an append onto a non-empty one. The store now publishes the written
    // span, so `buffer_tokens` is exact.
    let mut buffered: Vec<u32> = {
        let store = stores.rs.lock().unwrap();
        let mut out = Vec::with_capacity(rows);
        for (row, id) in ids.iter().enumerate() {
            match store.buffer_tokens(*id) {
                Ok(tokens) => out.push(tokens),
                // A working set whose last fold had a device-resident length
                // has no exact occupancy any more, and classifying a fire
                // against a bound would either replay absorbed tokens (a
                // double fold) or drop live ones. The store says so; say it
                // back with the row that tripped it.
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

    // The fold length lives on device: only the engine will ever resolve it.
    // The host can still PLAN, because the engine CLAMPS the resolved count to
    // the bound the host publishes — the row's whole live buffer `b`. That
    // clamp is what makes the dispatch knowable in advance: every value the
    // device can name lies in `[1, b]`, and every value in `[1, b]` is the
    // same `FoldBuffered` replay. The fire's own new tokens are irrelevant to
    // the choice for the same reason they are irrelevant to any commit — the
    // linear layers do not compute them, they replay the slabs.
    //
    // The price is that the boundary can never land INSIDE this fire's own new
    // tokens under a device-resident length, because the clamp forbids it.
    // That is the interior-boundary case, which is refused for host-known
    // lengths too, so nothing is lost here that is available elsewhere.
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

    // Classify each row independently, then decide what the PASS can be.
    // `Fold` and `Buffer` mix freely: both run this fire's own tokens through
    // the full stack over the extended layout, and they differ only in whether
    // the row's recurrence persists — which the engine now expresses as a
    // per-row mask rather than a per-pass flag. `Commit` does not mix with
    // anything: it replays activations out of the slabs instead of computing
    // them, which is a different dispatch, not a different flag.
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
            // The guest said, in the only way that cannot be confused with
            // anything else, "I have nothing to compute; only move the
            // boundary". A row spanning no tokens is a PURE REPLAY: the
            // linear layers gather the buffered prefix [0, n) out of the
            // slabs and return before the output projection.
            //
            // This used to be inferred from `n <= b`, which is an incidental
            // property of a commit rather than a statement of intent -- and
            // it is exactly the condition a fire folding BEHIND its own new
            // tokens satisfies while meaning the opposite. Reading the
            // emptiness of the row directly frees `n <= b` to mean what it
            // says.
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
            // A pure append. Still the extended layout when b > 0, but the
            // folded boundary does not move.
            Position::Buffer
        } else if n == b + t {
            // The fold takes the WHOLE row. With nothing buffered that is the
            // plain in-forward advance and no buffer is involved at all; with
            // a non-empty buffer it is the same thing over the extended
            // `[b | t]` layout, and since the boundary is the last extended
            // token the ordinary end-of-sequence state writeback lands exactly
            // on it -- no `commit_len`, no kernel work.
            if b == 0 {
                Position::Fold
            } else {
                Position::Buffer
            }
        } else {
            // Everything else runs over the extended `[b | t]` layout, which
            // IS the row's buffer token space. The engine replays the b
            // buffered tokens ahead of the t new ones (every recurrence
            // initializes from `recurrent_state[slot]`, the state at F),
            // scatters all t new ones into the buffer, and snapshots the
            // recurrent state at extended token n. So one shape covers the
            // append (n == 0), the fold through a non-empty buffer
            // (n == b + t), the boundary landing strictly INSIDE this fire's
            // own new tokens (b < n < b + t), and the boundary landing BEHIND
            // them (n < b) -- folding a previous window's accepted prefix in
            // the very fire that writes the next one.
            //
            // The interior case is not `commit_len`: that TRUNCATES the
            // sequence, and a fire folding at an interior boundary still owes
            // logits for the tokens past it. The engine cuts the row in two
            // instead and runs the recurrence twice on one stream -- the head
            // persisting its end-of-sequence state onto the boundary, the tail
            // continuing from that state to produce the remaining outputs
            // without moving it again.
            Position::Buffer
        };
        kinds.push(here);
    }

    // A pure commit cannot share a fire. Its rows are not computed at all --
    // the linear layers gather already-buffered activations and return before
    // the output projection -- so there is no per-row switch that would let a
    // computing row ride along.
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
        // Mixed, or uniformly buffered. Either way the pass runs the buffered
        // shape; an `in_forward` row simply owns no buffer inside it.
        Position::Buffer
    };
    let in_forward: Vec<bool> = kinds.iter().map(|k| *k == Position::Fold).collect();
    for (row, forward) in in_forward.iter().enumerate() {
        if *forward {
            // Its boundary moves through its OWN tokens, not through a buffer,
            // so it carries no fold length: `validate_fold` measures against
            // buffered capacity, which this row does not have.
            fold_tokens[row] = 0;
            buffered[row] = 0;
            row_tokens[row] = 0;
        }
    }

    Ok(match pass {
        Position::Fold => rs::RsPlan::Fold,
        // Each row opens at its own exact occupancy. The planner still refuses
        // a non-zero one above until the recurrence can read the buffer, but
        // the plan carries the truth either way, so relaxing that refusal is a
        // one-line change rather than a re-derivation.
        Position::Buffer => rs::RsPlan::Buffer {
            start_tokens: buffered,
            row_tokens,
            fold_tokens,
            in_forward,
        },
        Position::Commit => rs::RsPlan::FoldBuffered {
            fold_len_is_device: false,
            // `fold_tokens`, not the raw `fold_len`: the WIT promises the
            // length is CLAMPED to the tail, which is what makes "fold
            // everything" a fire-invariant `u32::MAX`. Under the old rule a
            // commit was selected by `fold_len <= buffered`, so the raw value
            // was already clamped by construction and the distinction never
            // showed. A row that carries no tokens is a commit whatever its
            // fold length says, so an unclamped `u32::MAX` would now reach
            // `validate_fold` and be REFUSED as exceeding capacity.
            tokens: fold_tokens,
        },
    })
}

/// Drop the SYNTHESIZED read-out rows from a fire that replays the BUFFER.
///
/// A `FoldBuffered` fire scans activations that were computed by an earlier
/// fire and are already sitting in buffer slabs. It exists only to move the
/// folded boundary, so it returns from the linear layers before the output
/// projection and produces no logits; the engine refuses one that declares
/// sample rows. But an absent `readout` binding does not mean "sample nothing"
/// — it means "sample each lane's last row" — and an empty readout channel has
/// no expressible shape. So a guest that simply committed got a sample row it
/// never asked for and a fire that could not run.
///
/// **Only `FoldBuffered`.** A plain `RsPlan::Fold` also advances the folded
/// state, but it does so over the fire's OWN new tokens, running them through
/// the full stack exactly as a prefill does — because that is what a prefill
/// on a linear model IS. Its logits are ordinary and required. Suppressing
/// them left `sampled_rows` at zero and the fused epilogue failed to launch
/// with a zero extent, which is to say: every prefill on a linear model. The
/// engine draws this line correctly already — its `rs_is_fold` is
/// `mode == BufferFold`, nothing wider — so this must match it and not the
/// looser "does this fire fold at all".
///
/// Only the default is dropped. An EXPLICIT readout on a buffered fold still
/// reaches the engine and is still refused, loudly: asking for logits that
/// cannot exist is a guest bug worth reporting, not one worth papering over.
fn suppress_defaulted_readout_for_fold(
    req: &mut crate::engine::FireRequest,
    readout_defaulted: bool,
    plan: &rs::RsPlan,
) {
    let replays_buffer = matches!(plan, rs::RsPlan::FoldBuffered { .. });
    if !replays_buffer || !readout_defaulted {
        return;
    }
    // `Readout::None` is the contract's way of saying "this lane runs for its
    // cache writes alone", which is exactly what a buffered fold does. The
    // wire form had to say it by clearing an index vector and zeroing a CSR,
    // and could not tell that apart from a lane whose readout was empty for
    // some other reason.
    for lane in &mut req.lanes {
        lane.readout = crate::engine::Readout::None;
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

/// Prepare the host-projected KV write from the fire's grant, all under one
/// store lock: re-check the demand (the staleness gate), realize the
/// declaration, and back the writable frontier — draining exactly the
/// consumed prefix from the grant. On `Stale` nothing was consumed. The
/// grant's Drop returns whatever this did not consume; nothing is ever
/// hand-released.
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

/// A pipeline FIFO entry: a forward FIRE holding its ordered slot on the
/// stream; `take`/`read` drain entries in submit order. (KV cell moves are
/// awaited inline by `copy_into_inner` and never enter the FIFO.)
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

/// Test-only inert FIFO entry: enough to make a pass's shared fires queue
/// non-empty for `ForwardPass::can_close_native_on_drop`/`Drop` tests
/// (`pipeline::instance`'s `native_cleanup` test module), without needing a
/// live completion. `PendingFire`'s fields are private
/// to this module, so cross-module tests construct a stub through here
/// rather than reaching in directly.
#[cfg(test)]
pub(crate) fn test_pending_op_stub() -> PendingOp {
    PendingOp::TestStub
}

/// The open KV/arena transaction(s) one in-flight fire holds until it resolves.
/// Two shapes: the ordinary single-seq / MTP projection ([`kv`]), or a
/// device-geometry fire whose KV the engine resolves+writes itself (B2's
/// explicit-KV path) — the runtime only pins the [`lease::PageLease`]-granted
/// physical pages for the fire, released at finalize (per-fire arena txn; the
/// plan's "pin float bounded by run-ahead depth × B, riding the per-fire arena
/// txns").
enum FireKv {
    Host(Option<kv::KvTxn>),
    /// A device-geometry fire's prepared write over the lease-granted slots
    /// (B2's explicit-KV path): same commit/abort protocol, no host
    /// projection.
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
    rstxn: Option<rs::RsTxn>,
    ws_guard: KvFireLease,
    model: usize,
    engine: usize,
    /// The owning pass, to fail it on a fire error (rep — the guest may have
    /// dropped the handle; failure marking is then moot).
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

    // Resolution + model/engine validation live in the phase-A resolver;
    // only the scope claim is prepare-time work.
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
        // Staleness gate under the same lock as the prepare: if the demand
        // grew while the requester awaited its grant, nothing is consumed
        // and the caller re-acquires.
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

/// Drain EVERY in-flight fire on this pipeline to settlement.
///
/// This is not an RS ordering rule (RS mappings publish at prepare, in
/// submission order, so a recurrent-state fire never needs to wait for its
/// predecessors). It is the host-side ordering seam for out-of-band ops that
/// read committed physical ids and then act on them off the fire path —
/// `copy_into`, whose page translation must not race a same-WS CoW rebase.
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
        // Predecessor fires retire with their frames regardless of this
        // process's residency, so a plain await cannot deadlock the planner.
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

/// **THE LANE WORD, STAMPED ONCE THE LANE IS FINISHED** (`palo B-word`).
///
/// A lane's word is its fact bits, and `engine::fire::compose` turns it into
/// a class and therefore into the row WINDOW every guarded node runs over
/// (palo design §0, decision 18). Both lane-construction sites left it at
/// zero — the all-false class — so every decode lane went out composed as a
/// prefill one; the runtime could not do better while `model::catalog()` shipped
/// three columns, because a plan's `Guard::Fact(bit)` numbers its bits and no
/// reader outside a family's own module can say which one `qo_one` is. The
/// catalog carries a `ClassifyFn` now and `crate::model::Model::word` calls
/// it; nothing here reads a bit.
///
/// **WHY HERE AND NOT IN `ReqGeometry::lanes`.** A lane's word depends on its
/// mask, and a lane does not have its mask until the fire's mask has lowered:
/// `FireAttnMask::apply_to` is what cuts the mask CSR onto the lanes. So the
/// two facts the runtime states — how many rows a lane has, and whether it
/// carries a mask of its own — are only both true at one point in each path,
/// and that is where this is called.
///
/// `fire_wide_mask` is the device-resident case, which has no `Lane::mask` to
/// read: the engine's descriptor resolver reads the `AttnMask` channel cell on
/// every fire, so the mask is on the FIRE and covers every lane in it. A model
/// that splits on `masked` (gemma) would otherwise run those rows through the
/// arm for lanes that have no mask.
///
/// **AND IT IS THE COLLAPSE §0 WARNS ABOUT, STATED RATHER THAN HIDDEN.** One
/// dense mask in one channel cell, shared by every lane of the pass, is a
/// per-fire reading of a per-LANE axis — the runtime cannot do better here
/// because the device value it is reading has no per-lane state to cut
/// (`geometry::detect_pooled_device_geometry`'s ruling). What it must not do
/// is let that reading reach a masked arm with nothing behind it: the word
/// says `masked` and no `Lane::mask` follows, so the engine would send the
/// arm at a slab the fire never staged. The CUDA shell refuses exactly that
/// pairing by name — `Fault::MaskWord`, asked per lane against the class the
/// word resolved to — so the narrowing costs a refusal and never an answer.
///
/// **AND THE SIXTH FACT IS THE SUBMISSION'S, NOT THE LANE'S** (multimodal §15,
/// media-door §4). `media` says the submission this fire came from attached
/// spans — one reading of one `forward-pass.media` call, stamped on every lane
/// the submission fires, because the pass is one pass and its images are its
/// images whichever row groups it splits into. It joins here rather than
/// earlier for the reason `drafts` and `captures_scores` do: it imposes no
/// ordering of its own, so it takes the instant that already exists.
///
/// §15's constraint is that this argument and the verb that makes it
/// reachable land TOGETHER. A lane whose word said `false` while its
/// submission carried images would compose as the text-only class — and both
/// `qwen_3` and `gemma_4` guard their embed merge on `Facts::media`, so the
/// patch rows would have no node to land in and the pass would answer
/// fluently about an image it never saw. That is not a failure any test
/// downstream of here catches, which is why it is a constraint and not a
/// follow-up.
fn stamp_lane_words(
    req: &mut crate::engine::FireRequest,
    fire_wide_mask: bool,
    carries_media: bool,
) {
    let model = crate::model::model();
    for lane in &mut req.lanes {
        let rows = u32::try_from(lane.tokens.len()).unwrap_or(u32::MAX);
        // THE FACTS AFTER THE FIRST ARE READ OFF THE LANE ITSELF, and that is
        // the whole reason they are stamped here rather than anywhere
        // earlier: `word` and the intents it names have to be ONE READING OF
        // ONE LANE, or the shell refuses the fire (`Fault::AdapterWord`,
        // `Fault::DraftWord`, `Fault::ScoreWord`). Whoever put them on the
        // lane — the working set, `crate::engine::fire` — has already run by
        // now, so this asks the lane rather than re-deriving the answers from
        // the request they came from.
        //
        // **THE TWO EXPORT AXES NEEDED NO NEW INSTANT** (palo C3b/C4b). The
        // mask is why this function exists at all: a lane does not have its
        // mask until `FireAttnMask::apply_to` has cut the fire's, so a word
        // stated earlier would be stated before its inputs. `drafts` and
        // `captures_scores` are not like that — they are on the lane from the
        // moment the lane is built, and nothing downstream can change them —
        // so they impose no ordering of their own and simply join the reading
        // that already happens at the latest of the five instants.
        lane.word = model.word(
            rows,
            lane.mask.is_some() || fire_wide_mask,
            lane.adapter.is_some(),
            lane.drafts,
            lane.captures_scores,
            carries_media,
        );
    }
}

/// Seat this fire's lanes: `Lane::slot`, from the working set that owns it.
///
/// **THE SLOT HAD NO OWNER, AND THAT IS WHAT A MULTI-LANE FIRE FOUND OUT**
/// (`palo` build log 28's headline). Both paths built their lanes with the
/// field at zero — `ReqGeometry::lanes` under a comment saying "the caller
/// stamps it", `fire_device_geometry` through `Lane::default()` — and the
/// caller it named did not exist. A solo fire is one lane and zero is a
/// perfectly good seat for it, so every gate in the tree passed; the batcher
/// concatenates the lanes of every member of a step (`scheduler::batch`), so
/// the second concurrent guest in one wave made the fire two lanes both
/// seated at zero and `Step::validate` refused the batch by name.
/// Concurrent multi-lane serving was broken for exactly as long as nothing
/// covered it.
///
/// **WHO OWNS A SLOT.** It is the sequence's seat in the shell's pools — the
/// kv block a shell-owned page table hands it, the recurrent bank row
/// `Pools::clear` zeroes on the fire where `held == 0`, the id
/// `slot_ids[lane]` a linear-attention scan indexes with. The runtime's
/// per-sequence identity is the KV WORKING SET: it is one sequence's page
/// table, minted at create/fork/slice and released with its last handle. So
/// the working set owns the seat, [`crate::store::seat`] is the book it owns
/// it in, and this is where the book's answer meets the lanes.
///
/// One seat per LANE, because a request is lanes plural: a beam fires B row
/// groups against one page table and each of those rows is a sequence as far
/// as the pools are concerned. The book keeps the run, so lane `i` of every
/// fire of a working set sits in the same seat — which is what a recurrent
/// bank row depends on, and why a seat cannot be handed out per fire.
///
/// # Errors
///
/// The refusal, ready to return, when this fire would seat more sequences
/// than the deployment's pools hold (`Budgets::slots`, which the engine
/// advertises as `PoolFacts::state_slots`). Named here, with both numbers,
/// rather than reaching the shell and coming back as a `Fault::Ceiling`
/// naming a lane index.
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

/// **REWRITE EVERY LANE'S PAGE TABLE FROM WORKING-SET INDEXES TO POOL PAGE
/// IDS**, which is the only spelling [`KvDelta::pages`] has.
///
/// A guest states its `pages` channel in WorkingSet-RELATIVE indexes — it
/// asked for `ws.reserve(n)` and holds `0..n`, and `crate::store::kv` is what
/// maps those onto the pool. [`KvDelta::pages`] is the other thing:
/// `engine::store::kv::geometry_with` pushes each entry it is given STRAIGHT
/// into the page CSR, with no per-slot base and no lookup, because "the ids
/// are the runtime's and the bytes under them are the engine's" (article 8).
/// Between the guest's spelling and the engine's there was no translation, so
/// every lane in the system addressed pool pages `0, 1, …`:
///
/// * **alone it is invisible** — a lane reads back the pages it wrote, so a
///   sequence that never shares a fire is self-consistent whatever ids it used;
/// * **under a HOMOGENEOUS load it is invisible** — colliding lanes write the
///   same prompt and the same continuation, so the bytes they overwrite each
///   other with are the bytes that were there;
/// * **under a heterogeneous one it is a wrong answer** — a fresh sequence
///   prefilling six tokens over pool page 0 walks over the first six positions
///   of every other live sequence's cache, and the lane decoding beside it
///   attends a prompt it never had. Measured: a greedy lane over "The capital
///   of France is" answers " Paris.\nThe capital of France is Paris." alone and
///   " Paris.\nThe following are the results of the following:" with any
///   two-token neighbour co-resident, deterministically, from the fifth token.
///
/// **IT RUNS AFTER THE FIRE'S KV PREPARE.** `flat_table` answers the committed
/// mapping overlaid with the write targets this fire has just reserved, and a
/// first prefill's own pages are in it only then — which is also why this is
/// not folded into [`stamp_lane_slots`], whose instant is before the acquire.
///
/// A page a lane names and the working set has not mapped is a refusal: the
/// alternative is addressing a pool page belonging to somebody else, which is
/// the failure this function exists to end.
///
/// [`KvDelta::pages`]: crate::engine::KvDelta::pages
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
/// pool page backing WorkingSet-relative index `i`.
///
/// **ONE READER, BECAUSE ONE MAPPING** (article 8). [`map_lane_pages`] applies
/// it and [`stamp_lane_translation`] quotes it, and the two must be the same
/// bytes at the same instant or the host path and the device path would be
/// translating through two tables.
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

/// **HAND THE ENGINE THE TABLE, FOR THE ONE CLASS WHOSE PAGE REFERENCES THE
/// RUNTIME CANNOT REACH.**
///
/// [`map_lane_pages`] is the same rule applied one step earlier: for a
/// host-resolved geometry the runtime folds the guest's `Pages` port itself,
/// so it can rewrite the relative indexes into pool ids and ship the RESULT.
/// A device-geometry pass states its pages, its page CSR and its write
/// descriptor in channel cells its own epilogue wrote — `gather(pool_ids, ..)`
/// over a `ws.reserve` grant, which is relative like every other guest's — and
/// the host never reads them. There is nothing to rewrite, so the TABLE
/// crosses instead and the engine applies it where it resolves the cells.
///
/// The mapping still has one owner: this is `kv::build_translation`'s vector,
/// quoted onto the lanes, and the engine may only index it.
///
/// **AT THE SAME INSTANT [`map_lane_pages`] RUNS**, and for its reason: a
/// pooled pass prepares its whole writable span before this point, so the
/// table answered here already covers every cell the device may pick.
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

    /// **AND NOW IT CROSSES, ONTO THE LANE** (alto design §1 article 3,
    /// waves F2a and E).
    ///
    /// These heads and tails were `channel_expected_head` /
    /// `channel_expected_tail` on the old wire plan — a per-fire assertion,
    /// shipped, that a guest program's rings stood where the runtime thought —
    /// and F1 stopped shipping them because nothing on the far side could
    /// check one: the engine gated its own cursors on the host before it
    /// launched anything, so a stated claim had nothing to add.
    ///
    /// An engine with the pull-validate and commit-bump kernels checks them
    /// where the data is, and that is the whole of article 3: the host owns a
    /// prediction and the device owns the truth. F2a stated the reservation
    /// on the REQUEST and had `scheduler::batch` transcribe it onto the
    /// attached lane; wave E stamps it here instead, because this is the one
    /// place that mints the prediction and the one place that knows whether
    /// the instance's channels were ADOPTED — and two spellings of one number
    /// is what article 8 forbids.
    ///
    /// It lands on the request's FIRST lane, which is the lane the batcher's
    /// attachment names: a member that fires three row groups is one bound
    /// instance running one pass with one commit. And it lands only when
    /// every channel was adopted, because an engine with no device half
    /// refuses a stated prediction by name (`Lane::validate_for`) rather than
    /// ignoring it, and a prediction nobody checks is worse than none.
    ///
    /// What the reservation and the LIFO rollback below were always also for
    /// is the OTHER question — how many cells a frame of k fires could need
    /// before any of them settles — and that is runtime bookkeeping,
    /// unchanged.
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
/// advances (the engine's completion callback wakes the reader wait slot
/// directly), or the oldest in-flight pipeline op settles so the caller can
/// drain it. Errors surface poison/closure or a definitively empty channel
/// (no endpoint and nothing in flight: nothing can ever fill the cell).
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
    // palo D0: the guest thread's own submit, phase by phase. Compiles away
    // without `profile-fire` (see `scheduler::probe::HostSubmitProbes`).
    let submit_probe = crate::scheduler::probe::host_submit();
    let submit_clock = crate::scheduler::probe::ProbeClock::start();
    {
        // Device-geometry pass (Track B): the [B,P] geometry is
        // device-produced (the program traces the wire form in-graph) and
        // the engine resolves it pre-forward, so this pass leases physical
        // pages and fires prebuilt — but it RUNS AHEAD like any pass (the
        // FIFO carries it; NOT synchronous like the deleted host-replay beam
        // branch).
        if ctx.resources().get(&fwd)?.devgeo.is_some() {
            return fire_device_geometry(ctx, this, fwd, frame).await;
        }
        // Contention-probe marker: when the guest's WIT call reached the
        // host (vs `hp-acquire` below — the delta is the build preamble,
        // including the settlement drain).
        // W3.1: the PIPELINE owns the in-flight FIFO. Point each of this
        // pass's channels at this pipeline's queue so their `take`/`read`
        // await the right FIFO — enforcing the same-pipeline constraint
        // (§3.4): every pass binding a channel must submit on one pipeline.
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
        // Non-blocking settlement drain (plan §6): resolved fires' KV/RS
        // txns finalize here so arena pins stay bounded by run-ahead depth
        // even when the guest never takes.
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
        // An RS-binding pass needs no extra serialization here: its mapping
        // publishes at prepare, in submission order, so it runs ahead like
        // any other pass.
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
        ) = {
            let p = ctx.resources().get_mut(&fwd)?;
            if let Some(e) = &p.failed {
                return Ok(Err(format!(
                    "pipeline: forward-pass failed by an earlier fire: {e}"
                )));
            }
            // WHICH PORTS THE ENGINE WILL RESOLVE, AND NOTHING ELSE (`palo
            // B3`). A decode-envelope pass carries exactly one value the host
            // cannot fold — the sampled token, which the shadow commits
            // unknown — and the CUDA shell reads it off the ring the previous
            // epilogue wrote. Everything else the pass's epilogue carries is
            // arithmetic over the KV length, and folding it here is what
            // gives the fire a page table: the RUNTIME owns this working set's
            // physical pages, so a submission that stated none would leave
            // the engine deriving a block formula for a pool it does not
            // allocate from.
            //
            // An all-placeholder geometry is the other reading — an engine
            // that resolves the WHOLE envelope from its own page segments,
            // which is what `fire::envelope::compose` did one generation back
            // — and no shell in this workspace owns a page segment to resolve
            // one from. `DecodeEnvelope::template` was that reading written
            // down, and it is deleted rather than kept as an unreachable
            // alternative.
            let device_resolved = if p.decode_envelope.is_some() {
                PortMask::of(&[Port::EmbedTokens])
            } else {
                PortMask::NONE
            };
            let geometry_clock = crate::scheduler::probe::ProbeClock::start();
            let (geometry, attn_mask) = {
                let bound = &p.instance.program.bound;
                let (shadow, shadow_cells) = (&p.host_shadow, &p.cells);
                let mut known = |chan: u32| shadow.fire_value(bound, shadow_cells, chan);
                match geometry::map_geometry_evaluated_with(bound, &mut known, device_resolved) {
                    Ok((geometry, evaluated)) => {
                        // In-band -1 skips are the DEVICE-resolved contract
                        // (rank compaction happens in the compose kernels); a
                        // host-wire fire would embed the sentinel as a real
                        // token. Loud rejection, never silent execution
                        // (RV-12).
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
            (
                geometry,
                p.cells.clone(),
                p.kv_ws,
                p.rs_ws.clone(),
                p.rs_fold_len.clone(),
                p.kv_declaration,
                p.kv_declaration_realized,
                fwd.rep(),
                p.bound_instance.instance_id,
                p.scheduler.clone(),
                attn_mask,
                accesses,
                p.decode_envelope.clone(),
                reads_attn_score,
            )
        };
        let mut req = crate::engine::FireRequest::default();
        let readout_defaulted = geometry.readout_defaulted;
        geometry.apply_to(&mut req);
        // A BOUND PASS *IS* A GUEST PROGRAM AT THE FIRE'S BOUNDARY (`palo
        // B2`, design §9). Every fire that comes through here fires a
        // `BoundForwardPass` — an instance whose channels the engine carved
        // and whose stages it compiled — so the fire carries an attachment
        // for it and the engine runs its pass after the forward, with this
        // lane's logits row bound as the `logits` intrinsic. The requests
        // that do NOT come through here (a prebuilt rider, a geometry
        // lowering under test) leave the flag false and submit exactly the
        // fire they always did.
        req.boundary_program = true;
        // **AND A PROGRAM THAT READS THE SCORES IS A CAPTURING LANE**
        // (`.wiki/alto/attn-score.md` §4, wave S1). The capture bit was
        // hard-`false` on every path in this crate under a note saying the
        // port vocabulary named no capture port — and the note was right and
        // the port was never the answer. The observability contract is that
        // the graph writes and the EPILOGUE reads, so the ask and the read
        // are one act: an attached program that materializes
        // `IntrinsicId::AttnScore` is asking, and one that does not is not.
        //
        // Stamped on every lane of the member, because a member firing three
        // row groups is one bound instance running one pass and its epilogue
        // is pointed at the first lane's block — but the capture arm writes
        // whichever lanes the fire seats, and `Fault::ScoreWord` refuses a
        // lane whose word and whose ask disagree, so the two readings have to
        // be the same reading.
        if p_reads_attn_score {
            for lane in &mut req.lanes {
                lane.captures_scores = true;
            }
        }
        // A CLASS AND NOT A BOOL, and the bool beside it is gone (alto E).
        // `device_resolved_geometry` said the same fact with two values where
        // there are three ways to read a fire's geometry; its only readers
        // were the deleted co-batching rules, and `fire_device_geometry`
        // stamps the third class this path cannot reach.
        req.geometry = if decode_envelope.is_some() {
            GeometryClass::DecodeEnvelope
        } else {
            GeometryClass::Host
        };
        req.single_token_mode = req.lanes.iter().all(|lane| lane.tokens.len() == 1);
        // tart (0.3 re-port step 2): the pass's layer truncation rides
        // every fire; the scheduler's region table carries it to the
        // engine as per-region k.
        req.max_layers = {
            let p = ctx.resources().get(&fwd)?;
            p.max_layers
        };
        // **THE RUN SCAN** (`.wiki/alto/media-door.md` §3, wave MD-A).
        //
        // THE TOKENS ARE THE LEDGER AND THIS IS THE AUDIT. A span entered the
        // sequence as the run `image.tokens()` answered; the handle crossed
        // again beside it through `forward-pass.media` carrying only the
        // payload; nothing said where the two meet. So the host finds the
        // model's reserved placeholder runs in the submitted tokens — a
        // tokenizer never emits that id from text — and matches them to the
        // attached spans IN ORDER, refusing every disagreement by name.
        //
        // It stands HERE, at the first instant both halves are final: the
        // tokens are the lanes' only once `geometry.apply_to` has split them,
        // and the spans are the pass's from the moment `media` was called. A
        // scan any earlier would be a scan of half the fact, and any later
        // would be after the fire had staged.
        let media_spans = ctx.resources().get(&fwd)?.bindings.media.clone();
        let carries_media = !media_spans.is_empty();
        let matched = {
            let lane_tokens: Vec<&[u32]> =
                req.lanes.iter().map(|lane| lane.tokens.as_slice()).collect();
            let scanned = if carries_media {
                crate::pipeline::media::scan(&lane_tokens, &media_spans)
            } else {
                // THE OTHER DIRECTION, and it costs one pass over the tokens
                // on a text-only fire: a run with no span behind it would
                // otherwise embed the pad id as an ordinary token and decode
                // as nonsense nothing named.
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
        // The mask is on the FIRE when it stays device-resident, and there is
        // no `Lane::mask` to read it back off — so the class is read here,
        // before `apply_to` consumes it, and handed to the word stamp.
        let fire_wide_mask = matches!(attn_mask, geometry::FireAttnMask::Device);
        if let Err(error) = attn_mask.apply_to(&mut req) {
            return Ok(Err(format!("pipeline: fire attention mask: {error}")));
        }
        // THE LANES ARE FINISHED, so their words can be stated: rows from the
        // geometry above, mask from the lowering just now (`palo B-word`), and
        // the media fact from the scan above — multimodal §15's constraint,
        // landing in the same commit as the verb that makes it reachable.
        stamp_lane_words(&mut req, fire_wide_mask, carries_media);
        // **AND THE DOOR IS CUT** (media-door §6, wave MD-C). What stood here
        // was `Refusal::EngineDoor` — a typed stop, refusing rather than
        // dropping, because the rows the spans land in did not exist. They
        // exist: `engine::fire::StepMedia` is the contract's parallel slice
        // keyed by lane, `lane_media` answers it directly, and the request
        // carries it to `scheduler::batch`, which rebases each row's lane onto
        // the step it co-batches into.
        //
        // Everything above this line is unchanged and is MD-A's: the scan, its
        // four refusals, and the word stamp. The attachment is the last act
        // and it is a move, not a computation.
        if !matched.is_empty() {
            let lane_rows: Vec<u32> = req
                .lanes
                .iter()
                .map(|lane| u32::try_from(lane.tokens.len()).unwrap_or(u32::MAX))
                .collect();
            req.media = crate::pipeline::media::lane_media(&matched, &lane_rows);
        }
        crate::pipeline::offload::try_encode(&mut req).await;
        // Resource preparation is independent of token position: realize the
        // declaration once, back only its missing frontier, then snapshot the
        // WorkingSet translation.
        let kv_clock = crate::scheduler::probe::ProbeClock::start();
        let ws_res: Resource<KvWorkingSet> = Resource::new_borrow(ws_rep);
        let ws = ctx.resources().get(&ws_res)?.clone();
        let stores = crate::store::registry::get(ws.model, ws.engine);
        // THE LANES' SEATS, and they stand here rather than beside the word
        // stamp for one reason: the seat is the WORKING SET's, and the
        // working set is resolved on the line above. Nothing between the two
        // points touches `Lane::slot`, and everything after this reads a
        // seated fire.
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
        // Scope claim precedes acquisition (symmetric with the
        // device-geometry path).
        if let Err(owner) = ws.claim_pipeline_scope(&pipeline_scope) {
            return Ok(Err(format!(
                "pipeline: KV working set is already scoped to pipeline {owner:032x}"
            )));
        }
        // Phase A: pure demand for both resources, holding nothing. Phase
        // B: acquire the one grant (KV pages + RS slots — a fire never
        // half-succeeds); the acquire awaits are the only awaits in this
        // build, and the requester waits at the safe point with no pins, no
        // lease, no open transaction. Phase C: prepare from the grant; if
        // the demand drifted while waiting (a peer lane touched the same
        // working set), recompute both demands and re-acquire, bounded.
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
            // The lease is the suspend seal: acquired AFTER any park (a
            // parked ask must hold no lease, or the planner could never
            // quiesce it) and BEFORE the prepare (so an eviction either
            // sees this fire's lease and waits it out, or fenced first and
            // this fire backs off to wait out the eviction).
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
            // request order. Their CoW copies ride the scheduler's typed
            // pre-launch state copy so a copy failure rejects this fire
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
        // THE PAGE IDS BECOME THE ENGINE'S HERE, and this is the earliest
        // instant they can: the mapping is only complete once this fire's own
        // write targets are reserved (`map_lane_pages` argues it).
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
                rstxn: rstxns.into_inner(),
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

/// The body behind the interface-level `forward.submit(on, slots)` —
/// Vesuvius frame submission. Exactly `model.frame-size()` ordered slots;
/// slot i executes in wave i; `none` is a no-op. At k = 1 a single-slot frame
/// IS today's per-pass submit (identical semantics — a 1-slot frame runs the
/// same unified FramePolicy path). At k > 1 the frame validates structurally (Section 5 of the
/// design: staged / device-advanced / latest-value host-writer classes,
/// static reader-capacity overflow prevention), then prepares and enqueues
/// each slot in order under one frame stamp.
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
            // Mid-frame failure: the fires already submitted stand and
            // execute as a truncated frame; tell the scheduler how many
            // exist so the frame can still seal. This must cover the host
            // trap path too — returning through `?` without truncating
            // strands the frame arrival-incomplete, and the wait-all gate
            // then holds the whole fleet on a frame that can never
            // complete (CONTENTION_FOLLOWUP §20.8).
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
/// submits — every frame submitted before it must seal first, which is what
/// makes it legal to park with fires still outstanding.
///
/// Silent no-op when there is no wait-set to leave: `k == 1` never batches
/// across pipelines (`submit_frame` bypasses stamping entirely there), and a
/// closed pipeline has already left. The WIT signature returns nothing for
/// the same reason — parking is a statement about the future, and there is no
/// state in which it can fail to be true.
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

/// **STATIC ADMISSION FOR A FRAME** (alto design §1 article 4; survey §7 I8).
///
/// Walks the frame's steps in slot order and proves, against the channels'
/// DECLARED capacities, that no step of this frame can meet a gate the device
/// would refuse. Three classes, three proofs:
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
///
/// **THIS IS WHAT MAKES RETRY DELETABLE.** Article 4: everything a device
/// gate could refuse is proved impossible here, so a surviving refusal past
/// this door is a contract violation and not a replay. The engine lane no
/// longer sleeps on `Exhausted`, and a readiness miss at an attached pass is
/// a loud fault naming the instance and the channel
/// (`engine_cuda::serve::committed_or`).
///
/// # What stood here and is gone (wave E)
///
/// A fourth rule refused a frame whose slot *j* consumed a channel an earlier
/// slot published, whenever slot *j* "resolved its descriptors on the HOST".
/// Its entire justification was a C++ engine: `FramePrepare` ran every step's
/// host work at frame entry, and a step whose descriptor ports were
/// device-carried escaped that only if `try_device_composed_template` in
/// `crates/engine-cuda/csrc/src/pipeline/dispatch.cu` accepted it — so the
/// rule enumerated the two RS shapes and the one geometry shape that template
/// refused, and reached into the recurrent store to ask whether a row was
/// buffered. Neither the file nor the template nor the frame-entry host pass
/// exists in this tree: `Cuda::submit` drives `prepare`/`enqueue`/`settle` per
/// step, and a step's descriptor ports are read in ITS prepare, off the
/// committed front of the rings the step before it advanced. The
/// `EngineSpec::resolves_geometry_per_step` capability that let a backend opt
/// out of the rule is deleted with it — no shell in the workspace answered
/// anything but `false`, and the rule the `false` selected is gone.
fn validate_frame<C: FireContext>(
    ctx: &mut C,
    k: usize,
    fired: &[(u32, u32)],
) -> Anyhow<Result<(), String>> {
    // The walk is separated from the resource lookup so it can be tested
    // against channels rather than against a wasm store: `prove_frame_admissible`
    // is the proof and this is the gather. One `Arc` clone per channel per
    // slot, at k <= 4 and a handful of channels — the same clone the use map
    // made anyway.
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

/// The proof [`validate_frame`] gathers for — see its doc for the three
/// classes and why each one is what makes retry deletable.
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

            // ── DEVICE-ONLY RING OCCUPANCY, WALKED IN SLOT ORDER. The
            //    device publish gate admits a publish only while occupancy
            //    stays below capacity (+1 when the same fire also consumes,
            //    which is the same-fire-consume credit the counter form of
            //    the ring-full test gives). An accepted frame that
            //    structurally exceeds it would jam on the device with no
            //    retry left to absorb it. Consumes not yet reserved by any
            //    accepted fire grant no relief — this is a worst case, not a
            //    schedule. Seeded descriptor channels are exempt: their
            //    occupancy belongs to the seed protocol, not to the
            //    reserved-ticket ledger.
            //
            //    The capacity is the CHANNEL's own declared capacity
            //    (`ChannelCell::capacity`, set from the program's channel
            //    registration at bind), and the starting pressure is
            //    `device_ring_backlog()` — the reservations accepted fires
            //    hold and have not settled.
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
                    // ── HOST-WRITER, *staged* class: each consuming fire
                    //    drains one host cell, and every one of them must
                    //    already exist. A frame executes uninterrupted, so no
                    //    mid-frame host `put` can arrive to make up a
                    //    shortfall. The capacity here is not a ring size but
                    //    an inventory: `writer_available_cells()` is what the
                    //    guest has staged and not yet had taken.
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
                    // ── HOST-WRITER, *latest-value* class: a control word the
                    //    program only reads. One committed cell suffices; the
                    //    host's `set` may replace it at any time.
                    if !cell.has_committed_front() {
                        return Err(format!(
                            "pipeline: channel {}: latest-value control word has \
                             no committed cell at frame submit",
                            cell.global_id
                        ));
                    }
                }
                // publishes > 0: *device-advanced* — the program carries an
                // advance rule for it, so the host stages nothing.
            }
            Some(HostRole::Reader) if entry.publishes > 0 => {
                // ── READER RING, WORST CASE. Overflow is prevented here and
                //    never by back-pressure: the cells reserved by accepted
                //    unsettled fires, minus what the host has consumed, plus
                //    everything this frame writes, must fit the channel's
                //    declared capacity. `reader_ring_pressure()` is the first
                //    two numbers; `entry.publishes` is the third.
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

/// Compaction (Design-B lazy KV GC): move `n` token KV cells within `ws`,
/// all layers, from (`src_page_ids[i]`, `src_tok_idx[i]`) -> (`dst_page_ids[i]`,
/// `dst_tok_idx[i]`). The copy is submitted in pipeline order and awaited here,
/// so no separate pending-move lifetime or recycle epoch exists.
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

    // The WIT contract passes WorkingSet-RELATIVE page indexes (guests
    // never hold physical ids); translate through the flattened table so
    // the move lands on exactly the physical pages the fires read/write.
    // Translated at enqueue against the committed mapping: same-WS
    // in-flight fires that could remap these pages (a CoW rebase) are the
    // guest's ordering hazard, like any same-WS run-ahead write overlap.
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
                crate::engine::KvMove {
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
                // eviction and retry against the restored pages
                // (`wait_resident` posts the leave before parking).
                //
                // DIVERGENCE, kept deliberately un-"fixed": unlike the
                // fire-submit back-off (`settle_and_wait_resident`), this
                // park does not settle the process's pipeline tails first.
                // An unsettled fire elsewhere keeps its lease, and the
                // eviction's quiescence then depends on another finalizer —
                // a liveness question under watch (CONTENTION_FOLLOWUP.md
                // §16.2), not a cleanup-pass alignment.
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
/// settled FIFO entries are finalized opportunistically, but close never waits
/// for an unsettled fire: a full reader ring may require a post-close `take`
/// before the next submitted fire can settle. Channel reads and process
/// teardown retain the FIFO and finish that drain without cancelling work.
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
        // Measured at conc 512: 3 us p50 with zero pending fires, i.e. the
        // guest's run-ahead window is always already settled by the time it
        // closes. Close is not a boundary cost.
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
/// without blocking (plan §6): submit and take/read entry call this so
/// KV/RS transaction pins stay bounded by run-ahead depth while value
/// waiting rides the channel wait slots. Returns whether anything drained.
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

/// Pop and finalize EVERY pending op of one pipeline FIFO, in submit order,
/// under the queue's finalize guard — the full-drain sibling of
/// [`drain_settled`], shared by the residency gate, forward-pass drop, and
/// process teardown. `continue_on_error` is the teardown policy (log and
/// keep draining — the table is about to drop); the strict form propagates
/// the first failure.
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
/// Used by the idle-park drain (accessor-based long host awaits), which runs
/// before the scheduler freeze barrier. Device-geometry fires are excluded
/// because their lease reclamation lives on the `ForwardPass` resource and
/// still requires `FireContext`.
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
        // RS transactions have no Drop rollback. Settle them before the only
        // await below so process cancellation cannot leak their slots. The
        // mapping is already published (fail-stop either way), so settlement
        // cannot fail and does not depend on `success`.
        let rs_failure: Option<String> = if rstxn.is_some() {
            let mut rs_store = stores.rs.lock().unwrap();
            rs::settle(&mut rs_store, rstxn);
            None
        } else {
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

    // The fire's sequence retired: recycled slots (aborts, CoW'd tails,
    // collected suffixes) are allocatable now — wake parked asks. This is
    // also the planner's per-fire quiescence event: the lease this fire
    // held has just released.
    if let Some(planner) = crate::planner::planner() {
        planner.pages_freed();
    }

    // Values are already visible through the release-published tail words
    // (plan §4.5) — resolving the fire only classifies success and settles
    // the transactions above.
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

/// Device-geometry fire (Track B): the pass's [B,P] geometry is
/// DEVICE-produced (the program traces `page_indptr = CumSum(np)` + packed
/// live pages in-graph) and the engine resolves it pre-forward, so the host
/// neither replays the epilogue arithmetic nor projects per-lane KV. The
/// runtime leases `B` fresh physical pages, delivers them to the program as a
/// host-put on the `fresh` channel, submits the fire prebuilt (the host wire
/// geometry stays empty — `geometry::map_geometry_evaluated_with` maps what the
/// engine resolved), and fires it RUN-AHEAD onto the pipeline FIFO (unlike
/// the deleted synchronous host-replay beam branch).
/// The per-fire arena/write txns ride the `PendingFire`; `finalize_op`
/// commits/aborts them and reclaims continuing heirs' unused grants (w_cont).
async fn fire_device_geometry<C: FireContext>(
    ctx: &mut C,
    this: Resource<Pipeline>,
    fwd: Resource<ForwardPass>,
    frame: Option<crate::scheduler::FrameStamp>,
) -> Anyhow<Result<(), String>> {
    // **A DEVICE-RESOLVED GEOMETRY CANNOT CARRY MEDIA, AND SAYS SO FIRST**
    // (media-door §3). This path's token ids never reach the host — the
    // engine resolves the embed port in-graph — so there is no submitted
    // token list to scan for placeholder runs, and the door's whole safety
    // argument is that the correspondence is CHECKED rather than promised.
    // Taking the spans anyway would be taking the guest's word for exactly
    // the thing the scan exists to not take on trust.
    if !ctx.resources().get(&fwd)?.bindings.media.is_empty() {
        return Ok(Err(
            "pipeline: MediaDeviceGeometry: this pass attached media spans \
             and resolves its token ids on the device, so the host has no \
             token list to scan for their placeholder runs — media rides a \
             host-resolved geometry, where the runs can be checked"
                .to_string(),
        ));
    }
    // Contention-probe marker: when the guest's WIT call reached the host
    // (vs when its build reaches `acquire` — the delta is the build
    // preamble, including the settlement drain below).
    // Wire each of this pass's channels at this pipeline's FIFO (§3.4: all
    // passes binding a channel must submit on ONE pipeline — the entire
    // ordering/FIFO correctness argument).
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
    // Non-blocking settlement drain (plan §6), as in the ordinary submit.
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
        (pass.kv_ws, pass.rs_ws.clone(), pass.rs_fold_len.clone())
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

    // Grant B logical slots and size the physical demand — DevGeo STAYS in
    // the resource table throughout (no take/put-back dance); every error
    // path reverts the lease grant through `reclaim_pending_device_grant`.
    // The lease grant is purely logical (free-list reuse, then fresh
    // logical reserves) — it can never exhaust the pool, so it runs ONCE;
    // only the physical prepare below retries under contention.
    // A pool-owned pass resolves its own write targets in-graph, so the host
    // cannot name them: materialize the program's declared WRITABLE span and let
    // the translation cover every slot the device might pick. The declaration is
    // the containment promise — preparing outside it would copy-on-write a
    // shared prefix the program never touches. Resolved before the `devgeo`
    // borrow below, which would otherwise alias the pass's declaration.
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
            // Not pooled: the declaration is not needed to place writes, so a
            // span that will not resolve is not fatal here — it only costs the
            // containment bound below.
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
    let (grant_slots, write_indexes, fresh_dense, devgeo_b) = {
        let p = ctx.resources().get_mut(&fwd)?;
        let devgeo = p
            .devgeo
            .as_mut()
            .expect("fire_device_geometry on a non-device-geometry pass");
        let fresh_dense = devgeo.fresh_dense;
        let devgeo_b = devgeo.b;

        // Grant B slots: lease free-list first, then fresh logical
        // reserves. Purely logical — this can never exhaust the pool, so
        // it runs ONCE; only the physical prepare below retries under
        // contention. A pool-owned pass resolves its own write targets
        // in-graph, so it grants nothing.
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
        (grant_slots, write_indexes, fresh_dense, devgeo_b)
    };

    // Device geometry resolves B request rows in-graph. Validate the bound
    // RS list against that resolved arity before the launch and prepare one
    // folded target per row. The zero-valued `qo_indptr` carries only the
    // known row count; the engine still resolves its values in-graph.
    let resolved_qo_indptr = vec![0; devgeo_b + 1];
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
    // `pages` WAS READ ONLY BY THE WIRE FORM'S `last_page_len`: a
    // device-geometry fire leases pages and the DEVICE picks which of them a
    // token lands in, so the host's `kv_last_page_lens` for it was a guess
    // dressed as a fact (`if pages.is_empty() { 0 } else { page_size }`).
    // `KvDelta` has no seat for a guess.
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
        let rs_plan = match rs_plan_for(&rs_fold_len, &stores, &rs_ws_ids, &resolved_qo_indptr) {
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
            &resolved_qo_indptr,
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
    // channel — a shared-ring write the engine pulls before the pass (plan
    // §4.2/§4.3). The grants are WorkingSet-RELATIVE indexes — the
    // program's in-graph geometry stays logical end-to-end and the prepared
    // write above backs the physical pages behind it. This also matches the
    // bind-time seed (`reserve(b)`), which was already logical.
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

    // Device geometry and mask provenance are NOT independent, which is what
    // the previous version of this comment had wrong. For a device-geometry
    // instance the engine's descriptor resolver reads the `AttnMask` channel
    // cell on EVERY fire — that is what the class means — so lowering a
    // host-KNOWN mask to wire BRLE here reports the same mask twice, and the
    // engine fails the step loud:
    //
    //   ptir: structured mask pack collides with staged wire custom masks
    //   structured=[kind=3 sink=2 window=4 key_len=48] dense_mask_bytes=4
    //
    // Fire 0 is exactly where it bites: the mask arrives as a SEED, so the
    // host shadow knows it and took the wire path, while the seed is already
    // in the device cell for the resolver to read. Bind the whole pass to the
    // channel instead — one mask, one reader.
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

    // A DEVICE-GEOMETRY FIRE STATES ITS ROW SPLIT AND NOTHING ELSE. The wire
    // plan carried `qo_indptr` — a CSR whose only content here was "this many
    // lanes, one row each" — because the engine read the geometry out of a
    // channel the device wrote and the CSR was the only shape the host still
    // had to state. Lanes say it directly: one lane per entry, and its tokens
    // are the ones the device will resolve.
    let mut req = crate::engine::FireRequest {
        // As the host-geometry path above: this fire is a bound pass, so its
        // guest program runs at the boundary.
        boundary_program: true,
        // `Lane::word` is left at its default here and stamped below, once
        // the mask has lowered — see `stamp_lane_words`, which is the one
        // place either path states a word. `Lane::slot` is defaulted for the
        // same shape of reason and stamped at the same place by
        // `stamp_lane_slots`: a seat belongs to the working set, and B lanes
        // of one beam are B sequences that each need one.
        lanes: resolved_qo_indptr
            .windows(2)
            .map(|span| crate::engine::Lane {
                tokens: vec![0; (span[1] - span[0]) as usize],
                ..crate::engine::Lane::default()
            })
            .collect(),
        ..crate::engine::FireRequest::default()
    };
    rs_prepared.apply_to(&mut req);
    let fire_wide_mask = matches!(attn_mask, geometry::FireAttnMask::Device);
    if let Err(error) = attn_mask.apply_to(&mut req) {
        reclaim_pending_device_grant(ctx, &fwd);
        let reason = format!("pipeline: device-geometry attention mask: {error}");
        record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
        return Ok(Err(reason));
    }
    // The same stamp as the wire path, for the same reason and at the same
    // point: rows from the lanes above, mask from the lowering just now.
    //
    // **AND `false` IS THE ONLY WORD THIS PATH CAN STATE** (media-door §3). A
    // device-geometry fire's token ids are resolved ON THE DEVICE — the lanes
    // above are built with `vec![0; rows]` — so there is no host-side token
    // list to scan for placeholder runs, and a media submission here could
    // only be taken on the guest's word. `fire_device_geometry`'s entry
    // refuses that combination by name, so by this line the submission
    // carries no spans and `false` is a fact rather than a default.
    stamp_lane_words(&mut req, fire_wide_mask, false);
    // And the same seating, from the same owner: this pass's working set.
    // A device-geometry fire resolves its ROW SPLIT on the device and its
    // seats here, because a seat is not geometry — it is which sequence each
    // row group IS, which only the host knows (`stamp_lane_slots`).
    if let Err(refusal) = stamp_lane_slots(&mut req, &stores, ws.id) {
        reclaim_pending_device_grant(ctx, &fwd);
        record_submit_failure(ctx, &fwd, &pipeline_failure, &refusal);
        return Ok(Err(refusal));
    }
    // AND THE TABLE THE ENGINE RESOLVES THIS PASS'S PAGE REFERENCES THROUGH.
    // The host path's `map_lane_pages` has no work here — a device-geometry
    // lane's `KvDelta::pages` is empty, because its pages are in a channel —
    // so this is where the same translation crosses instead. See
    // `stamp_lane_translation`.
    if let Err(refusal) = stamp_lane_translation(&mut req, &stores, ws.id) {
        reclaim_pending_device_grant(ctx, &fwd);
        record_submit_failure(ctx, &fwd, &pipeline_failure, &refusal);
        return Ok(Err(refusal));
    }
    // The CLASS this fire is, said on the wire.
    //
    // A POOLED device-geometry fire states its pages in a channel and picks a
    // subset of them in-graph, so there is nothing for the engine to read in
    // the wire plan -- and until this line there was nothing in the wire
    // TABLES either: `scheduler::batch` stamps from the request, this path
    // never set anything, and the fire went out as class 0 (`Host`). Every
    // portable engine then looked for geometry in the plan, found none, and
    // refused with `no page span in a CSR of 0 entries` -- a true statement
    // about a plan that was never the place to look. The runtime's own log
    // ("executes as a pool-owned device-geometry pass") had already named the
    // class; it just never travelled.
    //
    // The NON-pooled device-geometry fire (the Track B page lease) keeps
    // class 0 deliberately. It is what CUDA runs today, its engine resolves
    // geometry from the device-composed template whatever the wire says, and
    // its scheduling was measured with the stamp it has. Nothing here is an
    // argument for moving it.
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
            rstxn: rstxns.into_inner(),
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
/// FIFO), enforcing the same-pipeline invariant (§3.4). Returns `Ok(Err(..))`
/// if a channel is already bound to a DIFFERENT pipeline.
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

/// Device-geometry per-fire page reclaim: read the harvested `w_cont`
/// (`[B]` bool: heir(true)/fork(false)) from its bound mirror, reclaim the
/// continuing heirs' UNUSED fresh page grants into the lease free-list, and
/// free those ws slots. No-op for a non-device-geometry pass.
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
    // working set discards or drops them (a discard here would shift live
    // indexes under the pass — see the pass-drop note).
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

    /// §8 guard contract, failure-injected: a prepared KV write whose fire
    /// never submits is rolled back by the guard's Drop — every page
    /// returns to the pool, exactly once, through the destructor.
    #[tokio::test(flavor = "current_thread")]
    async fn kv_txn_guard_drop_returns_every_prepared_page() -> anyhow::Result<()> {
        let model = crate::store::registry::register_model(16, &[8], &[0]);
        let stores = crate::store::registry::get(model, 0);
        let (ws, before) = crate::store::registry::with_kv_lock(&stores.kv, "test", |kv| {
            let ws = kv.create_working_set();
            kv.reserve(ws, 2).unwrap();
            (ws, kv.available_pages())
        });
        // Reserve pages as a grant would, lend them to the prepare, then
        // drop the armed guard without submitting.
        let (txn, leftover) = crate::store::registry::with_kv_lock(&stores.kv, "test", |kv| {
            let mut granted = kv.reserve_device_pages(3).expect("pool has pages");
            let (_, txn) =
                kv::realize_declaration_reserved(kv, ws, 0..2, &mut granted).expect("realize");
            let txn = kv.ensure_backed_reserved(ws, 2, &mut granted).map(|_| txn);
            (txn.expect("backed"), granted)
        });
        let guard = KvTxnGuard::new(model, 0, txn);
        // The surplus page returns through the store, the prepared pages
        // through the guard's abort — nothing needs a hand-written path.
        crate::store::registry::with_kv_lock(&stores.kv, "test", |kv| {
            kv.release_device_reservation(leftover);
        });
        drop(guard);
        crate::store::registry::with_kv_lock(&stores.kv, "test", |kv| {
            let epoch = kv.current_epoch();
            kv.release_working_set(ws, epoch);
            kv.retire_idle();
            assert_eq!(
                kv.available_pages(),
                before,
                "every page returned exactly once"
            );
        });
        Ok(())
    }

    /// The RS contract under publish-at-prepare: the guard SETTLES rather
    /// than rolls back. The folded slot the prepare adopted stays owned by
    /// the working set (fail-stop, as for KV), the unconsumed reservation
    /// returns immediately, and settling releases the in-flight hold so a
    /// later release retires everything.
    #[tokio::test(flavor = "current_thread")]
    async fn rs_txns_guard_drop_settles_without_rolling_back_the_mapping() -> anyhow::Result<()> {
        let model = crate::store::registry::register_model(16, &[4], &[4]);
        let stores = crate::store::registry::get(model, 0);
        let (txn, ws, before) = {
            let mut store = stores.rs.lock().unwrap();
            let ws = store.create_working_set(crate::store::rs::RsGeometry {
                state_size: 64,
                buffer_page_tokens: 4,
                fold_granularity: 1,
            });
            let before = store.available_slots();
            let mut granted = store.reserve_slots(2).expect("slots available");
            let prepared =
                rs::prepare_many_reserved(&mut store, &[ws], &rs::RsPlan::Fold, &mut granted)
                    .expect("prepare");
            let txn = prepared.txn;
            store.release_slot_reservation(granted);
            assert!(
                store.folded_slot(ws).expect("live working set").is_some(),
                "prepare publishes the folded slot before the fire is submitted"
            );
            (txn, ws, before)
        };
        drop(RsTxnsGuard::new(model, 0, txn));
        {
            let mut store = stores.rs.lock().unwrap();
            assert_eq!(
                store.available_slots(),
                before - 1,
                "the published folded slot stays owned by the working set"
            );
            let epoch = store.current_epoch();
            store.release_working_set(ws, epoch);
            store.retire_idle();
            assert_eq!(
                store.available_slots(),
                before,
                "settling released the in-flight hold, so release retires everything"
            );
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

/// **STATIC ADMISSION, PROVED AGAINST CHANNELS** (alto E; design §1 article 4,
/// survey §7 I8).
///
/// These drive [`prove_frame_admissible`] — the half of `validate_frame` that
/// is the proof rather than the resource lookup — because what article 4 asks
/// is a statement about rings and capacities, not about a wasm store. Each
/// test builds the channels a frame would touch, states what each slot does to
/// them, and asserts the frame is admitted or refused BY NAME.
///
/// The rule these replaced is in `validate_frame`'s own doc: a chained-slot
/// refusal justified by a `try_device_composed_template` in a `csrc/` this
/// tree does not contain.
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

    /// Reserve `n` publish tickets on a channel — what an accepted, unsettled
    /// fire leaves behind. This is the state the walk starts from.
    fn reserve_publishes(cell: &Arc<Mutex<ChannelCell>>, n: usize) {
        for _ in 0..n {
            cell.lock().unwrap().reserve_device_ticket(false, true);
        }
    }

    /// **A DEVICE-ONLY RING IS WALKED IN SLOT ORDER**, and a frame whose net
    /// growth fits its declared capacity is admitted.
    ///
    /// Two slots, each publishing one cell and consuming the one before it:
    /// occupancy goes 0 → 1 → 1, which a capacity of 2 holds with room. The
    /// walk has to be ordered for this to be provable at all — the same two
    /// accesses counted as a set say "two publishes, two consumes" and cannot
    /// tell this frame from one that publishes both before consuming either.
    #[test]
    fn a_device_ring_frame_that_fits_in_slot_order_is_admitted() {
        let ring = channel(HostRole::None, 2, false);
        let slots = [
            slot(std::slice::from_ref(&ring), &[(false, true)]),
            slot(std::slice::from_ref(&ring), &[(true, true)]),
        ];
        assert_eq!(prove_frame_admissible(2, &slots), Ok(()));
    }

    /// The same shape past the capacity is refused, and the refusal names the
    /// channel, the slot and the capacity.
    ///
    /// This is the whole of why retry is deletable: a frame that structurally
    /// overfills a device-only ring would jam on the device with nothing left
    /// to absorb it, so it must not be admitted rather than admitted and
    /// re-offered.
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

    /// **THE BACKLOG OF ACCEPTED FIRES IS WHERE THE WALK STARTS.** A ring with
    /// room for two that already owes one publish to an unsettled fire has
    /// room for one more, not two.
    #[test]
    fn a_reserved_backlog_counts_against_the_frame() {
        let ring = channel(HostRole::None, 2, false);
        reserve_publishes(&ring, 1);
        let one = [slot(std::slice::from_ref(&ring), &[(false, true)])];
        assert_eq!(prove_frame_admissible(2, &one), Ok(()));

        let two = [
            slot(std::slice::from_ref(&ring), &[(false, true)]),
            slot(std::slice::from_ref(&ring), &[(false, true)]),
        ];
        let refusal = prove_frame_admissible(2, &two).expect_err("backlog + 2 > capacity 2");
        assert!(refusal.contains("device-ring occupancy past capacity 2"), "{refusal}");
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

    /// **EVERY HOST-WRITER CELL A FRAME DRAINS MUST ALREADY BE STAGED.**
    ///
    /// A frame executes uninterrupted, so a `put` arriving mid-frame is not a
    /// thing that can happen: two consuming slots against one staged cell is
    /// a frame that would run dry, and it is refused at submit with the two
    /// numbers in the message.
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

    /// **THE READER RING IS SIZED FOR THE WORST CASE**, which is the guest
    /// draining nothing before the frame executes.
    ///
    /// Capacity `2k - 1` is the message's own advice at k = 2: a frame of two
    /// publishing slots needs two cells, and a ring of one is refused with the
    /// number it would have needed.
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

    /// **A CHAINED SLOT IS ADMITTED**, which is the rule wave E deleted.
    ///
    /// Slot 0 publishes a device-only descriptor ring and slot 1 consumes it —
    /// the shape `validate_frame` used to refuse whenever slot 1's descriptors
    /// "resolved on the HOST", on behalf of a `FramePrepare` that ran every
    /// step's host work at frame entry. `Cuda::submit` prepares each step in
    /// turn, off the committed front the step before it advanced, so the
    /// frame is ordinary and the only question left is whether the ring holds
    /// it.
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
