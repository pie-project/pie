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
//! (`PendingForward`, Option A) applied to this engine. Pure-attention KV step
//! t+1 is prepared against t's OPTIMISTIC post-state (`committed_tokens`
//! advances at submit); RS step t+1 waits for t's committed folded mapping.
//! `channel.take`/`read` also finalize in-flight fires FIFO until the cell
//! fills. A failed fire **poisons** the pass's host-reader channels and fails
//! the pass for further submits.
//!
//! **Layering.** The orchestration functions below (`submit_pass`,
//! `finalize_fire`, `drain_settled`, `wire_channels_to_pipeline`,
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

pub mod context;
pub mod geometry;
pub mod kv;
pub mod lease;
pub mod rs;

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use wasmtime::component::Resource;

pub use context::FireContext;

use crate::pipeline::Pipeline;
use crate::pipeline::channel::{BoundCells, Channel, ChannelError};
use crate::pipeline::instance::ForwardPass;
use crate::store::kv::working_set::{KvFireLease, KvWorkingSet};
use crate::store::rs::working_set::RsWorkingSet;
use lease::DevGeo;
use pie_ptir::container::HostRole;

/// A pass's in-flight fires, submit order. Plain mutex: never held across an
/// await (the op is popped out, then awaited).
pub type PendingFires = Arc<Mutex<VecDeque<PendingOp>>>;
pub type PipelineFailure = Arc<Mutex<Option<String>>>;

#[derive(Clone)]
struct AcquireCancellation {
    cancelled: Arc<AtomicBool>,
    notify: Arc<tokio::sync::Notify>,
}

impl AcquireCancellation {
    async fn cancelled(&self) {
        if self.cancelled.load(Ordering::Acquire) {
            return;
        }
        let notified = self.notify.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        if self.cancelled.load(Ordering::Acquire) {
            return;
        }
        notified.await;
    }
}

struct AcquireOwner {
    cancellation: AcquireCancellation,
}

impl AcquireOwner {
    fn new() -> Self {
        Self {
            cancellation: AcquireCancellation {
                cancelled: Arc::new(AtomicBool::new(false)),
                notify: Arc::new(tokio::sync::Notify::new()),
            },
        }
    }

    fn cancellation(&self) -> AcquireCancellation {
        self.cancellation.clone()
    }
}

impl Drop for AcquireOwner {
    fn drop(&mut self) {
        self.cancellation.cancelled.store(true, Ordering::Release);
        self.cancellation.notify.notify_waiters();
    }
}

fn reclaimable_probe(
    model: usize,
    driver: usize,
    process_id: uuid::Uuid,
    working_sets: std::collections::HashSet<crate::store::kv::page_table::WorkingSetId>,
) -> crate::store::reclaim::ReclaimableProbe {
    Arc::new(move || {
        let stores = crate::store::registry::get(model, driver);
        match stores
            .kv
            .lock()
            .unwrap()
            .post_drain_reclaimable_page_count(&working_sets)
        {
            Ok(pages) => pages > 0,
            Err(error) => {
                tracing::warn!(
                    pid = %process_id,
                    %error,
                    "failed to refresh KV reclaimability"
                );
                false
            }
        }
    })
}

/// A pipeline FIFO entry: a forward FIRE or a KV cell MOVE (Design-B
/// compaction). Both hold an ordered slot on the same stream — the B3
/// happens-before invariant; `take`/`read` drain them in submit order.
pub enum PendingOp {
    Fire(PendingFire),
    Move(PendingMove),
}

impl PendingOp {
    /// Non-blocking probe: whether the op's completion has settled.
    fn is_settled(&self) -> bool {
        match self {
            PendingOp::Fire(fire) => fire.completion.is_settled(),
            PendingOp::Move(mv) => mv.completion.is_settled(),
        }
    }

    /// An owned, payload-free await on this op's completion (cloned so the
    /// pipeline queue lock is not held across the await). The outcome is
    /// ignored; the FIFO drain reads the real result.
    fn completion_signal(&self) -> OpSignal {
        match self {
            PendingOp::Fire(fire) => OpSignal::Fire(fire.completion.clone()),
            PendingOp::Move(mv) => OpSignal::Move(mv.completion.clone()),
        }
    }

    fn request_cancel(&self) {
        if let PendingOp::Fire(fire) = self {
            fire.completion.request_cancel();
            crate::scheduler::nudge(0);
        }
    }

    pub(crate) fn is_preemption_safe_unprepared(&self) -> bool {
        matches!(
            self,
            PendingOp::Fire(PendingFire {
                kv: FireKv::Deferred(slot),
                ..
            }) if slot.lock().unwrap().is_none()
        )
    }

    pub(crate) fn is_preemption_detachable(&self) -> bool {
        matches!(
            self,
            PendingOp::Move(_)
                | PendingOp::Fire(PendingFire {
                    kv: FireKv::Deferred(_),
                    ..
                })
        )
    }

    pub(crate) fn preemption_signal(
        &self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> {
        Box::pin(self.completion_signal())
    }
}

/// See [`PendingOp::completion_signal`].
enum OpSignal {
    Fire(crate::driver::WorkItemCompletion),
    Move(crate::driver::SubmissionCompletion),
}

impl std::future::Future for OpSignal {
    type Output = ();

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<()> {
        match self.get_mut() {
            OpSignal::Fire(completion) => std::pin::Pin::new(completion).poll(cx).map(|_| ()),
            OpSignal::Move(completion) => std::pin::Pin::new(completion).poll(cx).map(|_| ()),
        }
    }
}

/// One in-flight KV cell MOVE (`pipeline.copy-into`). It holds an ordered slot
/// in the pipeline FIFO and finalizes by awaiting its payload-free completion.
pub struct PendingMove {
    completion: crate::driver::SubmissionCompletion,
    failure: PipelineFailure,
}

/// Test-only inert FIFO entry: enough to make a pass's shared fires queue
/// non-empty for `ForwardPass::can_close_native_on_drop`/`Drop` tests
/// (`pipeline::instance`'s `native_cleanup` test module), without needing a
/// live completion. `PendingMove`/`PendingFire`'s fields are private
/// to this module, so cross-module tests construct a stub through here
/// rather than reaching in directly.
#[cfg(test)]
pub(crate) fn test_pending_move_stub() -> PendingOp {
    PendingOp::Move(PendingMove {
        completion: crate::driver::SubmissionCompletion::ready(),
        failure: Arc::new(Mutex::new(None)),
    })
}

/// The open KV/arena transaction(s) one in-flight fire holds until it resolves.
/// Two shapes: the ordinary single-seq / MTP projection ([`kv`]), or a
/// device-geometry fire whose KV the driver resolves+writes itself (B2's
/// explicit-KV path) — the runtime only pins the [`lease::PageLease`]-granted
/// physical pages for the fire, released at finalize (per-fire arena txn; the
/// plan's "pin float bounded by run-ahead depth × B, riding the per-fire arena
/// txns").
enum FireKv {
    Deferred(Arc<Mutex<Option<kv::KvTxn>>>),
    /// A device-geometry fire's prepared write over the lease-granted slots
    /// (B2's explicit-KV path): same commit/abort protocol, no host
    /// projection.
    DeviceGeom {
        kvtxn: kv::KvTxn,
    },
}

/// One in-flight fire: the work item completion plus everything needed to
/// finalize when it resolves — the open KV/RS txns (pins/CoW held until
/// commit/abort) and the bound cells whose mirror epochs become visible.
pub struct PendingFire {
    completion: crate::driver::WorkItemCompletion,
    kv: FireKv,
    rstxns: Vec<rs::RsTxn>,
    ws_guard: KvFireLease,
    ws_rep: u32,
    rs_reps: Vec<u32>,
    /// The owning pass, to fail it on a fire error (rep — the guest may have
    /// dropped the handle; failure marking is then moot).
    fwd_rep: u32,
    instance_id: u64,
    cells: BoundCells,
    failure: PipelineFailure,
}

type PreparedRs = (Vec<u32>, Vec<u8>, Vec<u32>, Vec<u32>, Vec<rs::RsTxn>);

fn prepare_bound_rs<C: FireContext>(
    ctx: &mut C,
    stores: &crate::store::registry::Stores,
    model: usize,
    driver: usize,
    rs_reps: &[u32],
    qo_indptr: &[u32],
    pipeline_scope: usize,
) -> Anyhow<Result<PreparedRs, String>> {
    let has_recurrent_state = pie_model::model().rs_caps().state_size > 0;
    if let Err(error) = rs::validate_count(rs_reps.len(), qo_indptr, has_recurrent_state) {
        return Ok(Err(format!("pipeline: recurrent-state binding: {error}")));
    }

    let mut working_sets = Vec::with_capacity(rs_reps.len());
    for (row, &rep) in rs_reps.iter().enumerate() {
        let resource: Resource<RsWorkingSet> = Resource::new_borrow(rep);
        let rs = ctx.resources().get(&resource)?.clone();
        if rs.model != model || rs.driver as usize != driver {
            return Ok(Err(format!(
                "pipeline: rs-working-set at request row {row} belongs to model/driver \
                 ({}, {}), expected ({model}, {driver})",
                rs.model, rs.driver
            )));
        }
        if let Err(owner) = rs.claim_pipeline_scope(pipeline_scope) {
            return Ok(Err(format!(
                "pipeline: rs-working-set at request row {row} is already scoped to pipeline \
                 {owner:#x}"
            )));
        }
        working_sets.push(rs.id);
    }

    let prepared = {
        let mut store = stores.rs.lock().unwrap();
        rs::prepare_many(&mut store, &working_sets)
    };
    Ok(prepared
        .map(|(ids, flags, (copy_src, copy_dst), txns)| (ids, flags, copy_src, copy_dst, txns))
        .map_err(|error| format!("pipeline: rs prepare: {error}")))
}

fn abort_rs_transactions(stores: &crate::store::registry::Stores, txns: Vec<rs::RsTxn>) {
    if txns.is_empty() {
        return;
    }
    let mut store = stores.rs.lock().unwrap();
    rs::abandon_many(&mut store, txns);
}

pub(crate) async fn drain_rs_predecessors<C: FireContext>(
    ctx: &mut C,
    fires: &PendingFires,
) -> Anyhow<()> {
    loop {
        let completion = fires
            .lock()
            .unwrap()
            .front()
            .map(PendingOp::completion_signal);
        let Some(mut completion) = completion else {
            return Ok(());
        };
        if let Some(preemption) = ctx.preemption_signal() {
            let notified = preemption.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            tokio::select! {
                _ = &mut completion => {}
                _ = &mut notified => {
                    ctx.honor_preemption().await?;
                    continue;
                }
            }
        } else {
            completion.await;
        }

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

/// Poison every host-reader cell of a pass with the failed fire's error —
/// under run-ahead this IS the error channel (`take`/`read` surface it).
fn poison_readers(cells: &BoundCells, reason: &str) {
    for cell in cells {
        let mut c = cell.lock().unwrap();
        if c.role == Some(HostRole::Reader) {
            c.poison(reason);
            // A waiter parked on the reader wait slot must observe the poison.
            if let Some(endpoint) = c.endpoint() {
                pie_waker::WakerTable::global().wake(endpoint.registered().reader_wait_id);
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

    fn apply_to(&self, request: &mut crate::driver::LaunchPlan) {
        request.channel_expected_head.clone_from(&self.heads);
        request.channel_expected_tail.clone_from(&self.tails);
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
/// advances (the driver's completion callback wakes the reader wait slot
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

/// The body behind `forward-pass.submit(on)`.
pub async fn submit_pass<C: FireContext>(
    ctx: &mut C,
    this: Resource<Pipeline>,
    fwd: Resource<ForwardPass>,
) -> Anyhow<Result<(), String>> {
    {
        // Device-geometry pass (Track B): the [B,P] geometry is
        // device-produced (the program traces the wire form in-graph) and
        // the driver resolves it pre-forward, so this pass leases physical
        // pages + fires solo/prebuilt via `map_geometry_relaxed` — but it
        // RUNS AHEAD like any pass (the FIFO carries it; NOT synchronous like
        // the deleted host-replay beam branch).
        if ctx.resources().get(&fwd)?.devgeo.is_some() {
            return fire_device_geometry(ctx, this, fwd).await;
        }
        // W3.1: the PIPELINE owns the in-flight FIFO. Point each of this
        // pass's channels at this pipeline's queue so their `take`/`read`
        // await the right FIFO — enforcing the same-pipeline constraint
        // (§3.4): every pass binding a channel must submit on one pipeline.
        let pipe_fires = ctx.resources().get(&this)?.fires.clone();
        let pipeline_failure = ctx.resources().get(&this)?.failure.clone();
        // Non-blocking settlement drain (plan §6): resolved fires' KV/RS
        // txns finalize here so arena pins stay bounded by run-ahead depth
        // even when the guest never takes.
        drain_settled(ctx, Some(&pipe_fires)).await?;
        if let Some(reason) = pipeline_failure.lock().unwrap().clone() {
            return Ok(Err(format!("pipeline: pipeline failed: {reason}")));
        }
        if let Err(error) = wire_channels_to_pipeline(ctx, &fwd, &pipe_fires)? {
            return Ok(Err(error));
        }
        if !ctx.resources().get(&fwd)?.rs_ws.is_empty() {
            // RS mappings publish only at finalize. Correctness-first
            // serialization prevents a later run-ahead fire from preparing
            // against stale committed state (double RESET / repeated CoW).
            drain_rs_predecessors(ctx, &pipe_fires).await?;
            if let Some(reason) = pipeline_failure.lock().unwrap().clone() {
                return Ok(Err(format!("pipeline: pipeline failed: {reason}")));
            }
        }
        let (
            geometry,
            cells,
            ws_rep,
            rs_reps,
            committed_tokens,
            fwd_rep,
            instance_id,
            program,
            completion,
            canonical_evidence,
            dense_mask,
            accesses,
        ) = {
            let p = ctx.resources().get_mut(&fwd)?;
            if let Some(e) = &p.failed {
                return Ok(Err(format!(
                    "pipeline: forward-pass failed by an earlier fire: {e}"
                )));
            }
            let geometry = p
                .instance
                .fire_geometry(crate::pipeline::program::model_profile().page_size)
                .ok();
            let canonical_evidence = kv::canonical_fire_evidence(
                p.canonical_kv,
                &p.instance.program.bound.container,
                &p.cells,
                &p.instance.seeds,
                p.fired_once,
            );
            let accesses = p.instance.program.channel_accesses.clone();
            (
                geometry,
                p.cells.clone(),
                p.kv_ws,
                p.rs_ws.clone(),
                p.committed_tokens,
                fwd.rep(),
                p.bound_instance.instance_id,
                Arc::clone(&p.instance.program),
                p.bound_instance.reserve_completion(),
                canonical_evidence,
                p.dense_mask,
                accesses,
            )
        };
        let mut req = crate::driver::LaunchPlan::default();
        if let Some(g) = &geometry {
            g.apply_to(&mut req);
        }
        // A dense device mask (AttnMask channel) marks the fire
        // mask-carrying: the scheduler batches it SOLO (the composed
        // multi-program batch does not merge dense device masks — v1
        // scope).
        req.has_user_mask = dense_mask;
        crate::pipeline::offload::try_encode(&mut req).await;
        // Prepare the guest-owned KV working set for this fire via
        // `pipeline::fire::kv` over the typed KvStore (reserve growth +
        // fresh / in-place / CoW classification + geometry projection).
        // The held `KvTxn` rides the PendingFire across the async fire;
        // finalized (commit → mapping publishes / abort → pending slots
        // release) when a take/read/drop drains it.
        let new_tokens: Vec<u32> = req.token_ids.clone();
        let ws_res: Resource<KvWorkingSet> = Resource::new_borrow(ws_rep);
        let ws = ctx.resources().get(&ws_res)?.clone();
        let stores = crate::store::registry::get(ws.model, ws.driver as usize);
        let pid = ctx.process_id();
        let process_working_sets = ctx.kv_working_sets();
        if let Err(owner) = ws.claim_pipeline_scope(pid.as_u128()) {
            return Ok(Err(format!(
                "pipeline: KV working set is already scoped to pipeline {owner:032x}"
            )));
        }
        let ws_guard = match ws.fire_lease() {
            Ok(lease) => lease,
            Err(error) => return Ok(Err(format!("pipeline: KV working set: {error}"))),
        };

        // Recurrent-state rows are lowered independently, in resolved request
        // order. Their CoW copies ride the scheduler's typed pre-launch state
        // copy so a copy failure rejects this fire before model execution.
        let (rs_slot_ids, rs_slot_flags, rs_copy_src, rs_copy_dst, rstxns) = match prepare_bound_rs(
            ctx,
            &stores,
            ws.model,
            ws.driver as usize,
            &rs_reps,
            &req.qo_indptr,
            Arc::as_ptr(&pipe_fires) as usize,
        )? {
            Ok(prepared) => prepared,
            Err(error) => return Ok(Err(error)),
        };
        req.rs_slot_ids = rs_slot_ids;
        req.rs_slot_flags = rs_slot_flags;

        // Mapping allocation is deferred, but extent order is not: publish the
        // submitted token extent now so another pass sharing this WS cannot
        // prepare over it before this fire reaches dispatch.
        let committed_floor = match {
            let kv = stores.kv.lock().unwrap();
            kv.visible_token_len(ws.id, ws.page_size)
        } {
            Ok(floor) => floor,
            Err(error) => {
                abort_rs_transactions(&stores, rstxns);
                return Ok(Err(format!("pipeline: KV visible token extent: {error}")));
            }
        };
        let extent_reservation = match ws.reserve_token_extent(
            u64::from(committed_tokens),
            committed_floor,
            new_tokens.len() as u64,
        ) {
            Ok(reservation) => reservation,
            Err(error) => {
                abort_rs_transactions(&stores, rstxns);
                return Ok(Err(format!("pipeline: KV working set: {error}")));
            }
        };
        let committed_tokens = match u32::try_from(extent_reservation.start_token()) {
            Ok(cursor) => cursor,
            Err(_) => {
                abort_rs_transactions(&stores, rstxns);
                return Ok(Err("pipeline: KV token cursor exceeds u32".to_string()));
            }
        };
        let reserved_end = match committed_tokens.checked_add(new_tokens.len() as u32) {
            Some(end) => end,
            None => {
                abort_rs_transactions(&stores, rstxns);
                return Ok(Err("pipeline: KV token cursor exceeds u32".to_string()));
            }
        };
        // Canonical gate, fire-time half: hash under the host-verified token
        // values only when they cover this fire's full-context append.
        let hash_tokens: Option<Vec<u32>> = canonical_evidence.and_then(|(toks, kv_len)| {
            (toks.len() == new_tokens.len() && kv_len == Some(reserved_end)).then_some(toks)
        });

        // ── Prefix-cache graft (kv_refact.md, the matching consumer) ──
        // First canonical fire of a FRESH working set: probe the CAS for
        // the longest cached full-page prefix of this fire's embed, adopt
        // it (structurally shared pages — writes CoW like any shared
        // path), and TRIM the launch to the uncached suffix. A pure
        // execution-strategy swap, invisible to the guest: the total
        // attended context (KvLen) is unchanged, the kv projection covers
        // the full context from the store, sampled rows keep their
        // identity (shifted to the computed suffix), and the chain state
        // continues from the boundary hash — so the launch is
        // indistinguishable from a continuation fire over `kts` committed
        // tokens. Any miss or ineligible shape falls through to the full
        // computation.
        let (mut committed_tokens, mut new_tokens, mut hash_tokens) = {
            let mut committed = committed_tokens;
            let mut toks_new = new_tokens;
            let mut toks_hash = hash_tokens;
            let n = toks_new.len();
            // Eligible: nothing committed yet, host-verified tokens for
            // the whole embed, and the exact single-lane canonical shape.
            let eligible = committed == 0
                && n > 1
                && req.qo_indptr == [0, n as u32]
                && req.position_ids.len() == n
                && req.sampling_indices.iter().all(|&s| (s as usize) < n);
            if eligible {
                if let Some(toks) = toks_hash.as_deref() {
                    // A sampled row must stay in the computed suffix: cap
                    // the probe so no adopted page covers one.
                    let min_sample = req
                        .sampling_indices
                        .iter()
                        .copied()
                        .min()
                        .map(|s| s as usize)
                        .unwrap_or(usize::MAX);
                    let cap = n.min(min_sample.saturating_add(1));
                    let adopted = {
                        let mut kv = stores.kv.lock().unwrap();
                        kv::match_prefix(&mut kv, ws.id, &toks[..cap], ws.page_size)
                    };
                    match adopted {
                        Ok(Some(pages)) if pages > 0 => {
                            let kts = pages as usize * ws.page_size as usize;
                            debug_assert!(kts < n && kts <= min_sample);
                            req.token_ids.drain(..kts);
                            req.position_ids.drain(..kts);
                            req.qo_indptr = vec![0, (n - kts) as u32];
                            for s in &mut req.sampling_indices {
                                *s -= kts as u32;
                            }
                            toks_new.drain(..kts);
                            toks_hash = toks_hash.map(|t| t[kts..].to_vec());
                            committed = kts as u32;
                        }
                        Ok(_) => {}
                        Err(e) => {
                            tracing::debug!("pipeline prefix-cache probe miss: {e}");
                        }
                    }
                }
            }
            (committed, toks_new, toks_hash)
        };

        if committed_tokens == 0
            && let Some(tokens) = hash_tokens.as_deref()
            && let Some(adoption) = crate::pipeline::offload::try_prefill(
                ws.model,
                ws.driver as usize,
                ws.id,
                ws.page_size,
                &mut req,
                tokens,
                &program,
                instance_id,
                ws.fire_lease().ok(),
            )
            .await
        {
            let adopted = adoption.token_count;
            new_tokens.drain(..adopted);
            hash_tokens = hash_tokens.map(|tokens| tokens[adopted..].to_vec());
            committed_tokens += adopted as u32;
        }

        debug_assert_eq!(
            u64::from(committed_tokens) + new_tokens.len() as u64,
            u64::from(reserved_end)
        );
        let next_committed = reserved_end;
        let kvtxn_slot = Arc::new(Mutex::new(None));

        let model = ws.model;
        let driver = ws.driver as usize;
        let ws_id = ws.id;
        let page_size = ws.page_size;
        let deferred_slot = kvtxn_slot.clone();
        let mut deferred_new_tokens = Some(new_tokens);
        let mut deferred_hash_tokens = Some(hash_tokens);
        let acquire_started = Arc::new(AtomicBool::new(false));
        let acquire_result = Arc::new(Mutex::new(
            None::<Result<crate::store::reclaim::Acquired, String>>,
        ));
        let self_suspend_wait = Arc::new(AtomicBool::new(false));
        let acquire_owner = AcquireOwner::new();
        let runtime = tokio::runtime::Handle::current();
        let preparation: crate::scheduler::LaunchPreparation = Box::new(move |request| {
            if let Some(orchestrator) = crate::store::reclaim::contention()
                && !orchestrator.is_running(pid)
            {
                if orchestrator.is_registered(pid) {
                    return Err(crate::scheduler::LaunchPreparationError::Blocked(
                        "pipeline is quiesced for KV suspension".to_string(),
                    ));
                }
                return Err(crate::scheduler::LaunchPreparationError::Failed(
                    "pipeline terminated during KV preparation".to_string(),
                ));
            }
            let new_tokens = deferred_new_tokens
                .as_ref()
                .expect("dispatch preparation payload remains live");
            let hash_tokens = deferred_hash_tokens
                .as_ref()
                .expect("dispatch preparation hash payload remains live");
            if self_suspend_wait.load(Ordering::Acquire) {
                if crate::store::reclaim::contention()
                    .is_some_and(|orchestrator| orchestrator.is_running(pid))
                {
                    self_suspend_wait.store(false, Ordering::Release);
                } else {
                    return Err(crate::scheduler::LaunchPreparationError::Blocked(
                        "KV requester is self-suspended".to_string(),
                    ));
                }
            }
            let grant = match acquire_result.lock().unwrap().take() {
                Some(Ok(crate::store::reclaim::Acquired::Granted(grant))) => {
                    acquire_started.store(false, Ordering::Release);
                    Some(grant.into_pages())
                }
                Some(Ok(crate::store::reclaim::Acquired::SelfSuspendFirst)) => {
                    acquire_started.store(false, Ordering::Release);
                    self_suspend_wait.store(true, Ordering::Release);
                    return Err(crate::scheduler::LaunchPreparationError::Blocked(
                        "KV requester must self-suspend".to_string(),
                    ));
                }
                Some(Err(error)) => {
                    acquire_started.store(false, Ordering::Release);
                    return Err(crate::scheduler::LaunchPreparationError::Failed(format!(
                        "pipeline: KV contention: {error}"
                    )));
                }
                None => None,
            };
            let stores = crate::store::registry::get(model, driver);
            if grant.is_none()
                && let Some(orchestrator) = crate::store::reclaim::contention()
                && orchestrator.allocation_requires_grant()
            {
                let requested = {
                    let mut kv_store = stores.kv.lock().unwrap();
                    kv::required_pages(
                        &mut kv_store,
                        ws_id,
                        committed_tokens,
                        new_tokens.len() as u32,
                        page_size,
                    )
                    .map_err(|error| {
                        crate::scheduler::LaunchPreparationError::Failed(format!(
                            "pipeline: KV grant sizing: {error}"
                        ))
                    })?
                };
                if requested > 0 && !acquire_started.swap(true, Ordering::AcqRel) {
                    let reclaimable =
                        reclaimable_probe(model, driver, pid, process_working_sets.clone());
                    let result = acquire_result.clone();
                    let cancellation = acquire_owner.cancellation();
                    runtime.spawn(async move {
                        let acquire = orchestrator.acquire_or_self_suspend_live(
                            pid,
                            requested as u32,
                            reclaimable,
                        );
                        tokio::pin!(acquire);
                        tokio::select! {
                            outcome = &mut acquire => {
                                *result.lock().unwrap() =
                                    Some(outcome.map_err(|error| error.to_string()));
                                crate::scheduler::nudge(driver);
                            }
                            _ = cancellation.cancelled() => {}
                        }
                    });
                }
                if requested > 0 {
                    return Err(crate::scheduler::LaunchPreparationError::Blocked(
                        "KV allocation waits for an older grant".to_string(),
                    ));
                }
            }
            let prepared = {
                let mut kv_store = stores.kv.lock().unwrap();
                match grant {
                    Some(granted) => kv::prepare_granted(
                        &mut kv_store,
                        ws_id,
                        committed_tokens,
                        new_tokens,
                        page_size,
                        hash_tokens.as_deref(),
                        granted,
                    ),
                    None => kv::prepare(
                        &mut kv_store,
                        ws_id,
                        committed_tokens,
                        new_tokens,
                        page_size,
                        hash_tokens.as_deref(),
                    ),
                }
            };
            match prepared {
                Ok((projection, (copy_src, copy_dst), translation, txn)) => {
                    let translation_version = txn.mapping_version();
                    request.kv_translation = translation;
                    *deferred_slot.lock().unwrap() = Some(txn);
                    deferred_new_tokens.take();
                    deferred_hash_tokens.take();
                    Ok(crate::scheduler::PreparedLaunch {
                        page_refs: (0..projection.physical_page_ids.len() as u32).collect(),
                        last_page_len: projection.last_page_len,
                        kv_translation_version: translation_version,
                        copy_src,
                        copy_dst,
                    })
                }
                Err(kv::KvError::OutOfPages { requested, .. }) => {
                    let mut kv_store = stores.kv.lock().unwrap();
                    let epoch = kv_store.current_epoch();
                    let reclaimed = kv_store.drop_unused_cache_leases(epoch);
                    kv_store.retire_idle();
                    drop(kv_store);
                    if reclaimed == 0
                        && let Some(orchestrator) = crate::store::reclaim::contention()
                    {
                        if !acquire_started.swap(true, Ordering::AcqRel) {
                            let reclaimable =
                                reclaimable_probe(model, driver, pid, process_working_sets.clone());
                            let result = acquire_result.clone();
                            let cancellation = acquire_owner.cancellation();
                            runtime.spawn(async move {
                                let acquire = orchestrator.acquire_or_self_suspend_live(
                                    pid,
                                    requested as u32,
                                    reclaimable,
                                );
                                tokio::pin!(acquire);
                                tokio::select! {
                                    outcome = &mut acquire => {
                                        *result.lock().unwrap() =
                                            Some(outcome.map_err(|error| error.to_string()));
                                        crate::scheduler::nudge(driver);
                                    }
                                    _ = cancellation.cancelled() => {}
                                }
                            });
                        }
                        return Err(crate::scheduler::LaunchPreparationError::Blocked(
                            "KV pages unavailable at dispatch".to_string(),
                        ));
                    }
                    Err(crate::scheduler::LaunchPreparationError::Retry(
                        "KV pages unavailable at dispatch".to_string(),
                    ))
                }
                Err(error) => Err(crate::scheduler::LaunchPreparationError::Failed(format!(
                    "pipeline: kv dispatch preparation: {error}"
                ))),
            }
        });

        // Fire through the scheduler → dispatch-time KV preparation → driver.
        let ticket_reservation = TicketReservation::new(&cells, &accesses);
        ticket_reservation.apply_to(&mut req);
        let retry_cells = cells.clone();
        let retry_accesses = accesses.clone();
        let retry_classifier: crate::scheduler::RetryClassifier = Box::new(move || {
            retry_cells
                .iter()
                .zip(&retry_accesses)
                .find_map(|(cell, &(consume, publish))| {
                    cell.lock()
                        .unwrap()
                        .permanent_retry_cause(consume || publish)
                })
        });
        let submit_error = crate::scheduler::submit_async_deferred_with_rs_copy(
            req,
            0,
            instance_id,
            Some(pid),
            Some(u64::from(ws_rep)),
            completion.clone(),
            preparation,
            Some(retry_classifier),
            rs_copy_src,
            rs_copy_dst,
        )
        .err()
        .map(|error| format!("{error:#}"));
        if let Some(error) = submit_error {
            let reason = format!("pipeline: submit failed: {error}");
            abort_rs_transactions(&stores, rstxns);
            ctx.resources().get_mut(&fwd)?.failed = Some(reason.clone());
            return Ok(Err(reason));
        }
        ticket_reservation.commit();
        extent_reservation.commit();

        // Optimistic cursor advance: the NEXT submit prepares against this
        // fire's post-state (the run-ahead overlap). A failed fire fails
        // the pass instead of rewinding.
        {
            let p = ctx.resources().get_mut(&fwd)?;
            p.committed_tokens = next_committed;
            p.fired_once = true; // seeds are consumed; the seed rule is off
        }

        pipe_fires
            .lock()
            .unwrap()
            .push_back(PendingOp::Fire(PendingFire {
                completion,
                kv: FireKv::Deferred(kvtxn_slot),
                rstxns,
                ws_guard,
                ws_rep,
                rs_reps,
                fwd_rep,
                instance_id,
                cells,
                failure: pipeline_failure,
            }));
        Ok(Ok(()))
    }
}

/// Compaction (Design-B lazy KV GC): move `n` token KV cells within `ws`,
/// all layers, from (`src_page_ids[i]`, `src_tok_idx[i]`) -> (`dst_page_ids[i]`,
/// `dst_tok_idx[i]`). Enqueues a `PendingOp::Move` on the SAME pipeline FIFO /
/// stream as forward fires (submitted solo/prebuilt), so ordering is automatic
/// (happens-after prior fires' KV writes, happens-before later fires' reads) —
/// no drain barrier (K6 dissolved). The guest's page ids are the PHYSICAL KV
/// block ids it also binds to the `Pages`/`WSlot` ports (Design B works in
/// physical page ids directly), so they pass straight through to the move — the
/// wire / kernel operate on exactly the physical pages the fires read/write.
/// The guest computed the post-move layout itself, so no result is taken.
pub async fn copy_into_inner<C: FireContext>(
    ctx: &mut C,
    this: Resource<Pipeline>,
    ws: Resource<KvWorkingSet>,
    dst_page_ids: Vec<u32>,
    dst_tok_idx: Vec<u32>,
    src_page_ids: Vec<u32>,
    src_tok_idx: Vec<u32>,
) -> Anyhow<Result<(), String>> {
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

    // The WIT contract passes WorkingSet-RELATIVE page indexes (guests
    // never hold physical ids); translate through the flattened table so
    // the move lands on exactly the physical pages the fires read/write.
    // Translated at enqueue against the committed mapping: same-WS
    // in-flight fires that could remap these pages (a CoW rebase) are the
    // guest's ordering hazard, like any same-WS run-ahead write overlap.
    let (kv_move_dst_pages, kv_move_src_pages): (Vec<u32>, Vec<u32>) = {
        let ws = ctx.resources().get(&ws)?.clone();
        let stores = crate::store::registry::get(ws.model, ws.driver as usize);
        let pid = ctx.process_id();
        if let Err(owner) = ws.claim_pipeline_scope(pid.as_u128()) {
            return Ok(Err(format!(
                "pipeline: KV working set is already scoped to pipeline {owner:032x}"
            )));
        }
        let mut kv_store = stores.kv.lock().unwrap();
        let (_, flat) = kv_store
            .flat_table(ws.id)
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
            (Ok(d), Ok(s)) => (d, s),
            (Err(e), _) | (_, Err(e)) => return Ok(Err(e)),
        }
    };

    // Point this pipeline's FIFO at the move so a later `take`/`read` on any
    // channel fed by this pipeline drains it in submit order.
    let pipe_fires = ctx.resources().get(&this)?.fires.clone();
    let pipeline_failure = ctx.resources().get(&this)?.failure.clone();
    if let Some(reason) = pipeline_failure.lock().unwrap().clone() {
        return Ok(Err(format!("pipeline: pipeline failed: {reason}")));
    }

    let cells = kv_move_dst_pages
        .into_iter()
        .zip(dst_tok_idx.into_iter())
        .zip(kv_move_src_pages.into_iter().zip(src_tok_idx.into_iter()))
        .map(
            |((dst_page_id, dst_token_offset), (src_page_id, src_token_offset))| {
                pie_driver_abi::PieKvMoveCell {
                    dst_page_id,
                    dst_token_offset,
                    src_page_id,
                    src_token_offset,
                }
            },
        )
        .collect::<Vec<_>>();
    let completion = match crate::scheduler::copy_kv_cells(0, cells) {
        Ok(completion) => completion,
        Err(e) => return Ok(Err(format!("pipeline copy_into: submit failed: {e:#}"))),
    };
    pipe_fires
        .lock()
        .unwrap()
        .push_back(PendingOp::Move(PendingMove {
            completion,
            failure: pipeline_failure,
        }));
    Ok(Ok(()))
}

pub async fn pipeline_close<C: FireContext>(ctx: &mut C, this: Resource<Pipeline>) -> Anyhow<()> {
    // Signal no further submissions: drain the pipeline's in-flight FIFO so
    // its fires' prepared KV/RS writes (snapshot pins) finalize before the
    // pipeline goes away (the pin-safety drain follows the FIFO — W3.1).
    let fires = ctx.resources().get(&this).ok().map(|p| p.fires.clone());
    if let Some(fires) = fires {
        for fire in fires.lock().unwrap().iter() {
            fire.request_cancel();
        }
        loop {
            let fire = fires.lock().unwrap().pop_front();
            match fire {
                Some(f) => {
                    let _ = finalize_op(ctx, f).await;
                }
                None => break,
            }
        }
    }
    Ok(())
}

pub async fn pipeline_drop<C: FireContext>(ctx: &mut C, this: Resource<Pipeline>) -> Anyhow<()> {
    // Drain the pipeline's in-flight FIFO before releasing it: each fire
    // holds a prepared KV/RS write (snapshot pins, pending slots) and the
    // GPU may still be writing — await + finalize, never abandon
    // mid-flight (W3.1: the pin-safety drain lives here, not on the pass).
    let fires = ctx.resources().get(&this).ok().map(|p| p.fires.clone());
    if let Some(fires) = fires {
        for fire in fires.lock().unwrap().iter() {
            fire.request_cancel();
        }
        loop {
            let fire = fires.lock().unwrap().pop_front();
            match fire {
                Some(f) => {
                    let _ = finalize_op(ctx, f).await;
                }
                None => break,
            }
        }
    }
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

/// Drain one pipeline FIFO entry in submit order: a forward fire finalizes
/// its KV/RS txns and exposes mirror epochs; a KV cell MOVE awaits its
/// payload-free completion. Move failures are logged because no channel is
/// associated with the operation.
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
    let mut drained = false;
    loop {
        let op = {
            let mut queue = fires.lock().unwrap();
            if queue.front().is_some_and(PendingOp::is_settled) {
                queue.pop_front()
            } else {
                None
            }
        };
        match op {
            Some(op) => {
                finalize_op(ctx, op).await?;
                drained = true;
            }
            None => return Ok(drained),
        }
    }
}

pub async fn finalize_op<C: FireContext>(ctx: &mut C, op: PendingOp) -> Anyhow<()> {
    match op {
        PendingOp::Fire(fire) => finalize_fire(ctx, fire).await,
        PendingOp::Move(mv) => {
            if let Err(e) = mv.completion.await {
                let reason = format!("pipeline kv-move (copy_into) failed: {e:#}");
                let mut failure = mv.failure.lock().unwrap();
                if failure.is_none() {
                    *failure = Some(reason.clone());
                }
                tracing::warn!("{reason}");
            }
            Ok(())
        }
    }
}

/// Finalize an ordinary host-geometry op without a `ResourceTable` borrow.
/// Used by accessor-based long host awaits after the scheduler freeze barrier.
/// Device-geometry fires are excluded because their lease reclamation lives on
/// the `ForwardPass` resource and still requires `FireContext`.
pub(crate) async fn finalize_op_detached(op: PendingOp) -> Anyhow<()> {
    match op {
        PendingOp::Move(mv) => {
            if let Err(error) = mv.completion.await {
                let reason = format!("pipeline kv-move (copy_into) failed: {error:#}");
                let mut failure = mv.failure.lock().unwrap();
                if failure.is_none() {
                    *failure = Some(reason.clone());
                }
                tracing::warn!("{reason}");
            }
            Ok(())
        }
        PendingOp::Fire(fire) => {
            let PendingFire {
                completion,
                kv,
                rstxns,
                ws_guard,
                ws_rep: _,
                rs_reps: _,
                fwd_rep: _,
                instance_id: _,
                cells,
                failure,
            } = fire;
            let FireKv::Deferred(slot) = kv else {
                unreachable!("device-geometry fires require FireContext finalization");
            };
            let prior_failure = failure.lock().unwrap().clone();
            let result = completion.await;
            let success = result.is_ok() && prior_failure.is_none();
            let stores = crate::store::registry::get(0, 0);
            if let Some(kvtxn) = slot.lock().unwrap().take() {
                let mut kv_store = stores.kv.lock().unwrap();
                let _ = kv::finalize(&mut kv_store, kvtxn, success);
            }
            let rs_failure = if rstxns.is_empty() {
                None
            } else {
                let mut rs_store = stores.rs.lock().unwrap();
                rs::finalize_many(&mut rs_store, rstxns, success)
                    .err()
                    .map(|error| format!("pipeline: recurrent-state finalize failed: {error}"))
            };
            if let Some(orchestrator) = crate::store::reclaim::contention() {
                orchestrator.on_blocks_freed();
                orchestrator.on_fire_retired();
            }
            let reason = prior_failure
                .or_else(|| {
                    result
                        .err()
                        .map(|error| format!("pipeline: forward failed: {error:#}"))
                })
                .or(rs_failure);
            if let Some(reason) = reason {
                poison_readers(&cells, &reason);
                let mut domain = failure.lock().unwrap();
                if domain.is_none() {
                    *domain = Some(reason);
                }
            }
            drop(ws_guard);
            Ok(())
        }
    }
}

/// Resolve one in-flight fire: await the payload-free callback, finalize the
/// KV/RS txns, and expose the release-published mirror tails. Values remain
/// in driver-owned channel memory until `channel.take` or `channel.read`.
async fn finalize_fire<C: FireContext>(ctx: &mut C, fire: PendingFire) -> Anyhow<()> {
    let PendingFire {
        completion,
        kv,
        rstxns,
        ws_guard,
        ws_rep,
        rs_reps,
        fwd_rep,
        instance_id,
        cells,
        failure,
    } = fire;
    let prior_failure = failure.lock().unwrap().clone();
    let result = completion.await;
    let success = result.is_ok() && prior_failure.is_none();

    let rs_failure = {
        // Single-model, single-driver v1: fires are keyed to stores (0,0).
        let _ = ws_rep;
        let _ = rs_reps;
        let stores = crate::store::registry::get(0, 0);
        match kv {
            FireKv::DeviceGeom { kvtxn } => {
                let mut kv_store = stores.kv.lock().unwrap();
                let _ = kv::finalize(&mut kv_store, kvtxn, success);
            }
            FireKv::Deferred(slot) => {
                if let Some(kvtxn) = slot.lock().unwrap().take() {
                    let mut kv_store = stores.kv.lock().unwrap();
                    let _ = kv::finalize(&mut kv_store, kvtxn, success);
                }
            }
        }
        if rstxns.is_empty() {
            None
        } else {
            let mut rs_store = stores.rs.lock().unwrap();
            rs::finalize_many(&mut rs_store, rstxns, success)
                .err()
                .map(|error| format!("pipeline: recurrent-state finalize failed: {error}"))
        }
    }; // store locks released before the contention drain re-locks pools

    // The fire's sequence retired: recycled slots (aborts, CoW'd tails,
    // collected suffixes) are allocatable now — wake parked allocators
    // and drain-gated lanes.
    if let Some(orch) = crate::store::reclaim::contention() {
        orch.on_blocks_freed();
        orch.on_fire_retired();
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
        .or(rs_failure);
    if let Some(reason) = failure_reason {
        poison_readers(&cells, &reason);
        fail_pass(ctx, fwd_rep, &reason);
        let mut domain = failure.lock().unwrap();
        if domain.is_none() {
            *domain = Some(reason);
        }
    } else {
        reclaim_device_geometry_grants(ctx, fwd_rep, instance_id);
    }
    drop(ws_guard);
    Ok(())
}

/// Mark a pass failed (first failure wins). The guest may have dropped
/// the pass handle already — then there is nothing to mark.
fn fail_pass<C: FireContext>(ctx: &mut C, fwd_rep: u32, reason: &str) {
    let res: Resource<ForwardPass> = Resource::new_borrow(fwd_rep);
    if let Ok(p) = ctx.resources().get_mut(&res) {
        if p.failed.is_none() {
            p.failed = Some(reason.to_string());
        }
    }
}

/// Device-geometry fire (Track B): the pass's [B,P] geometry is
/// DEVICE-produced (the program traces `page_indptr = CumSum(np)` + packed
/// live pages in-graph) and the driver resolves it pre-forward, so the host
/// neither replays the epilogue arithmetic nor projects per-lane KV. The
/// runtime leases `B` fresh physical pages, delivers them to the program as a
/// host-put on the `fresh` channel, marks the fire solo/prebuilt via
/// `map_geometry_relaxed` (wire fields empty), and fires it RUN-AHEAD onto the
/// pipeline FIFO (unlike the deleted synchronous host-replay beam branch).
/// The per-fire arena/write txns ride the `PendingFire`; `finalize_fire`
/// commits/aborts them and reclaims continuing heirs' unused grants (w_cont).
///
/// BRING-UP (4090, shadow-verify): the exact fresh-page materialization
/// (`cow_write_slot`), the fire-0 seed source, and the physical-page ids fed
/// on `fresh` are validated against the beam goldens on device; the geometry
/// contract itself is host-verified (`ptir-dsl` `beam_designb_goldens`).
async fn fire_device_geometry<C: FireContext>(
    ctx: &mut C,
    this: Resource<Pipeline>,
    fwd: Resource<ForwardPass>,
) -> Anyhow<Result<(), String>> {
    // Wire each of this pass's channels at this pipeline's FIFO (§3.4: all
    // passes binding a channel must submit on ONE pipeline — the entire
    // ordering/FIFO correctness argument).
    let pipe_fires = ctx.resources().get(&this)?.fires.clone();
    let pipeline_failure = ctx.resources().get(&this)?.failure.clone();
    // Non-blocking settlement drain (plan §6), as in the ordinary submit.
    drain_settled(ctx, Some(&pipe_fires)).await?;
    if let Some(reason) = pipeline_failure.lock().unwrap().clone() {
        return Ok(Err(format!("pipeline: pipeline failed: {reason}")));
    }
    if let Err(e) = wire_channels_to_pipeline(ctx, &fwd, &pipe_fires)? {
        return Ok(Err(e));
    }
    if !ctx.resources().get(&fwd)?.rs_ws.is_empty() {
        drain_rs_predecessors(ctx, &pipe_fires).await?;
        if let Some(reason) = pipeline_failure.lock().unwrap().clone() {
            return Ok(Err(format!("pipeline: pipeline failed: {reason}")));
        }
    }

    let (ws_rep, rs_reps) = {
        let pass = ctx.resources().get(&fwd)?;
        (pass.kv_ws, pass.rs_ws.clone())
    };
    let ws_res: Resource<KvWorkingSet> = Resource::new_borrow(ws_rep);
    let ws = ctx.resources().get(&ws_res)?.clone();
    let page_size = ws.page_size;
    let stores = crate::store::registry::get(ws.model, ws.driver as usize);
    let pid = ctx.process_id();
    if let Err(owner) = ws.claim_pipeline_scope(pid.as_u128()) {
        return Ok(Err(format!(
            "pipeline: KV working set is already scoped to pipeline {owner:032x}"
        )));
    }
    let ws_guard = match ws.fire_lease() {
        Ok(lease) => lease,
        Err(error) => return Ok(Err(format!("pipeline: KV working set: {error}"))),
    };

    // Grant B pages, allocating their physical backing under one prepared
    // write (no borrow held across submit).
    let (
        completion,
        instance_id,
        cells,
        fwd_rep,
        kvtxn,
        kv_translation_version,
        kv_translation,
        wire_pages,
        copy_src,
        copy_dst,
        dense_mask,
        accesses,
        rs_slot_ids,
        rs_slot_flags,
        rs_copy_src,
        rs_copy_dst,
        resolved_qo_indptr,
        rstxns,
    ) = {
        // Fail-fast.
        {
            let p = ctx.resources().get_mut(&fwd)?;
            if let Some(e) = &p.failed {
                return Ok(Err(format!(
                    "pipeline: forward-pass failed by an earlier fire: {e}"
                )));
            }
        }

        // Take the DevGeo out so the lease grant can use the store
        // (distinct table resources can't be borrowed mutably at once).
        let mut devgeo: DevGeo = ctx
            .resources()
            .get_mut(&fwd)?
            .devgeo
            .take()
            .expect("fire_device_geometry on a non-device-geometry pass");

        // Grant B slots: lease free-list first, then fresh logical
        // reserves. Purely logical — this can never exhaust the pool, so
        // it runs ONCE; only the physical prepare below retries under
        // contention.
        let grant_slots = {
            let mut kv_store = stores.kv.lock().unwrap();
            devgeo.lease.grant(|| {
                kv_store
                    .reserve(ws.id, 1)
                    .map(|r| r.start as u32)
                    .unwrap_or(0)
            })
        };
        let mut allocation_grant: Option<crate::store::reclaim::AllocationGrant> = None;
        let (pages, (copy_src, copy_dst), kv_translation, kvtxn) = loop {
            if allocation_grant.is_none()
                && let Some(orchestrator) = crate::store::reclaim::contention()
                && orchestrator.allocation_requires_grant()
            {
                let required = (|| -> Result<usize, kv::KvError> {
                    let mut kv_store = stores.kv.lock().unwrap();
                    let mapped = kv_store
                        .visible_mapped_len(ws.id)
                        .map_err(kv::KvError::from)?;
                    let mut write_indexes: Vec<u64> =
                        grant_slots.iter().map(|&slot| u64::from(slot)).collect();
                    if let Some(max_fresh) =
                        write_indexes.iter().copied().filter(|&i| i >= mapped).max()
                    {
                        write_indexes.extend(mapped..max_fresh);
                    }
                    write_indexes.sort_unstable();
                    write_indexes.dedup();
                    kv_store
                        .required_pages(ws.id, &write_indexes)
                        .map_err(kv::KvError::from)
                })();
                let required = match required {
                    Ok(required) => required,
                    Err(error) => {
                        ctx.resources().get_mut(&fwd)?.devgeo = Some(devgeo);
                        return Ok(Err(format!(
                            "pipeline: device-geometry grant sizing: {error}"
                        )));
                    }
                };
                if required > 0 {
                    let reclaimable =
                        reclaimable_probe(ws.model, ws.driver as usize, pid, ctx.kv_working_sets());
                    match orchestrator
                        .acquire_or_self_suspend_live(pid, required as u32, reclaimable)
                        .await
                    {
                        Ok(crate::store::reclaim::Acquired::Granted(grant)) => {
                            allocation_grant = Some(grant);
                        }
                        Ok(crate::store::reclaim::Acquired::SelfSuspendFirst) => {
                            if let Err(error) = ctx.honor_preemption().await {
                                ctx.resources().get_mut(&fwd)?.devgeo = Some(devgeo);
                                return Err(error);
                            }
                        }
                        Err(error) => {
                            ctx.resources().get_mut(&fwd)?.devgeo = Some(devgeo);
                            return Ok(Err(format!("pipeline: device-geometry grant: {error}")));
                        }
                    }
                    continue;
                }
            }
            let prepared = (|| {
                let mut kv_store = stores.kv.lock().unwrap();
                // One prepared write covers the grants plus any unwritten
                // gap up to the mapped end (left by an aborted earlier
                // fire), so fresh publication stays contiguous. The gap
                // is recomputed per attempt: a parked retry may resume
                // after another of this pass's fires committed.
                let mapped = kv_store
                    .visible_mapped_len(ws.id)
                    .map_err(kv::KvError::from)?;
                let mut write_indexes: Vec<u64> = grant_slots.iter().map(|&s| s as u64).collect();
                if let Some(max_fresh) =
                    write_indexes.iter().copied().filter(|&i| i >= mapped).max()
                {
                    write_indexes.extend(mapped..max_fresh);
                }
                write_indexes.sort_unstable();
                write_indexes.dedup();
                match allocation_grant.take() {
                    Some(grant) => kv::prepare_explicit_granted(
                        &mut kv_store,
                        ws.id,
                        &write_indexes,
                        grant.into_pages(),
                    ),
                    None => kv::prepare_explicit(&mut kv_store, ws.id, &write_indexes),
                }
            })(); // kv lock dropped before any contention await below
            match prepared {
                Ok(v) => break v,
                Err(e @ kv::KvError::OutOfPages { requested, .. }) => {
                    // Same contention-ladder seam as the host-geometry
                    // path (see submit_pass), including the Error-mode
                    // inline rung 1.
                    let Some(orch) = crate::store::reclaim::contention() else {
                        let freed = {
                            let mut kv_store = stores.kv.lock().unwrap();
                            let epoch = kv_store.current_epoch();
                            let freed = kv_store.drop_unused_cache_leases(epoch);
                            kv_store.retire_idle();
                            freed
                        };
                        if freed > 0 {
                            continue;
                        }
                        ctx.resources().get_mut(&fwd)?.devgeo = Some(devgeo);
                        return Ok(Err(format!("pipeline: device-geometry grant: {e}")));
                    };
                    let reclaimable =
                        reclaimable_probe(ws.model, ws.driver as usize, pid, ctx.kv_working_sets());
                    match orch
                        .acquire_or_self_suspend_live(pid, requested as u32, reclaimable)
                        .await
                    {
                        Ok(crate::store::reclaim::Acquired::Granted(grant)) => {
                            allocation_grant = Some(grant)
                        }
                        Ok(crate::store::reclaim::Acquired::SelfSuspendFirst) => {
                            if let Err(error) = ctx.honor_preemption().await {
                                ctx.resources().get_mut(&fwd)?.devgeo = Some(devgeo);
                                return Err(error);
                            }
                        }
                        Err(hard) => {
                            ctx.resources().get_mut(&fwd)?.devgeo = Some(devgeo);
                            return Ok(Err(format!("pipeline: device-geometry grant: {hard}")));
                        }
                    }
                }
                Err(e) => {
                    ctx.resources().get_mut(&fwd)?.devgeo = Some(devgeo);
                    return Ok(Err(format!("pipeline: device-geometry grant: {e}")));
                }
            }
        };
        // Device geometry resolves B request rows in-graph. Validate the bound
        // RS list against that resolved arity before the launch and prepare one
        // folded target per row. The zero-valued `qo_indptr` carries only the
        // known row count; the driver still resolves its values in-graph.
        let resolved_qo_indptr = vec![0; devgeo.b + 1];
        let prepared_rs = prepare_bound_rs(
            ctx,
            &stores,
            ws.model,
            ws.driver as usize,
            &rs_reps,
            &resolved_qo_indptr,
            Arc::as_ptr(&pipe_fires) as usize,
        );
        let (rs_slot_ids, rs_slot_flags, rs_copy_src, rs_copy_dst, rstxns) = match prepared_rs {
            Ok(Ok(prepared)) => prepared,
            outcome => {
                {
                    let mut kv_store = stores.kv.lock().unwrap();
                    let _ = kv::finalize(&mut kv_store, kvtxn, false);
                }
                devgeo.lease.reclaim_after_fire(&vec![true; devgeo.b]);
                ctx.resources().get_mut(&fwd)?.devgeo = Some(devgeo);
                return match outcome {
                    Ok(Err(error)) => Ok(Err(error)),
                    Err(error) => Err(error),
                    Ok(Ok(_)) => unreachable!(),
                };
            }
        };
        // Wire page refs (scheduler capacity accounting): this fire's
        // physical write targets.
        let wire_pages: Vec<u32> = pages.iter().map(|&(_, dst)| dst).collect();
        // Deliver the fresh grant to the program as a direct put on its
        // `fresh` channel — a shared-ring write the driver pulls before
        // the pass (plan §4.2/§4.3). The grants are WorkingSet-RELATIVE
        // indexes — the program's in-graph geometry stays logical
        // end-to-end and the driver translates its resolved
        // `Pages`/`WSlot` values through `kv_translation` (which the
        // prepared write above backs physically). This also matches the
        // bind-time seed (`reserve(b)`), which was already logical.
        let fresh_dense = devgeo.fresh_dense;
        let dense_mask = devgeo.has_mask;
        let fresh_error = {
            let p = ctx.resources().get_mut(&fwd)?;
            let bytes: Vec<u8> = grant_slots.iter().flat_map(|s| s.to_le_bytes()).collect();
            let error = match p.cells.get(fresh_dense) {
                Some(cell) => cell.lock().unwrap().put(bytes).err(),
                None => Some(ChannelError::Empty),
            };
            p.devgeo = Some(devgeo);
            error
        };
        if let Some(error) = fresh_error {
            abort_rs_transactions(&stores, rstxns);
            let mut kv_store = stores.kv.lock().unwrap();
            let _ = kv::finalize(&mut kv_store, kvtxn, false);
            return Ok(Err(format!(
                "pipeline: device-geometry fresh grant put: {error}"
            )));
        }

        let p = ctx.resources().get_mut(&fwd)?;
        let completion = p.bound_instance.reserve_completion();
        let accesses = p.instance.program.channel_accesses.clone();
        let kv_translation_version = kvtxn.mapping_version();
        (
            completion,
            p.bound_instance.instance_id,
            p.cells.clone(),
            fwd.rep(),
            kvtxn,
            kv_translation_version,
            kv_translation,
            wire_pages,
            copy_src,
            copy_dst,
            dense_mask,
            accesses,
            rs_slot_ids,
            rs_slot_flags,
            rs_copy_src,
            rs_copy_dst,
            resolved_qo_indptr,
            rstxns,
        )
    };

    let mut req = crate::driver::LaunchPlan::default();
    req.qo_indptr = resolved_qo_indptr;
    req.kv_translation = kv_translation;
    req.kv_translation_version = kv_translation_version;
    req.rs_slot_ids = rs_slot_ids;
    req.rs_slot_flags = rs_slot_flags;
    // A dense device mask (AttnMask channel) marks the fire mask-carrying;
    // the scheduler batches it SOLO (the composed multi-program batch
    // does not merge dense device masks — v1 scope).
    req.has_user_mask = dense_mask;
    let ticket_reservation = TicketReservation::new(&cells, &accesses);
    ticket_reservation.apply_to(&mut req);
    let last_page_len = wire_pages.last().map(|_| page_size).unwrap_or(0);
    let submit_error = crate::scheduler::submit_prebuilt_tracked_async_with_kv_and_rs_copy(
        req,
        0,
        instance_id,
        pid,
        wire_pages,
        last_page_len,
        completion.clone(),
        copy_src,
        copy_dst,
        rs_copy_src,
        rs_copy_dst,
    )
    .err()
    .map(|error| format!("{error:#}"));
    if let Some(error) = submit_error {
        let reason = format!("pipeline: device-geometry submit failed: {error}");
        abort_rs_transactions(&stores, rstxns);
        {
            let mut kv_store = stores.kv.lock().unwrap();
            let _ = kv::finalize(&mut kv_store, kvtxn, false);
        }
        ctx.resources().get_mut(&fwd)?.failed = Some(reason.clone());
        return Ok(Err(reason));
    }
    ticket_reservation.commit();
    ctx.resources().get_mut(&fwd)?.fired_once = true;

    pipe_fires
        .lock()
        .unwrap()
        .push_back(PendingOp::Fire(PendingFire {
            completion,
            kv: FireKv::DeviceGeom { kvtxn },
            rstxns,
            ws_guard,
            ws_rep,
            rs_reps,
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
    if let Some(existing) = &ctx.resources().get(fwd)?.fires {
        if !Arc::ptr_eq(existing, pipe_fires) {
            return Ok(Err(
                "pipeline: a pass cannot submit across different pipelines".into(),
            ));
        }
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
    let (ws_rep, reclaimed) = {
        let Ok(p) = ctx.resources().get_mut(&res) else {
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
        let reclaimed = devgeo.lease.reclaim_after_fire(&w_cont);
        (p.kv_ws, reclaimed)
    };
    // Reclaimed grants return to the lease free-list (done above) and are
    // re-granted to later fires; the store mapping keeps their committed
    // pages until the working set discards or drops them (a discard here
    // would shift live indexes under the pass — see the pass-drop note).
    let _ = (ws_rep, reclaimed);
}
