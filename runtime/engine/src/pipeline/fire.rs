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
//! (`PendingForward`, Option A) applied to this engine. Pure-attention fires
//! read a standing WorkingSet translation; RS step t+1 waits for t's committed
//! folded mapping.
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
pub mod shadow;

use std::collections::VecDeque;
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
}

pub type PendingFires = Arc<PendingFireQueue>;
pub type PipelineFailure = Arc<Mutex<Option<String>>>;

struct CopyCompletionGuard {
    completion: Option<crate::driver::SubmissionCompletion>,
    lease: Option<KvFireLease>,
    model: usize,
    driver: usize,
    ws: crate::store::kv::page_table::WorkingSetId,
    indexes: Vec<u32>,
}

impl CopyCompletionGuard {
    fn invalidate(
        model: usize,
        driver: usize,
        ws: crate::store::kv::page_table::WorkingSetId,
        indexes: &[u32],
    ) {
        let stores = crate::store::registry::get(model, driver);
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
        Self::invalidate(self.model, self.driver, self.ws, &self.indexes);
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
        let driver = self.driver;
        let ws = self.ws;
        let indexes = std::mem::take(&mut self.indexes);
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            Self::invalidate(model, driver, ws, &indexes);
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
            Self::invalidate(model, driver, ws, &indexes);
            drop(lease);
        });
    }
}

type PreparedExplicitKv = (Vec<(u64, u32)>, (Vec<u32>, Vec<u32>), Vec<u32>, kv::KvTxn);

fn reclaim_cache_roots(stores: &crate::store::registry::Stores) -> usize {
    crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv_store| {
        let epoch = kv_store.current_epoch();
        let reclaimed = kv_store.drop_unused_cache_leases(epoch);
        kv_store.retire_idle();
        reclaimed
    })
}

fn realize_host_declaration(
    stores: &crate::store::registry::Stores,
    ws: &KvWorkingSet,
    writable: std::ops::Range<u64>,
) -> Result<((Vec<u32>, Vec<u32>), Option<kv::KvTxn>), String> {
    let realize = || {
        crate::store::registry::with_kv_lock(&stores.kv, "host-other", |store| {
            kv::realize_declaration(store, ws.id, writable.clone())
        })
    };
    match realize() {
        Ok(realized) => Ok(realized),
        Err(kv::KvError::OutOfPages { .. }) if reclaim_cache_roots(stores) != 0 => {
            realize().map_err(|error| format!("pipeline: KV declaration realization: {error}"))
        }
        Err(error) => Err(format!("pipeline: KV declaration realization: {error}")),
    }
}

fn ensure_host_backing(
    stores: &crate::store::registry::Stores,
    ws: &KvWorkingSet,
    end: u64,
) -> Result<(), String> {
    let ensure = || {
        crate::store::registry::with_kv_lock(&stores.kv, "host-other", |store| {
            store.ensure_backed(ws.id, end)
        })
    };
    match ensure() {
        Ok(_) => Ok(()),
        Err(crate::store::kv::KvStoreError::OutOfPages { .. })
            if reclaim_cache_roots(stores) != 0 =>
        {
            ensure()
                .map(|_| ())
                .map_err(|error| format!("pipeline: KV backing frontier: {error}"))
        }
        Err(error) => Err(format!("pipeline: KV backing frontier: {error}")),
    }
}

fn prepare_explicit_kv(
    stores: &crate::store::registry::Stores,
    ws: &KvWorkingSet,
    write_indexes: &[u64],
) -> Result<PreparedExplicitKv, String> {
    let prepare = || {
        crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv_store| {
            kv::prepare_explicit(kv_store, ws.id, write_indexes)
        })
    };
    match prepare() {
        Ok(prepared) => Ok(prepared),
        Err(kv::KvError::OutOfPages { .. }) => {
            let reclaimed =
                crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv_store| {
                    let epoch = kv_store.current_epoch();
                    let reclaimed = kv_store.drop_unused_cache_leases(epoch);
                    kv_store.retire_idle();
                    reclaimed
                });
            if reclaimed == 0 {
                return Err(
                    "pipeline: in-lease device-geometry allocation failed; reservation accounting is inconsistent"
                        .to_string(),
                );
            }
            prepare().map_err(|error| format!("pipeline: device-geometry grant: {error}"))
        }
        Err(error) => Err(format!("pipeline: device-geometry grant: {error}")),
    }
}

/// A pipeline FIFO entry: a forward FIRE or a KV cell MOVE (Design-B
/// compaction). Both hold an ordered slot on the same stream — the B3
/// happens-before invariant; `take`/`read` drain them in submit order.
pub enum PendingOp {
    Fire(PendingFire),
    #[cfg(test)]
    TestStub,
}

impl PendingOp {
    /// Non-blocking probe: whether the op's completion has settled.
    fn is_settled(&self) -> bool {
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

    pub(crate) fn is_preemption_safe_unprepared(&self) -> bool {
        false
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

    pub(crate) fn preemption_signal(
        &self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> {
        Box::pin(self.completion_signal())
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
    Fire(crate::driver::WorkItemCompletion),
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
/// device-geometry fire whose KV the driver resolves+writes itself (B2's
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
/// finalize when it resolves — the open KV/RS txns (pins/CoW held until
/// commit/abort) and the bound cells whose mirror epochs become visible.
pub struct PendingFire {
    completion: crate::driver::WorkItemCompletion,
    kv: FireKv,
    rstxns: Vec<rs::RsTxn>,
    ws_guard: KvFireLease,
    model: usize,
    driver: usize,
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
    pipeline_scope: &crate::store::PipelineScope,
) -> Anyhow<Result<PreparedRs, String>> {
    let has_recurrent_state = pie_model::model().rs_caps().state_size > 0;
    if let Err(error) = rs::validate_count(rs_reps.len(), qo_indptr, has_recurrent_state) {
        return Ok(Err(format!("pipeline: recurrent-state binding: {error}")));
    }
    if rs_reps.is_empty() {
        return Ok(Ok((
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )));
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
        let timing_enabled = ctx.fire_timing_requested();
        let (
            geometry,
            cells,
            ws_rep,
            rs_reps,
            kv_declaration,
            kv_declaration_realized,
            fwd_rep,
            instance_id,
            scheduler,
            completion,
            dense_mask,
            accesses,
            decode_envelope,
        ) = {
            let p = ctx.resources().get_mut(&fwd)?;
            if let Some(e) = &p.failed {
                return Ok(Err(format!(
                    "pipeline: forward-pass failed by an earlier fire: {e}"
                )));
            }
            let geometry = if let Some(envelope) = &p.decode_envelope {
                match envelope.template(&p.instance.program.bound.container) {
                    Ok(template) => template,
                    Err(error) => {
                        return Ok(Err(format!("pipeline: fire geometry: {error:?}")));
                    }
                }
            } else {
                let bound = &p.instance.program.bound;
                let (shadow, shadow_cells) = (&p.host_shadow, &p.cells);
                let mut known = |chan: u32| shadow.fire_value(bound, shadow_cells, chan);
                match geometry::map_geometry_evaluated(
                    bound,
                    &mut known,
                    crate::pipeline::program::model_profile().page_size,
                ) {
                    Ok((geometry, _)) => {
                        // In-band -1 skips are the DEVICE-resolved contract
                        // (rank compaction happens in the compose kernels); a
                        // host-wire fire would embed the sentinel as a real
                        // token. Loud rejection, never silent execution
                        // (RV-12).
                        if geometry.token_ids.contains(&u32::MAX) {
                            return Ok(Err(
                                "pipeline: fire geometry: in-band -1 skip tokens require a \
                                 device-resolved geometry class; this fire resolved on the \
                                 host wire"
                                    .to_string(),
                            ));
                        }
                        geometry
                    }
                    Err(error) => {
                        return Ok(Err(format!("pipeline: fire geometry: {error}")));
                    }
                }
            };
            let accesses = p.instance.program.channel_accesses.clone();
            (
                geometry,
                p.cells.clone(),
                p.kv_ws,
                p.rs_ws.clone(),
                p.kv_declaration,
                p.kv_declaration_realized,
                fwd.rep(),
                p.bound_instance.instance_id,
                p.scheduler.clone(),
                p.bound_instance.reserve_completion(),
                p.dense_mask,
                accesses,
                p.decode_envelope.clone(),
            )
        };
        let mut req = crate::driver::LaunchPlan::default();
        geometry.apply_to(&mut req);
        req.device_resolved_geometry = decode_envelope.is_some();
        req.single_token_mode = req.token_ids.len() + 1 == req.qo_indptr.len()
            && req.qo_indptr.windows(2).all(|lane| lane[1] - lane[0] == 1);
        // A dense device mask (AttnMask channel) marks the fire
        // mask-carrying: the scheduler batches it SOLO (the composed
        // multi-program batch does not merge dense device masks — v1
        // scope).
        req.has_user_mask = dense_mask;
        crate::pipeline::offload::try_encode(&mut req).await;
        // Resource preparation is independent of token position: realize the
        // declaration once, back only its missing frontier, then snapshot the
        // WorkingSet translation.
        let ws_res: Resource<KvWorkingSet> = Resource::new_borrow(ws_rep);
        let ws = ctx.resources().get(&ws_res)?.clone();
        let stores = crate::store::registry::get(ws.model, ws.driver as usize);
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
        let pid = ctx.process_id();
        if let Err(owner) = ws.claim_pipeline_scope(&pipeline_scope) {
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
            &pipeline_scope,
        )? {
            Ok(prepared) => prepared,
            Err(error) => return Ok(Err(error)),
        };
        req.rs_slot_ids = rs_slot_ids;
        req.rs_slot_flags = rs_slot_flags;

        if writable_pages.is_empty() {
            abort_rs_transactions(&stores, rstxns);
            return Ok(Err(
                "pipeline: writable KV page declaration is empty".to_string()
            ));
        }
        if decode_envelope.is_none() {
            if let Some(&page) = req
                .kv_page_indices
                .iter()
                .find(|&&page| !readable_pages.contains(&u64::from(page)))
            {
                abort_rs_transactions(&stores, rstxns);
                return Ok(Err(format!(
                    "pipeline: KV read page {page} escapes the readable declaration"
                )));
            }
        }
        let page_size = u64::from(ws.page_size);
        req.kv_write_lower_bounds = vec![writable_pages.start * page_size];
        req.kv_write_upper_bounds = vec![writable_pages.end * page_size];
        let model = ws.model;
        let driver = ws.driver as usize;
        let ((copy_src, copy_dst), kvtxn) = if kv_declaration_realized {
            ((Vec::new(), Vec::new()), None)
        } else {
            match realize_host_declaration(&stores, &ws, writable_pages.clone()) {
                Ok(realized) => realized,
                Err(error) => {
                    abort_rs_transactions(&stores, rstxns);
                    return Ok(Err(error));
                }
            }
        };
        if let Err(error) = ensure_host_backing(&stores, &ws, writable_pages.end) {
            abort_rs_transactions(&stores, rstxns);
            if let Some(kvtxn) = kvtxn {
                crate::store::registry::with_kv_lock(&stores.kv, "host-other", |store| {
                    let _ = kv::finalize(store, kvtxn, false);
                });
                record_submit_failure(ctx, &fwd, &pipeline_failure, &error);
            }
            return Ok(Err(error));
        }
        let (translation_version, translation) = match ws.translation() {
            Ok(translation) => translation,
            Err(error) => {
                abort_rs_transactions(&stores, rstxns);
                if let Some(kvtxn) = kvtxn {
                    crate::store::registry::with_kv_lock(&stores.kv, "host-other", |store| {
                        let _ = kv::finalize(store, kvtxn, false);
                    });
                }
                let reason = format!("pipeline: KV translation: {error}");
                record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
                return Ok(Err(reason));
            }
        };
        req.kv_translation_version = translation_version;
        req.kv_translation = translation.as_ref().to_vec();
        let last_page_len = req.kv_last_page_lens.last().copied().unwrap_or(0);

        // Preparation is complete in guest order; the scheduler sees only
        // launch-ready work.
        let ticket_reservation = TicketReservation::new(&cells, &accesses);
        ticket_reservation.apply_to(&mut req);
        let retry_cells = cells.clone();
        let retry_accesses = accesses.clone();
        let closed_scope = pipeline_scope.clone();
        let retry_classifier: crate::scheduler::RetryClassifier = Box::new(move || {
            retry_cells
                .iter()
                .zip(&retry_accesses)
                .find_map(|(cell, &(consume, publish))| {
                    let cell = cell.lock().unwrap();
                    cell.permanent_retry_cause(consume || publish).or_else(|| {
                        // Close is the decidability point: a put
                        // still blocked on a full ring with no attached
                        // consumer and no host role can never commit.
                        (publish && closed_scope.is_closed() && cell.is_consumerless_device_ring())
                            .then(|| {
                                format!(
                                    "channel {} put can never commit: the pipeline \
                                 closed with no consumer attached \
                                 (device-only ring, single pass) — a definite \
                                 deadlock",
                                    cell.global_id
                                )
                            })
                    })
                })
        });
        let submit_error = crate::scheduler::submit_prebuilt_tracked_async_with_kv_and_rs_copy_on(
            &scheduler,
            req,
            instance_id,
            pid,
            last_page_len,
            completion.clone(),
            copy_src,
            copy_dst,
            rs_copy_src,
            rs_copy_dst,
            Some(retry_classifier),
            timing_enabled,
        )
        .err()
        .map(|error| format!("{error:#}"));
        if let Some(error) = submit_error {
            let reason = format!("pipeline: submit failed: {error}");
            abort_rs_transactions(&stores, rstxns);
            if let Some(kvtxn) = kvtxn {
                crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv_store| {
                    let _ = kv::finalize(kv_store, kvtxn, false);
                });
            }
            ctx.resources().get_mut(&fwd)?.failed = Some(reason.clone());
            let mut failure = pipeline_failure.lock().unwrap();
            if failure.is_none() {
                *failure = Some(reason.clone());
            }
            return Ok(Err(reason));
        }
        ctx.commit_fire_timing(timing_enabled);
        ticket_reservation.commit();

        {
            let p = ctx.resources().get_mut(&fwd)?;
            let p = p.bound_mut().map_err(anyhow::Error::msg)?;
            p.kv_declaration_realized = true;
            let (shadow, bound, shadow_cells) =
                (&mut p.host_shadow, &p.instance.program.bound, &p.cells);
            shadow.advance(bound, shadow_cells);
        }

        pipe_fires
            .lock()
            .unwrap()
            .push_back(PendingOp::Fire(PendingFire {
                completion,
                kv: FireKv::Host(kvtxn),
                rstxns,
                ws_guard,
                model,
                driver,
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
    drain_rs_predecessors(ctx, &pipe_fires).await?;
    if let Some(reason) = pipeline_failure.lock().unwrap().clone() {
        return Ok(Err(format!("pipeline: pipeline failed: {reason}")));
    }
    let ws_handle = ctx.resources().get(&ws)?.clone();

    // The WIT contract passes WorkingSet-RELATIVE page indexes (guests
    // never hold physical ids); translate through the flattened table so
    // the move lands on exactly the physical pages the fires read/write.
    // Translated at enqueue against the committed mapping: same-WS
    // in-flight fires that could remap these pages (a CoW rebase) are the
    // guest's ordering hazard, like any same-WS run-ahead write overlap.
    let (kv_move_dst_pages, kv_move_src_pages): (Vec<u32>, Vec<u32>) = {
        let stores = crate::store::registry::get(ws_handle.model, ws_handle.driver as usize);
        if let Err(owner) = ws_handle.claim_pipeline_scope(&pipeline_scope) {
            return Ok(Err(format!(
                "pipeline: KV working set is already scoped to pipeline {owner:032x}"
            )));
        }
        let translated = crate::store::registry::with_kv_lock(
            &stores.kv,
            "host-other",
            |kv_store| -> anyhow::Result<Result<(Vec<u32>, Vec<u32>), String>> {
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
    let lease = match ws_handle.fire_lease() {
        Ok(lease) => lease,
        Err(error) => return Ok(Err(format!("pipeline copy_into: {error}"))),
    };
    let completion = match crate::scheduler::copy_kv_cells(0, cells).await {
        Ok(completion) => completion,
        Err(e) => return Ok(Err(format!("pipeline copy_into: submit failed: {e:#}"))),
    };
    let result = CopyCompletionGuard {
        completion: Some(completion),
        lease: Some(lease),
        model: ws_handle.model,
        driver: ws_handle.driver as usize,
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
/// later submissions, releases the scheduler wait-set immediately, then
/// drains every already-submitted FIFO operation to settlement without
/// cancelling it. Finalization leaves committed reader cells available to
/// channel `take` after close returns. Drop is exactly the same lifecycle
/// plus deletion of the WIT resource.
async fn pipeline_close_and_drain<C: FireContext>(
    ctx: &mut C,
    this: &Resource<Pipeline>,
) -> Anyhow<()> {
    let state = ctx.resources().get(this).ok().map(|pipeline| {
        let first_close = pipeline.scope.close();
        (first_close, pipeline.fires.clone())
    });
    if let Some((first_close, fires)) = state {
        if first_close {
            crate::scheduler::worker::notify_pipeline_leave(
                ctx.process_id(),
                crate::scheduler::worker::LeaveKind::Close,
            );
        }
        let _finalize_guard = fires.finalize_guard().await;
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

pub async fn pipeline_close<C: FireContext>(ctx: &mut C, this: Resource<Pipeline>) -> Anyhow<()> {
    pipeline_close_and_drain(ctx, &this).await
}

pub async fn pipeline_drop<C: FireContext>(ctx: &mut C, this: Resource<Pipeline>) -> Anyhow<()> {
    pipeline_close_and_drain(ctx, &this).await?;
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
/// Used by accessor-based long host awaits after the scheduler freeze barrier.
/// Device-geometry fires are excluded because their lease reclamation lives on
/// the `ForwardPass` resource and still requires `FireContext`.
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
/// in driver-owned channel memory until `channel.take` or `channel.read`.
async fn finalize_fire_await(fire: PendingFire) -> Anyhow<FinalizeOutcome> {
    let PendingFire {
        completion,
        kv,
        rstxns,
        ws_guard,
        model,
        driver,
        ws_rep,
        rs_reps,
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
        let _ = ws_rep;
        let _ = rs_reps;
        let stores = crate::store::registry::get(model, driver);
        // RS transactions have no Drop rollback. Retire them before the only
        // await below so process cancellation cannot leak their slots.
        let rs_failure = if rstxns.is_empty() {
            None
        } else {
            let mut rs_store = stores.rs.lock().unwrap();
            rs::finalize_many(&mut rs_store, rstxns, success)
                .err()
                .map(|error| format!("pipeline: recurrent-state finalize failed: {error}"))
        };
        let kv_failure = match kv {
            FireKv::DeviceGeom { kvtxn } => {
                crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv_store| {
                    kv::finalize(kv_store, kvtxn, success)
                        .err()
                        .map(|error| format!("pipeline: KV finalize failed: {error}"))
                })
            }
            FireKv::Host(kvtxn) => kvtxn.and_then(|kvtxn| {
                crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv_store| {
                    kv::finalize(kv_store, kvtxn, success)
                        .err()
                        .map(|error| format!("pipeline: KV finalize failed: {error}"))
                })
            }),
        };
        (kv_failure, rs_failure)
    }; // store locks released before the contention drain re-locks pools

    // The fire's sequence retired: recycled slots (aborts, CoW'd tails,
    // collected suffixes) are allocatable now — wake parked allocators
    // and drain-gated lanes.
    if let Some(orch) = crate::store::reclaim::contention() {
        orch.on_blocks_freed();
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
    if let Ok(p) = ctx.resources().get_mut(&res) {
        if p.failed.is_none() {
            p.failed = Some(reason.to_string());
        }
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
    let timing_enabled = ctx.fire_timing_requested();

    let (ws_rep, rs_reps) = {
        let pass = ctx.resources().get(&fwd)?;
        (pass.kv_ws, pass.rs_ws.clone())
    };
    let ws_res: Resource<KvWorkingSet> = Resource::new_borrow(ws_rep);
    let ws = ctx.resources().get(&ws_res)?.clone();
    let page_size = ws.page_size;
    let stores = crate::store::registry::get(ws.model, ws.driver as usize);
    let pid = ctx.process_id();
    if let Err(owner) = ws.claim_pipeline_scope(&pipeline_scope) {
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
        scheduler,
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
        let grant_slots =
            crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv_store| {
                devgeo.lease.grant(|| {
                    kv_store
                        .reserve(ws.id, 1)
                        .map(|r| r.start as u32)
                        .unwrap_or(0)
                })
            });
        let mut write_indexes: Vec<u64> = grant_slots.iter().map(|&slot| u64::from(slot)).collect();
        write_indexes.sort_unstable();
        write_indexes.dedup();
        let (pages, (copy_src, copy_dst), kv_translation, kvtxn) =
            match prepare_explicit_kv(&stores, &ws, &write_indexes) {
                Ok(prepared) => prepared,
                Err(error) => {
                    ctx.resources().get_mut(&fwd)?.devgeo = Some(devgeo);
                    return Ok(Err(error));
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
            &pipeline_scope,
        );
        let (rs_slot_ids, rs_slot_flags, rs_copy_src, rs_copy_dst, rstxns) = match prepared_rs {
            Ok(Ok(prepared)) => prepared,
            outcome => {
                crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv_store| {
                    let _ = kv::finalize(kv_store, kvtxn, false);
                });
                devgeo.lease.reclaim_after_fire(&vec![true; devgeo.b]);
                ctx.resources().get_mut(&fwd)?.devgeo = Some(devgeo);
                return match outcome {
                    Ok(Err(error)) => {
                        record_submit_failure(ctx, &fwd, &pipeline_failure, &error);
                        Ok(Err(error))
                    }
                    Err(error) => {
                        let reason =
                            format!("pipeline: device-geometry RS prepare failed: {error:#}");
                        record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
                        Err(error)
                    }
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
            crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv_store| {
                let _ = kv::finalize(kv_store, kvtxn, false);
            });
            reclaim_pending_device_grant(ctx, &fwd);
            let reason = format!("pipeline: device-geometry fresh grant put: {error}");
            record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
            return Ok(Err(reason));
        }

        let p = ctx.resources().get_mut(&fwd)?;
        let completion = p.bound_instance.reserve_completion();
        let accesses = p.instance.program.channel_accesses.clone();
        let kv_translation_version = kvtxn.mapping_version();
        (
            completion,
            p.bound_instance.instance_id,
            p.scheduler.clone(),
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
    let submit_error = crate::scheduler::submit_prebuilt_tracked_async_with_kv_and_rs_copy_on(
        &scheduler,
        req,
        instance_id,
        pid,
        last_page_len,
        completion.clone(),
        copy_src,
        copy_dst,
        rs_copy_src,
        rs_copy_dst,
        None,
        timing_enabled,
    )
    .err()
    .map(|error| format!("{error:#}"));
    if let Some(error) = submit_error {
        let reason = format!("pipeline: device-geometry submit failed: {error}");
        abort_rs_transactions(&stores, rstxns);
        crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv_store| {
            let _ = kv::finalize(kv_store, kvtxn, false);
        });
        reclaim_pending_device_grant(ctx, &fwd);
        record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
        return Ok(Err(reason));
    }
    ctx.commit_fire_timing(timing_enabled);
    ticket_reservation.commit();

    pipe_fires
        .lock()
        .unwrap()
        .push_back(PendingOp::Fire(PendingFire {
            completion,
            kv: FireKv::DeviceGeom { kvtxn },
            rstxns,
            ws_guard,
            model: ws.model,
            driver: ws.driver as usize,
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
        let reclaimed = devgeo.lease.reclaim_after_fire(&w_cont);
        (p.kv_ws, reclaimed)
    };
    // Reclaimed grants return to the lease free-list (done above) and are
    // re-granted to later fires; the store mapping keeps their committed
    // pages until the working set discards or drops them (a discard here
    // would shift live indexes under the pass — see the pass-drop note).
    let _ = (ws_rep, reclaimed);
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

        async fn honor_preemption(&mut self) -> anyhow::Result<()> {
            Ok(())
        }

        fn preemption_signal(&self) -> Option<Arc<tokio::sync::Notify>> {
            None
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
