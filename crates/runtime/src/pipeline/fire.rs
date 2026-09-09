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

type PreparedHostKv = kv::RealizedDeclaration;

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

    fn mapping_version(&self) -> Option<u64> {
        self.txn.as_ref().map(kv::KvTxn::mapping_version)
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
            kv::abandon(store, txn);
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
            let page = store
                .geometry(*id)
                .map_err(|e| e.to_string())?
                .buffer_page_tokens
                .max(1);
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
    let mut buffered: Vec<u32> = {
        let store = stores.rs.lock().unwrap();
        let mut out = Vec::with_capacity(rows);
        for (row, id) in ids.iter().enumerate() {
            match store.buffer_tokens(*id) {
                Ok(tokens) => out.push(tokens),
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
            Position::Buffer
        } else if n == b + t {
            if b == 0 {
                Position::Fold
            } else {
                Position::Buffer
            }
        } else {
            Position::Buffer
        };
        kinds.push(here);
    }

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
        Position::Buffer
    };
    let in_forward: Vec<bool> = kinds.iter().map(|k| *k == Position::Fold).collect();
    for (row, forward) in in_forward.iter().enumerate() {
        if *forward {
            fold_tokens[row] = 0;
            buffered[row] = 0;
            row_tokens[row] = 0;
        }
    }

    Ok(match pass {
        Position::Fold => rs::RsPlan::Fold,
        Position::Buffer => rs::RsPlan::Buffer {
            start_tokens: buffered,
            row_tokens,
            fold_tokens,
            in_forward,
        },
        Position::Commit => rs::RsPlan::FoldBuffered {
            fold_len_is_device: false,
            tokens: fold_tokens,
        },
    })
}

fn suppress_defaulted_readout_for_fold(
    req: &mut crate::engine::FireRequest,
    readout_defaulted: bool,
    plan: &rs::RsPlan,
) {
    let replays_buffer = matches!(plan, rs::RsPlan::FoldBuffered { .. });
    if !replays_buffer || !readout_defaulted {
        return;
    }
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

fn prepare_host_kv_reserved(
    stores: &crate::store::registry::Stores,
    ws: &KvWorkingSet,
    writable: std::ops::Range<u64>,
    declaration_realized: bool,
    grant: &mut crate::planner::AllocationGrant,
) -> Result<PreparedHostKv, ReservedError> {
    if stores.context_pages != 0 && writable.end > stores.context_pages {
        let page = u64::from(stores.kv_page_size);
        return Err(ReservedError::Fatal(format!(
            "pipeline: this fire would grow its sequence to {} KV page(s) (up to {} tokens), \
             past the model's max_context of {} tokens ({} page(s)); the sequence is at \
             its ceiling and cannot take another token",
            writable.end,
            writable.end.saturating_mul(page),
            stores.context_pages.saturating_mul(page),
            stores.context_pages,
        )));
    }
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

enum FireKv {
    Host(Option<kv::KvTxn>),
    DeviceGeom { kvtxn: kv::KvTxn },
}

pub struct PendingFire {
    completion: crate::engine::WorkItemCompletion,
    kv: FireKv,
    rstxn: RsTxnsGuard,
    ws_guard: KvFireLease,
    model: usize,
    engine: usize,
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

fn stamp_denoise(
    req: &mut crate::engine::FireRequest,
    payload: Option<crate::pipeline::instance::SelfCondPayload>,
) -> Result<(), String> {
    let wanted: usize = req.lanes.iter().map(|lane| lane.tokens.len()).sum();
    let mut cursor = 0usize;
    for lane in &mut req.lanes {
        lane.bidirectional = true;
        if lane.mask.is_none() {
            lane.mask = Some(::engine::Masking::Extent(::engine::Mask::new(
                vec![0, u32::MAX],
                u64::from(u32::MAX),
            )));
        }
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

fn stamp_lane_words(
    req: &mut crate::engine::FireRequest,
    fire_wide_mask: bool,
    carries_media: bool,
) {
    let model = crate::model::model();
    for lane in &mut req.lanes {
        let rows = u32::try_from(lane.tokens.len()).unwrap_or(u32::MAX);
        lane.word = model.word(
            rows,
            lane.mask.is_some() || fire_wide_mask,
            lane.adapter.is_some(),
            lane.drafts,
            lane.captures_scores,
            carries_media,
            lane.block_draft,
            lane.bidirectional,
            crate::pipeline::instance::stream_of_lane(lane.stream),
            lane.reading,
        );
    }
}

pub(crate) fn stamp_lane_slots(
    req: &mut crate::engine::FireRequest,
    stores: &crate::store::registry::Stores,
    ws: crate::store::kv::page_table::WorkingSetId,
) -> Result<(), crate::store::seat::SeatError> {
    let seats = stores.seats.lock().unwrap().seats(ws, req.lanes.len())?;
    for (lane, &seat) in req.lanes.iter_mut().zip(&seats) {
        lane.slot = seat;
    }
    Ok(())
}

const SEAT_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

pub(crate) async fn seat_lane_slots(
    req: &mut crate::engine::FireRequest,
    stores: &crate::store::registry::Stores,
    ws: crate::store::kv::page_table::WorkingSetId,
) -> Result<(), String> {
    use crate::store::seat::SeatError;
    let deadline = tokio::time::Instant::now() + SEAT_WAIT;
    loop {
        let mut freed = Box::pin(stores.seats_freed.notified());
        freed.as_mut().enable();
        let refusal = match stamp_lane_slots(req, stores, ws) {
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
        let now = tokio::time::Instant::now();
        if now >= deadline {
            return Err(format!(
                "pipeline: seating this fire's lanes: {refusal}; waited {:?} for a peer to \
                 release seats and none did",
                SEAT_WAIT
            ));
        }
        let _ = tokio::time::timeout(deadline - now, freed).await;
    }
}

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

pub(crate) async fn await_channel_progress(
    cell: &Arc<Mutex<crate::pipeline::channel::ChannelCell>>,
    fires: Option<&PendingFires>,
) -> Result<(), String> {
    let wait = cell.lock().unwrap().reader_wait_state();
    let oldest = fires.and_then(|f| f.lock().unwrap().front().map(|op| op.completion_signal()));
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
        ctx.resources().get(&fwd)?.lane.stamp(&mut req);
        let group = ctx.resources().get(&fwd)?.lane.group;
        let peer = ctx.resources().get(&fwd)?.lane.peer;
        req.cohort = crate::pipeline::instance::cohort_of(ctx.resources(), group, peer);
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
        {
            let pass = ctx.resources().get_mut(&fwd)?;
            if pass.bindings.canvas == Some(crate::pipeline::instance::CanvasMode::Denoise) {
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
                if let Err(error) = stamp_denoise(&mut req, payload) {
                    return Ok(Err(format!("pipeline: self-conditioning: {error}")));
                }
            }
        }
        stamp_lane_words(&mut req, fire_wide_mask, carries_media);
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
        let kv_clock = crate::scheduler::probe::ProbeClock::start();
        let ws_res: Resource<KvWorkingSet> = Resource::new_borrow(ws_rep);
        let ws = ctx.resources().get(&ws_res)?.clone();
        let stores = crate::store::registry::get(ws.model, ws.engine);
        if let Err(refusal) = seat_lane_slots(&mut req, &stores, ws.id).await {
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
                    continue;
                }
                Err(ReservedError::Stale) => return Ok(Err(stale_demand_error())),
                Err(ReservedError::Fatal(error)) => return Ok(Err(error)),
            };
            let kvtxn = KvTxnGuard::new(model, engine, kvtxn);
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
                    continue;
                }
                Err(ReservedError::Stale) => return Ok(Err(stale_demand_error())),
                Err(ReservedError::Fatal(error)) => return Ok(Err(error)),
            }
        };
        crate::probe_fire_record!(submit_probe.kv_prepare_us, kv_clock.elapsed());
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
            |((dst_page_id, dst_token_offset), (src_page_id, src_token_offset))| ::engine::KvMove {
                dst_page_id,
                dst_token_offset,
                src_page_id,
                src_token_offset,
            },
        )
        .collect::<Vec<_>>();
    let lease = loop {
        match ws_handle.fire_lease() {
            Ok(lease) => break lease,
            Err(crate::store::kv::working_set::FireLeaseError::Fenced) => {
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
        let rs_failure: Option<String> = {
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
    };

    if let Some(planner) = crate::planner::planner() {
        planner.pages_freed();
    }

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
    drain_settled(ctx, Some(&pipe_fires)).await?;
    if let Some(error) = pipeline_failed(&pipeline_failure) {
        return Ok(Err(error));
    }
    if let Err(e) = wire_channels_to_pipeline(ctx, &fwd, &pipe_fires)? {
        return Ok(Err(e));
    }

    let (ws_rep, rs_reps, rs_fold_len) = {
        let pass = ctx.resources().get(&fwd)?;
        let device_fold_len = pass
            .instance
            .program
            .bound
            .container
            .ports
            .iter()
            .any(|binding| {
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
    {
        let p = ctx.resources().get(&fwd)?;
        if let Some(e) = &p.failed {
            return Ok(Err(format!(
                "pipeline: forward-pass failed by an earlier fire: {e}"
            )));
        }
    }

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
        (
            grant_slots,
            write_indexes,
            fresh_dense,
            devgeo_b,
            devgeo_split,
        )
    };

    let wide = devgeo_split
        .as_ref()
        .is_some_and(|split| split.windows(2).any(|lane| lane[1] > lane[0] + 1));
    let resolved_qo_indptr = match &devgeo_split {
        Some(split) if wide => split.clone(),
        _ => vec![0; devgeo_b + 1],
    };
    let rs_qo_indptr = devgeo_split
        .clone()
        .unwrap_or_else(|| resolved_qo_indptr.clone());
    let rs_ws_ids = match bound_rs_working_set_ids(ctx, ws.model, ws.engine, &rs_reps)? {
        Ok(ids) => ids,
        Err(error) => {
            reclaim_pending_device_grant(ctx, &fwd);
            return Ok(Err(error));
        }
    };
    let mut attempts = 0;
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

    let (completion, instance_id, scheduler, cells, fwd_rep, accesses) = {
        let p = ctx.resources().get_mut(&fwd)?;
        let bytes: Vec<u8> = grant_slots.iter().flat_map(|s| s.to_le_bytes()).collect();
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
    ctx.resources().get(&fwd)?.lane.stamp(&mut req);
    let group = ctx.resources().get(&fwd)?.lane.group;
    let peer = ctx.resources().get(&fwd)?.lane.peer;
    req.cohort = crate::pipeline::instance::cohort_of(ctx.resources(), group, peer);
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
    {
        let pass = ctx.resources().get_mut(&fwd)?;
        if pass.bindings.canvas == Some(crate::pipeline::instance::CanvasMode::Denoise) {
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
            if let Err(error) = stamp_denoise(&mut req, payload) {
                reclaim_pending_device_grant(ctx, &fwd);
                let reason = format!("pipeline: self-conditioning: {error}");
                record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
                return Ok(Err(reason));
            }
        }
    }
    {
        let program = &ctx.resources().get(&fwd)?.instance.program;
        let (drafts, captures) = (program.reads_mtp_logits, program.reads_attn_score);
        for lane in &mut req.lanes {
            lane.drafts = drafts;
            lane.captures_scores = captures;
        }
    }
    stamp_lane_words(&mut req, fire_wide_mask, false);
    if let Err(refusal) = seat_lane_slots(&mut req, &stores, ws.id).await {
        reclaim_pending_device_grant(ctx, &fwd);
        record_submit_failure(ctx, &fwd, &pipeline_failure, &refusal);
        return Ok(Err(refusal));
    }
    if let Err(refusal) = stamp_lane_translation(&mut req, &stores, ws.id) {
        reclaim_pending_device_grant(ctx, &fwd);
        record_submit_failure(ctx, &fwd, &pipeline_failure, &refusal);
        return Ok(Err(refusal));
    }
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
    fn fire_every_case() {
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
            .put(vec![0u8; 4])
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
            .put(vec![0u8; 4])
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
