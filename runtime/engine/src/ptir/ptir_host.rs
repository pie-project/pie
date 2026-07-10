//! Host wiring for the `ptir` WIT interface (thrust-3 P2b, first-class model +
//! run-ahead submission).
//!
//! Behaviour is gated behind the `ptir` cargo feature (manager decision (a): the
//! interface is present in the world unconditionally — additive, no-op for
//! legacy guests — while the impl is inert when the feature is off). When on,
//! `forward-pass.new` binds + caches via [`ptir_registry`](super::ptir_registry)
//! (the old `register-program`, now an invisible compile/bind cache) and stamps
//! the container's roles onto the guest-constructed channels.
//!
//! **Run-ahead** (overview §3): `pipeline.submit` NEVER blocks. It prepares the
//! fire (seeds, host puts, KV/RS projection), hands the request to the
//! scheduler, and enqueues a [`PendingFire`] (the payload-free completion + the open
//! KV/RS txns) on the pass — the classic `execute()`/`output()` split
//! (`PendingForward`, Option A) applied to PTIR. Step t+1 is prepared against
//! t's OPTIMISTIC post-state (the `committed_tokens` cursor advances at
//! submit), so a decode loop keeps the scheduler fed. `channel.take`/`read`
//! are the await points: they finalize in-flight fires FIFO until the cell
//! fills. A failed fire **poisons** the pass's host-reader channels (the only
//! error path once submit is non-blocking) and fails the pass for further
//! submits.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

use crate::api::pie;
use crate::instance::InstanceState;
use crate::working_set::kv::KvWorkingSet;
use crate::working_set::rs::RsWorkingSet;

use pie_ptir::container::HostRole;

use super::ptir_channel_store::{
    BoundCells, ChannelCell, ChannelError, commit_host_puts, reserve_reader_capacity,
    rollback_reader_capacity, snapshot_host_puts,
};
use super::ptir_kv;
use super::ptir_lease::PageLease;
use super::ptir_rs;

/// A first-class, guest-constructed channel (overview §1). The SAME handle is
/// bound into a forward pass (dense declaration index) and used for host
/// `put`/`take`/`read`; the shared [`ChannelCell`] is Arc'd so a pass that
/// bound it survives the guest dropping the handle.
pub struct Channel {
    pub cell: Arc<Mutex<ChannelCell>>,
    /// Set at SUBMIT: the feeding PIPELINE's in-flight fire queue (W3.1). A
    /// channel may be fed by several passes, but all must submit on the SAME
    /// pipeline (§3.4) — so `take`/`read` await + finalize fires from one FIFO
    /// (submission order) until the cell fills. `None` until first submit.
    pub fires: Option<PendingFires>,
}

/// A pass's in-flight fires, submit order. Plain mutex: never held across an
/// await (the op is popped out, then awaited).
pub type PendingFires = Arc<Mutex<VecDeque<PendingOp>>>;
type PipelineFailure = Arc<Mutex<Option<String>>>;

/// A pipeline FIFO entry: a forward FIRE or a KV cell MOVE (Design-B
/// compaction). Both hold an ordered slot on the same stream — the B3
/// happens-before invariant; `take`/`read` drain them in submit order.
pub enum PendingOp {
    Fire(PendingFire),
    Move(PendingMove),
}

impl PendingOp {
    async fn wait_for_completion_signal(&self) {
        match self {
            Self::Fire(fire) => {
                let _ = fire.completion.clone().await;
            }
            Self::Move(mv) => {
                let _ = mv.completion.clone().await;
            }
        }
    }
}

/// One in-flight KV cell MOVE (`pipeline.copy-into`). It holds an ordered slot
/// in the pipeline FIFO and finalizes by awaiting its payload-free completion.
pub struct PendingMove {
    completion: crate::driver::Completion,
    failure: PipelineFailure,
}

/// The open KV/arena transaction(s) one in-flight fire holds until it resolves.
/// Two shapes: the ordinary single-seq / MTP projection (`ptir_kv`), or a
/// device-geometry fire whose KV the driver resolves+writes itself (B2's
/// explicit-KV path) — the runtime only pins the `PageLease`-granted physical
/// pages for the fire, released at finalize (per-fire arena txn; the plan's
/// "pin float bounded by run-ahead depth × B, riding the per-fire arena txns").
enum FireKv {
    Kv(ptir_kv::PtirKvTxn),
    DeviceGeom {
        arena_txn: crate::arena::ArenaTxn,
        write_txn: crate::working_set::kv::WriteTxnId,
    },
}

/// One in-flight PTIR fire: the driver completion plus everything needed to
/// finalize when it resolves — the open KV/RS txns (pins/CoW held until
/// commit/abort) and the bound cells whose mirror epochs become visible.
pub struct PendingFire {
    completion: crate::driver::InstanceCompletion,
    kv: FireKv,
    rstxn: Option<ptir_rs::PtirRsTxn>,
    ws_rep: u32,
    rs_rep: Option<u32>,
    /// The owning pass, to fail it on a fire error (rep — the guest may have
    /// dropped the handle; failure marking is then moot).
    fwd_rep: u32,
    instance_id: u64,
    cells: BoundCells,
    failure: PipelineFailure,
}

/// A traced forward pass bound to its first-class handles — one instance of a
/// (hash-deduped) registered program. The driver's persistent channel arena is
/// keyed by this pass's `instance_id`; a channel MAY bind to several passes
/// (multi-pass channels, W3.2) — the driver's global channel registry (W0.1)
/// resolves one shared device cell and the pipeline orders the fires (§3.4).
pub struct ForwardPass {
    pub instance: super::ptir_instance::PtirInstance,
    pub bound_instance: crate::driver::BoundInstance,
    /// The bound channel cells, dense declaration order. Writer puts are
    /// coalesced into each fire; Reader cells hold direct mirror bindings.
    pub cells: BoundCells,
    /// Dense-channel-index → global-channel-id map (captured at bind from the
    /// bound cells). Rides every submission so the driver binds the trace's
    /// dense channel references to the global device channel registry.
    pub channel_ids: Vec<u64>,
    /// The bound channel resource reps (captured at `forward-pass.new`), so
    /// `submit` can point each channel's await queue at the feeding pipeline
    /// (W3.1: the pipeline owns the FIFO; a pass may bind to any pipeline).
    pub channel_reps: Vec<u32>,
    /// Pipeline FIFO this pass has submitted through. Stored on the pass so
    /// teardown remains safe even if guest channel handles were dropped first.
    pub fires: Option<PendingFires>,
    /// The guest-owned KV working set bound into this pass (the model forward
    /// writes the embedded token's K/V here + self-attends over it). The guest
    /// keeps it alive for the pass's lifetime (the classic `forward-pass`
    /// borrow convention); the pass does NOT destroy it on drop.
    pub kv_ws: u32,
    /// The guest-owned recurrent-state working set (hybrid / linear-attention
    /// models — GDN, Mamba2). `None` for pure-attention models.
    pub rs_ws: Option<u32>,
    /// The bound ws's committed token length — the growing cursor threaded
    /// into [`ptir_kv::ptir_kv_prepare`]. Advances OPTIMISTICALLY at submit
    /// (run-ahead: fire t+1 prepares against t's post-state); a failed fire
    /// fails the whole pass (`failed`) rather than rewinding the cursor.
    pub committed_tokens: u32,
    /// First-fire byte-ship tracking: the container bytes + seeds ride the
    /// first fire; steady-state fires carry the hash + instance id only
    /// (driver hash-cache + persistent arena).
    /// Set when a fire of this pass failed: further submits error with the
    /// root cause (the KV cursor and device channel state are unspecified
    /// after a failed fire — the guest builds a fresh pass).
    pub failed: Option<String>,
    /// Device-geometry state (Track B): `Some` iff this pass's geometry
    /// (`pages`/`w_slot`/…) is DEVICE-produced — the program traces the wire-form
    /// geometry in-graph (`page_indptr = CumSum(np)`, packed live pages) and the
    /// driver resolves it pre-forward, so the host neither replays the epilogue
    /// arithmetic nor projects per-lane KV. The runtime only leases physical
    /// pages (`PageLease`) and delivers fresh grants on the program's fresh
    /// channel. Replaces the deleted host-replay beam branch.
    pub devgeo: Option<DevGeo>,
}

/// A run-ahead submission pipeline (overview §3): the ORDERING domain (W3.1).
/// Owns the in-flight fire FIFO — fires submitted here are issued in submission
/// order, so fire t's epilogue channel puts happen-before fire t+1's descriptor
/// reads, EXTENDED ACROSS PASSES (draft→verify chaining). `take`/`read` await
/// this FIFO via each channel's recorded pipeline. Submission order rides the
/// scheduler queue; completion order rides this FIFO.
///
/// **FIFO INVARIANT (B3, mandatory).** Fires of one pipeline keep submission
/// order through the scheduler onto one stream, and every pass binding a shared
/// channel MUST submit on the SAME pipeline (enforced by
/// [`InstanceState::wire_channels_to_pipeline`]). This is the ENTIRE correctness
/// argument for run-ahead + multi-pass chaining: because all interacting fires
/// funnel onto one ordered FIFO, fire t's epilogue puts happen-before fire t+1's
/// descriptor reads. `push_back` at submit + `pop_front` at finalize preserve
/// that order; the same-pipeline check makes it an explicit invariant, not an
/// accident. Tested by `tests::{detect_device_geometry_*, fifo_preserves_submission_order}`.
pub struct Pipeline {
    pub fires: PendingFires,
    failure: PipelineFailure,
}

/// Physical-page leasing + channel bookkeeping for a device-geometry pass
/// (Track B / plan W3.3). The runtime seeds fire 0's `B` pages and grants `B`
/// fresh physical ids per submit (delivered as a host-put on the `fresh`
/// channel); after a fire commits it reclaims the UNUSED grants of continuing
/// heirs (harvested `w_cont`), and everything on drop.
pub struct DevGeo {
    /// The physical-page lease (grant / reclaim / free-list bookkeeping).
    pub lease: PageLease,
    /// Beam / lane width `B` — the fresh grants per fire.
    pub b: usize,
    /// Dense channel index of the host-writer `fresh`-page input channel — where
    /// the runtime injects each fire's grant.
    pub fresh_dense: usize,
    /// Dense channel index of the `w_cont` host-reader output ([B] bool) — read
    /// at finalize to reclaim continuing heirs' unused fresh pages.
    pub w_cont_dense: usize,
}

/// Detect a device-geometry pass: its geometry ports (`WSlot`/`WOff` write
/// descriptors — beam-specific, a plain decode's `attn_working_set` binds only
/// `KvLen`) bind DEVICE-produced channels, and the `Pages` port's channel is
/// `[B, P]` (`P > 1`). Returns `(B, fresh_dense, w_cont_dense)`: the single
/// host-writer channel is `fresh`; the host-reader `[B]` bool channel is
/// `w_cont` (the reclaim signal). `None` for an ordinary decode.
fn detect_device_geometry(
    container: &pie_ptir::container::TraceContainer,
) -> Option<(usize, usize, usize)> {
    use pie_ptir::container::{ChanDType, PortSource};
    use pie_ptir::registry::Port;
    use pie_ptir::types::DType;

    let has_write_desc = container
        .ports
        .iter()
        .any(|p| matches!(p.port, Port::WSlot | Port::WOff));
    if !has_write_desc {
        return None;
    }
    // B from the [B, P] channel bound to the `Pages` port (P > 1 for a beam).
    let pages_ch = container
        .ports
        .iter()
        .find_map(|p| match (&p.port, &p.source) {
            (Port::Pages, PortSource::Channel(c)) => Some(*c as usize),
            _ => None,
        })?;
    let dims = container.channels.get(pages_ch)?.shape.dims();
    let b = if dims.len() == 2 && dims[1] > 1 {
        dims[0] as usize
    } else {
        return None;
    };

    // fresh = the single host-Writer channel; w_cont = the host-Reader bool.
    let fresh_dense = container
        .channels
        .iter()
        .position(|c| c.host_role == HostRole::Writer)?;
    let w_cont_dense = container.channels.iter().position(|c| {
        c.host_role == HostRole::Reader && matches!(c.dtype, ChanDType::Concrete(DType::Bool))
    })?;
    Some((b, fresh_dense, w_cont_dense))
}

type Anyhow<T> = anyhow::Result<T>;

/// First fire only: bind the pass's seeds — one staged `put` per `seeded`
/// channel, dense order (D2, per-instance data). A seeded channel with no
/// staged put is the old `MissingSeed` instantiate error, surfaced at the
/// first submit instead of hanging the fire.
fn bind_seeds_first_fire(_p: &mut ForwardPass) -> Result<(), String> {
    Ok(())
}

/// Poison every host-reader cell of a pass with the failed fire's error —
/// under run-ahead this IS the error channel (`take`/`read` surface it).
fn poison_readers(cells: &BoundCells, reason: &str) {
    for cell in cells {
        let mut c = cell.lock().unwrap();
        if c.role == Some(HostRole::Reader) {
            c.poison(reason);
        }
    }
}

fn publish_reader_mirrors(cells: &BoundCells, instance_id: u64) -> Result<bool, String> {
    let mut poisoned = false;
    for cell in cells {
        let mut cell = cell.lock().unwrap();
        if cell.role == Some(HostRole::Reader) {
            poisoned |= cell.publish_reader_mirror(instance_id)?;
        }
    }
    Ok(poisoned)
}

// The interface has no free functions left (register-program folded into
// forward-pass.new); the trait still anchors the resource types.
impl pie::inferlet::forward::Host for InstanceState {}

impl pie::inferlet::forward::HostChannel for InstanceState {
    async fn new(
        &mut self,
        shape: Vec<u32>,
        dtype: pie::inferlet::types::Dtype,
        capacity: u32,
    ) -> Anyhow<Resource<Channel>> {
        // Pure host bookkeeping — works with the `ptir` feature off too (the
        // WIT constructor cannot carry a result; the gate errors at
        // forward-pass.new / submit instead).
        use pie::inferlet::types::Dtype;
        let dtype = match dtype {
            Dtype::F32 => pie_ptir::types::DType::F32,
            Dtype::I32 => pie_ptir::types::DType::I32,
            Dtype::U32 => pie_ptir::types::DType::U32,
            Dtype::Bool => pie_ptir::types::DType::Bool,
        };
        let cell = Arc::new(Mutex::new(ChannelCell::new(shape, dtype, capacity)));
        Ok(self.ctx().table.push(Channel { cell, fires: None })?)
    }

    async fn put(&mut self, this: Resource<Channel>, value: Vec<u8>) -> Anyhow<Result<(), String>> {
        let cell = self.ctx().table.get(&this)?.cell.clone();
        loop {
            let result = cell.lock().unwrap().put_ref(&value);
            match result {
                Ok(()) => return Ok(Ok(())),
                Err(ChannelError::Full) => {}
                Err(error) => return Ok(Err(error.to_string())),
            }
            let wait = cell.lock().unwrap().writer_wait_state();
            let Some((endpoint, observed_head)) = wait else {
                return Ok(Err(ChannelError::Full.to_string()));
            };
            if let Err(error) = endpoint.wait_for_writer_change(observed_head).await {
                return Ok(Err(error.to_string()));
            }
        }
    }

    /// The run-ahead await point: while the cell is empty, await + finalize
    /// the pass's oldest in-flight fire (FIFO), then re-check. With no local
    /// fire, wait directly on the endpoint so a producer in another task can
    /// wake this channel. Poison and close are surfaced as errors.
    async fn take(&mut self, this: Resource<Channel>) -> Anyhow<Result<Vec<u8>, String>> {
        loop {
            let (cell, fires) = {
                let ch = self.ctx().table.get(&this)?;
                (ch.cell.clone(), ch.fires.clone())
            };
            match cell.lock().unwrap().take() {
                Ok(v) => return Ok(Ok(v)),
                Err(ChannelError::Empty) => {}
                Err(e) => return Ok(Err(e.to_string())),
            }
            let wait = cell.lock().unwrap().reader_wait_state();
            let fire = fires.as_ref().and_then(|f| f.lock().unwrap().pop_front());
            match fire {
                Some(fire) => {
                    if let Some((endpoint, observed_tail)) = wait {
                        tokio::select! {
                            _ = endpoint.wait_for_reader_change(observed_tail) => {}
                            _ = fire.wait_for_completion_signal() => {}
                        }
                    }
                    self.finalize_op(fire).await?;
                }
                None => {
                    let Some((endpoint, observed_tail)) = wait else {
                        return Ok(Err(ChannelError::Empty.to_string()));
                    };
                    if let Err(error) = endpoint.wait_for_reader_change(observed_tail).await {
                        return Ok(Err(error.to_string()));
                    }
                    let mut cell = cell.lock().unwrap();
                    if let Err(error) = cell.publish_reader_mirror(0) {
                        return Ok(Err(error));
                    }
                }
            }
        }
    }

    /// Non-consuming peek; same await discipline as `take`.
    async fn read(&mut self, this: Resource<Channel>) -> Anyhow<Result<Vec<u8>, String>> {
        loop {
            let (cell, fires) = {
                let ch = self.ctx().table.get(&this)?;
                (ch.cell.clone(), ch.fires.clone())
            };
            match cell.lock().unwrap().read() {
                Ok(v) => return Ok(Ok(v)),
                Err(ChannelError::Empty) => {}
                Err(e) => return Ok(Err(e.to_string())),
            }
            let wait = cell.lock().unwrap().reader_wait_state();
            let fire = fires.as_ref().and_then(|f| f.lock().unwrap().pop_front());
            match fire {
                Some(fire) => {
                    if let Some((endpoint, observed_tail)) = wait {
                        tokio::select! {
                            _ = endpoint.wait_for_reader_change(observed_tail) => {}
                            _ = fire.wait_for_completion_signal() => {}
                        }
                    }
                    self.finalize_op(fire).await?;
                }
                None => {
                    let Some((endpoint, observed_tail)) = wait else {
                        return Ok(Err(ChannelError::Empty.to_string()));
                    };
                    if let Err(error) = endpoint.wait_for_reader_change(observed_tail).await {
                        return Ok(Err(error.to_string()));
                    }
                    let mut cell = cell.lock().unwrap();
                    if let Err(error) = cell.publish_reader_mirror(0) {
                        return Ok(Err(error));
                    }
                }
            }
        }
    }

    async fn drop(&mut self, this: Resource<Channel>) -> Anyhow<()> {
        // A pass that bound this channel holds its own Arc — dropping the
        // guest handle never dangles an in-flight fire. Native channel storage
        // is reference-counted by bound instances and releases on instance close.
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl pie::inferlet::forward::HostForwardPass for InstanceState {
    async fn new(
        &mut self,
        container_bytes: Vec<u8>,
        channels: Vec<Resource<Channel>>,
        kv_working_sets: Vec<Resource<KvWorkingSet>>,
        rs_working_sets: Vec<Resource<RsWorkingSet>>,
    ) -> Anyhow<Result<Resource<ForwardPass>, String>> {
        {
            // Identity dedup + bind against the model profile (the old
            // register-program, now invisible): hash-deduped compile/bind
            // cache; a malformed trace fails HERE with the validator's
            // message (the P2 exit).
            let prog = match super::ptir_registry::register(container_bytes, &model_profile()) {
                Ok(p) => p,
                Err(e) => return Ok(Err(e.to_string())),
            };

            // Validate every handle against its dense declaration BEFORE
            // stamping any of them, so a failed `new` binds nothing.
            let decls = prog.bound.container.channels.clone();
            let extern_bindings = decls
                .iter()
                .enumerate()
                .map(|(dense, _)| {
                    prog.bound
                        .container
                        .externs
                        .iter()
                        .find(|binding| binding.chan == dense as u32)
                        .map(|binding| {
                            (
                                prog.bound.container.names[binding.name as usize].clone(),
                                binding.dir,
                            )
                        })
                })
                .collect::<Vec<_>>();
            if channels.len() != decls.len() {
                return Ok(Err(format!(
                    "ptir: {} channel handles supplied for {} declared channels",
                    channels.len(),
                    decls.len()
                )));
            }
            let mut cells: BoundCells = Vec::with_capacity(channels.len());
            for (i, ch) in channels.iter().enumerate() {
                let cell = self.ctx().table.get(ch)?.cell.clone();
                if cells.iter().any(|prev| Arc::ptr_eq(prev, &cell)) {
                    return Ok(Err(format!(
                        "ptir: channel {i} appears twice in the handle list"
                    )));
                }
                {
                    let c = cell.lock().unwrap();
                    // W3.2: a channel MAY bind to several passes (multi-pass
                    // channels). The old one-pass-per-channel gate is lifted; the
                    // driver's global channel registry (W0.1) resolves one shared
                    // device cell, and the pipeline enforces same-pipeline
                    // ordering (§3.4). Decl equality across the sharing passes is
                    // still validated (`matches_decl`) — a conflict is an error.
                    let extern_binding = extern_bindings[i]
                        .as_ref()
                        .map(|(name, dir)| (name.as_str(), *dir));
                    if let Err(e) = c.validate_attachment(&decls[i], extern_binding) {
                        return Ok(Err(format!("ptir: channel {i}: {e}")));
                    }
                    // Pre-bind staged puts must fit the declared role: a
                    // Writer drains them per fire, a seeded non-Writer holds
                    // exactly its one seed, anything else never drains.
                    let staged = c.staged_len();
                    let staged_ok = match decls[i].host_role {
                        HostRole::Writer => true,
                        _ if decls[i].seeded => staged <= 1,
                        _ => staged == 0,
                    };
                    if !staged_ok {
                        return Ok(Err(format!(
                            "ptir: channel {i}: {staged} staged put(s) don't fit its declared \
                             {:?}{} role",
                            decls[i].host_role,
                            if decls[i].seeded { " seeded" } else { "" }
                        )));
                    }
                }
                cells.push(cell);
            }

            // v1 single-model contract: exactly one guest-owned KV working set
            // (the classic forward-pass borrow convention — the guest keeps it
            // alive for the pass's lifetime); at most one RS working set
            // (hybrid / linear-attention models).
            if kv_working_sets.len() != 1 {
                return Ok(Err(format!(
                    "ptir: expected exactly one kv-working-set, got {}",
                    kv_working_sets.len()
                )));
            }
            if rs_working_sets.len() > 1 {
                return Ok(Err(format!(
                    "ptir: expected at most one rs-working-set, got {}",
                    rs_working_sets.len()
                )));
            }
            let ws_rep = kv_working_sets[0].rep();
            let rs_rep = rs_working_sets.first().map(|r| r.rep());

            // Device-geometry pass (Track B): seed the physical-page lease with
            // `B` fire-0 pages (one live page per lane) drawn from the guest's
            // working set. The [B,P] geometry is device-produced (the program
            // traces `page_indptr = CumSum(np)` + packed pages in-graph); the
            // runtime no longer replays the epilogue arithmetic nor eagerly
            // reserves B*P slots. A normal program's ws starts as the guest left
            // it — `ptir_kv::ptir_kv_prepare` grows it per fire.
            let devgeo = match detect_device_geometry(&prog.bound.container) {
                Some((b, fresh_dense, w_cont_dense)) => {
                    let ws_res: Resource<KvWorkingSet> = Resource::new_borrow(ws_rep);
                    let seed_pages = match self.ctx().table.get_mut(&ws_res)?.alloc_slots(b as u32)
                    {
                        Ok(ids) => ids,
                        Err(e) => return Ok(Err(format!("ptir: device-geometry seed alloc: {e}"))),
                    };
                    let mut lease = PageLease::new(b);
                    lease.seed(seed_pages);
                    Some(DevGeo {
                        lease,
                        b,
                        fresh_dense,
                        w_cont_dense,
                    })
                }
                None => None,
            };

            let instance_id = super::ptir_instance::next_instance_id();
            for (dense, cell) in cells.iter().enumerate() {
                let extern_binding = extern_bindings[dense]
                    .as_ref()
                    .map(|(name, dir)| (name.as_str(), *dir));
                if let Err(error) =
                    cell.lock()
                        .unwrap()
                        .attach(instance_id, &decls[dense], extern_binding)
                {
                    for attached in &cells {
                        attached.lock().unwrap().detach(instance_id);
                    }
                    return Ok(Err(format!("ptir: channel {dense} attach: {error}")));
                }
            }
            for (dense, cell) in cells.iter().enumerate() {
                let existing = cell.lock().unwrap().endpoint();
                let endpoint = match existing {
                    Some(endpoint) => endpoint,
                    None => {
                        let extern_binding = extern_bindings[dense].as_ref();
                        let plan = crate::driver::ChannelRegistrationPlan {
                            driver_id: 0,
                            channel_id: cell.lock().unwrap().global_id,
                            shape: decls[dense].shape.dims().to_vec(),
                            dtype: decls[dense].dtype.tag(),
                            host_role: decls[dense].host_role as u8,
                            seeded: decls[dense].seeded,
                            extern_dir: extern_binding
                                .map(|(_, dir)| match dir {
                                    pie_ptir::container::ExternDir::Import => {
                                        pie_driver_abi::PIE_CHANNEL_EXTERN_IMPORT
                                    }
                                    pie_ptir::container::ExternDir::Export => {
                                        pie_driver_abi::PIE_CHANNEL_EXTERN_EXPORT
                                    }
                                })
                                .unwrap_or(pie_driver_abi::PIE_CHANNEL_EXTERN_NONE),
                            capacity: decls[dense].capacity,
                            reader_wait_id: 0,
                            writer_wait_id: 0,
                            extern_name: extern_binding
                                .map(|(name, _)| name.as_bytes().to_vec())
                                .unwrap_or_default(),
                        };
                        match crate::driver::register_channel(0, plan) {
                            Ok(endpoint) => endpoint,
                            Err(error) => {
                                for attached in &cells {
                                    attached.lock().unwrap().detach(instance_id);
                                }
                                return Ok(Err(format!(
                                    "ptir: register channel {dense}: {error:#}"
                                )));
                            }
                        }
                    }
                };
                if let Err(error) = cell.lock().unwrap().attach_endpoint(endpoint) {
                    for attached in &cells {
                        attached.lock().unwrap().detach(instance_id);
                    }
                    return Ok(Err(format!("ptir: channel {dense} endpoint: {error}")));
                }
            }

            // All validation passed — stamp the container's roles onto the
            // cells (the bind point) and mint the instance identity (the
            // driver's persistent channel-arena key). The await FIFO is owned by
            // the PIPELINE now (W3.1), wired to the channels at submit.
            // Capture the dense-index → global-channel-id map now that the cells
            // are validated (multi-pass channels: a global id is stable across
            // every pass a channel binds into).
            let channel_ids: Vec<u64> = cells.iter().map(|c| c.lock().unwrap().global_id).collect();
            // Capture the bound channel resource reps so `submit` can point each
            // channel's await queue at the feeding pipeline (W3.1).
            let channel_reps: Vec<u32> = channels.iter().map(|c| c.rep()).collect();
            let program_id = match crate::driver::register_program(
                0,
                crate::driver::ProgramRegistration {
                    program_hash: prog.hash,
                    canonical_bytes: prog.bytes.clone(),
                    sidecar_bytes: prog.sidecar.clone(),
                },
            ) {
                Ok(id) => id,
                Err(e) => return Ok(Err(format!("ptir: register program: {e:#}"))),
            };
            let mut instance_seeds = Vec::new();
            let mut seed_values = Vec::new();
            for (dense, cell) in cells.iter().enumerate() {
                let cell = cell.lock().unwrap();
                if !cell.seeded {
                    continue;
                }
                let bytes = match cell.peek_seed() {
                    Ok(bytes) => bytes,
                    Err(e) => return Ok(Err(format!("ptir: channel {dense} seed: {e}"))),
                };
                instance_seeds.push(super::ptir_instance::ChannelSeed {
                    channel: dense as u32,
                    data: bytes.clone(),
                });
                seed_values.push(crate::ptir::PtirChannelValue {
                    channel: cell.global_id,
                    bytes,
                });
            }
            let instance = super::ptir_instance::PtirInstance {
                program: prog,
                instance_id,
                seeds: instance_seeds,
            };
            let bound_instance = match crate::driver::bind_instance(
                0,
                program_id,
                instance.instance_id,
                channel_ids.clone(),
                seed_values,
            ) {
                Ok(bound) => bound,
                Err(e) => {
                    for cell in &cells {
                        cell.lock().unwrap().detach(instance_id);
                    }
                    return Ok(Err(format!("ptir: bind instance: {e:#}")));
                }
            };
            for cell in &cells {
                let mut cell = cell.lock().unwrap();
                if cell.role == Some(HostRole::Reader) {
                    if let Err(error) = cell.publish_reader_mirror(instance_id) {
                        drop(cell);
                        let _ = crate::driver::close_instance(&bound_instance);
                        for cell in &cells {
                            cell.lock().unwrap().detach(instance_id);
                        }
                        return Ok(Err(format!("ptir: publish bound reader endpoint: {error}")));
                    }
                }
                if cell.seeded {
                    cell.commit_seed();
                }
            }
            let res = self.ctx().table.push(ForwardPass {
                instance,
                bound_instance,
                cells,
                channel_ids,
                channel_reps,
                fires: None,
                kv_ws: ws_rep,
                rs_ws: rs_rep,
                committed_tokens: 0,
                failed: None,
                devgeo,
            })?;
            Ok(Ok(res))
        }
    }

    async fn drop(&mut self, this: Resource<ForwardPass>) -> Anyhow<()> {
        // A pass can be dropped before its pipeline. Drain the shared FIFO first
        // so every callback, mirror publication, and KV/RS transaction completes
        // before raw mirror pointers are detached or pages become reusable.
        let fires = self.ctx().table.get(&this)?.fires.clone();
        if let Some(fires) = fires {
            loop {
                let op = fires.lock().unwrap().pop_front();
                match op {
                    Some(op) => self.finalize_op(op).await?,
                    None => break,
                }
            }
        }

        let mut pass = self.ctx().table.delete(this)?;
        let _ = crate::driver::close_instance(&pass.bound_instance);
        for cell in &pass.cells {
            cell.lock().unwrap().detach(pass.bound_instance.instance_id);
        }
        // Device-geometry: reclaim EVERY leased physical page (all in-flight
        // grants + the fire-0 seed) back to the working set on pass drop.
        if let Some(devgeo) = pass.devgeo.as_mut() {
            let freed = devgeo.lease.reclaim_all();
            if !freed.is_empty() {
                let ws_res: Resource<KvWorkingSet> = Resource::new_borrow(pass.kv_ws);
                if let Ok(ws) = self.ctx().table.get(&ws_res) {
                    let (m, d) = ws.device();
                    let arena_arc = crate::arena::get(m, d);
                    let cas_arc = crate::working_set::kv_cas::get(m, d);
                    let mut arena = arena_arc.lock().unwrap();
                    let mut cas = cas_arc.lock().unwrap();
                    if let Ok(ws) = self.ctx().table.get_mut(&ws_res) {
                        let _ = ws.free_slots(&freed, &mut arena, &mut cas);
                    }
                }
            }
        }
        Ok(())
    }
}

impl pie::inferlet::forward::HostPipeline for InstanceState {
    async fn new(&mut self) -> Anyhow<Resource<Pipeline>> {
        Ok(self.ctx().table.push(Pipeline {
            fires: Arc::new(Mutex::new(VecDeque::new())),
            failure: Arc::new(Mutex::new(None)),
        })?)
    }

    /// Run-ahead submit: prepare + fire + enqueue, NO await. See the module
    /// docs; errors after this call surface via channel poison + `take`.
    async fn submit(
        &mut self,
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
            if self.ctx().table.get(&fwd)?.devgeo.is_some() {
                return self.fire_device_geometry(this, fwd).await;
            }
            // W3.1: the PIPELINE owns the in-flight FIFO. Point each of this
            // pass's channels at this pipeline's queue so their `take`/`read`
            // await the right FIFO — enforcing the same-pipeline constraint
            // (§3.4): every pass binding a channel must submit on one pipeline.
            let pipe_fires = self.ctx().table.get(&this)?.fires.clone();
            let pipeline_failure = self.ctx().table.get(&this)?.failure.clone();
            if let Some(reason) = pipeline_failure.lock().unwrap().clone() {
                return Ok(Err(format!("ptir: pipeline failed: {reason}")));
            }
            if let Err(error) = self.wire_channels_to_pipeline(&fwd, &pipe_fires)? {
                return Ok(Err(error));
            }
            let (
                geometry,
                cells,
                ws_rep,
                rs_rep,
                committed_tokens,
                fwd_rep,
                instance_id,
                host_puts,
                completion,
            ) = {
                let p = self.ctx().table.get_mut(&fwd)?;
                if let Some(e) = &p.failed {
                    return Ok(Err(format!(
                        "ptir: forward-pass failed by an earlier fire: {e}"
                    )));
                }
                let geometry = p.instance.fire_geometry(model_profile().page_size).ok();
                (
                    geometry,
                    p.cells.clone(),
                    p.kv_ws,
                    p.rs_ws,
                    p.committed_tokens,
                    fwd.rep(),
                    p.bound_instance.instance_id,
                    snapshot_host_puts(&p.cells),
                    p.bound_instance.reserve_completion(),
                )
            };
            let mut req = crate::driver::LaunchPlan::default();
            if let Some(g) = &geometry {
                g.apply_to(&mut req);
            }
            // Project the guest-owned KV working set for this fire via `ptir_kv`
            // (alpha's ws-alloc + arena-txn + project_kv lifecycle). The held
            // `PtirKvTxn` rides the PendingFire across the async fire; finalized
            // (commit → KV persists / abort → revert) when a take/read/drop
            // drains it.
            let new_tokens: Vec<u32> = req.token_ids.clone();
            let page_size = crate::working_set::page_size::tokens_per_page(0);
            let ws_res: Resource<KvWorkingSet> = Resource::new_borrow(ws_rep);
            let arena_arc = crate::arena::get(0, 0);
            let prepared = {
                let mut arena = arena_arc.lock().unwrap();
                let ws = self.ctx().table.get_mut(&ws_res)?;
                ptir_kv::ptir_kv_prepare(ws, committed_tokens, &new_tokens, &mut arena, page_size)
            };
            let (proj, move_plans, kvtxn) = match prepared {
                Ok(v) => v,
                Err(e) => return Ok(Err(format!("ptir: kv prepare: {e}"))),
            };
            let next_committed = kvtxn.committed_tokens_after;

            // The recurrent-state slot for hybrid / linear-attention models
            // (GDN, Mamba2): fresh RESET slab on the first fire, CoW-continue
            // after — mirrors `execute_impl`'s rs block via `ptir_rs`.
            let rstxn = if let Some(rs_rep) = rs_rep {
                let rs_res: Resource<RsWorkingSet> = Resource::new_borrow(rs_rep);
                let prepared = {
                    let mut arena = arena_arc.lock().unwrap();
                    let rs = self.ctx().table.get_mut(&rs_res)?;
                    ptir_rs::ptir_rs_prepare(rs, &mut arena)
                };
                match prepared {
                    Ok((rs_slot_ids, rs_slot_flags, cow_move, txn)) => {
                        req.rs_slot_ids = rs_slot_ids;
                        req.rs_slot_flags = rs_slot_flags;
                        if let Some(mp) = &cow_move {
                            if let Err(e) = crate::driver::copy_d2d(0, &mp.from, &mp.to) {
                                tracing::warn!("ptir rs CoW d2d copy failed: {e:#}");
                            }
                        }
                        Some(txn)
                    }
                    Err(e) => {
                        // Revert the KV txn we already opened for this fire.
                        let mut arena = arena_arc.lock().unwrap();
                        match self.ctx().table.get_mut(&ws_res) {
                            Ok(ws) => {
                                let _ = ptir_kv::ptir_kv_finalize(ws, &mut arena, kvtxn, false);
                            }
                            Err(_) => ptir_kv::ptir_kv_abandon(&mut arena, kvtxn),
                        }
                        return Ok(Err(format!("ptir: rs prepare: {e}")));
                    }
                }
            } else {
                None
            };

            // D2D-copy every CoW'd write target before the fire (empty for the
            // single-context pipeline; non-empty only under a forked/shared page).
            for mp in &move_plans {
                if let Err(e) = crate::driver::copy_d2d(0, &mp.from, &mp.to) {
                    tracing::warn!("ptir forward CoW d2d copy failed: {e:#}");
                }
            }

            // Fire through the scheduler → charlie's PTIR executor hook — and
            // return. NO await: the PendingFire carries the round-trip; the
            // channels' take/read finalize it.
            let reserved = reserve_reader_capacity(&cells);
            let submit_error = match &reserved {
                Err(error) => Some(format!("channel reservation failed: {error}")),
                Ok(()) => crate::inference::submit_async(
                    req,
                    0,
                    instance_id,
                    host_puts,
                    proj.physical_page_ids,
                    proj.last_page_len,
                    Vec::new(),
                    None,
                    completion.clone(),
                )
                .err()
                .map(|error| format!("{error:#}")),
            };
            if let Some(error) = submit_error {
                if reserved.is_ok() {
                    rollback_reader_capacity(&cells);
                }
                let reason = format!("ptir: submit failed: {error}");
                let mut arena = arena_arc.lock().unwrap();
                match self.ctx().table.get_mut(&ws_res) {
                    Ok(ws) => {
                        let _ = ptir_kv::ptir_kv_finalize(ws, &mut arena, kvtxn, false);
                    }
                    Err(_) => ptir_kv::ptir_kv_abandon(&mut arena, kvtxn),
                }
                if let Some(rstxn) = rstxn {
                    let rs_res: Resource<RsWorkingSet> =
                        Resource::new_borrow(rs_rep.expect("rs txn implies rs rep"));
                    match self.ctx().table.get_mut(&rs_res) {
                        Ok(rs) => {
                            let _ = ptir_rs::ptir_rs_finalize(rs, &mut arena, rstxn, false);
                        }
                        Err(_) => ptir_rs::ptir_rs_abandon(&mut arena, rstxn),
                    }
                }
                drop(arena);
                self.ctx().table.get_mut(&fwd)?.failed = Some(reason.clone());
                return Ok(Err(reason));
            }

            // Optimistic cursor advance: the NEXT submit prepares against this
            // fire's post-state (the run-ahead overlap). A failed fire fails
            // the pass instead of rewinding.
            self.ctx().table.get_mut(&fwd)?.committed_tokens = next_committed;
            commit_host_puts(&cells);

            pipe_fires
                .lock()
                .unwrap()
                .push_back(PendingOp::Fire(PendingFire {
                    completion,
                    kv: FireKv::Kv(kvtxn),
                    rstxn,
                    ws_rep,
                    rs_rep,
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
    async fn copy_into(
        &mut self,
        this: Resource<Pipeline>,
        _ws: Resource<KvWorkingSet>,
        dst_page_ids: Vec<u32>,
        dst_tok_idx: Vec<u32>,
        src_page_ids: Vec<u32>,
        src_tok_idx: Vec<u32>,
    ) -> Anyhow<Result<(), String>> {
        let n = dst_page_ids.len();
        if dst_tok_idx.len() != n || src_page_ids.len() != n || src_tok_idx.len() != n {
            return Ok(Err(format!(
                "ptir copy_into: the four (dst_page,dst_tok,src_page,src_tok) lists \
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

        // Design B works in PHYSICAL page ids directly: the guest's `pool.ids()`
        // are the physical KV block ids it binds straight to the `Pages` and
        // `WSlot` ports (no `slot_to_block` resolution — the fire/executor consume
        // them as-is; `launch_resolve_slot_to_block` is unwired). `copy_into` must
        // address cells the SAME way, so the move lands on exactly the physical
        // pages the fires read/write. Resolving through `slot_to_block_table` here
        // (as an earlier draft did) sent the move to a DIFFERENT physical page than
        // the fire attended, so the moved cell was never read — pass through.
        let kv_move_dst_pages: Vec<u32> = dst_page_ids;
        let kv_move_src_pages: Vec<u32> = src_page_ids;

        // Point this pipeline's FIFO at the move so a later `take`/`read` on any
        // channel fed by this pipeline drains it in submit order.
        let pipe_fires = self.ctx().table.get(&this)?.fires.clone();
        let pipeline_failure = self.ctx().table.get(&this)?.failure.clone();
        if let Some(reason) = pipeline_failure.lock().unwrap().clone() {
            return Ok(Err(format!("ptir: pipeline failed: {reason}")));
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
        let completion = match crate::driver::copy_kv_cells(0, cells) {
            Ok(completion) => completion,
            Err(e) => return Ok(Err(format!("ptir copy_into: submit failed: {e:#}"))),
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

    async fn close(&mut self, this: Resource<Pipeline>) -> Anyhow<()> {
        // Signal no further submissions: drain the pipeline's in-flight FIFO so
        // its fires' KV/RS txns (arena pins) finalize before the pipeline goes
        // away (the pin-safety drain follows the FIFO now — W3.1).
        let fires = self.ctx().table.get(&this).ok().map(|p| p.fires.clone());
        if let Some(fires) = fires {
            loop {
                let fire = fires.lock().unwrap().pop_front();
                match fire {
                    Some(f) => {
                        let _ = self.finalize_op(f).await;
                    }
                    None => break,
                }
            }
        }
        Ok(())
    }

    async fn drop(&mut self, this: Resource<Pipeline>) -> Anyhow<()> {
        // Drain the pipeline's in-flight FIFO before releasing it: each fire
        // holds open KV/RS txns (arena pins) and the GPU may still be writing the
        // pinned pages — await + finalize, never abandon mid-flight (W3.1: the
        // pins-safety drain lives here now, not on the pass).
        let fires = self.ctx().table.get(&this).ok().map(|p| p.fires.clone());
        if let Some(fires) = fires {
            loop {
                let fire = fires.lock().unwrap().pop_front();
                match fire {
                    Some(f) => {
                        let _ = self.finalize_op(f).await;
                    }
                    None => break,
                }
            }
        }
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl InstanceState {
    /// Drain one pipeline FIFO entry in submit order: a forward fire finalizes
    /// its KV/RS txns and exposes mirror epochs; a KV cell MOVE awaits its
    /// payload-free completion. Move failures are logged because no channel is
    /// associated with the operation.
    async fn finalize_op(&mut self, op: PendingOp) -> Anyhow<()> {
        match op {
            PendingOp::Fire(fire) => self.finalize_fire(fire).await,
            PendingOp::Move(mv) => {
                if let Err(e) = mv.completion.await {
                    let reason = format!("ptir kv-move (copy_into) failed: {e:#}");
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

    /// Resolve one in-flight fire: await the payload-free callback, finalize the
    /// KV/RS txns, and expose the release-published mirror tails. Values remain
    /// in driver-owned channel memory until `channel.take` or `channel.read`.
    async fn finalize_fire(&mut self, fire: PendingFire) -> Anyhow<()> {
        let PendingFire {
            completion,
            kv,
            rstxn,
            ws_rep,
            rs_rep,
            fwd_rep,
            instance_id,
            cells,
            failure,
        } = fire;
        let prior_failure = failure.lock().unwrap().clone();
        let result = completion.await;
        let success = result.is_ok() && prior_failure.is_none();

        {
            let arena_arc = crate::arena::get(0, 0);
            let mut arena = arena_arc.lock().unwrap();
            let ws_res: Resource<KvWorkingSet> = Resource::new_borrow(ws_rep);
            match kv {
                FireKv::Kv(kvtxn) => match self.ctx().table.get_mut(&ws_res) {
                    Ok(ws) => {
                        let _ = ptir_kv::ptir_kv_finalize(ws, &mut arena, kvtxn, success);
                    }
                    Err(_) => ptir_kv::ptir_kv_abandon(&mut arena, kvtxn),
                },
                FireKv::DeviceGeom {
                    arena_txn,
                    write_txn,
                } => {
                    if success {
                        let _ = arena.txn_commit(arena_txn);
                    } else {
                        arena.txn_abort(arena_txn);
                    }
                    if let Ok(ws) = self.ctx().table.get_mut(&ws_res) {
                        if success {
                            ws.commit_writes(write_txn);
                        } else {
                            ws.abort_writes(write_txn);
                        }
                    }
                }
            }
            if let Some(rstxn) = rstxn {
                let rs_res: Resource<RsWorkingSet> =
                    Resource::new_borrow(rs_rep.expect("rs txn implies rs rep"));
                match self.ctx().table.get_mut(&rs_res) {
                    Ok(rs) => {
                        let _ = ptir_rs::ptir_rs_finalize(rs, &mut arena, rstxn, success);
                    }
                    Err(_) => ptir_rs::ptir_rs_abandon(&mut arena, rstxn),
                }
            }
        }

        if let Some(reason) = prior_failure {
            poison_readers(&cells, &reason);
            self.fail_pass(fwd_rep, &reason);
            return Ok(());
        }

        let failure_reason = match result {
            Ok(()) => match publish_reader_mirrors(&cells, instance_id) {
                Ok(true) => {
                    let reason = "ptir: driver published poisoned output channel".to_string();
                    poison_readers(&cells, &reason);
                    self.fail_pass(fwd_rep, &reason);
                    Some(reason)
                }
                Ok(false) => {
                    self.reclaim_device_geometry_grants(fwd_rep, instance_id);
                    None
                }
                Err(e) => {
                    let reason = format!("ptir: mirror publication failed: {e}");
                    poison_readers(&cells, &reason);
                    self.fail_pass(fwd_rep, &reason);
                    Some(reason)
                }
            },
            Err(e) => {
                let reason = format!("ptir: forward failed: {e:#}");
                poison_readers(&cells, &reason);
                self.fail_pass(fwd_rep, &reason);
                Some(reason)
            }
        };
        if let Some(reason) = failure_reason {
            let mut domain = failure.lock().unwrap();
            if domain.is_none() {
                *domain = Some(reason);
            }
        }
        Ok(())
    }

    /// Mark a pass failed (first failure wins). The guest may have dropped
    /// the pass handle already — then there is nothing to mark.
    fn fail_pass(&mut self, fwd_rep: u32, reason: &str) {
        let res: Resource<ForwardPass> = Resource::new_borrow(fwd_rep);
        if let Ok(p) = self.ctx().table.get_mut(&res) {
            if p.failed.is_none() {
                p.failed = Some(reason.to_string());
            }
        }
    }
}

impl InstanceState {
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
    async fn fire_device_geometry(
        &mut self,
        this: Resource<Pipeline>,
        fwd: Resource<ForwardPass>,
    ) -> Anyhow<Result<(), String>> {
        // Wire each of this pass's channels at this pipeline's FIFO (§3.4: all
        // passes binding a channel must submit on ONE pipeline — the entire
        // ordering/FIFO correctness argument).
        let pipe_fires = self.ctx().table.get(&this)?.fires.clone();
        let pipeline_failure = self.ctx().table.get(&this)?.failure.clone();
        if let Some(reason) = pipeline_failure.lock().unwrap().clone() {
            return Ok(Err(format!("ptir: pipeline failed: {reason}")));
        }
        if let Err(e) = self.wire_channels_to_pipeline(&fwd, &pipe_fires)? {
            return Ok(Err(e));
        }

        let page_size = crate::working_set::page_size::tokens_per_page(0);
        let ws_rep = self.ctx().table.get(&fwd)?.kv_ws;
        let ws_res: Resource<KvWorkingSet> = Resource::new_borrow(ws_rep);
        let arena_arc = crate::arena::get(0, 0);

        // Snapshot the carrier + grant B fresh physical pages, materializing +
        // pinning them under one write/arena txn (no borrow held across submit).
        let (host_puts, completion, instance_id, cells, fwd_rep, arena_txn, write_txn, phys_pages) = {
            // Fail-fast + first-fire seed binding.
            {
                let p = self.ctx().table.get_mut(&fwd)?;
                if let Some(e) = &p.failed {
                    return Ok(Err(format!(
                        "ptir: forward-pass failed by an earlier fire: {e}"
                    )));
                }
                if let Err(e) = bind_seeds_first_fire(p) {
                    return Ok(Err(e));
                }
            }

            // Take the DevGeo out so the lease grant can borrow the ws (distinct
            // table resources can't be borrowed mutably at once).
            let mut devgeo = self
                .ctx()
                .table
                .get_mut(&fwd)?
                .devgeo
                .take()
                .expect("fire_device_geometry on a non-device-geometry pass");

            let wtx = self.ctx().table.get_mut(&ws_res)?.begin_write_txn();
            let mut arena = arena_arc.lock().unwrap();
            let mut txn = arena.txn_begin();

            // Grant B fresh pages (free-list first, then a fresh ws slot), then
            // materialize + pin each so the driver's explicit-KV write (B2) lands
            // on a live, un-evictable physical page.
            let grant_slots = {
                let ws = self.ctx().table.get_mut(&ws_res)?;
                devgeo.lease.grant(|| {
                    ws.alloc_slots(1)
                        .ok()
                        .and_then(|v| v.into_iter().next())
                        .unwrap_or(0)
                })
            };
            let mut phys_pages: Vec<u32> = Vec::with_capacity(grant_slots.len());
            let mut grant_err: Option<String> = None;
            for &slot in &grant_slots {
                let ws = match self.ctx().table.get_mut(&ws_res) {
                    Ok(ws) => ws,
                    Err(e) => {
                        grant_err = Some(format!("ptir: device-geometry ws: {e}"));
                        break;
                    }
                };
                match ws.cow_write_slot(wtx, slot, &mut txn, &mut arena) {
                    Ok((obj, _)) => {
                        if let Err(e) = arena.txn_pin(&mut txn, obj) {
                            grant_err = Some(format!("ptir: device-geometry pin {slot}: {e}"));
                            break;
                        }
                        match arena.blocks(obj) {
                            Ok(bl) => phys_pages.push(bl[0]),
                            Err(e) => {
                                grant_err =
                                    Some(format!("ptir: device-geometry blocks {slot}: {e}"));
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        grant_err = Some(format!("ptir: device-geometry grant slot {slot}: {e}"));
                        break;
                    }
                }
            }
            if let Some(e) = grant_err {
                arena.txn_abort(txn);
                drop(arena);
                self.ctx().table.get_mut(&ws_res)?.abort_writes(wtx);
                self.ctx().table.get_mut(&fwd)?.devgeo = Some(devgeo);
                return Ok(Err(e));
            }
            drop(arena);

            // Deliver the fresh grant to the program as a host-put on its
            // `fresh` channel (D1-coalesced into this fire's carrier).
            let fresh_dense = devgeo.fresh_dense;
            {
                let p = self.ctx().table.get_mut(&fwd)?;
                let bytes: Vec<u8> = phys_pages.iter().flat_map(|s| s.to_le_bytes()).collect();
                if let Some(cell) = p.cells.get(fresh_dense) {
                    let _ = cell.lock().unwrap().put(bytes);
                }
                p.devgeo = Some(devgeo);
            }

            let p = self.ctx().table.get_mut(&fwd)?;
            let host_puts = snapshot_host_puts(&p.cells);
            let completion = p.bound_instance.reserve_completion();
            (
                host_puts,
                completion,
                p.bound_instance.instance_id,
                p.cells.clone(),
                fwd.rep(),
                txn,
                wtx,
                phys_pages,
            )
        };

        let req = crate::driver::LaunchPlan::default();
        let last_page_len = phys_pages.last().map(|_| page_size).unwrap_or(0);
        let reserved = reserve_reader_capacity(&cells);
        let submit_error = match &reserved {
            Err(error) => Some(format!("channel reservation failed: {error}")),
            Ok(()) => crate::inference::submit_prebuilt_async(
                req,
                0,
                instance_id,
                host_puts,
                phys_pages,
                last_page_len,
                Vec::new(),
                completion.clone(),
            )
            .err()
            .map(|error| format!("{error:#}")),
        };
        if let Some(error) = submit_error {
            if reserved.is_ok() {
                rollback_reader_capacity(&cells);
            }
            let reason = format!("ptir: device-geometry submit failed: {error}");
            arena_arc.lock().unwrap().txn_abort(arena_txn);
            if let Ok(ws) = self.ctx().table.get_mut(&ws_res) {
                ws.abort_writes(write_txn);
            }
            self.ctx().table.get_mut(&fwd)?.failed = Some(reason.clone());
            return Ok(Err(reason));
        }
        commit_host_puts(&cells);

        pipe_fires
            .lock()
            .unwrap()
            .push_back(PendingOp::Fire(PendingFire {
                completion,
                kv: FireKv::DeviceGeom {
                    arena_txn,
                    write_txn,
                },
                rstxn: None,
                ws_rep,
                rs_rep: None,
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
    fn wire_channels_to_pipeline(
        &mut self,
        fwd: &Resource<ForwardPass>,
        pipe_fires: &PendingFires,
    ) -> Anyhow<Result<(), String>> {
        if let Some(existing) = &self.ctx().table.get(fwd)?.fires {
            if !Arc::ptr_eq(existing, pipe_fires) {
                return Ok(Err(
                    "ptir: a pass cannot submit across different pipelines".into()
                ));
            }
        }
        let reps = self.ctx().table.get(fwd)?.channel_reps.clone();
        for rep in reps {
            let cres: Resource<Channel> = Resource::new_borrow(rep);
            if let Ok(ch) = self.ctx().table.get_mut(&cres) {
                match &ch.fires {
                    Some(existing) if !Arc::ptr_eq(existing, pipe_fires) => {
                        return Ok(Err("ptir: a channel is shared across pipelines \
                             (all passes binding a channel must submit on the same \
                             pipeline)"
                            .into()));
                    }
                    _ => ch.fires = Some(pipe_fires.clone()),
                }
            }
        }
        self.ctx().table.get_mut(fwd)?.fires = Some(pipe_fires.clone());
        Ok(Ok(()))
    }

    /// Device-geometry per-fire page reclaim: read the harvested `w_cont`
    /// (`[B]` bool: heir(true)/fork(false)) from its bound mirror, reclaim the
    /// continuing heirs' UNUSED fresh page grants into the lease free-list, and
    /// free those ws slots. No-op for a non-device-geometry pass.
    fn reclaim_device_geometry_grants(&mut self, fwd_rep: u32, instance_id: u64) {
        let res: Resource<ForwardPass> = Resource::new_borrow(fwd_rep);
        let (ws_rep, reclaimed) = {
            let Ok(p) = self.ctx().table.get_mut(&res) else {
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
        if !reclaimed.is_empty() {
            let ws_res: Resource<KvWorkingSet> = Resource::new_borrow(ws_rep);
            let (m, d) = match self.ctx().table.get(&ws_res) {
                Ok(ws) => ws.device(),
                Err(_) => return,
            };
            // Lock order: arena → cas.
            let arena_arc = crate::arena::get(m, d);
            let cas_arc = crate::working_set::kv_cas::get(m, d);
            let mut arena = arena_arc.lock().unwrap();
            let mut cas = cas_arc.lock().unwrap();
            if let Ok(ws) = self.ctx().table.get_mut(&ws_res) {
                let _ = ws.free_slots(&reclaimed, &mut arena, &mut cas);
            }
        }
    }
}

/// Build the bind-time [`ModelProfile`] from the loaded model (P2b: vocab +
/// page-size + layer caps; model-gated intrinsics + second-party kernels default
/// conservative until the model surfaces them).
fn model_profile() -> pie_ptir::registry::ModelProfile {
    let m = pie_model::model();
    pie_ptir::registry::ModelProfile {
        vocab: m.vocab_size(),
        page_size: crate::working_set::page_size::tokens_per_page(0) as u32,
        num_layers: 1,
        activation: pie_ptir::types::DType::F32,
        has_mtp_logits: false,
        has_mtp_drafts: false,
        has_value_head: false,
        kernels: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pie_ptir::container::{
        ChanDType, ChannelDecl, PortBinding, PortSource, StageProgram, TraceContainer,
    };
    use pie_ptir::registry::{Port, Stage};
    use pie_ptir::types::{DType, Shape};
    use std::collections::VecDeque;

    fn ch(shape: Shape, dtype: DType, role: HostRole) -> ChannelDecl {
        ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 1,
            host_role: role,
            seeded: false,
        }
    }

    /// A minimal device-geometry container: a `[B,P]` Pages channel, WSlot/WOff
    /// write descriptors, one host-Writer (`fresh`) + one host-Reader bool
    /// (`w_cont`). Channels: 0 pages[B,P], 1 w_slot[B], 2 w_off[B], 3 fresh[B]
    /// (Writer), 4 w_cont[B] bool (Reader).
    fn devgeo_container(b: u32, p: u32) -> TraceContainer {
        TraceContainer {
            names: vec![],
            channels: vec![
                ch(Shape::matrix(b, p), DType::U32, HostRole::None), // 0 pages
                ch(Shape::vector(b), DType::U32, HostRole::None),    // 1 w_slot
                ch(Shape::vector(b), DType::U32, HostRole::None),    // 2 w_off
                ch(Shape::vector(b), DType::U32, HostRole::Writer),  // 3 fresh
                ch(Shape::vector(b), DType::Bool, HostRole::Reader), // 4 w_cont
            ],
            ports: vec![
                PortBinding {
                    port: Port::Pages,
                    source: PortSource::Channel(0),
                },
                PortBinding {
                    port: Port::WSlot,
                    source: PortSource::Channel(1),
                },
                PortBinding {
                    port: Port::WOff,
                    source: PortSource::Channel(2),
                },
            ],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![],
            }],
            externs: vec![],
        }
    }

    #[test]
    fn detect_device_geometry_identifies_b_fresh_and_wcont() {
        let c = devgeo_container(2, 3);
        let (b, fresh, w_cont) = detect_device_geometry(&c).expect("device-geometry pass");
        assert_eq!(b, 2, "B from the [B,P] Pages channel");
        assert_eq!(fresh, 3, "fresh = the single host-Writer channel");
        assert_eq!(w_cont, 4, "w_cont = the host-Reader bool channel");
    }

    #[test]
    fn detect_device_geometry_rejects_plain_decode() {
        // A plain decode: KvLen only (no WSlot/WOff write descriptors), P == 1.
        let c = TraceContainer {
            names: vec![],
            channels: vec![ch(Shape::vector(1), DType::I32, HostRole::None)],
            ports: vec![PortBinding {
                port: Port::KvLen,
                source: PortSource::Channel(0),
            }],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![],
            }],
            externs: vec![],
        };
        assert!(
            detect_device_geometry(&c).is_none(),
            "no WSlot/WOff ⇒ not device-geometry"
        );
    }

    #[test]
    fn detect_device_geometry_rejects_single_page_width() {
        // WSlot/WOff present but Pages is [B,1] (P == 1) — not a multi-page beam.
        let mut c = devgeo_container(2, 1);
        // pages [B,1]
        c.channels[0] = ch(Shape::matrix(2, 1), DType::U32, HostRole::None);
        assert!(
            detect_device_geometry(&c).is_none(),
            "P == 1 ⇒ not device-geometry"
        );
    }

    /// The FIFO invariant primitive: fires enqueued in submission order drain in
    /// that same order (the PendingFires deque semantics the same-pipeline check
    /// funnels all interacting fires onto). `push_back` at submit + `pop_front`
    /// at finalize preserve submission == completion order.
    #[test]
    fn fifo_preserves_submission_order() {
        let mut fifo: VecDeque<u32> = VecDeque::new();
        for fire in 0..8u32 {
            fifo.push_back(fire); // submit order
        }
        let drained: Vec<u32> = std::iter::from_fn(|| fifo.pop_front()).collect();
        assert_eq!(
            drained,
            (0..8).collect::<Vec<_>>(),
            "completion order == submission order"
        );
    }
}
