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
//! scheduler, and enqueues a [`PendingFire`] (the driver round-trip + the open
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
use crate::inference::paging;
use crate::instance::InstanceState;
use crate::working_set::kv::KvWorkingSet;
use crate::working_set::rs::RsWorkingSet;

use pie_ptir::container::HostRole;

use super::ptir_channel_store::drain_host_puts;
use super::ptir_channel_store::{
    marshal_response, BoundCells, ChannelCell, ChannelError,
};
use super::ptir_kv;
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
/// await (the fire is popped out, then awaited).
pub type PendingFires = Arc<Mutex<VecDeque<PendingFire>>>;

/// One in-flight PTIR fire: the driver round-trip plus everything needed to
/// finalize when it resolves — the open KV/RS txns (pins/CoW held until
/// commit/abort) and the bound cells to marshal outputs into (or poison on
/// failure). The PTIR mirror of the classic `PendingForward`.
pub struct PendingFire {
    rx: tokio::sync::oneshot::Receiver<anyhow::Result<crate::inference::ForwardOutput>>,
    kvtxn: ptir_kv::PtirKvTxn,
    rstxn: Option<ptir_rs::PtirRsTxn>,
    ws_rep: u32,
    rs_rep: Option<u32>,
    /// The owning pass, to fail it on a fire error (rep — the guest may have
    /// dropped the handle; failure marking is then moot).
    fwd_rep: u32,
    cells: BoundCells,
}

/// A traced forward pass bound to its first-class handles — one instance of a
/// (hash-deduped) registered program. The driver's persistent channel arena is
/// keyed by this pass's `instance_id`; a channel MAY bind to several passes
/// (multi-pass channels, W3.2) — the driver's global channel registry (W0.1)
/// resolves one shared device cell and the pipeline orders the fires (§3.4).
pub struct ForwardPass {
    pub instance: super::ptir_instance::PtirInstance,
    /// The bound channel cells, dense declaration order: Writer puts staged
    /// here are D1-coalesced into each fire's carrier; Reader cells the fire
    /// produces are marshaled back here for the guest to `take`/`read`.
    pub cells: BoundCells,
    /// Dense-channel-index → global-channel-id map (captured at bind from the
    /// bound cells). Rides every submission so the driver binds the trace's
    /// dense channel references to the global device channel registry.
    pub channel_ids: Vec<u64>,
    /// The bound channel resource reps (captured at `forward-pass.new`), so
    /// `submit` can point each channel's await queue at the feeding pipeline
    /// (W3.1: the pipeline owns the FIFO; a pass may bind to any pipeline).
    pub channel_reps: Vec<u32>,
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
    pub shipped: bool,
    /// Set when a fire of this pass failed: further submits error with the
    /// root cause (the KV cursor and device channel state are unspecified
    /// after a failed fire — the guest builds a fresh pass).
    pub failed: Option<String>,
    /// §6.2 beam host-replay state (Design X). `Some` iff this is a beam
    /// program whose [B,P] geometry is device-produced (`fire_geometry` can't
    /// resolve it) — then each submit fires the replayed multi-lane batch
    /// instead of the single-page projection. NOTE: the beam replay derives
    /// fire t+1's geometry from t's harvested outputs, so the beam path is
    /// inherently synchronous (submit awaits) — it does NOT run ahead. Slated
    /// for deletion once the driver resolves device-produced geometry.
    pub beam: Option<BeamRun>,
}

/// A run-ahead submission pipeline (overview §3): the ORDERING domain (W3.1).
/// Owns the in-flight fire FIFO — fires submitted here are issued in submission
/// order, so fire t's epilogue channel puts happen-before fire t+1's descriptor
/// reads, EXTENDED ACROSS PASSES (draft→verify chaining). `take`/`read` await
/// this FIFO via each channel's recorded pipeline. Submission order rides the
/// scheduler queue; completion order rides this FIFO.
pub struct Pipeline {
    pub fires: PendingFires,
}

/// Per-instance beam replay state carried across fires (Design X).
pub struct BeamRun {
    /// The host-tracked freeze/heir geometry state.
    pub state: super::ptir_beam::BeamState,
    /// The [B,P] geometry for the NEXT fire (seeded for fire 0; re-derived by
    /// `BeamState::step` after each fire from the harvested `out_par`).
    pub geom: super::ptir_beam::BeamGeometry,
    /// `[B]` tokens to embed next fire (seeded prompt; then the harvested `out`).
    pub toks: Vec<u32>,
    /// `[B]` decode positions (advance by 1 per fire).
    pub pos: Vec<u32>,
}

/// Detect a §6.2-style beam program: its geometry ports (`Pages`/`KvLen`) bind
/// DEVICE-produced channels `fire_geometry` can't resolve. Returns `(B, P)` read
/// from the `pages` channel's `[B, P]` shape (channel 0 by the beam convention).
fn detect_beam(container: &pie_ptir::container::TraceContainer) -> Option<(usize, usize)> {
    use pie_ptir::container::PortSource;
    use pie_ptir::registry::Port;
    // STRUCTURAL beam signal: the WSlot/WOff write descriptors are beam-specific
    // — a plain decode's `attn_working_set` binds only KvLen. This is robust to
    // seeding (unlike `fire_geometry`'s MissingChannelValue, which host-known
    // seeds for pages/klen/w_slot defeat → the beam would take the trivial path).
    let has_write_desc = container
        .ports
        .iter()
        .any(|p| matches!(p.port, Port::WSlot | Port::WOff));
    if !has_write_desc {
        return None;
    }
    // B, P from the [B, P] channel bound to the `Pages` port (P > 1 for a beam).
    let pages_ch = container.ports.iter().find_map(|p| match (&p.port, &p.source) {
        (Port::Pages, PortSource::Channel(c)) => Some(*c as usize),
        _ => None,
    })?;
    let dims = container.channels.get(pages_ch)?.shape.dims();
    if dims.len() == 2 && dims[1] > 1 {
        Some((dims[0] as usize, dims[1] as usize))
    } else {
        None
    }
}

type Anyhow<T> = anyhow::Result<T>;

/// Process-wide accumulator of PTIR device-resource release markers (W0.3):
/// global channel ids + instance ids whose device storage the driver must free.
/// Populated by `channel.drop` / `forward-pass.drop`, drained into the next
/// submitted `ForwardRequest` (rides any fire; a drop with no imminent fire
/// flushes on the runtime's next submit). Fixes the pre-existing driver
/// instance-map leak too.
static PENDING_RELEASE: Mutex<(Vec<u64>, Vec<u64>)> = Mutex::new((Vec::new(), Vec::new()));

/// Queue a dropped channel's global id for device release.
fn queue_channel_release(global_id: u64) {
    PENDING_RELEASE.lock().unwrap().0.push(global_id);
}

/// Queue a dropped pass's instance id for device release.
fn queue_instance_release(instance_id: u64) {
    PENDING_RELEASE.lock().unwrap().1.push(instance_id);
}

/// Drain queued release markers into a request about to be submitted.
fn drain_releases_into(req: &mut pie_driver_abi::ForwardRequest) {
    let mut g = PENDING_RELEASE.lock().unwrap();
    req.ptir_release_channel_ids.append(&mut g.0);
    req.ptir_release_instance_ids.append(&mut g.1);
}


/// First fire only: bind the pass's seeds — one staged `put` per `seeded`
/// channel, dense order (D2, per-instance data). A seeded channel with no
/// staged put is the old `MissingSeed` instantiate error, surfaced at the
/// first submit instead of hanging the fire.
fn bind_seeds_first_fire(p: &mut ForwardPass) -> Result<(), String> {
    if p.shipped {
        return Ok(());
    }
    let seeded: Vec<bool> =
        p.instance.program.bound.container.channels.iter().map(|c| c.seeded).collect();
    let mut seeds = Vec::new();
    for (i, is_seeded) in seeded.into_iter().enumerate() {
        if !is_seeded {
            continue;
        }
        // Multi-pass channels (W3.2): a shared seeded channel's staged seed is
        // consumed by the first pass to ship it; a later pass sharing the same
        // channel finds `seed_taken` and skips it (the device is already seeded
        // — the seed table is de-duped by global id, W0.2).
        {
            if p.cells[i].lock().unwrap().seed_taken {
                continue;
            }
        }
        match p.cells[i].lock().unwrap().take_seed() {
            Ok(data) => {
                seeds.push(super::ptir_instance::ChannelSeed { channel: i as u32, data })
            }
            Err(e) => return Err(format!("ptir: channel {i}: {e}")),
        }
    }
    p.instance.seeds = seeds;
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

// The interface has no free functions left (register-program folded into
// forward-pass.new); the trait still anchors the resource types.
impl pie::core::ptir::Host for InstanceState {}

impl pie::core::ptir::HostChannel for InstanceState {
    async fn new(
        &mut self,
        shape: Vec<u32>,
        dtype: pie::core::tensor::Dtype,
        capacity: u32,
    ) -> Anyhow<Resource<Channel>> {
        // Pure host bookkeeping — works with the `ptir` feature off too (the
        // WIT constructor cannot carry a result; the gate errors at
        // forward-pass.new / submit instead).
        use pie::core::tensor::Dtype;
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
        Ok(cell.lock().unwrap().put(value).map_err(|e| e.to_string()))
    }

    /// The run-ahead await point: while the cell is empty, await + finalize
    /// the pass's oldest in-flight fire (FIFO), then re-check. Errors when no
    /// in-flight fire remains (nothing will ever fill the cell — a genuinely
    /// blocking cross-task wait needs the wasi-p3 surface; a p2 guest awaiting
    /// here without a prior submit would deadlock itself) or the channel is
    /// poisoned (a fire that feeds it failed).
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
            let fire = fires.as_ref().and_then(|f| f.lock().unwrap().pop_front());
            match fire {
                Some(fire) => self.finalize_fire(fire).await?,
                None => return Ok(Err(ChannelError::Empty.to_string())),
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
            let fire = fires.as_ref().and_then(|f| f.lock().unwrap().pop_front());
            match fire {
                Some(fire) => self.finalize_fire(fire).await?,
                None => return Ok(Err(ChannelError::Empty.to_string())),
            }
        }
    }

    async fn drop(&mut self, this: Resource<Channel>) -> Anyhow<()> {
        // A pass that bound this channel holds its own Arc — dropping the
        // guest handle never dangles an in-flight fire. Queue the channel's
        // global id for device release (W0.3): device lifetime follows the WIT
        // resource drop.
        let ch = self.ctx().table.delete(this)?;
        let global_id = ch.cell.lock().unwrap().global_id;
        queue_channel_release(global_id);
        Ok(())
    }
}

impl pie::core::ptir::HostForwardPass for InstanceState {
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
                    if let Err(e) = c.matches_decl(&decls[i]) {
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

            // §6.2 beam: eagerly reserve B*P slots on the GUEST's working set
            // (the runtime owns the [B,P] layout: beam `l`'s pages occupy
            // [l*P, l*P+P)) + seed the replay state. A normal program's ws
            // starts as the guest left it — `ptir_kv::ptir_kv_prepare` grows
            // it per fire (the growing-KV decode/MTP lifecycle).
            let page_size = crate::working_set::page_size::tokens_per_page(0);
            let beam = match detect_beam(&prog.bound.container) {
                Some((b, p)) => {
                    let ws_res: Resource<KvWorkingSet> = Resource::new_borrow(ws_rep);
                    if let Err(e) = self.ctx().table.get_mut(&ws_res)?.alloc((b * p) as u32) {
                        return Ok(Err(format!("ptir: beam kv working-set alloc: {e}")));
                    }
                    let slot0: Vec<u32> = (0..b).map(|l| (l * p) as u32).collect();
                    let state = super::ptir_beam::BeamState::seeded(b, p, page_size, &slot0);
                    let geom = state.geometry();
                    Some(BeamRun {
                        state,
                        geom,
                        // Placeholder prompt token per beam (the toks seed in
                        // echo's beam_trace); the real prompt is refined
                        // during the 4090 bring-up.
                        toks: vec![1u32; b],
                        pos: vec![0u32; b],
                    })
                }
                None => None,
            };

            // All validation passed — stamp the container's roles onto the
            // cells (the bind point) and mint the instance identity (the
            // driver's persistent channel-arena key). The await FIFO is owned by
            // the PIPELINE now (W3.1), wired to the channels at submit.
            for (cell, decl) in cells.iter().zip(decls.iter()) {
                cell.lock().unwrap().bind(decl);
            }
            // Capture the dense-index → global-channel-id map now that the cells
            // are validated (multi-pass channels: a global id is stable across
            // every pass a channel binds into).
            let channel_ids: Vec<u64> =
                cells.iter().map(|c| c.lock().unwrap().global_id).collect();
            // Capture the bound channel resource reps so `submit` can point each
            // channel's await queue at the feeding pipeline (W3.1).
            let channel_reps: Vec<u32> = channels.iter().map(|c| c.rep()).collect();
            let instance = super::ptir_instance::PtirInstance {
                program: prog,
                instance_id: super::ptir_instance::next_instance_id(),
                // Seeds ride the channels' own staged puts — bound at the
                // first submit (D2, never part of identity).
                seeds: Vec::new(),
            };
            let res = self.ctx().table.push(ForwardPass {
                instance,
                cells,
                channel_ids,
                channel_reps,
                kv_ws: ws_rep,
                rs_ws: rs_rep,
                committed_tokens: 0,
                shipped: false,
                failed: None,
                beam,
            })?;
            Ok(Ok(res))
        }
    }

    async fn drop(&mut self, this: Resource<ForwardPass>) -> Anyhow<()> {
        // The pass's in-flight fires live on the PIPELINE's FIFO now (W3.1), not
        // the pass — draining them (pins safety) follows the pipeline's
        // drop/close, not the pass. The guest owns the channels (Arc'd cells)
        // and the working sets (their own resources); the pass releases only
        // itself. Queue the instance id for device release (W0.3): frees the
        // driver's persistent per-instance view (also fixes the map leak).
        let pass = self.ctx().table.delete(this)?;
        queue_instance_release(pass.instance.instance_id);
        Ok(())
    }
}

impl pie::core::ptir::HostPipeline for InstanceState {
    async fn new(&mut self) -> Anyhow<Resource<Pipeline>> {
        Ok(self.ctx().table.push(Pipeline {
            fires: Arc::new(Mutex::new(VecDeque::new())),
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
            // §6.2 beam: the [B,P] geometry is device-produced and replayed
            // from each fire's harvested outputs, so the beam path is
            // inherently synchronous — it awaits inside (see `ForwardPass::beam`).
            if self.ctx().table.get(&fwd)?.beam.is_some() {
                return self.fire_beam(fwd).await;
            }
            // W3.1: the PIPELINE owns the in-flight FIFO. Point each of this
            // pass's channels at this pipeline's queue so their `take`/`read`
            // await the right FIFO — enforcing the same-pipeline constraint
            // (§3.4): every pass binding a channel must submit on one pipeline.
            let pipe_fires = self.ctx().table.get(&this)?.fires.clone();
            {
                let reps = self.ctx().table.get(&fwd)?.channel_reps.clone();
                for rep in reps {
                    let cres: Resource<Channel> = Resource::new_borrow(rep);
                    if let Ok(ch) = self.ctx().table.get_mut(&cres) {
                        match &ch.fires {
                            Some(existing) if !Arc::ptr_eq(existing, &pipe_fires) => {
                                return Ok(Err(
                                    "ptir: a channel is shared across pipelines \
                                     (all passes binding a channel must submit on \
                                     the same pipeline)"
                                        .into(),
                                ));
                            }
                            _ => ch.fires = Some(pipe_fires.clone()),
                        }
                    }
                }
            }
            // Build the PTIR carrier for this fire (thrust-3 P2c host emit): ship
            // the container bytes + the seeds (drained off the seeded channels'
            // staged puts) on the pass's first fire, the hash + instance id only
            // thereafter (driver compile-cache + persistent arena); attach the
            // D1-coalesced host-puts drained from the bound Writer cells.
            let (submission, geometry, cells, ws_rep, rs_rep, committed_tokens, fwd_rep) = {
                let p = self.ctx().table.get_mut(&fwd)?;
                if let Some(e) = &p.failed {
                    return Ok(Err(format!(
                        "ptir: forward-pass failed by an earlier fire: {e}"
                    )));
                }
                if let Err(e) = bind_seeds_first_fire(p) {
                    return Ok(Err(e));
                }
                let ship = !p.shipped;
                p.shipped = true;
                let channel_ids = p.channel_ids.clone();
                let host_puts = drain_host_puts(&p.cells);
                let submission = p.instance.submission(ship, channel_ids, host_puts);
                // Host-known geometry prefill (token/positions/qo/readout for
                // seeded ports, e.g. a §3 single-seq decode). `None` ⇒ a port
                // binds a device-derived / ws / run-ahead channel the host can't
                // resolve — the driver fills the descriptor ports itself.
                let geometry = p.instance.fire_geometry(model_profile().page_size).ok();
                (
                    submission,
                    geometry,
                    p.cells.clone(),
                    p.kv_ws,
                    p.rs_ws,
                    p.committed_tokens,
                    fwd.rep(),
                )
            };
            let mut req = pie_driver_abi::ForwardRequest::default();
            req.push_ptir_program(&submission);
            if let Some(g) = &geometry {
                g.apply_to(&mut req);
            }
            // Ride any queued device-resource release markers on this fire (W0.3).
            drain_releases_into(&mut req);

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
            let rx = match crate::inference::submit_async(
                req,
                0,
                proj.physical_page_ids,
                proj.last_page_len,
                Vec::new(),
                None,
            ) {
                Ok(rx) => rx,
                Err(e) => {
                    let reason = format!("ptir: submit failed: {e:#}");
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
            };

            // Optimistic cursor advance: the NEXT submit prepares against this
            // fire's post-state (the run-ahead overlap). A failed fire fails
            // the pass instead of rewinding.
            self.ctx().table.get_mut(&fwd)?.committed_tokens = next_committed;

            pipe_fires.lock().unwrap().push_back(PendingFire {
                rx,
                kvtxn,
                rstxn,
                ws_rep,
                rs_rep,
                fwd_rep,
                cells,
            });
            Ok(Ok(()))
        }
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
                        let _ = self.finalize_fire(f).await;
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
                        let _ = self.finalize_fire(f).await;
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
    /// Resolve one in-flight fire: await the driver round-trip, finalize the
    /// KV/RS txns (commit on success / abort on failure; abandon if the guest
    /// dropped the working set mid-flight), then marshal the produced Reader
    /// cells — or, on failure, poison the pass's Reader channels + fail the
    /// pass. The PTIR mirror of the classic `await_and_finalize`.
    async fn finalize_fire(&mut self, fire: PendingFire) -> Anyhow<()> {
        let PendingFire { rx, kvtxn, rstxn, ws_rep, rs_rep, fwd_rep, cells } = fire;
        let result = rx.await;
        let success = matches!(result, Ok(Ok(_)));

        {
            let arena_arc = crate::arena::get(0, 0);
            let mut arena = arena_arc.lock().unwrap();
            let ws_res: Resource<KvWorkingSet> = Resource::new_borrow(ws_rep);
            match self.ctx().table.get_mut(&ws_res) {
                Ok(ws) => {
                    let _ = ptir_kv::ptir_kv_finalize(ws, &mut arena, kvtxn, success);
                }
                Err(_) => ptir_kv::ptir_kv_abandon(&mut arena, kvtxn),
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

        match result {
            Ok(Ok(crate::inference::ForwardOutput::Response(resp))) => {
                // The rich response carries the harvested Reader-channel cells;
                // marshal program 0's outputs back into the bound cells so the
                // guest's `take`/`read` see them.
                let produced: Vec<(u64, Vec<u8>)> = resp
                    .ptir_output_at(0)
                    .unwrap_or_default()
                    .into_iter()
                    .map(|c| (c.channel, c.bytes))
                    .collect();
                if let Err(e) = marshal_response(&cells, &produced) {
                    let reason = format!("ptir: output marshal failed: {e}");
                    poison_readers(&cells, &reason);
                    self.fail_pass(fwd_rep, &reason);
                }
            }
            Ok(Ok(_)) => {
                // A non-Response output (legacy token fast-path) carries no
                // PTIR channel cells — nothing to marshal.
            }
            Ok(Err(e)) => {
                let reason = format!("ptir: forward failed: {e:#}");
                poison_readers(&cells, &reason);
                self.fail_pass(fwd_rep, &reason);
            }
            Err(e) => {
                let reason = format!("ptir: forward channel closed: {e}");
                poison_readers(&cells, &reason);
                self.fail_pass(fwd_rep, &reason);
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

/// Parse a beam program-output channel's bytes as `[n]` little-endian u32 (tokens
/// are non-negative i32 → the same bit pattern).
fn beam_channel_u32(
    produced: &[pie_driver_abi::PtirChannelValue],
    channel: u64,
    n: usize,
) -> Option<Vec<u32>> {
    let cell = produced.iter().find(|c| c.channel == channel)?;
    if cell.bytes.len() < n * 4 {
        return None;
    }
    Some(
        cell.bytes[..n * 4]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

impl InstanceState {
    /// §6.2 beam host-replay fire (Design X). Builds the [B,P] decode batch from
    /// the replayed geometry (`BeamState`), resolves each beam's slots→physical in
    /// Rust (read pages via `resolve_read`; the write page via
    /// `write_slot_shared_inplace` for a HEIR continuing a shared tail, else
    /// `cow_write_slot` for a FORK), folds the B lanes via the existing batch
    /// assembly, fires the pre-assembled batch (bravo's `submit_prebuilt_async`),
    /// marshals the [B] program outputs back to the bound cells, and replays the
    /// epilogue (`BeamState::step` on the harvested `out_par`) for the next fire.
    ///
    /// SYNCHRONOUS BY CONSTRUCTION: fire t+1's geometry comes from t's harvested
    /// outputs, so this path awaits inside submit (no run-ahead). Slated for
    /// deletion once the driver resolves device-produced geometry.
    ///
    /// BRING-UP (4090, with charlie): the fire-0 / prompt seeding + the fresh-slot
    /// lifecycle (currently runtime-issued via `alloc_slots`) are refined against
    /// the value-verify vs the 3 beam goldens; the freeze/heir REPLAY itself is
    /// golden-verified host-side (`ptir_beam::tests`).
    async fn fire_beam(&mut self, fwd: Resource<ForwardPass>) -> Anyhow<Result<(), String>> {
        use crate::inference::request;

        let page_size = crate::working_set::page_size::tokens_per_page(0);

        // 1) Snapshot the beam geometry + PTIR carrier (no borrow across await).
        let (geom, toks, pos, submission, cells, channel_ids, ws_rep, b, p) = {
            let pl = self.ctx().table.get_mut(&fwd)?;
            if let Some(e) = &pl.failed {
                return Ok(Err(format!("ptir: forward-pass failed by an earlier fire: {e}")));
            }
            if let Err(e) = bind_seeds_first_fire(pl) {
                return Ok(Err(e));
            }
            let beam = pl.beam.as_ref().expect("fire_beam on a non-beam pass");
            let geom = beam.geom.clone();
            let toks = beam.toks.clone();
            let pos = beam.pos.clone();
            let (b, p) = (beam.state.b, beam.state.p);
            let ship = !pl.shipped;
            pl.shipped = true;
            let channel_ids = pl.channel_ids.clone();
            let host_puts = drain_host_puts(&pl.cells);
            let submission = pl.instance.submission(ship, channel_ids.clone(), host_puts);
            (geom, toks, pos, submission, pl.cells.clone(), channel_ids, pl.kv_ws, b, p)
        };

        // kvm → per-beam BRLE masks (1 query/beam).
        let (masks, _mask_indptr) = geom.masks(p, page_size);

        // 2) Resolve each beam's slots→physical + fold the B-lane batch under one
        //    KV write txn. HEIR writes go in-place (shared page preserved, alpha's
        //    `write_slot_shared_inplace`); FORK writes CoW-alloc a fresh page.
        let ws_res: Resource<KvWorkingSet> = Resource::new_borrow(ws_rep);
        let arena_arc = crate::arena::get(0, 0);
        let wtx = self.ctx().table.get_mut(&ws_res)?.begin_write_txn();

        let built = {
            let mut arena = arena_arc.lock().unwrap();
            let mut txn = arena.txn_begin();
            let ws = self.ctx().table.get_mut(&ws_res)?;

            let mut batch = request::new_batched_forward_request_with_capacity(b);
            let mut union_phys = Vec::new();
            let mut err: Option<String> = None;

            'lanes: for lane in 0..b {
                let np_b = geom.np[lane] as usize;
                // Read pages = the beam's live pages before the tail; the tail
                // (index np_b-1 = w_slot) is the write page.
                let mut read_pages = Vec::with_capacity(np_b.saturating_sub(1));
                for pp in 0..np_b.saturating_sub(1) {
                    let slot = geom.pages[lane * p + pp];
                    match ws.resolve_read(slot, 1).map(|o| o[0]) {
                        Ok(obj) => {
                            if let Err(e) = arena.txn_pin(&mut txn, obj) {
                                err = Some(format!("beam pin read {lane}/{pp}: {e}"));
                                break 'lanes;
                            }
                            match arena.blocks(obj) {
                                Ok(bl) => read_pages.push(bl[0]),
                                Err(e) => {
                                    err = Some(format!("beam blocks read {lane}/{pp}: {e}"));
                                    break 'lanes;
                                }
                            }
                        }
                        Err(e) => {
                            err = Some(format!("beam resolve_read slot {slot}: {e}"));
                            break 'lanes;
                        }
                    }
                }
                // Write page: heir (shared in-place) vs fork (CoW fresh).
                let w_slot = geom.w_slot[lane];
                let write_obj = if geom.w_cont[lane] {
                    match ws.write_slot_shared_inplace(wtx, w_slot) {
                        Ok(o) => o,
                        Err(e) => {
                            err = Some(format!("beam heir write slot {w_slot}: {e}"));
                            break 'lanes;
                        }
                    }
                } else {
                    match ws.cow_write_slot(wtx, w_slot, &mut txn, &mut arena) {
                        Ok((o, _)) => o,
                        Err(e) => {
                            err = Some(format!("beam fork write slot {w_slot}: {e}"));
                            break 'lanes;
                        }
                    }
                };
                if let Err(e) = arena.txn_pin(&mut txn, write_obj) {
                    err = Some(format!("beam pin write {lane}: {e}"));
                    break 'lanes;
                }
                let write_page = match arena.blocks(write_obj) {
                    Ok(bl) => bl[0],
                    Err(e) => {
                        err = Some(format!("beam blocks write {lane}: {e}"));
                        break 'lanes;
                    }
                };

                let write = paging::KvWrite {
                    slot_index: (np_b - 1) as u32,
                    page: write_page,
                    valid_len: geom.w_off[lane] + 1,
                };
                let ctx_valid = np_b.saturating_sub(1) as u32 * page_size;
                let proj = match paging::project_kv(&read_pages, ctx_valid, &[write], page_size) {
                    Ok(pr) => pr,
                    Err(e) => {
                        err = Some(format!("beam project_kv {lane}: {e:?}"));
                        break 'lanes;
                    }
                };
                union_phys.extend_from_slice(&proj.physical_page_ids);

                // Per-lane decode request (1 token, custom kvm mask) folded into
                // the batch by the EXISTING assembly (physical kv_page_indices,
                // kv_last_page_lens, qo_indptr=[0..=B]).
                let req_l = request::new_per_request(
                    0,
                    vec![toks[lane]],
                    vec![pos[lane]],
                    vec![masks[lane].clone()],
                    true,
                    None,
                    vec![0],
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    false,
                    None,
                    None,
                );
                request::append_request_with_options(
                    &mut batch,
                    &req_l,
                    &proj.physical_page_ids,
                    proj.last_page_len,
                    page_size,
                    false,
                );
            }

            match err {
                None => Ok((txn, batch, union_phys)),
                Some(e) => {
                    arena.txn_abort(txn);
                    Err(e)
                }
            }
        };

        let (txn, mut batch, union_phys) = match built {
            Ok(v) => v,
            Err(e) => {
                self.ctx().table.get_mut(&ws_res)?.abort_writes(wtx);
                return Ok(Err(e));
            }
        };

        // 3) Attach the PTIR carrier + fire the pre-assembled B-lane batch.
        batch.push_ptir_program(&submission);
        drain_releases_into(&mut batch);
        let rx = match crate::inference::submit_prebuilt_async(batch, 0, union_phys, 0, Vec::new()) {
            Ok(rx) => rx,
            Err(e) => {
                arena_arc.lock().unwrap().txn_abort(txn);
                self.ctx().table.get_mut(&ws_res)?.abort_writes(wtx);
                return Ok(Err(format!("beam submit_prebuilt: {e:#}")));
            }
        };
        let result = rx.await;

        // 4) Finalize the KV txn.
        {
            let mut arena = arena_arc.lock().unwrap();
            let ws = self.ctx().table.get_mut(&ws_res)?;
            if matches!(result, Ok(Ok(_))) {
                let _ = arena.txn_commit(txn);
                ws.commit_writes(wtx);
            } else {
                arena.txn_abort(txn);
                ws.abort_writes(wtx);
            }
        }

        // 5) Marshal the [B] program outputs back to the bound cells + replay the
        //    epilogue (step on out_par) for the next fire.
        match result {
            Ok(Ok(crate::inference::ForwardOutput::Response(resp))) => {
                let produced = resp.ptir_output_at(0).unwrap_or_default();
                let tuples: Vec<(u64, Vec<u8>)> =
                    produced.iter().map(|c| (c.channel, c.bytes.clone())).collect();
                if let Err(e) = marshal_response(&cells, &tuples) {
                    return Ok(Err(format!("beam output marshal failed: {e}")));
                }
                // out_par (dense ch14) = parent [B]; out (dense ch13) = survivor
                // tokens [B]. Resolve the dense indices to global ids for the
                // response lookup (outputs are keyed by global id under the new ABI).
                let out_par_id = channel_ids.get(14).copied().unwrap_or(14);
                let out_id = channel_ids.get(13).copied().unwrap_or(13);
                let parent = beam_channel_u32(&produced, out_par_id, b);
                let out = beam_channel_u32(&produced, out_id, b);
                // Runtime-issued fresh slots for the next step's forks.
                let fresh = match self.ctx().table.get_mut(&ws_res)?.alloc_slots(b as u32) {
                    Ok(f) => f,
                    Err(e) => return Ok(Err(format!("beam fresh alloc: {e}"))),
                };
                let pl = self.ctx().table.get_mut(&fwd)?;
                if let (Some(beam), Some(parent), Some(out)) = (pl.beam.as_mut(), parent, out) {
                    beam.geom = beam.state.step(&parent, &fresh);
                    beam.toks = out;
                    beam.pos.iter_mut().for_each(|x| *x += 1);
                }
                Ok(Ok(()))
            }
            Ok(Ok(_)) => Ok(Ok(())),
            Ok(Err(e)) => {
                let reason = format!("beam forward failed: {e:#}");
                poison_readers(&cells, &reason);
                self.fail_pass(fwd.rep(), &reason);
                Ok(Err(reason))
            }
            Err(e) => {
                let reason = format!("beam forward channel closed: {e}");
                poison_readers(&cells, &reason);
                self.fail_pass(fwd.rep(), &reason);
                Ok(Err(reason))
            }
        }
    }
}

/// Build the bind-time [`ModelProfile`] from the loaded model (P2b: vocab +
/// page-size + layer caps; model-gated intrinsics + second-party kernels default
/// conservative until the model surfaces them).
fn model_profile() -> pie_ptir::registry::ModelProfile {
    let m = crate::model::model();
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
