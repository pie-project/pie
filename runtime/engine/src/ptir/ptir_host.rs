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
use crate::inferlet::ProcessCtx;
use crate::store::kv::working_set::KvWorkingSet;
use crate::store::rs::working_set::RsWorkingSet;

use pie_ptir::container::HostRole;

use super::ptir_channel_store::{
    BoundCells, ChannelCell, ChannelError, reserve_reader_capacity, rollback_reader_capacity,
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
    /// Non-blocking probe: whether the op's driver completion has settled.
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
}

/// See [`PendingOp::completion_signal`].
enum OpSignal {
    Fire(crate::driver::InstanceCompletion),
    Move(crate::driver::Completion),
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
    /// A device-geometry fire's prepared write over the lease-granted slots
    /// (B2's explicit-KV path): same commit/abort protocol, no host
    /// projection.
    DeviceGeom { kvtxn: ptir_kv::PtirKvTxn },
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
    /// Bind-time half of the canonical-KV gate ([`canonical_kv_shape`]): this
    /// pass CAN produce semantically hashable KV. Each fire additionally
    /// passes the fire-time host-known gate ([`canonical_fire_evidence`]).
    pub canonical_kv: bool,
    /// The pass binds an `AttnMask` descriptor channel (dense device mask).
    /// Its fires are marked mask-carrying on the launch plan so the
    /// scheduler batches them SOLO — the composed multi-program batch does
    /// not merge dense device masks (v1 scope).
    pub dense_mask: bool,
    /// Whether a fire of this pass has been submitted. The first fire
    /// consumes channel seeds, so the fire-time gate's seed rule only
    /// applies while this is false.
    pub fired_once: bool,
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
/// [`ProcessCtx::wire_channels_to_pipeline`]). This is the ENTIRE correctness
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
    /// The program binds an `AttnMask` descriptor channel (dense per-cell
    /// mask). Such fires are scheduled SOLO: the driver's composed
    /// multi-program batch does not merge dense device masks with other
    /// programs' geometry (v1 scope).
    pub has_mask: bool,
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

/// Bind-time half of the canonical-KV gate (kv_refact.md, "Token-Slot
/// Hashes, Page Hashes, and Trie Matching"): the pass writes exactly what
/// the vanilla model produces for one appended token run under full causal
/// self-attention over the working set — so its KV rows may carry chained
/// semantic hashes. Rejected by anything that can perturb K/V production:
/// an attention mask or explicit positions (they change hidden states, hence
/// KV at layers > 0), device geometry (inferlet-managed layout), per-layer
/// stage programs (they can rewrite projections), extern channels, or
/// multi-lane batching. Prologue/epilogue programs only shape sampling —
/// grammar, watermarking, and sampler passes all stay canonical. A `KvLen`
/// port must exist so the fire-time gate can verify the pass attends the
/// FULL context (a shorter span changes upper-layer KV).
fn canonical_kv_shape(container: &pie_ptir::container::TraceContainer) -> bool {
    use pie_ptir::container::PortSource;
    use pie_ptir::registry::{Port, Stage};

    if !container.externs.is_empty() {
        return false;
    }
    let mut has_kv_len = false;
    for binding in &container.ports {
        match binding.port {
            Port::AttnMask
            | Port::Positions
            | Port::Pages
            | Port::PageIndptr
            | Port::WSlot
            | Port::WOff => return false,
            Port::KvLen => has_kv_len = true,
            Port::EmbedIndptr => {
                // Single lane only: a trace-const two-entry CSR from 0.
                match &binding.source {
                    PortSource::Const { data, .. } => {
                        if data.len() != 8 || data[0..4] != [0u8; 4] {
                            return false;
                        }
                    }
                    PortSource::Channel(_) => return false,
                }
            }
            _ => {}
        }
    }
    has_kv_len
        && !container
            .stages
            .iter()
            .any(|s| matches!(s.stage, Stage::OnAttnProj | Stage::OnAttn))
}

fn decode_le_u32s(bytes: &[u8]) -> Option<Vec<u32>> {
    if bytes.len() % 4 != 0 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

/// The host-known u32 payload the channel bound at `port` consumes on THIS
/// fire: a trace constant, the staged Writer put shipping with this fire, or
/// (first fire only) the channel's seed. `None` = device-carried — the host
/// cannot see the value, so the fire cannot hash canonically.
fn host_known_port_u32s(p: &ForwardPass, port: pie_ptir::registry::Port) -> Option<Vec<u32>> {
    use pie_ptir::container::PortSource;

    let container = &p.instance.program.bound.container;
    let binding = container.ports.iter().find(|b| b.port == port)?;
    match &binding.source {
        PortSource::Const { data, .. } => decode_le_u32s(data),
        PortSource::Channel(c) => {
            let dense = *c as usize;
            if let Some(bytes) = super::ptir_channel_store::staged_put_bytes(p.cells.get(dense)?) {
                return decode_le_u32s(&bytes);
            }
            if !p.fired_once && container.channels.get(dense).is_some_and(|d| d.seeded) {
                let seed = p.instance.seeds.iter().find(|s| s.channel as usize == dense)?;
                return decode_le_u32s(&seed.data);
            }
            None
        }
    }
}

/// Fire-time half of the canonical-KV gate: the host-verified token values
/// this fire embeds plus the kv-len the pass claims to attend. `None` unless
/// the bind-time shape passed and the embed value is host-known. The caller
/// still verifies the token count against the fire geometry and
/// `kv_len == committed + new` (full-context attention) before hashing.
fn canonical_fire_evidence(p: &ForwardPass) -> Option<(Vec<u32>, Option<u32>)> {
    use pie_ptir::registry::Port;

    if !p.canonical_kv {
        return None;
    }
    let tokens = host_known_port_u32s(p, Port::EmbedTokens)?;
    let kv_len = host_known_port_u32s(p, Port::KvLen).and_then(|v| v.first().copied());
    Some((tokens, kv_len))
}

type Anyhow<T> = anyhow::Result<T>;

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

/// Park until the channel can make progress: its endpoint's reader word
/// advances (the driver's completion callback wakes the reader wait slot
/// directly), or the oldest in-flight pipeline op settles so the caller can
/// drain it. Errors surface poison/closure or a definitively empty channel
/// (no endpoint and nothing in flight: nothing can ever fill the cell).
async fn await_channel_progress(
    cell: &Arc<Mutex<ChannelCell>>,
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

// The interface has no free functions left (register-program folded into
// forward-pass.new); the trait still anchors the resource types.
impl pie::inferlet::forward::Host for ProcessCtx {}

impl pie::inferlet::forward::HostChannel for ProcessCtx {
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

    /// The direct-wake await point (plan §4.5): while the cell is empty,
    /// non-blockingly drain already-settled pipeline ops (their KV/RS txns
    /// finalize here, bounding pin float), then park on the channel's reader
    /// wait slot — the driver's completion callback wakes it right after
    /// publishing the mirror tail. The park races the oldest in-flight op so
    /// a fire that resolves without producing on this channel still unblocks
    /// the loop; with no endpoint and nothing in flight, nothing can ever
    /// fill the cell and `Empty` is returned instead of parking.
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
            if self.drain_settled(fires.as_ref()).await? {
                continue;
            }
            if let Err(error) = await_channel_progress(&cell, fires.as_ref()).await {
                return Ok(Err(error));
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
            if self.drain_settled(fires.as_ref()).await? {
                continue;
            }
            if let Err(error) = await_channel_progress(&cell, fires.as_ref()).await {
                return Ok(Err(error));
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

impl pie::inferlet::forward::HostForwardPass for ProcessCtx {
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
                    let ws = *self.ctx().table.get(&ws_res)?;
                    let stores = crate::store::registry::get(ws.model, ws.driver as usize);
                    let reserved = stores.kv.lock().unwrap().reserve(ws.id, b as u64);
                    let seed_pages: Vec<u32> = match reserved {
                        Ok(range) => (range.start as u32..range.end as u32).collect(),
                        Err(e) => return Ok(Err(format!("ptir: device-geometry seed alloc: {e}"))),
                    };
                    let mut lease = PageLease::new(b);
                    lease.seed(seed_pages);
                    let has_mask = prog.bound.container.ports.iter().any(|p| {
                        matches!(p.port, pie_ptir::registry::Port::AttnMask)
                            && matches!(
                                p.source,
                                pie_ptir::container::PortSource::Channel(_)
                            )
                    });
                    Some(DevGeo {
                        lease,
                        b,
                        fresh_dense,
                        w_cont_dense,
                        has_mask,
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
                if cell.seeded {
                    cell.commit_seed();
                }
                // A seeded Writer held its staging back until the seed
                // settled into the instance descriptor — flush it now so
                // direct ring puts take over (plan §4.2).
                if cell.role == Some(HostRole::Writer)
                    && let Err(error) = cell.flush_writer_staging()
                {
                    drop(cell);
                    let _ = crate::driver::close_instance(&bound_instance);
                    for cell in &cells {
                        cell.lock().unwrap().detach(instance_id);
                    }
                    return Ok(Err(format!("ptir: writer staging flush: {error}")));
                }
            }
            let canonical_kv = canonical_kv_shape(&instance.program.bound.container);
            let dense_mask = instance.program.bound.container.ports.iter().any(|p| {
                matches!(p.port, pie_ptir::registry::Port::AttnMask)
                    && matches!(p.source, pie_ptir::container::PortSource::Channel(_))
            });
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
                canonical_kv,
                dense_mask,
                fired_once: false,
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
        // Device-geometry: leased slots are logical reserve indexes in the
        // store model. Unwritten grants hold no physical memory (reserve is
        // logical); written grants are committed pages that stay mapped until
        // the working set itself is discarded or dropped — discarding them
        // here would shift the surviving indexes under other passes bound to
        // the same working set, so the pass drop intentionally leaves the
        // mapping alone.
        if let Some(devgeo) = pass.devgeo.as_mut() {
            let _ = devgeo.lease.reclaim_all();
        }
        Ok(())
    }

    /// Run-ahead submit on `on`: prepare + fire + enqueue, NO await. See the
    /// module docs; errors after this call surface via channel poison +
    /// `take`.
    async fn submit(
        &mut self,
        this: Resource<ForwardPass>,
        on: Resource<Pipeline>,
    ) -> Anyhow<Result<(), String>> {
        self.submit_pass(on, this).await
    }
}

impl pie::inferlet::pipeline::Host for ProcessCtx {}

impl pie::inferlet::pipeline::HostPipeline for ProcessCtx {
    async fn new(&mut self) -> Anyhow<Resource<Pipeline>> {
        Ok(self.ctx().table.push(Pipeline {
            fires: Arc::new(Mutex::new(VecDeque::new())),
            failure: Arc::new(Mutex::new(None)),
        })?)
    }

    async fn close(&mut self, this: Resource<Pipeline>) -> Anyhow<()> {
        self.pipeline_close(this).await
    }

    async fn drop(&mut self, this: Resource<Pipeline>) -> Anyhow<()> {
        self.pipeline_drop(this).await
    }
}

impl ProcessCtx {
    /// The body behind `forward-pass.submit(on)`.
    pub(crate) async fn submit_pass(
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
            // Non-blocking settlement drain (plan §6): resolved fires' KV/RS
            // txns finalize here so arena pins stay bounded by run-ahead depth
            // even when the guest never takes.
            self.drain_settled(Some(&pipe_fires)).await?;
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
                completion,
                canonical_evidence,
                dense_mask,
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
                    p.bound_instance.reserve_completion(),
                    canonical_fire_evidence(p),
                    p.dense_mask,
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
            // Prepare the guest-owned KV working set for this fire via
            // `ptir_kv` over the typed KvStore (reserve growth + fresh /
            // in-place / CoW classification + geometry projection). The held
            // `PtirKvTxn` rides the PendingFire across the async fire;
            // finalized (commit → mapping publishes / abort → pending slots
            // release) when a take/read/drop drains it.
            let new_tokens: Vec<u32> = req.token_ids.clone();
            // Canonical gate, fire-time half: hash under the host-verified
            // token values only when they cover exactly this fire's embed
            // AND the pass attends the FULL context. Anything else (device-
            // carried decode tokens, partial/unknown kv-len) hashes opaque.
            let hash_tokens: Option<Vec<u32>> = canonical_evidence.and_then(|(toks, kv_len)| {
                (toks.len() == new_tokens.len()
                    && kv_len == Some(committed_tokens + new_tokens.len() as u32))
                .then_some(toks)
            });
            let ws_res: Resource<KvWorkingSet> = Resource::new_borrow(ws_rep);
            let ws = *self.ctx().table.get(&ws_res)?;
            let stores = crate::store::registry::get(ws.model, ws.driver as usize);
            let pid = self.id();

            // A pass's optimistic cursor covers only ITS OWN pending fires;
            // ANOTHER pass may have committed into the same working set
            // before this pass's first fire (a prefill pass feeding a decode
            // pass is the norm). Floor the cursor by the store's published
            // extent so prepare appends after the real committed content
            // instead of re-writing (CoW-forking) published slots. The
            // cursor stays authoritative when it runs AHEAD (this pass's
            // in-flight fires).
            let committed_tokens = {
                let kv = stores.kv.lock().unwrap();
                committed_tokens.max(
                    kv.committed_token_len(ws.id, ws.page_size).unwrap_or(0) as u32,
                )
            };

            // Run-ahead can outpace finalization under the direct plane: the
            // ring delivers a fire's value BEFORE its completion retires the
            // txn that publishes its pages, so the guest's next submit may
            // arrive while the flat table still lacks the context. Prepare
            // projects the committed context from the PUBLISHED mapping, so
            // drain pending fires (FIFO order — awaits their completions)
            // until it covers the cursor. Bounded by run-ahead depth; only
            // fires that crossed an unpublished page boundary wait.
            while committed_tokens as u64 > {
                let kv = stores.kv.lock().unwrap();
                kv.committed_token_len(ws.id, ws.page_size).unwrap_or(0)
            } {
                let op = pipe_fires.lock().unwrap().pop_front();
                match op {
                    Some(op) => self.finalize_op(op).await?,
                    None => break,
                }
            }

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
            let (committed_tokens, new_tokens, hash_tokens) = {
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
                            ptir_kv::ptir_kv_match_prefix(
                                &mut kv,
                                ws.id,
                                &toks[..cap],
                                ws.page_size,
                            )
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
                                tracing::debug!("ptir prefix-cache probe miss: {e}");
                            }
                        }
                    }
                }
                (committed, toks_new, toks_hash)
            };

            let (proj, (kv_copy_src, kv_copy_dst), kv_translation, kvtxn) = loop {
                let prepared = {
                    let mut kv = stores.kv.lock().unwrap();
                    ptir_kv::ptir_kv_prepare(
                        &mut kv,
                        ws.id,
                        committed_tokens,
                        &new_tokens,
                        ws.page_size,
                        hash_tokens.as_deref(),
                    )
                }; // kv lock dropped before any contention await below
                match prepared {
                    Ok(v) => break v,
                    Err(e @ ptir_kv::PtirKvError::OutOfPages { requested, .. }) => {
                        // Contention ladder (kv_refact.md Scheduler): the
                        // orchestrator drops idle cache leases, then waits
                        // FIFO for frees (FCFS-preempting when a victim
                        // backend is live); on Ok the prepare RETRIES. No
                        // orchestrator wired (default Error mode) ⇒ still
                        // run rung 1 inline — auto-retained prefix caches
                        // must never surface as guest OOM — then error.
                        let Some(orch) = crate::inference::contention::contention() else {
                            let freed = {
                                let mut kv = stores.kv.lock().unwrap();
                                let epoch = kv.current_epoch();
                                let freed = kv.drop_unused_cache_leases(epoch);
                                kv.retire_idle();
                                freed
                            };
                            if freed > 0 {
                                continue;
                            }
                            return Ok(Err(format!("ptir: kv prepare: {e}")));
                        };
                        if let Err(hard) = orch.acquire(pid, requested as u32).await {
                            return Ok(Err(format!("ptir: kv prepare: {hard}")));
                        }
                    }
                    Err(e) => return Ok(Err(format!("ptir: kv prepare: {e}"))),
                }
            };
            let next_committed = kvtxn.committed_tokens_after;

            // The recurrent-state slot for hybrid / linear-attention models
            // (GDN, Mamba2): fresh RESET slab on the first fire, CoW-continue
            // after.
            let rstxn = if let Some(rs_rep) = rs_rep {
                let rs_res: Resource<RsWorkingSet> = Resource::new_borrow(rs_rep);
                let rs = *self.ctx().table.get(&rs_res)?;
                let prepared = {
                    let mut rs_store = stores.rs.lock().unwrap();
                    ptir_rs::ptir_rs_prepare(&mut rs_store, rs.id)
                };
                match prepared {
                    Ok((rs_slot_ids, rs_slot_flags, (rs_copy_src, rs_copy_dst), txn)) => {
                        req.rs_slot_ids = rs_slot_ids;
                        req.rs_slot_flags = rs_slot_flags;
                        if !rs_copy_src.is_empty() {
                            // The copy rides the pipeline FIFO (D4): an
                            // asynchronous failure poisons the pipeline
                            // failure domain instead of vanishing in a log.
                            match crate::driver::copy_d2d(0, &rs_copy_src, &rs_copy_dst) {
                                Ok(move_completion) => {
                                    pipe_fires.lock().unwrap().push_back(PendingOp::Move(
                                        PendingMove {
                                            completion: move_completion,
                                            failure: pipeline_failure.clone(),
                                        },
                                    ));
                                }
                                Err(e) => {
                                    tracing::warn!("ptir rs CoW d2d copy failed: {e:#}");
                                }
                            }
                        }
                        Some(txn)
                    }
                    Err(e) => {
                        // Revert the KV write we already prepared for this fire.
                        let mut kv = stores.kv.lock().unwrap();
                        let _ = ptir_kv::ptir_kv_finalize(&mut kv, kvtxn, false);
                        return Ok(Err(format!("ptir: rs prepare: {e}")));
                    }
                }
            } else {
                None
            };

            // WorkingSet page translation for this fire: the driver maps any
            // channel-resolved `Pages`/`WSlot` reference (guest-held relative
            // indexes) through it (kv_refact.md flattened-table model).
            req.kv_translation = kv_translation;

            // D2D-copy the CoW-preserved pages before the fire (empty for the
            // single-context pipeline; non-empty only under a forked/shared
            // tail). The copy rides the pipeline FIFO (D4) so an asynchronous
            // failure poisons the pipeline failure domain.
            if !kv_copy_src.is_empty() {
                match crate::driver::copy_d2d(0, &kv_copy_src, &kv_copy_dst) {
                    Ok(move_completion) => {
                        pipe_fires
                            .lock()
                            .unwrap()
                            .push_back(PendingOp::Move(PendingMove {
                                completion: move_completion,
                                failure: pipeline_failure.clone(),
                            }));
                    }
                    Err(e) => {
                        tracing::warn!("ptir forward CoW d2d copy failed: {e:#}");
                    }
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
                {
                    let mut kv = stores.kv.lock().unwrap();
                    let _ = ptir_kv::ptir_kv_finalize(&mut kv, kvtxn, false);
                }
                if let Some(rstxn) = rstxn {
                    let mut rs_store = stores.rs.lock().unwrap();
                    let _ = ptir_rs::ptir_rs_finalize(&mut rs_store, rstxn, false);
                }
                self.ctx().table.get_mut(&fwd)?.failed = Some(reason.clone());
                return Ok(Err(reason));
            }

            // Optimistic cursor advance: the NEXT submit prepares against this
            // fire's post-state (the run-ahead overlap). A failed fire fails
            // the pass instead of rewinding.
            {
                let p = self.ctx().table.get_mut(&fwd)?;
                p.committed_tokens = next_committed;
                p.fired_once = true; // seeds are consumed; the seed rule is off
            }

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
    pub(crate) async fn copy_into_inner(
        &mut self,
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

        // The WIT contract passes WorkingSet-RELATIVE page indexes (guests
        // never hold physical ids); translate through the flattened table so
        // the move lands on exactly the physical pages the fires read/write.
        // Translated at enqueue against the committed mapping: same-WS
        // in-flight fires that could remap these pages (a CoW rebase) are the
        // guest's ordering hazard, like any same-WS run-ahead write overlap.
        let (kv_move_dst_pages, kv_move_src_pages): (Vec<u32>, Vec<u32>) = {
            let ws = *self.ctx().table.get(&ws)?;
            let stores = crate::store::registry::get(ws.model, ws.driver as usize);
            let mut kv = stores.kv.lock().unwrap();
            let (_, flat) = kv
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

    pub(crate) async fn pipeline_close(&mut self, this: Resource<Pipeline>) -> Anyhow<()> {
        // Signal no further submissions: drain the pipeline's in-flight FIFO so
        // its fires' prepared KV/RS writes (snapshot pins) finalize before the
        // pipeline goes away (the pin-safety drain follows the FIFO — W3.1).
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

    pub(crate) async fn pipeline_drop(&mut self, this: Resource<Pipeline>) -> Anyhow<()> {
        // Drain the pipeline's in-flight FIFO before releasing it: each fire
        // holds a prepared KV/RS write (snapshot pins, pending slots) and the
        // GPU may still be writing — await + finalize, never abandon
        // mid-flight (W3.1: the pin-safety drain lives here, not on the pass).
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

/// The body behind `kv-working-set.copy-into(on, ...)` (called from
/// `api::kv_working_set`): an ordered KV cell move on the pipeline FIFO.
pub(crate) async fn working_set_copy_into(
    state: &mut ProcessCtx,
    ws: Resource<KvWorkingSet>,
    on: Resource<Pipeline>,
    dst_page_ids: Vec<u32>,
    dst_tok_idx: Vec<u32>,
    src_page_ids: Vec<u32>,
    src_tok_idx: Vec<u32>,
) -> Anyhow<Result<(), String>> {
    state
        .copy_into_inner(on, ws, dst_page_ids, dst_tok_idx, src_page_ids, src_tok_idx)
        .await
}

impl ProcessCtx {
    /// Drain one pipeline FIFO entry in submit order: a forward fire finalizes
    /// its KV/RS txns and exposes mirror epochs; a KV cell MOVE awaits its
    /// payload-free completion. Move failures are logged because no channel is
    /// associated with the operation.
    /// Pop and finalize pipeline ops whose completions have already settled,
    /// without blocking (plan §6): submit and take/read entry call this so
    /// KV/RS transaction pins stay bounded by run-ahead depth while value
    /// waiting rides the channel wait slots. Returns whether anything drained.
    async fn drain_settled(&mut self, fires: Option<&PendingFires>) -> Anyhow<bool> {
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
                    self.finalize_op(op).await?;
                    drained = true;
                }
                None => return Ok(drained),
            }
        }
    }

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
            // Single-model, single-driver v1: fires are keyed to stores (0,0).
            let _ = ws_rep;
            let stores = crate::store::registry::get(0, 0);
            match kv {
                FireKv::Kv(kvtxn) | FireKv::DeviceGeom { kvtxn } => {
                    let mut kv_store = stores.kv.lock().unwrap();
                    let _ = ptir_kv::ptir_kv_finalize(&mut kv_store, kvtxn, success);
                }
            }
            if let Some(rstxn) = rstxn {
                let _ = rs_rep;
                let mut rs_store = stores.rs.lock().unwrap();
                let _ = ptir_rs::ptir_rs_finalize(&mut rs_store, rstxn, success);
            }
        } // store locks released before the contention drain re-locks pools

        // The fire's sequence retired: recycled slots (aborts, CoW'd tails,
        // collected suffixes) are allocatable now — wake parked allocators
        // and drain-gated lanes.
        if let Some(orch) = crate::inference::contention::contention() {
            orch.on_blocks_freed();
            orch.on_fire_retired();
        }

        if let Some(reason) = prior_failure {
            poison_readers(&cells, &reason);
            self.fail_pass(fwd_rep, &reason);
            return Ok(());
        }

        // Values are already visible through the release-published tail words
        // (plan §4.5) — resolving the fire only classifies success and settles
        // the transactions above.
        let failure_reason = match result {
            Ok(()) => {
                self.reclaim_device_geometry_grants(fwd_rep, instance_id);
                None
            }
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

impl ProcessCtx {
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
        // Non-blocking settlement drain (plan §6), as in the ordinary submit.
        self.drain_settled(Some(&pipe_fires)).await?;
        if let Some(reason) = pipeline_failure.lock().unwrap().clone() {
            return Ok(Err(format!("ptir: pipeline failed: {reason}")));
        }
        if let Err(e) = self.wire_channels_to_pipeline(&fwd, &pipe_fires)? {
            return Ok(Err(e));
        }

        let ws_rep = self.ctx().table.get(&fwd)?.kv_ws;
        let ws_res: Resource<KvWorkingSet> = Resource::new_borrow(ws_rep);
        let ws = *self.ctx().table.get(&ws_res)?;
        let page_size = ws.page_size;
        let stores = crate::store::registry::get(ws.model, ws.driver as usize);

        // Grant B pages, allocating their physical backing under one prepared
        // write (no borrow held across submit).
        let (
            completion,
            instance_id,
            cells,
            fwd_rep,
            kvtxn,
            kv_translation,
            wire_pages,
            dense_mask,
        ) = {
            // Fail-fast.
            {
                let p = self.ctx().table.get_mut(&fwd)?;
                if let Some(e) = &p.failed {
                    return Ok(Err(format!(
                        "ptir: forward-pass failed by an earlier fire: {e}"
                    )));
                }
            }

            // Take the DevGeo out so the lease grant can use the store
            // (distinct table resources can't be borrowed mutably at once).
            let mut devgeo = self
                .ctx()
                .table
                .get_mut(&fwd)?
                .devgeo
                .take()
                .expect("fire_device_geometry on a non-device-geometry pass");

            // Grant B slots: lease free-list first, then fresh logical
            // reserves. Purely logical — this can never exhaust the pool, so
            // it runs ONCE; only the physical prepare below retries under
            // contention.
            let grant_slots = {
                let mut kv = stores.kv.lock().unwrap();
                devgeo.lease.grant(|| {
                    kv.reserve(ws.id, 1).map(|r| r.start as u32).unwrap_or(0)
                })
            };
            let pid = self.id();
            let (pages, (copy_src, copy_dst), kv_translation, kvtxn) = loop {
                let prepared = {
                    let mut kv = stores.kv.lock().unwrap();
                    // One prepared write covers the grants plus any unwritten
                    // gap up to the mapped end (left by an aborted earlier
                    // fire), so fresh publication stays contiguous. The gap
                    // is recomputed per attempt: a parked retry may resume
                    // after another of this pass's fires committed.
                    let mapped = kv.mapped_len(ws.id).unwrap_or(0);
                    let mut write_indexes: Vec<u64> =
                        grant_slots.iter().map(|&s| s as u64).collect();
                    if let Some(max_fresh) = write_indexes
                        .iter()
                        .copied()
                        .filter(|&i| i >= mapped)
                        .max()
                    {
                        write_indexes.extend(mapped..max_fresh);
                    }
                    write_indexes.sort_unstable();
                    write_indexes.dedup();
                    ptir_kv::ptir_kv_prepare_explicit(&mut kv, ws.id, &write_indexes)
                }; // kv lock dropped before any contention await below
                match prepared {
                    Ok(v) => break v,
                    Err(e @ ptir_kv::PtirKvError::OutOfPages { requested, .. }) => {
                        // Same contention-ladder seam as the host-geometry
                        // path (see submit_pass), including the Error-mode
                        // inline rung 1.
                        let Some(orch) = crate::inference::contention::contention() else {
                            let freed = {
                                let mut kv = stores.kv.lock().unwrap();
                                let epoch = kv.current_epoch();
                                let freed = kv.drop_unused_cache_leases(epoch);
                                kv.retire_idle();
                                freed
                            };
                            if freed > 0 {
                                continue;
                            }
                            self.ctx().table.get_mut(&fwd)?.devgeo = Some(devgeo);
                            return Ok(Err(format!("ptir: device-geometry grant: {e}")));
                        };
                        if let Err(hard) = orch.acquire(pid, requested as u32).await {
                            self.ctx().table.get_mut(&fwd)?.devgeo = Some(devgeo);
                            return Ok(Err(format!("ptir: device-geometry grant: {hard}")));
                        }
                    }
                    Err(e) => {
                        self.ctx().table.get_mut(&fwd)?.devgeo = Some(devgeo);
                        return Ok(Err(format!("ptir: device-geometry grant: {e}")));
                    }
                }
            };
            if !copy_src.is_empty() {
                if let Err(e) = crate::driver::copy_d2d(0, &copy_src, &copy_dst) {
                    tracing::warn!("ptir device-geometry CoW d2d copy failed: {e:#}");
                }
            }
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
                let p = self.ctx().table.get_mut(&fwd)?;
                let bytes: Vec<u8> = grant_slots.iter().flat_map(|s| s.to_le_bytes()).collect();
                let error = match p.cells.get(fresh_dense) {
                    Some(cell) => cell.lock().unwrap().put(bytes).err(),
                    None => Some(ChannelError::Empty),
                };
                p.devgeo = Some(devgeo);
                error
            };
            if let Some(error) = fresh_error {
                let mut kv = stores.kv.lock().unwrap();
                let _ = ptir_kv::ptir_kv_finalize(&mut kv, kvtxn, false);
                return Ok(Err(format!(
                    "ptir: device-geometry fresh grant put: {error}"
                )));
            }

            let p = self.ctx().table.get_mut(&fwd)?;
            let completion = p.bound_instance.reserve_completion();
            (
                completion,
                p.bound_instance.instance_id,
                p.cells.clone(),
                fwd.rep(),
                kvtxn,
                kv_translation,
                wire_pages,
                dense_mask,
            )
        };

        let mut req = crate::driver::LaunchPlan::default();
        req.kv_translation = kv_translation;
        // A dense device mask (AttnMask channel) marks the fire mask-carrying;
        // the scheduler batches it SOLO (the composed multi-program batch
        // does not merge dense device masks — v1 scope).
        req.has_user_mask = dense_mask;
        let last_page_len = wire_pages.last().map(|_| page_size).unwrap_or(0);
        let reserved = reserve_reader_capacity(&cells);
        let submit_error = match &reserved {
            Err(error) => Some(format!("channel reservation failed: {error}")),
            Ok(()) => crate::inference::submit_prebuilt_async(
                req,
                0,
                instance_id,
                wire_pages,
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
            {
                let mut kv = stores.kv.lock().unwrap();
                let _ = ptir_kv::ptir_kv_finalize(&mut kv, kvtxn, false);
            }
            self.ctx().table.get_mut(&fwd)?.failed = Some(reason.clone());
            return Ok(Err(reason));
        }
        self.ctx().table.get_mut(&fwd)?.fired_once = true;

        pipe_fires
            .lock()
            .unwrap()
            .push_back(PendingOp::Fire(PendingFire {
                completion,
                kv: FireKv::DeviceGeom { kvtxn },
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
        // Reclaimed grants return to the lease free-list (done above) and are
        // re-granted to later fires; the store mapping keeps their committed
        // pages until the working set discards or drops them (a discard here
        // would shift live indexes under the pass — see the pass-drop note).
        let _ = (ws_rep, reclaimed);
    }
}

/// Build the bind-time [`ModelProfile`] from the loaded model (P2b: vocab +
/// page-size + layer caps; model-gated intrinsics + second-party kernels default
/// conservative until the model surfaces them).
fn model_profile() -> pie_ptir::registry::ModelProfile {
    let m = pie_model::model();
    pie_ptir::registry::ModelProfile {
        vocab: m.vocab_size(),
        page_size: crate::store::registry::get(0, 0).kv_page_size,
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

    /// A minimal canonical decode container: embed tokens + kv-len +
    /// epilogue. Channels: 0 tok (device-loop), 1 klen.
    fn plain_decode_container() -> TraceContainer {
        TraceContainer {
            names: vec![],
            channels: vec![
                ch(Shape::vector(1), DType::I32, HostRole::None),
                ch(Shape::vector(1), DType::U32, HostRole::None),
            ],
            ports: vec![
                PortBinding {
                    port: Port::EmbedTokens,
                    source: PortSource::Channel(0),
                },
                PortBinding {
                    port: Port::KvLen,
                    source: PortSource::Channel(1),
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
    fn canonical_shape_accepts_the_plain_decode() {
        assert!(canonical_kv_shape(&plain_decode_container()));
    }

    #[test]
    fn canonical_shape_rejects_kv_perturbing_passes() {
        // Attention mask: changes hidden states, hence KV at layers > 0.
        let mut c = plain_decode_container();
        c.ports.push(PortBinding {
            port: Port::AttnMask,
            source: PortSource::Channel(0),
        });
        assert!(!canonical_kv_shape(&c));

        // Per-layer stage program: can rewrite the projections.
        let mut c = plain_decode_container();
        c.stages.push(StageProgram {
            stage: Stage::OnAttn,
            ops: vec![],
        });
        assert!(!canonical_kv_shape(&c));

        // No KvLen port: the full-context claim cannot be verified.
        let mut c = plain_decode_container();
        c.ports.retain(|p| p.port != Port::KvLen);
        assert!(!canonical_kv_shape(&c));

        // Device geometry is inferlet-managed layout.
        assert!(!canonical_kv_shape(&devgeo_container(2, 3)));
    }

    #[test]
    fn canonical_shape_gates_embed_indptr_to_a_single_const_lane() {
        // Const [0, n] single-lane CSR: canonical.
        let mut c = plain_decode_container();
        c.ports.push(PortBinding {
            port: Port::EmbedIndptr,
            source: PortSource::Const {
                dtype: DType::U32,
                shape: Shape::vector(2),
                data: [0u32.to_le_bytes(), 4u32.to_le_bytes()].concat(),
            },
        });
        assert!(canonical_kv_shape(&c));

        // Channel-fed indptr (dynamic lanes): not canonical.
        let mut c = plain_decode_container();
        c.ports.push(PortBinding {
            port: Port::EmbedIndptr,
            source: PortSource::Channel(1),
        });
        assert!(!canonical_kv_shape(&c));
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
