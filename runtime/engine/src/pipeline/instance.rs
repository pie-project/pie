//! Instance construction (thrust-3 P2.2) — the host-side `instantiate`
//! logic behind the WIT `pipeline.instantiate`.
//!
//! An **instance** is one binding of a registered program (the trace identity)
//! to its per-instance state: the seed values for its `seeded` channels
//! (`Channel::from` data — D2, never part of registration/identity) and the set
//! of host-facing channels the guest can drive. The device channel arena is
//! allocated driver-side from the declared shapes; the host instance carries the
//! validated seeds to send at instantiation plus the host-channel index map.
//!
//! Complete pipeline domain API: some methods here (relaxed geometry
//! variants, per-channel introspection, the pure `instantiate`/registry
//! probe entry points, device-geometry lease internals) are not yet
//! called by the current single-model/mock-driver fire path, but are
//! exercised by this module's own unit tests and reserved for upcoming
//! wiring (multi-pass channels, device-geometry beams) — kept rather
//! than deleted, allowed rather than silently masked.
#![allow(dead_code)]

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use pie_ptir::container::{self, HostRole};

use super::program::{RegisteredProgram, lookup};

/// Process-wide monotonic source of instance identities (0 reserved as a null /
/// "no instance" sentinel). Each `instantiate` mints a fresh id: the driver
/// caches one channel arena per id, so distinct instances of the same program
/// (same `hash`) hold independent, persistent channel state.
static NEXT_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

/// Mint the next process-wide instance identity.
pub fn next_instance_id() -> u64 {
    NEXT_INSTANCE_ID.fetch_add(1, Ordering::Relaxed)
}

/// A per-instance channel seed value, by dense channel index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChannelSeed {
    pub channel: u32,
    pub data: Vec<u8>,
}

/// A constructed instance: the registered program + its validated seeds.
#[derive(Debug)]
pub struct Instance {
    pub program: Arc<RegisteredProgram>,
    /// This instance's identity — the driver's channel-arena cache key (stable
    /// across all its fires). Seeds bind on its first fire; the arena persists.
    pub instance_id: u64,
    /// Validated seeds, one per `seeded` channel, in channel order.
    pub seeds: Vec<ChannelSeed>,
}

impl Instance {
    /// The dense indices of host-facing channels (`host-role != none`) — the
    /// only channels the guest may obtain a host endpoint on.
    pub fn host_channels(&self) -> Vec<u32> {
        self.program
            .bound
            .container
            .channels
            .iter()
            .enumerate()
            .filter(|(_, c)| c.host_role != HostRole::None)
            .map(|(i, _)| i as u32)
            .collect()
    }

    /// The host role of channel `index`, or `None` if the index is out of range.
    pub fn host_role(&self, index: u32) -> Option<HostRole> {
        self.program
            .bound
            .container
            .channels
            .get(index as usize)
            .map(|c| c.host_role)
    }

    /// Assemble the host-known per-channel values for a fire's geometry map:
    /// seeded channels carry their seed value; every other channel is
    /// host-unknown (device-derived, working-set, or not-yet-produced) → `None`,
    /// and the driver fills those descriptor ports itself. Sufficient to
    /// host-prefill a single-sequence decode whose token/geometry ports bind
    /// seeded channels (§3).
    pub fn channel_values(&self) -> Vec<Option<Vec<u8>>> {
        let mut v = vec![None; self.program.bound.container.channels.len()];
        for s in &self.seeds {
            v[s.channel as usize] = Some(s.data.clone());
        }
        v
    }

    /// Map this instance's descriptor ports → the fire's [`ReqGeometry`] from the
    /// host-known channel values (thrust-3 P2c-fire). Errors
    /// [`GeometryError::MissingChannelValue`] if a port binds a channel whose
    /// value isn't host-known — the signal that the geometry needs driver / ws /
    /// run-ahead resolution rather than a pure host prefill.
    pub fn fire_geometry(
        &self,
        page_size: u32,
    ) -> Result<
        crate::pipeline::fire::geometry::ReqGeometry,
        crate::pipeline::fire::geometry::GeometryError,
    > {
        crate::pipeline::fire::geometry::map_geometry(
            &self.program.bound.container,
            &self.channel_values(),
            page_size,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvPageSpan {
    pub start: u64,
    pub end: Option<u64>,
}

impl KvPageSpan {
    pub const fn open(start: u64) -> Self {
        Self { start, end: None }
    }

    pub fn resolve(self, page_len: u64) -> Result<std::ops::Range<u64>, String> {
        let end = self.end.unwrap_or(page_len);
        if self.start > end || end > page_len {
            return Err(format!(
                "KV page declaration {}..{} exceeds lease extent {page_len}",
                self.start, end
            ));
        }
        Ok(self.start..end)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvDeclaration {
    pub ws_rep: u32,
    pub readable: KvPageSpan,
    pub writable: KvPageSpan,
}

impl KvDeclaration {
    pub const fn all(ws_rep: u32) -> Self {
        Self {
            ws_rep,
            readable: KvPageSpan::open(0),
            writable: KvPageSpan::open(0),
        }
    }
}

#[derive(Default)]
pub struct ForwardBindings {
    pub embed: Option<EmbedBinding>,
    pub attention: Option<AttentionBinding>,
    pub readout: Option<u32>,
    pub rs_ws: Vec<u32>,
}

#[derive(Clone, Copy)]
pub struct EmbedBinding {
    pub tokens: u32,
    pub indptr: u32,
}

#[derive(Clone, Copy)]
pub struct AttentionBinding {
    pub kv_ws: u32,
    pub readable: KvPageSpan,
    pub writable: KvPageSpan,
    pub kv_len: u32,
    pub pages: u32,
    pub page_indptr: u32,
    pub w_slot: u32,
    pub w_off: u32,
    pub positions: u32,
    pub mask: Option<u32>,
}

/// The WIT forward-pass builder. It starts empty and acquires a native bound
/// pass only when canonical program bytes are attached.
pub struct ForwardPass {
    pub bindings: ForwardBindings,
    bound: Option<Box<BoundForwardPass>>,
}

impl ForwardPass {
    pub fn new() -> Self {
        Self {
            bindings: ForwardBindings::default(),
            bound: None,
        }
    }

    pub fn is_bound(&self) -> bool {
        self.bound.is_some()
    }

    pub fn attach_bound(&mut self, bound: BoundForwardPass) -> Result<(), String> {
        if self.bound.is_some() {
            return Err("forward pass program is already attached".to_string());
        }
        self.bound = Some(Box::new(bound));
        Ok(())
    }

    pub fn bound(&self) -> Result<&BoundForwardPass, String> {
        self.bound
            .as_deref()
            .ok_or_else(|| "forward pass program is not attached".to_string())
    }

    pub fn bound_mut(&mut self) -> Result<&mut BoundForwardPass, String> {
        self.bound
            .as_deref_mut()
            .ok_or_else(|| "forward pass program is not attached".to_string())
    }
}

impl Default for ForwardPass {
    fn default() -> Self {
        Self::new()
    }
}

impl std::ops::Deref for ForwardPass {
    type Target = BoundForwardPass;

    fn deref(&self) -> &Self::Target {
        self.bound
            .as_deref()
            .expect("forward-pass runtime use requires an attached program")
    }
}

impl std::ops::DerefMut for ForwardPass {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.bound
            .as_deref_mut()
            .expect("forward-pass runtime use requires an attached program")
    }
}

/// A traced forward pass bound to its first-class handles — one instance of a
/// hash-deduped registered program. The driver's persistent channel arena is keyed by this pass's
/// `instance_id`; a channel MAY bind to several passes (multi-pass channels,
/// W3.2) — the driver's global channel registry (W0.1) resolves one shared
/// device cell and the pipeline orders the fires (§3.4). Domain state (not
/// WIT glue), so it lives here rather than in `inferlet::host::forward`,
/// which only holds the `Host`/`HostForwardPass` impls that push/get/delete
/// it from the WASM component resource table.
pub struct BoundForwardPass {
    pub instance: Instance,
    pub bound_instance: crate::driver::BoundInstance,
    /// Bind-time scheduler address. Stable for the instance lifetime, so
    /// per-token submits never re-enter the global driver-handle registry.
    pub scheduler: crate::scheduler::worker::SchedulerHandle,
    /// The bound channel cells, dense declaration order. Writer puts are
    /// coalesced into each fire; Reader cells hold direct mirror bindings.
    pub cells: crate::pipeline::channel::BoundCells,
    /// Dense-channel-index → global-channel-id map (captured at bind from the
    /// bound cells). Rides every submission so the driver binds the trace's
    /// dense channel references to the global device channel registry.
    pub channel_ids: Vec<u64>,
    /// The bound channel resource reps (captured at `forward-pass.program`), so
    /// `submit` can point each channel's await queue at the feeding pipeline
    /// (W3.1: the pipeline owns the FIFO; a pass may bind to any pipeline).
    pub channel_reps: Vec<u32>,
    /// Pipeline FIFO this pass has submitted through. Stored on the pass so
    /// teardown remains safe even if guest channel handles were dropped first.
    pub fires: Option<crate::pipeline::fire::PendingFires>,
    /// The guest-owned KV working set bound into this pass (the model forward
    /// writes the embedded token's K/V here + self-attends over it). The guest
    /// keeps it alive for the pass's lifetime (the classic `forward-pass`
    /// borrow convention); the pass does NOT destroy it on drop.
    pub kv_ws: u32,
    pub kv_declaration: KvDeclaration,
    /// Guest-owned recurrent-state working sets (hybrid / linear-attention
    /// models — GDN, Mamba2), in resolved forward-request order. Empty for
    /// pure-attention models.
    pub rs_ws: Vec<u32>,
    /// Whether the currently bound writable declaration has performed its
    /// one-shot COW against the sharing shape visible at first submit.
    pub kv_declaration_realized: bool,
    /// Set when a fire of this pass failed: further submits error with the
    /// root cause (the KV cursor and device channel state are unspecified
    /// after a failed fire — the guest builds a fresh pass).
    pub failed: Option<String>,
    /// Device-geometry state (Track B): `Some` iff this pass's geometry
    /// (`pages`/`w_slot`/…) is DEVICE-produced — the program traces the wire-form
    /// geometry in-graph (`page_indptr = CumSum(np)`, packed live pages) and the
    /// driver resolves it pre-forward, so the host neither replays the epilogue
    /// arithmetic nor projects per-lane KV. The runtime only leases physical
    /// pages ([`crate::pipeline::fire::lease::PageLease`]) and delivers fresh
    /// grants on the program's fresh channel. Replaces the deleted host-replay
    /// beam branch.
    pub devgeo: Option<crate::pipeline::fire::lease::DevGeo>,
    /// Shape-derived decode layout whose values are resolved by the driver.
    pub decode_envelope: Option<crate::pipeline::fire::geometry::DecodeEnvelope>,
    /// The pass binds an `AttnMask` descriptor channel (dense device mask).
    /// Its fires are marked mask-carrying on the launch plan so the
    /// scheduler batches them SOLO (the composed multi-program batch does
    /// not merge dense device masks — v1 scope).
    pub dense_mask: bool,
    /// Host mirror of the instance's committed channel state (seeds, then
    /// per-fire stage folds): the value oracle for evaluated fire geometry.
    /// Advances at submit.
    pub host_shadow: crate::pipeline::fire::shadow::HostShadow,
    /// Idempotency guard for [`ForwardPass::close_native`]. Set the first
    /// time native cleanup runs — either the explicit WIT `drop` (after it
    /// drains the pass's pending FIFO) or, when a `ResourceTable`/
    /// `ProcessCtx` teardown drops this value directly and bypasses the WIT
    /// path, this type's own `Drop` fallback. Guards a second call from
    /// double-closing the driver instance or double-detaching a cell.
    pub(crate) closed: bool,
}

impl BoundForwardPass {
    /// Replace only the recurrent-state resource reps. The native instance and
    /// every other pass field remain untouched. Rebinding is legal only at an
    /// empty pipeline FIFO boundary so no in-flight fire can retain the old
    /// request-row mapping.
    pub fn replace_rs_working_sets(&mut self, reps: Vec<u32>) -> Result<(), String> {
        let pending = self
            .fires
            .as_ref()
            .map(|fifo| fifo.lock().unwrap().len())
            .unwrap_or(0);
        if pending != 0 {
            return Err(format!(
                "cannot replace rs-working-sets while {pending} operation(s) remain in the pass FIFO"
            ));
        }
        self.rs_ws = reps;
        Ok(())
    }

    pub fn replace_kv_declaration(
        &mut self,
        ws_rep: u32,
        readable: KvPageSpan,
        writable: KvPageSpan,
    ) -> Result<(), String> {
        let pending = self
            .fires
            .as_ref()
            .map(|fifo| fifo.lock().unwrap().len())
            .unwrap_or(0);
        if pending != 0 {
            return Err(format!(
                "cannot replace KV working-set declaration while {pending} operation(s) remain in the pass FIFO"
            ));
        }
        if self.devgeo.is_some() {
            return Err("a device-geometry pass cannot redeclare its KV working set".to_string());
        }
        if self.decode_envelope.is_some()
            && (ws_rep != self.kv_ws || readable.start != 0 || readable.end.is_some())
        {
            return Err(
                "a device-resolved pass cannot replace its KV working set or narrow readable pages"
                    .to_string(),
            );
        }
        self.kv_ws = ws_rep;
        self.kv_declaration = KvDeclaration {
            ws_rep,
            readable,
            writable,
        };
        self.kv_declaration_realized = false;
        Ok(())
    }

    /// Idempotent ordered native teardown: enqueues the bound driver's close,
    /// detaches this pass's `instance_id` from every bound `ChannelCell`, and
    /// reclaims any outstanding device-geometry page grants. Safe to call more
    /// than once — every step is gated by `closed`, so a repeat call (explicit
    /// `drop` followed by this type's `Drop`, or vice versa) is a pure no-op.
    /// Never panics and never awaits: scheduler-side failures are logged,
    /// since teardown has no result channel left to report to.
    ///
    /// Callers MUST first confirm [`Self::can_close_native_on_drop`] (or
    /// have otherwise already drained the pass's fires FIFO, as
    /// `HostForwardPass::drop` does) — this method does not itself check
    /// for in-flight fires, since the explicit path calls it only after
    /// draining and re-checking here would just be a second, redundant
    /// lock/scan.
    pub fn close_native(&mut self) {
        if std::mem::replace(&mut self.closed, true) {
            return;
        }
        crate::pipeline::offload::close_home_instance(self.bound_instance.instance_id);
        if let Err(error) = self.scheduler.close_instance(
            self.bound_instance.instance_id,
            self.bound_instance.pacing_wait_id,
        ) {
            tracing::warn!(
                instance_id = self.bound_instance.instance_id,
                %error,
                "forward-pass native cleanup: close_instance failed"
            );
        }
        for cell in &self.cells {
            cell.lock().unwrap().detach(self.bound_instance.instance_id);
        }
        // Device-geometry: leased slots are logical reserve indexes in the
        // store model. Unwritten grants hold no physical memory (reserve is
        // logical); written grants are committed pages that stay mapped
        // until the working set itself is discarded or dropped — discarding
        // them here would shift the surviving indexes under other passes
        // bound to the same working set, so cleanup intentionally leaves the
        // mapping alone and only clears the lease's own pending/free
        // bookkeeping.
        if let Some(devgeo) = self.devgeo.as_mut() {
            let _ = devgeo.lease.reclaim_all();
        }
    }

    /// Whether it is safe to run [`Self::close_native`] from the `Drop`
    /// fallback right now: the pass's shared fires FIFO (`None`, or `Some`
    /// but empty) must hold no in-flight fire/move. A non-empty FIFO means
    /// at least one op is still awaiting its driver completion and hasn't
    /// been async-finalized (`crate::pipeline::fire::finalize_op` commits or
    /// aborts its KV/RS txns and publishes its mirror epoch) — closing the
    /// native instance, detaching channel cells, or reclaiming
    /// device-geometry pages out from under that pending completion would
    /// race a still-live driver write against page reuse (a use-after-free).
    /// `Drop` cannot `.await` that drain, so it must check this predicate
    /// instead of finalizing; the explicit `HostForwardPass::drop` path is
    /// unaffected — it always awaits the drain first and so always finds
    /// this true by the time it calls `close_native`.
    pub(crate) fn can_close_native_on_drop(&self) -> bool {
        match &self.fires {
            None => true,
            Some(fifo) => fifo.lock().unwrap().is_empty(),
        }
    }
}

impl Drop for BoundForwardPass {
    /// Fallback for when a `ResourceTable`/`ProcessCtx` teardown drops this
    /// value directly, bypassing `HostForwardPass::drop`'s async FIFO drain +
    /// explicit `close_native` call (thrust-3 process-teardown hardening).
    /// Idempotent with the explicit path via `closed`.
    ///
    /// Refuses to run native teardown while [`Self::can_close_native_on_drop`]
    /// is false (a fire/move still in flight): `Drop` has no `.await` to
    /// drain the FIFO safely, so forcing `close_native` here would risk a
    /// use-after-free / premature device-geometry page reuse against that
    /// pending completion. In that case this logs a high-signal error and
    /// leaves the native instance, channel attachments, and page lease
    /// alone — a bounded leak (reclaimed at process exit) rather than
    /// silent memory corruption. This should not happen in practice: it
    /// means a `ForwardPass` with in-flight fires was dropped without going
    /// through `HostForwardPass::drop`, which is itself the bug to fix at
    /// the call site.
    fn drop(&mut self) {
        if self.closed {
            return;
        }
        if self.can_close_native_on_drop() {
            self.close_native();
        } else {
            tracing::error!(
                instance_id = self.bound_instance.instance_id,
                pending_fires = self
                    .fires
                    .as_ref()
                    .map(|fifo| fifo.lock().unwrap().len())
                    .unwrap_or(0),
                "forward-pass dropped with its fires FIFO non-empty, bypassing \
                 HostForwardPass::drop's async drain; skipping native teardown \
                 (close_instance / channel detach / device-geometry reclaim) to avoid \
                 racing a live driver completion into a use-after-free or premature \
                 page reuse — this leaks the driver instance and its channel \
                 attachments until process exit"
            );
        }
    }
}

/// An instantiation failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InstantiateError {
    /// No program with this identity hash is registered.
    UnknownProgram(u64),
    /// A seed targets a channel that was not declared `seeded`.
    SeedForNonSeeded { channel: u32 },
    /// A seed targets a channel index outside the container.
    SeedChannelOutOfRange { channel: u32 },
    /// Two seeds target the same channel.
    DuplicateSeed { channel: u32 },
    /// A declared `seeded` channel has no seed value.
    MissingSeed { channel: u32 },
    /// A seed's byte length does not match its channel's shape×dtype.
    SeedShapeMismatch {
        channel: u32,
        expected: usize,
        got: usize,
    },
}

impl fmt::Display for InstantiateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use InstantiateError::*;
        match self {
            UnknownProgram(h) => write!(f, "no registered program with identity {h:#018x}"),
            SeedForNonSeeded { channel } => {
                write!(
                    f,
                    "channel {channel}: a seed was supplied but the channel is not seeded"
                )
            }
            SeedChannelOutOfRange { channel } => write!(f, "seed channel {channel} out of range"),
            DuplicateSeed { channel } => write!(f, "channel {channel}: duplicate seed"),
            MissingSeed { channel } => write!(f, "channel {channel}: seeded but no seed supplied"),
            SeedShapeMismatch {
                channel,
                expected,
                got,
            } => write!(
                f,
                "channel {channel}: seed is {got} bytes, expected {expected} (shape×dtype)"
            ),
        }
    }
}
impl std::error::Error for InstantiateError {}

/// Instantiate a registered program with per-channel seeds (looks the program up
/// in the process-wide registry). Validates that exactly the `seeded` channels
/// are seeded, each with the right byte length. Seeds are per-instance data (D2)
/// — never part of the program identity.
///
/// The live WIT `forward-pass.program` bind path (`inferlet::host::forward`) does
/// its own inline validation (dense channel decls, staged puts, extern
/// bindings) alongside seed checking, so it does not call this pure
/// entry point; kept + unit-tested as the minimal instantiate contract.
pub fn instantiate(program: u64, seeds: Vec<ChannelSeed>) -> Result<Instance, InstantiateError> {
    let prog = lookup(program).ok_or(InstantiateError::UnknownProgram(program))?;
    let validated = validate_seeds(&prog, seeds)?;
    Ok(Instance {
        program: prog,
        instance_id: next_instance_id(),
        seeds: validated,
    })
}

/// Validate + order the seeds against a program's channel declarations.
pub fn validate_seeds(
    prog: &RegisteredProgram,
    seeds: Vec<ChannelSeed>,
) -> Result<Vec<ChannelSeed>, InstantiateError> {
    let channels = &prog.bound.container.channels;

    // Index the supplied seeds; reject out-of-range, non-seeded, and duplicates.
    let mut by_channel: Vec<Option<Vec<u8>>> = vec![None; channels.len()];
    for s in seeds {
        let idx = s.channel as usize;
        let decl = channels
            .get(idx)
            .ok_or(InstantiateError::SeedChannelOutOfRange { channel: s.channel })?;
        if !decl.seeded {
            return Err(InstantiateError::SeedForNonSeeded { channel: s.channel });
        }
        if by_channel[idx].is_some() {
            return Err(InstantiateError::DuplicateSeed { channel: s.channel });
        }
        let expected =
            decl.shape.numel() as usize * container::const_elem_size(decl.dtype.program_dtype());
        if s.data.len() != expected {
            return Err(InstantiateError::SeedShapeMismatch {
                channel: s.channel,
                expected,
                got: s.data.len(),
            });
        }
        by_channel[idx] = Some(s.data);
    }

    // Every seeded channel must have a seed; assemble in channel order.
    let mut out = Vec::new();
    for (i, decl) in channels.iter().enumerate() {
        match (decl.seeded, by_channel[i].take()) {
            (true, Some(data)) => out.push(ChannelSeed {
                channel: i as u32,
                data,
            }),
            (true, None) => return Err(InstantiateError::MissingSeed { channel: i as u32 }),
            (false, _) => {}
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::program::{Registry, register};
    use pie_ptir::container::{
        ChanDType, ChannelDecl, PortBinding, PortSource, StageProgram, TraceContainer,
    };
    use pie_ptir::op::{IntrinsicId, Op};
    use pie_ptir::registry::{ModelProfile, Port, Stage};
    use pie_ptir::types::{DType, Shape};
    use std::num::NonZeroUsize;

    const VOCAB: u32 = 32;

    fn chan(shape: Shape, dtype: DType, role: HostRole, seeded: bool) -> ChannelDecl {
        ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 1,
            host_role: role,
            seeded,
        }
    }

    fn u32_port(port: Port, shape: Shape, values: &[u32]) -> PortBinding {
        PortBinding {
            port,
            source: PortSource::Const {
                dtype: DType::U32,
                shape,
                data: values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            },
        }
    }

    /// tok (i32 [1], seeded, device) + out (i32 [1], host-reader).
    fn greedy() -> TraceContainer {
        let ops = vec![
            Op::IntrinsicVal {
                intr: IntrinsicId::Logits,
                shape: Shape::matrix(1, VOCAB),
                dtype: DType::F32,
            },
            Op::Reshape {
                value: 0,
                shape: Shape::vector(VOCAB),
            },
            Op::ReduceArgmax(1),
            Op::Reshape {
                value: 2,
                shape: Shape::vector(1),
            },
            Op::ChanPut { chan: 1, value: 3 },
        ];
        TraceContainer {
            names: vec![],
            externs: vec![],
            channels: vec![
                chan(Shape::vector(1), DType::I32, HostRole::None, true),
                chan(Shape::vector(1), DType::I32, HostRole::Reader, false),
            ],
            ports: vec![
                PortBinding {
                    port: Port::EmbedTokens,
                    source: PortSource::Channel(0),
                },
                PortBinding {
                    port: Port::EmbedIndptr,
                    source: PortSource::Const {
                        dtype: DType::U32,
                        shape: Shape::vector(2),
                        data: [0u32, 1].iter().flat_map(|w| w.to_le_bytes()).collect(),
                    },
                },
                u32_port(Port::Positions, Shape::vector(1), &[0]),
                u32_port(Port::Pages, Shape::vector(1), &[0]),
                u32_port(Port::PageIndptr, Shape::vector(2), &[0, 1]),
                PortBinding {
                    port: Port::KvLen,
                    source: PortSource::Const {
                        dtype: DType::U32,
                        shape: Shape::vector(1),
                        data: 1u32.to_le_bytes().to_vec(),
                    },
                },
                u32_port(Port::WSlot, Shape::vector(1), &[0]),
                u32_port(Port::WOff, Shape::vector(1), &[0]),
            ],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops,
            }],
        }
    }

    fn registered() -> Arc<RegisteredProgram> {
        let mut r = Registry::new(NonZeroUsize::new(8).unwrap());
        r.register(
            greedy().encode(),
            &ModelProfile {
                vocab: VOCAB,
                ..ModelProfile::dummy()
            },
        )
        .unwrap()
    }

    fn seed(ch: u32, val: i32) -> ChannelSeed {
        ChannelSeed {
            channel: ch,
            data: val.to_le_bytes().to_vec(),
        }
    }

    #[test]
    fn valid_seeds_construct_instance() {
        let prog = registered();
        let inst = Instance {
            program: prog.clone(),
            instance_id: 1,
            seeds: validate_seeds(&prog, vec![seed(0, 1)]).unwrap(),
        };
        assert_eq!(inst.seeds, vec![seed(0, 1)]);
        assert_eq!(inst.host_channels(), vec![1], "only `out` is host-facing");
        assert_eq!(inst.host_role(0), Some(HostRole::None));
        assert_eq!(inst.host_role(1), Some(HostRole::Reader));
    }

    #[test]
    fn missing_seed_for_seeded_channel_fails() {
        let prog = registered();
        let e = validate_seeds(&prog, vec![]).unwrap_err();
        assert_eq!(e, InstantiateError::MissingSeed { channel: 0 });
    }

    #[test]
    fn seed_for_non_seeded_channel_fails() {
        let prog = registered();
        let e = validate_seeds(&prog, vec![seed(0, 1), seed(1, 9)]).unwrap_err();
        assert_eq!(e, InstantiateError::SeedForNonSeeded { channel: 1 });
    }

    #[test]
    fn wrong_seed_length_fails() {
        let prog = registered();
        let bad = ChannelSeed {
            channel: 0,
            data: vec![1, 2],
        }; // 2 bytes, need 4
        let e = validate_seeds(&prog, vec![bad]).unwrap_err();
        assert_eq!(
            e,
            InstantiateError::SeedShapeMismatch {
                channel: 0,
                expected: 4,
                got: 2
            }
        );
    }

    #[test]
    fn duplicate_and_out_of_range_seeds_fail() {
        let prog = registered();
        assert_eq!(
            validate_seeds(&prog, vec![seed(0, 1), seed(0, 2)]).unwrap_err(),
            InstantiateError::DuplicateSeed { channel: 0 }
        );
        assert_eq!(
            validate_seeds(&prog, vec![seed(0, 1), seed(9, 2)]).unwrap_err(),
            InstantiateError::SeedChannelOutOfRange { channel: 9 }
        );
    }

    #[test]
    fn instantiate_unknown_program_fails() {
        let e = instantiate(0xdead_beef, vec![]).unwrap_err();
        assert_eq!(e, InstantiateError::UnknownProgram(0xdead_beef));
    }

    #[test]
    fn instantiate_via_process_registry_roundtrips() {
        let bytes = greedy().encode();
        let prog = register(
            bytes.clone(),
            &ModelProfile {
                vocab: VOCAB,
                ..ModelProfile::dummy()
            },
        )
        .unwrap();
        let inst = instantiate(prog.hash, vec![seed(0, 42)]).unwrap();
        assert_eq!(inst.program.hash, prog.hash);
        assert_eq!(inst.seeds, vec![seed(0, 42)]);
    }

    #[test]
    fn fire_geometry_prefills_request_from_seed() {
        // §3-shaped greedy: embed_tokens ← the seeded `tok` channel, embed_indptr
        // const [0,1]. The host-known geometry (token + qo_indptr + default
        // positions/readout) prefills a LaunchPlan — no driver/ws needed.
        let prog = register(
            greedy().encode(),
            &ModelProfile {
                vocab: VOCAB,
                ..ModelProfile::dummy()
            },
        )
        .unwrap();
        let inst = instantiate(prog.hash, vec![seed(0, 42)]).unwrap();

        let g = inst.fire_geometry(16).unwrap();
        assert_eq!(g.token_ids, vec![42], "the seeded token embeds");
        assert_eq!(g.qo_indptr, vec![0, 1], "one lane, one token");
        assert_eq!(g.position_ids, vec![0]);
        assert_eq!(g.sampling_indices, vec![0], "read out the only token");

        let mut req = crate::driver::LaunchPlan::default();
        g.apply_to(&mut req);
        assert_eq!(req.token_ids, vec![42]);
        assert_eq!(req.qo_indptr, vec![0, 1]);
        assert_eq!(req.sampling_indices, vec![0]);
    }

    /// P2/P3 exit gate: register → instantiate → run on echo's reference
    /// interpreter (the mock driver) → golden token. Proves the runtime
    /// register/instantiate/submit-round-trip path end to end against the same
    /// oracle the CUDA backend diffs against — the greedy program argmaxes the
    /// supplied logits and publishes the token to the host-reader `out` channel.
    #[test]
    fn instantiate_mints_distinct_persistent_identities() {
        let prog = register(
            greedy().encode(),
            &ModelProfile {
                vocab: VOCAB,
                ..ModelProfile::dummy()
            },
        )
        .unwrap();
        let inst = instantiate(prog.hash, vec![seed(0, 42)]).unwrap();
        assert_ne!(inst.instance_id, 0, "a fresh instance identity is minted");
        assert_eq!(inst.seeds.len(), 1);
        assert_eq!(inst.seeds[0].channel, 0);
        assert_eq!(inst.seeds[0].data, 42i32.to_le_bytes().to_vec());

        // Two instances of the SAME program get DISTINCT identities (independent
        // channel arenas) though they share one compiled `hash`.
        let inst2 = instantiate(prog.hash, vec![seed(0, 7)]).unwrap();
        assert_eq!(
            inst2.program.hash, inst.program.hash,
            "same compiled program"
        );
        assert_ne!(
            inst2.instance_id, inst.instance_id,
            "distinct persistent instances"
        );
    }

    #[test]
    fn register_instantiate_run_on_mock_interp() {
        use pie_ptir::interp::Value;
        use pie_ptir::interp::{Instance as Interp, NoKernels, PassInputs};

        let prog = register(
            greedy().encode(),
            &ModelProfile {
                vocab: VOCAB,
                ..ModelProfile::dummy()
            },
        )
        .unwrap();
        // Instance construction (seed validation) — the WIT `instantiate` core.
        let inst = instantiate(prog.hash, vec![seed(0, 1)]).unwrap();
        assert_eq!(inst.host_channels(), vec![1], "only `out` is host-facing");

        // Run the bound trace on the mock driver (echo's interp) with the same
        // per-instance seeds the runtime would ship.
        let mut mock = Interp::new(&prog.bound, &[(0, Value::I32(vec![1]))]).unwrap();
        let out_i = prog
            .bound
            .container
            .channels
            .iter()
            .position(|c| c.host_role == HostRole::Reader)
            .unwrap() as u32;

        let mut l = vec![0.0f32; VOCAB as usize];
        l[5] = 9.0; // argmax at index 5
        let r = mock
            .step(
                &prog.bound,
                &PassInputs {
                    logits: Some(Value::F32(l)),
                    ..Default::default()
                },
                &mut NoKernels,
            )
            .unwrap();
        assert!(r.committed, "the pass commits");
        assert_eq!(
            mock.host_take(&prog.bound, out_i).unwrap(),
            Value::I32(vec![5]),
            "greedy token = argmax = 5"
        );
    }

    /// `ForwardPass::close_native` process-teardown fallback coverage
    /// (thrust-3): a `ForwardPass` value constructed exactly like
    /// `HostForwardPass::new` binds one, then DROPPED DIRECTLY — never
    /// routed through `HostForwardPass::drop`'s FIFO drain + explicit
    /// `close_native` call — must still close the bound driver instance and
    /// detach every one of its cell attachments. This is the host-level
    /// analog of the CUDA-contention "solo lane exhausts an 8-page pool at
    /// lane 8" repro: every dropped pass that skips native cleanup leaves a
    /// `close_channel: channel still has instance attachments` liability for
    /// the next lane.
    mod native_cleanup {
        use super::*;
        use crate::driver::{
            self, BoundInstance, ChannelRegistrationPlan, ChannelValue, DriverSpec,
            ProgramRegistration, SchedulerLimits,
        };
        use crate::pipeline::channel::ChannelCell;
        use crate::scheduler::worker::BatchScheduler;
        use pie_driver_dummy_lib::DummyDriverOptions;
        use std::sync::Mutex;

        /// A trivial driver-registerable program (no `Logits`/vocab
        /// dependency — this harness only exercises bind/close, never a
        /// fire): one seeded `HostRole::None` channel, one `Reader` channel.
        fn dummy_program() -> ProgramRegistration {
            let bytes = TraceContainer {
                names: vec![],
                externs: vec![],
                channels: vec![
                    chan(Shape::vector(1), DType::U32, HostRole::None, true),
                    chan(Shape::vector(1), DType::U32, HostRole::Reader, false),
                ],
                ports: vec![],
                stages: vec![StageProgram {
                    stage: Stage::Epilogue,
                    ops: vec![
                        Op::ChanTake(0),
                        Op::Const(pie_ptir::types::Literal::U32(1)),
                        Op::Add(0, 1),
                        Op::ChanPut { chan: 0, value: 2 },
                        Op::ChanPut { chan: 1, value: 2 },
                    ],
                }],
            }
            .encode();
            ProgramRegistration {
                program_hash: pie_ptir::container_hash(&bytes),
                canonical_bytes: bytes,
                sidecar_bytes: Vec::new(),
            }
        }

        /// Real scheduler + dummy driver, one bound instance, two bound
        /// `ChannelCell`s (dense order matching `dummy_program`) — the exact
        /// shape `HostForwardPass::new` binds, built directly so the test
        /// can drop the resulting `ForwardPass` WITHOUT calling
        /// `HostForwardPass::drop`.
        async fn setup(
            operation_log: Arc<Mutex<Vec<String>>>,
        ) -> (
            BatchScheduler,
            BoundInstance,
            Vec<Arc<Mutex<ChannelCell>>>,
            u64,
        ) {
            let driver_id = driver::register_driver_backend(
                DriverSpec {
                    num_kv_pages: 16,
                    limits: SchedulerLimits {
                        max_forward_requests: 1,
                        max_forward_tokens: 64,
                        max_page_refs: 64,
                    },
                    device_geometry_port_mask: 0,
                },
                driver::DriverBackend::Dummy(driver::DummyDriver::new(DummyDriverOptions {
                    operation_log: Some(operation_log),
                    ..DummyDriverOptions::default()
                })),
            );
            let scheduler = BatchScheduler::new(
                driver_id,
                driver_id,
                16,
                SchedulerLimits {
                    max_forward_requests: 1,
                    max_forward_tokens: 64,
                    max_page_refs: 64,
                },
                1,
                1,
            );

            let program_id = crate::scheduler::register_program(driver_id, dummy_program())
                .await
                .unwrap();

            let decls = [
                chan(Shape::vector(1), DType::U32, HostRole::None, true),
                chan(Shape::vector(1), DType::U32, HostRole::Reader, false),
            ];
            let cells: Vec<Arc<Mutex<ChannelCell>>> = decls
                .iter()
                .map(|decl| {
                    Arc::new(Mutex::new(ChannelCell::new(
                        decl.shape.dims().to_vec(),
                        decl.dtype.program_dtype(),
                        decl.capacity,
                    )))
                })
                .collect();
            let channel_ids: Vec<u64> = cells.iter().map(|c| c.lock().unwrap().global_id).collect();

            let instance_id = next_instance_id();
            for (cell, decl) in cells.iter().zip(&decls) {
                cell.lock()
                    .unwrap()
                    .attach(instance_id, decl, None)
                    .unwrap();
            }
            for (cell, decl) in cells.iter().zip(&decls) {
                let global_id = cell.lock().unwrap().global_id;
                let endpoint = crate::scheduler::register_channel(
                    driver_id,
                    ChannelRegistrationPlan {
                        driver_id,
                        channel_id: global_id,
                        shape: decl.shape.dims().to_vec(),
                        dtype: decl.dtype.tag(),
                        host_role: decl.host_role as u8,
                        seeded: decl.seeded,
                        extern_dir: pie_driver_abi::PIE_CHANNEL_EXTERN_NONE,
                        capacity: decl.capacity,
                        reader_wait_id: 0,
                        writer_wait_id: 0,
                        extern_name: Vec::new(),
                    },
                )
                .await
                .unwrap();
                cell.lock().unwrap().attach_endpoint(endpoint).unwrap();
            }

            let seed_values = vec![ChannelValue {
                channel: channel_ids[0],
                bytes: 1u32.to_le_bytes().to_vec(),
            }];
            let bound = crate::scheduler::bind_instance(
                driver_id,
                None,
                program_id,
                instance_id,
                channel_ids,
                seed_values,
            )
            .await
            .unwrap();

            (scheduler, bound, cells, instance_id)
        }

        async fn wait_for_operation(operation_log: &Arc<Mutex<Vec<String>>>, operation: &str) {
            tokio::time::timeout(std::time::Duration::from_secs(5), async {
                loop {
                    if operation_log
                        .lock()
                        .unwrap()
                        .iter()
                        .any(|entry| entry == operation)
                    {
                        return;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("scheduler operation must complete");
        }

        /// Assemble a `ForwardPass` exactly like `HostForwardPass::new`
        /// would (minus channel-rep/devgeo/KV-ws plumbing this harness has
        /// no use for) around an already-bound instance + cells.
        fn make_pass(
            bound_instance: BoundInstance,
            cells: Vec<Arc<Mutex<ChannelCell>>>,
        ) -> ForwardPass {
            let program = registered();
            let host_shadow = crate::pipeline::fire::shadow::HostShadow::new(&program.bound, &[]);
            let bound = BoundForwardPass {
                instance: Instance {
                    program,
                    instance_id: bound_instance.instance_id,
                    seeds: Vec::new(),
                },
                host_shadow,
                scheduler: crate::scheduler::scheduler_handle(bound_instance.driver_id).unwrap(),
                bound_instance,
                cells,
                channel_ids: Vec::new(),
                channel_reps: Vec::new(),
                fires: None,
                kv_ws: 0,
                kv_declaration: KvDeclaration::all(0),
                rs_ws: Vec::new(),
                kv_declaration_realized: false,
                failed: None,
                devgeo: None,
                decode_envelope: None,
                dense_mask: false,
                closed: false,
            };
            let mut pass = ForwardPass::new();
            pass.attach_bound(bound).unwrap();
            pass
        }

        #[tokio::test(flavor = "current_thread")]
        async fn rs_rebind_is_in_place_and_preserves_pass_state() {
            let operation_log = Arc::new(Mutex::new(Vec::new()));
            let (_scheduler, bound, cells, _) = setup(operation_log).await;
            let mut pass = make_pass(bound, cells);
            pass.kv_declaration_realized = true;
            pass.devgeo = Some(crate::pipeline::fire::lease::DevGeo {
                lease: crate::pipeline::fire::lease::PageLease::new(2),
                b: 2,
                fresh_dense: 0,
                w_cont_dense: 1,
                has_mask: true,
            });
            let instance_id = pass.bound_instance.instance_id;

            pass.replace_rs_working_sets(vec![101, 202]).unwrap();

            assert_eq!(pass.rs_ws, vec![101, 202]);
            assert_eq!(pass.bound_instance.instance_id, instance_id);
            assert!(pass.kv_declaration_realized);
            let devgeo = pass.devgeo.as_ref().expect("device geometry preserved");
            assert_eq!(devgeo.b, 2);
            assert!(devgeo.has_mask);
        }

        #[tokio::test(flavor = "current_thread")]
        async fn rs_rebind_rejects_nonempty_fifo_without_mutation() {
            let operation_log = Arc::new(Mutex::new(Vec::new()));
            let (_scheduler, bound, cells, _) = setup(operation_log).await;
            let mut pass = make_pass(bound, cells);
            pass.rs_ws = vec![7];
            let fifo = Arc::new(crate::pipeline::fire::PendingFireQueue::from_queue(
                std::collections::VecDeque::from([crate::pipeline::fire::test_pending_op_stub()]),
            ));
            pass.fires = Some(fifo.clone());

            assert!(pass.replace_rs_working_sets(vec![8, 9]).is_err());
            assert_eq!(pass.rs_ws, vec![7]);

            fifo.lock().unwrap().clear();
        }

        #[tokio::test(flavor = "current_thread")]
        async fn kv_rebind_is_in_place_and_resets_declaration_realization() {
            let operation_log = Arc::new(Mutex::new(Vec::new()));
            let (_scheduler, bound, cells, _) = setup(operation_log).await;
            let mut pass = make_pass(bound, cells);
            pass.kv_declaration_realized = true;
            let instance_id = pass.bound_instance.instance_id;

            pass.replace_kv_declaration(9, KvPageSpan::open(0), KvPageSpan::open(2))
                .unwrap();

            assert_eq!(pass.kv_ws, 9);
            assert_eq!(pass.kv_declaration.ws_rep, 9);
            assert_eq!(pass.kv_declaration.writable.start, 2);
            assert!(!pass.kv_declaration_realized);
            assert_eq!(pass.bound_instance.instance_id, instance_id);
        }

        #[tokio::test(flavor = "current_thread")]
        async fn kv_rebind_rejects_nonempty_fifo_without_mutation() {
            let operation_log = Arc::new(Mutex::new(Vec::new()));
            let (_scheduler, bound, cells, _) = setup(operation_log).await;
            let mut pass = make_pass(bound, cells);
            pass.kv_ws = 7;
            let fifo = Arc::new(crate::pipeline::fire::PendingFireQueue::from_queue(
                std::collections::VecDeque::from([crate::pipeline::fire::test_pending_op_stub()]),
            ));
            pass.fires = Some(fifo.clone());

            assert!(
                pass.replace_kv_declaration(8, KvPageSpan::open(0), KvPageSpan::open(1))
                    .is_err()
            );
            assert_eq!(pass.kv_ws, 7);

            fifo.lock().unwrap().clear();
        }

        #[tokio::test(flavor = "current_thread")]
        async fn device_geometry_rejects_kv_recall() {
            let operation_log = Arc::new(Mutex::new(Vec::new()));
            let (_scheduler, bound, cells, _) = setup(operation_log).await;
            let mut pass = make_pass(bound, cells);
            pass.kv_ws = 7;
            pass.devgeo = Some(crate::pipeline::fire::lease::DevGeo {
                lease: crate::pipeline::fire::lease::PageLease::new(2),
                b: 2,
                fresh_dense: 0,
                w_cont_dense: 1,
                has_mask: false,
            });

            assert!(
                pass.replace_kv_declaration(7, KvPageSpan::open(0), KvPageSpan::open(0))
                    .is_err()
            );
            assert_eq!(pass.kv_ws, 7);
        }

        #[tokio::test(flavor = "current_thread")]
        async fn device_resolved_recall_keeps_ws_and_full_readable_span() {
            let operation_log = Arc::new(Mutex::new(Vec::new()));
            let (_scheduler, bound, cells, _) = setup(operation_log).await;
            let mut pass = make_pass(bound, cells);
            pass.kv_ws = 7;
            pass.decode_envelope = Some(crate::pipeline::fire::geometry::DecodeEnvelope {
                token_count: 1,
                lane_count: 1,
                token_indptr: vec![0, 1],
                loop_carried: true,
                device_positions: true,
                has_mask: false,
            });

            assert!(
                pass.replace_kv_declaration(8, KvPageSpan::open(0), KvPageSpan::open(0))
                    .is_err()
            );
            assert!(
                pass.replace_kv_declaration(7, KvPageSpan::open(1), KvPageSpan::open(0))
                    .is_err()
            );
            pass.replace_kv_declaration(7, KvPageSpan::open(0), KvPageSpan::open(2))
                .unwrap();
            assert_eq!(pass.kv_declaration.writable.start, 2);
            assert!(!pass.kv_declaration_realized);
        }

        #[tokio::test(flavor = "current_thread")]
        async fn drop_without_host_drop_closes_instance_and_detaches_channels() {
            let operation_log = Arc::new(Mutex::new(Vec::new()));
            let (_scheduler, bound, cells, instance_id) = setup(operation_log.clone()).await;
            let pass = make_pass(bound, cells.clone());

            // Simulate a `ResourceTable`/`ProcessCtx` teardown dropping the
            // `ForwardPass` value directly — `HostForwardPass::drop` (which
            // drains the pending FIFO, then calls `close_native`) is never
            // invoked.
            drop(pass);
            wait_for_operation(&operation_log, "close_instance").await;

            assert!(
                operation_log
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|op| op == "close_instance"),
                "the Drop fallback closed the native instance"
            );

            // Every cell's attachment for `instance_id` must be gone: a
            // FRESH instance can attach to the SAME cells without hitting
            // channel.rs's "a private or host-visible channel may attach to
            // only one pass" rule — which is exactly the driver-level
            // symptom (`close_channel: channel still has instance
            // attachments`) this fallback prevents.
            let decls = [
                chan(Shape::vector(1), DType::U32, HostRole::None, true),
                chan(Shape::vector(1), DType::U32, HostRole::Reader, false),
            ];
            let other_instance = next_instance_id();
            assert_ne!(other_instance, instance_id);
            for (cell, decl) in cells.iter().zip(&decls) {
                cell.lock()
                    .unwrap()
                    .attach(other_instance, decl, None)
                    .expect(
                        "the Drop fallback detached the old instance; a fresh \
                         attach must succeed",
                    );
            }
        }

        #[tokio::test(flavor = "current_thread")]
        async fn explicit_close_native_is_idempotent_with_the_drop_fallback() {
            let operation_log = Arc::new(Mutex::new(Vec::new()));
            let (_scheduler, bound, cells, _instance_id) = setup(operation_log.clone()).await;
            let mut pass = make_pass(bound, cells.clone());

            // The explicit path (what `HostForwardPass::drop` does after
            // draining the FIFO).
            pass.close_native();
            wait_for_operation(&operation_log, "close_instance").await;
            let closes_after_explicit = operation_log
                .lock()
                .unwrap()
                .iter()
                .filter(|op| *op == "close_instance")
                .count();
            assert_eq!(closes_after_explicit, 1);

            // The value's own `Drop` fires next when `pass` goes out of
            // scope — must be a pure no-op (`closed` already set): no
            // second `close_instance` dispatch, which would otherwise
            // error against an already-removed native instance.
            drop(pass);
            let closes_after_drop = operation_log
                .lock()
                .unwrap()
                .iter()
                .filter(|op| *op == "close_instance")
                .count();
            assert_eq!(
                closes_after_drop, 1,
                "close_native is idempotent: the Drop fallback must not double-close"
            );
        }

        #[tokio::test(flavor = "current_thread")]
        async fn can_close_native_on_drop_is_true_with_no_or_empty_fifo() {
            let operation_log = Arc::new(Mutex::new(Vec::new()));
            let (_scheduler, bound, cells, _instance_id) = setup(operation_log).await;
            let mut pass = make_pass(bound, cells);

            assert!(pass.fires.is_none());
            assert!(
                pass.can_close_native_on_drop(),
                "no shared FIFO at all is trivially safe"
            );

            pass.fires = Some(Arc::new(crate::pipeline::fire::PendingFireQueue::new()));
            assert!(
                pass.can_close_native_on_drop(),
                "a FIFO that exists but is drained (empty) is just as safe as None"
            );
        }

        #[tokio::test(flavor = "current_thread")]
        async fn can_close_native_on_drop_is_false_with_a_pending_fifo_entry() {
            let operation_log = Arc::new(Mutex::new(Vec::new()));
            let (_scheduler, bound, cells, _instance_id) = setup(operation_log).await;
            let mut pass = make_pass(bound, cells);

            let mut fifo = std::collections::VecDeque::new();
            fifo.push_back(crate::pipeline::fire::test_pending_op_stub());
            pass.fires = Some(Arc::new(
                crate::pipeline::fire::PendingFireQueue::from_queue(fifo),
            ));

            assert!(
                !pass.can_close_native_on_drop(),
                "an in-flight fire/move must refuse native cleanup on drop"
            );
        }

        /// The actual review finding this guards: a `ForwardPass` dropped
        /// directly (bypassing `HostForwardPass::drop`'s FIFO drain) while a
        /// fire/move is still in flight must NOT close the native instance,
        /// detach cell attachments, or reclaim device-geometry pages — `Drop`
        /// cannot `.await` the pending completion's async finalize, so doing
        /// so would race a live driver write into a use-after-free /
        /// premature page reuse. It must instead leave everything alone
        /// (log-and-skip), unlike the empty-FIFO fallback case which DOES
        /// run `close_native` (see `drop_without_host_drop_closes_instance_and_detaches_channels`).
        #[tokio::test(flavor = "current_thread")]
        async fn drop_with_pending_fifo_entry_skips_native_teardown() {
            let operation_log = Arc::new(Mutex::new(Vec::new()));
            let (_scheduler, bound, cells, instance_id) = setup(operation_log.clone()).await;
            let mut pass = make_pass(bound, cells.clone());

            let mut fifo = std::collections::VecDeque::new();
            fifo.push_back(crate::pipeline::fire::test_pending_op_stub());
            pass.fires = Some(Arc::new(
                crate::pipeline::fire::PendingFireQueue::from_queue(fifo),
            ));

            // Simulate a `ResourceTable`/`ProcessCtx` teardown dropping the
            // `ForwardPass` value directly while its FIFO is non-empty.
            drop(pass);

            assert!(
                !operation_log
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|op| op == "close_instance"),
                "the Drop fallback must not close the native instance while a fire/move \
                 is still in flight"
            );

            // The old instance's attachment must still be present: a FRESH
            // instance attaching to the same cells must be REJECTED by
            // channel.rs's "a private or host-visible channel may attach to
            // only one pass" rule — proof detach did not run.
            let decls = [
                chan(Shape::vector(1), DType::U32, HostRole::None, true),
                chan(Shape::vector(1), DType::U32, HostRole::Reader, false),
            ];
            let other_instance = next_instance_id();
            assert_ne!(other_instance, instance_id);
            for (cell, decl) in cells.iter().zip(&decls) {
                let error = cell
                    .lock()
                    .unwrap()
                    .attach(other_instance, decl, None)
                    .expect_err(
                        "the old instance's attachment must remain: skipping teardown \
                         also skips detach",
                    );
                assert!(
                    error.contains("only one pass"),
                    "unexpected attach error: {error}"
                );
            }
        }
    }
}
