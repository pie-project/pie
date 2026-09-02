//! Instance construction — host-side `instantiate` logic behind WIT
//! `pipeline.instantiate`. An instance binds a registered program to its
//! per-instance state: seed values for `seeded` channels, plus the
//! host-facing channel set.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};


use super::program::RegisteredProgram;

/// Process-wide monotonic instance-id source (0 reserved as null). Each
/// `instantiate` mints a fresh id; the engine caches one channel arena per id.
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
    /// This instance's identity — the engine's channel-arena cache key,
    /// stable across its fires.
    pub instance_id: u64,
    /// Validated seeds, one per `seeded` channel, in channel order.
    pub seeds: Vec<ChannelSeed>,
}

impl Instance {

    /// Assemble host-known per-channel values for a fire's geometry map:
    /// seeded channels carry their seed; everything else is host-unknown
    /// (`None`), left for the engine to resolve.
    pub fn channel_values(&self) -> Vec<Option<Vec<u8>>> {
        let mut v = vec![None; self.program.bound.container.channels.len()];
        for s in &self.seeds {
            v[s.channel as usize] = Some(s.data.clone());
        }
        v
    }

    /// Map this instance's descriptor ports → the fire's [`ReqGeometry`] from
    /// host-known channel values. Errs with
    /// [`GeometryError::MissingChannelValue`] when a port binds a channel
    /// whose value isn't host-known (needs engine/ws/run-ahead resolution).
    pub fn fire_geometry(
        &self,
    ) -> Result<
        crate::pipeline::fire::geometry::ReqGeometry,
        crate::pipeline::fire::geometry::GeometryError,
    > {
        crate::pipeline::fire::geometry::map_geometry(
            &self.program.bound.container,
            &self.channel_values(),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvPageSpan {
    pub start: u64,
    pub end: Option<u64>,
}

impl KvPageSpan {
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

/// The pass's declared KV window; the owning working set is
/// [`BoundForwardPass::kv_ws`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvDeclaration {
    pub readable: KvPageSpan,
    pub writable: KvPageSpan,
}

/// Which WIT forward interface a pass was built through. One Rust
/// `ForwardPass` backs all three interfaces; this field makes a
/// mis-selected interface fail loudly rather than silently run wrong logic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassKind {
    Attention,
    Recurrent,
    Hybrid,
}

impl PassKind {
    pub fn name(self) -> &'static str {
        match self {
            PassKind::Attention => "attention",
            PassKind::Recurrent => "recurrent",
            PassKind::Hybrid => "hybrid",
        }
    }

    /// The interface a guest must use for this kind.
    pub fn interface(self) -> &'static str {
        match self {
            PassKind::Attention => "pie:inferlet/forward",
            PassKind::Recurrent => "pie:inferlet/forward-recurrent",
            PassKind::Hybrid => "pie:inferlet/forward-hybrid",
        }
    }
}

#[derive(Default)]
pub struct ForwardBindings {
    pub embed: Option<EmbedBinding>,
    pub attention: Option<AttentionBinding>,
    pub readout: Option<u32>,
    pub rs_ws: Vec<u32>,
    pub rs_geom: Option<RsGeometryBinding>,
    /// Host-known `rs-geometry.fold-len`, one per bound working set. `None`
    /// when the channel had no seed — the fold length is device-computed
    /// and the host plans against an upper bound.
    pub rs_fold_len: Option<Vec<u32>>,
    /// Run layers [0, k) and take the head at k for this pass's fires;
    /// `None` = full model.
    pub max_layers: Option<u32>,
    /// The spans this pass's tokens carry, in the order `forward-pass.media`
    /// attached them. Empty on every text-only pass. `Arc` so a decoded
    /// image submitted to two passes is decoded once.
    pub media: Vec<std::sync::Arc<models::media::EncodedSpan>>,
}

/// Where a fire's folded boundary lands — host mirror of WIT
/// `rs-geometry`. Absent only when no recurrent state is bound. Bound with
/// the working sets, since the boundary is an input to the recurrence on
/// every fire. Registry tags 10-14 stay reserved for compiled containers.
#[derive(Clone, Copy, Debug)]
pub struct RsGeometryBinding {
    /// How far the folded boundary advances, per request, over `[buffer |
    /// this fire's tokens]` — the twin of `kv-len`. Host-known value lives
    /// in [`ForwardBindings::rs_fold_len`] (needs per-row token counts).
    pub fold_len: u32,
    /// Capacity grant: how many buffer pages this fire may occupy — the
    /// only buffer decision left to the guest; everything else is derived
    /// from the store's own occupancy.
    #[allow(
        dead_code,
        reason = "written at bind time from the guest's `rs-geometry` record \
                  (`host/forward.rs`, where `page_span` still rejects a malformed \
                  span) and read by nothing. That is not a missing check. The WIT \
                  doc justifies the field by saying a fire needing an ungranted \
                  page must fail rather than have the runtime quietly find one -- \
                  and the runtime has no way to find one: `RsStore::alloc_buffer` \
                  is reachable ONLY from the guest's own `alloc-buffer` call, so \
                  nothing grows a buffer mid-fire. What could actually overrun is \
                  a fold, and `validate_fold` refuses that against both physical \
                  capacity and live occupancy on every `prepare_*` path. So the \
                  grant is the SEVENTH buffer-addressing channel, not a ceiling: \
                  the six above it were deleted for exactly this reason -- the \
                  runtime derives them from the store it is already authoritative \
                  for. Removing it is a WIT change and wants the deliberation the \
                  other six got, so it is described here rather than done here"
    )]
    pub buffer: KvPageSpan,
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
    /// The interface this pass was constructed through.
    pub kind: PassKind,
    pub bindings: ForwardBindings,
    bound: Option<Box<BoundForwardPass>>,
}

impl ForwardPass {
    pub fn new(kind: PassKind) -> Self {
        Self {
            kind,
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

/// A traced forward pass bound to its first-class handles — one instance
/// of a hash-deduped registered program. The engine's persistent channel
/// arena is keyed by this pass's `instance_id`; a channel may bind to
/// several passes, and the global channel registry resolves one shared
/// device cell while the pipeline orders the fires.
pub struct BoundForwardPass {
    pub instance: Instance,
    pub bound_instance: crate::engine::BoundInstance,
    /// Bind-time scheduler address, stable for the instance lifetime.
    pub scheduler: crate::scheduler::worker::SchedulerHandle,
    /// Bound channel cells, dense declaration order; writer puts coalesce
    /// into each fire, reader cells hold direct mirrors.
    pub cells: crate::pipeline::channel::BoundCells,
    /// Bound channel resource reps, so `submit` can point each channel's
    /// await queue at the feeding pipeline.
    pub channel_reps: Vec<u32>,
    /// Pipeline FIFO this pass has submitted through; kept on the pass so
    /// teardown is safe even if guest channel handles dropped first.
    pub fires: Option<crate::pipeline::fire::PendingFires>,
    /// Guest-owned KV working set bound into this pass (forward writes the
    /// embedded token's K/V here and self-attends over it). Guest keeps it
    /// alive for the pass's lifetime; the pass does not destroy it on drop.
    pub kv_ws: u32,
    pub kv_declaration: KvDeclaration,
    /// This pass's layer truncation, copied from
    /// [`ForwardBindings::max_layers`] at bind; stamped onto every fire's
    /// [`crate::engine::LaunchPlan`].
    pub max_layers: Option<u32>,
    /// Guest-owned recurrent-state working sets (hybrid/linear-attention
    /// models), in resolved forward-request order; empty for pure attention.
    pub rs_ws: Vec<u32>,
    /// How this pass treats recurrent state: fold, buffer, or replay.
    /// Host-known `fold-len` per working set; `None` when device-resident.
    pub rs_fold_len: Option<Vec<u32>>,
    /// Whether the bound writable declaration has performed its one-shot
    /// COW against the sharing shape at first submit.
    pub kv_declaration_realized: bool,
    /// Set when a fire of this pass failed: further submits error with the
    /// root cause (KV cursor and device channel state are unspecified after).
    pub failed: Option<String>,
    /// Device-geometry state: `Some` iff this pass's geometry is
    /// device-produced (traced in-graph, resolved pre-forward), so the
    /// host neither replays epilogue arithmetic nor projects per-lane KV.
    /// Leases physical pages and delivers grants on the program's channel.
    pub devgeo: Option<crate::pipeline::fire::lease::DevGeo>,
    /// Shape-derived decode layout whose values are resolved by the engine.
    pub decode_envelope: Option<crate::pipeline::fire::geometry::DecodeEnvelope>,
    /// Host mirror of the instance's committed channel state (seeds, then
    /// per-fire stage folds) — the value oracle for evaluated fire geometry.
    pub host_shadow: crate::pipeline::fire::shadow::HostShadow,
    /// Idempotency guard for [`ForwardPass::close_native`], set the first
    /// time native cleanup runs (explicit WIT drop, or this type's `Drop`
    /// fallback), guarding against double-closing.
    pub(crate) closed: bool,
}

impl BoundForwardPass {
    /// Replace only the recurrent-state resource reps. Legal only at an
    /// empty pipeline FIFO boundary, so no in-flight fire retains the old
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

    /// Idempotent ordered native teardown: closes the bound engine, detaches
    /// this pass's `instance_id` from every bound `ChannelCell`, and
    /// reclaims outstanding device-geometry page grants. Gated by `closed`,
    /// so repeat calls are no-ops; never panics or awaits.
    ///
    /// Callers must first confirm [`Self::can_close_native_on_drop`] (or
    /// have already drained the fires FIFO).
    pub fn close_native(&mut self) {
        if std::mem::replace(&mut self.closed, true) {
            return;
        }
        crate::offload::close_home_instance(self.bound_instance.instance_id);
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
        // Leased slots are logical reserve indexes; discarding them here
        // would shift surviving indexes under other passes on the same
        // working set, so cleanup only clears the lease's own bookkeeping.
        if let Some(devgeo) = self.devgeo.as_mut() {
            let _ = devgeo.lease.reclaim_all();
        }
    }

    /// Whether it's safe to run [`Self::close_native`] from the `Drop`
    /// fallback: the fires FIFO must hold no in-flight fire, since closing
    /// mid-completion would race a live engine write against page reuse
    /// (use-after-free). `Drop` can't await the drain, so it checks this
    /// instead; the explicit path always finds it true.
    pub(crate) fn can_close_native_on_drop(&self) -> bool {
        match &self.fires {
            None => true,
            Some(fifo) => fifo.lock().unwrap().is_empty(),
        }
    }
}

impl Drop for BoundForwardPass {
    /// Fallback for when a `ResourceTable`/`ProcessCtx` teardown drops this
    /// value directly, bypassing `HostForwardPass::drop`'s FIFO drain.
    /// Idempotent with the explicit path via `closed`.
    ///
    /// Refuses to run teardown while a fire is still in flight (`Drop` has
    /// no `.await` to drain safely): logs an error and leaves the instance,
    /// attachments and lease alone — a bounded leak, not corruption.
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
                 racing a live engine completion into a use-after-free or premature \
                 page reuse — this leaks the engine instance and its channel \
                 attachments until process exit"
            );
        }
    }
}

