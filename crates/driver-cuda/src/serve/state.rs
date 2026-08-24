//! The shell's state, and the device-lifetime things hung off it.
//!
//! A leaf: nothing here calls into `load`, `launch`, `encode` or `transfer`;
//! they all call in. The types are the driver's nouns; the verbs are next door.

use crate::fire::scratch::Scratch;
use driver_api::completion::{CompletionBroker, CompletionTarget};
use driver_api::local::PIE_STATUS_DRIVER_ERROR;

/// The shell's state: the receiver of every driver entry-point method.
pub struct Shell {
    /// What this device is, parsed once at create from the capabilities JSON.
    pub(crate) facts: driver_api::DeviceFacts,
    // `boot_config` STOOD HERE — `[model] config`, the path to a
    // `config.json` for snapshots that embedded none. The LEGACY LOAD
    // CONTRACT was its only reader: `Encoding::from_config_json` parsed it to
    // tell the contract how the numbers were stored. R3 deleted the contract;
    // the SKU's text states its own dtypes and `produce` reads the checkpoint
    // through the import table, so this driver parses no `config.json` at all.
    /// `[model] id` from the boot TOML, or `None` to read the tensors.
    pub(crate) boot_model_id: Option<String>,
    /// Does this driver hand completions to a stream callback and return before
    /// the fire retires? From [`boot::Boot::runahead`](crate::boot::Boot::runahead).
    pub(crate) runahead: bool,
    /// The parse of this driver's knobs; see [`crate::boot`].
    pub(crate) boot: crate::boot::Boot,
    /// How the KV pages are stored — `[driver] kv_cache_dtype`; the layout plans
    /// scale planes per scheme and the paged-attention kernels switch on it.
    pub(crate) kv_format: crate::layout::KvCacheFormat,
    /// The cuBLAS handle, created once (`cublasDestroy` costs ~3.2 ms); stream
    /// rebound per fire via `cublasSetStream`.
    pub(crate) cublas:
        Option<crate::device::cublas::CublasHandle<cudarc::cublas::sys::cublasHandle_t>>,
    // `preds` and `peel_win` STOOD HERE — the device words a captured union
    // read its guards and its peel split out of. Both are the legacy walk's
    // and both are gone with it.
    /// The pinned host buffer the logits D2H lands in, grown and reused. The
    /// shell's, not the fire's: a stream callback may not free it.
    pub(crate) logits_staging: Option<crate::device::PinnedBuf>,
    /// Staging buffers a wider fire replaced, held until nothing is in flight:
    /// `cudaFreeHost` is not stream-ordered, so a queued fire may still read them.
    pub(crate) retired_staging: Vec<crate::device::PinnedBuf>,
    /// Is this boot measuring rather than serving? See `[driver] calibrate_planner`.
    pub(crate) calibrating: bool,
    /// The CUDA device ordinal this driver binds; `build_tp_plane` all-gathers
    /// it so the group's device map matches what each rank bound.
    pub(crate) device_ordinal: i32,
    /// This driver's place in its tensor-parallel group. Both feed the load plan
    /// and KV geometry: a rank cannot be one width for weights, another for KV.
    pub(crate) tp_rank: u32,
    pub(crate) tp_size: u32,
    /// The loaded model, once `load_model` succeeds.
    pub(crate) model: Option<LoadedModel>,
    /// THE PROGRAM THIS DRIVER FIRES: one `Program` per lane, its weights,
    /// and the geometry read off its plan. Built at `load_model` and dropped
    /// with `model`, because its banks are device addresses in an arena the
    /// next load would free.
    ///
    /// `None` only before a load, or after one that refused. A `Some` here
    /// is what makes a fire possible at all — `step_impl` refuses by name
    /// rather than reaching for a second path, because there is not one.
    pub(crate) baker: Option<crate::baker::Baked>,
    /// Which load this is, from one: the identity model-keyed caches need. Not a
    /// path hash — a reload reallocates, so must not reuse the id.
    pub(crate) load_generation: u64,
    /// Registered programs by id; the C3 hash is the dedup key.
    pub(crate) programs: std::collections::BTreeMap<u64, ProgramEntry>,
    /// Bound instances by id.
    pub(crate) instances: std::collections::BTreeMap<u64, InstanceEntry>,
    /// The next never-used id; programs and instances share the counter.
    pub(crate) next_id: u64,
    /// Who to tell when work finishes, from `create`: a `CompletionBroker`
    /// (`Clone + Send`) a stream callback publishes through by name.
    pub(crate) broker: CompletionBroker,
    /// The hybrid's GDN state slabs, allocated on first hybrid launch.
    pub(crate) gdn: Option<GdnState>,
    // `supergraph` STOOD HERE — `fire::recordings::Recordings`, the
    // instantiated graphs the legacy walk replayed, one per (R, N) bucket.
    // The baker walk is eager; the perf debt is named at the walk.
    /// The per-fire device arrays, pooled so a capture can outlive the fire that
    /// recorded it (see [`Scratch`]). Dropped after the execs that address it.
    pub(crate) fire_arrays: Scratch,
    /// This rank's custom P2P all-reduce plane (`Some` for any rank of a group).
    /// Declared after [`Self::supergraph`] so its captured execs drop first, and
    /// its raw device pointers owned by the creating thread make `Shell` `!Send`.
    pub(crate) all_reduce: Option<crate::fire::all_reduce::ResidentPlane>,
    /// The key this rank's tensor-parallel group rendezvouses on (`[driver]
    /// tp_group_id`), for `layout::rendezvous` and the memory planner.
    pub(crate) tp_group_id: String,
    /// The driver-owned KV pools, allocated on first launch and grown on demand.
    pub(crate) kv: Option<KvState>,
    /// Registered channels: the pinned host ring endpoints the engine maps.
    pub(crate) channels: std::collections::BTreeMap<u64, ChannelState>,
    /// The host-pinned KV swap pool where `copy_kv`'s host domain lands:
    /// page-granular, per layer, both planes. Grown by highest page id touched.
    pub(crate) swap: Option<SwapPool>,
    // `lora_arena` STOOD HERE, with `fire::lora` behind it. The adapter it
    // staged was fired by exactly one thing, `bind::dispatch`'s
    // `gemm::lora_qkv_correction` arm, and that arm is deleted.
    /// The fire scratch held per driver: the attention workspace and both
    /// FlashInfer plan caches, created on first launch.
    pub(crate) scratch: Option<FireScratch>,
    /// The fire stream and its allocator, held per driver, not per fire, so a
    /// stream outlives the fire that queued work on it — run-ahead needs that.
    pub(crate) fire_stream: Option<crate::device::OwnedStream>,
    /// The fire still running, if any — see [`InFlight`]. One slot.
    pub(crate) in_flight: std::collections::VecDeque<InFlight>,
    /// The allocator every fire's transient device memory comes from.
    pub(crate) fire_alloc: Option<crate::device::Allocator>,
    /// The PTIR plane: [`crate::program::Runtime`] is the cache (a shared
    /// stage's second registration is free), `ptir_programs` owns the modules.
    pub(crate) ptir: crate::program::Runtime,
    /// The compiled form of each registered program, by program id.
    pub(crate) ptir_programs: crate::program::Programs,
    /// The control kernels, compiled once for this device's architecture. Lazy:
    /// a driver that never fires a program should not pay NVRTC.
    pub(crate) ptir_control: Option<crate::program::Control>,
    /// Every registered channel's device ring, by driver-wide slot: one
    /// registry, so a channel two instances name is one ring. Built lazily.
    pub(crate) ptir_rings: Option<crate::program::channel::Rings>,
    /// Which slot each registered channel's ring lives at, by channel id.
    /// Assigned on first use — no allocator until the first fire.
    pub(crate) ptir_channel_slots: std::collections::BTreeMap<u64, u32>,
    /// One instance's dense channel index -> registry slot, by instance id: per instance
    /// since a program numbers channels by its own `channel_ids`, and rebuilding renumbers.
    pub(crate) ptir_sessions: std::collections::BTreeMap<u64, crate::program::session::Session>,
    /// The adopted plans, by program id — kept apart from the compiled modules
    /// since a program can be adopted then rejected, yet a launch must report why.
    pub(crate) ptir_plans: std::collections::BTreeMap<u64, driver::ExecPlan>,
}

/// One raised attention schedule and the workspace it was carved in.
///
/// ONE WORKSPACE PER SCHEDULE, and that is a FlashInfer fact rather than a
/// budget: a plan writes its work list into the workspace it was raised
/// against, so two live schedules sharing one would each read the other's.
///
/// GENERIC OVER THE PLAN, because the two readings differ in NOTHING ELSE. A
/// decode work list and a prefill one are carved by different planners into
/// different caches, and everything around them — the class they are filed
/// under, the workspace that holds them, the fence that publishes them — is
/// one sentence said twice. [`DecodeSchedule`] and [`PrefillSchedule`] are
/// that sentence's two readings.
pub(crate) struct Schedule<P> {
    /// The class this was planned at — `(head_dim, window)` as the lane's
    /// statements state them, plus the kv heads the pool row carries.
    pub class: crate::baker::DecodeClass,
    pub ws: crate::fire::attention_workspace::AttentionWorkspace<cudarc::runtime::sys::cudaEvent_t>,
    pub plan: P,
}

/// What `attention.decode` reads, one per class its lane states.
pub(crate) type DecodeSchedule = Schedule<crate::bind::DecodePlan>;

/// What `attention.masked` reads, one per class its lane states — and, for a
/// lane that states no masked arm, the cache the planless prefill leg carves
/// into per statement and asks for classless.
pub(crate) type PrefillSchedule = Schedule<crate::bind::PrefillPlan>;

/// Driver-lifetime fire scratch.
pub(crate) struct FireScratch {
    /// One entry per decode CLASS any lane of this model has fired, grown on
    /// demand and never shrunk. `decode_plan_full` and a fixed pair STOOD
    /// HERE: the legacy walk kept a second decode schedule for a stack whose
    /// layer kinds disagree about head dim and picked between them off a
    /// `LaunchSpec`'s window. A `Program` is one lane, its statements name
    /// their own class (`kernels::raises::Class`), and this is the same two
    /// schedules addressed by what the statement says rather than by what the
    /// lowering remembered.
    pub decode: Vec<DecodeSchedule>,
    /// The same table for the PREFILL cache, and a fixed pair — one plan, one
    /// workspace — stood here for the mirror of the same reason:
    /// `attention.masked` refused a stated window, so a lane could state one
    /// masked geometry or already be refused. It serves the window now, so
    /// gemma's masked lane states two, and this holds one entry per masked
    /// class under the same growth rule.
    ///
    /// A LANE THAT STATES NO MASKED ARM ASKS FOR EXACTLY ONE, the deployment's
    /// own widest unwindowed geometry, and gets it STAMPED rather than planned:
    /// the planless prefill leg replans that cache per statement out of the
    /// host CSR mirrors, so what it needs is the cache and a stamped
    /// workspace. One ask, one allocation — which is what a fire of every
    /// other SKU has always made, measured.
    ///
    /// BOTH TABLES ARE KEYED BY GEOMETRY, not by the class a body asks with,
    /// and gemma is where the two spellings visibly differ: its fallback
    /// geometry — a 512-wide head, unwindowed, 2 kv heads — IS its second
    /// masked class, so ONE entry serves both roles and the tower stages three
    /// prefill workspaces rather than four. That is sound because a `Program`
    /// is one lane and a lane states one arm: a masked fire plans this entry
    /// before reading it, an unmasked fire stamps it before reading it, and
    /// neither ever sees the other's leftovers. Which schedule a BODY reaches
    /// is a separate question, answered by class in `bind::views`. See
    /// `Baked::attn_ask`.
    pub prefill: Vec<PrefillSchedule>,
    // `tail_plan` and `tail_ws` STOOD HERE TOO, and served a peel. Nothing
    // peels.
}

/// The pinned swap pool: `layers × [pages × page_bytes]` per plane.
pub(crate) struct SwapPool {
    /// One pinned region per `(layer, buffer)`, in `plan.buffers()` order: a quantized
    /// cache has up to four buffers/layer and varying head dims, so count and width vary.
    pub regions: Vec<*mut std::ffi::c_void>,
    /// The plan those regions were allocated against; what this pool can serve.
    pub plan: crate::pools::swap_pool::SwapPoolLayout,
    /// The two stream roles the plan asked for, kept for the driver's life: an
    /// eviction queued behind a restore is the stall the second stream avoids.
    pub evict: Option<crate::device::OwnedStream>,
    pub restore: Option<crate::device::OwnedStream>,
}

impl SwapPool {
    pub(crate) fn free(&self) {
        use crate::fire::attention_workspace::{LiveStagingOps, StagingOps};
        let mut ops = LiveStagingOps;
        for &b in &self.regions {
            ops.free_host(b);
        }
    }

    /// The host base of one `(layer, buffer)` region.
    pub(crate) fn region(&self, layer: u32, buffer: u32) -> Option<*mut u8> {
        let i = self
            .plan
            .buffers()
            .iter()
            .position(|b| b.layer == layer && b.buffer == buffer)?;
        Some(self.regions.get(i)?.cast::<u8>())
    }
}

/// One channel's host endpoint: the pinned mirror and the four control words.
#[derive(Clone, Copy)]
pub(crate) struct ChannelState {
    pub mirror: *mut std::ffi::c_void,
    pub words: *mut std::ffi::c_void,
    pub mirror_bytes: usize,
    /// wire bytes per cell — bit-packed for bools.
    pub cell_bytes: usize,
    /// `capacity + 1` — the ring modulus.
    pub ring: u32,
    pub host_role: u8,
    /// Lanes in one cell, and the cell's element type. Needed because
    /// `cell_bytes` is not invertible (a bool cell packs 8 lanes/byte).
    pub numel: usize,
    pub dtype: driver::tensor_ir::DType,
    /// `PIE_CHANNEL_EXTERN_*`: private to one instance, or crossing programs?
    /// Recorded but not acted on; `bind_instance` refuses one.
    pub extern_dir: u8,
}

impl ChannelState {
    /// Does this channel cross a program boundary? An extern channel is shared
    /// by two programs over one ring; `bind_instance` still refuses one.
    pub const fn is_extern(&self) -> bool {
        self.extern_dir != driver_api::local::PIE_CHANNEL_EXTERN_NONE
    }

    /// This channel as the device rings want it.
    pub(crate) fn shape(&self) -> crate::program::channel::ChannelShape {
        crate::program::channel::ChannelShape {
            numel: self.numel,
            dtype: self.dtype,
            // `ring` is `capacity + 1`; `ChannelShape` wants the capacity.
            capacity: self.ring.saturating_sub(1),
        }
    }
}

/// A fire's transient device memory, kept alive until it retires: `cudaFree`
/// synchronizes and CUDA forbids runtime calls from a host callback.
pub(crate) struct InFlight {
    pub done: crate::device::Event,
    /// Ordinary scratch, held only so dropping it does not synchronize at the
    /// wrong moment — nothing here is meant to be read again.
    #[expect(dead_code, reason = "owned to defer the drop; see the type's doc")]
    pub(crate) scratch: Vec<crate::device::DeviceBuffer>,
    /// Channels closed while this fire was queued, freed when it retires:
    /// freeing the mirror early would race the callback still writing to it.
    pub closed_channels: Vec<ChannelState>,
}

/// Give back what a retired fire held: a `ChannelState` is a pair of raw host
/// allocations this shell owns and must free by hand.
pub(crate) fn retire(fire: InFlight) {
    for ch in &fire.closed_channels {
        ch.free();
    }
}

/// How many fires the driver may have queued ahead of the GPU. Backpressure
/// by scratch, not by time: the bound is on how much the driver is carrying.
pub(crate) const RUNAHEAD_DEPTH: usize = 2;

// `LoweringKey`, `digest_rows` and `LoweredFire` STOOD HERE — the cache key
// and the cached triple (`ForwardPlan`, `Lowered`, `DispatchPlan`) that made
// the legacy trace-lower-join chain affordable at ~3.3 ms a fire shape. A
// `Program` is built once, at load, so there is nothing per-fire left to
// cache and nothing to key it on.

/// Everything a fire still owes at enqueue, paid from a stream-ordered callback
/// that cannot borrow — every field owned, no `cudaFreeHost` on live staging.
pub(crate) struct FireDebt {
    /// The bf16 logits D2H'd into the shell's pinned staging, as (pointer,
    /// length): pageable host memory would block `cudaMemcpyAsync`.
    pub staging: Option<(*const u8, usize)>,
    /// One `(reader channel, logits row)` per request. The row is not the index:
    /// request `r`'s answer is at `qo_indptr[r + 1] - 1` (equal to `r` on decode).
    pub readouts: Vec<(ChannelState, usize)>,
    pub vocab: usize,
    /// The terminal cells this frame publishes, and the completion awaited.
    pub cells: Vec<*mut driver_api::local::TerminalCell>,
    pub completion: CompletionTarget,
    pub(crate) broker: CompletionBroker,
}

// The debt crosses to a CUDA callback thread. Every field is owned bytes or a
// raw pointer into memory the runtime keeps alive for the driver's lifetime.
unsafe impl Send for FireDebt {}

/// The stream-ordered callback: pay the debt, then drop it.
///
/// # Safety
///
/// `data` is a `Box<FireDebt>` leaked by the enqueuing side. CUDA forbids
/// calling back into the runtime from here, and nothing below does.
pub(crate) unsafe extern "C" fn retire_fire(data: *mut std::ffi::c_void) {
    if data.is_null() {
        return;
    }
    let debt = unsafe { Box::from_raw(data.cast::<FireDebt>()) };

    // The logits, widened bf16 -> f32: the ring's cell is f32, the device bf16.
    if let Some(&(ptr, len)) = debt.staging.as_ref()
        && debt.vocab > 0
    {
        // SAFETY: the shell's staging buffer, alive for the driver's lifetime;
        // the D2H that filled it is ordered before this callback on one stream.
        let staged = unsafe { std::slice::from_raw_parts(ptr, len) };
        for (ch, row) in &debt.readouts {
            let mut cell = vec![0u8; debt.vocab * 4];
            for t in 0..debt.vocab {
                let off = (row * debt.vocab + t) * 2;
                if off + 1 < staged.len() {
                    let bits = u16::from_le_bytes([staged[off], staged[off + 1]]);
                    cell[t * 4..t * 4 + 4].copy_from_slice(&(u32::from(bits) << 16).to_le_bytes());
                }
            }
            if !ch.publish(&cell) {
                eprintln!("[driver-cuda] launch: logits ring full; a request dropped its output");
            }
        }
    }

    // Then the terminal cells, then the notify: `publish` is a release store
    // pairing with the runtime's `load(Acquire)` once the notify lands.
    for &cell in &debt.cells {
        if !cell.is_null() {
            unsafe {
                (*cell).publish(driver_api::local::PIE_TERMINAL_OUTCOME_SUCCESS);
            }
        }
    }
    // Fenced before the notify: the channel publishes above are a different
    // plane, and the runtime reads those on the notify too.
    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
    debt.broker
        .notify(debt.completion.wait_id, debt.completion.target_epoch);
}

impl ChannelState {
    /// The host (writer) plane of this channel's ring: release-ordered cursor writes that
    /// must agree byte-for-byte with the engine's poller. `close_channel` defers the
    /// mirror/words free onto an in-flight fire, so they outlive any view.
    pub(crate) fn host_plane(&self) -> crate::program::channel::HostChannel {
        debug_assert!(self.cell_bytes * self.ring as usize <= self.mirror_bytes);
        unsafe {
            crate::program::channel::HostChannel::new(
                self.mirror,
                self.words,
                self.cell_bytes,
                self.ring,
                self.host_role,
            )
        }
    }

    pub(crate) fn publish(&self, cell: &[u8]) -> bool {
        self.host_plane().publish(cell)
    }
}

impl ChannelState {
    pub(crate) fn free(&self) {
        use crate::fire::attention_workspace::{LiveStagingOps, StagingOps};
        let mut ops = LiveStagingOps;
        ops.free_host(self.mirror);
        ops.free_host(self.words);
    }
}

/// The shell's KV: one (k, v) pool per layer, plus page capacity. A `None` row owns no
/// pages (KV-shared trailing layers alias their source); `_held` gives the `Drop` `KvCache` lacks.
pub(crate) struct KvState {
    pub cache: crate::pools::kv_cache_live::KvCache<crate::pools::kv_cache_live::AllResident>,
    /// Backing store for `cache`; dropping this frees the pages.
    pub _held: Vec<crate::device::DeviceBuffer>,
    pub num_pages: u32,
}

impl KvState {
    /// The pages `layer` owns, or `None` if it reads through another's.
    pub(crate) fn owned(
        &self,
        layer: usize,
    ) -> Option<(*mut core::ffi::c_void, *mut core::ffi::c_void)> {
        let l = i32::try_from(layer).ok()?;
        let slot = self.cache.layout().slots().get(layer)?;
        if slot.is_alias() {
            return None;
        }
        Some((self.cache.k(l), self.cache.v(l)))
    }

    /// Bytes of one page at `layer` — its own stride, so the two-head-dim
    /// families move the right amount per layer.
    pub(crate) fn page_bytes(&self, layer: usize) -> Option<usize> {
        let slot = self.cache.layout().slots().get(layer)?;
        let k = slot.k.as_ref()?;
        Some(usize::try_from(k.nbytes()).unwrap_or(0) / self.num_pages.max(1) as usize)
    }

    /// How many layers the cache describes.
    pub(crate) fn layers(&self) -> usize {
        self.cache.layout().slots().len()
    }

    /// `layer`'s head dim. The config's single number is wrong for the families
    /// whose layers disagree, and this is the extent the copy actually strides.
    pub(crate) fn head_dim(&self, layer: usize) -> Option<i32> {
        let slot = self.cache.layout().slots().get(layer)?;
        slot.k.as_ref()?;
        Some(self.cache.layout().head_dim_at(i32::try_from(layer).ok()?))
    }

    /// What a kernel is handed for each layer.
    pub(crate) fn views(&self) -> Vec<crate::bind::abi::KvCacheLayerView> {
        (0..self.layers())
            .map(|l| self.cache.layer_view(i32::try_from(l).unwrap_or(0)))
            .collect()
    }
}

/// The hybrid's driver-owned GDN state via the ported [`RecurrentStateCache`]: one
/// allocation per buffer kind, addressed `linear_index * max_slots * stride + slot * stride`.
pub(crate) struct GdnState {
    /// The cache: the layout, the strides, and what to do to the buffers.
    pub cache: crate::pools::recurrent_state_cache::RecurrentStateCache,
    /// The two pooled allocations, in `Buffer` order. `mtp` is absent until
    /// the MTP pending-hidden row has a writer.
    pub conv: crate::device::DeviceBuffer,
    pub recurrent: crate::device::DeviceBuffer,
    /// Which model layers are linear, mapping a model layer to a linear index.
    pub is_linear: Vec<bool>,
    pub num_slots: u32,
    pub conv_stride_elems: i64,
    pub state_stride_elems: i64,
}

impl GdnState {
    /// The device base of one model layer's conv window, or 0 where the layer
    /// is full-attention. What `GdnCtx::conv_state` carries per layer.
    pub(crate) fn conv_base(&self, layer: usize) -> u64 {
        self.base(layer, true)
    }

    /// The same for the recurrent state.
    pub(crate) fn recurrent_base(&self, layer: usize) -> u64 {
        self.base(layer, false)
    }

    fn base(&self, layer: usize, conv: bool) -> u64 {
        let l = match u32::try_from(layer) {
            Ok(l) => l,
            Err(_) => return 0,
        };
        let addr = if conv {
            self.cache.layout().conv_state(l, 0)
        } else {
            self.cache.layout().recurrent_state(l, 0)
        };
        let Some(addr) = addr else { return 0 };
        let pool = if conv { &self.conv } else { &self.recurrent };
        (pool.as_ptr() as u64).wrapping_add(addr.offset)
    }

    /// Run the ops a `RecurrentStateCache` routine asked for: it returns [`StateOp`]s
    /// rather than calling CUDA (checkable without a GPU); here they become calls.
    pub(crate) fn apply(
        &self,
        ops: &[crate::pools::recurrent_state_cache::StateOp],
        stream: crate::device::StreamRef<'_>,
    ) -> Result<(), i32> {
        use crate::pools::recurrent_state_cache::{Buffer, StateOp};
        use cudarc::runtime::sys::{
            cudaError, cudaMemcpy2DAsync, cudaMemcpyAsync, cudaMemcpyKind, cudaMemset2DAsync,
            cudaMemsetAsync,
        };
        let base = |b: Buffer| -> Option<*mut u8> {
            match b {
                Buffer::Conv => Some(self.conv.as_ptr().cast::<u8>()),
                Buffer::Recurrent => Some(self.recurrent.as_ptr().cast::<u8>()),
                // No writer yet, so no allocation; an op against a missing
                // buffer is skipped, which `has_mtp_hidden` gates.
                Buffer::MtpHidden => None,
            }
        };
        let at = |p: *mut u8, off: u64| unsafe { p.add(usize::try_from(off).unwrap_or(0)) };
        let n = |v: u64| usize::try_from(v).unwrap_or(0);
        for op in ops {
            let code = match *op {
                StateOp::Memset {
                    buffer,
                    offset,
                    len,
                } => {
                    let Some(p) = base(buffer) else { continue };
                    unsafe { cudaMemsetAsync(at(p, offset).cast(), 0, n(len), stream.as_raw()) }
                }
                StateOp::Memset2D {
                    buffer,
                    offset,
                    pitch,
                    width,
                    rows,
                } => {
                    let Some(p) = base(buffer) else { continue };
                    unsafe {
                        cudaMemset2DAsync(
                            at(p, offset).cast(),
                            n(pitch),
                            0,
                            n(width),
                            n(rows),
                            stream.as_raw(),
                        )
                    }
                }
                StateOp::Memcpy {
                    buffer,
                    dst,
                    src,
                    len,
                } => {
                    let Some(p) = base(buffer) else { continue };
                    unsafe {
                        cudaMemcpyAsync(
                            at(p, dst).cast(),
                            at(p, src).cast_const().cast(),
                            n(len),
                            cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                            stream.as_raw(),
                        )
                    }
                }
                StateOp::Memcpy2D {
                    buffer,
                    dst,
                    src,
                    pitch,
                    width,
                    rows,
                } => {
                    let Some(p) = base(buffer) else { continue };
                    unsafe {
                        cudaMemcpy2DAsync(
                            at(p, dst).cast(),
                            n(pitch),
                            at(p, src).cast_const().cast(),
                            n(pitch),
                            n(width),
                            n(rows),
                            cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                            stream.as_raw(),
                        )
                    }
                }
                // Needs two device arrays this shell does not build; the reset
                // instead fires earlier from the host via `PIE_RS_FLAG_RESET`.
                StateOp::ZeroSlotsIfFresh { .. } => continue,
            };
            if code != cudaError::cudaSuccess {
                return Err(PIE_STATUS_DRIVER_ERROR);
            }
        }
        Ok(())
    }

    /// Grow to cover `need` slots, migrating survivors per linear layer (growing
    /// `max_slots` restrides every layer), and bump `epoch` if the pools moved:
    /// a captured graph replaying against the old bases would hit freed memory.
    pub(crate) fn ensure_slots(
        &mut self,
        need: u32,
        epoch: &mut crate::fire::scratch::PlanEpoch,
        alloc: &crate::device::Allocator,
        stream: &crate::device::OwnedStream,
    ) -> Result<bool, i32> {
        if self.num_slots >= need {
            return Ok(false);
        }
        let grown =
            crate::pools::recurrent_state_cache::RecurrentStateCache::allocate_bf16_recurrent(
                &self.is_linear,
                self.cache.conv_dim(),
                self.cache.conv_kernel(),
                self.cache.v_heads(),
                self.cache.head_k_dim(),
                self.cache.head_v_dim(),
                i32::try_from(need).unwrap_or(i32::MAX),
            );
        let (conv_n, rec_n) = (
            usize::try_from(grown.layout().conv_total_bytes())
                .unwrap_or(0)
                .max(1),
            usize::try_from(grown.layout().recurrent_total_bytes())
                .unwrap_or(0)
                .max(1),
        );
        let mut conv = alloc.alloc(conv_n)?;
        let mut recurrent = alloc.alloc(rec_n)?;
        conv.memset(0, stream.as_ref())?;
        recurrent.memset(0, stream.as_ref())?;
        {
            use cudarc::runtime::sys::{cudaError, cudaMemcpyAsync, cudaMemcpyKind};
            let keep = self.num_slots;
            let copy = |dst: *mut u8, src: *const u8, bytes: u64| -> Result<(), i32> {
                if bytes == 0 {
                    return Ok(());
                }
                let code = unsafe {
                    cudaMemcpyAsync(
                        dst.cast(),
                        src.cast(),
                        usize::try_from(bytes).unwrap_or(0),
                        cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                        stream.as_ref().as_raw(),
                    )
                };
                (code == cudaError::cudaSuccess)
                    .then_some(())
                    .ok_or(PIE_STATUS_DRIVER_ERROR)
            };
            for l in 0..self.is_linear.len() {
                let lu = u32::try_from(l).unwrap_or(0);
                for conv_side in [true, false] {
                    let (old_a, new_a) = if conv_side {
                        (
                            self.cache.layout().conv_state(lu, 0),
                            grown.layout().conv_state(lu, 0),
                        )
                    } else {
                        (
                            self.cache.layout().recurrent_state(lu, 0),
                            grown.layout().recurrent_state(lu, 0),
                        )
                    };
                    let (Some(old_a), Some(new_a)) = (old_a, new_a) else {
                        continue;
                    };
                    let (old_pool, new_pool) = if conv_side {
                        (self.conv.as_ptr().cast::<u8>(), conv.as_ptr().cast::<u8>())
                    } else {
                        (
                            self.recurrent.as_ptr().cast::<u8>(),
                            recurrent.as_ptr().cast::<u8>(),
                        )
                    };
                    copy(
                        unsafe { new_pool.add(usize::try_from(new_a.offset).unwrap_or(0)) },
                        unsafe { old_pool.add(usize::try_from(old_a.offset).unwrap_or(0)) }
                            .cast_const(),
                        new_a.len * u64::from(keep),
                    )?;
                }
            }
        }
        stream.as_ref().synchronize()?;
        self.cache = grown;
        self.conv = conv;
        self.recurrent = recurrent;
        self.num_slots = need;
        // grew, so every capture that recorded a slab base is stale.
        epoch.bump();
        Ok(true)
    }
}

/// Install a rebuilt KV pool and bump the generation: the pages moved, so centralizing
/// the bump stops either rebuild path replaying a captured launch against freed pages.
pub(crate) fn install_kv(
    kv: &mut Option<KvState>,
    epoch: &mut crate::fire::scratch::PlanEpoch,
    next: KvState,
) {
    *kv = Some(next);
    epoch.bump();
}

/// What registration keeps of a program today: the identity the engine dedups
/// on. The launch package itself is deep-copied when the `launch` arm lands.
pub(crate) struct ProgramEntry {
    pub program_hash: u64,
    #[allow(dead_code)] // read when launch's compile cache lands
    pub emitter_version: u32,
}

/// A bound instance: which program, the geometry the binding echoed, and its
/// attached channels.
pub(crate) struct InstanceEntry {
    #[allow(dead_code)] // read when launch resolves frames to instances
    pub program_id: u64,
    #[allow(dead_code)]
    pub geometry_class: u32,
    pub channel_ids: Vec<u64>,
    /// The value each seeded channel starts with, in wire form, by channel id: held not
    /// applied at bind (no allocator there); `launch::ensure_sessions` applies it later.
    pub seeds: Vec<(u64, Vec<u8>)>,
}

/// What a successful `load_model` leaves behind: which SKU this is, the
/// pool geometry read off its plan, and the caps that were published from
/// both. The WEIGHTS are not here — they are `baker::Baked::banks`, produced
/// through the SKU's import table and held by the lane that fires them.
pub(crate) struct LoadedModel {
    /// The catalog SKU this checkpoint matched — a `&'static str` that
    /// reaches `DriverCapabilities::model_id` and, through it, the host's
    /// chat template. One id space since R3: this is the same string
    /// `model::catalog()` files the row under.
    pub id: &'static str,
    /// What this checkpoint is, read once at load off the same `Plan` the
    /// lane's program is built from. Carries no family name by design, so
    /// the fire path cannot special-case on one.
    pub deployment: model::deployment::Deployment,
    /// What `load_model` answered with, TYPED.
    ///
    /// It was the JSON, and `load_model` returns a `DriverCapabilities`, so
    /// the document was built, serialised, stored, and then parsed back —
    /// once to answer the caller and three more times inside the driver
    /// (`warm_lane`'s rectangle, `calibrate_planner`'s ceiling, the P2P
    /// plane's token bound). Four `serde_json::from_slice` calls against
    /// bytes this process had just written.
    ///
    /// Each one also manufactured a failure that cannot happen — "the caps
    /// did not parse" — and two of those printed a refusal sentence and
    /// returned early. A branch on a parse that cannot fail is a branch no
    /// test can enter and no reader can price.
    pub caps: driver_api::DriverCapabilities,
}

// `LoadedModel::tp_size` STOOD HERE — "the group this rank's weights were
// sharded for, carried from the shell so a family's facts and its load plan
// cannot disagree on rank width". It had one reader, `capabilities_json`,
// which copied it out because the planner needed it after the borrow of
// `state` had ended. The caps are built BEFORE the model is stored now, so
// the planner reads `Shell::tp_size` directly and the copy that existed to
// survive a borrow has no borrow to survive. Two fields that must agree are
// one field again.

// `LoadedModel::{weights, owned, aliases, layer_scalars}` and the `weight()`
// resolver STOOD HERE — the LEGACY LOAD CONTRACT's arena and its name map.
// `model::produce` reads the checkpoint through the SKU's own import table
// now and `baker::Baked::banks` is the only weight map a fire touches, so a
// second residency with a second spelling has nothing left to answer for.

/// The facts a scheduler reads. Storage facts import from `StorageTarget::for_backend` so
/// they can't drift; `native_mxfp4_moe` false is a trap — a native MXFP4 *GEMM* (an unported
/// Marlin repack), not "reads MXFP4".
pub(crate) fn device_facts() -> driver_api::DeviceFacts {
    driver_api::DeviceFacts {
        abi_version: driver_api::PIE_DRIVER_ABI_VERSION,
        backend: "cuda".to_string(),
        // false: a discrete card's KV pool and host do not share physical
        // memory, so "the device is full" is about the card alone.
        unified_memory: false,
        // true: fp8 quantize kernels and fp8 page storage the kernels read.
        fp8_native: true,
        native_mxfp4_moe: false,
        storage_alignment: 256,
        storage_max_tile_bytes: 64 * 1024 * 1024,
        // Not transcribed: the bits are 1, 4 and 128, not the 1, 2, 4 a reader
        // would guess, so this imports `CUDA_TILE_MAP_MASK`.
        storage_tile_map_mask: model_loader::plan::passes::tile::CUDA_TILE_MAP_MASK,
        // The paged KV pool's rows per page, the unit of every `kv_translation`
        // index; `boot::KV_PAGE_SIZE` is the same sixteen the kernels compile for.
        page_size: crate::boot::KV_PAGE_SIZE.unsigned_abs(),
    }
}

/// The device-ring shapes of one instance's channels, in program index order — the
/// contract: a missing channel is a refusal, not a gap (skipping renumbers). `None` if unheld.
#[cfg(feature = "abi")]
pub(crate) fn instance_ring_shapes(
    instance: &InstanceEntry,
    channels: &std::collections::BTreeMap<u64, ChannelState>,
) -> Option<Vec<crate::program::channel::ChannelShape>> {
    instance
        .channel_ids
        .iter()
        .map(|id| channels.get(id).map(ChannelState::shape))
        .collect()
}

/// The wire dtype byte, as the tensor IR names it — a lookup (vocabularies agree on 0..=3).
/// `PIE_CHANNEL_DTYPE_ACT` (4) is an activation channel, not a dtype, so it reads as `F32`.
pub(crate) fn channel_dtype(byte: u8) -> driver::tensor_ir::DType {
    driver::tensor_ir::DType::from_wire(byte).unwrap_or(driver::tensor_ir::DType::F32)
}
