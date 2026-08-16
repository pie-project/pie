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
    /// `[model] config` from the boot TOML, for snapshots with no embedded config.
    pub(crate) boot_config: Option<std::path::PathBuf>,
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
    /// Cached traced-and-lowered program per fire shape (~3.3 ms/fire), keyed
    /// including union-asked (a union may be declined).
    pub(crate) lowerings: std::collections::BTreeMap<LoweringKey, LoweredFire>,
    /// The cuBLAS handle, created once (`cublasDestroy` costs ~3.2 ms); stream
    /// rebound per fire via `cublasSetStream`.
    pub(crate) cublas:
        Option<crate::device::cublas::CublasHandle<cudarc::cublas::sys::cublasHandle_t>>,
    /// The fire's predicate word, allocated once: `cudaFree` synchronizes the
    /// device, and a captured graph bakes this address, so it must not move.
    pub(crate) preds: Option<crate::device::PredicateWord>,
    /// The fire's peel-window word, allocated once; same reasoning as `preds`.
    pub(crate) peel_win: Option<crate::device::PeelWindowWord>,
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
    /// The unionized supergraph's instantiated graphs, one per (R, N) bucket;
    /// empty unless `PIE_CUDA_SUPERGRAPH` armed it. Declared before
    /// [`Self::fire_arrays`], whose addresses its execs hold, so it drops first.
    pub(crate) supergraph: crate::fire::recordings::Recordings,
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
    pub(crate) swap: Option<SwapPool>,    /// The adapter staging's bump arena, driver-lifetime: reset each fire, grown
    /// on demand. Must never retire a block an in-flight fire may still read.
    pub(crate) lora_arena: crate::fire::lora::LoraStageArena,
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

/// Driver-lifetime fire scratch.
pub(crate) struct FireScratch {
    pub ws: crate::fire::attention_workspace::AttentionWorkspace<cudarc::runtime::sys::cudaEvent_t>,
    /// The prefill plan's own workspace: a FlashInfer plan writes its schedule
    /// into the workspace it was raised against, so sharing would clobber.
    pub prefill_ws:
        crate::fire::attention_workspace::AttentionWorkspace<cudarc::runtime::sys::cudaEvent_t>,
    pub decode_plan: crate::bind::DecodePlan,
    /// gemma-4's second decode plan — the full layers' 512-wide geometry;
    /// single-kind families never plan it.
    pub decode_plan_full: crate::bind::DecodePlan,
    pub prefill_plan: crate::bind::PrefillPlan,
    /// A peel tail's decode plan and its own workspace — like `prefill_ws`, and
    /// a tail serves `[split, N)`, a different request count, hence its schedule.
    pub tail_plan: crate::bind::DecodePlan,
    pub tail_ws:
        crate::fire::attention_workspace::AttentionWorkspace<cudarc::runtime::sys::cudaEvent_t>,
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

/// What a lowering can depend on: see [`Shell::lowerings`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) struct LoweringKey {
    pub model_id: u64,
    pub class: model_ir::trace::FireClass,
    pub rows: u32,
    /// A digest of the row axes, not just the count: the lowering resolves
    /// per-row guards a plain row count cannot distinguish.
    pub rows_digest: u64,
    pub union_asked: bool,
}

/// FNV-1a over the rows' axes. Not a hash of the struct: `Row` is not `Hash`,
/// so naming the axes means a new one is added to the key deliberately.
pub(crate) fn digest_rows(rows: &[model_compiler::lower::Row]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |b: u64| {
        h ^= b;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for r in rows {
        eat(u64::from(r.multi_token));
        eat(u64::from(r.custom_mask));
        eat(u64::from(r.hooked));
        eat(u64::from(r.lora));
        eat(u64::from(r.write_desc));
        eat(u64::from(r.wants_scores));
        eat(u64::from(r.samples));
        eat(r.depth_k.map_or(u64::MAX, u64::from));
    }
    h
}

/// A traced, lowered and joined program, and whether it kept its union.
pub(crate) struct LoweredFire {
    pub plan: model_ir::trace::ForwardPlan,
    pub lowered: model_compiler::lower::Lowered,
    pub dplan: crate::bind::DispatchPlan,
    pub union: bool,
}

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
        epoch: &mut crate::fire::recordings::PlanEpoch,
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
    epoch: &mut crate::fire::recordings::PlanEpoch,
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

/// What a successful `load_model` leaves behind: the parsed config and every
/// weight resident on the device, keyed by checkpoint (and fused trace) name.
pub(crate) struct LoadedModel {
    /// The catalog row this checkpoint matched, by id — a `&'static str` that
    /// reaches `DriverCapabilities::model_id` and the host's chat template.
    pub id: &'static str,
    /// What this checkpoint is, derived once at load. Carries no family name by
    /// design, so the fire path cannot special-case on one.
    pub deployment: model::deployment::Deployment,
    /// The caps JSON `load_model` answered with; owned like `Shell::caps`.
    pub load_caps: Vec<u8>,
    /// Every tensor the plan named, as an arena span not an allocation: a resident plan
    /// lays the model out contiguously, so a weight is an offset into one buffer.
    pub weights: std::collections::BTreeMap<String, crate::weights::stage::WeightSpan>,
    /// The arena, and anything the plan published outside it. Held so the spans
    /// above stay valid; never indexed.
    #[allow(dead_code)]
    pub owned: Vec<crate::device::DeviceBuffer>,
    /// Trace-name renames onto checkpoint names (`layer.3.attn_norm` →
    /// `model.layers.3.input_layernorm.weight`) — a row here, not a second copy.
    pub aliases: std::collections::BTreeMap<String, String>,
    /// The per-layer `layer_scalar` [1] tensors, read to host once at load: the
    /// fused sandwich norm's multiplier. Empty where wiring names none.
    pub layer_scalars: Vec<f32>,
    /// The group this rank's weights were sharded for, carried from the shell so
    /// a family's facts and its load plan cannot disagree on rank width.
    pub(crate) tp_size: u32,
}

impl LoadedModel {
    /// The device pointer for a name — the live half of the executor's
    /// `Resolver::weight`. Checkpoint names, fused names and aliases all answer.
    #[allow(dead_code)]
    pub(crate) fn weight(&self, name: &str) -> Option<*const std::ffi::c_void> {
        if let Some(b) = self.weights.get(name) {
            return Some(b.ptr.cast_const());
        }
        let target = self.aliases.get(name)?;
        self.weights.get(target).map(|b| b.ptr.cast_const())
    }
}

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

