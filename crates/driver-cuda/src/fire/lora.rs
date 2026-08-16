//! The `lora` sink's vocabulary and the fire-scoped staging — gate-lora,
//! slice A.
//!
//! Ports `model/lora.hpp` (the site vocabulary and the lane/table views)
//! and the staging half of `llama_like.cpp`'s lora machinery:
//! [`LoraStageArena`], [`LoraFireState`]'s construction — validation,
//! bf16 casts, same-shape grouping, the grouped xA^T layout, the pointer
//! slab — and [`stage_qkv_adapters`] with its splitmix fingerprint.
//! `apply()`, the body-time launches, is slice B and lands with the
//! emitter work that generates its call sites.
//!
//! # Grouping, the short version
//!
//! Same-shape lanes (equal `(rank, d_in, d_out)`) share one grouped-GEMM
//! launch per correction GEMM — measured at up to 24.75× over separate
//! launches when shapes share (stage0-l40s §3.1). The precondition is
//! pairwise-disjoint token spans (one grouped call runs its beta=1
//! accumulations concurrently); overlap falls back to per-lane pairs,
//! which stay correct. Groups of one are pruned — nothing to share.

use std::ffi::c_void;

use super::sideband_arena::DeviceMemory;

// ── THE LAUNCH HALF OF THIS FILE IS `kernels_cuda::gemm::lora` ────────
//
// `LoraFireState::apply` -- the three passes, the slot arithmetic and the
// four kinds of matmul in them -- descended under
// `.wiki/kernel-x/refactor-plan.md` §6.3, and `Lane`, `Group`, `bf16_row`
// and the site vocabulary went with it. The symbol went too:
// `pie_lora_qkv_correction` is `gemm::lora_qkv_correction`, so that its row
// can derive from a `fn` in a family instead of being hand-written in
// `not_yet_crossed.rs` -- a bare symbol has no namespace, and a namespace is
// what `Family::symbol` is half made of.
//
// **What stayed is what could not go**: `stage` reaches
// `sideband_arena::DeviceMemory`, a trait with five implementors in this
// crate, and lays its casts down in a per-FIRE bump arena with
// retire-on-grow. `kernels_cuda`'s two device-memory shapes are per-CALL
// (`Ctx`) and per-PROCESS (`jit::device`'s scratch); neither is a fire. So
// the arithmetic went and the lifetime stayed, which is the line FA2's
// descent drew as well.
//
// The vocabulary is re-exported rather than re-spelled: `fire::launch` and
// `tests/lora_stage_parity.rs` name these through this module, and a
// re-export makes every one of them keep resolving to the one definition.
use kernels_cuda::gemm::lora::{Group, Lane, Staged, bf16_row};
pub use kernels_cuda::gemm::lora::{
    LORA_SITE_DOWN, LORA_SITE_GATE_UP, LORA_SITE_K, LORA_SITE_O, LORA_SITE_Q, LORA_SITE_V,
    LORA_SITES_CONSUMED, LORA_SITES_KNOWN, LoraForm, LoraLaneView,
};
use kernels_cuda::jit::abi::bf16;

/// The launch's resolved lora configuration — a borrowed view, valid for
/// the fire. Ports `LoraTable`.
#[derive(Debug, Clone, Copy)]
pub struct LoraTable<'a> {
    /// One entry per lane whose program carries the sink.
    pub lanes: &'a [LoraLaneView],
}

impl LoraTable<'_> {
    /// The C++ `usable()`.
    #[must_use]
    pub const fn usable(&self) -> bool {
        !self.lanes.is_empty()
    }
}

/// The per-fire bump arena the staging draws from. Ports
/// `LoraStageArena` (`model/workspace.hpp`): 256-aligned allocs, a
/// doubling growth with a 1 MiB floor, retire-on-grow (the old block may
/// still be read by an in-flight fire), reset per fire.
#[derive(Debug, Default)]
pub struct LoraStageArena {
    buf: *mut c_void,
    buf_size: usize,
    used: usize,
    retired: Vec<*mut c_void>,
}

impl LoraStageArena {
    /// Reclaim the space, stream-ordered behind the previous fire.
    pub const fn reset(&mut self) {
        self.used = 0;
    }

    /// The current backing block's base — what the fingerprint mixes.
    #[must_use]
    pub const fn base(&self) -> *mut c_void {
        self.buf
    }

    /// Bump-allocate `bytes`, growing the backing when it does not fit.
    pub fn alloc<M: DeviceMemory>(&mut self, mem: &mut M, bytes: usize) -> *mut c_void {
        const ALIGN: usize = 256;
        let at = self.used.div_ceil(ALIGN) * ALIGN;
        if at + bytes > self.buf_size {
            let mut want = (at + bytes) * 2;
            if want < 1 << 20 {
                want = 1 << 20;
            }
            if self.buf_size > 0 {
                self.retired.push(self.buf);
            }
            self.buf = mem.alloc(want).unwrap_or(std::ptr::null_mut());
            self.buf_size = want;
        }
        self.used = at + bytes;
        // SAFETY: `at + bytes <= buf_size` by construction; the pointer
        // is only ever handed to the ops seam, never dereferenced here.
        unsafe { self.buf.cast::<u8>().add(at).cast() }
    }

    /// Free every block through the seam — the C++ leaves this to
    /// `DeviceBuffer` RAII; the port's usual explicit release.
    pub fn release<M: DeviceMemory>(&mut self, mem: &mut M) {
        if !self.buf.is_null() {
            mem.free(self.buf);
            self.buf = std::ptr::null_mut();
        }
        for p in self.retired.drain(..) {
            mem.free(p);
        }
        self.buf_size = 0;
        self.used = 0;
    }
}

/// The stream work staging issues — recorders in the parity test, CUDA in
/// the real driver.
pub trait LoraOps {
    /// `kernels::quant::cast_fp32_to_bf16`.
    fn cast_fp32_to_bf16(&mut self, src: *const c_void, dst: *mut c_void, elems: usize);
    /// The pointer-slab upload (`cudaMemcpyAsync`, host to device) — the
    /// slab must be device-resident because `cublasGemmGroupedBatchedEx`
    /// does not consume its pointer arrays synchronously.
    fn upload_slab(&mut self, dst: *mut c_void, slots: &[*const c_void]);
}

/// The live [`LoraOps`] (retirement plan phase B), behind `_cuda` for the
/// cast launch. The slab upload is `cudaMemcpyAsync` of the host pointer
/// array — device-resident because `cublasGemmGroupedBatchedEx` does not
/// consume its pointer arrays synchronously, which is the trait doc's own
/// sentence and the reason this is an upload rather than an argument.
#[cfg(feature = "_cuda")]
#[derive(Debug, Clone, Copy)]
pub struct LiveLoraOps {
    stream: *mut c_void,
}

#[cfg(feature = "_cuda")]
impl LiveLoraOps {
    /// Ops ordered on the fire's stream.
    #[must_use]
    pub const fn new(stream: *mut c_void) -> Self {
        Self { stream }
    }
}

/// The staging arena's allocator, which is the same three CUDA calls
/// the sideband arena makes.
///
/// `LiveLoraOps` implements BOTH traits because the staging needs both
/// — memory to lay the slab in and kernels to cast with — and splitting
/// them would mean two handles onto one stream.
#[cfg(feature = "_cuda")]
impl crate::fire::sideband_arena::DeviceMemory for LiveLoraOps {
    fn alloc(&mut self, bytes: usize) -> Option<*mut c_void> {
        use cudarc::runtime::sys::{cudaError, cudaMalloc};
        let mut p: *mut c_void = core::ptr::null_mut();
        let code = unsafe { cudaMalloc(std::ptr::from_mut(&mut p), bytes) };
        (code == cudaError::cudaSuccess && !p.is_null()).then_some(p)
    }

    fn free(&mut self, ptr: *mut c_void) {
        let _ = unsafe { cudarc::runtime::sys::cudaFree(ptr) };
    }

    fn synchronize(&mut self) -> bool {
        use cudarc::runtime::sys::{cudaError, cudaStreamSynchronize};
        unsafe { cudaStreamSynchronize(self.stream.cast()) == cudaError::cudaSuccess }
    }
}

#[cfg(feature = "_cuda")]
impl LoraOps for LiveLoraOps {
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // seam method; recorders share it
    fn cast_fp32_to_bf16(&mut self, src: *const c_void, dst: *mut c_void, elems: usize) {
        // SAFETY: the caller holds `self.stream` live across the launch —
        // the same assertion this method made when it handed the stream to
        // `ffi::pie_k_quant_cast_fp32_to_bf16`, which put it in a `<<<>>>`.
        let fired = unsafe {
            let ctx = kernels_cuda::jit::Ctx::on(self.stream);
            kernels_cuda::quant::cast_fp32_to::<bf16>(
                &ctx,
                src.cast::<f32>(),
                dst.cast::<kernels_cuda::jit::abi::bf16>(),
                elems,
            )
        };
        empty_or_panic("quant::cast_fp32_to_bf16", fired);
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)] // seam method; recorders share it
    fn upload_slab(&mut self, dst: *mut c_void, slots: &[*const c_void]) {
        use cudarc::runtime::sys::{cudaError, cudaMemcpyAsync, cudaMemcpyKind};
        let code = unsafe {
            cudaMemcpyAsync(
                dst,
                slots.as_ptr().cast(),
                std::mem::size_of_val(slots),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                self.stream.cast(),
            )
        };
        assert!(code == cudaError::cudaSuccess, "cudaMemcpyAsync: {code:?}");
    }
}

/// Why staging refused a lane table. Every message is the C++'s
/// `runtime_error` text — the refusals are part of the surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoraStageError(pub String);

impl std::fmt::Display for LoraStageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LoraStageError {}

/// `PIE_LORA_GROUPED` as a pure function — unset or first byte not `'0'`
/// means the grouped lowering is armed.
#[must_use]
pub fn lora_grouped_enabled_from(value: Option<&std::ffi::OsStr>) -> bool {
    match value {
        None => true,
        Some(v) => {
            let b = v.as_encoded_bytes();
            b.is_empty() || b[0] != b'0'
        }
    }
}

/// The rows the stage phase reads off the fire's workspace — the C++
/// passes `Workspace&` and picks five buffers; the port names them.
#[derive(Debug, Clone, Copy)]
/// The fire buffers a staging pass reads.
///
/// # The one thing still missing, stated precisely
///
/// Every other piece of the adapter path exists: [`read_lora_sink`]
/// resolves the plan, [`lane_for_instance`] resolves the addresses,
/// [`stage_qkv_adapters`] builds the state, and the executor's
/// `gemm::lora_qkv_correction` arm applies it. What is not decided is
/// WHICH of the fire's named values these five pointers are.
///
/// It is not plumbing. The staging reads `rows.q` and `rows.v` to build
/// its pointer slab, and those are the q and v PROJECTION OUTPUTS —
/// which the lowering names, differently per family text. `rows.gate`
/// is scratch the driver can supply from its own arena; `rows.norm_x`
/// and `rows.y` are the projection input under pre- and post-norm
/// placement, and which of the two applies is
/// `LlamaLikeFacts::norm_placement`.
///
/// So the resolution is: find the launch that states
/// `gemm::lora_qkv_correction`, read its operand join the way
/// [`crate::fire::launch::attention_pins`] reads attention's, and
/// take q and v from there. That is the same shape as the fix that
/// replaced attention's positional read, and for the same reason — a
/// value found by counting launches is a fact derived from where a
/// statement SITS, which is false under `GuardMode::Union`.
pub struct LoraStageRows {
    /// `ws.y` — the projection input under POST-norm placement.
    pub y: *const c_void,
    /// `ws.norm_x` — the projection input under PRE-norm placement.
    pub norm_x: *const c_void,
    /// `ws.q` — the q-site output rows.
    pub q: *mut c_void,
    /// `ws.v` — the v-site output rows.
    pub v: *mut c_void,
    /// `ws.gate` — the xA^T scratch alias, `[max_tokens, I]`.
    pub gate: *mut c_void,
}

/// The fire-scoped staging — `LoraFireState`, construction half.
///
/// `Lane` and `Group` are `kernels_cuda::gemm::lora`'s now, because every
/// field of both exists to be read by the launch this state was staged for.
/// This type writes them and hands them over as a [`Staged`]; see
/// [`Self::staged`].
#[derive(Debug)]
pub struct LoraFireState {
    lanes: Vec<Lane>,
    groups: Vec<Group>,
    ptr_slab: *mut c_void,
    slab_stride: usize,
    grouped_enabled: bool,
    /// EVERYTHING A CAPTURED LORA BODY BAKES, mixed into one number.
    ///
    /// `stage_qkv_adapters` computed this and RETURNED it, and the call
    /// site wrote `let _ = fingerprint`. Carried on the state instead, so
    /// the value and the thing it describes cannot be separated — the
    /// bucket key needs it, and a key that has to be given a number
    /// separately is a key that will one day be given the wrong one.
    ///
    /// Its readers — `recordings::capture_digest` and the bucket key in
    /// `fire::launch` — are both `feature = "abi"`, and this file is not, so
    /// a build that cannot fire computes the fingerprint and has nothing to
    /// key on. That is the honest shape rather than a defect: staging an
    /// adapter is a host job the portable half still does, and gating the
    /// FIELD would mean gating every line that fills it.
    #[cfg_attr(
        not(feature = "abi"),
        expect(dead_code, reason = "both readers are behind `abi`")
    )]
    pub(crate) capture_fingerprint: u64,
}

// `bf16_row` STOOD HERE and is `kernels_cuda::gemm::lora::bf16_row`. It
// was the one function this file's two halves BOTH used -- the staging
// addresses a row to put it in the slab, the launch addresses the same row to
// accumulate into it -- so leaving a copy behind would have been the same row
// stride computed on two sides of a crate boundary. That is the shape §6.3
// exists to find, and finding it in the file being split is the cheapest
// place it can be found.

impl LoraFireState {
    /// The constructor: validate every lane, stage the bf16 casts, group
    /// same-shape lanes, lay out the grouped scratch and the pointer
    /// slab. Ports `LoraFireState::LoraFireState` line for line;
    /// `grouped_enabled` carries the env gate as a value.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    pub fn stage<O: DeviceMemory + LoraOps>(
        ops: &mut O,
        arena: &mut LoraStageArena,
        table: &LoraTable<'_>,
        num_hidden_layers: i32,
        n: i32,
        h: i32,
        hq: i32,
        hk: i32,
        i_width: i32,
        tp_size: i32,
        rows: &LoraStageRows,
        grouped_enabled: bool,
    ) -> Result<Self, LoraStageError> {
        arena.reset();
        if tp_size != 1 {
            return Err(LoraStageError(
                "lora is not supported under tensor parallelism".into(),
            ));
        }
        let mut me = Self {
            // Filled by `stage_qkv_adapters`, which is the only thing
            // that knows what the capture will bake.
            capture_fingerprint: 0,
            lanes: Vec::with_capacity(table.lanes.len()),
            groups: Vec::new(),
            ptr_slab: std::ptr::null_mut(),
            slab_stride: 0,
            grouped_enabled,
        };
        for lane in table.lanes {
            let scale_lane = lane.form == LoraForm::Scale;
            if lane.a.is_null() || (!scale_lane && lane.b.is_null()) {
                return Err(LoraStageError(
                    "lora lane carries a null adapter address".into(),
                ));
            }
            if lane.sites_bits == 0 {
                return Err(LoraStageError("lora lane names no site".into()));
            }
            if lane.sites_bits & !LORA_SITES_KNOWN != 0 {
                return Err(LoraStageError(format!(
                    "lora SITES names bits outside the site vocabulary (bits {})",
                    lane.sites_bits
                )));
            }
            if lane.sites_bits & !LORA_SITES_CONSUMED != 0 {
                return Err(LoraStageError(format!(
                    "lora site not implemented by this forward (v0 applies q \
                     and v only; SITES bits {})",
                    lane.sites_bits
                )));
            }
            if i64::from(lane.num_layers) != i64::from(num_hidden_layers) {
                return Err(LoraStageError(format!(
                    "lora adapter declares {} layers, model has {}",
                    lane.num_layers, num_hidden_layers
                )));
            }
            if !scale_lane && i64::from(lane.d_in) != i64::from(h) {
                return Err(LoraStageError(format!(
                    "lora adapter d_in {} != hidden size {}",
                    lane.d_in, h
                )));
            }
            let require_width = |bit: u64, width: i32, site: &str| {
                if lane.sites_bits & bit != 0 && i64::from(lane.d_out) != i64::from(width) {
                    return Err(LoraStageError(format!(
                        "lora adapter d_out {} != {site} projection width {width}",
                        lane.d_out
                    )));
                }
                Ok(())
            };
            require_width(LORA_SITE_Q, hq, "q")?;
            require_width(LORA_SITE_V, hk, "v")?;
            if scale_lane {
                if i64::from(lane.token_start) > i64::from(n)
                    || i64::from(lane.token_count) > i64::from(n) - i64::from(lane.token_start)
                {
                    return Err(LoraStageError(
                        "lora scale lane token span exceeds the fire".into(),
                    ));
                }
                if lane.token_count == 0 {
                    continue;
                }
                let l_elems = lane.num_layers as usize * lane.d_out as usize;
                let a_bf16 = arena.alloc(ops, l_elems * 2);
                ops.cast_fp32_to_bf16(lane.a, a_bf16, l_elems);
                me.lanes.push(Lane {
                    view: *lane,
                    a_bf16,
                    b_bf16: std::ptr::null_mut(),
                    xa_offset: 0,
                    grouped: false,
                });
                continue;
            }
            if lane.rank == 0 || i64::from(lane.rank) > i64::from(i_width) {
                return Err(LoraStageError(format!(
                    "lora rank {} is zero or exceeds the scratch width {i_width}",
                    lane.rank
                )));
            }
            if i64::from(lane.token_start) > i64::from(n)
                || i64::from(lane.token_count) > i64::from(n) - i64::from(lane.token_start)
            {
                return Err(LoraStageError(format!(
                    "lora lane token span [{}, +{}) exceeds the fire's {n} rows",
                    lane.token_start, lane.token_count
                )));
            }
            if lane.token_count == 0 {
                continue;
            }
            let a_elems = lane.num_layers as usize * lane.rank as usize * lane.d_in as usize;
            let b_elems = lane.num_layers as usize * lane.d_out as usize * lane.rank as usize;
            let a_bf16 = arena.alloc(ops, a_elems * 2);
            let b_bf16 = arena.alloc(ops, b_elems * 2);
            ops.cast_fp32_to_bf16(lane.a, a_bf16, a_elems);
            ops.cast_fp32_to_bf16(lane.b, b_bf16, b_elems);
            me.lanes.push(Lane {
                view: *lane,
                a_bf16,
                b_bf16,
                xa_offset: 0,
                grouped: false,
            });
        }

        // Same-shape lane grouping. Key: the GEMM-shape tuple
        // (rank, d_in, d_out); precondition: pairwise-disjoint spans.
        if grouped_enabled && me.lanes.len() >= 2 && me.lane_spans_disjoint() {
            for i in 0..me.lanes.len() {
                let v = me.lanes[i].view;
                if v.form == LoraForm::Scale {
                    continue;
                }
                let pos = me.groups.iter().position(|g| {
                    g.rank == v.rank as i32 && g.d_in == v.d_in as i32 && g.d_out == v.d_out as i32
                });
                let g = match pos {
                    Some(p) => &mut me.groups[p],
                    None => {
                        me.groups.push(Group {
                            rank: v.rank as i32,
                            d_in: v.d_in as i32,
                            d_out: v.d_out as i32,
                            ..Group::default()
                        });
                        me.groups.last_mut().expect("just pushed")
                    }
                };
                g.members.push(i);
            }
            me.groups.retain(|g| g.members.len() >= 2);
            // The grouped xA^T scratch: exclusive [t, R] regions, packed.
            let mut xa_total: usize = 0;
            for gi in 0..me.groups.len() {
                for mi in 0..me.groups[gi].members.len() {
                    let idx = me.groups[gi].members[mi];
                    me.lanes[idx].grouped = true;
                    me.lanes[idx].xa_offset = xa_total;
                    xa_total +=
                        me.lanes[idx].view.token_count as usize * me.lanes[idx].view.rank as usize;
                }
            }
            let bound = usize::try_from(n.max(0)).unwrap_or(0)
                * usize::try_from(i_width.max(0)).unwrap_or(0);
            if xa_total > bound {
                return Err(LoraStageError(format!(
                    "lora grouped xA^T scratch layout ({xa_total} elems) \
                     exceeds the {n}x{i_width} ws.gate alias bound"
                )));
            }
            // Pointer-slab layout: per layer, per group, [x a xa](n each)
            // [q_act q_w q_y](nq each) [v_act v_w v_y](nv each).
            for gi in 0..me.groups.len() {
                let members: Vec<usize> = me.groups[gi].members.clone();
                for &idx in &members {
                    let bits = me.lanes[idx].view.sites_bits;
                    if bits & LORA_SITE_Q != 0 {
                        me.groups[gi].nq += 1;
                    }
                    if bits & LORA_SITE_V != 0 {
                        me.groups[gi].nv += 1;
                    }
                }
                me.groups[gi].slab_off = me.slab_stride;
                me.slab_stride += 3 * members.len()
                    + 3 * usize::try_from(me.groups[gi].nq).unwrap_or(0)
                    + 3 * usize::try_from(me.groups[gi].nv).unwrap_or(0);
            }
            if me.slab_stride > 0 {
                let layers = usize::try_from(num_hidden_layers.max(0)).unwrap_or(0);
                me.ptr_slab = arena.alloc(
                    ops,
                    layers * me.slab_stride * std::mem::size_of::<*const c_void>(),
                );
                let mut slab_host: Vec<*const c_void> =
                    vec![std::ptr::null(); layers * me.slab_stride];
                for layer in 0..layers {
                    for gi in 0..me.groups.len() {
                        let g_rank = me.groups[gi].rank;
                        let g_d_in = me.groups[gi].d_in;
                        let g_d_out = me.groups[gi].d_out;
                        let members = me.groups[gi].members.clone();
                        let mut staged = Vec::new();
                        let mut a_run = Vec::new();
                        let mut xa_run = Vec::new();
                        let (mut q_act, mut q_w, mut q_y) = (Vec::new(), Vec::new(), Vec::new());
                        let (mut v_act, mut v_w, mut v_y) = (Vec::new(), Vec::new(), Vec::new());
                        if layer == 0 {
                            let g = &mut me.groups[gi];
                            g.m.clear();
                            g.mq.clear();
                            g.mv.clear();
                        }
                        for &idx in &members {
                            let lane = &me.lanes[idx];
                            let v = lane.view;
                            // SAFETY: layer-strided offsets into the staged
                            // casts; pointers cross the ops seam only.
                            let a_l = unsafe {
                                lane.a_bf16
                                    .cast::<u16>()
                                    .add(layer * g_rank as usize * g_d_in as usize)
                            };
                            let b_l = unsafe {
                                lane.b_bf16
                                    .cast::<u16>()
                                    .add(layer * g_d_out as usize * g_rank as usize)
                            };
                            let xa = unsafe { rows.gate.cast::<u16>().add(lane.xa_offset) };
                            staged.push(bf16_row(rows.norm_x, v.token_start, h));
                            a_run.push(a_l.cast_const().cast::<c_void>());
                            xa_run.push(xa.cast_const().cast::<c_void>());
                            if layer == 0 {
                                me.groups[gi]
                                    .m
                                    .push(i32::try_from(v.token_count).unwrap_or(0));
                            }
                            if v.sites_bits & LORA_SITE_Q != 0 {
                                q_act.push(xa.cast_const().cast::<c_void>());
                                q_w.push(b_l.cast_const().cast::<c_void>());
                                q_y.push(bf16_row(rows.q.cast_const(), v.token_start, hq));
                                if layer == 0 {
                                    me.groups[gi]
                                        .mq
                                        .push(i32::try_from(v.token_count).unwrap_or(0));
                                }
                            }
                            if v.sites_bits & LORA_SITE_V != 0 {
                                v_act.push(xa.cast_const().cast::<c_void>());
                                v_w.push(b_l.cast_const().cast::<c_void>());
                                v_y.push(bf16_row(rows.v.cast_const(), v.token_start, hk));
                                if layer == 0 {
                                    me.groups[gi]
                                        .mv
                                        .push(i32::try_from(v.token_count).unwrap_or(0));
                                }
                            }
                        }
                        let mut slot = layer * me.slab_stride + me.groups[gi].slab_off;
                        for run in [
                            &staged, &a_run, &xa_run, &q_act, &q_w, &q_y, &v_act, &v_w, &v_y,
                        ] {
                            for &p in run {
                                slab_host[slot] = p;
                                slot += 1;
                            }
                        }
                    }
                }
                ops.upload_slab(me.ptr_slab, &slab_host);
            }
        }
        Ok(me)
    }

    /// True iff no two lanes' token spans overlap. Ports
    /// `lane_spans_disjoint`.
    fn lane_spans_disjoint(&self) -> bool {
        let mut by_start: Vec<&LoraLaneView> = self.lanes.iter().map(|l| &l.view).collect();
        by_start.sort_by_key(|v| v.token_start);
        for w in by_start.windows(2) {
            if w[0].token_start + w[0].token_count > w[1].token_start {
                return false;
            }
        }
        true
    }

    /// Whether this staged state may be recorded into a union capture.
    ///
    /// Only the GROUPED path may. `apply`'s solo path is a host-side loop
    /// over lanes whose launch count and shapes follow the fire's adapter
    /// set — its rank, its token spans, which sites each lane touches — so
    /// a capture bakes the lanes it happened to see and a later fire with
    /// a different set needs a different launch sequence. That is not a
    /// pointer to hand in; the arm is a PROGRAM whose shape is a variant
    /// axis with unbounded cardinality, and no conditional node folds one.
    ///
    /// The grouped path is slot arithmetic over `ptr_slab` and a fixed
    /// launch per group — which is what `apply` says where it does it,
    /// "the slab was fully staged at fire setup … what a captured body
    /// requires". It was written for this.
    ///
    /// So a fire joins a union only if EVERY lane grouped. This is an
    /// eligibility rule of the same shape the C++ arc used for mixed
    /// peels: what cannot be replayed stays eager.
    ///
    /// Note the group SHAPE is still baked by a capture (the member count
    /// and the `m` vector reach the launcher as arguments), so a bucket
    /// key that admits LoRA has to carry it. `true` here means "may be
    /// captured", not "may share any exec".
    #[must_use]
    pub fn union_capture_safe(&self) -> bool {
        self.grouped_enabled && self.lanes.iter().all(|l| l.grouped)
    }

    /// The one-line grouping summary `PIE_LORA_FIRE_TRACE` prints. Ports
    /// `grouping_desc`.
    #[must_use]
    pub fn grouping_desc(&self) -> String {
        if !self.grouped_enabled {
            return "off".into();
        }
        let solo = self.lanes.iter().filter(|l| !l.grouped).count();
        if self.groups.is_empty() {
            return format!("none({solo} solo)");
        }
        let mut s = String::new();
        for g in &self.groups {
            if !s.is_empty() {
                s.push(',');
            }
            s.push_str(&format!("{}xr{}", g.members.len(), g.rank));
        }
        if solo > 0 {
            s.push_str(&format!("+{solo}solo"));
        }
        s
    }

    /// What the launch half needs, as one borrow.
    ///
    /// The whole of this type's output. `apply` used to be a method here and
    /// is `kernels_cuda::gemm::lora_qkv_correction` now; this is the seam
    /// between the two, and it is a BORROW rather than a copy because the two
    /// vectors are the fire's and outlive every launch made from them.
    ///
    /// It is also the reason the fields stay private. A `Staged` is what any
    /// reader outside this file may have, and it is exactly what the launcher
    /// reads -- so nothing can reach `capture_fingerprint` or `grouped_\
    /// enabled` through it, and those two have their own readers for their own
    /// reasons.
    #[must_use]
    pub fn staged(&self) -> Staged<'_> {
        Staged {
            lanes: &self.lanes,
            groups: &self.groups,
            ptr_slab: self.ptr_slab,
            slab_stride: self.slab_stride,
        }
    }
}

// `LoraFireState::apply` STOOD HERE, and `grouped_or_panic` under it. Both
// are `kernels_cuda::gemm::lora`'s -- see the note at the top of this
// file for what moved and what could not.
//
// `grouped_or_panic` did not survive the move, and its own doc is why: it
// existed because *"a staged LoRA apply has nowhere to put a `Declined`"*,
// `apply` having returned `()`. `gemm::lora_qkv_correction` returns
// `Result<(), Refusal>`, so the three call sites propagate with `?` and the
// abort is a value the dispatch arm reports like any other decline. The
// unreachability argument that doc made still holds and now lives beside the
// guards that establish it, in `stage` above.

/// An empty extent is a no-op; every other decline is a bug.
///
/// `fire::dtype_cast` stood between this file and the two `quant` casts and
/// returned `()`, because a `bind::jit::fire` returns `()` and swallows its
/// own refusals. `x::quant::{cast_fp32_to_bf16, scale_rows_bf16}` return
/// `#[must_use] Fired` instead, and the two outcomes they can produce are not
/// the same outcome:
///
///   * `Declined(Empty)` is `dtype_cast.cu:50`'s `if (n == 0) return;` and
///     `:65`'s `if (rows == 0 || width == 0) return;`, moved from the C++ into
///     the host program unchanged. Callers relied on it — a LoRA lane with no
///     tokens reaches here — so it must stay a no-op HERE, on the caller's
///     side, which is where the loader's no-op lived all along.
///   * Anything else means the host program refused an argument this file
///     built, which the C++ had no way to say and this file has no way to
///     handle.
///
/// So the arm is named and the second case aborts with the symbol, for the
/// reason its one remaining caller cannot escape: `let _ =` would spell
/// "it declined" like "it ran", and `LoraOps::cast_fp32_to_bf16` returns
/// `()` because the trait's other implementor is a recorder with no device
/// to be refused by. The launch half had the same problem and no longer
/// does -- see the note where `apply` stood.
fn empty_or_panic(symbol: &str, fired: Result<(), kernels_cuda::Refusal>) {
    if let Err(why) = fired
        && !matches!(why, kernels_cuda::Refusal::Empty { .. })
    {
        panic!("{symbol} declined: {why:?}");
    }
}

/// splitmix64's finalizer — the fingerprint's mixer.
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Stage the fire's lora state and answer a fingerprint of everything a
/// captured lora body bakes: the lane structure, the adapter device
/// pointers, the grouping mode, and the post-staging arena base (a growth
/// changes addresses and must recapture). `0` means no lora. Ports
/// `llama_like_lora_stage` (the C++'s name); the staged state and the
/// table it was staged from go to the caller — the `lora_staged` slot in
/// C++ — rather than being written here, because the plan state's `L`
/// parameter is exactly this type.
///
/// NAMED FOR WHAT IT DOES, not for where it came from. The C++ spells
/// it `llama_like_lora_stage`, and provenance belongs in a doc line:
/// nothing about staging q/k/v adapters is llama's, and a function
/// named after a family teaches the next reader that there is a family
/// axis here. `tests/no_family_names.rs` is what noticed.
#[allow(clippy::too_many_arguments)]
pub fn stage_qkv_adapters<O: DeviceMemory + LoraOps>(
    ops: &mut O,
    arena: &mut LoraStageArena,
    table: Option<&LoraTable<'_>>,
    num_hidden_layers: i32,
    total_tokens: i32,
    h: i32,
    hq: i32,
    hk: i32,
    i_width: i32,
    tp_size: i32,
    post_norm: bool,
    rows: &LoraStageRows,
    grouped_enabled: bool,
) -> Result<(u64, Option<LoraFireState>), LoraStageError> {
    let Some(table) = table else {
        return Ok((0, None));
    };
    if !table.usable() {
        return Ok((0, None));
    }
    let qkv_in = if post_norm { rows.y } else { rows.norm_x };
    let staged = LoraFireState::stage(
        ops,
        arena,
        table,
        num_hidden_layers,
        total_tokens,
        h,
        hq,
        hk,
        i_width,
        tp_size,
        &LoraStageRows {
            norm_x: qkv_in,
            ..*rows
        },
        grouped_enabled,
    )?;

    let mut h64 = mix(u64::try_from(table.lanes.len()).unwrap_or(0));
    h64 ^= mix(u64::try_from(total_tokens.max(0)).unwrap_or(0));
    h64 ^= mix(if grouped_enabled { 1 } else { 2 });
    h64 ^= mix(arena.base() as u64);
    for v in table.lanes {
        h64 ^= mix(u64::from(v.rank))
            .wrapping_add(mix(u64::from(v.d_in)).wrapping_mul(3))
            .wrapping_add(mix(u64::from(v.d_out)).wrapping_mul(5));
        h64 ^= mix(v.sites_bits)
            .wrapping_add(mix(u64::from(v.token_start)).wrapping_mul(7))
            .wrapping_add(mix(u64::from(v.token_count)).wrapping_mul(11));
        h64 ^= mix(v.a as u64);
        h64 ^= mix(v.b as u64);
    }
    let h64 = if h64 == 0 { 1 } else { h64 };
    // CARRIED, not merely returned. The bucket key needs it and the call
    // site used to drop it; a value that travels beside the thing it
    // describes cannot be dropped without dropping both.
    let staged = LoraFireState {
        capture_fingerprint: h64,
        ..staged
    };
    Ok((h64, Some(staged)))
}

// ── Reading the sink: a plan says WHICH adapter, the caller says where ──

/// The `lora` sink's operand layout, read out of a prologue stage plan.
///
/// `fwd.adapter(site, |x, y| expr)` recognises LoRA `y + mm(b, mm(a, x))`,
/// IA3 `scale(y, l)` and the DoRA composite, and lowers all of them to ONE
/// pass-wide sink in the program's PROLOGUE: `SinkCall { name: "lora" }`
/// with three args for the low-rank form (`a`, `b`, `sites`) or two for
/// the scale form (`l`, `sites`). The ARITY selects the form — see
/// `tensor-compiler/src/codegen/cuda/validate.rs`, which refuses any other
/// count and any stage but the prologue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoraSink {
    /// The channel the `a` (or, on a scale sink, the `l`) tensor arrives on
    /// — a PROGRAM-GLOBAL dense channel index, which is what
    /// `LaunchStagePlan::channel_bindings` resolves a stage-local slot to.
    pub a_channel: u32,
    /// The `b` channel, absent on a scale sink.
    pub b_channel: Option<u32>,
    /// The sites bitmask, a trace-known literal.
    pub sites_bits: u64,
    /// Layers, taken from `a`'s first dimension.
    pub num_layers: u32,
    /// The rank, `a`'s second dimension; zero on a scale sink.
    pub rank: u32,
    /// `a`'s last dimension; zero on a scale sink.
    pub d_in: u32,
    /// `b`'s middle dimension, or `l`'s width on a scale sink.
    pub d_out: u32,
    /// Which form the arity chose.
    pub form: LoraForm,
}

/// Why a plan's `lora` sink could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SinkRefusal {
    /// The stage states no `lora` sink. Not an error — most stages do not.
    Absent,
    /// It states one, and something about it does not resolve.
    Malformed(&'static str),
}

/// `PTIR_OP_SINK_CALL`.
const SINK_CALL: u16 = 0xA2;

/// Find and resolve the `lora` sink in one stage plan.
///
/// # Why this reads three tables rather than one
///
/// A sink's operands are stage-local VALUE ids, and the ops that produce
/// them are not in `plan.ops` at all: `codegen/launch.rs::lower_stages`
/// turns `chan_take`, `chan_read`, `const` and `intrinsic_val` into stage
/// EFFECTS, so the op list has no producer to walk back to. What survives
/// is `channel_rules` — one `{ value, local }` per channel-touching op —
/// which is exactly the value→channel map, and `value_types`, which is
/// indexed by the same value ids and carries the dims.
///
/// So: `ops` says which values the sink consumes, `channel_rules` says
/// which channel each of them came off, `channel_bindings` widens a
/// stage-local slot to the program's dense numbering, and `value_types`
/// says what shape it is. A caller then turns a dense index into an
/// address, which is the one thing this cannot do — see
/// [`LoraSink::a_channel`].
///
/// # Errors
///
/// [`SinkRefusal::Absent`] when the stage states no sink, which is the
/// ordinary case. [`SinkRefusal::Malformed`] when it states one that does
/// not resolve — and that is worth distinguishing, because a program that
/// asked for an adapter and silently ran without it returns an answer
/// indistinguishable from a correct one.
pub fn read_lora_sink(
    plan: &driver::driver_api::plan::LaunchStagePlan,
) -> Result<LoraSink, SinkRefusal> {
    let op = plan
        .ops
        .iter()
        .find(|op| {
            op.code == SINK_CALL
                && plan
                    .names
                    .get(op.name_index as usize)
                    .is_some_and(|n| n == "lora")
        })
        .ok_or(SinkRefusal::Absent)?;

    let form = match op.args.len() {
        3 => LoraForm::LowRank,
        2 => LoraForm::Scale,
        _ => {
            return Err(SinkRefusal::Malformed(
                "a lora sink takes two or three args",
            ));
        }
    };
    // The SITES literal is the last arg either way, and it is a constant
    // the trace knew: `lit_bits` on the op that defined it. That op is a
    // `const`, which does reach `plan.ops`, so unlike the channel operands
    // it can be found by walking.
    let sites_value = *op.args.last().expect("checked arity");
    let sites_bits = plan
        .ops
        .iter()
        .enumerate()
        .find_map(|(i, o)| (i as u32 == sites_value).then_some(u64::from(o.lit_bits)))
        .ok_or(SinkRefusal::Malformed("the sites operand names no op"))?;
    if sites_bits & !LORA_SITES_KNOWN != 0 {
        return Err(SinkRefusal::Malformed(
            "the sites mask names a site this driver has no arm for",
        ));
    }

    let channel_of = |value: u32| -> Option<u32> {
        let local = plan
            .channel_rules
            .iter()
            .find(|rule| rule.value == value)
            .map(|rule| rule.local)?;
        plan.channel_bindings.get(local as usize).copied()
    };
    let dims_of = |value: u32| -> Option<&[u32]> {
        plan.value_types
            .get(value as usize)
            .map(|v| v.dims.as_slice())
    };

    let a_value = op.args[0];
    let a_channel = channel_of(a_value).ok_or(SinkRefusal::Malformed(
        "the adapter's first operand is not a channel",
    ))?;
    let a_dims = dims_of(a_value).ok_or(SinkRefusal::Malformed(
        "the adapter's first operand has no type",
    ))?;

    match form {
        // `a` is [layers, R, d_in] and `b` is [layers, d_out, R].
        LoraForm::LowRank => {
            let &[layers, rank, d_in] = a_dims else {
                return Err(SinkRefusal::Malformed("a low-rank A is [layers, R, d_in]"));
            };
            let b_value = op.args[1];
            let b_channel = channel_of(b_value).ok_or(SinkRefusal::Malformed(
                "the adapter's B operand is not a channel",
            ))?;
            let b_dims = dims_of(b_value).ok_or(SinkRefusal::Malformed(
                "the adapter's B operand has no type",
            ))?;
            let &[b_layers, d_out, b_rank] = b_dims else {
                return Err(SinkRefusal::Malformed("a low-rank B is [layers, d_out, R]"));
            };
            // CHECKED RATHER THAN ASSUMED, because the two tensors arrive
            // on separate channels and nothing upstream pairs them: a
            // caller that re-seeded one and not the other would otherwise
            // get a GEMM over mismatched inner dimensions, which is a
            // wrong answer and not a fault.
            if b_layers != layers || b_rank != rank {
                return Err(SinkRefusal::Malformed(
                    "the adapter's A and B disagree on layers or rank",
                ));
            }
            Ok(LoraSink {
                a_channel,
                b_channel: Some(b_channel),
                sites_bits,
                num_layers: layers,
                rank,
                d_in,
                d_out,
                form,
            })
        }
        // `l` is [layers, d_out]: a scale has no rank and no input width.
        LoraForm::Scale => {
            let &[layers, d_out] = a_dims else {
                return Err(SinkRefusal::Malformed("a scale L is [layers, d_out]"));
            };
            Ok(LoraSink {
                a_channel,
                b_channel: None,
                sites_bits,
                num_layers: layers,
                rank: 0,
                d_in: 0,
                d_out,
                form,
            })
        }
    }
}

impl LoraSink {
    /// This sink as one lane's view, given the addresses the caller
    /// resolved and the token span the lane owns.
    ///
    /// The split is deliberate: everything above is a pure read of the
    /// plan and is testable without a device, while an ADDRESS is a fact
    /// about a live session's rings. Keeping the two apart is what lets
    /// the resolution be proved and the binding be simple.
    #[must_use]
    pub const fn lane(
        &self,
        a: *const c_void,
        b: *const c_void,
        token_start: u32,
        token_count: u32,
    ) -> LoraLaneView {
        LoraLaneView {
            a,
            b,
            sites_bits: self.sites_bits,
            token_start,
            token_count,
            num_layers: self.num_layers,
            rank: self.rank,
            d_in: self.d_in,
            d_out: self.d_out,
            form: self.form,
        }
    }
}

#[cfg(test)]
mod sink_tests {
    use super::*;
    use driver::driver_api::plan::{LaunchChannelRule, LaunchOp, LaunchPlanValue, LaunchStagePlan};

    /// Value ids are op INDICES here, which is what the plan's stage-local
    /// numbering makes them for a straight-line stage.
    fn value(dims: &[u32]) -> LaunchPlanValue {
        LaunchPlanValue {
            dtype: 0,
            extents: vec![0; dims.len()],
            dims: dims.to_vec(),
        }
    }

    fn const_op(bits: u32) -> LaunchOp {
        LaunchOp {
            code: 0x81,
            lit_bits: bits,
            ..LaunchOp::default()
        }
    }

    fn sink(args: Vec<u32>) -> LaunchOp {
        LaunchOp {
            code: SINK_CALL,
            name_index: 0,
            args,
            ..LaunchOp::default()
        }
    }

    /// A prologue whose sink is the low-rank form: A on channel slot 0,
    /// B on slot 1, and a sites literal.
    fn low_rank_plan() -> LaunchStagePlan {
        LaunchStagePlan {
            names: vec!["lora".to_owned()],
            // 0: A [4, 8, 64]   1: B [4, 128, 8]   2: sites   3: the sink
            value_types: vec![
                value(&[4, 8, 64]),
                value(&[4, 128, 8]),
                value(&[]),
                value(&[]),
            ],
            ops: vec![
                LaunchOp::default(),
                LaunchOp::default(),
                const_op(u32::try_from(LORA_SITE_Q | LORA_SITE_V).expect("fits")),
                sink(vec![0, 1, 2]),
            ],
            channel_rules: vec![
                LaunchChannelRule { value: 0, local: 0 },
                LaunchChannelRule { value: 1, local: 1 },
            ],
            // Stage-local slot 0 is the program's channel 5, slot 1 is 9.
            channel_bindings: vec![5, 9],
            ..LaunchStagePlan::default()
        }
    }

    #[test]
    fn a_low_rank_sink_names_its_channels_its_sites_and_its_geometry() {
        let read = read_lora_sink(&low_rank_plan()).expect("the sink resolves");
        assert_eq!(
            read.form,
            LoraForm::LowRank,
            "three args is the low-rank form"
        );
        assert_eq!(
            (read.a_channel, read.b_channel),
            (5, Some(9)),
            "the stage-local slots widen through channel_bindings"
        );
        assert_eq!(read.sites_bits, LORA_SITE_Q | LORA_SITE_V);
        assert_eq!(
            (read.num_layers, read.rank, read.d_in, read.d_out),
            (4, 8, 64, 128),
            "A is [layers, R, d_in] and B is [layers, d_out, R]"
        );
    }

    /// Two args is IA3, and it has no rank and no input width. Reading it
    /// as a low-rank sink would give a GEMM two dimensions it does not
    /// have.
    #[test]
    fn a_two_arg_sink_is_the_scale_form() {
        let mut plan = low_rank_plan();
        plan.value_types[0] = value(&[4, 128]);
        plan.ops[3] = sink(vec![0, 2]);
        let read = read_lora_sink(&plan).expect("the scale sink resolves");
        assert_eq!(read.form, LoraForm::Scale);
        assert_eq!(read.b_channel, None, "a scale has no B");
        assert_eq!(
            (read.rank, read.d_in),
            (0, 0),
            "and neither a rank nor an input width"
        );
        assert_eq!(read.d_out, 128);
    }

    /// The ordinary case. Most stages state no sink and must not be read
    /// as stating a broken one.
    #[test]
    fn a_stage_with_no_sink_is_absent_rather_than_malformed() {
        let plan = LaunchStagePlan {
            ops: vec![LaunchOp::default()],
            ..LaunchStagePlan::default()
        };
        assert_eq!(read_lora_sink(&plan), Err(SinkRefusal::Absent));
    }

    /// A and B arrive on SEPARATE channels and nothing upstream pairs
    /// them, so a caller that re-seeded one and not the other would hand
    /// the GEMM mismatched inner dimensions — a wrong answer, not a fault.
    #[test]
    fn an_a_and_b_that_disagree_are_refused() {
        let mut plan = low_rank_plan();
        plan.value_types[1] = value(&[4, 128, 16]); // rank 16, not 8
        assert!(
            matches!(read_lora_sink(&plan), Err(SinkRefusal::Malformed(_))),
            "a rank mismatch between A and B is refused"
        );
    }

    /// A site this driver has no arm for is refused rather than dropped.
    /// A program that asked for an adapter and silently ran without it
    /// returns an answer indistinguishable from a correct one.
    #[test]
    fn an_unknown_site_is_refused_rather_than_masked_off() {
        let mut plan = low_rank_plan();
        plan.ops[2] = const_op(1 << 20);
        assert!(matches!(
            read_lora_sink(&plan),
            Err(SinkRefusal::Malformed(_))
        ));
    }

    /// An operand that is not a channel cannot be an adapter tensor: the
    /// whole point of the channel form is that swapping an adapter is
    /// re-seeding rather than re-tracing.
    #[test]
    fn an_operand_off_no_channel_is_refused() {
        let mut plan = low_rank_plan();
        plan.channel_rules.clear();
        assert!(matches!(
            read_lora_sink(&plan),
            Err(SinkRefusal::Malformed(_))
        ));
    }
}

// ── The address half: a plan says WHICH channel, a session says WHERE ──

/// Resolve one instance's `lora` sink into a lane view.
///
/// [`read_lora_sink`] answers *which* channels the adapter arrives on
/// and what shape it is; this answers *where*, which is a fact about a
/// live session's rings rather than about a plan. The two are separate
/// functions because the first is testable without a device and the
/// second is not.
///
/// # Why this could not exist until sessions were hoisted
///
/// A cell's address comes from `Rings`, and the driver built one lazily
/// inside `run_program` — the sampler, which runs AFTER the forward.
/// The `lora` sink lives in the program's PROLOGUE, which is before. The
/// blocker was never a missing function; it was the order two things
/// happened in. `launch::ensure_sessions` fixed the order.
///
/// # Errors
///
/// `None` when the instance has no program, its program states no
/// prologue, the prologue states no `lora` sink, or the sink's channels
/// are not ones this session rings. Each of those is an ordinary
/// adapter-free fire rather than a failure — a program without an
/// adapter is the common case.
// Same seam as `fire::launch`: it reads the shell's instance table,
// which is the door's own state.
// `pub(crate)` rather than `pub`, because its `instances` parameter is a map
// of `serve::state::InstanceEntry` and that type is the shell's own — an entry
// in the instance table, meaningless outside the door that keeps it. Exported
// at `pub` the signature named a type nobody outside could spell, which rustc
// reports as "more private than the item" and which is really the visibility
// having been chosen for the function alone rather than for its arguments.
// `fire::launch` is the only caller and is in this crate.
#[cfg(feature = "abi")]
#[must_use]
pub(crate) fn lane_for_instance(
    programs: &crate::program::Programs,
    sessions: &std::collections::BTreeMap<u64, crate::program::session::Session>,
    rings: &crate::program::channel::Rings,
    instances: &std::collections::BTreeMap<u64, crate::serve::state::InstanceEntry>,
    instance_id: u64,
    token_start: u32,
    token_count: u32,
    stream: crate::device::StreamRef<'_>,
) -> Option<LoraLaneView> {
    let instance = instances.get(&instance_id)?;
    let compiled = programs.get(instance.program_id)?;
    // THE PROLOGUE'S plan, by KIND. `plans.first()` would be the
    // epilogue on every program in the tree today, which is the same
    // accident `run_program` was fixed for.
    let stage = compiled.stage_of_kind(crate::program::runtime::stage_kind::PROLOGUE)?;
    let plan = compiled.plans.get(stage)?;
    let sink = read_lora_sink(plan).ok()?;

    let session = sessions.get(&instance_id)?;
    let cursors = rings.cursors(stream).ok()?;
    // THE COMMITTED CELL, not the pending one. An adapter is SEEDED —
    // the host publishes it once and the fire reads it — so what a fire
    // wants is the last value that was committed, which is what `head`
    // names. Reading `tail` would be reading a cell nobody filled.
    //
    // THROUGH THE SESSION'S SLOT MAP, because the rings are the driver's now
    // and `dense` is this instance's own numbering. Indexing the registry with
    // a dense number would hand the adapter whichever channel happened to be
    // registered at that position — a real cell, and somebody else's.
    let address = |dense: u32| -> Option<*const std::ffi::c_void> {
        let c = session.slot(dense as usize)? as usize;
        let cursor = cursors.get(c)?;
        let at = rings.cell_address(c, cursor.head).ok()?;
        (at != 0).then(|| at as *const std::ffi::c_void)
    };

    let a = address(sink.a_channel)?;
    let b = match sink.b_channel {
        // A SCALE SINK HAS NO B, and null is the form `LoraFireState`
        // expects for it — see `LoraForm::Scale`, whose `b` is null and
        // whose rank and `d_in` are zero.
        None => core::ptr::null(),
        Some(dense) => address(dense)?,
    };
    Some(sink.lane(a, b, token_start, token_count))
}
