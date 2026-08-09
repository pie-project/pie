//! The `lora` sink's vocabulary and the fire-scoped staging — gate-lora,
//! slice A.
//!
//! Ports `model/lora.hpp` (the site vocabulary and the lane/table views)
//! and the staging half of `llama_like.cpp`'s lora machinery:
//! [`LoraStageArena`], [`LoraFireState`]'s construction — validation,
//! bf16 casts, same-shape grouping, the grouped xA^T layout, the pointer
//! slab — and [`llama_like_lora_stage`] with its splitmix fingerprint.
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

/// `q_proj` — consumed.
pub const LORA_SITE_Q: u64 = 1 << 0;
/// `k_proj` — reserved.
pub const LORA_SITE_K: u64 = 1 << 1;
/// `v_proj` — consumed.
pub const LORA_SITE_V: u64 = 1 << 2;
/// `o_proj` — reserved.
pub const LORA_SITE_O: u64 = 1 << 3;
/// gate/up — reserved.
pub const LORA_SITE_GATE_UP: u64 = 1 << 4;
/// `down_proj` — reserved.
pub const LORA_SITE_DOWN: u64 = 1 << 5;
/// Every bit the vocabulary defines.
pub const LORA_SITES_KNOWN: u64 = LORA_SITE_Q
    | LORA_SITE_K
    | LORA_SITE_V
    | LORA_SITE_O
    | LORA_SITE_GATE_UP
    | LORA_SITE_DOWN;
/// The bits v0 actually applies. A lane naming any other known bit binds
/// fine and is refused loudly at first use — a silently ignored site
/// would be a request whose adapter never applied while every sample
/// still returned.
pub const LORA_SITES_CONSUMED: u64 = LORA_SITE_Q | LORA_SITE_V;

/// The adapter FORM — the sink's arity selects it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum LoraForm {
    /// 3 args: low-rank `y += B(Ax)`.
    #[default]
    LowRank = 0,
    /// 2 args: SCALE `y = l ⊙ y` (IA3) — `a` holds the l vector, `b`
    /// null, rank/d_in zero.
    Scale = 1,
}

/// One lane's resolved `lora` sink. Ports `LoraLaneView`; the header
/// carries the field-by-field story.
#[derive(Debug, Clone, Copy)]
pub struct LoraLaneView {
    /// The A channel's committed cell, f32, `[num_layers, R, d_in]`.
    pub a: *const c_void,
    /// The B channel's committed cell, f32, `[num_layers, d_out, R]`;
    /// the LoRA scale is folded into the contents.
    pub b: *const c_void,
    /// The SITES placement bitmask — structure, not contents.
    pub sites_bits: u64,
    /// The lane's span in fire token rows.
    pub token_start: u32,
    /// Rows in the span.
    pub token_count: u32,
    /// Adapter geometry, element counts.
    pub num_layers: u32,
    /// The rank; zero on a scale lane.
    pub rank: u32,
    /// Input width; zero on a scale lane.
    pub d_in: u32,
    /// Output width.
    pub d_out: u32,
    /// Low-rank or scale.
    pub form: LoraForm,
}

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

/// The live [`LoraOps`] (retirement plan phase B), behind `bridge` for the
/// cast launch. The slab upload is `cudaMemcpyAsync` of the host pointer
/// array — device-resident because `cublasGemmGroupedBatchedEx` does not
/// consume its pointer arrays synchronously, which is the trait doc's own
/// sentence and the reason this is an upload rather than an argument.
#[cfg(feature = "bridge")]
#[derive(Debug, Clone, Copy)]
pub struct LiveLoraOps {
    stream: *mut c_void,
}

#[cfg(feature = "bridge")]
impl LiveLoraOps {
    /// Ops ordered on the fire's stream.
    #[must_use]
    pub const fn new(stream: *mut c_void) -> Self {
        Self { stream }
    }
}

#[cfg(feature = "bridge")]
impl LoraOps for LiveLoraOps {
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // seam method; recorders share it
    fn cast_fp32_to_bf16(&mut self, src: *const c_void, dst: *mut c_void, elems: usize) {
        unsafe {
            crate::launch::ffi::pie_k_quant_cast_fp32_to_bf16(src, dst, elems, self.stream);
        }
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

#[derive(Debug)]
struct Lane {
    view: LoraLaneView,
    a_bf16: *mut c_void,
    b_bf16: *mut c_void,
    xa_offset: usize,
    grouped: bool,
}

#[derive(Debug, Default)]
struct Group {
    rank: i32,
    d_in: i32,
    d_out: i32,
    members: Vec<usize>,
    nq: i32,
    nv: i32,
    m: Vec<i32>,
    mq: Vec<i32>,
    mv: Vec<i32>,
    slab_off: usize,
}

/// The fire-scoped staging — `LoraFireState`, construction half. Slice B
/// adds `apply`.
#[derive(Debug)]
pub struct LoraFireState {
    lanes: Vec<Lane>,
    groups: Vec<Group>,
    ptr_slab: *mut c_void,
    slab_stride: usize,
    grouped_enabled: bool,
}

/// `base + row * width` bf16 elements — the C++ `bf16_row`.
fn bf16_row(base: *const c_void, row: u32, width: i32) -> *const c_void {
    let off = row as usize * usize::try_from(width.max(0)).unwrap_or(0) * 2;
    // SAFETY: offset arithmetic only; the result crosses the ops seam.
    unsafe { base.cast::<u8>().add(off).cast() }
}

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
                    || i64::from(lane.token_count)
                        > i64::from(n) - i64::from(lane.token_start)
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
                    g.rank == v.rank as i32
                        && g.d_in == v.d_in as i32
                        && g.d_out == v.d_out as i32
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
                    xa_total += me.lanes[idx].view.token_count as usize
                        * me.lanes[idx].view.rank as usize;
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
                        let (mut q_act, mut q_w, mut q_y) =
                            (Vec::new(), Vec::new(), Vec::new());
                        let (mut v_act, mut v_w, mut v_y) =
                            (Vec::new(), Vec::new(), Vec::new());
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
                            let xa = unsafe {
                                rows.gate.cast::<u16>().add(lane.xa_offset)
                            };
                            staged.push(bf16_row(rows.norm_x, v.token_start, h));
                            a_run.push(a_l.cast_const().cast::<c_void>());
                            xa_run.push(xa.cast_const().cast::<c_void>());
                            if layer == 0 {
                                me.groups[gi].m.push(
                                    i32::try_from(v.token_count).unwrap_or(0),
                                );
                            }
                            if v.sites_bits & LORA_SITE_Q != 0 {
                                q_act.push(xa.cast_const().cast::<c_void>());
                                q_w.push(b_l.cast_const().cast::<c_void>());
                                q_y.push(bf16_row(rows.q.cast_const(), v.token_start, hq));
                                if layer == 0 {
                                    me.groups[gi].mq.push(
                                        i32::try_from(v.token_count).unwrap_or(0),
                                    );
                                }
                            }
                            if v.sites_bits & LORA_SITE_V != 0 {
                                v_act.push(xa.cast_const().cast::<c_void>());
                                v_w.push(b_l.cast_const().cast::<c_void>());
                                v_y.push(bf16_row(rows.v.cast_const(), v.token_start, hk));
                                if layer == 0 {
                                    me.groups[gi].mv.push(
                                        i32::try_from(v.token_count).unwrap_or(0),
                                    );
                                }
                            }
                        }
                        let mut slot = layer * me.slab_stride + me.groups[gi].slab_off;
                        for run in [
                            &staged, &a_run, &xa_run, &q_act, &q_w, &q_y, &v_act, &v_w,
                            &v_y,
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

    /// SLICE B: the fire-time launches over the staged state — ports
    /// `LoraFireState::apply` argument for argument. Three passes, in
    /// the C++'s order: solo lanes (xAᵀ into the shared scratch base,
    /// then (xAᵀ)Bᵀ accumulated β=1 into the q/v row windows per the
    /// lane's SITES bits), grouped lanes (slot arithmetic over the
    /// staged pointer slab + three grouped GEMMs), then the SCALE pass
    /// last — after every delta, so a same-site low-rank + scale
    /// composes as s ⊙ (y + B(Ax)), DoRA's order; a lone scale lane is
    /// IA3 unchanged. The LAYER is the op tag's (never `param1` — the
    /// bug the C++'s first live A/B caught).
    #[cfg(feature = "bridge")]
    #[allow(clippy::too_many_arguments, clippy::not_unsafe_ptr_arg_deref)]
    pub fn apply(
        &self,
        cublas: *mut c_void,
        layer: i32,
        qkv_in: *const c_void,
        h: i32,
        hq: i32,
        hk: i32,
        q_out: *mut c_void,
        v_out: *mut c_void,
        xa_scratch: *mut c_void,
        stream: *mut c_void,
    ) {
        use crate::launch::ffi;
        let layer_u = usize::try_from(layer).unwrap_or(0);
        for lane in &self.lanes {
            if lane.grouped {
                continue;
            }
            let v = &lane.view;
            if v.form == LoraForm::Scale {
                continue; // the scale pass below, after every delta
            }
            let t = i32::try_from(v.token_count).unwrap_or(0);
            let r = i32::try_from(v.rank).unwrap_or(0);
            let a_l = bf16_row(
                lane.a_bf16.cast_const(),
                u32::try_from(layer_u * v.rank as usize).unwrap_or(0),
                i32::try_from(v.d_in).unwrap_or(0),
            );
            let b_l = bf16_row(
                lane.b_bf16.cast_const(),
                u32::try_from(layer_u * v.d_out as usize).unwrap_or(0),
                r,
            );
            let x = bf16_row(qkv_in, v.token_start, h);
            unsafe {
                ffi::pie_k_gemm_act_x_wt_bf16(cublas, x, a_l, xa_scratch, t, r, h, 0.0);
                if v.sites_bits & LORA_SITE_Q != 0 {
                    ffi::pie_k_gemm_act_x_wt_bf16(
                        cublas,
                        xa_scratch.cast_const(),
                        b_l,
                        bf16_row(q_out.cast_const(), v.token_start, hq).cast_mut(),
                        t,
                        i32::try_from(v.d_out).unwrap_or(0),
                        r,
                        1.0,
                    );
                }
                if v.sites_bits & LORA_SITE_V != 0 {
                    ffi::pie_k_gemm_act_x_wt_bf16(
                        cublas,
                        xa_scratch.cast_const(),
                        b_l,
                        bf16_row(v_out.cast_const(), v.token_start, hk).cast_mut(),
                        t,
                        i32::try_from(v.d_out).unwrap_or(0),
                        r,
                        1.0,
                    );
                }
            }
        }
        for g in &self.groups {
            let n = g.members.len();
            // The slab was fully staged at fire setup — slot arithmetic
            // and launches, nothing else (what a captured body requires).
            let slot = unsafe {
                self.ptr_slab
                    .cast::<*const c_void>()
                    .add(layer_u * self.slab_stride + g.slab_off)
            };
            let x_ptrs = slot.cast_const();
            unsafe {
                let a_ptrs = x_ptrs.add(n);
                let xa_ptrs = x_ptrs.add(2 * n);
                ffi::pie_k_gemm_grouped_act_x_wt_bf16(
                    cublas,
                    x_ptrs,
                    a_ptrs,
                    xa_ptrs.cast::<*mut c_void>().cast_mut(),
                    g.m.as_ptr(),
                    i32::try_from(n).unwrap_or(0),
                    g.rank,
                    g.d_in,
                    0.0,
                );
                if g.nq > 0 {
                    let base = x_ptrs.add(3 * n);
                    ffi::pie_k_gemm_grouped_act_x_wt_bf16(
                        cublas,
                        base,
                        base.add(g.nq as usize),
                        base.add(2 * g.nq as usize).cast::<*mut c_void>().cast_mut(),
                        g.mq.as_ptr(),
                        g.nq,
                        g.d_out,
                        g.rank,
                        1.0,
                    );
                }
                if g.nv > 0 {
                    let base = x_ptrs.add(3 * n + 3 * g.nq as usize);
                    ffi::pie_k_gemm_grouped_act_x_wt_bf16(
                        cublas,
                        base,
                        base.add(g.nv as usize),
                        base.add(2 * g.nv as usize).cast::<*mut c_void>().cast_mut(),
                        g.mv.as_ptr(),
                        g.nv,
                        g.d_out,
                        g.rank,
                        1.0,
                    );
                }
            }
        }
        for lane in &self.lanes {
            let v = &lane.view;
            if v.form != LoraForm::Scale {
                continue;
            }
            let t = i32::try_from(v.token_count).unwrap_or(0);
            let l_l = bf16_row(
                lane.a_bf16.cast_const(),
                u32::try_from(layer_u).unwrap_or(0),
                i32::try_from(v.d_out).unwrap_or(0),
            );
            unsafe {
                if v.sites_bits & LORA_SITE_Q != 0 {
                    ffi::pie_k_quant_scale_rows_bf16(
                        bf16_row(q_out.cast_const(), v.token_start, hq).cast_mut(),
                        l_l,
                        t,
                        i32::try_from(v.d_out).unwrap_or(0),
                        stream,
                    );
                }
                if v.sites_bits & LORA_SITE_V != 0 {
                    ffi::pie_k_quant_scale_rows_bf16(
                        bf16_row(v_out.cast_const(), v.token_start, hk).cast_mut(),
                        l_l,
                        t,
                        i32::try_from(v.d_out).unwrap_or(0),
                        stream,
                    );
                }
            }
        }
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
/// `llama_like_lora_stage`; the staged state and the table it was staged
/// from go to the caller — the plan state's `lora_staged` slot in the
/// C++ — rather than being written here, because the plan state's `L`
/// parameter is exactly this type.
#[allow(clippy::too_many_arguments)]
pub fn llama_like_lora_stage<O: DeviceMemory + LoraOps>(
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
        &LoraStageRows { norm_x: qkv_in, ..*rows },
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
    Ok((if h64 == 0 { 1 } else { h64 }, Some(staged)))
}
