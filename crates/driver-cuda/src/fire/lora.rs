//! The `lora` sink's vocabulary and the fire-scoped staging — gate-lora,
//! slice A: [`LoraStageArena`], [`LoraFireState`]'s construction, and
//! [`stage_qkv_adapters`]. Same-shape lanes share one grouped-GEMM launch,
//! valid only for pairwise-disjoint token spans; overlap falls back to
//! per-lane pairs. `apply()` is slice B.

use std::ffi::c_void;

use super::sideband_arena::DeviceMemory;

// `apply` and the matmul passes live in `kernels_cuda::gemm::lora`; the
// types below are re-exported so callers keep one definition.

use kernels_cuda::gemm::lora::{Group, Lane, Staged, bf16_row};

pub use kernels_cuda::gemm::lora::{
    LORA_SITE_DOWN, LORA_SITE_GATE_UP, LORA_SITE_K, LORA_SITE_O, LORA_SITE_Q, LORA_SITE_V,
    LORA_SITES_CONSUMED, LORA_SITES_KNOWN, LoraForm, LoraLaneView,
};
use kernels_cuda::jit::abi::bf16;

/// The launch's resolved lora configuration: a borrowed view valid for the fire.
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

/// The per-fire bump arena the staging draws from: 256-aligned allocs,
/// doubling growth with a 1 MiB floor. Retire-on-grow, not free-on-grow —
/// the old block may still be read by an in-flight fire — reset per fire.
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

    /// Free every block through the seam — explicit, since there is no RAII here.
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
    /// The pointer-slab upload, host to device — must be device-resident,
    /// since `cublasGemmGroupedBatchedEx` doesn't consume it synchronously.
    fn upload_slab(&mut self, dst: *mut c_void, slots: &[*const c_void]);
}

/// The live [`LoraOps`], behind `_cuda` for the cast launch.
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

/// Implements both `DeviceMemory` and `LoraOps`: one handle for memory and casts alike.
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
        // SAFETY: the caller holds `self.stream` live across the launch.
        let fired = unsafe {
            let ctx = kernels_cuda::jit::Ctx::on(self.stream);
            kernels_cuda::quant::cast_fp32_to::<bf16>(
                &ctx,
                kernels::routine::In { ptr: src.cast::<f32>(), rows: 0, width: 0 },
                kernels::routine::Out { ptr: dst.cast::<kernels_cuda::jit::abi::bf16>(), rows: 0, width: 0 },
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

/// Why staging refused a lane table; each message is part of the surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoraStageError(pub String);

impl std::fmt::Display for LoraStageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LoraStageError {}

/// `PIE_LORA_GROUPED` as a pure function: unset, or first byte not `'0'`, arms grouping.
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

#[derive(Debug, Clone, Copy)]
/// The fire buffers a staging pass reads. Which named values these five
/// pointers are is resolved from the launch's operand join, not launch
/// position — which is false under `GuardMode::Union`.
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
/// `Lane`/`Group` live in `kernels_cuda::gemm::lora`, since the launch
/// reads every field; this type writes them and hands them over as [`Staged`].
#[derive(Debug)]
pub struct LoraFireState {
    lanes: Vec<Lane>,
    groups: Vec<Group>,
    ptr_slab: *mut c_void,
    slab_stride: usize,
    grouped_enabled: bool,
    /// Everything a captured lora body bakes, mixed into one number for the
    /// bucket key. Unused unless `feature = "abi"`, which this file isn't.
    #[cfg_attr(
        not(feature = "abi"),
        expect(dead_code, reason = "both readers are behind `abi`")
    )]
    pub(crate) capture_fingerprint: u64,
}

// `bf16_row` (shared with the launch half) computes the row stride both use.

impl LoraFireState {
    /// The constructor: validates lanes, stages bf16 casts, groups same-shape
    /// lanes, and lays out the grouped scratch/pointer slab.
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
            // Filled by `stage_qkv_adapters`, the only thing that knows the bake.
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
            // Pointer-slab layout, per layer/group: [x a xa](n) [q_act q_w q_y](nq)
            // [v_act v_w v_y](nv).
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

    /// True iff no two lanes' token spans overlap.
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
    /// Only if every lane grouped: the solo path's launch shape follows the
    /// fire's adapter set, so a different adapter set needs a different capture.
    /// `true` means "may be captured", not "may share any exec".
    #[must_use]
    pub fn union_capture_safe(&self) -> bool {
        self.grouped_enabled && self.lanes.iter().all(|l| l.grouped)
    }

    /// The one-line grouping summary `PIE_LORA_FIRE_TRACE` prints.
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

    /// What the launch half needs, as a borrow (not a copy — the vectors
    /// outlive every launch made from them). Excludes the private fields.
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

/// An empty extent is a no-op (a LoRA lane with no tokens reaches here);
/// every other decline means this file built an argument the host program
/// refused, which it cannot handle — abort.
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
/// captured lora body bakes (lane structure, adapter pointers, grouping mode,
/// arena base — a growth changes addresses and must recapture). `0` means no lora.
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
    // Carried on the state so the bucket key can't be handed it separately.
    let staged = LoraFireState {
        capture_fingerprint: h64,
        ..staged
    };
    Ok((h64, Some(staged)))
}

// ── Reading the sink: a plan says WHICH adapter, the caller says where ──

/// The `lora` sink's operand layout, read out of a prologue stage plan.
///
/// `fwd.adapter` lowers LoRA, IA3 and DoRA to one `SinkCall { name: "lora" }`:
/// three args for the low-rank form (`a`, `b`, `sites`), two for scale
/// (`l`, `sites`) — arity selects the form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoraSink {
    /// The channel the `a` (or, on a scale sink, `l`) tensor arrives on — a
    /// program-global dense index, resolved from the stage-local slot.
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
/// [`SinkRefusal::Absent`] is the ordinary case (no sink). `Malformed` is
/// distinguished because silently running without a requested adapter
/// would return an answer indistinguishable from a correct one.
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
    // The SITES arg is a `const` op, which (unlike the channel operands) does
    // reach `plan.ops`, so it can be found by walking to `lit_bits`.
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
            // A and B arrive on separate channels with nothing to pair them; a
            // mismatched pair silently produces a wrong GEMM, not a fault.
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
    /// This sink as one lane's view, given the resolved addresses and the
    /// lane's token span. The plan read above is device-free.
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

    /// Value ids are op INDICES here, per the plan's stage-local numbering.
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

    /// A prologue whose sink is the low-rank form: A on slot 0, B on slot 1.
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

    /// Two args is IA3: no rank, no input width.
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

    /// Most stages state no sink and must read as Absent, not Malformed.
    #[test]
    fn a_stage_with_no_sink_is_absent_rather_than_malformed() {
        let plan = LaunchStagePlan {
            ops: vec![LaunchOp::default()],
            ..LaunchStagePlan::default()
        };
        assert_eq!(read_lora_sink(&plan), Err(SinkRefusal::Absent));
    }

    /// A and B on separate channels with mismatched rank must be refused.
    #[test]
    fn an_a_and_b_that_disagree_are_refused() {
        let mut plan = low_rank_plan();
        plan.value_types[1] = value(&[4, 128, 16]); // rank 16, not 8
        assert!(
            matches!(read_lora_sink(&plan), Err(SinkRefusal::Malformed(_))),
            "a rank mismatch between A and B is refused"
        );
    }

    /// An unknown site is refused, not silently masked off.
    #[test]
    fn an_unknown_site_is_refused_rather_than_masked_off() {
        let mut plan = low_rank_plan();
        plan.ops[2] = const_op(1 << 20);
        assert!(matches!(
            read_lora_sink(&plan),
            Err(SinkRefusal::Malformed(_))
        ));
    }

    /// A non-channel operand cannot be an adapter tensor.
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

/// Resolve one instance's `lora` sink into a lane view — the *where*
/// [`read_lora_sink`]'s *which* leaves open. `None` covers every ordinary
/// adapter-free fire (no program, no prologue, no sink, unringed channel).
// `pub(crate)`: `instances` is the shell's own `InstanceEntry`, and
// `fire::launch` is the only caller.
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
    // By KIND, not `plans.first()`, which would be the epilogue today.
    let stage = compiled.stage_of_kind(crate::program::runtime::stage_kind::PROLOGUE)?;
    let plan = compiled.plans.get(stage)?;
    let sink = read_lora_sink(plan).ok()?;

    let session = sessions.get(&instance_id)?;
    let cursors = rings.cursors(stream).ok()?;
    // `head` (committed), not `tail` (pending): an adapter is seeded once and
    // read by the fire.
    //
    // `dense` is this instance's own numbering; indexing the rings with it
    // directly would hand the adapter somebody else's channel.
    let address = |dense: u32| -> Option<*const std::ffi::c_void> {
        let c = session.slot(dense as usize)? as usize;
        let cursor = cursors.get(c)?;
        let at = rings.cell_address(c, cursor.head).ok()?;
        (at != 0).then(|| at as *const std::ffi::c_void)
    };

    let a = address(sink.a_channel)?;
    let b = match sink.b_channel {
        // A scale sink has no B; null is the form `LoraFireState` expects.
        None => core::ptr::null(),
        Some(dense) => address(dense)?,
    };
    Some(sink.lane(a, b, token_start, token_count))
}
