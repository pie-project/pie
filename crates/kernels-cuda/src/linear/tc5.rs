//! Native datacenter-Blackwell GEMMs: one persistent tcgen05 kernel
//! (`linear/tc5.cuh`) specialized per kind and tile at JIT time, fed by
//! TMA tensor maps encoded here. `y[m][n] = alpha * sum_k act[m][k] * w[n][k]`
//! with both operands K-major and `y` bf16, `m x n` with row stride `n`.
//!
//! Block scales are 512-byte atoms of 128 rows x 4 scale columns (see
//! [`atom_offset`]), atoms of one 128-row block consecutive along K, rows
//! padded to whole blocks with finite (zero) bytes. A scale column covers
//! `gran` K: 16 for NVFP4's ue4m3, 32 or 128 for ue8m0.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use cudarc::driver::sys as dr;
use cudarc::runtime::sys as rt;

use crate::error::Error;
use crate::jit::{ArgValue, Ctx, Fire, Launch, refuse, symbol};

const OP: &str = "linear.tc5";
const FILE: &str = "linear/tc5.cuh";
const THREADS: u32 = 224;
const SMEM_CAP: u32 = 232_448;
/// The routing entries a grouped launch holds in shared memory.
const MAX_ROUTES: u32 = 1028;

/// An operand's element type as it lies in global memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Elem {
    Bf16,
    /// fp8 e4m3, one a byte.
    E4m3,
    /// fp4 e2m1, two a byte, the even element in the low nibble.
    E2m1,
}

/// A block-scale tensor in the atom layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scales {
    pub ptr: u64,
    /// The K a scale covers: 16 (ue4m3), 32 or 128 (ue8m0).
    pub gran: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Operand {
    pub ptr: u64,
    pub elem: Elem,
    pub scales: Option<Scales>,
}

impl Operand {
    #[must_use]
    pub const fn plain(ptr: u64, elem: Elem) -> Self {
        Self {
            ptr,
            elem,
            scales: None,
        }
    }

    #[must_use]
    pub const fn scaled(ptr: u64, elem: Elem, scales: u64, gran: u32) -> Self {
        Self {
            ptr,
            elem,
            scales: Some(Scales { ptr: scales, gran }),
        }
    }
}

/// The byte of scale `(row, col)` in an atom-layout tensor whose rows carry
/// `cols` scale columns.
#[must_use]
pub const fn atom_offset(row: usize, col: usize, cols: usize) -> usize {
    let atoms_k = cols.div_ceil(4);
    ((row / 128) * atoms_k + col / 4) * 512 + (row % 32) * 16 + ((row / 32) % 4) * 4 + col % 4
}

/// The bytes an atom-layout scale tensor for `rows x k` at `gran` takes.
#[must_use]
pub const fn atom_bytes(rows: usize, k: usize, gran: usize) -> usize {
    rows.div_ceil(128) * k.div_ceil(gran).div_ceil(4) * 512
}

/// Whether this device and NVRTC run the tcgen05 kernels.
#[must_use]
pub fn supported() -> bool {
    crate::jit::cache::features().tcgen05
}

/// bf16 x bf16 -> bf16, f32 accumulation.
pub fn gemm_bf16(ctx: &Ctx, act: u64, w: u64, y: u64, m: u32, n: u32, k: u32) -> Result<(), Error> {
    gemm(
        ctx,
        Operand::plain(act, Elem::Bf16),
        Operand::plain(w, Elem::Bf16),
        y,
        [m, n, k],
        1.0,
    )
}

/// fp8 e4m3 W8A8 with DeepSeek's blocks: one ue8m0 scale per 1x128 of the
/// activation and per 128x128 of the weight, the weight's spread to every
/// row of its block (gran 128 on both).
#[allow(clippy::too_many_arguments)]
pub fn gemm_fp8_block(
    ctx: &Ctx,
    act: u64,
    act_scales: u64,
    w: u64,
    w_scales: u64,
    y: u64,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), Error> {
    gemm(
        ctx,
        Operand::scaled(act, Elem::E4m3, act_scales, 128),
        Operand::scaled(w, Elem::E4m3, w_scales, 128),
        y,
        [m, n, k],
        1.0,
    )
}

/// NVFP4 W4A4: e2m1 with a ue4m3 scale a 16, `alpha` the product of the
/// two global scales.
#[allow(clippy::too_many_arguments)]
pub fn gemm_nvfp4(
    ctx: &Ctx,
    act: u64,
    act_scales: u64,
    w: u64,
    w_scales: u64,
    y: u64,
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
) -> Result<(), Error> {
    gemm(
        ctx,
        Operand::scaled(act, Elem::E2m1, act_scales, 16),
        Operand::scaled(w, Elem::E2m1, w_scales, 16),
        y,
        [m, n, k],
        alpha,
    )
}

/// W4A8: MXFP4 weights (e2m1, ue8m0 a 32) against e4m3 activations with a
/// ue8m0 scale a `act_gran` (32 or 128).
#[allow(clippy::too_many_arguments)]
pub fn gemm_w4a8(
    ctx: &Ctx,
    act: u64,
    act_scales: u64,
    act_gran: u32,
    w: u64,
    w_scales: u64,
    y: u64,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), Error> {
    gemm(
        ctx,
        Operand::scaled(act, Elem::E4m3, act_scales, act_gran),
        Operand::scaled(w, Elem::E2m1, w_scales, 32),
        y,
        [m, n, k],
        1.0,
    )
}

/// The kernel's operand kind, and each side's format code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Bf16,
    Fp8,
    Mx,
    Nvf4,
}

impl Kind {
    const fn code(self) -> u32 {
        match self {
            Self::Bf16 => 0,
            Self::Fp8 => 1,
            Self::Mx => 2,
            Self::Nvf4 => 3,
        }
    }

    const fn stage_k(self) -> u32 {
        match self {
            Self::Bf16 => 64,
            Self::Nvf4 => 256,
            Self::Fp8 | Self::Mx => 128,
        }
    }

    fn of(act: Operand, w: Operand) -> Result<Self, String> {
        let gran = |o: Operand| o.scales.map(|s| s.gran);
        match (act.elem, w.elem, gran(act), gran(w)) {
            (Elem::Bf16, Elem::Bf16, None, None) => Ok(Self::Bf16),
            (Elem::E4m3, Elem::E4m3, None, None) => Ok(Self::Fp8),
            (Elem::E2m1, Elem::E2m1, Some(16), Some(16)) => Ok(Self::Nvf4),
            (Elem::E4m3 | Elem::E2m1, Elem::E4m3 | Elem::E2m1, Some(a), Some(b))
                if matches!(a, 32 | 128) && matches!(b, 32 | 128) =>
            {
                Ok(Self::Mx)
            }
            other => Err(format!(
                "no tcgen05 kind multiplies {other:?} (act elem, w elem, act gran, w gran): \
                 bf16 x bf16, e4m3 x e4m3, nvfp4 (gran 16) or e4m3/e2m1 with ue8m0 at 32/128"
            )),
        }
    }

    /// The instruction descriptor's format code for one side.
    const fn format(self, elem: Elem) -> u32 {
        match (self, elem) {
            (Self::Bf16, _) | (Self::Nvf4, _) => 1,
            (_, Elem::E2m1) => 5,
            _ => 0,
        }
    }
}

const fn atoms_k(kind: Kind, gran: u32) -> u32 {
    match kind {
        Kind::Bf16 | Kind::Fp8 => 0,
        Kind::Mx | Kind::Nvf4 => {
            let atoms = kind.stage_k() / gran / 4;
            if atoms == 0 { 1 } else { atoms }
        }
    }
}

/// The dynamic shared memory the kernel lays out, mirrored from it, plus
/// the 1024 bytes it may spend aligning the ring.
const fn smem_bytes(
    kind: Kind,
    gran_p: u32,
    gran_q: u32,
    plan: Plan,
    pt: u32,
    p_blocks: u32,
) -> u32 {
    let Plan {
        swap,
        pair,
        bn,
        stages,
        ..
    } = plan;
    let stage = 128 * 128 * pt
        + bn / if pair { 2 } else { 1 } * 128
        + atoms_k(kind, gran_p) * 512 * pt * p_blocks
        + atoms_k(kind, gran_q) * bn.div_ceil(128) * 512;
    let out = (stages * stage).div_ceil(1024) * 1024;
    let bar = out + if swap { 1024 } else { 4 * 2 * 2048 };
    // The routing a grouped launch holds is reserved for every launch.
    bar + (2 * stages + 16) * 8 + 8 + MAX_ROUTES * 8 + 1024
}

/// One shape's specialization: which side is the MMA's M, whether two CTAs
/// share a 256-row MMA, the Q tile, and the ring depth that fits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub swap: bool,
    /// A cluster of two CTAs runs each 256-row tile (`cta_group::2`).
    pub pair: bool,
    pub bn: u32,
    pub stages: u32,
    /// K splits of a tile, summed in f32; 0 picks enough to fill the
    /// SMs.
    pub splits: u32,
}

impl Plan {
    /// Rows of 64 or fewer put the weight on the 128-row side, the rows on
    /// a narrow Q tile. Otherwise a pair of CTAs runs 256 x 256 tiles, each
    /// streaming half of both operands' bytes a lone CTA would, unless lone
    /// 128 x 128 tiles finish sooner for filling the `sms` better (a lone
    /// SM runs at about 0.55 a paired one's rate). Either splits K when its
    /// tiles leave SMs idle; see [`Plan::splits_for`].
    #[must_use]
    pub fn choose(m: u32, n: u32, k: u32, sms: u32) -> Self {
        if m <= 64 {
            return Self {
                swap: true,
                pair: false,
                bn: m.next_multiple_of(32).max(32),
                stages: 0,
                splits: 0,
            };
        }
        // Time in units of a lone 128 x 128 x 128 step on one SM.
        let steps = k.div_ceil(128);
        let time = |rows: u32, ctas: u32, rate: f64| {
            let tiles = m.div_ceil(rows) * n.div_ceil(rows);
            let splits = Self::splits_for(tiles, steps, sms / ctas);
            let waves = (tiles * splits).div_ceil((sms / ctas).max(1));
            let area = f64::from(rows * rows) / f64::from(128 * 128 * ctas);
            let penalty = if splits > 1 { 1.15 } else { 1.0 };
            f64::from(waves) * area * f64::from(steps.div_ceil(splits)) / rate * penalty
        };
        let pair = time(256, 2, 1.8) <= time(128, 1, 1.0);
        Self {
            swap: false,
            pair,
            bn: if pair { 256 } else { 128 },
            stages: 0,
            splits: 0,
        }
    }

    /// K splits for `tiles` tiles of `steps` K steps on `slots` CTAs (or
    /// pairs): enough to fill them once three or more fit, each keeping four
    /// steps to pipeline. Fewer do not repay the f32 sum's zeroing, its
    /// reductions (one a split per output) and its rounding pass (two
    /// splits of a long-K mid-M tile measured slower for fp8 and NVFP4),
    /// and more than one wave of splits only multiplies the reductions.
    #[must_use]
    pub fn splits_for(tiles: u32, steps: u32, slots: u32) -> u32 {
        let fit = slots / tiles.max(1);
        if fit < 3 {
            1
        } else {
            fit.min((steps / 4).max(1))
        }
    }

    fn staged(mut self, kind: Kind, gran_p: u32, gran_q: u32, pt: u32, p_blocks: u32) -> Self {
        let mut stages = 8;
        while stages > 2
            && smem_bytes(kind, gran_p, gran_q, Self { stages, ..self }, pt, p_blocks) > SMEM_CAP
        {
            stages -= 1;
        }
        self.stages = stages;
        self
    }
}

/// `y = alpha * act . w^T` for `[m, n, k]`, choosing the plan.
pub fn gemm(
    ctx: &Ctx,
    act: Operand,
    w: Operand,
    y: u64,
    [m, n, k]: [u32; 3],
    alpha: f32,
) -> Result<(), Error> {
    let sms = ctx.multiprocessors().unwrap_or(148);
    gemm_with(ctx, act, w, y, [m, n, k], alpha, Plan::choose(m, n, k, sms))
}

/// As [`gemm`], under a stated plan (its `stages` is refit to the kind).
#[allow(clippy::too_many_arguments)]
pub fn gemm_with(
    ctx: &Ctx,
    act: Operand,
    w: Operand,
    y: u64,
    [m, n, k]: [u32; 3],
    alpha: f32,
    plan: Plan,
) -> Result<(), Error> {
    gemm_epi_with(ctx, act, w, y, [m, n, k], alpha, Epilogue::STORE, plan)
}

/// A gate/up pair's activation, the epilogue reads off the product.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Glu {
    /// `silu(g) * u` (`mlp_swiglu`).
    Swiglu,
    /// `silu(min(g, limit)) * clamp(u, ±limit)` (`mlp_swiglu_clamp`).
    SwigluClamp { limit: f32 },
    /// `(clamp(u, ±limit) + 1) * g' * sigmoid(alpha * g')`, `g' = min(g,
    /// limit)` (`mlp_swiglu_clamp_alpha`).
    SwigluClampAlpha { limit: f32, alpha: f32 },
}

/// Where the epilogue lands its `rows x n` result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Out {
    /// bf16 at `y`, row stride `n`.
    Bf16,
    /// The next GEMM's fp8 operand: e4m3 `[rows][n]` at `data`, and a ue8m0
    /// scale a 1x128 block (`2^ceil(log2(max(amax, 1e-4) / 448))`, values
    /// clamped to ±448) in the atom layout at `scales`, `n / 128` columns.
    /// Needs a GLU and `n % 128 == 0`.
    Fp8Block { data: u64, scales: u64 },
    /// The next GEMM's NVFP4 operand: e2m1 `[rows][n / 2]` at `data`, and a
    /// ue4m3 scale a 16 (`e4m3(gscale * amax / 6)`, vLLM's and modelopt's)
    /// in the atom layout at `scales`, `n / 16` columns. `gscale` is the
    /// global scale, `1 / input_scale`; the next GEMM's alpha carries
    /// `input_scale`. Needs a GLU and `n % 128 == 0`.
    Nvfp4 { data: u64, scales: u64, gscale: f32 },
}

/// What the epilogue does past `alpha * act . w^T`.
///
/// With a GLU the weight holds the gate's `n` rows, then the up's `n`
/// (each expert's so, in a bank), and the output is `rows x n`. Output rows
/// index scales as the activation would: a masked group's pad each expert to
/// whole 128-row blocks ([`masked_atom_offset`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Epilogue {
    pub glu: Option<Glu>,
    pub out: Out,
    /// A device `f32` a output row multiplying it last (the routed weight
    /// in a down projection's epilogue), or 0.
    pub row_scale: u64,
}

impl Epilogue {
    pub const STORE: Self = Self {
        glu: None,
        out: Out::Bf16,
        row_scale: 0,
    };

    const fn glu_code(&self) -> u32 {
        match self.glu {
            None => 0,
            Some(Glu::Swiglu) => 1,
            Some(Glu::SwigluClamp { .. }) => 2,
            Some(Glu::SwigluClampAlpha { .. }) => 3,
        }
    }

    const fn out_code(&self) -> u32 {
        match self.out {
            Out::Bf16 => 0,
            Out::Fp8Block { .. } => 1,
            Out::Nvfp4 { .. } => 2,
        }
    }

    /// Fits a plan to the epilogue: an unswapped GLU tile is gate 128 | up
    /// 128, a swapped one at most 128 wide (more Q tiles past it), and
    /// nothing splits K past a GLU.
    fn fit(&self, mut plan: Plan) -> Result<Plan, Error> {
        if self.glu.is_none() {
            if self.out != Out::Bf16 {
                return Err(refuse(OP, "a quantized store follows a GLU"));
            }
            return Ok(plan);
        }
        if plan.swap {
            if plan.bn > 128 {
                // Two accumulators of 128 fill a stage's TMEM: wider rows take
                // more Q tiles.
                plan.bn = 128;
            }
        } else {
            plan.bn = 256;
        }
        plan.splits = 1;
        Ok(plan)
    }

    fn args(&self, n: u32, pad: u32) -> EpiArgs {
        let (limit, galpha) = match self.glu {
            Some(Glu::SwigluClamp { limit }) => (limit, 1.0),
            Some(Glu::SwigluClampAlpha { limit, alpha }) => (limit, alpha),
            _ => (0.0, 1.0),
        };
        let (q, qs, gscale) = match self.out {
            Out::Bf16 => (0, 0, 1.0),
            Out::Fp8Block { data, scales } => (data, scales, 1.0),
            Out::Nvfp4 {
                data,
                scales,
                gscale,
            } => (data, scales, gscale),
        };
        EpiArgs {
            inter: i32::try_from(n).unwrap_or(i32::MAX),
            limit,
            galpha,
            q,
            qs,
            gscale,
            row_scale: self.row_scale,
            pad: i32::try_from(pad).unwrap_or(i32::MAX),
        }
    }
}

/// The kernel's `Epi`, field for field.
#[repr(C)]
#[derive(Clone, Copy)]
struct EpiArgs {
    inter: i32,
    limit: f32,
    galpha: f32,
    q: u64,
    qs: u64,
    gscale: f32,
    row_scale: u64,
    pad: i32,
}

/// `y = epilogue(alpha * act . w^T)` for `[m, n, k]`, `n` the output's
/// columns, choosing the plan.
#[allow(clippy::too_many_arguments)]
pub fn gemm_epi(
    ctx: &Ctx,
    act: Operand,
    w: Operand,
    y: u64,
    [m, n, k]: [u32; 3],
    alpha: f32,
    epi: Epilogue,
) -> Result<(), Error> {
    let sms = ctx.multiprocessors().unwrap_or(148);
    let plan = Plan::choose(m, if epi.glu.is_some() { 2 * n } else { n }, k, sms);
    gemm_epi_with(ctx, act, w, y, [m, n, k], alpha, epi, plan)
}

/// As [`gemm_epi`], under a stated plan.
#[allow(clippy::too_many_arguments)]
pub fn gemm_epi_with(
    ctx: &Ctx,
    act: Operand,
    w: Operand,
    y: u64,
    [m, n, k]: [u32; 3],
    alpha: f32,
    epi: Epilogue,
    plan: Plan,
) -> Result<(), Error> {
    if m == 0 || n == 0 {
        return Ok(());
    }
    let plan = epi.fit(plan)?;
    let kind = admit(act, w, n, k, plan)?;
    let glu = epi.glu.is_some();
    if glu && !n.is_multiple_of(128) {
        return Err(refuse(OP, "a GLU's gate rows are whole 128-row blocks"));
    }
    let w_rows = if glu { 2 * n } else { n };
    let (p, q) = if plan.swap { (w, act) } else { (act, w) };
    let (rows_p, rows_q) = if plan.swap { (n, m) } else { (m, n) };
    let (map_p, map_q) = if plan.swap { (w_rows, m) } else { (m, w_rows) };
    let ctas = if plan.pair { 2 } else { 1 };
    let q_step = if glu && !plan.swap { 128 } else { plan.bn };
    launch(
        ctx,
        Job {
            kind,
            plan,
            p,
            q,
            rows: [rows_p, rows_q],
            map_rows: [map_p, map_q],
            scale_rows: [map_p, map_q],
            y_rows: m,
            tiles: rows_p.div_ceil(128 * ctas) * rows_q.div_ceil(q_step),
            group: Group::NONE,
            epi,
            pad: 0,
        },
        y,
        [n, k],
        alpha,
    )
}

/// How a grouped GEMM's activation rows meet a bank of `experts` weights
/// `[experts][n][k]` (each expert's scales atoms of its own whole row blocks,
/// so `n` is a multiple of 128 for a scaled kind).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Groups {
    /// The prefill layout: rows sorted by expert, expert e's rows at
    /// `[offsets[e], offsets[e + 1])` of a `rows`-row activation (and of
    /// `y`, `rows x n`). `offsets` is a device `i32[experts + 1]`, every
    /// entry a multiple of [`GROUP_ALIGN`]; the pad rows between experts
    /// are computed and land, unread. Rows past `offsets[experts]` are not
    /// touched.
    Contiguous { offsets: u64, rows: u32 },
    /// The decode layout: expert e's rows at `[e * m_max, e * m_max +
    /// counts[e])` of an `experts * m_max`-row activation and of `y`.
    /// `counts` is a device `i32[experts]`, each at most `m_max <= 256`.
    /// The activation's scale atoms pad each expert to whole row blocks:
    /// see [`masked_atom_offset`]. Rows past a count are not touched.
    Masked { counts: u64, m_max: u32 },
}

/// Where a contiguous group's rows start: a 64-row half of a scale row
/// block, so a tile's warps never straddle experts and its scales start at
/// a column the MMA takes (it faults on odd 32-row quarters of the A side's
/// scales). 128 served before and still does.
pub const GROUP_ALIGN: u32 = 64;

/// A grouped GEMM: each routed row against its expert's weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grouped {
    pub experts: u32,
    pub groups: Groups,
    /// A device `f32[experts]` scaling each expert's product besides
    /// `alpha` (NVFP4's per-expert global scale), or 0.
    pub alpha_e: u64,
}

/// The byte of scale `(row, col)` of expert `e`'s activation in the masked
/// layout, rows carrying `cols` scale columns.
#[must_use]
pub const fn masked_atom_offset(
    e: usize,
    row: usize,
    col: usize,
    m_max: usize,
    cols: usize,
) -> usize {
    atom_offset(e * m_max.div_ceil(128) * 128 + row, col, cols)
}

/// The bytes a masked activation's scale atoms take.
#[must_use]
pub const fn masked_atom_bytes(experts: usize, m_max: usize, k: usize, gran: usize) -> usize {
    atom_bytes(experts * m_max.div_ceil(128) * 128, k, gran)
}

impl Plan {
    /// A grouped GEMM's plan. Contiguous groups run unswapped on lone CTAs
    /// (a pair's 256-row tile would cross experts at 128-row alignment);
    /// masked groups swap, the Q tile the rows an expert may hold. `routed`
    /// is the rows the routing is expected to fill in all (a decode's
    /// tokens x top-k), which sizes the K splits without reading counts.
    #[must_use]
    pub fn grouped(g: &Grouped, n: u32, k: u32, routed: u32, sms: u32) -> Self {
        let steps = k.div_ceil(128);
        match g.groups {
            Groups::Contiguous { .. } => {
                let row_tiles = routed.div_ceil(128) + g.experts.min(routed);
                let tiles = |bn: u32| row_tiles * n.div_ceil(bn);
                let bn = if n.is_multiple_of(256) && tiles(256) >= sms {
                    256
                } else {
                    128
                };
                Self {
                    swap: false,
                    pair: false,
                    bn,
                    stages: 0,
                    splits: Self::splits_for(tiles(bn), steps, sms),
                }
            }
            Groups::Masked { m_max, .. } => {
                let bn = m_max.next_multiple_of(32).clamp(32, 256);
                let busy = g.experts.min(routed.max(1)) * m_max.div_ceil(bn) * n.div_ceil(128);
                // The masked sum's zeroing and rounding passes walk every
                // expert's rows: they repay only a launch that leaves most
                // SMs idle.
                let splits = if busy * 4 <= sms {
                    Self::splits_for(busy, steps, sms)
                } else {
                    1
                };
                Self {
                    swap: true,
                    pair: false,
                    bn,
                    stages: 0,
                    splits,
                }
            }
        }
    }
}

/// `y = alpha * alpha_e[e] * act . w[e]^T`, each row against its expert
/// `e`'s weight, for a bank of `[experts][n][k]`. `routed` sizes the plan;
/// see [`Plan::grouped`].
#[allow(clippy::too_many_arguments)]
pub fn grouped_gemm(
    ctx: &Ctx,
    act: Operand,
    w: Operand,
    y: u64,
    g: Grouped,
    [n, k]: [u32; 2],
    alpha: f32,
    routed: u32,
) -> Result<(), Error> {
    grouped_gemm_epi(ctx, act, w, y, g, [n, k], alpha, Epilogue::STORE, routed)
}

/// As [`grouped_gemm_epi`] with no epilogue past the product.
#[allow(clippy::too_many_arguments)]
pub fn grouped_gemm_with(
    ctx: &Ctx,
    act: Operand,
    w: Operand,
    y: u64,
    g: Grouped,
    [n, k]: [u32; 2],
    alpha: f32,
    plan: Plan,
) -> Result<(), Error> {
    grouped_gemm_epi_with(ctx, act, w, y, g, [n, k], alpha, Epilogue::STORE, plan)
}

/// [`grouped_gemm`] under an epilogue: `n` the output's columns, a GLU's
/// bank `[experts][2n][k]` (each expert's gate rows, then its up rows).
#[allow(clippy::too_many_arguments)]
pub fn grouped_gemm_epi(
    ctx: &Ctx,
    act: Operand,
    w: Operand,
    y: u64,
    g: Grouped,
    [n, k]: [u32; 2],
    alpha: f32,
    epi: Epilogue,
    routed: u32,
) -> Result<(), Error> {
    let sms = ctx.multiprocessors().unwrap_or(148);
    let plan = Plan::grouped(&g, n, k, routed, sms);
    grouped_gemm_epi_with(ctx, act, w, y, g, [n, k], alpha, epi, plan)
}

/// As [`grouped_gemm_epi`], under a stated plan: unswapped and lone for a
/// contiguous group, swapped and lone for a masked one.
#[allow(clippy::too_many_arguments)]
pub fn grouped_gemm_epi_with(
    ctx: &Ctx,
    act: Operand,
    w: Operand,
    y: u64,
    g: Grouped,
    [n, k]: [u32; 2],
    alpha: f32,
    epi: Epilogue,
    plan: Plan,
) -> Result<(), Error> {
    if g.experts == 0 || n == 0 {
        return Ok(());
    }
    let plan = epi.fit(plan)?;
    let glu = epi.glu.is_some();
    if glu && !n.is_multiple_of(128) {
        return Err(refuse(OP, "a GLU's gate rows are whole 128-row blocks"));
    }
    let n_per = if glu { 2 * n } else { n };
    if g.experts >= MAX_ROUTES {
        return Err(refuse(
            OP,
            format!(
                "{} experts; a launch routes fewer than {MAX_ROUTES}",
                g.experts
            ),
        ));
    }
    let kind = admit(act, w, n, k, plan)?;
    if act.scales.is_some() && !n.is_multiple_of(128) {
        return Err(refuse(
            OP,
            "a scaled bank's experts are whole 128-row blocks",
        ));
    }
    let bank = g.experts * n_per;
    let q_step = if glu { 128 } else { plan.bn };
    let job = match g.groups {
        Groups::Contiguous { offsets, rows } => {
            if plan.swap || plan.pair {
                return Err(refuse(OP, "a contiguous group runs unswapped on lone CTAs"));
            }
            if rows == 0 {
                return Ok(());
            }
            Job {
                kind,
                plan,
                p: act,
                q: w,
                rows: [rows, bank],
                map_rows: [rows, bank],
                scale_rows: [rows, bank],
                y_rows: rows,
                tiles: rows.div_ceil(128) * n.div_ceil(q_step),
                group: Group {
                    mode: 1,
                    ptr: offsets,
                    experts: g.experts,
                    n: n_per,
                    m_max: 0,
                    alpha_e: g.alpha_e,
                },
                epi,
                pad: 0,
            }
        }
        Groups::Masked { counts, m_max } => {
            if !plan.swap || plan.pair || m_max > 256 {
                return Err(refuse(
                    OP,
                    "a masked group runs swapped on lone CTAs, m_max <= 256",
                ));
            }
            if m_max == 0 {
                return Ok(());
            }
            Job {
                kind,
                plan,
                p: w,
                q: act,
                rows: [bank, g.experts * m_max],
                map_rows: [bank, g.experts * m_max],
                scale_rows: [bank, g.experts * m_max.div_ceil(128) * 128],
                y_rows: g.experts * m_max,
                tiles: g.experts * n.div_ceil(128) * m_max.div_ceil(plan.bn),
                group: Group {
                    mode: 2,
                    ptr: counts,
                    experts: g.experts,
                    n: n_per,
                    m_max,
                    alpha_e: g.alpha_e,
                },
                epi,
                pad: m_max.div_ceil(128) * 128,
            }
        }
    };
    launch(ctx, job, y, [n, k], alpha)
}

/// The kind of `act x w`, once the device, shape and plan admit it.
fn admit(act: Operand, w: Operand, n: u32, k: u32, plan: Plan) -> Result<Kind, Error> {
    if k == 0 {
        return Err(refuse(OP, "an empty contraction"));
    }
    if !supported() {
        return Err(refuse(
            OP,
            "this device has no tcgen05 (sm_100a/sm_103a), or NVRTC is older than 12.9",
        ));
    }
    let kind = Kind::of(act, w).map_err(|why| refuse(OP, why))?;
    if !n.is_multiple_of(8) {
        return Err(refuse(OP, format!("n = {n} is not a whole number of 8s")));
    }
    // Whole 32-row quarters: a scaled Q tile narrower than its scale atom
    // starts its scale columns at its own quarter.
    if !matches!(plan.bn, 32 | 64 | 96 | 128 | 160 | 192 | 224 | 256) {
        return Err(refuse(OP, format!("a Q tile of {} rows", plan.bn)));
    }
    if plan.pair && (plan.swap || plan.bn < 64) {
        return Err(refuse(
            OP,
            "a CTA pair splits an unswapped Q tile of 64 or more",
        ));
    }
    Ok(kind)
}

/// The kernel's grouping arguments; mode 0 is none.
#[derive(Clone, Copy)]
struct Group {
    mode: u32,
    ptr: u64,
    experts: u32,
    n: u32,
    m_max: u32,
    alpha_e: u64,
}

impl Group {
    const NONE: Self = Self {
        mode: 0,
        ptr: 0,
        experts: 0,
        n: 0,
        m_max: 0,
        alpha_e: 0,
    };
}

/// One launch: the MMA's sides, their tensors' rows and their scales' rows,
/// the output's rows (`y_rows x n`, or the split sum's), and the tiles the
/// grid and the splits are sized by.
struct Job {
    kind: Kind,
    plan: Plan,
    p: Operand,
    q: Operand,
    /// The kernel's extents: rows of P and Q it tiles (a GLU's output
    /// columns on the weight side).
    rows: [u32; 2],
    /// The P and Q tensors' rows, and their scale tensors'.
    map_rows: [u32; 2],
    scale_rows: [u32; 2],
    y_rows: u32,
    tiles: u32,
    group: Group,
    epi: Epilogue,
    /// A masked group's scale rows an expert.
    pad: u32,
}

fn launch(ctx: &Ctx, job: Job, y: u64, [n, k]: [u32; 2], alpha: f32) -> Result<(), Error> {
    let Job {
        kind,
        p,
        q,
        group,
        epi,
        ..
    } = job;
    let glu = epi.glu.is_some();
    // A GLU swapped reads two weight tiles a stage; unswapped, its Q tile
    // comes in two 128-row halves.
    let pt = if glu && job.plan.swap { 2 } else { 1 };
    let halves = glu && !job.plan.swap;
    let gran = |o: Operand| o.scales.map_or(128, |s| s.gran);
    // A contiguous tile starting mid-block reads two row blocks of scales.
    let p_blocks = if group.mode == 1 { 2 } else { 1 };
    let plan = job.plan.staged(kind, gran(p), gran(q), pt, p_blocks);
    let smem = smem_bytes(kind, gran(p), gran(q), plan, pt, p_blocks);
    let ctas = if plan.pair { 2 } else { 1 };
    let [rows_p, rows_q] = job.rows;

    let map_p = tensor_map(kind, p, job.map_rows[0], k, 128)?;
    let q_box = if halves { 128 } else { plan.bn / ctas };
    let map_q = tensor_map(kind, q, job.map_rows[1], k, q_box)?;
    let map_y = if plan.swap || epi.out != Out::Bf16 {
        [0; 16]
    } else {
        output_map(y, job.y_rows, n)?
    };
    let atoms = |o: Operand, rows: u32, row_blocks: u32| match o.scales {
        Some(s) => scale_map(s, rows, k, atoms_k(kind, s.gran), row_blocks),
        None => Ok([0; 16]),
    };
    let map_sfp = atoms(p, job.scale_rows[0], p_blocks)?;
    let q_blocks = if halves { 1 } else { plan.bn.div_ceil(128) };
    let map_sfq = atoms(q, job.scale_rows[1], q_blocks)?;
    let epi_args = epi.args(n, job.pad);
    let sms = ctx.multiprocessors().unwrap_or(148);
    // Too few tiles to fill the SMs split K, each split keeping four K steps
    // or more to pipeline; the splits sum in f32 and one pass rounds.
    let steps = k.div_ceil(kind.stage_k());
    let splits = match plan.splits {
        0 => Plan::splits_for(job.tiles, steps, sms / ctas),
        s => s.clamp(1, steps),
    };
    let splits = steps.div_ceil(steps.div_ceil(splits));
    let int = |v: u32| i32::try_from(v).map_err(|_| refuse(OP, format!("{v} overflows an int")));
    let routed = group.mode != 0;
    let part = if splits > 1 {
        let bytes = job.y_rows as usize * n as usize * 4;
        let part = ctx.scratch_stream(OP, SPLIT_PARTIAL, bytes)?;
        zero_once(ctx, part, bytes)?;
        part as u64
    } else {
        0
    };
    let units = job.tiles * splits;
    let grid = ctas * units.min(sms / ctas).max(1);
    let entry = symbol(&format!(
        "::pie::linear::tc5_gemm<{}, {}, {}, {}, {}, {}, {}, {}, {ctas}, {}, {}, {}>",
        kind.code(),
        kind.format(p.elem),
        kind.format(q.elem),
        gran(p),
        gran(q),
        plan.bn,
        plan.stages,
        u32::from(plan.swap),
        group.mode,
        epi.glu_code(),
        epi.out_code(),
    ));
    let map = |m: &[u64; 16]| ArgValue::Bytes {
        ptr: m.as_ptr().cast(),
        len: 128,
    };
    let values = [
        map(&map_p),
        map(&map_q),
        map(&map_y),
        map(&map_sfp),
        map(&map_sfq),
        ArgValue::Ptr(y),
        ArgValue::I32(int(rows_p)?),
        ArgValue::I32(int(rows_q)?),
        ArgValue::I32(int(k)?),
        ArgValue::I32(int(n)?),
        ArgValue::F32(alpha),
        ArgValue::Ptr(part),
        ArgValue::I32(int(splits)?),
        ArgValue::Ptr(group.ptr),
        ArgValue::I32(int(group.experts)?),
        ArgValue::I32(int(group.n)?),
        ArgValue::I32(int(group.m_max)?),
        ArgValue::Ptr(group.alpha_e),
        ArgValue::Bytes {
            ptr: (&raw const epi_args).cast(),
            len: core::mem::size_of::<EpiArgs>(),
        },
    ];
    ctx.fire(
        OP,
        Fire::at(FILE, entry).apply(
            Launch::grid([grid, 1, 1], [THREADS, 1, 1])
                .smem(smem)
                .cluster(ctas),
        ),
        &values,
    )?;
    if splits == 1 {
        return Ok(());
    }
    if routed {
        return fire_routed(
            ctx,
            "::pie::linear::tc5_finalize_routed",
            part,
            y,
            group,
            job.y_rows,
            n,
        );
    }
    let quads = job.y_rows * n / 4;
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::linear::tc5_finalize").apply(Launch::flat(quads, 256)),
        &[
            ArgValue::Ptr(part),
            ArgValue::Ptr(y),
            ArgValue::I32(int(quads)?),
        ],
    )
}

/// A pass over a grouped GEMM's routed rows of `rows`, a block a row.
fn fire_routed(
    ctx: &Ctx,
    entry: &'static str,
    part: u64,
    y: u64,
    group: Group,
    rows: u32,
    n: u32,
) -> Result<(), Error> {
    let int = |v: u32| i32::try_from(v).map_err(|_| refuse(OP, format!("{v} overflows an int")));
    ctx.fire(
        OP,
        Fire::at(FILE, entry).apply(Launch::per_row(rows, 128)),
        &[
            ArgValue::Ptr(part),
            ArgValue::Ptr(y),
            ArgValue::Ptr(group.ptr),
            ArgValue::I32(int(group.experts)?),
            ArgValue::I32(int(group.m_max)?),
            ArgValue::I32(int(n)?),
        ],
    )
}

/// The f32 split-K sum, one a stream: the finalize after the GEMM on the
/// same stream consumes it.
const SPLIT_PARTIAL: &str = "linear.tc5.split_partial";

/// The bytes from the start of each split-sum slab known to be zero.
fn zeroed() -> std::sync::MutexGuard<'static, HashMap<usize, usize>> {
    static ZEROED: OnceLock<Mutex<HashMap<usize, usize>>> = OnceLock::new();
    ZEROED
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Slabs were freed: an address may come back holding anything.
pub(crate) fn forget_zeroed() {
    zeroed().clear();
}

/// Keeps the split sum's slab zero past `bytes`. Every finalize zeroes what
/// it rounds, so a slab is cleared once, when first seen or first asked for
/// more than before, and never again a launch.
fn zero_once(ctx: &Ctx, part: *mut core::ffi::c_void, bytes: usize) -> Result<(), Error> {
    let mut zeroed = zeroed();
    let have = zeroed.entry(part.addr()).or_insert(0);
    if bytes <= *have {
        return Ok(());
    }
    // SAFETY: a slab of at least `bytes` this context owns, on its stream.
    let code = unsafe {
        rt::cudaMemsetAsync(
            part.cast::<u8>().add(*have).cast(),
            0,
            bytes - *have,
            ctx.stream().cast(),
        )
    };
    if code != rt::cudaError::cudaSuccess {
        return Err(refuse(OP, format!("cudaMemsetAsync answered {code:?}")));
    }
    *have = bytes;
    Ok(())
}

/// An atom-layout scale tensor as TMA reads it: [row block][atom][128 words],
/// boxes of `atoms` atoms by `row_blocks` row blocks, zero past either end.
fn scale_map(
    s: Scales,
    rows: u32,
    k: u32,
    atoms: u32,
    row_blocks: u32,
) -> Result<[u64; 16], Error> {
    if !s.ptr.is_multiple_of(16) {
        return Err(refuse(OP, "TMA needs the scales 16-byte aligned"));
    }
    let along = k.div_ceil(s.gran).div_ceil(4);
    encode(Map {
        ptr: s.ptr,
        ty: dr::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32,
        dims: [128, along, rows.div_ceil(128)],
        strides: [512, u64::from(along) * 512],
        boxes: [128, atoms, row_blocks],
        swizzle: dr::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    })
}

/// The tensor map of a K-major `rows x k` operand, boxes of 128 bytes of K
/// by `box_rows`, swizzled 128B.
fn tensor_map(
    kind: Kind,
    o: Operand,
    rows: u32,
    k: u32,
    box_rows: u32,
) -> Result<[u64; 16], Error> {
    use dr::CUtensorMapDataType as T;

    // (type, inner extent in the type's elements, row stride in bytes, box width)
    let (ty, inner, stride, box_k) = match (kind, o.elem) {
        (_, Elem::Bf16) => (T::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, k, u64::from(k) * 2, 64),
        (_, Elem::E4m3) => (T::CU_TENSOR_MAP_DATA_TYPE_UINT8, k, u64::from(k), 128),
        (Kind::Nvf4, Elem::E2m1) => (
            T::CU_TENSOR_MAP_DATA_TYPE_UINT8,
            k / 2,
            u64::from(k) / 2,
            128,
        ),
        (_, Elem::E2m1) => (
            T::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
            k,
            u64::from(k) / 2,
            128,
        ),
    };
    // Packed fp4 rows are 16-byte strides; unpacked (16U4_ALIGN16B) ones
    // must be whole 64-byte runs.
    let whole = if ty == T::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B {
        128
    } else {
        32
    };
    if o.elem == Elem::E2m1 && !k.is_multiple_of(whole) {
        return Err(refuse(
            OP,
            format!("an fp4 k of {k} is not a whole number of {whole}s for this kind"),
        ));
    }
    if !stride.is_multiple_of(16) || !o.ptr.is_multiple_of(16) {
        return Err(refuse(
            OP,
            format!("TMA needs 16-byte rows and base; this operand's row is {stride} bytes"),
        ));
    }
    encode(Map {
        ptr: o.ptr,
        ty,
        dims: [inner, rows, 0],
        strides: [stride, 0],
        boxes: [box_k, box_rows, 1],
        swizzle: dr::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
    })
}

/// The bf16 `m x n` output, in the 32 x 32 boxes an epilogue warp stores
/// from its 64-byte-swizzled stage.
fn output_map(y: u64, m: u32, n: u32) -> Result<[u64; 16], Error> {
    if !y.is_multiple_of(16) {
        return Err(refuse(OP, "TMA needs the output 16-byte aligned"));
    }
    encode(Map {
        ptr: y,
        ty: dr::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        dims: [n, m, 0],
        strides: [u64::from(n) * 2, 0],
        boxes: [32, 32, 1],
        swizzle: dr::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
    })
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Map {
    ptr: u64,
    ty: dr::CUtensorMapDataType,
    /// A zero third extent makes the map 2D.
    dims: [u32; 3],
    strides: [u64; 2],
    boxes: [u32; 3],
    swizzle: dr::CUtensorMapSwizzle,
}

/// Encoding is host-only and cheap; the cache spares the driver call on a
/// replayed shape.
fn encode(m: Map) -> Result<[u64; 16], Error> {
    static MAPS: OnceLock<Mutex<HashMap<Map, [u64; 16]>>> = OnceLock::new();
    let maps = MAPS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(found) = maps
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&m)
    {
        return Ok(*found);
    }
    let dims = m.dims.map(u64::from);
    let steps = [1u32, 1, 1];
    let rank = if m.dims[2] == 0 { 2 } else { 3 };
    let mut map: dr::CUtensorMap = unsafe { core::mem::zeroed() };
    // SAFETY: every pointer is to a live local of the length the rank says.
    let code = unsafe {
        dr::cuTensorMapEncodeTiled(
            &raw mut map,
            m.ty,
            rank,
            m.ptr as *mut core::ffi::c_void,
            dims.as_ptr(),
            m.strides.as_ptr(),
            m.boxes.as_ptr(),
            steps.as_ptr(),
            dr::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            m.swizzle,
            dr::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            dr::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };
    if code != dr::CUresult::CUDA_SUCCESS {
        return Err(refuse(
            OP,
            format!(
                "cuTensorMapEncodeTiled answered {code:?} for a {:?} {} x {} at {:#x}",
                m.ty, m.dims[1], m.dims[0], m.ptr
            ),
        ));
    }
    // SAFETY: CUtensorMap is 128 opaque bytes, 64-aligned.
    let raw: [u64; 16] = unsafe { core::mem::transmute(map) };
    maps.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(m, raw);
    Ok(raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_atom_is_the_layout_tcgen05_cp_lands() {
        assert_eq!(atom_offset(0, 0, 4), 0);
        assert_eq!(atom_offset(0, 3, 4), 3);
        assert_eq!(atom_offset(32, 0, 4), 4);
        assert_eq!(atom_offset(1, 0, 4), 16);
        assert_eq!(atom_offset(127, 3, 4), 511);
        assert_eq!(
            atom_offset(0, 4, 8),
            512,
            "atoms of a row block run along K"
        );
        assert_eq!(atom_offset(128, 0, 8), 1024);
        assert_eq!(atom_bytes(129, 4096, 16), 2 * 64 * 512);
    }

    #[test]
    fn a_decode_splits_and_a_full_wave_does_not() {
        assert_eq!(
            Plan::splits_for(32, 32, 148),
            4,
            "32 tiles leave 116 SMs idle"
        );
        assert_eq!(Plan::splits_for(128, 112, 148), 1);
        assert_eq!(Plan::splits_for(4096, 32, 148), 1);
        assert_eq!(
            Plan::splits_for(8, 8, 148),
            2,
            "four steps a split at least"
        );
    }

    #[test]
    fn every_kind_fits_its_ring() {
        for (kind, g) in [
            (Kind::Bf16, 128),
            (Kind::Fp8, 128),
            (Kind::Mx, 32),
            (Kind::Nvf4, 16),
        ] {
            for bn in [32, 64, 128, 256] {
                let plan = Plan {
                    swap: false,
                    pair: bn >= 64,
                    bn,
                    stages: 0,
                    splits: 0,
                }
                .staged(kind, g, g, 1, 1);
                assert!(plan.stages >= 3, "{kind:?} bn={bn}: {plan:?}");
                assert!(smem_bytes(kind, g, g, plan, 1, 1) <= SMEM_CAP);
            }
        }
    }
}
