//! Activations quantized for a low-precision product (`linear/act_quant.cuh`):
//! NVFP4 (e2m1 under an e4m3 scale a 16, under the weight's static global
//! `1 / input_scale`) or e4m3 under a ue8m0 scale a `gran` (DeepSeek's
//! `act_quant`). Each lands as the tcgen05 GEMM reads it ([`quantize`]:
//! codes, and scales in its atoms) or read back to bf16 without the global
//! ([`read_back`]: `q · sf`, exact in bf16), for a bf16 product that
//! multiplies by the global once.
//!
//! [`matmul`] is the dense product: activations quantized, the weight's
//! scales laid out in atoms, one [`tc5`] GEMM. [`select`] is the routed one:
//! the routes laid out by expert on the device, each quantized into its
//! expert's rows (under its expert's global), one grouped [`tc5`] GEMM, and
//! the rows gathered back to route order.

use cudarc::runtime::sys as rt;
use dtype::Dtype;

use crate::error::Error;
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, refuse, stated, symbol};
use crate::linear::tc5;
use crate::tensor::Tensor;

const FILE: &str = "linear/act_quant.cuh";

const BLOCK: u32 = 256;
/// The routes one `route_*_tally` block lays out (`kTallyRoutes`).
const TALLY_ROUTES: u32 = 2048;
/// The most routes `route_lay_out_small` lays out in one block
/// (`kOffsetThreads * kSmallPer`).
const SMALL_LAYOUT: u32 = 1024 * 16;

/// The precision activations are quantized to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// e2m1 under an e4m3 scale a 16, under a static global.
    Nvfp4,
    /// e4m3 under a ue8m0 scale a `gran` (32 or 128).
    E4m3 { gran: u32 },
}

impl Format {
    /// The K one scale covers.
    #[must_use]
    pub const fn gran(self) -> u32 {
        match self {
            Self::Nvfp4 => 16,
            Self::E4m3 { gran } => gran,
        }
    }

    /// A row's code bytes.
    #[must_use]
    pub const fn row_bytes(self, k: u32) -> u32 {
        match self {
            Self::Nvfp4 => k / 2,
            Self::E4m3 { .. } => k,
        }
    }

    fn check(self, op: &'static str, k: u32) -> Result<(), Error> {
        let gran = self.gran();
        if let Self::E4m3 { gran } = self
            && !matches!(gran, 32 | 128)
        {
            return Err(refuse(
                op,
                format!("an e4m3 scale covers 32 or 128, not {gran}"),
            ));
        }
        if k == 0 || !k.is_multiple_of(gran) {
            return Err(refuse(
                op,
                format!("K is {k}, not a whole number of {gran}-wide scale blocks"),
            ));
        }
        Ok(())
    }
}

/// Where a quantization reads: `rows` out rows, row `r` reading `x` row
/// `r / div`, its global (NVFP4) its route's expert's with `routes`, else
/// `input_scales[0]`; with `pos`, a device `i32[rows]`, row `r`'s codes land
/// at row `pos[r]` of a grouped layout (none where it is negative), the
/// masked one's scales padded to whole row blocks an expert when `m_max` is
/// stated.
#[derive(Clone, Copy, Debug)]
pub struct Source {
    pub x: Tensor,
    pub rows: u32,
    pub div: u32,
    pub routes: Option<Tensor>,
    pub input_scales: Option<Tensor>,
    pub pos: u64,
    pub m_max: u32,
}

impl Source {
    /// `x` row for row.
    #[must_use]
    pub const fn rows(x: Tensor, input_scales: Option<Tensor>) -> Self {
        Self {
            x,
            rows: x.rows,
            div: 1,
            routes: None,
            input_scales,
            pos: 0,
            m_max: 0,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn launch(
    ctx: &Ctx,
    op: &'static str,
    fmt: Format,
    mode: u32,
    src: Source,
    codes: u64,
    atoms: u64,
    values: u64,
) -> Result<(), Error> {
    let k = src.x.width;
    fmt.check(op, k)?;
    if src.x.dtype != Dtype::Bf16 {
        return Err(refuse(
            op,
            format!("activations quantize from bf16, not {:?}", src.x.dtype),
        ));
    }
    if src.div == 0 || src.rows.div_ceil(src.div) > src.x.rows {
        return Err(refuse(
            op,
            format!(
                "{} rows a {} of {} read past the activations",
                src.rows, src.div, src.x.rows
            ),
        ));
    }
    let blocks = u64::from(src.rows) * u64::from(k / fmt.gran());
    let ptr = |t: Option<Tensor>| t.map_or(ArgValue::Ptr(0), |t| t.arg());
    let int = |v: u32| stated(op, v).map(|v| v.arg());
    match fmt {
        Format::Nvfp4 => {
            if src.input_scales.is_none() {
                return Err(refuse(
                    op,
                    "NVFP4 activations quantize under the weight's input scales",
                ));
            }
            let threads = u32::try_from(blocks)
                .map_err(|_| refuse(op, format!("{blocks} blocks overflow a launch")))?;
            ctx.fire(
                op,
                Fire::at(
                    FILE,
                    symbol(&format!("::pie::linear::act_quant_nvfp4<{mode}>")),
                )
                .apply(Launch::flat(threads, BLOCK)),
                &[
                    src.x.arg(),
                    ptr(src.routes),
                    ptr(src.input_scales),
                    ArgValue::Ptr(src.pos),
                    int(src.m_max)?,
                    ArgValue::Ptr(codes),
                    ArgValue::Ptr(atoms),
                    ArgValue::Ptr(values),
                    int(src.rows)?,
                    int(k)?,
                    int(src.div)?,
                ],
            )
        }
        Format::E4m3 { gran } => {
            if src.routes.is_some() {
                return Err(refuse(
                    op,
                    "e4m3 activations quantize alike for every expert",
                ));
            }
            let threads = u32::try_from(blocks * 32)
                .map_err(|_| refuse(op, format!("{blocks} blocks overflow a launch")))?;
            ctx.fire(
                op,
                Fire::at(
                    FILE,
                    symbol(&format!("::pie::linear::act_quant_e4m3<{mode}, {gran}>")),
                )
                .apply(Launch::flat(threads, BLOCK)),
                &[
                    src.x.arg(),
                    ArgValue::Ptr(src.pos),
                    int(src.m_max)?,
                    ArgValue::Ptr(codes),
                    ArgValue::Ptr(atoms),
                    ArgValue::Ptr(values),
                    int(src.rows)?,
                    int(k)?,
                    int(src.div)?,
                ],
            )
        }
    }
}

/// Codes (`fmt.row_bytes(k)` a row) and atom-layout scales, `atom_bytes`
/// of them zeroed first so the padding the GEMM reads is finite, as [`tc5`]
/// reads them.
pub fn quantize(
    ctx: &Ctx,
    op: &'static str,
    fmt: Format,
    src: Source,
    codes: u64,
    atoms: u64,
    atom_bytes: usize,
) -> Result<(), Error> {
    zero(ctx, op, atoms, atom_bytes)?;
    launch(ctx, op, fmt, 0, src, codes, atoms, 0)
}

/// The quantized activations read back to bf16 into `out` (`rows x k`),
/// each `q · sf` without the global: exact in bf16, so a bf16 product of
/// them rounds nothing until its accumulator.
pub fn read_back(
    ctx: &Ctx,
    op: &'static str,
    fmt: Format,
    src: Source,
    out: u64,
) -> Result<Tensor, Error> {
    launch(ctx, op, fmt, 1, src, 0, 0, out)?;
    Ok(Tensor::new(out, src.rows, src.x.width, Dtype::Bf16))
}

/// A row-major plane of one-byte scales, `cols` a row, laid into atoms for
/// `rows` rows (zeroed first), atom row `r` reading plane row `r / div`.
pub fn scales_to_atoms(
    ctx: &Ctx,
    op: &'static str,
    plane: u64,
    atoms: u64,
    rows: u32,
    cols: u32,
    div: u32,
) -> Result<(), Error> {
    let bytes = rows.div_ceil(128) as usize * cols.div_ceil(4) as usize * 512;
    zero(ctx, op, atoms, bytes)?;
    let n = u32::try_from(u64::from(rows) * u64::from(cols)).map_err(|_| {
        refuse(
            op,
            format!("{rows} rows of {cols} scales overflow a launch"),
        )
    })?;
    ctx.fire(
        op,
        Fire::at(FILE, "::pie::linear::scales_to_atoms").apply(Launch::flat(n, BLOCK)),
        &[
            ArgValue::Ptr(plane),
            ArgValue::Ptr(atoms),
            stated(op, rows)?.arg(),
            stated(op, cols)?.arg(),
            stated(op, div.max(1))?.arg(),
        ],
    )
}

fn zero(ctx: &Ctx, op: &'static str, ptr: u64, bytes: usize) -> Result<(), Error> {
    // SAFETY: a slab of `bytes` the caller owns, on this context's stream.
    let code = unsafe { rt::cudaMemsetAsync(ptr as *mut _, 0, bytes, ctx.stream().cast()) };
    if code != rt::cudaError::cudaSuccess {
        return Err(refuse(op, format!("cudaMemsetAsync answered {code:?}")));
    }
    Ok(())
}

/// The activations' codes and scales, one a stream: the GEMM after them on
/// the same stream consumes them.
const ACT_CODES: &str = "linear.act_quant.codes";
const ACT_ATOMS: &str = "linear.act_quant.atoms";
const WEIGHT_ATOMS: &str = "linear.act_quant.weight_atoms";

/// A dense projection's weight as the tcgen05 GEMM reads it: `codes` its
/// `n` rows of `k` codes (two e2m1 a byte, the even one low), `scales` its
/// one-byte scales (row-major, or already in atoms when `laid`), `gran` K a
/// scale, ue8m0 (MXFP4) or e4m3 (NVFP4).
#[derive(Clone, Copy, Debug)]
pub struct Weight {
    pub codes: u64,
    pub scales: u64,
    pub n: u32,
    pub k: u32,
    pub elem: tc5::Elem,
    pub gran: u32,
    pub laid: bool,
}

/// `y = quant(act) · w^T` on the tcgen05 GEMM, `alpha` the product's global
/// (1 for ue8m0 scales).
pub fn matmul(
    ctx: &Ctx,
    op: &'static str,
    fmt: Format,
    act: Source,
    w: Weight,
    y: &mut Tensor,
    alpha: f32,
) -> Result<(), Error> {
    let (m, k) = (act.rows, act.x.width);
    if w.k != k || y.rows != m || y.width != w.n {
        return Err(refuse(
            op,
            format!(
                "{m} x {k} activations against an {} x {} weight into {} x {}",
                w.n, w.k, y.rows, y.width
            ),
        ));
    }
    if m == 0 {
        return Ok(());
    }
    let codes = ctx.scratch_stream(op, ACT_CODES, m as usize * fmt.row_bytes(k) as usize)?;
    let atom_bytes = tc5::atom_bytes(m as usize, k as usize, fmt.gran() as usize);
    let atoms = ctx.scratch_stream(op, ACT_ATOMS, atom_bytes)?;
    quantize(ctx, op, fmt, act, codes as u64, atoms as u64, atom_bytes)?;
    let w_atoms = if w.laid {
        w.scales
    } else {
        let bytes = tc5::atom_bytes(w.n as usize, k as usize, w.gran as usize);
        let atoms = ctx.scratch_stream(op, WEIGHT_ATOMS, bytes)? as u64;
        scales_to_atoms(ctx, op, w.scales, atoms, w.n, k / w.gran, 1)?;
        atoms
    };
    let a_elem = match fmt {
        Format::Nvfp4 => tc5::Elem::E2m1,
        Format::E4m3 { .. } => tc5::Elem::E4m3,
    };
    tc5::gemm(
        ctx,
        tc5::Operand::scaled(codes as u64, a_elem, atoms as u64, fmt.gran()),
        tc5::Operand::scaled(w.codes, w.elem, w_atoms, w.gran),
        y.ptr,
        [m, w.n, k],
        alpha,
    )
}

/// A routed bank as the grouped tcgen05 GEMM reads it: `codes`
/// `[experts][n][k]`, its scales laid into atoms of each expert's whole row
/// blocks, `gran` K a scale, and `alpha_e` a device `f32[experts]` scaling
/// each expert's product (NVFP4's `input_scale · tensor scale`) or 0.
#[derive(Clone, Copy, Debug)]
pub struct Bank {
    pub codes: u64,
    pub atoms: u64,
    pub experts: u32,
    pub n: u32,
    pub k: u32,
    pub elem: tc5::Elem,
    pub gran: u32,
    pub alpha_e: u64,
    /// Where `alpha_e` comes from, when the select lays it: NVFP4's input
    /// scales and tensor scales (`rows` an expert), `alpha_e[e] =
    /// input_scales[e] · tensor_scales[e · rows]` ([`expert_alpha`]),
    /// written with the fire's route layout rather than in a launch of its
    /// own.
    pub alpha_from: Option<AlphaFrom>,
}

/// NVFP4's per-expert product scale's sources (see [`Bank::alpha_from`]).
#[derive(Clone, Copy, Debug)]
pub struct AlphaFrom {
    pub input_scales: u64,
    pub tensor_scales: u64,
    pub rows: u32,
}

/// A fire of this many routes (or tokens, when no token names an expert
/// twice) or fewer lays them out masked, each expert's rows a block as long
/// as the fire's routes (tokens); a wider one lays them out contiguous.
const MASKED_FIRE: u32 = 64;

/// A select's experts start whole 128-row scale blocks apart when its
/// activations carry block scales (the atoms it lands are whole blocks).
const BLOCK_ROWS: u32 = 128;
const _: () = assert!(BLOCK_ROWS.is_multiple_of(tc5::GROUP_ALIGN));

const ROUTE_COUNTS: &str = "linear.act_quant.route_counts";
const ROUTE_OFFSETS: &str = "linear.act_quant.route_offsets";
const ROUTE_CURSOR: &str = "linear.act_quant.route_cursor";
const ROUTE_POS: &str = "linear.act_quant.route_pos";
const GROUPED_Y: &str = "linear.act_quant.grouped_y";

/// `y[route] = quant(x[route or its token]) · W[routes[route]]^T` on one
/// grouped tcgen05 GEMM, each route's activations quantized under its
/// expert's global (NVFP4, `input_scales` one an expert); a dropped route
/// lands zeros.
#[allow(clippy::too_many_arguments)]
pub fn select(
    ctx: &Ctx,
    op: &'static str,
    fmt: Format,
    x: Tensor,
    routes: Tensor,
    input_scales: Option<Tensor>,
    bank: Bank,
    y: &mut Tensor,
) -> Result<(), Error> {
    let Bank { experts, n, k, .. } = bank;
    let tokens = routes.rows;
    let count = tokens * routes.width;
    if x.width != k || y.rows != count || y.width != n {
        return Err(refuse(
            op,
            format!(
                "{} x {} activations, {count} routes, against [{experts}][{n}][{k}] into {} x {}",
                x.rows, x.width, y.rows, y.width
            ),
        ));
    }
    if count == 0 {
        return Ok(());
    }
    let div = if x.rows == count { 1 } else { routes.width };
    let layout = lay_out_with(
        ctx,
        op,
        routes,
        experts,
        BLOCK_ROWS,
        false,
        bank.alpha_from.map(|a| (a, bank.alpha_e)),
    )?;
    let Layout {
        pos,
        rows,
        groups,
        m_max,
        ..
    } = layout;
    let (e, gran) = (experts as usize, fmt.gran() as usize);
    let atom_bytes = if m_max > 0 {
        tc5::masked_atom_bytes(e, m_max as usize, k as usize, gran)
    } else {
        tc5::atom_bytes(rows as usize, k as usize, gran)
    };
    let codes = ctx.scratch_stream(op, ACT_CODES, rows as usize * fmt.row_bytes(k) as usize)?;
    let atoms = ctx.scratch_stream(op, ACT_ATOMS, atom_bytes)?;
    let src = Source {
        x,
        rows: count,
        div,
        // A dropped route lands nowhere; NVFP4 reads each route's expert's
        // global.
        routes: (fmt == Format::Nvfp4).then_some(routes),
        input_scales,
        pos,
        m_max,
    };
    quantize(ctx, op, fmt, src, codes as u64, atoms as u64, atom_bytes)?;
    let grouped = ctx.scratch_stream(op, GROUPED_Y, rows as usize * n as usize * 2)? as u64;
    let a_elem = match fmt {
        Format::Nvfp4 => tc5::Elem::E2m1,
        Format::E4m3 { .. } => tc5::Elem::E4m3,
    };
    tc5::grouped_gemm_epi(
        ctx,
        tc5::Operand::scaled(codes as u64, a_elem, atoms as u64, fmt.gran()),
        tc5::Operand::scaled(bank.codes, bank.elem, bank.atoms, bank.gran),
        grouped,
        tc5::Grouped {
            experts,
            groups,
            alpha_e: bank.alpha_e,
        },
        [n, k],
        1.0,
        // An expert's padding rows land nowhere.
        tc5::Epilogue {
            live_rows: layout.live_rows(),
            ..tc5::Epilogue::STORE
        },
        count,
    )?;
    gather_back(ctx, op, grouped, pos, routes.width, y)
}

/// Routes laid out by expert on the device, as the grouped [`tc5`] GEMM
/// reads them: route `r` at row `pos[r]` of `rows` (negative: dropped, or
/// past the context's live window, whose rows are left alone). A fire of
/// [`MASKED_FIRE`] routes or fewer is masked, each expert `m_max` = its
/// route count of rows (its count of them live) -- or of tokens or fewer
/// when `distinct` (no token names an expert twice: `m_max` = its tokens);
/// a wider one contiguous, each expert's rows `align` apart.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Layout {
    pub pos: u64,
    pub rows: u32,
    pub groups: tc5::Groups,
    pub m_max: u32,
    /// Each expert's live rows (`i32 [experts]`), the rest of a contiguous
    /// expert's block padding.
    pub counts: u64,
}

impl Layout {
    /// The epilogue's live-row clip for this layout: a contiguous group's
    /// counts (a masked one's tiles already stop at them).
    pub(crate) fn live_rows(&self) -> u64 {
        match self.groups {
            tc5::Groups::Contiguous { .. } => self.counts,
            tc5::Groups::Masked { .. } => 0,
        }
    }
}

/// Lays `routes` (`i32 [tokens][top]`, an expert of `experts` or negative)
/// out by expert: counts, offsets and each route's row, on the device.
pub(crate) fn lay_out(
    ctx: &Ctx,
    op: &'static str,
    routes: Tensor,
    experts: u32,
    align: u32,
    distinct: bool,
) -> Result<Layout, Error> {
    lay_out_with(ctx, op, routes, experts, align, distinct, None)
}

/// [`lay_out`], laying NVFP4's per-expert alpha too where `alpha` names its
/// sources and destination: in the narrow fire's one launch, else apart.
fn lay_out_with(
    ctx: &Ctx,
    op: &'static str,
    routes: Tensor,
    experts: u32,
    align: u32,
    distinct: bool,
    alpha: Option<(AlphaFrom, u64)>,
) -> Result<Layout, Error> {
    let tokens = routes.rows;
    let count = tokens * routes.width;
    let m_max = match distinct {
        true if tokens <= MASKED_FIRE => tokens,
        false if count <= MASKED_FIRE => count,
        _ => 0,
    };
    let e = experts as usize;
    let i32s = |name, len: usize| -> Result<u64, Error> {
        Ok(ctx.scratch_stream(op, name, len * 4)? as u64)
    };
    let counts = i32s(ROUTE_COUNTS, e)?;
    let offsets = i32s(ROUTE_OFFSETS, e + 1)?;
    let cursor = i32s(ROUTE_CURSOR, e)?;
    let pos = i32s(ROUTE_POS, count as usize)?;
    let int = |v: u32| stated(op, v).map(|v| v.arg());
    let laid = |pos| {
        if m_max > 0 {
            Layout {
                pos,
                rows: experts * m_max,
                groups: tc5::Groups::Masked { counts, m_max },
                m_max,
                counts,
            }
        } else {
            // Each expert's rows start a whole alignment apart.
            let rows = (count + (align - 1) * experts.min(count)).next_multiple_of(align);
            Layout {
                pos,
                rows,
                groups: tc5::Groups::Contiguous { offsets, rows },
                m_max,
                counts,
            }
        }
    };
    // A narrow fire lays out in one block (`route_lay_out_small`): one launch,
    // not a clear and three.
    if count <= SMALL_LAYOUT {
        ctx.fire(
            op,
            Fire::at(FILE, "::pie::linear::route_lay_out_small")
                .apply(Launch::flat(1024, 1024).smem(8 * experts)),
            &[
                routes.arg(),
                int(count)?,
                int(routes.width)?,
                ctx.stage(),
                int(experts)?,
                int(m_max)?,
                int(align)?,
                ArgValue::Ptr(counts),
                ArgValue::Ptr(offsets),
                ArgValue::Ptr(pos),
                ArgValue::Ptr(alpha.map_or(0, |(a, _)| a.input_scales)),
                ArgValue::Ptr(alpha.map_or(0, |(a, _)| a.tensor_scales)),
                ArgValue::Ptr(alpha.map_or(0, |(_, out)| out)),
                int(alpha.map_or(0, |(a, _)| a.rows))?,
            ],
        )?;
        return Ok(laid(pos));
    }
    if let Some((a, out)) = alpha {
        expert_alpha(
            ctx,
            op,
            a.input_scales,
            a.tensor_scales,
            out,
            experts,
            a.rows,
        )?;
    }
    zero(ctx, op, counts, e * 4)?;
    // A block of routes tallies its experts in shared memory and touches the
    // global count once an expert (`route_count_tally`).
    let tally = Launch::flat(count.div_ceil(TALLY_ROUTES) * BLOCK, BLOCK).smem(8 * experts);
    ctx.fire(
        op,
        Fire::at(FILE, "::pie::linear::route_count_tally").apply(tally),
        &[
            routes.arg(),
            int(count)?,
            int(routes.width)?,
            ctx.stage(),
            int(experts)?,
            ArgValue::Ptr(counts),
        ],
    )?;
    ctx.fire(
        op,
        // One block scans the experts' rows into their starts.
        Fire::at(FILE, "::pie::linear::route_offsets_scan").apply(Launch::flat(1024, 1024)),
        &[
            ArgValue::Ptr(counts),
            int(experts)?,
            int(m_max)?,
            int(align)?,
            ArgValue::Ptr(offsets),
            ArgValue::Ptr(cursor),
        ],
    )?;
    ctx.fire(
        op,
        Fire::at(FILE, "::pie::linear::route_place_tally").apply(tally),
        &[
            routes.arg(),
            int(count)?,
            int(routes.width)?,
            ctx.stage(),
            int(experts)?,
            ArgValue::Ptr(cursor),
            ArgValue::Ptr(pos),
        ],
    )?;
    Ok(laid(pos))
}

/// `y[r] = grouped[pos[r]]`, a dropped route's row zeros: the grouped
/// product back in route order.
pub(crate) fn gather_back(
    ctx: &Ctx,
    op: &'static str,
    grouped: u64,
    pos: u64,
    top: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    let (count, n) = (y.rows, y.width);
    // Whole 16-byte runs where the rows and both buffers allow them: the same
    // values, a quarter of the threads.
    let vec8 = n.is_multiple_of(8) && grouped % 16 == 0 && y.ptr % 16 == 0;
    let (entry, per) = if vec8 {
        ("::pie::linear::route_gather_vec8", 8)
    } else {
        ("::pie::linear::route_gather", 2)
    };
    let threads = u32::try_from(u64::from(count) * u64::from(n / per))
        .map_err(|_| refuse(op, format!("{count} routes of {n} overflow a launch")))?;
    let int = |v: u32| stated(op, v).map(|v| v.arg());
    ctx.fire(
        op,
        Fire::at(FILE, entry).apply(Launch::flat(threads, BLOCK)),
        &[
            ArgValue::Ptr(grouped),
            ArgValue::Ptr(pos),
            y.arg(),
            int(count)?,
            int(n)?,
            int(top)?,
            ctx.stage(),
        ],
    )
}

/// `y[t] = sum_j weights[t][j] * grouped[pos[t * top + j]]`, in f32 and in
/// `j` order, a dropped route adding nothing: `moe_weighted_sum` of the
/// grouped product gathered back, read where it lies. Under a gate, plus
/// `shared[t] * sigmoid(gate[t])`, rounded once. Tokens past the live window
/// are untouched.
pub(crate) fn weighted_gather(
    ctx: &Ctx,
    op: &'static str,
    grouped: u64,
    pos: u64,
    weights: Tensor,
    gate: Option<(Tensor, Tensor)>,
    y: &mut Tensor,
) -> Result<(), Error> {
    let (tokens, n, top) = (y.rows, y.width, weights.width);
    // Under a gate, the shared expert's row and its gate column (a pitch a
    // row); without, nothing.
    let (shared, gate, pitch) = gate.map_or((0, 0, 0), |(s, g)| (s.ptr, g.ptr, g.width));
    if !n.is_multiple_of(8) || !grouped.is_multiple_of(16) || !y.ptr.is_multiple_of(16) {
        return Err(refuse(
            op,
            format!("a weighted gather of {n}-wide rows in 16-byte runs"),
        ));
    }
    let int = |v: u32| stated(op, v).map(|v| v.arg());
    ctx.fire(
        op,
        Fire::at(FILE, "::pie::linear::route_weighted_gather_vec8").apply(Launch::grid(
            [tokens, (n / 8).div_ceil(BLOCK), 1],
            [BLOCK, 1, 1],
        )),
        &[
            ArgValue::Ptr(grouped),
            ArgValue::Ptr(pos),
            weights.arg(),
            ArgValue::Ptr(shared),
            ArgValue::Ptr(gate),
            int(pitch)?,
            y.arg(),
            int(n)?,
            int(top)?,
            ctx.stage(),
        ],
    )
}

/// `grouped[pos[r]] = x[r / div]` for each of `count` routes (a dropped one
/// lands nowhere): bf16 activations in the layout's rows.
pub(crate) fn scatter_rows(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    div: u32,
    pos: u64,
    count: u32,
    grouped: u64,
) -> Result<(), Error> {
    if x.dtype != Dtype::Bf16 || !x.width.is_multiple_of(8) {
        return Err(refuse(
            op,
            format!(
                "rows scatter as bf16 in 16-byte runs, not {:?} x {}",
                x.dtype, x.width
            ),
        ));
    }
    let runs = u32::try_from(u64::from(count) * u64::from(x.width / 8)).map_err(|_| {
        refuse(
            op,
            format!("{count} routes of {} overflow a launch", x.width),
        )
    })?;
    let int = |v: u32| stated(op, v).map(|v| v.arg());
    ctx.fire(
        op,
        Fire::at(FILE, "::pie::linear::route_scatter_rows").apply(Launch::flat(runs, BLOCK)),
        &[
            x.arg(),
            ArgValue::Ptr(pos),
            ArgValue::Ptr(grouped),
            int(count)?,
            int(x.width)?,
            int(div)?,
        ],
    )
}

/// `alpha[e] = input_scales[e] · tensor_scales[e][0]`: an NVFP4 bank's
/// per-expert product scale, its first row's tensor scale standing for the
/// expert's (vLLM reads a gated bank's gate leg's for both legs).
pub fn expert_alpha(
    ctx: &Ctx,
    op: &'static str,
    input_scales: u64,
    tensor_scales: u64,
    alpha: u64,
    experts: u32,
    rows: u32,
) -> Result<(), Error> {
    ctx.fire(
        op,
        Fire::at(FILE, "::pie::linear::expert_alpha").apply(Launch::flat(experts, BLOCK)),
        &[
            ArgValue::Ptr(input_scales),
            ArgValue::Ptr(tensor_scales),
            ArgValue::Ptr(alpha),
            stated(op, experts)?.arg(),
            stated(op, rows)?.arg(),
        ],
    )
}
