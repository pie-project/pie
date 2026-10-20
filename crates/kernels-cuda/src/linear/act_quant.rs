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
            // Codes for a token's routes: each token's blocks read and
            // quantized once (its routes' globals are one) and landed at
            // each route's row; under `FAN_TOKENS` a thread a route.
            let fan = mode == 0
                && src.div > 1
                && src.rows / src.div >= FAN_TOKENS
                && src.pos != 0
                && src.routes.is_some()
                && src.rows.is_multiple_of(src.div)
                && src.x.ptr.is_multiple_of(16)
                && codes.is_multiple_of(8);
            let blocks = if fan {
                blocks / u64::from(src.div)
            } else {
                blocks
            };
            let threads = u32::try_from(blocks)
                .map_err(|_| refuse(op, format!("{blocks} blocks overflow a launch")))?;
            ctx.fire(
                op,
                Fire::at(
                    FILE,
                    symbol(&format!("::pie::linear::act_quant_nvfp4<{mode}, {fan}>")),
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
                    int(if fan { src.rows / src.div } else { src.rows })?,
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
            // Eight elements a lane where the planes align to it (a 16-byte
            // load, an 8-byte store), else one a lane: the same bytes.
            let v8 = src.x.ptr.is_multiple_of(16)
                && codes.is_multiple_of(8)
                && values.is_multiple_of(16);
            let lanes = if v8 { u64::from(gran / 8) } else { 32 };
            let wide = if v8 { "_v8" } else { "" };
            // The eight-a-lane kernels wait on their producer in-kernel.
            let pdl = v8 && ctx.tuning().pdl;
            // Codes for a token's routes: each token's blocks quantized once
            // and landed at each of its routes' rows.
            let fan = mode == 0 && src.div > 1 && src.pos != 0 && src.rows.is_multiple_of(src.div);
            let rows = if fan { src.rows / src.div } else { src.rows };
            let lanes_all = u64::from(rows) * u64::from(k / gran) * lanes;
            let threads = u32::try_from(lanes_all.next_multiple_of(32))
                .map_err(|_| refuse(op, format!("{blocks} blocks overflow a launch")))?;
            ctx.fire(
                op,
                Fire::at(
                    FILE,
                    symbol(&format!(
                        "::pie::linear::act_quant_e4m3{wide}<{mode}, {gran}, {fan}>"
                    )),
                )
                .apply(Launch::flat(threads, BLOCK).programmatic(pdl)),
                &[
                    src.x.arg(),
                    ArgValue::Ptr(src.pos),
                    int(src.m_max)?,
                    ArgValue::Ptr(codes),
                    ArgValue::Ptr(atoms),
                    ArgValue::Ptr(if fan { 0 } else { values }),
                    int(rows)?,
                    int(k)?,
                    int(src.div)?,
                ],
            )
        }
    }
}

/// Codes (`fmt.row_bytes(k)` a row) and atom-layout scales, as [`tc5`]
/// reads them. Only the live rows' atoms land: the padding past them (a
/// block's rows past its live ones) keeps what it held, which the GEMM
/// reads for rows no store keeps. It must be finite, so `atoms` is a slab
/// zeroed when fresh that only ever holds `fmt`'s scales (a ue8m0 1.0,
/// 0x7f, is an e4m3 NaN): [`atoms_of`]. `atom_bytes` is the extent the GEMM
/// reads.
pub fn quantize(
    ctx: &Ctx,
    op: &'static str,
    fmt: Format,
    src: Source,
    codes: u64,
    atoms: u64,
    atom_bytes: usize,
) -> Result<(), Error> {
    let _ = atom_bytes;
    launch(ctx, op, fmt, 0, src, codes, atoms, 0)
}

/// A gate-up's bf16 rows (`[rows][gate inter | up inter]`) through SwiGLU
/// into e4m3 codes in 1x128 blocks and their atoms (`m_max` a masked
/// layout's), the GLU's values rounded to bf16 first as apart: `act_quant`'s
/// eight a lane over the GLU's output.
#[allow(clippy::too_many_arguments)]
pub fn swiglu_quant(
    ctx: &Ctx,
    op: &'static str,
    h: u64,
    rows: u32,
    inter: u32,
    m_max: u32,
    codes: u64,
    atoms: u64,
) -> Result<(), Error> {
    if !inter.is_multiple_of(128) || !h.is_multiple_of(16) || !codes.is_multiple_of(8) {
        return Err(refuse(
            op,
            format!("a {inter}-wide GLU lands whole 128-wide e4m3 blocks"),
        ));
    }
    let lanes = u64::from(rows) * u64::from(inter / 8);
    let threads = u32::try_from(lanes.next_multiple_of(32))
        .map_err(|_| refuse(op, format!("{lanes} lanes overflow a launch")))?;
    let int = |v: u32| stated(op, v).map(|v| v.arg());
    ctx.fire(
        op,
        Fire::at(FILE, "::pie::linear::swiglu_quant_e4m3_v8")
            .apply(Launch::flat(threads, BLOCK).programmatic(ctx.tuning().pdl)),
        &[
            ArgValue::Ptr(h),
            int(rows)?,
            int(inter)?,
            int(m_max)?,
            ArgValue::Ptr(codes),
            ArgValue::Ptr(atoms),
        ],
    )
}

/// A stream's slab named `name` (one name a format) of `bytes` activation
/// atoms, zeroed when fresh, for [`quantize`]: no clear a launch.
pub(crate) fn atoms_of(
    ctx: &Ctx,
    op: &'static str,
    name: &'static str,
    bytes: usize,
) -> Result<u64, Error> {
    Ok(ctx.scratch_stream_zeroed(op, name, bytes)? as u64)
}

/// The quantizer's own atom slab for `fmt`.
fn act_atoms(fmt: Format) -> &'static str {
    match fmt {
        Format::Nvfp4 => ACT_ATOMS_NVFP4,
        Format::E4m3 { .. } => ACT_ATOMS_E4M3,
    }
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

/// A 128x128-block FP8 projection (`n x k` e4m3 codes at `codes`, their
/// ue8m0 exponents laid in atoms a row a 128 columns) at a decode's few rows
/// as a GEMV (`linear/fp8_gemv.cuh`): the activations as they are, bf16, the
/// codes streamed once. `Ok(false)` past `kernel.gemm.fp8_gemv_until` rows
/// or for a shape it does not read (`k` a whole number of 512s, the rows'
/// activations within its shared memory).
pub fn fp8_block_gemv(
    ctx: &Ctx,
    act: Tensor,
    codes: u64,
    atoms: u64,
    y: &mut Tensor,
) -> Result<bool, Error> {
    fp8_block_gemv_in(ctx, act, codes, atoms, 128, y)
}

/// [`fp8_block_gemv`] over blocks of `gran` (128, or MXFP8's 32) columns.
pub fn fp8_block_gemv_in(
    ctx: &Ctx,
    act: Tensor,
    codes: u64,
    atoms: u64,
    gran: u32,
    y: &mut Tensor,
) -> Result<bool, Error> {
    const OP: &str = "linear.fp8_block_gemv";
    const WARPS: u32 = 8;
    const ROWS: u32 = 4;
    let (m, n, k) = (act.rows, y.width, act.width);
    if m == 0
        || m > ctx.tuning().gemm_fp8_gemv_until
        || act.dtype != Dtype::Bf16
        || y.dtype != Dtype::Bf16
        || y.rows != m
        || k == 0
        || !k.is_multiple_of(512)
        || n == 0
        || !codes.is_multiple_of(16)
        || !act.ptr.is_multiple_of(16)
        || !matches!(gran, 32 | 128)
    {
        return Ok(false);
    }
    // Warps a row group: each at least two 512-code passes of K.
    let splits = (k / 1024).clamp(1, WARPS).next_power_of_two().min(WARPS);
    let rows_pad: u32 = match m {
        1 => 1,
        2 => 2,
        3 | 4 => 4,
        _ => return Ok(false),
    };
    ctx.fire(
        OP,
        Fire::at(
            "linear/fp8_gemv.cuh",
            symbol(&format!(
                "::pie::linear::fp8_block_gemv<{WARPS}, {ROWS}, {rows_pad}, {gran}>"
            )),
        )
        .apply(Launch::grid(
            [n.div_ceil(ROWS * (WARPS / splits)), 1, 1],
            [WARPS * 32, 1, 1],
        )),
        &[
            act.arg(),
            stated(OP, act.width)?.arg(),
            ArgValue::Ptr(codes),
            ArgValue::Ptr(atoms),
            y.arg(),
            stated(OP, y.width)?.arg(),
            stated(OP, m)?.arg(),
            stated(OP, n)?.arg(),
            stated(OP, k)?.arg(),
            stated(OP, splits)?.arg(),
        ],
    )?;
    Ok(true)
}

/// A block-FP8 plane (`rows x cols` e4m3 codes at `codes`) under f32 factors
/// (`[rows / per][cols / 128]` at `factors`, `per` rows a factor: 128, or 1
/// for a row's own) requantized in place to ue8m0 exponents (`exps`, the
/// factors' shape): each factor rounded up to its power of two, each code
/// re-rounded under it, as vLLM's DeepGEMM path requantizes a block-FP8
/// release on Blackwell.
#[allow(clippy::too_many_arguments)]
pub fn fp8_tile_requant(
    ctx: &Ctx,
    op: &'static str,
    codes: u64,
    factors: u64,
    exps: u64,
    rows: u32,
    cols: u32,
    per: u32,
) -> Result<(), Error> {
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    ctx.fire(
        op,
        Fire::at(FILE, "::pie::linear::fp8_tile_requant").apply(Launch::grid(
            [cols.div_ceil(128), rows.div_ceil(128), 1],
            [32, 8, 1],
        )),
        &[
            ArgValue::Ptr(codes),
            ArgValue::Ptr(factors),
            ArgValue::Ptr(exps),
            stated(op, rows)?.arg(),
            stated(op, cols)?.arg(),
            stated(op, per)?.arg(),
        ],
    )
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
const ACT_ATOMS_NVFP4: &str = "linear.act_quant.atoms_nvfp4";
const ACT_ATOMS_E4M3: &str = "linear.act_quant.atoms_e4m3";
const WEIGHT_ATOMS: &str = "linear.act_quant.weight_atoms";
const HELD_CODES: &str = "linear.act_quant.held_codes";
const HELD_ATOMS: &str = "linear.act_quant.held_atoms";

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
    let atoms = atoms_of(ctx, op, act_atoms(fmt), atom_bytes)?;
    quantize(ctx, op, fmt, act, codes as u64, atoms, atom_bytes)?;
    product(ctx, op, fmt, [codes as u64, atoms], m, w, y, alpha)
}

/// Where a producer lands `rows x k` of DeepSeek's fp8 1x128 activations
/// itself (`e4m3_block_store` in act_quant.cuh, row `o` at row `o`,
/// `m_max` 0) for [`matmul_held`]: the codes and their atoms, apart from
/// the quantizer's own planes so a quantize between the producer and its
/// consumer leaves them be. The atoms are zeroed when fresh, and every
/// store lands a finite scale, so the padding the GEMM reads past `rows`
/// stays finite with no clear a launch.
pub fn held_planes(ctx: &Ctx, op: &'static str, rows: u32, k: u32) -> Result<[u64; 2], Error> {
    let fmt = Format::E4m3 { gran: 128 };
    fmt.check(op, k)?;
    let codes = ctx.scratch_stream(op, HELD_CODES, rows as usize * fmt.row_bytes(k) as usize)?;
    let atom_bytes = tc5::atom_bytes(rows as usize, k as usize, 128);
    let atoms = ctx.scratch_stream_zeroed(op, HELD_ATOMS, atom_bytes)?;
    Ok([codes as u64, atoms as u64])
}

/// [`matmul`] of activations a producer already landed in `planes`
/// ([`held_planes`]), `rows` of them: no quantize launch.
pub fn matmul_held(
    ctx: &Ctx,
    op: &'static str,
    planes: [u64; 2],
    rows: u32,
    w: Weight,
    y: &mut Tensor,
    alpha: f32,
) -> Result<(), Error> {
    let fmt = Format::E4m3 { gran: 128 };
    fmt.check(op, w.k)?;
    if y.rows != rows || y.width != w.n {
        return Err(refuse(
            op,
            format!(
                "{rows} held rows against an {} x {} weight into {} x {}",
                w.n, w.k, y.rows, y.width
            ),
        ));
    }
    if rows == 0 {
        return Ok(());
    }
    product(ctx, op, fmt, planes, rows, w, y, alpha)
}

/// The tcgen05 product of quantized activations `[codes, atoms]`, `m` rows.
#[allow(clippy::too_many_arguments)]
fn product(
    ctx: &Ctx,
    op: &'static str,
    fmt: Format,
    [codes, atoms]: [u64; 2],
    m: u32,
    w: Weight,
    y: &mut Tensor,
    alpha: f32,
) -> Result<(), Error> {
    let k = w.k;
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
        tc5::Operand::scaled(codes, a_elem, atoms, fmt.gran()),
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
pub(crate) const MASKED_FIRE: u32 = 64;

/// The fewest tokens an NVFP4 quantization takes a token at a time (fewer:
/// a thread a route, more of them in flight).
const FAN_TOKENS: u32 = 64;

/// A select's experts start whole 128-row scale blocks apart when its
/// activations carry block scales (the atoms it lands are whole blocks).
pub(crate) const BLOCK_ROWS: u32 = 128;
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
        Some(false),
        bank.alpha_from.map(|a| (a, bank.alpha_e)).as_slice(),
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
    let atoms = atoms_of(ctx, op, act_atoms(fmt), atom_bytes)?;
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
    quantize(ctx, op, fmt, src, codes as u64, atoms, atom_bytes)?;
    let grouped = ctx.scratch_stream(op, GROUPED_Y, rows as usize * n as usize * 2)? as u64;
    let a_elem = match fmt {
        Format::Nvfp4 => tc5::Elem::E2m1,
        Format::E4m3 { .. } => tc5::Elem::E4m3,
    };
    tc5::grouped_gemm_epi(
        ctx,
        tc5::Operand::scaled(codes as u64, a_elem, atoms, fmt.gran()),
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
    lay_out_with(ctx, op, routes, experts, align, Some(distinct), &[])
}

/// The most routes and experts [`lay_out_quant_nvfp4`] takes in one launch.
pub(crate) const FUSED_ROUTES: u32 = 1024;
pub(crate) const FUSED_EXPERTS: u32 = 512;

/// [`lay_out_with`] contiguous (each expert's rows `align` apart) and the
/// NVFP4 quantize of `x`'s tokens to their routes' rows (`codes`, atoms in
/// `atoms`) under `input_scales` in one launch, for a fire of at most
/// [`FUSED_ROUTES`] routes over [`FUSED_EXPERTS`] experts; each leg's alpha
/// laid as [`lay_out_with`] lays it. The layout's rows are
/// [`contiguous_rows`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn lay_out_quant_nvfp4(
    ctx: &Ctx,
    op: &'static str,
    routes: Tensor,
    experts: u32,
    align: u32,
    alphas: [(AlphaFrom, u64); 2],
    x: Tensor,
    input_scales: Tensor,
    codes: u64,
    atoms: u64,
) -> Result<Layout, Error> {
    let count = routes.rows * routes.width;
    let k = x.width;
    if count > FUSED_ROUTES
        || experts > FUSED_EXPERTS
        || !k.is_multiple_of(16)
        || x.rows < routes.rows
    {
        return Err(refuse(
            op,
            "the fused layout takes 1024 routes over 512 experts, k a whole number of 16s",
        ));
    }
    let e = experts as usize;
    let i32s = |name, len: usize| -> Result<u64, Error> {
        Ok(ctx.scratch_stream(op, name, len * 4)? as u64)
    };
    let counts = i32s(ROUTE_COUNTS, e)?;
    let offsets = i32s(ROUTE_OFFSETS, e + 1)?;
    let pos = i32s(ROUTE_POS, count as usize)?;
    let int = |v: u32| stated(op, v).map(|v| v.arg());
    let items = count * (k / 16);
    let grid = items
        .div_ceil(256)
        .clamp(1, 4 * ctx.multiprocessors().unwrap_or(148));
    let smem = (33 * experts + count) * 4;
    let [(up, up_alpha), (down, down_alpha)] = alphas;
    ctx.fire(
        op,
        Fire::at(FILE, "::pie::linear::route_lay_out_quant_nvfp4").apply(
            Launch::grid([grid, 1, 1], [256, 1, 1])
                .smem(smem)
                .programmatic(ctx.tuning().pdl),
        ),
        &[
            routes.arg(),
            int(count)?,
            int(routes.width)?,
            ctx.stage(),
            int(experts)?,
            int(align)?,
            ArgValue::Ptr(counts),
            ArgValue::Ptr(offsets),
            ArgValue::Ptr(pos),
            ArgValue::Ptr(up.tensor_scales),
            ArgValue::Ptr(up_alpha),
            int(up.rows)?,
            ArgValue::Ptr(down.tensor_scales),
            ArgValue::Ptr(down.input_scales),
            ArgValue::Ptr(down_alpha),
            int(down.rows)?,
            x.arg(),
            input_scales.arg(),
            ArgValue::Ptr(codes),
            ArgValue::Ptr(atoms),
            int(k)?,
        ],
    )?;
    let rows = contiguous_rows(count, experts, align);
    Ok(Layout {
        pos,
        rows,
        // A router names an expert once a token.
        groups: tc5::Groups::Contiguous {
            offsets,
            rows,
            cap: routes.rows,
        },
        m_max: 0,
        counts,
    })
}

/// The rows a contiguous layout of `count` routes over `experts` holds,
/// each expert's start `align` apart.
pub(crate) fn contiguous_rows(count: u32, experts: u32, align: u32) -> u32 {
    (count + (align - 1) * experts.min(count)).next_multiple_of(align)
}

/// [`lay_out`], laying NVFP4's per-expert alpha too for each leg `alphas`
/// names the sources and destination of (at most two): in the narrow
/// fire's one launch, else apart. `distinct` `None`: contiguous however
/// narrow the fire.
pub(crate) fn lay_out_with(
    ctx: &Ctx,
    op: &'static str,
    routes: Tensor,
    experts: u32,
    align: u32,
    distinct: Option<bool>,
    alphas: &[(AlphaFrom, u64)],
) -> Result<Layout, Error> {
    let tokens = routes.rows;
    let count = tokens * routes.width;
    let m_max = match distinct {
        Some(true) if tokens <= MASKED_FIRE => tokens,
        Some(false) if count <= MASKED_FIRE => count,
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
            let rows = contiguous_rows(count, experts, align);
            Layout {
                pos,
                rows,
                groups: tc5::Groups::Contiguous {
                    offsets,
                    rows,
                    cap: if distinct == Some(true) { tokens } else { 0 },
                },
                m_max,
                counts,
            }
        }
    };
    // A leg's alpha sources and destination, or nulls.
    let leg = |i: usize| {
        alphas.get(i).map_or((0, 0, 0, 0), |&(a, out)| {
            (a.input_scales, a.tensor_scales, out, a.rows)
        })
    };
    // A narrow fire lays out in one block (`route_lay_out_small`): one launch,
    // not a clear and three.
    if count <= SMALL_LAYOUT && alphas.len() <= 2 {
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
                ArgValue::Ptr(leg(0).0),
                ArgValue::Ptr(leg(0).1),
                ArgValue::Ptr(leg(0).2),
                int(leg(0).3)?,
                ArgValue::Ptr(leg(1).0),
                ArgValue::Ptr(leg(1).1),
                ArgValue::Ptr(leg(1).2),
                int(leg(1).3)?,
            ],
        )?;
        return Ok(laid(pos));
    }
    for &(a, out) in alphas {
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
    let vec8 = n.is_multiple_of(8) && grouped.is_multiple_of(16) && y.ptr.is_multiple_of(16);
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn weighted_gather(
    ctx: &Ctx,
    op: &'static str,
    grouped: u64,
    pos: u64,
    weights: Tensor,
    gate: Option<(Tensor, Tensor)>,
    norm: Option<(Tensor, f32)>,
    y: &mut Tensor,
) -> Result<(), Error> {
    let (tokens, n, top) = (y.rows, y.width, weights.width);
    // Under a gate, the shared expert's row and its gate column (a pitch a
    // row); without, nothing.
    let (shared, gate, pitch) = gate.map_or((0, 0, 0), |(s, g)| (s.ptr, g.ptr, g.width));
    let (scale, eps) = norm.map_or((0, 0.0), |(s, eps)| (s.ptr, eps));
    if !n.is_multiple_of(8)
        || !grouped.is_multiple_of(16)
        || !y.ptr.is_multiple_of(16)
        || !scale.is_multiple_of(16)
        || (scale != 0 && !(n / 8).is_multiple_of(32))
        || (scale != 0 && n / 8 > 512)
    {
        return Err(refuse(
            op,
            format!("a weighted gather of {n}-wide rows in 16-byte runs (a block a normed row)"),
        ));
    }
    let int = |v: u32| stated(op, v).map(|v| v.arg());
    // A normed row is a block's.
    let (blocks, block) = if scale != 0 {
        (1, n / 8)
    } else {
        ((n / 8).div_ceil(BLOCK), BLOCK)
    };
    // A normed fire of `kernel.moe.gather_ahead_until` tokens or fewer has
    // eight routes in flight a round (its blocks too few to fill the
    // multiprocessors), else two.
    let ahead = if tokens <= ctx.tuning().moe_gather_ahead_until {
        8
    } else {
        2
    };
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::linear::route_weighted_gather_vec8<{}, {ahead}>",
                scale != 0
            )),
        )
        .apply(Launch::grid([tokens, blocks, 1], [block, 1, 1])),
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
            ArgValue::Ptr(scale),
            eps.arg(),
            ctx.stage(),
        ],
    )
}

/// `y[t] = shared[t] + sum_j grouped[pos[t * top + j]]`, in f32 and in `j`
/// order, rounded once: `moe_sum_add` of the grouped product gathered back
/// (its routes already weighted), read where it lies.
pub(crate) fn sum_gather(
    ctx: &Ctx,
    op: &'static str,
    grouped: u64,
    pos: u64,
    top: u32,
    shared: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    let (tokens, n) = (y.rows, y.width);
    if !n.is_multiple_of(8)
        || !grouped.is_multiple_of(16)
        || !y.ptr.is_multiple_of(16)
        || !shared.ptr.is_multiple_of(16)
        || (shared.rows, shared.width) != (tokens, n)
    {
        return Err(refuse(
            op,
            format!("a summed gather of {n}-wide rows in 16-byte runs"),
        ));
    }
    let int = |v: u32| stated(op, v).map(|v| v.arg());
    ctx.fire(
        op,
        Fire::at(FILE, "::pie::linear::route_weighted_gather_vec8<false>").apply(Launch::grid(
            [tokens, (n / 8).div_ceil(BLOCK), 1],
            [BLOCK, 1, 1],
        )),
        &[
            ArgValue::Ptr(grouped),
            ArgValue::Ptr(pos),
            ArgValue::Ptr(0),
            shared.arg(),
            ArgValue::Ptr(0),
            int(0)?,
            y.arg(),
            int(n)?,
            int(top)?,
            ArgValue::Ptr(0),
            0f32.arg(),
            ctx.stage(),
        ],
    )
}

/// `out[pos[r]] = v[r]` for each of `count` routes: a route-order f32 a
/// route (its weight, say) landed at its row of the layout.
pub(crate) fn scatter_f32(
    ctx: &Ctx,
    op: &'static str,
    v: u64,
    pos: u64,
    count: u32,
    out: u64,
) -> Result<(), Error> {
    let int = |x: u32| stated(op, x).map(|x| x.arg());
    ctx.fire(
        op,
        Fire::at(FILE, "::pie::linear::route_scatter_f32").apply(Launch::flat(count, BLOCK)),
        &[
            ArgValue::Ptr(v),
            ArgValue::Ptr(pos),
            ArgValue::Ptr(out),
            int(count)?,
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
