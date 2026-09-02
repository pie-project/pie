//! `Lora`: the correction class. One entry, [`correct`], for
//! `linear.lora_correct` — two launches, two matmuls with a rank-wide waist
//! between them:
//!
//! ```text
//! t[row, 0..r] = A[routes[row]] . x[row]      <- linear/moe.cuh, verbatim
//! y[row, 0..n] += B[routes[row]] . t[row]     <- linear/lora.cuh, new
//! ```
//!
//! The first half is `moe`'s dense select GEMV at fan-out one, fired through
//! [`moe::select_gemv`](super::moe::select_gemv). The second (`lora_combine`)
//! is the accumulate a routed op never needs, since a correction adds to a
//! value the trunk already wrote rather than assigning its own row.
//!
//! The waist `t` (`rows x rank`) lives in [`Ctx::scratch`] between the two
//! launches (an entry may not allocate per fire under graph capture). It
//! carries the activation's dtype (what the projection writes); the
//! arithmetic inside both halves stays f32.

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "linear/lora.cuh";

const BLOCK: u32 = 256;

/// The rank-wide waist between the two halves, per fire. One name for one
/// slab: every correction site in a plan shares it.
const WAIST: &str = "linear.lora.waist";

/// `Fallback::Grouped`, as the few numbers a launch needs for it: a
/// correction whose window covers several intervals rather than one, called
/// once over their union, with segments saying where the intervals are so
/// the combine can skip foreign rows between them.
///
/// `count` is this fire's; `cap` is the artifact's (`engine::fire::max_runs`)
/// and sizes the grid's `z` extent so the grid does not move with the
/// composition — the shell cannot rebind a captured launch's extent, so the
/// blocks between the two return immediately (max-grid plus early exit).
#[derive(Debug, Clone, Copy)]
pub struct Segments {
    /// `[count, 2]` i32, on the device: (first row, row count), ascending,
    /// non-overlapping.
    pub list: Tensor,
    /// How many entries `list` holds this fire.
    pub count: u32,
    /// The load-time bound on `count` — the grid's `z` extent.
    pub cap: u32,
    /// The largest `rows` any entry states — the grid's `y` extent.
    pub max_rows: u32,
}

/// `y += B[a]*(A[a]*x)` over the rows `routes` gives an adapter.
///
/// `x` is `[rows, in]`, `y` is `[rows, out]`, `bank_a` is `[adapters,
/// rank*in]`, `bank_b` is `[adapters, out*rank]`. The rank is not an
/// argument: it is `bank_a.width / x.width`, checked against `bank_b`.
///
/// `segments` is `None` when every row of the rectangle is a row of the
/// correction, `Some` when the rectangle is the union of several intervals
/// and the segments say which rows are actually corrected (see
/// [`Segments`]). The projection half is not told: `select_gemv` runs over
/// the whole rectangle either way, computing a wasted waist row for the
/// gap rows the combine then never reads.
///
/// # Errors
///
/// [`Error::Backend`] for banks whose widths do not divide into a
/// common rank, for a bank pair that names two different adapter counts, or
/// for a fire past the GEMV's grid; [`Error::DtypeUnsupported`] for an
/// activation this plane has no instantiation for.
pub fn correct(
    ctx: &Ctx,
    x: Tensor,
    bank_a: Tensor,
    bank_b: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    segments: Option<Segments>,
) -> Result<(), Error> {
    const OP: &str = "linear.lora_correct";

    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(routes.dtype, Dtype::I32, "`{OP}` walks i32 adapter ids");
    debug_assert_eq!(bank_a.dtype, x.dtype, "the adapter bank rides the activation's dtype");
    debug_assert_eq!(bank_b.dtype, x.dtype, "the adapter bank rides the activation's dtype");

    let rows = nonzero(OP, "rows", x.rows)?;
    let in_width = nonzero(OP, "the correction's input width", x.width)?;
    let out_width = nonzero(OP, "the correction's output width", y.width)?;
    if bank_a.width % in_width != 0 {
        return Err(refuse(
            OP,
            format!(
                "the down bank is {} wide over an input of {in_width}, which is not a \
                 whole number of ranks",
                bank_a.width
            ),
        ));
    }
    let rank = nonzero(OP, "the adapter bank's rank", bank_a.width / in_width)?;
    if bank_b.width != out_width.saturating_mul(rank) {
        return Err(refuse(
            OP,
            format!(
                "the up bank is {} wide where {out_width} x {rank} is {}; the two \
                 planes of one bank state two ranks",
                bank_b.width,
                out_width.saturating_mul(rank),
            ),
        ));
    }
    if bank_a.rows != bank_b.rows {
        return Err(refuse(
            OP,
            format!(
                "the bank's two planes seat {} and {} adapters",
                bank_a.rows, bank_b.rows
            ),
        ));
    }
    debug_assert_eq!(y.rows, x.rows, "a correction lands one row per input row");
    debug_assert_eq!(routes.rows, x.rows, "one adapter id per token row");

    // Two bytes an element: only bf16 and f16 are instantiated above.
    let bytes = (rows as usize)
        .saturating_mul(rank as usize)
        .saturating_mul(2);
    let waist = ctx.scratch(OP, WAIST, bytes)?;
    let mut projected = Tensor::new(waist as u64, rows, rank, x.dtype);

    // `routes` is `[rows, 1]`: fan-out one, `t[row] = A[routes[row]] * x[row]`,
    // zero row where the id is negative (an adapterless row the combine
    // never reads). The adapter bank is resident by construction, so this
    // call is byte-for-byte the launch it was.
    super::moe::select_gemv(
        ctx,
        OP,
        x,
        bank_a,
        routes,
        &mut projected,
        super::moe::ExpertTable::RESIDENT,
    )?;

    let rank_i = stated(OP, rank)?;
    let out_i = stated(OP, out_width)?;
    let stride = i64::from(out_i) * i64::from(rank_i);
    // Without segments: one block row per rectangle row, z of one. With
    // them: (segment, row within segment), at the artifact's bound.
    let (grid, list, segs) = match segments {
        None => ([out_width.div_ceil(BLOCK), rows, 1], ArgValue::ABSENT, 0i32),
        Some(segments) => (
            [
                out_width.div_ceil(BLOCK),
                segments.max_rows.max(1),
                segments.cap.max(segments.count).max(1),
            ],
            segments.list.arg(),
            stated(OP, segments.count)?,
        ),
    };
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::linear::lora_combine<{t}>")))
            .apply(Launch::grid(grid, [BLOCK, 1, 1])),
        &[
            routes.arg(),
            projected.arg(),
            bank_b.arg(),
            y.arg(),
            list,
            segs.arg(),
            rank_i.arg(),
            out_i.arg(),
            stride.arg(),
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
        ],
    )
}
