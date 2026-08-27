//! `Lora`: the correction class (palo design §8, decision 17).
//!
//! One entry, [`correct`], for one IR variant — `linear.lora_correct` — and it
//! is two launches because the correction is two matmuls with a rank-wide
//! waist between them:
//!
//! ```text
//! t[row, 0..r] = A[routes[row]] · x[row]      ← linear/moe.cuh, verbatim
//! y[row, 0..n] += B[routes[row]] · t[row]     ← linear/lora.cuh, new
//! ```
//!
//! **THE FIRST HALF COMPOSES AND THE SECOND DOES NOT.** A routed projection at
//! fan-out one IS `moe`'s dense select GEMV: same bank stride, same per-route
//! index, same negative-route zero. It is fired through
//! [`moe::select_gemv`](super::moe::select_gemv) under this op's own name, so
//! no second copy of that ladder exists to drift. What has no counterpart is
//! the ACCUMULATE — every routed op in this plane assigns its output row,
//! because a routed expert owns the row it computes, and a correction by
//! definition does not: it adds to a value the trunk already wrote. That is
//! `lora_combine`, and it is the only device text this axis added.
//!
//! # The waist is a scratch slab, and it has to be
//!
//! `t` is `rows x rank` and lives for the two launches between which it is
//! written and read. An entry may not allocate per fire (graph capture forbids
//! it), so it comes from [`Ctx::scratch`] — named, process-global, grown but
//! never shrunk, and warmed by the shell's eager pass before any capture, on
//! the same contract every other scratch-consuming entry here answers to.
//!
//! It carries the ACTIVATION's dtype rather than f32, because that is what the
//! composed projection writes. The arithmetic inside both halves is f32 (the
//! GEMV accumulates in f32 and `lora_combine` reads back through
//! `Elem<T>::to_f32`); what is bf16 is the rank-wide waist, which is `r`
//! numbers per row against the `n` the trunk already rounded. Stated rather
//! than hidden: a fused single-launch form would keep the waist in registers
//! and is what a rank-diverse deployment should measure next.

use kernels::KernelError;
use model_ir::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "linear/lora.cuh";

const BLOCK: u32 = 256;

/// The rank-wide waist between the two halves, per fire.
///
/// One name for one slab: every correction site in a plan writes and consumes
/// it inside its own dispatch, so the layers share it the way they share every
/// other scratch here.
const WAIST: &str = "linear.lora.waist";

/// `y += B[a]·(A[a]·x)` over the rows `routes` gives an adapter.
///
/// `x` is `[rows, in]`, `y` is `[rows, out]`, `bank_a` is `[adapters,
/// rank·in]` and `bank_b` is `[adapters, out·rank]` — the weight table's
/// `rows x width` reading of the `[adapters, rank, in]` and `[adapters, out,
/// rank]` shapes the model text declared. The rank is not an argument: it is
/// `bank_a.width / x.width`, and the two banks are checked against each other,
/// because a rank stated twice is a rank free to disagree with itself.
///
/// # Errors
///
/// [`KernelError::Backend`] for banks whose widths do not divide into a
/// common rank, for a bank pair that names two different adapter counts, or
/// for a fire past the GEMV's grid; [`KernelError::DtypeUnsupported`] for an
/// activation this plane has no instantiation for.
pub fn correct(
    ctx: &Ctx,
    x: Tensor,
    bank_a: Tensor,
    bank_b: Tensor,
    routes: Tensor,
    y: &mut Tensor,
) -> Result<(), KernelError> {
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

    // ── the waist ──────────────────────────────────────────────────────
    // Two bytes an element, and the `dtype_dispatch` above is what says so:
    // the only activations this entry instantiates for are bf16 and f16.
    let bytes = (rows as usize)
        .saturating_mul(rank as usize)
        .saturating_mul(2);
    let waist = ctx.scratch(OP, WAIST, bytes)?;
    let mut projected = Tensor::new(waist as u64, rows, rank, x.dtype);

    // ── half one: the routed projection, somebody else's launch ────────
    //
    // `routes` is `[rows, 1]`, so the select's fan-out is one and its route
    // run is the row run: `t[row] = A[routes[row]] · x[row]`, with a zero row
    // wherever the id is negative. That zero is what an adapterless row inside
    // an adapter window computes, and the combine below returns on it before
    // it reads the bank at all.
    super::moe::select_gemv(ctx, OP, x, bank_a, routes, &mut projected)?;

    // ── half two: the accumulate ───────────────────────────────────────
    let rank_i = stated(OP, rank)?;
    let out_i = stated(OP, out_width)?;
    let stride = i64::from(out_i) * i64::from(rank_i);
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::linear::lora_combine<{t}>")))
            .apply(Launch::grid([out_width.div_ceil(BLOCK), rows, 1], [BLOCK, 1, 1])),
        &[
            routes.arg(),
            projected.arg(),
            bank_b.arg(),
            y.arg(),
            rank_i.arg(),
            out_i.arg(),
            stride.arg(),
        ],
    )
}
