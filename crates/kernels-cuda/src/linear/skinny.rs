//! The skinny skeleton (`skinny_skeleton`, `linear/contract.cuh`): up to 64 rows
//! on mma.sync, the weight streamed once past the activation's rows in
//! shared memory, its products handed to an epilogue body 8 columns of one
//! row at a time. A packed bank's parts (a GLU's `[gate | up]`) share a
//! block, so each `(m, n)` has every part's product at once. A body that
//! combines a head's columns (its span: a head's norm, a rotary pair) takes
//! a block of whole heads, as many warps as that wants (8 for a 128-wide
//! head, 16 for a 256-wide one, over shorter steps of K so the ring fits).

use crate::error::Error;
use crate::jit::{ArgValue, Ctx, Fire, Kernel, Launch, aligned16, refuse};
use crate::linear::skeleton::{Epilogue, In};

pub const ROWS: u32 = 64;

const WARPS: u32 = 4;
const STAGES: u32 = 2;
const STEP: u32 = 128;

/// A block's warps and its step of K for a body of `span` over `parts`:
/// the fewest warps whose columns are whole heads, a step short enough past
/// four warps that the ring fits a block's shared memory.
const fn shape(span: u32, parts: u32) -> Option<(u32, u32)> {
    let mut warps = WARPS;
    while warps <= 16 {
        let cols = warps * 16 / parts;
        if span <= 1 || (cols % span == 0 && span % 8 == 0 && span / 8 <= 32) {
            return Some((warps, if warps == WARPS { STEP } else { 64 }));
        }
        warps *= 2;
    }
    None
}

const fn smem(warps: u32, step: u32) -> u32 {
    STAGES * (warps * 16 + ROWS) * (step + 8) * 2
}

/// The columns a block lands: its `WARPS * 16` weight rows over the parts.
const fn cols(parts: u32) -> u32 {
    WARPS * 16 / parts
}

/// Whether the skinny skeleton serves `m x n x k` over `parts` packed banks
/// with at most `rows` rows (`kernel.gemm.skinny_rows`, never past its
/// 64-row tile).
#[must_use]
pub fn covers(m: i32, n: i32, k: i32, parts: u32, rows: u32) -> bool {
    covers_span(m, n, k, parts, rows, 1)
}

/// [`covers`] for a body that combines the columns of each `span` (1: none).
#[must_use]
pub fn covers_span(m: i32, n: i32, k: i32, parts: u32, rows: u32, span: u32) -> bool {
    if !matches!(parts, 1 | 2 | 4 | 8) {
        return false;
    }
    let Some((warps, step)) = shape(span, parts) else {
        return false;
    };
    let cols = (warps * 16 / parts) as i32;
    m >= 1 && m <= rows.min(ROWS) as i32 && n > 0 && n % cols == 0 && k > 0 && k % step as i32 == 0
}

/// The skinny skeleton's kernel with `epi` filled in.
pub(crate) fn kernel(epi: &Epilogue) -> &'static Kernel {
    let (warps, step) = shape(epi.span(), epi.parts()).unwrap_or((WARPS, STEP));
    epi.kernel(
        "skinny_skeleton",
        &format!("{warps}, {STAGES}, {step}, {}, {}", epi.span(), epi.table()),
    )
}

/// A generated body holds no fields; C++ gives the empty struct one byte.
const NO_FIELDS: u8 = 0;

/// `outs = epi(act . w^T)` over `m` rows (1..=64) on the skinny skeleton: `w`
/// the weight (`parts · n x k`), `ins` the epilogue's inputs and `outs` its
/// results, each `m x n`, every plane 16-byte aligned.
#[allow(clippy::too_many_arguments)]
pub fn skinny(
    ctx: &Ctx,
    epi: &Epilogue,
    weight: u64,
    act: u64,
    ins: &[In],
    outs: &[u64],
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), Error> {
    const OP: &str = "linear.skinny";
    let parts = epi.parts();
    let span = epi.span();
    if !covers_span(m, n, k, parts, ROWS, span) {
        return Err(refuse(
            OP,
            format!(
                "covers 1 <= m <= 64, n in whole {}s and k in whole {STEP}s over {parts} part(s), \
                 and was handed m={m}, n={n}, k={k}",
                cols(parts.max(1)) * shape(span, parts.max(1)).map_or(1, |(w, _)| w / WARPS)
            ),
        ));
    }
    for (address, what) in [(weight, "the weight"), (act, "the activation")]
        .into_iter()
        .chain(outs.iter().map(|&out| (out, "a result")))
    {
        if address == 0 {
            return Err(refuse(OP, format!("{what} is null")));
        }
        if !aligned16(address) {
            return Err(refuse(
                OP,
                format!("{what} is not 16-byte aligned, and cp.async and the drain demand it"),
            ));
        }
    }
    let planes = epi.planes(ins, outs, n.unsigned_abs())?;
    let (warps, step) = shape(span, parts).unwrap_or((WARPS, STEP));
    ctx.fire(
        OP,
        Fire::at(kernel(epi)).apply(
            Launch::grid(
                [n.unsigned_abs() / (warps * 16 / parts), 1, 1],
                [32 * warps, 1, 1],
            )
            .smem(smem(warps, step))
            .programmatic(ctx.tuning().pdl),
        ),
        &[
            ArgValue::Ptr(weight),
            ArgValue::Ptr(act),
            ArgValue::of_slice(&planes),
            ArgValue::of(&NO_FIELDS),
            ArgValue::I32(m),
            ArgValue::I32(n),
            ArgValue::I32(k),
        ],
    )
}
