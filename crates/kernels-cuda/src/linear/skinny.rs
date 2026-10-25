//! The skinny skeleton (`skinny_skeleton`, `linear/contract.cuh`): up to 64 rows
//! on mma.sync, the weight streamed once past the activation's rows in
//! shared memory, its products handed to an epilogue body 8 columns of one
//! row at a time. A packed bank's parts (a GLU's `[gate | up]`) share a
//! block, so each `(m, n)` has every part's product at once.

use crate::error::Error;
use crate::jit::{ArgValue, Ctx, Fire, Kernel, Launch, aligned16, refuse};
use crate::linear::skeleton::{Epilogue, In};

pub const ROWS: u32 = 64;

const WARPS: u32 = 4;
const STAGES: u32 = 2;
const STEP: u32 = 128;

const SMEM_BYTES: u32 = STAGES * (WARPS * 16 + ROWS) * (STEP + 8) * 2;

/// The columns a block lands: its `WARPS * 16` weight rows over the parts.
const fn cols(parts: u32) -> u32 {
    WARPS * 16 / parts
}

/// Whether the skinny skeleton serves `m x n x k` over `parts` packed banks
/// with at most `rows` rows (`kernel.gemm.skinny_rows`, never past its
/// 64-row tile).
#[must_use]
pub fn covers(m: i32, n: i32, k: i32, parts: u32, rows: u32) -> bool {
    if !matches!(parts, 1 | 2 | 4 | 8) {
        return false;
    }
    let cols = cols(parts) as i32;
    m >= 1 && m <= rows.min(ROWS) as i32 && n > 0 && n % cols == 0 && k > 0 && k % STEP as i32 == 0
}

/// The skinny skeleton's kernel with `epi` filled in.
pub(crate) fn kernel(epi: &Epilogue) -> &'static Kernel {
    epi.kernel("skinny_skeleton", &format!("{WARPS}, {STAGES}, {STEP}"))
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
    if !covers(m, n, k, parts, ROWS) {
        return Err(refuse(
            OP,
            format!(
                "covers 1 <= m <= 64, n in whole {}s and k in whole {STEP}s over {parts} part(s), \
                 and was handed m={m}, n={n}, k={k}",
                cols(parts.max(1))
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
    ctx.fire(
        OP,
        Fire::at(kernel(epi)).apply(
            Launch::grid([n.unsigned_abs() / cols(parts), 1, 1], [32 * WARPS, 1, 1])
                .smem(SMEM_BYTES)
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
