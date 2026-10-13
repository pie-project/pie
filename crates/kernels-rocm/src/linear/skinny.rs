//! The decode projection with a fused epilogue (a soft cap, or GEGLU over a
//! packed `[gate | up]` bank), on the streaming skinny kernel of
//! `linear/gemv.hip`: the weight read once past the activation rows in LDS.

use crate::error::Error;
use crate::jit::{Ctx, refuse};
#[cfg(feature = "rocm")]
use crate::linear::gemv::{self, SkinnyEpilogue};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Epilogue {
    Store,
    Softcap(f32),
    Geglu,
}

/// The most activation rows the kernel takes.
pub const ROWS: u32 = 8;

/// The LDS the activation rows may take (one block a compute unit).
const LDS: u32 = 64 << 10;

/// Whether the kernel's shape takes `m x n x k`: up to `ROWS` rows of K in
/// whole eights, the rows within LDS.
pub(crate) fn fits(m: i32, n: i32, k: i32) -> bool {
    (1..=ROWS as i32).contains(&m)
        && n > 0
        && k > 0
        && k % 8 == 0
        && m.unsigned_abs() * k.unsigned_abs() * 2 <= LDS
}

/// Whether the skinny kernel serves `m x n x k` with at most `rows` rows
/// (`kernel.gemm.skinny_rows`, never past its own).
#[must_use]
pub fn covers(m: i32, n: i32, k: i32, _epilogue: Epilogue, rows: u32) -> bool {
    m <= rows.min(ROWS) as i32 && fits(m, n, k)
}

/// `out[m, n] = epilogue(act[m, k] · weight[n, k]ᵀ)`; under `Geglu` the
/// weight is the packed `[gate | up]` bank of 2n rows.
#[allow(clippy::too_many_arguments)]
pub fn skinny_bf16(
    ctx: &Ctx,
    weight: u64,
    act: u64,
    out: u64,
    m: i32,
    n: i32,
    k: i32,
    epilogue: Epilogue,
) -> Result<(), Error> {
    if !covers(m, n, k, epilogue, ROWS) {
        return Err(refuse(
            "linear.skinny",
            format!(
                "covers 1 <= m <= {ROWS} and k in whole eights whose rows fit LDS, and was \
                 handed m={m}, n={n}, k={k} for {epilogue:?}"
            ),
        ));
    }
    if let Epilogue::Softcap(cap) = epilogue
        && !(cap.is_finite() && cap > 0.0)
    {
        return Err(refuse(
            "linear.skinny",
            format!("a softcap of {cap} is not a positive number"),
        ));
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = (ctx, weight, act, out);
        Err(crate::jit::runtimeless("linear.skinny"))
    }
    #[cfg(feature = "rocm")]
    {
        let epilogue = match epilogue {
            Epilogue::Store => SkinnyEpilogue::Store,
            Epilogue::Geglu => SkinnyEpilogue::Geglu,
            Epilogue::Softcap(cap) => SkinnyEpilogue::Softcap(cap),
        };
        gemv::skinny(ctx, weight, act, out, m, n, k, epilogue)
    }
}
