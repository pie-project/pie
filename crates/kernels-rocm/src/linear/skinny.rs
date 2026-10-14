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

/// Whether the kernel's shape takes `m x n x k`: up to `ROWS` rows of K in
/// whole eights (rows past LDS are read from L2).
pub(crate) fn fits(m: i32, n: i32, k: i32) -> bool {
    (1..=ROWS as i32).contains(&m) && n > 0 && k > 0 && k % 8 == 0
}

/// Whether the skinny kernel serves `m x n x k` with at most `rows` rows
/// (`kernel.gemm.skinny_rows`, never past its own). Past two rows an
/// activation beyond LDS is left to the unfused path unless the bank is
/// wide enough for the whole-product kernel: the diagonal kernel's L2
/// rereads lose to hipBLASLt there.
#[must_use]
pub fn covers(m: i32, n: i32, k: i32, epilogue: Epilogue, rows: u32) -> bool {
    #[cfg(feature = "rocm")]
    {
        let bank = i64::from(n) * if epilogue == Epilogue::Geglu { 2 } else { 1 };
        let reread = m > 2
            && 2 * i64::from(m) * i64::from(k) > i64::from(gemv::SKINNY_LDS)
            && bank < i64::from(gemv::QUAD_MIN_ROWS);
        m <= rows.min(ROWS) as i32 && fits(m, n, k) && !reread
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = epilogue;
        m <= rows.min(ROWS) as i32 && fits(m, n, k)
    }
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

/// [`skinny_bf16`] under an explicit walk, for sweeping it:
/// `(yt, unrl, waves, block_waves, grid, act_global)`.
#[doc(hidden)]
#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
pub fn skinny_probe(
    ctx: &Ctx,
    weight: u64,
    act: u64,
    out: u64,
    m: i32,
    n: i32,
    k: i32,
    epilogue: Epilogue,
    walk: (u32, u32, u32, u32, u32, bool),
    quad: bool,
    ks: u32,
) -> Result<(), Error> {
    let epilogue = match epilogue {
        Epilogue::Store => SkinnyEpilogue::Store,
        Epilogue::Geglu => SkinnyEpilogue::Geglu,
        Epilogue::Softcap(cap) => SkinnyEpilogue::Softcap(cap),
    };
    let (yt, unrl, waves, block_waves, grid, act_global) = walk;
    gemv::skinny_planned(
        ctx,
        weight,
        act,
        out,
        m,
        n,
        k,
        epilogue,
        gemv::SkinnyPlan { yt, unrl, waves, block_waves, grid, act_global, quad, ks },
    )
}

/// The walk [`skinny_bf16`] takes for `m x n x k` (`n` output columns):
/// `(yt, unrl, waves, block_waves, grid, act_global, ks, quad)`.
#[doc(hidden)]
#[cfg(feature = "rocm")]
#[must_use]
pub fn skinny_walk(
    m: u32,
    n: u32,
    k: u32,
    epilogue: Epilogue,
) -> (u32, u32, u32, u32, u32, bool, u32, bool) {
    let glu = epilogue == Epilogue::Geglu;
    let rows = if glu { 2 * n } else { n };
    let p = gemv::skinny_plan(m, rows, k, glu, crate::tuning::Tuning::DEFAULT.gemm_skinny_quad_from);
    (p.yt, p.unrl, p.waves, p.block_waves, p.grid, p.act_global, p.ks, p.quad)
}

/// The decode-batch engine's projection (`linear/dense_batch.rs`) forced,
/// for comparing it against the skinny kernels. `Ok(false)` where it does
/// not cover the shape.
#[doc(hidden)]
#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
pub fn dense_batch_probe(
    ctx: &Ctx,
    weight: u64,
    act: u64,
    out: u64,
    m: i32,
    n: i32,
    k: i32,
    swiglu: bool,
) -> Result<bool, Error> {
    crate::linear::dense_batch::dense_batch_bf16(ctx, weight, act, out, m, n, k, swiglu)
}
