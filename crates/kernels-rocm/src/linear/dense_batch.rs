//! A bf16 projection for a decode batch, `y[m, n] = act[m, k] · w[n, k]ᵀ`
//! (and the fused `silu(act · gateᵀ) · (act · upᵀ)`), on the MoE batch
//! engine (`linear/moe_kit.hip`) as one expert every row routes to: the
//! weight read in whole rows ahead of its use, at the card's streaming rate.
//! Past 64 rows the batch is several items over the same 64-row tile, which
//! run on one XCD at once, so all but the first find the weight in its L2.
//! Refuses what it does not cover.

use crate::error::Error;
use crate::jit::{ArgValue, Ctx, Fire, Launch, aligned16};

const OP: &str = "linear.dense_batch";

/// The XCDs a launch's blocks are dealt round (`kDenseXcds`).
const XCDS: u32 = 8;
/// Output rows a block takes (four waves of 16).
const TILE: u32 = 64;

/// The rows an item takes for a batch of `m`: the least of 4, 8, 16, 32 and
/// 64 covering it, else 64.
const fn item_rows(m: u32) -> u32 {
    match m {
        0..=4 => 4,
        5..=8 => 8,
        9..=16 => 16,
        17..=32 => 32,
        _ => 64,
    }
}

/// Blocks a launch wants: one a CU. Past it, slicing K costs a partial and a
/// sum (and, under SwiGLU, more than the blocks win).
const WANT_BLOCKS: u32 = 304;
/// The most slices of K a launch takes, and the fewest stages a slice.
const MAX_SPLITS: u32 = 8;
const MIN_STAGES: u32 = 4;

/// `y = act · wᵀ` (`swiglu`: `y = silu(act · gateᵀ) · (act · upᵀ)` over a
/// packed `[gate | up]` bank of 2n rows, y n wide). `Ok(false)` for a shape
/// it does not cover. Too few 64-row tiles to fill the card slice K, each
/// slice's partial summed in slice order after (a fixed workspace; the same
/// slices for the fused SwiGLU as for its bank's plain matmul).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dense_batch_bf16(
    ctx: &Ctx,
    w: u64,
    act: u64,
    y: u64,
    m: i32,
    n: i32,
    k: i32,
    swiglu: bool,
) -> Result<bool, Error> {
    if m <= 0 || n <= 0 || k <= 0 || k % 8 != 0 {
        return Ok(false);
    }
    // The weight's byte offsets are 32-bit.
    let rows = i64::from(n) * if swiglu { 2 } else { 1 };
    if rows * i64::from(k) * 2 >= 0xffff_f000 || !aligned16(w) || !aligned16(act) {
        return Ok(false);
    }
    let (mu, nu, ku) = (m.unsigned_abs(), n.unsigned_abs(), k.unsigned_abs());
    let mt = item_rows(mu);
    let items = mu.div_ceil(mt);
    // A SwiGLU tile's 64 bank rows are 32 gate rows and their up rows.
    let tiles = nu.div_ceil(if swiglu { TILE / 2 } else { TILE });
    // A stage is 128 elements of a row short of 32 rows an item, 64 from
    // (the fused SwiGLU slices as its bank's matmul does).
    let stages = ku.div_ceil(if mt >= 32 { 64 } else { 128 });
    let bank_tiles = (rows as u32).div_ceil(TILE);
    let mut splits = 1;
    while splits < MAX_SPLITS
        && bank_tiles * items * splits < WANT_BLOCKS
        && stages / (2 * splits) >= MIN_STAGES
        && nu.is_multiple_of(4)
    {
        splits *= 2;
    }
    let per = stages.div_ceil(splits);
    let part = if splits > 1 {
        let width = if swiglu { 2 * nu } else { nu } as usize;
        ctx.scratch(
            OP,
            "linear.dense_batch_ws",
            splits as usize * mu as usize * width * 4,
        )? as u64
    } else {
        0
    };
    let blocks = tiles.div_ceil(XCDS) * XCDS * items * splits;
    let symbol = format!(
        "::pie::linear::dense_batch_bf16<{mt}, {swiglu}, {}, {}>",
        splits > 1,
        items == 1
    );
    ctx.fire(
        OP,
        Fire::at("linear/moe.hip", crate::jit::symbol(&symbol))
            .apply(Launch::grid([blocks, 1, 1], [256, 1, 1])),
        &[
            ArgValue::Ptr(act),
            ArgValue::Ptr(w),
            ArgValue::Ptr(y),
            if splits > 1 {
                ArgValue::Ptr(part)
            } else {
                ArgValue::ABSENT
            },
            ArgValue::I32(m),
            ArgValue::I32(n),
            ArgValue::I32(k),
            ArgValue::I32(items as i32),
            ArgValue::I32(tiles as i32),
            ArgValue::I32(splits as i32),
            ArgValue::I32(per as i32),
        ],
    )?;
    if splits > 1 {
        let grid = (mu * nu / 4).div_ceil(256).min(4096);
        ctx.fire(
            OP,
            Fire::at(
                "linear/mgemm.hip",
                crate::jit::symbol(&format!("::pie::linear::mgemm_finalize<{swiglu}>")),
            )
            .apply(Launch::grid([grid, 1, 1], [256, 1, 1])),
            &[
                ArgValue::Ptr(part),
                ArgValue::Ptr(y),
                ArgValue::I32(m),
                ArgValue::I32(n),
                ArgValue::I32(splits as i32),
            ],
        )?;
    }
    Ok(true)
}
