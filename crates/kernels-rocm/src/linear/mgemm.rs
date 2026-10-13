//! The decode batch's projection, `y[m, n] = act[m, k] · w[n, k]ᵀ` for
//! 1..=256 rows, on `linear/mgemm.hip`: the weight streamed once into
//! registers, the batch staged through LDS, the matrix cores multiplying.
//! hipBLASLt's tiles leave most of the bandwidth on the table at these
//! shapes. Refuses what it does not cover.

use crate::error::Error;
use crate::jit::{ArgValue, Ctx, Fire, Launch, aligned16};

const OP: &str = "linear.mgemm";

/// The batch rows covered, in 16-row blocks.
pub(crate) const MAX_ROWS: i32 = 256;

const WAVES: u32 = 4;
const PAD: u32 = 8;

/// A launch's tiling.
#[derive(Clone, Copy, Debug)]
struct Plan {
    mb: u32,
    nbw: u32,
    kc: u32,
    depth: u32,
    blocks: u32,
    splits: u32,
    chunks_per_split: u32,
    smem: u32,
}

/// A tiling forced from outside (nbw, kc, depth, splits), for sweeping
/// configurations; zero fields keep the heuristic's choice.
static FORCED: std::sync::Mutex<Option<[u32; 4]>> = std::sync::Mutex::new(None);

pub fn force(tiling: Option<[u32; 4]>) {
    *FORCED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = tiling;
}

fn plan(m: u32, n: u32, k: u32, swiglu: bool) -> Option<Plan> {
    let forced = *FORCED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let pick = |at: usize, default: u32| {
        forced.map_or(default, |f| if f[at] == 0 { default } else { f[at] })
    };
    let mb = m.div_ceil(16);
    // Hand-issued reads fill their registers behind the compiler's back, so
    // a tiling must fit the VGPRs outright: past 256 the compiler copies
    // values to AGPRs, a read still in flight among them.
    let nbw = pick(0, 1).min(if mb <= 8 { 2 } else { 1 });
    let kc = pick(1, if mb <= 8 { 64 } else { 32 });
    let depth = pick(2, 2);
    let rows = WAVES * nbw * 16;
    if !k.is_multiple_of(kc) || !n.is_multiple_of(16) || (swiglu && !n.is_multiple_of(rows / 2)) {
        return None;
    }
    let blocks = if swiglu {
        n / (rows / 2)
    } else {
        n.div_ceil(rows)
    };
    let chunks = k / kc;
    let cus = super::gemv::compute_units();
    // Slice K until the grid covers the device; each slice's f32 partial is
    // written and read once more, so no further.
    let mut splits = 1;
    while blocks * splits < cus && splits * 2 <= chunks && splits < 16 {
        splits *= 2;
    }
    let splits = pick(3, splits).min(chunks);
    let chunks_per_split = chunks.div_ceil(splits);
    let splits = chunks.div_ceil(chunks_per_split);
    let stage = 3 * 16 * mb * (kc + PAD) * 2;
    let exchange = if swiglu { 2 * mb * 64 * 4 * 4 } else { 0 };
    Some(Plan {
        mb,
        nbw,
        kc,
        depth,
        blocks,
        splits,
        chunks_per_split,
        smem: stage.max(exchange),
    })
}

/// `y = act · wᵀ` (`swiglu`: `y = silu(act · gateᵀ) · (act · upᵀ)` over a
/// packed `[gate | up]` bank of 2n rows). `Ok(false)` for a shape it does
/// not cover.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mgemm_bf16(
    ctx: &Ctx,
    w: u64,
    act: u64,
    y: u64,
    m: i32,
    n: i32,
    k: i32,
    swiglu: bool,
) -> Result<bool, Error> {
    if !(1..=MAX_ROWS).contains(&m) || n <= 0 || k <= 0 {
        return Ok(false);
    }
    if !aligned16(w) || !aligned16(act) || !y.is_multiple_of(8) {
        return Ok(false);
    }
    let (mu, nu, ku) = (m.unsigned_abs(), n.unsigned_abs(), k.unsigned_abs());
    let Some(p) = plan(mu, nu, ku, swiglu) else {
        return Ok(false);
    };
    let split = p.splits > 1;
    let ws = if split {
        let width = if swiglu { 2 * nu } else { nu };
        let bytes = p.splits as usize * mu as usize * width as usize * 4;
        ctx.scratch(OP, "linear.mgemm_ws", bytes)? as u64
    } else {
        0
    };
    let symbol = format!(
        "::pie::linear::mgemm_bf16<{}, {}, {}, {}, {split}, {swiglu}>",
        p.mb, p.nbw, p.kc, p.depth
    );
    ctx.fire(
        OP,
        Fire::at("linear/mgemm.hip", crate::jit::symbol(&symbol))
            .apply(Launch::grid([p.blocks, p.splits, 1], [64, WAVES, 1]).smem(p.smem)),
        &[
            ArgValue::Ptr(w),
            ArgValue::Ptr(act),
            ArgValue::Ptr(y),
            if split {
                ArgValue::Ptr(ws)
            } else {
                ArgValue::ABSENT
            },
            ArgValue::I32(m),
            ArgValue::I32(n),
            ArgValue::I32(k),
            ArgValue::I32(p.chunks_per_split as i32),
        ],
    )?;
    if split {
        let cells = mu * nu / 4;
        let grid = cells.div_ceil(256).min(4096);
        let symbol = format!("::pie::linear::mgemm_finalize<{swiglu}>");
        ctx.fire(
            OP,
            Fire::at("linear/mgemm.hip", crate::jit::symbol(&symbol))
                .apply(Launch::grid([grid, 1, 1], [256, 1, 1])),
            &[
                ArgValue::Ptr(ws),
                ArgValue::Ptr(y),
                ArgValue::I32(m),
                ArgValue::I32(n),
                ArgValue::I32(p.splits as i32),
            ],
        )?;
    }
    Ok(true)
}
