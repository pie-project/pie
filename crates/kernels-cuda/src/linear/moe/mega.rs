//! The mega-MoE on tcgen05 (`linear/moe/mega_sm100.cu`): a routed MLP's
//! gate-up product, GLU, down product and route fold in one persistent
//! launch, a CTA an SM. Two forms: DeepSeek-V4's W4A8 experts (MXFP4 banks
//! against e4m3 rows in 1x128 blocks, each route's weight scaling its GLU
//! row, the routes summed with the shared row added), and bf16 banks under
//! either fold. A fire the kernel does not take answers `Ok(false)`; the
//! caller runs the four apart.

use super::{GluBank, RoutedMlp};
use crate::error::Error;
use crate::jit::{ArgValue, Ctx, Fire, Launch, kernel, refuse};
use crate::linear::gemm_sm100::{self, Glu, Map, encode};
use crate::tensor::Tensor;
use cudarc::driver::sys as dr;
use dtype::Dtype;

const FILE: &str = "linear/moe/mega_sm100.cu";
const OP: &str = "fused.moe_mega";

const POOL_X: &str = "linear.moe_mega_x";
const POOL_X_SF: &str = "linear.moe_mega_x_atoms";
const POOL_H: &str = "linear.moe_mega_h";
const POOL_H_SF: &str = "linear.moe_mega_h_atoms";
const ROUTED: &str = "linear.moe_mega_routed";
const SYNC: &str = "linear.moe_mega_sync";

/// `pie::linear::mega::Params`, field for field.
#[repr(C)]
struct Params {
    x: u64,
    routes: u64,
    weights: u64,
    shared: u64,
    y: u64,
    up_atoms: u64,
    down_atoms: u64,
    xpool: u64,
    xpool_sf: u64,
    hpool: u64,
    hpool_sf: u64,
    routed: u64,
    sync: u64,
    win: u64,
    tokens: i32,
    top: i32,
    experts: i32,
    hidden: i32,
    inter: i32,
    limit: f32,
}

/// `pie::linear::mega` bounds: experts, routes a fire, 128-column output
/// tiles (a hidden width of 8192), gate-up tiles a block's ready word.
const MAX_EXPERTS: u32 = 512;
const MAX_ROUTES: u32 = 2048;
const MAX_COL_TILES: u32 = 64;
const MAX_GLU_TILES: u32 = 32;
/// `pie::linear::mega::ws_words`: the flags, zero at rest.
const SYNC_WORDS: usize = 2 * MAX_ROUTES as usize + MAX_COL_TILES as usize + 1;
const THREADS: u32 = 256;

/// `pie::linear::mega::stage_bytes` and `tail_bytes`: a ring stage (two
/// 128 x 128-byte weight boxes, two kBN x 128-byte row boxes, the weights'
/// two scale atoms and the rows' one); the tail (barriers, the amax exchange,
/// the routing: three 513-entry int tables, two 2048-entry short ones and
/// two ints).
const fn stage_bytes(bn: u32) -> u32 {
    2 * 128 * 128 + 2 * bn * 128 + 1024 + 512
}
const ROUTING_BYTES: u32 = 3 * 513 * 4 + 2 * 2048 * 2 + 8;
const fn tail_bytes(bn: u32) -> u32 {
    256 + 16 * bn + ROUTING_BYTES
}
/// The dynamic shared memory a CTA may take (227 KiB).
const SMEM_CAP: u32 = 232_448;

const fn smem_bytes(bn: u32, stages: u32) -> u32 {
    stages * stage_bytes(bn) + tail_bytes(bn) + 1024
}

/// The deepest ring that fits.
const fn stages_for(bn: u32) -> u32 {
    let mut s = 8;
    while s > 2 && smem_bytes(bn, s) > SMEM_CAP {
        s -= 1;
    }
    s
}

/// How a token's routes fold into its row.
#[derive(Clone, Copy, Debug)]
pub enum Fold {
    /// Each route's weight scales its GLU row before the down product
    /// (DeepSeek's); the routes sum plainly, `shared` added where given.
    First(Option<Tensor>),
    /// Each route's down row under its weight, summed.
    Weighted,
}

/// `pie::linear::mega::kPoolStride`: a block's pool rows 64 apart.
const POOL_STRIDE: u32 = 64;

/// The pool rows a fire's blocks of `bn` take: every route its own block at
/// worst, or each expert's last block short, in whole 128-row atoms.
fn pool_rows(count: u32, experts: u32, bn: u32) -> u32 {
    let blocks = count.min(count.div_ceil(bn) + experts.min(count));
    (blocks * POOL_STRIDE).div_ceil(128) * 128
}

/// A K-major `rows x k` operand's map: boxes of 128 bytes of K by `box_rows`,
/// swizzled 128B (an fp4 one landing a byte an element).
fn operand_map(
    ptr: u64,
    w4: bool,
    fp8: bool,
    rows: u32,
    k: u32,
    box_rows: u32,
) -> Result<[u64; 16], Error> {
    use dr::CUtensorMapDataType as T;
    let (ty, stride, box_k) = if w4 {
        (
            T::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
            u64::from(k) / 2,
            128,
        )
    } else if fp8 {
        (T::CU_TENSOR_MAP_DATA_TYPE_UINT8, u64::from(k), 128)
    } else {
        (T::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, u64::from(k) * 2, 64)
    };
    if !ptr.is_multiple_of(16) || !stride.is_multiple_of(16) {
        return Err(refuse(OP, "TMA needs 16-byte rows and base"));
    }
    encode(Map {
        ptr,
        ty,
        dims: [k, rows, 0],
        strides: [stride, 0],
        boxes: [box_k, box_rows, 1],
        swizzle: dr::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
    })
}

/// Whether [`mega`] takes a fire of `routes` (asked before its operands are
/// gathered): tcgen05, and `kernel.moe.mega_until` routes or fewer.
#[must_use]
pub fn mega_takes(ctx: &Ctx, routes: Tensor) -> bool {
    let count = routes.rows * routes.width;
    gemm_sm100::supported() && count > 0 && count <= ctx.tuning().moe_mega_until.min(MAX_ROUTES)
}

/// `y[t] = fold over t's routes r of down_e(glu(up_e . x[t]))`, the routed
/// MLP in one launch. `Ok(false)` where the kernel does not take the fire.
pub fn mega(ctx: &Ctx, m: RoutedMlp, fold: Fold, y: &mut Tensor) -> Result<bool, Error> {
    let RoutedMlp {
        x,
        up,
        glu,
        down,
        routes,
        weights,
        inter,
    } = m;
    let (tokens, top) = (routes.rows, routes.width);
    let count = tokens * top;
    let hidden = x.width;
    if !mega_takes(ctx, routes)
        || x.dtype != Dtype::Bf16
        || x.rows != tokens
        || y.dtype != Dtype::Bf16
        || y.rows != tokens
        || y.width != hidden
        || routes.dtype != Dtype::I32
        || weights.dtype != Dtype::F32
        || weights.rows != tokens
        || weights.width != top
        || top > 32
        || !hidden.is_multiple_of(512)
        || hidden / 128 > MAX_COL_TILES
        || !inter.is_multiple_of(512)
        || inter / 128 > MAX_GLU_TILES
    {
        return Ok(false);
    }
    let shared = match fold {
        Fold::First(Some(s)) => {
            if s.dtype != Dtype::Bf16
                || s.rows != tokens
                || s.width != hidden
                || !s.ptr.is_multiple_of(16)
            {
                return Ok(false);
            }
            s.ptr
        }
        _ => 0,
    };
    let first = matches!(fold, Fold::First(_));
    // (kind, gate-up codes, its atoms, down codes, its atoms, experts)
    let (w4, up_codes, up_atoms, down_codes, down_atoms, experts) = match (up, down) {
        (
            GluBank::Mxfp4A8 {
                codes: uc,
                atoms: ua,
            },
            GluBank::Mxfp4A8 {
                codes: dc,
                atoms: da,
            },
        ) => (true, uc, ua.ptr, dc, da.ptr, uc.rows),
        (GluBank::Bf16(uc), GluBank::Bf16(dc)) => (false, uc, 0, dc, 0, uc.rows),
        _ => return Ok(false),
    };
    let elements = |w: Tensor| u64::from(w.width) * if w4 { 2 } else { 1 };
    let (h, i) = (u64::from(hidden), u64::from(inter));
    if experts == 0
        || experts > MAX_EXPERTS
        || down_codes.rows != experts
        || elements(up_codes) != 2 * i * h
        || elements(down_codes) != h * i
        || (!w4 && (up_codes.dtype != Dtype::Bf16 || down_codes.dtype != Dtype::Bf16))
    {
        return Ok(false);
    }
    // W4A8 lands its GLU rows as the apart e4m3 GLU does: clamped, each
    // under its route's weight.
    let (code, limit) = match glu {
        Glu::SwigluClamp { limit } => (2, limit),
        Glu::Swiglu if !w4 => (1, f32::INFINITY),
        _ => return Ok(false),
    };
    if w4 && !first {
        return Ok(false);
    }
    let Some(sms) = ctx.multiprocessors() else {
        return Ok(false);
    };
    let bn: u32 = 32;
    let stages = stages_for(bn);
    let smem = smem_bytes(bn, stages);
    let pool = pool_rows(count, experts, bn);
    let eb = if w4 { 1 } else { 2 };
    let rows = pool as usize;
    let xpool = ctx.scratch_stream(OP, POOL_X, rows * hidden as usize * eb)? as u64;
    let hpool = ctx.scratch_stream(OP, POOL_H, rows * inter as usize * eb)? as u64;
    let (xpool_sf, hpool_sf) = if w4 {
        (
            ctx.scratch_stream(
                OP,
                POOL_X_SF,
                gemm_sm100::atom_bytes(rows, hidden as usize, 128),
            )? as u64,
            ctx.scratch_stream(
                OP,
                POOL_H_SF,
                gemm_sm100::atom_bytes(rows, inter as usize, 128),
            )? as u64,
        )
    } else {
        (0, 0)
    };
    let routed = ctx.scratch_stream(OP, ROUTED, count as usize * hidden as usize * 2)? as u64;
    let sync = ctx.scratch_stream_zeroed(OP, SYNC, SYNC_WORDS * 4)? as u64;
    let map_up = operand_map(up_codes.ptr, w4, false, experts * 2 * inter, hidden, 128)?;
    let map_down = operand_map(down_codes.ptr, w4, false, experts * hidden, inter, 128)?;
    let map_x = operand_map(xpool, false, w4, pool, hidden, bn)?;
    let map_h = operand_map(hpool, false, w4, pool, inter, bn)?;
    let int = |v: u32| i32::try_from(v).map_err(|_| refuse(OP, "a shape past i32"));
    let win = match ctx.stage() {
        ArgValue::Ptr(at) => at,
        _ => 0,
    };
    let params = Params {
        x: x.ptr,
        routes: routes.ptr,
        weights: weights.ptr,
        shared,
        y: y.ptr,
        up_atoms,
        down_atoms,
        xpool,
        xpool_sf,
        hpool,
        hpool_sf,
        routed,
        sync,
        win,
        tokens: int(tokens)?,
        top: int(top)?,
        experts: int(experts)?,
        hidden: int(hidden)?,
        inter: int(inter)?,
        limit,
    };
    let kind = u32::from(w4);
    let first = u32::from(first);
    ctx.fire(
        OP,
        Fire::at(kernel(
            FILE,
            &format!("::pie::linear::mega_moe<{kind}, {bn}, {code}, {first}, {stages}>"),
        ))
        .apply(
            Launch::grid([sms, 1, 1], [THREADS, 1, 1])
                .smem(smem)
                .cooperative(),
        ),
        &[
            ArgValue::of(&map_up),
            ArgValue::of(&map_down),
            ArgValue::of(&map_x),
            ArgValue::of(&map_h),
            ArgValue::of(&params),
        ],
    )?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_block_width_fits_a_ring() {
        for bn in [32, 64] {
            let stages = stages_for(bn);
            assert!(stages >= 4, "{bn}: {stages}");
            assert!(smem_bytes(bn, stages) <= SMEM_CAP);
        }
        assert_eq!(pool_rows(6, 256, 32), 384);
        assert_eq!(pool_rows(1536, 256, 32), (48 + 256) * 64);
    }
}
