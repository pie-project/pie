//! Block-FP8 projections against bf16 activations (W8A16) where no FP8 tensor
//! core is (`linear/w8a16.cuh`), the codes widened in registers on mma.sync:
//! a decode's few rows GEMV-shaped, a wider fire's as a tiled GEMM; a
//! prefill's against the plane widened whole to bf16 (`widen`) on cuBLAS.

use dtype::Dtype;

use crate::error::Error;
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, refuse, stated, symbol};
use crate::linear::gemm;
use crate::tensor::Tensor;

const DECODED_WEIGHT: &str = "linear.quant.decoded_weight";
const SPLIT_PARTIALS: &str = "linear.w8a16.split_partials";

/// The most ways the tiled GEMM splits K for a plane too narrow to fill the card.
const MOST_SPLITS: u32 = 16;

const FILE: &str = "linear/w8a16.cuh";

/// The most rows the GEMV-shaped mma.sync kernel takes (K split across
/// warps, the codes streamed once); wider fires take the tiled GEMM.
pub const MMA_UNTIL: u32 = 32;

/// The most rows the tiled GEMM takes; a prefill's wider fires widen the
/// plane once for cuBLAS, which leads there.
pub const GEMM_UNTIL: u32 = 64;

/// One CTA shape of the mma.sync kernel: `16 * mt` rows, `8 * n_sub *
/// warps_n` columns, `warps_k` groups over K.
#[derive(Clone, Copy)]
struct Shape {
    mt: u32,
    n_sub: u32,
    warps_n: u32,
    warps_k: u32,
}

const STAGES: u32 = 4;

impl Shape {
    /// The shape for `rows` of at most [`MMA_UNTIL`].
    const fn for_rows(rows: u32) -> Shape {
        if rows <= 16 {
            Shape {
                mt: 1,
                n_sub: 2,
                warps_n: 1,
                warps_k: 8,
            }
        } else {
            Shape {
                mt: 2,
                n_sub: 2,
                warps_n: 2,
                warps_k: 4,
            }
        }
    }

    const fn columns(self) -> u32 {
        8 * self.n_sub * self.warps_n
    }

    const fn smem(self) -> u32 {
        let ring = self.warps_k * STAGES * 16 * self.mt * 128;
        let fold = if self.warps_k > 1 {
            self.warps_k * 16 * self.mt * self.columns() * 4
        } else {
            0
        };
        if ring > fold { ring } else { fold }
    }
}

/// `y = act . W^T` for an `n x k` e4m3 plane at `codes` under ue8m0 exponents
/// laid in atoms at `atoms`, one a row a `gran` (32 or 128) columns.
/// `Ok(false)` for a shape this point does not read.
pub fn matmul(
    ctx: &Ctx,
    op: &'static str,
    act: Tensor,
    codes: u64,
    atoms: u64,
    gran: u32,
    y: &mut Tensor,
) -> Result<bool, Error> {
    let (m, n, k) = (act.rows, y.width, act.width);
    if act.dtype != Dtype::Bf16
        || y.dtype != Dtype::Bf16
        || !matches!(gran, 32 | 128)
        || k == 0
        || n == 0
        || !k.is_multiple_of(128)
        || !codes.is_multiple_of(16)
        || !act.ptr.is_multiple_of(16)
    {
        return Ok(false);
    }
    if m == 0 {
        return Ok(true);
    }
    // A plane at a batch's rows that tiles too few CTAs to fill the card
    // splits K, the codes still read once.
    let sms = ctx.multiprocessors().unwrap_or(108);
    let tiles = n.div_ceil(128) * m.div_ceil(64);
    let splits = if m > MMA_UNTIL && tiles < 2 * sms && n.is_multiple_of(2) {
        (2 * sms)
            .div_ceil(tiles)
            .min(MOST_SPLITS)
            .min(k / 128)
            .max(1)
    } else {
        1
    };
    if m > GEMM_UNTIL {
        // A prefill's rows against the plane widened once, on cuBLAS.
        let bytes = (n as usize).saturating_mul(k as usize).saturating_mul(2);
        let tile = ctx.scratch_stream(op, DECODED_WEIGHT, bytes)? as usize as u64;
        widen(ctx, op, codes, atoms, gran, n, k, tile)?;
        gemm::act_x_wt(ctx, op, act, Tensor::new(tile, n, k, Dtype::Bf16), None, y)?;
        return Ok(true);
    }
    let args = [
        act.arg(),
        stated(op, k)?.arg(),
        ArgValue::Ptr(codes),
        ArgValue::Ptr(atoms),
        y.arg(),
        stated(op, n)?.arg(),
        stated(op, m)?.arg(),
        stated(op, n)?.arg(),
        stated(op, k)?.arg(),
        ctx.stage(),
    ];
    if m > MMA_UNTIL {
        // The tiled GEMM: 64-row CTAs while that keeps the grid wide, 128 past it.
        let (bm, stages) = if m <= 128 { (64u32, 4u32) } else { (128, 3) };
        // The stages, and the CTA's block exponents laid beside them.
        let smem = stages * (bm * 64 * 2 + 128 * 64) + 128 * (k / gran);
        let part = if splits > 1 {
            let bytes = u64::from(splits) * u64::from(m) * u64::from(n) * 4;
            ctx.scratch_stream(
                op,
                SPLIT_PARTIALS,
                usize::try_from(bytes).unwrap_or(usize::MAX),
            )? as u64
        } else {
            0
        };
        let mut args = args.to_vec();
        args.extend([ArgValue::Ptr(part), stated(op, splits)?.arg()]);
        // Hopper's wgmma, the codes widened once into its A registers.
        let fire = if crate::linear::moe::hopper_wgmma() {
            const WG_STAGES: u32 = 3;
            Fire::at(
                "linear/moe_wgmma.cuh",
                symbol(&format!(
                    "::pie::linear::fp8_w8a16_wgmma<{gran}, {WG_STAGES}>"
                )),
            )
            .apply(
                Launch::grid([n.div_ceil(128), 1, splits], [128, 1, 1])
                    .smem(WG_STAGES * 33792 + 1024),
            )
        } else {
            Fire::at(
                FILE,
                symbol(&format!(
                    "::pie::linear::fp8_w8a16_gemm<{bm}, {stages}, {gran}>"
                )),
            )
            .apply(Launch::grid([n.div_ceil(128), m.div_ceil(bm), splits], [256, 1, 1]).smem(smem))
        };
        ctx.fire(op, fire, &args)?;
        if splits > 1 {
            let pairs = m * n / 2;
            ctx.fire(
                op,
                Fire::at(FILE, symbol("::pie::linear::fp8_w8a16_fold"))
                    .apply(Launch::grid([pairs.div_ceil(256), 1, 1], [256, 1, 1])),
                &[
                    ArgValue::Ptr(part),
                    y.arg(),
                    stated(op, n)?.arg(),
                    stated(op, m)?.arg(),
                    stated(op, n)?.arg(),
                    stated(op, splits)?.arg(),
                    ctx.stage(),
                ],
            )?;
        }
        return Ok(true);
    }
    let fire = {
        let s = Shape::for_rows(m);
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::linear::fp8_w8a16_mma<{}, {}, {}, {}, {STAGES}, {gran}>",
                s.mt, s.n_sub, s.warps_n, s.warps_k
            )),
        )
        .apply(
            Launch::grid(
                [n.div_ceil(s.columns()), m.div_ceil(16 * s.mt), 1],
                [32 * s.warps_n * s.warps_k, 1, 1],
            )
            .smem(s.smem()),
        )
    };
    ctx.fire(op, fire, &args)?;
    Ok(true)
}

/// The plane widened whole to bf16 at `out` (`n x k`).
#[allow(clippy::too_many_arguments)]
pub fn widen(
    ctx: &Ctx,
    op: &'static str,
    codes: u64,
    atoms: u64,
    gran: u32,
    n: u32,
    k: u32,
    out: u64,
) -> Result<(), Error> {
    if !matches!(gran, 32 | 128) || !k.is_multiple_of(gran) || !k.is_multiple_of(16) {
        return Err(refuse(
            op,
            format!(
                "a block-fp8 plane widens in 16-column runs of whole {gran}-wide blocks, not K {k}"
            ),
        ));
    }
    let chunks = u64::from(n) * u64::from(k / 16);
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!("::pie::linear::fp8_atoms_to_bf16<{gran}>")),
        )
        .apply(Launch::grid(
            [
                u32::try_from(chunks.div_ceil(256)).unwrap_or(u32::MAX),
                1,
                1,
            ],
            [256, 1, 1],
        )),
        &[
            ArgValue::Ptr(codes),
            ArgValue::Ptr(atoms),
            ArgValue::Ptr(out),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
        ],
    )
}
