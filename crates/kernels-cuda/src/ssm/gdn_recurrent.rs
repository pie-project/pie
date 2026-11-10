use super::*;
use crate::jit::kernel;

const GDN_RECURRENT: &str = "ssm/gdn_recurrent.cu";

#[allow(clippy::too_many_arguments)]
pub fn gated_delta(
    ctx: &Ctx,
    qkv: Tensor,
    z: Tensor,
    gates: Tensor,
    state: &RecurrentPool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.gated_delta";

    const SMEM_ARM_WIDTH: u32 = 128;

    const SMEM_BV: u32 = 128;

    let _ = z;
    dtype_dispatch!(OP, qkv.dtype, { Bf16 => () });
    debug_assert_eq!(gates.dtype, Dtype::F32, "`{OP}` reads an f32 decay row");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Delta::of(OP, qkv, gates, y, k_heads, v_heads, k_dim, v_dim)?;
    seated(OP, state, "gdn_step", true, false, false)?;
    let rows = state.rows(OP, state.slab, state.slot_stride_elems, v_heads)?;
    if fused_fits(&shape, state, &rows, qkv, y) {
        return shape.decode_fused(ctx, OP, qkv, gates, state, y);
    }
    let staged = shape.stage(ctx, OP, qkv, gates, StagePlanes::Step)?;

    let (entrypoint, launch) = if v_dim == SMEM_ARM_WIDTH && k_dim == SMEM_ARM_WIDTH {
        (
            "::pie::ssm::gdn_step_smem<::pie::ssm::gqa_smem_bv>",
            Launch::grid([v_dim.div_ceil(SMEM_BV), shape.n, v_heads], [SMEM_BV, 1, 1])
                .smem(k_dim * SMEM_BV * 2 + 2 * k_dim * FLOAT),
        )
    } else {
        (
            "::pie::ssm::gdn_step<::pie::ssm::state_bf16, false>",
            Launch::grid([shape.n, v_heads, 1], [GDN_BLOCK, 1, 1]).smem(2 * k_dim * FLOAT),
        )
    };
    ctx.fire(
        OP,
        Fire::at(kernel(GDN_RECURRENT, entrypoint)).apply(launch),
        &[
            ArgValue::Ptr(staged.q_norm),
            ArgValue::Ptr(staged.k_norm),
            ArgValue::Ptr(staged.v),
            ArgValue::Ptr(staged.g_log),
            ArgValue::Ptr(staged.beta),
            rows.arg(),
            y.arg(),
            stated(OP, k_heads)?.arg(),
            stated(OP, v_heads)?.arg(),
            stated(OP, k_dim)?.arg(),
            stated(OP, v_dim)?.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            ctx.stage(),
        ],
    )
}

/// The value columns one warp of the fused step owns.
const FUSED_BV: u32 = 16;

/// The key rows one warp of the fused step holds: sixteen row pairs, eight
/// rows each.
const FUSED_BK: u32 = 128;

/// Whether the fused step's vector reads land whole: every row it touches
/// starts on a 16-byte boundary and the key width is whole sixteen-row steps.
fn fused_fits(
    shape: &Delta,
    state: &RecurrentPool,
    rows: &StateRows,
    qkv: Tensor,
    y: &Tensor,
) -> bool {
    const VEC: u32 = 8;
    shape.conv_dim.is_multiple_of(VEC)
        && (shape.k_heads * shape.k_dim).is_multiple_of(VEC)
        && shape.v_dim.is_multiple_of(FUSED_BV)
        && shape.k_dim.is_multiple_of(16)
        && rows.keeps(i64::from(VEC))
        && shape.k_dim <= FUSED_BK
        && [qkv.ptr, y.ptr, state.slab.ptr]
            .iter()
            .all(|&at| aligned16(at))
}

/// Warps per block of the fused step: the warps are persistent, so the
/// grid is at most what the SMs seat at once (`kernel.ssm.gdn_step_blocks`
/// blocks an SM at most).
const FUSED_WARPS: u32 = 4;

/// Warps per block of the fused step where the SM's shared memory seats two
/// four-warp blocks (100 KiB: A40, L40S): the same eight warps in one block,
/// which streams the eight value tiles of a head's state rows together where
/// two blocks each streamed half of every row. Qwen3.8-27B's mini at 128
/// rows on an A40: 818 to 780 us a step of a layer.
const FUSED_WARPS_WIDE: u32 = 8;

/// The bytes one warp's stage of the fused step streams: its state tile, the
/// q and k rows, its value slice and the two gates.
const FUSED_STAGE: u32 = 4096 + 512 + 32 + 32;

/// The shared memory the runtime keeps a block.
const BLOCK_RESERVED: u32 = 1024;

/// The fused step's shared memory a block of `warps`, double-buffered.
const fn fused_smem(warps: u32) -> u32 {
    warps * 2 * FUSED_STAGE
}

/// The fused step's blocks of `warps` one SM seats: `most`, or fewer where
/// the SM's shared memory (less the 1 KiB the runtime keeps a block) holds
/// fewer.
fn fused_blocks_per_sm(smem_per_sm: Option<u32>, most: u32, warps: u32) -> u32 {
    let most = most.max(1);
    smem_per_sm.map_or(most, |bytes| {
        (bytes / (fused_smem(warps) + BLOCK_RESERVED)).clamp(1, most)
    })
}

/// The fused step's warps a block and blocks an SM: four-warp blocks, or
/// one eight-warp block where the SM's shared memory holds just two of them.
fn fused_seats(smem_per_sm: Option<u32>, most: u32) -> (u32, u32) {
    let blocks = fused_blocks_per_sm(smem_per_sm, most, FUSED_WARPS);
    let fits_wide = smem_per_sm.is_some_and(|bytes| {
        bytes / (fused_smem(FUSED_WARPS_WIDE) + BLOCK_RESERVED) >= 1
            && bytes / (fused_smem(FUSED_WARPS) + BLOCK_RESERVED) == 2
    });
    if blocks == 2 && fits_wide {
        (FUSED_WARPS_WIDE, 1)
    } else {
        (FUSED_WARPS, blocks)
    }
}

impl Delta {
    fn decode_fused(
        self,
        ctx: &Ctx,
        op: &'static str,
        qkv: Tensor,
        gates: Tensor,
        state: &RecurrentPool,
        y: &mut Tensor,
    ) -> Result<(), Error> {
        #[allow(clippy::cast_precision_loss)]
        let q_scale = (self.k_dim as f32).sqrt().recip();
        let rows = state.rows(op, state.slab, state.slot_stride_elems, self.v_heads)?;
        let sms = ctx.multiprocessors().unwrap_or(148);
        let tiles = self
            .n
            .checked_mul(self.v_heads * (self.v_dim / FUSED_BV))
            .ok_or_else(|| {
                refuse(
                    op,
                    format!("{} rows of value tiles overflow a grid", self.n),
                )
            })?;
        let (warps, per_sm) = fused_seats(ctx.smem_per_sm(), ctx.tuning().ssm_gdn_step_blocks);
        let blocks = tiles.div_ceil(warps).min(sms * per_sm);
        let entrypoint = if warps == FUSED_WARPS_WIDE {
            "::pie::ssm::gdn_step_fused<8>"
        } else {
            "::pie::ssm::gdn_step_fused<4>"
        };
        ctx.fire(
            op,
            Fire::at(kernel(GDN_RECURRENT, entrypoint))
                .apply(Launch::grid([blocks, 1, 1], [32 * warps, 1, 1]).smem(fused_smem(warps))),
            &[
                qkv.arg(),
                gates.arg(),
                rows.arg(),
                y.arg(),
                stated(op, self.k_heads)?.arg(),
                stated(op, self.v_heads)?.arg(),
                stated(op, self.k_dim)?.arg(),
                stated(op, self.v_dim)?.arg(),
                stated(op, self.conv_dim)?.arg(),
                q_scale.arg(),
                state.write_state.arg(),
                state.write_state_mask.arg(),
                stated(op, tiles)?.arg(),
                ctx.stage(),
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Four blocks of the fused step where the SM holds them, as on a B200
    /// (228 KiB) or an A100 (164 KiB); one block of eight warps on a 100 KiB
    /// SM, which holds two of four; the profile's fewer where it asks
    /// (A100's three).
    #[test]
    fn the_fused_step_seats_what_the_sm_holds() {
        assert_eq!(fused_seats(Some(233_472), 4), (4, 4));
        assert_eq!(fused_seats(Some(167_936), 4), (4, 4));
        assert_eq!(fused_seats(Some(102_400), 4), (8, 1));
        assert_eq!(fused_seats(Some(102_400), 3), (8, 1));
        assert_eq!(fused_seats(Some(102_400), 1), (4, 1));
        assert_eq!(fused_seats(Some(16_384), 4), (4, 1));
        assert_eq!(fused_seats(None, 4), (4, 4));
        assert_eq!(fused_seats(Some(167_936), 3), (4, 3));
        assert_eq!(fused_seats(Some(167_936), 2), (4, 2));
    }
}
