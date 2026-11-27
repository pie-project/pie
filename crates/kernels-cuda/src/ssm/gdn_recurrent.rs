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

/// The shared memory a block of the carried step stages a chunk of buffered
/// rows in, double-buffered with its state: [`crate::ssm::CARRY_CHUNK`]
/// rows of a head's keys (at most 128 bf16), corrected values (at most 128
/// f32) and `a` gate word.
const CARRIED_ROWS_SMEM: u32 = 2 * crate::ssm::CARRY_CHUNK * (128 * 2 + 128 * 4 + 4);

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
/// The greatest common divisor of `a` and `b`.
const fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

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
        let (warps, per_sm) = fused_seats(ctx.smem_per_sm(), ctx.tuning().ssm_gdn_step_blocks());
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

/// The gate prep a gated-delta step's `gates` come from: its `[b | a]` rows
/// and its banks, which a carried row's gates are derived from as
/// `gdn_gates` derives them.
#[derive(Clone, Copy, Debug)]
pub struct GdnGated {
    pub ba: Tensor,
    pub dt_bias: Tensor,
    pub a_log: Tensor,
}

/// The gated-delta decode step over each lane's buffered rows then its own
/// ([`Carry`]: planes `[b | a]`, keys, corrected values): the fused step's
/// tile, loaded once, folds the carried rows the lane commits, reads the
/// rest as a linear attention for the own row's corrected value and output,
/// folds the own row where it commits too, and is stored once; the own
/// row's `[b | a]` and corrected value land in the lane's buffer. The fused
/// step's shapes only: another is refused by name.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_carried(
    ctx: &Ctx,
    qkv: Tensor,
    gates: Tensor,
    state: &RecurrentPool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: &mut Tensor,
    gated: GdnGated,
    carry: &Carry,
) -> Result<(), Error> {
    const OP: &str = "ssm.gated_delta";
    dtype_dispatch!(OP, qkv.dtype, { Bf16 => () });
    dtype_dispatch!(OP, gated.ba.dtype, { Bf16 => () });
    dtype_dispatch!(OP, gated.dt_bias.dtype, { Bf16 => () });
    debug_assert_eq!(gates.dtype, Dtype::F32, "`{OP}` reads an f32 decay row");
    debug_assert_eq!(
        gated.a_log.dtype,
        Dtype::F32,
        "`{OP}` reads an f32 decay bank"
    );
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Delta::of(OP, qkv, gates, y, k_heads, v_heads, k_dim, v_dim)?;
    seated(OP, state, "gdn_step_carried", true, false, false)?;
    let rows = state.rows(OP, state.slab, state.slot_stride_elems, v_heads)?;
    if !fused_fits(&shape, state, &rows, qkv, y) {
        return Err(refuse(
            OP,
            format!(
                "a lane's carried rows step in the fused tile only: {k_heads} key heads of \
                 {k_dim} and {v_heads} value heads of {v_dim} do not lie in it"
            ),
        ));
    }
    #[allow(clippy::cast_precision_loss)]
    let q_scale = (k_dim as f32).sqrt().recip();
    let sms = ctx.multiprocessors().unwrap_or(148);
    let tiles = shape
        .n
        .checked_mul(v_heads * (v_dim / FUSED_BV))
        .ok_or_else(|| {
            refuse(
                OP,
                format!("{} rows of value tiles overflow a grid", shape.n),
            )
        })?;
    let (seated_warps, per_sm) = fused_seats(ctx.smem_per_sm(), ctx.tuning().ssm_gdn_step_blocks());
    // A block's warps share one (lane, head)'s buffered rows: they are a
    // whole number of the head's value tiles, so a head of fewer tiles than
    // the seated warps (a narrow head) takes a block of as many as it has.
    let head_tiles = v_dim / FUSED_BV;
    let warps = gcd(seated_warps, head_tiles.max(1));
    if head_tiles == 0 || v_dim > 128 {
        return Err(refuse(
            OP,
            format!(
                "a block stages one head's buffered rows in whole value tiles, and a \
                 {v_dim}-wide head has {head_tiles} of them (at most 128 wide)"
            ),
        ));
    }
    // A splitting fire (room given: some lane carries more than a chunk)
    // launches every block the device seats with the carried step's shared
    // memory, so a unit whose rows run long takes the blocks the units
    // leave idle; any other fire a block a unit at most.
    let carried_smem = fused_smem(warps) + CARRIED_ROWS_SMEM + BLOCK_RESERVED;
    let resident = ctx
        .smem_per_sm()
        .map_or(per_sm, |bytes| (bytes / carried_smem).clamp(1, per_sm));
    let seats = sms * resident;
    let split_room = carry.split != 0
        && u64::from(seats) * CARRY_SPLIT_BYTES_PER_BLOCK <= carry_split_bytes(sms);
    let blocks = if split_room {
        seats
    } else {
        tiles.div_ceil(warps).min(seats)
    };
    let entrypoint = format!("::pie::ssm::gdn_step_carried<{warps}, {split_room}>");
    ctx.fire(
        OP,
        Fire::at(kernel(GDN_RECURRENT, &entrypoint)).apply(
            Launch::grid([blocks, 1, 1], [32 * warps, 1, 1])
                .smem(fused_smem(warps) + CARRIED_ROWS_SMEM),
        ),
        &[
            qkv.arg(),
            gates.arg(),
            gated.ba.arg(),
            gated.dt_bias.arg(),
            gated.a_log.arg(),
            rows.arg(),
            y.arg(),
            stated(OP, k_heads)?.arg(),
            stated(OP, v_heads)?.arg(),
            stated(OP, k_dim)?.arg(),
            stated(OP, v_dim)?.arg(),
            stated(OP, shape.conv_dim)?.arg(),
            q_scale.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            stated(OP, tiles)?.arg(),
            carry.arg(),
            ctx.stage(),
        ],
    )
}

/// The rows a chunked gated-delta fire's lanes land in their buffer past
/// their commit, made ready for a later decode ([`gated_delta_carried`]):
/// each row's corrected value and keys into the lane's buffer (planes 2 and
/// 1; its `[b | a]`, plane 0, the gate prep lands), stepped in order from
/// the state the fire left. `qkv` and `gates` are the fire's own rows by
/// `qo_indptr`, or with `ext` its extended rows (row `RsLane::ext + j`).
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_tail(
    ctx: &Ctx,
    qkv: Tensor,
    gates: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    state: &RecurrentPool,
    qo_indptr: Tensor,
    ext: bool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    carry: &Carry,
) -> Result<(), Error> {
    const OP: &str = "ssm.gated_delta_chunked";
    const WARPS: u32 = 4;
    if !carry.buffers() {
        return Ok(());
    }
    dtype_dispatch!(OP, qkv.dtype, { Bf16 => () });
    let lanes = qo_indptr.rows.saturating_sub(1);
    if lanes == 0 {
        return Ok(());
    }
    if !(k_dim.is_multiple_of(16) && k_dim <= FUSED_BK && v_dim.is_multiple_of(FUSED_BV)) {
        return Err(refuse(
            OP,
            format!(
                "a buffered row's corrected value steps in the fused tile only: key heads of \
                 {k_dim} and value heads of {v_dim} do not lie in it"
            ),
        ));
    }
    let rows = state.rows(OP, state.slab, state.slot_stride_elems, v_heads)?;
    let tiles = v_heads * (v_dim / FUSED_BV);
    let conv_dim = 2 * k_heads * k_dim + v_heads * v_dim;
    ctx.fire(
        OP,
        Fire::at(kernel(GDN_RECURRENT, "::pie::ssm::gdn_tail_corrected<4>")).apply(Launch::grid(
            [tiles.div_ceil(WARPS), lanes, 1],
            [32 * WARPS, 1, 1],
        )),
        &[
            qkv.arg(),
            gates.arg(),
            dt_bias.arg(),
            a_log.arg(),
            rows.arg(),
            qo_indptr.arg(),
            ext.arg(),
            stated(OP, k_heads)?.arg(),
            stated(OP, v_heads)?.arg(),
            stated(OP, k_dim)?.arg(),
            stated(OP, v_dim)?.arg(),
            stated(OP, conv_dim)?.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            stated(OP, lanes)?.arg(),
            carry.arg(),
            ctx.stage(),
        ],
    )
}

/// Folds each beam group's fold rows (`carry`'s
/// planes `[b | a]`, keys, `u`) into its gated-delta state, as a decode step
/// folds a token and stores it, the state stored once.
/// The groups that fold lead `folds` (`groups` entries at most, the first of
/// no rows ending them); a block folds each in turn, so a fire none of whose
/// groups folds costs one read a block.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_fold(
    ctx: &Ctx,
    state: &RecurrentPool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    dt_bias: Tensor,
    a_log: Tensor,
    carry: &Carry,
    folds: Tensor,
    groups: u32,
) -> Result<(), Error> {
    const OP: &str = "ssm.gated_delta group fold";
    const BLOCK: u32 = 256;
    if groups == 0 {
        return Ok(());
    }
    dtype_dispatch!(OP, dt_bias.dtype, { Bf16 => () });
    debug_assert_eq!(a_log.dtype, Dtype::F32, "`{OP}` reads an f32 decay bank");
    if k_dim > 128
        || !k_dim.is_multiple_of(16)
        || !v_dim.is_multiple_of(16)
        || !v_heads.is_multiple_of(k_heads.max(1))
    {
        return Err(refuse(
            OP,
            format!(
                "a fold holds a value tile of a head's {k_dim} key rows (at most 128, in 16s) \
                 and {v_dim}-wide values in 16-column tiles, over {v_heads} value heads \
                 sharing {k_heads} key heads"
            ),
        ));
    }
    let rows = state.rows(OP, state.slab, state.slot_stride_elems, v_heads)?;
    ctx.fire(
        OP,
        Fire::at(fold_kernel()).apply(Launch::grid([v_heads, 1, 1], [BLOCK, 1, 1])),
        &[
            rows.arg(),
            dt_bias.arg(),
            a_log.arg(),
            stated(OP, k_heads)?.arg(),
            stated(OP, v_heads)?.arg(),
            stated(OP, k_dim)?.arg(),
            stated(OP, v_dim)?.arg(),
            carry.arg(),
            folds.arg(),
            stated(OP, groups)?.arg(),
        ],
    )
}

/// The bytes the room to split a decode step's carried rows takes on a
/// device of `sms` multiprocessors: [`CARRY_SPLIT_BYTES_PER_BLOCK`] a block of
/// a grid of every block it seats, at most eight a multiprocessor.
#[must_use]
pub fn carry_split_bytes(sms: u32) -> u64 {
    u64::from(sms) * 8 * CARRY_SPLIT_BYTES_PER_BLOCK
}

/// The gated delta's group fold kernel ([`gdn_fold`]).
pub(super) fn fold_kernel() -> &'static crate::jit::Kernel {
    kernel(GDN_RECURRENT, "::pie::ssm::gdn_fold")
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
