use crate::error::Error;
use dtype::Dtype;

use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, aligned16, dtype_dispatch, nonzero, refuse, stated, symbol,
};
use crate::tensor::{RaggedTensor, RecurrentPool, Tensor};

const FILE: &str = "attn/ssm.cuh";

const BLOCK: u32 = 256;

const WARP: u32 = 32;

const FLOAT: u32 = 4;

const PREP_BLOCK: u32 = 128;

const CONV_VEC: u32 = 8;

const CONV_K_MAX: u32 = 8;

const CONV_BLOCK: u32 = 64;

const GDN_BLOCK: u32 = 128;

const fn kda_shmem(d: u32) -> u32 {
    3u32.saturating_mul(d).saturating_mul(FLOAT)
}

fn requests(op: &'static str, x: RaggedTensor) -> Result<u32, Error> {
    if x.indptr.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the query CSR's boundaries are {:?}, and this scan walks an i32 indptr",
                x.indptr.dtype
            ),
        ));
    }
    match x.indptr.rows.checked_sub(1) {
        Some(lanes) if lanes > 0 => Ok(lanes),
        _ => Err(refuse(op, "the query CSR this fire names spans no request")),
    }
}

#[derive(Clone, Copy)]
enum StagePlanes {
    Step,
    Chunk,
}

impl StagePlanes {
    const fn names(self) -> (&'static str, &'static str, &'static str) {
        match self {
            StagePlanes::Step => (
                "attn.ssm_gdn_step_qk",
                "attn.ssm_gdn_step_v",
                "attn.ssm_gdn_step_gates",
            ),
            StagePlanes::Chunk => (
                "attn.ssm_gdn_chunk_qk",
                "attn.ssm_gdn_chunk_v",
                "attn.ssm_gdn_chunk_gates",
            ),
        }
    }
}

fn plane(ctx: &Ctx, op: &'static str, name: &'static str, elems: u64) -> Result<u64, Error> {
    let bytes = elems.checked_mul(u64::from(FLOAT)).ok_or_else(|| {
        refuse(
            op,
            format!("the {elems}-element staging plane will not size"),
        )
    })?;
    let bytes = usize::try_from(bytes)
        .map_err(|_| refuse(op, format!("{bytes} staging bytes do not fit this host")))?;
    Ok(ctx.scratch_shared(op, name, bytes)? as u64)
}

fn seated(
    op: &'static str,
    state: &RecurrentPool,
    arm: &'static str,
    mask: bool,
    commit: bool,
    begin: bool,
) -> Result<(), Error> {
    if !mask && !state.write_state_mask.is_absent() {
        return Err(refuse(
            op,
            format!(
                "this fire carries a per-request fold predicate and `{arm}` has no seat for \
                 one, so a refused pass would fold anyway"
            ),
        ));
    }
    if !commit && !state.commit_len.is_absent() {
        return Err(refuse(
            op,
            format!(
                "this fire carries a commit length and `{arm}` has no seat for one, so the \
                 replay would fold the whole buffered window instead of its accepted prefix"
            ),
        ));
    }
    if !begin && !state.begin_at.is_absent() {
        return Err(refuse(
            op,
            format!(
                "this fire cuts a row at an interior fold boundary and `{arm}` has no seat for \
                 the segment's origin, so the tail would replay the head's tokens from the state \
                 the head just folded"
            ),
        ));
    }
    if !state.write_state && !mask {
        return Err(refuse(
            op,
            format!("`{arm}` folds its boundary unconditionally and this fire asked it not to"),
        ));
    }
    Ok(())
}

fn conv_extents(
    op: &'static str,
    x: Tensor,
    y: &Tensor,
    conv_width: u32,
) -> Result<(u32, i32, i32), Error> {
    let channels = nonzero(op, "the conv's channel count", x.width)?;
    debug_assert!(
        y.rows == x.rows && y.width == x.width,
        "the conv lands the row it convolves"
    );
    Ok((
        channels,
        stated(op, channels)?,
        stated(
            op,
            nonzero(op, "the conv width this statement states", conv_width)?,
        )?,
    ))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ConvOut {
    Silu,
    Residual,
}

impl ConvOut {
    const fn args(self) -> &'static str {
        match self {
            ConvOut::Silu => "<::pie::bf16>",
            ConvOut::Residual => "<::pie::bf16, false, true>",
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d(
    ctx: &Ctx,
    x: Tensor,
    weight: Tensor,
    bias: Option<Tensor>,
    state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_causal_conv1d";
    conv1d_update(
        ctx,
        OP,
        x,
        weight,
        bias,
        state,
        conv_width,
        dilation,
        y,
        ConvOut::Silu,
    )
}

pub fn short_conv(
    ctx: &Ctx,
    x: Tensor,
    weight: Tensor,
    state: &RecurrentPool,
    conv_width: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.short_conv";
    conv1d_update(
        ctx,
        OP,
        x,
        weight,
        None,
        state,
        conv_width,
        1,
        y,
        ConvOut::Residual,
    )
}

#[allow(clippy::too_many_arguments)]
fn conv1d_update(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    weight: Tensor,
    bias: Option<Tensor>,
    state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    y: &mut Tensor,
    out: ConvOut,
) -> Result<(), Error> {
    dtype_dispatch!(op, x.dtype, { Bf16 => () });
    let (channels, c, k) = conv_extents(op, x, y, conv_width)?;
    let dil = stated(op, nonzero(op, "the conv's dilation", dilation)?)?;
    let rows = nonzero(op, "rows", x.rows)?;
    seated(
        op,
        state,
        "ssm_causal_conv1d_update_batched",
        true,
        false,
        false,
    )?;
    let vectors = channels % CONV_VEC == 0
        && conv_width <= CONV_K_MAX
        && dilation == 1
        && state.conv_stride % i64::from(CONV_VEC) == 0
        && aligned16(x.ptr)
        && aligned16(y.ptr)
        && aligned16(weight.ptr)
        && bias.is_none_or(|b| aligned16(b.ptr))
        && aligned16(state.conv_slab.ptr);
    let mut args = vec![
        x.arg(),
        weight.arg(),
        bias.map_or(ArgValue::ABSENT, |b| b.arg()),
        state.conv_slab.arg(),
        state.slot_ids.arg(),
        state.conv_stride.arg(),
        y.arg(),
        stated(op, rows)?.arg(),
        c.arg(),
        k.arg(),
    ];
    let (entrypoint, launch) = if vectors {
        (
            symbol(&format!(
                "::pie::attn::ssm_causal_conv1d_update_batched_vec8{}",
                out.args()
            )),
            Launch::grid(
                [(channels / CONV_VEC).div_ceil(CONV_BLOCK), rows, 1],
                [CONV_BLOCK, 1, 1],
            ),
        )
    } else {
        args.push(dil.arg());
        (
            symbol(&format!(
                "::pie::attn::ssm_causal_conv1d_update_batched{}",
                out.args()
            )),
            Launch::grid([channels.div_ceil(BLOCK), rows, 1], [BLOCK, 1, 1]),
        )
    };
    args.extend([
        state.write_state.arg(),
        state.write_state_mask.arg(),
        ctx.stage(),
    ]);
    ctx.fire(op, Fire::at(FILE, entrypoint).apply(launch), &args)
}

#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_chunked(
    ctx: &Ctx,
    x: RaggedTensor,
    weight: Tensor,
    bias: Option<Tensor>,
    state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_causal_conv1d_chunked";
    conv1d_chunked(
        ctx,
        OP,
        x,
        weight,
        bias,
        state,
        conv_width,
        dilation,
        y,
        ConvOut::Silu,
    )
}

pub fn short_conv_chunked(
    ctx: &Ctx,
    x: RaggedTensor,
    weight: Tensor,
    state: &RecurrentPool,
    conv_width: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.short_conv_chunked";
    conv1d_chunked(
        ctx,
        OP,
        x,
        weight,
        None,
        state,
        conv_width,
        1,
        y,
        ConvOut::Residual,
    )
}

#[allow(clippy::too_many_arguments)]
fn conv1d_chunked(
    ctx: &Ctx,
    op: &'static str,
    x: RaggedTensor,
    weight: Tensor,
    bias: Option<Tensor>,
    state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    y: &mut Tensor,
    out: ConvOut,
) -> Result<(), Error> {
    const CHANNEL_TILE_FROM: u32 = 8;

    const TILE_BLOCK: u32 = 128;
    const CONV_TT: u32 = 32;

    const PER_CHANNEL_BLOCK: u32 = 64;

    dtype_dispatch!(op, x.data.dtype, { Bf16 => () });
    let (channels, c, k) = conv_extents(op, x.data, y, conv_width)?;
    let dil = stated(op, nonzero(op, "the conv's dilation", dilation)?)?;
    let lanes = requests(op, x)?;
    seated(
        op,
        state,
        "ssm_causal_conv1d_chunked_batched",
        true,
        true,
        true,
    )?;
    let reach = (conv_width.saturating_sub(1)).saturating_mul(dilation);
    let row_tiled = reach <= CONV_TT;
    let vec8 = conv_width == 4
        && dilation == 1
        && channels.is_multiple_of(8)
        && state.conv_stride % 8 == 0
        && [x.data.ptr, y.ptr, weight.ptr, state.conv_slab.ptr]
            .iter()
            .all(|&at| aligned16(at));
    let (entrypoint, launch) = if vec8 && ctx.tuning().ssm_conv_vec4 {
        (
            symbol(&format!(
                "::pie::attn::ssm_causal_conv1d_chunked_batched_row_tile_vec4{}",
                out.args()
            )),
            Launch::grid(
                [
                    (channels / 4).div_ceil(TILE_BLOCK),
                    x.data.rows.div_ceil(CONV_TT).max(1),
                    1,
                ],
                [TILE_BLOCK, 1, 1],
            ),
        )
    } else if vec8 {
        (
            symbol(&format!(
                "::pie::attn::ssm_causal_conv1d_chunked_batched_row_tile_vec8{}",
                out.args()
            )),
            Launch::grid(
                [
                    (channels / 8).div_ceil(TILE_BLOCK),
                    x.data.rows.div_ceil(CONV_TT).max(1),
                    1,
                ],
                [TILE_BLOCK, 1, 1],
            ),
        )
    } else if row_tiled {
        (
            symbol(&format!(
                "::pie::attn::ssm_causal_conv1d_chunked_batched_row_tile{}",
                out.args()
            )),
            Launch::grid(
                [
                    channels.div_ceil(TILE_BLOCK),
                    x.data.rows.div_ceil(CONV_TT).max(1),
                    1,
                ],
                [TILE_BLOCK, 1, 1],
            ),
        )
    } else if lanes >= CHANNEL_TILE_FROM {
        (
            symbol(&format!(
                "::pie::attn::ssm_causal_conv1d_chunked_batched_channel_tile{}",
                out.args()
            )),
            Launch::grid(
                [channels.div_ceil(TILE_BLOCK), lanes, 1],
                [TILE_BLOCK, 1, 1],
            ),
        )
    } else {
        (
            symbol(&format!(
                "::pie::attn::ssm_causal_conv1d_chunked_batched{}",
                out.args()
            )),
            Launch::grid([channels, lanes, 1], [PER_CHANNEL_BLOCK, 1, 1]),
        )
    };
    ctx.fire(
        op,
        Fire::at(FILE, entrypoint).apply(launch),
        &[
            x.data.arg(),
            weight.arg(),
            bias.map_or(ArgValue::ABSENT, |b| b.arg()),
            y.arg(),
            state.conv_slab.arg(),
            state.slot_ids.arg(),
            x.indptr.arg(),
            state.conv_stride.arg(),
            c.arg(),
            k.arg(),
            dil.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            state.commit_len.arg(),
            state.begin_at.arg(),
            ctx.stage(),
            stated(op, lanes)?.arg(),
        ],
    )?;
    if !row_tiled || !state.write_state {
        return Ok(());
    }
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol("::pie::attn::ssm_causal_conv1d_state_commit<::pie::bf16>"),
        )
        .apply(Launch::grid(
            [channels.div_ceil(TILE_BLOCK), lanes, 1],
            [TILE_BLOCK, 1, 1],
        )),
        &[
            x.data.arg(),
            state.conv_slab.arg(),
            state.slot_ids.arg(),
            x.indptr.arg(),
            state.conv_stride.arg(),
            c.arg(),
            k.arg(),
            dil.arg(),
            state.write_state_mask.arg(),
            state.commit_len.arg(),
            state.begin_at.arg(),
            ctx.stage(),
            stated(op, lanes)?.arg(),
        ],
    )
}

pub fn gdn_prep(
    ctx: &Ctx,
    ba: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    gates: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_gdn_prep";
    dtype_dispatch!(OP, ba.dtype, { Bf16 => () });
    debug_assert_eq!(a_log.dtype, Dtype::F32, "`{OP}` reads an f32 decay bank");
    debug_assert_eq!(gates.dtype, Dtype::F32, "`{OP}` lands an f32 decay row");
    if ba.width == 0 || !ba.width.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide `[b | a]` projection does not halve into value heads",
                ba.width
            ),
        ));
    }
    let v_heads = ba.width / 2;
    debug_assert!(
        gates.rows == ba.rows && gates.width == ba.width,
        "the fused `[g_log | beta]` row rides the projection it is derived from"
    );
    let rows = nonzero(OP, "rows", ba.rows)?;
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::attn::ssm_gdn_prep_ba_gates<::pie::bf16>").apply(Launch::grid(
            [rows, v_heads.div_ceil(BLOCK), 1],
            [BLOCK, 1, 1],
        )),
        &[
            ba.arg(),
            a_log.arg(),
            dt_bias.arg(),
            gates.arg(),
            stated(OP, rows)?.arg(),
            stated(OP, v_heads)?.arg(),
            ctx.stage(),
        ],
    )
}

#[derive(Clone, Copy)]
struct Delta {
    n: u32,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    conv_dim: u32,
}

#[derive(Clone, Copy)]
struct DeltaStaged {
    q_norm: u64,
    k_norm: u64,
    v: u64,
    g_log: u64,
    beta: u64,
}

impl Delta {
    #[allow(clippy::too_many_arguments)]
    fn of(
        op: &'static str,
        qkv: Tensor,
        gates: Tensor,
        y: &Tensor,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
    ) -> Result<Self, Error> {
        nonzero(op, "the key heads this statement states", k_heads)?;
        nonzero(op, "the value heads this statement states", v_heads)?;
        nonzero(op, "the key head width this statement states", k_dim)?;
        nonzero(op, "the value head width this statement states", v_dim)?;
        if !v_heads.is_multiple_of(k_heads) {
            return Err(refuse(
                op,
                format!(
                    "the {v_heads} value heads are not a whole number of the {k_heads} key heads"
                ),
            ));
        }
        debug_assert_eq!(
            u64::from(qkv.width),
            2 * u64::from(k_heads) * u64::from(k_dim) + u64::from(v_heads) * u64::from(v_dim),
            "the post-convolution qkv's row is the four stated head numbers"
        );
        debug_assert!(
            gates.rows == qkv.rows && gates.width == 2 * v_heads,
            "the fused `[g_log | beta]` row is two entries per value head"
        );
        debug_assert!(
            y.rows == qkv.rows && u64::from(y.width) == u64::from(v_heads) * u64::from(v_dim),
            "the recurrence lands one value plane per row"
        );
        nonzero(op, "rows", qkv.rows)?;
        Ok(Self {
            n: qkv.rows,
            k_heads,
            v_heads,
            k_dim,
            v_dim,
            conv_dim: qkv.width,
        })
    }

    const fn elems(self, heads: u32, width: u32) -> u64 {
        self.n as u64 * heads as u64 * width as u64
    }

    fn stage(
        self,
        ctx: &Ctx,
        op: &'static str,
        qkv: Tensor,
        gates: Tensor,
        planes: StagePlanes,
    ) -> Result<DeltaStaged, Error> {
        let key = self.elems(self.k_heads, self.k_dim);
        let val = self.elems(self.v_heads, self.v_dim);
        let decay = self.elems(self.v_heads, 1);

        let (qk_name, v_name, gates_name) = planes.names();
        let qk = plane(ctx, op, qk_name, 2 * key)?;
        let v = plane(ctx, op, v_name, val)?;
        let gb = plane(ctx, op, gates_name, 2 * decay)?;
        let staged = DeltaStaged {
            q_norm: qk,
            k_norm: qk + key * u64::from(FLOAT),
            v,
            g_log: gb,
            beta: gb + decay * u64::from(FLOAT),
        };

        #[allow(clippy::cast_precision_loss)]
        let q_scale = (self.k_dim as f32).sqrt().recip();
        ctx.fire(
            op,
            Fire::at(FILE, "::pie::attn::ssm_gdn_prep_qk_norm<::pie::bf16, 128>")
                .apply(Launch::grid([self.n, self.k_heads, 1], [PREP_BLOCK, 1, 1])),
            &[
                qkv.arg(),
                ArgValue::Ptr(staged.q_norm),
                ArgValue::Ptr(staged.k_norm),
                stated(op, self.k_heads)?.arg(),
                stated(op, self.k_dim)?.arg(),
                stated(op, self.conv_dim)?.arg(),
                q_scale.arg(),
                ctx.stage(),
            ],
        )?;
        ctx.fire(
            op,
            Fire::at(FILE, "::pie::attn::ssm_gdn_prep_v_gates<::pie::bf16, 128>")
                .apply(Launch::grid([self.n, self.v_heads, 1], [PREP_BLOCK, 1, 1])),
            &[
                qkv.arg(),
                gates.arg(),
                ArgValue::Ptr(staged.v),
                ArgValue::Ptr(staged.g_log),
                ArgValue::Ptr(staged.beta),
                stated(op, self.k_heads)?.arg(),
                stated(op, self.v_heads)?.arg(),
                stated(op, self.k_dim)?.arg(),
                stated(op, self.v_dim)?.arg(),
                stated(op, self.conv_dim)?.arg(),
                ctx.stage(),
            ],
        )?;
        Ok(staged)
    }
}

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
    const OP: &str = "attention.ssm_gated_delta";

    const SMEM_ARM_WIDTH: u32 = 128;

    const SMEM_BV: u32 = 128;

    let _ = z;
    dtype_dispatch!(OP, qkv.dtype, { Bf16 => () });
    debug_assert_eq!(gates.dtype, Dtype::F32, "`{OP}` reads an f32 decay row");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Delta::of(OP, qkv, gates, y, k_heads, v_heads, k_dim, v_dim)?;
    seated(
        OP,
        state,
        "ssm_gated_delta_step_batched_gqa",
        true,
        false,
        false,
    )?;
    if fused_fits(&shape, state, qkv, y) {
        return shape.decode_fused(ctx, OP, qkv, gates, state, y);
    }
    let staged = shape.stage(ctx, OP, qkv, gates, StagePlanes::Step)?;

    let (entrypoint, launch) = if v_dim == SMEM_ARM_WIDTH && k_dim == SMEM_ARM_WIDTH {
        (
            "::pie::attn::ssm_gated_delta_step_batched_gqa_smem<::pie::attn::gqa_smem_bv>",
            Launch::grid([v_dim.div_ceil(SMEM_BV), shape.n, v_heads], [SMEM_BV, 1, 1])
                .smem(k_dim * SMEM_BV * 2 + 2 * k_dim * FLOAT),
        )
    } else {
        (
            "::pie::attn::ssm_gated_delta_step_batched_gqa<::pie::attn::state_bf16, false>",
            Launch::grid([shape.n, v_heads, 1], [GDN_BLOCK, 1, 1]).smem(2 * k_dim * FLOAT),
        )
    };
    ctx.fire(
        OP,
        Fire::at(FILE, entrypoint).apply(launch),
        &[
            ArgValue::Ptr(staged.q_norm),
            ArgValue::Ptr(staged.k_norm),
            ArgValue::Ptr(staged.v),
            ArgValue::Ptr(staged.g_log),
            ArgValue::Ptr(staged.beta),
            state.slab.arg(),
            state.slot_ids.arg(),
            state.slot_stride_elems.arg(),
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
fn fused_fits(shape: &Delta, state: &RecurrentPool, qkv: Tensor, y: &Tensor) -> bool {
    const VEC: u32 = 8;
    shape.conv_dim.is_multiple_of(VEC)
        && (shape.k_heads * shape.k_dim).is_multiple_of(VEC)
        && shape.v_dim.is_multiple_of(FUSED_BV)
        && shape.k_dim.is_multiple_of(16)
        && state.slot_stride_elems % i64::from(VEC) == 0
        && shape.k_dim <= FUSED_BK
        && [qkv.ptr, y.ptr, state.slab.ptr]
            .iter()
            .all(|&at| aligned16(at))
}

/// Warps per block of the fused step, and the most blocks each SM holds:
/// the warps are persistent, so the grid is at most what the SMs seat at
/// once.
const FUSED_WARPS: u32 = 4;
const FUSED_BLOCKS_PER_SM: u32 = 3;
/// The bytes one warp's stage of the fused step streams: its state tile, the
/// q and k rows, its value slice and the two gates.
const FUSED_STAGE: u32 = 4096 + 512 + 32 + 32;
/// The fused step's shared memory a block, double-buffered.
const FUSED_SMEM: u32 = FUSED_WARPS * 2 * FUSED_STAGE;

/// The fused step's blocks one SM seats: three, or fewer where the SM's
/// shared memory (less the 1 KiB the runtime keeps a block) holds fewer.
fn fused_blocks_per_sm(smem_per_sm: Option<u32>) -> u32 {
    const RESERVED: u32 = 1024;
    smem_per_sm.map_or(FUSED_BLOCKS_PER_SM, |bytes| {
        (bytes / (FUSED_SMEM + RESERVED)).clamp(1, FUSED_BLOCKS_PER_SM)
    })
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
        let blocks = tiles
            .div_ceil(FUSED_WARPS)
            .min(sms * fused_blocks_per_sm(ctx.smem_per_sm()));
        ctx.fire(
            op,
            Fire::at(FILE, "::pie::attn::ssm_gdn_decode_step<4>")
                .apply(Launch::grid([blocks, 1, 1], [32 * FUSED_WARPS, 1, 1]).smem(FUSED_SMEM)),
            &[
                qkv.arg(),
                gates.arg(),
                state.slab.arg(),
                state.slot_ids.arg(),
                state.slot_stride_elems.arg(),
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

/// The value columns one block of the chunked scan carries: half a head,
/// so a lone long request still spreads over twice its heads' blocks. A
/// captured fire states its bucket's lanes, not the live ones, so the width
/// cannot follow the batch.
const SCAN_BV: u32 = 64;

/// The shared memory one block of the chunked scan holds at key width
/// `k_dim`: its prep is smaller.
const fn scan_smem(k_dim: u32) -> u32 {
    const CH: u32 = 64;
    const LD: u32 = 136;
    const LDC: u32 = 72;
    let ldv = SCAN_BV + 8;
    (3 * CH * LD + CH * ldv + CH * LDC + k_dim * ldv + 2 * CH * ldv) * 2 + 2 * CH * FLOAT
}

/// Whether a 128-wide prefill takes the two-launch scan: the profile turns
/// it on, and this device's blocks can opt into its shared memory.
fn fused_scan(ctx: &Ctx, k_dim: u32) -> bool {
    ctx.tuning().ssm_gdn_fused_scan
        && ctx
            .smem_per_block_optin()
            .is_some_and(|cap| cap >= scan_smem(k_dim))
}

/// `pie::attn::gdn::Params`, field for field.
#[cfg(feature = "cuda")]
#[repr(C)]
struct Tc5Params {
    gates: u64,
    state: u64,
    slot_ids: u64,
    qo_indptr: u64,
    commit_len: u64,
    begin_at: u64,
    write_state_mask: u64,
    out: u64,
    win: u64,
    slot_stride: i64,
    lanes: i32,
    k_heads: i32,
    v_heads: i32,
    write_state: i32,
    q_scale: f32,
    out_bf16: i32,
}

/// The one-launch tcgen05 walk (`attn/gdn_tc5.cuh`), where the profile
/// asks for it and the device and operands admit it: 128-wide heads, rows
/// TMA can read. `Ok(false)` leaves the prefill to the scans.
#[cfg(feature = "cuda")]
fn gated_delta_chunked_tc5(
    ctx: &Ctx,
    op: &'static str,
    shape: &Delta,
    qkv: RaggedTensor,
    gates: Tensor,
    state: &RecurrentPool,
    y: &mut Tensor,
) -> Result<bool, Error> {
    use crate::linear::tc5::{Map, encode, supported};
    use cudarc::driver::sys as dr;
    const FILE_TC5: &str = "attn/gdn_tc5.cuh";
    /// `pie::attn::gdn::smem_bytes()`.
    const SMEM: u32 = 213_504 + 1024;
    if !(ctx.tuning().ssm_gdn_tc5
        && supported()
        && shape.k_dim == 128
        && shape.v_dim == 128
        && shape.v_heads.is_multiple_of(shape.k_heads.max(1))
        && (shape.conv_dim * 2).is_multiple_of(16)
        && aligned16(qkv.data.ptr)
        && aligned16(y.ptr)
        && state.slot_stride_elems > 0)
    {
        return Ok(false);
    }
    let lanes = requests(op, qkv)?;
    seated(op, state, "gdn_chunked_tc5", true, true, true)?;
    let map = crate::attn::bf16_map(
        qkv.data.ptr,
        [shape.conv_dim, qkv.data.rows, 1],
        [
            u64::from(shape.conv_dim) * 2,
            u64::from(shape.conv_dim) * 2 * u64::from(qkv.data.rows.max(1)),
        ],
        [64, 64, 1],
    )?;
    // A whole chunk's output, 64 rows of a value head, stored by TMA, in
    // bf16 where the trace narrowed it for its gated norm.
    let out_bf16 = y.dtype == Dtype::Bf16;
    let out_bytes: u64 = if out_bf16 { 2 } else { 4 };
    let out_width = shape.v_heads * shape.v_dim;
    let map_out = encode(Map {
        ptr: y.ptr,
        ty: if out_bf16 {
            dr::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
        } else {
            dr::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32
        },
        dims: [out_width, y.rows, 1],
        strides: [
            u64::from(out_width) * out_bytes,
            u64::from(out_width) * out_bytes * u64::from(y.rows.max(1)),
        ],
        boxes: [128, 64, 1],
        swizzle: dr::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    })?;
    #[allow(clippy::cast_precision_loss)]
    let q_scale = (shape.k_dim as f32).sqrt().recip();
    let int = |v: u32| i32::try_from(v).unwrap_or(i32::MAX);
    let ArgValue::Ptr(win) = ctx.stage() else {
        return Ok(false);
    };
    let params = Tc5Params {
        gates: gates.ptr,
        state: state.slab.ptr,
        slot_ids: state.slot_ids.ptr,
        qo_indptr: qkv.indptr.ptr,
        commit_len: state.commit_len.ptr,
        begin_at: state.begin_at.ptr,
        write_state_mask: state.write_state_mask.ptr,
        out: y.ptr,
        win,
        slot_stride: state.slot_stride_elems,
        lanes: int(lanes),
        k_heads: int(shape.k_heads),
        v_heads: int(shape.v_heads),
        write_state: i32::from(state.write_state),
        q_scale,
        out_bf16: i32::from(out_bf16),
    };
    let tiles = lanes.saturating_mul(shape.v_heads);
    let grid = tiles.min(ctx.multiprocessors().unwrap_or(148)).max(1);
    ctx.fire(
        op,
        Fire::at(FILE_TC5, symbol("::pie::attn::gdn::gdn_chunked_tc5"))
            .apply(Launch::grid([grid, 1, 1], [384, 1, 1]).smem(SMEM)),
        &[
            ArgValue::of(&map),
            ArgValue::of(&map_out),
            ArgValue::of(&params),
        ],
    )?;
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_chunked_fla(
    ctx: &Ctx,
    op: &'static str,
    shape: &Delta,
    qkv: RaggedTensor,
    gates: Tensor,
    state: &RecurrentPool,
    y: &mut Tensor,
) -> Result<(), Error> {
    const CH: u32 = 64;
    const BLOCK: u32 = 128;
    const LD: u32 = 136;
    const LDC: u32 = 72;
    const LDA: u32 = 68;
    let lanes = requests(op, qkv)?;
    seated(op, state, "ssm_gdn_chunk", true, true, true)?;
    let (k_dim, v_dim, v_heads, k_heads) = (shape.k_dim, shape.v_dim, shape.v_heads, shape.k_heads);
    let slots = shape.n.div_ceil(CH) + lanes;
    let padded = Delta {
        n: shape.n + CH,
        ..*shape
    };
    let gcum = plane(ctx, op, "attn.ssm_gdn_chunk_gcum", padded.elems(v_heads, 1))?;
    let qn = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_qn",
        padded.elems(k_heads, k_dim),
    )?;
    let kn = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_kn",
        padded.elems(k_heads, k_dim),
    )?;
    let w = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_w",
        padded.elems(v_heads, k_dim),
    )?;
    let u = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_u",
        padded.elems(v_heads, v_dim),
    )?;
    let pm = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_p",
        u64::from(slots) * u64::from(v_heads) * u64::from(CH * CH),
    )?;
    #[allow(clippy::cast_precision_loss)]
    let q_scale = (k_dim as f32).sqrt().recip();
    let prep_smem =
        2 * CH * LD * 2 + 2 * CH * LDC * 2 + CH * LDA * FLOAT + 3 * CH * FLOAT + 4 * 256 * FLOAT;
    ctx.fire(
        op,
        Fire::at(FILE, "::pie::attn::ssm_gdn_chunk_prep<128, 128>")
            .apply(Launch::grid([v_heads, slots, 1], [2 * BLOCK, 1, 1]).smem(prep_smem)),
        &[
            qkv.data.arg(),
            gates.arg(),
            state.slot_ids.arg(),
            qkv.indptr.arg(),
            state.commit_len.arg(),
            state.begin_at.arg(),
            ArgValue::Ptr(gcum),
            ArgValue::Ptr(qn),
            ArgValue::Ptr(kn),
            ArgValue::Ptr(w),
            ArgValue::Ptr(u),
            ArgValue::Ptr(pm),
            stated(op, lanes)?.arg(),
            stated(op, k_heads)?.arg(),
            stated(op, v_heads)?.arg(),
            stated(op, shape.conv_dim)?.arg(),
            q_scale.arg(),
            ctx.stage(),
        ],
    )?;
    let scan_smem = scan_smem(k_dim);
    ctx.fire(
        op,
        Fire::at(FILE, "::pie::attn::ssm_chunk_scan<64>").apply(
            Launch::grid(
                [lanes * v_heads * (v_dim / SCAN_BV), 1, 1],
                [2 * BLOCK, 1, 1],
            )
            .smem(scan_smem),
        ),
        &[
            ArgValue::Ptr(gcum),
            ArgValue::Ptr(qn),
            ArgValue::Ptr(kn),
            ArgValue::Ptr(w),
            ArgValue::Ptr(u),
            ArgValue::Ptr(pm),
            state.slab.arg(),
            state.slot_ids.arg(),
            qkv.indptr.arg(),
            state.slot_stride_elems.arg(),
            state.commit_len.arg(),
            state.begin_at.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            y.arg(),
            stated(op, lanes)?.arg(),
            stated(op, k_heads)?.arg(),
            stated(op, v_heads)?.arg(),
            ctx.stage(),
        ],
    )
}

/// The chunked rule in three wmma launches over the staged planes: the
/// update directions per chunk, the state walk per sixteen value columns,
/// and the output per chunk. No block holds more than 41 KiB of shared
/// memory, so every card runs it.
#[allow(clippy::too_many_arguments)]
fn gated_delta_chunked_split(
    ctx: &Ctx,
    op: &'static str,
    shape: &Delta,
    staged: &DeltaStaged,
    qkv: RaggedTensor,
    state: &RecurrentPool,
    y: &mut Tensor,
) -> Result<(), Error> {
    const CH: u32 = 64;
    const BVT: u32 = 16;
    const BLOCK: u32 = 256;
    let lanes = requests(op, qkv)?;
    seated(op, state, "ssm_gdn_chunk", true, true, true)?;
    let (k_dim, v_dim, v_heads, k_heads) = (shape.k_dim, shape.v_dim, shape.v_heads, shape.k_heads);
    let slots = shape.n.div_ceil(CH) + lanes;
    let padded = Delta {
        n: shape.n + CH,
        ..*shape
    };
    let per_slot = |heads: u32, width: u32| u64::from(slots) * u64::from(heads) * u64::from(width);
    let gcum = plane(ctx, op, "attn.ssm_gdn_chunk_gcum", shape.elems(v_heads, 1))?;
    let qb = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_qb",
        padded.elems(k_heads, k_dim),
    )?;
    let kb = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_kb",
        padded.elems(k_heads, k_dim),
    )?;
    let kt = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_kt",
        per_slot(k_heads, k_dim * CH),
    )?;
    let w = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_w",
        padded.elems(v_heads, k_dim),
    )?;
    let u = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_u",
        padded.elems(v_heads, v_dim),
    )?;
    let vnt = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_vnt",
        per_slot(v_heads, v_dim * CH),
    )?;
    let hct = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_hct",
        per_slot(v_heads, v_dim * k_dim),
    )?;
    let wu_smem = CH * CH * FLOAT + CH * k_dim * 2 + CH * CH * 2 + 4 * CH * FLOAT;
    ctx.fire(
        op,
        Fire::at(FILE, "::pie::attn::ssm_gdn_chunk_wu<128, 128>")
            .apply(Launch::grid([slots, v_heads, 1], [BLOCK, 1, 1]).smem(wu_smem)),
        &[
            ArgValue::Ptr(staged.q_norm),
            ArgValue::Ptr(staged.k_norm),
            ArgValue::Ptr(staged.v),
            ArgValue::Ptr(staged.g_log),
            ArgValue::Ptr(staged.beta),
            state.slot_ids.arg(),
            qkv.indptr.arg(),
            state.commit_len.arg(),
            state.begin_at.arg(),
            ArgValue::Ptr(gcum),
            ArgValue::Ptr(qb),
            ArgValue::Ptr(kb),
            ArgValue::Ptr(kt),
            ArgValue::Ptr(w),
            ArgValue::Ptr(u),
            stated(op, lanes)?.arg(),
            stated(op, k_heads)?.arg(),
            stated(op, v_heads)?.arg(),
            ctx.stage(),
        ],
    )?;
    ctx.fire(
        op,
        Fire::at(
            FILE,
            "::pie::attn::ssm_gdn_chunk_h<::pie::attn::state_bf16, 128, 128, 16>",
        )
        .apply(Launch::grid(
            [lanes * v_heads * (v_dim / BVT), 1, 1],
            [BLOCK, 1, 1],
        )),
        &[
            ArgValue::Ptr(gcum),
            ArgValue::Ptr(kt),
            ArgValue::Ptr(w),
            ArgValue::Ptr(u),
            ArgValue::Ptr(vnt),
            ArgValue::Ptr(hct),
            state.slab.arg(),
            state.slot_ids.arg(),
            qkv.indptr.arg(),
            state.slot_stride_elems.arg(),
            state.commit_len.arg(),
            state.begin_at.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            stated(op, lanes)?.arg(),
            stated(op, k_heads)?.arg(),
            stated(op, v_heads)?.arg(),
            ctx.stage(),
        ],
    )?;
    ctx.fire(
        op,
        Fire::at(FILE, "::pie::attn::ssm_gdn_chunk_o<128, 128, 16>")
            .apply(Launch::grid([slots, v_heads, 1], [BLOCK, 1, 1])),
        &[
            ArgValue::Ptr(gcum),
            ArgValue::Ptr(qb),
            ArgValue::Ptr(kb),
            ArgValue::Ptr(vnt),
            ArgValue::Ptr(hct),
            state.slot_ids.arg(),
            qkv.indptr.arg(),
            state.commit_len.arg(),
            state.begin_at.arg(),
            y.arg(),
            stated(op, lanes)?.arg(),
            stated(op, k_heads)?.arg(),
            stated(op, v_heads)?.arg(),
            ctx.stage(),
        ],
    )
}

fn plane_bf16(ctx: &Ctx, op: &'static str, name: &'static str, elems: u64) -> Result<u64, Error> {
    let bytes = usize::try_from(elems.saturating_mul(2))
        .map_err(|_| refuse(op, format!("{elems} bf16 elements do not fit this host")))?;
    Ok(ctx.scratch_shared(op, name, bytes)? as u64)
}

#[allow(clippy::too_many_arguments)]
pub fn gated_delta_chunked_serial(
    ctx: &Ctx,
    qkv: RaggedTensor,
    z: Tensor,
    gates: Tensor,
    state: &RecurrentPool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    SERIAL_SCAN.with(|flag| flag.set(true));
    let out = gated_delta_chunked(ctx, qkv, z, gates, state, k_heads, v_heads, k_dim, v_dim, y);
    SERIAL_SCAN.with(|flag| flag.set(false));
    out
}

thread_local! {
    static SERIAL_SCAN: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

#[allow(clippy::too_many_arguments)]
pub fn gated_delta_chunked(
    ctx: &Ctx,
    qkv: RaggedTensor,
    z: Tensor,
    gates: Tensor,
    state: &RecurrentPool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_gated_delta_chunked";

    const BK_MAX_FLA: u32 = 128;

    const BV_FLA: u32 = 128;

    const WARP_TILED_K_MAX: u32 = 256;

    const SCAN_WARPS: u32 = 4;

    let _ = z;
    dtype_dispatch!(OP, qkv.data.dtype, { Bf16 => () });
    debug_assert_eq!(gates.dtype, Dtype::F32, "`{OP}` reads an f32 decay row");
    if y.dtype == Dtype::Bf16 {
        // Narrowed for its gated norm: the tcgen05 walk lands bf16 itself;
        // any other walk lands f32 in a plane, cast after.
        let wide_y = Tensor::new(y.ptr, y.rows, y.width, Dtype::F32);
        let shape = Delta::of(OP, qkv.data, gates, &wide_y, k_heads, v_heads, k_dim, v_dim)?;
        #[cfg(feature = "cuda")]
        if k_dim == 128
            && v_dim == 128
            && !SERIAL_SCAN.with(core::cell::Cell::get)
            && gated_delta_chunked_tc5(ctx, OP, &shape, qkv, gates, state, y)?
        {
            return Ok(());
        }
        let _ = shape;
        // A staged window lands its rows past its base, within twice the
        // fire's rows.
        let plane_at = plane(ctx, OP, "attn.ssm_gdn_wide_out", 2 * y.elements())?;
        let mut wide = Tensor::new(plane_at, y.rows, y.width, Dtype::F32);
        gated_delta_chunked(
            ctx, qkv, z, gates, state, k_heads, v_heads, k_dim, v_dim, &mut wide,
        )?;
        let width = stated(OP, y.width)?;
        let rows = nonzero(OP, "rows", y.rows)?;
        return ctx.fire(
            OP,
            Fire::at(FILE, "::pie::attn::ssm_gdn_narrow_rows")
                .apply(Launch::grid([rows, 1, 1], [128, 1, 1])),
            &[wide.arg(), y.arg(), width.arg(), ctx.stage()],
        );
    }
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Delta::of(OP, qkv.data, gates, y, k_heads, v_heads, k_dim, v_dim)?;
    let lanes = requests(OP, qkv)?;
    let wide = k_dim == BK_MAX_FLA && v_dim == BV_FLA && !SERIAL_SCAN.with(core::cell::Cell::get);
    #[cfg(feature = "cuda")]
    if wide && gated_delta_chunked_tc5(ctx, OP, &shape, qkv, gates, state, y)? {
        return Ok(());
    }
    let fused = wide && fused_scan(ctx, k_dim);
    if fused
        && shape.conv_dim.is_multiple_of(8)
        && (k_heads * k_dim).is_multiple_of(8)
        && state.slot_stride_elems % 8 == 0
        && [qkv.data.ptr, y.ptr, state.slab.ptr]
            .iter()
            .all(|&at| aligned16(at))
    {
        return gated_delta_chunked_fla(ctx, OP, &shape, qkv, gates, state, y);
    }
    let staged = shape.stage(ctx, OP, qkv.data, gates, StagePlanes::Chunk)?;
    if wide && !fused {
        return gated_delta_chunked_split(ctx, OP, &shape, &staged, qkv, state, y);
    }
    if k_dim <= BK_MAX_FLA && v_dim.is_multiple_of(BV_FLA) {
        seated(
            OP,
            state,
            "ssm_gated_delta_chunked_batched_fla",
            true,
            true,
            true,
        )?;
        return ctx.fire(
            OP,
            Fire::at(
                FILE,
                "::pie::attn::ssm_gated_delta_chunked_batched_fla<::pie::attn::state_bf16, 128, 128>",
            )
            .apply(
                Launch::grid([v_dim / BV_FLA, lanes, v_heads], [BV_FLA, 1, 1])
                    .smem(2 * BK_MAX_FLA * FLOAT),
            ),
            &[
                ArgValue::Ptr(staged.q_norm),
                ArgValue::Ptr(staged.k_norm),
                ArgValue::Ptr(staged.v),
                ArgValue::Ptr(staged.g_log),
                ArgValue::Ptr(staged.beta),
                state.slab.arg(),
                state.slot_ids.arg(),
                qkv.indptr.arg(),
                state.slot_stride_elems.arg(),
                y.arg(),
                stated(OP, k_heads)?.arg(),
                stated(OP, v_heads)?.arg(),
                stated(OP, k_dim)?.arg(),
                stated(OP, v_dim)?.arg(),
                state.write_state.arg(),
                state.commit_len.arg(),
                state.write_state_mask.arg(),
                state.begin_at.arg(),
                state.fused_decay.arg(),
                ctx.stage(),
            ],
        );
    }

    if k_dim <= WARP_TILED_K_MAX {
        seated(
            OP,
            state,
            "ssm_gated_delta_chunked_batched_warp_tiled_gqa",
            true,
            false,
            false,
        )?;
        return ctx.fire(
            OP,
            Fire::at(
                FILE,
                "::pie::attn::ssm_gated_delta_chunked_batched_warp_tiled_gqa<::pie::attn::state_bf16, false>",
            )
            .apply(Launch::grid(
                [lanes, v_heads, v_dim.div_ceil(SCAN_WARPS)],
                [SCAN_WARPS * WARP, 1, 1],
            )),
            &[
                ArgValue::Ptr(staged.q_norm),
                ArgValue::Ptr(staged.k_norm),
                ArgValue::Ptr(staged.v),
                ArgValue::Ptr(staged.g_log),
                ArgValue::Ptr(staged.beta),
                state.slab.arg(),
                state.slot_ids.arg(),
                qkv.indptr.arg(),
                state.slot_stride_elems.arg(),
                y.arg(),
                stated(OP, k_heads)?.arg(),
                stated(OP, v_heads)?.arg(),
                stated(OP, k_dim)?.arg(),
                stated(OP, v_dim)?.arg(),
                state.write_state.arg(),
                state.write_state_mask.arg(),
                ctx.stage(),
            ],
        );
    }

    seated(
        OP,
        state,
        "ssm_gated_delta_chunked_batched",
        false,
        false,
        false,
    )?;
    let (q_norm, k_norm) = if v_heads == k_heads {
        (staged.q_norm, staged.k_norm)
    } else {
        let wide = shape.elems(v_heads, k_dim);
        let repeated = plane(ctx, OP, "attn.ssm_gdn_chunk_repeat", 2 * wide)?;
        let (q, k) = (repeated, repeated + wide * u64::from(FLOAT));
        for (src, dst) in [(staged.q_norm, q), (staged.k_norm, k)] {
            ctx.fire(
                OP,
                Fire::at(
                    FILE,
                    "::pie::attn::repeat_interleave_heads_fp32<::pie::attn::f32>",
                )
                .apply(Launch::grid([shape.n, v_heads, 1], [BLOCK, 1, 1])),
                &[
                    ArgValue::Ptr(src),
                    ArgValue::Ptr(dst),
                    stated(OP, k_heads)?.arg(),
                    stated(OP, v_heads)?.arg(),
                    stated(OP, k_dim)?.arg(),
                    stated(OP, v_heads / k_heads)?.arg(),
                    ctx.stage(),
                ],
            )?;
        }
        (q, k)
    };
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            "::pie::attn::ssm_gated_delta_chunked_batched<::pie::attn::state_bf16, false>",
        )
        .apply(Launch::grid([lanes, v_heads, 1], [GDN_BLOCK, 1, 1]).smem(2 * k_dim * FLOAT)),
        &[
            ArgValue::Ptr(q_norm),
            ArgValue::Ptr(k_norm),
            ArgValue::Ptr(staged.v),
            ArgValue::Ptr(staged.g_log),
            ArgValue::Ptr(staged.beta),
            state.slab.arg(),
            state.slot_ids.arg(),
            qkv.indptr.arg(),
            state.slot_stride_elems.arg(),
            y.arg(),
            stated(OP, v_heads)?.arg(),
            stated(OP, k_dim)?.arg(),
            stated(OP, v_dim)?.arg(),
            ctx.stage(),
        ],
    )
}

#[derive(Clone, Copy)]
struct Kda {
    n: u32,
    heads: u32,
    head_dim: u32,
    width: u32,
}

#[derive(Clone, Copy)]
struct KdaStaged {
    q_norm: u64,
    k_norm: u64,
    v: u64,
    gate: u64,
    beta: u64,
}

impl Kda {
    fn of(
        op: &'static str,
        mixed: Tensor,
        f: Tensor,
        b: Tensor,
        y: &Tensor,
        heads: u32,
        head_dim: u32,
    ) -> Result<Self, Error> {
        nonzero(op, "the KDA heads this statement states", heads)?;
        nonzero(op, "the KDA head width this statement states", head_dim)?;
        let width = heads.checked_mul(head_dim).ok_or_else(|| {
            refuse(
                op,
                format!("the KDA plane will not size: {heads} heads x {head_dim} wide"),
            )
        })?;
        debug_assert_eq!(
            u64::from(mixed.width),
            3 * u64::from(width),
            "the post-convolution `[q | k | v]` row is three head planes"
        );
        debug_assert!(
            f.rows == mixed.rows && f.width == width,
            "the forget projection's row is one head plane"
        );
        debug_assert!(
            b.rows == mixed.rows && b.width == heads,
            "the beta projection's row is one entry per head"
        );
        debug_assert!(
            y.rows == mixed.rows && y.width == width,
            "the recurrence lands one head plane per row"
        );
        nonzero(op, "rows", mixed.rows)?;
        Ok(Self {
            n: mixed.rows,
            heads,
            head_dim,
            width,
        })
    }

    /// The scan walks each head's `D x D` state in place, so a fire in which
    /// some row may not fold its boundary (no fold asked, or a predicate
    /// naming the lanes that do) gets a working copy per row and head; a
    /// fire that folds everywhere walks the slots themselves.
    fn work(
        self,
        ctx: &Ctx,
        op: &'static str,
        state: &RecurrentPool,
        rows: u32,
    ) -> Result<u64, Error> {
        if state.write_state && state.write_state_mask.is_absent() {
            return Ok(0);
        }
        let cells = u64::from(self.head_dim) * u64::from(self.head_dim);
        plane(
            ctx,
            op,
            "attn.ssm_kda_work",
            u64::from(rows) * u64::from(self.heads) * cells,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn stage(
        self,
        ctx: &Ctx,
        op: &'static str,
        mixed: Tensor,
        f: Tensor,
        b: Tensor,
        dt_bias: Tensor,
        a_log: Tensor,
        norm_eps: f32,
        gate_floor: f32,
    ) -> Result<KdaStaged, Error> {
        const PLANES: u32 = 3;

        let wide = u64::from(self.n) * u64::from(self.width);
        let decay = u64::from(self.n) * u64::from(self.heads);

        let qkv = plane(ctx, op, "attn.ssm_kda_qkv", 3 * wide)?;
        let gb = plane(ctx, op, "attn.ssm_kda_gates", wide + decay)?;
        let staged = KdaStaged {
            q_norm: qkv,
            k_norm: qkv + wide * u64::from(FLOAT),
            v: qkv + 2 * wide * u64::from(FLOAT),
            gate: gb,
            beta: gb + wide * u64::from(FLOAT),
        };

        ctx.fire(
            op,
            Fire::at(FILE, "::pie::attn::ssm_kda_qkv_prep<::pie::bf16, 128>")
                .apply(Launch::grid([self.n, PLANES, 1], [PREP_BLOCK, 1, 1])),
            &[
                mixed.arg(),
                ArgValue::Ptr(staged.q_norm),
                ArgValue::Ptr(staged.k_norm),
                ArgValue::Ptr(staged.v),
                stated(op, self.width)?.arg(),
                stated(op, self.head_dim)?.arg(),
                norm_eps.arg(),
                ctx.stage(),
            ],
        )?;
        // A row a block, four channels a thread.
        let gates = if self.head_dim.is_multiple_of(4) {
            Fire::at(FILE, "::pie::attn::ssm_kda_gate_beta_rows<::pie::bf16>")
                .apply(Launch::grid([self.n, 1, 1], [256, 1, 1]))
        } else {
            Fire::at(FILE, "::pie::attn::ssm_kda_gate_beta<::pie::bf16>").apply(Launch::grid(
                [self.n, self.heads, 1],
                [self.head_dim.clamp(WARP, PREP_BLOCK), 1, 1],
            ))
        };
        ctx.fire(
            op,
            gates,
            &[
                f.arg(),
                b.arg(),
                a_log.arg(),
                dt_bias.arg(),
                ArgValue::Ptr(staged.gate),
                ArgValue::Ptr(staged.beta),
                stated(op, self.n)?.arg(),
                stated(op, self.heads)?.arg(),
                stated(op, self.head_dim)?.arg(),
                gate_floor.arg(), // the decay's lower bound; zero leaves it unbounded
                ctx.stage(),
            ],
        )?;
        Ok(staged)
    }
}

fn f32_state(op: &'static str, state: &RecurrentPool) -> Result<(), Error> {
    if state.slab.dtype != Dtype::F32 {
        return Err(refuse(
            op,
            format!(
                "the KDA recurrence keeps an f32 state and this cache row is declared {:?}",
                state.slab.dtype
            ),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn kda_step(
    ctx: &Ctx,
    mixed: Tensor,
    f: Tensor,
    b: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    state: &RecurrentPool,
    heads: u32,
    head_dim: u32,
    norm_eps: f32,
    gate_floor: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_kda_step";

    const STEP_BLOCK: u32 = 256;

    dtype_dispatch!(OP, mixed.dtype, { Bf16 => () });
    debug_assert_eq!(dt_bias.dtype, Dtype::F32, "`{OP}` reads an f32 decay bias");
    debug_assert_eq!(a_log.dtype, Dtype::F32, "`{OP}` reads an f32 decay bank");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Kda::of(OP, mixed, f, b, y, heads, head_dim)?;
    f32_state(OP, state)?;
    seated(OP, state, "ssm_kda_step_batched", true, false, false)?;
    let work = shape.work(ctx, OP, state, shape.n)?;
    if ctx.tuning().ssm_kda_step_fused && matches!(shape.head_dim, 32 | 64 | 128) {
        // Enough blocks to fill the device twice: each (row, head) split
        // over up to eight slices of its value rows.
        let sms = ctx.multiprocessors().unwrap_or(148);
        let blocks = shape.n.saturating_mul(shape.heads).max(1);
        let splits = (2 * sms)
            .div_ceil(blocks)
            .clamp(1, 8)
            .min(shape.head_dim.max(1));
        return ctx.fire(
            OP,
            Fire::at(
                FILE,
                symbol(&format!(
                    "::pie::attn::ssm_kda_step_fused<::pie::bf16, {}>",
                    shape.head_dim / WARP
                )),
            )
            .apply(
                Launch::grid([shape.n, shape.heads, splits], [STEP_BLOCK, 1, 1])
                    .smem(kda_shmem(shape.head_dim) + shape.head_dim * FLOAT),
            ),
            &[
                mixed.arg(),
                f.arg(),
                b.arg(),
                a_log.arg(),
                dt_bias.arg(),
                gate_floor.arg(),
                norm_eps.arg(),
                state.slab.arg(),
                state.slot_ids.arg(),
                state.slot_stride_elems.arg(),
                y.arg(),
                stated(OP, shape.heads)?.arg(),
                stated(OP, shape.head_dim)?.arg(),
                state.write_state.arg(),
                state.write_state_mask.arg(),
                ArgValue::Ptr(work),
                ctx.stage(),
            ],
        );
    }
    let staged = shape.stage(ctx, OP, mixed, f, b, dt_bias, a_log, norm_eps, gate_floor)?;
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::attn::ssm_kda_step_batched").apply(
            Launch::grid([shape.n, shape.heads, 1], [STEP_BLOCK, 1, 1])
                .smem(kda_shmem(shape.head_dim)),
        ),
        &[
            ArgValue::Ptr(staged.q_norm),
            ArgValue::Ptr(staged.k_norm),
            ArgValue::Ptr(staged.v),
            ArgValue::Ptr(staged.gate),
            ArgValue::Ptr(staged.beta),
            state.slab.arg(),
            state.slot_ids.arg(),
            state.slot_stride_elems.arg(),
            y.arg(),
            stated(OP, shape.heads)?.arg(),
            stated(OP, shape.head_dim)?.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            ArgValue::Ptr(work),
            ctx.stage(),
        ],
    )
}

/// The raw decay and beta rows the tensor-core prep forms its gates from.
struct KdaGates {
    f: Tensor,
    b: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    floor: f32,
}

/// [`kda_chunked`] on the tensor cores in sixteen-token chunks: two warps a
/// (chunk, head) land the chunk's decayed queries and fold keys, its decay,
/// its update directions `w` and `u` and its decayed causal scores; then a
/// block a (request, head, 64 value columns) walks the chunks with the state
/// in f32. 128-wide heads.
#[allow(clippy::too_many_arguments)]
fn kda_chunked_mma(
    ctx: &Ctx,
    op: &'static str,
    shape: Kda,
    gates: KdaGates,
    mixed: RaggedTensor,
    norm_eps: f32,
    state: &RecurrentPool,
    lanes: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const CH: u32 = 16;
    const D: u32 = 128;
    const TASKS: u32 = 2;
    // `KDA16_TASK_SMEM` and `KDA16_SCAN_SMEM`.
    const LD: u32 = 136;
    const LDT: u32 = 24;
    const LDV: u32 = 72;
    const TASK_SMEM: u32 = 3 * CH * LD * 2 + CH * (CH + 1) * FLOAT + CH * LDT * 2 + 3 * D * FLOAT;
    const STAGE: u32 = 3 * CH * LD * 2 + CH * LDV * 2 + CH * LDT * 2 + D * FLOAT;
    const SCAN_SMEM: u32 = 2 * STAGE + D * LDV * 2 + CH * LDV * 2;
    let heads = shape.heads;
    let slots = shape.n.div_ceil(CH) + lanes;
    let rows = u64::from(shape.n + CH) * u64::from(heads) * u64::from(D);
    let qg = plane_bf16(ctx, op, "attn.ssm_kda_chunk_qg", rows)?;
    let kg = plane_bf16(ctx, op, "attn.ssm_kda_chunk_kg", rows)?;
    let w = plane_bf16(ctx, op, "attn.ssm_kda_chunk_w", rows)?;
    let u = plane_bf16(ctx, op, "attn.ssm_kda_chunk_u", rows)?;
    let pm = plane_bf16(
        ctx,
        op,
        "attn.ssm_kda_chunk_p",
        u64::from(slots) * u64::from(heads) * u64::from(CH * CH),
    )?;
    let gl = plane(
        ctx,
        op,
        "attn.ssm_kda_chunk_gl",
        u64::from(slots) * u64::from(heads) * u64::from(D),
    )?;
    ctx.fire(
        op,
        Fire::at(FILE, "::pie::attn::ssm_kda16_prep").apply(
            Launch::grid([heads.div_ceil(TASKS), slots, 1], [TASKS * 2 * WARP, 1, 1])
                .smem(TASKS * TASK_SMEM),
        ),
        &[
            mixed.data.arg(),
            norm_eps.arg(),
            gates.f.arg(),
            gates.b.arg(),
            gates.a_log.arg(),
            gates.dt_bias.arg(),
            gates.floor.arg(),
            state.slot_ids.arg(),
            mixed.indptr.arg(),
            state.commit_len.arg(),
            state.begin_at.arg(),
            ArgValue::Ptr(qg),
            ArgValue::Ptr(kg),
            ArgValue::Ptr(w),
            ArgValue::Ptr(u),
            ArgValue::Ptr(pm),
            ArgValue::Ptr(gl),
            stated(op, lanes)?.arg(),
            stated(op, heads)?.arg(),
            ctx.stage(),
        ],
    )?;
    ctx.fire(
        op,
        Fire::at(FILE, "::pie::attn::ssm_kda16_scan")
            .apply(Launch::grid([lanes * heads * (D / 64), 1, 1], [128, 1, 1]).smem(SCAN_SMEM)),
        &[
            ArgValue::Ptr(gl),
            ArgValue::Ptr(qg),
            ArgValue::Ptr(kg),
            ArgValue::Ptr(w),
            ArgValue::Ptr(u),
            ArgValue::Ptr(pm),
            state.slab.arg(),
            state.slot_ids.arg(),
            mixed.indptr.arg(),
            state.slot_stride_elems.arg(),
            state.commit_len.arg(),
            state.begin_at.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            y.arg(),
            stated(op, lanes)?.arg(),
            stated(op, heads)?.arg(),
            ctx.stage(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn kda_chunked(
    ctx: &Ctx,
    mixed: RaggedTensor,
    f: Tensor,
    b: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    state: &RecurrentPool,
    heads: u32,
    head_dim: u32,
    norm_eps: f32,
    gate_floor: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_kda_chunked";

    const PREFILL_MAX_WARPS: u32 = 32;

    dtype_dispatch!(OP, mixed.data.dtype, { Bf16 => () });
    debug_assert_eq!(dt_bias.dtype, Dtype::F32, "`{OP}` reads an f32 decay bias");
    debug_assert_eq!(a_log.dtype, Dtype::F32, "`{OP}` reads an f32 decay bank");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Kda::of(OP, mixed.data, f, b, y, heads, head_dim)?;
    let lanes = requests(OP, mixed)?;
    f32_state(OP, state)?;
    seated(OP, state, "ssm_kda_chunked_batched", true, true, true)?;
    if ctx.tuning().ssm_kda_chunked && shape.head_dim == 128 {
        // The tensor-core prep reads the raw rows and forms the norms and
        // gates itself.
        let gates = KdaGates {
            f,
            b,
            dt_bias,
            a_log,
            floor: gate_floor,
        };
        return kda_chunked_mma(ctx, OP, shape, gates, mixed, norm_eps, state, lanes, y);
    }
    let staged = shape.stage(
        ctx, OP, mixed.data, f, b, dt_bias, a_log, norm_eps, gate_floor,
    )?;
    let work = shape.work(ctx, OP, state, lanes)?;
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::attn::ssm_kda_chunked_batched").apply(
            Launch::grid(
                [lanes, shape.heads, 1],
                [shape.head_dim.min(PREFILL_MAX_WARPS) * WARP, 1, 1],
            )
            .smem(kda_shmem(shape.head_dim)),
        ),
        &[
            ArgValue::Ptr(staged.q_norm),
            ArgValue::Ptr(staged.k_norm),
            ArgValue::Ptr(staged.v),
            ArgValue::Ptr(staged.gate),
            ArgValue::Ptr(staged.beta),
            state.slab.arg(),
            state.slot_ids.arg(),
            mixed.indptr.arg(),
            state.slot_stride_elems.arg(),
            y.arg(),
            stated(OP, shape.heads)?.arg(),
            stated(OP, shape.head_dim)?.arg(),
            state.write_state.arg(),
            state.commit_len.arg(),
            state.write_state_mask.arg(),
            state.begin_at.arg(),
            ArgValue::Ptr(work),
            ctx.stage(),
        ],
    )
}

/// Mamba-2's per-head weights: `A_log`, `dt_bias` and `D` (f32 `[heads]`),
/// and the gated group norm's bf16 `[heads * 64]` weight and epsilon.
#[derive(Clone, Copy)]
pub struct Mamba2 {
    pub a_log: Tensor,
    pub dt_bias: Tensor,
    pub d: Tensor,
    pub norm: Tensor,
    pub eps: f32,
    pub heads: u32,
    pub groups: u32,
}

impl Mamba2 {
    /// Refuses all but 64-wide heads over a 128-wide state, the width of
    /// `xbc` ( `x`, then each group's `B`, then its `C`) and `dt` against
    /// the heads, and a group count that does not split them.
    fn check(
        &self,
        op: &'static str,
        xbc: Tensor,
        dt: Tensor,
        z: Tensor,
        y: &Tensor,
        state_dim: u32,
    ) -> Result<(), Error> {
        dtype_dispatch!(op, xbc.dtype, { Bf16 => () });
        let inner = self.heads * 64;
        if state_dim != 128
            || self.groups == 0
            || !self.heads.is_multiple_of(self.groups)
            || xbc.width != inner + 2 * self.groups * state_dim
            || dt.width != self.heads
            || z.width != inner
            || y.width != inner
        {
            return Err(refuse(
                op,
                format!(
                    "Mamba-2 runs 64-wide heads over a 128-wide state in whole groups; this \
                     fire states {} heads in {} groups of a {}-wide state, a {}-wide xbc, \
                     {}-wide dt, a {}-wide z and a {}-wide output",
                    self.heads, self.groups, state_dim, xbc.width, dt.width, z.width, y.width
                ),
            ));
        }
        for (what, t) in [("dt", dt.dtype), ("z", z.dtype), ("output", y.dtype)] {
            if t != Dtype::Bf16 {
                return Err(refuse(
                    op,
                    format!("Mamba-2 reads and lands bf16, and its {what} is {t:?}"),
                ));
            }
        }
        Ok(())
    }

    /// The gated output's group norm, in place.
    fn norm(&self, ctx: &Ctx, op: &'static str, y: &mut Tensor) -> Result<(), Error> {
        let group = y.width / self.groups;
        crate::elemwise::norm::rmsnorm_grouped(ctx, op, *y, self.norm, group, self.eps, false, y)
    }
}

/// One Mamba-2 step a request over the convolved `xbc` rows: lands
/// `(C S + D x) * silu(z)`, group-normed, in `y`.
#[allow(clippy::too_many_arguments)]
pub fn mamba2_step(
    ctx: &Ctx,
    xbc: Tensor,
    dt: Tensor,
    z: Tensor,
    m: Mamba2,
    state: &RecurrentPool,
    state_dim: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_mamba2_step";
    m.check(OP, xbc, dt, z, y, state_dim)?;
    f32_state(OP, state)?;
    seated(OP, state, "ssm_mamba2_step", true, false, false)?;
    let rows = nonzero(OP, "rows", xbc.rows)?;
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::attn::ssm_mamba2_step")
            .apply(Launch::grid([rows, m.heads, 1], [BLOCK, 1, 1])),
        &[
            xbc.arg(),
            dt.arg(),
            z.arg(),
            m.a_log.arg(),
            m.dt_bias.arg(),
            m.d.arg(),
            state.slab.arg(),
            state.slot_ids.arg(),
            state.slot_stride_elems.arg(),
            y.arg(),
            stated(OP, m.heads)?.arg(),
            stated(OP, m.groups)?.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            ctx.stage(),
        ],
    )?;
    m.norm(ctx, OP, y)
}

/// A prompt's Mamba-2 walk in sixteen-token chunks: a block a (request,
/// head) lands `(C S + D x) * silu(z)` for each row, then the group norm.
#[allow(clippy::too_many_arguments)]
pub fn mamba2_chunked(
    ctx: &Ctx,
    xbc: RaggedTensor,
    dt: Tensor,
    z: Tensor,
    m: Mamba2,
    state: &RecurrentPool,
    state_dim: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_mamba2_chunked";
    // `M2_SCAN_SMEM`: two stages of C, B, x and dt, the bf16 state, `dt x`
    // twice, `P`, and the chunk's steps and decay sums.
    const CH: u32 = 16;
    const LDV: u32 = 72;
    const STAGE: u32 = 2 * CH * 136 * 2 + CH * LDV * 2 + CH * FLOAT;
    const SMEM: u32 = 2 * STAGE + 128 * LDV * 2 + 2 * CH * LDV * 2 + CH * 24 * 2 + 2 * CH * FLOAT;
    m.check(OP, xbc.data, dt, z, y, state_dim)?;
    let lanes = requests(OP, xbc)?;
    f32_state(OP, state)?;
    seated(OP, state, "ssm_mamba2_scan", true, true, true)?;
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::attn::ssm_mamba2_scan")
            .apply(Launch::grid([lanes * m.heads, 1, 1], [128, 1, 1]).smem(SMEM)),
        &[
            xbc.data.arg(),
            dt.arg(),
            z.arg(),
            m.a_log.arg(),
            m.dt_bias.arg(),
            m.d.arg(),
            state.slab.arg(),
            state.slot_ids.arg(),
            xbc.indptr.arg(),
            state.slot_stride_elems.arg(),
            state.commit_len.arg(),
            state.begin_at.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            y.arg(),
            stated(OP, lanes)?.arg(),
            stated(OP, m.heads)?.arg(),
            stated(OP, m.groups)?.arg(),
            ctx.stage(),
        ],
    )?;
    m.norm(ctx, OP, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A B200 or H100 block opts into 227 KiB, an A100 into 163 KiB; an
    /// RTX 30/40/50 or a DGX Spark into 99 KiB, short of the scan.
    #[test]
    fn the_scan_fits_a_datacenter_block_and_not_a_99_kib_one() {
        assert!(scan_smem(128) <= 232_448);
        assert!(scan_smem(128) <= 166_912);
        assert!(scan_smem(128) > 101_376);
    }

    /// Three blocks of the fused step where the SM holds them, as on a B200
    /// (228 KiB) or an A100 (164 KiB); two on a 100 KiB SM.
    #[test]
    fn the_fused_step_seats_what_the_sm_holds() {
        assert_eq!(fused_blocks_per_sm(Some(233_472)), 3);
        assert_eq!(fused_blocks_per_sm(Some(167_936)), 3);
        assert_eq!(fused_blocks_per_sm(Some(102_400)), 2);
        assert_eq!(fused_blocks_per_sm(Some(16_384)), 1);
        assert_eq!(fused_blocks_per_sm(None), 3);
    }
}
