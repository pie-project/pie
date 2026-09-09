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

fn plane(ctx: &Ctx, op: &'static str, name: &'static str, elems: u64) -> Result<u64, Error> {
    let bytes = elems.checked_mul(u64::from(FLOAT)).ok_or_else(|| {
        refuse(
            op,
            format!("the {elems}-element staging plane will not size"),
        )
    })?;
    let bytes = usize::try_from(bytes)
        .map_err(|_| refuse(op, format!("{bytes} staging bytes do not fit this host")))?;
    Ok(ctx.scratch(op, name, bytes)? as u64)
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

pub fn causal_conv1d(
    ctx: &Ctx,
    x: Tensor,
    weight: Tensor,
    state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_causal_conv1d";
    conv1d_update(ctx, OP, x, weight, state, conv_width, dilation, y, ConvOut::Silu)
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
    conv1d_update(ctx, OP, x, weight, state, conv_width, 1, y, ConvOut::Residual)
}

#[allow(clippy::too_many_arguments)]
fn conv1d_update(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    weight: Tensor,
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
        false,
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
        && aligned16(state.conv_slab.ptr);
    let mut args = vec![
        x.arg(),
        weight.arg(),
        ArgValue::ABSENT,
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
    args.push(ctx.stage());
    ctx.fire(op, Fire::at(FILE, entrypoint).apply(launch), &args)
}

pub fn causal_conv1d_chunked(
    ctx: &Ctx,
    x: RaggedTensor,
    weight: Tensor,
    state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_causal_conv1d_chunked";
    conv1d_chunked(ctx, OP, x, weight, state, conv_width, dilation, y, ConvOut::Silu)
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
    conv1d_chunked(ctx, OP, x, weight, state, conv_width, 1, y, ConvOut::Residual)
}

#[allow(clippy::too_many_arguments)]
fn conv1d_chunked(
    ctx: &Ctx,
    op: &'static str,
    x: RaggedTensor,
    weight: Tensor,
    state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    y: &mut Tensor,
    out: ConvOut,
) -> Result<(), Error> {

    const CHANNEL_TILE_FROM: u32 = 8;

    const TILE_BLOCK: u32 = 128;

    const PER_CHANNEL_BLOCK: u32 = 64;

    dtype_dispatch!(op, x.data.dtype, { Bf16 => () });
    let (channels, c, k) = conv_extents(op, x.data, y, conv_width)?;
    let dil = stated(op, nonzero(op, "the conv's dilation", dilation)?)?;
    let lanes = requests(op, x)?;
    seated(op, state, "ssm_causal_conv1d_chunked_batched", true, true, true)?;
    let (entrypoint, launch) = if lanes >= CHANNEL_TILE_FROM {
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
            ArgValue::ABSENT,
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
    ) -> Result<DeltaStaged, Error> {
        let key = self.elems(self.k_heads, self.k_dim);
        let val = self.elems(self.v_heads, self.v_dim);
        let decay = self.elems(self.v_heads, 1);

        let qk = plane(ctx, op, "attn.ssm_gdn_chunk_qk", 2 * key)?;
        let v = plane(ctx, op, "attn.ssm_gdn_chunk_v", val)?;
        let gb = plane(ctx, op, "attn.ssm_gdn_chunk_gates", 2 * decay)?;
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
    seated(OP, state, "ssm_gated_delta_step_batched_gqa", false, false, false)?;
    if fused_fits(&shape, state) {
        return shape.decode_fused(ctx, OP, qkv, gates, state, y);
    }
    let staged = shape.stage(ctx, OP, qkv, gates)?;

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
            ctx.stage(),
        ],
    )
}

const FUSED_BLOCK: u32 = 256;

const FUSED_BV: u32 = 16;

fn fused_fits(shape: &Delta, state: &RecurrentPool) -> bool {
    const VEC: u32 = 8;
    shape.conv_dim.is_multiple_of(VEC)
        && (shape.k_heads * shape.k_dim).is_multiple_of(VEC)
        && shape.v_dim.is_multiple_of(FUSED_BV)
        && (shape.k_dim * shape.v_dim).is_multiple_of(VEC)
        && state.slot_stride_elems % i64::from(VEC) == 0
        && shape.k_dim <= FUSED_BLOCK * VEC / FUSED_BV
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
        ctx.fire(
            op,
            Fire::at(FILE, "::pie::attn::ssm_gdn_decode_step<16>").apply(Launch::grid(
                [self.v_dim / FUSED_BV, self.n, self.v_heads],
                [FUSED_BLOCK, 1, 1],
            )),
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
                ctx.stage(),
            ],
        )
    }
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
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Delta::of(OP, qkv.data, gates, y, k_heads, v_heads, k_dim, v_dim)?;
    let lanes = requests(OP, qkv)?;
    let staged = shape.stage(ctx, OP, qkv.data, gates)?;

    if k_dim <= BK_MAX_FLA && v_dim.is_multiple_of(BV_FLA) {
        seated(OP, state, "ssm_gated_delta_chunked_batched_fla", true, true, true)?;
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

    seated(OP, state, "ssm_gated_delta_chunked_batched", false, false, false)?;
    let (q_norm, k_norm) = if v_heads == k_heads {
        (staged.q_norm, staged.k_norm)
    } else {
        let wide = shape.elems(v_heads, k_dim);
        let repeated = plane(ctx, OP, "attn.ssm_gdn_chunk_repeat", 2 * wide)?;
        let (q, k) = (repeated, repeated + wide * u64::from(FLOAT));
        for (src, dst) in [(staged.q_norm, q), (staged.k_norm, k)] {
            ctx.fire(
                OP,
                Fire::at(FILE, "::pie::attn::repeat_interleave_heads_fp32<::pie::attn::f32>")
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
        ctx.fire(
            op,
            Fire::at(FILE, "::pie::attn::ssm_kda_gate_beta<::pie::bf16>").apply(Launch::grid(
                [self.n, self.heads, 1],
                [self.head_dim.clamp(WARP, PREP_BLOCK), 1, 1],
            )),
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
    seated(OP, state, "ssm_kda_step_batched", false, false, false)?;
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
    seated(OP, state, "ssm_kda_chunked_batched", false, false, false)?;
    let staged = shape.stage(ctx, OP, mixed.data, f, b, dt_bias, a_log, norm_eps, gate_floor)?;
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
            ctx.stage(),
        ],
    )
}
