//! `Attention`'s recurrent-state mixers — causal conv, gated delta nets,
//! KDA: the `Ssm*` variants, whose sequence cache is a recurrent state
//! rather than a kv pool. One entry per IR variant; [`RecurrentPool`] is
//! that state, updated in place.
//! The chunked forms are the prefill path: they take the fire's ragged view
//! and launch one scan per request instead of one per token.
//!
//! The delta and KDA recurrences stage first: a prep launch widens and
//! normalises the packed projection into f32 planes in process-global
//! scratch (grown, never shrunk — an entry may not allocate per fire), and
//! the scan reads the planes. Both launches ride the same stream, so the
//! staging is ordered like everything else and nothing synchronises.

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated};
use crate::tensor::{RaggedTensor, RecurrentPool, Tensor};

const FILE: &str = "attn/ssm.cuh";

const BLOCK: u32 = 256;

const WARP: u32 = 32;

/// Bytes per f32, for shared-memory and scratch sizing.
const FLOAT: u32 = 4;

/// The prep launches' block.
const PREP_BLOCK: u32 = 128;

/// The delta scans' block.
const GDN_BLOCK: u32 = 128;

/// The KDA kernels' dynamic shared memory: three f32 rows of the head width.
const fn kda_shmem(d: u32) -> u32 {
    3u32.saturating_mul(d).saturating_mul(FLOAT)
}

/// The request count a ragged fire spans: the indptr is `[lanes + 1]`. The
/// boundary vector is driver-assembled, not an operand the validator sees,
/// so a wrong dtype is refused, not asserted (the boundary rule at
/// [`refuse`]).
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

/// A named f32 scratch plane, returned as the address the launch binds.
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

/// **The RS seats, checked against the arm that is about to run.**
///
/// `attn/ssm.cuh` does not carry the fold predicate, the commit length and
/// the segment origin on every instantiation: the CHUNKED conv takes all
/// three, the chunked delta scan takes all three on its fla arm and the
/// predicate alone on the warp-tiled one, and the DECODE (per-step) kernels
/// take none — a step kernel updates the
/// bank in place, interleaved with the output it is computing, so predicating
/// it would need a shadow slot the pool does not carry.
///
/// So a pool that CARRIES a seat this arm has no parameter for is refused by
/// name. The alternative is the one failure a typed seat exists to prevent:
/// a speculative fire whose refused pass silently folds anyway, or a replay
/// that folds the whole buffered window instead of the accepted prefix of it.
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

/// The conv's stated extents, shared by both forms.
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
    dtype_dispatch!(OP, x.dtype, { Bf16 => () });
    let (channels, c, k) = conv_extents(OP, x, y, conv_width)?;
    let dil = stated(OP, nonzero(OP, "the conv's dilation", dilation)?)?;
    let rows = nonzero(OP, "rows", x.rows)?;
    seated(
        OP,
        state,
        "ssm_causal_conv1d_update_batched",
        false,
        false,
        false,
    )?;
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::attn::ssm_causal_conv1d_update_batched<::pie::bf16>")
            .apply(Launch::grid([channels.div_ceil(BLOCK), rows, 1], [BLOCK, 1, 1])),
        &[
            x.arg(),
            weight.arg(),
            ArgValue::ABSENT, // the bias seat; this point carries none
            state.conv_slab.arg(),
            state.slot_ids.arg(),
            state.conv_stride.arg(),
            y.arg(),
            stated(OP, rows)?.arg(),
            c.arg(),
            k.arg(),
            dil.arg(),
            // ctx.stage(): the region's live-rows word, or ABSENT.
            ctx.stage(),
        ],
    )
}

/// Prefill form: walks the fire's request boundaries, one grid row per
/// request. Two instantiations; the channel-tiled form wins once the fire
/// is wide enough to fill it.
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

    const CHANNEL_TILE_FROM: u32 = 8;

    const TILE_BLOCK: u32 = 128;

    const PER_CHANNEL_BLOCK: u32 = 64;

    dtype_dispatch!(OP, x.data.dtype, { Bf16 => () });
    let (channels, c, k) = conv_extents(OP, x.data, y, conv_width)?;
    let dil = stated(OP, nonzero(OP, "the conv's dilation", dilation)?)?;
    let lanes = requests(OP, x)?;
    seated(OP, state, "ssm_causal_conv1d_chunked_batched", true, true, true)?;
    let (entrypoint, launch) = if lanes >= CHANNEL_TILE_FROM {
        (
            "::pie::attn::ssm_causal_conv1d_chunked_batched_channel_tile<::pie::bf16>",
            Launch::grid(
                [channels.div_ceil(TILE_BLOCK), lanes, 1],
                [TILE_BLOCK, 1, 1],
            ),
        )
    } else {
        (
            "::pie::attn::ssm_causal_conv1d_chunked_batched<::pie::bf16>",
            Launch::grid([channels, lanes, 1], [PER_CHANNEL_BLOCK, 1, 1]),
        )
    };
    ctx.fire(
        OP,
        Fire::at(FILE, entrypoint).apply(launch),
        &[
            x.data.arg(),
            weight.arg(),
            ArgValue::ABSENT, // the bias seat
            y.arg(),
            state.conv_slab.arg(),
            state.slot_ids.arg(),
            x.indptr.arg(),
            state.conv_stride.arg(),
            c.arg(),
            k.arg(),
            dil.arg(),
            // The three RS seats: fold predicate, commit-length truncation,
            // and the accepted boundary.
            state.write_state.arg(),
            state.write_state_mask.arg(),
            state.commit_len.arg(),
            // The segment's origin: bound only on the tail launch of a row
            // whose fold boundary falls inside its own tokens.
            state.begin_at.arg(),
            // Read on the lane axis (both arms above grid on requests):
            // passed unconditionally, so pointers are always pre-shifted.
            ctx.stage(),
        ],
    )
}

/// Folds `ba` with the dt bias and A-log into per-head decay gates.
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
    if ba.width == 0 || ba.width % 2 != 0 {
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
            // ctx.stage(): the region's live-rows word, or ABSENT.
            ctx.stage(),
        ],
    )
}

/// The gated-delta shape: four stated head numbers against the fused rows.
#[derive(Clone, Copy)]
struct Delta {
    n: u32,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    /// The packed post-convolution row's width, which the preps stride by.
    conv_dim: u32,
}

/// The staged f32 planes a delta scan reads: addresses into the scratch
/// slabs.
#[derive(Clone, Copy)]
struct DeltaStaged {
    q_norm: u64,
    k_norm: u64,
    v: u64,
    g_log: u64,
    beta: u64,
}

impl Delta {
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
        if v_heads % k_heads != 0 {
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

    /// Widen and normalise the packed projection into f32 planes: q/k
    /// L2-normed per head, v copied, the fused gates split.
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
                // ctx.stage(): the region's live-rows word, or ABSENT.
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
                // ctx.stage(): the region's live-rows word, or ABSENT.
                ctx.stage(),
            ],
        )?;
        Ok(staged)
    }
}

/// The gated-delta recurrent step, one token per lane. `z` goes unread:
/// this plane gates afterwards (`elementwise.rmsnorm_gated`).
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

    /// Both head widths at exactly this hit the shared-memory arm.
    const SMEM_ARM_WIDTH: u32 = 128;

    const SMEM_BV: u32 = 128;

    let _ = z;
    dtype_dispatch!(OP, qkv.dtype, { Bf16 => () });
    debug_assert_eq!(gates.dtype, Dtype::F32, "`{OP}` reads an f32 decay row");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Delta::of(OP, qkv, gates, y, k_heads, v_heads, k_dim, v_dim)?;
    seated(OP, state, "ssm_gated_delta_step_batched_gqa", false, false, false)?;
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
            // ctx.stage(): the region's live-rows word, or ABSENT.
            ctx.stage(),
        ],
    )
}

/// Prefill form of [`gated_delta`]: one chunked scan per request. Three
/// instantiations, picked on head geometry: fla tiling when the value width
/// fills it, a warp-tiled sweep for narrow keys, else the plain block scan
/// (key planes repeated across the GQA fan). `z` goes unread, as above.
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

    if k_dim <= BK_MAX_FLA && v_dim % BV_FLA == 0 {
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
                // The one chunked arm carrying all three: fold predicate
                // and accepted length, which is why a buffered replay is
                // dispatchable at all.
                state.write_state.arg(),
                state.commit_len.arg(),
                state.write_state_mask.arg(),
                // The segment's origin (tail of an interior split) and the
                // decay rounding policy.
                state.begin_at.arg(),
                state.fused_decay.arg(),
                // Read on the lane axis; the staged scratch and window CSR
                // stay launch-local.
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
                // Read on the lane axis, as the fla arm above.
                ctx.stage(),
            ],
        );
    }

    seated(OP, state, "ssm_gated_delta_chunked_batched", false, false, false)?;
    // The plain scan reads one key plane per value head; a GQA fan repeats
    // the staged planes across it first.
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
                    // ctx.stage(): the region's live-rows word, or ABSENT.
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
            // Read on the lane axis, as the fla arm above.
            ctx.stage(),
        ],
    )
}

/// The KDA shape: two stated head numbers against the mixed rows.
#[derive(Clone, Copy)]
struct Kda {
    n: u32,
    heads: u32,
    head_dim: u32,
    /// `heads x head_dim`, the plane every staged buffer is one of.
    width: u32,
}

/// The staged f32 planes the KDA recurrence reads.
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

    /// Split, norm and widen `[q | k | v]`, then fold the forget/beta
    /// projections with the decay weights into f32 gates.
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
    ) -> Result<KdaStaged, Error> {
        /// q, k, v — the prep's grid-y axis.
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
                norm_eps.arg(),
                // ctx.stage(): the region's live-rows word, or ABSENT.
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
                0.0_f32.arg(), // the decay's lower bound, unbounded here
                // ctx.stage(): the region's live-rows word, or ABSENT.
                ctx.stage(),
            ],
        )?;
        Ok(staged)
    }
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
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_kda_step";

    const STEP_BLOCK: u32 = 256;

    dtype_dispatch!(OP, mixed.dtype, { Bf16 => () });
    debug_assert_eq!(dt_bias.dtype, Dtype::F32, "`{OP}` reads an f32 decay bias");
    debug_assert_eq!(a_log.dtype, Dtype::F32, "`{OP}` reads an f32 decay bank");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Kda::of(OP, mixed, f, b, y, heads, head_dim)?;
    seated(OP, state, "ssm_kda_step_batched", false, false, false)?;
    let staged = shape.stage(ctx, OP, mixed, f, b, dt_bias, a_log, norm_eps)?;
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
            // ctx.stage(): the region's live-rows word, or ABSENT.
            ctx.stage(),
        ],
    )
}

/// Prefill form of [`kda_step`]: one scan per request.
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
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_kda_chunked";

    /// The widest block the prefill scan spans, in warps.
    const PREFILL_MAX_WARPS: u32 = 32;

    dtype_dispatch!(OP, mixed.data.dtype, { Bf16 => () });
    debug_assert_eq!(dt_bias.dtype, Dtype::F32, "`{OP}` reads an f32 decay bias");
    debug_assert_eq!(a_log.dtype, Dtype::F32, "`{OP}` reads an f32 decay bank");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Kda::of(OP, mixed.data, f, b, y, heads, head_dim)?;
    let lanes = requests(OP, mixed)?;
    seated(OP, state, "ssm_kda_chunked_batched", false, false, false)?;
    let staged = shape.stage(ctx, OP, mixed.data, f, b, dt_bias, a_log, norm_eps)?;
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
            // Read on the lane axis; the staged scratch and window CSR
            // stay launch-local.
            ctx.stage(),
        ],
    )
}
