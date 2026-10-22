mod dyn_conv;
pub use dyn_conv::dyn_conv;

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol,
};
use crate::tensor::{RaggedTensor, RecurrentPool, Tensor};

const FILE: &str = "attn/ssm.hip";

const BLOCK: u32 = 256;

const WAVE: u32 = 64;

const FLOAT: u32 = 4;

const PREP_BLOCK: u32 = 128;

/// A thread a channel for the decode conv.
const CONV_UPDATE_BLOCK: u32 = 256;

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
    Chunk,
}

impl StagePlanes {
    const fn names(self) -> (&'static str, &'static str, &'static str) {
        match self {
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

    /// The template arguments with the tap count after the element type.
    fn args_with(self, taps: &str) -> String {
        match self {
            ConvOut::Silu => format!("<::pie::bf16, {taps}>"),
            ConvOut::Residual => format!("<::pie::bf16, {taps}, false, true>"),
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
    const OP: &str = "ssm.causal_conv1d";
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
    const OP: &str = "ssm.short_conv";
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
    // A thread owns one channel of a row.
    let entrypoint = symbol(&format!("::pie::attn::ssm_conv1d_update{}", out.args()));
    let launch = Launch::grid(
        [channels.div_ceil(CONV_UPDATE_BLOCK), rows, 1],
        [CONV_UPDATE_BLOCK, 1, 1],
    );
    let args = [
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
        dil.arg(),
        state.write_state.arg(),
        state.write_state_mask.arg(),
        ctx.stage(),
    ];
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
    const OP: &str = "ssm.causal_conv1d_chunked";
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
    const OP: &str = "ssm.short_conv_chunked";
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
    const TILE_BLOCK: u32 = 128;
    const CONV_TT: u32 = 32;

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
    // Dilation 1 at 2..=4 taps over whole 16-byte channel runs takes the
    // eight-channel sliding walk (`ssm_conv1d_chunked_v8`).
    let vectored = dil == 1
        && (2..=4).contains(&k)
        && channels % 8 == 0
        && crate::jit::aligned16(x.data.ptr)
        && crate::jit::aligned16(y.ptr)
        && crate::jit::aligned16(state.conv_slab.ptr)
        && state.conv_stride % 8 == 0;
    if vectored {
        let entrypoint = symbol(&format!(
            "::pie::attn::ssm_conv1d_chunked_v8{}",
            out.args_with(&format!("{k}"))
        ));
        let launch = Launch::grid(
            [
                (channels / 8).div_ceil(TILE_BLOCK),
                x.data.rows.div_ceil(CONV_TT).max(1),
                1,
            ],
            [TILE_BLOCK, 1, 1],
        );
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
                state.commit_len.arg(),
                state.begin_at.arg(),
                ctx.stage(),
                stated(op, lanes)?.arg(),
            ],
        )?;
    } else {
        let entrypoint = symbol(&format!("::pie::attn::ssm_conv1d_chunked{}", out.args()));
        let launch = Launch::grid(
            [
                channels.div_ceil(TILE_BLOCK),
                x.data.rows.div_ceil(CONV_TT).max(1),
                1,
            ],
            [TILE_BLOCK, 1, 1],
        );
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
    }
    if !state.write_state {
        return Ok(());
    }
    let [snap, snap_rows, snap_slots] = state.snapshots.args();
    ctx.fire(
        op,
        Fire::at(FILE, symbol("::pie::attn::ssm_conv1d_state_commit")).apply(Launch::grid(
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
            snap,
            snap_rows,
            snap_slots,
            ctx.stage(),
            stated(op, lanes)?.arg(),
        ],
    )
}

pub fn gdn_gates(
    ctx: &Ctx,
    ba: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    gates: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.gdn_gates";
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
        Fire::at(FILE, "::pie::attn::ssm_gdn_prep_ba_gates").apply(Launch::grid(
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
            Fire::at(FILE, "::pie::attn::ssm_gdn_prep_qk_norm<128>")
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
            Fire::at(FILE, "::pie::attn::ssm_gdn_prep_v_gates<128>")
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
    const OP: &str = "ssm.gated_delta";

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
    // A block a (head, row) at any batch: even one row of 48 heads runs in
    // under half the tiled step's time.
    if shape.k_dim == CHUNK_HEAD
        && shape.v_dim == CHUNK_HEAD
        && state.slot_stride_elems % 8 == 0
        && shape.conv_dim.is_multiple_of(8)
    {
        return shape.decode_rows(ctx, OP, qkv, gates, state, y);
    }
    if fused_fits(&shape, state) {
        return shape.decode_fused(ctx, OP, qkv, gates, state, y);
    }
    Err(refuse(
        OP,
        format!(
            "the decode step is built for 8-aligned rows, a value head in whole 16-wide \
             tiles and a key head of at most 128; this one is {k_dim} x {v_dim}"
        ),
    ))
}

/// The tickets every fused step launch draws from (a key head a row).
const STEP_TICKETS: u32 = crate::jit::TICKETS;

/// A gated-delta layer's decode step whole, from the in-projections' rows:
/// `[qkv | z] = split(qkvz)`, the conv update of `qkv`, the gate prep of
/// `ba` and the recurrence, in one launch (`ssm_gdn_step_fused`), landing
/// the split's halves (`qkv`, `z`) and the recurrence's f32 row `y`.
/// `Ok(false)` where the shape is not the one it is built for (128-wide
/// heads, a conv of at most four taps, no dilation), which leaves the step
/// to its primitives.
#[allow(clippy::too_many_arguments)]
pub fn gdn_step_fused(
    ctx: &Ctx,
    qkvz: Tensor,
    conv_weight: Tensor,
    conv_state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    ba: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    state: &RecurrentPool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qkv: &mut Tensor,
    z: &mut Tensor,
    y: &mut Tensor,
) -> Result<bool, Error> {
    const OP: &str = "attention.ssm_gdn_step";
    let channels = 2 * k_heads * k_dim + v_heads * v_dim;
    let covered = k_dim == CHUNK_HEAD
        && v_dim == CHUNK_HEAD
        && k_heads > 0
        && v_heads.is_multiple_of(k_heads)
        && (2..=4).contains(&conv_width)
        && dilation == 1
        && qkvz.dtype == Dtype::Bf16
        && ba.dtype == Dtype::Bf16
        && dt_bias.dtype == Dtype::Bf16
        && a_log.dtype == Dtype::F32
        && y.dtype == Dtype::F32
        && qkvz.width == channels + v_heads * v_dim
        && qkv.width == channels
        && z.width == v_heads * v_dim
        && ba.width == 2 * v_heads
        && u64::from(conv_weight.rows) * u64::from(conv_weight.width)
            == u64::from(channels) * u64::from(conv_width)
        && state.slot_stride_elems % 8 == 0
        && state.write_state_mask.is_absent() == conv_state.write_state_mask.is_absent()
        && conv_state.commit_len.is_absent()
        && conv_state.begin_at.is_absent()
        && state.commit_len.is_absent()
        && state.begin_at.is_absent()
        && (conv_state.write_state || !conv_state.write_state_mask.is_absent())
        && (state.write_state || !state.write_state_mask.is_absent());
    if !covered {
        return Ok(false);
    }
    let rows = nonzero(OP, "rows", qkvz.rows)?;
    if u64::from(rows) * u64::from(k_heads) > u64::from(STEP_TICKETS) {
        return Ok(false);
    }
    let tickets = ctx.tickets(OP, STEP_TICKETS)?;
    #[allow(clippy::cast_precision_loss)]
    let q_scale = (k_dim as f32).sqrt().recip();
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::attn::ssm_gdn_step_fused<128, 128, 256, 4>")
            .apply(Launch::grid([v_heads, rows, 1], [FUSED_BLOCK, 1, 1])),
        &[
            qkvz.arg(),
            conv_weight.arg(),
            conv_state.conv_slab.arg(),
            conv_state.slot_ids.arg(),
            conv_state.conv_stride.arg(),
            conv_state.write_state.arg(),
            conv_state.write_state_mask.arg(),
            ba.arg(),
            a_log.arg(),
            dt_bias.arg(),
            state.slab.arg(),
            state.slot_ids.arg(),
            state.slot_stride_elems.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            qkv.arg(),
            z.arg(),
            y.arg(),
            ArgValue::Ptr(tickets),
            stated(OP, k_heads)?.arg(),
            stated(OP, v_heads)?.arg(),
            stated(OP, conv_width)?.arg(),
            q_scale.arg(),
            ctx.stage(),
        ],
    )?;
    Ok(true)
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

/// From this many (row, head) pairs the KDA step's block a whole head fills
/// the device; under it each head's rows split over D / 32 blocks.
const ROWS_STEP_FROM: u32 = 256;

impl Delta {
    fn decode_rows(
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
            Fire::at(FILE, "::pie::attn::ssm_gdn_decode_rows<128, 128, 256>")
                .apply(Launch::grid([self.v_heads, self.n, 1], [FUSED_BLOCK, 1, 1])),
            &[
                qkv.arg(),
                gates.arg(),
                state.slab.arg(),
                state.slot_ids.arg(),
                state.slot_stride_elems.arg(),
                y.arg(),
                stated(op, self.k_heads)?.arg(),
                stated(op, self.v_heads)?.arg(),
                stated(op, self.conv_dim)?.arg(),
                q_scale.arg(),
                state.write_state.arg(),
                state.write_state_mask.arg(),
                ctx.stage(),
            ],
        )
    }

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
                state.write_state.arg(),
                state.write_state_mask.arg(),
                ctx.stage(),
            ],
        )
    }
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
    const OP: &str = "ssm.gated_delta_chunked";

    /// State columns a scan block owns.
    const SCAN_BV: u32 = 128;

    let _ = z;
    dtype_dispatch!(OP, qkv.data.dtype, { Bf16 => () });
    debug_assert_eq!(gates.dtype, Dtype::F32, "`{OP}` reads an f32 decay row");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Delta::of(OP, qkv.data, gates, y, k_heads, v_heads, k_dim, v_dim)?;
    let lanes = requests(OP, qkv)?;
    if k_dim == CHUNK_HEAD && v_dim == CHUNK_HEAD && !SERIAL_SCAN.with(core::cell::Cell::get) {
        return chunk_parallel(ctx, OP, &shape, qkv, gates, state, y);
    }
    let staged = shape.stage(ctx, OP, qkv.data, gates, StagePlanes::Chunk)?;
    // The recurrence, token by token per request: a thread per state
    // column, its key-head entries in registers.
    let bk = match k_dim {
        0..=128 => 128,
        129..=256 => 256,
        _ => {
            return Err(refuse(
                OP,
                format!("the scan holds a key head of at most 256 and this one is {k_dim}"),
            ));
        }
    };
    seated(OP, state, "ssm_gdn_scan", true, true, true)?;
    let bv = v_dim.min(SCAN_BV);
    let [snap, snap_rows, snap_slots] = state.snapshots.args();
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!("::pie::attn::ssm_gdn_scan<{bk}, {bv}>")),
        )
        .apply(Launch::grid(
            [v_dim.div_ceil(bv), lanes, v_heads],
            [bv, 1, 1],
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
            state.commit_len.arg(),
            state.write_state_mask.arg(),
            state.begin_at.arg(),
            state.fused_decay.arg(),
            snap,
            snap_rows,
            snap_slots,
            ctx.stage(),
        ],
    )
}

/// The head width the chunk-parallel passes are built for.
const CHUNK_HEAD: u32 = 128;

/// Tokens per chunk.
const CHUNK: u32 = 64;

/// Value columns per state-walking block.
const CHUNK_SLICE: u32 = 16;

/// The recurrence as three chunk-parallel passes (see `ssm_gdn_chunk_*`).
fn chunk_parallel(
    ctx: &Ctx,
    op: &'static str,
    shape: &Delta,
    qkv: RaggedTensor,
    gates: Tensor,
    state: &RecurrentPool,
    y: &mut Tensor,
) -> Result<(), Error> {
    let lanes = requests(op, qkv)?;
    seated(op, state, "ssm_gdn_chunk", true, true, true)?;
    let Delta {
        k_dim,
        v_dim,
        v_heads,
        k_heads,
        ..
    } = *shape;
    // A request's chunks start at ⌊t0 / CH⌋ + r, so `n / CH + lanes` slots
    // cover them all; the row planes carry a chunk of tail for the tiles
    // that read past a request's last row.
    let slots = shape.n.div_ceil(CHUNK) + lanes + state.snapshots.count;
    let padded = Delta {
        n: shape.n + CHUNK,
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
        per_slot(k_heads, k_dim * CHUNK),
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
        "attn.ssm_gdn_chunk_ut",
        per_slot(v_heads, v_dim * CHUNK),
    )?;
    let vnt = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_vnt",
        per_slot(v_heads, v_dim * CHUNK),
    )?;
    let hct = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_hct",
        per_slot(v_heads, v_dim * k_dim),
    )?;
    #[allow(clippy::cast_precision_loss)]
    let q_scale = (k_dim as f32).sqrt().recip();
    let snap = state.snapshots.args();
    let common = |args: &mut Vec<ArgValue>| -> Result<(), Error> {
        args.push(stated(op, lanes)?.arg());
        args.push(stated(op, k_heads)?.arg());
        args.push(stated(op, v_heads)?.arg());
        args.extend(snap);
        args.push(ctx.stage());
        Ok(())
    };

    let args = vec![
        qkv.data.arg(),
        gates.arg(),
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
        stated(op, shape.conv_dim)?.arg(),
        q_scale.arg(),
        snap[0],
        snap[1],
        ctx.stage(),
    ];
    ctx.fire(
        op,
        Fire::at(FILE, "::pie::attn::ssm_gdn_chunk_wu<128, 128>")
            .apply(Launch::grid([slots, v_heads, 1], [2 * BLOCK, 1, 1])),
        &args,
    )?;

    let mut args = vec![
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
    ];
    common(&mut args)?;
    let from = ctx.tuning().ssm_gdn_wide_from;
    let slices = lanes * v_heads * (v_dim / CHUNK_SLICE);
    let bvt = if from > 0 && slices >= from {
        2 * CHUNK_SLICE
    } else {
        CHUNK_SLICE
    };
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!("::pie::attn::ssm_gdn_chunk_h<128, 128, {bvt}>")),
        )
        .apply(Launch::grid(
            [lanes * v_heads * (v_dim / bvt), 1, 1],
            [BLOCK, 1, 1],
        )),
        &args,
    )?;

    let mut args = vec![
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
    ];
    common(&mut args)?;
    ctx.fire(
        op,
        Fire::at(FILE, "::pie::attn::ssm_gdn_chunk_o<128, 128>")
            .apply(Launch::grid([slots, v_heads, 1], [BLOCK, 1, 1])),
        &args,
    )
}

fn plane_bf16(ctx: &Ctx, op: &'static str, name: &'static str, elems: u64) -> Result<u64, Error> {
    let bytes = usize::try_from(elems.saturating_mul(2))
        .map_err(|_| refuse(op, format!("{elems} bf16 elements do not fit this host")))?;
    Ok(ctx.scratch_shared(op, name, bytes)? as u64)
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
        /// Heads a staging wavefront norms.
        const PREP_HEADS: u32 = 4;

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
            Fire::at(FILE, "::pie::attn::ssm_kda_qkv_prep<128>").apply(Launch::grid(
                [
                    self.n,
                    PLANES,
                    self.heads.div_ceil(PREP_HEADS * (PREP_BLOCK / WAVE)),
                ],
                [PREP_BLOCK, 1, 1],
            )),
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
            Fire::at(FILE, "::pie::attn::ssm_kda_gate_beta").apply(Launch::grid(
                [self.n, self.width.div_ceil(4 * PREP_BLOCK), 1],
                [PREP_BLOCK, 1, 1],
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

/// The token kernels hold a KDA state row on eight lanes in whole 32-wide
/// pieces, up to this wide.
const KDA_MAX_HEAD: u32 = 128;

/// Value rows a decode block walks when the batch is too small for a block
/// per row and head.
const KDA_TILE_ROWS: u32 = 32;

fn kda_head(op: &'static str, head_dim: u32) -> Result<(), Error> {
    if !head_dim.is_multiple_of(32) || head_dim > KDA_MAX_HEAD {
        return Err(refuse(
            op,
            format!(
                "the KDA recurrence holds a head of whole 32-wide pieces, at most \
                 {KDA_MAX_HEAD}, and this one is {head_dim}"
            ),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn kda(
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
    const OP: &str = "ssm.kda";

    const STEP_BLOCK: u32 = 256;

    dtype_dispatch!(OP, mixed.dtype, { Bf16 => () });
    debug_assert_eq!(dt_bias.dtype, Dtype::F32, "`{OP}` reads an f32 decay bias");
    debug_assert_eq!(a_log.dtype, Dtype::F32, "`{OP}` reads an f32 decay bank");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Kda::of(OP, mixed, f, b, y, heads, head_dim)?;
    f32_state(OP, state)?;
    kda_head(OP, head_dim)?;
    seated(OP, state, "ssm_kda_step_batched", true, false, false)?;
    // A block a (row, head) once the batch fills the device; under that,
    // each head's rows split over D / 32 blocks.
    let rows = if shape.n.saturating_mul(shape.heads) >= ROWS_STEP_FROM {
        head_dim
    } else {
        KDA_TILE_ROWS
    };
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::attn::ssm_kda_step_batched<{head_dim}, {rows}>"
            )),
        )
        .apply(Launch::grid(
            [shape.heads, shape.n, head_dim / rows],
            [STEP_BLOCK, 1, 1],
        )),
        &[
            mixed.arg(),
            f.arg(),
            b.arg(),
            a_log.arg(),
            dt_bias.arg(),
            state.slab.arg(),
            state.slot_ids.arg(),
            state.slot_stride_elems.arg(),
            y.arg(),
            stated(OP, shape.heads)?.arg(),
            norm_eps.arg(),
            gate_floor.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            ctx.stage(),
        ],
    )
}

/// `kda_chunked` through the token-by-token scan whatever the head: the
/// reference the chunk-parallel passes answer to.
#[allow(clippy::too_many_arguments)]
pub fn kda_chunked_serial(
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
    SERIAL_SCAN.with(|flag| flag.set(true));
    let out = kda_chunked(
        ctx, mixed, f, b, dt_bias, a_log, state, heads, head_dim, norm_eps, gate_floor, y,
    );
    SERIAL_SCAN.with(|flag| flag.set(false));
    out
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
    const OP: &str = "ssm.kda_chunked";

    dtype_dispatch!(OP, mixed.data.dtype, { Bf16 => () });
    debug_assert_eq!(dt_bias.dtype, Dtype::F32, "`{OP}` reads an f32 decay bias");
    debug_assert_eq!(a_log.dtype, Dtype::F32, "`{OP}` reads an f32 decay bank");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Kda::of(OP, mixed.data, f, b, y, heads, head_dim)?;
    let lanes = requests(OP, mixed)?;
    f32_state(OP, state)?;
    kda_head(OP, head_dim)?;
    seated(OP, state, "ssm_kda_chunked_batched", true, true, true)?;
    if head_dim == CHUNK_HEAD && !SERIAL_SCAN.with(core::cell::Cell::get) {
        return kda_chunk_parallel(
            ctx,
            OP,
            &shape,
            mixed,
            [f, b, dt_bias, a_log],
            state,
            lanes,
            [norm_eps, gate_floor],
            y,
        );
    }
    let staged = shape.stage(
        ctx, OP, mixed.data, f, b, dt_bias, a_log, norm_eps, gate_floor,
    )?;
    let [snap, snap_rows, snap_slots] = state.snapshots.args();
    // A request a block, a value row to eight lanes.
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!("::pie::attn::ssm_kda_chunked_batched<{head_dim}>")),
        )
        .apply(Launch::grid([lanes, shape.heads, 1], [8 * head_dim, 1, 1])),
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
            snap,
            snap_rows,
            snap_slots,
            ctx.stage(),
        ],
    )
}

/// The recurrence as two chunk-parallel passes (`ssm_kda_chunk_wu`, then
/// `ssm_kda_chunk_ho`),
/// over the chunk slots the gated delta rule's passes use.
#[allow(clippy::too_many_arguments)]
fn kda_chunk_parallel(
    ctx: &Ctx,
    op: &'static str,
    shape: &Kda,
    mixed: RaggedTensor,
    [f, b, dt_bias, a_log]: [Tensor; 4],
    state: &RecurrentPool,
    lanes: u32,
    [norm_eps, gate_floor]: [f32; 2],
    y: &mut Tensor,
) -> Result<(), Error> {
    let (d, heads) = (shape.head_dim, shape.heads);
    let slots = shape.n.div_ceil(CHUNK) + lanes + state.snapshots.count;
    let per_slot = |width: u32| u64::from(slots) * u64::from(heads) * u64::from(width);
    let kt = plane_bf16(ctx, op, "attn.ssm_kda_chunk_kt", per_slot(d * CHUNK))?;
    let w = plane_bf16(ctx, op, "attn.ssm_kda_chunk_w", per_slot(CHUNK * d))?;
    let ut = plane_bf16(ctx, op, "attn.ssm_kda_chunk_ut", per_slot(d * CHUNK))?;
    let qg = plane_bf16(ctx, op, "attn.ssm_kda_chunk_qg", per_slot(CHUNK * d))?;
    let p = plane_bf16(ctx, op, "attn.ssm_kda_chunk_p", per_slot(CHUNK * CHUNK))?;
    let gl = plane(ctx, op, "attn.ssm_kda_chunk_gl", per_slot(d))?;
    let snap = state.snapshots.args();
    let tail = |args: &mut Vec<ArgValue>| -> Result<(), Error> {
        args.push(stated(op, lanes)?.arg());
        args.push(stated(op, heads)?.arg());
        args.extend(snap);
        args.push(ctx.stage());
        Ok(())
    };

    let args = vec![
        mixed.data.arg(),
        f.arg(),
        b.arg(),
        a_log.arg(),
        dt_bias.arg(),
        state.slot_ids.arg(),
        mixed.indptr.arg(),
        state.commit_len.arg(),
        state.begin_at.arg(),
        ArgValue::Ptr(kt),
        ArgValue::Ptr(w),
        ArgValue::Ptr(ut),
        ArgValue::Ptr(qg),
        ArgValue::Ptr(p),
        ArgValue::Ptr(gl),
        stated(op, lanes)?.arg(),
        stated(op, heads)?.arg(),
        norm_eps.arg(),
        gate_floor.arg(),
        snap[0],
        snap[1],
        ctx.stage(),
    ];
    ctx.fire(
        op,
        Fire::at(FILE, "::pie::attn::ssm_kda_chunk_wu<128>")
            .apply(Launch::grid([slots, heads, 1], [2 * BLOCK, 1, 1])),
        &args,
    )?;

    // The state walk lands each chunk's output on the way: its entering
    // state and V' stay in the walk's LDS.
    let mut args = vec![
        ArgValue::Ptr(kt),
        ArgValue::Ptr(w),
        ArgValue::Ptr(ut),
        ArgValue::Ptr(gl),
        ArgValue::Ptr(qg),
        ArgValue::Ptr(p),
        y.arg(),
        state.slab.arg(),
        state.slot_ids.arg(),
        mixed.indptr.arg(),
        state.slot_stride_elems.arg(),
        state.commit_len.arg(),
        state.begin_at.arg(),
        state.write_state.arg(),
        state.write_state_mask.arg(),
    ];
    tail(&mut args)?;
    ctx.fire(
        op,
        Fire::at(FILE, "::pie::attn::ssm_kda_chunk_ho<128>").apply(Launch::grid(
            [lanes * heads * (d / CHUNK_SLICE), 1, 1],
            [BLOCK, 1, 1],
        )),
        &args,
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
    /// `xbc` (`x`, then each group's `B`, then its `C`) and `dt` against
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
pub fn mamba2(
    ctx: &Ctx,
    xbc: Tensor,
    dt: Tensor,
    z: Tensor,
    m: Mamba2,
    state: &RecurrentPool,
    state_dim: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.mamba2";
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

/// A prompt's Mamba-2 walk, token by token: a block a (request, head) lands
/// `(C S + D x) * silu(z)` for each row, then the group norm.
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
    const OP: &str = "ssm.mamba2_chunked";
    m.check(OP, xbc.data, dt, z, y, state_dim)?;
    let lanes = requests(OP, xbc)?;
    f32_state(OP, state)?;
    seated(OP, state, "ssm_mamba2_scan", true, true, true)?;
    let [snap, snap_rows, snap_slots] = state.snapshots.args();
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::attn::ssm_mamba2_scan")
            .apply(Launch::grid([lanes, m.heads, 1], [BLOCK, 1, 1])),
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
            snap,
            snap_rows,
            snap_slots,
            y.arg(),
            stated(OP, m.heads)?.arg(),
            stated(OP, m.groups)?.arg(),
            ctx.stage(),
        ],
    )?;
    m.norm(ctx, OP, y)
}
