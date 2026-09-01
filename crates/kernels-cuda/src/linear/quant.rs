//! `quant`: dtype casts and the weight quantizers the loader drives. No IR
//! variant lands here directly; the engine and loader call these entries
//! while staging checkpoints and recurrent state.
//!
//! The jit-stamped transcode descriptors (`DecodeBf16`,
//! `DecodeFp8E4m3PerGroup`, `EncodeMxfp4`) are by-value aggregates; they
//! arrive with the attn wave, together with the `by_value!` abi machinery
//! that measures their layouts. The moe wave settled its half of that
//! deferral without them: an mxfp4 bank travels as the explicit
//! `(codes, scales)` pair and marshals as two plain pointers
//! (`moe::matmul_select_bias`), so no descriptor is needed on the routed
//! path.

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::linear::gemm;
use crate::linear::moe::GroupSeat;
use crate::tensor::Tensor;

const FILE: &str = "linear/quant.cuh";

const BLOCK: u32 = 256;

const WARP: u32 = 32;

/// **THE DECODED WEIGHT THE PREFILL ARM PROJECTS THROUGH** — one bf16
/// `[n, k]` tile, grown to the widest projection a fire has asked for and
/// never shrunk ([`Ctx::scratch`]).
///
/// One name for one slab, on `linear::lora`'s rule: every dense affine
/// projection in a plan decodes and consumes it inside its own dispatch, so
/// the layers share it the way they share every other scratch here, and the
/// arena keys it per stream so two fires never hold it at once.
const DECODED_WEIGHT: &str = "linear.quant.decoded_weight";

/// One block per row, sized to the row in whole warps.
fn route_rows(rows: u32, width: u32) -> Launch {
    const MAX_BLOCK: u32 = 1024;

    Launch::per_row(
        rows,
        width
            .div_ceil(WARP)
            .max(1)
            .saturating_mul(WARP)
            .min(MAX_BLOCK),
    )
}

/// A 32-bit launch extent, refused rather than truncated — a clamped grid
/// would launch over the low 32 bits and leave the rest of the destination
/// unwritten.
fn extent(op: &'static str, n: u64) -> Result<u32, Error> {
    u32::try_from(n).map_err(|_| {
        refuse(
            op,
            format!("{n} elements do not fit a 32-bit launch extent"),
        )
    })
}

/// **THE OFFSET ARM, DECLARED AND THEN CHECKED** — the launch side of
/// `linear/quant.cuh`'s `kOffset` axis, and the mirror of
/// `dtype::Off` rather than that type itself: the algebra is the
/// contract's currency and this is an ABI, so what crosses is the four arms
/// this point is stamped for, not a term the kernels would have to parse.
///
/// The arm is the CALLER's declaration, on `linear::fp8::Form`'s rule: the
/// planes are then checked against it and never inferred from it, because a
/// `Post` bias plane and a `PreReal` zero plane are the same rectangle of
/// the same reals and only the caller knows which fold they belong to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OffsetKind {
    /// `s·c + b`, `b` a factor-dtype real per group: an offset in the VALUE
    /// domain — MLX's affine triplet, `OffSub::Post(L(f))`.
    Post,
    /// `s·(c − z)`, `z` an unsigned code-domain integer per group: GPTQ and
    /// AWQ, `OffSub::Pre(L(U(b)))`.
    PreInt,
    /// `s·(c − z)`, `z` a factor-dtype REAL per group: HQQ,
    /// `OffSub::Pre(L(f))`. It is a separate arm from [`PreInt`](Self::PreInt)
    /// for the reason `OffSub` keeps the two apart — a pre-scale zero cannot
    /// be assumed to share the codes' dtype.
    PreReal,
    /// `s·(c − 2^(bits−1))`, with no offset plane at all: the excess-binary
    /// symmetric rows (`Leaf::I(b)` — Q4_0, Q8_0 and Int4B8 after canon),
    /// whose constant recentering IS a pre-offset once it reaches a dot.
    /// A symmetric `n`-offset term over excess leaves lands here, which is
    /// why there is no unoffset arm: a point that only ever multiplied would
    /// be `linear::nvfp4`'s, not this one.
    PreConst,
}

impl OffsetKind {
    /// The `kOffset` constant this arm is stamped as.
    const fn axis(self) -> &'static str {
        match self {
            Self::Post => "::pie::linear::kOffPost",
            Self::PreInt => "::pie::linear::kOffPreInt",
            Self::PreReal => "::pie::linear::kOffPreReal",
            Self::PreConst => "::pie::linear::kOffPreConst",
        }
    }

    /// The arm's name in a refusal, in the algebra's own words.
    const fn spelling(self) -> &'static str {
        match self {
            Self::Post => "a post-offset arm (`s·c + b`)",
            Self::PreInt => "an integer pre-offset arm (`s·(c − z)`)",
            Self::PreReal => "a real pre-offset arm (`s·(c − z)`)",
            Self::PreConst => "a constant pre-offset arm (`s·(c − 2^(bits−1))`)",
        }
    }
}

/// `linear.matmul` over a weight the store seats as codes plus a per-group
/// factor plane — the dense twin of `moe_matmul_select_quant`, and this
/// shell's spelling of `kernels_metal::linear::quant::matmul`.
///
/// Two things the caller STATES and this entry checks rather than infers:
/// the offset arm ([`OffsetKind`], with the plane it reads or the absence it
/// requires), and `factor`, the dtype of the scales plane — which binds as a
/// `U8` byte rectangle, so nothing about its bytes says whether they are
/// bf16 or f16. What is still inferred is what a rectangle can only mean:
/// the bit width off the codes row (a `[n, k]` weight stores `k` bytes a row
/// at eight bits and `k / 2` at four) and the group off the factor row.
#[allow(clippy::too_many_arguments)]
pub fn matmul(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    offset: OffsetKind,
    biases: Option<Tensor>,
    factor: Dtype,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    dense_affine(
        ctx,
        "linear.matmul",
        act,
        codes,
        scales,
        offset,
        biases,
        factor,
        y,
        seat,
    )
}

/// [`matmul`] under the head's own op name, `linear::gemm`'s pairing kept.
#[allow(clippy::too_many_arguments)]
pub fn lm_head(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    offset: OffsetKind,
    biases: Option<Tensor>,
    factor: Dtype,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    dense_affine(
        ctx,
        "linear.lm_head",
        act,
        codes,
        scales,
        offset,
        biases,
        factor,
        y,
        seat,
    )
}

/// **THE PLANE LADDER, READ ONCE FOR BOTH ARMS.** Everything a dense affine
/// fire can only learn by measuring the rectangles it was handed — the code
/// width, the group, and the two extents — with every refusal that reading
/// them raises. It is one function because there are now two points over
/// these planes ([`dense_affine`] and [`dense_affine_via_dense`]), and two
/// copies of this ladder would be two chances to disagree about what one
/// stored form means.
#[derive(Clone, Copy)]
struct Affine {
    /// Four or eight, off the codes row against `k`.
    bits: u32,
    /// The codes under one factor, off the factor row against `k`.
    group: u32,
    /// The rows this projection lands, which are the weight's rows.
    n: u32,
    /// The contraction it walks, which is the weight's width.
    k: u32,
}

/// [`Affine`], measured and checked. `y`'s ROW count is deliberately not
/// read here: an empty fire is each point's own no-op, and this ladder must
/// answer the same on a fire with rows and on one without.
fn affine(
    op: &'static str,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    offset: OffsetKind,
    biases: Option<Tensor>,
    y: &Tensor,
) -> Result<Affine, Error> {
    debug_assert_eq!(codes.dtype, Dtype::U8, "a packed plane binds as bytes");
    debug_assert_eq!(scales.dtype, Dtype::U8, "a packed plane binds as bytes");
    debug_assert!(
        biases.is_none_or(|b| b.dtype == Dtype::U8),
        "a packed plane binds as bytes"
    );
    // **THE ARM AND THE PLANE MUST AGREE.** A `PreConst` fire handed an
    // offset plane would read none of it and answer a plausible number off
    // the constant instead — the silent-wrongness class — and the three
    // arms that DO read one have no fallback to fold if it is missing. An
    // mxfp4 dense weight arrives here as the second of those: it has no
    // gemm point on this plane, and the offset it lacks is the refusal.
    match (offset, biases) {
        (OffsetKind::PreConst, Some(_)) => {
            return Err(refuse(
                op,
                format!(
                    "{} reads no offset plane, and one was bound",
                    offset.spelling()
                ),
            ));
        }
        (OffsetKind::Post | OffsetKind::PreInt | OffsetKind::PreReal, None) => {
            return Err(refuse(
                op,
                format!("{} was declared with no offset plane to read", offset.spelling()),
            ));
        }
        _ => {}
    }
    debug_assert_eq!(
        act.rows, y.rows,
        "the activation's rows are the rows the result lands"
    );
    let n = nonzero(op, "N, the columns this projection lands", y.width)?;
    let k = nonzero(op, "K, the contraction this projection walks", act.width)?;
    // **THE GROUP COMES OFF THE FACTOR PLANE**, the only thing that states
    // it: the plane is `[n, k / group]` factor-dtype reals bound as its byte
    // rectangle, so half its width is the group COUNT and the group follows.
    // Thirty-two, sixty-four and a hundred and twenty-eight all fold; a
    // width that leaves the row in fractional groups is refused rather than
    // mis-scaled.
    if scales.width == 0 || scales.width % 2 != 0 {
        return Err(refuse(
            op,
            format!(
                "a {}-byte factor row is not a whole number of two-byte factors",
                scales.width
            ),
        ));
    }
    let groups = scales.width / 2;
    if !k.is_multiple_of(groups) {
        return Err(refuse(
            op,
            format!("{groups} factors do not group a {k}-wide row into whole groups"),
        ));
    }
    let group = k / groups;
    let bits: u32 = if codes.width == k {
        8
    } else if codes.width * 2 == k {
        4
    } else {
        return Err(refuse(
            op,
            format!(
                "a {}-byte code row stores a {k}-wide row at neither four nor eight bits",
                codes.width
            ),
        ));
    };
    // The kernel reads codes a 32-bit word at a time, so a group is a whole
    // number of words or its second half would be read against the next
    // group's factor.
    let per_word = 32 / bits;
    if !group.is_multiple_of(per_word) {
        return Err(refuse(
            op,
            format!("a {group}-code group is not a whole number of {per_word}-code words"),
        ));
    }
    // **ONE BYTE PER GROUP IS THE `PreInt` CONTAINER** — the zero is a `u4`
    // for GPTQ and AWQ and the plane could hold two of them to the byte, but
    // that halves a plane that is already one byte per group of codes, and
    // it would cost a shift and a mask on every group of every row. The zero
    // travels widened instead, so a four-bit form's zero plane is the same
    // `[n, k / group]` byte rectangle an eight-bit form's is.
    if let Some(plane) = biases {
        let want = if offset == OffsetKind::PreInt {
            groups
        } else {
            scales.width
        };
        if plane.width != want {
            return Err(refuse(
                op,
                format!(
                    "{} states a {want}-byte offset row over a {k}-wide row, and the \
                     plane holds {}",
                    offset.spelling(),
                    plane.width
                ),
            ));
        }
    }
    Ok(Affine { bits, group, n, k })
}

/// The one launch behind both dense entries, over all four offset arms.
/// Every arm folds the same per-group pair — `s·Σ c·x` against `Σ x` — and
/// they differ only in what the second is multiplied by, which is why this
/// is one point and not four (`linear/quant.cuh`). A fire with no rows is
/// the same silent no-op the dense gemm keeps, and for the same capture
/// reason.
#[allow(clippy::too_many_arguments)]
fn dense_affine(
    ctx: &Ctx,
    op: &'static str,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    offset: OffsetKind,
    biases: Option<Tensor>,
    factor: Dtype,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    const ROWS_PER_WARP: u32 = 4;
    const BLOCK_LANES: u32 = 128;

    let t = dtype_dispatch!(op, act.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let f = dtype_dispatch!(op, factor, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let Affine { bits, group, n, k } = affine(op, act, codes, scales, offset, biases, y)?;
    if y.rows == 0 {
        return Ok(());
    }
    let tile = (BLOCK_LANES / WARP) * ROWS_PER_WARP;
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::linear::matmul_affine<{t}, {f}, ::pie::i32({bits}), {}, \
                 ::pie::i32({group}), ::pie::i32({ROWS_PER_WARP})>",
                offset.axis()
            )),
        )
        .apply(Launch::grid(
            [y.rows, n.div_ceil(tile), 1],
            [BLOCK_LANES, 1, 1],
        )),
        &[
            act.arg(),
            codes.arg(),
            scales.arg(),
            biases.map_or(ArgValue::ABSENT, |b| b.arg()),
            y.arg(),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
            ArgValue::Ptr(seat.cell),
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
        ],
    )
}

/// **THE PREFILL ARM OF [`matmul`]** — the same stored planes, decoded once
/// into a transient scratch tile and projected by the dense cuBLAS point.
///
/// Same signature as [`matmul`], because the choice between them is the
/// CALLER's and it is made on shape alone. [`matmul`]'s point carves one
/// block column per activation row and re-reads the whole weight inside each
/// of them: parity with cuBLAS bf16 at one token, and 98–189× slower than it
/// over 128–2048 prefill rows. This arm reads the weight once and pays an
/// `n·k` bf16 slab and a second launch for it.
///
/// **THE RESIDENT FORM DOES NOT CHANGE.** The store still seats codes and a
/// factor plane; the decoded tile is a kernel's workspace, alive for the
/// fire and named nowhere in the plan. That is the tree's serve-as-stored
/// ruling kept rather than bent — what is SERVED is the row the checkpoint
/// stated, and where a kernel decodes it on the way is implementation.
///
/// **AND IT IS RESIDENT-ONLY.** A streamed seat ([`GroupSeat::streams`])
/// moves its planes between fires, so there is no fixed rectangle to decode
/// into a slab; that path is the tier's and its staging is the moe point's
/// business. The refusal is typed rather than a quiet fall back to the fused
/// arm, because a caller that reached here reached here on purpose.
///
/// **THE NUMERICS, STATED SO NOBODY GOLDENS AGAINST THE WRONG THING.** The
/// decode rounds every weight element to bf16 exactly once; the fused point
/// rounds none of them, folding `s·c + b` in f32 inside the dot. So the two
/// arms answer the SAME NUMBERS AND NOT THE SAME BITS, and a logit off the
/// head differs in its last places depending on which one fired. Callers
/// pick the arm per shape — they do not expect bit-equality across arms, and
/// no golden in this tree asks for it.
#[allow(clippy::too_many_arguments)]
pub fn matmul_via_dense(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    offset: OffsetKind,
    biases: Option<Tensor>,
    factor: Dtype,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    dense_affine_via_dense(
        ctx,
        "linear.matmul",
        act,
        codes,
        scales,
        offset,
        biases,
        factor,
        y,
        seat,
    )
}

/// [`matmul_via_dense`] under the head's own op name — [`lm_head`]'s prefill
/// twin, and the same pairing `linear::gemm` keeps.
#[allow(clippy::too_many_arguments)]
pub fn lm_head_via_dense(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    offset: OffsetKind,
    biases: Option<Tensor>,
    factor: Dtype,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    dense_affine_via_dense(
        ctx,
        "linear.lm_head",
        act,
        codes,
        scales,
        offset,
        biases,
        factor,
        y,
        seat,
    )
}

/// The two launches behind both prefill entries: `dequant_affine` into the
/// slab, then `linear::gemm`'s dense point over it. The plane ladder is
/// [`affine`]'s, shared with the fused arm, so the two points refuse the
/// same stored forms for the same reasons.
#[allow(clippy::too_many_arguments)]
fn dense_affine_via_dense(
    ctx: &Ctx,
    op: &'static str,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    offset: OffsetKind,
    biases: Option<Tensor>,
    factor: Dtype,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    // The dense point is stamped for bf16 activations alone, so an f16 fire
    // is refused HERE and not after a decode nobody would have read.
    dtype_dispatch!(op, act.dtype, { Bf16 => () });
    let f = dtype_dispatch!(op, factor, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    if seat.streams() {
        return Err(refuse(
            op,
            "the decoded-tile arm serves resident dense projections only, and these \
             planes are seated by a streaming tier",
        ));
    }
    let Affine { bits, group, n, k } = affine(op, act, codes, scales, offset, biases, y)?;
    if y.rows == 0 {
        return Ok(());
    }
    // Two bytes an element: the tile is bf16, which is the only weight dtype
    // the dense point reads.
    let bytes = (n as usize).saturating_mul(k as usize).saturating_mul(2);
    let tile = ctx.scratch(op, DECODED_WEIGHT, bytes)? as usize as u64;
    // One thread per code WORD — the kernel writes `32 / bits` elements from
    // each — so the extent is the weight's words and not its elements.
    let words = extent(op, u64::from(n) * u64::from(k) / u64::from(32 / bits))?;
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::linear::dequant_affine<{f}, ::pie::i32({bits}), {}, \
                 ::pie::i32({group})>",
                offset.axis()
            )),
        )
        .apply(Launch::flat(words, BLOCK)),
        &[
            codes.arg(),
            scales.arg(),
            biases.map_or(ArgValue::ABSENT, |b| b.arg()),
            ArgValue::Ptr(tile),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
        ],
    )?;
    // What the slab now holds is the rectangle the dense point has always
    // taken: `[n, k]` bf16, read as `act x w^T` under the caller's op name.
    gemm::act_x_wt(ctx, op, act, Tensor::new(tile, n, k, Dtype::Bf16), y)
}

pub fn cast_fp32_to(ctx: &Ctx, src: Tensor, dst: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "linear.quant_cast_fp32_to";
    debug_assert_eq!(src.dtype, Dtype::F32, "`{OP}` casts an f32 source");
    let t = dtype_dispatch!(OP, dst.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let n = dst.elements();
    let lanes = nonzero(OP, "the cast's element count", extent(OP, n)?)?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::linear::cast_f32_to<{t}>")))
            .apply(Launch::flat(lanes, BLOCK)),
        &[src.arg(), dst.arg(), n.arg()],
    )
}

/// Scales each row of `buf` by the matching row of `l`, in place.
pub fn scale_rows(ctx: &Ctx, l: Tensor, buf: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "linear.quant_scale_rows";
    let t = dtype_dispatch!(OP, buf.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    nonzero(OP, "rows", buf.rows)?;
    let width = stated(OP, nonzero(OP, "width", buf.width)?)?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::linear::scale_rows<{t}>")))
            .apply(route_rows(buf.rows, buf.width)),
        &[buf.arg(), l.arg(), width.arg()],
    )
}

// **TWO ENCODE ENTRIES STOOD HERE** — `quantize_bf16_to_mxfp4_e2m1_per_block`
// and `quantize_bf16_to_fp8_e4m3_per_channel`, the device half of the
// load-time quantization §M-3 shut. Nothing quantizes a weight on the way to
// a device any more: the stored form IS the served form, and the one family
// that declared otherwise now has its codes written by `pie model import`,
// on the host, once. The device text went with them.
