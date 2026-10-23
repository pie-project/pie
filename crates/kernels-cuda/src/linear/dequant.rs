//! Quantized planes widened to bf16 (`linear/dequant.cu`) and the dense
//! products that read them: the affine planes, and NVFP4's.

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, refuse, stated, symbol};
use crate::linear::affine::Affine;
use crate::linear::affine::OffsetKind;
use crate::linear::affine::affine;
use crate::linear::affine::extent;
use crate::linear::gemm;
use crate::linear::moe::GroupSeat;
use crate::tensor::Tensor;

/// The bf16 tile a plane is decoded into before the dense gemm reads it:
/// one a stream, the size of the widest plane fired on it, since the decode
/// and the gemm are consecutive on the stream. Keyed by region it was one a
/// region, the whole model in bf16 across a load.
const DECODED_WEIGHT: &str = "linear.quant.decoded_weight";

const DEQUANT: &str = "linear/dequant.cu";

/// Takes this stream's decoded-weight tile at `bytes` now, so the first
/// fire that decodes a plane finds it and no capture ever has to grow it —
/// a load whose bodies are held to nothing arms no body at load, and its
/// first fire is a capture.
pub fn warm_decoded_weight(ctx: &Ctx, bytes: u64) -> Result<(), Error> {
    if bytes == 0 {
        return Ok(());
    }
    ctx.scratch_stream(
        "linear.matmul",
        DECODED_WEIGHT,
        usize::try_from(bytes).unwrap_or(usize::MAX),
    )
    .map(|_| ())
}

#[allow(clippy::too_many_arguments)]
pub fn decoded_plane(
    ctx: &Ctx,
    op: &'static str,
    codes: Tensor,
    scales: Tensor,
    offset: OffsetKind,
    biases: Option<Tensor>,
    factor: Dtype,
    n: u32,
    k: u32,
    seat: GroupSeat,
) -> Result<Tensor, Error> {
    let f = dtype_dispatch!(op, factor, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    if seat.streams() {
        return Err(refuse(
            op,
            "a decoded plane serves resident projections only, and these planes are \
             seated by a streaming tier",
        ));
    }
    if codes.rows != n {
        return Err(refuse(
            op,
            format!(
                "the code plane holds {} rows and the entry states {n}",
                codes.rows
            ),
        ));
    }
    let bits: u32 = if codes.width == k {
        8
    } else if codes.width * 2 == k {
        4
    } else if codes.width * 4 == k {
        2
    } else {
        return Err(refuse(
            op,
            format!(
                "a {}-byte code row stores a {k}-wide row at neither two, four nor eight bits",
                codes.width
            ),
        ));
    };
    let groups = scales.width / 2;
    if groups == 0 || !k.is_multiple_of(groups) {
        return Err(refuse(
            op,
            format!("{groups} factors do not group a {k}-wide row into whole groups"),
        ));
    }
    let group = k / groups;
    let bytes = (n as usize).saturating_mul(k as usize).saturating_mul(2);
    let tile = ctx.scratch_stream(op, DECODED_WEIGHT, bytes)? as usize as u64;
    let words = extent(op, u64::from(n) * u64::from(k) / u64::from(32 / bits))?;
    ctx.fire(
        op,
        Fire::at(
            DEQUANT,
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
    Ok(Tensor::new(tile, n, k, Dtype::Bf16))
}

#[allow(clippy::too_many_arguments)]
pub fn decode_into(
    ctx: &Ctx,
    op: &'static str,
    codes: Tensor,
    scales: Tensor,
    offset: OffsetKind,
    biases: Option<Tensor>,
    factor: Dtype,
    dst: u64,
    n: u32,
    k: u32,
) -> Result<Tensor, Error> {
    let f = dtype_dispatch!(op, factor, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    if codes.rows != n {
        return Err(refuse(
            op,
            format!(
                "the code plane holds {} rows and the entry states {n}",
                codes.rows
            ),
        ));
    }
    let bits: u32 = if codes.width == k {
        8
    } else if codes.width * 2 == k {
        4
    } else if codes.width * 4 == k {
        2
    } else {
        return Err(refuse(
            op,
            format!(
                "a {}-byte code row stores a {k}-wide row at neither two, four nor eight bits",
                codes.width
            ),
        ));
    };
    let groups = scales.width / 2;
    if groups == 0 || !k.is_multiple_of(groups) {
        return Err(refuse(
            op,
            format!("{groups} factors do not group a {k}-wide row into whole groups"),
        ));
    }
    let group = k / groups;
    let words = extent(op, u64::from(n) * u64::from(k) / u64::from(32 / bits))?;
    ctx.fire(
        op,
        Fire::at(
            DEQUANT,
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
            ArgValue::Ptr(dst),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
        ],
    )?;
    Ok(Tensor::new(dst, n, k, Dtype::Bf16))
}

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
    let bytes = (n as usize).saturating_mul(k as usize).saturating_mul(2);
    let tile = ctx.scratch_stream(op, DECODED_WEIGHT, bytes)? as usize as u64;
    let words = extent(op, u64::from(n) * u64::from(k) / u64::from(32 / bits))?;
    ctx.fire(
        op,
        Fire::at(
            DEQUANT,
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
    gemm::act_x_wt(ctx, op, act, Tensor::new(tile, n, k, Dtype::Bf16), None, y)
}

// NVFP4 projections read at bf16 (W4A16): `codes` are e2m1 nibble pairs,
// `scales` one e4m3 per 16 codes, `tensor_scales` one f32 a row, each plane
// bound as bytes, one row of the plane an expert (or the whole projection).
// A weight is `e2m1(code) · e4m3(scale) · tensor_scale`; the activations
// stay bf16, so a checkpoint's activation scales are not read here.

/// Codes a 16-byte load holds: the by-route arm's step along K.
const CHUNK: u32 = 32;

/// The decoded bf16 tile a dense product reads, as the affine arm keeps it.
const NVFP4_DECODED: &str = "linear.nvfp4.decoded_weight";

const BLOCK: u32 = 256;

pub(crate) struct Planes {
    pub(crate) n: u32,
    pub(crate) k: u32,
}

/// That the three planes hold `slabs` slabs of `n` rows of `k` codes.
pub(crate) fn planes(
    op: &'static str,
    codes: Tensor,
    scales: Tensor,
    tensor_scales: Tensor,
    n: u32,
    k: u32,
) -> Result<Planes, Error> {
    debug_assert_eq!(codes.dtype, Dtype::Nvfp4, "nvfp4 codes keep their dtype");
    if k == 0 || !k.is_multiple_of(CHUNK) {
        return Err(refuse(
            op,
            format!("K is {k}, not a whole number of {CHUNK}-code nvfp4 chunks"),
        ));
    }
    let rows = u64::from(n);
    let wants = [
        ("code", codes, rows * u64::from(k / 2)),
        ("scale", scales, rows * u64::from(k / 16)),
        ("tensor scale", tensor_scales, rows * 4),
    ];
    for (what, plane, bytes) in wants {
        if u64::from(plane.width) != bytes {
            return Err(refuse(
                op,
                format!(
                    "a slab's {what} plane is {} bytes, and {n} rows of {k} codes are {bytes}",
                    plane.width
                ),
            ));
        }
    }
    if scales.rows != codes.rows || tensor_scales.rows != codes.rows {
        return Err(refuse(
            op,
            format!(
                "the planes hold {}, {} and {} slabs; an nvfp4 bank's planes are one \
                 slab a row each",
                codes.rows, scales.rows, tensor_scales.rows
            ),
        ));
    }
    Ok(Planes { n, k })
}

/// The planes of one `[n, k]` projection decoded into a bf16 tile.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_plane(
    ctx: &Ctx,
    op: &'static str,
    codes: Tensor,
    scales: Tensor,
    tensor_scales: Tensor,
    n: u32,
    k: u32,
    seat: GroupSeat,
) -> Result<Tensor, Error> {
    if seat.streams() {
        return Err(refuse(
            op,
            "a decoded plane serves resident projections only, and these planes are \
             seated by a streaming tier",
        ));
    }
    let (codes, scales, tensor_scales) = whole(codes, scales, tensor_scales, n)?;
    planes(op, codes, scales, tensor_scales, n, k)?;
    let bytes = (n as usize).saturating_mul(k as usize).saturating_mul(2);
    let tile = ctx.scratch_stream(op, NVFP4_DECODED, bytes)? as usize as u64;
    let pairs = u32::try_from(u64::from(n) * u64::from(k / 2)).map_err(|_| {
        refuse(
            op,
            format!("{n} rows of {k} codes do not fit a 32-bit launch extent"),
        )
    })?;
    ctx.fire(
        op,
        Fire::at(DEQUANT, symbol("::pie::linear::dequant_nvfp4<::pie::bf16>"))
            .apply(Launch::flat(pairs, BLOCK)),
        &[
            codes.arg(),
            scales.arg(),
            tensor_scales.arg(),
            ArgValue::Ptr(tile),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
        ],
    )?;
    Ok(Tensor::new(tile, n, k, Dtype::Bf16))
}

/// A dense projection's planes are bound one row a weight row; read whole,
/// they are one slab.
fn whole(
    codes: Tensor,
    scales: Tensor,
    tensor_scales: Tensor,
    n: u32,
) -> Result<(Tensor, Tensor, Tensor), Error> {
    let fold = |plane: Tensor| -> Tensor {
        if plane.rows == n && n > 1 {
            Tensor::new(plane.ptr, 1, plane.rows * plane.width, plane.dtype)
        } else {
            plane
        }
    };
    Ok((fold(codes), fold(scales), fold(tensor_scales)))
}

/// The dense product `y = act · Wᵀ`: through a decoded bf16 tile and the
/// dense gemm for bf16 activations, the by-route arm over one expert
/// otherwise.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_matmul(
    ctx: &Ctx,
    op: &'static str,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    tensor_scales: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    let t = dtype_dispatch!(op, act.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let (n, k) = (y.width, act.width);
    if act.rows == 0 {
        return Ok(());
    }
    if act.dtype == Dtype::Bf16 && act.rows > 1 && !seat.streams() {
        let tile = nvfp4_plane(ctx, op, codes, scales, tensor_scales, n, k, seat)?;
        return gemm::act_x_wt(ctx, op, act, tile, None, y);
    }
    let (codes, scales, tensor_scales) = whole(codes, scales, tensor_scales, n)?;
    planes(op, codes, scales, tensor_scales, n, k)?;
    crate::linear::moe::select_nvfp4_routed(
        ctx,
        op,
        t,
        act,
        None,
        codes,
        scales,
        tensor_scales,
        y,
        seat,
        act.rows,
        1,
        1,
    )
}
