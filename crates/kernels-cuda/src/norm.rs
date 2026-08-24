use crate::jit::abi::Tensor;
use crate::jit::abi::{bf16, f16};
use crate::jit::{Ctx, Launch, aligned16};
use kernels::Refusal;
use kernels::Region;
use kernels::Stride;
use kernels::routine::{Const, In, InOut, Out};
use kernels::{Bind, Fire};

use core::ffi::c_void;
use core::ptr::NonNull;

const BLOCK: u32 = 256;

const VBLOCK: u32 = 512;

const WARP: u32 = 32;

pub const ALTUP_EPS: f32 = 1e-5;

pub const RASR_VEC512_ABOVE: i32 = 2560;

pub const MAX_HC_MULT: i32 = 8;

#[must_use]
const fn per_row(rows: i32) -> Launch {
    Launch::per_row(rows.unsigned_abs(), BLOCK)
}

#[must_use]
const fn gated_rms(rows: i32, heads: i32) -> Launch {
    Launch::grid(
        [rows.unsigned_abs(), heads.unsigned_abs(), 1],
        [BLOCK, 1, 1],
    )
}

#[must_use]
const fn elementwise(n: i32) -> Launch {
    Launch::flat(n.unsigned_abs(), BLOCK)
}

#[must_use]
fn elementwise_wide(n: i64) -> Launch {
    let blocks = (n + i64::from(BLOCK) - 1) / i64::from(BLOCK);
    Launch::grid(
        [u32::try_from(blocks).unwrap_or(u32::MAX), 1, 1],
        [BLOCK, 1, 1],
    )
}

#[must_use]
const fn route_rows(rows: i32, width: i32) -> Launch {
    const MAX_BLOCK: u32 = 1024;

    let warps = width.unsigned_abs().div_ceil(WARP);
    let warps = if warps == 0 { 1 } else { warps };
    let block = warps.saturating_mul(WARP);
    let block = if block > MAX_BLOCK { MAX_BLOCK } else { block };
    Launch::grid([rows.unsigned_abs(), 1, 1], [block, 1, 1])
}

fn rows_per_head<P>(dst: &Region<P>, stated_head_dim: i32) -> Result<Launch, Refusal> {
    if stated_head_dim == 0 {
        return Ok(per_row(dst.rows));
    }
    let (w, hd) = (dst.width.unsigned_abs(), stated_head_dim.unsigned_abs());
    if !w.is_multiple_of(hd) {
        return Err(Refusal::Narrow {
            what: "a row that divides by head_dim",
            at: i64::from(dst.width),
        });
    }
    let blocks = dst
        .rows
        .unsigned_abs()
        .checked_mul(w / hd)
        .ok_or(Refusal::Narrow {
            what: "a row count that fits a grid",
            at: i64::from(dst.rows),
        })?;
    Ok(Launch::per_row(blocks, BLOCK))
}

#[must_use]
fn vec8_ok(
    x: *const c_void,
    y: *const c_void,
    weight: *const c_void,
    hidden: i32,
    x_row_stride: Stride,
    y_row_stride: Stride,
) -> bool {
    hidden % 8 == 0
        && *x_row_stride % 8 == 0
        && *y_row_stride % 8 == 0
        && aligned16(x)
        && aligned16(y)
        && aligned16(weight)
}

pub(crate) fn heads<P>(row: &Region<P>, head_dim: i32) -> Result<i32, Refusal> {
    let width = row.width;
    if head_dim <= 0 {
        return Err(Refusal::Empty { what: "head_dim" });
    }
    if width % head_dim != 0 {
        return Err(Refusal::Narrow {
            what: "the row is not a whole number of heads",
            at: i64::from(width),
        });
    }
    Ok(width / head_dim)
}

/// `per_head_dim = 0`: the reduction runs over the whole row. Every other
/// value is the head width the statement pins it to.
const WHOLE_ROW: i32 = 0;

/// `WEIGHT_PLUS_ONE = false`: the scale IS the weight.
fn absolute_bank<T: kernels::Elem>() -> String {
    format!("::pie::norm::rmsnorm<{}, 256>", T::CPP)
}

/// `WEIGHT_PLUS_ONE = true`: the scale is `1 + weight`, folded in float.
/// See `Norm::rmsnorm_plus_one` for why the other convention is a separate
/// point and not a flag.
fn offset_bank<T: kernels::Elem>() -> String {
    format!("::pie::norm::rmsnorm_gemma<{}, 256>", T::CPP)
}

fn streams<P>(row: &Region<P>, hidden_size: i32) -> Result<i32, Refusal> {
    let width = row.width;
    if hidden_size <= 0 {
        return Err(Refusal::Empty {
            what: "the hidden width",
        });
    }
    if width % hidden_size != 0 {
        return Err(Refusal::Narrow {
            what: "the row is not a whole number of hyper-connection streams",
            at: i64::from(width),
        });
    }
    let hc_mult = width / hidden_size;
    hc_mult_ok(hc_mult)?;
    Ok(hc_mult)
}

/// The `Norm` family, claimed. Every body is the launch itself, deriving
/// from the operands what the declaration does not state: a whole-row norm
/// passes [`WHOLE_ROW`] where a per-head one passes the stated head width,
/// and the row count is the result rectangle's.
///
/// FOUR POINTS FIRE ONE LAUNCH, which is why [`rms_row`] is a function and
/// the four bodies are two numbers each: `rmsnorm`, `rmsnorm_per_head`,
/// `rmsnorm_plus_one` and `rmsnorm_per_head_plus_one` differ in the AXIS and
/// in the bank convention, and the convention arrives as the entrypoint's
/// name because `WEIGHT_PLUS_ONE` is a template parameter.
///
/// One point stays on the floor's default body, and the absence is a
/// measured row rather than an oversight:
///
/// * `norm.res_blend` — kimi's variadic ledger item. The text states one
///   value per earlier block and the count grows with the layer, so the
///   statement's arity is a function of where it stands. It resolves
///   through [`crate::CANON`] until the floor carries a `Vararg` mark.
#[kernels_macros::claims]
impl kernels::points::Norm for Ctx<'_> {
    fn rmsnorm_per_head<T: kernels::points::Scalar>(
        &self,
        x: In<Tensor<T>>,
        weight: Const<Tensor<T>>,
        head_dim: u32,
        eps: f32,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        let head_dim = i32::try_from(head_dim).map_err(|_| Refusal::Wide {
            what: "the head width this norm states",
            at: i64::from(head_dim),
            max: i64::from(i32::MAX),
        })?;
        rms_row(self, &absolute_bank::<T>(), x, weight, y, head_dim, eps)
    }

    fn rmsnorm_gated_by<T: kernels::points::Scalar>(
        &self,
        x: In<Tensor<f32>>,
        gate: In<Tensor<T>>,
        weight: Const<Tensor<f32>>,
        heads: u32,
        eps: f32,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        let rect = x.all("the row the heads divide")?;
        let heads = i32::try_from(heads).map_err(|_| Refusal::Wide {
            what: "the head count this norm states",
            at: i64::from(heads),
            max: i64::from(i32::MAX),
        })?;
        let d = rect.width / heads.max(1);
        // `ssm/kda.cuh`'s own grid: one block per (row, head), and the head
        // width as the block width clamped into a launchable range.
        const KDA_BLOCK_MIN: u32 = WARP;
        const KDA_BLOCK_MAX: u32 = 128;
        self.fire(
            Fire::at(
                "ssm/kda.cuh",
                crate::jit::symbol(&format!("::pie::ssm::kda_o_norm_gated<{}>", T::CPP)),
            )
            .apply(Launch::grid(
                [y.rows.unsigned_abs(), heads.unsigned_abs(), 1],
                [d.unsigned_abs().clamp(KDA_BLOCK_MIN, KDA_BLOCK_MAX), 1, 1],
            )),
            &[
                x.arg(),
                gate.arg(),
                weight.arg(),
                y.arg(),
                heads.arg(),
                d.arg(),
                eps.arg(),
            ],
        )
    }

    fn rmsnorm<T: kernels::points::Scalar>(
        &self,
        x: In<Tensor<T>>,
        weight: Const<Tensor<T>>,
        eps: f32,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        rms_row(self, &absolute_bank::<T>(), x, weight, y, WHOLE_ROW, eps)
    }

    /// The offset-bank pair, both delegating to `rmsnorm_gemma` -- the same
    /// `rmsnorm_row` body as `rmsnorm`, instantiated at
    /// `WEIGHT_PLUS_ONE = true` (`kernels/norm/rmsnorm.cuh:226-237`). One
    /// kernel, two conventions, and the declaration is what picks.
    fn rmsnorm_plus_one<T: kernels::points::Scalar>(
        &self,
        x: In<Tensor<T>>,
        weight: Const<Tensor<T>>,
        eps: f32,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        rms_row(self, &offset_bank::<T>(), x, weight, y, WHOLE_ROW, eps)
    }

    fn rmsnorm_per_head_plus_one<T: kernels::points::Scalar>(
        &self,
        x: In<Tensor<T>>,
        weight: Const<Tensor<T>>,
        head_dim: u32,
        eps: f32,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        let head_dim = i32::try_from(head_dim).map_err(|_| Refusal::Wide {
            what: "the head width this norm states",
            at: i64::from(head_dim),
            max: i64::from(i32::MAX),
        })?;
        rms_row(self, &offset_bank::<T>(), x, weight, y, head_dim, eps)
    }

    fn rmsnorm_no_scale<T: kernels::points::Scalar>(
        &self,
        x: In<Tensor<T>>,
        head_dim: u32,
        eps: f32,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        let head_dim = i32::try_from(head_dim).map_err(|_| Refusal::Wide {
            what: "the head width this norm states",
            at: i64::from(head_dim),
            max: i64::from(i32::MAX),
        })?;
        let dst = y.all("the normalised row's width")?;
        let hidden = if head_dim == 0 { dst.width } else { head_dim };
        let launch = rows_per_head(&dst, head_dim)?;
        if launch.empty() {
            return Err(Refusal::Empty { what: "num_rows" });
        }
        self.fire(
            Fire::at(
                "norm/rmsnorm.cuh",
                crate::jit::symbol(&format!("::pie::norm::rmsnorm_no_scale<{}, 256>", T::CPP)),
            )
            .apply(launch),
            &[x.arg(), y.arg(), hidden.arg(), eps.arg()],
        )
    }

    fn rmsnorm_gated<T: kernels::points::Scalar>(
        &self,
        x: In<Tensor<f32>>,
        gate: In<Tensor<T>>,
        weight: Const<Tensor<f32>>,
        head_dim: u32,
        eps: f32,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        // `per_head_dim`, and passing `0` here was the bug the declaration's
        // new `head_dim` exists to close: zero means "the whole row" to
        // `rmsnorm_gated_fp32_in`, so a `[value_heads * value_head_dim]`
        // mixer output was reduced as one vector and `weight[i]` walked off
        // the end of a one-head bank.
        let head_dim = i32::try_from(head_dim).map_err(|_| Refusal::Wide {
            what: "the head width this gated norm states",
            at: i64::from(head_dim),
            max: i64::from(i32::MAX),
        })?;
        let dst = y.all("the normalised row's width")?;
        let hidden = if head_dim == 0 { dst.width } else { head_dim };
        let launch = rows_per_head(&dst, head_dim)?;
        if launch.empty() {
            return Err(Refusal::Empty { what: "num_rows" });
        }
        self.fire(
            Fire::at(
                "norm/rmsnorm.cuh",
                crate::jit::symbol(&format!(
                    "::pie::norm::rmsnorm_gated_f32_in<{}, 256>",
                    T::CPP
                )),
            )
            .apply(launch),
            &[
                x.arg(),
                gate.arg(),
                weight.arg(),
                y.arg(),
                hidden.arg(),
                eps.arg(),
            ],
        )
    }

    fn residual_add<T: kernels::points::Scalar>(
        &self,
        x: In<Tensor<T>>,
        y: InOut<Tensor<T>>,
    ) -> Result<(), Refusal> {
        residual_add(self, y, x)
    }

    fn add_bias<T: kernels::points::Scalar>(
        &self,
        bias: Const<Tensor<T>>,
        out: InOut<Tensor<T>>,
    ) -> Result<(), Refusal> {
        add_bias(self, out, bias)
    }

    fn mul_scalar<T: kernels::points::Scalar>(
        &self,
        s: f32,
        x: InOut<Tensor<T>>,
    ) -> Result<(), Refusal> {
        let rect = x.all("the rectangle's row width")?;
        let Ok(n) = usize::try_from(rect.elements()) else {
            return Err(Refusal::Empty {
                what: "the scaled rectangle's element count",
            });
        };
        self.fire(
            Fire::at(
                "norm/elementwise.cuh",
                crate::jit::symbol(&format!("::pie::norm::scalar_mul<{}>", T::CPP)),
            )
            .apply(Launch::flat(u32::try_from(n).unwrap_or(u32::MAX), BLOCK)),
            &[x.arg(), s.arg(), n.arg()],
        )
    }

    /// `x *= s[0]`, the factor a `[1]` bank on the device.
    ///
    /// The launcher, not a delegation — see this impl's header. Everything
    /// about the geometry is `norm.mul_scalar`'s: one thread per element,
    /// 256 to a block, the count rounded up and the tail threads told to
    /// stop. What differs is one operand's MARK, and a mark is who binds
    /// the slot: `mul_scalar`'s factor comes off the fire's params run,
    /// this one's off the load-time parameter table.
    ///
    /// The `[1]` shape is not checked here and could not be. A
    /// `Const<Tensor<T>>` carries the weight's ADDRESS and no rectangle
    /// (`bind/table.rs`: "a weight's shape is the MODEL's, not the
    /// statement's"), so the one element this reads is the model text's
    /// claim about its own checkpoint, verified where that claim is made —
    /// `baker_load`'s join, which reads gemma's `layer.{l}.ple_scalar` as
    /// `[1]` or refuses.
    fn scale<T: kernels::points::Scalar>(
        &self,
        s: Const<Tensor<T>>,
        x: InOut<Tensor<T>>,
    ) -> Result<(), Refusal> {
        let rect = x.all("the scaled rectangle's row width")?;
        let Ok(n) = usize::try_from(rect.elements()) else {
            return Err(Refusal::Empty {
                what: "the scaled rectangle's element count",
            });
        };
        self.fire(
            Fire::at(
                "norm/elementwise.cuh",
                crate::jit::symbol(&format!("::pie::norm::scale<{}>", T::CPP)),
            )
            .apply(elementwise(rect.elements())),
            &[x.arg(), s.arg(), n.arg()],
        )
    }
}

pub fn rmsnorm_strided_bf16_at(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    weight: Const<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    eps: f32,
) -> Result<(), Refusal> {
    let dst = y.all("the normalised row's width")?;
    let src = x.all("the source row's pitch")?;

    let vec_ok = vec8_ok(
        x.ptr.cast(),
        y.ptr.cast_const().cast(),
        weight.v.cast(),
        dst.width,
        src.stride,
        dst.stride,
    );
    if vec_ok {
        return ctx.fire(
            Fire::at(
                "norm/rmsnorm.cuh",
                "::pie::norm::rmsnorm_vec8<::pie::i32(512), false, false>",
            )
            .apply(Launch::per_row(dst.rows.unsigned_abs(), VBLOCK)),
            &[
                x.arg(),
                weight.arg(),
                y.arg(),
                None::<NonNull<f16>>.arg(),
                dst.width.arg(),
                src.stride.arg(),
                dst.stride.arg(),
                eps.arg(),
            ],
        );
    }
    ctx.fire(
        Fire::at("norm/rmsnorm.cuh", "::pie::norm::rmsnorm<::pie::bf16, 256>")
            .apply(per_row(dst.rows)),
        &[
            x.arg(),
            weight.arg(),
            y.arg(),
            dst.width.arg(),
            src.stride.arg(),
            dst.stride.arg(),
            eps.arg(),
        ],
    )
}

/// `norm/rmsnorm.cuh`'s row norm, and the only place its two entrypoints
/// are named.
///
/// FOUR POINTS FIRE THIS ONE LAUNCH, which is why it is a function rather
/// than four copies of itself in the impl above: `rmsnorm`,
/// `rmsnorm_per_head`, `rmsnorm_plus_one` and `rmsnorm_per_head_plus_one`
/// differ in the AXIS the reduction runs over and in the bank convention,
/// and in nothing else. The axis is a number the caller has; the convention
/// is a separate `__global__` because `WEIGHT_PLUS_ONE` is a template
/// parameter, so it arrives as the entrypoint's name.
fn rms_row<T: crate::RoutineElem>(
    ctx: &Ctx<'_>,
    entrypoint: &str,
    x: In<Tensor<T>>,
    weight: Const<Tensor<T>>,
    y: Out<Tensor<T>>,
    per_head_dim: i32,
    eps: f32,
) -> Result<(), Refusal> {
    let dst = y.all("the normalised row's width")?;
    let hidden = if per_head_dim == 0 {
        dst.width
    } else {
        per_head_dim
    };
    let launch = rows_per_head(&dst, per_head_dim)?;
    if launch.empty() {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    ctx.fire(
        Fire::at("norm/rmsnorm.cuh", crate::jit::symbol(entrypoint)).apply(launch),
        &[
            x.arg(),
            weight.arg(),
            y.arg(),
            hidden.arg(),
            hidden.arg(),
            hidden.arg(),
            eps.arg(),
        ],
    )
}

pub fn rmsnorm_no_scale_at(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    per_head_dim: i32,
    eps: f32,
) -> Result<(), Refusal> {
    let dst = y.all("the normalised row's width")?;
    let hidden = if per_head_dim == 0 {
        dst.width
    } else {
        per_head_dim
    };
    let launch = rows_per_head(&dst, per_head_dim)?;
    if launch.empty() {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    ctx.fire(
        Fire::at(
            "norm/rmsnorm.cuh",
            crate::jit::symbol("::pie::norm::rmsnorm_no_scale<::pie::bf16, 256>"),
        )
        .apply(launch),
        &[x.arg(), y.arg(), hidden.arg(), eps.arg()],
    )
}

pub(crate) fn add_bias<T: crate::RoutineElem>(
    ctx: &Ctx<'_>,
    out: InOut<Tensor<T>>,
    bias: Const<Tensor<T>>,
) -> Result<(), Refusal> {
    let dst = out.all("the biased row's width")?;
    ctx.fire(
        Fire::at(
            "norm/add_bias.cuh",
            crate::jit::symbol(&format!("::pie::norm::add_bias<{}>", T::CPP)),
        )
        .apply(route_rows(dst.rows, dst.width)),
        &[out.arg(), bias.arg(), dst.width.arg()],
    )
}

/// The canon point `norm.residual_add`: add a residual into an accumulator.
///
/// The claim was on `norm::residual_add_rmsnorm` -- the FUSED form -- for as
/// long as the gate that would have said so could not compile. See the note
/// there for what that cost.
pub(crate) fn residual_add<T: crate::RoutineElem>(
    ctx: &Ctx<'_>,
    y: InOut<Tensor<T>>,
    x: In<Tensor<T>>,
) -> Result<(), Refusal> {
    let rect = y.all("the rectangle's row width")?;
    let Ok(n) = usize::try_from(rect.elements()) else {
        return Err(Refusal::Empty {
            what: "the residual rectangle's element count",
        });
    };
    let launch = Launch::flat(u32::try_from(n).unwrap_or(u32::MAX), BLOCK);
    ctx.fire(
        Fire::at(
            "norm/elementwise.cuh",
            crate::jit::symbol(&format!("::pie::norm::residual_add<{}>", T::CPP)),
        )
        .apply(launch),
        &[y.arg(), x.arg(), n.arg()],
    )
}

/// The `Hc` family, claimed. Four of five points land, and every body is the
/// launch itself: one `__global__` out of `norm/dsv4_hc.cuh`.
///
/// EVERY BODY DROPS THE STATED `stream_count`. The declaration states it
/// because the collapsed row is an `Out` the statement allocates and a
/// divisor has to exist before the row does; a body has both rectangles in
/// hand and reads the count back off them (`streams`
/// above, the stack's width over the collapsed one). The `moe.experts`
/// reading exactly.
///
/// `hc.rmsnorm_f32` crosses a bf16 PIN BY NAME rather than with a cast no
/// kernel stands behind: `::pie::norm::hc_rmsnorm_to_f32` is instantiated at
/// bf16 by a literal symbol and nowhere else, so a second element wants a
/// second instantiation in the `.cuh`. The `gate.sigmoid_mul` precedent.
///
/// One point stays on the floor's default body:
///
/// * `hc.collapse` — `hc_head_postprocess` reads TWO planes where the
///   statement names one: an `[N, streams]` f32 `mixes` (the head gate
///   logits, "after GEMM" in the kernel's own comment) beside the
///   `[N, streams, hidden]` residual stack. The legacy call site passed the
///   bf16 stack for the f32 `mixes` slot — which reads a stack's leading
///   bytes as gates — and that is a caller's bug, not a shape to reproduce.
///   [`hc_head_postprocess`] keeps the launch and [`crate::CANON`] keeps the
///   resolution; the day a text states the projection the kernel asks for,
///   the body is four lines.
#[kernels_macros::claims]
impl kernels::points::Hc for Ctx<'_> {
    fn expand<T: kernels::points::Scalar>(
        &self,
        x: In<Tensor<T>>,
        streams: u32,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = streams;
        let dst = y.all("the hyper-connection row's width")?;
        let hc_mult = self::streams(&dst, x.width)?;
        let total = i64::from(x.rows) * i64::from(x.width);
        self.fire(
            Fire::at(
                "norm/dsv4_hc.cuh",
                crate::jit::symbol(&format!("::pie::norm::hc_expand<{}>", T::CPP)),
            )
            .apply(elementwise_wide(total)),
            &[
                x.arg(),
                y.arg(),
                x.rows.arg(),
                hc_mult.arg(),
                x.width.arg(),
            ],
        )
    }

    fn rmsnorm_f32<T: kernels::points::Scalar>(
        &self,
        streams: In<Tensor<T>>,
        eps: f32,
        y: Out<Tensor<f32>>,
    ) -> Result<(), Refusal> {
        if T::CPP != <bf16 as kernels::Elem>::CPP {
            return Err(Refusal::Absent {
                what: "hc.rmsnorm_f32 at an element other than bf16",
            });
        }
        let input: In<Tensor<bf16>> = In {
            ptr: streams.ptr.cast::<bf16>(),
            rows: streams.rows,
            width: streams.width,
        };
        let dst = y.all("the normalised row's width")?;
        self.fire(
            Fire::at(
                "norm/dsv4_hc.cuh",
                "::pie::norm::hc_rmsnorm_to_f32<::pie::bf16, 256>",
            )
            .apply(per_row(dst.rows)),
            &[input.arg(), y.arg(), dst.width.arg(), eps.arg()],
        )
    }

    fn gates<T: kernels::points::Scalar>(
        &self,
        normed: In<Tensor<f32>>,
        streams: In<Tensor<T>>,
        scale: Const<Tensor<f32>>,
        base: Const<Tensor<f32>>,
        stream_count: u32,
        gate_eps: f32,
        alpha: f32,
        sinkhorn: u32,
        x: Out<Tensor<T>>,
        post_mix: Out<Tensor<f32>>,
        comb_mix: Out<Tensor<f32>>,
    ) -> Result<(), Refusal> {
        let _ = stream_count;
        let sinkhorn = i32::try_from(sinkhorn).map_err(|_| Refusal::Wide {
            what: "the Sinkhorn iteration count this statement states",
            at: i64::from(sinkhorn),
            max: i64::from(i32::MAX),
        })?;
        // THE STATEMENT'S RESULT ORDER IS NOT THE LAUNCH'S. A text reads
        // `(x, post_mix, comb_mix)` — the row it runs on first, because that
        // is the one it consumes — and the kernel writes `post_mix`,
        // `comb_mix`, `layer_input`. The mapping is here, named, once.
        //
        // THE RECTANGLE ANSWERS THREE OF THE SIX. `n` is the row count and
        // `hidden_size` the collapsed row's width, both on `x`; the stream
        // count is what the residual stack is wider by, which is what
        // `streams` computes and what `Hc::fold` reads the same way. Only
        // the three CONSTANTS are asked for.
        let dst = x.all("the hyper-connection row's width")?;
        let (n, hidden_size) = (dst.rows, dst.width);
        let hc_mult = self::streams(&streams.all("the residual row's width")?, hidden_size)?;
        self.fire(
            Fire::at(
                "norm/dsv4_hc.cuh",
                crate::jit::symbol(&format!("::pie::norm::hc_pre_postprocess<{}, 256>", T::CPP)),
            )
            .apply(per_row(n)),
            &[
                normed.arg(),
                scale.arg(),
                base.arg(),
                streams.arg(),
                post_mix.arg(),
                comb_mix.arg(),
                x.arg(),
                hc_mult.arg(),
                hidden_size.arg(),
                gate_eps.arg(),
                alpha.arg(),
                sinkhorn.arg(),
            ],
        )
    }

    fn fold<T: kernels::points::Scalar>(
        &self,
        x: In<Tensor<T>>,
        streams: In<Tensor<T>>,
        post_mix: In<Tensor<f32>>,
        comb_mix: In<Tensor<f32>>,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        let dst = y.all("the hyper-connection row's width")?;
        let hc_mult = self::streams(&dst, x.width)?;
        let total = i64::from(dst.rows) * i64::from(x.width);
        self.fire(
            Fire::at(
                "norm/dsv4_hc.cuh",
                crate::jit::symbol(&format!("::pie::norm::hc_post<{}>", T::CPP)),
            )
            .apply(elementwise_wide(total)),
            &[
                x.arg(),
                streams.arg(),
                post_mix.arg(),
                comb_mix.arg(),
                y.arg(),
                dst.rows.arg(),
                hc_mult.arg(),
                x.width.arg(),
            ],
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn hc_head_postprocess<T: crate::RoutineElem>(
    ctx: &Ctx<'_>,
    mixes: In<Tensor<f32>>,
    scale: Const<Tensor<f32>>,
    base: Const<Tensor<f32>>,
    residual: In<Tensor<T>>,
    out: Out<Tensor<T>>,
    hc_eps: Const<f32>,
) -> Result<(), Refusal> {
    // [`hc_pre_postprocess`]'s reading, off the one result this form declares.
    let dst = out.all("the collapsed row's width")?;
    let (n, hidden_size) = (dst.rows, dst.width);
    let hc_mult = streams(&residual.all("the residual row's width")?, hidden_size)?;
    let hc_eps = *hc_eps;
    ctx.fire(
        Fire::at(
            "norm/dsv4_hc.cuh",
            crate::jit::symbol(&format!(
                "::pie::norm::hc_head_postprocess<{}, 256>",
                T::CPP
            )),
        )
        .apply(per_row(n)),
        &[
            mixes.arg(),
            scale.arg(),
            base.arg(),
            residual.arg(),
            out.arg(),
            hc_mult.arg(),
            hidden_size.arg(),
            hc_eps.arg(),
        ],
    )
}

/// NO CANON AND NO POINT ANSWERS THROUGH THIS ONE. It reads the sink at
/// f32 and does not rebase the lse, which is neither half of what
/// `attention.sink` states: the checkpoints ship the sink at the model's
/// element and the lse arrives in base two. `attn::attn_sink_rescale` is
/// the kernel that answers the point; this launcher stays only because the
/// legacy dsv4 text still fires it, and dies with `model-dsl-legacy`.
pub fn attn_sink_correction<T: crate::RoutineElem>(
    ctx: &Ctx<'_>,
    out: InOut<Tensor<T>>,
    lse: In<Tensor<f32>>,
    sink: Const<Tensor<f32>>,
    head_dim: Const<i32>,
) -> Result<(), Refusal> {
    let head_dim = *head_dim;

    let dst = out.all("the row whose heads are counted")?;
    let num_heads = heads(&dst, head_dim)?;
    ctx.fire(
        Fire::at(
            "norm/dsv4_hc.cuh",
            crate::jit::symbol(&format!("::pie::norm::attn_sink_correction<{}>", T::CPP)),
        )
        .apply(gated_rms(dst.rows, num_heads)),
        &[
            out.arg(),
            lse.arg(),
            sink.arg(),
            num_heads.arg(),
            head_dim.arg(),
        ],
    )
}

fn hc_mult_ok(hc_mult: i32) -> Result<(), Refusal> {
    if hc_mult > MAX_HC_MULT {
        return Err(Refusal::Wide {
            what: "hc_mult, which `hc_post` unrolls into a register array",
            at: i64::from(hc_mult),
            max: i64::from(MAX_HC_MULT),
        });
    }
    Ok(())
}
