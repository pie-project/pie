use crate::jit::abi::Tensor;
use crate::jit::abi::{bf16, f16};
use crate::jit::{Ctx, Launch, aligned16};
use kernels::Refusal;
use kernels::Region;
use kernels::Stride;
use kernels::plane::{Const, In, InOut, Out};
use kernels::{Bind, Fire};

use core::ffi::c_void;
use core::ptr::NonNull;

const BLOCK: u32 = 256;

const VBLOCK: u32 = 512;

const WARP: u32 = 32;

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

const WHOLE_ROW: i32 = 0;

fn absolute_bank<T: kernels::Elem>() -> String {
    format!("::pie::norm::rmsnorm<{}, 256>", T::CPP)
}

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
            &[x.arg(), y.arg(), x.rows.arg(), hc_mult.arg(), x.width.arg()],
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
