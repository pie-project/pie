use kernels::Grid;
use kernels::plane::Refusal;
use kernels::points::Scalar;

use crate::plane::{
    Bind, Const, Ctx, Fire, In, InOut, Out, Tensor, bf16, elementwise, elementwise_rows,
};
use crate::points::{self, Handle};

fn rms_threads(axis: i32) -> Result<u32, Refusal> {
    if axis <= 0 {
        return Err(Refusal::Empty { what: "axis" });
    }
    Ok(axis.unsigned_abs().div_ceil(4).min(1024))
}

fn rms_grid(width: i32, axis: i32, rows: i32) -> Result<([u32; 3], [u32; 3]), Refusal> {
    if width <= 0 {
        return Err(Refusal::Empty { what: "width" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if axis > width {
        return Err(Refusal::Wide {
            what: "axis",
            at: i64::from(axis),
            max: i64::from(width),
        });
    }
    let t = rms_threads(axis)?;
    let norms = width.unsigned_abs() / axis.unsigned_abs();
    let lanes = t
        .checked_mul(norms)
        .and_then(|n| n.checked_mul(rows.unsigned_abs()))
        .ok_or(Refusal::Grid {
            what: "axis threads * norms per row * rows",
            at: i64::from(t) * i64::from(norms) * i64::from(rows),
        })?;
    Ok(([lanes, 1, 1], [t, 1, 1]))
}

fn head_row_grid(threads: u32, heads: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if heads <= 0 {
        return Err(Refusal::Empty { what: "heads" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    Ok([threads, heads.unsigned_abs(), rows.unsigned_abs()])
}

#[allow(clippy::too_many_arguments)]
fn rms_row(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    w: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    eps: f32,
    axis: i32,
    w_stride: u32,
    plus_one: u32,
    gain: f32,
    rows: i32,
) -> Result<(), Refusal> {
    let (lanes, group) = rms_grid(x.width, axis, rows)?;
    ctx.fire(
        Fire::at("norm/rms.metal", "rms_single_row_bfloat16").apply(Grid::of(lanes, group)),
        &[
            x.arg(),
            w.arg(),
            out.arg(),
            eps.arg(),
            axis.arg(),
            w_stride.arg(),
            plus_one.arg(),
            gain.arg(),
        ],
    )
}

fn head_width(vd: i32) -> Result<u32, Refusal> {
    if vd <= 0 {
        return Err(Refusal::Empty { what: "vd" });
    }
    if vd > 1024 {
        return Err(Refusal::Wide {
            what: "vd",
            at: i64::from(vd),
            max: 1024,
        });
    }
    Ok(vd.unsigned_abs())
}

const DENSE_BANK: u32 = 1;

const ABSOLUTE_BANK: u32 = 0;

const OFFSET_BANK: u32 = 1;

const UNIT_GAIN: f32 = 1.0;

#[kernels_macros::claims]
impl kernels::points::Norm for Ctx<'_> {
    fn rmsnorm<T: Scalar>(
        &self,
        x: In<Handle<T>>,
        weight: Const<Handle<T>>,
        eps: f32,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`norm.rmsnorm`, at an element this plane does not stamp";
        let x = points::input::<T, bf16>(x, WHAT)?;
        rms_row(
            self,
            x,
            points::weight::<T, bf16>(weight, WHAT)?,
            points::result::<T, bf16>(y, WHAT)?,
            eps,
            x.width,
            DENSE_BANK,
            ABSOLUTE_BANK,
            UNIT_GAIN,
            x.rows,
        )
    }

    fn rmsnorm_per_head<T: Scalar>(
        &self,
        x: In<Handle<T>>,
        weight: Const<Handle<T>>,
        head_dim: u32,
        eps: f32,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`norm.rmsnorm_per_head`, at an element this plane does not stamp";
        let x = points::input::<T, bf16>(x, WHAT)?;
        rms_row(
            self,
            x,
            points::weight::<T, bf16>(weight, WHAT)?,
            points::result::<T, bf16>(y, WHAT)?,
            eps,
            points::stated(head_dim, "the head width this norm states")?,
            DENSE_BANK,
            ABSOLUTE_BANK,
            UNIT_GAIN,
            x.rows,
        )
    }

    fn rmsnorm_plus_one<T: Scalar>(
        &self,
        x: In<Handle<T>>,
        weight: Const<Handle<T>>,
        eps: f32,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`norm.rmsnorm_plus_one`, at an element this plane does not stamp";
        let x = points::input::<T, bf16>(x, WHAT)?;
        rms_row(
            self,
            x,
            points::weight::<T, bf16>(weight, WHAT)?,
            points::result::<T, bf16>(y, WHAT)?,
            eps,
            x.width,
            DENSE_BANK,
            OFFSET_BANK,
            UNIT_GAIN,
            x.rows,
        )
    }

    fn rmsnorm_per_head_plus_one<T: Scalar>(
        &self,
        x: In<Handle<T>>,
        weight: Const<Handle<T>>,
        head_dim: u32,
        eps: f32,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str =
            "`norm.rmsnorm_per_head_plus_one`, at an element this plane does not stamp";
        let x = points::input::<T, bf16>(x, WHAT)?;
        rms_row(
            self,
            x,
            points::weight::<T, bf16>(weight, WHAT)?,
            points::result::<T, bf16>(y, WHAT)?,
            eps,
            points::stated(head_dim, "the head width this norm states")?,
            DENSE_BANK,
            OFFSET_BANK,
            UNIT_GAIN,
            x.rows,
        )
    }

    fn rmsnorm_no_scale<T: Scalar>(
        &self,
        x: In<Handle<T>>,
        head_dim: u32,
        eps: f32,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`norm.rmsnorm_no_scale`, at an element this plane does not stamp";
        let x = points::input::<T, bf16>(x, WHAT)?;
        let y = points::result::<T, bf16>(y, WHAT)?;
        let axis = points::stated(head_dim, "the head width this norm states")?;
        let (lanes, group) = rms_grid(x.width, axis, x.rows)?;
        self.fire(
            Fire::at("norm/vector.metal", "vnorm_single_row_bfloat16")
                .apply(Grid::of(lanes, group)),
            &[x.arg(), y.arg(), eps.arg(), axis.arg()],
        )
    }

    fn rmsnorm_gated<T: Scalar>(
        &self,
        x: In<Handle<f32>>,
        gate: In<Handle<T>>,
        weight: Const<Handle<f32>>,
        head_dim: u32,
        eps: f32,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`norm.rmsnorm_gated`, at an element this plane does not stamp";
        let x = points::input::<f32, f32>(x, WHAT)?;
        let vd = points::stated(head_dim, "the value-head width this fold states")?;
        let t = head_width(vd)?;

        if x.width <= 0 || x.width % vd != 0 {
            return Err(Refusal::Narrow {
                what: "the normed row does not divide by the value-head width this fold states",
                at: i64::from(x.width),
            });
        }
        let grid = head_row_grid(t, x.width / vd, x.rows)?;
        self.fire(
            Fire::at("norm/gated_rms.metal", "gated_rms_f32_bfloat16")
                .apply(Grid::of(grid, [t, 1, 1])),
            &[
                x.arg(),
                points::input::<T, bf16>(gate, WHAT)?.arg(),
                points::weight::<f32, f32>(weight, WHAT)?.arg(),
                points::result::<T, bf16>(y, WHAT)?.arg(),
                eps.arg(),
                vd.arg(),
            ],
        )
    }

    fn rmsnorm_gated_by<T: Scalar>(
        &self,
        x: In<Handle<f32>>,
        gate: In<Handle<T>>,
        weight: Const<Handle<f32>>,
        heads: u32,
        eps: f32,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`norm.rmsnorm_gated_by`, at an element this plane does not stamp";
        let x = points::input::<f32, f32>(x, WHAT)?;
        let heads = points::stated(heads, "the head count this norm states")?;
        if heads <= 0 {
            return Err(Refusal::Empty { what: "heads" });
        }
        if x.width <= 0 || x.width % heads != 0 {
            return Err(Refusal::Narrow {
                what: "the normed row does not divide by the head count this norm states",
                at: i64::from(x.width),
            });
        }
        let vd = x.width / heads;
        let t = head_width(vd)?;
        let grid = head_row_grid(t, heads, x.rows)?;
        self.fire(
            Fire::at("norm/gated_rms.metal", "gated_rms_by_f32_bfloat16")
                .apply(Grid::of(grid, [t, 1, 1])),
            &[
                x.arg(),
                points::input::<T, bf16>(gate, WHAT)?.arg(),
                points::weight::<f32, f32>(weight, WHAT)?.arg(),
                points::result::<T, bf16>(y, WHAT)?.arg(),
                eps.arg(),
                vd.arg(),
            ],
        )
    }

    fn residual_add<T: Scalar>(
        &self,
        x: In<Handle<T>>,
        y: InOut<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`norm.residual_add`, at an element this plane does not stamp";
        let y = points::in_place::<T, bf16>(y, WHAT)?;
        self.fire(
            Fire::at("norm/residual_add.metal", "residual_add_bfloat16")
                .apply(Grid::of(elementwise(y.width, y.rows)?, [256, 1, 1])),
            &[
                points::input::<T, bf16>(x, WHAT)?.arg(),
                points::read_half(y).arg(),
                points::write_half(y).arg(),
            ],
        )
    }

    fn add_bias<T: Scalar>(
        &self,
        bias: Const<Handle<T>>,
        out: InOut<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`norm.add_bias`, at an element this plane does not stamp";
        let out = points::in_place::<T, bf16>(out, WHAT)?;
        let lanes = elementwise_rows(out.width, out.rows)?;
        self.fire(
            Fire::at("norm/add_bias.metal", "add_bias_bfloat16")
                .apply(Grid::of(lanes, [lanes[0].min(256), 1, 1])),
            &[
                out.arg(),
                points::weight::<T, bf16>(bias, WHAT)?.arg(),
                out.width.arg(),
            ],
        )
    }

    fn mul_scalar<T: Scalar>(&self, s: f32, x: InOut<Handle<T>>) -> Result<(), Refusal> {
        const WHAT: &str = "`norm.mul_scalar`, at an element this plane does not stamp";
        let x = points::in_place::<T, bf16>(x, WHAT)?;
        self.fire(
            Fire::at(
                "norm/layer_scalar.metal",
                "layer_scalar_mul_stated_bfloat16",
            )
            .apply(Grid::of(elementwise(x.width, x.rows)?, [256, 1, 1])),
            &[
                points::read_half(x).arg(),
                s.arg(),
                points::write_half(x).arg(),
            ],
        )
    }

    fn scale<T: Scalar>(&self, s: Const<Handle<T>>, x: InOut<Handle<T>>) -> Result<(), Refusal> {
        const WHAT: &str = "`norm.scale`, at an element this plane does not stamp";
        let x = points::in_place::<T, bf16>(x, WHAT)?;
        self.fire(
            Fire::at("norm/layer_scalar.metal", "layer_scalar_mul_bfloat16")
                .apply(Grid::of(elementwise(x.width, x.rows)?, [256, 1, 1])),
            &[
                points::read_half(x).arg(),
                points::weight::<T, bf16>(s, WHAT)?.arg(),
                points::write_half(x).arg(),
            ],
        )
    }
}

#[kernels_macros::claims]
impl kernels::points::Hc for Ctx<'_> {}
