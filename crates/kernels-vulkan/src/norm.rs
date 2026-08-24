use crate::plane::{Bind, Const, Ctx, Fire, In, InOut, Out, elementwise, elementwise_rows};
use kernels::plane::Refusal;

pub fn per_axis(width: i32, axis: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if width <= 0 {
        return Err(Refusal::Empty { what: "width" });
    }
    if axis <= 0 {
        return Err(Refusal::Empty { what: "axis" });
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
    let axes = (width.unsigned_abs() / axis.unsigned_abs()) * rows.unsigned_abs();
    let lanes = axes.checked_mul(256).ok_or(Refusal::Grid {
        what: "axes * the workgroup width",
        at: i64::from(axes) * 256,
    })?;
    Ok([lanes, 1, 1])
}

pub fn per_head_row(heads: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if heads <= 0 {
        return Err(Refusal::Empty { what: "heads" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    Ok([256, heads.unsigned_abs(), rows.unsigned_abs()])
}

#[kernels_macros::claims]
impl kernels::points::Norm for Ctx<'_> {
    fn rmsnorm<T: kernels::points::Scalar>(
        &self,
        x: In<crate::points::Handle<T>>,
        weight: Const<crate::points::Handle<T>>,
        eps: f32,
        y: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>("norm.rmsnorm, at an element this plane does not instantiate")?;
        let row = x.all("the normalised row's width")?;
        self.fire(
            Fire::at(
                crate::plane::module_path("rms_single_row_bfloat16", self.best()),
                "rms_single_row_bfloat16",
            )
            .apply(per_axis(row.width, row.width, row.rows)?),
            &[
                x.arg(),
                weight.arg(),
                y.arg(),
                eps.arg(),
                row.width.arg(),
                1u32.arg(),
                0u32.arg(),
                1.0f32.arg(),
            ],
        )
    }

    fn rmsnorm_per_head<T: kernels::points::Scalar>(
        &self,
        x: In<crate::points::Handle<T>>,
        weight: Const<crate::points::Handle<T>>,
        head_dim: u32,
        eps: f32,
        y: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "norm.rmsnorm_per_head, at an element this plane does not instantiate",
        )?;
        let row = x.all("the row the heads divide")?;
        let axis = crate::points::stated("the head width this norm states", head_dim)?;
        self.fire(
            Fire::at(
                crate::plane::module_path("rms_single_row_bfloat16", self.best()),
                "rms_single_row_bfloat16",
            )
            .apply(per_axis(row.width, axis, row.rows)?),
            &[
                x.arg(),
                weight.arg(),
                y.arg(),
                eps.arg(),
                axis.arg(),
                1u32.arg(),
                0u32.arg(),
                1.0f32.arg(),
            ],
        )
    }

    fn rmsnorm_plus_one<T: kernels::points::Scalar>(
        &self,
        x: In<crate::points::Handle<T>>,
        weight: Const<crate::points::Handle<T>>,
        eps: f32,
        y: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "norm.rmsnorm_plus_one, at an element this plane does not instantiate",
        )?;
        let row = x.all("the normalised row's width")?;
        self.fire(
            Fire::at(
                crate::plane::module_path("rms_single_row_bfloat16", self.best()),
                "rms_single_row_bfloat16",
            )
            .apply(per_axis(row.width, row.width, row.rows)?),
            &[
                x.arg(),
                weight.arg(),
                y.arg(),
                eps.arg(),
                row.width.arg(),
                1u32.arg(),
                1u32.arg(),
                1.0f32.arg(),
            ],
        )
    }

    fn rmsnorm_per_head_plus_one<T: kernels::points::Scalar>(
        &self,
        x: In<crate::points::Handle<T>>,
        weight: Const<crate::points::Handle<T>>,
        head_dim: u32,
        eps: f32,
        y: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "norm.rmsnorm_per_head_plus_one, at an element this plane does not instantiate",
        )?;
        let row = x.all("the row the heads divide")?;
        let axis = crate::points::stated("the head width this norm states", head_dim)?;
        self.fire(
            Fire::at(
                crate::plane::module_path("rms_single_row_bfloat16", self.best()),
                "rms_single_row_bfloat16",
            )
            .apply(per_axis(row.width, axis, row.rows)?),
            &[
                x.arg(),
                weight.arg(),
                y.arg(),
                eps.arg(),
                axis.arg(),
                1u32.arg(),
                1u32.arg(),
                1.0f32.arg(),
            ],
        )
    }

    fn rmsnorm_no_scale<T: kernels::points::Scalar>(
        &self,
        x: In<crate::points::Handle<T>>,
        head_dim: u32,
        eps: f32,
        y: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "norm.rmsnorm_no_scale, at an element this plane does not instantiate",
        )?;
        let row = x.all("the row the heads divide")?;
        let axis = crate::points::stated("the head width this norm states", head_dim)?;
        self.fire(
            Fire::at(
                crate::plane::module_path("vnorm_single_row_bfloat16", self.best()),
                "vnorm_single_row_bfloat16",
            )
            .apply(per_axis(row.width, axis, row.rows)?),
            &[x.arg(), y.arg(), eps.arg(), axis.arg()],
        )
    }

    fn rmsnorm_gated<T: kernels::points::Scalar>(
        &self,
        x: In<crate::points::Handle<f32>>,
        gate: In<crate::points::Handle<T>>,
        weight: Const<crate::points::Handle<f32>>,
        head_dim: u32,
        eps: f32,
        y: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "norm.rmsnorm_gated, at a gate element this plane does not instantiate",
        )?;
        let row = x.all("the row the value heads divide")?;
        let vd = crate::points::stated("the head width this gated norm states", head_dim)?;
        let heads =
            crate::points::heads("the value heads this gated norm divides by", row.width, vd)?;

        self.fire(
            Fire::at(
                crate::plane::module_path("gated_rms_bfloat16", self.best()),
                "gated_rms_bfloat16",
            )
            .apply(per_head_row(heads, row.rows)?),
            &[
                x.arg(),
                gate.arg(),
                weight.arg(),
                y.arg(),
                eps.arg(),
                vd.arg(),
            ],
        )
    }

    fn rmsnorm_gated_by<T: kernels::points::Scalar>(
        &self,
        x: In<crate::points::Handle<f32>>,
        gate: In<crate::points::Handle<T>>,
        weight: Const<crate::points::Handle<f32>>,
        heads: u32,
        eps: f32,
        y: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "norm.rmsnorm_gated_by, at a gate element this plane does not instantiate",
        )?;
        let row = x.all("the row the heads divide")?;
        let n = crate::points::stated("the head count this norm states", heads)?;
        let vd = crate::points::heads("the head width this norm divides into", row.width, n)?;

        self.fire(
            Fire::at(
                crate::plane::module_path("gated_rms_bfloat16", self.best()),
                "gated_rms_bfloat16",
            )
            .apply(per_head_row(n, row.rows)?),
            &[
                x.arg(),
                gate.arg(),
                weight.arg(),
                y.arg(),
                eps.arg(),
                vd.arg(),
            ],
        )
    }

    fn residual_add<T: kernels::points::Scalar>(
        &self,
        x: In<crate::points::Handle<T>>,
        y: InOut<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "norm.residual_add, at an element this plane does not instantiate",
        )?;
        let row = y.all("the residual stream's rectangle")?;
        self.fire(
            Fire::at(
                crate::plane::module_path("residual_add_bfloat16", self.best()),
                "residual_add_bfloat16",
            )
            .apply(elementwise(row.width, row.rows)?),
            &[y.ptr.arg(), x.arg(), y.arg()],
        )
    }

    fn add_bias<T: kernels::points::Scalar>(
        &self,
        bias: Const<crate::points::Handle<T>>,
        out: InOut<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "norm.add_bias, at an element this plane does not instantiate",
        )?;
        let row = out.all("the biased rectangle's row width")?;
        self.fire(
            Fire::at(
                crate::plane::module_path("add_bias_bfloat16", self.best()),
                "add_bias_bfloat16",
            )
            .apply(elementwise_rows(row.width, row.rows)?),
            &[out.arg(), bias.arg(), row.width.arg()],
        )
    }

    fn scale<T: kernels::points::Scalar>(
        &self,
        s: Const<crate::points::Handle<T>>,
        x: InOut<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>("norm.scale, at an element this plane does not instantiate")?;
        let row = x.all("the scaled rectangle's row width")?;
        self.fire(
            Fire::at(
                crate::plane::module_path("layer_scalar_mul_bfloat16", self.best()),
                "layer_scalar_mul_bfloat16",
            )
            .apply(elementwise(row.width, row.rows)?),
            &[x.ptr.arg(), s.arg(), x.arg()],
        )
    }
}
