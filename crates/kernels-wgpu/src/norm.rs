use crate::plane::{Bind, Const, Ctx, Fire, In, InOut, Out, elementwise_rows};
use crate::points::{Payload, at_bf16};
use kernels::plane::Refusal;

fn per_axis(width: i32, axis: i32, rows: i32) -> Result<[u32; 3], Refusal> {
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

#[kernels_macros::claims]
impl kernels::points::Norm for Ctx<'_> {
    fn rmsnorm<T: kernels::points::Scalar>(
        &self,
        x: In<Payload<T>>,
        weight: Const<Payload<T>>,
        eps: f32,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("norm.rmsnorm at an element other than bf16")?;
        rms_row(self, x, weight, y, eps, x.width, 0)
    }

    fn rmsnorm_per_head<T: kernels::points::Scalar>(
        &self,
        x: In<Payload<T>>,
        weight: Const<Payload<T>>,
        head_dim: u32,
        eps: f32,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("norm.rmsnorm_per_head at an element other than bf16")?;
        rms_row(self, x, weight, y, eps, axis_of(head_dim)?, 0)
    }

    fn rmsnorm_plus_one<T: kernels::points::Scalar>(
        &self,
        x: In<Payload<T>>,
        weight: Const<Payload<T>>,
        eps: f32,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("norm.rmsnorm_plus_one at an element other than bf16")?;
        rms_row(self, x, weight, y, eps, x.width, 1)
    }

    fn rmsnorm_per_head_plus_one<T: kernels::points::Scalar>(
        &self,
        x: In<Payload<T>>,
        weight: Const<Payload<T>>,
        head_dim: u32,
        eps: f32,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("norm.rmsnorm_per_head_plus_one at an element other than bf16")?;
        rms_row(self, x, weight, y, eps, axis_of(head_dim)?, 1)
    }

    fn rmsnorm_no_scale<T: kernels::points::Scalar>(
        &self,
        x: In<Payload<T>>,
        head_dim: u32,
        eps: f32,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("norm.rmsnorm_no_scale at an element other than bf16")?;
        let axis = axis_of(head_dim)?;
        self.fire(
            Fire::at("norm/vector.wgsl", "vnorm_single_row_bfloat16")
                .apply(per_axis(x.width, axis, x.rows)?),
            &[x.arg(), y.arg(), eps.arg(), axis.arg()],
        )
    }

    fn residual_add<T: kernels::points::Scalar>(
        &self,
        x: In<Payload<T>>,
        y: InOut<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("norm.residual_add at an element other than bf16")?;
        self.fire(
            Fire::at("norm/residual_add.wgsl", "residual_add_bfloat16")
                .apply(elementwise_rows(y.width, y.rows)?),
            &[x.arg(), y.ptr.arg(), y.arg()],
        )
    }

    fn add_bias<T: kernels::points::Scalar>(
        &self,
        bias: Const<Payload<T>>,
        out: InOut<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("norm.add_bias at an element other than bf16")?;
        let width = out.width;
        self.fire(
            Fire::at("norm/add_bias.wgsl", "add_bias_bfloat16")
                .apply(elementwise_rows(width, out.rows)?),
            &[out.arg(), bias.arg(), width.arg()],
        )
    }

    fn scale<T: kernels::points::Scalar>(
        &self,
        s: Const<Payload<T>>,
        x: InOut<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("norm.scale at an element other than bf16")?;
        self.fire(
            Fire::at("norm/layer_scalar.wgsl", "layer_scalar_mul_bfloat16")
                .apply(elementwise_rows(x.width, x.rows)?),
            &[x.ptr.arg(), s.arg(), x.arg()],
        )
    }
}

fn axis_of(head_dim: u32) -> Result<i32, Refusal> {
    i32::try_from(head_dim).map_err(|_| Refusal::Wide {
        what: "the head width this norm states",
        at: i64::from(head_dim),
        max: i64::from(i32::MAX),
    })
}

fn rms_row<T: kernels::points::Scalar>(
    ctx: &Ctx<'_>,
    x: In<Payload<T>>,
    weight: Const<Payload<T>>,
    y: Out<Payload<T>>,
    eps: f32,
    axis: i32,
    plus_one: u32,
) -> Result<(), Refusal> {
    ctx.fire(
        Fire::at("norm/rms.wgsl", "rms_single_row_bfloat16")
            .apply(per_axis(x.width, axis, x.rows)?),
        &[
            x.arg(),
            weight.arg(),
            y.arg(),
            eps.arg(),
            axis.arg(),
            1u32.arg(),
            plus_one.arg(),
            1.0f32.arg(),
        ],
    )
}
