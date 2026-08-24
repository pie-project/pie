use crate::plane::{Bind, Ctx, Fire, In, Out, elementwise};
use kernels::plane::Refusal;

#[kernels_macros::claims]
impl kernels::points::Mlp for Ctx<'_> {
    fn swiglu<T: kernels::points::Scalar>(
        &self,
        packed: In<crate::points::Handle<T>>,
        intermediate: u32,
        y: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>("mlp.swiglu, at an element this plane does not instantiate")?;
        let cut = halves(self, packed, intermediate)?;
        self.fire(
            Fire::at(
                crate::plane::module_path("silu_mul_bfloat16", self.best()),
                "silu_mul_bfloat16",
            )
            .apply(elementwise(cut.width, cut.rows)?),
            &[cut.gate.arg(), cut.up.arg(), y.arg()],
        )
    }

    fn geglu_tanh<T: kernels::points::Scalar>(
        &self,
        gate: In<crate::points::Handle<T>>,
        up: In<crate::points::Handle<T>>,
        y: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "mlp.geglu_tanh, at an element this plane does not instantiate",
        )?;
        let row = gate.all("the gate half's rectangle")?;
        self.fire(
            Fire::at(
                crate::plane::module_path("geglu_tanh_bfloat16", self.best()),
                "geglu_tanh_bfloat16",
            )
            .apply(elementwise(row.width, row.rows)?),
            &[gate.arg(), up.arg(), y.arg()],
        )
    }

    fn geglu_tanh_packed<T: kernels::points::Scalar>(
        &self,
        packed: In<crate::points::Handle<T>>,
        intermediate: u32,
        y: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "mlp.geglu_tanh_packed, at an element this plane does not instantiate",
        )?;
        let cut = halves(self, packed, intermediate)?;
        self.fire(
            Fire::at(
                crate::plane::module_path("geglu_tanh_bfloat16", self.best()),
                "geglu_tanh_bfloat16",
            )
            .apply(elementwise(cut.width, cut.rows)?),
            &[cut.gate.arg(), cut.up.arg(), y.arg()],
        )
    }

    fn swiglu_clamp_alpha<T: kernels::points::Scalar>(
        &self,
        packed: In<crate::points::Handle<T>>,
        intermediate: u32,
        limit: f32,
        alpha: f32,
        y: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "mlp.swiglu_clamp_alpha, at an element this plane does not instantiate",
        )?;
        let cut = halves(self, packed, intermediate)?;
        self.fire(
            Fire::at(
                crate::plane::module_path("gptoss_swiglu_bfloat16", self.best()),
                "gptoss_swiglu_bfloat16",
            )
            .apply(elementwise(cut.width, cut.rows)?),
            &[
                cut.gate.arg(),
                cut.up.arg(),
                y.arg(),
                limit.arg(),
                alpha.arg(),
            ],
        )
    }
}

struct Halves<T> {
    gate: crate::points::Handle<T>,

    up: crate::points::Handle<T>,

    rows: i32,
    width: i32,
}

fn halves<T: kernels::points::Scalar>(
    ctx: &Ctx<'_>,
    packed: In<crate::points::Handle<T>>,
    intermediate: u32,
) -> Result<Halves<T>, Refusal> {
    use crate::points::Staged;

    let row = packed.all("the packed `[gate | up]` row")?;
    let half = crate::points::stated("the intermediate width this statement states", intermediate)?;
    if row.width != half.saturating_mul(2) {
        return Err(Refusal::Narrow {
            what: "the packed `[gate | up]` row, against the intermediate width it states",
            at: i64::from(row.width),
        });
    }
    Ok(Halves {
        gate: ctx.window(packed.ptr, 0, half)?,
        up: ctx.window(packed.ptr, i64::from(half), half)?,
        rows: row.rows,
        width: half,
    })
}
