use crate::plane::{Bind, Ctx, Fire, In, Out, elementwise};
use crate::points::{Payload, at_bf16};
use kernels::plane::Refusal;

struct Halves {
    words: [u32; 3],
    intermediate: u32,
}

fn halves<T: kernels::points::Scalar>(
    packed: In<Payload<T>>,
    intermediate: u32,
    y: Out<Payload<T>>,
    what: &'static str,
) -> Result<Halves, Refusal> {
    at_bf16::<T>(what)?;
    let half = i32::try_from(intermediate).map_err(|_| Refusal::Wide {
        what: "the intermediate width this statement states",
        at: i64::from(intermediate),
        max: i64::from(i32::MAX),
    })?;
    if half <= 0 {
        return Err(Refusal::Empty {
            what: "the intermediate width this statement states",
        });
    }
    if packed.width != half.saturating_mul(2) {
        return Err(Refusal::Narrow {
            what: "the packed `[gate | up]` row, against the intermediate width it states",
            at: i64::from(packed.width),
        });
    }
    if y.width != half {
        return Err(Refusal::Narrow {
            what: "the activation's row, against the intermediate width it states",
            at: i64::from(y.width),
        });
    }
    if y.rows != packed.rows {
        return Err(Refusal::Narrow {
            what: "the activation's rows, against the packed row it halves",
            at: i64::from(y.rows),
        });
    }
    let elements = elementwise(y.width, y.rows)?;
    Ok(Halves {
        words: [elements[0].div_ceil(2), 1, 1],
        intermediate,
    })
}

#[kernels_macros::claims]
impl kernels::points::Mlp for Ctx<'_> {
    fn swiglu<T: kernels::points::Scalar>(
        &self,
        packed: In<Payload<T>>,
        intermediate: u32,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        let cut = halves(
            packed,
            intermediate,
            y,
            "mlp.swiglu at an element other than bf16",
        )?;
        self.fire(
            Fire::at("mlp/packed.wgsl", "packed_swiglu_bfloat16").apply(cut.words),
            &[packed.arg(), y.arg(), cut.intermediate.arg()],
        )
    }

    fn swiglu_clamp<T: kernels::points::Scalar>(
        &self,
        packed: In<Payload<T>>,
        intermediate: u32,
        limit: f32,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        let cut = halves(
            packed,
            intermediate,
            y,
            "mlp.swiglu_clamp at an element other than bf16",
        )?;
        self.fire(
            Fire::at("mlp/packed.wgsl", "packed_swiglu_clamp_bfloat16").apply(cut.words),
            &[packed.arg(), y.arg(), cut.intermediate.arg(), limit.arg()],
        )
    }

    fn swiglu_clamp_alpha<T: kernels::points::Scalar>(
        &self,
        packed: In<Payload<T>>,
        intermediate: u32,
        limit: f32,
        alpha: f32,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        let cut = halves(
            packed,
            intermediate,
            y,
            "mlp.swiglu_clamp_alpha at an element other than bf16",
        )?;
        self.fire(
            Fire::at("mlp/packed.wgsl", "packed_gptoss_swiglu_bfloat16").apply(cut.words),
            &[
                packed.arg(),
                y.arg(),
                cut.intermediate.arg(),
                limit.arg(),
                alpha.arg(),
            ],
        )
    }

    fn geglu_tanh<T: kernels::points::Scalar>(
        &self,
        gate: In<Payload<T>>,
        up: In<Payload<T>>,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("mlp.geglu_tanh at an element other than bf16")?;
        self.fire(
            Fire::at("mlp/gated.wgsl", "geglu_tanh_bfloat16")
                .apply(elementwise(gate.width, gate.rows)?),
            &[gate.arg(), up.arg(), y.arg()],
        )
    }

    fn geglu_tanh_packed<T: kernels::points::Scalar>(
        &self,
        packed: In<Payload<T>>,
        intermediate: u32,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        let cut = halves(
            packed,
            intermediate,
            y,
            "mlp.geglu_tanh_packed at an element other than bf16",
        )?;
        self.fire(
            Fire::at("mlp/packed.wgsl", "packed_geglu_tanh_bfloat16").apply(cut.words),
            &[packed.arg(), y.arg(), cut.intermediate.arg()],
        )
    }

    fn situ<T: kernels::points::Scalar>(
        &self,
        packed: In<Payload<T>>,
        intermediate: u32,
        beta: f32,
        up_cap: f32,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        if beta == 0.0 {
            return Err(Refusal::Empty {
                what: "`mlp.situ`'s beta, which the gate divides by",
            });
        }
        let cut = halves(
            packed,
            intermediate,
            y,
            "mlp.situ at an element other than bf16",
        )?;
        self.fire(
            Fire::at("mlp/packed.wgsl", "packed_situ_bfloat16").apply(cut.words),
            &[
                packed.arg(),
                y.arg(),
                cut.intermediate.arg(),
                beta.arg(),
                up_cap.arg(),
            ],
        )
    }
}
