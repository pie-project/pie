use kernels::Grid;
use kernels::plane::Refusal;
use kernels::points::Scalar;
use kernels::shader::Tensor;

use crate::plane::{Bind, Ctx, Fire, In, Out, bf16, elementwise, elementwise_rows};
use crate::points::{self, Handle};

const GROUP: u32 = 256;

struct Halves {
    packed: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    grid: Grid,
    intermediate: u32,
}

fn halves<T: Scalar>(
    packed: In<Handle<T>>,
    intermediate: u32,
    y: Out<Handle<T>>,
    what: &'static str,
) -> Result<Halves, Refusal> {
    let packed = points::input::<T, bf16>(packed, what)?;
    let y = points::result::<T, bf16>(y, what)?;
    let half = points::stated(intermediate, "the intermediate width this statement states")?;
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
    let lanes = elementwise_rows(y.width, y.rows)?;
    Ok(Halves {
        packed,
        y,
        grid: Grid::of(lanes, [lanes[0].min(GROUP), 1, 1]),
        intermediate,
    })
}

#[kernels_macros::claims]
impl kernels::points::Mlp for Ctx<'_> {
    fn swiglu<T: Scalar>(
        &self,
        packed: In<Handle<T>>,
        intermediate: u32,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        let cut = halves(
            packed,
            intermediate,
            y,
            "`mlp.swiglu`, at an element this plane does not stamp",
        )?;
        self.fire(
            Fire::at("mlp/packed.metal", "packed_swiglu_bfloat16").apply(cut.grid),
            &[cut.packed.arg(), cut.y.arg(), cut.intermediate.arg()],
        )
    }

    fn swiglu_clamp<T: Scalar>(
        &self,
        packed: In<Handle<T>>,
        intermediate: u32,
        limit: f32,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        let cut = halves(
            packed,
            intermediate,
            y,
            "`mlp.swiglu_clamp`, at an element this plane does not stamp",
        )?;
        self.fire(
            Fire::at("mlp/packed.metal", "packed_swiglu_clamp_bfloat16").apply(cut.grid),
            &[
                cut.packed.arg(),
                cut.y.arg(),
                cut.intermediate.arg(),
                limit.arg(),
            ],
        )
    }

    fn swiglu_clamp_alpha<T: Scalar>(
        &self,
        packed: In<Handle<T>>,
        intermediate: u32,
        limit: f32,
        alpha: f32,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        let cut = halves(
            packed,
            intermediate,
            y,
            "`mlp.swiglu_clamp_alpha`, at an element this plane does not stamp",
        )?;
        self.fire(
            Fire::at("mlp/packed.metal", "packed_gptoss_swiglu_bfloat16").apply(cut.grid),
            &[
                cut.packed.arg(),
                cut.y.arg(),
                cut.intermediate.arg(),
                limit.arg(),
                alpha.arg(),
            ],
        )
    }

    fn geglu_tanh<T: Scalar>(
        &self,
        gate: In<Handle<T>>,
        up: In<Handle<T>>,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`mlp.geglu_tanh`, at an element this plane does not stamp";
        let gate = points::input::<T, bf16>(gate, WHAT)?;
        self.fire(
            Fire::at("mlp/gated.metal", "geglu_tanh_bfloat16")
                .apply(Grid::of(elementwise(gate.width, gate.rows)?, [GROUP, 1, 1])),
            &[
                gate.arg(),
                points::input::<T, bf16>(up, WHAT)?.arg(),
                points::result::<T, bf16>(y, WHAT)?.arg(),
            ],
        )
    }

    fn geglu_tanh_packed<T: Scalar>(
        &self,
        packed: In<Handle<T>>,
        intermediate: u32,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        let cut = halves(
            packed,
            intermediate,
            y,
            "`mlp.geglu_tanh_packed`, at an element this plane does not stamp",
        )?;
        self.fire(
            Fire::at("mlp/packed.metal", "packed_geglu_tanh_bfloat16").apply(cut.grid),
            &[cut.packed.arg(), cut.y.arg(), cut.intermediate.arg()],
        )
    }

    fn situ<T: Scalar>(
        &self,
        packed: In<Handle<T>>,
        intermediate: u32,
        beta: f32,
        up_cap: f32,
        y: Out<Handle<T>>,
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
            "`mlp.situ`, at an element this plane does not stamp",
        )?;
        self.fire(
            Fire::at("mlp/packed.metal", "packed_situ_bfloat16").apply(cut.grid),
            &[
                cut.packed.arg(),
                cut.y.arg(),
                cut.intermediate.arg(),
                beta.arg(),
                up_cap.arg(),
            ],
        )
    }
}
