use kernels::Grid;
use kernels::plane::Refusal;
use kernels::points::Scalar;

use crate::plane::{Bind, Ctx, Fire, In, InOut, Tensor, bf16};
use crate::points::{self, Handle};

fn rope_grid(rotary: i32, width: i32, head_dim: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if rotary <= 0 {
        return Err(Refusal::Empty { what: "rotary" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if head_dim <= 0 {
        return Err(Refusal::Empty { what: "head_dim" });
    }
    if rotary % 2 != 0 {
        return Err(Refusal::Narrow {
            what: "rotary is not a whole number of pairs",
            at: i64::from(rotary),
        });
    }
    if width <= 0 || width % head_dim != 0 {
        return Err(Refusal::Narrow {
            what: "width is not a whole number of heads",
            at: i64::from(width),
        });
    }
    Ok([
        rotary.unsigned_abs() / 2,
        width.unsigned_abs() / head_dim.unsigned_abs(),
        rows.unsigned_abs(),
    ])
}

const fn rope_group(lanes: [u32; 3]) -> [u32; 3] {
    [lanes[0], 1, 1]
}

#[allow(clippy::too_many_arguments)]
fn rotate_geometric(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    positions: In<Tensor<i32>>,
    scale: f32,
    base: f32,
    head_dim: i32,
    rotary: i32,
    rows: i32,
) -> Result<(), Refusal> {
    let lanes = rope_grid(rotary, x.width, head_dim, rows)?;
    ctx.fire(
        Fire::at("rope/neox.metal", "neox_mb_bfloat16").apply(Grid::of(lanes, rope_group(lanes))),
        &[
            x.arg(),
            positions.ptr.arg(),
            scale.arg(),
            base.arg(),
            head_dim.arg(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn rotate_proportional(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    positions: In<Tensor<i32>>,
    scale: f32,
    base: f32,
    head_dim: i32,
    rotary: i32,
    rows: i32,
) -> Result<(), Refusal> {
    let lanes = rope_grid(rotary, x.width, head_dim, rows)?;
    ctx.fire(
        Fire::at("rope/neox.metal", "neox_prop_mb_bfloat16")
            .apply(Grid::of(lanes, rope_group(lanes))),
        &[
            x.arg(),
            positions.ptr.arg(),
            scale.arg(),
            base.arg(),
            head_dim.arg(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn rotate_tail(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    positions: In<Tensor<i32>>,
    base: f32,
    head_dim: i32,
    rotary: i32,
    interleaved: bool,
    rows: i32,
) -> Result<(), Refusal> {
    let lanes = rope_grid(rotary, x.width, head_dim, rows)?;
    ctx.fire(
        Fire::at("rope/neox.metal", "neox_last_mb_bfloat16")
            .apply(Grid::of(lanes, rope_group(lanes))),
        &[
            x.arg(),
            positions.ptr.arg(),
            base.arg(),
            head_dim.arg(),
            i32::from(interleaved).arg(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn rotate_ramped(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    positions: In<Tensor<i32>>,
    base: f32,
    head_dim: i32,
    ramp: Ramp,
    interleaved: bool,
    rows: i32,
) -> Result<(), Refusal> {
    let lanes = rope_grid(head_dim, x.width, head_dim, rows)?;
    ctx.fire(
        Fire::at("rope/neox.metal", "neox_yarn_mb_bfloat16")
            .apply(Grid::of(lanes, rope_group(lanes))),
        &[
            x.arg(),
            positions.ptr.arg(),
            base.arg(),
            head_dim.arg(),
            ramp.factor.arg(),
            ramp.low_dim.arg(),
            ramp.high_dim.arg(),
            ramp.mscale.arg(),
            i32::from(interleaved).arg(),
        ],
    )
}

#[derive(Clone, Copy)]
struct Ramp {
    factor: f32,
    low_dim: f32,
    high_dim: f32,
    mscale: f32,
}

#[allow(clippy::cast_precision_loss)]
fn ramp_bounds(
    span: i32,
    theta: f32,
    beta_fast: f32,
    beta_slow: f32,
    original_max_position: i32,
) -> (f32, f32) {
    let ln_theta = theta.ln();
    let corr_dim = |rot: f32| -> f32 {
        span as f32 * (original_max_position as f32 / (rot * core::f32::consts::TAU)).ln()
            / (2.0 * ln_theta)
    };
    let low_dim = corr_dim(beta_fast).floor().max(0.0);
    let high_dim = corr_dim(beta_slow)
        .ceil()
        .min((span / 2) as f32 - 1.0)
        .max(low_dim);
    (low_dim, high_dim)
}

const UNSCALED: f32 = 1.0;

#[kernels_macros::claims]
impl kernels::points::Rope for Ctx<'_> {
    fn full<T: Scalar>(
        &self,
        q: InOut<Handle<T>>,
        k: InOut<Handle<T>>,
        positions: In<Handle<i32>>,
        head_dim: u32,
        theta: f32,
        interleaved: bool,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`rope.full`, at an element this plane does not stamp";

        if interleaved {
            return Err(Refusal::Absent {
                what: "`rope.full` with interleaved pairs: every neox arm rotates halves",
            });
        }
        let head_dim = points::stated(head_dim, "the head width this rotation states")?;
        let q = points::in_place::<T, bf16>(q, WHAT)?;
        let k = points::in_place::<T, bf16>(k, WHAT)?;
        let positions = points::input::<i32, i32>(positions, "`rope.full`'s position stream")?;

        let base = theta.log2();
        rotate_geometric(
            self, q, positions, UNSCALED, base, head_dim, head_dim, q.rows,
        )?;
        rotate_geometric(
            self, k, positions, UNSCALED, base, head_dim, head_dim, k.rows,
        )
    }

    fn partial<T: Scalar>(
        &self,
        q: InOut<Handle<T>>,
        k: InOut<Handle<T>>,
        positions: In<Handle<i32>>,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`rope.partial`, at an element this plane does not stamp";
        let rotary = points::stated(rotary_dim, "the rotated width this statement states")?;
        let head_dim = points::stated(head_dim, "the head width this rotation states")?;
        let q = points::in_place::<T, bf16>(q, WHAT)?;
        let k = points::in_place::<T, bf16>(k, WHAT)?;
        let positions = points::input::<i32, i32>(positions, "`rope.partial`'s position stream")?;
        let base = theta.log2();
        rotate_proportional(self, q, positions, UNSCALED, base, head_dim, rotary, q.rows)?;
        rotate_proportional(self, k, positions, UNSCALED, base, head_dim, rotary, k.rows)
    }

    fn partial_q<T: Scalar>(
        &self,
        q: InOut<Handle<T>>,
        positions: In<Handle<i32>>,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`rope.partial_q`, at an element this plane does not stamp";
        let rotary = points::stated(rotary_dim, "the rotated width this statement states")?;
        let head_dim = points::stated(head_dim, "the head width this rotation states")?;
        let q = points::in_place::<T, bf16>(q, WHAT)?;
        let positions = points::input::<i32, i32>(positions, "`rope.partial_q`'s position stream")?;
        rotate_proportional(
            self,
            q,
            positions,
            UNSCALED,
            theta.log2(),
            head_dim,
            rotary,
            q.rows,
        )
    }

    fn partial_last<T: Scalar>(
        &self,
        q: InOut<Handle<T>>,
        positions: In<Handle<i32>>,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
        interleaved: bool,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`rope.partial_last`, at an element this plane does not stamp";
        let rotary = points::stated(rotary_dim, "the rotated width this statement states")?;
        let head_dim = points::stated(head_dim, "the head width this rotation states")?;
        if rotary > head_dim {
            return Err(Refusal::Wide {
                what: "the rotated tail is wider than the head it sits at the end of",
                at: i64::from(rotary),
                max: i64::from(head_dim),
            });
        }
        let q = points::in_place::<T, bf16>(q, WHAT)?;
        let positions =
            points::input::<i32, i32>(positions, "`rope.partial_last`'s position stream")?;
        rotate_tail(
            self,
            q,
            positions,
            theta.log2(),
            head_dim,
            rotary,
            interleaved,
            q.rows,
        )
    }

    fn yarn<T: Scalar>(
        &self,
        q: InOut<Handle<T>>,
        k: InOut<Handle<T>>,
        positions: In<Handle<i32>>,
        head_dim: u32,
        theta: f32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        attention_factor: f32,
        original_max_position: u32,
        interleaved: bool,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`rope.yarn`, at an element this plane does not stamp";
        let head_dim = points::stated(head_dim, "the head width this rotation states")?;
        let span = points::stated(
            original_max_position,
            "the position span this checkpoint's YaRN block states",
        )?;
        if span <= 0 {
            return Err(Refusal::Unstated {
                what: "the checkpoint's YaRN block",
            });
        }
        let q = points::in_place::<T, bf16>(q, WHAT)?;
        let k = points::in_place::<T, bf16>(k, WHAT)?;
        let positions = points::input::<i32, i32>(positions, "`rope.yarn`'s position stream")?;

        let (low_dim, high_dim) = ramp_bounds(head_dim, theta, beta_fast, beta_slow, span);
        let ramp = Ramp {
            factor,
            low_dim,
            high_dim,
            mscale: attention_factor,
        };
        let base = theta.log2();
        rotate_ramped(
            self,
            q,
            positions,
            base,
            head_dim,
            ramp,
            interleaved,
            q.rows,
        )?;
        rotate_ramped(
            self,
            k,
            positions,
            base,
            head_dim,
            ramp,
            interleaved,
            k.rows,
        )
    }
}
