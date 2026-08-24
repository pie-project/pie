use crate::plane::{Bind, Ctx, Fire, In, InOut};
use crate::points::{Payload, at_bf16};
use kernels::plane::Refusal;

#[kernels_macros::claims]
impl kernels::points::Rope for Ctx<'_> {
    fn full<T: kernels::points::Scalar>(
        &self,
        q: InOut<Payload<T>>,
        k: InOut<Payload<T>>,
        positions: In<Payload<i32>>,
        head_dim: u32,
        theta: f32,
        interleaved: bool,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("rope.full at an element other than bf16")?;
        neox_pairing(interleaved, "rope.full")?;
        let head_dim = head_of(head_dim)?;
        neox_mb_at(self, q, positions, head_dim, head_dim, theta)?;
        neox_mb_at(self, k, positions, head_dim, head_dim, theta)
    }

    fn partial<T: kernels::points::Scalar>(
        &self,
        q: InOut<Payload<T>>,
        k: InOut<Payload<T>>,
        positions: In<Payload<i32>>,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("rope.partial at an element other than bf16")?;
        let rotary = head_of(rotary_dim)?;
        let head_dim = head_of(head_dim)?;
        neox_mb_at(self, q, positions, rotary, head_dim, theta)?;
        neox_mb_at(self, k, positions, rotary, head_dim, theta)
    }

    fn partial_q<T: kernels::points::Scalar>(
        &self,
        q: InOut<Payload<T>>,
        positions: In<Payload<i32>>,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("rope.partial_q at an element other than bf16")?;
        neox_mb_at(
            self,
            q,
            positions,
            head_of(rotary_dim)?,
            head_of(head_dim)?,
            theta,
        )
    }
}

fn neox_pairing(interleaved: bool, what: &'static str) -> Result<(), Refusal> {
    if interleaved {
        Err(Refusal::Absent { what })
    } else {
        Ok(())
    }
}

fn head_of(width: u32) -> Result<i32, Refusal> {
    i32::try_from(width).map_err(|_| Refusal::Wide {
        what: "the head width this rotation states",
        at: i64::from(width),
        max: i64::from(i32::MAX),
    })
}

fn neox_mb_at<T: kernels::points::Scalar>(
    ctx: &Ctx<'_>,
    x: InOut<Payload<T>>,
    positions: In<Payload<i32>>,
    rotary: i32,
    head_dim: i32,
    theta: f32,
) -> Result<(), Refusal> {
    ctx.fire(
        Fire::at("rope/neox.wgsl", "neox_mb_bfloat16")
            .apply(rope_grid(rotary, x.width, head_dim, x.rows)?),
        &[
            x.arg(),
            positions.arg(),
            1.0f32.arg(),
            theta.log2().arg(),
            head_dim.arg(),
            rotary.arg(),
        ],
    )
}

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

    let pairs = rotary.unsigned_abs() / 2;
    Ok([
        pairs.div_ceil(2).div_ceil(32),
        width.unsigned_abs() / head_dim.unsigned_abs(),
        rows.unsigned_abs(),
    ])
}
