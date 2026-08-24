use crate::plane::{Bind, Ctx, Fire, In, InOut};
use kernels::plane::Refusal;

pub fn rope_grid(rotary: i32, width: i32, head_dim: i32, rows: i32) -> Result<[u32; 3], Refusal> {
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

#[kernels_macros::claims]
impl kernels::points::Rope for Ctx<'_> {
    fn full<T: kernels::points::Scalar>(
        &self,
        q: InOut<crate::points::Handle<T>>,
        k: InOut<crate::points::Handle<T>>,
        positions: In<crate::points::Handle<i32>>,
        head_dim: u32,
        theta: f32,
        interleaved: bool,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>("rope.full, at an element this plane does not instantiate")?;
        let hd = crate::points::stated("the head width this rotation states", head_dim)?;
        neox(self, q, positions, hd, hd, theta, interleaved)?;
        neox(self, k, positions, hd, hd, theta, interleaved)
    }

    fn partial<T: kernels::points::Scalar>(
        &self,
        q: InOut<crate::points::Handle<T>>,
        k: InOut<crate::points::Handle<T>>,
        positions: In<crate::points::Handle<i32>>,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>("rope.partial, at an element this plane does not instantiate")?;
        let hd = crate::points::stated("the head width this rotation states", head_dim)?;
        let rot = crate::points::stated("the rotary width this rotation states", rotary_dim)?;

        neox(self, q, positions, hd, rot, theta, false)?;
        neox(self, k, positions, hd, rot, theta, false)
    }

    fn partial_q<T: kernels::points::Scalar>(
        &self,
        q: InOut<crate::points::Handle<T>>,
        positions: In<crate::points::Handle<i32>>,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "rope.partial_q, at an element this plane does not instantiate",
        )?;
        let hd = crate::points::stated("the head width this rotation states", head_dim)?;
        let rot = crate::points::stated("the rotary width this rotation states", rotary_dim)?;
        neox(self, q, positions, hd, rot, theta, false)
    }

    fn yarn<T: kernels::points::Scalar>(
        &self,
        q: InOut<crate::points::Handle<T>>,
        k: InOut<crate::points::Handle<T>>,
        positions: In<crate::points::Handle<i32>>,
        head_dim: u32,
        theta: f32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        attention_factor: f32,
        original_max_position: u32,
        interleaved: bool,
    ) -> Result<(), Refusal> {
        use crate::points::Staged;

        crate::points::at_bf16::<T>("rope.yarn, at an element this plane does not instantiate")?;
        if interleaved {
            return Err(NOT_NEOX);
        }
        let hd = crate::points::stated("the head width this rotation states", head_dim)?;

        let _ = (theta, factor, beta_fast, beta_slow, original_max_position);

        let inv_freq = self.stream::<f32>("rope.yarn_inv_freq")?;
        for x in [q, k] {
            let row = x.all("the rotated row")?;
            self.fire(
                Fire::at(
                    crate::plane::module_path("neox_freqs_mb_bfloat16", self.best()),
                    "neox_freqs_mb_bfloat16",
                )
                .apply(rope_grid(hd, row.width, hd, row.rows)?),
                &[
                    x.arg(),
                    positions.arg(),
                    1.0f32.arg(),
                    inv_freq.arg(),
                    hd.arg(),
                    attention_factor.arg(),
                ],
            )?;
        }
        Ok(())
    }
}

const NOT_NEOX: Refusal = Refusal::Absent {
    what: "a rotation under GPT-J pairing: `neox.slang` pairs `i` with \
           `i + rotary/2` and this plane stamps no interleaved instantiation",
};

fn neox<T: kernels::points::Scalar>(
    ctx: &Ctx<'_>,
    x: InOut<crate::points::Handle<T>>,
    positions: In<crate::points::Handle<i32>>,
    head_dim: i32,
    rotary: i32,
    theta: f32,
    interleaved: bool,
) -> Result<(), Refusal> {
    if interleaved {
        return Err(NOT_NEOX);
    }
    let row = x.all("the rotated row")?;
    ctx.fire(
        Fire::at(
            crate::plane::module_path("neox_mb_bfloat16", ctx.best()),
            "neox_mb_bfloat16",
        )
        .apply(rope_grid(rotary, row.width, head_dim, row.rows)?),
        &[
            x.arg(),
            positions.arg(),
            1.0f32.arg(),
            theta.log2().arg(),
            head_dim.arg(),
        ],
    )
}
