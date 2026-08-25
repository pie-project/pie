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
        crate::points::at_bf16::<T>("rope.yarn, at an element this plane does not instantiate")?;
        if interleaved {
            return Err(NOT_NEOX);
        }
        let hd = crate::points::stated("the head width this rotation states", head_dim)?;

        // THE FREQUENCIES ARE COMPUTED, NOT FETCHED. This asked
        // `Staged::stream("rope.yarn_inv_freq")` for a per-fire plane holding a
        // precomputed inverse-frequency table -- a legacy mechanism, and one
        // this floor has no door for: `Encode::staged` refuses a runtime plane
        // asked for BY NAME, because the walk hands a body every operand its
        // point declares and nothing else. So `rope.yarn` was in the claim
        // table and could not fire for any SKU that states it, which is
        // gpt-oss, and nothing said so until a whole tower was walked here.
        //
        // `neox.slang`'s `PIE_YARN` arm derives the frequency from the same
        // six numbers `kernels-wgpu`'s `rope/yarn.wgsl` takes. `theta` crosses
        // as its LOG so the shader can use `exp2`, which is the trade every
        // other arm of that file already makes.
        let span = i32::try_from(original_max_position).ok().filter(|n| *n > 0);
        let Some(span) = span else {
            return Err(Refusal::Unstated {
                what: "the checkpoint's YaRN block: its original context length",
            });
        };
        let (low, high) = ramp_bounds(hd, theta, beta_fast, beta_slow, span);
        for x in [q, k] {
            let row = x.all("the rotated row")?;
            self.fire(
                Fire::at(
                    crate::plane::module_path("neox_yarn_mb_bfloat16", self.best()),
                    "neox_yarn_mb_bfloat16",
                )
                .apply(rope_grid(hd, row.width, hd, row.rows)?),
                &[
                    x.arg(),
                    positions.arg(),
                    1.0f32.arg(),
                    theta.log2().arg(),
                    hd.arg(),
                    factor.arg(),
                    low.arg(),
                    high.arg(),
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

/// The two dimensions a YaRN ramp runs between, from the checkpoint's block.
///
/// A FUNCTION OF THE MODEL AND NOT OF THE ROW, which is why it is here and not
/// in the shader: `beta_fast`/`beta_slow` and the original context length are
/// the same for every invocation of every fire, so deriving them per lane
/// would be one number computed a few thousand times. `kernels-wgpu`'s
/// `rope::ramp_bounds` is the same arithmetic on the other plane -- stated
/// twice on purpose, because the two are different crates and a shared one
/// would be an abstraction serving two callers and nothing else.
fn ramp_bounds(span: i32, theta: f32, beta_fast: f32, beta_slow: f32, original: i32) -> (f32, f32) {
    let ln_theta = theta.ln();
    let corr = |rot: f32| {
        span as f32 * (original as f32 / (rot * core::f32::consts::TAU)).ln() / (2.0 * ln_theta)
    };
    let low = corr(beta_fast).floor().max(0.0);
    let high = corr(beta_slow).ceil().min((span / 2) as f32 - 1.0).max(low);
    (low, high)
}
