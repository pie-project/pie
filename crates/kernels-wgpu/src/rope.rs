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

    fn partial_last<T: kernels::points::Scalar>(
        &self,
        q: InOut<Payload<T>>,
        positions: In<Payload<i32>>,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
        interleaved: bool,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("rope.partial_last at an element other than bf16")?;
        let rotary = head_of(rotary_dim)?;
        let head_dim = head_of(head_dim)?;
        if rotary > head_dim {
            return Err(Refusal::Wide {
                what: "the rotated tail is wider than the head it sits at the end of",
                at: i64::from(rotary),
                max: i64::from(head_dim),
            });
        }
        if (head_dim - rotary) % 2 != 0 {
            return Err(Refusal::Narrow {
                what: "the untouched lead is not a whole number of bf16 pairs",
                at: i64::from(head_dim - rotary),
            });
        }
        let entrypoint = if interleaved {
            "gptj_last_mb_bfloat16"
        } else {
            if rotary % 4 != 0 {
                return Err(Refusal::Narrow {
                    what: "a NeoX tail whose partner distance is not a whole number of bf16 pairs",
                    at: i64::from(rotary),
                });
            }
            "neox_last_mb_bfloat16"
        };
        self.fire(
            Fire::at("rope/partial_last.wgsl", entrypoint)
                .apply(pair_lanes(rotary, q.width, head_dim, q.rows)?),
            &[
                q.arg(),
                positions.arg(),
                theta.log2().arg(),
                head_dim.arg(),
                rotary.arg(),
            ],
        )
    }

    fn yarn<T: kernels::points::Scalar>(
        &self,
        q: InOut<Payload<T>>,
        k: InOut<Payload<T>>,
        positions: In<Payload<i32>>,
        head_dim: u32,
        theta: f32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        attention_factor: f32,
        original_max_position: u32,
        interleaved: bool,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("rope.yarn at an element other than bf16")?;
        let head_dim = head_of(head_dim)?;
        if head_dim % 4 != 0 && !interleaved {
            return Err(Refusal::Narrow {
                what: "a NeoX head whose partner distance is not a whole number of bf16 pairs",
                at: i64::from(head_dim),
            });
        }
        let span = i32::try_from(original_max_position).ok().filter(|n| *n > 0);
        let Some(span) = span else {
            return Err(Refusal::Unstated {
                what: "the checkpoint's YaRN block: its original context length",
            });
        };
        let (low, high) = ramp_bounds(head_dim, theta, beta_fast, beta_slow, span);
        let entrypoint = if interleaved {
            "gptj_yarn_mb_bfloat16"
        } else {
            "neox_yarn_mb_bfloat16"
        };
        for x in [q, k] {
            self.fire(
                Fire::at("rope/yarn.wgsl", entrypoint)
                    .apply(pair_lanes(head_dim, x.width, head_dim, x.rows)?),
                &[
                    x.arg(),
                    positions.arg(),
                    theta.log2().arg(),
                    head_dim.arg(),
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

#[allow(clippy::cast_precision_loss)]
fn ramp_bounds(span: i32, theta: f32, beta_fast: f32, beta_slow: f32, original: i32) -> (f32, f32) {
    let ln_theta = theta.ln();
    let corr = |rot: f32| {
        span as f32 * (original as f32 / (rot * core::f32::consts::TAU)).ln() / (2.0 * ln_theta)
    };
    let low = corr(beta_fast).floor().max(0.0);
    let high = corr(beta_slow).ceil().min((span / 2) as f32 - 1.0).max(low);
    (low, high)
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
    pair_lanes(rotary, width, head_dim, rows)
}

fn pair_lanes(rotary: i32, width: i32, head_dim: i32, rows: i32) -> Result<[u32; 3], Refusal> {
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
        pairs.div_ceil(2),
        width.unsigned_abs() / head_dim.unsigned_abs(),
        rows.unsigned_abs(),
    ])
}
