use crate::routine::{Bind, Ctx, Fire, In, InOut};
use kernels::routine::Refusal;

/// The rotation's grid: one lane per channel PAIR, per head, per row.
///
/// `neox.slang` pairs channel `i` with `i + rotary/2`, so the x extent is
/// half the rotary width and not the whole of it — the one place on this
/// plane where a grid axis is not the extent it walks. The head count is
/// the row over the head width, which is the derivation every per-head
/// point here makes.
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

/// The `Rope` family, claimed. Four of five points land; the fifth is a
/// measured backlog row and the reason is a pairing, not a parameter.
///
/// # `theta` is stated, `base` is what the shader wants
///
/// `neox.slang` computes `theta = scale * position * exp2(-d * pc.base)`,
/// so the push word is `log2(theta)` and not `theta`. The legacy lowering
/// says so in as many words beside its own dispatch
/// (`model-dsl-legacy/src/metal/mod.rs`: "the base is `log2(theta)` because
/// the shader raises two to it"), and it is the one number a body here has
/// to TRANSFORM rather than pass. `scale` is the position scaling and is
/// `1.0` at every declared point — the declarations carry no rope-scaling
/// slot, and YaRN's factor is folded into the frequency table instead.
///
/// # Two rectangles, two fires
///
/// Every point but `partial_q` rotates `q` AND `k`, and this plane's
/// entrypoint rotates ONE binding: `neox_mb_bfloat16` takes a single `x`
/// and grids by that rectangle's own head count. The legacy lowering fused
/// the pair into one dispatch when the two happened to be contiguous
/// halves of one projection; a point carries two rectangles the arena
/// minted separately, so the honest form is two launches. They are ordered
/// on one queue, so the pair is atomic to everything downstream.
///
/// # `interleaved` is refused, never ignored
///
/// `neox.slang` pairs channel `i` with `i + rotary/2` — NeoX. GPT-J's
/// pairing is `2d` with `2d + 1`, a different index arithmetic and a
/// different kernel, and this plane has none. `kernels::points` states
/// exactly why the flag cannot be defaulted away: "the texts disagree —
/// gpt-oss's YaRN rotation is NeoX, deepseek-v4's trailing rotation is
/// not. A point that fixed it would be right for one checkpoint and
/// silently wrong for the other." So the bodies refuse it by name.
///
/// # One point stays on the floor's default body
///
/// * `rope.partial_last` — the head's TRAILING `rotary_dim` channels
///   rotate. `neox.slang` addresses `i1 = row_base + h * head_dim + i` with
///   `i` counted from zero, so the leading slice is the only slice it can
///   reach; a trailing rotation needs `+ (head_dim - rotary_dim)` in the
///   shader. That is a `PIE_TRAILING` instantiation — shader work, not
///   plumbing — and until it exists the point is a measured row.
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
        // NeoX throughout, which is what the declaration says: "the kernel
        // that rotates a leading slice branches on nothing, so nothing is
        // stated". `false` here is that sentence, not a default.
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

    /// The whole head, on YaRN-interpolated frequencies.
    ///
    /// SEAM, AND IT IS A PLANE, NOT A NUMBER. `neox_freqs_mb_bfloat16`
    /// reads `inv_freq[i]` from a THIRD binding and applies `mscale` on the
    /// way out; the declaration states the YaRN block flattened — `factor`,
    /// `beta_fast`, `beta_slow`, `attention_factor`,
    /// `original_max_position` — and carries no slot for a frequency
    /// table, because a table is not a statement operand. Somebody has to
    /// build one `head_dim / 2` wide out of those five numbers, and on this
    /// plane nobody does: there is no shader that computes it and no
    /// load-time hook that uploads it. `attention_factor` IS `mscale`, so
    /// that half is already answered.
    ///
    /// What P5 needs: either the table as a load-time-derived weight (which
    /// is what the legacy lowering did on the host), or a shader that
    /// writes it into scratch from the five numbers. The body below is
    /// whole against the first reading — it asks for the plane by name and
    /// fires — so the door is [`crate::points::Staged::stream`].
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
        // The five YaRN numbers are the TABLE's, not the launch's: they say
        // what `inv_freq` holds and nothing the shader reads. `theta` is
        // among them for the same reason — the geometric base is the
        // interpolation's input, not a push word here.
        let _ = (theta, factor, beta_fast, beta_slow, original_max_position);
        // SEAM: the interpolated frequency plane. See this method's doc.
        let inv_freq = self.stream::<f32>("rope.yarn_inv_freq")?;
        for x in [q, k] {
            let row = x.all("the rotated row")?;
            self.fire(
                Fire::at(
                    crate::routine::module_path("neox_freqs_mb_bfloat16", self.best()),
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

/// What every point of this family answers when a text asks for GPT-J
/// pairing.
///
/// `neox.slang` pairs channel `i` with `i + rotary/2` and stamps no
/// interleaved instantiation, so the refusal is the plane's and not the
/// point's — one constant rather than a tag threaded through the launcher.
/// Which STATEMENT asked is named one call out, by the dispatch arm.
const NOT_NEOX: Refusal = Refusal::Absent {
    what: "a rotation under GPT-J pairing: `neox.slang` pairs `i` with \
           `i + rotary/2` and this plane stamps no interleaved instantiation",
};

/// One `neox_mb_bfloat16` launch over one rectangle: the arithmetic every
/// claimed point of this family is made of.
///
/// A free function and not a method, because it is the whole of what the
/// four claimed points share — the branch on `interleaved`, the `log2` the
/// push block wants, and the grid the shader addresses.
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
            crate::routine::module_path("neox_mb_bfloat16", ctx.best()),
            "neox_mb_bfloat16",
        )
        .apply(rope_grid(rotary, row.width, head_dim, row.rows)?),
        &[
            x.arg(),
            positions.arg(),
            // The position scaling. `1.0` because no declared point of this
            // family carries one — a scaled rope is `yarn`, and there the
            // factor lives in the frequency table.
            1.0f32.arg(),
            // `exp2(-d * base)`, so the word is the LOG of the base the
            // text states. The one transformed number in this family.
            theta.log2().arg(),
            head_dim.arg(),
        ],
    )
}
