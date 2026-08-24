use crate::points::{Payload, at_bf16};
use crate::routine::{Bind, Ctx, Fire, In, InOut};
use kernels::routine::Refusal;

/// The `Rope` family, claimed. Three of five points land; the two that do
/// not are a MISSING SHADER and a MISSING SLOT respectively, and the
/// difference matters.
///
/// # `rope.partial_last` — missing shader
///
/// The point rotates a head's TRAILING `rotary_dim` channels, which is MLA's
/// `[nope | pe]` head. `rope/neox.wgsl` has three angle spellings and a
/// fourth addressing shape, and not one of them starts the rotation at an
/// offset: every arm bases at `row_base + h * head_dim + i0` with `i0` from
/// zero. The point's own doc says it is "different arithmetic, not a
/// re-parameterisation", and this plane agrees — it wants a `PIE_LAST`
/// instantiation that adds `head_dim - rotary` to the base.
///
/// # `rope.yarn` — missing slot, and the slot is on the FLOOR side
///
/// `neox_freqs_mb_bfloat16` is the YaRN arm and it reads its frequencies out
/// of a `inv_freq: array<f32>` BUFFER at `@group(0) @binding(2)`, plus an
/// `mscale` word. `rope.yarn` states neither: it declares the checkpoint's
/// YaRN block flattened into six scalars — `factor`, `beta_fast`,
/// `beta_slow`, `attention_factor`, `original_max_position`, `theta` — and
/// leaves the ramp to the plane.
///
/// Those two are not the same statement. Building the table from the six
/// numbers is `head_dim/2` transcendentals of host arithmetic and a device
/// upload, which is exactly the kind of per-fire host staging a claim is
/// allowed to do on cuda (`Ctx::scratch`) and cannot do here: `Ctx` is
/// `dyn Encode`, there is no scratch allocator behind it, and a `Fire`
/// carries no bytes. `attention_factor` maps cleanly onto `mscale` and would
/// be the easy half.
///
/// **SEAM (P5):** either a `Const<Self::Tensor<f32>>` frequency bank on the
/// declaration — which is the honest reading, since the table is a function
/// of the CHECKPOINT and not of the fire — or an `Encode` that can stage a
/// small host-computed buffer. The first is the floor's call and would suit
/// every plane; cuda builds its ramp inside the kernel, so it is the one
/// plane the question does not press on.
///
/// # The fourth angle spelling has no point at all
///
/// `neox_prop_mb_bfloat16` (`PIE_PROP`) is gemma-4's proportional slice: the
/// exponent divides by the WHOLE head while only `rotary` channels turn, and
/// the rotary partner is half a HEAD away rather than half the rotary. The
/// shader's own comment says confusing it with the plain partial arm "reads
/// as a model that has merely gone slightly stupid". `rope.partial` states
/// the plain one — "NeoX pairing throughout ... branches on nothing" — so
/// this claim fires the plain arm and gemma's third convention is a POINT
/// nobody has declared. It is the `rmsnorm` / `rmsnorm_plus_one` situation
/// exactly: a fact about the checkpoint that a text states for its whole
/// life, so it wants its own point rather than a flag.
///
/// # `interleaved` is refused, not ignored
///
/// Two of the five points carry `interleaved: bool`. Every entrypoint in
/// `rope/neox.wgsl` pairs channel `i` with channel `i + dist` for a `dist`
/// of half the rotary or half the head — NEOX, in the file's name and in its
/// arithmetic. The interleaved (GPT-J) pairing is `2i` with `2i + 1` and no
/// shader here spells it, so a claim that accepted the flag and rotated NEOX
/// anyway would compute a different model in silence. Refused by name.
#[kernels_macros::claims]
impl kernels::points::Rope for Ctx<'_> {
    /// The whole head turns, on `q` and then on `k`.
    ///
    /// TWO FIRES OF ONE SHADER, and the head counts are read separately.
    /// `neox_mb_bfloat16` rotates ONE tensor in place — the routine layer
    /// took a single `x: InOut` — while the point states the pair, because a
    /// statement rotates a query and its key together. Each fire's grid
    /// divides ITS OWN operand's width by `head_dim`, so GQA falls out
    /// without being stated: `q` is `q_heads * head_dim` wide and `k` is
    /// `kv_heads * head_dim`, and neither number appears here.
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

    /// The head's LEADING `rotary_dim` channels turn; the tail passes
    /// through.
    ///
    /// The same entrypoint at a `rotary` narrower than the head. The shader
    /// reads `pairs = rotary >> 1` off the uniform (not off the grid — see
    /// the file's "the way out was already in the signature") and returns at
    /// `i0 >= pairs`, so the tail is untouched rather than rotated by zero.
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

    /// [`kernels::points::Rope::partial`] with no `k`: one fire.
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

/// The interleaved (GPT-J) pairing, refused by name. See the `Rope` block.
fn neox_pairing(interleaved: bool, what: &'static str) -> Result<(), Refusal> {
    if interleaved {
        Err(Refusal::Absent { what })
    } else {
        Ok(())
    }
}

/// A stated head or rotary width, as the shader's `i32`.
fn head_of(width: u32) -> Result<i32, Refusal> {
    i32::try_from(width).map_err(|_| Refusal::Wide {
        what: "the head width this rotation states",
        at: i64::from(width),
        max: i64::from(i32::MAX),
    })
}

/// `neox_mb_bfloat16` over one tensor, in place.
///
/// The four words after the two buffers are `Params { scale, base, head_dim,
/// rotary }` in order. `scale` is the position multiplier and is ONE for
/// every deployment without a rope scaling factor, which is every one that
/// reaches this point — a scaled deployment states `rope.yarn`, which this
/// plane does not claim. `base` is `log2(theta)` and not theta, because the
/// shader raises TWO to it (`exp2(-(i / pairs) * base)`); handing it theta
/// rotates by a frequency ladder that is wrong from the second channel on.
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
