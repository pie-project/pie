use kernels::Grid;
use kernels::points::Scalar;
use kernels::routine::Refusal;
use kernels_macros::routine;

use crate::plane::{self, Handle};
use crate::routine::{Bind, Const, Ctx, Fire, In, InOut, Tensor, bf16};

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

#[routine(out(x = like(x)))]
pub fn neox_decode(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    base: Const<f32>,
    head_dim: Const<i32>,
    rotary: Const<i32>,
    positions: In<Tensor<i32>>,
) -> Result<(), Refusal> {
    let position = positions.ptr;
    let width = x.width;
    let lanes = rope_grid(*rotary, width, *head_dim, 1)?;
    ctx.fire(
        Fire::at("rope/neox.metal", "neox_decode_bfloat16")
            .apply(Grid::of(lanes, rope_group(lanes))),
        &[
            x.arg(),
            position.arg(),
            scale.arg(),
            base.arg(),
            head_dim.arg(),
        ],
    )
}

/// One rectangle rotated GEOMETRICALLY: `rope/neox.metal` infers both the
/// pair offset and the frequency divisor from `grid.x`, which is
/// `rotary / 2`, so the exponent divides by the ROTARY width and the pair is
/// `(i, i + rotary / 2)`.
///
/// TWO CALLS PER STATEMENT. `rope.full` states the query AND the key at one
/// position stream and this shader takes one `device T* x`, so the point
/// fires it twice; the two rectangles are disjoint and each thread owns a
/// disjoint `(i, i + half)` pair inside its own, so the order between them is
/// free.
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

/// One rectangle rotated PROPORTIONALLY: the exponent divides by the HEAD
/// width and the pair is `(i, i + head_dim / 2)`, with only the first
/// `rotary` channels moving because the grid is `rotary / 2` wide.
///
/// THIS IS `::pie::rope::rotate_partial`, ARITHMETIC FOR ARITHMETIC. Cuda's
/// partial rotations both fire that kernel, whose header records the two ways
/// a previous draft got it wrong — `rotary_dim` as the frequency denominator,
/// and `rotary_dim / 2` as the pair offset — and `rope_neox_prop_mb` computes
/// `d = 2i / head_dim` and pairs across `head_dim / 2`, which is the reading
/// that survived. So the CHOICE this plane could not make from the
/// declaration alone is one the cuda claim body already made, and it
/// transfers because the two kernels compute the same thing.
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

// INLINED into impl Rope; dies with the routine layer.
#[routine(canon = "rope.full", out(x = like(x)))]
pub fn neox_mb(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    base: Const<f32>,
    head_dim: Const<i32>,
    rotary: Const<i32>,
    positions: In<Tensor<i32>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    rotate_geometric(
        ctx, x, positions, *scale, *base, *head_dim, *rotary, *rows,
    )
}

#[routine(out(x = like(x)))]
pub fn neox_freqs_decode(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    head_dim: Const<i32>,
    mscale: Const<f32>,
    rotary: Const<i32>,
    rope_freqs: In<Tensor<f32>>,
    positions: In<Tensor<i32>>,
) -> Result<(), Refusal> {
    let inv_freq = rope_freqs.ptr;

    let position = positions.ptr;
    let width = x.width;
    let lanes = rope_grid(*rotary, width, *head_dim, 1)?;
    ctx.fire(
        Fire::at("rope/neox.metal", "neox_freqs_decode_bfloat16")
            .apply(Grid::of(lanes, rope_group(lanes))),
        &[
            x.arg(),
            position.arg(),
            scale.arg(),
            inv_freq.arg(),
            head_dim.arg(),
            mscale.arg(),
        ],
    )
}

#[routine(out(x = like(x)))]
pub fn neox_freqs_mb(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    head_dim: Const<i32>,
    mscale: Const<f32>,
    rotary: Const<i32>,
    rope_freqs: In<Tensor<f32>>,
    positions: In<Tensor<i32>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let inv_freq = rope_freqs.ptr;

    let position = positions.ptr;
    let width = x.width;
    let rows = *rows;
    let lanes = rope_grid(*rotary, width, *head_dim, rows)?;
    ctx.fire(
        Fire::at("rope/neox.metal", "neox_freqs_mb_bfloat16")
            .apply(Grid::of(lanes, rope_group(lanes))),
        &[
            x.arg(),
            position.arg(),
            scale.arg(),
            inv_freq.arg(),
            head_dim.arg(),
            mscale.arg(),
        ],
    )
}

#[routine(out(x = like(x)))]
pub fn neox_prop_decode(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    base: Const<f32>,
    head_dim: Const<i32>,
    rotary: Const<i32>,
    positions: In<Tensor<i32>>,
) -> Result<(), Refusal> {
    let position = positions.ptr;
    let width = x.width;
    let lanes = rope_grid(*rotary, width, *head_dim, 1)?;
    ctx.fire(
        Fire::at("rope/neox.metal", "neox_prop_decode_bfloat16")
            .apply(Grid::of(lanes, rope_group(lanes))),
        &[
            x.arg(),
            position.arg(),
            scale.arg(),
            base.arg(),
            head_dim.arg(),
        ],
    )
}

// INLINED into impl Rope; dies with the routine layer.
#[routine(out(x = like(x)))]
pub fn neox_prop_mb(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    base: Const<f32>,
    head_dim: Const<i32>,
    rotary: Const<i32>,
    positions: In<Tensor<i32>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    rotate_proportional(
        ctx, x, positions, *scale, *base, *head_dim, *rotary, *rows,
    )
}

#[routine(out(x = like(x)))]
pub fn neox_strided(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    base: Const<f32>,
    head_dim: Const<i32>,
    rotary: Const<i32>,
    row_pitch: Const<i32>,
    positions: In<Tensor<i32>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let position = positions.ptr;
    let width = x.width;
    let rows = *rows;
    if *row_pitch < width {
        return Err(Refusal::Narrow {
            what: "row_pitch is narrower than the row it strides over",
            at: i64::from(*row_pitch),
        });
    }
    let lanes = rope_grid(*rotary, width, *head_dim, rows)?;
    ctx.fire(
        Fire::at("rope/neox.metal", "neox_strided_bfloat16")
            .apply(Grid::of(lanes, rope_group(lanes))),
        &[
            x.arg(),
            position.arg(),
            scale.arg(),
            base.arg(),
            head_dim.arg(),
            row_pitch.arg(),
        ],
    )
}

/// `scale`: the rotation's output factor, `1 / scaling_factor`. A
/// deployment that RESCALES its ladder does it in the frequency table
/// (`neox_freqs_*`) or in the proportional arm, and the geometric one this
/// point delegates to takes the identity — which is what every text in this
/// tree passed it.
const UNSCALED: f32 = 1.0;

/// The `Rope` family, claimed — three points of five, each body the launch
/// itself.
///
/// TWO LAUNCHES FOR ONE STATEMENT, twice over. `rope/neox.metal` takes one
/// `device T* x` and rotates it in place; `rope.full` and `rope.partial` both
/// state the query AND the key at one position stream, so both fire their
/// shader twice. That is the shape `model-dsl-legacy`'s `rope_one` already
/// had and the shape `mla.latents_rope` takes on cuda.
///
/// THE BASE IS `log2(theta)` AND NOT THETA. Both bodies in this file compute
/// `inv_freq = exp2(-d * base)`, so handing them theta rotates by a frequency
/// ladder wrong from the second channel on — a defect this tree has already
/// paid for once, and the one number these claims compute.
///
/// THE GEOMETRIC/PROPORTIONAL CHOICE IS SETTLED, and it was settled on the
/// other plane. G3 left `rope.partial` and `rope.partial_q` unclaimed because
/// the kernels were both here and the declaration carried no word saying
/// which — `neox_mb` divides the exponent by the rotary width and pairs
/// `(i, i + rotary/2)`, `neox_prop_mb` divides by the HEAD width and pairs
/// `(i, i + head_dim/2)`, and those are different rotations the moment
/// `rotary < head_dim`. The word is not in the declaration and did not need
/// to be: `kernels_cuda::points::Rope::partial` and `::partial_q` both fire
/// `::pie::rope::rotate_partial`, which computes
/// `powf(theta, -2 * dim_pair / head_dim)` and pairs across `head_dim / 2`.
/// [`rotate_proportional`] is that arithmetic, so the two planes now answer
/// the same point the same way — which is the only reading under which one
/// declaration means one thing.
///
/// `rope.full` IS THE SAME ARITHMETIC READ THE OTHER WAY: the point's whole
/// content is that the rotary width IS the head width, and at
/// `rotary == head_dim` the two bodies compute the same `d = 2i / head_dim`
/// and the same pair offset. `rope/neox.metal` says so itself — "on a sliding
/// layer rotary_dims == head_dim and it reduces exactly to
/// `rope_neox_decode`" — so [`rotate_geometric`] is the one that answers it.
///
/// Two points stay on the floor's default body:
///
/// * `rope.partial_last` — SEAM: it rotates the LAST `rotary_dim` channels of
///   the head rather than the first (cuda's `rotate_partial_last` offsets by
///   `head_dim - rotary_dim`), and every `.metal` arm here starts at channel
///   zero.
/// * `rope.yarn` — SEAM: no `.metal` kernel states YaRN's ramp. `neox_freqs_*`
///   reads an inverse-frequency table the DRIVER derives at load, which is a
///   different contract from a point that states `factor`, `beta_fast`,
///   `beta_slow` and the original position span and expects the ramp computed
///   from them.
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
        // NEOX PAIRS `(i, i + half)`, WHICH IS NOT INTERLEAVING. Every
        // `rope/neox.metal` entrypoint rotates the halves of a head against
        // each other; a checkpoint that pairs `(2i, 2i + 1)` is a different
        // rotation and no kernel here performs it. The declaration carries
        // the word, so the refusal can name it.
        if interleaved {
            return Err(Refusal::Absent {
                what: "`rope.full` with interleaved pairs: every neox arm rotates halves",
            });
        }
        let head_dim = plane::stated(head_dim, "the head width this rotation states")?;
        let q = plane::in_place::<T, bf16>(q, WHAT)?;
        let k = plane::in_place::<T, bf16>(k, WHAT)?;
        let positions = plane::input::<i32, i32>(positions, "`rope.full`'s position stream")?;
        // THE WHOLE HEAD ROTATES, which is what this point says its rotary
        // width is: `rotary` and `head_dim` are one number here, and the
        // grid takes half of it per head.
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
        let rotary = plane::stated(rotary_dim, "the rotated width this statement states")?;
        let head_dim = plane::stated(head_dim, "the head width this rotation states")?;
        let q = plane::in_place::<T, bf16>(q, WHAT)?;
        let k = plane::in_place::<T, bf16>(k, WHAT)?;
        let positions = plane::input::<i32, i32>(positions, "`rope.partial`'s position stream")?;
        let base = theta.log2();
        rotate_proportional(
            self, q, positions, UNSCALED, base, head_dim, rotary, q.rows,
        )?;
        rotate_proportional(
            self, k, positions, UNSCALED, base, head_dim, rotary, k.rows,
        )
    }

    /// The query alone, which is one launch and not two — cuda's own claim
    /// hands `rotate_partial` the query pointer for both slots and a key
    /// width of zero, so its key loop runs no heads. A second launch over a
    /// rectangle with no heads in it is what that means here.
    fn partial_q<T: Scalar>(
        &self,
        q: InOut<Handle<T>>,
        positions: In<Handle<i32>>,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`rope.partial_q`, at an element this plane does not stamp";
        let rotary = plane::stated(rotary_dim, "the rotated width this statement states")?;
        let head_dim = plane::stated(head_dim, "the head width this rotation states")?;
        let q = plane::in_place::<T, bf16>(q, WHAT)?;
        let positions = plane::input::<i32, i32>(positions, "`rope.partial_q`'s position stream")?;
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
}
