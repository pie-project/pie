use kernels::Grid;
use kernels::points::Scalar;
use kernels::routine::Refusal;

use crate::plane::{self, Handle};
use crate::routine::{Bind, Ctx, Fire, In, Out, bf16, elementwise};

/// The `Mlp` family, claimed, and it claims the ONE point of the six whose
/// operands this plane's kernels already carry.
///
/// `mlp/gated.metal` is one binding contract for three activations: `(gate,
/// up, out)` at buffers 0, 1, 2. TWO PLANES IN, and that is the whole of what
/// separates the claimed point from the five below it — `mlp.geglu_tanh` is
/// the one point of this family that states gate and up as two rows, so it is
/// the one whose statement fits the contract.
///
/// Five points stay on the floor's default body, and four of them share one
/// absence:
///
/// * `mlp.swiglu`, `mlp.swiglu_clamp`, `mlp.swiglu_clamp_alpha`,
///   `mlp.geglu_tanh_packed` — SEAM: THE PACKED ROW HAS NO KERNEL HERE. Each
///   states one `[gate | up]` rectangle and an `intermediate` width, and
///   every arm `mlp/gated.metal` stamps — `silu_mul`, `gptoss_swiglu`,
///   `geglu_tanh` — reads two separate bases. A packed row cannot be handed over as two: an operand on this
///   plane is a BINDING HANDLE with no offset (`plane::Handle` carries a
///   `u32` and nothing else), so the `up` half's base — the same buffer,
///   `intermediate` elements in — is not something a body can construct.
///   `geglu_tanh_strided` takes three pitches and still cannot: it reads
///   `up[m * up_pitch + k]`, which reaches `m * 2I + k` and never
///   `m * 2I + I + k`. Cuda closed the same gap by writing `chunked_swiglu`,
///   `chunked_geglu_tanh`, `chunked_situ` and `chunked_gpt_oss_glu` beside
///   the two-plane forms; this plane wants the same four, and a delegation
///   that cut the row with a pointer would be inventing an operand.
/// * `mlp.situ` — SEAM: kimi's `beta`-scaled silu with a capped linear term,
///   and no `.metal` entrypoint computes it in either form, packed or split.
#[kernels_macros::claims]
impl kernels::points::Mlp for Ctx<'_> {
    fn geglu_tanh<T: Scalar>(
        &self,
        gate: In<Handle<T>>,
        up: In<Handle<T>>,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`mlp.geglu_tanh`, at an element this plane does not stamp";
        let gate = plane::input::<T, bf16>(gate, WHAT)?;
        self.fire(
            Fire::at("mlp/gated.metal", "geglu_tanh_bfloat16")
                .apply(Grid::of(elementwise(gate.width, gate.rows)?, [256, 1, 1])),
            &[
                gate.arg(),
                plane::input::<T, bf16>(up, WHAT)?.arg(),
                plane::result::<T, bf16>(y, WHAT)?.arg(),
            ],
        )
    }
}
