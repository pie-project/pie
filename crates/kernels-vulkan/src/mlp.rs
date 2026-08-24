use crate::routine::{
    Asks, Bind, Const, Ctx, Fire, In, Out, Tensor, bf16, elementwise, elementwise_rows,
};
use kernels::routine::Refusal;
use kernels_macros::routine;

// INLINED into impl Mlp; dies with the routine layer. (mlp.geglu_tanh)
#[routine(out(out = like(gate)))]
pub fn geglu_tanh(
    ctx: &Ctx<'_>,
    gate: In<Tensor<bf16>>,
    up: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let width = gate.width;
    let rows = *rows;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("geglu_tanh_bfloat16", ctx.best()),
            "geglu_tanh_bfloat16",
        )
        .apply(elementwise(width, rows)?),
        &[gate.arg(), up.arg(), out.arg()],
    )
}

#[routine(out(out = rows(gate) x const(stated_width)))]
pub fn geglu_tanh_strided(
    ctx: &Ctx<'_>,
    gate: In<Tensor<bf16>>,
    up: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    stated_width: Const<u32>,
    stated_rows: Const<u32>,
    gate_pitch: Const<u32>,
    up_pitch: Const<u32>,
    out_pitch: Const<u32>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let width = gate.width;
    let rows = *rows;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("geglu_tanh_strided_bfloat16", ctx.best()),
            "geglu_tanh_strided_bfloat16",
        )
        .apply(elementwise(width, rows)?),
        &[
            gate.arg(),
            up.arg(),
            out.arg(),
            stated_width.arg(),
            stated_rows.arg(),
            gate_pitch.arg(),
            up_pitch.arg(),
            out_pitch.arg(),
        ],
    )
}

// INLINED into impl Mlp; dies with the routine layer. (mlp.swiglu_clamp_alpha)
#[routine(out(out = like(gate)))]
pub fn gptoss_swiglu(
    ctx: &Ctx<'_>,
    gate: In<Tensor<bf16>>,
    up: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    _stated_elements: Const<u32>,
    limit: Const<f32>,
    alpha: Const<f32>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let width = gate.width;
    let rows = *rows;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("gptoss_swiglu_bfloat16", ctx.best()),
            "gptoss_swiglu_bfloat16",
        )
        .apply(elementwise(width, rows)?),
        &[gate.arg(), up.arg(), out.arg(), limit.arg(), alpha.arg()],
    )
}

// INLINED into impl Mlp; dies with the routine layer. (mlp.swiglu)
#[routine(out(out = like(gate)))]
pub fn silu_mul(
    ctx: &Ctx<'_>,
    gate: In<Tensor<bf16>>,
    up: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let width = gate.width;
    let rows = *rows;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("silu_mul_bfloat16", ctx.best()),
            "silu_mul_bfloat16",
        )
        .apply(elementwise(width, rows)?),
        &[gate.arg(), up.arg(), out.arg()],
    )
}

#[routine]
pub fn silu_mul_strided(
    ctx: &Ctx<'_>,
    gate: In<Tensor<bf16>>,
    up: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let row_pitch = ctx.param(1)?;
    let width = gate.width;
    let rows = *rows;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("silu_mul_strided_bfloat16", ctx.best()),
            "silu_mul_strided_bfloat16",
        )
        .apply(elementwise_rows(width, rows)?),
        &[gate.arg(), up.arg(), out.arg(), row_pitch.arg()],
    )
}

/// The `Mlp` family, claimed. One point lands as a launcher, three are
/// written whole against a door this plane does not have, and two are
/// measured backlog rows.
///
/// # The family's shape is the seam
///
/// `kernels::points` opens this family by saying "every point but
/// `geglu_tanh` reads ONE packed `[gate | up]` row" — and `gated.slang`
/// binds `gate` and `up` as TWO descriptors that both start at element
/// zero. Even the strided arm does: it addresses
/// `gate[m * gate_pitch + k]` and `up[m * up_pitch + k]`, two pitches over
/// two bindings and no base between them. So the one unpacked point is the
/// one this plane can fire today, and the three packed ones wait on
/// [`crate::points::Staged::window`] — a binding with an offset, or a
/// `base` word in the shader's push block.
///
/// This is not an accident of the port. A cuda body cuts a packed row by
/// advancing a pointer, which is free; a descriptor set names allocations,
/// so the same cut is either a driver-minted sub-handle or a shader
/// parameter. P5 picks which, and the three bodies below are already
/// written for the first.
///
/// # Two points stay on the floor's default body
///
/// * `mlp.swiglu_clamp` — a clamped swiglu with NO alpha and no linear
///   fold. `gptoss_swiglu` is not a re-parameterisation of it: it computes
///   `(g * sigmoid(alpha * g)) * (u + 1.0)`, and both the `alpha` and the
///   `+ 1.0` are gpt-oss's own. Passing `alpha = 1.0` would leave the
///   fold, which is different numbers.
/// * `mlp.situ` — kimi's beta-scaled silu with an `up_cap`. No
///   instantiation in `mlp/gated.slang` computes it.
#[kernels_macros::claims]
impl kernels::points::Mlp for Ctx<'_> {
    /// `y = silu(gate) * up` over a packed `[gate | up]` row.
    ///
    /// SEAM: the cut. `intermediate` IS the result's row and the second
    /// half starts at exactly that column, so the two operands the shader
    /// wants are `packed[.., 0..intermediate]` and
    /// `packed[.., intermediate..2*intermediate]`. Both are windows of one
    /// binding and this plane cannot express one — see the impl header.
    fn swiglu<T: kernels::points::Scalar>(
        &self,
        packed: In<crate::points::Handle<T>>,
        intermediate: u32,
        y: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>("mlp.swiglu, at an element this plane does not instantiate")?;
        let cut = halves(self, packed, intermediate)?;
        self.fire(
            Fire::at(
                crate::routine::module_path("silu_mul_bfloat16", self.best()),
                "silu_mul_bfloat16",
            )
            .apply(elementwise(cut.width, cut.rows)?),
            &[cut.gate.arg(), cut.up.arg(), y.arg()],
        )
    }

    /// The unpacked geglu: two rows in, one out.
    ///
    /// THE ONE POINT OF THIS FAMILY THIS PLANE FIRES TODAY, and the reason
    /// is the whole of the impl header: gate and up arrive as separate
    /// rectangles, so the two bindings the shader wants are the two the
    /// statement carries.
    fn geglu_tanh<T: kernels::points::Scalar>(
        &self,
        gate: In<crate::points::Handle<T>>,
        up: In<crate::points::Handle<T>>,
        y: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "mlp.geglu_tanh, at an element this plane does not instantiate",
        )?;
        let row = gate.all("the gate half's rectangle")?;
        self.fire(
            Fire::at(
                crate::routine::module_path("geglu_tanh_bfloat16", self.best()),
                "geglu_tanh_bfloat16",
            )
            .apply(elementwise(row.width, row.rows)?),
            &[gate.arg(), up.arg(), y.arg()],
        )
    }

    /// [`Self::geglu_tanh`] over one packed row. SEAM: the cut — see
    /// [`Self::swiglu`].
    fn geglu_tanh_packed<T: kernels::points::Scalar>(
        &self,
        packed: In<crate::points::Handle<T>>,
        intermediate: u32,
        y: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "mlp.geglu_tanh_packed, at an element this plane does not instantiate",
        )?;
        let cut = halves(self, packed, intermediate)?;
        self.fire(
            Fire::at(
                crate::routine::module_path("geglu_tanh_bfloat16", self.best()),
                "geglu_tanh_bfloat16",
            )
            .apply(elementwise(cut.width, cut.rows)?),
            &[cut.gate.arg(), cut.up.arg(), y.arg()],
        )
    }

    /// gpt-oss's clamped, alpha-scaled swiglu with the linear fold.
    ///
    /// The two scalars are exactly `gptoss_swiglu`'s push block and the
    /// arithmetic is its body verbatim — `min(g, limit)`,
    /// `clamp(u, -limit, limit)`, `sigmoid(alpha * g)`, `* (u + 1)`. What
    /// is missing is the same cut as [`Self::swiglu`]'s and nothing else,
    /// so of the four packed points this is the one closest to firing.
    fn swiglu_clamp_alpha<T: kernels::points::Scalar>(
        &self,
        packed: In<crate::points::Handle<T>>,
        intermediate: u32,
        limit: f32,
        alpha: f32,
        y: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "mlp.swiglu_clamp_alpha, at an element this plane does not instantiate",
        )?;
        let cut = halves(self, packed, intermediate)?;
        self.fire(
            Fire::at(
                crate::routine::module_path("gptoss_swiglu_bfloat16", self.best()),
                "gptoss_swiglu_bfloat16",
            )
            .apply(elementwise(cut.width, cut.rows)?),
            &[
                cut.gate.arg(),
                cut.up.arg(),
                y.arg(),
                limit.arg(),
                alpha.arg(),
            ],
        )
    }
}

/// The packed row's two halves, and the rectangle they write.
///
/// SEAM in one place for all three packed points: `intermediate` is the
/// result's row and the cut falls there, so the halves are
/// `packed[0..intermediate]` and `packed[intermediate..]` — two windows of
/// one binding. [`crate::points::Staged::window`] is the door.
struct Halves<T> {
    /// The leading `intermediate` columns.
    gate: crate::points::Handle<T>,
    /// The trailing `intermediate` columns.
    up: crate::points::Handle<T>,
    /// The rectangle both halves — and the result — are.
    rows: i32,
    width: i32,
}

fn halves<T: kernels::points::Scalar>(
    ctx: &Ctx<'_>,
    packed: In<crate::points::Handle<T>>,
    intermediate: u32,
) -> Result<Halves<T>, Refusal> {
    use crate::points::Staged;

    let row = packed.all("the packed `[gate | up]` row")?;
    let half = crate::points::stated("the intermediate width this statement states", intermediate)?;
    if row.width != half.saturating_mul(2) {
        return Err(Refusal::Narrow {
            what: "the packed `[gate | up]` row, against the intermediate width it states",
            at: i64::from(row.width),
        });
    }
    Ok(Halves {
        gate: ctx.window(packed.ptr, 0, half)?,
        up: ctx.window(packed.ptr, i64::from(half), half)?,
        rows: row.rows,
        width: half,
    })
}
