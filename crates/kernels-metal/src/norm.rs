use kernels::Grid;
use kernels::points::Scalar;
use kernels::routine::Refusal;

use crate::plane::{self, Handle};
use crate::routine::{
    Bind, Const, Ctx, Fire, In, InOut, Out, Tensor, bf16, elementwise, elementwise_rows,
};

fn rms_threads(axis: i32) -> Result<u32, Refusal> {
    if axis <= 0 {
        return Err(Refusal::Empty { what: "axis" });
    }
    Ok(axis.unsigned_abs().div_ceil(4).min(1024))
}

fn rms_grid(width: i32, axis: i32, rows: i32) -> Result<([u32; 3], [u32; 3]), Refusal> {
    if width <= 0 {
        return Err(Refusal::Empty { what: "width" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if axis > width {
        return Err(Refusal::Wide {
            what: "axis",
            at: i64::from(axis),
            max: i64::from(width),
        });
    }
    let t = rms_threads(axis)?;
    let norms = width.unsigned_abs() / axis.unsigned_abs();
    let lanes = t
        .checked_mul(norms)
        .and_then(|n| n.checked_mul(rows.unsigned_abs()))
        .ok_or(Refusal::Grid {
            what: "axis threads * norms per row * rows",
            at: i64::from(t) * i64::from(norms) * i64::from(rows),
        })?;
    Ok(([lanes, 1, 1], [t, 1, 1]))
}

fn head_row_grid(threads: u32, heads: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if heads <= 0 {
        return Err(Refusal::Empty { what: "heads" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    Ok([threads, heads.unsigned_abs(), rows.unsigned_abs()])
}

/// `norm/rms.metal`'s whole-row norm, and the only place its entrypoint is
/// named.
///
/// FOUR POINTS FIRE THIS ONE LAUNCH, which is why it is a function rather
/// than four copies of itself in the impl below: `rmsnorm`,
/// `rmsnorm_per_head`, `rmsnorm_plus_one` and `rmsnorm_per_head_plus_one`
/// differ in the AXIS and in `plus_one` and in nothing else. The parameters
/// are bare numbers because the impl bodies hold bare numbers.
#[allow(clippy::too_many_arguments)]
fn rms_row(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    w: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    eps: f32,
    axis: i32,
    w_stride: u32,
    plus_one: u32,
    gain: f32,
    rows: i32,
) -> Result<(), Refusal> {
    let (lanes, group) = rms_grid(x.width, axis, rows)?;
    ctx.fire(
        Fire::at("norm/rms.metal", "rms_single_row_bfloat16").apply(Grid::of(lanes, group)),
        &[
            x.arg(),
            w.arg(),
            out.arg(),
            eps.arg(),
            axis.arg(),
            w_stride.arg(),
            plus_one.arg(),
            gain.arg(),
        ],
    )
}

fn head_width(vd: i32) -> Result<u32, Refusal> {
    if vd <= 0 {
        return Err(Refusal::Empty { what: "vd" });
    }
    if vd > 1024 {
        return Err(Refusal::Wide {
            what: "vd",
            at: i64::from(vd),
            max: 1024,
        });
    }
    Ok(vd.unsigned_abs())
}

// ── the `rms_*` scalars, named ──────────────────────────────────────────
//
// `norm/rms.metal` takes five words beside its three buffers and a
// delegation has to state all five. Three of them are the same number for
// every statement a `points` declaration can make, and they are constants
// here rather than literals at the four call sites so that what each one
// MEANS is written once.

/// `w_stride`: the shader reads `w[w_stride * i]`. A bank this plane is
/// handed is contiguous — the strided reading was MLX's, for a weight
/// shared across a strided axis, and no text in this tree states one.
const DENSE_BANK: u32 = 1;

/// `plus_one = 0`: the scale IS the weight. See `Norm::rmsnorm_plus_one`
/// for why the other convention is a separate point and not a flag.
const ABSOLUTE_BANK: u32 = 0;

/// `plus_one = 1`: the scale is `1 + weight`, folded in float exactly as
/// MLX folds it.
const OFFSET_BANK: u32 = 1;

/// `gain`: an extra whole-tensor factor `norm/rms.metal`'s fused sandwich
/// arms carry. A
/// `points` statement states its scale as `norm.scale` or
/// `norm.mul_scalar` — its own point, with its own operand — so every claim
/// here passes the identity.
const UNIT_GAIN: f32 = 1.0;

/// The `Norm` family, claimed. Every body is the launch itself — the grid
/// this file's helpers compute and the argument run `norm/*.metal` reads —
/// deriving from the operands what the declaration does not state: the axis a
/// whole-row norm reduces over is the rectangle's width, the row count is the
/// rectangle's rows, and the three bank conventions are the constants above.
///
/// ONE KERNEL, FOUR POINTS. [`rms_row`] answers `rmsnorm`,
/// `rmsnorm_per_head`, `rmsnorm_plus_one` and `rmsnorm_per_head_plus_one`,
/// and the four differ in exactly two numbers: the AXIS (the whole row, or
/// one head of it — the grid takes `width / axis` threadgroups either way
/// and each re-reads the weight from zero, which is what makes a one-head
/// bank right for a many-head row) and `plus_one`. That is the shape cuda's
/// `Norm` impl has against the same two template parameters, and it is why
/// that one launch is a function here rather than four transcriptions of
/// itself.
///
/// Three points stay on the floor's default body, and each absence is a
/// measured row rather than an oversight:
///
/// * `norm.rmsnorm_gated_by` — KDA's per-head form. SEAM: no `.metal` kernel
///   divides a row by a head COUNT; `gated_rms` divides by a head WIDTH, and
///   the two agree only when the row is one head wide.
/// * `norm.mul_scalar` — SEAM: the factor is a host `f32` and no `.metal`
///   entrypoint takes one. `layer_scalar_mul` reads a DEVICE `[1]` bank and
///   answers `norm.scale` below; nothing here multiplies by a word off the
///   params run.
/// * `norm.res_blend` — kimi's variadic ledger item, and no `.metal` kernel
///   at all.
///
/// `norm.rmsnorm_gated` LANDED HERE BY A SHADER EDIT, and the edit is the
/// honest half of the claim rather than a convenience. The point declares
/// `x: In<Tensor<f32>>` and `weight: Const<Tensor<f32>>` because the
/// gated-delta core leaves its output in float; `norm/gated_rms.metal` was
/// one template over ONE `T` for all four buffers and said in as many words
/// that a `float*` there "reads element 2i+1 in place of i". It is now two
/// templates — `X` for the two planes the declaration pins to float, `T` for
/// the two that ride the statement — with `<bfloat, bfloat>` keeping the name
/// and the ABI the legacy driver fires and `<float, bfloat>` stamped beside
/// it as the arm this body names. THE SHADER IS METAL-COMPILE-UNVERIFIED:
/// nothing in this checkout can build a `.metal` file.
#[kernels_macros::claims]
impl kernels::points::Norm for Ctx<'_> {
    fn rmsnorm<T: Scalar>(
        &self,
        x: In<Handle<T>>,
        weight: Const<Handle<T>>,
        eps: f32,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`norm.rmsnorm`, at an element this plane does not stamp";
        let x = plane::input::<T, bf16>(x, WHAT)?;
        rms_row(
            self,
            x,
            plane::weight::<T, bf16>(weight, WHAT)?,
            plane::result::<T, bf16>(y, WHAT)?,
            eps,
            x.width,
            DENSE_BANK,
            ABSOLUTE_BANK,
            UNIT_GAIN,
            x.rows,
        )
    }

    fn rmsnorm_per_head<T: Scalar>(
        &self,
        x: In<Handle<T>>,
        weight: Const<Handle<T>>,
        head_dim: u32,
        eps: f32,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`norm.rmsnorm_per_head`, at an element this plane does not stamp";
        let x = plane::input::<T, bf16>(x, WHAT)?;
        rms_row(
            self,
            x,
            plane::weight::<T, bf16>(weight, WHAT)?,
            plane::result::<T, bf16>(y, WHAT)?,
            eps,
            plane::stated(head_dim, "the head width this norm states")?,
            DENSE_BANK,
            ABSOLUTE_BANK,
            UNIT_GAIN,
            x.rows,
        )
    }

    fn rmsnorm_plus_one<T: Scalar>(
        &self,
        x: In<Handle<T>>,
        weight: Const<Handle<T>>,
        eps: f32,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`norm.rmsnorm_plus_one`, at an element this plane does not stamp";
        let x = plane::input::<T, bf16>(x, WHAT)?;
        rms_row(
            self,
            x,
            plane::weight::<T, bf16>(weight, WHAT)?,
            plane::result::<T, bf16>(y, WHAT)?,
            eps,
            x.width,
            DENSE_BANK,
            OFFSET_BANK,
            UNIT_GAIN,
            x.rows,
        )
    }

    fn rmsnorm_per_head_plus_one<T: Scalar>(
        &self,
        x: In<Handle<T>>,
        weight: Const<Handle<T>>,
        head_dim: u32,
        eps: f32,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str =
            "`norm.rmsnorm_per_head_plus_one`, at an element this plane does not stamp";
        let x = plane::input::<T, bf16>(x, WHAT)?;
        rms_row(
            self,
            x,
            plane::weight::<T, bf16>(weight, WHAT)?,
            plane::result::<T, bf16>(y, WHAT)?,
            eps,
            plane::stated(head_dim, "the head width this norm states")?,
            DENSE_BANK,
            OFFSET_BANK,
            UNIT_GAIN,
            x.rows,
        )
    }

    /// The weightless form, and a different kernel rather than the same one
    /// with a null bank: `norm/vector.metal` takes three buffers where
    /// `rms.metal` takes four, because there is no weight to bind and no
    /// gain to apply. Gemma's V-norm is the only statement of it.
    fn rmsnorm_no_scale<T: Scalar>(
        &self,
        x: In<Handle<T>>,
        head_dim: u32,
        eps: f32,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`norm.rmsnorm_no_scale`, at an element this plane does not stamp";
        let x = plane::input::<T, bf16>(x, WHAT)?;
        let y = plane::result::<T, bf16>(y, WHAT)?;
        let axis = plane::stated(head_dim, "the head width this norm states")?;
        let (lanes, group) = rms_grid(x.width, axis, x.rows)?;
        self.fire(
            Fire::at("norm/vector.metal", "vnorm_single_row_bfloat16").apply(Grid::of(lanes, group)),
            &[x.arg(), y.arg(), eps.arg(), axis.arg()],
        )
    }

    /// The gated-delta fold: `weight * rmsnorm(x) * silu(gate)`, per value
    /// head.
    ///
    /// TWO ELEMENTS IN ONE LAUNCH, and they are the declaration's. `x` is the
    /// recurrence's f32 output and `weight` is the f32 norm row staged beside
    /// it; the gate and the result ride the statement's `T`. See this block's
    /// header for the shader edit that made the crossing honest.
    fn rmsnorm_gated<T: Scalar>(
        &self,
        x: In<Handle<f32>>,
        gate: In<Handle<T>>,
        weight: Const<Handle<f32>>,
        head_dim: u32,
        eps: f32,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`norm.rmsnorm_gated`, at an element this plane does not stamp";
        let x = plane::input::<f32, f32>(x, WHAT)?;
        let vd = plane::stated(head_dim, "the value-head width this fold states")?;
        let t = head_width(vd)?;
        // THE HEAD COUNT IS THE ROW OVER THE STATED WIDTH, and the shader
        // reads it back as `threadgroups_per_grid.y`, so the grid is where it
        // has to be right.
        if x.width <= 0 || x.width % vd != 0 {
            return Err(Refusal::Narrow {
                what: "the normed row does not divide by the value-head width this fold states",
                at: i64::from(x.width),
            });
        }
        let grid = head_row_grid(t, x.width / vd, x.rows)?;
        self.fire(
            Fire::at("norm/gated_rms.metal", "gated_rms_f32_bfloat16")
                .apply(Grid::of(grid, [t, 1, 1])),
            &[
                x.arg(),
                plane::input::<T, bf16>(gate, WHAT)?.arg(),
                plane::weight::<f32, f32>(weight, WHAT)?.arg(),
                plane::result::<T, bf16>(y, WHAT)?.arg(),
                eps.arg(),
                vd.arg(),
            ],
        )
    }

    /// `y += x`, and the destination is one of the operands — see
    /// [`crate::plane::read_half`] for why handing the shader the same handle
    /// twice is what an `InOut` means here.
    fn residual_add<T: Scalar>(
        &self,
        x: In<Handle<T>>,
        y: InOut<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`norm.residual_add`, at an element this plane does not stamp";
        let y = plane::in_place::<T, bf16>(y, WHAT)?;
        self.fire(
            Fire::at("norm/residual_add.metal", "residual_add_bfloat16")
                .apply(Grid::of(elementwise(y.width, y.rows)?, [256, 1, 1])),
            &[
                plane::input::<T, bf16>(x, WHAT)?.arg(),
                plane::read_half(y).arg(),
                plane::write_half(y).arg(),
            ],
        )
    }

    fn add_bias<T: Scalar>(
        &self,
        bias: Const<Handle<T>>,
        out: InOut<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`norm.add_bias`, at an element this plane does not stamp";
        let out = plane::in_place::<T, bf16>(out, WHAT)?;
        let lanes = elementwise_rows(out.width, out.rows)?;
        self.fire(
            Fire::at("norm/add_bias.metal", "add_bias_bfloat16")
                .apply(Grid::of(lanes, [lanes[0].min(256), 1, 1])),
            &[
                out.arg(),
                plane::weight::<T, bf16>(bias, WHAT)?.arg(),
                out.width.arg(),
            ],
        )
    }

    /// `x *= s[0]`, the factor a `[1]` bank on the device — gemma's
    /// per-layer scalar. The `[1]` shape is not checked here and could not
    /// be: a `Const` carries the weight's address and no rectangle, so the
    /// one element this reads is the model text's claim about its own
    /// checkpoint, verified where that claim is made.
    fn scale<T: Scalar>(&self, s: Const<Handle<T>>, x: InOut<Handle<T>>) -> Result<(), Refusal> {
        const WHAT: &str = "`norm.scale`, at an element this plane does not stamp";
        let x = plane::in_place::<T, bf16>(x, WHAT)?;
        self.fire(
            Fire::at("norm/layer_scalar.metal", "layer_scalar_mul_bfloat16")
                .apply(Grid::of(elementwise(x.width, x.rows)?, [256, 1, 1])),
            &[
                plane::read_half(x).arg(),
                plane::weight::<T, bf16>(s, WHAT)?.arg(),
                plane::write_half(x).arg(),
            ],
        )
    }
}

/// The `Hc` family, implemented and claiming nothing.
///
/// THE BLOCK IS THE MEASUREMENT. Hyper-connections are five points and this
/// plane has no kernel for any of them — `hc.expand` widens a row into
/// streams, `hc.rmsnorm_f32` norms the widened block into float,
/// `hc.gates` runs a sinkhorn over the stream mixture, and `hc.fold` and
/// `hc.collapse` put the streams back — and the `.metal` tree carries no
/// entrypoint that touches a stream axis at all. SEAM: five kernels, not five
/// delegations. The block exists so `Ctx` implements the family and its
/// default bodies are the backlog rows the resolution counts, which is what
/// `kernels_cuda::attn::POOL_CLAIMS` is on the other plane.
#[kernels_macros::claims]
impl kernels::points::Hc for Ctx<'_> {}
