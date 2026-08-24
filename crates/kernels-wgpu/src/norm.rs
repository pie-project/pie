use crate::points::{Payload, at_bf16};
use crate::routine::{Bind, Const, Ctx, Fire, In, InOut, Out, elementwise_rows};
use kernels::routine::Refusal;

fn per_axis(width: i32, axis: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if width <= 0 {
        return Err(Refusal::Empty { what: "width" });
    }
    if axis <= 0 {
        return Err(Refusal::Empty { what: "axis" });
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
    let axes = (width.unsigned_abs() / axis.unsigned_abs()) * rows.unsigned_abs();
    let lanes = axes.checked_mul(256).ok_or(Refusal::Grid {
        what: "axes * the workgroup width",
        at: i64::from(axes) * 256,
    })?;
    Ok([lanes, 1, 1])
}

/// The `Norm` family, claimed. Seven of twelve points land as launchers; the
/// other five are measured backlog rows and each absence is a stated one.
///
/// # The five that do not land
///
/// * `norm.rmsnorm_gated` and `norm.rmsnorm_gated_by` — THE SHADER READS A
///   DIFFERENT ELEMENT THAN THE DECLARATION STATES. Both points spell
///   `x: In<Self::Tensor<f32>>` and `weight: Const<Self::Tensor<f32>>`,
///   because the GDN mixer's output is carried in f32 and rounded once at
///   the gate. `norm/gated_rms.wgsl` binds `x`, `z` and `w` as `array<u32>`
///   and unpacks each through `pie_bf16_at` — all three are bf16 — so a
///   claim here would hand an f32 rectangle to a shader that reads it as two
///   bf16 per word. That is not a refusal a body can make conditionally: the
///   mismatch is unconditional, so the honest row is the family's default,
///   and the fix is a `PIE_F32_IN` instantiation of `gated_rms.wgsl` rather
///   than a line here. Nothing fires either `gated_rms` entrypoint now; the
///   launchers that claimed the point by name went with the routine layer.
/// * `norm.mul_scalar` — the factor is a HOST scalar (`s: f32`) and every
///   scale shader on this plane reads a `[1]` bank off a binding.
///   `norm/layer_scalar.wgsl` is the bank form and it lands as `norm.scale`;
///   nothing here multiplies by a number carried in the uniform block.
/// * `norm.res_blend` — kimi's variadic ledger item, unclaimed on cuda too
///   and for the reason its `Norm` block gives: the statement's arity is a
///   function of which layer states it.
///
/// # `w_stride`, `plus_one` and `gain` are stated by the CLAIM, not the plan
///
/// `norm/rms.wgsl` reads five words out of its `@group(1)` uniform and three
/// of them are conventions rather than measurements: `w_stride` is the
/// distance between consecutive channels of the gain vector, which is one
/// for every contiguous bank this tree loads; `plus_one` is gemma's `1 + w`
/// reading; `gain` is a post-multiplier no `Norm` point states. The routine
/// layer made all three `Const` marks and asked the PLAN to place them,
/// which is how `norm.rmsnorm` and `norm.rmsnorm_plus_one` came to be one
/// row with a flag. The declaration splits them into two POINTS — a fact
/// about the checkpoint, decided once for a text's whole life — so the
/// bodies below state the words and no plan carries them.
///
/// # `rows` is read off the mark
///
/// Every launcher this file used to carry ended with `rows: Const<i32>`,
/// because a fire's extent is not recoverable from a bare handle and the
/// lowering spliced the number in. A points mark is `In { ptr, rows, width }`
/// and `Backend::region` fills both off `ArgValue::Shaped`, so these bodies
/// read `x.rows` and the trailing param is gone.
///
/// # `norm/rms_rope.wgsl` is a fused arm no point states
///
/// One dispatch that does the per-head RMS norm and the NEOX rotation
/// together, written against `model-dsl::metal`'s `rms_rope` declaration and
/// fired here through the same table. `kernels::points` cuts the two apart
/// (`Norm::rmsnorm_per_head` then `Rope::partial`), so the fusion is a POINT
/// nobody has declared and its launcher went with the routine layer.
///
/// The one refusal that was this backend's and not the family's goes with
/// it, and is restated here because the WGSL states the word discipline but
/// not the host's guard: two bf16 share a four-byte word, the kernel's
/// rotation walks `[base, base + rotary)` in WORDS and its tail walks
/// `[base + rotary, base + axis)`, so `rotary`, `axis_size` and `row_pitch`
/// must all be EVEN or an invocation races the neighbour holding the other
/// half. They are, in every checkpoint this tree loads; a deployment where
/// they are not wants the unfused pair rather than a torn head.
#[kernels_macros::claims]
impl kernels::points::Norm for Ctx<'_> {
    fn rmsnorm<T: kernels::points::Scalar>(
        &self,
        x: In<Payload<T>>,
        weight: Const<Payload<T>>,
        eps: f32,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("norm.rmsnorm at an element other than bf16")?;
        rms_row(self, x, weight, y, eps, x.width, 0)
    }

    /// The row divided into `head_dim`-wide axes, each normalised alone.
    ///
    /// ONE SHADER, TWO POINTS, AND THE GRID IS WHAT DIFFERS.
    /// `rms_single_row_bfloat16` bases at `wg.x * axis_size`, so a launch of
    /// `(width / axis) * rows` workgroups over an `axis` of `head_dim` walks
    /// the heads and a launch over an `axis` of the whole width walks the
    /// rows. The weight is addressed from the AXIS's start either way
    /// (`normed` passes `at`, the row-relative index, not the absolute one),
    /// which is exactly what a one-head-wide bank wants.
    fn rmsnorm_per_head<T: kernels::points::Scalar>(
        &self,
        x: In<Payload<T>>,
        weight: Const<Payload<T>>,
        head_dim: u32,
        eps: f32,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("norm.rmsnorm_per_head at an element other than bf16")?;
        rms_row(self, x, weight, y, eps, axis_of(head_dim)?, 0)
    }

    /// [`kernels::points::Norm::rmsnorm`] against a bank stored as an OFFSET:
    /// the scale is `1 + weight`.
    ///
    /// The same shader at `plus_one = 1`, and the fold happens in FLOAT
    /// before the bf16 round (`gain_at`) — MLX's choice, and the one a parity
    /// walk against it has to make too.
    fn rmsnorm_plus_one<T: kernels::points::Scalar>(
        &self,
        x: In<Payload<T>>,
        weight: Const<Payload<T>>,
        eps: f32,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("norm.rmsnorm_plus_one at an element other than bf16")?;
        rms_row(self, x, weight, y, eps, x.width, 1)
    }

    fn rmsnorm_per_head_plus_one<T: kernels::points::Scalar>(
        &self,
        x: In<Payload<T>>,
        weight: Const<Payload<T>>,
        head_dim: u32,
        eps: f32,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("norm.rmsnorm_per_head_plus_one at an element other than bf16")?;
        rms_row(self, x, weight, y, eps, axis_of(head_dim)?, 1)
    }

    /// The weightless per-head norm — `norm/vector.wgsl`, not `rms.wgsl`.
    ///
    /// A SEPARATE MODULE AND NOT `rms.wgsl` WITH A NULL BANK, because this
    /// plane cannot bind a null: `driver-wgpu` builds a bind group out of the
    /// module's own declarations, so a declared-and-unfilled slot is a layout
    /// the dispatch cannot satisfy (`norm/rms.wgsl`'s header says so where it
    /// explains why deleting `params` renumbered `r` and `s`). The no-scale
    /// reading is therefore a module with two bindings instead of three, and
    /// that module is `vnorm_single_row_bfloat16`.
    fn rmsnorm_no_scale<T: kernels::points::Scalar>(
        &self,
        x: In<Payload<T>>,
        head_dim: u32,
        eps: f32,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("norm.rmsnorm_no_scale at an element other than bf16")?;
        let axis = axis_of(head_dim)?;
        self.fire(
            Fire::at("norm/vector.wgsl", "vnorm_single_row_bfloat16")
                .apply(per_axis(x.width, axis, x.rows)?),
            &[x.arg(), y.arg(), eps.arg(), axis.arg()],
        )
    }

    /// `y += x`.
    ///
    /// ONE HANDLE IN TWO BINDINGS, and this is the shape every read-modify
    /// point takes on this plane. `norm/residual_add.wgsl` declares three
    /// storage buffers — `x`, `residual`, `out_` — because a routine states
    /// three rectangles. The point states two, and the second is read AND
    /// written, so the fire binds `y`'s handle at binding 1 and again at
    /// binding 2.
    ///
    /// SAFE HERE, AND THE REASON IS THE INDEXING RATHER THAN THE SPEC: every
    /// invocation of this shader reads `residual[i]` and writes `out_[i]` for
    /// its own `i` alone, so the two views of the one buffer never cross.
    /// WebGPU permits a buffer to appear in two bindings and promises nothing
    /// stronger; aliasing a shader whose write index differed from its read
    /// index would be leaning on a promise nobody made.
    ///
    /// **SEAM (P5):** `driver-wgpu::lowering::bind` has never been handed the
    /// same handle twice in one bind group. It must neither dedupe the entry
    /// nor refuse it.
    fn residual_add<T: kernels::points::Scalar>(
        &self,
        x: In<Payload<T>>,
        y: InOut<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("norm.residual_add at an element other than bf16")?;
        self.fire(
            Fire::at("norm/residual_add.wgsl", "residual_add_bfloat16")
                .apply(elementwise_rows(y.width, y.rows)?),
            &[x.arg(), y.ptr.arg(), y.arg()],
        )
    }

    /// `out += bias`, the bias one row wide and broadcast down.
    ///
    /// The one read-modify point whose shader ALREADY declares the
    /// destination once, so nothing aliases: `norm/add_bias.wgsl` binds
    /// `out_` at 0 and reads and writes it there.
    fn add_bias<T: kernels::points::Scalar>(
        &self,
        bias: Const<Payload<T>>,
        out: InOut<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("norm.add_bias at an element other than bf16")?;
        let width = out.width;
        self.fire(
            Fire::at("norm/add_bias.wgsl", "add_bias_bfloat16")
                .apply(elementwise_rows(width, out.rows)?),
            &[out.arg(), bias.arg(), width.arg()],
        )
    }

    /// `x *= s[0]`, the factor a `[1]` bank on the device.
    ///
    /// gemma's per-layer-embedding scalar. The `[1]` shape is not checked
    /// here and could not be — a `Const` carries the weight's handle and no
    /// rectangle — so the one element this reads is the model text's claim
    /// about its own checkpoint, verified where that claim is made.
    ///
    /// Aliased like [`kernels::points::Norm::residual_add`], for the same
    /// reason and with the same per-element indexing behind it.
    fn scale<T: kernels::points::Scalar>(
        &self,
        s: Const<Payload<T>>,
        x: InOut<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("norm.scale at an element other than bf16")?;
        self.fire(
            Fire::at("norm/layer_scalar.wgsl", "layer_scalar_mul_bfloat16")
                .apply(elementwise_rows(x.width, x.rows)?),
            &[x.ptr.arg(), s.arg(), x.arg()],
        )
    }
}

/// The head width a per-head point states, as the shader's `axis_size`.
fn axis_of(head_dim: u32) -> Result<i32, Refusal> {
    i32::try_from(head_dim).map_err(|_| Refusal::Wide {
        what: "the head width this norm states",
        at: i64::from(head_dim),
        max: i64::from(i32::MAX),
    })
}

/// `rms_single_row_bfloat16`, at whichever axis and convention the point
/// picked. The four weighted `Norm` points differ in exactly those two
/// numbers and in nothing else, so they say so once.
///
/// The five words after the buffers are the shader's `Params` in order —
/// eps, axis_size, w_stride, plus_one, gain — and a body that reordered them
/// would read a gain as an epsilon with no error anywhere, which is
/// `norm/rms.wgsl`'s own warning about its own block.
fn rms_row<T: kernels::points::Scalar>(
    ctx: &Ctx<'_>,
    x: In<Payload<T>>,
    weight: Const<Payload<T>>,
    y: Out<Payload<T>>,
    eps: f32,
    axis: i32,
    plus_one: u32,
) -> Result<(), Refusal> {
    ctx.fire(
        Fire::at("norm/rms.wgsl", "rms_single_row_bfloat16")
            .apply(per_axis(x.width, axis, x.rows)?),
        &[
            x.arg(),
            weight.arg(),
            y.arg(),
            eps.arg(),
            axis.arg(),
            1u32.arg(),
            plus_one.arg(),
            1.0f32.arg(),
        ],
    )
}
