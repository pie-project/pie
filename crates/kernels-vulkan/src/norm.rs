use crate::routine::{Bind, Const, Ctx, Fire, In, InOut, Out, elementwise, elementwise_rows};
use kernels::routine::Refusal;

/// The whole-row norm's grid: one 256-wide workgroup per `axis`-wide slice.
///
/// `rms.slang` addresses `row_base = group.x * axis_size`, so the slice
/// count is the row over the stated axis and the workgroup is 256 lanes
/// cooperating on one reduction. A dispatch here is in TOTAL THREADS, which
/// is why the product is multiplied out rather than handed over as groups.
pub fn per_axis(width: i32, axis: i32, rows: i32) -> Result<[u32; 3], Refusal> {
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

/// The per-head norm's grid: one 256-wide workgroup per (head, row).
///
/// Where [`per_axis`] multiplies the slices out onto x, this spreads them
/// over y and z — the shape `gated_rms.slang` and `rms_strided*` address,
/// which read the head off `group.y` because their rows are not dense.
pub fn per_head_row(heads: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if heads <= 0 {
        return Err(Refusal::Empty { what: "heads" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    Ok([256, heads.unsigned_abs(), rows.unsigned_abs()])
}

/// The `Norm` family, claimed. Ten of twelve points land as LAUNCHERS —
/// `.wiki/baker.md`'s endpoint shape, where the impl method fires and
/// nothing stands under it — and the two absences are stated rather than
/// left to be counted.
///
/// # One kernel, four points
///
/// `rms_single_row_bfloat16`'s push block carries `axis_size` and
/// `plus_one`, and those two words are the whole difference between
/// `rmsnorm`, `rmsnorm_per_head`, `rmsnorm_plus_one` and
/// `rmsnorm_per_head_plus_one`. That is exactly the split the declarations
/// draw and the reason they draw it: `plus_one` is a fact about the
/// CHECKPOINT (`.wiki/baker.md`'s gemma law, and W3's finding that the
/// premise was false for gemma-4 but true for qwen3.5), so it is a point
/// and not a flag, and `axis_size` is the head width a `Const` weight
/// cannot carry. Four bodies over one entrypoint is what a declaration
/// floor buys.
///
/// `w_stride` is `1` and `gain` is `1.0` at every one of them. Neither is a
/// number any declaration here states: `w[w_stride * i]` exists for a
/// weight plane packed against a wider bank, and `gain` for a fused scale
/// the fused `rms_rope`/`rms_residual` forms fold in. A point that wanted
/// either would state it, and none does.
///
/// # The grid is read, never stated
///
/// `per_axis(width, axis, rows)` grids one workgroup per `axis`-wide slice,
/// and `rms.slang`'s `row_base` is `group.x * axis_size` — so the whole-row
/// forms pass `axis = width` and the per-head forms pass `axis = head_dim`,
/// and in both readings the ROWS come off the operand's own rectangle. The
/// retired routine layer took a `rows: Const<i32>` beside every one of
/// these launches; W10's law is that an executor hands a kernel dense
/// rectangles, and a dense rectangle knows how many rows it has.
///
/// # A FUSED ARM THE FLOOR'S DECOMPOSITION DOES NOT REACH
///
/// `norm/rms_rope.slang` stamps six entrypoints that do the norm AND the
/// rotation in one launch, over the geometric, tabulated and proportional
/// frequency readings and their decode cuts. No point names one: the floor
/// declares `norm.rmsnorm` and the `Rope` family separately and both land
/// here, so the fused arm is a PERFORMANCE reading of a decomposition this
/// plane can already fire — unlike [`crate::ssm`]'s fusion, which is a
/// decomposition disagreement and blocks every point of its family. It
/// becomes reachable when a text may name a fused point, which
/// `.wiki/baker.md` reserves for tier-2.
///
/// The same reading covers the four strided arms (`rms_strided_row`,
/// `rms_strided_head_row`, `gated_rms_strided`, `residual_add_strided`):
/// they take a row pitch, a mark carries a dense rectangle, and W10's law
/// is that an executor hands a kernel dense rectangles.
///
/// # Two points stay on the floor's default body
///
/// * `norm.mul_scalar` — multiply by a HOST `f32`. `layer_scalar.slang`
///   reads its factor from a BINDING (`PIE_BUFFER_RO(1, scalar)`), which is
///   `norm.scale`'s mark and not this one's. There is no push-constant
///   multiply on this plane, so the gap is a shader instantiation and not
///   plumbing — a measured backlog row.
/// * `norm.res_blend` — kimi's variadic ledger item, unclaimed on every
///   plane for the reason `kernels::points` states at length.
///
/// # SEAM, and it is the same one twice
///
/// `rmsnorm_gated` and `rmsnorm_gated_by` declare `x` and `weight` at
/// `f32` — a decay core is accumulated, not activated, and cuda's
/// `rmsnorm_gated_fp32_in` reads it that way. `gated_rms.slang` binds all
/// four of its planes through `PIE_BUFFER_*`, which is `PIE_ACT`, which is
/// `bfloat16` at every instantiation this tree stamps. The two bodies below
/// are the launch as it stands and the crossing is marked where it happens;
/// what closes it is an `_f32_core` instantiation of that shader, which is
/// shader work. It is not urgent on this plane for a stated reason: the
/// only thing that produces an f32 GDN core is `ssm.gated_delta`, and this
/// plane does not claim it (see [`crate::ssm`]).
#[kernels_macros::claims]
impl kernels::points::Norm for Ctx<'_> {
    fn rmsnorm<T: kernels::points::Scalar>(
        &self,
        x: In<crate::points::Handle<T>>,
        weight: Const<crate::points::Handle<T>>,
        eps: f32,
        y: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>("norm.rmsnorm, at an element this plane does not instantiate")?;
        let row = x.all("the normalised row's width")?;
        self.fire(
            Fire::at(
                crate::routine::module_path("rms_single_row_bfloat16", self.best()),
                "rms_single_row_bfloat16",
            )
            .apply(per_axis(row.width, row.width, row.rows)?),
            &[
                x.arg(),
                weight.arg(),
                y.arg(),
                eps.arg(),
                row.width.arg(),
                1u32.arg(),
                0u32.arg(),
                1.0f32.arg(),
            ],
        )
    }

    fn rmsnorm_per_head<T: kernels::points::Scalar>(
        &self,
        x: In<crate::points::Handle<T>>,
        weight: Const<crate::points::Handle<T>>,
        head_dim: u32,
        eps: f32,
        y: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "norm.rmsnorm_per_head, at an element this plane does not instantiate",
        )?;
        let row = x.all("the row the heads divide")?;
        let axis = crate::points::stated("the head width this norm states", head_dim)?;
        self.fire(
            Fire::at(
                crate::routine::module_path("rms_single_row_bfloat16", self.best()),
                "rms_single_row_bfloat16",
            )
            .apply(per_axis(row.width, axis, row.rows)?),
            &[
                x.arg(),
                weight.arg(),
                y.arg(),
                eps.arg(),
                axis.arg(),
                1u32.arg(),
                0u32.arg(),
                1.0f32.arg(),
            ],
        )
    }

    /// [`Self::rmsnorm`] against an OFFSET bank: `plus_one = 1`, and that
    /// word is the whole of the difference. One kernel, two conventions,
    /// and the declaration is what picks — the same sentence
    /// `kernels-cuda`'s pair carries, over a push constant instead of a
    /// template argument.
    fn rmsnorm_plus_one<T: kernels::points::Scalar>(
        &self,
        x: In<crate::points::Handle<T>>,
        weight: Const<crate::points::Handle<T>>,
        eps: f32,
        y: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "norm.rmsnorm_plus_one, at an element this plane does not instantiate",
        )?;
        let row = x.all("the normalised row's width")?;
        self.fire(
            Fire::at(
                crate::routine::module_path("rms_single_row_bfloat16", self.best()),
                "rms_single_row_bfloat16",
            )
            .apply(per_axis(row.width, row.width, row.rows)?),
            &[
                x.arg(),
                weight.arg(),
                y.arg(),
                eps.arg(),
                row.width.arg(),
                1u32.arg(),
                1u32.arg(),
                1.0f32.arg(),
            ],
        )
    }

    fn rmsnorm_per_head_plus_one<T: kernels::points::Scalar>(
        &self,
        x: In<crate::points::Handle<T>>,
        weight: Const<crate::points::Handle<T>>,
        head_dim: u32,
        eps: f32,
        y: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "norm.rmsnorm_per_head_plus_one, at an element this plane does not instantiate",
        )?;
        let row = x.all("the row the heads divide")?;
        let axis = crate::points::stated("the head width this norm states", head_dim)?;
        self.fire(
            Fire::at(
                crate::routine::module_path("rms_single_row_bfloat16", self.best()),
                "rms_single_row_bfloat16",
            )
            .apply(per_axis(row.width, axis, row.rows)?),
            &[
                x.arg(),
                weight.arg(),
                y.arg(),
                eps.arg(),
                axis.arg(),
                1u32.arg(),
                1u32.arg(),
                1.0f32.arg(),
            ],
        )
    }

    /// The weightless norm, per head.
    ///
    /// A CORRECTION RIDES HERE, and it is why this is a body rather than a
    /// delegation to `vnorm_single_row` below. That routine grids with
    /// `per_axis(width, x.width, rows)` — the whole row — while passing
    /// `axis_size` to the shader separately, and `vector.slang` addresses
    /// `base = group.x * axis_size`. The two agree only when the stated
    /// axis IS the row: at any narrower `head_dim` the launch runs `rows`
    /// workgroups over a row that holds `width / head_dim` of them, so all
    /// but the first head of every row are left untouched. The grid below
    /// is `per_axis(width, head_dim, rows)`, which is what the shader
    /// addresses.
    fn rmsnorm_no_scale<T: kernels::points::Scalar>(
        &self,
        x: In<crate::points::Handle<T>>,
        head_dim: u32,
        eps: f32,
        y: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "norm.rmsnorm_no_scale, at an element this plane does not instantiate",
        )?;
        let row = x.all("the row the heads divide")?;
        let axis = crate::points::stated("the head width this norm states", head_dim)?;
        self.fire(
            Fire::at(
                crate::routine::module_path("vnorm_single_row_bfloat16", self.best()),
                "vnorm_single_row_bfloat16",
            )
            .apply(per_axis(row.width, axis, row.rows)?),
            &[x.arg(), y.arg(), eps.arg(), axis.arg()],
        )
    }

    fn rmsnorm_gated<T: kernels::points::Scalar>(
        &self,
        x: In<crate::points::Handle<f32>>,
        gate: In<crate::points::Handle<T>>,
        weight: Const<crate::points::Handle<f32>>,
        head_dim: u32,
        eps: f32,
        y: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "norm.rmsnorm_gated, at a gate element this plane does not instantiate",
        )?;
        let row = x.all("the row the value heads divide")?;
        let vd = crate::points::stated("the head width this gated norm states", head_dim)?;
        let heads =
            crate::points::heads("the value heads this gated norm divides by", row.width, vd)?;
        // SEAM: `x` and `weight` are declared `f32` and `gated_rms.slang`
        // binds both through `PIE_BUFFER_*` — `PIE_ACT`, `bfloat16` at
        // every instantiation this tree stamps. Closing it is an
        // `_f32_core` instantiation of that shader, not plumbing; see this
        // impl's header for why it is not this plane's first debt.
        self.fire(
            Fire::at(
                crate::routine::module_path("gated_rms_bfloat16", self.best()),
                "gated_rms_bfloat16",
            )
            .apply(per_head_row(heads, row.rows)?),
            &[
                x.arg(),
                gate.arg(),
                weight.arg(),
                y.arg(),
                eps.arg(),
                vd.arg(),
            ],
        )
    }

    /// [`Self::rmsnorm_gated`] with the HEAD COUNT stated instead of the
    /// head width. The shader takes `vd`, so the body divides — the
    /// mirror image of `rmsnorm_gated`'s multiply, and the reason both
    /// points exist is that a text holds one number or the other.
    fn rmsnorm_gated_by<T: kernels::points::Scalar>(
        &self,
        x: In<crate::points::Handle<f32>>,
        gate: In<crate::points::Handle<T>>,
        weight: Const<crate::points::Handle<f32>>,
        heads: u32,
        eps: f32,
        y: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "norm.rmsnorm_gated_by, at a gate element this plane does not instantiate",
        )?;
        let row = x.all("the row the heads divide")?;
        let n = crate::points::stated("the head count this norm states", heads)?;
        let vd = crate::points::heads("the head width this norm divides into", row.width, n)?;
        // SEAM: see [`Self::rmsnorm_gated`] — the same f32/`PIE_ACT`
        // crossing, on the same entrypoint.
        self.fire(
            Fire::at(
                crate::routine::module_path("gated_rms_bfloat16", self.best()),
                "gated_rms_bfloat16",
            )
            .apply(per_head_row(n, row.rows)?),
            &[
                x.arg(),
                gate.arg(),
                weight.arg(),
                y.arg(),
                eps.arg(),
                vd.arg(),
            ],
        )
    }

    /// `y += x`, and the two operands go to the shader's `x` and
    /// `residual` in the order cuda's claim uses: the running stream
    /// first. `residual_add.slang` adds them, so the order is a reading
    /// convention rather than arithmetic — what matters is that `out_` is
    /// the `InOut`'s binding.
    ///
    /// ONE BINDING, TWICE. `out_` and `x` are the same handle here, which
    /// this shader permits because it is elementwise 1:1: every lane reads
    /// and writes the one index it owns. The general aliasing question the
    /// floor has open (`.wiki/baker-todo.md`, "Arena liveness/reuse +
    /// InOut aliasing") is about points where that is not true.
    fn residual_add<T: kernels::points::Scalar>(
        &self,
        x: In<crate::points::Handle<T>>,
        y: InOut<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "norm.residual_add, at an element this plane does not instantiate",
        )?;
        let row = y.all("the residual stream's rectangle")?;
        self.fire(
            Fire::at(
                crate::routine::module_path("residual_add_bfloat16", self.best()),
                "residual_add_bfloat16",
            )
            .apply(elementwise(row.width, row.rows)?),
            // THE READ SIDE IS BOUND READ-ONLY. `y` is an `InOut`, so
            // `y.arg()` is `buffer_mut` — right for `out_` at binding 2
            // and wrong for `x` at binding 0, which `residual_add.slang`
            // declares `PIE_BUFFER_RO`. `y.ptr.arg()` is the same handle
            // through the read spelling.
            &[y.ptr.arg(), x.arg(), y.arg()],
        )
    }

    fn add_bias<T: kernels::points::Scalar>(
        &self,
        bias: Const<crate::points::Handle<T>>,
        out: InOut<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "norm.add_bias, at an element this plane does not instantiate",
        )?;
        let row = out.all("the biased rectangle's row width")?;
        self.fire(
            Fire::at(
                crate::routine::module_path("add_bias_bfloat16", self.best()),
                "add_bias_bfloat16",
            )
            .apply(elementwise_rows(row.width, row.rows)?),
            &[out.arg(), bias.arg(), row.width.arg()],
        )
    }

    /// `x *= s[0]`, the factor a `[1]` bank on the device.
    ///
    /// The `[1]` shape is not checked here and could not be: a `Const`
    /// carries the weight's binding and no rectangle, so the one element
    /// this reads is the model text's claim about its own checkpoint,
    /// verified where that claim is made. `kernels-cuda`'s `norm.scale`
    /// states the same thing about the same point.
    fn scale<T: kernels::points::Scalar>(
        &self,
        s: Const<crate::points::Handle<T>>,
        x: InOut<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>("norm.scale, at an element this plane does not instantiate")?;
        let row = x.all("the scaled rectangle's row width")?;
        self.fire(
            Fire::at(
                crate::routine::module_path("layer_scalar_mul_bfloat16", self.best()),
                "layer_scalar_mul_bfloat16",
            )
            .apply(elementwise(row.width, row.rows)?),
            // Read-only at binding 0, writable at binding 2 — one
            // handle, two spellings; see `residual_add` above.
            &[x.ptr.arg(), s.arg(), x.arg()],
        )
    }
}
