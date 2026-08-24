use crate::routine::{Bind, Const, Ctx, Fire, In, Out, Tensor, bf16};
use kernels::routine::Refusal;
use kernels::shader::{elementwise, elementwise_rows};
use kernels_macros::routine;

fn affine_point(group: i32, bits: i32) -> Result<usize, Refusal> {
    let g = match group {
        32 => 0,
        64 => 1,
        128 => 2,
        _ => {
            return Err(Refusal::Narrow {
                what: "affine group size",
                at: i64::from(group),
            });
        }
    };
    let b = match bits {
        4 => 0,
        8 => 1,
        _ => {
            return Err(Refusal::Narrow {
                what: "affine bit width",
                at: i64::from(bits),
            });
        }
    };
    Ok(g * 2 + b)
}

#[routine]
pub fn embed_gather_4bit(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    group: Const<i32>,
    bits: Const<i32>,
    token_ids: In<Tensor<i32>>,
) -> Result<(), Refusal> {
    let id = token_ids.ptr;
    let hidden = out.width;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(
                [
                    "embed_gather_4bit_bfloat16_gs_32_b_4",
                    "embed_gather_4bit_bfloat16_gs_32_b_8",
                    "embed_gather_4bit_bfloat16_gs_64_b_4",
                    "embed_gather_4bit_bfloat16_gs_64_b_8",
                    "embed_gather_4bit_bfloat16_gs_128_b_4",
                    "embed_gather_4bit_bfloat16_gs_128_b_8",
                ][affine_point(*group, *bits)?],
                ctx.best(),
            ),
            [
                "embed_gather_4bit_bfloat16_gs_32_b_4",
                "embed_gather_4bit_bfloat16_gs_32_b_8",
                "embed_gather_4bit_bfloat16_gs_64_b_4",
                "embed_gather_4bit_bfloat16_gs_64_b_8",
                "embed_gather_4bit_bfloat16_gs_128_b_4",
                "embed_gather_4bit_bfloat16_gs_128_b_8",
            ][affine_point(*group, *bits)?],
        )
        .apply(elementwise(hidden, 1)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            id.arg(),
            out.arg(),
            hidden.arg(),
        ],
    )
}

// INLINED into impl Layout; dies with the routine layer. (layout.embed)
#[routine(canon = "layout.embed")]
pub fn embed_gather_mb_4bit(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    group: Const<i32>,
    bits: Const<i32>,
    token_ids: In<Tensor<i32>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let id = token_ids.ptr;
    let hidden = out.width;
    let rows = *rows;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(
                [
                    "embed_gather_mb_4bit_bfloat16_gs_32_b_4",
                    "embed_gather_mb_4bit_bfloat16_gs_32_b_8",
                    "embed_gather_mb_4bit_bfloat16_gs_64_b_4",
                    "embed_gather_mb_4bit_bfloat16_gs_64_b_8",
                    "embed_gather_mb_4bit_bfloat16_gs_128_b_4",
                    "embed_gather_mb_4bit_bfloat16_gs_128_b_8",
                ][affine_point(*group, *bits)?],
                ctx.best(),
            ),
            [
                "embed_gather_mb_4bit_bfloat16_gs_32_b_4",
                "embed_gather_mb_4bit_bfloat16_gs_32_b_8",
                "embed_gather_mb_4bit_bfloat16_gs_64_b_4",
                "embed_gather_mb_4bit_bfloat16_gs_64_b_8",
                "embed_gather_mb_4bit_bfloat16_gs_128_b_4",
                "embed_gather_mb_4bit_bfloat16_gs_128_b_8",
            ][affine_point(*group, *bits)?],
        )
        .apply(elementwise_rows(hidden, rows)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            id.arg(),
            out.arg(),
            hidden.arg(),
        ],
    )
}

#[routine]
pub fn embed_gather_scaled_4bit(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    embed_scale: Const<f32>,
    group: Const<i32>,
    bits: Const<i32>,
    token_ids: In<Tensor<i32>>,
) -> Result<(), Refusal> {
    let id = token_ids.ptr;
    let hidden = out.width;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(
                [
                    "embed_gather_scaled_4bit_bfloat16_gs_32_b_4",
                    "embed_gather_scaled_4bit_bfloat16_gs_32_b_8",
                    "embed_gather_scaled_4bit_bfloat16_gs_64_b_4",
                    "embed_gather_scaled_4bit_bfloat16_gs_64_b_8",
                    "embed_gather_scaled_4bit_bfloat16_gs_128_b_4",
                    "embed_gather_scaled_4bit_bfloat16_gs_128_b_8",
                ][affine_point(*group, *bits)?],
                ctx.best(),
            ),
            [
                "embed_gather_scaled_4bit_bfloat16_gs_32_b_4",
                "embed_gather_scaled_4bit_bfloat16_gs_32_b_8",
                "embed_gather_scaled_4bit_bfloat16_gs_64_b_4",
                "embed_gather_scaled_4bit_bfloat16_gs_64_b_8",
                "embed_gather_scaled_4bit_bfloat16_gs_128_b_4",
                "embed_gather_scaled_4bit_bfloat16_gs_128_b_8",
            ][affine_point(*group, *bits)?],
        )
        .apply(elementwise(hidden, 1)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            id.arg(),
            out.arg(),
            hidden.arg(),
            embed_scale.arg(),
        ],
    )
}

#[routine]
pub fn embed_gather_scaled_mb_4bit(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    embed_scale: Const<f32>,
    group: Const<i32>,
    bits: Const<i32>,
    token_ids: In<Tensor<i32>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let id = token_ids.ptr;
    let hidden = out.width;
    let rows = *rows;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(
                [
                    "embed_gather_scaled_mb_4bit_bfloat16_gs_32_b_4",
                    "embed_gather_scaled_mb_4bit_bfloat16_gs_32_b_8",
                    "embed_gather_scaled_mb_4bit_bfloat16_gs_64_b_4",
                    "embed_gather_scaled_mb_4bit_bfloat16_gs_64_b_8",
                    "embed_gather_scaled_mb_4bit_bfloat16_gs_128_b_4",
                    "embed_gather_scaled_mb_4bit_bfloat16_gs_128_b_8",
                ][affine_point(*group, *bits)?],
                ctx.best(),
            ),
            [
                "embed_gather_scaled_mb_4bit_bfloat16_gs_32_b_4",
                "embed_gather_scaled_mb_4bit_bfloat16_gs_32_b_8",
                "embed_gather_scaled_mb_4bit_bfloat16_gs_64_b_4",
                "embed_gather_scaled_mb_4bit_bfloat16_gs_64_b_8",
                "embed_gather_scaled_mb_4bit_bfloat16_gs_128_b_4",
                "embed_gather_scaled_mb_4bit_bfloat16_gs_128_b_8",
            ][affine_point(*group, *bits)?],
        )
        .apply(elementwise_rows(hidden, rows)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            id.arg(),
            out.arg(),
            hidden.arg(),
            embed_scale.arg(),
        ],
    )
}

#[routine(out(out = like(proj)))]
pub fn ple_combine(
    ctx: &Ctx<'_>,
    proj: In<Tensor<bf16>>,
    token: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    inv_sqrt2: Const<f32>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let width = proj.width;
    let rows = *rows;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("ple_combine_bfloat16", ctx.best()),
            "ple_combine_bfloat16",
        )
        .apply(elementwise(width, rows)?),
        &[proj.arg(), token.arg(), out.arg(), inv_sqrt2.arg()],
    )
}

#[routine]
pub fn row_gather(
    ctx: &Ctx<'_>,
    input: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    width: Const<u32>,
    sampling_indices: In<Tensor<u32>>,
    count: Const<u32>,
    row_count: Const<i32>,
) -> Result<(), Refusal> {
    let rows = sampling_indices.ptr;

    let count = *count;
    let row_count = *row_count;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("row_gather_bfloat16", ctx.best()),
            "row_gather_bfloat16",
        )
        .apply(elementwise_rows(input.width, row_count)?),
        &[input.arg(), out.arg(), rows.arg(), width.arg(), count.arg()],
    )
}

/// One workgroup grid per `(head_dim, heads, rows)`, which is what every
/// per-head cut on this plane addresses: `x` walks the head's channels,
/// `y` the heads, `z` the token rows. The same shape `attn.rs` grids its
/// `q_gate_split` and `kv_append` by, transcribed here beside the claim
/// that uses it.
fn head_grid(head_dim: i32, heads: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if head_dim <= 0 {
        return Err(Refusal::Empty {
            what: "the head width",
        });
    }
    if heads <= 0 {
        return Err(Refusal::Empty { what: "heads" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "tokens" });
    }
    Ok([
        head_dim.unsigned_abs(),
        heads.unsigned_abs(),
        rows.unsigned_abs(),
    ])
}

/// The `Layout` family, claimed. Three of five points land; two are
/// measured backlog rows and both absences are one missing kernel each.
///
/// # `embed` is the whole `Bank<R: Repr>` seam in one point
///
/// `layout.embed` declares `table: Const<Self::Tensor<T>>` — ONE address.
/// `embed_gather.slang` binds THREE — `StructuredBuffer<uint> w`, then
/// `scales` and `biases` at the activation element — and picks among six
/// modules on the `(group, bits)` pair. That is not a shape a body can
/// derive from a handle: it is what the LOAD contract bound to the
/// weight's name, and `.wiki/baker-todo.md` has it on the ledger as the
/// `Bank<R: Repr>` floor type. The body below is written against
/// [`crate::points::Staged::bank`], which is that type's stub.
///
/// There is no dense embedding gather on this plane at all — every
/// instantiation in `layout/embed_gather.slang` is `_4bit` — so this seam
/// is not "the quantised path also exists", it is the only path.
///
/// # Two points stay on the floor's default body
///
/// * `layout.split_rows` — the plain two-way divide at `width`.
///   `split_qkv.slang` writes THREE bindings from stated `q_width` and
///   `kv_width`; there is no two-way form, and firing the three-way one
///   with a zero-width third result is a launch this plane refuses at its
///   grid. A `PIE_SPLIT2` instantiation closes it.
/// * `layout.select` — one layer's `[rows, width]` slice of a
///   `[rows, layers * width]` relay, at column `layer * width`. That is a
///   STRIDED per-row copy and `row_gather.slang` is not one: it gathers
///   whole rows by index (`out[r] = input[rows[r]]`), which is the
///   sampling cut and a different arithmetic. Gemma's PLE relay is what
///   wants this, and gemma does not serve on this plane for other reasons
///   as well (`ple_combine` has no point at all).
#[kernels_macros::claims]
impl kernels::points::Layout for Ctx<'_> {
    fn embed<T: kernels::points::Scalar>(
        &self,
        ids: In<crate::points::Handle<i32>>,
        table: Const<crate::points::Handle<T>>,
        vocab: u32,
        y: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        use crate::points::Staged;

        // STATED AND UNREAD: `embed_gather_mb_4bit` takes no vocab bound and
        // does not clamp. The out-of-range id it would read is a seam of this
        // plane's shader, not of the statement, which says the number.
        let _ = vocab;
        crate::points::at_bf16::<T>("layout.embed, at an element this plane does not instantiate")?;
        // THE WIDTH IS READ OFF THE RESULT, and the declaration says so:
        // "a `Const` table carries an address and no rectangle, so the
        // width is not in the operands to read at all. The statement
        // allocates the result and the plane reads the width back off it."
        let out = y.all("the embedded row's width")?;
        // SEAM: the three planes and two numbers behind the one `Const`.
        let bank = self.bank(table)?;
        let at = affine_point(bank.group, bank.bits)?;
        let entrypoint = [
            "embed_gather_mb_4bit_bfloat16_gs_32_b_4",
            "embed_gather_mb_4bit_bfloat16_gs_32_b_8",
            "embed_gather_mb_4bit_bfloat16_gs_64_b_4",
            "embed_gather_mb_4bit_bfloat16_gs_64_b_8",
            "embed_gather_mb_4bit_bfloat16_gs_128_b_4",
            "embed_gather_mb_4bit_bfloat16_gs_128_b_8",
        ][at];
        self.fire(
            Fire::at(
                crate::routine::module_path(entrypoint, self.best()),
                entrypoint,
            )
            .apply(elementwise_rows(out.width, out.rows)?),
            &[
                bank.words.arg(),
                bank.scales.arg(),
                bank.biases.arg(),
                ids.arg(),
                y.arg(),
                out.width.arg(),
            ],
        )
    }

    /// Cut one packed `[q | k | v]` projection into three.
    ///
    /// THE ONLY CUT ON THIS PLANE THAT NEEDS NO WINDOW, and it is worth
    /// saying why while the sibling seams are open: `split_qkv.slang`
    /// binds the packed row once and the three results separately, and
    /// does its own column arithmetic from `q_width` and `kv_width`. A cut
    /// whose halves are RESULTS is expressible here; a cut whose halves
    /// are OPERANDS — every `Mlp` packed point — is not, because a
    /// descriptor has no base. See [`crate::points::Staged::window`].
    fn split_qkv<T: kernels::points::Scalar>(
        &self,
        packed: In<crate::points::Handle<T>>,
        q_width: u32,
        kv_width: u32,
        q: Out<crate::points::Handle<T>>,
        k: Out<crate::points::Handle<T>>,
        v: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "layout.split_qkv, at an element this plane does not instantiate",
        )?;
        let row = packed.all("the packed projection's row")?;
        let qw = crate::points::stated("the query width this cut states", q_width)?;
        let kw = crate::points::stated("the key/value width this cut states", kv_width)?;
        // The statement's three widths must be the row it divides. The
        // shader reads past the end otherwise, and the numbers are the
        // text's rather than the rectangle's, so this is the one place the
        // two meet.
        if qw.saturating_add(kw.saturating_mul(2)) != row.width {
            return Err(Refusal::Narrow {
                what: "the packed `[q | k | v]` row, against the widths this cut states",
                at: i64::from(row.width),
            });
        }
        self.fire(
            Fire::at(
                crate::routine::module_path("split_qkv_bf16", self.best()),
                "split_qkv_bf16",
            )
            .apply(elementwise_rows(row.width, row.rows)?),
            &[
                packed.arg(),
                q.arg(),
                k.arg(),
                v.arg(),
                q_width.arg(),
                kv_width.arg(),
            ],
        )
    }

    /// Cut an INTERLEAVED per-head `[query | gate]` row into its halves.
    ///
    /// `head_dim` is the PITCH the cut walks and not a width either half
    /// carries — the declaration says so — so the head count comes off the
    /// RESULT's width, which is `heads * head_dim`. The packed row is
    /// twice that; the two row strides the shader takes are the two
    /// rectangles' own widths.
    fn split_q_gate<T: kernels::points::Scalar>(
        &self,
        packed: In<crate::points::Handle<T>>,
        head_dim: u32,
        q: Out<crate::points::Handle<T>>,
        gate: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "layout.split_q_gate, at an element this plane does not instantiate",
        )?;
        let src = packed.all("the interleaved `[query | gate]` row")?;
        let dst = q.all("the query half this cut writes")?;
        let hd = crate::points::stated("the head width this cut walks by", head_dim)?;
        let heads = crate::points::heads("the heads this cut divides by", dst.width, hd)?;
        self.fire(
            Fire::at(
                crate::routine::module_path("q_gate_split_bfloat16", self.best()),
                "q_gate_split_bfloat16",
            )
            .apply(head_grid(hd, heads, src.rows)?),
            &[
                packed.arg(),
                q.arg(),
                gate.arg(),
                hd.arg(),
                src.width.arg(),
                dst.width.arg(),
            ],
        )
    }
}
