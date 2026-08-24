use kernels_macros::routine;

use crate::attn::head_grid;
use crate::points::{Payload, at_bf16};
use crate::routine::{Bind, Const, Ctx, Fire, In, Out, Tensor, bf16};
use kernels::routine::Refusal;
use kernels::shader::{elementwise, elementwise_rows};

/// The `Layout` family, claimed. Two of five points land, and the three that
/// do not split into two causes.
///
/// # `layout.embed` — the `Bank<R: Repr>` seam, and this plane feels it first
///
/// The point declares ONE weight: `table: Const<Self::Tensor<T>>`, an
/// embedding table of the same element the row it gathers rides. There is no
/// such table on this plane. `layout/embed_gather.wgsl` gathers out of an
/// AFFINE-QUANTISED bank — `w: array<u32>` packed 4 or 8 bits to the
/// element, `scales` and `biases` one per group of 32/64/128 — and the six
/// entrypoints are the group-size × bit-width cross. Every embedding this
/// backend has ever served came out of that bank; nothing here reads a dense
/// `bf16` table at all.
///
/// So the absence is not "wgpu has no embed", it is "the declaration has no
/// way to name what wgpu embeds from". `.wiki/baker.md` already names the
/// type — `Bank<R: Repr>`, "quantized banks: blocks+scales, NOT a second
/// Elem" — and baker-todo queues it behind `moe.matmul_select_bias`. THIS
/// PLANE NEEDS IT SOONER AND WIDER: see `quant.rs`, where the entire `Gemm`
/// family fails the same way.
///
/// **SEAM (floor):** `Bank<R: Repr>`, and see `quant.rs` for the shape this
/// plane measures for it.
///
/// One further number the floor would owe: the group size and bit width pick
/// the ENTRYPOINT (`affine_point` below turns them into an index), so
/// whatever `Bank` becomes has to carry both as facts a body can read, not
/// as scalars a statement restates.
///
/// # `layout.split_rows` and `layout.select` — missing shaders
///
/// `split_rows` is the plain two-way cut. `attn/split_qkv.wgsl` is a
/// three-way one and its three destinations are three declared bindings, so
/// it cannot serve a two-way statement: a bind group with an unfilled slot
/// is a layout `driver-wgpu` cannot build. `select` is gemma's per-layer PLE
/// slice, which W3 wrote for cuda and no shader plane has yet.
#[kernels_macros::claims]
impl kernels::points::Layout for Ctx<'_> {
    /// The fused QKV projection cut three ways.
    ///
    /// `q_width` and `kv_width` are the two the statement carries and the
    /// shader reads both out of its uniform; `v` is `kv_width` wide too and
    /// the declaration says so by giving it no number of its own. The grid is
    /// the PACKED rectangle — one lane per element of the source, not of any
    /// destination — because the cut is a walk over what is being cut.
    fn split_qkv<T: kernels::points::Scalar>(
        &self,
        packed: In<Payload<T>>,
        q_width: u32,
        kv_width: u32,
        q: Out<Payload<T>>,
        k: Out<Payload<T>>,
        v: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("layout.split_qkv at an element other than bf16")?;
        self.fire(
            Fire::at("attn/split_qkv.wgsl", "split_qkv_bf16")
                .apply(elementwise_rows(packed.width, packed.rows)?),
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

    /// qwen3.5's interleaved `[query | gate]` row, cut per head.
    ///
    /// # The query-head count is read off the grid, and that is the shader's
    ///
    /// `q_gate_split_bfloat16` takes `head_dim` and two row pitches in its
    /// uniform and then reads `let n_q = groups.y;` — "the y extent is the
    /// query-head count, which nothing hands this kernel as a scalar; the
    /// grid IS the statement". The routine layer restated it as
    /// `q_heads: Const<i32>` so the plan could place a number the kernel
    /// takes off the launch. The point states neither, and it does not need
    /// to: the packed row holds `2 * head_dim` per head, so the count is
    /// `packed.width / (2 * head_dim)` and a `Const` restating it could
    /// disagree with the rectangle it divides. Read, not stated — the
    /// `ssm.gdn_prep` rule (W10) applied to a different packing.
    ///
    /// The two pitches are the operands' own widths, which is what a dense
    /// arena rectangle makes them.
    fn split_q_gate<T: kernels::points::Scalar>(
        &self,
        packed: In<Payload<T>>,
        head_dim: u32,
        q: Out<Payload<T>>,
        gate: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("layout.split_q_gate at an element other than bf16")?;
        let head_dim = i32::try_from(head_dim).map_err(|_| Refusal::Wide {
            what: "the head width this split states",
            at: i64::from(head_dim),
            max: i64::from(i32::MAX),
        })?;
        let pair = head_dim.checked_mul(2).ok_or(Refusal::Wide {
            what: "the interleaved `[query | gate]` head",
            at: i64::from(head_dim) * 2,
            max: i64::from(i32::MAX),
        })?;
        if pair <= 0 || packed.width % pair != 0 {
            return Err(Refusal::Narrow {
                what: "the packed `[query | gate]` row, which divides into \
                       `2 * head_dim` per head",
                at: i64::from(packed.width),
            });
        }
        let q_heads = packed.width / pair;
        self.fire(
            Fire::at("attn/gate.wgsl", "q_gate_split_bfloat16").apply(head_grid(
                head_dim,
                q_heads,
                packed.rows,
            )?),
            &[
                packed.arg(),
                q.arg(),
                gate.arg(),
                head_dim.arg(),
                packed.width.arg(),
                q.width.arg(),
            ],
        )
    }
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
    if width <= 0 {
        return Err(Refusal::Empty { what: "width" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }

    let lanes = width.unsigned_abs() * rows.unsigned_abs();
    ctx.fire(
        Fire::at("layout/ple_combine.wgsl", "ple_combine_bfloat16").apply([lanes, 1, 1]),
        &[proj.arg(), token.arg(), out.arg(), inv_sqrt2.arg()],
    )
}

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
    let rows = 1;
    let lanes = elementwise(hidden, rows)?;
    ctx.fire(
        Fire::at(
            "layout/embed_gather.wgsl",
            [
                "embed_gather_4bit_bfloat16_gs_32_b_4",
                "embed_gather_4bit_bfloat16_gs_32_b_8",
                "embed_gather_4bit_bfloat16_gs_64_b_4",
                "embed_gather_4bit_bfloat16_gs_64_b_8",
                "embed_gather_4bit_bfloat16_gs_128_b_4",
                "embed_gather_4bit_bfloat16_gs_128_b_8",
            ][affine_point(*group, *bits)?],
        )
        .apply(lanes),
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
    let lanes = elementwise_rows(hidden, rows)?;
    ctx.fire(
        Fire::at(
            "layout/embed_gather.wgsl",
            [
                "embed_gather_mb_4bit_bfloat16_gs_32_b_4",
                "embed_gather_mb_4bit_bfloat16_gs_32_b_8",
                "embed_gather_mb_4bit_bfloat16_gs_64_b_4",
                "embed_gather_mb_4bit_bfloat16_gs_64_b_8",
                "embed_gather_mb_4bit_bfloat16_gs_128_b_4",
                "embed_gather_mb_4bit_bfloat16_gs_128_b_8",
            ][affine_point(*group, *bits)?],
        )
        .apply(lanes),
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
    let lanes = elementwise(hidden, 1)?;
    ctx.fire(
        Fire::at(
            "layout/embed_gather.wgsl",
            [
                "embed_gather_scaled_4bit_bfloat16_gs_32_b_4",
                "embed_gather_scaled_4bit_bfloat16_gs_32_b_8",
                "embed_gather_scaled_4bit_bfloat16_gs_64_b_4",
                "embed_gather_scaled_4bit_bfloat16_gs_64_b_8",
                "embed_gather_scaled_4bit_bfloat16_gs_128_b_4",
                "embed_gather_scaled_4bit_bfloat16_gs_128_b_8",
            ][affine_point(*group, *bits)?],
        )
        .apply(lanes),
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
    let lanes = elementwise_rows(hidden, rows)?;
    ctx.fire(
        Fire::at(
            "layout/embed_gather.wgsl",
            [
                "embed_gather_scaled_mb_4bit_bfloat16_gs_32_b_4",
                "embed_gather_scaled_mb_4bit_bfloat16_gs_32_b_8",
                "embed_gather_scaled_mb_4bit_bfloat16_gs_64_b_4",
                "embed_gather_scaled_mb_4bit_bfloat16_gs_64_b_8",
                "embed_gather_scaled_mb_4bit_bfloat16_gs_128_b_4",
                "embed_gather_scaled_mb_4bit_bfloat16_gs_128_b_8",
            ][affine_point(*group, *bits)?],
        )
        .apply(lanes),
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
    let lanes = elementwise_rows(input.width, row_count)?;
    ctx.fire(
        Fire::at("layout/row_gather.wgsl", "row_gather_bfloat16").apply(lanes),
        &[input.arg(), out.arg(), rows.arg(), width.arg(), count.arg()],
    )
}
