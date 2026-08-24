use kernels::Grid;
use kernels::points::Scalar;
use kernels::routine::Refusal;
use kernels_macros::routine;

use crate::plane::{self, Handle};
use crate::routine::{
    Bind, Const, Ctx, Fire, In, Out, Tensor, bf16, elementwise, elementwise_rows,
};

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
            "layout/embed_gather.metal",
            [
                "embed_gather_4bit_bfloat16_gs_32_b_4",
                "embed_gather_4bit_bfloat16_gs_32_b_8",
                "embed_gather_4bit_bfloat16_gs_64_b_4",
                "embed_gather_4bit_bfloat16_gs_64_b_8",
                "embed_gather_4bit_bfloat16_gs_128_b_4",
                "embed_gather_4bit_bfloat16_gs_128_b_8",
            ][affine_point(*group, *bits)?],
        )
        .apply(Grid::of(elementwise(hidden, 1)?, [256, 1, 1])),
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
    ctx.fire(
        Fire::at(
            "layout/embed_gather.metal",
            [
                "embed_gather_mb_4bit_bfloat16_gs_32_b_4",
                "embed_gather_mb_4bit_bfloat16_gs_32_b_8",
                "embed_gather_mb_4bit_bfloat16_gs_64_b_4",
                "embed_gather_mb_4bit_bfloat16_gs_64_b_8",
                "embed_gather_mb_4bit_bfloat16_gs_128_b_4",
                "embed_gather_mb_4bit_bfloat16_gs_128_b_8",
            ][affine_point(*group, *bits)?],
        )
        .apply(Grid::of(elementwise_rows(hidden, rows)?, [256, 1, 1])),
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
            "layout/embed_gather.metal",
            [
                "embed_gather_scaled_4bit_bfloat16_gs_32_b_4",
                "embed_gather_scaled_4bit_bfloat16_gs_32_b_8",
                "embed_gather_scaled_4bit_bfloat16_gs_64_b_4",
                "embed_gather_scaled_4bit_bfloat16_gs_64_b_8",
                "embed_gather_scaled_4bit_bfloat16_gs_128_b_4",
                "embed_gather_scaled_4bit_bfloat16_gs_128_b_8",
            ][affine_point(*group, *bits)?],
        )
        .apply(Grid::of(elementwise(hidden, 1)?, [256, 1, 1])),
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
            "layout/embed_gather.metal",
            [
                "embed_gather_scaled_mb_4bit_bfloat16_gs_32_b_4",
                "embed_gather_scaled_mb_4bit_bfloat16_gs_32_b_8",
                "embed_gather_scaled_mb_4bit_bfloat16_gs_64_b_4",
                "embed_gather_scaled_mb_4bit_bfloat16_gs_64_b_8",
                "embed_gather_scaled_mb_4bit_bfloat16_gs_128_b_4",
                "embed_gather_scaled_mb_4bit_bfloat16_gs_128_b_8",
            ][affine_point(*group, *bits)?],
        )
        .apply(Grid::of(elementwise_rows(hidden, rows)?, [256, 1, 1])),
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
        Fire::at("layout/ple_combine.metal", "ple_combine_bfloat16")
            .apply(Grid::of(elementwise(width, rows)?, [256, 1, 1])),
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
        Fire::at("layout/row_gather.metal", "row_gather_bfloat16").apply(Grid::of(
            elementwise_rows(input.width, row_count)?,
            [256, 1, 1],
        )),
        &[input.arg(), out.arg(), rows.arg(), width.arg(), count.arg()],
    )
}

/// The `Layout` family, claimed — and both bodies fire out of `attn/`, which
/// is where their kernels have always lived. A family is ONE impl block and
/// its points may fire out of two shader directories; cuda's `Mla` reaches
/// into `gemm/` the same way.
///
/// Three points stay on the floor's default body:
///
/// * `layout.embed` — SEAM: THE ONE GATHER THIS PLANE HAS IS QUANTIZED.
///   Every `embed_gather*` arm above takes a `[vocab, hidden]` bank as three
///   operands (packed 4-bit words, per-group scales, per-group biases) plus
///   the group size and the bit width that say how to decode them, where the
///   declaration states ONE `Const<Tensor<T>>` and no scalars. That is the
///   `Bank<R: Repr>` payload the floor does not carry yet, so the point
///   cannot be claimed without either lying about the operand list or writing
///   a dense gather no text on this plane would call. The
///   `#[routine(canon = "layout.embed")]` on `embed_gather_mb_4bit` keeps
///   answering, which is what claim-only means.
/// * `layout.split_rows` — SEAM: the plain two-way divide, and no `.metal`
///   kernel cuts a row at a stated width. The three cuts this plane has are
///   the qkv one, the interleaved q/gate one and quant's own, and none of the
///   three takes a single boundary and two dense halves.
/// * `layout.select` — SEAM: gemma's per-layer PLE slice, which is
///   `y[m, ..] = table[m, layer * width ..]` — a strided per-row COLUMN copy.
///   `row_gather` above gathers whole ROWS by an index plane, which is a
///   different arithmetic; cuda wrote `layout/deinterleave.cuh`'s slice for
///   this point in W3 and this tree has no counterpart.
#[kernels_macros::claims]
impl kernels::points::Layout for Ctx<'_> {
    fn split_qkv<T: Scalar>(
        &self,
        packed: In<Handle<T>>,
        q_width: u32,
        kv_width: u32,
        q: Out<Handle<T>>,
        k: Out<Handle<T>>,
        v: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`layout.split_qkv`, at an element this plane does not stamp";
        let packed = plane::input::<T, bf16>(packed, WHAT)?;
        self.fire(
            Fire::at("attn/split_qkv.metal", "split_qkv_bf16").apply(Grid::of(
                elementwise_rows(packed.width, packed.rows)?,
                [256, 1, 1],
            )),
            &[
                packed.arg(),
                plane::result::<T, bf16>(q, WHAT)?.arg(),
                plane::result::<T, bf16>(k, WHAT)?.arg(),
                plane::result::<T, bf16>(v, WHAT)?.arg(),
                q_width.arg(),
                kv_width.arg(),
            ],
        )
    }

    /// The interleaved cut, and the three numbers it derives are all the
    /// operands' own. `head_dim` is the pitch the declaration states; the
    /// two row strides are the packed row and the half row, which is what a
    /// DENSE rectangle's pitch is on either side; and the head count is the
    /// half row over the stated pitch — the shader reads it back as
    /// `grid.y`, so the grid is where it has to be right.
    fn split_q_gate<T: Scalar>(
        &self,
        packed: In<Handle<T>>,
        head_dim: u32,
        q: Out<Handle<T>>,
        gate: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`layout.split_q_gate`, at an element this plane does not stamp";
        let head_dim = plane::stated(head_dim, "the head width this cut walks")?;
        if head_dim <= 0 {
            return Err(Refusal::Empty {
                what: "the head width this cut walks",
            });
        }
        let packed = plane::input::<T, bf16>(packed, WHAT)?;
        let q = plane::result::<T, bf16>(q, WHAT)?;
        if q.width <= 0 || q.width % head_dim != 0 {
            return Err(Refusal::Narrow {
                what: "the query half does not divide by the head width this cut states",
                at: i64::from(q.width),
            });
        }
        let lanes = crate::attn::head_grid(head_dim, q.width / head_dim, packed.rows)?;
        self.fire(
            Fire::at("attn/gate.metal", "q_gate_split_bfloat16")
                .apply(Grid::of(lanes, crate::attn::head_group(lanes))),
            &[
                packed.arg(),
                q.arg(),
                plane::result::<T, bf16>(gate, WHAT)?.arg(),
                head_dim.arg(),
                packed.width.arg(),
                q.width.arg(),
            ],
        )
    }
}

/// The `Gemm` family, implemented and claiming nothing.
///
/// EVERY MATMUL ON THIS PLANE IS QUANTIZED, and that is the whole of it.
/// `gemm.matmul`, `gemm.lm_head` and `gemm.attention_landing` each state one
/// `Const<Self::Tensor<T>>` — one dense bank at the activation's element —
/// and the `.metal` tree stamps no such kernel: `quant/qmm_t.metal` and
/// `quant/qmv.metal` are affine (packed words, per-group scales, per-group
/// biases, a group size and a bit width), `moe/route.metal`'s routed arms are
/// the same three operands with a permutation, and `third_party/mlx/steel_*`
/// are the tiles those two are built out of rather than entrypoints of their
/// own.
///
/// SEAM: `Bank<R: Repr>`, the floor payload `.wiki/baker.md` names beside
/// `Tensor<T>` and the floor does not carry yet. It is the same gap
/// `layout.embed` above and `moe.matmul_select*` next door are stated
/// against, and it is one gap and not three — when a point can state a
/// quantized bank, all six land together and every SKU's `gemm.matmul` row
/// with them. Writing a dense bf16 GEMM instead would be a kernel no text on
/// this plane calls.
#[kernels_macros::claims]
impl kernels::points::Gemm for Ctx<'_> {}
