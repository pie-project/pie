use kernels::Grid;
use kernels::plane::Refusal;
use kernels::points::Scalar;

use crate::plane::{Bind, Const, Ctx, Fire, In, Out, Tensor, bf16, elementwise_rows};
use crate::points::{self, Handle};

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

#[allow(clippy::too_many_arguments)]
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
        let packed = points::input::<T, bf16>(packed, WHAT)?;
        self.fire(
            Fire::at("attn/split_qkv.metal", "split_qkv_bf16").apply(Grid::of(
                elementwise_rows(packed.width, packed.rows)?,
                [256, 1, 1],
            )),
            &[
                packed.arg(),
                points::result::<T, bf16>(q, WHAT)?.arg(),
                points::result::<T, bf16>(k, WHAT)?.arg(),
                points::result::<T, bf16>(v, WHAT)?.arg(),
                q_width.arg(),
                kv_width.arg(),
            ],
        )
    }

    fn split_q_gate<T: Scalar>(
        &self,
        packed: In<Handle<T>>,
        head_dim: u32,
        q: Out<Handle<T>>,
        gate: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`layout.split_q_gate`, at an element this plane does not stamp";
        let head_dim = points::stated(head_dim, "the head width this cut walks")?;
        if head_dim <= 0 {
            return Err(Refusal::Empty {
                what: "the head width this cut walks",
            });
        }
        let packed = points::input::<T, bf16>(packed, WHAT)?;
        let q = points::result::<T, bf16>(q, WHAT)?;
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
                points::result::<T, bf16>(gate, WHAT)?.arg(),
                head_dim.arg(),
                packed.width.arg(),
                q.width.arg(),
            ],
        )
    }

    fn embed<T: Scalar>(
        &self,
        ids: In<Handle<i32>>,
        table: Const<Handle<T>>,
        vocab: u32,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`layout.embed`, at an element this plane does not stamp";
        let vocab = points::stated(vocab, "the row count this embedding table states")?;
        if vocab <= 0 {
            return Err(Refusal::Empty {
                what: "the row count this embedding table states",
            });
        }
        let ids = points::input::<i32, i32>(ids, "`layout.embed`'s token stream")?;
        let y = points::result::<T, bf16>(y, WHAT)?;
        if ids.rows != y.rows {
            return Err(Refusal::Narrow {
                what: "the token ids handed over, against the rows this gather lands",
                at: i64::from(ids.rows),
            });
        }
        self.fire(
            Fire::at("layout/embed.metal", "embed_bfloat16")
                .apply(Grid::of(elementwise_rows(y.width, y.rows)?, [256, 1, 1])),
            &[
                ids.ptr.arg(),
                points::weight::<T, bf16>(table, WHAT)?.arg(),
                y.arg(),
                y.width.arg(),
                vocab.arg(),
            ],
        )
    }

    fn split_rows<T: Scalar>(
        &self,
        x: In<Handle<T>>,
        width: u32,
        left: Out<Handle<T>>,
        right: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`layout.split_rows`, at an element this plane does not stamp";
        let width = points::stated(width, "the column this cut falls at")?;
        let x = points::input::<T, bf16>(x, WHAT)?;
        let left = points::result::<T, bf16>(left, WHAT)?;
        let right = points::result::<T, bf16>(right, WHAT)?;
        if left.width != width {
            return Err(Refusal::Narrow {
                what: "the left half is not the width this cut states",
                at: i64::from(left.width),
            });
        }
        if left.width <= 0 || right.width <= 0 {
            return Err(Refusal::Empty {
                what: "a half of this cut",
            });
        }
        if left.width.checked_add(right.width) != Some(x.width) {
            return Err(Refusal::Narrow {
                what: "the two halves do not cover the packed row",
                at: i64::from(x.width),
            });
        }
        self.fire(
            Fire::at("layout/deinterleave.metal", "split_rows_bfloat16")
                .apply(Grid::of(elementwise_rows(x.width, x.rows)?, [256, 1, 1])),
            &[
                x.arg(),
                left.arg(),
                right.arg(),
                left.width.arg(),
                right.width.arg(),
            ],
        )
    }

    fn select<T: Scalar>(
        &self,
        table: In<Handle<T>>,
        layer: u32,
        width: u32,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`layout.select`, at an element this plane does not stamp";
        let width = points::stated(width, "the slice width this select states")?;
        let table = points::input::<T, bf16>(table, WHAT)?;
        let y = points::result::<T, bf16>(y, WHAT)?;
        if y.width != width {
            return Err(Refusal::Narrow {
                what: "the selected slice is not the width the statement states",
                at: i64::from(y.width),
            });
        }
        let offset = points::stated(layer, "the layer this select names")?
            .checked_mul(width)
            .ok_or(Refusal::Wide {
                what: "the column this layer's slice starts at",
                at: i64::from(layer) * i64::from(width),
                max: i64::from(i32::MAX),
            })?;
        if offset
            .checked_add(width)
            .is_none_or(|end| end > table.width)
        {
            return Err(Refusal::Narrow {
                what: "the relayed row does not reach this layer's slice",
                at: i64::from(table.width),
            });
        }
        self.fire(
            Fire::at("layout/deinterleave.metal", "select_slice_bfloat16")
                .apply(Grid::of(elementwise_rows(y.width, y.rows)?, [256, 1, 1])),
            &[
                table.arg(),
                y.arg(),
                table.width.arg(),
                offset.arg(),
                width.arg(),
            ],
        )
    }
}

#[kernels_macros::claims]
impl kernels::points::Gemm for Ctx<'_> {
    fn matmul<T: Scalar>(
        &self,
        act: In<Handle<T>>,
        w: Const<Handle<T>>,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`gemm.matmul`, at an element this plane does not stamp";
        crate::gemm::act_x_wt(
            self,
            points::input::<T, bf16>(act, WHAT)?,
            points::weight::<T, bf16>(w, WHAT)?,
            points::result::<T, bf16>(y, WHAT)?,
        )
    }

    fn lm_head<T: Scalar>(
        &self,
        act: In<Handle<T>>,
        w: Const<Handle<T>>,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        self.matmul(act, w, y)
    }

    fn attention_landing<T: Scalar>(
        &self,
        act: In<Handle<T>>,
        w: Const<Handle<T>>,
        layer: u32,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        let _ = layer;
        self.matmul(act, w, y)
    }
}
