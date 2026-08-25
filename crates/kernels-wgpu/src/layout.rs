use crate::attn::head_grid;
use crate::plane::{Bind, Const, Ctx, Fire, In, Out};
use crate::points::{Payload, at_bf16};
use kernels::plane::Refusal;
use kernels::shader::elementwise_rows;

const CUTS: &str = "layout/deinterleave.wgsl";

fn stated(v: u32, what: &'static str) -> Result<i32, Refusal> {
    i32::try_from(v).map_err(|_| Refusal::Wide {
        what,
        at: i64::from(v),
        max: i64::from(i32::MAX),
    })
}

fn paired(width: i32, what: &'static str) -> Result<(), Refusal> {
    if width % 2 == 0 {
        Ok(())
    } else {
        Err(Refusal::Misaligned { what })
    }
}

#[kernels_macros::claims]
impl kernels::points::Layout for Ctx<'_> {
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

    fn embed<T: kernels::points::Scalar>(
        &self,
        ids: In<Payload<i32>>,
        table: Const<Payload<T>>,
        vocab: u32,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("`layout.embed`, at an element this plane does not stamp")?;
        let vocab = stated(vocab, "the row count this embedding table states")?;
        if vocab <= 0 {
            return Err(Refusal::Empty {
                what: "the row count this embedding table states",
            });
        }
        if ids.rows != y.rows {
            return Err(Refusal::Narrow {
                what: "the token ids handed over, against the rows this gather lands",
                at: i64::from(ids.rows),
            });
        }
        paired(
            y.width,
            "the embedded row, which this plane gathers one whole word at a time",
        )?;
        self.fire(
            Fire::at("layout/embed.wgsl", "embed_bfloat16")
                .apply(elementwise_rows(y.width, y.rows)?),
            &[ids.arg(), table.arg(), y.arg(), y.width.arg(), vocab.arg()],
        )
    }

    fn split_rows<T: kernels::points::Scalar>(
        &self,
        x: In<Payload<T>>,
        width: u32,
        left: Out<Payload<T>>,
        right: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("`layout.split_rows`, at an element this plane does not stamp")?;
        let width = stated(width, "the column this cut falls at")?;
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
        paired(
            left.width,
            "the column this cut falls at, which this plane cuts between words",
        )?;
        paired(
            x.width,
            "the packed row, which this plane cuts one whole word at a time",
        )?;
        self.fire(
            Fire::at(CUTS, "split_rows_bfloat16").apply(elementwise_rows(x.width, x.rows)?),
            &[
                x.arg(),
                left.arg(),
                right.arg(),
                left.width.arg(),
                right.width.arg(),
            ],
        )
    }

    fn select<T: kernels::points::Scalar>(
        &self,
        table: In<Payload<T>>,
        layer: u32,
        width: u32,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("`layout.select`, at an element this plane does not stamp")?;
        let width = stated(width, "the slice width this select states")?;
        if y.width != width {
            return Err(Refusal::Narrow {
                what: "the selected slice is not the width the statement states",
                at: i64::from(y.width),
            });
        }
        let offset = stated(layer, "the layer this select names")?
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
        paired(
            width,
            "the slice width this select states, which this plane copies one \
             whole word at a time",
        )?;
        paired(
            table.width,
            "the relayed row, whose pitch is where this slice's words start",
        )?;
        self.fire(
            Fire::at(CUTS, "select_slice_bfloat16").apply(elementwise_rows(y.width, y.rows)?),
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
