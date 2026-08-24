use crate::attn::head_grid;
use crate::plane::{Bind, Ctx, Fire, In, Out};
use crate::points::{Payload, at_bf16};
use kernels::plane::Refusal;
use kernels::shader::elementwise_rows;

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
}
