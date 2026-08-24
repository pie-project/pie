use crate::plane::{Bind, Const, Ctx, Fire, In, Out};
use kernels::plane::Refusal;
use kernels::shader::elementwise_rows;

pub fn affine_point(group: i32, bits: i32) -> Result<usize, Refusal> {
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

        let _ = vocab;
        crate::points::at_bf16::<T>("layout.embed, at an element this plane does not instantiate")?;

        let out = y.all("the embedded row's width")?;

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
                crate::plane::module_path(entrypoint, self.best()),
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

        if qw.saturating_add(kw.saturating_mul(2)) != row.width {
            return Err(Refusal::Narrow {
                what: "the packed `[q | k | v]` row, against the widths this cut states",
                at: i64::from(row.width),
            });
        }
        self.fire(
            Fire::at(
                crate::plane::module_path("split_qkv_bf16", self.best()),
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
                crate::plane::module_path("q_gate_split_bfloat16", self.best()),
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
