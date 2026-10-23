//! MLA's latent prologue (`attn/mla/prep.cu`): the kv_a split and norm, the
//! q_b split, and the latent rows written into the pool.

use crate::error::Error;

use super::rms;
use crate::attn::kv::lanes_of;
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, count, dtype_dispatch, refuse, stated};
use crate::tensor::{KvPool, Tensor};

const PREP: &str = "attn/mla/prep.cu";

pub(crate) const BLOCK: u32 = 256;

pub(crate) fn split_kv_a_norm(
    ctx: &Ctx,
    op: &'static str,
    kv_a: Tensor,
    weight: Tensor,
    eps: f32,
    kv_c: &mut Tensor,
    k_pe: &mut Tensor,
) -> Result<(), Error> {
    let t = dtype_dispatch!(op, kv_a.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let kv_lora = count(op, "the latent width this cut states", kv_c.width)?;
    let rope = stated(op, k_pe.width)?;
    let src_row_stride = stated(op, kv_a.width)?;
    if src_row_stride < kv_lora + rope {
        return Err(refuse(
            op,
            format!(
                "the {src_row_stride}-wide source row does not hold the {kv_lora}-wide \
                 latent beside the {rope}-wide rope plane"
            ),
        ));
    }
    ctx.fire(
        op,
        Fire::at(
            PREP,
            crate::jit::symbol(&format!("::pie::attn::mla::mla_latents<{t}, 256>")),
        )
        .apply(rms(kv_a.rows).programmatic(ctx.tuning().pdl)),
        &[
            kv_a.arg(),
            weight.arg(),
            kv_c.arg(),
            k_pe.arg(),
            kv_lora.arg(),
            rope.arg(),
            src_row_stride.arg(),
            eps.arg(),
            ctx.stage(),
        ],
    )
}

pub fn split_q_b(
    ctx: &Ctx,
    q_b: Tensor,
    heads: u32,
    nope_dim: u32,
    rope_dim: u32,
    q_nope: &mut Tensor,
    q_pe: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.mla_split_q_b";
    let t = dtype_dispatch!(OP, q_b.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let heads = count(OP, "the head count this cut states", heads)?;
    let nope = stated(OP, nope_dim)?;
    let rope = stated(OP, rope_dim)?;
    let width = i64::from(heads) * (i64::from(nope) + i64::from(rope));
    // Whole 16-byte runs a half: eight elements a thread, and an i64 count
    // (a prefill's cut passes i32 elements).
    if nope % 8 == 0 && rope % 8 == 0 && [q_b.ptr, q_nope.ptr, q_pe.ptr].iter().all(|p| p % 16 == 0)
    {
        let runs = i64::from(q_b.rows) * width / 8;
        let blocks = u32::try_from((runs + i64::from(BLOCK) - 1) / i64::from(BLOCK))
            .map_err(|_| refuse(OP, format!("{runs} runs overflow a launch")))?;
        return ctx.fire(
            OP,
            Fire::at(
                PREP,
                crate::jit::symbol(&format!("::pie::attn::mla::mla_split_q_b_vec8<{t}>")),
            )
            .apply(Launch::grid([blocks, 1, 1], [BLOCK, 1, 1]).programmatic(ctx.tuning().pdl)),
            &[
                q_b.arg(),
                q_nope.arg(),
                q_pe.arg(),
                ArgValue::I64(runs),
                heads.arg(),
                nope.arg(),
                rope.arg(),
                ctx.stage(),
            ],
        );
    }
    let total = i64::from(q_b.rows) * width;
    let total = i32::try_from(total).map_err(|_| {
        refuse(
            OP,
            format!("{total} split elements do not fit the kernel's int"),
        )
    })?;
    ctx.fire(
        OP,
        Fire::at(
            PREP,
            crate::jit::symbol(&format!("::pie::attn::mla::mla_split_q_b<{t}>")),
        )
        .apply(Launch::flat(total.unsigned_abs(), BLOCK)),
        &[
            q_b.arg(),
            q_nope.arg(),
            q_pe.arg(),
            total.arg(),
            heads.arg(),
            nope.arg(),
            rope.arg(),
            ctx.stage(),
        ],
    )
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
/// The latent rows `ckv` and their rotated dims `kpe` (64 a row) landed in
/// `pool` as standard fp8 rows (`attn/mla.cuh`'s `mla_kv_append_fp8`).
pub(crate) fn write_mla_fp8_rows(
    ctx: &Ctx,
    op: &'static str,
    ckv: Tensor,
    kpe: Tensor,
    indptr: Tensor,
    pool: &KvPool,
) -> Result<(), Error> {
    let rope = pool.keys.width.saturating_sub(512);
    if ckv.width != 512 || (rope > 0 && (kpe.width != rope || kpe.rows != ckv.rows)) {
        return Err(refuse(
            op,
            format!(
                "fp8 latent rows hold 512 latent and {rope} rotated dims a position, and this \
                 append brings {} x {} and {} x {}",
                ckv.rows, ckv.width, kpe.rows, kpe.width
            ),
        ));
    }
    let num_requests = lanes_of(op, indptr)?;
    ctx.fire(
        op,
        Fire::at(PREP, "::pie::attn::mla::mla_kv_append_fp8").apply(Launch::per_row(ckv.rows, 160)),
        &[
            ckv.arg(),
            kpe.arg(),
            pool.keys.arg(),
            indptr.arg(),
            pool.page_indices.arg(),
            pool.page_indptr.arg(),
            pool.last_page_lens.arg(),
            pool.row_valid.arg(),
            num_requests.arg(),
            pool.page_size.arg(),
            i32::try_from(rope).unwrap_or(0).arg(),
            ctx.stage(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn write_mla_to_pages(
    ctx: &Ctx,
    op: &'static str,
    ckv: Tensor,
    kpe: ArgValue,
    kpe_pages: ArgValue,
    indptr: Tensor,
    pool: &KvPool,
    kv_lora_rank: i32,
    rope_dim: i32,
) -> Result<(), Error> {
    const MLA_WRITE_BLOCK: u32 = 256;

    let num_requests = lanes_of(op, indptr)?;
    ctx.fire(
        op,
        Fire::at(PREP, "::pie::attn::mla::mla_kv_append")
            .apply(Launch::per_row(ckv.rows, MLA_WRITE_BLOCK)),
        &[
            ckv.arg(),
            kpe,
            pool.keys.arg(),
            kpe_pages,
            indptr.arg(),
            pool.page_indices.arg(),
            pool.page_indptr.arg(),
            pool.last_page_lens.arg(),
            pool.row_valid.arg(),
            num_requests.arg(),
            pool.page_size.arg(),
            kv_lora_rank.arg(),
            rope_dim.arg(),
            ctx.stage(),
        ],
    )
}
