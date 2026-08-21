
use kernels::{Bind, Fire};
use kernels_macros::routine;
use crate::jit::{Ctx, Launch};
use crate::jit::abi::Tensor;
use crate::jit::abi::bf16;
use kernels::Refusal;
use kernels::routine::{Const, In, Out};

#[must_use]
const fn rms(rows: i32) -> Launch {
    const ARGMAX_BLOCK: u32 = 256;

    Launch::per_row(rows.unsigned_abs(), ARGMAX_BLOCK).smem((ARGMAX_BLOCK / 32) * 4)
}

#[routine]
pub fn lm_head_gemv_argmax_int8(
    ctx: &Ctx<'_>,
    hidden_states: In<Tensor<bf16>>,
    lm_head_weight: Const<Tensor<i8>>,
    scale_inv: Const<Tensor<f32>>,
    token_ids: Out<Tensor<i32>>,
    vocab: Const<i32>) -> Result<(), Refusal> {

    let vocab = *vocab;

    let num_rows = token_ids.rows;

    if token_ids.width <= 0 {
        return Err(Refusal::Empty { what: "the token id column's width" });
    }

    if hidden_states.width <= 0 {
        return Err(Refusal::Empty { what: "hidden" });
    }

    let states = hidden_states.all("hidden")?;

    const GEMV_WARPS: u32 = 8;

    fn blocks_x(ctx: &Ctx<'_>, vocab: i32) -> Result<u32, Refusal> {
    const BLOCKS_PER_SM: u32 = 2;

    let max_blocks_x = ctx.multiprocessors()? * BLOCKS_PER_SM;
    let min_blocks_x = vocab.unsigned_abs().div_ceil(GEMV_WARPS);
    Ok(max_blocks_x.min(min_blocks_x).max(1))
    }

            const SELECT_BLOCK: u32 = 128;

            const GEMV_BLOCK_DIM: u32 = GEMV_WARPS * 32;

    let num_blocks_x = blocks_x(ctx, vocab)?;
    let rows = num_rows.unsigned_abs();
    let tiles = i32::try_from(num_blocks_x).unwrap_or(i32::MAX);
    let partial_pairs = ctx.scratch(
        "sample::lm_head_argmax_pairs",
        num_blocks_x as usize * rows as usize * core::mem::size_of::<u64>(),
    )?;
    let smem = states.width.unsigned_abs().saturating_mul(4);

    ctx.fire(Fire::at("sample/argmax.cuh", "::pie::sample::lm_head_gemv_argmax_int8<::pie::bf16>").apply(Launch::grid([num_blocks_x, rows, 1], [GEMV_BLOCK_DIM, 1, 1]).smem(smem)), &[
                states.ptr.arg(),
                lm_head_weight.arg(),
                scale_inv.arg(),
                partial_pairs.arg(),
                num_rows.arg(),
                states.stride.arg(),
                vocab.arg(),
                tiles.arg(),
            ])?;
        ctx.fire(Fire::at("sample/argmax.cuh", "::pie::sample::select_lm_head_argmax_pairs").apply(Launch::flat(rows, SELECT_BLOCK)), &[partial_pairs.cast_const().arg(), token_ids.arg(), num_rows.arg(), tiles.arg()])
}

#[routine(bf16, internal)]
pub fn argmax<T>(
    ctx: &Ctx<'_>,
    logits: In<Tensor<T>>,
    out: Out<Tensor<i32>>) -> Result<(), Refusal> {

    let rows = logits.rows;
    if logits.width <= 0 {
        return Err(Refusal::Empty { what: "vocab" });
    }

    let src = logits.all("vocab")?;

    ctx.fire(Fire::at("sample/argmax.cuh", crate::jit::symbol(&format!("::pie::sample::argmax<{}>", T::CPP))).apply(rms(rows)), &[src.ptr.arg(), out.arg(), src.width.arg()])
}

#[routine(internal)]
pub fn argmax_f32(ctx: &Ctx<'_>, logits: In<Tensor<f32>>, out: Out<Tensor<i32>>) -> Result<(), Refusal> {
    if logits.width <= 0 {
        return Err(Refusal::Empty { what: "vocab" });
    }

    let src = logits.all("vocab")?;
    ctx.fire(Fire::at("sample/argmax.cuh", "::pie::sample::argmax<\
                                             ::pie::sample::f32>").apply(rms(src.rows)), &[src.ptr.arg(), out.arg(), src.width.arg()])
}

#[routine(bf16, internal)]
pub fn argmax_compact_scatter<T>(
    ctx: &Ctx<'_>,
    logits: In<Tensor<T>>,
    row_indices: In<Tensor<i32>>,
    out: Out<Tensor<i32>>) -> Result<(), Refusal> {

    let rows = logits.rows;
    if logits.width <= 0 {
        return Err(Refusal::Empty { what: "vocab" });
    }

    let src = logits.all("vocab")?;

    ctx.fire(Fire::at("sample/argmax.cuh", crate::jit::symbol(&format!("::pie::sample::argmax_compact_scatter<{}>", T::CPP))).apply(rms(rows)), &[src.ptr.arg(), row_indices.arg(), out.arg(), src.width.arg()])
}
