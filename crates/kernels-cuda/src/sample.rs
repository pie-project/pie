#![allow(clippy::too_many_arguments)]

use crate::jit::{Ctx, Family, Launch, Routine};
use crate::routine;
use crate::jit::Abi;
use crate::jit::abi::Elem;
use crate::jit::abi::bf16;
use kernels::Refusal;

/// `LaunchRule::Rms`, as the expression it evaluates to.
#[must_use]
const fn rms(rows: i32) -> Launch {
    /// Threads per block of the three plain argmaxes — `LaunchRule::Rms`' 256,
    const ARGMAX_BLOCK: u32 = 256;

    Launch::per_row(rows.unsigned_abs(), ARGMAX_BLOCK).smem((ARGMAX_BLOCK / 32) * 4)
}
/// Greedy decode straight off an int8 LM head: `token_ids[r] = argmax_v
///
/// Two launches, ordered by the stream: the GEMV writes one partial pair per
/// (block, row), and the selector folds them.
///
/// What the caller must guarantee, as `call()` states it: every pointer
/// addresses live device memory of the extents `num_rows`, `hidden` and
/// `vocab` describe, and `token_ids` is writable for `num_rows` i32.
pub fn lm_head_gemv_argmax_int8(
    ctx: &Ctx,
    hidden_states: *const bf16,
    lm_head_weight: *const i8,
    scale_inv: *const f32,
    token_ids: *mut i32,
    num_rows: i32,
    hidden: i32,
    vocab: i32,
) -> Result<(), Refusal> {
    /// Vocab rows a block covers per grid step — `sample/argmax.cuh:153`.
    const GEMV_WARPS: u32 = 8;

    /// `grid.x` for the fused GEMV — `argmax.cu:101-107`, transcribed.
    ///
    /// # Errors
    ///
    /// [`Refusal::Device`] if the SM count cannot be read. There is no default:
    /// the number is the grid AND the operand the kernel strides the vocab by.
    fn blocks_x(ctx: &Ctx, vocab: i32) -> Result<u32, Refusal> {
    /// Resident blocks per SM the launcher aims for — `argmax.cu:103`'s
    const BLOCKS_PER_SM: u32 = 2;

    let max_blocks_x = ctx.multiprocessors()? * BLOCKS_PER_SM;
    let min_blocks_x = vocab.unsigned_abs().div_ceil(GEMV_WARPS);
    Ok(max_blocks_x.min(min_blocks_x).max(1))
    }

            /// Threads per block of the selector — `argmax.cu:134`'s `dim3
            const SELECT_BLOCK: u32 = 128;

            /// Threads per block of the fused GEMV — `sample/argmax.cuh:153-154`'s
            const GEMV_BLOCK_DIM: u32 = GEMV_WARPS * 32;

    let num_blocks_x = blocks_x(ctx, vocab)?;
    let rows = num_rows.unsigned_abs();
    let tiles = i32::try_from(num_blocks_x).unwrap_or(i32::MAX);
    let partial_pairs = ctx.scratch(
        "sample::lm_head_argmax_pairs",
        num_blocks_x as usize * rows as usize * core::mem::size_of::<u64>(),
    )?;
    let smem = hidden.unsigned_abs().saturating_mul(4);

    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as. The two launches are
    // ordered by the stream, and the second reads what the first wrote.
    unsafe {
        ctx.launch(
            "sample/argmax.cuh",
            "::pie::sample::lm_head_gemv_argmax_int8<::pie::bf16>",
            Launch::grid([num_blocks_x, rows, 1], [GEMV_BLOCK_DIM, 1, 1]).smem(smem),
            &[
                hidden_states.arg(),
                lm_head_weight.arg(),
                scale_inv.arg(),
                partial_pairs.arg(),
                num_rows.arg(),
                hidden.arg(),
                vocab.arg(),
                tiles.arg(),
            ],
        )?;
        ctx.launch(
            "sample/argmax.cuh",
            "::pie::sample::select_lm_head_argmax_pairs",
            Launch::flat(rows, SELECT_BLOCK),
            &[partial_pairs.cast_const().arg(), token_ids.arg(), num_rows.arg(), tiles.arg()],
        )
    }
}

/// `sample::argmax_bf16` — the greedy decode over a bf16 logit row.
///
/// What the caller must guarantee, as `call()` states it:
///
/// `logits` must address `rows * vocab` live bf16 elements and `out` `rows`
/// writable `i32`s.
pub fn argmax<T>(
    ctx: &Ctx,
    logits: *const T,
    out: *mut i32,
    rows: i32,
    vocab: i32,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
{
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch("sample/argmax.cuh", &format!("::pie::sample::argmax<{}>", T::CPP), rms(rows), &[logits.arg(), out.arg(), vocab.arg()])
    }
}

/// `sample::argmax_f32` — the fp32 twin.
///
/// [`argmax`]'s obligation, with `f32` for `bf16`.
pub fn argmax_f32(
    ctx: &Ctx,
    logits: *const f32,
    out: *mut i32,
    rows: i32,
    vocab: i32,
) -> Result<(), Refusal> {
    // SAFETY: as [`argmax`]'s.
    unsafe {
        ctx.launch("sample/argmax.cuh", "::pie::sample::argmax<\
                                             ::pie::sample::f32>", rms(rows), &[logits.arg(), out.arg(), vocab.arg()])
    }
}

/// `sample::argmax_compact_scatter_bf16` — the compact form.
///
/// What the caller must guarantee, as `call()` states it:
///
/// `logits` must address `rows * vocab` live bf16 elements, `row_indices`
/// `rows` live `i32`s, and `out` must be writable at every index
/// `row_indices` names.
pub fn argmax_compact_scatter<T>(
    ctx: &Ctx,
    logits: *const T,
    row_indices: *const i32,
    out: *mut i32,
    rows: i32,
    vocab: i32,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
{
    // SAFETY: as [`argmax`]'s, and `out` must be writable at every index
    // `row_indices` names.
    unsafe {
        ctx.launch(
            "sample/argmax.cuh",
            &format!("::pie::sample::argmax_compact_scatter<{}>", T::CPP),
            rms(rows),
            &[logits.arg(), row_indices.arg(), out.arg(), vocab.arg()],
        )
    }
}

/// This family's routines, and what a trace may say about each.
pub static ROUTINES: &[Routine] = &[
    routine!(lm_head_gemv_argmax_int8),
    routine!(argmax_bf16 = argmax::<bf16>),
    routine!(argmax_f32),
    routine!(argmax_compact_scatter_bf16 = argmax_compact_scatter::<bf16>),
];

/// `sample`, as a trace names it.
pub static FAMILY: Family = crate::family!(ROUTINES);
