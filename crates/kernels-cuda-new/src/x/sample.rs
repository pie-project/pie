#![allow(clippy::too_many_arguments)]

use crate::jit::{Ctx, Family, Launch, Root, Routine};
use crate::routine;
use crate::x::Abi;
use crate::x::abi::bf16;
use kernels::Refusal;

/// `sample/argmax.cuh` — the root every routine here compiles a symbol out of.
pub static ROOT: Root = Root::new(
    "sample/argmax",
    include_str!("../../csrc/src/sample/argmax.cuh"),
    "sample/argmax.cuh",
);

/// The template-ids NVRTC is handed, spelled as it is handed them.
mod inst {
    /// `argmax.cuh:219` — the greedy decode over bf16 logits.
    pub const ARGMAX_BF16: &str = "::pie_cuda_driver::kernels::sample::device::argmax\
         <::pie_cuda_driver::kernels::device::bf16>";
    /// The fp32 twin.
    pub const ARGMAX_F32: &str = "::pie_cuda_driver::kernels::sample::device::argmax\
         <::pie_cuda_driver::kernels::sample::device::f32>";
    /// `argmax.cuh:248` — the compact scatter form.
    pub const ARGMAX_COMPACT_SCATTER: &str = "::pie_cuda_driver::kernels::sample::device::argmax_compact_scatter\
         <::pie_cuda_driver::kernels::device::bf16>";
    /// `argmax.cuh:670` — the fused int8 LM-head GEMV.
    pub const LM_HEAD_GEMV: &str = "::pie_cuda_driver::kernels::sample::device::lm_head_gemv_argmax_int8\
         <::pie_cuda_driver::kernels::device::bf16>";
    /// `argmax.cuh:546` — its second half, folding the partial pairs.
    pub const SELECT_PAIRS: &str =
        "::pie_cuda_driver::kernels::sample::device::select_lm_head_argmax_pairs";
}

/// Vocab rows a block covers per grid step — `sample/argmax.cuh:153`.
const GEMV_WARPS: u32 = 8;

/// Threads per block of the fused GEMV — `sample/argmax.cuh:153-154`'s
const GEMV_BLOCK_DIM: u32 = GEMV_WARPS * 32;

/// Resident blocks per SM the launcher aims for — `argmax.cu:103`'s
const BLOCKS_PER_SM: u32 = 2;

/// Threads per block of the selector — `argmax.cu:134`'s `dim3
const SELECT_BLOCK: u32 = 128;

/// Threads per block of the three plain argmaxes — `LaunchRule::Rms`' 256,
const ARGMAX_BLOCK: u32 = 256;

/// `LaunchRule::Rms`, as the expression it evaluates to.
#[must_use]
const fn rms(rows: i32) -> Launch {
    Launch::per_row(rows.unsigned_abs(), ARGMAX_BLOCK).smem((ARGMAX_BLOCK / 32) * 4)
}

/// `grid.x` for the fused GEMV — `argmax.cu:101-107`, transcribed.
///
/// # Errors
///
/// [`Refusal::Device`] if the SM count cannot be read. There is no default:
/// the number is the grid AND the operand the kernel strides the vocab by.
fn blocks_x(ctx: &Ctx, vocab: i32) -> Result<u32, Refusal> {
    let max_blocks_x = ctx.multiprocessors()? * BLOCKS_PER_SM;
    let min_blocks_x = vocab.unsigned_abs().div_ceil(GEMV_WARPS);
    Ok(max_blocks_x.min(min_blocks_x).max(1))
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
    if num_rows <= 0 {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    if hidden <= 0 {
        return Err(Refusal::Empty { what: "hidden" });
    }
    if vocab <= 0 {
        return Err(Refusal::Empty { what: "vocab" });
    }

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
            &ROOT,
            inst::LM_HEAD_GEMV,
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
            &ROOT,
            inst::SELECT_PAIRS,
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
pub fn argmax_bf16(
    ctx: &Ctx,
    logits: *const bf16,
    out: *mut i32,
    rows: i32,
    vocab: i32,
) -> Result<(), Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if vocab <= 0 {
        return Err(Refusal::Empty { what: "vocab" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(&ROOT, inst::ARGMAX_BF16, rms(rows), &[logits.arg(), out.arg(), vocab.arg()])
    }
}

/// `sample::argmax_f32` — the fp32 twin.
///
/// [`argmax_bf16`]'s obligation, with `f32` for `bf16`.
pub fn argmax_f32(
    ctx: &Ctx,
    logits: *const f32,
    out: *mut i32,
    rows: i32,
    vocab: i32,
) -> Result<(), Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if vocab <= 0 {
        return Err(Refusal::Empty { what: "vocab" });
    }
    // SAFETY: as [`argmax_bf16`]'s.
    unsafe {
        ctx.launch(&ROOT, inst::ARGMAX_F32, rms(rows), &[logits.arg(), out.arg(), vocab.arg()])
    }
}

/// `sample::argmax_compact_scatter_bf16` — the compact form.
///
/// What the caller must guarantee, as `call()` states it:
///
/// `logits` must address `rows * vocab` live bf16 elements, `row_indices`
/// `rows` live `i32`s, and `out` must be writable at every index
/// `row_indices` names.
pub fn argmax_compact_scatter_bf16(
    ctx: &Ctx,
    logits: *const bf16,
    row_indices: *const i32,
    out: *mut i32,
    rows: i32,
    vocab: i32,
) -> Result<(), Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if vocab <= 0 {
        return Err(Refusal::Empty { what: "vocab" });
    }
    // SAFETY: as [`argmax_bf16`]'s, and `out` must be writable at every index
    // `row_indices` names.
    unsafe {
        ctx.launch(
            &ROOT,
            inst::ARGMAX_COMPACT_SCATTER,
            rms(rows),
            &[logits.arg(), row_indices.arg(), out.arg(), vocab.arg()],
        )
    }
}

/// This family's routines, and what a trace may say about each.
pub static ROUTINES: &[Routine] = &[
    routine!(lm_head_gemv_argmax_int8),
    routine!(argmax_bf16),
    routine!(argmax_f32),
    routine!(argmax_compact_scatter_bf16),
];

/// `sample`, as a trace names it.
pub static FAMILY: Family = Family { namespace: "sample", routines: ROUTINES };
