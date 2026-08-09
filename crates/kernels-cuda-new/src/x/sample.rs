#![allow(clippy::too_many_arguments)]

use crate::x::abi::bf16;
use crate::x::launch::Launch;
use core::ffi::c_void;

#[cfg(feature = "_cuda")]
use crate::x::contract::{Fired, Refusal};
#[cfg(feature = "_cuda")]
use std::sync::Mutex;

unit! {
    /// `sample`'s device text: the greedy argmax at two formats, the compact
    unit ARGMAX = "sample/argmax",
        text = include_str!("../../csrc/src/sample/argmax.cuh"),
        file = "sample/argmax.cuh";

    /// `argmax.cuh:219` — the greedy decode: one block per row, 256 threads
    fn argmax = "sample::device::argmax" <T> (
        logits: *const T,
        out: *mut i32,
        vocab: i32,
    ) where *const T {
        "sample::argmax_bf16" => where [T = bf16] "device::bf16",
        "sample::argmax_f32"  => where [T = f32] "sample::device::f32",
    }

    /// `argmax.cuh:248` — the compact form.
    fn argmax_compact_scatter = "sample::device::argmax_compact_scatter" <T> (
        logits: *const T,
        row_indices: *const i32,
        out: *mut i32,
        vocab: i32,
    ) where *const T {
        "sample::argmax_compact_scatter_bf16" => where [T = bf16] "device::bf16",
    }

    /// `argmax.cuh:670` — the fused int8 LM-head GEMV.
    fn lm_head_gemv_argmax_int8 = "sample::device::lm_head_gemv_argmax_int8" <T> (
        hidden_states: *const T,
        lm_head_weight: *const i8,
        scale_inv: *const f32,
        partial_pairs: *mut c_void,
        num_rows: i32,
        hidden: i32,
        vocab: i32,
        num_blocks_x: i32,
    ) where *const T {
        "sample::lm_head_gemv_argmax_int8_bf16" => where [T = bf16] "device::bf16",
    }

    /// `argmax.cuh:546` — the fused GEMV's second half: fold one partial pair
    fn select_lm_head_argmax_pairs = "sample::device::select_lm_head_argmax_pairs" (
        partial_pairs: *const c_void,
        out_tokens: *mut i32,
        num_rows: i32,
        num_tiles: i32,
    ) {
        "sample::select_lm_head_argmax_pairs" => crate::device::DeviceKernel::PLAIN,
    }
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

/// The growable pair scratch — `argmax.cu:113-114`'s two function-local
#[cfg(feature = "_cuda")]
static PAIRS: Mutex<Pairs> = Mutex::new(Pairs { ptr: 0, cap: 0 });

/// [`PAIRS`]' two numbers.
#[cfg(feature = "_cuda")]
struct Pairs {
    /// The device address, or 0 before the first allocation.
    ptr: usize,
    /// Capacity in `u64` PAIRS, not bytes — `argmax.cu:114`'s `s_pairs_cap`,
    cap: usize,
}

/// `grid.x` for the fused GEMV — `argmax.cu:101-107`, transcribed.
#[cfg(feature = "_cuda")]
fn blocks_x(vocab: i32) -> u32 {
    static SMS: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    let num_sms = *SMS.get_or_init(|| {
        use cudarc::runtime::sys as rt;

        let mut ordinal: i32 = 0;
        // SAFETY: `ordinal` is a live, writable out-parameter for the call.
        let code = unsafe { rt::cudaGetDevice(&raw mut ordinal) };
        assert!(
            code == rt::cudaError::cudaSuccess,
            "sample::lm_head_gemv_argmax_int8: cudaGetDevice failed ({code:?}), so the SM \
             count that sizes this launch cannot be read. There is no safe default: the \
             number is the grid AND the operand the kernel strides the vocab by"
        );
        let mut sms: i32 = 0;
        // SAFETY: `sms` is a live, writable out-parameter for the call.
        let code = unsafe {
            rt::cudaDeviceGetAttribute(
                &raw mut sms,
                rt::cudaDeviceAttr::cudaDevAttrMultiProcessorCount,
                ordinal,
            )
        };
        assert!(
            code == rt::cudaError::cudaSuccess && sms > 0,
            "sample::lm_head_gemv_argmax_int8: cudaDevAttrMultiProcessorCount failed on \
             device {ordinal} ({code:?}), and there is no safe default for it"
        );
        sms
    });
    let max_blocks_x = num_sms.unsigned_abs() * BLOCKS_PER_SM;
    let min_blocks_x = vocab.unsigned_abs().div_ceil(GEMV_WARPS);
    max_blocks_x.min(min_blocks_x).max(1)
}

/// Greedy decode straight off an int8 LM head: `token_ids[r] = argmax_v
///
/// # Safety
///
/// Every pointer must address live device memory of the extents `num_rows`,
/// `hidden` and `vocab` describe, `token_ids` must be writable for `num_rows`
/// i32, and `stream` must be live across BOTH launches.
#[cfg(feature = "_cuda")]
pub unsafe fn lm_head_gemv_argmax_int8(
    hidden_states: *const bf16,
    lm_head_weight: *const i8,
    scale_inv: *const f32,
    token_ids: *mut i32,
    num_rows: i32,
    hidden: i32,
    vocab: i32,
    stream: *mut c_void,
) -> Fired {
    if num_rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    if hidden <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden" });
    }
    if vocab <= 0 {
        return Fired::Declined(Refusal::Empty { what: "vocab" });
    }

    let num_blocks_x = blocks_x(vocab);
    let rows = num_rows.unsigned_abs();

    let pairs_elems = num_blocks_x as usize * rows as usize;
    let mut pairs = PAIRS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if pairs_elems > pairs.cap {
        use cudarc::runtime::sys as rt;

        if pairs.ptr != 0 {
            // SAFETY: the address came from `cudaMalloc` below and nothing
            let _ = unsafe { rt::cudaFree(pairs.ptr as *mut c_void) };
            pairs.ptr = 0;
            pairs.cap = 0;
        }
        let bytes = pairs_elems * core::mem::size_of::<u64>();
        let mut p: *mut c_void = core::ptr::null_mut();
        // SAFETY: `p` is a live, writable out-parameter.
        let code = unsafe { rt::cudaMalloc(&raw mut p, bytes) };
        assert!(
            code == rt::cudaError::cudaSuccess && !p.is_null(),
            "sample::lm_head_gemv_argmax_int8: cudaMalloc({bytes}) for the pair scratch \
             failed ({code:?}). The C++ ignored this return code and launched over a null \
             pointer; refusing here is the difference between a diagnosable failure and a \
             token id read out of unwritten memory"
        );
        pairs.ptr = p as usize;
        pairs.cap = pairs_elems;
    }
    let partial_pairs = pairs.ptr as *mut c_void;

    let smem = hidden.unsigned_abs().saturating_mul(4);
    unsafe {
        raw::lm_head_gemv_argmax_int8(
            "sample::lm_head_gemv_argmax_int8_bf16",
            Launch {
                grid: [num_blocks_x, rows, 1],
                block: [GEMV_BLOCK_DIM, 1, 1],
                smem,
                smem_opt_in: smem > crate::x::launch::OPT_IN_ABOVE,
            },
            hidden_states,
            lm_head_weight,
            scale_inv,
            partial_pairs,
            num_rows,
            hidden,
            vocab,
            i32::try_from(num_blocks_x).unwrap_or(i32::MAX),
            stream,
        );
        raw::select_lm_head_argmax_pairs(
            "sample::select_lm_head_argmax_pairs",
            Launch::flat(rows, SELECT_BLOCK),
            partial_pairs.cast_const(),
            token_ids,
            num_rows,
            i32::try_from(num_blocks_x).unwrap_or(i32::MAX),
            stream,
        );
    }

    drop(pairs);
    Fired::Launched
}

/// `sample::argmax_bf16` — the greedy decode over a bf16 logit row.
///
/// # Safety
///
/// `logits` must address `rows * vocab` live bf16 elements, `out` `rows`
/// writable `i32`s, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn argmax_bf16(
    logits: *const bf16,
    out: *mut i32,
    rows: i32,
    vocab: i32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if vocab <= 0 {
        return Fired::Declined(Refusal::Empty { what: "vocab" });
    }
    unsafe {
        raw::argmax("sample::argmax_bf16", rms(rows), logits, out, vocab, stream);
    }
    Fired::Launched
}

/// `sample::argmax_f32` — the fp32 twin.
///
/// # Safety
///
/// [`argmax_bf16`]'s, with `f32` for `bf16`.
#[cfg(feature = "_cuda")]
pub unsafe fn argmax_f32(
    logits: *const f32,
    out: *mut i32,
    rows: i32,
    vocab: i32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if vocab <= 0 {
        return Fired::Declined(Refusal::Empty { what: "vocab" });
    }
    unsafe {
        raw::argmax("sample::argmax_f32", rms(rows), logits, out, vocab, stream);
    }
    Fired::Launched
}

/// `sample::argmax_compact_scatter_bf16` — the compact form.
///
/// # Safety
///
/// `logits` must address `rows * vocab` live bf16 elements, `row_indices`
/// `rows` live `i32`s, `out` must be writable at every index `row_indices`
/// names, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn argmax_compact_scatter_bf16(
    logits: *const bf16,
    row_indices: *const i32,
    out: *mut i32,
    rows: i32,
    vocab: i32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if vocab <= 0 {
        return Fired::Declined(Refusal::Empty { what: "vocab" });
    }
    unsafe {
        raw::argmax_compact_scatter(
            "sample::argmax_compact_scatter_bf16",
            rms(rows),
            logits,
            row_indices,
            out,
            vocab,
            stream,
        );
    }
    Fired::Launched
}

contract! {
    /// Produces TOKEN IDS, not logits: a greedy-decode fast path that never
    LM_HEAD_GEMV_ARGMAX_INT8 = "sample::lm_head_gemv_argmax_int8" as lm_head_gemv_argmax_int8
}

#[cfg(feature = "_cuda")]
bind! {
    LM_HEAD_GEMV_ARGMAX_INT8 => { none:
        "sample::lm_head_gemv_argmax_int8: all eight operands were unsourced. \
         The two that still are: the int8 head and its per-row dequant scale \
         are named weights, and no model text states this symbol, so there is \
         no statement to read the two names off. The host program is public" },
}
