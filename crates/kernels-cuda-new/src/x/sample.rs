//! `sample` — turning logits into tokens, as two truths.
//!
//! One root of device text, `csrc/src/sample/argmax.cuh`; five `__global__`
//! instantiations over it; four host programs, one contract and one written
//! refusal. §5 step 5's second family.
//!
//! # What this replaces
//!
//! ```text
//!   before                                              lines
//!   kernels-cuda-new/src/families/sample.rs   5 rows       288
//!   kernels-cuda-new/src/table/sample.rs      1 row         33
//!   driver-cuda/src/fire/lm_head_argmax.rs    1 program    395
//!   driver-cuda/src/bind/service.rs           1 wrapper    ~54
//!                                                       ------
//!                                                          770
//!   after
//!   kernels-cuda-new/src/x/sample.rs    4 host programs, 5 device
//!                                        rows, 1 contract, 1 `none:`
//! ```
//!
//! # The archetype, and why it is in the boring batch
//!
//! [`lm_head_gemv_argmax_int8`] is the owner's principle with both clauses
//! true of one body:
//!
//! > Every CUDA kernel is compiled by NVRTC. Where host code is needed to
//! > compose kernels — because kernels produce intermediate results, or
//! > because device-specific tuning is involved — that host code is all Rust.
//!
//! *Intermediate results*: the first kernel writes one packed `(value,
//! token)` pair per tile per row into a scratch buffer, and the second folds
//! the tiles. Nothing in the deleted table row's operand list mentions that
//! buffer. *Device-specific tuning*: `grid.x` is `min(num_sms * 2, ceil(vocab
//! / 8))`, read straight off `cudaDevAttrMultiProcessorCount`.
//!
//! It is boring anyway, because it was already Rust. The move is a move.
//!
//! # The five things no row says, and where each one went
//!
//! `families/sample.rs`'s header listed them. They are all still true — none
//! is retracted by declaring the two kernels, because a declaration is a NAME
//! and a parameter list and never a geometry:
//!
//! | the fact | `argmax.cu` | here |
//! | --- | --- | --- |
//! | grid.x from an occupancy query | `:101-107` | [`blocks_x`] |
//! | a 2-D grid over (blocks, rows) | `:121` | the first `Launch` below |
//! | dynamic shmem `hidden * 4` | `:108-109` | the same `Launch` |
//! | a `static` scratch that grows | `:111-119` | [`PAIRS`] |
//! | two kernels per call | `:123`, `:136` | [`lm_head_gemv_argmax_int8`] |
//!
//! **No `LaunchRule` variant was ever added for them**: §10.5's bar is that a
//! rule must serve more kernels than the one that wants it, and a grid whose
//! extent comes from a device query serves exactly one. Under fn-world the
//! bar is not reached for, because there is no vocabulary left to grow.
//!
//! # One `none:`, and the sentence was already written
//!
//! `sample::lm_head_gemv_argmax_int8` is the family's only trace-facing
//! symbol and **all eight of its operands were unsourced**. `crate::abi` skips
//! a row with any `Source::Unbound` operand whole, so that row generated no
//! dispatch arm and claimed none; a trace that stated it got `UnknownKernel`
//! at fire time, silently. Here it is a refusal with a sentence, made at
//! model load. See [`LM_HEAD_GEMV_ARGMAX_INT8`]'s bind for the sentence and
//! for what would close it.
//!
//! # The scratch, reproduced rather than improved — and its inherited defect
//!
//! `argmax.cu:113-119` was a function-local `static device::u64*` with a
//! `static usize` capacity beside it, `cudaFree`d and re-`cudaMalloc`d
//! whenever `num_blocks_x * num_rows` outgrew it. [`PAIRS`] is that, as a
//! value, with the allocation under a lock so the GROWTH is not a data race.
//!
//! **The buffer is still shared by every caller, and that is a defect this
//! port inherits deliberately.** Two worker threads firing this symbol on two
//! streams write the same scratch, and holding a lock across the launches
//! does not fix it: a launch is asynchronous, so the second thread's GEMV can
//! land in the buffer before the first thread's selector has read it, and the
//! result is a plausible token id for the wrong row. The C++ had exactly this
//! hazard and neither the Rust port before this one nor this one changes it,
//! because *"the port's first duty is to reproduce today's launches rather
//! than to improve them"* — a port that changes the arithmetic cannot be
//! A/B'd against the arithmetic it replaced.
//!
//! What closes it, when someone wants to: the scratch belongs in the fire's
//! own per-fire pool, which is already pooled for the address-identity reason
//! a recorded graph needs. That is a change to where a buffer lives, not to
//! what the kernels compute, and it can be measured against this.
//!
//! # A failure is a refusal, never a fallback
//!
//! `argmax.cu` ignored `cudaMalloc`'s return code and launched over whatever
//! `s_partial_pairs` held — a null pointer on the first failure. This panics.
//! It also panics on a failed SM query, where a compute-capability read may
//! be defaulted: that kind of default can answer 4 because both arms compute
//! the same thing and one is slower, whereas here the queried number IS
//! `grid.x` and IS the `num_blocks_x` operand the kernel strides the vocab
//! by. A guess covers a subset of the vocabulary and reports the argmax of
//! it, which is a wrong answer wearing a right answer's shape.
//!
//! # `partial_pairs` is the one parameter this family cannot spell
//!
//! It is a `u64*` — one packed `(value, token)` pair per tile per row — and
//! `kernels::Ty` has no word for that. `x/abi.rs`'s note beside the missing
//! impl is this family's, and its conclusion is the one `families/sample.rs`
//! reached: the buffer crosses correctly today, it is opaque to the host, and
//! only its width is wrong in the tag. It is `*mut c_void` here, under
//! `Ty::BufMut`, which is what the deleted row already said. Closing it is
//! `Ty::U64sMut` plus its `cpp()`/`rust()`/`ArgValue` arms in `crates/kernels`
//! — a step-9 change, when `Ty` retires or becomes fn-world's alone.

#![allow(clippy::too_many_arguments)]

use crate::x::abi::bf16;
use crate::x::launch::Launch;
// Ungated, where `rope.rs` gates it: `c_void` appears in the `unit!` parameter
// lists below — `partial_pairs` is the `u64*` this family cannot spell — and
// `ROWS`/`PARAMS` are built with or without a GPU.
use core::ffi::c_void;

#[cfg(feature = "_cuda")]
use crate::x::contract::{Fired, Refusal};
#[cfg(feature = "_cuda")]
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Truth one, declared: the device text and its instantiations.
//
// One root and one `unit!`, so `UNITS`, `ROWS`, `PARAMS` and `mod raw` land
// at this module's scope directly — `rope`'s shape, not `layout`'s.
// ---------------------------------------------------------------------------

unit! {
    /// `sample`'s device text: the greedy argmax at two formats, the compact
    /// scatter, and the fused int8 LM-head GEMV with its selector.
    ///
    /// Twelve `__global__` templates, five instantiations. The ten with none
    /// are unchanged in kind and their reasons are `families/sample.rs`'s,
    /// re-read against fn-world:
    ///
    /// * `lm_head_gemv_argmax` — the bf16-weight twin of the int8 form.
    ///   Nothing calls it: `sample/argmax.cu` launched only the int8 form and
    ///   the table held only the int8 symbol. A declaration would name a
    ///   caller that does not exist.
    /// * `argmax_vec2` and `argmax_compact_scatter_vec2` — selected by
    ///   `argmax_vec2_usable(logits, vocab)`, a run-time test on an operand's
    ///   ADDRESS and on the parity of the vocab. **Under fn-world that is no
    ///   longer a reason** — a host `fn` may test a pointer, which is exactly
    ///   what `x::layout::vectorisable` does — and what still blocks them is
    ///   the smaller thing the row world's argument hid: firing the vec2 form
    ///   on an odd vocab puts every second row on a 2-byte boundary and
    ///   faults, and no caller in the tree asks for the wide form. They come
    ///   back with a caller and a host predicate, in that order.
    /// * `masked_embedding_argmax`, `topk_centroids` and
    ///   `masked_embedding_tile_argmax_pairs` — their launchers CLAMP
    ///   `centroid_top_k` to 64 before passing it, and the kernels index
    ///   `__shared__` arrays of exactly 64 with it. The clamp is load-bearing;
    ///   a caller that dropped it turns a truncation into a shared-memory
    ///   overrun on the first config that asks for more.
    /// * `argmax_accumulate` and `argmax_finalize` — 1024 threads and 32
    ///   threads, and two `bool` parameters. Nothing in the tree fires
    ///   either.
    ///
    /// # The `static_assert` `argmax.cu` carried
    ///
    /// `sample/argmax.cu:38-40` held `static_assert(device::kAccumWarps ==
    /// kArgmaxAccumSlots)` — the accumulator carries one slot per warp,
    /// `sample/argmax.hpp` published that count to its callers as
    /// `kArgmaxAccumSlots`, and the two files agreed by assertion rather than
    /// by assumption. Both files are deleted and the two constants are now
    /// one: `sample/argmax.cuh:150-151` defines `kAccumThreads = 1024` and
    /// `kAccumWarps = kAccumThreads / 32`, and nothing outside that header
    /// names a slot count. The assertion is not lost, it is unnecessary — but
    /// **a future caller of `argmax_accumulate` that sizes a scratch buffer
    /// must read `kAccumWarps` from the header rather than writing 32 by
    /// hand**, which is what the deleted `.hpp` constant existed to prevent.
    unit ARGMAX = "sample/argmax",
        text = include_str!("../../csrc/src/sample/argmax.cuh"),
        file = "sample/argmax.cuh";

    /// `argmax.cuh:219` — the greedy decode: one block per row, 256 threads
    /// striding the vocab.
    ///
    /// **Two instantiations, and the second costs a line.** These were TWO
    /// kernels in the ahead-of-time build — `argmax_bf16_kernel` and
    /// `argmax_fp32_kernel`, identical but for a load — because instantiating
    /// a template twice cost a translation unit. Under a JIT the second
    /// format costs a row, which is `norm/elementwise`'s measurement restated
    /// in a family that had already paid for it by hand.
    ///
    /// `sample::device::f32` is a `using` alias for `float` inside the device
    /// namespace, not a prelude type: `Elem<T>` has no `float` specialisation
    /// and should not grow one — there fp32 is what a kernel COMPUTES in, and
    /// a specialisation would make `Elem<float>::from_f32` an identity that
    /// reads like a conversion. The header's `Logit<T>` carries that one
    /// widening.
    fn argmax = "sample::device::argmax" <T> (
        logits: *const T,
        out: *mut i32,
        vocab: i32,
    ) where *const T {
        "sample::argmax_bf16" => where [T = bf16] "device::bf16",
        "sample::argmax_f32"  => where [T = f32] "sample::device::f32",
    }

    /// `argmax.cuh:248` — the compact form.
    ///
    /// Logits indexed by the COMPACT row, output by
    /// `row_indices[compact_row]`, so a fire that dropped rows writes its
    /// answers where the un-dropped batch expects them.
    fn argmax_compact_scatter = "sample::device::argmax_compact_scatter" <T> (
        logits: *const T,
        row_indices: *const i32,
        out: *mut i32,
        vocab: i32,
    ) where *const T {
        "sample::argmax_compact_scatter_bf16" => where [T = bf16] "device::bf16",
    }

    /// `argmax.cuh:670` — the fused int8 LM-head GEMV.
    ///
    /// **This parameter list is NOT the deleted table row's, and the
    /// difference is the point.** The table's
    /// `sample::lm_head_gemv_argmax_int8` stated `(hidden_states,
    /// lm_head_weight, scale_inv, token_ids, num_rows, hidden, vocab,
    /// stream)` — a launcher's list, with the CALLER's output and a stream.
    /// The kernel writes `partial_pairs`, never sees `token_ids`, takes
    /// `num_blocks_x` (its own grid extent, which it needs for the
    /// grid-stride bound) and takes no stream, because a stream is
    /// `cuLaunchKernel`'s sixth parameter. The two are different contracts
    /// over the same job, and the symbols differ so that `unit::unit_of`
    /// keeps answering `None` for the stated one.
    ///
    /// Rowed at bf16 alone, because that is the only instantiation
    /// `sample/argmax.cu` ever launched.
    ///
    /// `lm_head_weight` is `*const i8` and not `*const c_void`: the `Abi`
    /// impl for it exists for this parameter, and `x/abi.rs`'s note beside it
    /// says why the opaque spelling would be a bypass with no type error
    /// anywhere.
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
    /// per tile into one token per row.
    ///
    /// Not a template — the header says so in as many words, *"there is no
    /// element type in it, only packed pairs"* — so its instantiation is its
    /// bare path,
    /// [`DeviceKernel::PLAIN`](crate::device::DeviceKernel::PLAIN).
    ///
    /// `num_tiles` is `num_blocks_x` under another name: the producer's
    /// `grid.x`, which is how many pairs per row the scratch holds. It is the
    /// parameter `families/sample.rs` cites as the reason an `Elementwise`
    /// rule could not state this launch — the grid a rule computes, over an
    /// extent a device query produced.
    fn select_lm_head_argmax_pairs = "sample::device::select_lm_head_argmax_pairs" (
        partial_pairs: *const c_void,
        out_tokens: *mut i32,
        num_rows: i32,
        num_tiles: i32,
    ) {
        "sample::select_lm_head_argmax_pairs" => crate::device::DeviceKernel::PLAIN,
    }
}

// ---------------------------------------------------------------------------
// The numbers, once each.
// ---------------------------------------------------------------------------

/// Vocab rows a block covers per grid step — `sample/argmax.cuh:153`.
///
/// Also the divisor in `min_blocks_x = ceil(vocab / GEMV_WARPS)`: one warp
/// per vocab row, so the smallest grid that covers the vocab in one step is
/// the row count divided by the warps in a block. A grid computed with one
/// number and a kernel striding by another would either skip rows or repeat
/// them, and the kernel's `for (v = blockIdx.x * GEMV_WARPS + warp; v <
/// vocab; v += num_blocks_x * GEMV_WARPS)` cannot tell the difference.
const GEMV_WARPS: u32 = 8;

/// Threads per block of the fused GEMV — `sample/argmax.cuh:153-154`'s
/// `GEMV_WARPS = 8` times a warp, which the header spells `GEMV_BLOCK_DIM =
/// GEMV_WARPS * 32`.
///
/// Load-bearing twice, which is why it is derived from [`GEMV_WARPS`] rather
/// than written as 256: the kernel stages the hidden vector with `for (i =
/// threadIdx.x; i < hidden; i += GEMV_BLOCK_DIM)`, so a narrower block leaves
/// the tail of `sh_hidden` uninitialised and every dot product reads it.
const GEMV_BLOCK_DIM: u32 = GEMV_WARPS * 32;

/// Resident blocks per SM the launcher aims for — `argmax.cu:103`'s
/// `constexpr int kBlocksPerSm = 2`.
///
/// A persistent-block tuning constant with one reader, which is the shape a
/// tuning constant should have. It is not an occupancy guarantee: the kernel
/// asks for `hidden * sizeof(float)` of dynamic shared memory, so at a wide
/// hidden size two blocks per SM may not fit and the hardware simply runs
/// fewer. The grid is a bound on how much work is resident, not a promise.
const BLOCKS_PER_SM: u32 = 2;

/// Threads per block of the selector — `argmax.cu:134`'s `dim3
/// sel_block(128)`.
///
/// A plain elementwise fold over rows, one thread per row. 128 rather than
/// the 256 every other pointwise launcher in this tree uses, and it is
/// transcribed rather than harmonised: `num_rows` is a batch size, so the
/// grid is one or two blocks either way and the difference is unmeasurable —
/// which makes changing it a diff with no argument behind it.
const SELECT_BLOCK: u32 = 128;

/// Threads per block of the three plain argmaxes — `LaunchRule::Rms`' 256,
/// which is also the launcher's.
///
/// **The 256 is not merely what the launcher passed**: the kernels stride the
/// vocab by a compile-time `BLOCK = 256` and size their `__shared__`
/// reduction with it, so a block of any other width would fold over a buffer
/// it had not filled. That is why the three fitted `Rms` and the vectorised
/// pair — 128 threads doing two elements each — does not.
const ARGMAX_BLOCK: u32 = 256;

/// `LaunchRule::Rms`, as the expression it evaluates to.
///
/// `runtime/launch.rs:737-746` — `grid [rows, 1, 1]`, `block [256, 1, 1]`,
/// `(256 / 32) * 4` bytes of dynamic shared memory.
///
/// **The 32 bytes are asked for and never read**, and that was true under the
/// rule too. `families/sample.rs` recorded it: *"The rule also asks for 32
/// bytes of dynamic shared memory that these kernels never declare — an
/// unread allocation is not a behaviour, and the alternative was a fourth
/// rule that differs from `Rms` in a number nobody reads."* It is carried
/// here rather than dropped, because the port's duty is to reproduce today's
/// launches; a reader with a profile may drop it knowing it was deliberate.
#[must_use]
const fn rms(rows: i32) -> Launch {
    Launch::per_row(rows.unsigned_abs(), ARGMAX_BLOCK).smem((ARGMAX_BLOCK / 32) * 4)
}

// ---------------------------------------------------------------------------
// Truth two: the host programs.
// ---------------------------------------------------------------------------

/// The growable pair scratch — `argmax.cu:113-114`'s two function-local
/// `static`s, as one value.
///
/// The pointer is kept as a `usize` because a `*mut` is not `Send` and this
/// is reachable from every worker thread, exactly as the C++ `static` was.
/// That is a statement about the ADDRESS being shared, not a claim that
/// sharing it is safe — see this module's header, which names the hazard the
/// C++ had and this keeps.
///
/// Never freed at process exit, like the C++: the allocation is one buffer
/// whose lifetime is the process's, and a `Drop` that ran `cudaFree` during
/// static destruction would race the runtime's own teardown.
#[cfg(feature = "_cuda")]
static PAIRS: Mutex<Pairs> = Mutex::new(Pairs { ptr: 0, cap: 0 });

/// [`PAIRS`]' two numbers.
#[cfg(feature = "_cuda")]
struct Pairs {
    /// The device address, or 0 before the first allocation.
    ptr: usize,
    /// Capacity in `u64` PAIRS, not bytes — `argmax.cu:114`'s `s_pairs_cap`,
    /// which is compared against `pairs_elems` and never against a byte
    /// count.
    cap: usize,
}

/// `grid.x` for the fused GEMV — `argmax.cu:101-107`, transcribed.
///
/// ```text
/// cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
/// constexpr int kBlocksPerSm = 2;
/// const int max_blocks_x = num_sms * kBlocksPerSm;
/// const int min_blocks_x = (vocab + device::GEMV_WARPS - 1) / device::GEMV_WARPS;
/// const int num_blocks_x = std::min(max_blocks_x, min_blocks_x);
/// ```
///
/// The `min` is what makes the blocks persistent: enough to fill the machine,
/// never more than there is vocabulary for. It is also the kernel's
/// `num_blocks_x` operand, which is why this number is computed once and used
/// twice — a grid that disagreed with the operand would stride past the end
/// of the vocab or stop short of it, silently either way.
///
/// # The SM count is cached, and the query is on the CURRENT device
///
/// The C++ asked on every call and passed ordinal `0` unconditionally; this
/// asks once per process, on whichever device is current, and caches. On a
/// single-GPU process those are the same number. On a multi-GPU one the C++
/// was reading device 0's SM count to size a grid for whatever device the
/// stream belonged to — a bug that happens not to matter because every GPU in
/// a node is the same part, and it is not reproduced.
///
/// The driver's own `Device::sm_count` was the previous port's reader; this
/// crate has no `Device`, so the two `cudarc::runtime` calls the C++ made are
/// made here directly. Same attribute, same ordinal, one `OnceLock`.
///
/// # Panics
///
/// If the device cannot be read. See this module's header for why this is not
/// a defaulted value.
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
/// (hidden[r] . W[v]) * scale_inv[v]`.
///
/// `hidden_states` is bf16 `[num_rows, hidden]`, `lm_head_weight` is int8
/// `[vocab, hidden]` row-major, `scale_inv` is one fp32 dequant scale per
/// vocab row, and `token_ids` receives one i32 per row. Produces TOKEN IDS
/// and never materialises the vocab-wide logit row, which is why this is its
/// own statement rather than an `lm_head` followed by an argmax.
///
/// # TWO KERNELS IN ONE BODY, and the scratch between them
///
/// `argmax.cu:121-132`:
///
/// ```text
///     dim3 grid(num_blocks_x, num_rows);
///     dim3 block(device::GEMV_BLOCK_DIM);
///     device::lm_head_gemv_argmax_int8<device::bf16>
///         <<<grid, block, shmem_bytes, stream>>>(
/// ```
///
/// with `shmem_bytes = hidden * sizeof(float)` from `:108-109` — the staging
/// buffer for one row of the hidden vector, which the kernel declares
/// `extern __shared__` and therefore cannot size itself.
///
/// `argmax.cu:134-137`:
///
/// ```text
///     dim3 sel_block(128);
///     dim3 sel_grid((num_rows + sel_block.x - 1) / sel_block.x);
///     device::select_lm_head_argmax_pairs<<<sel_grid, sel_block, 0, stream>>>(
///         s_partial_pairs, token_ids, num_rows, num_blocks_x);
/// ```
///
/// `num_blocks_x` arrives as `num_tiles`: the same number, named for what it
/// means to the reader instead of for what it meant to the producer.
/// `partial_pairs` was written by the first launch, on this stream, so the
/// ordering that makes it readable in the second is the stream's.
///
/// # Panics
///
/// On a failed SM query, a failed scratch allocation, an NVRTC compile
/// failure, or a disagreement between this call and this file's declarations.
/// See this module's header.
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
    // `argmax.cu:99`. A decline and not a panic: an empty fire is a real
    // thing a batch produces and it was never an error.
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

    // `argmax.cu:111-119`. `pairs_elems` counts PAIRS; the allocation
    // multiplies by 8 and the capacity does not, exactly as `s_pairs_cap`
    // did.
    let pairs_elems = num_blocks_x as usize * rows as usize;
    let mut pairs = PAIRS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if pairs_elems > pairs.cap {
        use cudarc::runtime::sys as rt;

        if pairs.ptr != 0 {
            // SAFETY: the address came from `cudaMalloc` below and nothing
            // else frees it.
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
/// # No trace states this, and it is not dead
///
/// The deleted table row's comment is the whole story and it survives here:
///
/// > The plain `sample::argmax_bf16` is deliberately NOT here, though CSM's
/// > backbone fires it. A row was added and
/// > `the_table_is_exactly_the_dsl_surface` rejected it: this table and
/// > `dsl::cuda` are the same set, and a DSL statement is something a TRACE
/// > records. CSM's backbone is a hand-written forward, so nothing traces
/// > that argmax and the statement would have no caller.
///
/// A DEVICE declaration has no such constraint — it names an instantiation to
/// compile, not a statement to lower — and under fn-world neither does a
/// `fn`: this is a public function a hand-written forward calls directly,
/// which is what `x/driver_internal.rs` is the whole family of. So there is
/// no [`contract!`] for it and this is how a caller arrives.
///
/// # Geometry
///
/// `LaunchRule::Rms` ([`rms`]) — one block per row, 256 threads striding the
/// vocab, which is the launcher's `<<<rows, 256>>>` exactly.
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
/// The same template, the same rule, the same parameter order; it differs
/// from [`argmax_bf16`] in its instantiation and in nothing else, which is
/// the whole claim the JIT makes.
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
/// Logits indexed by the COMPACT row, output by `row_indices[compact_row]`,
/// so a fire that dropped rows writes its answers where the un-dropped batch
/// expects them. `rows` is the COMPACT count — the same count the launcher
/// passed, and the same one `Rms` read off the fire's rectangle.
///
/// [`argmax_bf16`]'s note about callers applies here too.
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

// ---------------------------------------------------------------------------
// The declaration the readers that cannot call read.
//
// One contract, because one symbol of this family is a thing a trace may say.
// The other three host programs above are public `fn`s a hand-written forward
// calls, which is the arrangement `x/driver_internal.rs` documents as the
// fourth: no contract, no `Entry`, no row.
// ---------------------------------------------------------------------------

contract! {
    /// Produces TOKEN IDS, not logits: a greedy-decode fast path that never
    /// materializes the vocab-wide row, which is why it is its own statement
    /// rather than `lm_head` followed by an argmax.
    LM_HEAD_GEMV_ARGMAX_INT8 = "sample::lm_head_gemv_argmax_int8" as lm_head_gemv_argmax_int8
}

// ---------------------------------------------------------------------------
// What happens when a trace says it.
//
// One arm, and it is a refusal. `#[cfg(feature = "_cuda")]` because an
// `Entry` holds a bind and a bind launches.
// ---------------------------------------------------------------------------

#[cfg(feature = "_cuda")]
bind! {
    // No bind, and the sentence was already written — as the ABSENCE of a
    // `Source` on all eight operands of the row this replaces.
    //
    // What each of the eight needed, in the row's own order:
    // `hidden_states` and `token_ids` are the statement's own In(0)/Out(0)
    // and would bind today. `lm_head_weight` and `scale_inv` are a QUANTISED
    // PAIR — an int8 weight and its per-row dequant scale — and neither is a
    // slot in the argument run: they are the statement's named weights, and
    // nothing in the tree states this symbol, so there is no statement to
    // read the two names off. `num_rows`, `hidden` and `vocab` are the
    // rectangle and the vocabulary, all three reachable now
    // (`Cx::rows`, `Cx::out_width`, `Cx::vocab`).
    //
    // So the refusal is smaller than the row's was, and it is one fact: the
    // two weight NAMES. `dsl::cuda::lm_head_gemv_argmax_int8` exists and has
    // ZERO call sites in `crates/model`; no `lower.rs::semantic()` mapping
    // names the symbol; nothing in the tree fires it. A bind written now
    // would be guessing which named weight is the head and which is the
    // scale, on behalf of a caller that does not exist — and a wrong guess
    // there is a launch with two pointers swapped, which is a wrong answer
    // and not a type error.
    //
    // What closes it: a model text that states it, whose two weight names
    // are then a fact rather than a guess. `lm_head_gemv_argmax_int8` above
    // is public, complete and fires correctly meanwhile — it is a direct
    // call away, which is exactly what `Route::Unbound` means and did not
    // mean under the row world.
    LM_HEAD_GEMV_ARGMAX_INT8 => { none:
        "sample::lm_head_gemv_argmax_int8: all eight operands were unsourced. \
         The two that still are: the int8 head and its per-row dequant scale \
         are named weights, and no model text states this symbol, so there is \
         no statement to read the two names off. The host program is public" },
}
