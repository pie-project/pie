//===-- attention_score_post.cuh - the capture post-kernels -----*- CUDA -*-===//
//
// THREE `__global__`s, no host code: the arithmetic that turns a captured
// attention-score buffer into something a policy can read.
//
//  * `attn_score_normalize`         -- decode: row-wise softmax, in place.
//  * `attn_prefill_score_normalize` -- prefill: the same, per window row, with
//                                      the causal support each row actually
//                                      has.
//  * `attn_prefill_score_fold`      -- prefill: `[head, window_row, kv]` down
//                                      to one averaged row per request.
//
// # Why they are here now, having been refused once
//
// `attention_flashinfer.cuh` -- the sibling that took `attn_score_fold_heads`
// out of the same `.cu` -- wrote the refusal down at the time:
//
//     `k_attn_score_normalize`, `k_attn_prefill_score_normalize` and
//     `k_attn_prefill_score_fold` stay. [...] No symbol in `kernels::table`
//     names any of them, nothing outside this translation unit can call
//     them, and a `.cuh` for a kernel no row can name is text nobody
//     compiles. THEY MOVE WHEN SOMETHING ASKS FOR THEM.
//
// Something asked. `attention_flashinfer.cu` left `crates/kernels-cuda` --
// the archive crate, not this one -- for `crates/driver-cuda/csrc/attn/`,
// because its two capture dispatches are
// host walks -- `switch (cache.head_dim)` over `src/kernels.def` into a
// FlashInfer instantiation -- and a host walk with a `cudaStream_t` in its
// signature is `driver-cuda`'s kind of object, which is the argument
// `driver-cuda/build.rs` already makes for the three vision towers. The
// device text a walk launches does not follow it. **This crate is where
// `__global__`s live**, whether or not a row names them, and the alternative
// -- shipping three kernels inside the driver's own `csrc/` -- is the second
// copy `new-horizon.md` §21.7 measured the cost of.
//
// The refusal's premise still holds and is now the *reason for the file
// rather than against it*: no table row names these, so no JIT unit compiles
// them, so this header is text ONLY nvcc reads. That is not a new shape.
// `attn/attention_score_capture.cuh` beside it -- the `LogitsTransform` hooks
// these three consume the output of -- is carried by exactly the same walk of
// `kernels/`, and was in exactly this position when this file was written:
// its one includer was `attention_flashinfer_common.cuh`, and NVRTC had never
// compiled a byte of it. Both halves of that have since moved -- the includer
// is deleted, and `attn/fa2.cuh` includes the capture hooks now, so NVRTC
// does compile them -- but the shape is one this tree has held before.
//
// # The rename, stated
//
// `k_attn_score_normalize` -> `attn_score_normalize`, and likewise for the
// other two. The `k_` prefix distinguished a `__global__` from its host
// launcher inside one `.cu`; here the namespace does that, and the sibling
// header already spells `attn_score_fold_heads` without one.
//
// A rename during a split is the exact hazard §21.7 measured -- fourteen
// drifted copies of `attn/kv_paged.cu` kernels passed every gate for a week
// because the gate compared NAMES and a split had renamed one side. So:
// **there is no other copy.** `attention_flashinfer.cu` no longer defines
// these; it includes this file. `tests/sources.rs::no_global_is_defined_twice`
// is the machine that keeps that true, and it reads both trees.
//
// # Written over the prelude, so nothing external is needed
//
// The `.cu` these came from opens with `<atomic>`, `<cstdio>`, `<cstdlib>`,
// the whole of `attention_flashinfer_common.cuh` and eleven FlashInfer
// headers. None of that is required to say what these three do, and requiring
// it would make the file unreadable to NVRTC forever rather than merely
// uncompiled by it today. So, exactly as the sibling header does:
//
//  * `std::int32_t`, `std::uint32_t` and `std::size_t` are `i32`, `u32` and
//    `usize` from the prelude -- the same three types under nvcc.
//  * `-INFINITY` is `neg_inf()`, `pie_device.cuh:359`, which is
//    `__int_as_float(0xff800000u)` and therefore the identical bit pattern.
//  * `min(a, b)` over two `int`s is written as the ternary it compiles to.
//    CUDA's `::min` arrives from `<device_functions.h>` under nvcc and is one
//    of the names NVRTC does not answer; the ternary is the same instruction
//    and needs nobody.
//
// `__expf` and `fmaxf` are compiler builtins in both front ends and are left
// alone. There is no `#ifdef __CUDACC_RTC__` below, for §14.3's reason: the
// guard is what lets two arms drift.
//
// # Linkage: SINGLE-INCLUDER
//
// §21.6, verbatim and measured again for `attention_flashinfer.cuh` on nvcc
// 13.0.88: a `.cuh` holding a non-template `__global__` may be included by
// exactly ONE translation unit, because the function and its host stub both
// take external linkage and a second includer is a hard `multiple definition`
// at link even when it never launches anything.
//
// The permitted includer is `crates/driver-cuda/csrc/attn/attention_flashinfer.cu`,
// which is the only file in the tree that launches these three -- three
// `<<<>>>`, one in `dispatch_attention_flashinfer_decode_capture_bf16` and
// two in `dispatch_attention_flashinfer_prefill_capture_bf16`. Both of those
// are `Execution::Walk` rows in `src/execution.rs`; the walk is what the
// driver owns, and this is what it launches.
//
// # THE CONSTRAINT IS NOW VACUOUS, and the named includer does not exist
//
// `crates/driver-cuda/csrc/attn/attention_flashinfer.cu` was deleted with the
// whole of `crates/driver-cuda/csrc`. This header has ZERO includers, so
// "exactly one" is satisfied by zero and the rule protects nothing.
//
// That is the harmless half. The harmful half is that the paragraph above
// reads as an instruction -- it names a permitted includer, and a reader who
// needs to include this header will go looking for that file, or worse,
// recreate it. The measurement is still true and still worth keeping (a
// non-template `__global__` in a `.cuh` really does collide at link on the
// second includer); what expired is the file it was pinned against. Keep the
// argument, do not act on the address.
//
// Nothing here is at risk while the count is zero. The constraint becomes
// live again the moment ANY translation unit includes this file, which under
// the JIT means a `unit!` root -- and this header is one
// (`families::attn::ATTENTION_SCORE_POST`), where NVRTC compiles it alone and
// the question does not arise.
//
// # A unit now, and the premise that changed
//
// This section used to read "No unit, and that is deliberate rather than
// pending", on the following argument:
//
//     `tests/units.rs::verdict` hard-fails a unit that declares no rows,
//     because a cubin nothing can fire is cached under an architecture and
//     satisfies nobody. Declaring one here would need three table rows, and
//     a table row is a thing a model text can STATE -- these are internal
//     steps of a dispatch that has its own row already.
//
// Every clause of that is still true and the conclusion has inverted, on one
// word: "a dispatch that has its own row already" meant the two capture
// dispatches in `driver-cuda/csrc/attn/attention_flashinfer.cu`, and those
// are HOST C++ that is becoming Rust. A Rust composer cannot fire a kernel it
// has no row for -- `unit::unit_of` resolves a SYMBOL -- so the rows exist
// for the DRIVER, not for a model text, and `families::attn`'s
// `ATTN_SCORE_POST` is where they are.
//
// The distinction the original refusal was drawing is worth keeping visible
// and is kept: none of these three is in `kernels::table`, no model text
// spells one, and `tests/consumer.rs` never sees them. A family row and a
// table row are different objects, and only the first was ever needed here.
// This is the same inversion `families::ssm`'s
// `causal_conv1d_prefill_noact_bf16` records -- an EMPTY consumer set was the
// whole of the objection, and a Rust caller is a consumer.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

namespace pie::attn {

/// Turn the captured scaled logits into attention probabilities, in place.
///
/// This is a plain row-wise softmax and NOT an approximation: at decode
/// `qo_len == 1`, so the row the variant captured is the complete set of
/// logits the kernel's own online softmax consumed. Recomputing the
/// denominator here is therefore exact, and it means the decode path does not
/// have to allocate or plumb an LSE buffer it otherwise never needs.
///
/// `kv_len` is derived from the page CSR rather than passed in. That is
/// deliberate: the CSR is the single source of truth for sequence length in
/// this driver (`kernels/geometry.cu`), and a second, independently-computed
/// length is exactly how a silent mis-attribution bug gets in.
__global__ void attn_score_normalize(
    float* __restrict__ scores,
    const i32* __restrict__ score_indptr,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    int page_size)
{
    constexpr int kThreads = 256;
    __shared__ float shared[kThreads];

    const int request = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int pages = static_cast<int>(kv_page_indptr[request + 1]) -
                      static_cast<int>(kv_page_indptr[request]);
    if (pages <= 0) return;
    const int kv_len =
        (pages - 1) * page_size + static_cast<int>(kv_last_page_lens[request]);
    if (kv_len <= 0) return;

    float* row = scores + static_cast<usize>(score_indptr[request]) +
                 static_cast<usize>(head) * static_cast<usize>(kv_len);

    float local = neg_inf();
    for (int i = threadIdx.x; i < kv_len; i += kThreads) {
        local = fmaxf(local, row[i]);
    }
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            shared[threadIdx.x] =
                fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float row_max = shared[0];
    __syncthreads();

    float total = 0.f;
    for (int i = threadIdx.x; i < kv_len; i += kThreads) {
        const float e = __expf(row[i] - row_max);
        row[i] = e;
        total += e;
    }
    shared[threadIdx.x] = total;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float denom = shared[0];
    if (denom <= 0.f) return;
    const float inv = 1.f / denom;
    for (int i = threadIdx.x; i < kv_len; i += kThreads) {
        row[i] *= inv;
    }
}

/// Prefill counterpart. Two things make this not just the decode kernel with
/// an extra grid dimension:
///
///  1. **Every window row has a different causal support.** The hook fires
///     before the kernel's mask (`LogitsMask` runs after `LogitsTransform`),
///     so a captured row contains real dot products at positions the softmax
///     is about to discard. Normalising over the full `kv_len` would spread
///     mass onto the future. Window row `w` belongs to the query at absolute
///     position `kv_len - rows + w`, so it may attend to
///     `kv_len - rows + w + 1` keys and no more. Everything past that is
///     zeroed here, which is also what makes the folded row a distribution
///     over the prefix.
///
///  2. **`rows` is `min(window, qo_len)`.** A prompt shorter than the
///     observation window contributes fewer rows; the rest of its slot is
///     never written by the kernel and must already be zero.
__global__ void attn_prefill_score_normalize(
    float* __restrict__ scores,
    const i32* __restrict__ score_indptr,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    int page_size,
    int window)
{
    constexpr int kThreads = 256;
    __shared__ float shared[kThreads];

    const int request = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int w = static_cast<int>(blockIdx.z);

    const int pages = static_cast<int>(kv_page_indptr[request + 1]) -
                      static_cast<int>(kv_page_indptr[request]);
    if (pages <= 0) return;
    const int kv_len =
        (pages - 1) * page_size + static_cast<int>(kv_last_page_lens[request]);
    if (kv_len <= 0) return;
    const int qo_len = static_cast<int>(qo_indptr[request + 1]) -
                       static_cast<int>(qo_indptr[request]);
    const int rows = window < qo_len ? window : qo_len;
    if (w >= rows) return;

    const int causal = kv_len - rows + w + 1;
    const int limit = causal < kv_len ? causal : kv_len;
    if (limit <= 0) return;

    float* row = scores + static_cast<usize>(score_indptr[request]) +
                 (static_cast<usize>(head) * static_cast<usize>(window) +
                  static_cast<usize>(w)) *
                     static_cast<usize>(kv_len);

    float local = neg_inf();
    for (int i = threadIdx.x; i < limit; i += kThreads) {
        local = fmaxf(local, row[i]);
    }
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            shared[threadIdx.x] =
                fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float row_max = shared[0];
    __syncthreads();

    // Accumulate the denominator WITHOUT storing the exponentials. Storing
    // them and rescaling in a second pass costs one more full write and one
    // more full read of a `heads * window * kv_len` buffer, which at 8K
    // context is 16 MB per layer; recomputing `__expf` in the final pass is a
    // handful of SFU cycles against a kernel that is entirely bandwidth-bound.
    float total = 0.f;
    for (int i = threadIdx.x; i < limit; i += kThreads) {
        total += __expf(row[i] - row_max);
    }
    shared[threadIdx.x] = total;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float denom = shared[0];
    // `denom >= 1` in exact arithmetic -- the argmax element contributes
    // `exp(0)` -- so this is unreachable. Zeroing rather than returning is
    // still the right failure: an early return would leave raw LOGITS in a
    // buffer the fold is about to average, and negative logits would read as
    // negative attention mass.
    const float inv = denom > 0.f ? 1.f / denom : 0.f;

    // One pass over the WHOLE row: positions at or past the causal limit were
    // computed by the kernel but never attended to (`LogitsMask` runs after
    // `LogitsTransform`), so they are zeroed here rather than in a separate
    // sweep. This is what makes the folded row a distribution over the prefix.
    for (int i = threadIdx.x; i < kv_len; i += kThreads) {
        row[i] = i < limit ? __expf(row[i] - row_max) * inv : 0.f;
    }
}

/// Fold `[head, window_row, kv]` down to one row per request.
///
/// Averaging rather than summing, for the same reason the decode fold
/// averages: every contributing row is a distribution over the prefix, so the
/// mean is one too, and a policy can threshold it in absolute terms without
/// knowing how many heads or window rows went into it. The divisor is
/// `heads * rows` with `rows = min(window, qo_len)` -- rows that do not exist
/// contribute nothing and must not be counted, or a short prompt's mass would
/// be scaled down.
///
/// The folded row lands at `score_indptr[r] / (heads * window)`, which is the
/// same derivation trick `attn_score_fold_heads` uses: the ragged offset
/// divided by the per-request multiplier is exactly the folded offset.
__global__ void attn_prefill_score_fold(
    const float* __restrict__ scores,
    float* __restrict__ folded,
    const i32* __restrict__ score_indptr,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    int page_size,
    int num_q_heads,
    int window)
{
    const int request = static_cast<int>(blockIdx.x);
    const int pages = static_cast<int>(kv_page_indptr[request + 1]) -
                      static_cast<int>(kv_page_indptr[request]);
    if (pages <= 0) return;
    const int kv_len =
        (pages - 1) * page_size + static_cast<int>(kv_last_page_lens[request]);
    if (kv_len <= 0) return;
    const int qo_len = static_cast<int>(qo_indptr[request + 1]) -
                       static_cast<int>(qo_indptr[request]);
    const int rows = window < qo_len ? window : qo_len;
    if (rows <= 0) return;

    const usize base = static_cast<usize>(score_indptr[request]);
    const usize out_base =
        base / (static_cast<usize>(num_q_heads) * static_cast<usize>(window));
    const float inv = 1.f / static_cast<float>(num_q_heads * rows);

    for (int k = static_cast<int>(threadIdx.x) +
                 static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.x);
         k < kv_len;
         k += static_cast<int>(blockDim.x) * static_cast<int>(gridDim.y)) {
        float acc = 0.f;
        for (int h = 0; h < num_q_heads; ++h) {
            for (int w = 0; w < rows; ++w) {
                acc += scores[base +
                              (static_cast<usize>(h) *
                                   static_cast<usize>(window) +
                               static_cast<usize>(w)) *
                                  static_cast<usize>(kv_len) +
                              static_cast<usize>(k)];
            }
        }
        folded[out_base + static_cast<usize>(k)] = acc * inv;
    }
}

}  // namespace pie::attn
