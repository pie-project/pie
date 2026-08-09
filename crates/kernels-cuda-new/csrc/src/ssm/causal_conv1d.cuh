//===-- causal_conv1d.cuh - the short causal convolution, as templates ---===//
//
// Five `__global__` templates and nothing else: no host function, no `<<<>>>`,
// no entry point. `causal_conv1d.cu` includes this file and keeps only its
// launchers, so there is EXACTLY ONE definition of each kernel in the tree.
//
// # Why the split, and what a copy would have cost
//
// A `.cu` holds both halves — the launcher and the `__global__` it launches —
// and the JIT needs only the second. The tempting move is to copy the kernels
// into a header and leave the `.cu` alone; `new-horizon.md` §10.10 records
// what that costs. Both halves compile, the archive gets one and NVRTC gets
// the other, and they drift with every test passing on whichever half it
// exercised. `norm/altup_aux` did exactly that for a release.
// `kernels-cuda/tests/sources.rs` now refuses two definitions of one
// namespace-qualified `__global__`, which is why this file INCLUDES rather
// than duplicates.
//
// # Why they are templates when the originals were `_bf16`
//
// An ahead-of-time build has to choose its instantiations, so the whole file
// was bf16 and only bf16 — not because the arithmetic is bf16's, but because
// a second element type cost a translation unit's worth of cicc for something
// nobody had asked for. Under the JIT the element type is the ROW's, so the
// kernels are written over `T` and reach the prelude's `Elem<T>` for the two
// conversions. The arithmetic is unchanged: widen to fp32, accumulate in
// fp32, narrow once, which is what the bf16 tolerance contract was measured
// against.
//
// `SILU` stays a template parameter rather than becoming a runtime flag,
// because the two instantiations are two kernels and that is what the naming
// rule asks a suffix to mean. Gemma-4's audio lconv1d wants this convolution
// with no activation, and `causal_conv1d_prefill_noact_bf16` in the `.cu` is
// that instantiation rather than a second kernel.
//
// # Which of these have rows, and which do not
//
// One: `causal_conv1d_update`, whose launcher is `ceil(C / BLOCK)` blocks of
// a fixed width over a pure map guarded by `c >= C` — `LaunchRule::Elementwise`
// exactly, by the coverage argument the norm pilot already measured. The
// other four compute their grid from a STATE LAYOUT: one block per channel
// with the token axis inside the block, or a `(channel-tile, request)`
// rectangle read out of `qo_indptr`. No rule in `kernels::LaunchRule` produces
// either, and three of them take the slot stride as a `long long` that
// `Args::bind` refuses. They are here because the split is per FILE — one
// `.cu`, one `.cuh`, one definition — and they are unrowed because inventing
// a rule to fit them would put a guess where a refusal belongs.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "pie_device.cuh"

namespace pie_cuda_driver::kernels::ssm::device {

// The scalar layer is the PRELUDE's, not this family's. Named here so the
// kernels below read as they always did, so a row may keep spelling its
// element type `device::bf16`, and so the LAUNCHERS in the sibling `.cu`
// files — which sit in `kernels::ssm` and reach these names unqualified
// through `device::` — resolve to the prelude's types rather than to nothing.
using ::pie_cuda_driver::kernels::device::Elem;
using ::pie_cuda_driver::kernels::device::bf16;
using ::pie_cuda_driver::kernels::device::bf16_to_f32;
using ::pie_cuda_driver::kernels::device::f16;
using ::pie_cuda_driver::kernels::device::f32_to_bf16;
using ::pie_cuda_driver::kernels::device::i32;
using ::pie_cuda_driver::kernels::device::u32;
using ::pie_cuda_driver::kernels::device::u8;
using ::pie_cuda_driver::kernels::device::usize;

/// SiLU, in the one spelling every kernel below shares.
///
/// `__device__` and not merely `inline`: nvcc infers a host function from an
/// unannotated one inside a `.cu` and says nothing, and NVRTC refuses it
/// outright — *"host functions are not allowed in JIT mode"*. That is
/// `new-horizon.md` §10.3's `yarn_original_ramp_bounds` defect, which lived in
/// a shared header for as long as the file existed because only one compiler
/// ever read it.
__device__ __forceinline__ float silu_f(float z) {
    return z / (1.f + __expf(-z));
}

/// One block per (channel, output token range). Each thread handles a
/// few output tokens in its block. The kernel size K is small (4 on
/// Qwen3.5), so the K accumulator unrolls trivially.
///
///     y[t, c] = silu( sum_{k=0..K-1} W[c, k] * x[t - K + 1 + k, c]  + bias[c] )
///
/// where `x[t<0, c]` is read from the prior state window. Fresh prompts
/// arrive with a zeroed state window, so this also implements causal
/// padding for first-chunk prefill. The trailing K input rows are written
/// back into `state_out[K, C]` (oldest first) so a follow-up decode or
/// mixed prefill chunk can resume from there.
template <class T, bool SILU>
__global__ void causal_conv1d_prefill(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ y,
    T* __restrict__ state_out,
    int N, int C, int K)
{
    const int c = blockIdx.x;       // one channel per block
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    if (c >= C) return;

    const float bias_v = bias ? Elem<T>::to_f32(bias[c]) : 0.f;

    // Each thread strides through tokens.
    for (int t = tid; t < N; t += block_size) {
        float acc = bias_v;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {  // unroll up to 8 (Qwen3.5 uses K=4)
            if (k >= K) break;
            const int src_t = t - (K - 1) + k;
            float xv = 0.f;
            if (src_t < 0) {
                if (state_out) {
                    xv = Elem<T>::to_f32(state_out[(K + src_t) * C + c]);
                }
            } else {
                xv = Elem<T>::to_f32(x[src_t * C + c]);
            }
            const float wv = Elem<T>::to_f32(weight[c * K + k]);
            acc += wv * xv;
        }
        y[t * C + c] = Elem<T>::from_f32(SILU ? silu_f(acc) : acc);
    }

    __syncthreads();

    // Persist the trailing K input rows into state_out (one thread does
    // this per channel; it's a tiny copy with strided indexing).
    if (state_out && tid == 0) {
        for (int s = 0; s < K; ++s) {
            const int src_t = N - K + s;  // token index for state slot s
            const float v = (src_t < 0)
                ? Elem<T>::to_f32(state_out[(K + src_t) * C + c])
                : Elem<T>::to_f32(x[src_t * C + c]);
            state_out[s * C + c] = Elem<T>::from_f32(v);
        }
    }
}

/// Decode update: state_in[K, C] holds the last K input rows; new x is
/// one row. After this kernel:
///   • y[c] = silu( sum_{k=0..K-1} W[c, k] * (k<K-1 ? state_in[k+1, c] : x[c])
///                 + bias[c] )
///   • state[K, C] is shifted: state[k] := state[k+1] for k<K-1, state[K-1] := x.
///
/// **The one kernel in this file a rule already states.** Every thread owns
/// one channel, reads only its own column and writes only its own, and the
/// guard is `c >= C` — so a grid that covers `C` threads computes all of it
/// and a wider one leaves the surplus idle. That is `LaunchRule::Elementwise`
/// by the same coverage argument `elementwise_rows_covers_every_channel`
/// measured for `mean_streams`: the launcher used 128-wide blocks and the rule
/// uses 256, the two cover the same channels, and a pure map answers the same
/// bits either way. A reduction could not be ported on those terms — the fold
/// order is part of the contract — and this is not one.
template <class T>
__global__ void causal_conv1d_update(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ state,
    T* __restrict__ y,
    int C, int K)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    const float bias_v = bias ? Elem<T>::to_f32(bias[c]) : 0.f;
    const float new_x  = Elem<T>::to_f32(x[c]);

    // Compute output: convolve over the K-window [state[1], ..., state[K-1], x].
    float acc = bias_v;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k >= K) break;
        float xv;
        if (k < K - 1) {
            xv = Elem<T>::to_f32(state[(k + 1) * C + c]);
        } else {
            xv = new_x;
        }
        const float wv = Elem<T>::to_f32(weight[c * K + k]);
        acc += wv * xv;
    }
    y[c] = Elem<T>::from_f32(silu_f(acc));

    // Update state: shift left by 1, new_x in the last slot.
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k >= K - 1) break;
        state[k * C + c] = state[(k + 1) * C + c];
    }
    state[(K - 1) * C + c] = Elem<T>::from_f32(new_x);
}

/// Multi-request batched prefill. Per-(channel, request) block; threads
/// stride through that request's tokens. Same math as the single-request
/// kernel; the (t0_r, Nr_r) window is read from qo_indptr at runtime,
/// source rows before that window are read from the request's existing
/// state slab, and the trailing K-window is persisted back to that slab.
///
/// **Unrowed.** The grid is `(C, R)` — a channel axis and a request axis —
/// and `slot_stride_elems` is a `long long`; no `LaunchRule` produces the
/// first and `Args::bind` refuses the second.
template <class T>
__global__ void causal_conv1d_prefill_batched(
    const T* __restrict__ x,                 // [N_total, C]
    const T* __restrict__ weight,            // [C, K]
    const T* __restrict__ bias,              // [C]
    T* __restrict__ y,                       // [N_total, C]
    T* __restrict__ state_out_base,          // [num_slots, K, C]
    const int* __restrict__ slot_ids,        // [R]
    const u32* __restrict__ qo_indptr,       // [R+1]
    long long slot_stride_elems,
    int C, int K, bool write_state,
    const u8* __restrict__ write_state_mask,
    const int* commit_len)
{
    const int c = blockIdx.x;
    const int r = blockIdx.y;
    if (c >= C) return;

    const int t0 = static_cast<int>(qo_indptr[r]);
    int Nr = static_cast<int>(qo_indptr[r + 1]) - t0;
    // Boundary-write (commit-advance): fold only the confirmed prefix into the
    // conv state; the trailing-K window then lands at the accepted boundary.
    if (commit_len != nullptr) {
        const int c = commit_len[r];
        if (c < Nr) Nr = c;
    }
    if (Nr <= 0) return;

    const int slot = slot_ids[r];
    if (slot < 0) return;
    const T* x_r = x + (long long)t0 * C;
    T* y_r = y + (long long)t0 * C;
    T* state = state_out_base + (long long)slot * slot_stride_elems;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const float bias_v = bias ? Elem<T>::to_f32(bias[c]) : 0.f;

    for (int t = tid; t < Nr; t += block_size) {
        float acc = bias_v;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (k >= K) break;
            const int src_t = t - (K - 1) + k;
            float xv = 0.f;
            if (src_t < 0) {
                xv = Elem<T>::to_f32(state[(K + src_t) * C + c]);
            } else {
                xv = Elem<T>::to_f32(x_r[src_t * C + c]);
            }
            const float wv = Elem<T>::to_f32(weight[c * K + k]);
            acc += wv * xv;
        }
        y_r[t * C + c] = Elem<T>::from_f32(silu_f(acc));
    }

    __syncthreads();

    // Frozen verify (write_state=false): leave the committed conv state at its
    // pre-verify value; the repair forward advances it through [input|accepted].
    if (state_out_base && write_state &&
        (write_state_mask == nullptr || write_state_mask[r] != 0) &&
        tid == 0) {
        for (int s = 0; s < K; ++s) {
            const int src_t = Nr - K + s;
            const float v = (src_t < 0)
                ? Elem<T>::to_f32(state[(K + src_t) * C + c])
                : Elem<T>::to_f32(x_r[src_t * C + c]);
            state[s * C + c] = Elem<T>::from_f32(v);
        }
    }
}

/// Multi-request batched prefill optimized for large request cohorts with
/// short prompts. One block covers a contiguous channel tile for one request;
/// each thread owns one channel and walks that request's tokens serially. This
/// avoids launching one tiny block per (request, channel) while keeping the
/// per-channel recurrence and state update identical to the reference kernel.
///
/// **Unrowed**, for `causal_conv1d_prefill_batched`'s reasons — and for one
/// more that is the launcher's rather than the kernel's: which of the two the
/// AOT path runs is decided at run time on `R >= 8`, so a JIT row for either
/// states half a contract. Under a JIT that choice is a row, or two rows and a
/// compile-time constant, and which of those it should be is a decision about
/// the kernel rather than a substitution.
template <class T>
__global__ void causal_conv1d_prefill_batched_channel_tile(
    const T* __restrict__ x,                 // [N_total, C]
    const T* __restrict__ weight,            // [C, K]
    const T* __restrict__ bias,              // [C]
    T* __restrict__ y,                       // [N_total, C]
    T* __restrict__ state_out_base,          // [num_slots, K, C]
    const int* __restrict__ slot_ids,        // [R]
    const u32* __restrict__ qo_indptr,       // [R+1]
    long long slot_stride_elems,
    int C, int K, bool write_state,
    const u8* __restrict__ write_state_mask,
    const int* commit_len)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (c >= C) return;

    const int t0 = static_cast<int>(qo_indptr[r]);
    int Nr = static_cast<int>(qo_indptr[r + 1]) - t0;
    // Boundary-write (commit-advance): fold only the confirmed prefix into the
    // conv state; the trailing-K window then lands at the accepted boundary.
    if (commit_len != nullptr) {
        const int c = commit_len[r];
        if (c < Nr) Nr = c;
    }
    if (Nr <= 0) return;

    const int slot = slot_ids[r];
    if (slot < 0) return;
    const T* x_r = x + static_cast<long long>(t0) * C;
    T* y_r = y + static_cast<long long>(t0) * C;
    T* state = state_out_base + static_cast<long long>(slot) * slot_stride_elems;

    const float bias_v = bias ? Elem<T>::to_f32(bias[c]) : 0.f;
    float wv[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        wv[k] = (k < K) ? Elem<T>::to_f32(weight[c * K + k]) : 0.f;
    }

    for (int t = 0; t < Nr; ++t) {
        float acc = bias_v;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (k >= K) break;
            const int src_t = t - (K - 1) + k;
            float xv = 0.f;
            if (src_t < 0) {
                xv = Elem<T>::to_f32(state[(K + src_t) * C + c]);
            } else {
                xv = Elem<T>::to_f32(x_r[src_t * C + c]);
            }
            acc += wv[k] * xv;
        }
        y_r[static_cast<long long>(t) * C + c] = Elem<T>::from_f32(silu_f(acc));
    }

    // Frozen verify (write_state=false): see the reference kernel above.
    if (state_out_base && write_state &&
        (write_state_mask == nullptr || write_state_mask[r] != 0)) {
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s >= K) break;
            const int src_t = Nr - K + s;
            const float v = (src_t < 0)
                ? Elem<T>::to_f32(state[(K + src_t) * C + c])
                : Elem<T>::to_f32(x_r[src_t * C + c]);
            state[s * C + c] = Elem<T>::from_f32(v);
        }
    }
}

/// Multi-request batched variant. Same math as the single-request
/// kernel; an outer R dimension picks the per-request input/output row
/// and the per-request slot in the state buffer. One block per
/// (request, channel-tile); threads parallelise channels in the tile.
///
/// **Unrowed.** The grid's second axis is the REQUEST, which no rule has, and
/// the row would take a `long long` slot stride the binder cannot marshal.
/// The single-request twin above is what a row can say today; this is what a
/// batch needs, and the gap between them is a `Ty::I64` and a grid axis rather
/// than anything about the arithmetic.
template <class T>
__global__ void causal_conv1d_update_batched(
    const T* __restrict__ x,                 // [R, C]
    const T* __restrict__ weight,            // [C, K]
    const T* __restrict__ bias,              // [C] nullable
    T* __restrict__ state_base,              // [num_slots, K, C]
    const int* __restrict__ slot_ids,        // [R]
    long long slot_stride_elems,             // K * C
    T* __restrict__ y,                       // [R, C]
    int R, int C, int K)
{
    const int r = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R || c >= C) return;

    const int slot = slot_ids[r];
    if (slot < 0) return;
    T* state = state_base + (long long)slot * slot_stride_elems;
    const T* x_r = x + (long long)r * C;
    T* y_r = y + (long long)r * C;

    const float bias_v = bias ? Elem<T>::to_f32(bias[c]) : 0.f;
    const float new_x  = Elem<T>::to_f32(x_r[c]);

    float acc = bias_v;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k >= K) break;
        float xv;
        if (k < K - 1) {
            xv = Elem<T>::to_f32(state[(k + 1) * C + c]);
        } else {
            xv = new_x;
        }
        const float wv = Elem<T>::to_f32(weight[c * K + k]);
        acc += wv * xv;
    }
    y_r[c] = Elem<T>::from_f32(silu_f(acc));

    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k >= K - 1) break;
        state[k * C + c] = state[(k + 1) * C + c];
    }
    state[(K - 1) * C + c] = Elem<T>::from_f32(new_x);
}

}  // namespace pie_cuda_driver::kernels::ssm::device
