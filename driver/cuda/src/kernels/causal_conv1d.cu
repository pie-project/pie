#include "kernels/causal_conv1d.hpp"

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

__device__ __forceinline__ float silu_f(float z) {
    return z / (1.f + __expf(-z));
}

// One block per (channel, output token range). Each thread handles a
// few output tokens in its block. The kernel size K is small (4 on
// Qwen3.5), so the K accumulator unrolls trivially.
//
//     y[t, c] = silu( sum_{k=0..K-1} W[c, k] * x[t - K + 1 + k, c]  + bias[c] )
//
// where `x[t<0, c]` is read from the prior state window. Fresh prompts
// arrive with a zeroed state window, so this also implements causal
// padding for first-chunk prefill. The trailing K input rows are written
// back into `state_out[K, C]` (oldest first) so a follow-up decode or
// mixed prefill chunk can resume from there.
__global__ void causal_conv1d_prefill_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ y,
    __nv_bfloat16* __restrict__ state_out,
    int N, int C, int K)
{
    const int c = blockIdx.x;       // one channel per block
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    if (c >= C) return;

    const float bias_v = bias ? __bfloat162float(bias[c]) : 0.f;

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
                    xv = __bfloat162float(state_out[(K + src_t) * C + c]);
                }
            } else {
                xv = __bfloat162float(x[src_t * C + c]);
            }
            const float wv = __bfloat162float(weight[c * K + k]);
            acc += wv * xv;
        }
        y[t * C + c] = __float2bfloat16(silu_f(acc));
    }

    __syncthreads();

    // Persist the trailing K input rows into state_out (one thread does
    // this per channel; it's a tiny copy with strided indexing).
    if (state_out && tid == 0) {
        for (int s = 0; s < K; ++s) {
            const int src_t = N - K + s;  // token index for state slot s
            const float v = (src_t < 0)
                ? __bfloat162float(state_out[(K + src_t) * C + c])
                : __bfloat162float(x[src_t * C + c]);
            state_out[s * C + c] = __float2bfloat16(v);
        }
    }
}

// Decode update: state_in[K, C] holds the last K input rows; new x is
// one row. After this kernel:
//   • y[c] = silu( sum_{k=0..K-1} W[c, k] * (k<K-1 ? state_in[k+1, c] : x[c])
//                 + bias[c] )
//   • state[K, C] is shifted: state[k] := state[k+1] for k<K-1, state[K-1] := x.
__global__ void causal_conv1d_update_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ y,
    int C, int K)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    const float bias_v = bias ? __bfloat162float(bias[c]) : 0.f;
    const float new_x  = __bfloat162float(x[c]);

    // Compute output: convolve over the K-window [state[1], ..., state[K-1], x].
    float acc = bias_v;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k >= K) break;
        float xv;
        if (k < K - 1) {
            xv = __bfloat162float(state[(k + 1) * C + c]);
        } else {
            xv = new_x;
        }
        const float wv = __bfloat162float(weight[c * K + k]);
        acc += wv * xv;
    }
    y[c] = __float2bfloat16(silu_f(acc));

    // Update state: shift left by 1, new_x in the last slot.
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k >= K - 1) break;
        state[k * C + c] = state[(k + 1) * C + c];
    }
    state[(K - 1) * C + c] = __float2bfloat16(new_x);
}

// Multi-request batched prefill. Per-(channel, request) block; threads
// stride through that request's tokens. Same math as the single-request
// kernel; the (t0_r, Nr_r) window is read from qo_indptr at runtime,
// source rows before that window are read from the request's existing
// state slab, and the trailing K-window is persisted back to that slab.
__global__ void causal_conv1d_prefill_batched_kernel(
    const __nv_bfloat16* __restrict__ x,           // [N_total, C]
    const __nv_bfloat16* __restrict__ weight,      // [C, K]
    const __nv_bfloat16* __restrict__ bias,        // [C]
    __nv_bfloat16* __restrict__ y,                 // [N_total, C]
    __nv_bfloat16* __restrict__ state_out_base,    // [num_slots, K, C]
    const int*       __restrict__ slot_ids,        // [R]
    const std::uint32_t* __restrict__ qo_indptr,   // [R+1]
    long long slot_stride_elems,
    int C, int K, bool write_state,
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
    const __nv_bfloat16* x_r = x + (long long)t0 * C;
    __nv_bfloat16* y_r = y + (long long)t0 * C;
    __nv_bfloat16* state =
        state_out_base + (long long)slot * slot_stride_elems;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const float bias_v = bias ? __bfloat162float(bias[c]) : 0.f;

    for (int t = tid; t < Nr; t += block_size) {
        float acc = bias_v;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (k >= K) break;
            const int src_t = t - (K - 1) + k;
            float xv = 0.f;
            if (src_t < 0) {
                xv = __bfloat162float(state[(K + src_t) * C + c]);
            } else {
                xv = __bfloat162float(x_r[src_t * C + c]);
            }
            const float wv = __bfloat162float(weight[c * K + k]);
            acc += wv * xv;
        }
        y_r[t * C + c] = __float2bfloat16(silu_f(acc));
    }

    __syncthreads();

    // Frozen verify (write_state=false): leave the committed conv state at its
    // pre-verify value; the repair forward advances it through [input|accepted].
    if (state_out_base && write_state && tid == 0) {
        for (int s = 0; s < K; ++s) {
            const int src_t = Nr - K + s;
            const float v = (src_t < 0)
                ? __bfloat162float(state[(K + src_t) * C + c])
                : __bfloat162float(x_r[src_t * C + c]);
            state[s * C + c] = __float2bfloat16(v);
        }
    }
}

// Multi-request batched prefill optimized for large request cohorts with
// short prompts. One block covers a contiguous channel tile for one request;
// each thread owns one channel and walks that request's tokens serially. This
// avoids launching one tiny block per (request, channel) while keeping the
// per-channel recurrence and state update identical to the reference kernel.
__global__ void causal_conv1d_prefill_batched_channel_tile_kernel(
    const __nv_bfloat16* __restrict__ x,           // [N_total, C]
    const __nv_bfloat16* __restrict__ weight,      // [C, K]
    const __nv_bfloat16* __restrict__ bias,        // [C]
    __nv_bfloat16* __restrict__ y,                 // [N_total, C]
    __nv_bfloat16* __restrict__ state_out_base,    // [num_slots, K, C]
    const int*       __restrict__ slot_ids,        // [R]
    const std::uint32_t* __restrict__ qo_indptr,   // [R+1]
    long long slot_stride_elems,
    int C, int K, bool write_state,
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
    const __nv_bfloat16* x_r = x + static_cast<long long>(t0) * C;
    __nv_bfloat16* y_r = y + static_cast<long long>(t0) * C;
    __nv_bfloat16* state =
        state_out_base + static_cast<long long>(slot) * slot_stride_elems;

    const float bias_v = bias ? __bfloat162float(bias[c]) : 0.f;
    float wv[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        wv[k] = (k < K) ? __bfloat162float(weight[c * K + k]) : 0.f;
    }

    for (int t = 0; t < Nr; ++t) {
        float acc = bias_v;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (k >= K) break;
            const int src_t = t - (K - 1) + k;
            float xv = 0.f;
            if (src_t < 0) {
                xv = __bfloat162float(state[(K + src_t) * C + c]);
            } else {
                xv = __bfloat162float(x_r[src_t * C + c]);
            }
            acc += wv[k] * xv;
        }
        y_r[static_cast<long long>(t) * C + c] = __float2bfloat16(silu_f(acc));
    }

    // Frozen verify (write_state=false): see the reference kernel above.
    if (state_out_base && write_state) {
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s >= K) break;
            const int src_t = Nr - K + s;
            const float v = (src_t < 0)
                ? __bfloat162float(state[(K + src_t) * C + c])
                : __bfloat162float(x_r[src_t * C + c]);
            state[s * C + c] = __float2bfloat16(v);
        }
    }
}

// Multi-request batched variant. Same math as the single-request
// kernel; an outer R dimension picks the per-request input/output row
// and the per-request slot in the state buffer. One block per
// (request, channel-tile); threads parallelise channels in the tile.
__global__ void causal_conv1d_update_batched_kernel(
    const __nv_bfloat16* __restrict__ x,           // [R, C]
    const __nv_bfloat16* __restrict__ weight,      // [C, K]
    const __nv_bfloat16* __restrict__ bias,        // [C] nullable
    __nv_bfloat16* __restrict__ state_base,        // [num_slots, K, C]
    const int* __restrict__ slot_ids,              // [R]
    long long slot_stride_elems,                   // K * C
    __nv_bfloat16* __restrict__ y,                 // [R, C]
    int R, int C, int K)
{
    const int r = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R || c >= C) return;

    const int slot = slot_ids[r];
    if (slot < 0) return;
    __nv_bfloat16* state = state_base + (long long)slot * slot_stride_elems;
    const __nv_bfloat16* x_r = x + (long long)r * C;
    __nv_bfloat16* y_r = y + (long long)r * C;

    const float bias_v = bias ? __bfloat162float(bias[c]) : 0.f;
    const float new_x  = __bfloat162float(x_r[c]);

    float acc = bias_v;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k >= K) break;
        float xv;
        if (k < K - 1) {
            xv = __bfloat162float(state[(k + 1) * C + c]);
        } else {
            xv = new_x;
        }
        const float wv = __bfloat162float(weight[c * K + k]);
        acc += wv * xv;
    }
    y_r[c] = __float2bfloat16(silu_f(acc));

    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k >= K - 1) break;
        state[k * C + c] = state[(k + 1) * C + c];
    }
    state[(K - 1) * C + c] = __float2bfloat16(new_x);
}

}  // namespace

void launch_causal_conv1d_prefill_bf16(
    const void* x, const void* weight, const void* bias,
    void* y, void* state_out,
    int N, int C, int K, cudaStream_t stream)
{
    if (N <= 0 || C <= 0 || K <= 0) return;
    constexpr int BLOCK = 64;
    dim3 grid(C);
    dim3 block(BLOCK);
    causal_conv1d_prefill_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<const __nv_bfloat16*>(bias),
        static_cast<__nv_bfloat16*>(y),
        static_cast<__nv_bfloat16*>(state_out),
        N, C, K);
}

void launch_causal_conv1d_update_bf16(
    const void* x, const void* weight, const void* bias,
    void* state, void* y,
    int C, int K, cudaStream_t stream)
{
    if (C <= 0 || K <= 0) return;
    constexpr int BLOCK = 128;
    dim3 grid((C + BLOCK - 1) / BLOCK);
    dim3 block(BLOCK);
    causal_conv1d_update_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<const __nv_bfloat16*>(bias),
        static_cast<__nv_bfloat16*>(state),
        static_cast<__nv_bfloat16*>(y),
        C, K);
}

void launch_causal_conv1d_update_batched_bf16(
    const void* x, const void* weight, const void* bias,
    void* state_base,
    const std::int32_t* slot_ids,
    long long slot_stride_elems,
    void* y,
    int R, int C, int K, cudaStream_t stream)
{
    if (R <= 0 || C <= 0 || K <= 0) return;
    constexpr int BLOCK = 128;
    dim3 grid((C + BLOCK - 1) / BLOCK, R);
    dim3 block(BLOCK);
    causal_conv1d_update_batched_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<const __nv_bfloat16*>(bias),
        static_cast<__nv_bfloat16*>(state_base),
        slot_ids,
        slot_stride_elems,
        static_cast<__nv_bfloat16*>(y),
        R, C, K);
}

void launch_causal_conv1d_prefill_batched_bf16(
    const void* x, const void* weight, const void* bias,
    void* y, void* state_out_base,
    const std::int32_t* slot_ids,
    const std::uint32_t* qo_indptr,
    long long slot_stride_elems,
    int R, int C, int K, cudaStream_t stream, bool write_state,
    const int* commit_len)
{
    if (R <= 0 || C <= 0 || K <= 0) return;
    if (R >= 8) {
        constexpr int TILE = 128;
        dim3 grid((C + TILE - 1) / TILE, R);
        dim3 block(TILE);
        causal_conv1d_prefill_batched_channel_tile_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x),
            static_cast<const __nv_bfloat16*>(weight),
            static_cast<const __nv_bfloat16*>(bias),
            static_cast<__nv_bfloat16*>(y),
            static_cast<__nv_bfloat16*>(state_out_base),
            slot_ids, qo_indptr,
            slot_stride_elems,
            C, K, write_state, commit_len);
        return;
    }
    constexpr int BLOCK = 64;
    dim3 grid(C, R);
    dim3 block(BLOCK);
    causal_conv1d_prefill_batched_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<const __nv_bfloat16*>(bias),
        static_cast<__nv_bfloat16*>(y),
        static_cast<__nv_bfloat16*>(state_out_base),
        slot_ids, qo_indptr,
        slot_stride_elems,
        C, K, write_state, commit_len);
}

}  // namespace pie_cuda_driver::kernels
