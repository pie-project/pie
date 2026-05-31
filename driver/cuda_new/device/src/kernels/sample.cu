#include "sample.cuh"

#include <cuda_bf16.h>
#include <cstdint>

// Lifted from driver/cuda/src/kernels/sample_temp.cu (the
// `launch_sample_temp_bf16` variant only). Verbatim apart from the namespace
// (`pie_cuda_driver` -> `pie_cuda_device`) and the dropped `launch_` prefix on
// the entry point. The SplitMix-style per-element hash (`hash_uniform`) and
// the Gumbel transform (`gumbel_noise`) are pulled in unchanged because the
// kernel uses them transitively. The other variants (_compact_scatter,
// argmax_*_with_offset, select_global_argmax*) are not lifted here.

namespace pie_cuda_device::kernels {

namespace {

constexpr int BLOCK = 256;

// SplitMix64 → [0, 1) float. Mixes a row seed with the column index so
// concurrent threads don't share PRNG state but each (seed, j) pair is
// stable across runs.
__device__ __forceinline__ float hash_uniform(std::uint64_t seed, int j) {
    std::uint64_t x = seed + 0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(j + 1);
    x ^= x >> 27; x *= 0x3C79AC492BA7B653ULL;
    x ^= x >> 33; x *= 0x1C69B3F74AC4AE35ULL;
    x ^= x >> 27;
    // High 24 bits → [0, 1). Avoids exact 0 by masking and offsetting.
    const std::uint32_t bits = static_cast<std::uint32_t>(x >> 40);
    return (static_cast<float>(bits) + 0.5f) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ float gumbel_noise(std::uint64_t seed, int j) {
    const float u = hash_uniform(seed, j);
    return -logf(-logf(u));
}

// Block-wide sum reduction of `val` through `buf` (BLOCK floats). Returns the
// total to every thread; leaves a trailing __syncthreads so `buf` is reusable.
__device__ __forceinline__ float block_sum(float* buf, int tid, float val) {
    buf[tid] = val;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) buf[tid] += buf[tid + off];
        __syncthreads();
    }
    const float total = buf[0];
    __syncthreads();
    return total;
}

__global__ void sample_temp_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const float* __restrict__ temperatures,
    const float* __restrict__ top_ps,
    const std::int32_t* __restrict__ top_ks,
    const float* __restrict__ min_ps,
    const std::uint32_t* __restrict__ seeds,
    std::int32_t* __restrict__ out,
    int vocab)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const __nv_bfloat16* row_ptr = logits + static_cast<long long>(row) * vocab;

    const float T = temperatures[row];
    const bool greedy = !(T > 0.f);
    const float inv_T = greedy ? 1.f : (1.f / T);
    const float min_p = min_ps ? min_ps[row] : 0.f;
    const float top_p = top_ps ? top_ps[row] : 1.f;
    const int   top_k = top_ks ? top_ks[row] : 0;
    const bool apply_min_p = (min_p > 0.f) && !greedy;
    const bool apply_top_p = (top_p > 0.f) && (top_p < 1.f) && !greedy;
    const bool apply_top_k = (top_k > 0) && !greedy;
    const bool need_max = apply_min_p || apply_top_p || apply_top_k;
    const std::uint64_t seed = static_cast<std::uint64_t>(seeds[row]) ^ 0xa5a5a5a5ull;

    __shared__ float reduce_buf[BLOCK];

    // Combined logit cutoff: tokens below `threshold` are excluded from the
    // draw. min-p / top-p / top-k each contribute a cutoff; the most
    // restrictive (largest) wins (= intersection of the kept sets).
    float threshold = -INFINITY;
    float max_logit = INFINITY;

    if (need_max) {
        float local_max = -INFINITY;
        for (int j = tid; j < vocab; j += BLOCK)
            local_max = fmaxf(local_max, __bfloat162float(row_ptr[j]));
        reduce_buf[tid] = local_max;
        __syncthreads();
        for (int off = BLOCK / 2; off > 0; off >>= 1) {
            if (tid < off) reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + off]);
            __syncthreads();
        }
        max_logit = reduce_buf[0];
        __syncthreads();
    }

    // ── min-p cutoff: logit >= max_logit + log(min_p).
    if (apply_min_p) threshold = fmaxf(threshold, max_logit + logf(min_p));

    // ── top-p cutoff: largest cutoff whose retained softmax mass still >= p.
    // Work in unnormalized exp(logit - max_logit); compare against p * Z.
    if (apply_top_p) {
        float local_z = 0.f;
        for (int j = tid; j < vocab; j += BLOCK)
            local_z += __expf(__bfloat162float(row_ptr[j]) - max_logit);
        const float target = top_p * block_sum(reduce_buf, tid, local_z);
        // Invariant: mass(lo) >= target (lo starts low enough to keep ~all).
        float lo = max_logit - 80.f, hi = max_logit;
        for (int it = 0; it < 40; ++it) {
            const float mid = 0.5f * (lo + hi);
            float local_s = 0.f;
            for (int j = tid; j < vocab; j += BLOCK) {
                const float lg = __bfloat162float(row_ptr[j]);
                if (lg >= mid) local_s += __expf(lg - max_logit);
            }
            const float mass = block_sum(reduce_buf, tid, local_s);
            if (mass >= target) lo = mid; else hi = mid;  // raise cutoff while covered
        }
        threshold = fmaxf(threshold, lo);
    }

    // ── top-k cutoff: largest cutoff retaining >= k tokens (the k-th logit).
    if (apply_top_k && top_k < vocab) {
        float lo = max_logit - 80.f, hi = max_logit;
        for (int it = 0; it < 40; ++it) {
            const float mid = 0.5f * (lo + hi);
            int local_c = 0;
            for (int j = tid; j < vocab; j += BLOCK)
                if (__bfloat162float(row_ptr[j]) >= mid) local_c++;
            const int count = static_cast<int>(block_sum(reduce_buf, tid, static_cast<float>(local_c)));
            if (count >= top_k) lo = mid; else hi = mid;  // raise cutoff while >= k kept
        }
        threshold = fmaxf(threshold, lo);
    }

    // ── Gumbel-max over the (possibly filtered) logits.
    float best_val = -INFINITY;
    int   best_idx = 0;

    for (int j = tid; j < vocab; j += BLOCK) {
        const float logit = __bfloat162float(row_ptr[j]);
        if (logit < threshold) continue;
        const float score = greedy
            ? logit
            : (logit * inv_T + gumbel_noise(seed, j));
        if (score > best_val || (score == best_val && j < best_idx)) {
            best_val = score;
            best_idx = j;
        }
    }

    __shared__ float vals[BLOCK];
    __shared__ int   idxs[BLOCK];
    vals[tid] = best_val;
    idxs[tid] = best_idx;
    __syncthreads();

    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) {
            const float ov = vals[tid + off];
            const int   oi = idxs[tid + off];
            if (ov > vals[tid] || (ov == vals[tid] && oi < idxs[tid])) {
                vals[tid] = ov;
                idxs[tid] = oi;
            }
        }
        __syncthreads();
    }

    if (tid == 0) out[row] = idxs[0];
}

}  // namespace

void sample_temp_bf16(
    const void* logits,
    const float* temperatures,
    const float* top_ps,
    const std::int32_t* top_ks,
    const float* min_ps,
    const std::uint32_t* seeds,
    std::int32_t* out,
    int num_rows, int vocab,
    cudaStream_t stream)
{
    dim3 grid(num_rows);
    dim3 block(BLOCK);
    sample_temp_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits),
        temperatures, top_ps, top_ks, min_ps, seeds, out, vocab);
}

}  // namespace pie_cuda_device::kernels
