#include "kernels/kda.hpp"

#include <algorithm>

#include <cuda_bf16.h>
#include <cstdint>

namespace pie_cuda_driver::kernels {

namespace {

__device__ __forceinline__ float sigmoidf(float x) {
    return 1.f / (1.f + __expf(-x));
}

// ── Gate + beta ────────────────────────────────────────────────────

__global__ void kda_gate_beta_kernel(
    const __nv_bfloat16* __restrict__ raw_g,
    const __nv_bfloat16* __restrict__ raw_beta,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ gate_out,
    float* __restrict__ beta_out,
    int T, int H, int D,
    float lower_bound)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    if (t >= T || h >= H) return;

    const float a = __expf(A_log[h]);
    const long long base = ((long long)t * H + h) * D;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float g = __bfloat162float(raw_g[base + d]) + dt_bias[(long long)h * D + d];
        float gate;
        if (lower_bound < 0.f) {
            gate = lower_bound * sigmoidf(a * g);
        } else {
            // softplus, guarded the way the reference kernel guards it: past
            // 20 the exp overflows and softplus(x) == x to fp32 precision.
            const float sp = (g > 20.f) ? g : __logf(1.f + __expf(g));
            gate = -a * sp;
        }
        gate_out[base + d] = gate;
    }

    if (threadIdx.x == 0) {
        beta_out[(long long)t * H + h] =
            sigmoidf(__bfloat162float(raw_beta[(long long)t * H + h]));
    }
}

// ── Recurrence ─────────────────────────────────────────────────────

// One block per (request, head), one **warp** per `v` row.
//
// The obvious mapping -- a thread per `v`, looping over `k` -- reads
// `state[v][k]` with 32 threads at 32 different rows, so every warp load
// touches 32 cache lines and returns four useful bytes from each. Giving a
// warp the row instead makes the lanes walk `k` contiguously, which is one
// line per load, and turns the two reductions into shuffles.
__global__ void kda_recurrent_step_batched_kernel(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ state_base,
    const std::int32_t* __restrict__ slot_ids,
    long long slot_stride_elems,
    float* __restrict__ out,
    int H, int D)
{
    const int r = blockIdx.x;
    const int h = blockIdx.y;

    const long long rh = (long long)r * H + h;
    const float* q_h = q_norm + rh * D;
    const float* k_h = k_norm + rh * D;
    const float* v_h = v      + rh * D;
    const float* g_h = gate   + rh * D;
    const float beta_h = beta[rh];

    float* st = state_base + (long long)slot_ids[r] * slot_stride_elems +
                (long long)h * D * D;
    float* out_h = out + rh * D;

    // q, k and the decay are read once per row, so stage them in shared
    // memory rather than re-reading them D times from L2.
    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + D;
    float* sg = smem + 2 * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
        sg[i] = __expf(g_h[i]);
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;

    for (int vi = warp; vi < D; vi += warps) {
        float* row = st + (long long)vi * D;
        float mem = 0.f;
        for (int ki = lane; ki < D; ki += 32) {
            const float sv = row[ki] * sg[ki];
            row[ki] = sv;
            mem += sv * sk[ki];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            mem += __shfl_down_sync(0xffffffffu, mem, off);
        }
        const float delta = __shfl_sync(0xffffffffu, (v_h[vi] - mem) * beta_h, 0);

        float acc = 0.f;
        for (int ki = lane; ki < D; ki += 32) {
            const float sv = row[ki] + sk[ki] * delta;
            row[ki] = sv;
            acc += sv * sq[ki];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        if (lane == 0) out_h[vi] = acc;
    }
}

// One block per (request, head); the block walks its whole window because the
// recurrence has a strict per-token state dependency. Same warp-per-`v` row
// mapping as the decode step, for the same coalescing reason.
__global__ void kda_prefill_batched_kernel(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ state_base,
    const std::int32_t* __restrict__ slot_ids,
    const std::uint32_t* __restrict__ qo_indptr,
    long long slot_stride_elems,
    float* __restrict__ out,
    int H, int D)
{
    const int r = blockIdx.x;
    const int h = blockIdx.y;

    const long long begin = qo_indptr[r];
    const long long end = qo_indptr[r + 1];
    if (end <= begin) return;

    float* st = state_base + (long long)slot_ids[r] * slot_stride_elems +
                (long long)h * D * D;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + D;
    float* sg = smem + 2 * D;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;

    for (long long t = begin; t < end; ++t) {
        const long long th = t * H + h;
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            sq[i] = q_norm[th * D + i];
            sk[i] = k_norm[th * D + i];
            sg[i] = __expf(gate[th * D + i]);
        }
        __syncthreads();

        const float beta_h = beta[th];
        for (int vi = warp; vi < D; vi += warps) {
            float* row = st + (long long)vi * D;
            float mem = 0.f;
            for (int ki = lane; ki < D; ki += 32) {
                const float sv = row[ki] * sg[ki];
                row[ki] = sv;
                mem += sv * sk[ki];
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                mem += __shfl_down_sync(0xffffffffu, mem, off);
            }
            const float delta =
                __shfl_sync(0xffffffffu, (v[th * D + vi] - mem) * beta_h, 0);

            float acc = 0.f;
            for (int ki = lane; ki < D; ki += 32) {
                const float sv = row[ki] + sk[ki] * delta;
                row[ki] = sv;
                acc += sv * sq[ki];
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                acc += __shfl_down_sync(0xffffffffu, acc, off);
            }
            if (lane == 0) out[th * D + vi] = acc;
        }
        // The next token reads the state this one just wrote, and reloads
        // shared memory over the values this one is still reading.
        __syncthreads();
    }
}

// ── Gated output norm ──────────────────────────────────────────────

// One block per (token, head). RMS over the head's D channels, then scale by
// `weight` and the sigmoid of the gate.
__global__ void kda_o_norm_gated_kernel(
    const float* __restrict__ o,
    const __nv_bfloat16* __restrict__ g,
    const float* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int H, int D, float eps)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const long long base = ((long long)t * H + h) * D;

    float acc = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float x = o[base + d];
        acc += x * x;
    }
    __shared__ float ssum;
    // Block-wide reduction through shared memory; D is 128 on K3, so one
    // warp-level reduction plus a single accumulator is enough.
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (threadIdx.x == 0) ssum = 0.f;
    __syncthreads();
    if ((threadIdx.x & (warpSize - 1)) == 0) atomicAdd(&ssum, acc);
    __syncthreads();

    const float scale = rsqrtf(ssum / static_cast<float>(D) + eps);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float gate = __bfloat162float(g[base + d]);
        const float y = o[base + d] * scale * weight[d] * sigmoidf(gate);
        out[base + d] = __float2bfloat16(y);
    }
}

}  // namespace

// ── Launchers ──────────────────────────────────────────────────────

void launch_kda_gate_beta_bf16(
    const void* raw_g, const void* raw_beta, const float* A_log,
    const float* dt_bias, float* gate_out, float* beta_out,
    int T, int H, int D, float lower_bound, cudaStream_t stream)
{
    if (T <= 0 || H <= 0 || D <= 0) return;
    const dim3 grid(static_cast<unsigned>(T), static_cast<unsigned>(H));
    const int threads = D < 256 ? D : 256;
    kda_gate_beta_kernel<<<grid, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(raw_g),
        static_cast<const __nv_bfloat16*>(raw_beta),
        A_log, dt_bias, gate_out, beta_out, T, H, D, lower_bound);
}

void launch_kda_recurrent_step_batched(
    const float* q_norm, const float* k_norm, const float* v,
    const float* gate, const float* beta, float* state_base,
    const std::int32_t* slot_ids, long long slot_stride_elems, float* out,
    int R, int H, int D, cudaStream_t stream)
{
    if (R <= 0 || H <= 0 || D <= 0) return;
    const dim3 grid(static_cast<unsigned>(R), static_cast<unsigned>(H));
    // A multiple of the warp size: the kernel gives one warp a `v` row.
    const int threads = 256;
    const std::size_t shmem = static_cast<std::size_t>(3 * D) * sizeof(float);
    kda_recurrent_step_batched_kernel<<<grid, threads, shmem, stream>>>(
        q_norm, k_norm, v, gate, beta, state_base, slot_ids,
        slot_stride_elems, out, H, D);
}

void launch_kda_prefill_batched(
    const float* q_norm, const float* k_norm, const float* v,
    const float* gate, const float* beta, float* state_base,
    const std::int32_t* slot_ids, const std::uint32_t* qo_indptr,
    long long slot_stride_elems, float* out,
    int R, int H, int D, cudaStream_t stream)
{
    if (R <= 0 || H <= 0 || D <= 0) return;
    const dim3 grid(static_cast<unsigned>(R), static_cast<unsigned>(H));
    // The recurrence serializes over tokens, so the only parallelism a block
    // has is across the state's `v` rows -- one warp each, `D / warps` rows per
    // warp per token. At 256 threads a 128-row state gives every warp 16 rows
    // to walk in sequence, and with a grid of only R*H blocks the whole kernel
    // was using a tenth of the machine. Widening the block is the entire fix:
    // 2.2x at T=2048 (26.2 ms -> 12.0 ms per layer, measured at K3's widths).
    // One warp per row is the useful limit; beyond that warps sit idle.
    const int warps = std::min(32, D);
    const int threads = warps * 32;
    const std::size_t shmem = static_cast<std::size_t>(3 * D) * sizeof(float);
    kda_prefill_batched_kernel<<<grid, threads, shmem, stream>>>(
        q_norm, k_norm, v, gate, beta, state_base, slot_ids, qo_indptr,
        slot_stride_elems, out, H, D);
}

void launch_kda_o_norm_gated_bf16(
    const float* o, const void* g, const float* weight, void* out,
    int T, int H, int D, float eps, cudaStream_t stream)
{
    if (T <= 0 || H <= 0 || D <= 0) return;
    const dim3 grid(static_cast<unsigned>(T), static_cast<unsigned>(H));
    const int threads = D < 256 ? D : 256;
    kda_o_norm_gated_kernel<<<grid, threads, 0, stream>>>(
        o, static_cast<const __nv_bfloat16*>(g), weight,
        static_cast<__nv_bfloat16*>(out), H, D, eps);
}

}  // namespace pie_cuda_driver::kernels
