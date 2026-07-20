#include "kernels/nemotron_h.hpp"

#include <cstdlib>

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

__device__ __forceinline__ float bf16_to_float(const __nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float v) {
    return __float2bfloat16(v);
}

__device__ __forceinline__ float softplus_f(float x) {
    // Stable enough for the dt range in Nemotron-H checkpoints.
    return x > 20.f ? x : log1pf(__expf(x));
}

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.f + __expf(-x));
}

__device__ __forceinline__ float warp_sum(float v) {
    unsigned mask = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(mask, v, off);
    }
    return v;
}

__device__ __forceinline__ float warp_broadcast_lane0(float v) {
    return __shfl_sync(0xffffffffu, v, 0);
}

__global__ void mamba_split_kernel(
    const __nv_bfloat16* __restrict__ projected,
    __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ conv_in,
    __nv_bfloat16* __restrict__ dt,
    int projection_dim,
    int intermediate,
    int conv_dim,
    int num_heads,
    int total)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int row = i / projection_dim;
    const int col = i - row * projection_dim;
    const auto v = projected[i];
    if (col < intermediate) {
        gate[static_cast<long long>(row) * intermediate + col] = v;
    } else if (col < intermediate + conv_dim) {
        conv_in[static_cast<long long>(row) * conv_dim +
                (col - intermediate)] = v;
    } else if (col < intermediate + conv_dim + num_heads) {
        dt[static_cast<long long>(row) * num_heads +
           (col - intermediate - conv_dim)] = v;
    }
}

__global__ void mamba_split_conv_dt_kernel(
    const __nv_bfloat16* __restrict__ projected,
    __nv_bfloat16* __restrict__ conv_in,
    __nv_bfloat16* __restrict__ dt,
    int projection_dim,
    int intermediate,
    int conv_dim,
    int num_heads,
    int total)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int row = i / (conv_dim + num_heads);
    const int col = i - row * (conv_dim + num_heads);
    const __nv_bfloat16* src =
        projected + static_cast<long long>(row) * projection_dim + intermediate;
    if (col < conv_dim) {
        conv_in[static_cast<long long>(row) * conv_dim + col] = src[col];
    } else {
        dt[static_cast<long long>(row) * num_heads + (col - conv_dim)] =
            src[col];
    }
}

__global__ void prepare_mamba_params_kernel(
    const __nv_bfloat16* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ D,
    const __nv_bfloat16* __restrict__ dt_bias,
    float* __restrict__ A,
    float* __restrict__ D_f32,
    float* __restrict__ dt_bias_f32,
    int num_heads)
{
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= num_heads) return;
    A[h] = -__expf(bf16_to_float(A_log[h]));
    D_f32[h] = bf16_to_float(D[h]);
    dt_bias_f32[h] = bf16_to_float(dt_bias[h]);
}

__global__ void prepare_mamba_dt_da_kernel(
    const __nv_bfloat16* __restrict__ dt_in,
    const float* __restrict__ A,
    const float* __restrict__ dt_bias,
    float* __restrict__ dt_out,
    float* __restrict__ dA_out,
    int total,
    int num_heads,
    float time_step_min)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int h = i - (i / num_heads) * num_heads;
    const float dt = fmaxf(
        softplus_f(bf16_to_float(dt_in[i]) + dt_bias[h]),
        time_step_min);
    dt_out[i] = dt;
    dA_out[i] = __expf(dt * A[h]);
}

// Same request/head ownership as mamba_ssm_batched_kernel, but maps one warp
// to one head dimension and reduces the state axis inside the warp. This avoids
// the shared-memory atomicAdd hot path in the generic kernel.
__global__ void mamba_ssm_batched_warp_kernel(
    const __nv_bfloat16* __restrict__ conv_out,
    const __nv_bfloat16* __restrict__ dt_in,
    const float* __restrict__ A,
    const float* __restrict__ D,
    const float* __restrict__ dt_bias,
    const float* __restrict__ dt_precomputed,
    const float* __restrict__ dA_precomputed,
    __nv_bfloat16* __restrict__ state_base,
    const std::int32_t* __restrict__ slot_ids,
    const std::uint32_t* __restrict__ qo_indptr,
    __nv_bfloat16* __restrict__ y,
    int num_heads,
    int head_dim,
    int state_size,
    int n_groups,
    int conv_dim,
    int intermediate,
    float time_step_min)
{
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int num_warps = blockDim.x >> 5;
    const int t0 = static_cast<int>(qo_indptr[r]);
    const int n_tokens = static_cast<int>(qo_indptr[r + 1]) - t0;
    if (n_tokens <= 0) return;

    const int slot = slot_ids ? slot_ids[r] : 0;
    if (slot < 0) return;
    const long long state_stride =
        static_cast<long long>(num_heads) * head_dim * state_size;
    __nv_bfloat16* state =
        state_base + static_cast<long long>(slot) * state_stride +
        static_cast<long long>(h) * head_dim * state_size;

    const int heads_per_group = num_heads / n_groups;
    const int group = h / heads_per_group;
    const float A_h = A[h];
    const float D_h = D[h];
    const float dt_b = dt_bias[h];
    const int bc_base = intermediate + group * state_size;
    const int c_base = intermediate + n_groups * state_size +
                       group * state_size;

    extern __shared__ float bc_smem[];
    float* b_s = bc_smem;
    float* c_s = bc_smem + state_size;

    for (int local_t = 0; local_t < n_tokens; ++local_t) {
        const int row = t0 + local_t;
        const long long dt_idx = static_cast<long long>(row) * num_heads + h;
        const float dt = dt_precomputed != nullptr
            ? dt_precomputed[dt_idx]
            : fmaxf(softplus_f(bf16_to_float(dt_in[dt_idx]) + dt_b),
                    time_step_min);
        const float dA = dA_precomputed != nullptr
            ? dA_precomputed[dt_idx]
            : __expf(dt * A_h);
        const __nv_bfloat16* row_conv =
            conv_out + static_cast<long long>(row) * conv_dim;

        for (int s = tid; s < state_size; s += blockDim.x) {
            b_s[s] = bf16_to_float(row_conv[bc_base + s]);
            c_s[s] = bf16_to_float(row_conv[c_base + s]);
        }
        __syncthreads();

        for (int dim = warp; dim < head_dim; dim += num_warps) {
            const float x = bf16_to_float(row_conv[h * head_dim + dim]);
            float sum = 0.f;
            for (int s = lane; s < state_size; s += 32) {
                const int idx = dim * state_size + s;
                const float old = bf16_to_float(state[idx]);
                const float next = old * dA + (dt * b_s[s]) * x;
                state[idx] = float_to_bf16(next);
                sum += next * c_s[s];
            }
            sum = warp_sum(sum);
            if (lane == 0) {
                y[static_cast<long long>(row) * intermediate +
                  h * head_dim + dim] = float_to_bf16(sum + D_h * x);
            }
        }
        __syncthreads();
    }
}

// Prefill-specialized recurrent SSM. Unlike the decode-oriented warp kernel,
// this keeps each lane's slice of the recurrent state in registers across the
// full scheduled token span and writes the cache only once at the end.
__global__ void mamba_ssm_batched_prefill_reg_kernel(
    const __nv_bfloat16* __restrict__ conv_out,
    const __nv_bfloat16* __restrict__ dt_in,
    const float* __restrict__ A,
    const float* __restrict__ D,
    const float* __restrict__ dt_bias,
    const float* __restrict__ dt_precomputed,
    const float* __restrict__ dA_precomputed,
    __nv_bfloat16* __restrict__ state_base,
    const std::int32_t* __restrict__ slot_ids,
    const std::uint32_t* __restrict__ qo_indptr,
    __nv_bfloat16* __restrict__ y,
    int num_heads,
    int head_dim,
    int state_size,
    int n_groups,
    int conv_dim,
    int intermediate,
    float time_step_min)
{
    constexpr int kMaxStatePerLane = 8;
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int num_warps = blockDim.x >> 5;
    const int dim = static_cast<int>(blockIdx.z) * num_warps + warp;
    const bool active_dim = dim < head_dim;
    if (state_size > 32 * kMaxStatePerLane) return;

    const int t0 = static_cast<int>(qo_indptr[r]);
    const int n_tokens = static_cast<int>(qo_indptr[r + 1]) - t0;
    if (n_tokens <= 0) return;

    const int slot = slot_ids ? slot_ids[r] : 0;
    if (slot < 0) return;
    const long long state_stride =
        static_cast<long long>(num_heads) * head_dim * state_size;
    __nv_bfloat16* state =
        active_dim
            ? state_base + static_cast<long long>(slot) * state_stride +
                  static_cast<long long>(h) * head_dim * state_size +
                  static_cast<long long>(dim) * state_size
            : nullptr;

    float state_vals[kMaxStatePerLane];
    int state_offsets[kMaxStatePerLane];
    int state_count = 0;
    if (active_dim) for (int s = lane; s < state_size; s += 32) {
        state_offsets[state_count] = s;
        state_vals[state_count] = bf16_to_float(state[s]);
        ++state_count;
    }

    const int heads_per_group = num_heads / n_groups;
    const int group = h / heads_per_group;
    const float A_h = A[h];
    const float D_h = D[h];
    const float dt_b = dt_bias[h];
    const int x_col = h * head_dim + dim;
    const int bc_base = intermediate + group * state_size;
    const int c_base = intermediate + n_groups * state_size +
                       group * state_size;
    extern __shared__ float bc_smem[];
    float* b_s = bc_smem;
    float* c_s = bc_smem + state_size;

    for (int local_t = 0; local_t < n_tokens; ++local_t) {
        const int row = t0 + local_t;
        const __nv_bfloat16* row_conv =
            conv_out + static_cast<long long>(row) * conv_dim;
        for (int s = tid; s < state_size; s += blockDim.x) {
            b_s[s] = bf16_to_float(row_conv[bc_base + s]);
            c_s[s] = bf16_to_float(row_conv[c_base + s]);
        }
        __syncthreads();

        const long long dt_idx = static_cast<long long>(row) * num_heads + h;
        float dt_lane0 = 0.f;
        float dA_lane0 = 0.f;
        if (lane == 0) {
            dt_lane0 = dt_precomputed != nullptr
                ? dt_precomputed[dt_idx]
                : fmaxf(softplus_f(bf16_to_float(dt_in[dt_idx]) + dt_b),
                        time_step_min);
            dA_lane0 = dA_precomputed != nullptr
                ? dA_precomputed[dt_idx]
                : __expf(dt_lane0 * A_h);
        }
        const float dt = warp_broadcast_lane0(dt_lane0);
        const float dA = warp_broadcast_lane0(dA_lane0);
        const float x = active_dim ? bf16_to_float(row_conv[x_col]) : 0.f;

        float sum = 0.f;
        #pragma unroll
        for (int i = 0; i < kMaxStatePerLane; ++i) {
            if (i >= state_count) break;
            const int s = state_offsets[i];
            const float b = b_s[s];
            const float c = c_s[s];
            const float next = state_vals[i] * dA + (dt * b) * x;
            state_vals[i] = next;
            sum += next * c;
        }
        sum = warp_sum(sum);
        if (active_dim && lane == 0) {
            y[static_cast<long long>(row) * intermediate +
              h * head_dim + dim] = float_to_bf16(sum + D_h * x);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < kMaxStatePerLane; ++i) {
        if (i >= state_count) break;
        state[state_offsets[i]] = float_to_bf16(state_vals[i]);
    }
}

// One CUDA block owns one (request, mamba-head) stream. Threads cooperate
// over the 64x128 state slab for that head, preserving token order inside
// the request while still parallelizing the expensive state update.
__global__ void mamba_ssm_batched_kernel(
    const __nv_bfloat16* __restrict__ conv_out,
    const __nv_bfloat16* __restrict__ dt_in,
    const float* __restrict__ A,
    const float* __restrict__ D,
    const float* __restrict__ dt_bias,
    __nv_bfloat16* __restrict__ state_base,
    const std::int32_t* __restrict__ slot_ids,
    const std::uint32_t* __restrict__ qo_indptr,
    __nv_bfloat16* __restrict__ y,
    int num_heads,
    int head_dim,
    int state_size,
    int n_groups,
    int conv_dim,
    int intermediate,
    float time_step_min)
{
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int t0 = static_cast<int>(qo_indptr[r]);
    const int n_tokens = static_cast<int>(qo_indptr[r + 1]) - t0;
    if (n_tokens <= 0) return;

    extern __shared__ float smem[];
    float* acc = smem;  // [head_dim]

    const int slot = slot_ids ? slot_ids[r] : 0;
    if (slot < 0) return;
    const long long state_stride =
        static_cast<long long>(num_heads) * head_dim * state_size;
    __nv_bfloat16* state =
        state_base + static_cast<long long>(slot) * state_stride +
        static_cast<long long>(h) * head_dim * state_size;

    const int heads_per_group = num_heads / n_groups;
    const int group = h / heads_per_group;
    const float A_h = A[h];
    const float D_h = D[h];
    const float dt_b = dt_bias[h];

    for (int local_t = 0; local_t < n_tokens; ++local_t) {
        const int row = t0 + local_t;
        if (tid < head_dim) acc[tid] = 0.f;
        __syncthreads();

        const float dt = fmaxf(
            softplus_f(bf16_to_float(
                dt_in[static_cast<long long>(row) * num_heads + h]) + dt_b),
            time_step_min);
        const float dA = __expf(dt * A_h);
        const __nv_bfloat16* row_conv =
            conv_out + static_cast<long long>(row) * conv_dim;
        const int bc_base = intermediate + group * state_size;
        const int c_base = intermediate + n_groups * state_size +
                           group * state_size;

        for (int idx = tid; idx < head_dim * state_size; idx += blockDim.x) {
            const int dim = idx / state_size;
            const int s = idx - dim * state_size;
            const float x = bf16_to_float(row_conv[h * head_dim + dim]);
            const float b = bf16_to_float(row_conv[bc_base + s]);
            const float c = bf16_to_float(row_conv[c_base + s]);
            const float old = bf16_to_float(state[idx]);
            const float next = old * dA + (dt * b) * x;
            state[idx] = float_to_bf16(next);
            atomicAdd(acc + dim, next * c);
        }
        __syncthreads();

        for (int dim = tid; dim < head_dim; dim += blockDim.x) {
            const float x = bf16_to_float(row_conv[h * head_dim + dim]);
            y[static_cast<long long>(row) * intermediate +
              h * head_dim + dim] = float_to_bf16(acc[dim] + D_h * x);
        }
        __syncthreads();
    }
}

__global__ void zamba_rmsnorm_gated_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ y,
    int hidden,
    int gate_stride,
    int group_size,
    float eps)
{
    const int row = blockIdx.x;
    const int group = blockIdx.y;
    const int tid = threadIdx.x;
    const int base = row * hidden + group * group_size;
    const long long gate_base =
        static_cast<long long>(row) * gate_stride + group * group_size;

    float local = 0.f;
    for (int i = tid; i < group_size; i += blockDim.x) {
        const float xv = bf16_to_float(x[base + i]);
        const float gv = bf16_to_float(gate[gate_base + i]);
        const float v = xv * silu_f(gv);
        local += v * v;
    }

    __shared__ float buf[256];
    buf[tid] = local;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) buf[tid] += buf[tid + off];
        __syncthreads();
    }
    const float inv_rms = rsqrtf(buf[0] / static_cast<float>(group_size) + eps);

    for (int i = tid; i < group_size; i += blockDim.x) {
        const float xv = bf16_to_float(x[base + i]);
        const float gv = bf16_to_float(gate[gate_base + i]);
        const float v = xv * silu_f(gv) * inv_rms;
        const int h = group * group_size + i;
        y[base + i] = float_to_bf16(v * bf16_to_float(weight[h]));
    }
}

__global__ void build_nemotron_moe_ptrs_decode_batched_kernel(
    const std::int32_t* __restrict__ topk_idx,
    const float* __restrict__ topk_w,
    const __nv_bfloat16* const* __restrict__ up_weight_ptrs,
    const __nv_bfloat16* const* __restrict__ down_weight_ptrs,
    const __nv_bfloat16* __restrict__ norm_x,
    __nv_bfloat16* __restrict__ expert_up,
    __nv_bfloat16* __restrict__ expert_act,
    __nv_bfloat16* __restrict__ expert_out,
    const __nv_bfloat16** __restrict__ a_up_ptrs,
    const __nv_bfloat16** __restrict__ b_up_ptrs,
    __nv_bfloat16** __restrict__ c_up_ptrs,
    const __nv_bfloat16** __restrict__ a_down_ptrs,
    const __nv_bfloat16** __restrict__ b_down_ptrs,
    __nv_bfloat16** __restrict__ c_down_ptrs,
    float* __restrict__ weights_out,
    int total,
    int top_k,
    int hidden,
    int intermediate)
{
    const int route = blockIdx.x * blockDim.x + threadIdx.x;
    if (route >= total) return;
    const int token = route / top_k;
    const int expert = topk_idx[route];

    a_up_ptrs[route] = up_weight_ptrs[expert];
    b_up_ptrs[route] = norm_x + static_cast<long long>(token) * hidden;
    c_up_ptrs[route] = expert_up + static_cast<long long>(route) * intermediate;

    a_down_ptrs[route] = down_weight_ptrs[expert];
    b_down_ptrs[route] = expert_act + static_cast<long long>(route) * intermediate;
    c_down_ptrs[route] = expert_out + static_cast<long long>(route) * hidden;

    weights_out[route] = topk_w[route];
}

__global__ void build_nemotron_moe_ptrs_aligned_kernel(
    const std::int32_t* __restrict__ expert_ids,
    const __nv_bfloat16* const* __restrict__ up_weight_ptrs,
    const __nv_bfloat16* const* __restrict__ down_weight_ptrs,
    const __nv_bfloat16* __restrict__ aligned_in,
    __nv_bfloat16* __restrict__ aligned_up,
    __nv_bfloat16* __restrict__ aligned_act,
    __nv_bfloat16* __restrict__ aligned_out,
    const __nv_bfloat16** __restrict__ a_up_ptrs,
    const __nv_bfloat16** __restrict__ b_up_ptrs,
    __nv_bfloat16** __restrict__ c_up_ptrs,
    const __nv_bfloat16** __restrict__ a_down_ptrs,
    const __nv_bfloat16** __restrict__ b_down_ptrs,
    __nv_bfloat16** __restrict__ c_down_ptrs,
    int max_blocks,
    int block_size,
    int hidden,
    int intermediate)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= max_blocks) return;
    int expert = expert_ids[b];
    if (expert < 0) expert = 0;
    const long long row = static_cast<long long>(b) * block_size;

    a_up_ptrs[b] = up_weight_ptrs[expert];
    b_up_ptrs[b] = aligned_in + row * hidden;
    c_up_ptrs[b] = aligned_up + row * intermediate;

    a_down_ptrs[b] = down_weight_ptrs[expert];
    b_down_ptrs[b] = aligned_act + row * intermediate;
    c_down_ptrs[b] = aligned_out + row * hidden;
}

}  // namespace

void launch_nemotron_mamba_split_bf16(
    const void* projected,
    void* gate,
    void* conv_in,
    void* dt,
    int N,
    int projection_dim,
    int intermediate,
    int conv_dim,
    int num_heads,
    cudaStream_t stream)
{
    const int total = N * projection_dim;
    if (total <= 0) return;
    constexpr int BLOCK = 256;
    if (gate == nullptr) {
        const int conv_dt_total = N * (conv_dim + num_heads);
        const int conv_dt_grid = (conv_dt_total + BLOCK - 1) / BLOCK;
        mamba_split_conv_dt_kernel<<<conv_dt_grid, BLOCK, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(projected),
            static_cast<__nv_bfloat16*>(conv_in),
            static_cast<__nv_bfloat16*>(dt),
            projection_dim, intermediate, conv_dim, num_heads,
            conv_dt_total);
        return;
    }
    const int grid = (total + BLOCK - 1) / BLOCK;
    mamba_split_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projected),
        static_cast<__nv_bfloat16*>(gate),
        static_cast<__nv_bfloat16*>(conv_in),
        static_cast<__nv_bfloat16*>(dt),
        projection_dim, intermediate, conv_dim, num_heads, total);
}

void launch_nemotron_prepare_mamba_params(
    const void* A_log,
    const void* D,
    const void* dt_bias,
    float* A,
    float* D_f32,
    float* dt_bias_f32,
    int num_heads,
    cudaStream_t stream)
{
    if (num_heads <= 0) return;
    constexpr int BLOCK = 256;
    const int grid = (num_heads + BLOCK - 1) / BLOCK;
    prepare_mamba_params_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(A_log),
        static_cast<const __nv_bfloat16*>(D),
        static_cast<const __nv_bfloat16*>(dt_bias),
        A, D_f32, dt_bias_f32, num_heads);
}

void launch_nemotron_prepare_mamba_dt_da(
    const void* dt,
    const float* A,
    const float* dt_bias,
    float* dt_out,
    float* dA_out,
    int N,
    int num_heads,
    float time_step_min,
    cudaStream_t stream)
{
    const int total = N * num_heads;
    if (total <= 0) return;
    constexpr int BLOCK = 256;
    const int grid = (total + BLOCK - 1) / BLOCK;
    prepare_mamba_dt_da_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(dt),
        A, dt_bias, dt_out, dA_out, total, num_heads, time_step_min);
}

void launch_nemotron_mamba_ssm_batched_bf16(
    const void* conv_out,
    const void* dt,
    const float* A,
    const float* D,
    const float* dt_bias,
    const float* dt_precomputed,
    const float* dA_precomputed,
    void* ssm_state_base,
    const std::int32_t* slot_ids,
    const std::uint32_t* qo_indptr,
    void* y,
    int R,
    int num_heads,
    int head_dim,
    int state_size,
    int n_groups,
    int conv_dim,
    int intermediate,
    float time_step_min,
    bool sequence_prefill,
    cudaStream_t stream)
{
    if (R <= 0 || num_heads <= 0 || head_dim <= 0 || state_size <= 0) return;
    constexpr int BLOCK = 256;
    const bool force_reg_kernel =
        std::getenv("PIE_NEMOTRON_FORCE_PREFILL_REG_SSM") != nullptr;
    if ((sequence_prefill || force_reg_kernel) &&
        std::getenv("PIE_NEMOTRON_DISABLE_PREFILL_REG_SSM") == nullptr) {
        constexpr int PREFILL_BLOCK = 512;
        const int num_warps = PREFILL_BLOCK / 32;
        dim3 grid(R, num_heads, (head_dim + num_warps - 1) / num_warps);
        const std::size_t shared =
            2ull * static_cast<std::size_t>(state_size) * sizeof(float);
        mamba_ssm_batched_prefill_reg_kernel<<<
            grid, PREFILL_BLOCK, shared, stream>>>(
            static_cast<const __nv_bfloat16*>(conv_out),
            static_cast<const __nv_bfloat16*>(dt),
            A,
            D,
            dt_bias,
            dt_precomputed,
            dA_precomputed,
            static_cast<__nv_bfloat16*>(ssm_state_base),
            slot_ids, qo_indptr,
            static_cast<__nv_bfloat16*>(y),
            num_heads, head_dim, state_size, n_groups,
            conv_dim, intermediate, time_step_min);
        return;
    }
    if (!sequence_prefill &&
        std::getenv("PIE_NEMOTRON_DECODE_TILE4_SSM") != nullptr) {
        constexpr int DECODE_TILE_BLOCK = 128;
        const int num_warps = DECODE_TILE_BLOCK / 32;
        dim3 grid(R, num_heads, (head_dim + num_warps - 1) / num_warps);
        const std::size_t shared =
            2ull * static_cast<std::size_t>(state_size) * sizeof(float);
        mamba_ssm_batched_prefill_reg_kernel<<<
            grid, DECODE_TILE_BLOCK, shared, stream>>>(
            static_cast<const __nv_bfloat16*>(conv_out),
            static_cast<const __nv_bfloat16*>(dt),
            A,
            D,
            dt_bias,
            dt_precomputed,
            dA_precomputed,
            static_cast<__nv_bfloat16*>(ssm_state_base),
            slot_ids, qo_indptr,
            static_cast<__nv_bfloat16*>(y),
            num_heads, head_dim, state_size, n_groups,
            conv_dim, intermediate, time_step_min);
        return;
    }
    dim3 grid(R, num_heads);
    static const bool use_warp_kernel =
        std::getenv("PIE_NEMOTRON_DISABLE_WARP_SSM") == nullptr;
    if (use_warp_kernel) {
        const std::size_t shared =
            2ull * static_cast<std::size_t>(state_size) * sizeof(float);
        mamba_ssm_batched_warp_kernel<<<grid, BLOCK, shared, stream>>>(
            static_cast<const __nv_bfloat16*>(conv_out),
            static_cast<const __nv_bfloat16*>(dt),
            A,
            D,
            dt_bias,
            dt_precomputed,
            dA_precomputed,
            static_cast<__nv_bfloat16*>(ssm_state_base),
            slot_ids, qo_indptr,
            static_cast<__nv_bfloat16*>(y),
            num_heads, head_dim, state_size, n_groups,
            conv_dim, intermediate, time_step_min);
        return;
    }
    const std::size_t shared = static_cast<std::size_t>(head_dim) * sizeof(float);
    mamba_ssm_batched_kernel<<<grid, BLOCK, shared, stream>>>(
        static_cast<const __nv_bfloat16*>(conv_out),
        static_cast<const __nv_bfloat16*>(dt),
        A,
        D,
        dt_bias,
        static_cast<__nv_bfloat16*>(ssm_state_base),
        slot_ids, qo_indptr,
        static_cast<__nv_bfloat16*>(y),
        num_heads, head_dim, state_size, n_groups,
        conv_dim, intermediate, time_step_min);
}

void launch_zamba_rmsnorm_gated_bf16(
    const void* x,
    const void* gate,
    const void* weight,
    void* y,
    int N,
    int hidden,
    int gate_stride,
    int group_size,
    float eps,
    cudaStream_t stream)
{
    if (N <= 0 || hidden <= 0 || group_size <= 0) return;
    if (gate_stride <= 0) gate_stride = hidden;
    constexpr int BLOCK = 256;
    const int groups = hidden / group_size;
    dim3 grid(N, groups);
    zamba_rmsnorm_gated_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const __nv_bfloat16*>(gate),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(y),
        hidden, gate_stride, group_size, eps);
}

void launch_build_nemotron_moe_ptrs_decode_batched_bf16(
    const std::int32_t* topk_idx,
    const float* topk_w,
    const void* const* up_weight_ptrs,
    const void* const* down_weight_ptrs,
    const void* norm_x,
    void* expert_up,
    void* expert_act,
    void* expert_out,
    const void** a_up_ptrs,
    const void** b_up_ptrs,
    void** c_up_ptrs,
    const void** a_down_ptrs,
    const void** b_down_ptrs,
    void** c_down_ptrs,
    float* weights_out,
    int N,
    int top_k,
    int hidden,
    int intermediate,
    cudaStream_t stream)
{
    const int routes = N * top_k;
    if (routes <= 0) return;
    constexpr int BLOCK = 256;
    const int blocks = (routes + BLOCK - 1) / BLOCK;
    build_nemotron_moe_ptrs_decode_batched_kernel<<<blocks, BLOCK, 0, stream>>>(
        topk_idx, topk_w,
        reinterpret_cast<const __nv_bfloat16* const*>(up_weight_ptrs),
        reinterpret_cast<const __nv_bfloat16* const*>(down_weight_ptrs),
        static_cast<const __nv_bfloat16*>(norm_x),
        static_cast<__nv_bfloat16*>(expert_up),
        static_cast<__nv_bfloat16*>(expert_act),
        static_cast<__nv_bfloat16*>(expert_out),
        reinterpret_cast<const __nv_bfloat16**>(a_up_ptrs),
        reinterpret_cast<const __nv_bfloat16**>(b_up_ptrs),
        reinterpret_cast<__nv_bfloat16**>(c_up_ptrs),
        reinterpret_cast<const __nv_bfloat16**>(a_down_ptrs),
        reinterpret_cast<const __nv_bfloat16**>(b_down_ptrs),
        reinterpret_cast<__nv_bfloat16**>(c_down_ptrs),
        weights_out, routes, top_k, hidden, intermediate);
}

void launch_build_nemotron_moe_ptrs_aligned_bf16(
    const std::int32_t* expert_ids,
    const void* const* up_weight_ptrs,
    const void* const* down_weight_ptrs,
    const void* aligned_in,
    void* aligned_up,
    void* aligned_act,
    void* aligned_out,
    const void** a_up_ptrs,
    const void** b_up_ptrs,
    void** c_up_ptrs,
    const void** a_down_ptrs,
    const void** b_down_ptrs,
    void** c_down_ptrs,
    int max_blocks,
    int block_size,
    int hidden,
    int intermediate,
    cudaStream_t stream)
{
    if (max_blocks <= 0 || block_size <= 0 || hidden <= 0 ||
        intermediate <= 0) {
        return;
    }
    constexpr int BLOCK = 256;
    const int blocks = (max_blocks + BLOCK - 1) / BLOCK;
    build_nemotron_moe_ptrs_aligned_kernel<<<blocks, BLOCK, 0, stream>>>(
        expert_ids,
        reinterpret_cast<const __nv_bfloat16* const*>(up_weight_ptrs),
        reinterpret_cast<const __nv_bfloat16* const*>(down_weight_ptrs),
        static_cast<const __nv_bfloat16*>(aligned_in),
        static_cast<__nv_bfloat16*>(aligned_up),
        static_cast<__nv_bfloat16*>(aligned_act),
        static_cast<__nv_bfloat16*>(aligned_out),
        reinterpret_cast<const __nv_bfloat16**>(a_up_ptrs),
        reinterpret_cast<const __nv_bfloat16**>(b_up_ptrs),
        reinterpret_cast<__nv_bfloat16**>(c_up_ptrs),
        reinterpret_cast<const __nv_bfloat16**>(a_down_ptrs),
        reinterpret_cast<const __nv_bfloat16**>(b_down_ptrs),
        reinterpret_cast<__nv_bfloat16**>(c_down_ptrs),
        max_blocks, block_size, hidden, intermediate);
}

}  // namespace pie_cuda_driver::kernels
