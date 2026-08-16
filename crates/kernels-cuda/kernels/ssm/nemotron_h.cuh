//===-- nemotron_h.cuh - Nemotron-H / Zamba device text ----------------===//
//
// Ten `__global__`s and six `__device__` helpers, with no host function and
// no `<<<>>>`. `nemotron_h.cu` includes this file and keeps only its
// launchers, so each kernel is defined ONCE in the tree. That is not tidiness:
// a header that COPIES rather than is included gives nvcc one text and NVRTC
// another, and the two drift with every test green on whichever half it
// exercised — `norm/altup_aux` shipped exactly that pair for a release, and
// `tests/device_sources.rs` now refuses a second definition of a
// namespace-qualified `__global__` because of it.
//
// # Three rows out of ten, and the seven refusals
//
// `prepare_mamba_params` and `prepare_mamba_dt_da` are pure maps behind an
// `i >= n` guard, launched as `ceil(n / 256)` blocks of 256 — which is
// `LaunchRule::Elementwise` verbatim, down to the block width.
//
// `zamba_rmsnorm_gated` is the third, and it was blocked on ONE fact:
// `dim3 grid(N, hidden / group_size)` at 256 threads is
// `LaunchRule::GatedRms` to the digit — `runtime::launch::gated_rms` cites
// this very launcher as the one the rule was derived from — and a plain
// `__global__` cannot be named by a row whatever its geometry, because
// `DeviceKernel::instantiation()` emits `path<Elem>` and stops. It is a
// template now. Nothing else about it changed: `Elem<T>::to_f32` and
// `from_f32` at `T = bf16` are this file's `bf16_to_float` and
// `float_to_bf16`, and the launcher instantiates it at `bf16`.
//
// The other seven state geometry no rule in `kernels::LaunchRule` produces,
// and the gap is the same one every time — CUDA's `Dims` carries `rows`,
// `width` and `in_width`, and these kernels put a HEAD, an EXPERT or a
// REQUEST on an axis:
//
//   • `mamba_ssm_batched_warp`, `..._prefill_reg`, `..._batched` — grid
//     `(requests, heads[, tiles])`, plus `2 * state_size * sizeof(float)` of
//     dynamic shared memory. `Launch` does carry an `smem` field, so the
//     third `<<<>>>` argument is not the blocker it looked like; but only
//     `Rms` ever fills it, and with a fixed 32 bytes — a rule-sized launch
//     would hand these three a zero-length `extern __shared__` and let them
//     stage `b` and `c` past the end of it. All three are also behind ONE
//     symbol: `nemotron_mamba_ssm_batched_bf16` picks between them on the
//     host from `sequence_prefill`, so a row for any one of them states a
//     third of a contract. They stay plain `__global__`s for that reason —
//     templating a kernel no row can name is a rename. (`mamba_ssm_batched`
//     is additionally dead: an unconditional `return` precedes its launch,
//     left standing as the reference the two fast paths are checked against.)
//   • `mamba_split`, `mamba_split_conv_dt` — flat over `rows * in_width`
//     (and over `rows * (conv_dim + num_heads)`, which is neither width).
//     The one flat rule, `Elementwise`, reads the OUTPUT extent;
//     `SplitPacked` reads `in_width` but puts the row on its own axis, and
//     the JIT runtime does not port it.
//   • `build_nemotron_moe_ptrs_decode_batched` — extent is `rows * top_k`,
//     an INPUT count where `Elementwise` reads output elements.
//   • `build_nemotron_moe_ptrs_aligned` — extent is a host scalar,
//     `max_blocks`, computed from a padded expert histogram. No shape on the
//     fire produces it.
//
// Those seven are here because the split is per FILE — one `.cu`, one `.cuh`,
// one definition — and unrowed because inventing a rule to fit them would put
// a guess where a refusal belongs.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

namespace pie::ssm {

// The scalar layer is the PRELUDE's. Named here so the kernels read as they
// did, so a row may spell its element type `bf16`, and so the
// launchers in `nemotron_h.cu` — which sit in `kernels::ssm`, where this
// namespace would otherwise shadow `kernels::device` — still resolve
// `bf16` and `i32` to the prelude's types.
__device__ __forceinline__ float bf16_to_float(const bf16 v) {
    return bf16_to_f32(v);
}

__device__ __forceinline__ bf16 float_to_bf16(float v) {
    return f32_to_bf16(v);
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

__global__ void mamba_split(
    const bf16* __restrict__ projected,
    bf16* __restrict__ gate,
    bf16* __restrict__ conv_in,
    bf16* __restrict__ dt,
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

__global__ void mamba_split_conv_dt(
    const bf16* __restrict__ projected,
    bf16* __restrict__ conv_in,
    bf16* __restrict__ dt,
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
    const bf16* src =
        projected + static_cast<long long>(row) * projection_dim + intermediate;
    if (col < conv_dim) {
        conv_in[static_cast<long long>(row) * conv_dim + col] = src[col];
    } else {
        dt[static_cast<long long>(row) * num_heads + (col - conv_dim)] =
            src[col];
    }
}

template <class T>
__global__ void prepare_mamba_params(
    const T* __restrict__ A_log,
    const T* __restrict__ D,
    const T* __restrict__ dt_bias,
    float* __restrict__ A,
    float* __restrict__ D_f32,
    float* __restrict__ dt_bias_f32,
    int num_heads)
{
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= num_heads) return;
    A[h] = -__expf(Elem<T>::to_f32(A_log[h]));
    D_f32[h] = Elem<T>::to_f32(D[h]);
    dt_bias_f32[h] = Elem<T>::to_f32(dt_bias[h]);
}

template <class T>
__global__ void prepare_mamba_dt_da(
    const T* __restrict__ dt_in,
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
        softplus_f(Elem<T>::to_f32(dt_in[i]) + dt_bias[h]),
        time_step_min);
    dt_out[i] = dt;
    dA_out[i] = __expf(dt * A[h]);
}

// Same request/head ownership as mamba_ssm_batched, but maps one warp
// to one head dimension and reduces the state axis inside the warp. This avoids
// the shared-memory atomicAdd hot path in the generic kernel.
__global__ void mamba_ssm_batched_warp(
    const bf16* __restrict__ conv_out,
    const bf16* __restrict__ dt_in,
    const float* __restrict__ A,
    const float* __restrict__ D,
    const float* __restrict__ dt_bias,
    const float* __restrict__ dt_precomputed,
    const float* __restrict__ dA_precomputed,
    bf16* __restrict__ state_base,
    const i32* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    bf16* __restrict__ y,
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
    bf16* state =
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
        const bf16* row_conv =
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
__global__ void mamba_ssm_batched_prefill_reg(
    const bf16* __restrict__ conv_out,
    const bf16* __restrict__ dt_in,
    const float* __restrict__ A,
    const float* __restrict__ D,
    const float* __restrict__ dt_bias,
    const float* __restrict__ dt_precomputed,
    const float* __restrict__ dA_precomputed,
    bf16* __restrict__ state_base,
    const i32* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    bf16* __restrict__ y,
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
    bf16* state =
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
        const bf16* row_conv =
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
__global__ void mamba_ssm_batched(
    const bf16* __restrict__ conv_out,
    const bf16* __restrict__ dt_in,
    const float* __restrict__ A,
    const float* __restrict__ D,
    const float* __restrict__ dt_bias,
    bf16* __restrict__ state_base,
    const i32* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    bf16* __restrict__ y,
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
    bf16* state =
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
        const bf16* row_conv =
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

/// The grouped, gated output norm: `y = rmsnorm(x * silu(gate)) * weight`, one
/// block per (row, group).
///
/// `LaunchRule::GatedRms` is this launcher and `runtime::launch::gated_rms`
/// cites it as the launcher the rule was derived from -- `dim3 grid(N,
/// hidden / group_size)` at 256 threads, digit for digit. The 256 is a
/// PRECONDITION and not a preference: `buf` below is a static
/// `__shared__ float[256]` and the fold walks `blockDim.x / 2`, so a wider
/// block indexes past the array (which the hardware does not report) and a
/// block that is not a power of two drops the odd lane and normalises by a sum
/// missing a term.
///
/// Templated because a row cannot name a kernel that is not a template:
/// `DeviceKernel::instantiation()` emits exactly one type argument. The
/// arithmetic is untouched -- widen to fp32, reduce, narrow once -- and
/// `Elem<T>::to_f32`/`from_f32` at `T = bf16` are the `bf16_to_float` and
/// `float_to_bf16` this kernel called, spelled through the prelude's trait
/// instead of this file's two aliases for it.
template <class T>
__global__ void zamba_rmsnorm_gated(
    const T* __restrict__ x,
    const T* __restrict__ gate,
    const T* __restrict__ weight,
    T* __restrict__ y,
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
        const float xv = Elem<T>::to_f32(x[base + i]);
        const float gv = Elem<T>::to_f32(gate[gate_base + i]);
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
        const float xv = Elem<T>::to_f32(x[base + i]);
        const float gv = Elem<T>::to_f32(gate[gate_base + i]);
        const float v = xv * silu_f(gv) * inv_rms;
        const int h = group * group_size + i;
        y[base + i] = Elem<T>::from_f32(v * Elem<T>::to_f32(weight[h]));
    }
}

__global__ void build_nemotron_moe_ptrs_decode_batched(
    const i32* __restrict__ topk_idx,
    const float* __restrict__ topk_w,
    const bf16* const* __restrict__ up_weight_ptrs,
    const bf16* const* __restrict__ down_weight_ptrs,
    const bf16* __restrict__ norm_x,
    bf16* __restrict__ expert_up,
    bf16* __restrict__ expert_act,
    bf16* __restrict__ expert_out,
    const bf16** __restrict__ a_up_ptrs,
    const bf16** __restrict__ b_up_ptrs,
    bf16** __restrict__ c_up_ptrs,
    const bf16** __restrict__ a_down_ptrs,
    const bf16** __restrict__ b_down_ptrs,
    bf16** __restrict__ c_down_ptrs,
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

__global__ void build_nemotron_moe_ptrs_aligned(
    const i32* __restrict__ expert_ids,
    const bf16* const* __restrict__ up_weight_ptrs,
    const bf16* const* __restrict__ down_weight_ptrs,
    const bf16* __restrict__ aligned_in,
    bf16* __restrict__ aligned_up,
    bf16* __restrict__ aligned_act,
    bf16* __restrict__ aligned_out,
    const bf16** __restrict__ a_up_ptrs,
    const bf16** __restrict__ b_up_ptrs,
    bf16** __restrict__ c_up_ptrs,
    const bf16** __restrict__ a_down_ptrs,
    const bf16** __restrict__ b_down_ptrs,
    bf16** __restrict__ c_down_ptrs,
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

}  // namespace pie::ssm
