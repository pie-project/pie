#include "kernels/gated_delta_net.hpp"

#include <cuda_bf16.h>
#include <cstdlib>
#include <stdexcept>

namespace pie_cuda_driver::kernels {

// ── Helpers ────────────────────────────────────────────────────────

namespace {

template <typename StateT>
__device__ __forceinline__ float state_load(const StateT* p) {
    return static_cast<float>(*p);
}

template <>
__device__ __forceinline__ float state_load<__nv_bfloat16>(
    const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}

template <typename StateT>
__device__ __forceinline__ void state_store(StateT* p, float v) {
    *p = static_cast<StateT>(v);
}

template <>
__device__ __forceinline__ void state_store<__nv_bfloat16>(
    __nv_bfloat16* p, float v) {
    *p = __float2bfloat16(v);
}

template <bool KLast>
__device__ __forceinline__ long long state_offset(
    int k_idx, int v_idx, int K_d, int V_d) {
    if constexpr (KLast) {
        return (long long)v_idx * K_d + k_idx;
    } else {
        return (long long)k_idx * V_d + v_idx;
    }
}

__device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffffu, x, offset);
    }
    return __shfl_sync(0xffffffffu, x, 0);
}

bool qwen_gdn_gqa_ilp2_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_GDN_GQA_ILP2");
        if (v == nullptr || v[0] == '\0') return false;
        return v[0] != '0';
    }();
    return enabled;
}

bool qwen_gdn_k_last_state_enabled() {
    // Default OFF: storing state as [k, v] (V-last) — the layout used
    // when KLast=false — lets threads indexed by v_idx access state at
    // contiguous offsets within a warp (stride 1, fully coalesced).
    // KLast=true stores [v, k] which forces a stride of K_d floats
    // between adjacent threads' state reads, fracturing every warp's
    // memory transaction into 32 cache-line accesses and amplifying
    // HBM traffic by ~30x on the recurrent step.
    //
    // Measured on Qwen/Qwen3.5-4B (cuda_native, H100 PCIe, 512 reqs x
    // 128 tok decode): KLast=true -> 1,414 tok/s; KLast=false -> 5,242
    // tok/s (+271%). Prefill-heavy workload (1k-tok prompts, 8-tok
    // output): KLast=true -> 29 tok/s; KLast=false -> 258 tok/s (+790%).
    // Output is bit-identical under both layouts (sha256 matches).
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_GDN_K_LAST_STATE");
        if (v == nullptr || v[0] == '\0') return false;
        return v[0] != '0';
    }();
    return enabled;
}

// Use the fused recurrent step kernel that caches state values in
// registers across the two analytical phases, halving HBM traffic on
// the state slab (2R+2W -> 1R+1W per element). Default OFF until
// parity is verified across all (K_d, V_d) combinations the kernel is
// instantiated for; turn ON for benchmarking the new path.
bool qwen_gdn_fused_step_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_GDN_FUSED_STEP");
        if (v == nullptr || v[0] == '\0') return false;
        return v[0] != '0';
    }();
    return enabled;
}

// SMEM read-only step kernel: stages the BF16 state slab into shared
// memory once, reads it from SMEM in both analytical phases, and
// writes the updated state straight to HBM (no SMEM writebacks). In a
// standalone microbench at R=511 saturated decode this drops the
// per-call wall from 2406 us (legacy bf16 v-last) to 1579 us — 34%
// faster. fp32 accumulate, rounded to BF16 once (same scheme as the
// FLA chunked-prefill default), so strictly less quantization than
// the legacy per-element-round kernel.
//
// Default ON: +32% end-to-end throughput on Qwen/Qwen3.5-4B
// (6924 -> 9166 tok/s). The launcher only routes here for the GQA
// bf16 V-last decode shape (V_d==K_d==128, !k_last); everything else
// falls back to the legacy kernel. Set PIE_QWEN35_GDN_SMEM_STEP=0 to
// force the fallback.
bool qwen_gdn_smem_step_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_GDN_SMEM_STEP");
        if (v == nullptr || v[0] == '\0') return true;
        return v[0] != '0';
    }();
    return enabled;
}

__global__ void bf16_to_fp32_kernel(
    const __nv_bfloat16* __restrict__ x, float* __restrict__ y, std::size_t n)
{
    const std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
    if (i < n) y[i] = __bfloat162float(x[i]);
}

__global__ void fp32_to_bf16_kernel(
    const float* __restrict__ x, __nv_bfloat16* __restrict__ y, std::size_t n)
{
    const std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16(x[i]);
}

__global__ void repeat_interleave_heads_fp32_kernel(
    const float* __restrict__ in, float* __restrict__ out,
    int K_h, int V_h, int D, int repeat)
{
    const int n   = blockIdx.x;
    const int h_v = blockIdx.y;
    const int d   = threadIdx.x;
    if (h_v >= V_h || d >= D) return;
    const int h_k = h_v / repeat;
    const long long src = ((long long)n * K_h + h_k) * D + d;
    const long long dst = ((long long)n * V_h + h_v) * D + d;
    if (d < D) out[dst] = in[src];
    // Iterate if D > blockDim.x.
    for (int dd = d + blockDim.x; dd < D; dd += blockDim.x) {
        out[((long long)n * V_h + h_v) * D + dd] =
            in[((long long)n * K_h + h_k) * D + dd];
    }
}

template <int BLOCK>
__global__ void l2norm_scale_bf16_to_fp32_kernel(
    const __nv_bfloat16* __restrict__ x,
    float*               __restrict__ y,
    int hidden, float scale, float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const __nv_bfloat16* xr = x + (long long)row * hidden;
    float*               yr = y + (long long)row * hidden;

    float local = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        const float v = __bfloat162float(xr[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    buf[tid] = local;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) buf[tid] += buf[tid + off];
        __syncthreads();
    }
    const float inv = rsqrtf(buf[0] + eps);

    for (int i = tid; i < hidden; i += BLOCK) {
        yr[i] = __bfloat162float(xr[i]) * inv * scale;
    }
}

// g_log[t, h] = -exp(A_log[h]) * softplus(a[t, h] + dt_bias[h])
// beta[t, h]  = sigmoid(b[t, h])
//
// HF Qwen3.5 stores `A_log` and the RMSNormGated weight in fp32 (matches
// the FLA fast-path expectation), even when the rest of the model is
// bf16. dt_bias stays bf16.
__global__ void g_beta_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const float*         __restrict__ A_log,
    const __nv_bfloat16* __restrict__ dt_bias,
    float*               __restrict__ g_log_out,
    float*               __restrict__ beta_out,
    int N, int V_h)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= N || h >= V_h) return;

    const float av  = __bfloat162float(a[(long long)t * V_h + h]);
    const float bv  = __bfloat162float(b[(long long)t * V_h + h]);
    const float Alh = A_log[h];
    const float dtb = __bfloat162float(dt_bias[h]);

    // softplus(z) = log1p(exp(z)). Numerically stable variant.
    const float z = av + dtb;
    const float sp = (z > 20.f) ? z : log1pf(__expf(z));

    g_log_out[(long long)t * V_h + h] = -__expf(Alh) * sp;
    beta_out[(long long)t * V_h + h]  = 1.f / (1.f + __expf(-bv));
}

template<int BLOCK>
__global__ void qwen_gdn_qk_norm_kernel(
    const __nv_bfloat16* __restrict__ qkv_post,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    int K_h, int K_d, int conv_dim,
    float q_scale)
{
    const int n = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int K_dim = K_h * K_d;
    const __nv_bfloat16* q_base =
        qkv_post + (long long)n * conv_dim + (long long)h * K_d;
    const __nv_bfloat16* k_base =
        qkv_post + (long long)n * conv_dim + K_dim + (long long)h * K_d;

    float q_sum = 0.f;
    float k_sum = 0.f;
    for (int i = tid; i < K_d; i += BLOCK) {
        const float qv = __bfloat162float(q_base[i]);
        const float kv = __bfloat162float(k_base[i]);
        q_sum += qv * qv;
        k_sum += kv * kv;
    }

    __shared__ float q_buf[BLOCK];
    __shared__ float k_buf[BLOCK];
    q_buf[tid] = q_sum;
    k_buf[tid] = k_sum;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) {
            q_buf[tid] += q_buf[tid + off];
            k_buf[tid] += k_buf[tid + off];
        }
        __syncthreads();
    }

    const float q_inv = rsqrtf(q_buf[0] + 1e-6f) * q_scale;
    const float k_inv = rsqrtf(k_buf[0] + 1e-6f);
    float* q_dst = q_out + ((long long)n * K_h + h) * K_d;
    float* k_dst = k_out + ((long long)n * K_h + h) * K_d;
    for (int i = tid; i < K_d; i += BLOCK) {
        q_dst[i] = __bfloat162float(q_base[i]) * q_inv;
        k_dst[i] = __bfloat162float(k_base[i]) * k_inv;
    }
}

template<int BLOCK>
__global__ void qwen_gdn_v_g_beta_kernel(
    const __nv_bfloat16* __restrict__ qkv_post,
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ dt_bias,
    float* __restrict__ v_out,
    float* __restrict__ g_log_out,
    float* __restrict__ beta_out,
    int K_h, int V_h, int K_d, int V_d, int conv_dim)
{
    const int n = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int K_dim = K_h * K_d;
    const __nv_bfloat16* v_base =
        qkv_post + (long long)n * conv_dim + 2 * K_dim + (long long)h * V_d;
    float* v_dst = v_out + ((long long)n * V_h + h) * V_d;
    for (int i = tid; i < V_d; i += BLOCK) {
        v_dst[i] = __bfloat162float(v_base[i]);
    }

    if (tid == 0) {
        const long long gh = (long long)n * V_h + h;
        const float av = __bfloat162float(a[gh]);
        const float bv = __bfloat162float(b[gh]);
        const float z = av + __bfloat162float(dt_bias[h]);
        const float sp = (z > 20.f) ? z : log1pf(__expf(z));
        g_log_out[gh] = -__expf(A_log[h]) * sp;
        beta_out[gh] = 1.f / (1.f + __expf(-bv));
    }
}

}  // namespace

void launch_bf16_to_fp32(
    const void* x, float* y, std::size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    constexpr int BLOCK = 256;
    const std::size_t grid = (n + BLOCK - 1) / BLOCK;
    bf16_to_fp32_kernel<<<(int)grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x), y, n);
}

void launch_fp32_to_bf16(
    const float* x, void* y, std::size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    constexpr int BLOCK = 256;
    const std::size_t grid = (n + BLOCK - 1) / BLOCK;
    fp32_to_bf16_kernel<<<(int)grid, BLOCK, 0, stream>>>(
        x, static_cast<__nv_bfloat16*>(y), n);
}

void launch_repeat_interleave_heads_fp32(
    const float* in, float* out,
    int N, int K_h, int V_h, int D,
    cudaStream_t stream)
{
    if (N <= 0 || K_h <= 0 || V_h <= 0 || D <= 0) return;
    const int repeat = V_h / K_h;
    const int block = (D < 128) ? 64 : 128;
    dim3 grid(N, V_h);
    repeat_interleave_heads_fp32_kernel<<<grid, block, 0, stream>>>(
        in, out, K_h, V_h, D, repeat);
}

void launch_l2norm_scale_bf16_to_fp32(
    const void* x, float* y,
    int N, int hidden,
    float scale, float eps,
    cudaStream_t stream)
{
    if (N <= 0 || hidden <= 0) return;
    constexpr int BLOCK = 128;
    dim3 grid(N);
    dim3 block(BLOCK);
    l2norm_scale_bf16_to_fp32_kernel<BLOCK><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x), y, hidden, scale, eps);
}

void launch_gated_delta_g_beta(
    const void* a, const void* b,
    const void* A_log, const void* dt_bias,
    float* g_log_out, float* beta_out,
    int N, int V_h, cudaStream_t stream)
{
    if (N <= 0 || V_h <= 0) return;
    constexpr int BLOCK = 64;
    dim3 grid(N, (V_h + BLOCK - 1) / BLOCK);
    dim3 block(BLOCK);
    g_beta_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(a),
        static_cast<const __nv_bfloat16*>(b),
        static_cast<const float*>(A_log),
        static_cast<const __nv_bfloat16*>(dt_bias),
        g_log_out, beta_out, N, V_h);
}

void launch_qwen_gdn_post_conv_prep_bf16(
    const void* qkv_post,
    const void* a,
    const void* b,
    const void* A_log,
    const void* dt_bias,
    float* q_norm_kh,
    float* k_norm_kh,
    float* v_fp32,
    float* g_log_out,
    float* beta_out,
    int N, int K_h, int V_h, int K_d, int V_d, int conv_dim,
    cudaStream_t stream)
{
    if (N <= 0 || K_h <= 0 || V_h <= 0 || K_d <= 0 || V_d <= 0) return;
    constexpr int BLOCK = 128;
    const float q_scale = rsqrtf(static_cast<float>(K_d));
    dim3 qk_grid(N, K_h);
    qwen_gdn_qk_norm_kernel<BLOCK><<<qk_grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(qkv_post),
        q_norm_kh, k_norm_kh, K_h, K_d, conv_dim, q_scale);
    dim3 vg_grid(N, V_h);
    qwen_gdn_v_g_beta_kernel<BLOCK><<<vg_grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(qkv_post),
        static_cast<const __nv_bfloat16*>(a),
        static_cast<const __nv_bfloat16*>(b),
        static_cast<const float*>(A_log),
        static_cast<const __nv_bfloat16*>(dt_bias),
        v_fp32, g_log_out, beta_out,
        K_h, V_h, K_d, V_d, conv_dim);
}

// ── Recurrent step kernel ──────────────────────────────────────────

namespace {

// One block per (request, head). Threads parallelize over v_idx in
// [0, V_d). Each thread loops over k_idx in [0, K_d) twice (once for
// the kv_mem accumulation, once for the post-update output).
//
// Shared memory layout: q[K_d] + k[K_d] fp32. Caller passes shmem of
// size 2*K_d*sizeof(float).
template <typename StateT, bool KLast>
__global__ void recurrent_step_kernel(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state,
    float*       __restrict__ out,
    int V_h, int K_d, int V_d)
{
    const int b = blockIdx.x;
    const int h = blockIdx.y;

    const long long bh = (long long)b * V_h + h;
    const float* q_h = q_norm + bh * K_d;
    const float* k_h = k_norm + bh * K_d;
    const float* v_h = v      + bh * V_d;
    const float  g_h = __expf(g_log[bh]);
    const float  beta_h = beta[bh];

    state += bh * (long long)K_d * V_d;
    out   += bh * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + K_d;

    // Load q/k into shmem cooperatively.
    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
    }
    __syncthreads();

    // Phase 1: state *= g, kv_mem[v] = Σ_k state[k, v] * k[k].
    // Output of this phase: kv_mem[v] in register `kv_mem` for each
    // thread that owns its v_idx.
    for (int v_idx = threadIdx.x; v_idx < V_d; v_idx += blockDim.x) {
        float kv_mem = 0.f;
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s = state_load(state + off) * g_h;
            state_store(state + off, s);
            kv_mem += s * sk[k_idx];
        }

        const float v_t   = v_h[v_idx];
        const float delta = (v_t - kv_mem) * beta_h;

        // Phase 2: state[k, v] += k[k] * delta; out[v] = Σ_k state[k,v]*q[k].
        float out_v = 0.f;
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s = state_load(state + off) + sk[k_idx] * delta;
            state_store(state + off, s);
            out_v += s * sq[k_idx];
        }
        out[v_idx] = out_v;
    }
}

// Multi-request batched chunked prefill. One block per (request, head);
// the block walks its T_r tokens sequentially (per-token state
// dependency), accumulating the recurrence into the request's state
// slab. Same per-token math as `recurrent_step_kernel`.
template <typename StateT, bool KLast>
__global__ void chunk_gated_delta_prefill_batched_kernel(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*       __restrict__ slot_ids,
    const std::uint32_t* __restrict__ qo_indptr,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int V_h, int K_d, int V_d)
{
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int t0 = static_cast<int>(qo_indptr[r]);
    const int T  = static_cast<int>(qo_indptr[r + 1]) - t0;
    if (T <= 0) return;

    const int slot = slot_ids[r];
    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + K_d;

    for (int t = 0; t < T; ++t) {
        const long long bh = (long long)(t0 + t) * V_h + h;
        const float* q_h = q_norm + bh * K_d;
        const float* k_h = k_norm + bh * K_d;
        const float* v_h = v      + bh * V_d;
        const float  g_h = __expf(g_log[bh]);
        const float  beta_h = beta[bh];
        float* out_bh = out + bh * V_d;

        for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
            sq[i] = q_h[i];
            sk[i] = k_h[i];
        }
        __syncthreads();

        for (int v_idx = threadIdx.x; v_idx < V_d; v_idx += blockDim.x) {
            float kv_mem = 0.f;
            for (int k_idx = 0; k_idx < K_d; ++k_idx) {
                const long long off =
                    state_offset<KLast>(k_idx, v_idx, K_d, V_d);
                const float s = state_load(state + off) * g_h;
                state_store(state + off, s);
                kv_mem += s * sk[k_idx];
            }

            const float v_t   = v_h[v_idx];
            const float delta = (v_t - kv_mem) * beta_h;

            float out_v = 0.f;
            for (int k_idx = 0; k_idx < K_d; ++k_idx) {
                const long long off =
                    state_offset<KLast>(k_idx, v_idx, K_d, V_d);
                const float s = state_load(state + off) + sk[k_idx] * delta;
                state_store(state + off, s);
                out_v += s * sq[k_idx];
            }
            out_bh[v_idx] = out_v;
        }
        // State must be globally visible before next-token's reads;
        // __syncthreads ensures the block sees its own writes — adjacent
        // blocks (different r or h) are independent.
        __syncthreads();
    }
}

template <typename StateT, bool KLast>
__global__ void chunk_gated_delta_prefill_batched_cached_kernel(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*       __restrict__ slot_ids,
    const std::uint32_t* __restrict__ qo_indptr,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int V_h, int K_d, int V_d,
    bool write_state)
{
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int t0 = static_cast<int>(qo_indptr[r]);
    const int T  = static_cast<int>(qo_indptr[r + 1]) - t0;
    if (T <= 0) return;

    const int slot = slot_ids[r];
    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;

    extern __shared__ float s_state[];
    const int state_elems = K_d * V_d;
    for (int i = threadIdx.x; i < state_elems; i += blockDim.x) {
        s_state[i] = state_load(state + i);
    }
    __syncthreads();

    for (int t = 0; t < T; ++t) {
        const long long bh = (long long)(t0 + t) * V_h + h;
        const float* q_h = q_norm + bh * K_d;
        const float* k_h = k_norm + bh * K_d;
        const float* v_h = v      + bh * V_d;
        const float  g_h = __expf(g_log[bh]);
        const float  beta_h = beta[bh];
        float* out_bh = out + bh * V_d;

        for (int v_idx = threadIdx.x; v_idx < V_d; v_idx += blockDim.x) {
            float kv_mem = 0.f;
            for (int k_idx = 0; k_idx < K_d; ++k_idx) {
                const long long off =
                    state_offset<KLast>(k_idx, v_idx, K_d, V_d);
                const float s = s_state[off] * g_h;
                s_state[off] = s;
                kv_mem += s * k_h[k_idx];
            }

            const float delta = (v_h[v_idx] - kv_mem) * beta_h;
            float out_v = 0.f;
            for (int k_idx = 0; k_idx < K_d; ++k_idx) {
                const long long off =
                    state_offset<KLast>(k_idx, v_idx, K_d, V_d);
                const float s = s_state[off] + k_h[k_idx] * delta;
                s_state[off] = s;
                out_v += s * q_h[k_idx];
            }
            out_bh[v_idx] = out_v;
        }
    }

    // Frozen verify (write_state=false): produce outputs but persist nothing,
    // leaving the committed slot at its pre-verify value (advanced later by the
    // repair forward).
    if (write_state) {
        __syncthreads();
        for (int i = threadIdx.x; i < state_elems; i += blockDim.x) {
            state_store(state + i, s_state[i]);
        }
    }
}

template <typename StateT, bool KLast>
__global__ void chunk_gated_delta_prefill_batched_warp_tiled_kernel(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*       __restrict__ slot_ids,
    const std::uint32_t* __restrict__ qo_indptr,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int V_h, int K_d, int V_d,
    bool write_state)
{
    constexpr int WARPS = 4;
    constexpr int MAX_K_PER_LANE = 8;  // supports K_d <= 256 with 32 lanes
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int v_tile = blockIdx.z * WARPS;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int v_idx = v_tile + warp;
    if (warp >= WARPS || v_idx >= V_d) return;

    const int t0 = static_cast<int>(qo_indptr[r]);
    const int T  = static_cast<int>(qo_indptr[r + 1]) - t0;
    if (T <= 0) return;

    const int slot = slot_ids[r];
    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;

    float s_vals[MAX_K_PER_LANE];
    int k_vals[MAX_K_PER_LANE];
    int n_k = 0;
    for (int k_idx = lane; k_idx < K_d && n_k < MAX_K_PER_LANE; k_idx += 32) {
        k_vals[n_k] = k_idx;
        s_vals[n_k] = state_load(
            state + state_offset<KLast>(k_idx, v_idx, K_d, V_d));
        ++n_k;
    }

    for (int t = 0; t < T; ++t) {
        const long long bh = (long long)(t0 + t) * V_h + h;
        const float* q_h = q_norm + bh * K_d;
        const float* k_h = k_norm + bh * K_d;
        const float* v_h = v + bh * V_d;
        const float g_h = __expf(g_log[bh]);
        const float beta_h = beta[bh];

        float kv_part = 0.f;
        #pragma unroll
        for (int i = 0; i < MAX_K_PER_LANE; ++i) {
            if (i < n_k) {
                const int k_idx = k_vals[i];
                const float s = s_vals[i] * g_h;
                s_vals[i] = s;
                kv_part += s * k_h[k_idx];
            }
        }
        const float kv_mem = warp_sum(kv_part);
        const float delta = (v_h[v_idx] - kv_mem) * beta_h;

        float out_part = 0.f;
        #pragma unroll
        for (int i = 0; i < MAX_K_PER_LANE; ++i) {
            if (i < n_k) {
                const int k_idx = k_vals[i];
                const float s = s_vals[i] + k_h[k_idx] * delta;
                s_vals[i] = s;
                out_part += s * q_h[k_idx];
            }
        }
        const float out_v = warp_sum(out_part);
        if (lane == 0) {
            out[bh * (long long)V_d + v_idx] = out_v;
        }
    }

    // Frozen verify (write_state=false): persist nothing.
    if (write_state) {
        #pragma unroll
        for (int i = 0; i < MAX_K_PER_LANE; ++i) {
            if (i < n_k) {
                state_store(
                    state + state_offset<KLast>(
                        k_vals[i], v_idx, K_d, V_d),
                    s_vals[i]);
            }
        }
    }
}

template <typename StateT, bool KLast>
__global__ void chunk_gated_delta_prefill_batched_warp_tiled_gqa_kernel(
    const float* __restrict__ q_norm_kh,
    const float* __restrict__ k_norm_kh,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*       __restrict__ slot_ids,
    const std::uint32_t* __restrict__ qo_indptr,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d,
    bool write_state)
{
    constexpr int WARPS = 4;
    constexpr int MAX_K_PER_LANE = 8;  // supports K_d <= 256 with 32 lanes
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int v_tile = blockIdx.z * WARPS;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int v_idx = v_tile + warp;
    if (warp >= WARPS || v_idx >= V_d) return;

    const int repeat = V_h / K_h;
    const int qk_h = h / repeat;
    const int t0 = static_cast<int>(qo_indptr[r]);
    const int T  = static_cast<int>(qo_indptr[r + 1]) - t0;
    if (T <= 0) return;

    const int slot = slot_ids[r];
    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;

    float s_vals[MAX_K_PER_LANE];
    int k_vals[MAX_K_PER_LANE];
    int n_k = 0;
    for (int k_idx = lane; k_idx < K_d && n_k < MAX_K_PER_LANE; k_idx += 32) {
        k_vals[n_k] = k_idx;
        s_vals[n_k] = state_load(
            state + state_offset<KLast>(k_idx, v_idx, K_d, V_d));
        ++n_k;
    }

    for (int t = 0; t < T; ++t) {
        const long long qk_bh = ((long long)(t0 + t) * K_h + qk_h);
        const long long vh = (long long)(t0 + t) * V_h + h;
        const float* q_h = q_norm_kh + qk_bh * K_d;
        const float* k_h = k_norm_kh + qk_bh * K_d;
        const float* v_h = v + vh * V_d;
        const float g_h = __expf(g_log[vh]);
        const float beta_h = beta[vh];

        float kv_part = 0.f;
        #pragma unroll
        for (int i = 0; i < MAX_K_PER_LANE; ++i) {
            if (i < n_k) {
                const int k_idx = k_vals[i];
                const float s = s_vals[i] * g_h;
                s_vals[i] = s;
                kv_part += s * k_h[k_idx];
            }
        }
        const float kv_mem = warp_sum(kv_part);
        const float delta = (v_h[v_idx] - kv_mem) * beta_h;

        float out_part = 0.f;
        #pragma unroll
        for (int i = 0; i < MAX_K_PER_LANE; ++i) {
            if (i < n_k) {
                const int k_idx = k_vals[i];
                const float s = s_vals[i] + k_h[k_idx] * delta;
                s_vals[i] = s;
                out_part += s * q_h[k_idx];
            }
        }
        const float out_v = warp_sum(out_part);
        if (lane == 0) {
            out[vh * (long long)V_d + v_idx] = out_v;
        }
    }

    // Frozen verify (write_state=false): walk the state in registers for
    // correct draft outputs but persist nothing, leaving the committed slot at
    // its pre-verify value (the implicit speculative snapshot). The repair
    // forward over [input|accepted] advances it afterward.
    if (write_state) {
        #pragma unroll
        for (int i = 0; i < MAX_K_PER_LANE; ++i) {
            if (i < n_k) {
                state_store(
                    state + state_offset<KLast>(
                        k_vals[i], v_idx, K_d, V_d),
                    s_vals[i]);
            }
        }
    }
}

template <typename StateT, bool KLast>
__global__ void chunk_gated_delta_prefill_batched_warp_tiled_gqa_ilp2_kernel(
    const float* __restrict__ q_norm_kh,
    const float* __restrict__ k_norm_kh,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*       __restrict__ slot_ids,
    const std::uint32_t* __restrict__ qo_indptr,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d,
    bool write_state)
{
    constexpr int WARPS = 4;
    constexpr int ILP_V = 2;
    constexpr int TILE_V = WARPS * ILP_V;
    constexpr int MAX_K_PER_LANE = 8;  // supports K_d <= 256 with 32 lanes
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int v0 = blockIdx.z * TILE_V + warp * ILP_V;
    const int v1 = v0 + 1;
    if (warp >= WARPS || v0 >= V_d) return;
    const bool has_v1 = v1 < V_d;

    const int repeat = V_h / K_h;
    const int qk_h = h / repeat;
    const int t0 = static_cast<int>(qo_indptr[r]);
    const int T  = static_cast<int>(qo_indptr[r + 1]) - t0;
    if (T <= 0) return;

    const int slot = slot_ids[r];
    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;

    float s0[MAX_K_PER_LANE];
    float s1[MAX_K_PER_LANE];
    int k_vals[MAX_K_PER_LANE];
    int n_k = 0;
    for (int k_idx = lane; k_idx < K_d && n_k < MAX_K_PER_LANE; k_idx += 32) {
        k_vals[n_k] = k_idx;
        s0[n_k] = state_load(
            state + state_offset<KLast>(k_idx, v0, K_d, V_d));
        s1[n_k] = has_v1
            ? state_load(state + state_offset<KLast>(k_idx, v1, K_d, V_d))
            : 0.f;
        ++n_k;
    }

    for (int t = 0; t < T; ++t) {
        const long long qk_bh = ((long long)(t0 + t) * K_h + qk_h);
        const long long vh = (long long)(t0 + t) * V_h + h;
        const float* q_h = q_norm_kh + qk_bh * K_d;
        const float* k_h = k_norm_kh + qk_bh * K_d;
        const float* v_h = v + vh * V_d;
        const float g_h = __expf(g_log[vh]);
        const float beta_h = beta[vh];

        float kv_part0 = 0.f;
        float kv_part1 = 0.f;
        #pragma unroll
        for (int i = 0; i < MAX_K_PER_LANE; ++i) {
            if (i < n_k) {
                const int k_idx = k_vals[i];
                const float k_val = k_h[k_idx];
                const float s_v0 = s0[i] * g_h;
                s0[i] = s_v0;
                kv_part0 += s_v0 * k_val;
                if (has_v1) {
                    const float s_v1 = s1[i] * g_h;
                    s1[i] = s_v1;
                    kv_part1 += s_v1 * k_val;
                }
            }
        }
        const float kv_mem0 = warp_sum(kv_part0);
        const float kv_mem1 = has_v1 ? warp_sum(kv_part1) : 0.f;
        const float delta0 = (v_h[v0] - kv_mem0) * beta_h;
        const float delta1 = has_v1 ? (v_h[v1] - kv_mem1) * beta_h : 0.f;

        float out_part0 = 0.f;
        float out_part1 = 0.f;
        #pragma unroll
        for (int i = 0; i < MAX_K_PER_LANE; ++i) {
            if (i < n_k) {
                const int k_idx = k_vals[i];
                const float k_val = k_h[k_idx];
                const float q_val = q_h[k_idx];
                const float new_s0 = s0[i] + k_val * delta0;
                s0[i] = new_s0;
                out_part0 += new_s0 * q_val;
                if (has_v1) {
                    const float new_s1 = s1[i] + k_val * delta1;
                    s1[i] = new_s1;
                    out_part1 += new_s1 * q_val;
                }
            }
        }
        const float out_v0 = warp_sum(out_part0);
        const float out_v1 = has_v1 ? warp_sum(out_part1) : 0.f;
        if (lane == 0) {
            out[vh * (long long)V_d + v0] = out_v0;
            if (has_v1) out[vh * (long long)V_d + v1] = out_v1;
        }
    }

    // Frozen verify (write_state=false): persist nothing — see the non-ILP2
    // GQA kernel above.
    if (write_state) {
        #pragma unroll
        for (int i = 0; i < MAX_K_PER_LANE; ++i) {
            if (i < n_k) {
                state_store(
                    state + state_offset<KLast>(k_vals[i], v0, K_d, V_d),
                    s0[i]);
                if (has_v1) {
                    state_store(
                        state + state_offset<KLast>(k_vals[i], v1, K_d, V_d),
                        s1[i]);
                }
            }
        }
    }
}

// Batched variant with slot indirection. State for request r lives at
// `state_base + slot_ids[r] * slot_stride_elems`. Otherwise the
// per-(request, head) compute is identical to `recurrent_step_kernel`.
template <typename StateT, bool KLast>
__global__ void recurrent_step_batched_kernel(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*   __restrict__ slot_ids,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int V_h, int K_d, int V_d)
{
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int slot = slot_ids[r];

    const long long bh = (long long)r * V_h + h;
    const float* q_h = q_norm + bh * K_d;
    const float* k_h = k_norm + bh * K_d;
    const float* v_h = v      + bh * V_d;
    const float  g_h = __expf(g_log[bh]);
    const float  beta_h = beta[bh];

    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;
    float* out_bh = out + bh * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + K_d;

    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
    }
    __syncthreads();

    for (int v_idx = threadIdx.x; v_idx < V_d; v_idx += blockDim.x) {
        float kv_mem = 0.f;
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s = state_load(state + off) * g_h;
            state_store(state + off, s);
            kv_mem += s * sk[k_idx];
        }

        const float v_t   = v_h[v_idx];
        const float delta = (v_t - kv_mem) * beta_h;

        float out_v = 0.f;
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s = state_load(state + off) + sk[k_idx] * delta;
            state_store(state + off, s);
            out_v += s * sq[k_idx];
        }
        out_bh[v_idx] = out_v;
    }
}

template <typename StateT, bool KLast>
__global__ void recurrent_step_batched_gqa_kernel(
    const float* __restrict__ q_norm_kh,
    const float* __restrict__ k_norm_kh,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const std::int32_t* __restrict__ slot_ids,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d)
{
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int repeat = V_h / K_h;
    const int h_k = h / repeat;
    const int slot = slot_ids[r];

    const long long qh = ((long long)r * K_h + h_k) * K_d;
    const long long vh = (long long)r * V_h + h;
    const float* q_h = q_norm_kh + qh;
    const float* k_h = k_norm_kh + qh;
    const float* v_h = v + vh * V_d;
    const float  g_h = __expf(g_log[vh]);
    const float  beta_h = beta[vh];

    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;
    float* out_bh = out + vh * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + K_d;

    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
    }
    __syncthreads();

    for (int v_idx = threadIdx.x; v_idx < V_d; v_idx += blockDim.x) {
        float kv_mem = 0.f;
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s = state_load(state + off) * g_h;
            state_store(state + off, s);
            kv_mem += s * sk[k_idx];
        }

        const float v_t = v_h[v_idx];
        const float delta = (v_t - kv_mem) * beta_h;

        float out_v = 0.f;
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s = state_load(state + off) + sk[k_idx] * delta;
            state_store(state + off, s);
            out_v += s * sq[k_idx];
        }
        out_bh[v_idx] = out_v;
    }
}

// Fused recurrent-step kernel: same per-token math as
// `recurrent_step_kernel` / `recurrent_step_batched_kernel`, but
// reorganized to halve the state slab HBM traffic. The original
// kernel reads state, scales by g and writes back, then reads state
// again to apply delta and writes again (4 ops per element). The
// fused variant:
//
//   1. Reads state ONCE into a per-thread register cache `s_cache[K_d]`.
//   2. Accumulates sum_s_sk = Σ_k s[k]*sk[k] (proxy for kv_mem)
//      and  sum_s_sq = Σ_k s[k]*sq[k] (proxy for partial out_v).
//   3. Computes kv_mem = g * sum_s_sk, delta = (v - kv_mem) * beta.
//   4. Computes out_v = g * sum_s_sq + delta * sum_sk_sq, where
//      sum_sk_sq is a per-block constant precomputed once in shmem.
//   5. Writes the updated state once: state[k,v] = s_cache[k]*g + sk[k]*delta.
//
// Memory traffic per (head, batch, v_idx): K_d state reads + K_d
// state writes (1R+1W) vs the original's 2R+2W. Register footprint:
// K_d floats per thread, fine for K_d up to 256 on H100 (255-reg cap).
//
// Output equivalence:
//   final state[k,v] = s_initial[k,v] * g + sk[k] * delta
//   out_v          = Σ_k (s_initial[k,v]*g + sk[k]*delta) * sq[k]
//                  = g * Σ_k s_initial*sq + delta * Σ_k sk*sq
//                  = g * sum_s_sq + delta * sum_sk_sq
// — exactly the value the original kernel computes via its second
// state read. The analytical decomposition introduces no extra FLOPs
// (same 3 K_d FMAs per element) while saving half the state I/O.
//
// Template `K_D_MAX` bounds the per-thread state cache. We
// dispatch at launch time on the actual K_d.
template <typename StateT, bool KLast, int K_D_MAX>
__global__ void recurrent_step_batched_fused_kernel(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*   __restrict__ slot_ids,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int V_h, int K_d, int V_d)
{
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int slot = slot_ids[r];

    const long long bh = (long long)r * V_h + h;
    const float* q_h = q_norm + bh * K_d;
    const float* k_h = k_norm + bh * K_d;
    const float* v_h = v      + bh * V_d;
    const float  g_h = __expf(g_log[bh]);
    const float  beta_h = beta[bh];

    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;
    float* out_bh = out + bh * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + K_d;
    // sum_sk_sq is a per-block scalar (same value for every v_idx
    // since sk·sq depends only on the head). Reduce it cooperatively
    // and broadcast via shared memory; saves K_d FMAs per thread.
    float* sm_scalars = smem + 2 * K_d;  // [sum_sk_sq]

    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
    }
    __syncthreads();

    // Cooperative reduction of sum_sk_sq across the block.
    float partial = 0.f;
    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        partial += sk[i] * sq[i];
    }
    // Warp + block reduce.
    for (int offset = 16; offset > 0; offset /= 2) {
        partial += __shfl_xor_sync(0xffffffffu, partial, offset);
    }
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp_id] = partial;
    __syncthreads();
    if (warp_id == 0) {
        const int num_warps = (blockDim.x + 31) >> 5;
        float w = (threadIdx.x < num_warps) ? warp_sums[lane] : 0.f;
        for (int offset = 16; offset > 0; offset /= 2) {
            w += __shfl_xor_sync(0xffffffffu, w, offset);
        }
        if (threadIdx.x == 0) sm_scalars[0] = w;
    }
    __syncthreads();
    const float sum_sk_sq = sm_scalars[0];

    // Per-thread state cache. K_D_MAX bounds the static array; we
    // only ever touch [0, K_d). Sized for the worst case across
    // instantiations (currently K_d <= 256 for Qwen3.5 family).
    float s_cache[K_D_MAX];

    for (int v_idx = threadIdx.x; v_idx < V_d; v_idx += blockDim.x) {
        // Pass 1: read state, cache, accumulate kv_mem & out_v partials.
        float sum_s_sk = 0.f;
        float sum_s_sq = 0.f;
        #pragma unroll 4
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s = state_load(state + off);
            s_cache[k_idx] = s;
            sum_s_sk += s * sk[k_idx];
            sum_s_sq += s * sq[k_idx];
        }

        const float kv_mem = g_h * sum_s_sk;
        const float v_t    = v_h[v_idx];
        const float delta  = (v_t - kv_mem) * beta_h;
        // out_v = g * Σ s*sq + delta * Σ sk*sq, an algebraic
        // rewrite of the original Phase-2 reduction.
        const float out_v  = g_h * sum_s_sq + delta * sum_sk_sq;
        out_bh[v_idx] = out_v;

        // Pass 2: write updated state.
        #pragma unroll 4
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s_new = s_cache[k_idx] * g_h + sk[k_idx] * delta;
            state_store(state + off, s_new);
        }
    }
}

template <typename StateT, bool KLast, int K_D_MAX>
__global__ void recurrent_step_batched_gqa_fused_kernel(
    const float* __restrict__ q_norm_kh,
    const float* __restrict__ k_norm_kh,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const std::int32_t* __restrict__ slot_ids,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d)
{
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int repeat = V_h / K_h;
    const int h_k = h / repeat;
    const int slot = slot_ids[r];

    const long long qh = ((long long)r * K_h + h_k) * K_d;
    const long long vh = (long long)r * V_h + h;
    const float* q_h = q_norm_kh + qh;
    const float* k_h = k_norm_kh + qh;
    const float* v_h = v + vh * V_d;
    const float  g_h = __expf(g_log[vh]);
    const float  beta_h = beta[vh];

    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;
    float* out_bh = out + vh * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + K_d;
    float* sm_scalars = smem + 2 * K_d;

    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
    }
    __syncthreads();

    float partial = 0.f;
    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        partial += sk[i] * sq[i];
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        partial += __shfl_xor_sync(0xffffffffu, partial, offset);
    }
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp_id] = partial;
    __syncthreads();
    if (warp_id == 0) {
        const int num_warps = (blockDim.x + 31) >> 5;
        float w = (threadIdx.x < num_warps) ? warp_sums[lane] : 0.f;
        for (int offset = 16; offset > 0; offset /= 2) {
            w += __shfl_xor_sync(0xffffffffu, w, offset);
        }
        if (threadIdx.x == 0) sm_scalars[0] = w;
    }
    __syncthreads();
    const float sum_sk_sq = sm_scalars[0];

    float s_cache[K_D_MAX];

    for (int v_idx = threadIdx.x; v_idx < V_d; v_idx += blockDim.x) {
        float sum_s_sk = 0.f;
        float sum_s_sq = 0.f;
        #pragma unroll 4
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s = state_load(state + off);
            s_cache[k_idx] = s;
            sum_s_sk += s * sk[k_idx];
            sum_s_sq += s * sq[k_idx];
        }

        const float kv_mem = g_h * sum_s_sk;
        const float v_t    = v_h[v_idx];
        const float delta  = (v_t - kv_mem) * beta_h;
        const float out_v  = g_h * sum_s_sq + delta * sum_sk_sq;
        out_bh[v_idx] = out_v;

        #pragma unroll 4
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s_new = s_cache[k_idx] * g_h + sk[k_idx] * delta;
            state_store(state + off, s_new);
        }
    }
}

// ── FLA-style recurrent step (KLast=false / V-last storage only) ──
// C++ port of fla-org/flash-linear-attention's
// fused_recurrent_gated_delta_rule_fwd_kernel for T=1 decode.
//
// Grid: (NV, R, V_h) where NV = ceil(V_d / BV).
// Block: BV threads — each owns one V column of the [K_d, BV] state
//        tile, keeping the per-column state vector in registers and
//        sharing the q/k vectors via shmem.
//
// Microbench on H100 PCIe at R=511, V_h=K_d=V_d=128:
//   legacy kernel    1.92 ms/call  (567 GB/s, 19% of HBM3 peak)
//   FLA-port BV=64   1.22 ms/call  (894 GB/s, +37%, bit-identical)
//
// Only handles KLast=false (V-last) — KLast=true is non-coalesced
// and is the layout the production default already moved off.
template <typename StateT, int BV, int BK_MAX>
__global__ void recurrent_step_batched_fla_kernel(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*   __restrict__ slot_ids,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int V_h, int K_d, int V_d)
{
    const int vt = blockIdx.x;
    const int r  = blockIdx.y;
    const int h  = blockIdx.z;
    const int slot = slot_ids[r];

    const int v_idx = vt * BV + threadIdx.x;
    if (v_idx >= V_d) return;

    const long long bh = (long long)r * V_h + h;
    const float* q_h = q_norm + bh * K_d;
    const float* k_h = k_norm + bh * K_d;
    const float* v_h = v      + bh * V_d;
    const float  g_h = __expf(g_log[bh]);
    const float  beta_h = beta[bh];

    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;
    float* out_bh = out + bh * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + BK_MAX;
    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
    }
    __syncthreads();

    float bh_state[BK_MAX];
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        bh_state[k_idx] = state_load(state + (long long)k_idx * V_d + v_idx);
    }
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        bh_state[k_idx] *= g_h;
    }
    float kv_mem = 0.f;
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        kv_mem += bh_state[k_idx] * sk[k_idx];
    }
    const float v_t   = v_h[v_idx];
    const float delta = (v_t - kv_mem) * beta_h;
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        bh_state[k_idx] += sk[k_idx] * delta;
    }
    float out_v = 0.f;
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        out_v += bh_state[k_idx] * sq[k_idx];
    }
    out_bh[v_idx] = out_v;
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        state_store(state + (long long)k_idx * V_d + v_idx, bh_state[k_idx]);
    }
}

template <typename StateT, int BV, int BK_MAX>
__global__ void recurrent_step_batched_gqa_fla_kernel(
    const float* __restrict__ q_norm_kh,
    const float* __restrict__ k_norm_kh,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*   __restrict__ slot_ids,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d)
{
    const int vt = blockIdx.x;
    const int r  = blockIdx.y;
    const int h  = blockIdx.z;
    const int repeat = V_h / K_h;
    const int h_k = h / repeat;
    const int slot = slot_ids[r];

    const int v_idx = vt * BV + threadIdx.x;
    if (v_idx >= V_d) return;

    const long long qh = ((long long)r * K_h + h_k) * K_d;
    const long long vh = (long long)r * V_h + h;
    const float* q_h = q_norm_kh + qh;
    const float* k_h = k_norm_kh + qh;
    const float* v_h = v + vh * V_d;
    const float  g_h = __expf(g_log[vh]);
    const float  beta_h = beta[vh];

    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;
    float* out_bh = out + vh * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + BK_MAX;
    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
    }
    __syncthreads();

    float bh_state[BK_MAX];
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        bh_state[k_idx] = state_load(state + (long long)k_idx * V_d + v_idx);
    }
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        bh_state[k_idx] *= g_h;
    }
    float kv_mem = 0.f;
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        kv_mem += bh_state[k_idx] * sk[k_idx];
    }
    const float v_t   = v_h[v_idx];
    const float delta = (v_t - kv_mem) * beta_h;
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        bh_state[k_idx] += sk[k_idx] * delta;
    }
    float out_v = 0.f;
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        out_v += bh_state[k_idx] * sq[k_idx];
    }
    out_bh[v_idx] = out_v;
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        state_store(state + (long long)k_idx * V_d + v_idx, bh_state[k_idx]);
    }
}

// ── SMEM read-only step kernel (KLast=false / V-last only) ───────
// Stages the V-last state slab into SMEM ONCE (read-only thereafter),
// computes both analytical phases reading from SMEM, and writes the
// updated state straight to HBM from phase 2 — no SMEM writebacks.
// Wins over the legacy 2R+2W kernel by halving HBM round-trips on the
// state slab; wins over the naive SMEM-staging variant by halving
// SMEM traffic (the binding bottleneck: the kernel is SMEM-read-
// bandwidth bound, not HBM bound). State is read once per phase from
// SMEM; phase 2 recomputes `s*g` (one extra FMA) rather than reading
// a stored scaled value, so the only SMEM writes are the one-time
// stage. q/k go into SMEM once with a [k][v_local] layout so adjacent
// threads touch adjacent offsets (coalesced).
//
// Microbench at R=511, V_h=32, K_d=V_d=128, BF16 state on H100 PCIe:
//   ref bf16 v-last (2R+2W)        2406 us  445 GB/s
//   FLA burst regs                 3632 us  296 GB/s (register spill)
//   cp.async SMEM regs             1989 us  539 GB/s
//   SMEM-staging (RMW writebacks)  1660 us  646 GB/s
//   fp32 SMEM (2x read bytes)      2691 us  398 GB/s
//   SMEM read-only (this)          1579 us  679 GB/s  ← best
//
// Precision: fp32 accumulate, rounded to BF16 once on the final
// store (same scheme as the FLA chunked-prefill path, which is the
// default). Not bit-identical to the legacy per-element-BF16-round
// kernel, but strictly less quantization.
template <int BV>
__global__ void recurrent_step_batched_gqa_smem_kernel(
    const float* __restrict__ q_norm_kh,
    const float* __restrict__ k_norm_kh,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ state_base,
    const std::int32_t* __restrict__ slot_ids,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d)
{
    const int vt = blockIdx.x;
    const int r  = blockIdx.y;
    const int h  = blockIdx.z;
    const int v_idx = vt * BV + threadIdx.x;
    if (v_idx >= V_d) return;
    const int repeat = V_h / K_h;
    const int h_k = h / repeat;
    const int slot = slot_ids[r];

    const long long qh = ((long long)r * K_h + h_k) * K_d;
    const long long vh = (long long)r * V_h + h;
    const float* v_h = v + vh * V_d;
    const float g_h = __expf(g_log[vh]);
    const float beta_h = beta[vh];

    __nv_bfloat16* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;
    float* out_bh = out + vh * V_d;

    extern __shared__ __nv_bfloat16 smem_smem_step[];
    __nv_bfloat16* s_state = smem_smem_step;          // read-only after stage
    float* sq = (float*)(smem_smem_step + K_d * BV);
    float* sk = sq + K_d;

    // Stage state HBM → SMEM. Adjacent threads at fixed k load
    // adjacent v indices → coalesced HBM reads; SMEM layout is
    // [k][v_local] → coalesced SMEM writes.
    for (int k = 0; k < K_d; ++k) {
        s_state[k * BV + threadIdx.x] =
            state[(long long)k * V_d + v_idx];
    }
    const float* q_h = q_norm_kh + qh;
    const float* k_h = k_norm_kh + qh;
    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
    }
    __syncthreads();

    // Phase 1: kv_mem = Σ (state*g) * sk.  No SMEM writes.
    float kv_mem = 0.f;
    for (int k = 0; k < K_d; ++k) {
        float s = __bfloat162float(s_state[k * BV + threadIdx.x]) * g_h;
        kv_mem += s * sk[k];
    }
    const float delta = (v_h[v_idx] - kv_mem) * beta_h;

    // Phase 2: recompute s = state*g + sk*delta; accumulate out_v;
    // write the updated state straight to HBM (skips SMEM writeback).
    float out_v = 0.f;
    for (int k = 0; k < K_d; ++k) {
        float s = __bfloat162float(s_state[k * BV + threadIdx.x]) * g_h
                + sk[k] * delta;
        out_v += s * sq[k];
        state[(long long)k * V_d + v_idx] = __float2bfloat16(s);
    }
    out_bh[v_idx] = out_v;
}

// Opt-in FLA-port path (PIE_QWEN35_GDN_FLA_STEP=1).
inline bool qwen_gdn_fla_step_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_GDN_FLA_STEP");
        if (v == nullptr || v[0] == '\0') return false;
        return v[0] != '0';
    }();
    return enabled;
}

// ── FLA-style chunked prefill kernel (KLast=false / V-last storage) ──
// State stays in per-thread registers across the whole T-token chunk,
// only one HBM round-trip per (request, head). The reference
// `chunk_gated_delta_prefill_batched_kernel` does 2R+2W per state
// element PER TOKEN — i.e. T-fold blowup of state HBM traffic.
//
// Microbench at production shapes (R=512, V_h=K_d=V_d=128, T=36 tokens
// per request) on H100 PCIe:
//   ref (per-token HBM)  47.5 ms/call  (~38 GB of state IO per call)
//   v1 FLA BV=128         5.27 ms/call (~1 GB,  9.0x faster, bit-identical)
//   v1 FLA BV=64          5.57 ms/call (8.5x)
//   v1 FLA BV=32          6.32 ms/call (7.5x)
//
// Picked BV=128 (matches the legacy block shape — 128 threads per
// (request, head) block, one v_idx per thread, state cached in
// 128 floats of registers). Output is bit-identical to the legacy
// kernel — the math is exactly the same per-token recurrence, only
// the in-block state staging changes.
template <typename StateT, int BV, int BK_MAX>
__global__ void chunk_gated_delta_prefill_batched_fla_kernel(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*       __restrict__ slot_ids,
    const std::uint32_t* __restrict__ qo_indptr,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d,
    bool write_state,
    const int* __restrict__ commit_len)
{
    // GQA: q/k are stored compact in K_h heads; g/beta/v/out are V_h heads.
    // Block head h (0..V_h) maps to its K-head h_k = h / (V_h/K_h). When
    // K_h == V_h (no GQA) this is the identity h_k = h, so callers that pass
    // the already-expanded V_h layout with K_h = V_h are unaffected.
    const int gqa_repeat = V_h / K_h;
    // Grid: (NV, R, V_h) where NV = ceil(V_d / BV). Each block owns
    // one V-tile of BV columns. Lowering BV from V_d (single block per
    // (r, h)) to smaller values raises grid parallelism, reduces
    // per-block register pressure, and lets the SM scheduler hide
    // memory latency across more in-flight warps.
    const int vt = blockIdx.x;
    const int r  = blockIdx.y;
    const int h  = blockIdx.z;
    const int h_k = h / gqa_repeat;   // K-head feeding this V-head (GQA)
    const int v_idx = vt * BV + threadIdx.x;
    if (v_idx >= V_d) return;

    const int t0 = static_cast<int>(qo_indptr[r]);
    int T  = static_cast<int>(qo_indptr[r + 1]) - t0;
    // Boundary-write (recurrent-only commit-advance): the in-projection ran
    // over the full [input|drafts] window (matching the verify's GEMM tiling),
    // but we only fold the confirmed prefix [input|accepted] into the committed
    // state. Walk just commit_len[r] tokens; the final writeback then lands at
    // the accepted boundary. Rejected drafts (later positions) are never folded.
    if (commit_len != nullptr) {
        const int c = commit_len[r];
        if (c < T) T = c;
    }
    if (T <= 0) return;

    const int slot = slot_ids[r];
    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + BK_MAX;

    // Load the state column [0..K_d, v_idx] into a per-thread register cache,
    // PACKED 2 K-values per register as __nv_bfloat162. This halves the
    // register footprint of the dominant array (128 floats -> 64 bf162),
    // raising occupancy (~19% -> ~31%) so the SM can hide the state-I/O
    // latency that bounds this kernel. All arithmetic still accumulates in
    // fp32 (the packed state is unpacked to float2 per access); only the
    // per-token state value is held in bf16 — which matches the bf16 storage
    // precision the committed slot already uses.
    __nv_bfloat162 bh_state[BK_MAX / 2];
    #pragma unroll
    for (int j = 0; j < BK_MAX / 2; ++j) {
        const int k0 = 2 * j;
        if (k0 >= K_d) break;
        const int k1 = k0 + 1;
        const float s0 = state_load(state + (long long)k0 * V_d + v_idx);
        const float s1 = (k1 < K_d)
            ? state_load(state + (long long)k1 * V_d + v_idx)
            : 0.f;
        bh_state[j] = __floats2bfloat162_rn(s0, s1);
    }

    // COMMIT-LEN-GATED rounding (verified on the 4090): the SAME kernel
    // serves two ops with DIFFERENT bit-exactness references:
    //   * commit_len == nullptr (plain PREFILL, K=0 & K=2 initial): fold N fresh
    //     tokens into a reset state. The HF reference (T0 GDN_GOLDEN) matches the
    //     DOUBLE-round trajectory here — single-round diverges (T0 glitch, the
    //     warp-tiled-bug signature). So plain prefill KEEPS double-round.
    //   * commit_len != nullptr (COMMIT-ADVANCE replay [input|accepted]): must
    //     bit-match the K=0 decode-step kernel (single bf16 round/token) so the
    //     spec-verify is lossless → SINGLE-round (fixes T1 K=0==K=2).
    // Verified: gating → T0 HF-exact (golden) AND T1 K=0==K=2 both green; ungated
    // single-round gave T1 green but T0 red (both glitch); double-round-only gave
    // T0 green but T1 red. (:1625 is shared prefill+commit-advance.)
    const bool single_round = (commit_len != nullptr);

    // Walk T tokens; state stays in registers.
    for (int t = 0; t < T; ++t) {
        const long long bh = (long long)(t0 + t) * V_h + h;
        const float  g_h = __expf(g_log[bh]);
        const float  beta_h = beta[bh];

        // Reload q/k into shmem for this token. q/k are compact in K_h heads.
        const long long bh_qk = (long long)(t0 + t) * K_h + h_k;
        const float* q_h_t = q_norm + bh_qk * K_d;
        const float* k_h_t = k_norm + bh_qk * K_d;
        for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
            sq[i] = q_h_t[i];
            sk[i] = k_h_t[i];
        }
        __syncthreads();

        // Phase 1: accumulate kv_mem = Σ (state*g)·sk (fp32). single_round leaves
        // bh_state untouched (Phase 2 recomputes state*g from the original);
        // double_round re-packs the g-scaled state into bh_state (the extra round).
        float kv_mem = 0.f;
        #pragma unroll
        for (int j = 0; j < BK_MAX / 2; ++j) {
            const int k0 = 2 * j;
            if (k0 >= K_d) break;
            const int k1 = k0 + 1;
            float2 s = __bfloat1622float2(bh_state[j]);
            s.x *= g_h;
            if (k1 < K_d) s.y *= g_h;
            if (!single_round) bh_state[j] = __floats2bfloat162_rn(s.x, s.y);
            kv_mem += s.x * sk[k0];
            if (k1 < K_d) kv_mem += s.y * sk[k1];
        }
        const float v_t   = v[bh * V_d + v_idx];
        const float delta = (v_t - kv_mem) * beta_h;

        // Phase 2: state = state*g + k·δ, accumulate out_v (fp32). single_round
        // recomputes state*g fresh from the ORIGINAL bh_state (one round total);
        // double_round reloads the already-g-scaled-and-rounded bh_state from
        // Phase 1 and adds k·δ (a second round) — matches HF for the plain prefill.
        float out_v = 0.f;
        #pragma unroll
        for (int j = 0; j < BK_MAX / 2; ++j) {
            const int k0 = 2 * j;
            if (k0 >= K_d) break;
            const int k1 = k0 + 1;
            float2 s = __bfloat1622float2(bh_state[j]);
            float sx, sy;
            if (single_round) {
                sx = s.x * g_h + sk[k0] * delta;
                sy = (k1 < K_d) ? (s.y * g_h + sk[k1] * delta) : s.y;
            } else {
                // bh_state already holds round(state*g) from Phase 1.
                sx = s.x + sk[k0] * delta;
                sy = (k1 < K_d) ? (s.y + sk[k1] * delta) : s.y;
            }
            bh_state[j] = __floats2bfloat162_rn(sx, sy);
            out_v += sx * sq[k0];
            if (k1 < K_d) out_v += sy * sq[k1];
        }
        out[bh * V_d + v_idx] = out_v;
        __syncthreads();
    }

    // Single state writeback at the (possibly commit_len-clamped) chunk end.
    // ACTIVATION REPLAY: honor write_state. Frozen verify (write_state=false)
    // persists nothing — it emits draft logits against the correct committed
    // state (the model's true ~54%/c256, ~65%/c1 acceptance; the not-frozen 90%
    // is corrupt co-drift). On accept, the commit-advance replays the recurrence
    // over just the accepted prefix (write_state=true) — lossless, cheap. This is
    // the SoTA (Snakes-and-Ladders / STree) and the right design for the low-
    // concurrency regime MTP targets.
    if (!write_state) return;
    #pragma unroll
    for (int j = 0; j < BK_MAX / 2; ++j) {
        const int k0 = 2 * j;
        if (k0 >= K_d) break;
        const int k1 = k0 + 1;
        const float2 s = __bfloat1622float2(bh_state[j]);
        state_store(state + (long long)k0 * V_d + v_idx, s.x);
        if (k1 < K_d) state_store(state + (long long)k1 * V_d + v_idx, s.y);
    }
}

// Same kernel for GQA variant (only the (q, k) indexing differs —
// they're loaded once per token via the shared shmem so the structural
// loop is identical to the non-GQA path; only the per-token bh index
// changes).
template <typename StateT, int BV, int BK_MAX>
__global__ void chunk_gated_delta_prefill_batched_gqa_fla_kernel(
    const float* __restrict__ q_norm_kh,
    const float* __restrict__ k_norm_kh,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*       __restrict__ slot_ids,
    const std::uint32_t* __restrict__ qo_indptr,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d)
{
    const int vt = blockIdx.x;
    const int r  = blockIdx.y;
    const int h  = blockIdx.z;
    const int v_idx = vt * BV + threadIdx.x;
    if (v_idx >= V_d) return;
    const int repeat = V_h / K_h;
    const int h_k = h / repeat;

    const int t0 = static_cast<int>(qo_indptr[r]);
    const int T  = static_cast<int>(qo_indptr[r + 1]) - t0;
    if (T <= 0) return;

    const int slot = slot_ids[r];
    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + BK_MAX;

    float bh_state[BK_MAX];
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        bh_state[k_idx] = state_load(state + (long long)k_idx * V_d + v_idx);
    }

    for (int t = 0; t < T; ++t) {
        const long long qh = ((long long)(t0 + t) * K_h + h_k) * K_d;
        const long long vh = (long long)(t0 + t) * V_h + h;
        const float  g_h = __expf(g_log[vh]);
        const float  beta_h = beta[vh];

        const float* q_h_t = q_norm_kh + qh;
        const float* k_h_t = k_norm_kh + qh;
        for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
            sq[i] = q_h_t[i];
            sk[i] = k_h_t[i];
        }
        __syncthreads();

        float kv_mem = 0.f;
        #pragma unroll
        for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
            if (k_idx >= K_d) break;
            bh_state[k_idx] *= g_h;
            kv_mem += bh_state[k_idx] * sk[k_idx];
        }
        const float v_t   = v[vh * V_d + v_idx];
        const float delta = (v_t - kv_mem) * beta_h;

        float out_v = 0.f;
        #pragma unroll
        for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
            if (k_idx >= K_d) break;
            bh_state[k_idx] += sk[k_idx] * delta;
            out_v += bh_state[k_idx] * sq[k_idx];
        }
        out[vh * V_d + v_idx] = out_v;
        __syncthreads();
    }

    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        state_store(state + (long long)k_idx * V_d + v_idx, bh_state[k_idx]);
    }
}

inline bool qwen_gdn_fla_prefill_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_GDN_FLA_PREFILL");
        // Default ON: 9x speedup, bit-identical to the legacy kernel
        // at production shapes (V_d=128, K_d<=128). Set the env var
        // to '0' to fall back to the legacy per-token HBM kernel.
        if (v == nullptr || v[0] == '\0') return true;
        return v[0] != '0';
    }();
    return enabled;
}

}  // namespace

void launch_recurrent_gated_delta_step(
    const float* q_norm, const float* k_norm, const float* v,
    const float* g_log, const float* beta,
    float* state, float* out,
    int B, int V_h, int K_d, int V_d,
    cudaStream_t stream)
{
    if (B <= 0 || V_h <= 0 || K_d <= 0 || V_d <= 0) return;
    constexpr int BLOCK = 128;
    dim3 grid(B, V_h);
    dim3 block(BLOCK);
    const int shmem_bytes = 2 * K_d * sizeof(float);
    if (qwen_gdn_k_last_state_enabled()) {
        recurrent_step_kernel<float, true><<<grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta, state, out, V_h, K_d, V_d);
    } else {
        recurrent_step_kernel<float, false><<<grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta, state, out, V_h, K_d, V_d);
    }
}

void launch_recurrent_gated_delta_step_state_bf16(
    const float* q_norm, const float* k_norm, const float* v,
    const float* g_log, const float* beta,
    void* state, float* out,
    int B, int V_h, int K_d, int V_d,
    cudaStream_t stream)
{
    if (B <= 0 || V_h <= 0 || K_d <= 0 || V_d <= 0) return;
    constexpr int BLOCK = 128;
    dim3 grid(B, V_h);
    dim3 block(BLOCK);
    const int shmem_bytes = 2 * K_d * sizeof(float);
    if (qwen_gdn_k_last_state_enabled()) {
        recurrent_step_kernel<__nv_bfloat16, true><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta,
            static_cast<__nv_bfloat16*>(state), out, V_h, K_d, V_d);
    } else {
        recurrent_step_kernel<__nv_bfloat16, false><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta,
            static_cast<__nv_bfloat16*>(state), out, V_h, K_d, V_d);
    }
}

void launch_recurrent_gated_delta_step_batched(
    const float* q_norm, const float* k_norm, const float* v,
    const float* g_log, const float* beta,
    float* state_base,
    const std::int32_t* slot_ids,
    long long slot_stride_elems,
    float* out,
    int R, int V_h, int K_d, int V_d,
    cudaStream_t stream)
{
    if (R <= 0 || V_h <= 0 || K_d <= 0 || V_d <= 0) return;
    constexpr int BLOCK = 128;
    dim3 grid(R, V_h);
    dim3 block(BLOCK);
    // Fused kernel needs the existing sq+sk shmem plus one float
    // scalar (sum_sk_sq broadcast); legacy kernel only needs the
    // first two arrays.
    const bool fused = qwen_gdn_fused_step_enabled() && K_d <= 256;
    const int shmem_bytes = (2 * K_d + (fused ? 1 : 0)) * sizeof(float);
    if (fused) {
        // K_d up to 256 covers every qwen3_5 GDN config currently in
        // production (E4B family is K_d=128). Dispatch on the bound
        // so the per-thread state_cache array is small enough to fit
        // in registers without spilling. We dispatch on the maximum
        // K_d, not the actual: the kernel only iterates [0, K_d) so
        // unused slots are dead code.
        if (qwen_gdn_k_last_state_enabled()) {
            recurrent_step_batched_fused_kernel<float, true, 256><<<
                grid, block, shmem_bytes, stream>>>(
                q_norm, k_norm, v, g_log, beta, state_base,
                slot_ids, slot_stride_elems, out, V_h, K_d, V_d);
        } else {
            recurrent_step_batched_fused_kernel<float, false, 256><<<
                grid, block, shmem_bytes, stream>>>(
                q_norm, k_norm, v, g_log, beta, state_base,
                slot_ids, slot_stride_elems, out, V_h, K_d, V_d);
        }
        return;
    }
    if (qwen_gdn_k_last_state_enabled()) {
        recurrent_step_batched_kernel<float, true><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta, state_base,
            slot_ids, slot_stride_elems, out, V_h, K_d, V_d);
    } else {
        recurrent_step_batched_kernel<float, false><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta, state_base,
            slot_ids, slot_stride_elems, out, V_h, K_d, V_d);
    }
}

void launch_recurrent_gated_delta_step_batched_state_bf16(
    const float* q_norm, const float* k_norm, const float* v,
    const float* g_log, const float* beta,
    void* state_base,
    const std::int32_t* slot_ids,
    long long slot_stride_elems,
    float* out,
    int R, int V_h, int K_d, int V_d,
    cudaStream_t stream)
{
    if (R <= 0 || V_h <= 0 || K_d <= 0 || V_d <= 0) return;
    constexpr int BLOCK = 128;
    dim3 grid(R, V_h);
    dim3 block(BLOCK);
    const bool fused = qwen_gdn_fused_step_enabled() && K_d <= 256;
    const int shmem_bytes = (2 * K_d + (fused ? 1 : 0)) * sizeof(float);
    if (fused) {
        if (qwen_gdn_k_last_state_enabled()) {
            recurrent_step_batched_fused_kernel<__nv_bfloat16, true, 256><<<
                grid, block, shmem_bytes, stream>>>(
                q_norm, k_norm, v, g_log, beta,
                static_cast<__nv_bfloat16*>(state_base),
                slot_ids, slot_stride_elems, out, V_h, K_d, V_d);
        } else {
            recurrent_step_batched_fused_kernel<__nv_bfloat16, false, 256><<<
                grid, block, shmem_bytes, stream>>>(
                q_norm, k_norm, v, g_log, beta,
                static_cast<__nv_bfloat16*>(state_base),
                slot_ids, slot_stride_elems, out, V_h, K_d, V_d);
        }
        return;
    }
    if (qwen_gdn_k_last_state_enabled()) {
        recurrent_step_batched_kernel<__nv_bfloat16, true><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta,
            static_cast<__nv_bfloat16*>(state_base),
            slot_ids, slot_stride_elems, out, V_h, K_d, V_d);
    } else {
        recurrent_step_batched_kernel<__nv_bfloat16, false><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta,
            static_cast<__nv_bfloat16*>(state_base),
            slot_ids, slot_stride_elems, out, V_h, K_d, V_d);
    }
}

void launch_recurrent_gated_delta_step_batched_gqa(
    const float* q_norm_kh, const float* k_norm_kh, const float* v,
    const float* g_log, const float* beta,
    float* state_base,
    const std::int32_t* slot_ids,
    long long slot_stride_elems,
    float* out,
    int R, int K_h, int V_h, int K_d, int V_d,
    cudaStream_t stream)
{
    if (R <= 0 || K_h <= 0 || V_h <= 0 || K_d <= 0 || V_d <= 0) return;
    if (V_h % K_h != 0) return;
    // FLA-style fast path (opt-in via PIE_QWEN35_GDN_FLA_STEP=1).
    // Requires KLast=false (the production default) and K_d <= 128.
    constexpr int BK_MAX_FLA = 128;
    constexpr int BV_FLA     = 64;
    if (qwen_gdn_fla_step_enabled() &&
        !qwen_gdn_k_last_state_enabled() &&
        K_d <= BK_MAX_FLA && V_d % BV_FLA == 0) {
        const int NV = V_d / BV_FLA;
        dim3 grid_fla(NV, R, V_h);
        dim3 block_fla(BV_FLA);
        const int shmem_bytes = 2 * BK_MAX_FLA * sizeof(float);
        recurrent_step_batched_gqa_fla_kernel<float, BV_FLA, BK_MAX_FLA><<<
            grid_fla, block_fla, shmem_bytes, stream>>>(
            q_norm_kh, k_norm_kh, v, g_log, beta, state_base,
            slot_ids, slot_stride_elems, out, K_h, V_h, K_d, V_d);
        return;
    }
    constexpr int BLOCK = 128;
    dim3 grid(R, V_h);
    dim3 block(BLOCK);
    const bool fused = qwen_gdn_fused_step_enabled() && K_d <= 256;
    const int shmem_bytes = (2 * K_d + (fused ? 1 : 0)) * sizeof(float);
    if (fused) {
        if (qwen_gdn_k_last_state_enabled()) {
            recurrent_step_batched_gqa_fused_kernel<float, true, 256><<<
                grid, block, shmem_bytes, stream>>>(
                q_norm_kh, k_norm_kh, v, g_log, beta, state_base,
                slot_ids, slot_stride_elems, out, K_h, V_h, K_d, V_d);
        } else {
            recurrent_step_batched_gqa_fused_kernel<float, false, 256><<<
                grid, block, shmem_bytes, stream>>>(
                q_norm_kh, k_norm_kh, v, g_log, beta, state_base,
                slot_ids, slot_stride_elems, out, K_h, V_h, K_d, V_d);
        }
        return;
    }
    if (qwen_gdn_k_last_state_enabled()) {
        recurrent_step_batched_gqa_kernel<float, true><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm_kh, k_norm_kh, v, g_log, beta, state_base,
            slot_ids, slot_stride_elems, out, K_h, V_h, K_d, V_d);
    } else {
        recurrent_step_batched_gqa_kernel<float, false><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm_kh, k_norm_kh, v, g_log, beta, state_base,
            slot_ids, slot_stride_elems, out, K_h, V_h, K_d, V_d);
    }
}

void launch_recurrent_gated_delta_step_batched_gqa_state_bf16(
    const float* q_norm_kh, const float* k_norm_kh, const float* v,
    const float* g_log, const float* beta,
    void* state_base,
    const std::int32_t* slot_ids,
    long long slot_stride_elems,
    float* out,
    int R, int K_h, int V_h, int K_d, int V_d,
    cudaStream_t stream)
{
    if (R <= 0 || K_h <= 0 || V_h <= 0 || K_d <= 0 || V_d <= 0) return;
    if (V_h % K_h != 0) return;
    // SMEM-only fast path — wins on the saturated decode shape used by
    // Qwen3.5. Requires KLast=false (V-last) state storage, which is
    // the default since the KLast bool flip.
    if (qwen_gdn_smem_step_enabled() &&
        !qwen_gdn_k_last_state_enabled() &&
        V_d == 128 && K_d == 128) {
        constexpr int BV = 128;
        dim3 grid_smem((V_d + BV - 1) / BV, R, V_h);
        dim3 block_smem(BV);
        const int shmem_bytes_smem =
            K_d * BV * sizeof(__nv_bfloat16) + 2 * K_d * sizeof(float);
        recurrent_step_batched_gqa_smem_kernel<BV><<<
            grid_smem, block_smem, shmem_bytes_smem, stream>>>(
            q_norm_kh, k_norm_kh, v, g_log, beta,
            static_cast<__nv_bfloat16*>(state_base),
            slot_ids, slot_stride_elems, out, K_h, V_h, K_d, V_d);
        return;
    }
    constexpr int BLOCK = 128;
    dim3 grid(R, V_h);
    dim3 block(BLOCK);
    const bool fused = qwen_gdn_fused_step_enabled() && K_d <= 256;
    const int shmem_bytes = (2 * K_d + (fused ? 1 : 0)) * sizeof(float);
    if (fused) {
        if (qwen_gdn_k_last_state_enabled()) {
            recurrent_step_batched_gqa_fused_kernel<__nv_bfloat16, true, 256><<<
                grid, block, shmem_bytes, stream>>>(
                q_norm_kh, k_norm_kh, v, g_log, beta,
                static_cast<__nv_bfloat16*>(state_base),
                slot_ids, slot_stride_elems, out, K_h, V_h, K_d, V_d);
        } else {
            recurrent_step_batched_gqa_fused_kernel<__nv_bfloat16, false, 256><<<
                grid, block, shmem_bytes, stream>>>(
                q_norm_kh, k_norm_kh, v, g_log, beta,
                static_cast<__nv_bfloat16*>(state_base),
                slot_ids, slot_stride_elems, out, K_h, V_h, K_d, V_d);
        }
        return;
    }
    if (qwen_gdn_k_last_state_enabled()) {
        recurrent_step_batched_gqa_kernel<__nv_bfloat16, true><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm_kh, k_norm_kh, v, g_log, beta,
            static_cast<__nv_bfloat16*>(state_base),
            slot_ids, slot_stride_elems, out, K_h, V_h, K_d, V_d);
    } else {
        recurrent_step_batched_gqa_kernel<__nv_bfloat16, false><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm_kh, k_norm_kh, v, g_log, beta,
            static_cast<__nv_bfloat16*>(state_base),
            slot_ids, slot_stride_elems, out, K_h, V_h, K_d, V_d);
    }
}

// Chunked prefill — for now, implemented as a sequential per-token
// loop over `launch_recurrent_gated_delta_step`. Mathematically
// identical to the chunked algorithm, just leaves chunk-parallelism
// on the table. Each recurrent step is a single grid launch of
// (1, V_h) blocks, so a T-token prefill costs T launches plus the
// state-dependent recurrence chain — roughly the same FLOPs as the
// fast chunked path but no chunk-level parallelism.
//
// TODO(perf): replace with the chunked algorithm from
// `torch_chunk_gated_delta_rule` once the recurrent path is parity-
// validated. The chunked version exposes per-chunk parallelism via
// (Schur-expanded) triangular inverse + batched GEMMs, which on
// 2k+ token prefills is the difference between launch-bound and
// SM-bound.
void launch_chunk_gated_delta_prefill(
    const float* q_norm, const float* k_norm, const float* v,
    const float* g_log, const float* beta,
    float* state, float* out,
    int T, int V_h, int K_d, int V_d,
    int chunk_size,
    cudaStream_t stream)
{
    (void)chunk_size;  // unused in the sequential implementation
    if (T <= 0 || V_h <= 0 || K_d <= 0 || V_d <= 0) return;
    const long long stride_qk = (long long)V_h * K_d;
    const long long stride_v  = (long long)V_h * V_d;
    const long long stride_h  = (long long)V_h;
    for (int t = 0; t < T; ++t) {
        launch_recurrent_gated_delta_step(
            q_norm + t * stride_qk,
            k_norm + t * stride_qk,
            v      + t * stride_v,
            g_log  + t * stride_h,
            beta   + t * stride_h,
            state,
            out    + t * stride_v,
            /*B=*/1, V_h, K_d, V_d, stream);
    }
}

void launch_chunk_gated_delta_prefill_state_bf16(
    const float* q_norm, const float* k_norm, const float* v,
    const float* g_log, const float* beta,
    void* state, float* out,
    int T, int V_h, int K_d, int V_d,
    int chunk_size,
    cudaStream_t stream)
{
    (void)chunk_size;
    if (T <= 0 || V_h <= 0 || K_d <= 0 || V_d <= 0) return;
    const long long stride_qk = (long long)V_h * K_d;
    const long long stride_v  = (long long)V_h * V_d;
    const long long stride_h  = (long long)V_h;
    auto* state_bf16 = static_cast<__nv_bfloat16*>(state);
    for (int t = 0; t < T; ++t) {
        launch_recurrent_gated_delta_step_state_bf16(
            q_norm + t * stride_qk,
            k_norm + t * stride_qk,
            v      + t * stride_v,
            g_log  + t * stride_h,
            beta   + t * stride_h,
            state_bf16,
            out    + t * stride_v,
            /*B=*/1, V_h, K_d, V_d, stream);
    }
}

void launch_chunk_gated_delta_prefill_batched(
    const float* q_norm, const float* k_norm, const float* v,
    const float* g_log, const float* beta,
    float* state_base,
    const std::int32_t* slot_ids,
    const std::uint32_t* qo_indptr,
    long long slot_stride_elems,
    float* out,
    int R, int K_h, int V_h, int K_d, int V_d,
    cudaStream_t stream, bool write_state, const int* commit_len)
{
    if (R <= 0 || V_h <= 0 || K_d <= 0 || V_d <= 0) return;
    // FLA-style chunked prefill: keeps state in registers across the
    // T-token loop, only one HBM round-trip per (request, head).
    // 9x faster than the legacy per-token-IO kernel at production
    // shapes (microbench: 47.5 ms -> 5.3 ms). Bit-identical output.
    // The fla kernel is GQA-aware (reads compact K_h-head q/k); the legacy
    // fallback below is not, so it requires the expanded layout (K_h==V_h).
    constexpr int BK_MAX_FLA = 128;
    constexpr int BV_FLA     = 128;
    if (qwen_gdn_fla_prefill_enabled() &&
        !qwen_gdn_k_last_state_enabled() &&
        K_d <= BK_MAX_FLA && V_d % BV_FLA == 0) {
        const int NV = V_d / BV_FLA;
        dim3 grid_fla(NV, R, V_h);
        dim3 block_fla(BV_FLA);
        const int shmem_bytes_fla = 2 * BK_MAX_FLA * sizeof(float);
        chunk_gated_delta_prefill_batched_fla_kernel<float, BV_FLA, BK_MAX_FLA><<<
            grid_fla, block_fla, shmem_bytes_fla, stream>>>(
            q_norm, k_norm, v, g_log, beta, state_base,
            slot_ids, qo_indptr, slot_stride_elems,
            out, K_h, V_h, K_d, V_d, write_state, commit_len);
        return;
    }
    constexpr int BLOCK = 128;
    dim3 grid(R, V_h);
    dim3 block(BLOCK);
    const int shmem_bytes = 2 * K_d * sizeof(float);
    if (qwen_gdn_k_last_state_enabled()) {
        chunk_gated_delta_prefill_batched_kernel<float, true><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta, state_base,
            slot_ids, qo_indptr, slot_stride_elems,
            out, V_h, K_d, V_d);
    } else {
        chunk_gated_delta_prefill_batched_kernel<float, false><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta, state_base,
            slot_ids, qo_indptr, slot_stride_elems,
            out, V_h, K_d, V_d);
    }
}

void launch_chunk_gated_delta_prefill_batched_state_bf16(
    const float* q_norm, const float* k_norm, const float* v,
    const float* g_log, const float* beta,
    void* state_base,
    const std::int32_t* slot_ids,
    const std::uint32_t* qo_indptr,
    long long slot_stride_elems,
    float* out,
    int R, int K_h, int V_h, int K_d, int V_d,
    cudaStream_t stream, bool write_state, const int* commit_len)
{
    if (R <= 0 || V_h <= 0 || K_d <= 0 || V_d <= 0) return;
    constexpr int BK_MAX_FLA = 128;
    constexpr int BV_FLA     = 128;
    if (qwen_gdn_fla_prefill_enabled() &&
        !qwen_gdn_k_last_state_enabled() &&
        K_d <= BK_MAX_FLA && V_d % BV_FLA == 0) {
        const int NV = V_d / BV_FLA;
        dim3 grid_fla(NV, R, V_h);
        dim3 block_fla(BV_FLA);
        const int shmem_bytes_fla = 2 * BK_MAX_FLA * sizeof(float);
        chunk_gated_delta_prefill_batched_fla_kernel<__nv_bfloat16, BV_FLA, BK_MAX_FLA><<<
            grid_fla, block_fla, shmem_bytes_fla, stream>>>(
            q_norm, k_norm, v, g_log, beta,
            static_cast<__nv_bfloat16*>(state_base),
            slot_ids, qo_indptr, slot_stride_elems,
            out, K_h, V_h, K_d, V_d, write_state, commit_len);
        return;
    }
    constexpr int BLOCK = 128;
    dim3 grid(R, V_h);
    dim3 block(BLOCK);
    const int shmem_bytes = 2 * K_d * sizeof(float);
    if (qwen_gdn_k_last_state_enabled()) {
        chunk_gated_delta_prefill_batched_kernel<__nv_bfloat16, true><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta,
            static_cast<__nv_bfloat16*>(state_base),
            slot_ids, qo_indptr, slot_stride_elems,
            out, V_h, K_d, V_d);
    } else {
        chunk_gated_delta_prefill_batched_kernel<__nv_bfloat16, false><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta,
            static_cast<__nv_bfloat16*>(state_base),
            slot_ids, qo_indptr, slot_stride_elems,
            out, V_h, K_d, V_d);
    }
}

void launch_chunk_gated_delta_prefill_batched_cached(
    const float* q_norm, const float* k_norm, const float* v,
    const float* g_log, const float* beta,
    float* state_base,
    const std::int32_t* slot_ids,
    const std::uint32_t* qo_indptr,
    long long slot_stride_elems,
    float* out,
    int R, int V_h, int K_d, int V_d,
    cudaStream_t stream, bool write_state)
{
    if (R <= 0 || V_h <= 0 || K_d <= 0 || V_d <= 0) return;
    constexpr int BLOCK = 128;
    dim3 grid(R, V_h);
    dim3 block(BLOCK);
    const int shmem_bytes = K_d * V_d * static_cast<int>(sizeof(float));
    static int configured_shmem_bytes = 0;
    const bool k_last = qwen_gdn_k_last_state_enabled();
    if (shmem_bytes > 48 * 1024 && shmem_bytes > configured_shmem_bytes) {
        if (k_last) {
            cudaFuncSetAttribute(
                chunk_gated_delta_prefill_batched_cached_kernel<float, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shmem_bytes);
        } else {
            cudaFuncSetAttribute(
                chunk_gated_delta_prefill_batched_cached_kernel<float, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shmem_bytes);
        }
        configured_shmem_bytes = shmem_bytes;
    }
    if (k_last) {
        chunk_gated_delta_prefill_batched_cached_kernel<float, true><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta, state_base,
            slot_ids, qo_indptr, slot_stride_elems,
            out, V_h, K_d, V_d, write_state);
    } else {
        chunk_gated_delta_prefill_batched_cached_kernel<float, false><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta, state_base,
            slot_ids, qo_indptr, slot_stride_elems,
            out, V_h, K_d, V_d, write_state);
    }
}

void launch_chunk_gated_delta_prefill_batched_cached_state_bf16(
    const float* q_norm, const float* k_norm, const float* v,
    const float* g_log, const float* beta,
    void* state_base,
    const std::int32_t* slot_ids,
    const std::uint32_t* qo_indptr,
    long long slot_stride_elems,
    float* out,
    int R, int V_h, int K_d, int V_d,
    cudaStream_t stream, bool write_state)
{
    if (R <= 0 || V_h <= 0 || K_d <= 0 || V_d <= 0) return;
    constexpr int BLOCK = 128;
    dim3 grid(R, V_h);
    dim3 block(BLOCK);
    const int shmem_bytes = K_d * V_d * static_cast<int>(sizeof(float));
    static int configured_shmem_bytes = 0;
    const bool k_last = qwen_gdn_k_last_state_enabled();
    if (shmem_bytes > 48 * 1024 && shmem_bytes > configured_shmem_bytes) {
        if (k_last) {
            cudaFuncSetAttribute(
                chunk_gated_delta_prefill_batched_cached_kernel<__nv_bfloat16, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shmem_bytes);
        } else {
            cudaFuncSetAttribute(
                chunk_gated_delta_prefill_batched_cached_kernel<__nv_bfloat16, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shmem_bytes);
        }
        configured_shmem_bytes = shmem_bytes;
    }
    if (k_last) {
        chunk_gated_delta_prefill_batched_cached_kernel<__nv_bfloat16, true><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta,
            static_cast<__nv_bfloat16*>(state_base),
            slot_ids, qo_indptr, slot_stride_elems,
            out, V_h, K_d, V_d, write_state);
    } else {
        chunk_gated_delta_prefill_batched_cached_kernel<__nv_bfloat16, false><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta,
            static_cast<__nv_bfloat16*>(state_base),
            slot_ids, qo_indptr, slot_stride_elems,
            out, V_h, K_d, V_d, write_state);
    }
}

void launch_chunk_gated_delta_prefill_batched_warp_tiled(
    const float* q_norm, const float* k_norm, const float* v,
    const float* g_log, const float* beta,
    float* state_base,
    const std::int32_t* slot_ids,
    const std::uint32_t* qo_indptr,
    long long slot_stride_elems,
    float* out,
    int R, int V_h, int K_d, int V_d,
    cudaStream_t stream, bool write_state)
{
    if (R <= 0 || V_h <= 0 || K_d <= 0 || V_d <= 0) return;
    if (K_d > 256) {
        launch_chunk_gated_delta_prefill_batched(
            q_norm, k_norm, v, g_log, beta, state_base,
            slot_ids, qo_indptr, slot_stride_elems, out,
            R, V_h, V_h, K_d, V_d, stream);
        return;
    }
    constexpr int WARPS = 4;
    constexpr int BLOCK = WARPS * 32;
    dim3 grid(R, V_h, (V_d + WARPS - 1) / WARPS);
    dim3 block(BLOCK);
    if (qwen_gdn_k_last_state_enabled()) {
        chunk_gated_delta_prefill_batched_warp_tiled_kernel<float, true><<<
            grid, block, 0, stream>>>(
            q_norm, k_norm, v, g_log, beta, state_base,
            slot_ids, qo_indptr, slot_stride_elems,
            out, V_h, K_d, V_d, write_state);
    } else {
        chunk_gated_delta_prefill_batched_warp_tiled_kernel<float, false><<<
            grid, block, 0, stream>>>(
            q_norm, k_norm, v, g_log, beta, state_base,
            slot_ids, qo_indptr, slot_stride_elems,
            out, V_h, K_d, V_d, write_state);
    }
}

void launch_chunk_gated_delta_prefill_batched_warp_tiled_state_bf16(
    const float* q_norm, const float* k_norm, const float* v,
    const float* g_log, const float* beta,
    void* state_base,
    const std::int32_t* slot_ids,
    const std::uint32_t* qo_indptr,
    long long slot_stride_elems,
    float* out,
    int R, int V_h, int K_d, int V_d,
    cudaStream_t stream, bool write_state)
{
    if (R <= 0 || V_h <= 0 || K_d <= 0 || V_d <= 0) return;
    if (K_d > 256) {
        launch_chunk_gated_delta_prefill_batched_state_bf16(
            q_norm, k_norm, v, g_log, beta, state_base,
            slot_ids, qo_indptr, slot_stride_elems, out,
            R, V_h, V_h, K_d, V_d, stream);
        return;
    }
    constexpr int WARPS = 4;
    constexpr int BLOCK = WARPS * 32;
    dim3 grid(R, V_h, (V_d + WARPS - 1) / WARPS);
    dim3 block(BLOCK);
    if (qwen_gdn_k_last_state_enabled()) {
        chunk_gated_delta_prefill_batched_warp_tiled_kernel<__nv_bfloat16, true><<<
            grid, block, 0, stream>>>(
            q_norm, k_norm, v, g_log, beta,
            static_cast<__nv_bfloat16*>(state_base),
            slot_ids, qo_indptr, slot_stride_elems,
            out, V_h, K_d, V_d, write_state);
    } else {
        chunk_gated_delta_prefill_batched_warp_tiled_kernel<__nv_bfloat16, false><<<
            grid, block, 0, stream>>>(
            q_norm, k_norm, v, g_log, beta,
            static_cast<__nv_bfloat16*>(state_base),
            slot_ids, qo_indptr, slot_stride_elems,
            out, V_h, K_d, V_d, write_state);
    }
}

void launch_chunk_gated_delta_prefill_batched_warp_tiled_gqa(
    const float* q_norm_kh, const float* k_norm_kh, const float* v,
    const float* g_log, const float* beta,
    float* state_base,
    const std::int32_t* slot_ids,
    const std::uint32_t* qo_indptr,
    long long slot_stride_elems,
    float* out,
    int R, int K_h, int V_h, int K_d, int V_d,
    cudaStream_t stream, bool write_state)
{
    if (R <= 0 || K_h <= 0 || V_h <= 0 || K_d <= 0 || V_d <= 0) return;
    if (K_d > 256 || V_h % K_h != 0) {
        throw std::runtime_error(
            "launch_chunk_gated_delta_prefill_batched_warp_tiled_gqa: "
            "unsupported GQA dimensions");
    }
    constexpr int WARPS = 4;
    constexpr int BLOCK = WARPS * 32;
    const bool k_last = qwen_gdn_k_last_state_enabled();
    if (qwen_gdn_gqa_ilp2_enabled()) {
        constexpr int TILE_V = WARPS * 2;
        dim3 grid(R, V_h, (V_d + TILE_V - 1) / TILE_V);
        dim3 block(BLOCK);
        if (k_last) {
            chunk_gated_delta_prefill_batched_warp_tiled_gqa_ilp2_kernel<float, true><<<
                grid, block, 0, stream>>>(
                q_norm_kh, k_norm_kh, v, g_log, beta, state_base,
                slot_ids, qo_indptr, slot_stride_elems,
                out, K_h, V_h, K_d, V_d, write_state);
        } else {
            chunk_gated_delta_prefill_batched_warp_tiled_gqa_ilp2_kernel<float, false><<<
                grid, block, 0, stream>>>(
                q_norm_kh, k_norm_kh, v, g_log, beta, state_base,
                slot_ids, qo_indptr, slot_stride_elems,
                out, K_h, V_h, K_d, V_d, write_state);
        }
        return;
    }
    dim3 grid(R, V_h, (V_d + WARPS - 1) / WARPS);
    dim3 block(BLOCK);
    if (k_last) {
        chunk_gated_delta_prefill_batched_warp_tiled_gqa_kernel<float, true><<<
            grid, block, 0, stream>>>(
            q_norm_kh, k_norm_kh, v, g_log, beta, state_base,
            slot_ids, qo_indptr, slot_stride_elems,
            out, K_h, V_h, K_d, V_d, write_state);
    } else {
        chunk_gated_delta_prefill_batched_warp_tiled_gqa_kernel<float, false><<<
            grid, block, 0, stream>>>(
            q_norm_kh, k_norm_kh, v, g_log, beta, state_base,
            slot_ids, qo_indptr, slot_stride_elems,
            out, K_h, V_h, K_d, V_d, write_state);
    }
}

void launch_chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16(
    const float* q_norm_kh, const float* k_norm_kh, const float* v,
    const float* g_log, const float* beta,
    void* state_base,
    const std::int32_t* slot_ids,
    const std::uint32_t* qo_indptr,
    long long slot_stride_elems,
    float* out,
    int R, int K_h, int V_h, int K_d, int V_d,
    cudaStream_t stream, bool write_state)
{
    if (R <= 0 || K_h <= 0 || V_h <= 0 || K_d <= 0 || V_d <= 0) return;
    if (K_d > 256 || V_h % K_h != 0) {
        throw std::runtime_error(
            "launch_chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16: "
            "unsupported GQA dimensions");
    }
    constexpr int WARPS = 4;
    constexpr int BLOCK = WARPS * 32;
    const bool k_last = qwen_gdn_k_last_state_enabled();
    if (qwen_gdn_gqa_ilp2_enabled()) {
        constexpr int TILE_V = WARPS * 2;
        dim3 grid(R, V_h, (V_d + TILE_V - 1) / TILE_V);
        dim3 block(BLOCK);
        if (k_last) {
            chunk_gated_delta_prefill_batched_warp_tiled_gqa_ilp2_kernel<__nv_bfloat16, true><<<
                grid, block, 0, stream>>>(
                q_norm_kh, k_norm_kh, v, g_log, beta,
                static_cast<__nv_bfloat16*>(state_base),
                slot_ids, qo_indptr, slot_stride_elems,
                out, K_h, V_h, K_d, V_d, write_state);
        } else {
            chunk_gated_delta_prefill_batched_warp_tiled_gqa_ilp2_kernel<__nv_bfloat16, false><<<
                grid, block, 0, stream>>>(
                q_norm_kh, k_norm_kh, v, g_log, beta,
                static_cast<__nv_bfloat16*>(state_base),
                slot_ids, qo_indptr, slot_stride_elems,
                out, K_h, V_h, K_d, V_d, write_state);
        }
        return;
    }
    dim3 grid(R, V_h, (V_d + WARPS - 1) / WARPS);
    dim3 block(BLOCK);
    if (k_last) {
        chunk_gated_delta_prefill_batched_warp_tiled_gqa_kernel<__nv_bfloat16, true><<<
            grid, block, 0, stream>>>(
            q_norm_kh, k_norm_kh, v, g_log, beta,
            static_cast<__nv_bfloat16*>(state_base),
            slot_ids, qo_indptr, slot_stride_elems,
            out, K_h, V_h, K_d, V_d, write_state);
    } else {
        chunk_gated_delta_prefill_batched_warp_tiled_gqa_kernel<__nv_bfloat16, false><<<
            grid, block, 0, stream>>>(
            q_norm_kh, k_norm_kh, v, g_log, beta,
            static_cast<__nv_bfloat16*>(state_base),
            slot_ids, qo_indptr, slot_stride_elems,
            out, K_h, V_h, K_d, V_d, write_state);
    }
}

}  // namespace pie_cuda_driver::kernels
