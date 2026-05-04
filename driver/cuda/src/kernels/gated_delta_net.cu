#include "kernels/gated_delta_net.hpp"

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

// ── Helpers ────────────────────────────────────────────────────────

namespace {

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

// ── Recurrent step kernel ──────────────────────────────────────────

namespace {

// One block per (request, head). Threads parallelize over v_idx in
// [0, V_d). Each thread loops over k_idx in [0, K_d) twice (once for
// the kv_mem accumulation, once for the post-update output).
//
// Shared memory layout: q[K_d] + k[K_d] fp32. Caller passes shmem of
// size 2*K_d*sizeof(float).
__global__ void recurrent_step_kernel(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    float*       __restrict__ state,
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
            const long long off = (long long)k_idx * V_d + v_idx;
            const float s = state[off] * g_h;
            state[off] = s;
            kv_mem += s * sk[k_idx];
        }

        const float v_t   = v_h[v_idx];
        const float delta = (v_t - kv_mem) * beta_h;

        // Phase 2: state[k, v] += k[k] * delta; out[v] = Σ_k state[k,v]*q[k].
        float out_v = 0.f;
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off = (long long)k_idx * V_d + v_idx;
            const float s = state[off] + sk[k_idx] * delta;
            state[off] = s;
            out_v += s * sq[k_idx];
        }
        out[v_idx] = out_v;
    }
}

// Multi-request batched chunked prefill. One block per (request, head);
// the block walks its T_r tokens sequentially (per-token state
// dependency), accumulating the recurrence into the request's state
// slab. Same per-token math as `recurrent_step_kernel`.
__global__ void chunk_gated_delta_prefill_batched_kernel(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    float*       __restrict__ state_base,
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
    float* state = state_base
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
                const long long off = (long long)k_idx * V_d + v_idx;
                const float s = state[off] * g_h;
                state[off] = s;
                kv_mem += s * sk[k_idx];
            }

            const float v_t   = v_h[v_idx];
            const float delta = (v_t - kv_mem) * beta_h;

            float out_v = 0.f;
            for (int k_idx = 0; k_idx < K_d; ++k_idx) {
                const long long off = (long long)k_idx * V_d + v_idx;
                const float s = state[off] + sk[k_idx] * delta;
                state[off] = s;
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

// Batched variant with slot indirection. State for request r lives at
// `state_base + slot_ids[r] * slot_stride_elems`. Otherwise the
// per-(request, head) compute is identical to `recurrent_step_kernel`.
__global__ void recurrent_step_batched_kernel(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    float*       __restrict__ state_base,
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

    float* state = state_base
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
            const long long off = (long long)k_idx * V_d + v_idx;
            const float s = state[off] * g_h;
            state[off] = s;
            kv_mem += s * sk[k_idx];
        }

        const float v_t   = v_h[v_idx];
        const float delta = (v_t - kv_mem) * beta_h;

        float out_v = 0.f;
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off = (long long)k_idx * V_d + v_idx;
            const float s = state[off] + sk[k_idx] * delta;
            state[off] = s;
            out_v += s * sq[k_idx];
        }
        out_bh[v_idx] = out_v;
    }
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
    recurrent_step_kernel<<<grid, block, shmem_bytes, stream>>>(
        q_norm, k_norm, v, g_log, beta, state, out, V_h, K_d, V_d);
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
    const int shmem_bytes = 2 * K_d * sizeof(float);
    recurrent_step_batched_kernel<<<grid, block, shmem_bytes, stream>>>(
        q_norm, k_norm, v, g_log, beta,
        state_base, slot_ids, slot_stride_elems,
        out, V_h, K_d, V_d);
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

void launch_chunk_gated_delta_prefill_batched(
    const float* q_norm, const float* k_norm, const float* v,
    const float* g_log, const float* beta,
    float* state_base,
    const std::int32_t* slot_ids,
    const std::uint32_t* qo_indptr,
    long long slot_stride_elems,
    float* out,
    int R, int V_h, int K_d, int V_d,
    cudaStream_t stream)
{
    if (R <= 0 || V_h <= 0 || K_d <= 0 || V_d <= 0) return;
    constexpr int BLOCK = 128;
    dim3 grid(R, V_h);
    dim3 block(BLOCK);
    const int shmem_bytes = 2 * K_d * sizeof(float);
    chunk_gated_delta_prefill_batched_kernel<<<grid, block, shmem_bytes, stream>>>(
        q_norm, k_norm, v, g_log, beta,
        state_base, slot_ids, qo_indptr, slot_stride_elems,
        out, V_h, K_d, V_d);
}

}  // namespace pie_cuda_driver::kernels
