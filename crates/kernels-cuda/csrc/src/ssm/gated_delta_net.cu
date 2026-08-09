// Host launchers for Gated Delta Net. Every `__global__` they fire lives in
// one of two headers and is defined ONCE, there: `ssm/gated_delta_net_prep.cuh`
// holds the seven pre-recurrence kernels and `ssm/gated_delta_net.cuh` the
// fourteen recurrence ones. This file includes both rather than carrying a
// second copy — `new-horizon.md` §10.10 records what the copy costs, and
// `kernels-cuda/tests/sources.rs` now refuses a second definition of a
// namespace-qualified `__global__` because `norm/altup_aux` paid it.
//
// # Why two headers and not one
//
// The two halves are not the same kind of code. The prep half is pointwise
// and on the prelude, so it compiles under NVRTC and carries five rows. The
// recurrence half still reaches for `__nv_bfloat162` and
// `__floats2bfloat162_rn` — §10.5 records that conversion as REVERTED — and
// carries none: its grids are `(requests, heads)`, its templates take two and
// three parameters where a row can spell one type, and its slot stride is a
// `long long` the argument binder refuses. NVRTC compiles a unit whole, so a
// single header would let one unresolved packed-half intrinsic take the five
// good rows down with it.
//
// # What stays here
//
// Every launcher, unchanged. The five rows do not replace one: they are a
// second way to reach the same `__global__`, and the ahead-of-time path is
// still the only path for the sixteen kernels no `LaunchRule` can state.
//
// The dispatch POLICY stays here too — the `constexpr bool ..._enabled()`
// toggles, the env-var read, and `gdn_raise_shmem_cap`. They are host
// decisions about which kernel to fire and how much shared memory to ask a
// device for, not device text; `cudaFuncSetAttribute` and `std::getenv` do not
// exist on the other side of a `<<<>>>`.
#include "ssm/gated_delta_net.hpp"
#include "ssm/gated_delta_net.cuh"
#include "ssm/gated_delta_net_prep.cuh"

#include <cuda_bf16.h>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace pie_cuda_driver::kernels::ssm {

namespace {

constexpr bool qwen_gdn_gqa_ilp2_enabled() { return false; }

constexpr bool qwen_gdn_k_last_state_enabled() { return false; }

// Use the fused recurrent step kernel that caches state values in
// registers across the two analytical phases, halving HBM traffic on
// the state slab (2R+2W -> 1R+1W per element). Default OFF until
// parity is verified across all (K_d, V_d) combinations the kernel is
// instantiated for; turn ON for benchmarking the new path.
constexpr bool qwen_gdn_fused_step_enabled() { return false; }

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
// `cudaFuncSetAttribute` configures a kernel's dynamic shared-memory cap
// PER DEVICE. A process-global "already configured" flag therefore lies to
// every device but the first: under tensor parallelism rank 0 raises the
// cap on device 0, sets the flag, and rank 1 skips the call — then launches
// the same kernel on device 1 asking for more shared memory than that
// device allows. Track the high-water mark per device instead.
bool gdn_raise_shmem_cap(const void* func, int shmem_bytes) {
    if (shmem_bytes <= 48 * 1024) return true;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    static std::mutex mutex;
    static std::map<std::pair<int, const void*>, int> configured;
    std::lock_guard<std::mutex> guard(mutex);
    int& high_water = configured[{device, func}];
    if (shmem_bytes <= high_water) return true;
    const cudaError_t status = cudaFuncSetAttribute(
        func, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes);
    if (status != cudaSuccess) {
        static_cast<void>(cudaGetLastError());
        return false;
    }
    high_water = shmem_bytes;
    return true;
}

bool qwen_gdn_smem_step_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_GDN_SMEM_STEP");
        if (v == nullptr || v[0] == '\0') return true;
        return v[0] != '0';
    }();
    return enabled;
}
// 9x speedup over the legacy per-token HBM kernel, bit-identical at
// production shapes (V_d=128, K_d<=128).
constexpr bool qwen_gdn_fla_prefill_enabled() { return true; }

constexpr bool qwen_gdn_fla_step_enabled() { return false; }

}  // namespace

void bf16_to_fp32(
    const void* x, float* y, std::size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    constexpr int BLOCK = 256;
    const std::size_t grid = (n + BLOCK - 1) / BLOCK;
    device::widen<device::bf16><<<(int)grid, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(x), y, n);
}

void fp32_to_bf16(
    const float* x, void* y, std::size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    constexpr int BLOCK = 256;
    const std::size_t grid = (n + BLOCK - 1) / BLOCK;
    device::narrow<device::bf16><<<(int)grid, BLOCK, 0, stream>>>(
        x, static_cast<device::bf16*>(y), n);
}

void repeat_interleave_heads_fp32(
    const float* in, float* out,
    int N, int K_h, int V_h, int D,
    cudaStream_t stream)
{
    if (N <= 0 || K_h <= 0 || V_h <= 0 || D <= 0) return;
    const int repeat = V_h / K_h;
    const int block = (D < 128) ? 64 : 128;
    dim3 grid(N, V_h);
    device::repeat_interleave_heads_fp32<device::f32><<<grid, block, 0, stream>>>(
        in, out, K_h, V_h, D, repeat);
}

void l2norm_scale_bf16_to_fp32(
    const void* x, float* y,
    int N, int hidden,
    float scale, float eps,
    cudaStream_t stream)
{
    if (N <= 0 || hidden <= 0) return;
    constexpr int BLOCK = 128;
    dim3 grid(N);
    dim3 block(BLOCK);
    device::l2norm_scale<device::bf16, BLOCK><<<grid, block, 0, stream>>>(
        static_cast<const device::bf16*>(x), y, hidden, scale, eps);
}

void gated_delta_g_beta(
    const void* a, const void* b,
    const void* A_log, const void* dt_bias,
    float* g_log_out, float* beta_out,
    int N, int V_h, cudaStream_t stream)
{
    if (N <= 0 || V_h <= 0) return;
    constexpr int BLOCK = 64;
    dim3 grid(N, (V_h + BLOCK - 1) / BLOCK);
    dim3 block(BLOCK);
    device::g_beta<device::bf16><<<grid, block, 0, stream>>>(
        static_cast<const device::bf16*>(a),
        static_cast<const device::bf16*>(b),
        static_cast<const float*>(A_log),
        static_cast<const device::bf16*>(dt_bias),
        g_log_out, beta_out, N, V_h);
}

void qwen_gdn_post_conv_prep_bf16(
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
    device::qwen_gdn_qk_norm<device::bf16, BLOCK><<<qk_grid, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(qkv_post),
        q_norm_kh, k_norm_kh, K_h, K_d, conv_dim, q_scale);
    dim3 vg_grid(N, V_h);
    device::qwen_gdn_v_g_beta<device::bf16, BLOCK><<<vg_grid, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(qkv_post),
        static_cast<const device::bf16*>(a),
        static_cast<const device::bf16*>(b),
        static_cast<const float*>(A_log),
        static_cast<const device::bf16*>(dt_bias),
        v_fp32, g_log_out, beta_out,
        K_h, V_h, K_d, V_d, conv_dim);
}

// ── Recurrent step kernel ──────────────────────────────────────────

void recurrent_gated_delta_step(
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
        device::recurrent_step<float, true><<<grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta, state, out, V_h, K_d, V_d);
    } else {
        device::recurrent_step<float, false><<<grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta, state, out, V_h, K_d, V_d);
    }
}

void recurrent_gated_delta_step_state_bf16(
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
        device::recurrent_step<__nv_bfloat16, true><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta,
            static_cast<__nv_bfloat16*>(state), out, V_h, K_d, V_d);
    } else {
        device::recurrent_step<__nv_bfloat16, false><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta,
            static_cast<__nv_bfloat16*>(state), out, V_h, K_d, V_d);
    }
}

void recurrent_gated_delta_step_batched(
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
            device::recurrent_step_batched_fused<float, true, 256><<<
                grid, block, shmem_bytes, stream>>>(
                q_norm, k_norm, v, g_log, beta, state_base,
                slot_ids, slot_stride_elems, out, V_h, K_d, V_d);
        } else {
            device::recurrent_step_batched_fused<float, false, 256><<<
                grid, block, shmem_bytes, stream>>>(
                q_norm, k_norm, v, g_log, beta, state_base,
                slot_ids, slot_stride_elems, out, V_h, K_d, V_d);
        }
        return;
    }
    if (qwen_gdn_k_last_state_enabled()) {
        device::recurrent_step_batched<float, true><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta, state_base,
            slot_ids, slot_stride_elems, out, V_h, K_d, V_d);
    } else {
        device::recurrent_step_batched<float, false><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta, state_base,
            slot_ids, slot_stride_elems, out, V_h, K_d, V_d);
    }
}

void recurrent_gated_delta_step_batched_state_bf16(
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
            device::recurrent_step_batched_fused<__nv_bfloat16, true, 256><<<
                grid, block, shmem_bytes, stream>>>(
                q_norm, k_norm, v, g_log, beta,
                static_cast<__nv_bfloat16*>(state_base),
                slot_ids, slot_stride_elems, out, V_h, K_d, V_d);
        } else {
            device::recurrent_step_batched_fused<__nv_bfloat16, false, 256><<<
                grid, block, shmem_bytes, stream>>>(
                q_norm, k_norm, v, g_log, beta,
                static_cast<__nv_bfloat16*>(state_base),
                slot_ids, slot_stride_elems, out, V_h, K_d, V_d);
        }
        return;
    }
    if (qwen_gdn_k_last_state_enabled()) {
        device::recurrent_step_batched<__nv_bfloat16, true><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta,
            static_cast<__nv_bfloat16*>(state_base),
            slot_ids, slot_stride_elems, out, V_h, K_d, V_d);
    } else {
        device::recurrent_step_batched<__nv_bfloat16, false><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta,
            static_cast<__nv_bfloat16*>(state_base),
            slot_ids, slot_stride_elems, out, V_h, K_d, V_d);
    }
}

void recurrent_gated_delta_step_batched_gqa(
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
        device::recurrent_step_batched_gqa_fla<float, BV_FLA, BK_MAX_FLA><<<
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
            device::recurrent_step_batched_gqa_fused<float, true, 256><<<
                grid, block, shmem_bytes, stream>>>(
                q_norm_kh, k_norm_kh, v, g_log, beta, state_base,
                slot_ids, slot_stride_elems, out, K_h, V_h, K_d, V_d);
        } else {
            device::recurrent_step_batched_gqa_fused<float, false, 256><<<
                grid, block, shmem_bytes, stream>>>(
                q_norm_kh, k_norm_kh, v, g_log, beta, state_base,
                slot_ids, slot_stride_elems, out, K_h, V_h, K_d, V_d);
        }
        return;
    }
    if (qwen_gdn_k_last_state_enabled()) {
        device::recurrent_step_batched_gqa<float, true><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm_kh, k_norm_kh, v, g_log, beta, state_base,
            slot_ids, slot_stride_elems, out, K_h, V_h, K_d, V_d);
    } else {
        device::recurrent_step_batched_gqa<float, false><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm_kh, k_norm_kh, v, g_log, beta, state_base,
            slot_ids, slot_stride_elems, out, K_h, V_h, K_d, V_d);
    }
}

void recurrent_gated_delta_step_batched_gqa_state_bf16(
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
        device::recurrent_step_batched_gqa_smem<BV><<<
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
            device::recurrent_step_batched_gqa_fused<__nv_bfloat16, true, 256><<<
                grid, block, shmem_bytes, stream>>>(
                q_norm_kh, k_norm_kh, v, g_log, beta,
                static_cast<__nv_bfloat16*>(state_base),
                slot_ids, slot_stride_elems, out, K_h, V_h, K_d, V_d);
        } else {
            device::recurrent_step_batched_gqa_fused<__nv_bfloat16, false, 256><<<
                grid, block, shmem_bytes, stream>>>(
                q_norm_kh, k_norm_kh, v, g_log, beta,
                static_cast<__nv_bfloat16*>(state_base),
                slot_ids, slot_stride_elems, out, K_h, V_h, K_d, V_d);
        }
        return;
    }
    if (qwen_gdn_k_last_state_enabled()) {
        device::recurrent_step_batched_gqa<__nv_bfloat16, true><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm_kh, k_norm_kh, v, g_log, beta,
            static_cast<__nv_bfloat16*>(state_base),
            slot_ids, slot_stride_elems, out, K_h, V_h, K_d, V_d);
    } else {
        device::recurrent_step_batched_gqa<__nv_bfloat16, false><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm_kh, k_norm_kh, v, g_log, beta,
            static_cast<__nv_bfloat16*>(state_base),
            slot_ids, slot_stride_elems, out, K_h, V_h, K_d, V_d);
    }
}

// Chunked prefill — for now, implemented as a sequential per-token
// loop over `recurrent_gated_delta_step`. Mathematically
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
void chunk_gated_delta_prefill(
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
        recurrent_gated_delta_step(
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

void chunk_gated_delta_prefill_state_bf16(
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
        recurrent_gated_delta_step_state_bf16(
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

void chunk_gated_delta_prefill_batched(
    const float* q_norm, const float* k_norm, const float* v,
    const float* g_log, const float* beta,
    float* state_base,
    const std::int32_t* slot_ids,
    const std::uint32_t* qo_indptr,
    long long slot_stride_elems,
    float* out,
    int R, int K_h, int V_h, int K_d, int V_d,
    cudaStream_t stream, bool write_state, const int* commit_len,
    const std::uint8_t* write_state_mask)
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
        device::chunk_gated_delta_prefill_batched_fla<float, BV_FLA, BK_MAX_FLA><<<
            grid_fla, block_fla, shmem_bytes_fla, stream>>>(
            q_norm, k_norm, v, g_log, beta, state_base,
            slot_ids, qo_indptr, slot_stride_elems,
            out, K_h, V_h, K_d, V_d, write_state, commit_len,
            write_state_mask);
        return;
    }
    constexpr int BLOCK = 128;
    dim3 grid(R, V_h);
    dim3 block(BLOCK);
    const int shmem_bytes = 2 * K_d * sizeof(float);
    if (qwen_gdn_k_last_state_enabled()) {
        device::chunk_gated_delta_prefill_batched<float, true><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta, state_base,
            slot_ids, qo_indptr, slot_stride_elems,
            out, V_h, K_d, V_d);
    } else {
        device::chunk_gated_delta_prefill_batched<float, false><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta, state_base,
            slot_ids, qo_indptr, slot_stride_elems,
            out, V_h, K_d, V_d);
    }
}

void chunk_gated_delta_prefill_batched_state_bf16(
    const float* q_norm, const float* k_norm, const float* v,
    const float* g_log, const float* beta,
    void* state_base,
    const std::int32_t* slot_ids,
    const std::uint32_t* qo_indptr,
    long long slot_stride_elems,
    float* out,
    int R, int K_h, int V_h, int K_d, int V_d,
    cudaStream_t stream, bool write_state, const int* commit_len,
    const std::uint8_t* write_state_mask)
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
        device::chunk_gated_delta_prefill_batched_fla<__nv_bfloat16, BV_FLA, BK_MAX_FLA><<<
            grid_fla, block_fla, shmem_bytes_fla, stream>>>(
            q_norm, k_norm, v, g_log, beta,
            static_cast<__nv_bfloat16*>(state_base),
            slot_ids, qo_indptr, slot_stride_elems,
            out, K_h, V_h, K_d, V_d, write_state, commit_len,
            write_state_mask);
        return;
    }
    constexpr int BLOCK = 128;
    dim3 grid(R, V_h);
    dim3 block(BLOCK);
    const int shmem_bytes = 2 * K_d * sizeof(float);
    if (qwen_gdn_k_last_state_enabled()) {
        device::chunk_gated_delta_prefill_batched<__nv_bfloat16, true><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta,
            static_cast<__nv_bfloat16*>(state_base),
            slot_ids, qo_indptr, slot_stride_elems,
            out, V_h, K_d, V_d);
    } else {
        device::chunk_gated_delta_prefill_batched<__nv_bfloat16, false><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta,
            static_cast<__nv_bfloat16*>(state_base),
            slot_ids, qo_indptr, slot_stride_elems,
            out, V_h, K_d, V_d);
    }
}

void chunk_gated_delta_prefill_batched_cached(
    const float* q_norm, const float* k_norm, const float* v,
    const float* g_log, const float* beta,
    float* state_base,
    const std::int32_t* slot_ids,
    const std::uint32_t* qo_indptr,
    long long slot_stride_elems,
    float* out,
    int R, int V_h, int K_d, int V_d,
    cudaStream_t stream, bool write_state,
    const std::uint8_t* write_state_mask)
{
    if (R <= 0 || V_h <= 0 || K_d <= 0 || V_d <= 0) return;
    constexpr int BLOCK = 128;
    dim3 grid(R, V_h);
    dim3 block(BLOCK);
    const int shmem_bytes = K_d * V_d * static_cast<int>(sizeof(float));
    const bool k_last = qwen_gdn_k_last_state_enabled();
    gdn_raise_shmem_cap(
        k_last ? reinterpret_cast<const void*>(
                     device::chunk_gated_delta_prefill_batched_cached<float, true>)
               : reinterpret_cast<const void*>(
                     device::chunk_gated_delta_prefill_batched_cached<float, false>),
        shmem_bytes);
    if (k_last) {
        device::chunk_gated_delta_prefill_batched_cached<float, true><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta, state_base,
            slot_ids, qo_indptr, slot_stride_elems,
            out, V_h, K_d, V_d, write_state, write_state_mask);
    } else {
        device::chunk_gated_delta_prefill_batched_cached<float, false><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta, state_base,
            slot_ids, qo_indptr, slot_stride_elems,
            out, V_h, K_d, V_d, write_state, write_state_mask);
    }
}

void chunk_gated_delta_prefill_batched_cached_state_bf16(
    const float* q_norm, const float* k_norm, const float* v,
    const float* g_log, const float* beta,
    void* state_base,
    const std::int32_t* slot_ids,
    const std::uint32_t* qo_indptr,
    long long slot_stride_elems,
    float* out,
    int R, int V_h, int K_d, int V_d,
    cudaStream_t stream, bool write_state,
    const std::uint8_t* write_state_mask)
{
    if (R <= 0 || V_h <= 0 || K_d <= 0 || V_d <= 0) return;
    constexpr int BLOCK = 128;
    dim3 grid(R, V_h);
    dim3 block(BLOCK);
    const int shmem_bytes = K_d * V_d * static_cast<int>(sizeof(float));
    const bool k_last = qwen_gdn_k_last_state_enabled();
    gdn_raise_shmem_cap(
        k_last ? reinterpret_cast<const void*>(
                     device::chunk_gated_delta_prefill_batched_cached<__nv_bfloat16, true>)
               : reinterpret_cast<const void*>(
                     device::chunk_gated_delta_prefill_batched_cached<__nv_bfloat16, false>),
        shmem_bytes);
    if (k_last) {
        device::chunk_gated_delta_prefill_batched_cached<__nv_bfloat16, true><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta,
            static_cast<__nv_bfloat16*>(state_base),
            slot_ids, qo_indptr, slot_stride_elems,
            out, V_h, K_d, V_d, write_state, write_state_mask);
    } else {
        device::chunk_gated_delta_prefill_batched_cached<__nv_bfloat16, false><<<
            grid, block, shmem_bytes, stream>>>(
            q_norm, k_norm, v, g_log, beta,
            static_cast<__nv_bfloat16*>(state_base),
            slot_ids, qo_indptr, slot_stride_elems,
            out, V_h, K_d, V_d, write_state, write_state_mask);
    }
}



void chunk_gated_delta_prefill_batched_warp_tiled_gqa(
    const float* q_norm_kh, const float* k_norm_kh, const float* v,
    const float* g_log, const float* beta,
    float* state_base,
    const std::int32_t* slot_ids,
    const std::uint32_t* qo_indptr,
    long long slot_stride_elems,
    float* out,
    int R, int K_h, int V_h, int K_d, int V_d,
    cudaStream_t stream, bool write_state,
    const std::uint8_t* write_state_mask)
{
    if (R <= 0 || K_h <= 0 || V_h <= 0 || K_d <= 0 || V_d <= 0) return;
    if (K_d > 256 || V_h % K_h != 0) {
        throw std::runtime_error(
            "chunk_gated_delta_prefill_batched_warp_tiled_gqa: "
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
            device::chunk_gated_delta_prefill_batched_warp_tiled_gqa_ilp2<float, true><<<
                grid, block, 0, stream>>>(
                q_norm_kh, k_norm_kh, v, g_log, beta, state_base,
                slot_ids, qo_indptr, slot_stride_elems,
                out, K_h, V_h, K_d, V_d, write_state, write_state_mask);
        } else {
            device::chunk_gated_delta_prefill_batched_warp_tiled_gqa_ilp2<float, false><<<
                grid, block, 0, stream>>>(
                q_norm_kh, k_norm_kh, v, g_log, beta, state_base,
                slot_ids, qo_indptr, slot_stride_elems,
                out, K_h, V_h, K_d, V_d, write_state, write_state_mask);
        }
        return;
    }
    dim3 grid(R, V_h, (V_d + WARPS - 1) / WARPS);
    dim3 block(BLOCK);
    if (k_last) {
        device::chunk_gated_delta_prefill_batched_warp_tiled_gqa<float, true><<<
            grid, block, 0, stream>>>(
            q_norm_kh, k_norm_kh, v, g_log, beta, state_base,
            slot_ids, qo_indptr, slot_stride_elems,
            out, K_h, V_h, K_d, V_d, write_state, write_state_mask);
    } else {
        device::chunk_gated_delta_prefill_batched_warp_tiled_gqa<float, false><<<
            grid, block, 0, stream>>>(
            q_norm_kh, k_norm_kh, v, g_log, beta, state_base,
            slot_ids, qo_indptr, slot_stride_elems,
            out, K_h, V_h, K_d, V_d, write_state, write_state_mask);
    }
}

void chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16(
    const float* q_norm_kh, const float* k_norm_kh, const float* v,
    const float* g_log, const float* beta,
    void* state_base,
    const std::int32_t* slot_ids,
    const std::uint32_t* qo_indptr,
    long long slot_stride_elems,
    float* out,
    int R, int K_h, int V_h, int K_d, int V_d,
    cudaStream_t stream, bool write_state,
    const std::uint8_t* write_state_mask)
{
    if (R <= 0 || K_h <= 0 || V_h <= 0 || K_d <= 0 || V_d <= 0) return;
    if (K_d > 256 || V_h % K_h != 0) {
        throw std::runtime_error(
            "chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16: "
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
            device::chunk_gated_delta_prefill_batched_warp_tiled_gqa_ilp2<__nv_bfloat16, true><<<
                grid, block, 0, stream>>>(
                q_norm_kh, k_norm_kh, v, g_log, beta,
                static_cast<__nv_bfloat16*>(state_base),
                slot_ids, qo_indptr, slot_stride_elems,
                out, K_h, V_h, K_d, V_d, write_state, write_state_mask);
        } else {
            device::chunk_gated_delta_prefill_batched_warp_tiled_gqa_ilp2<__nv_bfloat16, false><<<
                grid, block, 0, stream>>>(
                q_norm_kh, k_norm_kh, v, g_log, beta,
                static_cast<__nv_bfloat16*>(state_base),
                slot_ids, qo_indptr, slot_stride_elems,
                out, K_h, V_h, K_d, V_d, write_state, write_state_mask);
        }
        return;
    }
    dim3 grid(R, V_h, (V_d + WARPS - 1) / WARPS);
    dim3 block(BLOCK);
    if (k_last) {
        device::chunk_gated_delta_prefill_batched_warp_tiled_gqa<__nv_bfloat16, true><<<
            grid, block, 0, stream>>>(
            q_norm_kh, k_norm_kh, v, g_log, beta,
            static_cast<__nv_bfloat16*>(state_base),
            slot_ids, qo_indptr, slot_stride_elems,
            out, K_h, V_h, K_d, V_d, write_state, write_state_mask);
    } else {
        device::chunk_gated_delta_prefill_batched_warp_tiled_gqa<__nv_bfloat16, false><<<
            grid, block, 0, stream>>>(
            q_norm_kh, k_norm_kh, v, g_log, beta,
            static_cast<__nv_bfloat16*>(state_base),
            slot_ids, qo_indptr, slot_stride_elems,
            out, K_h, V_h, K_d, V_d, write_state, write_state_mask);
    }
}

}  // namespace pie_cuda_driver::kernels::ssm