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
// toggles and `gdn_raise_shmem_cap`. They are host decisions about which
// kernel to fire and how much shared memory to ask a device for, not device
// text; `cudaFuncSetAttribute` does not exist on the other side of a `<<<>>>`.
//
// # What is NOT policy, and is gone
//
// `PIE_QWEN35_GDN_SMEM_STEP` used to pick, at fire time, which of two
// `__global__`s served ONE stated symbol
// (`recurrent_gated_delta_step_batched_gqa_state_bf16`). That is not a
// tuning knob in the shape a knob may take: the same trace, the same
// weights and the same GPU ran different code, and nothing in the plan
// recorded which — so a replay could not reproduce a run and no other
// backend could implement the symbol, because the selector was invisible to
// everything above this file. §30 of `new-horizon.md` measured what it
// selected between: **the two arms are byte-identical**, at eight shapes, on
// both results, including the two the gate excludes. It is deleted, and with
// it the `<cstdlib>` this file included for exactly one call.
#include "ssm/gated_delta_net.hpp"
#include "ssm/gated_delta_net.cuh"
#include "ssm/gated_delta_net_prep.cuh"

#include <cuda_bf16.h>
#include <cstdint>
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
// faster, and +32% end-to-end throughput on Qwen/Qwen3.5-4B
// (6924 -> 9166 tok/s).
//
// It rounds where the legacy kernel rounds, on purpose and not by luck —
// `gated_delta_net.cuh`'s phase 2 says why — so the two are the SAME
// FUNCTION and the choice between them is a choice of speed only. Which is
// why there is no toggle for it any more: see the gate in
// `recurrent_gated_delta_step_batched_gqa_state_bf16` below.
//
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

// 9x speedup over the legacy per-token HBM kernel, bit-identical at
// production shapes (V_d=128, K_d<=128).
constexpr bool qwen_gdn_fla_prefill_enabled() { return true; }

}  // namespace

// §43 deleted `bf16_to_fp32`, `fp32_to_bf16` and `l2norm_scale_bf16_to_fp32`
// here. All three are named by `device::JIT_DISPATCHED`, so the shim
// generates no entry for them and nothing in any language reached these
// ahead-of-time launchers. The kernels are unchanged in
// `ssm/gated_delta_net.cuh`, which is what the rows fire through NVRTC.
// `repeat_interleave_heads_fp32` below is NOT routed and stays.
//
// The anonymous-namespace helper `qwen_gdn_smem_step_enabled` is also gone,
// but NOT with them, and the difference matters to anyone reading this for
// what is safe to delete. §30 removed it, and its only caller was
// `recurrent_gated_delta_step_batched_gqa_state_bf16` at `:201` — which is
// live, is what Qwen3.5 decode calls, and is still here. It went because a
// `std::getenv` may not pick a kernel, not because its caller went; the
// argument is at `:213-246`.

// `repeat_interleave_heads_fp32` was deleted by §43.9 — it is now routed, and
// the note above (written when it was not) is superseded.

// `gated_delta_g_beta` was deleted here by §43. `families::ssm` names it
// as the one launcher in the tree a row "would name a kernel no trace can
// ask for" -- and §41 confirmed the other half: nothing asks for it at all.

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

// `recurrent_gated_delta_step` and `recurrent_gated_delta_step_state_bf16`
// were deleted here by §43 -- the unbatched pair, superseded by the four
// `_batched` launchers below, which `families::ssm` records as doing the same
// job. Their last caller was `chunk_gated_delta_prefill`, itself unreachable:
// orphaned at one remove, which is §41's whole shape.

// `recurrent_gated_delta_step_batched` was deleted by §43.9 — routed, no shim
// entry, no C++ caller. **It was a four-arm selector and the row inherits all
// of it**: `fused = qwen_gdn_fused_step_enabled() && K_d <= 256` picks
// `recurrent_step_batched_fused` over `recurrent_step_batched`, and
// `qwen_gdn_k_last_state_enabled()` picks the `KLast` template argument.
// Shared memory is `(2 * K_d + (fused ? 1 : 0)) * sizeof(float)` — the fused
// kernel needs the sq+sk arrays plus one float for the `sum_sk_sq` broadcast.
// The 256 bound is a dispatch on the MAXIMUM K_d, not the actual: the kernel
// iterates `[0, K_d)` so unused slots are dead code, and the bound keeps the
// per-thread `state_cache` in registers without spilling. K_d up to 256 covers
// every qwen3_5 GDN config in production (the E4B family is K_d=128).

// `recurrent_gated_delta_step_batched_state_bf16` was deleted by §43.9 —
// routed, no shim entry, no C++ caller. Same four-arm selector as the fp32
// form above, with `__nv_bfloat16` as the state type.

// `recurrent_gated_delta_step_batched_gqa` was deleted by §43.9 — routed, no
// shim entry, no C++ caller. **Five arms, and the row inherits every one.**
// Beyond the fused/KLast pair above there is an FLA-style fast path, opt-in
// via `PIE_QWEN35_GDN_FLA_STEP=1` and currently `constexpr false`, taken only
// when `!KLast && K_d <= 128 && V_d % 64 == 0`; it launches a THREE-axis grid
// `(V_d / 64, R, V_h)` with a 64-wide block and `2 * 128 * sizeof(float)` of
// shared memory, where every other arm launches `(R, V_h)` with 128 threads.
// It also guarded `V_h % K_h != 0` by returning without launching.

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
    //
    // # The predicate is the SHAPE, and it used to be the environment
    //
    // This gate read `qwen_gdn_smem_step_enabled() && ...`, which read
    // `std::getenv("PIE_QWEN35_GDN_SMEM_STEP")` — one stated symbol, two
    // `__global__`s, and the selector nowhere in the plan. What survives is
    // `V_d == 128 && K_d == 128`: a fact about the fire, which the model
    // already states, a table can already carry (`new-horizon.md` §26.10(b)),
    // and any backend can read. The env var is deleted rather than moved,
    // because the measurement says there was nothing to move:
    //
    //   arm A  recurrent_step_batched_gqa_smem<128>       (this one)
    //   arm B  recurrent_step_batched_gqa<__nv_bfloat16, false>
    //
    // are BYTE-IDENTICAL — state slab and `out`, zero differing bytes, at
    // R = 1, 7, 13 (with a `slot_ids[r] < 0` hole and a reversed slot map),
    // 64, 511, at a 2-byte-aligned slab that takes the scalar staging path
    // instead of the `uint4` one, and at the two shapes this gate EXCLUDES.
    // §30 of `new-horizon.md` has the run, and the controls that say the
    // comparison can see a difference: a permutation moved 88.92% of the slab
    // with `out` unchanged, a truncation left half the v axis at its input.
    //
    // That is not a coincidence to be re-measured on every edit: the SMEM
    // kernel rounds `state*g` to bf16 before adding delta for no reason but
    // to land where the legacy kernel's HBM round trip lands, and
    // `gated_delta_net.cuh` says so at the line that does it. The two arms
    // are one function with two speeds. A switch between them could only
    // ever choose the slower one (1.48x at R=511, measured on an L40S), and
    // a switch whose reachable effect is "identical, but later" is bring-up
    // scaffolding, not configuration.
    if (!qwen_gdn_k_last_state_enabled() && V_d == 128 && K_d == 128) {
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

// `chunk_gated_delta_prefill` and `chunk_gated_delta_prefill_state_bf16` were
// deleted here by §43. They were the sequential per-token loop the comment
// above described as a placeholder for the chunked algorithm -- and the
// chunked algorithm arrived: `chunk_gated_delta_prefill_batched` below is it.
// `families::ssm` had already measured both as second names for a job a
// `_batched` row does, and §41 measured what was left: no shim entry, no row,
// no caller in any language.

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



// `chunk_gated_delta_prefill_batched_warp_tiled_gqa` was deleted by §43.9 —
// routed, no shim entry, no C++ caller. Three host facts the row inherits:
// it THREW `std::runtime_error("unsupported GQA dimensions")` when
// `K_d > 256 || V_h % K_h != 0` rather than returning quietly;
// `qwen_gdn_gqa_ilp2_enabled()` selected an `_ilp2` kernel whose third grid
// axis divides `V_d` by `WARPS * 2 = 8` instead of `WARPS = 4`; and
// `qwen_gdn_k_last_state_enabled()` picked the `KLast` template argument.

// `chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16` was deleted by
// §43.9 — routed, no shim entry, no C++ caller. Same three host facts as the
// fp32 form above, with `__nv_bfloat16` as the state type.

}  // namespace pie_cuda_driver::kernels::ssm