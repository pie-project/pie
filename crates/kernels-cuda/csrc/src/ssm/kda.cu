// Host launchers for Kimi Delta Attention. The `__global__` templates they
// fire live in `ssm/kda.cuh` and are defined ONCE, there — this file includes
// that header rather than carrying a second copy, so the archive and any
// JIT cubin come from the same text.
//
// Every launcher stays, and no row replaces one: all four kernels put the
// HEAD on a grid axis that `LaunchRule::Dims` cannot name, two size dynamic
// shared memory on `D`, and two take the slot stride as a `long long` the
// argument binder refuses. See the header for the reasoning; the ahead-of-time
// path is still the only path that fires these.
#include "pie_device.cuh"
#include "ssm/kda.cuh"
#include "ssm/kda.hpp"

// Host-only. `std::min` clamps a block width and `std::size_t` sizes a
// dynamic shared-memory request; neither crosses into device text, which is
// why they are here and not in the `.cuh` — NVRTC answers no standard header
// at all, measured 0 of 31 in §13.
#include <algorithm>

#include <cstdint>

namespace pie_cuda_driver::kernels::ssm {

// ── Launchers ──────────────────────────────────────────────────────

void kda_gate_beta_bf16(
    const void* raw_g, const void* raw_beta, const float* A_log,
    const float* dt_bias, float* gate_out, float* beta_out,
    int T, int H, int D, float lower_bound, cudaStream_t stream)
{
    if (T <= 0 || H <= 0 || D <= 0) return;
    const dim3 grid(static_cast<unsigned>(T), static_cast<unsigned>(H));
    const int threads = D < 256 ? D : 256;
    device::kda_gate_beta<device::bf16><<<grid, threads, 0, stream>>>(
        static_cast<const device::bf16*>(raw_g),
        static_cast<const device::bf16*>(raw_beta),
        A_log, dt_bias, gate_out, beta_out, T, H, D, lower_bound);
}

void kda_recurrent_step_batched(
    const float* q_norm, const float* k_norm, const float* v,
    const float* gate, const float* beta, float* state_base,
    const device::i32* slot_ids, long long slot_stride_elems, float* out,
    int R, int H, int D, cudaStream_t stream)
{
    if (R <= 0 || H <= 0 || D <= 0) return;
    const dim3 grid(static_cast<unsigned>(R), static_cast<unsigned>(H));
    // A multiple of the warp size: the kernel gives one warp a `v` row.
    const int threads = 256;
    const std::size_t shmem = static_cast<std::size_t>(3 * D) * sizeof(float);
    device::kda_recurrent_step_batched<<<grid, threads, shmem, stream>>>(
        q_norm, k_norm, v, gate, beta, state_base, slot_ids,
        slot_stride_elems, out, H, D);
}

void kda_prefill_batched(
    const float* q_norm, const float* k_norm, const float* v,
    const float* gate, const float* beta, float* state_base,
    const device::i32* slot_ids, const device::u32* qo_indptr,
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
    device::kda_prefill_batched<<<grid, threads, shmem, stream>>>(
        q_norm, k_norm, v, gate, beta, state_base, slot_ids, qo_indptr,
        slot_stride_elems, out, H, D);
}

void kda_o_norm_gated_bf16(
    const float* o, const void* g, const float* weight, void* out,
    int T, int H, int D, float eps, cudaStream_t stream)
{
    if (T <= 0 || H <= 0 || D <= 0) return;
    const dim3 grid(static_cast<unsigned>(T), static_cast<unsigned>(H));
    const int threads = D < 256 ? D : 256;
    device::kda_o_norm_gated<device::bf16><<<grid, threads, 0, stream>>>(
        o, static_cast<const device::bf16*>(g), weight,
        static_cast<device::bf16*>(out), H, D, eps);
}

}  // namespace pie_cuda_driver::kernels::ssm
