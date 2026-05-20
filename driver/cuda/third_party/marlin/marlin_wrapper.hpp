#pragma once

// Torch-free C++ entry point for vLLM's marlin W4A16 GEMM. The vendored
// `marlin.cu` exposes both the upstream torch wrapper (`marlin_gemm(
// torch::Tensor& ...)` — gated under `PIE_MARLIN_KEEP_TORCH_WRAPPER`) and
// the internal raw-pointer launcher (`marlin::marlin_mm(void*, ...)`).
// This header is the intended entry point for pie's GEMM dispatcher.
//
// The wrapper takes column-parallel `[N, K]` row-major weight tensors that
// the loader has already repacked into marlin's expected layout via
// `gptq_marlin_repack` (also vendored). Activations are bf16 row-major
// `[M, K]`; output is bf16 `[M, N]`. Per-group scales `[num_groups, N]`
// where `num_groups = K / group_size` (typically `group_size=128`; pass
// `group_size = -1` for per-channel scales with `num_groups = 1`).

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::marlin {

/// Compute `out = act @ dequant(W_q4)^T` with per-group scales. Activation
/// dtype = bf16. Weight stored as packed int4 (2 nibbles per byte) in
/// marlin's interleaved layout. Output dtype = bf16.
///
/// `workspace` must be a device buffer sized at least
/// `marlin_gptq_workspace_bytes(M, N, K, group_size)`. The caller is
/// responsible for keeping it alive across multiple GEMMs (one workspace
/// per-rank is fine).
///
/// `qzeros_marlin` is non-null for AWQ (asymmetric int4); pass `nullptr`
/// for GPTQ-symmetric (kU4B8). When non-null, the kernel runs in `kU4`
/// mode with `has_zp=true`.
///
/// `use_fp32_reduce` enables fp32-accumulator reduction on Ampere when
/// the M×K work is large enough to benefit from the extra precision.
/// On A100 we default this to true.
void launch_gptq_gemm_w4a16_bf16(
    const void* act_bf16,        // [M, K] bf16
    const void* w_q4_packed,     // marlin-repacked int4
    const void* scales_bf16,     // [num_groups, N] bf16
    const void* qzeros_marlin,   // [num_groups, N/8] int32, or nullptr
    void*       out_bf16,        // [M, N] bf16
    void*       workspace,       // device scratch
    int         M,
    int         N,
    int         K,
    int         group_size,      // 128 / 64 / -1 (per-channel)
    bool        use_fp32_reduce,
    cudaStream_t stream);

/// Compute `out = act @ dequant(W_mxfp4)^T` with GPT-OSS MXFP4 weights.
/// Activation dtype = bf16. Weight values are FE2M1 packed in Marlin's W4
/// layout; scales are raw E8M0 bytes in Marlin's `[K / 32, N]` scale layout.
void launch_mxfp4_gemm_w4a16_bf16(
    const void* act_bf16,        // [M, K] bf16
    const void* w_mxfp4_packed,  // marlin-repacked FE2M1
    const void* scales_e8m0,     // [K / 32, N] E8M0 bytes
    void*       out_bf16,        // [M, N] bf16
    void*       reduce_scratch,  // fp32 split-K scratch
    void*       workspace,       // device scratch
    int         M,
    int         N,
    int         K,
    cudaStream_t stream);

/// Required device workspace size for the gptq_gemm path. Conservative
/// upper bound; safe to allocate once at engine init.
std::size_t marlin_gptq_workspace_bytes(int M, int N, int K, int group_size);

/// Repack a GPTQ-formatted INT4 weight into marlin's tile-packed layout.
/// `qweight_in` is the raw checkpoint bytes — `[size_k / 8, size_n]`
/// int32 packed (8 nibbles per int32, GPTQ convention). `repacked_out`
/// must be sized to `(size_k / tile_size) * (size_n * tile_size / 8)`
/// int32s where `tile_size = 16`. Symmetric (no permutation, no zp);
/// callers with `desc_act=true` need the upstream torch path which
/// pie does not yet wire.
void launch_gptq_repack_w4_no_perm(
    const void*    qweight_in,    // [K/8, N] int32 (GPTQ packed)
    void*          repacked_out,  // [K/16, N*16/8] int32 (marlin layout)
    int            size_k,
    int            size_n,
    cudaStream_t   stream);

/// AWQ INT4 repack — same output layout as GPTQ but the input is packed
/// along the N axis instead of K (`[size_k, size_n / 8]` int32). AWQ
/// also has zero-points (handled separately by the caller via the
/// existing QuantMeta zero_point side tensor).
void launch_awq_repack_w4(
    const void*    qweight_in,    // [K, N/8] int32 (AWQ packed)
    void*          repacked_out,  // [K/16, N*16/8] int32 (marlin layout)
    int            size_k,
    int            size_n,
    cudaStream_t   stream);

}  // namespace pie_cuda_driver::marlin
