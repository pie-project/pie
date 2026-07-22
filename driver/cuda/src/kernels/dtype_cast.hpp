#pragma once

// Element-wise dtype casts used by the loader to bring non-bf16
// checkpoints into our standard bf16 format.

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

/// `dst[i] = (bf16)src[i]` for `n` elements.
void launch_cast_fp16_to_bf16(
    const void*   src_fp16,
    void*         dst_bf16,
    std::size_t   n,
    cudaStream_t  stream);

/// `dst[i] = (bf16)src[i]` for `n` fp32 elements. Used when ckpts ship
/// projection scales as fp32 but our GEMM dispatcher (or the cuBLASLt
/// scale path) expects bf16.
void launch_cast_fp32_to_bf16(
    const void*   src_fp32,
    void*         dst_bf16,
    std::size_t   n,
    cudaStream_t  stream);

/// `dst[i] = (fp32)src[i]` for `n` bf16 elements. Used by the
/// compressed-tensors FP8 loader when scales ship as bf16 but the
/// dispatcher (cuBLASLt scale-pointer / dequant fallback) requires
/// fp32. Equivalent to `(int32(bf16_bits) << 16) reinterpreted as fp32`.
void launch_cast_bf16_to_fp32(
    const void*   src_bf16,
    void*         dst_fp32,
    std::size_t   n,
    cudaStream_t  stream);

/// In-place marlin scale permutation. Marlin's gptq W4A16 kernel reads
/// per-group scales in a specific 64-wide column-interleaved layout
/// rather than the row-major `[groups, N]` shape that GPTQ checkpoints
/// ship. This kernel applies the same permutation as vLLM's
/// `marlin_permute_scales` (see `vllm/model_executor/layers/
/// quantization/utils/marlin_utils.py::get_scale_perms`):
///
///   For per-group (group_size < size_k):
///     Flatten to `[-1, 64]`, permute columns by
///         perm[i*8 + j] = i + 8*j   for i,j in 0..8.
///   For per-channel (group_size == -1 or == size_k):
///     Use a different 64-wide perm. Not implemented yet — refuse.
///
/// Runs in-place: `bf16_scales` shape `[groups, size_n]` is rewritten.
void launch_marlin_permute_scales_bf16(
    void*         bf16_scales,
    int           groups,
    int           size_n,
    int           group_size,
    int           size_k,
    cudaStream_t  stream);

/// AWQ → marlin zero-point repack. AWQ stores per-group zero-points as
/// packed int4 in `[groups, N/8]` int32 (each int32 holds 8 nibbles
/// along the N axis with the AWQ-specific [0,2,4,6,1,3,5,7] interleave).
/// Marlin's W4 kernel expects them in `[groups, N/8]` int32 too, but
/// zero-points need BOTH the 64-wide column permutation that scales undergo
/// (`marlin_permute_scales`) AND an additional 8-wide column interleave
/// `[0,2,4,6,1,3,5,7]` (`marlin_zero_points`), applied AFTER the AWQ
/// undo-interleave. This kernel does the full pipeline: undo AWQ interleave →
/// 64-wide scale_perm → 8-wide interleave → repack int4. Verified to match
/// vLLM's `awq_to_marlin_zero_points`.
void launch_awq_qzero_to_marlin_w4(
    const void*   awq_qzeros_in,    // [groups, N/8] int32 (AWQ packing)
    void*         qzeros_marlin_out,// [groups, N/8] int32 (marlin packing)
    int           groups,
    int           size_n,           // N (must be multiple of 64)
    cudaStream_t  stream);

/// AWQ qweight → GPTQ-format qweight conversion. AWQ stores `[K, N/8]`
/// int32 packed along N with the AWQ-specific [0,2,4,6,1,3,5,7] bit
/// interleave; GPTQ stores `[K/8, N]` int32 packed along K with linear
/// bit order. After this conversion the standard `gptq_marlin_repack`
/// can process the result. Mirrors vLLM's `_convert_awq_tensor_layout`
/// qweight branch (awq_marlin.py:99-108).
void launch_awq_qweight_to_gptq_w4(
    const void*   awq_qweight_in,   // [K, N/8] int32 (AWQ packing)
    void*         gptq_qweight_out, // [K/8, N] int32 (GPTQ packing)
    int           size_k,
    int           size_n,
    cudaStream_t  stream);

/// Direct AWQ dequant to bf16 — bypasses marlin entirely. Produces
/// `bf16[N, K]` (HF Linear-compatible row-major) from the AWQ triplet
/// (qweight, qzeros, scales). Used as a drop-in replacement for the
/// marlin path on sm80 where the marlin pipeline is broken; trades the
/// memory savings of W4 for correctness. Numerically identical to the
/// numpy reference: `(unpack(qweight) - unpack(qzeros)[g(k)]) *
/// scales[g(k), n]`.
///
/// Inputs:
///   * `qweight_in`  — `[K, N/8]` int32 (AWQ packing).
///   * `qzeros_in`   — `[groups, N/8]` int32 (AWQ packing).
///   * `scales_in`   — `[groups, N]` bf16. (FP16 scales must be cast
///                     to bf16 by the caller.)
///   * `bf16_out`    — `[N, K]` bf16 (transposed for HF Linear).
void launch_awq_dequant_to_bf16(
    const void*   qweight_in,
    const void*   qzeros_in,
    const void*   scales_in,
    void*         bf16_out,        // [N, K] bf16
    int           size_k,
    int           size_n,
    int           group_size,
    cudaStream_t  stream);

/// Direct GPTQ dequant to bf16. GPTQ packs nibbles along K axis (axis 0
/// of qweight) sequentially (no interleave). Supports both `desc_act=
/// false` (g_idx = nullptr → use g = k / group_size) and `desc_act=true`
/// (g_idx[k] is the per-row scale group lookup). Symmetric (kU4B8)
/// ckpts ship qzeros set to 8 everywhere — the kernel applies
/// (nibble - zp) uniformly so this works for both sym and asym.
///
/// Inputs:
///   * `qweight_in` — `[K/8, N]` int32 (GPTQ packing along K).
///   * `qzeros_in`  — `[groups, N/8]` int32 (packing along N).
///   * `scales_in`  — `[groups, N]` bf16.
///   * `g_idx_in`   — `[K]` int32 or nullptr. When non-null, this is
///                    desc_act=true (act-order); each row k of the
///                    dequanted weight uses scale group g_idx[k].
///   * `bf16_out`   — `[N, K]` bf16 (transposed for HF Linear).
void launch_gptq_dequant_to_bf16(
    const void*   qweight_in,
    const void*   qzeros_in,
    const void*   scales_in,
    const void*   g_idx_in,        // nullable
    void*         bf16_out,        // [N, K] bf16
    int           size_k,
    int           size_n,
    int           group_size,
    cudaStream_t  stream);

}  // namespace pie_cuda_driver::kernels
