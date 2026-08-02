#pragma once

// AltUp auxiliary kernels for Gemma-3n's stream initialization /
// unembed:
//
//   * `magnitude_rescale_bf16` — at AltUp init time, the K-1 streams
//     produced by `altup_projections[i]` need to be rescaled so that
//     each row's RMS magnitude matches `target_rms` (computed from
//     stream 0). Same rescale at the end via `altup_unembed_projections`.
//
//        rms = sqrt(max(mean(x^2, dim=-1), eps))
//        x  *= target_rms / rms
//
//   * `mean_streams_bf16` — sum across the K-stream axis and divide by
//     K. Used at the end of the model: `mean(unembed_streams, dim=0)`
//     before the final RMSNorm + lm_head.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

// Compute per-row RMS of `ref` ([T, H]) into `target_rms_out` ([T]).
//   target_rms_out[t] = sqrt(max(mean(ref[t, :]^2), eps))
void launch_compute_rms_bf16(
    const void*  ref,
    float*       target_rms_out,
    int          T,
    int          H,
    float        eps,
    cudaStream_t stream);

// Rescale `x` ([T, H]) in place so each row matches `target_rms[t]`:
//   new_rms = sqrt(max(mean(x[t, :]^2), eps))
//   x[t, :] *= target_rms[t] / new_rms
void launch_magnitude_rescale_bf16(
    void*        x,
    const float* target_rms,
    int          T,
    int          H,
    float        eps,
    cudaStream_t stream);

// Mean across the K-stream axis: out[t, h] = (1/K) Σ_k streams[k, t, h].
void launch_mean_streams_bf16(
    const void*  streams,        // [K, T, H]
    void*        out,             // [T, H]
    int K, int T, int H,
    cudaStream_t stream);

// Convert bf16 `[T, K*K]` (the cuBLAS output of `prediction_coefs @
// modalities`) into the fp32 `[T, K, K]` coefficient tensor that
// `altup_predict` expects, applying the permute(last two) that HF's
// `prediction_coefs(...).reshape(*, K, K).permute(0, 1, 3, 2)` does:
//
//     coefs_out[t, j, k] = bfloat_in[t, k * K + j]
void launch_altup_unpack_predict_coefs(
    const void*  in_bf16,         // [T, K*K]
    float*       out_fp32,        // [T, K, K]
    int T, int K,
    cudaStream_t stream);

// Convert bf16 `[T, K]` (the cuBLAS output of `correction_coefs @
// modalities`) into the fp32 `[T, K]` tensor that `altup_correct`
// expects, adding +1.0 along the way to match HF's `+ 1.0`.
void launch_altup_unpack_correct_coefs(
    const void*  in_bf16,         // [T, K]
    float*       out_fp32,        // [T, K]
    int T, int K,
    cudaStream_t stream);

// Element-wise tanh, in-place. Used on AltUp's modality-router output
// (HF computes it in fp32 then casts back; we fold both into the
// kernel to avoid a round-trip).
void launch_tanh_bf16(
    void* x,
    int   numel,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
