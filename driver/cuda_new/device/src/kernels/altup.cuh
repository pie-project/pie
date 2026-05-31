#pragma once

// AltUp ("Alternating Updates") kernels for Gemma-3n / Gemma-4. Each layer
// maintains `K = altup_num_inputs` parallel residual streams; AltUp
// predicts the post-layer state from a learned linear combination of
// the streams and corrects via the active-stream's actual layer output.
// Launcher decls; bodies lifted verbatim from
// driver/cuda/src/kernels/altup.cu (altup_predict_kernel + altup_correct_kernel).
//
// `predict`:
//     predictions[k, t, h] = streams[k, t, h]
//                          + Σ_j coefs[t, j, k] · streams[j, t, h]
//
// `correct`:
//     corrected[k, t, h] = predictions[k, t, h]
//                        + (activated[t, h] - predictions[active, t, h])
//                          · (correction_coefs[t, k] + 1)

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_device::kernels {

// `coefs` is the [T, K, K] per-token coefficient tensor produced by
// `tanh(modality_router(router_norm(active) / H)) @ prediction_coefs`.
// We carry it as float so we can reduce in fp32; bf16 inputs would
// accumulate substantial round-off across the K-summation.
//
// Layout: streams / predictions are [K, T, H] row-major (stream-major,
// matching how the workspace lays them out).
void altup_predict_bf16(
    const void*  streams,        // bf16 [K, T, H]
    const float* coefs,          // fp32 [T, K, K]
    void*        predictions,    // bf16 [K, T, H]
    int K, int T, int H,
    cudaStream_t stream);

// `correction_coefs` is the [T, K] result of `correction_coefs(modalities)`.
// We pre-add the +1.0 here to keep the kernel small.
void altup_correct_bf16(
    const void*  predictions,    // bf16 [K, T, H]
    const void*  activated,      // bf16 [T, H]
    const float* correction_coefs_plus_one,  // fp32 [T, K]
    void*        corrected,      // bf16 [K, T, H]
    int K, int T, int H, int active_idx,
    cudaStream_t stream);

}  // namespace pie_cuda_device::kernels
