#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

// Extensible transcode dispatch (WEIGHT_LOADER_TODO.md A2.2). A transcode is a
// (source decoder) x (target encoder) pair; the kernel keeps the decoded
// float-group intermediate in registers (no HBM round-trip). Adding a source =
// one Decode functor in transcode.cuh + one arm in the source switch; adding a
// target = one Encode functor (with `kGroup`) + one arm in the target switch.
// The compiler then emits the source x target kernel set.
//
// Scope: this fuses QUANT->QUANT (and BF16->quant) where the value must be
// decoded to a float group before re-encoding. Per-channel targets (INT8/FP8,
// one scale per row) use a different per-row reduction and live in their own
// quantize_* kernels; a BF16 source has no intermediate to save, so its only
// benefit here is a single uniform entry point.

enum class TranscodeSource {
    Bf16,
    Fp8E4m3PerGroup,  // FP8 E4M3 + per-block FP32 scale (e.g. DeepSeek/GLM)
};

enum class TranscodeTarget {
    Mxfp4E2m1E8m0,    // group-32, E8M0 byte scales
    // Nvfp4E2m1Fp8,  // group-16, FP8 E4M3 scales — add EncodeNvfp4 + an arm
};

struct TranscodeParams {
    const void*   src = nullptr;        // row-major [rows, cols] source elements
    const float*  src_scale = nullptr;  // per-group FP32 scale (per-group decoders)
    int           src_group_size = 0;   // source block-scale group (per-group decoders)
    std::uint8_t* dst_packed = nullptr; // target packed output
    std::uint8_t* dst_scale = nullptr;  // target scale output (E8M0 bytes, …)
    int rows = 0;
    int cols = 0;
};

// Launch the fused transcode for (src, tgt). Throws on an unregistered pair.
void launch_transcode(
    TranscodeSource src, TranscodeTarget tgt,
    const TranscodeParams& params, cudaStream_t stream);

// Whether a fused kernel exists for (src, tgt). Single source of truth for the
// registered pairs; callers fall back to the decode->encode two-step otherwise.
bool transcode_supported(TranscodeSource src, TranscodeTarget tgt);

}  // namespace pie_cuda_driver::kernels
