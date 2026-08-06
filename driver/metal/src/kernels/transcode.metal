// Load-time transcode kernels: what a checkpoint ships, turned into what the
// decode kernels read.
//
// GPT-OSS experts now run directly from native MXFP4. The MXFP4->BF16 path
// remains for explicit load-plan transforms and compatibility artifacts; the
// affine encoders serve runtime-quantized tensors. Each transform runs once
// during staging over Shared storage.
//
// They are on the GPU because they are pure streaming: the pair moves roughly
// eighty gigabytes through DRAM for twenty billion weights, which a host loop
// serves at a few GB/s and unified memory serves at hundreds.

#include <metal_stdlib>
using namespace metal;

// ── MXFP4 -> BF16 ────────────────────────────────────────────────────────────
//
// The format itself is in `mxfp4_codec.h`, which the decode matvec includes
// too. It used to be written twice -- a table here and a table there -- and
// the two copies are read on different paths: this one when the loader
// dequantizes, that one when a kernel reads the published bytes. A drift
// between them is a driver that is right about a checkpoint it converts and
// wrong about the same checkpoint read as it shipped.
#include "mxfp4_codec.h"

struct DequantParams {
    uint blocks;      // E8M0 exponents, one per block
    uint block_size;  // elements per block
};

/// One thread per block. The block is 32 elements in gpt-oss, so this is 16
/// bytes read and 64 written -- wide enough that per-thread setup disappears
/// and narrow enough that the exponent stays in a register.
kernel void mxfp4_dequant_bf16(
    device const uchar* payload [[buffer(0)]],
    device const uchar* exponents [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant DequantParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.blocks) return;
    const uint pb = p.block_size;
    const float factor = mxfp4_block_scale(exponents[gid]);
    const uint first = gid * pb;
    for (uint i = 0; i < pb; i += 2) {
        const uchar byte = payload[(first + i) / 2];
        out[first + i] = bfloat(mxfp4_lo(byte) * factor);
        out[first + i + 1] = bfloat(mxfp4_hi(byte) * factor);
    }
}

// ── BF16/F32 -> MLX affine U4, groups of 64 ──────────────────────────────────

struct EncodeParams {
    uint groups;      // total groups across the whole tensor
    uint group_size;  // elements per group; 64 for every affine tensor here
};

/// The (scale, bias) MLX picks for one group.
///
/// Transcribed from `mlx/backend/cpu/quantized.cpp::quantize`, and mirrored in
/// `loader/src/testkit/host_executor.rs`, because a checkpoint this produces has
/// to be interchangeable with one `mlx_lm convert` produced. Two details are
/// not what an independently written quantizer would do:
///
///   * the scale is NEGATED unless the group's minimum is the larger in
///     magnitude, which puts code 0 on whichever end dominates;
///   * the ENDPOINT is snapped, not the scale -- `scale = edge / round(edge /
///     scale)` -- which keeps the largest magnitude in the group exact;
///   * `w_max` starts at ZERO, so an all-negative group is quantized over the
///     range up to zero rather than up to its own largest element.
///
/// `round` is half AWAY FROM ZERO. `rint` is half to even, and the difference
/// between them was an 8.2% disagreement with `mx.quantize` on an MXFP4-derived
/// expert bank -- every mismatch by exactly one, because those values sit on
/// half-integers by construction.
inline float2 mlx_affine_params(float w_min, float w_max) {
    const bool mask = abs(w_min) > abs(w_max);
    float scale = max((w_max - w_min) / 15.0f, 1e-7f);
    if (!mask) scale = -scale;
    const float edge = mask ? w_min : w_max;
    const float q0 = round(edge / scale);
    float bias = 0.0f;
    if (q0 != 0.0f) {
        scale = edge / q0;
        bias = edge;
    }
    return float2(scale, bias);
}

/// One thread per group: 64 elements in, one packed u32 quad plus a scale and a
/// bias out. Two passes over the group, both out of the same registers.
template <typename T>
inline void encode_affine_group(
    device const T* input,
    device uint* codes,
    device bfloat* scales,
    device bfloat* biases,
    constant EncodeParams& p,
    uint gid) {
    if (gid >= p.groups) return;
    const uint first = gid * p.group_size;

    float w_min = INFINITY;
    float w_max = 0.0f;
    for (uint i = 0; i < p.group_size; ++i) {
        const float value = float(input[first + i]);
        w_min = min(w_min, value);
        w_max = max(w_max, value);
    }
    const float2 params = mlx_affine_params(w_min, w_max);
    scales[gid] = bfloat(params.x);
    biases[gid] = bfloat(params.y);

    // The codes come from the f32 parameters, not their BF16 rounding: MLX
    // rounds only on store, and agreeing with MLX matters more than being
    // self-consistent with what the runtime will read back.
    for (uint w = 0; w < p.group_size / 8; ++w) {
        uint packed = 0;
        for (uint k = 0; k < 8; ++k) {
            const float value = float(input[first + w * 8 + k]);
            const float q = round((value - params.y) / params.x);
            packed |= uint(clamp(q, 0.0f, 15.0f)) << (4 * k);
        }
        codes[first / 8 + w] = packed;
    }
}

kernel void affine_encode_u4_bf16(
    device const bfloat* input [[buffer(0)]],
    device uint* codes [[buffer(1)]],
    device bfloat* scales [[buffer(2)]],
    device bfloat* biases [[buffer(3)]],
    constant EncodeParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    encode_affine_group<bfloat>(input, codes, scales, biases, p, gid);
}

kernel void affine_encode_u4_f32(
    device const float* input [[buffer(0)]],
    device uint* codes [[buffer(1)]],
    device bfloat* scales [[buffer(2)]],
    device bfloat* biases [[buffer(3)]],
    constant EncodeParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    encode_affine_group<float>(input, codes, scales, biases, p, gid);
}
