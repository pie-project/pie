#include <metal_stdlib>
using namespace metal;

constant float kMxfp4Lut[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

inline float mxfp4_lo(uint8_t byte) { return kMxfp4Lut[byte & 0xf]; }
inline float mxfp4_hi(uint8_t byte) { return kMxfp4Lut[byte >> 4]; }

inline float mxfp4_block_scale(uint8_t code) {
  return code == 0xff ? NAN : metal::ldexp(1.0f, int(code) - 127);
}

kernel void mxfp4_dequant_bf16(
    device const uchar* payload [[buffer(0)]],
    device const uchar* exponents [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    const constant uint& blocks [[buffer(3)]],
    const constant uint& block_size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= blocks) return;
    const uint pb = block_size;
    const float factor = mxfp4_block_scale(exponents[gid]);
    const uint first = gid * pb;
    for (uint i = 0; i < pb; i += 2) {
        const uchar byte = payload[(first + i) / 2];
        out[first + i] = bfloat(mxfp4_lo(byte) * factor);
        out[first + i + 1] = bfloat(mxfp4_hi(byte) * factor);
    }
}

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

template <typename T>
inline void encode_affine_group(
    device const T* input,
    device uint* codes,
    device bfloat* scales,
    device bfloat* biases,

    uint groups,
    uint group_size,
    uint gid) {
    if (gid >= groups) return;
    const uint first = gid * group_size;

    float w_min = INFINITY;
    float w_max = 0.0f;
    for (uint i = 0; i < group_size; ++i) {
        const float value = float(input[first + i]);
        w_min = min(w_min, value);
        w_max = max(w_max, value);
    }
    const float2 params = mlx_affine_params(w_min, w_max);
    scales[gid] = bfloat(params.x);
    biases[gid] = bfloat(params.y);

    for (uint w = 0; w < group_size / 8; ++w) {
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
    const constant uint& groups [[buffer(4)]],
    const constant uint& group_size [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    encode_affine_group<bfloat>(input, codes, scales, biases, groups, group_size, gid);
}

kernel void affine_encode_u4_f32(
    device const float* input [[buffer(0)]],
    device uint* codes [[buffer(1)]],
    device bfloat* scales [[buffer(2)]],
    device bfloat* biases [[buffer(3)]],
    const constant uint& groups [[buffer(4)]],
    const constant uint& group_size [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    encode_affine_group<float>(input, codes, scales, biases, groups, group_size, gid);
}
