// Load-time quant transcodes for the affine and MXFP4 formats.
//
// The affine encoder follows MLX's CPU quantizer where it is unusual: the scale
// sign chooses the dominant endpoint, endpoint snapping changes the scale rather
// than the code, `w_max` starts at zero, and ties round away from zero. MXFP4
// decode shares `common/mxfp4.slang` with runtime readers so paths cannot drift.
//
// BOTH SCALAR PAIRS RIDE A PUSH BLOCK, which is what `quant/transcode.wgsl`
// already did and this file did not. The two counts were a `DequantParams` and
// an `EncodeParams` storage struct apiece -- eight bytes on a descriptor -- and
// each is now the routine's own run: the group or block COUNT, which the body
// derives from `keys::Rows`, followed by the group or block SIZE, which the
// statement states as a `Const<i32>` mark. Same two words, same order, no
// binding.

#include "common/bf16.slang"
#include "common/mxfp4.slang"

#if defined(PIE_MXFP4)
[[vk::binding(0)]] StructuredBuffer<uint8_t> payload;
[[vk::binding(1)]] StructuredBuffer<uint8_t> exponents;
PIE_BUFFER_RW(2, out_);
struct Push { uint blocks; uint block_size; };
[[vk::push_constant]] ConstantBuffer<Push> pc;
#else
#if defined(PIE_F32_INPUT)
[[vk::binding(0)]] StructuredBuffer<float> input_f32;
#else
PIE_BUFFER_RO(0, input_bf16);
#endif
[[vk::binding(1)]] RWStructuredBuffer<uint> codes;
PIE_BUFFER_RW(2, scales);
PIE_BUFFER_RW(3, biases);
struct Push { uint groups; uint group_size; };
[[vk::push_constant]] ConstantBuffer<Push> pc;
#endif

float round_away(float x) { return x < 0.0 ? -floor(-x + 0.5) : floor(x + 0.5); }

#if defined(PIE_MXFP4)
// One thread per block, one workgroup per thread. That is not a chosen
// decomposition: no `layout(local_size_x)` was ever declared here, GLSL
// defaulted the workgroup to (1, 1, 1), and the port wrote down what was
// implicit because Slang requires `[numthreads]`.
//
// It is left at one rather than widened, and the reason is that widening it
// could not be PROVED here. All three of this file's rows -- the two affine
// encoders and this dequantiser -- name no launch rule, and
// `geometry::lanes` refuses an `Unstated` row before it computes a grid, so
// nothing in this workspace dispatches them; the transcodes a model actually
// needs are done on the host by `model-loader`. The rows exist for parity
// with `kernels-metal`, which states the same three. So a wider workgroup
// would be a change no test could run and no text could reach, and the
// honest thing is to keep the shape that was measured and say why.
[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint gid = tid.x;
    if (gid >= pc.blocks) return;
    float factor = mxfp4_block_scale(uint(exponents[gid]));
    uint first = gid * pc.block_size;
    for (uint i = 0u; i < pc.block_size; i += 2u) {
        uint byte_ = uint(payload[(first + i) / 2u]);
        out_[first + i] = PIE_STORE(mxfp4_lo(byte_) * factor);
        out_[first + i + 1u] = PIE_STORE(mxfp4_hi(byte_) * factor);
    }
}
#else
float input_at(uint i) {
#if defined(PIE_F32_INPUT)
    return input_f32[i];
#else
    return PIE_LOAD(input_bf16[i]);
#endif
}
float2 mlx_affine_params(float w_min, float w_max) {
    bool mask = abs(w_min) > abs(w_max);
    float scale = max((w_max - w_min) / 15.0, 1e-7);
    if (!mask) scale = -scale;
    float edge = mask ? w_min : w_max;
    float q0 = round_away(edge / scale);
    float bias = 0.0;
    if (q0 != 0.0) {
        scale = edge / q0;
        bias = edge;
    }
    return float2(scale, bias);
}
// One group per thread, for the reason written above the MXFP4 entrypoint.
[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint gid = tid.x;
    if (gid >= pc.groups) return;
    uint first = gid * pc.group_size;
    float w_min = asfloat(0x7f800000u);
    float w_max = 0.0;
    for (uint i = 0u; i < pc.group_size; i++) {
        float v = input_at(first + i);
        w_min = min(w_min, v);
        w_max = max(w_max, v);
    }
    float2 prm = mlx_affine_params(w_min, w_max);
    scales[gid] = PIE_STORE(prm.x);
    biases[gid] = PIE_STORE(prm.y);
    for (uint word = 0u; word < pc.group_size / 8u; word++) {
        uint packed = 0u;
        for (uint k = 0u; k < 8u; k++) {
            float v = input_at(first + word * 8u + k);
            float q = round_away((v - prm.y) / prm.x);
            packed |= uint(clamp(q, 0.0, 15.0)) << (4u * k);
        }
        codes[first / 8u + word] = packed;
    }
}
#endif

// pie:instantiate affine_encode_u4_bf16 PIE_ENCODE=1
// pie:instantiate affine_encode_u4_f32 PIE_ENCODE=1 PIE_F32_INPUT=1
// pie:instantiate mxfp4_dequant_bf16 PIE_MXFP4=1
