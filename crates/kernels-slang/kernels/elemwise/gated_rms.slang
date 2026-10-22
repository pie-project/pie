#include "prim/bf16.slang"
#include "prim/params.slang"
#include "prim/reduce.slang"

// A gated RMS norm over f32 rows, one workgroup a `vd`-wide head:
// `y = x / rms(x) * w * gate(z)`, the gate `z * sigmoid(z)` (SiLU) or, with
// `PIE_SIGMOID`, `sigmoid(z)`. The rows and the weight are f32, the gate and
// the landing bf16 two a word, so a head is an even number wide.

[[vk::binding(0)]] StructuredBuffer<float> x;
[[vk::binding(1)]] StructuredBuffer<uint> z;
[[vk::binding(2)]] StructuredBuffer<float> w;
[[vk::binding(3)]] RWStructuredBuffer<uint> y;

struct Params {
    float eps;
    int vd;
};
PIE_PARAMS(4, Params, params);

float gate(float zr) {
    float e = 1.0 / (1.0 + exp(-abs(zr)));
    float sig = zr < 0.0 ? 1.0 - e : e;
#ifdef PIE_SIGMOID
    return sig;
#else
    return zr * sig;
#endif
}

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint vd = uint(max(params.vd, 0));
    uint base = group.x * vd;
    uint lid = local.x;
    float acc = 0.0;
    for (uint i = lid; i < vd; i += uint(PIE_GROUP_X)) {
        float xi = x[base + i];
        acc += xi * xi;
    }
    float inv = pie_inv_rms(acc, lid, vd, params.eps);
    for (uint i = 2u * lid; i < vd; i += 2u * uint(PIE_GROUP_X)) {
        uint word = z[(base + i) >> 1];
        float lo = (x[base + i] * inv) * w[i] * gate(bf16_to_f32(word & 0xffffu));
        float hi = (x[base + i + 1u] * inv) * w[i + 1u] * gate(bf16_to_f32(word >> 16));
        y[(base + i) >> 1] = pack_bf16(lo, hi);
    }
}

// pie:instantiate elemwise_gated_rms_bf16 PIE_GROUP_X=256
// pie:instantiate elemwise_gated_rms_sigmoid_bf16 PIE_GROUP_X=256 PIE_SIGMOID=1
