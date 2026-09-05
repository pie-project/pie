#include "common/bf16.slang"
#include "common/reduce.slang"

#ifndef PIE_GROUP_X
#define PIE_GROUP_X 256
#endif

[[vk::binding(0)]] StructuredBuffer<float> x;
PIE_BUFFER_RO(1, z);
[[vk::binding(2)]] StructuredBuffer<float> w;
PIE_BUFFER_RW(3, out_);

struct Push {
    float eps;
    int vd;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint vd = uint(pc.vd);
    uint base = group.x * vd;
    uint lid = local.x;
    float acc = 0.0;
    for (uint i = lid; i < vd; i += uint(PIE_GROUP_X)) {
        float xi = x[base + i];
        acc += xi * xi;
    }
    float inv = pie_inv_rms(acc, vd, pc.eps);
    for (uint i = lid; i < vd; i += uint(PIE_GROUP_X)) {
        float zr = PIE_LOAD(z[base + i]);
        float e = 1.0 / (1.0 + exp(-abs(zr)));
        float sig = zr < 0.0 ? 1.0 - e : e;
#if defined(PIE_SIGMOID)
        float gate = sig;
#else
        float gate = zr * sig;
#endif
        out_[base + i] = PIE_STORE((x[base + i] * inv) * w[i] * gate);
    }
}

// pie:instantiate gated_rms_f32_bf16 PIE_GROUP_X=256
// pie:instantiate gated_rms_sigmoid_f32_bf16 PIE_GROUP_X=256 PIE_SIGMOID=1
