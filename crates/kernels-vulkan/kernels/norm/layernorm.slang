#include "common/bf16.slang"
#include "common/reduce.slang"

#ifndef PIE_GROUP_X
#define PIE_GROUP_X 256
#endif

PIE_BUFFER_RO(0, x);
PIE_BUFFER_RO(1, w);
PIE_BUFFER_RO(2, b);
PIE_BUFFER_RW(3, out_);

struct Push { float eps; uint axis_size; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint n = pc.axis_size;
    uint base = group.x * n;
    float acc = 0.0;
    for (uint i = local.x; i < n; i += uint(PIE_GROUP_X)) {
        acc += PIE_LOAD(x[base + i]);
    }
    float mean = pie_workgroup_sum(acc) / float(n);
    float spread = 0.0;
    for (uint i = local.x; i < n; i += uint(PIE_GROUP_X)) {
        float c = PIE_LOAD(x[base + i]) - mean;
        spread += c * c;
    }
    float inv = rsqrt(pie_workgroup_sum(spread) / float(n) + pc.eps);
    for (uint i = local.x; i < n; i += uint(PIE_GROUP_X)) {
        float c = (PIE_LOAD(x[base + i]) - mean) * inv;
        out_[base + i] = PIE_STORE(c * PIE_LOAD(w[i]) + PIE_LOAD(b[i]));
    }
}

// pie:instantiate layernorm_bf16 PIE_GROUP_X=256
