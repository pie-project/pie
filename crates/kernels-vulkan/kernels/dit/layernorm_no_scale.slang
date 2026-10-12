#include "dit/types.slang"
#include "common/reduce.slang"


[[vk::binding(0)]] StructuredBuffer<PIE_XT> x;
[[vk::binding(1)]] RWStructuredBuffer<PIE_XT> o;

struct Push { float eps; uint axis_size; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint n = pc.axis_size;
    uint base = group.x * n;
    float acc = 0.0;
    for (uint i = local.x; i < n; i += 256u) acc += PIE_XL(x[base + i]);
    float mean = pie_workgroup_sum(acc) / float(n);
    float spread = 0.0;
    for (uint i = local.x; i < n; i += 256u) {
        float c = PIE_XL(x[base + i]) - mean;
        spread += c * c;
    }
    float inv = rsqrt(pie_workgroup_sum(spread) / float(n) + pc.eps);
    for (uint i = local.x; i < n; i += 256u) o[base + i] = PIE_XS((PIE_XL(x[base + i]) - mean) * inv);
}

// pie:instantiate layernorm_no_scale_bf16
// pie:instantiate layernorm_no_scale_f32 PIE_X_F32=1
