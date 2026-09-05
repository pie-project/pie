#include "common/bf16.slang"

PIE_BUFFER_RO(0, o1);
[[vk::binding(1)]] StructuredBuffer<float> lse1;
PIE_BUFFER_RO(2, o2);
[[vk::binding(3)]] StructuredBuffer<float> lse2;
PIE_BUFFER_RW(4, o_out);
[[vk::binding(5)]] RWStructuredBuffer<float> lse_out;
struct Push { int head_dim; int heads; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

bool finite_(float x) {
    return abs(x) < 3.0e38;
}

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint d = gid.x;
    if (d >= uint(pc.head_dim)) return;
    uint h = gid.y;
    uint t = gid.z;
    uint col = t * uint(pc.heads) + h;
    uint i = col * uint(pc.head_dim) + d;
    float l1 = lse1[col];
    float l2 = lse2[col];
    if (!finite_(l2)) {
        o_out[i] = o1[i];
        if (d == 0u) lse_out[col] = l1;
        return;
    }
    if (!finite_(l1)) {
        o_out[i] = o2[i];
        if (d == 0u) lse_out[col] = l2;
        return;
    }
    float merged_max = max(l1, l2);
    float w1 = exp2(l1 - merged_max);
    float w2 = exp2(l2 - merged_max);
    float total = w1 + w2;
    float v1 = PIE_LOAD(o1[i]);
    float v2 = PIE_LOAD(o2[i]);
    o_out[i] = PIE_STORE((v1 * w1 + v2 * w2) / total);
    if (d == 0u) lse_out[col] = merged_max + log2(total);
}

// pie:instantiate merge_lse_combine_bf16
