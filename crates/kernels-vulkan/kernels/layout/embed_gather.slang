#include "common/bf16.slang"
#include "common/affine.slang"

#ifndef PIE_GROUP_X
#define PIE_GROUP_X 256
#endif

[[vk::binding(0)]] StructuredBuffer<uint> w;
PIE_BUFFER_RO(1, scales);
PIE_BUFFER_RO(2, biases);
[[vk::binding(3)]] StructuredBuffer<int> ids;
PIE_BUFFER_RW(4, out_);

struct Push {
    int hidden;
    int vocab;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint k = gid.x;
    uint m = gid.y;
    uint hidden = uint(pc.hidden);
    if (k >= hidden) {
        return;
    }
    int id = ids[m];
    uint at = m * hidden + k;
    if (id < 0 || id >= pc.vocab) {
        out_[at] = PIE_STORE(0.0);
        return;
    }
    uint row = uint(id);
    uint cpw = uint(PIE_CODES_PER_WORD);
    uint wpr = hidden * uint(PIE_BITS) / 32u;
    uint gpr = hidden / uint(PIE_GROUP);
    uint g = k / uint(PIE_GROUP);
    float code = pie_affine_code(w[row * wpr + k / cpw], k % cpw);
    float s = PIE_LOAD(scales[row * gpr + g]);
    float b = PIE_LOAD(biases[row * gpr + g]);
    out_[at] = PIE_STORE(code * s + b);
}

// pie:instantiate embed_gather_mb_bf16_gs_32_b_2 PIE_GROUP=32 PIE_BITS=2 PIE_GROUP_X=256
// pie:instantiate embed_gather_mb_bf16_gs_32_b_4 PIE_GROUP=32 PIE_BITS=4 PIE_GROUP_X=256
// pie:instantiate embed_gather_mb_bf16_gs_32_b_8 PIE_GROUP=32 PIE_BITS=8 PIE_GROUP_X=256
// pie:instantiate embed_gather_mb_bf16_gs_64_b_2 PIE_GROUP=64 PIE_BITS=2 PIE_GROUP_X=256
// pie:instantiate embed_gather_mb_bf16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_GROUP_X=256
// pie:instantiate embed_gather_mb_bf16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8 PIE_GROUP_X=256
// pie:instantiate embed_gather_mb_bf16_gs_128_b_2 PIE_GROUP=128 PIE_BITS=2 PIE_GROUP_X=256
// pie:instantiate embed_gather_mb_bf16_gs_128_b_4 PIE_GROUP=128 PIE_BITS=4 PIE_GROUP_X=256
// pie:instantiate embed_gather_mb_bf16_gs_128_b_8 PIE_GROUP=128 PIE_BITS=8 PIE_GROUP_X=256
