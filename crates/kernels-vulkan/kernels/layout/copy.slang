#ifndef PIE_GROUP_X
#define PIE_GROUP_X 256
#endif

[[vk::binding(0)]] StructuredBuffer<uint> src;
[[vk::binding(1)]] RWStructuredBuffer<uint> dst;

struct Push {
    int words;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    if (i >= uint(pc.words)) {
        return;
    }
    dst[i] = src[i];
}

// pie:instantiate copy_words PIE_GROUP_X=256
