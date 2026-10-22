#include "prim/pair.slang"
#include "prim/params.slang"

// `out[r, c] += bias[c]`, a pair of columns a lane and a row a grid row. The
// bias rides the row's element, or is bf16 under f32 rows (`PIE_BIAS_BF16`).

[[vk::binding(0)]] RWStructuredBuffer<PIE_ACT> out_;
#ifdef PIE_BIAS_BF16
[[vk::binding(1)]] StructuredBuffer<uint> bias;

float2 bias_pair(uint c, bool both) {
    return float2(bf16_at(bias, c), both ? bf16_at(bias, c + 1u) : 0.0);
}
#else
[[vk::binding(1)]] StructuredBuffer<PIE_ACT> bias;

float2 bias_pair(uint c, bool both) {
    return pair_at(bias, c, both);
}
#endif

struct Params {
    int width;
};
PIE_PARAMS(2, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint width = uint(max(params.width, 0));
    uint c = 2u * gid.x;
    if (c >= width) {
        return;
    }
    bool both = c + 1u < width;
    uint i = gid.y * width + c;
    pair_put(out_, i, both, pair_at(out_, i, both) + bias_pair(c, both));
}

// pie:instantiate elemwise_add_bias_bf16 PIE_GROUP_X=256
// pie:instantiate elemwise_add_bias_f32 PIE_GROUP_X=256 PIE_ACT_F32=1
// pie:instantiate elemwise_add_bias_f32_bf16 PIE_GROUP_X=256 PIE_ACT_F32=1 PIE_BIAS_BF16=1
