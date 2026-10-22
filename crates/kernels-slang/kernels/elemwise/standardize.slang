#include "prim/pair.slang"
#include "prim/params.slang"

// `out[r, c] = (out[r, c] - bias[c]) * scale[c]` in place, a pair of columns
// a lane and a row a grid row. The planes ride the row's element.

[[vk::binding(0)]] RWStructuredBuffer<PIE_ACT> out_;
[[vk::binding(1)]] StructuredBuffer<PIE_ACT> bias;
[[vk::binding(2)]] StructuredBuffer<PIE_ACT> scale;

struct Params {
    int width;
    int rows;
};
PIE_PARAMS(3, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint width = uint(max(params.width, 0));
    uint c = 2u * gid.x;
    if (c >= width || gid.y >= uint(max(params.rows, 0))) {
        return;
    }
    bool both = c + 1u < width;
    uint i = gid.y * width + c;
    float2 v = (pair_at(out_, i, both) - pair_at(bias, c, both)) * pair_at(scale, c, both);
    pair_put(out_, i, both, v);
}

// pie:instantiate elemwise_standardize_bf16 PIE_GROUP_X=256
