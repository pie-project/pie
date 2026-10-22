#include "prim/pair.slang"
#include "prim/params.slang"
#include "prim/reduce.slang"

// A residual blend, one workgroup a row and a pair of elements a lane: each
// candidate (the stacked blocks, then the prefix) is RMS-normed, weighted and
// dotted with a projection to a logit; the row is the softmax-weighted sum of
// the raw candidates. The blocks sit as stacked planes,
// `blocks[(j * rows + row) * hidden]`; rows are bf16 and an even width.

#define PIE_MAX_BLOCKS 32

[[vk::binding(0)]] StructuredBuffer<uint> prefix;
[[vk::binding(1)]] StructuredBuffer<uint> blocks;
[[vk::binding(2)]] StructuredBuffer<uint> norm_w;
[[vk::binding(3)]] StructuredBuffer<uint> proj_w;
[[vk::binding(4)]] RWStructuredBuffer<uint> y;

struct Params {
    int blocks_n;
    int hidden;
    int rows;
    float eps;
};
PIE_PARAMS(5, Params, params);

groupshared float pie_prob[PIE_MAX_BLOCKS + 1];

// Candidate `j`'s pair at column `c` of `row`.
float2 candidate(uint j, uint n_blocks, uint row, uint c) {
    uint hidden = uint(max(params.hidden, 0));
    if (j < n_blocks) {
        return pair_at(blocks, (j * uint(max(params.rows, 0)) + row) * hidden + c, true);
    }
    return pair_at(prefix, row * hidden + c, true);
}

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint row = group.x;
    uint hidden = uint(max(params.hidden, 0));
    uint n_blocks = uint(max(params.blocks_n, 0));
    uint cands = n_blocks + 1u;
    uint lid = local.x;
    uint stride = 2u * uint(PIE_GROUP_X);

    for (uint j = 0u; j < cands; j++) {
        float ss = 0.0;
        for (uint c = 2u * lid; c < hidden; c += stride) {
            float2 v = candidate(j, n_blocks, row, c);
            ss += dot(v, v);
        }
        float inv = pie_inv_rms(ss, lid, hidden, params.eps);

        float logit = 0.0;
        for (uint c = 2u * lid; c < hidden; c += stride) {
            float2 v = candidate(j, n_blocks, row, c) * inv;
            logit += dot(v * pair_at(norm_w, c, true), pair_at(proj_w, c, true));
        }
        logit = pie_workgroup_sum(logit, lid);
        if (lid == 0u) {
            pie_prob[j] = logit;
        }
    }

    if (lid == 0u) {
        float m = pie_prob[0];
        for (uint j = 1u; j < cands; j++) {
            m = max(m, pie_prob[j]);
        }
        float sum = 0.0;
        for (uint j = 0u; j < cands; j++) {
            pie_prob[j] = exp(pie_prob[j] - m);
            sum += pie_prob[j];
        }
        for (uint j = 0u; j < cands; j++) {
            pie_prob[j] /= sum;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint c = 2u * lid; c < hidden; c += stride) {
        float2 acc = float2(0.0, 0.0);
        for (uint j = 0u; j < cands; j++) {
            acc += pie_prob[j] * candidate(j, n_blocks, row, c);
        }
        pair_put(y, row * hidden + c, true, acc);
    }
}

// pie:instantiate elemwise_res_blend_bf16 PIE_GROUP_X=256
