#include "common/bf16.slang"
#include "common/reduce.slang"

#ifndef PIE_GROUP_X
#define PIE_GROUP_X 256
#endif

#define PIE_MAX_BLOCKS 32

PIE_BUFFER_RO(0, prefix);
PIE_BUFFER_RO(1, blocks);
PIE_BUFFER_RO(2, norm_w);
PIE_BUFFER_RO(3, proj_w);
PIE_BUFFER_RW(4, out_);

struct Push {
    int blocks_n;
    int hidden;
    int rows;
    float eps;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared float pie_prob[PIE_MAX_BLOCKS + 1];

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint row = group.x;
    uint hidden = uint(pc.hidden);
    uint n_blocks = uint(pc.blocks_n);
    uint cands = n_blocks + 1u;
    uint lid = local.x;
    uint token_off = row * hidden;

    for (uint j = 0u; j < cands; j++) {
        bool from_blocks = j < n_blocks;
        uint base = from_blocks ? ((j * uint(pc.rows) + row) * hidden) : token_off;

        float ss = 0.0;
        for (uint h = lid; h < hidden; h += uint(PIE_GROUP_X)) {
            float v = from_blocks ? PIE_LOAD(blocks[base + h]) : PIE_LOAD(prefix[base + h]);
            ss += v * v;
        }
        float inv = pie_inv_rms(ss, hidden, pc.eps);

        float dot = 0.0;
        for (uint h = lid; h < hidden; h += uint(PIE_GROUP_X)) {
            float v = from_blocks ? PIE_LOAD(blocks[base + h]) : PIE_LOAD(prefix[base + h]);
            dot += v * inv * PIE_LOAD(norm_w[h]) * PIE_LOAD(proj_w[h]);
        }
        dot = pie_workgroup_sum(dot);

        if (lid == 0u) {
            pie_prob[j] = dot;
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
        float inv_sum = 1.0 / sum;
        for (uint j = 0u; j < cands; j++) {
            pie_prob[j] *= inv_sum;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint h = lid; h < hidden; h += uint(PIE_GROUP_X)) {
        float acc = 0.0;
        for (uint j = 0u; j < cands; j++) {
            bool from_blocks = j < n_blocks;
            uint base = from_blocks ? ((j * uint(pc.rows) + row) * hidden) : token_off;
            float v = from_blocks ? PIE_LOAD(blocks[base + h]) : PIE_LOAD(prefix[base + h]);
            acc += pie_prob[j] * v;
        }
        out_[token_off + h] = PIE_STORE(acc);
    }
}

// pie:instantiate res_blend_bf16 PIE_GROUP_X=256
