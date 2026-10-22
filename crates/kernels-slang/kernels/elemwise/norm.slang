#include "prim/bf16.slang"
#include "prim/params.slang"
#include "prim/reduce.slang"

// RMS norm over a row's `axis_size`-wide runs, one workgroup a run:
// `y = gain * x / rms(x) * (plus_one ? 1 + w : w)`, the weight read every
// `w_stride` (1: a bank as wide as the run; 0: one scalar). Grouped, each run
// reads its own slice of a bank as wide as the row; unscaled
// (`PIE_UNSCALED`), there is no weight at all. Rows land two values a word,
// so a run is an even number wide.

#ifndef N_READS
#define N_READS 4
#endif

[[vk::binding(0)]] StructuredBuffer<uint> x;
#ifdef PIE_UNSCALED
[[vk::binding(1)]] RWStructuredBuffer<uint> y;

struct Params {
    float eps;
    int axis_size;
};
PIE_PARAMS(2, Params, params);

float gain_at(uint i, uint plane) {
    return 1.0;
}
#else
[[vk::binding(1)]] StructuredBuffer<uint> w;
[[vk::binding(2)]] RWStructuredBuffer<uint> y;

struct Params {
    float eps;
    int axis_size;
    uint w_stride;
    uint plus_one;
    float gain;
#ifdef PIE_GROUPED
    uint groups;
#endif
};
PIE_PARAMS(3, Params, params);

float gain_at(uint i, uint plane) {
    float wv = bf16_at(w, plane + params.w_stride * i);
    return params.gain * (params.plus_one != 0u ? 1.0 + wv : wv);
}
#endif

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint axis_size = uint(max(params.axis_size, 0));
    uint base = group.x * axis_size;
#ifdef PIE_GROUPED
    uint plane = (group.x % max(params.groups, 1u)) * axis_size * params.w_stride;
#else
    uint plane = 0u;
#endif
    uint span = uint(PIE_GROUP_X) * uint(N_READS);
    uint lid = local.x;

    float acc = 0.0;
    for (uint start = lid * uint(N_READS); start < axis_size; start += span) {
        for (uint i = 0u; i < uint(N_READS); i++) {
            if (start + i < axis_size) {
                float xi = bf16_at(x, base + start + i);
                acc += xi * xi;
            }
        }
    }
    float inv = pie_inv_rms(acc, lid, axis_size, params.eps);

    for (uint start = lid * uint(N_READS); start < axis_size; start += span) {
        for (uint i = 0u; i < uint(N_READS); i += 2u) {
            uint at = start + i;
            if (at < axis_size) {
                uint word = x[(base + at) >> 1];
                float lo = gain_at(at, plane) * (bf16_to_f32(word & 0xffffu) * inv);
                float hi = gain_at(at + 1u, plane) * (bf16_to_f32(word >> 16) * inv);
                y[(base + at) >> 1] = pack_bf16(lo, hi);
            }
        }
    }
}

// pie:instantiate elemwise_rms_bf16 PIE_GROUP_X=256 N_READS=4
// pie:instantiate elemwise_rms_grouped_bf16 PIE_GROUP_X=256 N_READS=4 PIE_GROUPED=1
// pie:instantiate elemwise_rms_unscaled_bf16 PIE_GROUP_X=256 N_READS=4 PIE_UNSCALED=1
