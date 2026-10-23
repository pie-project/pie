#include "prim/bf16.slang"
#include "prim/clip.slang"
#include "prim/params.slang"

// Group norm over clip tables: each clip's channel groups normalized by
// their mean and variance over the clip's every voxel, then the per-channel
// f32 affine and an optional SiLU. Three kernels, Welford moments
// throughout. `PIE_STATS`: a workgroup a (split of a clip's rows, clip,
// group) folds its rows' moments into `partials`. `PIE_FINALIZE`: a 32-lane
// workgroup a (group, clip) folds the splits into `stats`, the mean and
// `1 / sqrt(var + eps)`. `PIE_APPLY`: a lane a pair of channels (one bf16
// word) and a row a grid row lands the normalized row; a row no clip holds
// lands zero.

void welford_merge(inout float cnt, inout float mean, inout float m2, float cnt_b, float mean_b, float m2_b) {
    float n = cnt + cnt_b;
    if (n == 0.0) {
        return;
    }
    float delta = mean_b - mean;
    float f = cnt_b / n;
    mean += delta * f;
    m2 += m2_b + delta * delta * cnt * f;
    cnt = n;
}

#if defined(PIE_STATS) || defined(PIE_FINALIZE)
#ifdef PIE_STATS
#define THREADS 256
#else
#define THREADS 32
#endif
groupshared float gn_cnt[THREADS];
groupshared float gn_mean[THREADS];
groupshared float gn_m2[THREADS];

// Each 32-lane strip's moments merged across its lanes, by xor partners.
void butterfly(uint lid, inout float cnt, inout float mean, inout float m2) {
    for (uint off = 16u; off > 0u; off >>= 1u) {
        gn_cnt[lid] = cnt;
        gn_mean[lid] = mean;
        gn_m2[lid] = m2;
        GroupMemoryBarrierWithGroupSync();
        uint other = lid ^ off;
        float o_cnt = gn_cnt[other];
        float o_mean = gn_mean[other];
        float o_m2 = gn_m2[other];
        GroupMemoryBarrierWithGroupSync();
        welford_merge(cnt, mean, m2, o_cnt, o_mean, o_m2);
    }
}
#endif

#ifdef PIE_STATS
[[vk::binding(0)]] StructuredBuffer<uint> x;
[[vk::binding(1)]] StructuredBuffer<int> grid;
[[vk::binding(2)]] RWStructuredBuffer<float4> partials;
struct Params {
    uint channels;
    uint groups;
    uint splits;
};
PIE_PARAMS(3, Params, params);

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 tgid : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint lid = local.x;
    uint split = tgid.x;
    uint clip = tgid.y;
    uint group = tgid.z;
    Clip g = clip_at(grid, int(clip));
    uint n = uint(max(clip_voxels(g), 0));
    uint chunk = (n + params.splits - 1u) / params.splits;
    uint begin = split * chunk;
    uint end = min(n, begin + chunk);
    uint cg = params.channels / params.groups;
    uint c0 = group * cg;

    float cnt = 0.0;
    float mean = 0.0;
    float m2 = 0.0;
    for (uint v = begin; v < end; ++v) {
        uint row = (uint(g.off) + v) * params.channels + c0;
        for (uint c = lid; c < cg; c += uint(THREADS)) {
            float value = bf16_at(x, row + c);
            cnt += 1.0;
            float d = value - mean;
            mean += d / cnt;
            m2 += d * (value - mean);
        }
    }
    butterfly(lid, cnt, mean, m2);
    if (lid % 32u == 0u) {
        gn_cnt[lid / 32u] = cnt;
        gn_mean[lid / 32u] = mean;
        gn_m2[lid / 32u] = m2;
    }
    GroupMemoryBarrierWithGroupSync();
    if (lid != 0u) {
        return;
    }
    float tcnt = 0.0;
    float tmean = 0.0;
    float tm2 = 0.0;
    for (uint s = 0u; s < uint(THREADS) / 32u; ++s) {
        welford_merge(tcnt, tmean, tm2, gn_cnt[s], gn_mean[s], gn_m2[s]);
    }
    partials[(clip * params.splits + split) * params.groups + group] = float4(tcnt, tmean, tm2, 0.0);
}
#endif

#ifdef PIE_FINALIZE
[[vk::binding(0)]] StructuredBuffer<float4> partials;
[[vk::binding(1)]] RWStructuredBuffer<float2> stats;
struct Params {
    uint groups;
    uint splits;
    float eps;
};
PIE_PARAMS(2, Params, params);

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 tgid : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint lid = local.x;
    uint group = tgid.x;
    uint clip = tgid.y;
    float cnt = 0.0;
    float mean = 0.0;
    float m2 = 0.0;
    for (uint s = lid; s < params.splits; s += 32u) {
        float4 p = partials[(clip * params.splits + s) * params.groups + group];
        welford_merge(cnt, mean, m2, p.x, p.y, p.z);
    }
    butterfly(lid, cnt, mean, m2);
    if (lid == 0u) {
        float var = cnt > 0.0 ? m2 / cnt : 0.0;
        stats[clip * params.groups + group] = float2(mean, rsqrt(var + params.eps));
    }
}
#endif

#ifdef PIE_APPLY
[[vk::binding(0)]] StructuredBuffer<uint> x;
[[vk::binding(1)]] StructuredBuffer<int> grid;
[[vk::binding(2)]] StructuredBuffer<float2> stats;
[[vk::binding(3)]] StructuredBuffer<float> weight;
[[vk::binding(4)]] StructuredBuffer<float> bias;
[[vk::binding(5)]] RWStructuredBuffer<uint> y;
struct Params {
    uint channels;
    uint groups;
    uint clips;
    uint silu;
    uint rows;
    uint row_span;
};
PIE_PARAMS(6, Params, params);

float normed(uint l, uint row, uint col) {
    uint g = col / (params.channels / params.groups);
    float2 st = stats[l * params.groups + g];
    float v = fma((bf16_at(x, row * params.channels + col) - st.x) * st.y, weight[col], bias[col]);
    if (params.silu != 0u) {
        v = v / (1.0 + exp(-v));
    }
    return v;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID, uint3 group : SV_GroupID) {
    uint col = 2u * gid.x;
    uint row = clip_row(group, params.row_span);
    if (col >= params.channels || row >= params.rows) {
        return;
    }
    Clip box;
    int l = clip_of(grid, int(params.clips), int(row), box);
    float2 v = float2(0.0);
    if (l >= 0) {
        v = float2(normed(uint(l), row, col), normed(uint(l), row, col + 1u));
    }
    y[(row * params.channels + col) >> 1] = pack_bf16(v.x, v.y);
}
#endif

// pie:instantiate conv_group_norm_stats_bf16 PIE_STATS=1
// pie:instantiate conv_group_norm_finalize PIE_FINALIZE=1
// pie:instantiate conv_group_norm_apply_bf16 PIE_APPLY=1
