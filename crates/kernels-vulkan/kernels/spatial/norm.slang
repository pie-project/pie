#include "common/bf16.slang"
#include "spatial/grid.slang"


void welford_merge(inout float cnt, inout float mean, inout float m2,
                   float cnt_b, float mean_b, float m2_b) {
    float n = cnt + cnt_b;
    if (n == 0.0) return;
    float delta = mean_b - mean;
    float f = cnt_b / n;
    mean += delta * f;
    m2 += m2_b + delta * delta * cnt * f;
    cnt = n;
}

#if defined(PIE_STATS) || defined(PIE_FINALIZE)
#ifdef PIE_STATS
#define PIE_GN_THREADS 256
#else
#define PIE_GN_THREADS 32
#endif
groupshared float pie_gn_cnt[PIE_GN_THREADS];
groupshared float pie_gn_mean[PIE_GN_THREADS];
groupshared float pie_gn_m2[PIE_GN_THREADS];

void butterfly(uint lid, inout float cnt, inout float mean, inout float m2) {
    for (uint off = 16u; off > 0u; off >>= 1u) {
        pie_gn_cnt[lid] = cnt;
        pie_gn_mean[lid] = mean;
        pie_gn_m2[lid] = m2;
        GroupMemoryBarrierWithGroupSync();
        uint other = lid ^ off;
        float o_cnt = pie_gn_cnt[other];
        float o_mean = pie_gn_mean[other];
        float o_m2 = pie_gn_m2[other];
        GroupMemoryBarrierWithGroupSync();
        welford_merge(cnt, mean, m2, o_cnt, o_mean, o_m2);
    }
}
#endif

#ifdef PIE_STATS
PIE_BUFFER_RO(0, x);
[[vk::binding(1)]] StructuredBuffer<int> grid;
[[vk::binding(2)]] RWStructuredBuffer<float4> partials;
struct Push { uint channels; uint groups; uint splits; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GN_THREADS, 1, 1)]
void main(uint3 tgid : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint lid = local.x;
    uint split = tgid.x;
    uint clip = tgid.y;
    uint group = tgid.z;
    SpatialClip g = clip_at(grid, int(clip));
    uint n = uint(max(clip_voxels(g), 0));
    uint chunk = (n + pc.splits - 1u) / pc.splits;
    uint begin = split * chunk;
    uint end = min(n, begin + chunk);
    uint cg = pc.channels / pc.groups;
    uint c0 = group * cg;

    float cnt = 0.0, mean = 0.0, m2 = 0.0;
    for (uint v = begin; v < end; ++v) {
        uint row = (uint(g.off) + v) * pc.channels + c0;
        for (uint c = lid; c < cg; c += uint(PIE_GN_THREADS)) {
            float value = PIE_LOAD(x[row + c]);
            cnt += 1.0;
            float d = value - mean;
            mean += d / cnt;
            m2 += d * (value - mean);
        }
    }
    butterfly(lid, cnt, mean, m2);
    if (lid % 32u == 0u) {
        pie_gn_cnt[lid / 32u] = cnt;
        pie_gn_mean[lid / 32u] = mean;
        pie_gn_m2[lid / 32u] = m2;
    }
    GroupMemoryBarrierWithGroupSync();
    if (lid != 0u) return;
    float tcnt = 0.0, tmean = 0.0, tm2 = 0.0;
    for (uint s = 0u; s < uint(PIE_GN_THREADS) / 32u; ++s) {
        welford_merge(tcnt, tmean, tm2, pie_gn_cnt[s], pie_gn_mean[s], pie_gn_m2[s]);
    }
    partials[(clip * pc.splits + split) * pc.groups + group] = float4(tcnt, tmean, tm2, 0.0);
}
#endif

#ifdef PIE_FINALIZE
[[vk::binding(0)]] StructuredBuffer<float4> partials;
[[vk::binding(1)]] RWStructuredBuffer<float2> stats;
struct Push { uint groups; uint splits; float eps; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GN_THREADS, 1, 1)]
void main(uint3 tgid : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint lid = local.x;
    uint group = tgid.x;
    uint clip = tgid.y;
    float cnt = 0.0, mean = 0.0, m2 = 0.0;
    for (uint s = lid; s < pc.splits; s += 32u) {
        float4 p = partials[(clip * pc.splits + s) * pc.groups + group];
        welford_merge(cnt, mean, m2, p.x, p.y, p.z);
    }
    butterfly(lid, cnt, mean, m2);
    if (lid == 0u) {
        float var = cnt > 0.0 ? m2 / cnt : 0.0;
        stats[clip * pc.groups + group] = float2(mean, rsqrt(var + pc.eps));
    }
}
#endif

#ifdef PIE_APPLY
PIE_BUFFER_RO(0, x);
[[vk::binding(1)]] StructuredBuffer<int> grid;
[[vk::binding(2)]] StructuredBuffer<float2> stats;
[[vk::binding(3)]] StructuredBuffer<float> weight;
[[vk::binding(4)]] StructuredBuffer<float> bias;
PIE_BUFFER_RW(5, y);
struct Push { uint channels; uint groups; uint clips; uint silu; uint rows; uint row_span; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID, uint3 group : SV_GroupID) {
    uint col = gid.x;
    uint row = spatial_row(group, pc.row_span);
    if (col >= pc.channels || row >= pc.rows) return;
    uint e = row * pc.channels + col;
    SpatialClip box;
    int l = clip_of(grid, int(pc.clips), int(row), box);
    float v = 0.0;
    if (l >= 0) {
        uint g = col / (pc.channels / pc.groups);
        float2 st = stats[uint(l) * pc.groups + g];
        v = fma((PIE_LOAD(x[e]) - st.x) * st.y, weight[col], bias[col]);
        if (pc.silu != 0u) v = v / (1.0 + exp(-v));
    }
    y[e] = PIE_STORE(v);
}
#endif

// pie:instantiate spatial_group_norm_stats_bf16 PIE_STATS=1
// pie:instantiate spatial_group_norm_finalize PIE_FINALIZE=1
// pie:instantiate spatial_group_norm_apply_bf16 PIE_APPLY=1
