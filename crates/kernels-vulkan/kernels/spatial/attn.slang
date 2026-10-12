#include "common/bf16.slang"
#include "spatial/grid.slang"

#ifndef PIE_C_MAX
#error "spatial attention requires -DPIE_C_MAX"
#endif
#define PIE_STRIPS 4
#define PIE_LANES 32
#define PIE_THREADS (PIE_STRIPS * PIE_LANES)
#define PIE_VPT (PIE_C_MAX / PIE_LANES)

PIE_BUFFER_RO(0, q);
PIE_BUFFER_RO(1, k);
PIE_BUFFER_RO(2, v);
PIE_BUFFER_RW(3, y);
[[vk::binding(4)]] StructuredBuffer<int> grid;
struct Push { uint channels; uint clips; int seg_frames; float sm_scale; uint rows; uint row_span; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared float pie_sa_q[PIE_C_MAX];
groupshared float pie_sa_fold[PIE_STRIPS][PIE_LANES];
groupshared float pie_sa_acc[PIE_STRIPS * PIE_C_MAX];
groupshared float pie_sa_m[PIE_STRIPS];
groupshared float pie_sa_l[PIE_STRIPS];
groupshared int pie_sa_span[2];

[numthreads(PIE_THREADS, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    int row = int(spatial_row(group, pc.row_span));
    if (uint(row) >= pc.rows) return;
    int tid = int(local.x);
    int strip = tid / PIE_LANES;
    int lane = tid % PIE_LANES;
    uint c = pc.channels;

    if (tid == 0) {
        SpatialClip g;
        int l = clip_of(grid, int(pc.clips), row, g);
        int begin = 0, end = 0;
        if (l >= 0) segment_of(g, row - g.off, pc.seg_frames, begin, end);
        pie_sa_span[0] = begin;
        pie_sa_span[1] = end;
    }
    GroupMemoryBarrierWithGroupSync();
    int begin = pie_sa_span[0];
    int end = pie_sa_span[1];

    uint out_row = uint(row) * c;
    if (end <= begin) {
        for (uint i = uint(tid); i < c; i += PIE_THREADS) y[out_row + i] = PIE_STORE(0.0);
        return;
    }
    for (uint i = uint(tid); i < c; i += PIE_THREADS) pie_sa_q[i] = PIE_LOAD(q[out_row + i]);
    GroupMemoryBarrierWithGroupSync();

    float acc[PIE_VPT];
    for (int u = 0; u < PIE_VPT; u++) acc[u] = 0.0;
    float running_max = -3.0e38;
    float running_sum = 0.0;

    int steps = (end - begin + PIE_STRIPS - 1) / PIE_STRIPS;
    for (int n = 0; n < steps; n++) {
        int j = begin + n * PIE_STRIPS + strip;
        bool live = j < end;
        uint kv_row = live ? uint(j) * c : 0u;
        float dot = 0.0;
        if (live) {
            for (int u = 0; u < PIE_VPT; u++) {
                uint i = uint(lane + u * PIE_LANES);
                if (i < c) dot += pie_sa_q[i] * PIE_LOAD(k[kv_row + i]);
            }
        }
        pie_sa_fold[strip][lane] = dot;
        GroupMemoryBarrierWithGroupSync();
        for (int half_ = PIE_LANES >> 1; half_ > 0; half_ >>= 1) {
            if (lane < half_) pie_sa_fold[strip][lane] += pie_sa_fold[strip][lane + half_];
            GroupMemoryBarrierWithGroupSync();
        }
        dot = pie_sa_fold[strip][0];
        GroupMemoryBarrierWithGroupSync();
        if (live) {
            float score = dot * pc.sm_scale;
            float widened = max(running_max, score);
            float rescale = exp(running_max - widened);
            float weight = exp(score - widened);
            for (int u = 0; u < PIE_VPT; u++) {
                uint i = uint(lane + u * PIE_LANES);
                if (i < c) acc[u] = acc[u] * rescale + weight * PIE_LOAD(v[kv_row + i]);
            }
            running_sum = running_sum * rescale + weight;
            running_max = widened;
        }
    }

    if (lane == 0) {
        pie_sa_m[strip] = running_max;
        pie_sa_l[strip] = running_sum;
    }
    for (int u = 0; u < PIE_VPT; u++) {
        uint i = uint(lane + u * PIE_LANES);
        if (i < c) pie_sa_acc[strip * PIE_C_MAX + int(i)] = acc[u];
    }
    GroupMemoryBarrierWithGroupSync();

    float folded_max = -3.0e38;
    for (int w = 0; w < PIE_STRIPS; w++) folded_max = max(folded_max, pie_sa_m[w]);
    float denominator = 0.0;
    for (int w = 0; w < PIE_STRIPS; w++) denominator += pie_sa_l[w] * exp(pie_sa_m[w] - folded_max);
    float inv = denominator > 0.0 ? 1.0 / denominator : 0.0;
    for (uint i = uint(tid); i < c; i += PIE_THREADS) {
        float sum = 0.0;
        for (int w = 0; w < PIE_STRIPS; w++) sum += pie_sa_acc[w * PIE_C_MAX + int(i)] * exp(pie_sa_m[w] - folded_max);
        y[out_row + i] = PIE_STORE(sum * inv);
    }
}

// pie:instantiate spatial_attention_bf16_c_256 PIE_C_MAX=256
// pie:instantiate spatial_attention_bf16_c_512 PIE_C_MAX=512
// pie:instantiate spatial_attention_bf16_c_1024 PIE_C_MAX=1024
