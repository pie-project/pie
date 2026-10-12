#include "common/bf16.slang"
#include "spatial/grid.slang"

#define PIE_STRIPS 4
#define PIE_LANES 32
#define PIE_THREADS (PIE_STRIPS * PIE_LANES)

PIE_BUFFER_RO(0, x);
[[vk::binding(1)]] StructuredBuffer<int> grid;
PIE_BUFFER_RO(2, w);
[[vk::binding(3)]] StructuredBuffer<float> bias;
PIE_BUFFER_RO(4, cache);
PIE_BUFFER_RW(5, o);
[[vk::binding(6)]] StructuredBuffer<int> o_grid;
struct Push {
    int c_in; int c_out;
    int kt; int kh; int kw;
    int st; int sh; int sw;
    int pt; int ph; int pw;
    int causal;
    int replicate;
    int clips;
    int rows_out;
    int has_cache;
    uint has_bias;
    uint row_span;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared float pie_conv_fold[PIE_STRIPS][PIE_LANES];

int conv_tap_row(SpatialClip in_, SpatialVoxel v, int cache_base, int it, int ih, int iw,
                 out bool from_cache) {
    from_cache = false;
    int hi = v.h * pc.sh - pc.ph + ih;
    int wi = v.w * pc.sw - pc.pw + iw;
    if (hi < 0 || hi >= in_.h || wi < 0 || wi >= in_.w) return -1;
    int ti = v.t * pc.st - pc.pt + it;
    if (ti < 0) {
        if (pc.causal != 0 && pc.has_cache != 0) {
            from_cache = true;
            return cache_base + ((ti + pc.pt) * in_.h + hi) * in_.w + wi;
        }
        if (pc.replicate == 0) return -1;
        ti = 0;
    }
    if (ti >= in_.t) {
        if (pc.causal != 0 || pc.replicate == 0) return -1;
        ti = in_.t - 1;
    }
    return ravel(in_, ti, hi, wi);
}

[numthreads(PIE_THREADS, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    int row = int(spatial_row(group, pc.row_span));
    if (row >= pc.rows_out) return;
    int tid = int(local.x);
    int strip = tid / PIE_LANES;
    int lane = tid % PIE_LANES;
    int n = int(group.x) * PIE_STRIPS + strip;
    bool live = n < pc.c_out;
    uint at = uint(row) * uint(pc.c_out) + uint(n);

    SpatialClip og;
    int l = clip_of(o_grid, pc.clips, row, og);
    if (l < 0) {
        if (live && lane == 0) o[at] = PIE_STORE(0.0);
        return;
    }
    SpatialClip in_ = clip_at(grid, l);
    SpatialVoxel v = unravel(og, row - og.off);

    int planes = 0;
    for (int j = 0; j < l; ++j) planes += clip_plane(clip_at(grid, j));
    int cache_base = planes * pc.pt;

    int taps = pc.kt * pc.kh * pc.kw;
    float acc = 0.0;
    if (live) {
        uint wrow = uint(n) * uint(taps) * uint(pc.c_in);
        int plane = pc.kh * pc.kw;
        for (int tap = 0; tap < taps; ++tap) {
            int it = tap / plane;
            int rest = tap - it * plane;
            int ih = rest / pc.kw;
            int iw = rest - ih * pc.kw;
            bool from_cache = false;
            int src = conv_tap_row(in_, v, cache_base, it, ih, iw, from_cache);
            if (src < 0) continue;
            uint xr = uint(src) * uint(pc.c_in);
            uint wt = wrow + uint(tap) * uint(pc.c_in);
            if (from_cache) {
                for (int c = lane; c < pc.c_in; c += PIE_LANES) {
                    acc = fma(PIE_LOAD(cache[xr + uint(c)]), PIE_LOAD(w[wt + uint(c)]), acc);
                }
            } else {
                for (int c = lane; c < pc.c_in; c += PIE_LANES) {
                    acc = fma(PIE_LOAD(x[xr + uint(c)]), PIE_LOAD(w[wt + uint(c)]), acc);
                }
            }
        }
    }
    pie_conv_fold[strip][lane] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (int half_ = PIE_LANES >> 1; half_ > 0; half_ >>= 1) {
        if (lane < half_) pie_conv_fold[strip][lane] += pie_conv_fold[strip][lane + half_];
        GroupMemoryBarrierWithGroupSync();
    }
    if (live && lane == 0) {
        float sum = pie_conv_fold[strip][0];
        o[at] = PIE_STORE(pc.has_bias != 0u ? sum + bias[n] : sum);
    }
}

// pie:instantiate spatial_conv3d_bf16
