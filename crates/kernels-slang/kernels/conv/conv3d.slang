#include "prim/bf16.slang"
#include "prim/clip.slang"
#include "prim/params.slang"

// A 3-D convolution over clip tables, a 32-lane strip an output channel and
// four strips (four channels, two bf16 words) a workgroup, a workgroup row an
// output row. Each lane walks every tap's input channels at its stride, and
// the strip folds its lanes through workgroup memory. The output row's voxel
// is read from `o_grid`, its taps from the input clip `grid` names at the
// same index: strided, padded, causal in time (no future frame) and with
// time padding either zero or the edge frame repeated. A row no clip holds
// lands zero.
//
// The weight is `[c_out, taps * c_in]`, taps-major in `(t, h, w)` order, or
// `[c_out, taps]` for a depthwise one, each channel convolved alone; the
// taps sit `dilation` apart along each axis. The bias, when there is one,
// is an f32 per output channel. `c_out` is even.

#define STRIPS 4
#define LANES 32
#define THREADS (STRIPS * LANES)

[[vk::binding(0)]] StructuredBuffer<uint> x;
[[vk::binding(1)]] StructuredBuffer<int> grid;
[[vk::binding(2)]] StructuredBuffer<uint> w;
[[vk::binding(3)]] StructuredBuffer<float> bias;
[[vk::binding(4)]] RWStructuredBuffer<uint> o;
[[vk::binding(5)]] StructuredBuffer<int> o_grid;

struct Params {
    int c_in;
    int c_out;
    int kt;
    int kh;
    int kw;
    int st;
    int sh;
    int sw;
    int pt;
    int ph;
    int pw;
    int causal;
    int replicate;
    int clips;
    int rows_out;
    uint has_bias;
    uint row_span;
    // `x`'s and `o`'s row 0 in the grids' fire-wide rows (a segmented fire's window starts later).
    int x_base;
    int o_base;
    int dt;
    int dh;
    int dw;
    int depthwise;
};
PIE_PARAMS(6, Params, params);

groupshared float conv_fold[STRIPS][LANES];

// The input row tap `(it, ih, iw)` of output voxel `v` reads, or -1 for a
// padded one.
int tap_row(Clip in_, Voxel v, int it, int ih, int iw) {
    int hi = v.h * params.sh - params.ph + ih * params.dh;
    int wi = v.w * params.sw - params.pw + iw * params.dw;
    if (hi < 0 || hi >= in_.h || wi < 0 || wi >= in_.w) {
        return -1;
    }
    int ti = v.t * params.st - params.pt + it * params.dt;
    if (ti < 0) {
        if (params.replicate == 0) {
            return -1;
        }
        ti = 0;
    }
    if (ti >= in_.t) {
        if (params.causal != 0 || params.replicate == 0) {
            return -1;
        }
        ti = in_.t - 1;
    }
    return ravel(in_, ti, hi, wi);
}

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    int row = int(clip_row(group, params.row_span));
    if (row >= params.rows_out) {
        return;
    }
    int tid = int(local.x);
    int strip = tid / LANES;
    int lane = tid % LANES;
    int n = int(group.x) * STRIPS + strip;
    bool live = n < params.c_out;

    Clip og;
    int fire_row = row + params.o_base;
    int l = clip_of(o_grid, params.clips, fire_row, og);
    float acc = 0.0;
    if (l >= 0 && live) {
        Clip in_ = clip_at(grid, l);
        in_.off -= params.x_base;
        Voxel v = unravel(og, fire_row - og.off);
        int taps = params.kt * params.kh * params.kw;
        int plane = params.kh * params.kw;
        if (params.depthwise != 0) {
            // Each output channel reads its own input channel alone.
            for (int tap = lane; tap < taps; tap += LANES) {
                int it = tap / plane;
                int rest = tap - it * plane;
                int ih = rest / params.kw;
                int src = tap_row(in_, v, it, ih, rest - ih * params.kw);
                if (src < 0) {
                    continue;
                }
                acc = fma(bf16_at(x, uint(src) * uint(params.c_out) + uint(n)), bf16_at(w, uint(n) * uint(taps) + uint(tap)), acc);
            }
        } else {
            uint wrow = uint(n) * uint(taps) * uint(params.c_in);
            for (int tap = 0; tap < taps; ++tap) {
                int it = tap / plane;
                int rest = tap - it * plane;
                int ih = rest / params.kw;
                int iw = rest - ih * params.kw;
                int src = tap_row(in_, v, it, ih, iw);
                if (src < 0) {
                    continue;
                }
                uint xr = uint(src) * uint(params.c_in);
                uint wt = wrow + uint(tap) * uint(params.c_in);
                for (int c = lane; c < params.c_in; c += LANES) {
                    acc = fma(bf16_at(x, xr + uint(c)), bf16_at(w, wt + uint(c)), acc);
                }
            }
        }
    }
    conv_fold[strip][lane] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (int half_ = LANES >> 1; half_ > 0; half_ >>= 1) {
        if (lane < half_) {
            conv_fold[strip][lane] += conv_fold[strip][lane + half_];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    // Strips 0 and 2 land their channel and the next one's as a word.
    if (lane == 0 && (strip & 1) == 0 && live) {
        float2 sum = float2(conv_fold[strip][0], conv_fold[strip + 1][0]);
        if (l < 0) {
            sum = float2(0.0);
        } else if (params.has_bias != 0u) {
            sum += float2(bias[n], bias[n + 1]);
        }
        o[(uint(row) * uint(params.c_out) + uint(n)) >> 1] = pack_bf16(sum.x, sum.y);
    }
}

// pie:instantiate conv_conv3d_bf16
