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
// The weight is `[c_out, taps * c_in]`, taps-major in `(t, h, w)` order;
// the bias, when there is one, an f32 per output channel. `c_out` is even,
// except under `PIE_FLAT`: there a workgroup's strips are four consecutive
// outputs of the flattened `[rows_out, c_out]`, so an odd `c_out`'s words,
// which straddle rows, land whole from one workgroup.

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
};
PIE_PARAMS(6, Params, params);

groupshared float conv_fold[STRIPS][LANES];
#if defined(PIE_FLAT)
groupshared int flat_n[STRIPS];
groupshared int flat_l[STRIPS];
#endif

// The input row tap `(it, ih, iw)` of output voxel `v` reads, or -1 for a
// padded one.
int tap_row(Clip in_, Voxel v, int it, int ih, int iw) {
    int hi = v.h * params.sh - params.ph + ih;
    int wi = v.w * params.sw - params.pw + iw;
    if (hi < 0 || hi >= in_.h || wi < 0 || wi >= in_.w) {
        return -1;
    }
    int ti = v.t * params.st - params.pt + it;
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
    int tid = int(local.x);
    int strip = tid / LANES;
    int lane = tid % LANES;
#if defined(PIE_FLAT)
    uint first = clip_row(group, params.row_span) * uint(STRIPS);
    uint outputs = uint(params.rows_out) * uint(params.c_out);
    if (first >= outputs) {
        return;
    }
    uint slot = first + uint(strip);
    int row = int(slot / uint(params.c_out));
    int n = int(slot % uint(params.c_out));
    bool live = slot < outputs;
#else
    int row = int(clip_row(group, params.row_span));
    if (row >= params.rows_out) {
        return;
    }
    int n = int(group.x) * STRIPS + strip;
    bool live = n < params.c_out;
#endif

    Clip og;
    int fire_row = row + params.o_base;
    int l = live ? clip_of(o_grid, params.clips, fire_row, og) : -1;
    float acc = 0.0;
    if (l >= 0 && live) {
        Clip in_ = clip_at(grid, l);
        in_.off -= params.x_base;
        Voxel v = unravel(og, fire_row - og.off);
        int taps = params.kt * params.kh * params.kw;
        uint wrow = uint(n) * uint(taps) * uint(params.c_in);
        int plane = params.kh * params.kw;
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
    conv_fold[strip][lane] = acc;
#if defined(PIE_FLAT)
    if (lane == 0) {
        flat_n[strip] = n;
        flat_l[strip] = live ? l : -1;
    }
#endif
    GroupMemoryBarrierWithGroupSync();
    for (int half_ = LANES >> 1; half_ > 0; half_ >>= 1) {
        if (lane < half_) {
            conv_fold[strip][lane] += conv_fold[strip][lane + half_];
        }
        GroupMemoryBarrierWithGroupSync();
    }
#if defined(PIE_FLAT)
    // Strips 0 and 2 land their output and the next one's as a word, the
    // next one zero past the end.
    if (lane == 0 && (strip & 1) == 0 && live) {
        float2 sum = float2(0.0);
        if (flat_l[strip] >= 0) {
            sum.x = conv_fold[strip][0] + (params.has_bias != 0u ? bias[flat_n[strip]] : 0.0);
        }
        if (flat_l[strip + 1] >= 0) {
            sum.y = conv_fold[strip + 1][0] + (params.has_bias != 0u ? bias[flat_n[strip + 1]] : 0.0);
        }
        o[slot >> 1] = pack_bf16(sum.x, sum.y);
    }
#else
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
#endif
}

// pie:instantiate conv_conv3d_bf16
// pie:instantiate conv_conv3d_flat_bf16 PIE_FLAT=1
