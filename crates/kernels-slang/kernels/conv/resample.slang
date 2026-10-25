#include "prim/clip.slang"
#include "prim/pair.slang"
#include "prim/params.slang"

// Resampling over clip tables, a pair of output columns a lane and an output
// row a grid row: each output row's voxel is read from `o_grid` and its
// source from the input clip `grid` names at the same index; a row no clip
// holds lands zero. `PIE_UPSAMPLE`: nearest by `(r1, r2, r3)`, the first
// frame kept alone when `p3` (`(t - 1) * r1 + 1` frames). `PIE_SHUFFLE`:
// channel blocks of `r1 r2 r3` spread into space and time, the first `p3`
// frames trimmed. `PIE_UNSHUFFLE`: the inverse, space and time folded into
// channels. `PIE_AVG_DOWN`: the unshuffle averaged over runs of `p3`
// channels, time front-padded to a multiple of `r1` (padding frames not
// counted, the run still divided by `p3`). `channels` is the row the kernel
// indexes (the output's for upsample and shuffle, the input's otherwise).
// A bf16 row is even wide; an f32 one may be odd.

[[vk::binding(0)]] StructuredBuffer<PIE_ACT> x;
[[vk::binding(1)]] StructuredBuffer<int> grid;
[[vk::binding(2)]] RWStructuredBuffer<PIE_ACT> y;
[[vk::binding(3)]] StructuredBuffer<int> o_grid;

struct Params {
    uint channels;
    uint clips;
    uint r1;
    uint r2;
    uint r3;
#if !defined(PIE_UNSHUFFLE)
    uint p3;
#endif
    uint rows;
    uint row_span;
    // The rows `grid` and `o_grid` address `x`'s and `y`'s row 0 at: a
    // window of a segmented fire starts its tensors at its own rows while
    // the fire's grids hold fire-wide ones.
    uint x_base;
    uint y_base;
};
PIE_PARAMS(4, Params, params);

// Output column `col` of the voxel `o` from input clip `ig`.
float sample(Clip ig, Voxel o, uint col) {
#if defined(PIE_UPSAMPLE)
    int ti = params.p3 != 0u ? (o.t == 0 ? 0 : (o.t - 1) / int(params.r1) + 1) : o.t / int(params.r1);
    int src = ravel(ig, ti, o.h / int(params.r2), o.w / int(params.r3));
    return act_at(x, uint(src) * params.channels + col);
#elif defined(PIE_SHUFFLE)
    int r1 = int(params.r1);
    int r2 = int(params.r2);
    int r3 = int(params.r3);
    int ot = o.t + int(params.p3);
    int src = ravel(ig, ot / r1, o.h / r2, o.w / r3);
    int block = ((ot % r1) * r2 + (o.h % r2)) * r3 + (o.w % r3);
    uint r = params.r1 * params.r2 * params.r3;
    return act_at(x, uint(src) * params.channels * r + col * r + uint(block));
#elif defined(PIE_UNSHUFFLE)
    uint r = params.r1 * params.r2 * params.r3;
    uint cin = col / r;
    uint block = col - cin * r;
    uint i1 = block / (params.r2 * params.r3);
    uint i2 = (block / params.r3) % params.r2;
    uint i3 = block % params.r3;
    int src = ravel(ig, o.t * int(params.r1) + int(i1), o.h * int(params.r2) + int(i2), o.w * int(params.r3) + int(i3));
    return act_at(x, uint(src) * params.channels + cin);
#elif defined(PIE_AVG_DOWN)
    uint r = params.r1 * params.r2 * params.r3;
    int pad_t = (int(params.r1) - ig.t % int(params.r1)) % int(params.r1);
    float acc = 0.0;
    for (uint j = 0u; j < params.p3; ++j) {
        uint q = col * params.p3 + j;
        uint cin = q / r;
        uint block = q - cin * r;
        uint i1 = block / (params.r2 * params.r3);
        uint i2 = (block / params.r3) % params.r2;
        uint i3 = block % params.r3;
        int ti = o.t * int(params.r1) + int(i1) - pad_t;
        if (ti < 0) {
            continue;
        }
        int src = ravel(ig, ti, o.h * int(params.r2) + int(i2), o.w * int(params.r3) + int(i3));
        acc += act_at(x, uint(src) * params.channels + cin);
    }
    return acc / float(params.p3);
#endif
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID, uint3 group : SV_GroupID) {
    uint col = 2u * gid.x;
    uint row = clip_row(group, params.row_span);
    if (row >= params.rows) {
        return;
    }
#if defined(PIE_UNSHUFFLE)
    uint width = params.channels * params.r1 * params.r2 * params.r3;
#elif defined(PIE_AVG_DOWN)
    uint width = params.channels * params.r1 * params.r2 * params.r3 / params.p3;
#else
    uint width = params.channels;
#endif
    if (col >= width) {
        return;
    }
    bool both = col + 1u < width;
    Clip og;
    int fire_row = int(row) + int(params.y_base);
    int l = clip_of(o_grid, int(params.clips), fire_row, og);
    float2 v = float2(0.0);
    if (l >= 0) {
        Clip ig = clip_at(grid, l);
        ig.off -= int(params.x_base);
        Voxel o = unravel(og, fire_row - og.off);
        v = float2(sample(ig, o, col), both ? sample(ig, o, col + 1u) : 0.0);
    }
    pair_put(y, row * width + col, both, v);
}

// pie:instantiate conv_resample_upsample_bf16 PIE_UPSAMPLE=1
// pie:instantiate conv_resample_upsample_f32 PIE_UPSAMPLE=1 PIE_ACT_F32=1
// pie:instantiate conv_resample_shuffle_bf16 PIE_SHUFFLE=1
// pie:instantiate conv_resample_shuffle_f32 PIE_SHUFFLE=1 PIE_ACT_F32=1
// pie:instantiate conv_resample_unshuffle_bf16 PIE_UNSHUFFLE=1
// pie:instantiate conv_resample_unshuffle_f32 PIE_UNSHUFFLE=1 PIE_ACT_F32=1
// pie:instantiate conv_resample_avg_down_bf16 PIE_AVG_DOWN=1
