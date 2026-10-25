#include "prim/bf16.slang"
#include "prim/clip.slang"
#include "prim/params.slang"

// The convolution of `conv/conv3d.slang` as a register-tiled GEMM over
// `taps * c_in`: a workgroup a 64 x 64 tile of output rows and channels, a
// lane a 4 x 4 block of it, the contraction in 16-wide steps staged in
// workgroup memory. A step stays inside one tap, so `c_in` is a multiple of
// 16; a lane lands its four channels as two bf16 words, so `c_out` is even.
// The tile's rows find their voxels and clips once, then each step's tap
// row.

#define BM 64
#define BN 64
#define BK 16
#define TX (BN / 4)
#define TY (BM / 4)
#define THREADS (TX * TY)

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
    // The rows `grid` and `o_grid` address `x`'s and `o`'s row 0 at: a
    // window of a segmented fire starts its tensors at its own rows while
    // the fire's grids hold fire-wide ones.
    int x_base;
    int o_base;
};
PIE_PARAMS(6, Params, params);

groupshared float4 xs[BK * TY];
groupshared float4 ws[BK * TX];
// A tile row's first tap corner (t, h, w) and its clip index (-1: none).
groupshared int4 corner[BM];
// A tile row's input clip (t, h, w, off).
groupshared int4 clip_box[BM];
groupshared int src[BM];

int tap_row(int4 box, int4 at, int it, int ih, int iw) {
    int hi = at.y + ih;
    int wi = at.z + iw;
    if (at.w < 0 || hi < 0 || hi >= box.y || wi < 0 || wi >= box.z) {
        return -1;
    }
    int ti = at.x + it;
    if (ti < 0) {
        if (params.replicate == 0) {
            return -1;
        }
        ti = 0;
    }
    if (ti >= box.x) {
        if (params.causal != 0 || params.replicate == 0) {
            return -1;
        }
        ti = box.x - 1;
    }
    return box.w + (ti * box.y + hi) * box.z + wi;
}

[shader("compute")]
[numthreads(TX, TY, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint flat = local.x + TX * local.y;
    int row0 = int(group.y) * BM;
    int col0 = int(group.x) * BN;
    int k = params.kt * params.kh * params.kw * params.c_in;

    if (flat < uint(BM)) {
        int row = row0 + int(flat);
        int4 at = int4(0, 0, 0, -1);
        int4 box = int4(1, 1, 1, 0);
        if (row < params.rows_out) {
            Clip og;
            int fire_row = row + params.o_base;
            int l = clip_of(o_grid, params.clips, fire_row, og);
            if (l >= 0) {
                Clip in_ = clip_at(grid, l);
                in_.off -= params.x_base;
                Voxel v = unravel(og, fire_row - og.off);
                at = int4(v.t * params.st - params.pt, v.h * params.sh - params.ph, v.w * params.sw - params.pw, l);
                box = int4(in_.t, in_.h, in_.w, in_.off);
            }
        }
        corner[flat] = at;
        clip_box[flat] = box;
    }

    float4 acc[4];
    [ForceUnroll] for (uint i = 0u; i < 4u; i++) {
        acc[i] = float4(0.0);
    }

    int plane = params.kh * params.kw;
    for (int k0 = 0; k0 < k; k0 += BK) {
        int tap = k0 / params.c_in;
        int c0 = k0 - tap * params.c_in;
        GroupMemoryBarrierWithGroupSync();
        if (flat < uint(BM)) {
            int it = tap / plane;
            int rest = tap - it * plane;
            int ih = rest / params.kw;
            src[flat] = tap_row(clip_box[flat], corner[flat], it, ih, rest - ih * params.kw);
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint e = flat; e < uint(BK * BM); e += uint(THREADS)) {
            uint kk = e % uint(BK);
            uint r = e / uint(BK);
            int from = src[r];
            float v = from >= 0 ? bf16_at(x, uint(from) * uint(params.c_in) + uint(c0) + kk) : 0.0;
            xs[kk * uint(TY) + r / 4u][r % 4u] = v;
        }
        for (uint e = flat; e < uint(BK * BN); e += uint(THREADS)) {
            uint kk = e % uint(BK);
            uint c = e / uint(BK);
            int col = col0 + int(c);
            float v = col < params.c_out ? bf16_at(w, uint(col) * uint(k) + uint(k0) + kk) : 0.0;
            ws[kk * uint(TX) + c / 4u][c % 4u] = v;
        }
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll] for (uint kk = 0u; kk < uint(BK); kk++) {
            float4 a = xs[kk * uint(TY) + local.y];
            float4 b = ws[kk * uint(TX) + local.x];
            acc[0] += a.x * b;
            acc[1] += a.y * b;
            acc[2] += a.z * b;
            acc[3] += a.w * b;
        }
    }

    [ForceUnroll] for (uint r = 0u; r < 4u; r++) {
        int row = row0 + int(local.y * 4u + r);
        if (row >= params.rows_out) {
            continue;
        }
        bool live = corner[local.y * 4u + r].w >= 0;
        [ForceUnroll] for (uint c = 0u; c < 4u; c += 2u) {
            int col = col0 + int(local.x * 4u + c);
            if (col >= params.c_out) {
                continue;
            }
            float2 sum = float2(0.0);
            if (live) {
                sum = float2(acc[r][c], acc[r][c + 1u]);
                if (params.has_bias != 0u) {
                    sum += float2(bias[col], bias[col + 1]);
                }
            }
            o[(uint(row) * uint(params.c_out) + uint(col)) >> 1] = pack_bf16(sum.x, sum.y);
        }
    }
}

// pie:instantiate conv_conv3d_tiled_bf16
