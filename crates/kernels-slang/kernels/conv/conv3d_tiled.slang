#include "prim/bf16.slang"
#include "prim/clip.slang"
#include "prim/params.slang"

// The convolution of `conv/conv3d.slang` as a register-tiled GEMM over
// `taps * c_in`: a workgroup of 256 lanes a tile of `BM` output rows by `BN`
// channels, a lane `RM` rows by `CN` channels of it, the contraction in
// 16-wide steps staged in workgroup memory. A step stays inside one tap, so
// `c_in` is a multiple of 16; a lane lands its channels two to a bf16 word,
// so `c_out` is even. The tile's rows find their voxels and clips once, then
// each step's tap row; a tile no clip reaches computes nothing. The 4 x 4
// lane fits WGSL's 16 KiB of workgroup memory; the 8-row lanes are SPIR-V's.

#ifndef BN
#define BN 64
#endif
#ifndef CN
#define CN 4
#endif
#ifndef RM
#define RM 4
#endif
#define BK 16
#define TX (BN / CN)
#define TY (256 / TX)
#define BM (TY * RM)
#define THREADS 256

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
    // `x`'s and `o`'s row 0 in the grids' fire-wide rows (a segmented fire's window starts later).
    int x_base;
    int o_base;
    int dt;
    int dh;
    int dw;
};
PIE_PARAMS(6, Params, params);

// A step's operands as scalars, four a float4: a lane writing one component
// of a shared vector would race its neighbours on WGSL, which lowers it to a
// read-modify-write of the whole vector.
groupshared float xs[BK * BM];
groupshared float ws[BK * BN];
// A tile row's first tap corner (t, h, w) and its clip index (-1: none).
groupshared int4 corner[BM];
// A tile row's input clip (t, h, w, off).
groupshared int4 clip_box[BM];
groupshared int src[BM];
groupshared int live;

int tap_row(int4 box, int4 at, int it, int ih, int iw) {
    int hi = at.y + ih * params.dh;
    int wi = at.z + iw * params.dw;
    if (at.w < 0 || hi < 0 || hi >= box.y || wi < 0 || wi >= box.z) {
        return -1;
    }
    int ti = at.x + it * params.dt;
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

    if (flat == 0u) {
        live = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint r = flat; r < uint(BM); r += uint(THREADS)) {
        int row = row0 + int(r);
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
        corner[r] = at;
        clip_box[r] = box;
        if (at.w >= 0) {
            live = 1;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    // A tile past the grid's last clip computes nothing; a strided stack is
    // carved at its input's rows and leaves most of its tiles so. Read
    // uniformly: WGSL lets a barrier follow only workgroup-uniform control flow.
    if (workgroupUniformLoad(live) == 0) {
        return;
    }

    float4 acc[RM][CN / 4];
    [ForceUnroll] for (uint i = 0u; i < uint(RM); i++) {
        [ForceUnroll] for (uint j = 0u; j < uint(CN / 4); j++) {
            acc[i][j] = float4(0.0);
        }
    }

    int plane = params.kh * params.kw;
    uint x_words = uint(params.c_in) / 2u;
    uint w_words = uint(k) / 2u;
    for (int k0 = 0; k0 < k; k0 += BK) {
        int tap = k0 / params.c_in;
        int c0 = k0 - tap * params.c_in;
        GroupMemoryBarrierWithGroupSync();
        for (uint r = flat; r < uint(BM); r += uint(THREADS)) {
            int it = tap / plane;
            int rest = tap - it * plane;
            int ih = rest / params.kw;
            src[r] = tap_row(clip_box[r], corner[r], it, ih, rest - ih * params.kw);
        }
        GroupMemoryBarrierWithGroupSync();
        // The activations: a row's sixteen channels as two runs of four words.
        for (uint e = flat; e < uint(BM * 2); e += uint(THREADS)) {
            uint r = e / 2u;
            uint half_ = e % 2u;
            int from = src[r];
            uint base = uint(from) * x_words + uint(c0) / 2u + half_ * 4u;
            uint at = r / 4u;
            uint lane = r % 4u;
            [ForceUnroll] for (uint q = 0u; q < 4u; q++) {
                uint word = 0u;
                if (from >= 0) {
                    word = x[base + q];
                }
                uint kk = half_ * 8u + q * 2u;
                xs[(kk * uint(BM / 4) + at) * 4u + lane] = bf16_to_f32(word & 0xffffu);
                xs[((kk + 1u) * uint(BM / 4) + at) * 4u + lane] = bf16_to_f32(word >> 16);
            }
        }
        // The weights: a channel's sixteen taps-major values as four runs of two words.
        for (uint e = flat; e < uint(BN * 4); e += uint(THREADS)) {
            uint c = e / 4u;
            uint q = e % 4u;
            int col = col0 + int(c);
            uint base = uint(col) * w_words + uint(k0) / 2u + q * 2u;
            uint at = c / 4u;
            uint lane = c % 4u;
            [ForceUnroll] for (uint p = 0u; p < 2u; p++) {
                uint word = 0u;
                if (col < params.c_out) {
                    word = w[base + p];
                }
                uint kk = q * 4u + p * 2u;
                ws[(kk * uint(BN / 4) + at) * 4u + lane] = bf16_to_f32(word & 0xffffu);
                ws[((kk + 1u) * uint(BN / 4) + at) * 4u + lane] = bf16_to_f32(word >> 16);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll] for (uint kk = 0u; kk < uint(BK); kk++) {
            [ForceUnroll] for (uint q = 0u; q < uint(RM / 4); q++) {
                uint ai = (kk * uint(BM / 4) + local.y * uint(RM / 4) + q) * 4u;
                float4 a = float4(xs[ai], xs[ai + 1u], xs[ai + 2u], xs[ai + 3u]);
                [ForceUnroll] for (uint j = 0u; j < uint(CN / 4); j++) {
                    uint bi = (kk * uint(BN / 4) + local.x * uint(CN / 4) + j) * 4u;
                    float4 b = float4(ws[bi], ws[bi + 1u], ws[bi + 2u], ws[bi + 3u]);
                    acc[q * 4u][j] += a.x * b;
                    acc[q * 4u + 1u][j] += a.y * b;
                    acc[q * 4u + 2u][j] += a.z * b;
                    acc[q * 4u + 3u][j] += a.w * b;
                }
            }
        }
    }

    [ForceUnroll] for (uint r = 0u; r < uint(RM); r++) {
        int row = row0 + int(local.y * uint(RM) + r);
        if (row >= params.rows_out) {
            continue;
        }
        bool alive = corner[local.y * uint(RM) + r].w >= 0;
        [ForceUnroll] for (uint c = 0u; c < uint(CN); c += 2u) {
            int col = col0 + int(local.x * uint(CN) + c);
            if (col >= params.c_out) {
                continue;
            }
            float2 sum = float2(0.0);
            if (alive) {
                sum = float2(acc[r][c / 4u][c % 4u], acc[r][c / 4u][c % 4u + 1u]);
                if (params.has_bias != 0u) {
                    sum += float2(bias[col], bias[col + 1]);
                }
            }
            o[(uint(row) * uint(params.c_out) + uint(col)) >> 1] = pack_bf16(sum.x, sum.y);
        }
    }
}

// pie:instantiate conv_conv3d_tiled_bf16
// pie:instantiate conv_conv3d_tiled_bm_256_bf16 RM=8 CN=8 PIE_TARGET=spirv
// pie:instantiate conv_conv3d_tiled_bm_256_bn_32_bf16 RM=8 BN=32 CN=4 PIE_TARGET=spirv
