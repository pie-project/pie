#include "common/bf16.slang"
#include "spatial/grid.slang"

// A convolution as a register-tiled GEMM over `taps * c_in`, the weight taps-major.

#define PIE_BM 64
#define PIE_BN 64
#define PIE_BK 16
#define PIE_TX (PIE_BN / 4)
#define PIE_TY (PIE_BM / 4)
#define PIE_THREADS (PIE_TX * PIE_TY)

PIE_BUFFER_RO(0, x);
[[vk::binding(1)]] StructuredBuffer<int> grid;
PIE_BUFFER_RO(2, w);
[[vk::binding(3)]] StructuredBuffer<float> bias;
PIE_BUFFER_RW(4, o);
[[vk::binding(5)]] StructuredBuffer<int> o_grid;

struct Push {
    int c_in; int c_out;
    int kt; int kh; int kw;
    int st; int sh; int sw;
    int pt; int ph; int pw;
    int causal;
    int replicate;
    int clips;
    int rows_out;
    uint has_bias;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared float4 xs[PIE_BK * PIE_TY];
groupshared float4 ws[PIE_BK * PIE_TX];
groupshared int4 corner[PIE_BM];
groupshared int4 clip_box[PIE_BM];
groupshared int src[PIE_BM];

int tap_row(int4 box, int4 at, int it, int ih, int iw) {
    int hi = at.y + ih;
    int wi = at.z + iw;
    if (at.w < 0 || hi < 0 || hi >= box.y || wi < 0 || wi >= box.z) return -1;
    int ti = at.x + it;
    if (ti < 0) {
        if (pc.replicate == 0) return -1;
        ti = 0;
    }
    if (ti >= box.x) {
        if (pc.causal != 0 || pc.replicate == 0) return -1;
        ti = box.x - 1;
    }
    return box.w + (ti * box.y + hi) * box.z + wi;
}

[numthreads(PIE_TX, PIE_TY, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint flat = local.x + PIE_TX * local.y;
    int row0 = int(group.y) * PIE_BM;
    int col0 = int(group.x) * PIE_BN;
    int k = pc.kt * pc.kh * pc.kw * pc.c_in;

    if (flat < uint(PIE_BM)) {
        int row = row0 + int(flat);
        int4 at = int4(0, 0, 0, -1);
        int4 box = int4(1, 1, 1, 0);
        if (row < pc.rows_out) {
            SpatialClip og;
            int l = clip_of(o_grid, pc.clips, row, og);
            if (l >= 0) {
                SpatialClip in_ = clip_at(grid, l);
                SpatialVoxel v = unravel(og, row - og.off);
                at = int4(v.t * pc.st - pc.pt, v.h * pc.sh - pc.ph, v.w * pc.sw - pc.pw, l);
                box = int4(in_.t, in_.h, in_.w, in_.off);
            }
        }
        corner[flat] = at;
        clip_box[flat] = box;
    }

    float4 acc[4];
    [ForceUnroll] for (uint i = 0u; i < 4u; i++) acc[i] = float4(0.0);

    int plane = pc.kh * pc.kw;
    for (int k0 = 0; k0 < k; k0 += PIE_BK) {
        int tap = k0 / pc.c_in;
        int c0 = k0 - tap * pc.c_in;
        GroupMemoryBarrierWithGroupSync();
        if (flat < uint(PIE_BM)) {
            int it = tap / plane;
            int rest = tap - it * plane;
            int ih = rest / pc.kw;
            src[flat] = tap_row(clip_box[flat], corner[flat], it, ih, rest - ih * pc.kw);
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint e = flat; e < uint(PIE_BK * PIE_BM); e += uint(PIE_THREADS)) {
            uint kk = e % uint(PIE_BK);
            uint r = e / uint(PIE_BK);
            int from = src[r];
            float v = from >= 0 ? PIE_LOAD(x[uint(from) * uint(pc.c_in) + uint(c0) + kk]) : 0.0;
            xs[kk * uint(PIE_TY) + r / 4u][r % 4u] = v;
        }
        for (uint e = flat; e < uint(PIE_BK * PIE_BN); e += uint(PIE_THREADS)) {
            uint kk = e % uint(PIE_BK);
            uint c = e / uint(PIE_BK);
            int col = col0 + int(c);
            float v = col < pc.c_out ? PIE_LOAD(w[uint(col) * uint(k) + uint(k0) + kk]) : 0.0;
            ws[kk * uint(PIE_TX) + c / 4u][c % 4u] = v;
        }
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll] for (uint kk = 0u; kk < uint(PIE_BK); kk++) {
            float4 a = xs[kk * uint(PIE_TY) + local.y];
            float4 b = ws[kk * uint(PIE_TX) + local.x];
            acc[0] += a.x * b;
            acc[1] += a.y * b;
            acc[2] += a.z * b;
            acc[3] += a.w * b;
        }
    }

    [ForceUnroll] for (uint r = 0u; r < 4u; r++) {
        int row = row0 + int(local.y * 4u + r);
        if (row >= pc.rows_out) continue;
        bool live = corner[local.y * 4u + r].w >= 0;
        [ForceUnroll] for (uint c = 0u; c < 4u; c++) {
            int col = col0 + int(local.x * 4u + c);
            if (col >= pc.c_out) continue;
            float sum = live ? acc[r][c] + (pc.has_bias != 0u ? bias[col] : 0.0) : 0.0;
            o[uint(row) * uint(pc.c_out) + uint(col)] = PIE_STORE(sum);
        }
    }
}

// pie:instantiate spatial_conv3d_rt_bf16
