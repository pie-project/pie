#pragma once


#include "prelude/device.cuh"

namespace pie::spatial {

struct RuleGeom {
    int kind;
    int a0, a1, a2;
    int b0, b1, b2;
    int c0, c1, c2;
    int d0, d1, d2;
    int flag;
    int lanes;
};

__device__ __forceinline__ int conv_axis(int n, int k, int s, int front, int back, bool& ok) {
    const int span = n + front + back - k;
    if (span < 0) { ok = false; return 0; }
    return span / (s > 0 ? s : 1) + 1;
}

__global__ void grid_rule(const int* __restrict__ grid, int* __restrict__ out, RuleGeom g)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    long long off = 0;
    for (int l = 0; l < g.lanes; ++l) {
        const int t = grid[4 * l], h = grid[4 * l + 1], w = grid[4 * l + 2];
        int ot = 0, oh = 0, ow = 0;
        bool ok = true;
        switch (g.kind) {
            case 0: {
                const int back_t = g.flag ? 0 : g.d0;
                ot = conv_axis(t, g.a0, g.b0, g.c0, back_t, ok);
                oh = conv_axis(h, g.a1, g.b1, g.c1, g.d1, ok);
                ow = conv_axis(w, g.a2, g.b2, g.c2, g.d2, ok);
                break;
            }
            case 1:
                ot = (g.flag && t > 0) ? 1 + (t - 1) * g.a0 : t * g.a0;
                oh = h * g.a1;
                ow = w * g.a2;
                break;
            case 2:

                ot = t * g.a0 - g.flag;
                oh = h * g.a1;
                ow = w * g.a2;
                ok = ot > 0;
                break;
            case 3:
                ok = g.a0 > 0 && g.a1 > 0 && g.a2 > 0
                    && t % g.a0 == 0 && h % g.a1 == 0 && w % g.a2 == 0;
                if (ok) { ot = t / g.a0; oh = h / g.a1; ow = w / g.a2; }
                break;
            default:

                ok = g.a0 > 0 && g.a1 > 0 && g.a2 > 0
                    && h % g.a1 == 0 && w % g.a2 == 0;
                if (ok) { ot = (t + g.a0 - 1) / g.a0; oh = h / g.a1; ow = w / g.a2; }
                break;
        }
        if (!ok) { ot = 0; oh = 0; ow = 0; }
        out[4 * l] = ot;
        out[4 * l + 1] = oh;
        out[4 * l + 2] = ow;
        out[4 * l + 3] = static_cast<int>(off);
        off += static_cast<long long>(ot) * oh * ow;
    }
}

}
