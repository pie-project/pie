#include "prim/params.slang"

// The clip table a conv-group op lands, derived on the device from the one
// it reads, one lane for the whole table: each clip's (t, h, w) by the rule
// and its first row the running sum of the clips before it. Rules (`kind`):
// conv (0; `a` the kernel, `b` the stride, `c` the front pad, `d` the back
// pad, none in time when `flag` says causal), upsample (1; `a` the factor,
// the first frame kept alone when `flag`), shuffle (2; `a` the factor,
// `flag` frames trimmed), unshuffle (3; `a` the factor, which must divide),
// avg-down (4; `a` the factor, time rounded up). A clip the rule cannot
// shape lands empty.

[[vk::binding(0)]] StructuredBuffer<int> grid;
[[vk::binding(1)]] RWStructuredBuffer<int> out_;

struct Params {
    int kind;
    int a0;
    int a1;
    int a2;
    int b0;
    int b1;
    int b2;
    int c0;
    int c1;
    int c2;
    int d0;
    int d1;
    int d2;
    int flag;
    int clips;
};
PIE_PARAMS(2, Params, params);

int conv_axis(int n, int k, int s, int front, int back, inout bool ok) {
    int span = n + front + back - k;
    if (span < 0) {
        ok = false;
        return 0;
    }
    return span / (s > 0 ? s : 1) + 1;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    if (gid.x != 0u) {
        return;
    }
    int off = 0;
    for (int l = 0; l < params.clips; ++l) {
        int t = grid[4 * l];
        int h = grid[4 * l + 1];
        int w = grid[4 * l + 2];
        int ot = 0;
        int oh = 0;
        int ow = 0;
        bool ok = true;
        if (params.kind == 0) {
            int back_t = params.flag != 0 ? 0 : params.d0;
            ot = conv_axis(t, params.a0, params.b0, params.c0, back_t, ok);
            oh = conv_axis(h, params.a1, params.b1, params.c1, params.d1, ok);
            ow = conv_axis(w, params.a2, params.b2, params.c2, params.d2, ok);
        } else if (params.kind == 1) {
            ot = (params.flag != 0 && t > 0) ? 1 + (t - 1) * params.a0 : t * params.a0;
            oh = h * params.a1;
            ow = w * params.a2;
        } else if (params.kind == 2) {
            ot = t * params.a0 - params.flag;
            oh = h * params.a1;
            ow = w * params.a2;
            ok = ot > 0;
        } else if (params.kind == 3) {
            ok = params.a0 > 0 && params.a1 > 0 && params.a2 > 0 && t % params.a0 == 0 && h % params.a1 == 0
                && w % params.a2 == 0;
            if (ok) {
                ot = t / params.a0;
                oh = h / params.a1;
                ow = w / params.a2;
            }
        } else {
            ok = params.a0 > 0 && params.a1 > 0 && params.a2 > 0 && h % params.a1 == 0 && w % params.a2 == 0;
            if (ok) {
                ot = (t + params.a0 - 1) / params.a0;
                oh = h / params.a1;
                ow = w / params.a2;
            }
        }
        if (!ok) {
            ot = 0;
            oh = 0;
            ow = 0;
        }
        out_[4 * l] = ot;
        out_[4 * l + 1] = oh;
        out_[4 * l + 2] = ow;
        out_[4 * l + 3] = off;
        off += ot * oh * ow;
    }
}

// pie:instantiate conv_rule
