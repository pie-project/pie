#include "spatial/grid.slang"

// Rules: conv (0), upsample (1), shuffle (2), unshuffle (3), avg-down (4).
[[vk::binding(0)]] StructuredBuffer<int> grid;
[[vk::binding(1)]] RWStructuredBuffer<int> out_;
struct Push {
    int kind;
    int a0; int a1; int a2;
    int b0; int b1; int b2;
    int c0; int c1; int c2;
    int d0; int d1; int d2;
    int flag;
    int clips;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

int conv_axis(int n, int k, int s, int front, int back, inout bool ok) {
    int span = n + front + back - k;
    if (span < 0) { ok = false; return 0; }
    return span / (s > 0 ? s : 1) + 1;
}

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    if (gid.x != 0u) return;
    int off = 0;
    for (int l = 0; l < pc.clips; ++l) {
        int t = grid[4 * l], h = grid[4 * l + 1], w = grid[4 * l + 2];
        int ot = 0, oh = 0, ow = 0;
        bool ok = true;
        if (pc.kind == 0) {
            int back_t = pc.flag != 0 ? 0 : pc.d0;
            ot = conv_axis(t, pc.a0, pc.b0, pc.c0, back_t, ok);
            oh = conv_axis(h, pc.a1, pc.b1, pc.c1, pc.d1, ok);
            ow = conv_axis(w, pc.a2, pc.b2, pc.c2, pc.d2, ok);
        } else if (pc.kind == 1) {
            ot = (pc.flag != 0 && t > 0) ? 1 + (t - 1) * pc.a0 : t * pc.a0;
            oh = h * pc.a1;
            ow = w * pc.a2;
        } else if (pc.kind == 2) {
            ot = t * pc.a0 - pc.flag;
            oh = h * pc.a1;
            ow = w * pc.a2;
            ok = ot > 0;
        } else if (pc.kind == 3) {
            ok = pc.a0 > 0 && pc.a1 > 0 && pc.a2 > 0
                && t % pc.a0 == 0 && h % pc.a1 == 0 && w % pc.a2 == 0;
            if (ok) { ot = t / pc.a0; oh = h / pc.a1; ow = w / pc.a2; }
        } else {
            ok = pc.a0 > 0 && pc.a1 > 0 && pc.a2 > 0
                && h % pc.a1 == 0 && w % pc.a2 == 0;
            if (ok) { ot = (t + pc.a0 - 1) / pc.a0; oh = h / pc.a1; ow = w / pc.a2; }
        }
        if (!ok) { ot = 0; oh = 0; ow = 0; }
        out_[4 * l] = ot;
        out_[4 * l + 1] = oh;
        out_[4 * l + 2] = ow;
        out_[4 * l + 3] = off;
        off += ot * oh * ow;
    }
}

// pie:instantiate spatial_grid_rule
