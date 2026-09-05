#include "common/bf16.slang"

PIE_BUFFER_RO(0, x);
PIE_BUFFER_RW(1, out_);

struct Push { int n_rows; int m; int h; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint h = gid.x;
    uint n = gid.y;
    if (h >= uint(pc.h) || n >= uint(pc.n_rows)) {
        return;
    }
    uint H = uint(pc.h);
    PIE_ACT v = x[n * H + h];
    uint base = n * uint(pc.m) * H + h;
    for (uint m = 0; m < uint(pc.m); m++) {
        out_[base + m * H] = v;
    }
}

// pie:instantiate hc_expand_bf16
