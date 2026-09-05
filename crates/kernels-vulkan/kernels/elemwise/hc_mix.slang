#include "common/bf16.slang"

PIE_BUFFER_RO(0, gates);
PIE_BUFFER_RO(1, normed);
PIE_BUFFER_RW(2, y);

struct Push { int n_rows; int m; int h; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint h = gid.x;
    uint n = gid.y;
    uint M = uint(pc.m);
    uint H = uint(pc.h);
    if (h >= H || n >= uint(pc.n_rows)) {
        return;
    }
    uint base = n * M * H + h;
    float acc = 0.0;
    for (uint s = 0; s < M; s++) {
        float g = PIE_LOAD(gates[base + s * H]);
        float v = PIE_LOAD(normed[base + s * H]);
        acc += v / (1.0 + exp(-g));
    }
    y[n * H + h] = PIE_STORE(acc / float(M));
}

// pie:instantiate hc_mix_bf16
