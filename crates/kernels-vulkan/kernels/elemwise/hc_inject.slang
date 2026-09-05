#include "common/bf16.slang"

#define HC_MAX_MULT 8

PIE_BUFFER_RO(0, o);
PIE_BUFFER_RO(1, gates);
PIE_BUFFER_RW(2, hyper);

struct Push { int n_rows; int m; int h; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint h = gid.x;
    uint n = gid.y;
    uint M = uint(pc.m);
    uint H = uint(pc.h);
    if (M > uint(HC_MAX_MULT) || h >= H || n >= uint(pc.n_rows)) {
        return;
    }
    float ov = PIE_LOAD(o[n * H + h]);
    uint base = n * M * H + h;
    for (uint s = 0; s < M; s++) {
        float logit = PIE_LOAD(gates[n * M + s]) / float(M);
        float g = 2.0 / (1.0 + exp(-logit));
        hyper[base + s * H] = PIE_STORE(PIE_LOAD(hyper[base + s * H]) + g * ov);
    }
}

// pie:instantiate hc_inject_bf16
