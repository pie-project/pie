#include "common/bf16.slang"

#define HC_MAX_MULT 8

PIE_BUFFER_RO(0, x);
PIE_BUFFER_RO(1, residual);
[[vk::binding(2)]] StructuredBuffer<float> post_mix;
[[vk::binding(3)]] StructuredBuffer<float> comb_mix;
PIE_BUFFER_RW(4, out_);

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
    uint comb = n * M * M;
    uint post = n * M;
    float xh = PIE_LOAD(x[n * H + h]);
    uint res = n * M * H + h;
    float r[HC_MAX_MULT];
    for (uint i = 0; i < M; i++) {
        r[i] = PIE_LOAD(residual[res + i * H]);
    }
    for (uint j = 0; j < M; j++) {
        float acc = post_mix[post + j] * xh;
        for (uint i = 0; i < M; i++) {
            acc += comb_mix[comb + i * M + j] * r[i];
        }
        out_[res + j * H] = PIE_STORE(acc);
    }
}

// pie:instantiate hc_fold_bf16
