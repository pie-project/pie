#include "common/bf16.slang"

[[vk::binding(0)]] StructuredBuffer<int> ids;
[[vk::binding(1)]] StructuredBuffer<float> weights;
PIE_BUFFER_RO(2, table);
PIE_BUFFER_RW(3, y);
struct Push { int hidden; int vocab; int taps; int rows; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int c = int(gid.x);
    uint n = gid.y;
    if (c >= pc.hidden || n >= uint(pc.rows)) return;
    uint row_base = n * uint(pc.taps);
    float acc = 0.0;
    for (int t = 0; t < pc.taps; t++) {
        int raw = ids[row_base + uint(t)];
        uint at = uint((raw >= 0 && raw < pc.vocab) ? raw : 0);
        acc += weights[row_base + uint(t)] * PIE_LOAD(table[at * uint(pc.hidden) + uint(c)]);
    }
    y[n * uint(pc.hidden) + uint(c)] = PIE_STORE(acc);
}

// pie:instantiate embed_weighted_bf16
