#include "common/bf16.slang"

[[vk::binding(0)]] StructuredBuffer<int> ids;
PIE_BUFFER_RO(1, table);
PIE_BUFFER_RW(2, out_);

struct Push { int hidden; int vocab; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint hidden = uint(pc.hidden);
    uint c = gid.x;
    uint n = gid.y;
    if (c >= hidden) {
        return;
    }
    int raw = ids[n];
    uint at = n * hidden + c;
    if (raw < 0 || raw >= pc.vocab) {
        out_[at] = PIE_ACT(0);
        return;
    }
    out_[at] = table[uint(raw) * hidden + c];
}

// pie:instantiate embed_concat_bf16
