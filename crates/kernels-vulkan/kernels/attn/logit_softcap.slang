#include "common/bf16.slang"

PIE_BUFFER_RO(0, logits);
PIE_BUFFER_RW(1, out_);

struct Push { float cap; int n; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    if (gid.x >= uint(pc.n)) return;
    float x = PIE_LOAD(logits[gid.x]);
    out_[gid.x] = PIE_STORE(pc.cap * tanh(x / pc.cap));
}

// pie:instantiate logit_softcap_bfloat16
