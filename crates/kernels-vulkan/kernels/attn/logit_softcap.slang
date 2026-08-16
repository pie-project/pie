#include "common/bf16.slang"

// gemma4 final logit softcap. The old per-row vocabulary bound deliberately
// stays unused: the Vulkan elementwise launch already is the whole extent.

PIE_BUFFER_RO(0, logits);
PIE_BUFFER_RW(1, out_);

struct SoftcapParams { float cap; uint unused; };
PIE_PARAMS(2, SoftcapParams, p);

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    float x = PIE_LOAD(logits[gid.x]);
    out_[gid.x] = PIE_STORE(p.cap * tanh(x / p.cap));
}

// pie:instantiate logit_softcap_bfloat16
