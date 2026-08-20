#include "common/bf16.slang"

// gemma4 final logit softcap. The old per-row vocabulary bound deliberately
// stays gone: the Vulkan elementwise launch already is the whole extent.
//
// THE CAP IS A PUSH CONSTANT, NOT A STORAGE STRUCT. `SoftcapParams { cap,
// unused }` was Metal's buffer-2 layout carried over whole, second word and
// all, and a storage block for four live bytes cost this plane a descriptor,
// a staging slot and the sentinel `Holds::params_block` mints for a buffer
// with no address. `cap: Const<f32>` is the same word 0 of the same statement
// run, and four bytes is a push range on every device this targets.

PIE_BUFFER_RO(0, logits);
PIE_BUFFER_RW(1, out_);

struct Push { float cap; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    float x = PIE_LOAD(logits[gid.x]);
    out_[gid.x] = PIE_STORE(pc.cap * tanh(x / pc.cap));
}

// pie:instantiate logit_softcap_bfloat16
