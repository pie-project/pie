#include "common/bf16.slang"

#ifndef PIE_GROUP_X
#define PIE_GROUP_X 256
#endif

PIE_BUFFER_RO(0, x);
#if defined(PIE_LEARNED)
PIE_BUFFER_RO(1, lo_p);
PIE_BUFFER_RO(2, hi_p);
PIE_BUFFER_RW(3, out_);
struct Push { uint n; };
#else
PIE_BUFFER_RW(1, out_);
struct Push { float lo; float hi; uint n; };
#endif
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    if (i >= pc.n) {
        return;
    }
#if defined(PIE_LEARNED)
    float lo = PIE_LOAD(lo_p[0]);
    float hi = PIE_LOAD(hi_p[0]);
#else

    float lo = PIE_LOAD(PIE_STORE(pc.lo));
    float hi = PIE_LOAD(PIE_STORE(pc.hi));
#endif
    out_[i] = PIE_STORE(min(max(PIE_LOAD(x[i]), lo), hi));
}

// pie:instantiate clamp_bf16 PIE_GROUP_X=256
// pie:instantiate clamp_learned_bf16 PIE_GROUP_X=256 PIE_LEARNED=1
