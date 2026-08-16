// Residual add, flat and row-strided.
//
// This is the Metal `residual_add.metal` arithmetic moved without changing the
// aliasing rule: `out` may alias `x`, and every element is read before its one
// write. The strided variant moves Metal's scalar row pitch into push constants.

#include "common/bf16.slang"

PIE_BUFFER_RO(0, x);
PIE_BUFFER_RO(1, residual);
PIE_BUFFER_RW(2, out_);

#if defined(PIE_STRIDED)
struct Push { int row_pitch; };
[[vk::push_constant]] ConstantBuffer<Push> pc;
#endif

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
#if defined(PIE_STRIDED)
    uint i = gid.y * uint(pc.row_pitch) + gid.x;
#else
    uint i = gid.x;
#endif
    out_[i] = PIE_STORE(PIE_LOAD(x[i]) + PIE_LOAD(residual[i]));
}

// pie:instantiate residual_add_bfloat16
// pie:instantiate residual_add_strided_bfloat16 PIE_STRIDED=1
