#include "common/bf16.slang"

PIE_BUFFER_RO(0, x);

#if defined(PIE_STATED) || defined(PIE_SILU)
PIE_BUFFER_RW(1, out_);
struct Push { float scalar; uint n; };
#else
PIE_BUFFER_RO(1, scalar);
PIE_BUFFER_RW(2, out_);
struct Push { uint n; };
#endif
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    if (i >= pc.n) {
        return;
    }
#if defined(PIE_SILU)

    float v = PIE_LOAD(x[i]) * pc.scalar;
    out_[i] = PIE_STORE(v / (1.0 + exp(-v)));
#else
#if defined(PIE_STATED)
    float s = PIE_LOAD(PIE_STORE(pc.scalar));
#else
    float s = PIE_LOAD(scalar[0]);
#endif
    out_[i] = PIE_STORE(PIE_LOAD(x[i]) * s);
#endif
}

// pie:instantiate layer_scalar_mul_bf16
// pie:instantiate layer_scalar_mul_stated_bf16 PIE_STATED=1
// pie:instantiate silu_scaled_bf16 PIE_SILU=1
