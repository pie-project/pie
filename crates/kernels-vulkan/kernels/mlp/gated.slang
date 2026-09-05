#include "common/bf16.slang"

#if defined(PIE_GELU)
PIE_BUFFER_RO(0, x);
PIE_BUFFER_RW(1, out_);
struct Push { uint n; };
#elif defined(PIE_SWIGLU_CLAMP)
PIE_BUFFER_RO(0, gate);
PIE_BUFFER_RO(1, up);
PIE_BUFFER_RW(2, out_);
struct Push { float limit; uint n; };
#else
PIE_BUFFER_RO(0, gate);
PIE_BUFFER_RO(1, up);
PIE_BUFFER_RW(2, out_);
struct Push { uint n; };
#endif
[[vk::push_constant]] ConstantBuffer<Push> pc;

float gelu_tanh(float x) {
    const float k = 0.7978845608028654;
    float inner = k * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    if (i >= pc.n) {
        return;
    }
#if defined(PIE_GELU)
    out_[i] = PIE_STORE(gelu_tanh(PIE_LOAD(x[i])));
#elif defined(PIE_SWIGLU_CLAMP)
    float g = min(PIE_LOAD(gate[i]), pc.limit);
    float u = clamp(PIE_LOAD(up[i]), -pc.limit, pc.limit);
    out_[i] = PIE_STORE((g / (1.0 + exp(-g))) * u);
#else
    out_[i] = PIE_STORE(gelu_tanh(PIE_LOAD(gate[i])) * PIE_LOAD(up[i]));
#endif
}

// pie:instantiate geglu_tanh_bf16
// pie:instantiate swiglu_clamp_bf16 PIE_SWIGLU_CLAMP=1
// pie:instantiate mlp_gelu_tanh_bf16 PIE_GELU=1
