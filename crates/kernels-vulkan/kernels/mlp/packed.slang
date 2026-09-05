#include "common/bf16.slang"

PIE_BUFFER_RO(0, packed);
PIE_BUFFER_RW(1, out_);

#if defined(PIE_SITU)
struct Push { uint intermediate; float beta; float up_cap; };
#elif defined(PIE_GPTOSS)
struct Push { uint intermediate; float limit; float alpha; };
#elif defined(PIE_CLAMP)
struct Push { uint intermediate; float limit; };
#else
struct Push { uint intermediate; };
#endif
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    uint intermediate = pc.intermediate;
    if (i >= intermediate) {
        return;
    }
    uint row = gid.y * intermediate;
    uint packed_row = row * 2u;
    float g = PIE_LOAD(packed[packed_row + i]);
    float u = PIE_LOAD(packed[packed_row + intermediate + i]);
#if defined(PIE_SITU)

    float sg = pc.beta * tanh(g / pc.beta) / (1.0 + exp(-g));
    if (pc.up_cap > 0.0) {
        u = pc.up_cap * tanh(u / pc.up_cap);
    }
    out_[row + i] = PIE_STORE(sg * u);
#elif defined(PIE_GEGLU)
    const float k = 0.7978845608028654;
    float inner = k * (g + 0.044715 * g * g * g);
    float gelu = 0.5 * g * (1.0 + tanh(inner));
    out_[row + i] = PIE_STORE(gelu * u);
#elif defined(PIE_GPTOSS)
    g = min(g, pc.limit);
    u = clamp(u, -pc.limit, pc.limit);
    float sig = 1.0 / (1.0 + exp(-pc.alpha * g));
    out_[row + i] = PIE_STORE((g * sig) * (u + 1.0));
#elif defined(PIE_CLAMP)
    g = min(g, pc.limit);
    u = clamp(u, -pc.limit, pc.limit);
    out_[row + i] = PIE_STORE((g / (1.0 + exp(-g))) * u);
#else
    out_[row + i] = PIE_STORE((g / (1.0 + exp(-g))) * u);
#endif
}

// pie:instantiate packed_swiglu_bf16
// pie:instantiate packed_geglu_tanh_bf16 PIE_GEGLU=1
// pie:instantiate packed_swiglu_clamp_bf16 PIE_CLAMP=1
// pie:instantiate packed_gptoss_swiglu_bf16 PIE_GPTOSS=1
// pie:instantiate packed_situ_bf16 PIE_SITU=1
