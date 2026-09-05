#ifndef PIE_VULKAN_BF16_SLANG
#define PIE_VULKAN_BF16_SLANG

float bf16_to_f32(uint16_t v) {
    return asfloat(uint(v) << 16);
}

uint16_t f32_to_bf16(float f) {
    uint bits = asuint(f);
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u) {
        return uint16_t(0x7fc0u);
    }
    uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return uint16_t(rounded >> 16);
}

#define PIE_ACT uint16_t
#define PIE_LOAD(x) bf16_to_f32(x)
#define PIE_STORE(f) f32_to_bf16(f)

#define PIE_STRIDE uint2

#define PIE_LOW(s) ((s).x)

#define PIE_BUFFER_RO(binding_, field_) \
    [[vk::binding(binding_)]] StructuredBuffer<PIE_ACT> field_
#define PIE_BUFFER_RW(binding_, field_) \
    [[vk::binding(binding_)]] RWStructuredBuffer<PIE_ACT> field_

#define PIE_PARAMS(binding_, type_, field_) \
    layout(std430, binding = binding_) readonly buffer type_##Block { type_ field_; }

#endif
