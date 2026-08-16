// Quantized tied-embedding gather.
//
// This is Metal's affine-dequant gather moved to Vulkan's binding rule: table
// buffers remain descriptors, while `hidden` and the optional embedding scale
// are push constants. The affine axes are compile-time facts (`PIE_GROUP`,
// `PIE_BITS`) so a `_gs_64_b_4` module cannot be accidentally reused for the
// identical-looking `_gs_128_b_8` storage.

#include "common/bf16.slang"
#include "common/affine.slang"

[[vk::binding(0)]] StructuredBuffer<uint> w;
PIE_BUFFER_RO(1, scales);
PIE_BUFFER_RO(2, biases);
[[vk::binding(3)]] StructuredBuffer<int> id;
PIE_BUFFER_RW(4, out_);

#if defined(PIE_SCALED)
struct Push { int hidden; float embed_scale; };
[[vk::push_constant]] ConstantBuffer<Push> pc;
#else
struct Push { int hidden; };
[[vk::push_constant]] ConstantBuffer<Push> pc;
#endif

#if defined(PIE_MB)
[numthreads(16, 16, 1)]
#else
[numthreads(256, 1, 1)]
#endif
void main(uint3 gid : SV_DispatchThreadID) {
    uint k = gid.x;
#if defined(PIE_MB)
    uint m = gid.y;
#else
    uint m = 0u;
#endif
    uint hidden = uint(pc.hidden);
    if (k >= hidden) return;

    uint row = uint(id[m]);
    uint packs_per_row = hidden / uint(PIE_CODES_PER_WORD);
    uint groups_per_row = hidden / uint(PIE_GROUP);
    uint word = w[row * packs_per_row + k / uint(PIE_CODES_PER_WORD)];
    uint group = k / uint(PIE_GROUP);
    float s = PIE_LOAD(scales[row * groups_per_row + group]);
    float b = PIE_LOAD(biases[row * groups_per_row + group]);
    float value = s * pie_affine_code(word, k % uint(PIE_CODES_PER_WORD)) + b;
#if defined(PIE_SCALED)
    value *= pc.embed_scale;
#endif
    out_[m * hidden + k] = PIE_STORE(value);
}

// pie:instantiate embed_gather_4bit_bfloat16_gs_128_b_4 PIE_GROUP=128 PIE_BITS=4
// pie:instantiate embed_gather_4bit_bfloat16_gs_128_b_8 PIE_GROUP=128 PIE_BITS=8
// pie:instantiate embed_gather_4bit_bfloat16_gs_32_b_4 PIE_GROUP=32 PIE_BITS=4
// pie:instantiate embed_gather_4bit_bfloat16_gs_32_b_8 PIE_GROUP=32 PIE_BITS=8
// pie:instantiate embed_gather_4bit_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4
// pie:instantiate embed_gather_4bit_bfloat16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8
// pie:instantiate embed_gather_mb_4bit_bfloat16_gs_128_b_4 PIE_GROUP=128 PIE_BITS=4 PIE_MB=1
// pie:instantiate embed_gather_mb_4bit_bfloat16_gs_128_b_8 PIE_GROUP=128 PIE_BITS=8 PIE_MB=1
// pie:instantiate embed_gather_mb_4bit_bfloat16_gs_32_b_4 PIE_GROUP=32 PIE_BITS=4 PIE_MB=1
// pie:instantiate embed_gather_mb_4bit_bfloat16_gs_32_b_8 PIE_GROUP=32 PIE_BITS=8 PIE_MB=1
// pie:instantiate embed_gather_mb_4bit_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_MB=1
// pie:instantiate embed_gather_mb_4bit_bfloat16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8 PIE_MB=1
// pie:instantiate embed_gather_scaled_4bit_bfloat16_gs_128_b_4 PIE_GROUP=128 PIE_BITS=4 PIE_SCALED=1
// pie:instantiate embed_gather_scaled_4bit_bfloat16_gs_128_b_8 PIE_GROUP=128 PIE_BITS=8 PIE_SCALED=1
// pie:instantiate embed_gather_scaled_4bit_bfloat16_gs_32_b_4 PIE_GROUP=32 PIE_BITS=4 PIE_SCALED=1
// pie:instantiate embed_gather_scaled_4bit_bfloat16_gs_32_b_8 PIE_GROUP=32 PIE_BITS=8 PIE_SCALED=1
// pie:instantiate embed_gather_scaled_4bit_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_SCALED=1
// pie:instantiate embed_gather_scaled_4bit_bfloat16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8 PIE_SCALED=1
// pie:instantiate embed_gather_scaled_mb_4bit_bfloat16_gs_128_b_4 PIE_GROUP=128 PIE_BITS=4 PIE_MB=1 PIE_SCALED=1
// pie:instantiate embed_gather_scaled_mb_4bit_bfloat16_gs_128_b_8 PIE_GROUP=128 PIE_BITS=8 PIE_MB=1 PIE_SCALED=1
// pie:instantiate embed_gather_scaled_mb_4bit_bfloat16_gs_32_b_4 PIE_GROUP=32 PIE_BITS=4 PIE_MB=1 PIE_SCALED=1
// pie:instantiate embed_gather_scaled_mb_4bit_bfloat16_gs_32_b_8 PIE_GROUP=32 PIE_BITS=8 PIE_MB=1 PIE_SCALED=1
// pie:instantiate embed_gather_scaled_mb_4bit_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_MB=1 PIE_SCALED=1
// pie:instantiate embed_gather_scaled_mb_4bit_bfloat16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8 PIE_MB=1 PIE_SCALED=1
