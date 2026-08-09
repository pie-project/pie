// bf16, which Vulkan does not have.
//
// The activation dtype of this whole driver is bfloat16, and `GL_EXT_bfloat16`
// -- the extension that would give GLSL a `bfloat16_t` -- is new, optional, and
// missing from most of the drivers a Vulkan shell exists to reach. llama.cpp
// draws the line in exactly this place: its coopmat paths use the native type
// when `GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT` says the toolchain has it, and
// everything else stores bf16 as a 16-bit word and promotes it with a shift.
//
// This file is the second of those, and it is the only one the tree uses. The
// promotion is EXACT -- bf16 is the top half of an fp32, so widening is a shift
// and nothing else -- so the only place a choice is made is the way back down,
// where `f32_to_bf16` rounds to nearest even. That matters: truncation is what
// a naive `>> 16` does, it biases every rounded value toward zero, and over a
// 5120-wide accumulate the drift is visible against a reference.
//
// The storage type is `uint16_t` rather than `float16_t` deliberately. A
// `float16_t` array would let a driver's fp16 denormal flushing touch words
// that are not fp16 at all, and the reinterpretation is not free on every
// implementation. Sixteen bits of integer is what the bytes ARE.
//
// Index arithmetic here is 32-bit, and that is a proof rather than an
// assumption worth revisiting. `kernels-metal` widened its offsets to
// `int64_t` because a Metal buffer can be larger than 4 GiB and a `[N, D]`
// weight really does overflow an `int` at model scale. Vulkan cannot reach
// that case through a binding: `VkPhysicalDeviceLimits::maxStorageBufferRange`
// is a `uint32_t`, so the range a descriptor can expose is at most 4 GiB - 1
// and a byte index into it always fits in a `uint`. A tensor larger than that
// is not addressable by one descriptor at all, which is a binding problem for
// the shell and not an arithmetic one for the shader.
//
// The intermediates are safe too, and that is worth writing down because the
// first version of these shaders did not believe it. `kv_write` computes
// `h * k_head_stride + pos * k_seq_stride + d`, and the worry was that a
// product could overflow 32 bits on the way to a result that fits. It cannot:
// every term is unsigned, so each product is no larger than the sum it belongs
// to, and the sum is an element index into a bound range and therefore fits.
// A `uint` multiply that is exact modulo 2^32 is exact, full stop.
//
// That mattered more than tidiness. Those shaders carried `uint64_t` strides
// and so declared `OpCapability Int64`, which needs `shaderInt64` -- an
// OPTIONAL feature -- on what is supposed to be the baseline tier that every
// device can load. A validation layer found it; this driver had been building
// those pipelines anyway. The push ABI still carries 64 bits per stride
// because it is shared with `kernels-metal`, where the concern is real; the
// shaders read the low half through `PIE_STRIDE` / `PIE_LOW` below, which is
// byte-for-byte the same block with no capability attached.

#ifndef PIE_VULKAN_BF16_GLSL
#define PIE_VULKAN_BF16_GLSL

#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_control_flow_attributes : enable

/// A stride that the push ABI carries in 64 bits, declared without the `Int64`
/// capability. `uvec2` has the same size and the same 8-byte alignment as
/// `uint64_t`, so a block using it is byte-identical to one that does not.
#define PIE_STRIDE uvec2
/// The addressable half of a `PIE_STRIDE`. See the indexing note at the top of
/// this file for why the other half cannot matter to a bound descriptor.
#define PIE_LOW(s) ((s).x)

/// The exact widening: bf16 IS the high half of an fp32.
float bf16_to_f32(uint16_t v) {
    return uintBitsToFloat(uint(v) << 16);
}

/// The narrowing, rounded to nearest even.
///
/// `0x7fc0` is a quiet NaN in the top half, and the branch is there because the
/// round-to-nearest add can carry a NaN's mantissa into the exponent and turn
/// it into an infinity -- a silent one, since nothing downstream distinguishes
/// them and the model simply stops producing text.
uint16_t f32_to_bf16(float f) {
    uint bits = floatBitsToUint(f);
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u) {
        return uint16_t(0x7fc0u);
    }
    uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return uint16_t(rounded >> 16);
}

/// The dtype a buffer of activations is declared with, so a family's shader
/// says `PIE_ACT` rather than choosing. One point today (`bfloat16`), which is
/// the axis `axes::BF16` declares: a second activation dtype becomes a define,
/// not a rewrite.
#define PIE_ACT uint16_t
#define PIE_LOAD(x) bf16_to_f32(x)
#define PIE_STORE(f) f32_to_bf16(f)

/// A buffer of activations, at `binding`. Declared through a macro because
/// every family needs the same block with a different name and GLSL has no way
/// to say so -- the alternative is the same three lines in thirty files, which
/// is how two of them end up disagreeing about `readonly`.
#define PIE_BUFFER_RO(binding_, name_, field_) \
    layout(std430, binding = binding_) readonly buffer name_ { PIE_ACT field_[]; }
#define PIE_BUFFER_RW(binding_, name_, field_) \
    layout(std430, binding = binding_) buffer name_ { PIE_ACT field_[]; }

#endif  // PIE_VULKAN_BF16_GLSL
