// bf16, which Vulkan does not have. The Slang half of `common/bf16.glsl`.
//
// Everything the GLSL header says about the ARITHMETIC still holds and is not
// repeated here: the promotion is a shift because bf16 is the top half of an
// fp32, the narrowing rounds to nearest even because truncation biases a
// 5120-wide accumulate toward zero, the storage type is an integer because
// sixteen bits of integer is what the bytes ARE, and index arithmetic is 32-bit
// because `maxStorageBufferRange` is a `uint32_t` and so a byte index into a
// descriptor always fits. Read that file for the reasoning; this one records
// only what the change of language forced.
//
// ## What the change of language forced
//
// Three things, and no more than three.
//
// The bit casts are spelled `asfloat`/`asuint` rather than
// `uintBitsToFloat`/`floatBitsToUint`. Same operation, same `OpBitcast`.
//
// The buffer declaration stops being a macro. GLSL has no way to say "a
// storage buffer of `PIE_ACT` at binding n" other than by pasting the block, so
// `common/bf16.glsl` has `PIE_BUFFER_RO`/`PIE_BUFFER_RW` and every family
// invokes them. Slang has generics and a real `StructuredBuffer`, so the
// declaration is just a declaration and the macro has nothing left to do. It is
// kept as an alias anyway -- see below -- because a port that changes the
// spelling of every buffer in thirty files at the same time as it changes the
// language is a port whose first failure cannot be attributed.
//
// The read/write split is now in the TYPE. `StructuredBuffer<T>` is read-only
// and `RWStructuredBuffer<T>` is not, where GLSL needed a `readonly` qualifier
// that it was possible to forget. That is a real improvement and it is also a
// change in the SPIR-V: Slang decorates a `StructuredBuffer` `NonWritable`,
// which `glslc` only did where the GLSL said `readonly`. Nothing in this driver
// reads that decoration to decide anything -- the write mask a barrier is built
// from comes from the kernel table, which is the only place the distinction is
// stated for the launcher -- so it is a tightening of the module's declaration
// and not a change of behaviour. `Device::run_all`'s hole-skipping comment is
// where the one thing that DOES read decorations is written down.

#ifndef PIE_VULKAN_BF16_SLANG
#define PIE_VULKAN_BF16_SLANG

/// The exact widening: bf16 IS the high half of an fp32.
float bf16_to_f32(uint16_t v) {
    return asfloat(uint(v) << 16);
}

/// The narrowing, rounded to nearest even.
///
/// `0x7fc0` is a quiet NaN in the top half, and the branch is there because the
/// round-to-nearest add can carry a NaN's mantissa into the exponent and turn
/// it into an infinity -- a silent one, since nothing downstream distinguishes
/// them and the model simply stops producing text.
uint16_t f32_to_bf16(float f) {
    uint bits = asuint(f);
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

/// A stride that the push ABI carries in 64 bits, declared without the `Int64`
/// capability. `uint2` has the same size and the same 8-byte alignment as
/// `uint64_t`, so a block using it is byte-identical to one that does not.
#define PIE_STRIDE uint2
/// The addressable half of a `PIE_STRIDE`.
#define PIE_LOW(s) ((s).x)

/// A buffer of activations, at `binding`.
///
/// A macro still, but a thin one: it exists so the port can be read as a change
/// of language rather than a rewrite of every declaration site. What it expands
/// to is one ordinary Slang global, which is why there is no longer a `name_`
/// parameter -- GLSL needed a block name and Slang does not.
#define PIE_BUFFER_RO(binding_, field_) \
    [[vk::binding(binding_)]] StructuredBuffer<PIE_ACT> field_
#define PIE_BUFFER_RW(binding_, field_) \
    [[vk::binding(binding_)]] RWStructuredBuffer<PIE_ACT> field_

/// The scalar parameter block, at `binding`, holding one `type_` under the
/// name `field_`.
///
/// This one is NOT idiomatic Slang and cannot be, which is the single place the
/// port had to keep GLSL syntax. The driver locates a launch's parameter block
/// by its SIZE -- `binding::params` searches the reflected block extents for
/// one of exactly `4 * scalars` bytes, because two kernels put their block at a
/// binding with operands after it and position is therefore not enough. So the
/// module has to DECLARE a fixed extent.
///
/// `StructuredBuffer<T>` cannot. It is a runtime array: how many `T` are there
/// is the descriptor range's answer at the call, not the module's, so
/// reflection reports no size -- correctly -- and every launch is refused with
/// "2 scalars stated, room for 0". That is not a bug in the reflection; sizing
/// a runtime array from its `ArrayStride` would make a plain
/// `StructuredBuffer<uint>` tensor operand claim to be a 4-byte parameter block
/// and be picked by a one-scalar launch.
///
/// A GLSL-style `buffer` block states the extent, Slang accepts it under
/// `-allow-glsl` as a first-class input, and it emits exactly the
/// `OpTypeStruct` in `StorageBuffer` that `glslc` emitted. `readonly` is kept
/// because these are read-only in fact and it costs nothing to say so.
#define PIE_PARAMS(binding_, type_, field_) \
    layout(std430, binding = binding_) readonly buffer type_##Block { type_ field_; }

#endif  // PIE_VULKAN_BF16_SLANG
