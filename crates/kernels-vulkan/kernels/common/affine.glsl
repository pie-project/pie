// The affine codec: `value = scale * code + bias`, per group of `PIE_GROUP`
// elements, at `PIE_BITS` bits per code.
//
// The two numbers are ONE fact and neither is inferable from the tensors --
// g64/b8 and g128/b4 pack to identical shapes -- so a module compiled for the
// wrong pair does not fail, it reads the scales against the wrong weights and
// returns fluent nonsense. `quant/affine_format.hpp` on the Metal side is the
// same statement; here the pair arrives as two `-D`s and the entrypoint name
// carries both, which is what makes `_gs_64_b_4` a coordinate rather than a
// label.
//
// Packing is MLX's, because the checkpoints are: codes are little-endian within
// a 32-bit word, lowest code in the lowest bits, and a row of K elements is
// `K * PIE_BITS / 32` words. `scales` and `biases` are one bf16 each per group,
// laid out `[out_vec_size, in_vec_size / PIE_GROUP]`.
//
// What is NOT here is MLX's pre-divided dot. Metal's `qmv_fast` never unpacks:
// it scales the ACTIVATION by 1/16, 1/256, 1/4096 and multiplies the packed
// nibbles in place, which works because a Metal `bfloat` multiply of an integer
// mask is an fp32 multiply. The same trick is available in GLSL and the same
// four-element grouping applies, so `pie_affine_dot4` below is that, spelled
// with explicit masks.

#ifndef PIE_VULKAN_AFFINE_GLSL
#define PIE_VULKAN_AFFINE_GLSL

#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#ifndef PIE_GROUP
#error "an affine shader is compiled at a group size: -DPIE_GROUP=64"
#endif
#ifndef PIE_BITS
#error "an affine shader is compiled at a bit width: -DPIE_BITS=4"
#endif

#if PIE_BITS != 4 && PIE_BITS != 8
#error "the tree covers the two widths mlx_lm ships: 4 and 8"
#endif

/// Codes per 32-bit word.
#define PIE_CODES_PER_WORD (32 / PIE_BITS)
/// Words per group.
#define PIE_WORDS_PER_GROUP (PIE_GROUP / PIE_CODES_PER_WORD)
/// The mask one code occupies.
#define PIE_CODE_MASK ((1u << PIE_BITS) - 1u)

/// The `i`-th code of a packed word, as a float.
float pie_affine_code(uint word, uint i) {
    return float((word >> (i * PIE_BITS)) & PIE_CODE_MASK);
}

/// Four consecutive codes, dequantised.
vec4 pie_affine_dequant4(uint word, uint i, float scale, float bias) {
    return vec4(pie_affine_code(word, i + 0u),
                pie_affine_code(word, i + 1u),
                pie_affine_code(word, i + 2u),
                pie_affine_code(word, i + 3u)) * scale + bias;
}

/// `dot(x, dequant(word))` over the whole word.
///
/// The `sum * bias` term is the affine part factored out: every code in the
/// word shares one bias, so the bias contributes `bias * sum(x)` and does not
/// need to be added per element. That factoring is the entire reason an affine
/// GEMV is cheap, and it is exactly what MXFP4 cannot do -- see `mxfp4.glsl`,
/// whose codes are not linear.
float pie_affine_word_dot(uint word, float x[PIE_CODES_PER_WORD],
                          float scale, float bias) {
    float accum = 0.0;
    float sum = 0.0;
    [[unroll]] for (uint i = 0u; i < uint(PIE_CODES_PER_WORD); i++) {
        float xi = x[i];
        accum += xi * pie_affine_code(word, i);
        sum += xi;
    }
    return scale * accum + sum * bias;
}

/// Which group an element index falls in, and where its scale lives.
uint pie_affine_group_of(uint k) {
    return k / uint(PIE_GROUP);
}

#endif  // PIE_VULKAN_AFFINE_GLSL
