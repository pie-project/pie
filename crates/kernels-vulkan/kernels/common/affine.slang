#ifndef PIE_VULKAN_AFFINE_SLANG
#define PIE_VULKAN_AFFINE_SLANG

#ifndef PIE_GROUP
#error "an affine shader is compiled at a group size: -DPIE_GROUP=64"
#endif
#ifndef PIE_BITS
#error "an affine shader is compiled at a bit width: -DPIE_BITS=4"
#endif

#if PIE_BITS != 2 && PIE_BITS != 4 && PIE_BITS != 8
#error "the tree covers the three widths the catalog ships: 2, 4 and 8"
#endif

#define PIE_CODES_PER_WORD (32 / PIE_BITS)

#define PIE_WORDS_PER_GROUP (PIE_GROUP / PIE_CODES_PER_WORD)

#define PIE_CODE_MASK ((1u << PIE_BITS) - 1u)

float pie_affine_code(uint word, uint i) {
    return float((word >> (i * PIE_BITS)) & PIE_CODE_MASK);
}

float4 pie_affine_dequant4(uint word, uint i, float scale, float bias) {
    return float4(pie_affine_code(word, i + 0u),
                pie_affine_code(word, i + 1u),
                pie_affine_code(word, i + 2u),
                pie_affine_code(word, i + 3u)) * scale + bias;
}

float pie_affine_word_dot_presummed(uint word, float x[PIE_CODES_PER_WORD],
                                    float xsum, float scale, float bias) {
    float accum = 0.0;
    [ForceUnroll] for (uint i = 0u; i < uint(PIE_CODES_PER_WORD); i++) {
        accum += x[i] * pie_affine_code(word, i);
    }
    return scale * accum + xsum * bias;
}

float pie_affine_word_dot(uint word, float x[PIE_CODES_PER_WORD],
                          float scale, float bias) {
    float sum = 0.0;
    [ForceUnroll] for (uint i = 0u; i < uint(PIE_CODES_PER_WORD); i++) {
        sum += x[i];
    }
    return pie_affine_word_dot_presummed(word, x, sum, scale, bias);
}

uint pie_affine_group_of(uint k) {
    return k / uint(PIE_GROUP);
}

#endif
