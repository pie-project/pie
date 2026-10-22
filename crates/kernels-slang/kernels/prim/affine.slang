#pragma once

// Affine-quantized banks: `PIE_BITS`-bit codes packed low first into 32-bit
// words, a bf16 scale and bias per `PIE_GROUP` codes of a row; a value is
// `scale * code + bias`.

static const uint PIE_CODES_PER_WORD = 32u / uint(PIE_BITS);
static const uint PIE_CODE_MASK = (1u << uint(PIE_BITS)) - 1u;

float affine_code(uint word, uint i) {
    return float((word >> (i * uint(PIE_BITS))) & PIE_CODE_MASK);
}

float affine_value(uint word, uint i, float scale, float bias) {
    return scale * affine_code(word, i) + bias;
}

// The word holding code `k` of `row`, a row `row_len` codes long.
uint affine_word_of(uint row, uint row_len, uint k) {
    return row * (row_len / PIE_CODES_PER_WORD) + k / PIE_CODES_PER_WORD;
}

// Code `k`'s place within its word.
uint affine_code_of(uint k) {
    return k % PIE_CODES_PER_WORD;
}

// The scale (and bias) index of code `k` of `row`.
uint affine_scale_of(uint row, uint row_len, uint k) {
    return row * (row_len / uint(PIE_GROUP)) + k / uint(PIE_GROUP);
}
