const PIE_CODES_PER_WORD = 32 / PIE_BITS;

const PIE_WORDS_PER_GROUP = PIE_GROUP / PIE_CODES_PER_WORD;

const PIE_CODE_MASK: u32 = (1u << u32(PIE_BITS)) - 1u;

fn pie_affine_code(word: u32, i: u32) -> f32 {
    return f32((word >> (i * u32(PIE_BITS))) & PIE_CODE_MASK);
}

fn pie_affine_value(word: u32, i: u32, scale: f32, bias: f32) -> f32 {
    return scale * pie_affine_code(word, i) + bias;
}

fn pie_affine_dequant4(word: u32, i: u32, scale: f32, bias: f32) -> vec4<f32> {
    return vec4<f32>(
        pie_affine_code(word, i + 0u),
        pie_affine_code(word, i + 1u),
        pie_affine_code(word, i + 2u),
        pie_affine_code(word, i + 3u),
    ) * scale + bias;
}

fn pie_affine_word_dot(
    word: u32,
    x: array<f32, PIE_CODES_PER_WORD>,
    scale: f32,
    bias: f32,
) -> f32 {
    var accum = 0.0;
    var sum = 0.0;
    var xs = x;
    for (var i = 0u; i < u32(PIE_CODES_PER_WORD); i = i + 1u) {
        let xi = xs[i];
        accum = accum + xi * pie_affine_code(word, i);
        sum = sum + xi;
    }
    return scale * accum + sum * bias;
}

fn pie_affine_word_of(row: u32, row_len: u32, k: u32) -> u32 {
    return row * (row_len / u32(PIE_CODES_PER_WORD)) + k / u32(PIE_CODES_PER_WORD);
}

fn pie_affine_code_of(k: u32) -> u32 {
    return k % u32(PIE_CODES_PER_WORD);
}

fn pie_affine_scale_of(row: u32, row_len: u32, k: u32) -> u32 {
    return row * (row_len / u32(PIE_GROUP)) + k / u32(PIE_GROUP);
}
