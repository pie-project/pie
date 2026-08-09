// The affine codec: `value = scale * code + bias`, per group of `PIE_GROUP`
// elements, at `PIE_BITS` bits per code.
//
// The two numbers are ONE fact and neither is inferable from the tensors --
// g64/b8 and g128/b4 pack to identical shapes -- so a module compiled for the
// wrong pair does not fail, it reads the scales against the wrong weights and
// returns fluent nonsense. `quant/affine_format.hpp` on the Metal side is the
// same statement; here the pair arrives as two defines and the entrypoint name
// carries both, which is what makes `_gs_64_b_4` a coordinate rather than a
// label.
//
// Packing is MLX's, because the checkpoints are: codes are little-endian within
// a 32-bit word, lowest code in the lowest bits, and a row of K elements is
// `K * PIE_BITS / 32` words. `scales` and `biases` are one bf16 each per group,
// laid out `[rows, K / PIE_GROUP]` -- which in this backend means a HALF-index
// into an `array<u32>`, because WGSL has no 16-bit storage at all. See
// `common/bf16.inc.wgsl`.
//
// ## Why this fragment takes words and not buffers
//
// The obvious API would be `pie_affine_dequant(w, scales, biases, ...)` with
// three `ptr<storage, array<u32>, read>` parameters, which is how the same file
// would be written in any language with pointers. **naga rejects it.** Its
// validator allows a pointer argument only in the `private` and `function`
// address spaces -- `unrestricted_pointer_parameters` is a WGSL language
// extension naga 30 lists as UNIMPLEMENTED (gfx-rs/wgpu#5158) -- and the
// failure arrives from `create_shader_module` as
//
//   Argument 'w' at index 0 is a pointer of space Storage { access: LOAD },
//   which can't be passed into functions.
//
// So the triple crosses as the three VALUES a caller has already loaded, plus
// the index helpers below that say WHERE to load them from. That keeps the
// addressing -- the part a caller gets wrong -- in one place, which is the
// whole reason the fragment exists, and costs the caller three subscripts it
// would otherwise have written anyway.
//
// What is NOT here is MLX's pre-divided dot. Metal's `qmv_fast` never unpacks:
// it scales the ACTIVATION by 1/16, 1/256, 1/4096 and multiplies the packed
// nibbles in place. `pie_affine_word_dot` below is the same factoring spelled
// with explicit masks, which is what a GEMV wants; a gather wants
// `pie_affine_value` and one code at a time.

// `PIE_GROUP` and `PIE_BITS` are the caller's, and there is deliberately no
// guard for their absence. The expander has no `//#error` -- and needs none: a
// variant that forgot them declares no `const PIE_GROUP`, so the first line
// below fails to compile with "unknown identifier", which names the missing
// define at the line that wanted it.

// Codes per 32-bit word: 8 at four bits, 4 at eight.
const PIE_CODES_PER_WORD = 32 / PIE_BITS;
// Words per group. A group never straddles a word, at either width.
const PIE_WORDS_PER_GROUP = PIE_GROUP / PIE_CODES_PER_WORD;
// The mask one code occupies.
const PIE_CODE_MASK: u32 = (1u << u32(PIE_BITS)) - 1u;

// The `i`-th code of a packed word, as a float.
fn pie_affine_code(word: u32, i: u32) -> f32 {
    return f32((word >> (i * u32(PIE_BITS))) & PIE_CODE_MASK);
}

// One dequantised element: the whole codec, given the three values a caller
// loaded from the `w`/`scales`/`biases` triple through the index helpers below.
fn pie_affine_value(word: u32, i: u32, scale: f32, bias: f32) -> f32 {
    return scale * pie_affine_code(word, i) + bias;
}

// Four consecutive codes, dequantised. `i + 3` must be in the same word, which
// is what makes four the number: eight codes fit at four bits and four at
// eight, so a quarter-word step is the widest one both widths allow.
fn pie_affine_dequant4(word: u32, i: u32, scale: f32, bias: f32) -> vec4<f32> {
    return vec4<f32>(
        pie_affine_code(word, i + 0u),
        pie_affine_code(word, i + 1u),
        pie_affine_code(word, i + 2u),
        pie_affine_code(word, i + 3u),
    ) * scale + bias;
}

// `dot(x, dequant(word))` over the whole word.
//
// The `sum * bias` term is the affine part factored out: every code in the word
// shares one bias, so the bias contributes `bias * sum(x)` and does not need to
// be added per element. That factoring is the entire reason an affine GEMV is
// cheap, and it is exactly what MXFP4 cannot do -- its codes are not linear.
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

// ── Where the three tensors keep element `k` of row `row` ────────────────────
//
// `row_len` is the row's length in ELEMENTS (`hidden` for an embedding table,
// `in_vec_size` for a projection). Both quotients below are exact: a row is a
// whole number of words and a whole number of groups, or the checkpoint would
// not pack. That is also why an odd `row_len` cannot occur, which the bf16
// bodies rely on when they write a whole word at a time.

// Index into `w` (an `array<u32>` of packed codes).
fn pie_affine_word_of(row: u32, row_len: u32, k: u32) -> u32 {
    return row * (row_len / u32(PIE_CODES_PER_WORD)) + k / u32(PIE_CODES_PER_WORD);
}

// Which code of that word element `k` is.
fn pie_affine_code_of(k: u32) -> u32 {
    return k % u32(PIE_CODES_PER_WORD);
}

// Index into `scales` and `biases` -- a HALF-index, because both are bf16 and
// this backend stores bf16 two to a `u32`. A caller reads it as
// `pie_bf16_to_f32(select(w & 0xffffu, w >> 16u, (i & 1u) == 1u))` over
// `scales[i >> 1u]`; it cannot call `pie_load_bf16`, for the pointer reason
// this file's header gives.
fn pie_affine_scale_of(row: u32, row_len: u32, k: u32) -> u32 {
    return row * (row_len / u32(PIE_GROUP)) + k / u32(PIE_GROUP);
}
