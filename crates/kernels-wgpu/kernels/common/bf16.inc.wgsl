// bf16 in a language with no 16-bit storage type.
//
// WGSL's smallest addressable element in a storage buffer is FOUR BYTES. There
// is no `u16` and there is no `bf16`, and `enable f16` -- when an adapter even
// offers it -- adds an f16 *arithmetic* type, not an f16 storage one. So a bf16
// tensor crosses as `array<u32>`, two values to a word, low half first.
//
// That is a real divergence from `kernels-vulkan`, which has
// `GL_EXT_shader_16bit_storage` and declares `uint16_t[]` directly. Every
// address here is therefore a half-index that has to be split, and the split is
// in ONE place -- these four functions -- rather than open-coded per kernel.

// Widen one bf16 bit pattern to f32. Exact: bf16 IS the top half of an f32.
fn pie_bf16_to_f32(v: u32) -> f32 {
    return bitcast<f32>(v << 16u);
}

// Narrow an f32 to a bf16 bit pattern, round to nearest even.
//
// Truncating instead of rounding is a real accuracy loss over a long
// accumulation -- `kernels-vulkan` says so in its own copy of this and it is
// just as true here. The NaN branch is explicit because the rounding add can
// carry a NaN's mantissa to zero and turn it into an infinity.
fn pie_f32_to_bf16(x: f32) -> u32 {
    let bits = bitcast<u32>(x);
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        return 0x7fc0u;
    }
    let rounded = bits + 0x7fffu + ((bits >> 16u) & 1u);
    return rounded >> 16u;
}

// The `i`-th bf16 of a word ALREADY LOADED, low half first.
//
// A word and an index, not a pointer and an index. Core WGSL allows a pointer
// parameter only in the `function`, `private` and `workgroup` address spaces:
// `ptr<storage, ...>` needs the `unrestricted_pointer_parameters` language
// extension, which naga refuses and which no WebGPU implementation is obliged
// to have. The first draft of this file took the pointer, and 478 of the 480
// modules failed `naga::valid::Validator` on it -- while PARSING fine, which is
// why `every_module_validates` exists and why a parse-only check was not
// enough.
//
// The cost is that the caller writes `pie_bf16_at(x[i >> 1u], i)` and so states
// the half-index split at the load. That is worth something on its own: the
// split is the divergence from `kernels-vulkan`, which declares `uint16_t[]`
// and indexes it directly, and having it visible at every load is how a reader
// sees that this tree is addressing PAIRS.
fn pie_bf16_at(word: u32, i: u32) -> f32 {
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

// One half of a word replaced, for a caller that owns the whole word.
//
// Returns the new word rather than writing it, for the same pointer reason —
// and the shape is better anyway. `out_[at] = pie_bf16_into(out_[at], i, v)` is
// a read-modify-write the caller can SEE, so the question "does another
// invocation own the other half of this word?" is asked where it can be
// answered. WGSL has no sub-word atomic, so a body that cannot answer it must
// either write both halves at once with `pie_pack_bf16` or use an
// `atomicCompareExchangeWeak` loop over an `array<atomic<u32>>`.
fn pie_bf16_into(word: u32, i: u32, x: f32) -> u32 {
    let v = pie_f32_to_bf16(x);
    if ((i & 1u) == 1u) {
        return (word & 0x0000ffffu) | (v << 16u);
    }
    return (word & 0xffff0000u) | v;
}

// Both halves of one word at once: the store a body should prefer.
fn pie_pack_bf16(lo: f32, hi: f32) -> u32 {
    return pie_f32_to_bf16(lo) | (pie_f32_to_bf16(hi) << 16u);
}
