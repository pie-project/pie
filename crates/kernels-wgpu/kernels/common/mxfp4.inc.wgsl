// MXFP4, in the one place every kernel that reads it can see.
//
// E2M1: one sign bit, two exponent bits, one mantissa bit, with the all-zero
// exponent giving the +/-0.5 denormals. The values are NOT linear in the code --
// they step by 0.5 up to 2 and then by 1, 2, 2 -- which is the whole reason
// MXFP4 cannot borrow the affine dot, whose trick is that `scale * code + bias`
// is linear in the code and so factors the bias out of the inner product.
//
// E8M0: an unsigned power of two, 127-biased, one per 32-element block, with
// 0xff reserved for NaN.
//
// Both siblings say the same three things (`quant/mxfp4_codec.h`,
// `common/mxfp4.glsl`) and say why they have to live in one file: the decode
// path would be right on a prefill and wrong on a decode, or the other way
// round, and the output would still look like text.
//
// ## What is different here, and it is not the arithmetic
//
// A payload is `device const uchar*` on Metal and `uint8_t[]` on Vulkan behind
// `GL_EXT_shader_8bit_storage`. **WGSL has no `u8` and no 8-bit storage at
// all** -- four bytes is the smallest element a storage buffer can hold -- so
// an MXFP4 code plane and an E8M0 exponent plane both cross as `array<u32>`,
// four bytes to a word, LOWEST byte first. Every byte index in the two
// siblings' bodies is therefore an index that has to be split, and it is split
// HERE rather than open-coded per kernel: two readers of one plane that
// disagree about byte order produce a transposed weight, which is wrong by a
// plausible amount and looks like a bad fine-tune rather than like a bug.
//
// The split is stated as the two shifts a caller does at the subscript:
//
//     byte `i`  ->  pie_mxfp4_byte(plane[i >> 2u], i)     // four bytes to a word
//     code `i`  ->  pie_mxfp4_code(plane[i >> 3u], i)     // eight codes to a word
//
// and NOT as a function taking the plane, because naga 30 refuses a
// `ptr<storage, ...>` function parameter outright ("Argument is a pointer of
// space Storage, which can't be passed into functions"). A module that took
// one would parse and then fail `create_shader_module` on every device.

// The sixteen E2M1 values, by code. The sign is the high bit, so the second
// half mirrors the first.
//
// `var<private>` and not `const`: a `const` array is a VALUE, and indexing a
// value by a code only known at runtime is the case WGSL leaves to the
// implementation to materialise. A private array is a memory location, which is
// dynamically indexable by construction, and it is sixteen floats.
var<private> pie_mxfp4_lut: array<f32, 16> = array<f32, 16>(
     0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
);

// One value from a 4-bit code.
fn pie_mxfp4_value(code: u32) -> f32 {
    return pie_mxfp4_lut[code & 0xfu];
}

// Byte `i` of the word that holds it -- the caller subscripts `plane[i >> 2u]`.
fn pie_mxfp4_byte(word: u32, i: u32) -> u32 {
    return (word >> ((i & 3u) * 8u)) & 0xffu;
}

// The two codes in one byte, low nibble first. Swapping them transposes every
// adjacent pair of weights.
fn pie_mxfp4_lo(byte_: u32) -> f32 {
    return pie_mxfp4_value(byte_);
}

fn pie_mxfp4_hi(byte_: u32) -> f32 {
    return pie_mxfp4_value(byte_ >> 4u);
}

// Code `i` of the word that holds it -- the caller subscripts `plane[i >> 3u]`.
//
// Eight codes to a word, low nibble first, which is the same ordering the two
// halves of a byte have: nibble `i & 7` of the word IS nibble `i & 1` of byte
// `(i >> 1) & 3`. A reader that has the byte already should say `pie_mxfp4_lo`
// / `pie_mxfp4_hi`; a reader walking elements should say this.
fn pie_mxfp4_code(word: u32, i: u32) -> f32 {
    return pie_mxfp4_value(word >> ((i & 7u) * 4u));
}

// One block's E8M0 exponent as a multiplier. 0xff is the NaN encoding; every
// other code is an exact power of two.
//
// `ldexp` is the spelling rather than `pow(2.0, e)`: the exponent is an integer
// and `pow` is neither exact nor cheap for one.
fn pie_mxfp4_block_scale(code: u32) -> f32 {
    if (code == 0xffu) {
        return bitcast<f32>(0x7fc00000u);  // quiet NaN
    }
    return ldexp(1.0, i32(code) - 127);
}

// Elements per E8M0 block. Fixed by the FORMAT and not by a deployment's
// choice, which is why the routed MXFP4 rows are compiled at `_gs_32` alone.
const PIE_MXFP4_BLOCK = 32u;
