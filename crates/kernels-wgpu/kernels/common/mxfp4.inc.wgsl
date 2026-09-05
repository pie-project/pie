var<private> pie_mxfp4_lut: array<f32, 16> = array<f32, 16>(
     0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
);

fn pie_mxfp4_value(code: u32) -> f32 {
    return pie_mxfp4_lut[code & 0xfu];
}

fn pie_mxfp4_byte(word: u32, i: u32) -> u32 {
    return (word >> ((i & 3u) * 8u)) & 0xffu;
}

fn pie_mxfp4_lo(byte_: u32) -> f32 {
    return pie_mxfp4_value(byte_);
}

fn pie_mxfp4_hi(byte_: u32) -> f32 {
    return pie_mxfp4_value(byte_ >> 4u);
}

fn pie_mxfp4_code(word: u32, i: u32) -> f32 {
    return pie_mxfp4_value(word >> ((i & 7u) * 4u));
}

fn pie_mxfp4_block_scale(code: u32) -> f32 {
    if (code == 0xffu) {
        return bitcast<f32>(0x7fc00000u);
    }
    return ldexp(1.0, i32(code) - 127);
}

const PIE_MXFP4_BLOCK = 32u;
