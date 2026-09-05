fn pie_bf16_to_f32(v: u32) -> f32 {
    return bitcast<f32>(v << 16u);
}

fn pie_f32_to_bf16(x: f32) -> u32 {
    let bits = bitcast<u32>(x);
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        return 0x7fc0u;
    }
    let rounded = bits + 0x7fffu + ((bits >> 16u) & 1u);
    return rounded >> 16u;
}

fn pie_bf16_at(word: u32, i: u32) -> f32 {
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

fn pie_bf16_into(word: u32, i: u32, x: f32) -> u32 {
    let v = pie_f32_to_bf16(x);
    if ((i & 1u) == 1u) {
        return (word & 0x0000ffffu) | (v << 16u);
    }
    return (word & 0xffff0000u) | v;
}

fn pie_pack_bf16(lo: f32, hi: f32) -> u32 {
    return pie_f32_to_bf16(lo) | (pie_f32_to_bf16(hi) << 16u);
}
