#[must_use]
pub const fn ceil_div_u32(x: u32, y: u32) -> u32 {
    x.wrapping_add(y).wrapping_sub(1) / y
}

#[must_use]
pub const fn ceil_div_i32_in_u32(x: i32, y: u32) -> i32 {
    ((x as u32).wrapping_add(y).wrapping_sub(1) / y) as i32
}

#[must_use]
pub const fn ceil_div_i64(x: i64, y: i64) -> i64 {
    x.wrapping_add(y).wrapping_sub(1) / y
}

#[must_use]
pub const fn ceil_div_i32(x: i32, y: i32) -> i32 {
    x.wrapping_add(y).wrapping_sub(1) / y
}

#[must_use]
pub const fn fa2_determine_cta_tile_q(avg_packed_qo_len: i64, head_dim: u32, cc_major: i32) -> u32 {
    if head_dim >= 512 {
        if avg_packed_qo_len <= 32 {
            return 16;
        }
        return 32;
    }
    if avg_packed_qo_len > 64 && head_dim < 256 {
        128
    } else if cc_major >= 8 {
        if avg_packed_qo_len > 16 { 64 } else { 16 }
    } else {
        64
    }
}

#[must_use]
pub fn cost_function(qo_len: i32, kv_len: i32) -> f32 {
    2.0 * (qo_len as f32) + (kv_len as f32)
}

#[must_use]
pub const fn packed_causal_kv_end(
    qo_len: i32,
    kv_len: i32,
    qo_tile_idx: i32,
    cluster_tile_q: i32,
    num_qo_tiles: i32,
    group_size: i32,
) -> i32 {
    if qo_tile_idx + 1 == num_qo_tiles {
        return kv_len;
    }
    let kv_len_init = kv_len - qo_len;
    let end = kv_len_init.wrapping_add(ceil_div_i32(
        (qo_tile_idx + 1).wrapping_mul(cluster_tile_q),
        group_size,
    ));
    let clamped = if end < kv_len { end } else { kv_len };
    if clamped > 0 { clamped } else { 0 }
}

#[must_use]
pub const fn cuda_max_u32_i32(a: u32, b: i32) -> u32 {
    let b = b as u32;
    if a > b { a } else { b }
}
