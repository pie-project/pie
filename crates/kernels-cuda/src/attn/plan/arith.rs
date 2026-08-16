/// `ceil_div<uint32_t, uint32_t>` — the plain unsigned case.
#[must_use]
pub const fn ceil_div_u32(x: u32, y: u32) -> u32 {
    x.wrapping_add(y).wrapping_sub(1) / y
}

/// `ceil_div<int32_t, uint32_t>` — computed in `unsigned`, truncated back.
#[must_use]
pub const fn ceil_div_i32_in_u32(x: i32, y: u32) -> i32 {
    ((x as u32).wrapping_add(y).wrapping_sub(1) / y) as i32
}

/// `ceil_div<int64_t, ...>` — the widened case.
#[must_use]
pub const fn ceil_div_i64(x: i64, y: i64) -> i64 {
    x.wrapping_add(y).wrapping_sub(1) / y
}

/// `ceil_div<int, int>` — the 32-bit signed case the SM90 and MLA schedulers
#[must_use]
pub const fn ceil_div_i32(x: i32, y: i32) -> i32 {
    x.wrapping_add(y).wrapping_sub(1) / y
}

/// `FA2DetermineCtaTileQ` — the tile width every prefill count is derived from.
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

/// `cost_function(qo_len, kv_len)` — the load-balancer's only notion of work.
#[must_use]
pub fn cost_function(qo_len: i32, kv_len: i32) -> f32 {
    2.0 * (qo_len as f32) + (kv_len as f32)
}

/// `packed_causal_kv_end` — how much KV a causal QO tile actually reads.
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
    let end = kv_len_init
        .wrapping_add(ceil_div_i32((qo_tile_idx + 1).wrapping_mul(cluster_tile_q), group_size));
    let clamped = if end < kv_len { end } else { kv_len };
    if clamped > 0 { clamped } else { 0 }
}

/// `max(uint32_t, int32_t)` — CUDA's overload, not `std::max`.
#[must_use]
pub const fn cuda_max_u32_i32(a: u32, b: i32) -> u32 {
    let b = b as u32;
    if a > b { a } else { b }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three `ceil_div` instantiations disagree, which is why there are
    #[test]
    fn ceil_div_conversions_are_not_interchangeable() {
        assert_eq!(ceil_div_u32(9, 4), 3);
        assert_eq!(ceil_div_u32(8, 4), 2);
        assert_eq!(ceil_div_u32(0, 4), 0);
        assert_eq!(ceil_div_i32_in_u32(-1, 4), 0);
        assert_eq!(ceil_div_i64(-1, 4), 0);
        assert_eq!(ceil_div_i32_in_u32(-8, 4), 1_073_741_822);
        assert_eq!(ceil_div_i64(-8, 4), -1);
    }

    /// The compute-capability branch is the only device fact in the file, and
    #[test]
    fn cta_tile_q_reads_compute_capability_on_one_path() {
        assert_eq!(fa2_determine_cta_tile_q(8, 128, 8), 16);
        assert_eq!(fa2_determine_cta_tile_q(8, 128, 7), 64);
        assert_eq!(fa2_determine_cta_tile_q(65, 128, 8), 128);
        assert_eq!(fa2_determine_cta_tile_q(65, 256, 8), 64);
        assert_eq!(fa2_determine_cta_tile_q(32, 512, 9), 16);
        assert_eq!(fa2_determine_cta_tile_q(33, 512, 9), 32);
    }

    /// A causal tile that is not the last one stops at the diagonal.
    #[test]
    fn causal_kv_end_stops_at_the_diagonal() {
        assert_eq!(packed_causal_kv_end(64, 100, 1, 32, 2, 1), 100);
        assert_eq!(packed_causal_kv_end(64, 100, 0, 32, 2, 1), 36 + 32);
        assert_eq!(packed_causal_kv_end(64, 0, 0, 32, 2, 1), 0);
    }
}
