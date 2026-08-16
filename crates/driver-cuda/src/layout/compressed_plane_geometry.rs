//! Geometry for the DeepSeek-V4 compressor caches.
//!
//! A V4 layer with `compress_ratio > 0` pools KV into one compressed entry per
//! `ratio` tokens, over the `coff * ratio` tokens ending at a boundary. Three
//! per-token tensors survive across forward passes (so they are cache, not
//! scratch): `state_kv` and `state_score` (the `wkv`/`wgate` projections,
//! `coff * head_dim` wide) and `comp_kv` (the finished entry, `head_dim`
//! wide). Only the widths live here.

/// The compressor's window coefficient for a given ratio.
///
/// A ratio-4 layer pools over `2 * 4` tokens, not `4`; every other ratio pools
/// over its own span. So at ratio 4 `state_kv` is twice `head_dim` wide.
#[must_use]
pub const fn compressor_coff(ratio: i32) -> u32 {
    if ratio == 4 { 2 } else { 1 }
}

/// Width in elements of one layer's `state_kv` / `state_score` rows.
///
/// `None` for a layer that does not compress.
#[must_use]
pub const fn state_width(ratio: i32, head_dim: u32) -> Option<u32> {
    if ratio <= 0 {
        None
    } else {
        Some(compressor_coff(ratio) * head_dim)
    }
}

/// Bytes of compressor state per token, summed over every compressing layer.
///
/// The `2 *` is `state_kv` and `state_score`; the trailing `head_dim` is
/// `comp_kv`. Everything is BF16, so the element size is fixed at 2. The
/// planner adds this on top of the KV cache for a V4 model.
#[must_use]
pub fn compress_bytes_per_token(ratios: &[i32], head_dim: u32) -> u64 {
    ratios
        .iter()
        .filter_map(|&r| state_width(r, head_dim))
        .map(|w| (2 * u64::from(w) + u64::from(head_dim)) * 2)
        .sum()
}

/// Device bytes for the whole compressor cache at a given page geometry.
///
/// Zero when nothing compresses, when there are no pages, or when the page
/// size is zero. A ratios list shorter than `num_hidden_layers` is supported:
/// the trailing layers are left uncompressed.
#[must_use]
pub fn compress_cache_bytes(
    ratios: &[i32],
    num_hidden_layers: u32,
    head_dim: u32,
    num_pages: u32,
    page_size: u32,
) -> u64 {
    if ratios.is_empty() || num_pages == 0 || page_size == 0 {
        return 0;
    }
    let per_page_tokens = u64::from(num_pages) * u64::from(page_size);
    (0..num_hidden_layers as usize)
        .map(|li| ratios.get(li).copied().unwrap_or(0))
        .filter_map(|r| state_width(r, head_dim))
        .map(|w| per_page_tokens * (2 * u64::from(w) + u64::from(head_dim)) * 2)
        .sum()
}

/// Which layers compress, as a per-layer width table.
///
/// Index `i` is `Some(width)` when layer `i` compresses; length is always
/// `num_hidden_layers`, so lookup is an index rather than a search.
#[must_use]
pub fn layer_widths(ratios: &[i32], num_hidden_layers: u32, head_dim: u32) -> Vec<Option<u32>> {
    if ratios.is_empty() {
        return Vec::new();
    }
    (0..num_hidden_layers as usize)
        .map(|li| state_width(ratios.get(li).copied().unwrap_or(0), head_dim))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratio_four_is_the_one_wide_window() {
        assert_eq!(compressor_coff(4), 2);
        for r in [1, 2, 3, 5, 6, 8, 16, 32, -1, 0] {
            assert_eq!(compressor_coff(r), 1, "ratio {r}");
        }
    }

    #[test]
    fn a_ratio_four_layer_is_twice_as_wide_as_its_neighbours() {
        assert_eq!(state_width(4, 128), Some(256));
        assert_eq!(state_width(2, 128), Some(128));
        assert_eq!(state_width(8, 128), Some(128));
    }

    #[test]
    fn a_non_positive_ratio_means_the_layer_does_not_compress() {
        for r in [0, -1, -100] {
            assert_eq!(state_width(r, 128), None, "ratio {r}");
        }
    }

    #[test]
    fn bytes_per_token_counts_two_state_tensors_plus_one_output() {
        // 2 * width + head_dim elements, at 2 bytes each.
        assert_eq!(compress_bytes_per_token(&[2], 128), (2 * 128 + 128) * 2);
        assert_eq!(compress_bytes_per_token(&[4], 128), (2 * 256 + 128) * 2);
    }

    #[test]
    fn non_compressing_layers_contribute_nothing() {
        assert_eq!(compress_bytes_per_token(&[0, 0, 0], 128), 0);
        assert_eq!(compress_bytes_per_token(&[], 128), 0);
        assert_eq!(
            compress_bytes_per_token(&[2, 0, 2, -1], 128),
            2 * compress_bytes_per_token(&[2], 128)
        );
    }

    #[test]
    fn a_short_ratios_list_leaves_the_trailing_layers_uncompressed() {
        // Supported input, not a caller error: the C++ indexes defensively.
        let widths = layer_widths(&[2, 4], 8, 128);
        assert_eq!(widths.len(), 8);
        assert_eq!(widths[0], Some(128));
        assert_eq!(widths[1], Some(256));
        assert!(widths[2..].iter().all(Option::is_none));
    }

    #[test]
    fn an_empty_ratios_list_produces_no_cache_at_all() {
        // Empty `layers_` (early return) is distinct from "L layers, none
        // compressing", though both cost zero bytes.
        assert!(layer_widths(&[], 8, 128).is_empty());
        assert_eq!(layer_widths(&[0; 8], 8, 128).len(), 8);
        assert_eq!(compress_cache_bytes(&[], 8, 128, 16, 64), 0);
    }

    #[test]
    fn cache_bytes_are_per_token_bytes_times_the_token_capacity() {
        let ratios = [2, 0, 4, 8];
        let per_token = compress_bytes_per_token(&ratios, 128);
        assert_eq!(
            compress_cache_bytes(&ratios, 4, 128, 16, 64),
            per_token * 16 * 64
        );
    }

    #[test]
    fn an_empty_page_geometry_costs_nothing() {
        assert_eq!(compress_cache_bytes(&[2, 4], 2, 128, 0, 64), 0);
        assert_eq!(compress_cache_bytes(&[2, 4], 2, 128, 16, 0), 0);
    }

    #[test]
    fn cache_bytes_only_count_the_first_num_hidden_layers_ratios() {
        // A ratios list longer than the layer count is truncated by
        // `compress_cache_bytes` (walks layers) but not by
        // `compress_bytes_per_token` (walks ratios) — the two disagree.
        let ratios = [2, 2, 2, 2];
        assert_eq!(
            compress_cache_bytes(&ratios, 2, 128, 1, 1),
            compress_bytes_per_token(&[2, 2], 128)
        );
        assert_eq!(
            compress_bytes_per_token(&ratios, 128),
            4 * compress_bytes_per_token(&[2], 128)
        );
    }
}
