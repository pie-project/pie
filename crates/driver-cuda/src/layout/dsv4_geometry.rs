//! Geometry for the DeepSeek-V4 compressor caches.
//!
//! Port of the sizing logic in
//! `driver-cuda/csrc/src/store/dsv4_compress_cache.cpp`.
//!
//! A V4 layer with `compress_ratio > 0` runs a second attention over
//! *compressed* KV entries: one entry per `ratio` tokens, each a per-dimension
//! softmax pool over the `coff * ratio` tokens ending at a boundary position.
//! Three per-token tensors have to survive across forward passes:
//!
//! * `state_kv` and `state_score` -- the compressor's `wkv` / `wgate`
//!   projections, `coff * head_dim` wide. A boundary token pools projections
//!   written during *earlier* forward passes, which is why they are cache and
//!   not scratch.
//! * `comp_kv` -- the finished entry, `head_dim` wide.
//!
//! Only the widths are here. Everything else in that file is allocation and a
//! best-effort `cudaMemset`.

/// The compressor's window coefficient for a given ratio.
///
/// A ratio-4 layer pools over a window of `2 * 4` tokens rather than `4`;
/// every other ratio pools over exactly its own span. This single special case
/// is the whole function, and it is why `state_kv` is not simply `head_dim`
/// wide: at ratio 4 it is **twice** as wide as the naive reading suggests.
#[must_use]
pub const fn compressor_coff(ratio: i32) -> u32 {
    if ratio == 4 { 2 } else { 1 }
}

/// The width in elements of one layer's `state_kv` / `state_score` rows.
///
/// Returns `None` for a layer that does not compress, which is the same thing
/// the C++ expresses by leaving that layer's tensors empty.
#[must_use]
pub const fn state_width(ratio: i32, head_dim: u32) -> Option<u32> {
    if ratio <= 0 { None } else { Some(compressor_coff(ratio) * head_dim) }
}

/// Bytes of compressor state per token, summed over every compressing layer.
///
/// The `2 *` is `state_kv` and `state_score`; the trailing `head_dim` is
/// `comp_kv`. Everything is BF16, so the element size is fixed at 2 rather
/// than read from a dtype -- matching the C++, which hardcodes
/// `sizeof(std::uint16_t)` and allocates `DType::BF16`.
///
/// This is what the memory planner adds on top of the KV cache for a V4 model.
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
/// size is zero -- the C++ returns a default-constructed (empty) cache in all
/// three cases, and an empty cache costs nothing.
///
/// A ratios list shorter than `num_hidden_layers` leaves the trailing layers
/// uncompressed. That is the C++'s explicit `li < ratios.size() ? ... : 0`,
/// not an accident of iteration, so a short list is a supported input rather
/// than a caller error.
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
/// Index `i` is `Some(width)` when layer `i` compresses. Length is always
/// `num_hidden_layers`, mirroring the C++'s `layers_.resize(L)` -- every
/// layer gets a slot whether or not it compresses, so `has_layer` is an index
/// rather than a search.
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
        // The C++ returns early with an empty `layers_`, so there is no
        // per-layer table to index -- distinct from "L layers, none of which
        // compress", even though both cost zero bytes.
        assert!(layer_widths(&[], 8, 128).is_empty());
        assert_eq!(layer_widths(&[0; 8], 8, 128).len(), 8);
        assert_eq!(compress_cache_bytes(&[], 8, 128, 16, 64), 0);
    }

    #[test]
    fn cache_bytes_are_per_token_bytes_times_the_token_capacity() {
        let ratios = [2, 0, 4, 8];
        let per_token = compress_bytes_per_token(&ratios, 128);
        assert_eq!(compress_cache_bytes(&ratios, 4, 128, 16, 64), per_token * 16 * 64);
    }

    #[test]
    fn an_empty_page_geometry_costs_nothing() {
        assert_eq!(compress_cache_bytes(&[2, 4], 2, 128, 0, 64), 0);
        assert_eq!(compress_cache_bytes(&[2, 4], 2, 128, 16, 0), 0);
    }

    #[test]
    fn cache_bytes_only_count_the_first_num_hidden_layers_ratios() {
        // A ratios list LONGER than the layer count is truncated by
        // `compress_cache_bytes` (which walks layers) but not by
        // `compress_bytes_per_token` (which walks ratios). That asymmetry is
        // in the C++ and is worth knowing before trusting either number.
        let ratios = [2, 2, 2, 2];
        assert_eq!(
            compress_cache_bytes(&ratios, 2, 128, 1, 1),
            compress_bytes_per_token(&[2, 2], 128)
        );
        assert_eq!(compress_bytes_per_token(&ratios, 128), 4 * compress_bytes_per_token(&[2], 128));
    }
}
