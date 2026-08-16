//! How many device bytes a KV page costs, for a stack of layers that may not
//! all be the same shape.
//!
//! These sit on top of [`KvCacheFormat`], take only `HfConfig` integers, and
//! touch no device state. The memory planner scores every candidate layout by
//! multiplying a page count by the number these return; an error here makes it
//! refuse a model that would have fit, or admit one that later runs out of
//! arena.

use super::KvCacheFormat;
use crate::dtype::DType;

/// Device bytes for one KV page in this format, including dequant scratch.
///
/// Quantised caches aren't read by the attention kernels (they take BF16), so a
/// non-native format lands its dequantised K and V in scratch, charged here so
/// no caller forgets it or double-counts it. The `2 *` is K and V. The scratch
/// is sized on the *logical* `head_dim`, not the format's packed
/// `storage_head_dim`: it holds the unpacked values.
#[must_use]
pub fn device_bytes_per_page(
    format: &KvCacheFormat,
    page_size: u32,
    num_kv_heads: u32,
    head_dim: u32,
) -> u64 {
    let mut bytes = format.total_bytes_per_page(page_size, num_kv_heads, head_dim);
    if !format.is_native_bf16() {
        bytes += 2
            * u64::from(page_size)
            * u64::from(num_kv_heads)
            * u64::from(head_dim)
            * DType::Bf16.size_bytes() as u64;
    }
    bytes
}

/// Per-page bytes for a stack where every layer has the same shape.
///
/// Tensor parallelism shards the KV heads: each rank holds
/// `num_key_value_heads / tp_size`, truncating — a count not divisible by the
/// rank count loses the remainder rather than rounding up.
#[must_use]
pub fn page_bytes_homogeneous(
    num_hidden_layers: u32,
    num_key_value_heads: u32,
    head_dim_kernel: u32,
    tp_size: i32,
    format: &KvCacheFormat,
) -> u64 {
    let kv_heads = num_key_value_heads / (tp_size.max(1) as u32);
    u64::from(num_hidden_layers) * device_bytes_per_page(format, 1, kv_heads, head_dim_kernel)
}

/// Per-layer shape overrides for a heterogeneous stack.
///
/// Each field is independently optional; an absent one falls back to the
/// uniform value.
#[derive(Debug, Default, Clone)]
pub struct LayerShapes<'a> {
    /// Per-layer `head_dim`. Empty means every layer uses `head_dim_kernel`.
    pub head_dim: &'a [u32],
    /// Per-layer KV head count, unsharded — divided by `tp_size` here. Empty
    /// means every layer uses `num_key_value_heads / tp_size`.
    pub num_kv_heads: &'a [u32],
    /// Which layer owns each layer's pages: `source_layer[i] == i` owns its
    /// own, anything else aliases another layer's and is not charged. Empty
    /// means no sharing.
    pub source_layer: &'a [u32],
}

/// Per-page bytes for a stack where layers may differ in shape or share pages.
///
/// Three ways a layer can deviate: it aliases another layer's pages (only the
/// owner is charged; `source_layer[i] != i` contributes nothing), it has a
/// different `head_dim`, or it has a different KV head count. The last takes
/// the `num_kv_heads` override path, which must bill the width the allocator
/// actually reserves rather than the flat config number.
///
/// # Panics
///
/// If a non-empty override slice is shorter than `num_hidden_layers`; the C++
/// indexes these unguarded.
#[must_use]
pub fn page_bytes_per_layer(
    num_hidden_layers: u32,
    num_key_value_heads: u32,
    head_dim_kernel: u32,
    shapes: &LayerShapes<'_>,
    tp_size: i32,
    format: &KvCacheFormat,
) -> u64 {
    let n = num_hidden_layers as usize;
    for (name, slice) in [
        ("head_dim", shapes.head_dim),
        ("num_kv_heads", shapes.num_kv_heads),
        ("source_layer", shapes.source_layer),
    ] {
        assert!(
            slice.is_empty() || slice.len() >= n,
            "LayerShapes::{name} has {} entries for {n} layers",
            slice.len()
        );
    }

    let tp = tp_size.max(1) as u32;
    let uniform_kv_heads = num_key_value_heads / tp;
    let mut per_token = 0u64;
    for i in 0..n {
        let owns = shapes.source_layer.is_empty() || shapes.source_layer[i] as usize == i;
        if !owns {
            continue;
        }
        let hd = if shapes.head_dim.is_empty() {
            head_dim_kernel
        } else {
            shapes.head_dim[i]
        };
        let kv_heads = if shapes.num_kv_heads.is_empty() {
            uniform_kv_heads
        } else {
            shapes.num_kv_heads[i] / tp
        };
        per_token += device_bytes_per_page(format, 1, kv_heads, hd);
    }
    per_token
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fmt(name: &str) -> KvCacheFormat {
        KvCacheFormat::from_name(name).expect("alias is in the catalogue")
    }

    // Exhaustive agreement with the C++ lives in `tests/kv_geometry_parity.rs`.

    #[test]
    fn a_native_format_is_charged_no_dequant_scratch() {
        let bf16 = fmt("bf16");
        assert!(bf16.is_native_bf16());
        assert_eq!(
            device_bytes_per_page(&bf16, 16, 8, 128),
            bf16.total_bytes_per_page(16, 8, 128),
            "bf16 needs no scratch, so the two must agree exactly"
        );
    }

    #[test]
    fn a_quantised_format_is_charged_bf16_scratch_for_both_k_and_v() {
        let f = fmt("fp8_e4m3");
        assert!(!f.is_native_bf16());
        let scratch = device_bytes_per_page(&f, 16, 8, 128) - f.total_bytes_per_page(16, 8, 128);
        assert_eq!(
            scratch,
            2 * 16 * 8 * 128 * 2,
            "K and V, at 2 bytes per element"
        );
    }

    #[test]
    fn dequant_scratch_is_sized_on_logical_head_dim_not_the_packed_one() {
        // FP4 packs two values per byte, so `storage_head_dim` is half; the
        // scratch holds unpacked BF16 and must not inherit that halving.
        let f = fmt("nvfp4");
        assert_eq!(f.storage_head_dim(128), 64, "storage is packed");
        let scratch = device_bytes_per_page(&f, 16, 8, 128) - f.total_bytes_per_page(16, 8, 128);
        assert_eq!(scratch, 2 * 16 * 8 * 128 * 2, "but scratch is not");
    }

    #[test]
    fn homogeneous_is_exactly_the_per_layer_cost_times_the_layer_count() {
        let f = fmt("bf16");
        let one = device_bytes_per_page(&f, 1, 8, 128);
        assert_eq!(page_bytes_homogeneous(32, 8, 128, 1, &f), 32 * one);
        assert_eq!(page_bytes_homogeneous(0, 8, 128, 1, &f), 0);
    }

    #[test]
    fn tensor_parallelism_shards_kv_heads_by_truncating_division() {
        let f = fmt("bf16");
        // 3 heads across 2 ranks is 1 per rank, not 2 (truncating).
        assert_eq!(
            page_bytes_homogeneous(1, 3, 128, 2, &f),
            device_bytes_per_page(&f, 1, 1, 128)
        );
        // Sharded past the head count, a rank holds nothing.
        assert_eq!(
            page_bytes_homogeneous(1, 4, 128, 8, &f),
            device_bytes_per_page(&f, 1, 0, 128)
        );
        // tp of 0 means 1, not a division by zero.
        assert_eq!(
            page_bytes_homogeneous(4, 8, 128, 0, &f),
            page_bytes_homogeneous(4, 8, 128, 1, &f)
        );
    }

    #[test]
    fn empty_overrides_reproduce_the_homogeneous_answer() {
        let f = fmt("fp8_e4m3");
        for tp in [0, 1, 2, 4] {
            assert_eq!(
                page_bytes_per_layer(32, 8, 128, &LayerShapes::default(), tp, &f),
                page_bytes_homogeneous(32, 8, 128, tp, &f),
                "tp={tp}"
            );
        }
    }

    #[test]
    fn an_aliasing_layer_is_not_charged() {
        let f = fmt("bf16");
        // Odd layers alias their even predecessor, so half the stack pays
        // nothing.
        let src: Vec<u32> = (0..32u32)
            .map(|i| if i % 2 == 1 { i - 1 } else { i })
            .collect();
        let shapes = LayerShapes {
            source_layer: &src,
            ..Default::default()
        };
        assert_eq!(
            page_bytes_per_layer(32, 8, 128, &shapes, 1, &f),
            page_bytes_homogeneous(16, 8, 128, 1, &f)
        );
        // All layers aliasing layer 0 leaves exactly one charged.
        let all_zero = vec![0u32; 32];
        let shapes = LayerShapes {
            source_layer: &all_zero,
            ..Default::default()
        };
        assert_eq!(
            page_bytes_per_layer(32, 8, 128, &shapes, 1, &f),
            device_bytes_per_page(&f, 1, 8, 128)
        );
    }

    #[test]
    fn the_gemma4_wide_layers_are_charged_their_real_width() {
        // Every 4th layer 4x wider must cost more than the flat config number,
        // but exactly what the allocator reserves — not 4x every layer.
        let f = fmt("bf16");
        let kv: Vec<u32> = (0..32u32)
            .map(|i| if i % 4 == 0 { 32 } else { 8 })
            .collect();
        let shapes = LayerShapes {
            num_kv_heads: &kv,
            ..Default::default()
        };
        let got = page_bytes_per_layer(32, 8, 128, &shapes, 1, &f);

        let wide = device_bytes_per_page(&f, 1, 32, 128);
        let narrow = device_bytes_per_page(&f, 1, 8, 128);
        assert_eq!(got, 8 * wide + 24 * narrow);

        assert!(
            got > page_bytes_homogeneous(32, 8, 128, 1, &f),
            "wide layers must cost more"
        );
        assert!(
            got < 32 * wide,
            "but far less than charging every layer the wide count, which is \
             the mistake that made the model unloadable"
        );
    }

    #[test]
    fn per_layer_kv_heads_are_unsharded_and_get_divided_here() {
        let f = fmt("bf16");
        let kv = vec![16u32; 4];
        let shapes = LayerShapes {
            num_kv_heads: &kv,
            ..Default::default()
        };
        assert_eq!(
            page_bytes_per_layer(4, 999, 128, &shapes, 4, &f),
            4 * device_bytes_per_page(&f, 1, 4, 128),
            "the override is divided by tp, and the flat config field is ignored"
        );
    }

    #[test]
    fn per_layer_head_dim_overrides_the_uniform_one() {
        let f = fmt("bf16");
        let hd = vec![64u32, 256, 64, 256];
        let shapes = LayerShapes {
            head_dim: &hd,
            ..Default::default()
        };
        assert_eq!(
            page_bytes_per_layer(4, 8, 128, &shapes, 1, &f),
            2 * device_bytes_per_page(&f, 1, 8, 64) + 2 * device_bytes_per_page(&f, 1, 8, 256)
        );
    }

    #[test]
    fn overrides_compose_and_aliasing_wins_over_both() {
        let f = fmt("bf16");
        let hd = vec![64u32, 256, 64, 256];
        let kv = vec![32u32, 8, 32, 8];
        // Layers 1 and 3 alias, so their (wide, deep) shapes never register.
        let src = vec![0u32, 0, 2, 2];
        let shapes = LayerShapes {
            head_dim: &hd,
            num_kv_heads: &kv,
            source_layer: &src,
        };
        assert_eq!(
            page_bytes_per_layer(4, 8, 128, &shapes, 1, &f),
            2 * device_bytes_per_page(&f, 1, 32, 64)
        );
    }

    #[test]
    #[should_panic(expected = "has 2 entries for 32 layers")]
    fn a_short_override_slice_is_rejected_rather_than_read_past_the_end() {
        // The C++ indexes these unguarded; here the precondition is checked.
        let f = fmt("bf16");
        let hd = vec![64u32, 128];
        let shapes = LayerShapes {
            head_dim: &hd,
            ..Default::default()
        };
        let _ = page_bytes_per_layer(32, 8, 128, &shapes, 1, &f);
    }
}
