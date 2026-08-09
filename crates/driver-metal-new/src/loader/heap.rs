//! Fixed byte offsets for the one resident heap, computed from the geometry.
//!
//! Every region the decode step touches — weights, KV, GDN state, the
//! scratch pool, IO, and the paged-KV bridge — lives in a single heap laid
//! out once at load. The offsets are pure arithmetic over
//! [`DecodeGeometry`], so they are computed here, testable on any host; the
//! actual allocation and argument-table wiring layer on top.
//!
//! Sizing rules, from the C++ (`loader/heap_layout.hpp`):
//!
//! * **Weights** are sized from the LOADER manifest — actual per-tensor
//!   bytes, not re-derived shapes. The GDN in/out projections are
//!   model-specific; trust the loader.
//! * **KV / State / Scratch / IO** are derived exactly from geometry, here.
//!
//! The scratch slot's width is the one number this file used to derive for
//! itself, and the drift that caused is why it no longer does: see
//! [`scratch_slot_elems`]'s module. `plan_heap` calls the survivor.
//!
//! The paged-KV regions (`mb_io`, `kv_pool`) are additive: both zero when
//! [`DecodeGeometry::paged_kv_enabled`] is off, so the sealed M=1 layout is
//! preserved byte-for-byte.

use crate::batch::{DecodeGeometry, SCRATCH_POOL, scratch_slot_elems};
use crate::tuning::Tuning;

/// `n` rounded up to a multiple of `alignment` (256 is Metal's
/// buffer-offset alignment). An alignment of zero behaves as one.
#[must_use]
pub const fn align_up(n: u64, alignment: u64) -> u64 {
    let a = if alignment == 0 { 1 } else { alignment };
    n.div_ceil(a) * a
}

/// The knobs `plan_heap` takes beside the geometry.
///
/// Defaults are the sealed values: a 4096-token context window, fp32 GDN
/// state, bf16 activations, 256-byte region and slot alignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeapParams {
    /// KV capacity per stream, in tokens (the single-stream context window).
    pub max_ctx: u32,
    /// GDN recurrent/conv state element size (fp32 = 4).
    pub state_dtype_bytes: u32,
    /// Activation element size (bf16 = 2).
    pub act_dtype_bytes: u32,
    /// Alignment of each region's base and rounded size.
    pub region_alignment: u64,
    /// Alignment of the slots inside a region.
    pub slot_alignment: u64,
}

impl Default for HeapParams {
    fn default() -> Self {
        HeapParams {
            max_ctx: 4096,
            state_dtype_bytes: 4,
            act_dtype_bytes: 2,
            region_alignment: 256,
            slot_alignment: 256,
        }
    }
}

/// Per-region byte sizes and base offsets within the single heap.
///
/// The last five fields are sizing intermediates surfaced for tests and
/// logging; `total` is the heap allocation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HeapPlan {
    /// Load-once read-only weights: base offset.
    pub weights_off: u64,
    /// Load-once read-only weights: size.
    pub weights_bytes: u64,
    /// The M=1 contiguous KV ring (full-attention layers only): base offset.
    pub kv_off: u64,
    /// The M=1 contiguous KV ring: size.
    pub kv_bytes: u64,
    /// GDN resident state (per-slot slabs): base offset.
    pub state_off: u64,
    /// GDN resident state: size.
    pub state_bytes: u64,
    /// The activation ping-pong pool: base offset.
    pub scratch_off: u64,
    /// The activation ping-pong pool: size.
    pub scratch_bytes: u64,
    /// Per-token scalars and logits: base offset.
    pub io_off: u64,
    /// Per-token scalars and logits: size.
    pub io_bytes: u64,
    /// The multi-batch CSR buffers (paged fires only): base offset.
    pub mb_io_off: u64,
    /// The multi-batch CSR buffers: size, zero unless paged.
    pub mb_io_bytes: u64,
    /// The paged NHD KV pool (separate from the M=1 ring): base offset.
    pub kv_pool_off: u64,
    /// The paged NHD KV pool: size, zero unless paged.
    pub kv_pool_bytes: u64,
    /// The whole heap.
    pub total: u64,
    /// One ping-pong buffer, slot-aligned.
    pub scratch_slot_bytes: u64,
    /// K+V for one full-attention layer (M=1 HND ring).
    pub kv_per_layer: u64,
    /// Conv + recurrent state for one GDN layer, all slots.
    pub state_per_layer: u64,
    /// K+V for one full-attention layer (paged NHD pool).
    pub kv_pool_per_layer: u64,
    /// Flattened CSR capacity for one paged fire, in page REFERENCES — a
    /// physical page may legitimately occur in several requests' segments
    /// (shared prefixes, forks), so capacity is references, not unique
    /// pages.
    pub max_page_refs: u64,
}

/// Lay the heap out: back-to-back regions, each region-aligned.
///
/// `weights_bytes_from_manifest` is the loader's total — see the module docs
/// for why weights are never re-derived here. `tuning` reaches the one place
/// the scratch slot's width is decided.
#[must_use]
pub fn plan_heap(
    geometry: &DecodeGeometry,
    tuning: &Tuning,
    weights_bytes_from_manifest: u64,
    params: HeapParams,
) -> HeapPlan {
    let g = geometry;
    let act = u64::from(params.act_dtype_bytes);
    let state = u64::from(params.state_dtype_bytes);
    let align_region = |bytes: u64| align_up(bytes, params.region_alignment);
    let align_slot = |bytes: u64| align_up(bytes, params.slot_alignment);
    let mut p = HeapPlan {
        weights_bytes: align_region(weights_bytes_from_manifest),
        ..HeapPlan::default()
    };

    // KV: append-only, full-attention layers only. Per layer: k + v, each
    // [n_kv_heads, max_ctx, head_dim] in the activation dtype.
    let n_full = u64::from(g.full_attn_layers());
    p.kv_per_layer =
        2 * u64::from(g.n_kv_heads) * u64::from(params.max_ctx) * u64::from(g.head_dim) * act;
    p.kv_bytes = align_region(n_full * p.kv_per_layer);

    // State: per-slot slabs, S = max_slots; in-place at S=1. conv_state
    // [gdn_conv_dim, gdn_conv_k] is PING-PONG — an in-place conv shift races
    // the K-tap reads, hence the factor of two. recurrent_state [Vh, Vd, Kd]
    // is in-place: each (v-head, v-dim) row is owned by one threadgroup.
    // Slots pack at the NATURAL per-slot stride — the slotted kernel indexes
    // slot * (Kc*CDIM) and slot * (Hv*Vd*Dk) — and only the whole slab is
    // aligned, so max_slots=1 is byte-identical to the sealed single-slot
    // layout.
    let n_gdn = u64::from(g.n_layers) - n_full;
    let conv_state = u64::from(g.gdn_conv_dim) * u64::from(g.gdn_conv_k) * state;
    let recur_state =
        u64::from(g.gdn_v_heads) * u64::from(g.gdn_v_dim) * u64::from(g.gdn_k_dim) * state;
    let slots = u64::from(g.max_slots);
    p.state_per_layer = 2 * align_slot(slots * conv_state) + align_slot(slots * recur_state);
    p.state_bytes = align_region(n_gdn * p.state_per_layer);

    // Scratch: one slot must hold the largest activation that ping-pongs
    // through the DAG, and `scratch_slot_elems` is the ONE place that
    // decides how large that is. Paged fires store token-major
    // [max_tokens, width] activations in the same colored buffers; the
    // sealed M=1 allocation stays byte-identical.
    let scratch_rows = if g.paged_kv_enabled {
        g.max_tokens.max(1)
    } else {
        1
    };
    p.scratch_slot_bytes = align_slot(scratch_slot_elems(g, tuning, scratch_rows) * act);
    p.scratch_bytes = align_region(SCRATCH_POOL as u64 * p.scratch_slot_bytes);

    // IO: TokenId/Position/SeqLen/NextToken, u32[max_tokens] each, then the
    // logits. The historical M=1 allocation was four bytes per logit; it is
    // retained exactly there, while paged output is densely [N, vocab] in
    // the activation dtype.
    let scalars = 4 * align_slot(4 * u64::from(g.max_tokens));
    let logits = if g.paged_kv_enabled {
        align_slot(u64::from(g.vocab) * u64::from(scratch_rows) * act)
    } else {
        align_slot(u64::from(g.vocab) * 4)
    };
    p.io_bytes = align_region(scalars + logits);

    // MbIo: the multi-batch CSR buffers, sized from max_requests (R),
    // max_tokens (N) and total_pages. Additive — zero bytes unless paged.
    if g.paged_kv_enabled {
        let r = u64::from(g.max_requests);
        let n = u64::from(g.max_tokens);
        let qo_indptr = align_slot((r + 1) * 4);
        let kv_page_indptr = align_slot((r + 1) * 4);
        p.max_page_refs = r * u64::from(g.total_pages);
        let kv_page_indices = align_slot(p.max_page_refs * 4);
        let kv_last_page_len = align_slot(r * 4);
        let rs_slot_ids = align_slot(r * 4);
        let rs_slot_flags = align_slot(r);
        let req_of_token = align_slot(n * 4);
        let slot_of_token = align_slot(n * 4);
        let w_page = align_slot(n * 4);
        let w_off = align_slot(n * 4);
        let mask_stride = u64::from(g.total_pages) * u64::from(g.kv_page_size);
        let attn_mask = align_slot(n * mask_stride);
        let attn_mask_stride = align_slot(4);
        let attn_mask_enabled = align_slot(n);
        p.mb_io_bytes = align_region(
            qo_indptr
                + kv_page_indptr
                + kv_page_indices
                + kv_last_page_len
                + rs_slot_ids
                + rs_slot_flags
                + req_of_token
                + slot_of_token
                + w_page
                + w_off
                + attn_mask
                + attn_mask_stride
                + attn_mask_enabled,
        );
    }

    // KvPagePool: a SEPARATE page-major [num_pages, page_size, n_kv_heads,
    // head_dim] pool from the M=1 HND ring above — the paged append and SDPA
    // read THIS region; the ring is untouched. Additive, like MbIo.
    if g.paged_kv_enabled {
        p.kv_pool_per_layer = 2
            * u64::from(g.total_pages)
            * u64::from(g.kv_page_size)
            * u64::from(g.n_kv_heads)
            * u64::from(g.head_dim)
            * act;
        p.kv_pool_bytes = align_region(n_full * p.kv_pool_per_layer);
    }

    // Back-to-back, each region-aligned.
    let mut off = 0;
    p.weights_off = off;
    off = align_region(off + p.weights_bytes);
    p.kv_off = off;
    off = align_region(off + p.kv_bytes);
    p.state_off = off;
    off = align_region(off + p.state_bytes);
    p.scratch_off = off;
    off = align_region(off + p.scratch_bytes);
    p.io_off = off;
    off = align_region(off + p.io_bytes);
    p.mb_io_off = off;
    off = align_region(off + p.mb_io_bytes);
    p.kv_pool_off = off;
    off = align_region(off + p.kv_pool_bytes);
    p.total = off;
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sealed_dense_heap_has_no_paged_regions_and_aligned_offsets() {
        let geometry = DecodeGeometry::default();
        let tuning = Tuning::default();
        let plan = plan_heap(&geometry, &tuning, 1_000_003, HeapParams::default());
        assert_eq!(plan.mb_io_bytes, 0, "paged regions are opt-in");
        assert_eq!(plan.kv_pool_bytes, 0);
        for off in [
            plan.weights_off,
            plan.kv_off,
            plan.state_off,
            plan.scratch_off,
            plan.io_off,
            plan.total,
        ] {
            assert_eq!(off % 256, 0, "every region base is Metal-aligned");
        }
        // Back-to-back: each base is the previous base plus its rounded
        // size, and nothing overlaps.
        assert_eq!(plan.weights_off, 0);
        assert_eq!(plan.kv_off, align_up(plan.weights_bytes, 256));
        assert_eq!(plan.state_off, plan.kv_off + plan.kv_bytes);
        assert!(plan.total >= plan.io_off + plan.io_bytes);
    }

    #[test]
    fn the_regions_are_the_geometrys_arithmetic() {
        let geometry = DecodeGeometry::default();
        let tuning = Tuning::default();
        let plan = plan_heap(&geometry, &tuning, 0, HeapParams::default());
        // 24 layers at interval 4: six full-attention, eighteen GDN.
        assert_eq!(plan.kv_per_layer, 2 * 2 * 4096 * 256 * 2);
        assert_eq!(plan.kv_bytes, 6 * plan.kv_per_layer);
        // conv ping-pongs (x2), recurrent is in-place.
        let conv = align_up(6144 * 4 * 4, 256);
        let recur = align_up(16u64 * 128 * 128 * 4, 256);
        assert_eq!(plan.state_per_layer, 2 * conv + recur);
        assert_eq!(plan.state_bytes, 18 * plan.state_per_layer);
        // The slot is the one derivation: widest (6144 elems) in bf16.
        assert_eq!(plan.scratch_slot_bytes, align_up(6144 * 2, 256));
        assert_eq!(plan.scratch_bytes, 9 * plan.scratch_slot_bytes);
        // M=1 logits stay four bytes each, exactly as sealed.
        assert_eq!(
            plan.io_bytes,
            align_up(4 * 256 + align_up(248_320 * 4, 256), 256)
        );
    }

    #[test]
    fn a_paged_heap_adds_its_regions_without_moving_the_sealed_ones() {
        let tuning = Tuning::default();
        let dense = plan_heap(
            &DecodeGeometry::default(),
            &tuning,
            1 << 20,
            HeapParams::default(),
        );
        let paged_geometry = DecodeGeometry {
            paged_kv_enabled: true,
            max_tokens: 128,
            max_requests: 8,
            total_pages: 64,
            ..DecodeGeometry::default()
        };
        let paged = plan_heap(&paged_geometry, &tuning, 1 << 20, HeapParams::default());
        assert_eq!(paged.weights_off, dense.weights_off);
        assert_eq!(paged.kv_off, dense.kv_off);
        assert_eq!(paged.state_off, dense.state_off);
        assert!(paged.mb_io_bytes > 0);
        assert!(paged.kv_pool_bytes > 0);
        assert_eq!(paged.max_page_refs, 8 * 64, "references, not unique pages");
        assert_eq!(paged.kv_pool_per_layer, 2 * 64 * 32 * 2 * 256 * 2);
        assert_eq!(paged.total, paged.kv_pool_off + paged.kv_pool_bytes);
    }

    #[test]
    fn a_paged_fires_scratch_and_logits_scale_with_the_token_count() {
        let tuning = Tuning::default();
        let geometry = DecodeGeometry {
            paged_kv_enabled: true,
            max_tokens: 128,
            max_requests: 8,
            total_pages: 64,
            ..DecodeGeometry::default()
        };
        let plan = plan_heap(&geometry, &tuning, 0, HeapParams::default());
        assert_eq!(plan.scratch_slot_bytes, align_up(6144 * 128 * 2, 256));
        // Paged logits are dense [N, vocab] in bf16, not the M=1 f32 row.
        let scalars = 4 * align_up(4 * 128, 256);
        let logits = align_up(248_320 * 128 * 2, 256);
        assert_eq!(plan.io_bytes, align_up(scalars + logits, 256));
    }
}
