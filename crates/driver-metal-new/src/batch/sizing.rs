//! How many rows the mixture's sort can produce, and how wide a scratch slot
//! has to be — asked once, here.
//!
//! The C++ derived the slot width twice: once in `batch/scratch.hpp` and once
//! in `loader/heap_layout.hpp`, with a comment on the second copy asking that
//! the two be kept in sync. They drifted, in both directions. The heap's copy
//! never grew the mixture's terms, so on Qwen3.6-35B-A3B it sized the slot at
//! 8320 elements a row where the binder lays the prefill's rows 16384 apart —
//! every row past the halfway point wrote into the next colour — and it never
//! learned that a batched sort's padded stack is not `n` times the M=1 one.
//! Two derivations of one number is not a thing to keep in sync; it is a
//! thing to delete. This module is the survivor; the heap plan calls it.
//!
//! The other boundary this file keeps: [`sorted_rows`] is the KERNEL's bound
//! (mirrored from `pie/kernels/moe.h`, read off `moe_route.metal`'s sort),
//! and it takes the tile as a parameter rather than reading
//! [`Tuning`](crate::tuning::Tuning) itself. Which tile a batch gets is a
//! tuning decision — a sweep of one machine, overridable by an env var — and
//! a bound the kernel guarantees must not depend on one. The C++ once let the
//! tuning table be read from below the kernel boundary, which is a two-way
//! dependency with the arrow hidden; here the policy computes the tile and
//! passes it down.

use crate::tuning::Tuning;

use super::color::{Coloring, Use};
use super::dispatch::Dispatch;

use super::geometry::DecodeGeometry;

/// Whether the routed projections run as matmuls over the sorted stack, or
/// as a matvec one row at a time.
///
/// Only a matmul pads each touched expert's run to a whole tile, and the
/// padding is not free: at 32 lanes on a 256-expert mixture it is 4096 rows
/// carrying 256 real ones. A caller that will run the matvec must say so, or
/// it pays a batched layout to walk it one row at a time — which measured
/// 41.6 tok/s against 284.0 for the same fire, both correct.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutedProjection {
    /// The routed GEMM over the padded stack.
    Matmul,
    /// One row at a time; the sort is a pure grouping with no padding.
    Matvec,
}

/// How many sorted rows a batch of `n_pairs` can produce, at a given tile.
///
/// The kernel's bound, not this driver's (`pie/kernels/moe.h`). It is the
/// worst case rather than the actual: the real count depends on how the
/// router spread the rows, which is a number the GPU has and the host would
/// have to stall to read. Every touched expert can waste `tile - 1` rows and
/// at most `min(n_pairs, n_experts)` experts are touched, so this bound is
/// reached and cannot be tightened without the routing itself.
///
/// `tile` is a parameter and not a call to
/// [`Tuning::moe_tile_rows`](crate::tuning::Tuning::moe_tile_rows): see the
/// module docs.
#[must_use]
pub fn sorted_rows(n_pairs: u32, n_experts: u32, tile: u32) -> u64 {
    let n = u64::from(n_pairs);
    if tile <= 1 {
        return n;
    }
    let tile = u64::from(tile);
    let touched = n.min(u64::from(n_experts));
    let bound = n + touched * (tile - 1);
    bound.div_ceil(tile) * tile
}

/// The extent of the mixture's sorted `(token, slot)` stack for one fire.
///
/// It is the extent of every routed projection, of the SiluMul between them,
/// of the gather that fills them and of the pool slot that holds them, so it
/// is asked in ONE place: five call sites each deriving it would be five
/// chances to disagree, and a disagreement here is a read off the end of the
/// routing rather than a wrong answer.
#[must_use]
pub fn moe_sorted_rows(
    geometry: &DecodeGeometry,
    tuning: &Tuning,
    n_tokens: u32,
    run: RoutedProjection,
) -> u64 {
    let n = n_tokens.max(1);
    let pairs = u64::from(n) * u64::from(geometry.experts_per_token);
    // Unbatched, the sort is a pure grouping: one row per (token, slot) pair
    // and no padding, which is exactly what `sorted_rows` returns at tile 1.
    // Said here rather than by forcing the tile, so the two answers cannot
    // drift.
    if run == RoutedProjection::Matvec {
        return pairs;
    }
    let pairs = u32::try_from(pairs).unwrap_or(u32::MAX);
    let tile = tuning.moe_tile_rows(pairs, geometry.n_experts);
    sorted_rows(pairs, geometry.n_experts, tile)
}

/// The widest activation that ping-pongs through the DAG, in elements, at
/// one row per token.
///
/// The mixture's terms are the one place in this family where a dispatch
/// writes more than one result per token: the sort turns `(token, slot)`
/// pairs into ROWS, so the gathered input is `sorted × hidden` and the
/// gate/up stack is `sorted × moe_intermediate`. Sized like a dense
/// activation they would overrun by a factor of `experts_per_token` — and
/// the overrun is into the next pool slot, which is another live activation
/// rather than unmapped memory, so it corrupts instead of faulting.
#[must_use]
pub fn scratch_widest_elems(geometry: &DecodeGeometry, tuning: &Tuning) -> u64 {
    let g = geometry;
    let mut widest = u64::from(g.intermediate).max(u64::from(g.gdn_conv_dim));
    // fp32 prep scratch shares the BF16 byte pitch. It holds two gate scalars
    // per value head followed by the token-parallel V convolution — and, in
    // the other two slots, the normalized q and k rows, Kd wide per value
    // head. Kd and Vd are equal on every released member, so the V term alone
    // covered it; stated separately because nothing makes them equal.
    let gdn_prep =
        2 * (2 * u64::from(g.gdn_v_heads) + u64::from(g.gdn_v_heads) * u64::from(g.gdn_v_dim));
    let gdn_qk = 2 * u64::from(g.gdn_v_heads) * u64::from(g.gdn_k_dim);
    widest = widest.max(gdn_prep).max(gdn_qk);
    let q = u64::from(g.n_q_heads) * u64::from(g.head_dim);
    widest = widest.max(q);
    if g.is_moe() {
        let sorted = moe_sorted_rows(g, tuning, 1, RoutedProjection::Matmul);
        widest = widest
            .max(sorted * u64::from(g.hidden))
            .max(sorted * u64::from(g.moe_intermediate))
            // The router's logits are one per expert, which for a 512-expert
            // mixture is narrower than everything above — but it is a real
            // activation and the bound must not depend on that staying true.
            .max(u64::from(g.n_experts))
            // The shared expert's SwiGLU stack: one row per token like any
            // dense activation, almost certainly narrower than the sorted
            // stack — but the bound must not rest on a checkpoint's choice
            // of widths.
            .max(u64::from(g.shared_intermediate));
    }
    widest
}

/// One scratch slot's element count for a fire of up to `max_tokens` rows.
///
/// The coloring (how many pool buffers, which dispatch binds which) is
/// N-invariant — it derives from the fixed DAG dataflow. Only the slot's
/// byte footprint scales: each ping-pong buffer holds `[max_tokens, widest]`
/// token-major.
///
/// The mixture does NOT scale with the token count, and this is the one
/// place in the pool where "n rows of the M=1 footprint" is the wrong
/// answer: the sort pads every touched expert's run to a whole tile, so a
/// batch of `n` tokens produces more than `n × experts_per_token` rows —
/// for a 512-expert mixture at 512 tokens it is 12800 rows where the linear
/// bound says 5120, a factor of two and a half. The overrun lands in the
/// next pool slot, which is another live activation.
#[must_use]
pub fn scratch_slot_elems(geometry: &DecodeGeometry, tuning: &Tuning, max_tokens: u32) -> u64 {
    let g = geometry;
    let n = max_tokens.max(1);
    let mut elems = scratch_widest_elems(g, tuning) * u64::from(n);
    if g.is_moe() {
        let wide = u64::from(g.hidden.max(g.moe_intermediate));
        let sorted = moe_sorted_rows(g, tuning, n, RoutedProjection::Matmul) * wide;
        elems = elems.max(sorted);
    }
    elems
}

// ─────────────────── the elastic step demand ────────────────────────────────

/// One paged-pool row's byte size: `[n_kv_heads, head_dim]` at bf16, which is
/// what `kv_append.metal` and `kv_append_paged.metal` both instantiate.
///
/// The C++ has this twice: once as `kv_pool_row_bytes(g)` and once inlined
/// eighty lines away inside `ensure_elastic_storage`, as
/// `page_size * n_kv_heads * head_dim * 2u`. Two spellings of one row width is
/// how a pool comes to be committed at a size its writer does not use.
#[must_use]
pub const fn kv_pool_row_bytes(geometry: &DecodeGeometry) -> u64 {
    geometry.n_kv_heads as u64 * geometry.head_dim as u64 * 2
}

/// A byte target for one elastic slot, and whether the slot could meet it.
///
/// The C++ takes `min(bytes, slot.size)` and returns nothing, so a demand
/// larger than the slot is answered by committing the whole slot and saying it
/// worked. The caller then runs against a buffer that is *not* the size it
/// asked for, and the first thing that notices is the arithmetic downstream.
/// [`Target::clamped`] is that fact, kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target {
    /// Bytes to commit — never more than the slot holds.
    pub bytes: u64,
    /// The demand exceeded the slot and was cut down to it.
    pub clamped: bool,
}

impl Target {
    /// Clamp `want` to `slot_bytes`, remembering whether it had to.
    #[must_use]
    pub const fn of(want: u64, slot_bytes: u64) -> Self {
        if want > slot_bytes {
            Target {
                bytes: slot_bytes,
                clamped: true,
            }
        } else {
            Target {
                bytes: want,
                clamped: false,
            }
        }
    }
}

/// The M=1 ring's per-layer target: the fraction of the layer's full context
/// this fire actually addresses.
///
/// `max_ctx` is the slot's full extent, so the ratio is exact rather than a
/// guess — but it is taken as `slot_bytes * min(tokens, max_ctx) / max_ctx`,
/// multiply before divide, so a context that is not a divisor of the byte size
/// still lands on the right side of the boundary.
#[must_use]
pub fn ring_target_bytes(slot_bytes: u64, ring_tokens: u32, max_ctx: u32) -> Target {
    let ctx = u64::from(max_ctx.max(1));
    let want = slot_bytes * u64::from(ring_tokens).min(ctx) / ctx;
    Target::of(want, slot_bytes)
}

/// The paged pool's per-layer target for `kv_pages` pages.
#[must_use]
pub fn kv_pool_target_bytes(
    geometry: &DecodeGeometry,
    page_size: u32,
    kv_pages: u32,
    slot_bytes: u64,
) -> Target {
    let page_bytes = u64::from(page_size) * kv_pool_row_bytes(geometry);
    Target::of(u64::from(kv_pages) * page_bytes, slot_bytes)
}

/// A GDN layer's convolution-state target for `state_slots` slots.
#[must_use]
pub fn conv_state_target_bytes(
    geometry: &DecodeGeometry,
    state_slots: u32,
    slot_bytes: u64,
) -> Target {
    Target::of(
        u64::from(state_slots) * geometry.gdn_conv_stride_bytes(),
        slot_bytes,
    )
}

/// A GDN layer's recurrent-state target for `state_slots` slots.
#[must_use]
pub fn recurrent_state_target_bytes(
    geometry: &DecodeGeometry,
    state_slots: u32,
    slot_bytes: u64,
) -> Target {
    Target::of(
        u64::from(state_slots) * geometry.gdn_recurrent_stride_bytes(),
        slot_bytes,
    )
}

/// A row-scaled slot's target: the pool and scratch buffers, whose size is a
/// function of how many token rows the fire actually carries.
///
/// Rounded **up**: a fire of one row out of a 512-row capacity still needs the
/// whole of whatever a single row touches, and rounding down would commit a
/// buffer one byte short of the row it is for. The C++'s
/// `(size * min(rows, cap) + cap - 1) / cap` is the same ceiling and is kept.
#[must_use]
pub fn row_scaled_target_bytes(slot_bytes: u64, token_rows: u32, max_tokens: u32) -> Target {
    let capacity = u64::from(max_tokens.max(1));
    let rows = u64::from(token_rows.max(1)).min(capacity);
    Target::of(
        slot_bytes.saturating_mul(rows).div_ceil(capacity),
        slot_bytes,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kv_geometry() -> DecodeGeometry {
        DecodeGeometry {
            n_kv_heads: 8,
            head_dim: 128,
            gdn_conv_dim: 64,
            gdn_conv_k: 4,
            gdn_v_heads: 2,
            gdn_v_dim: 16,
            gdn_k_dim: 16,
            ..DecodeGeometry::default()
        }
    }

    #[test]
    fn a_pool_page_is_the_row_width_times_the_page_size_and_the_row_width_has_one_spelling() {
        let g = kv_geometry();
        assert_eq!(kv_pool_row_bytes(&g), 8 * 128 * 2);
        // The whole point of the shared helper: the page target IS
        // `page_size * row_bytes`, not a second product that happens to agree.
        let t = kv_pool_target_bytes(&g, 32, 4, u64::MAX);
        assert_eq!(t.bytes, 4 * 32 * kv_pool_row_bytes(&g));
        assert!(!t.clamped);
    }

    #[test]
    fn a_demand_larger_than_its_slot_says_so_instead_of_quietly_becoming_the_slot() {
        // The C++ takes `min(bytes, slot.size)` and returns nothing, so the
        // caller runs against a buffer that is not the size it asked for.
        let g = kv_geometry();
        let slot = 1024;
        let t = kv_pool_target_bytes(&g, 32, 1000, slot);
        assert_eq!(t.bytes, slot);
        assert!(t.clamped, "the demand did not fit and the caller must know");
    }

    #[test]
    fn the_ring_target_is_the_fraction_of_the_context_this_fire_addresses() {
        // A quarter of the context is a quarter of the bytes.
        assert_eq!(ring_target_bytes(4096, 512, 2048).bytes, 1024);
        // And a fire longer than the context is the whole of it, not more.
        let t = ring_target_bytes(4096, 9000, 2048);
        assert_eq!(t.bytes, 4096);
        assert!(!t.clamped, "min against max_ctx already bounded it");
    }

    #[test]
    fn a_row_scaled_slot_rounds_up_so_a_single_row_is_never_a_byte_short() {
        // 100 bytes over 8 rows: one row needs 12.5, and 12 would be short.
        assert_eq!(row_scaled_target_bytes(100, 1, 8).bytes, 13);
        assert_eq!(row_scaled_target_bytes(100, 8, 8).bytes, 100);
        // More rows than the capacity is the capacity, not an overrun.
        assert_eq!(row_scaled_target_bytes(100, 99, 8).bytes, 100);
        // A zero row count is one row: a fire always carries something.
        assert_eq!(row_scaled_target_bytes(100, 0, 8).bytes, 13);
    }

    #[test]
    fn a_zero_capacity_cannot_divide_by_zero() {
        // Both clamps the C++ writes as `max(1, ...)`, kept and pinned.
        assert_eq!(row_scaled_target_bytes(64, 4, 0).bytes, 64);
        assert_eq!(ring_target_bytes(64, 4, 0).bytes, 64);
    }

    #[test]
    fn the_two_gdn_state_targets_scale_by_their_own_strides() {
        let g = kv_geometry();
        let conv = conv_state_target_bytes(&g, 3, u64::MAX);
        let rec = recurrent_state_target_bytes(&g, 3, u64::MAX);
        assert_eq!(conv.bytes, 3 * g.gdn_conv_stride_bytes());
        assert_eq!(rec.bytes, 3 * g.gdn_recurrent_stride_bytes());
        assert_ne!(
            conv.bytes, rec.bytes,
            "the conv and recurrent states are different shapes and must not share a formula"
        );
    }

    fn moe_geometry() -> DecodeGeometry {
        DecodeGeometry {
            n_experts: 512,
            experts_per_token: 10,
            moe_intermediate: 768,
            ..DecodeGeometry::default()
        }
    }

    #[test]
    fn the_sorted_bound_is_reached_and_tile_aligned() {
        // 256 pairs over 32 experts at a 16-row tile: every expert touched,
        // each wasting up to 15 rows. 256 + 32*15 = 736, already a tile
        // multiple.
        assert_eq!(sorted_rows(256, 32, 16), 736);
        // Not a multiple: 5 pairs over 2 experts at tile 4 -> 5 + 2*3 = 11,
        // rounded up to 12.
        assert_eq!(sorted_rows(5, 2, 4), 12);
        // Tile 1 is a pure grouping: one row per pair.
        assert_eq!(sorted_rows(256, 32, 1), 256);
        assert_eq!(sorted_rows(0, 32, 16), 0);
    }

    #[test]
    fn an_unbatched_sort_is_a_grouping_not_a_padding() {
        let geometry = moe_geometry();
        let tuning = Tuning::default();
        assert_eq!(
            moe_sorted_rows(&geometry, &tuning, 512, RoutedProjection::Matvec),
            5120
        );
    }

    #[test]
    fn the_mixtures_stack_does_not_scale_linearly_with_tokens() {
        // The doc's own numbers: a 512-expert mixture at 512 tokens, ten
        // slots each. 5120 pairs, ten rows per expert -> the 16-row tile;
        // all 512 experts touched, 5120 + 512*15 = 12800 rows where the
        // linear bound says 5120.
        let geometry = moe_geometry();
        let tuning = Tuning::default();
        assert_eq!(
            moe_sorted_rows(&geometry, &tuning, 512, RoutedProjection::Matmul),
            12800
        );
    }

    #[test]
    fn the_default_shapes_widest_row_is_the_gdn_in_projection() {
        let geometry = DecodeGeometry::default();
        let tuning = Tuning::default();
        // intermediate 3584, gdn prep 4160, gdn qk 4096, packed q 2048 — the
        // GDN in-projection's 6144 wins.
        assert_eq!(scratch_widest_elems(&geometry, &tuning), 6144);
    }

    #[test]
    fn the_slot_holds_the_mixture_or_the_batch_whichever_is_wider() {
        let geometry = moe_geometry();
        let tuning = Tuning::default();
        // M=1: ten pairs over 512 experts do not batch (per-expert rows are
        // zero), so the sort is ten rows and the widest term is the gathered
        // input, 10 * hidden.
        let widest = scratch_widest_elems(&geometry, &tuning);
        assert_eq!(widest, 10 * 1024);
        // At 512 tokens the linear answer is widest * 512; the padded stack
        // is 12800 * hidden, which is wider, and the slot must hold it.
        assert_eq!(
            scratch_slot_elems(&geometry, &tuning, 512),
            12800 * 1024,
            "the padded stack beats n rows of the M=1 footprint"
        );
        // Dense shapes scale linearly: no sort, no padding.
        let dense = DecodeGeometry::default();
        assert_eq!(
            scratch_slot_elems(&dense, &tuning, 8),
            scratch_widest_elems(&dense, &tuning) * 8
        );
    }
}

/// Which row axis a scratch value scales on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RowAxis {
    /// One row per token.
    #[default]
    Body,
    /// One row per SAMPLED row — the tail after the row gather.
    Tail,
    /// One row per sorted `(token, slot)` pair, tile-padded — the
    /// expert-major stack, TALLER than `rows × k`.
    Sorted,
}

/// One scratch value's shape: its producer's output shape. The pool is
/// measured in ACTIVATION elements (two bytes); an `int32` value counts
/// two per entry — sizing the ids like the weights hands the kernel a
/// buffer half the length it writes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ValueExtent {
    /// Elements per row.
    pub elems: u32,
    /// The row axis.
    pub axis: RowAxis,
}

/// How many elements each pool COLOUR must hold, for a `rows`-token fire
/// sampling `head_rows`.
///
/// Read off the VALUES rather than the dispatch kinds, because the two
/// are not the same question: a routed layer's expert stack is
/// `experts_per_token` times taller than the dense tensor it shares a
/// colour with, and a colour is sized by the widest value in it — sizing
/// by kind would under-allocate the stack and let the last expert's
/// projection run off the end of the buffer. `extent_of` answers for the
/// WRITER (every value has exactly one producer; reading the extent off
/// a consumer is ambiguous for anything read twice), and a value written
/// more than once — the ropes and qk-norms write in place — keeps its
/// widest claim, so an in-place pass can never shrink a slot under the
/// value already living in it.
#[must_use]
#[allow(clippy::too_many_arguments)] // one summation; a params struct would hide it
pub fn pool_colour_elems(
    dag: &[Dispatch],
    uses: &[Use],
    coloring: &Coloring,
    extent_of: impl Fn(&Dispatch) -> ValueExtent,
    rows: u32,
    head_rows: u32,
    sorted: u32,
    fallback_elems: u64,
) -> Vec<u64> {
    let rows = rows.max(1);
    let head_rows = if head_rows == 0 {
        rows
    } else {
        head_rows.min(rows)
    };
    // The widest claim per value first, then the widest value per colour.
    let mut extents: Vec<ValueExtent> = vec![ValueExtent::default(); coloring.color.len()];
    for u in uses {
        if !u.is_write {
            continue;
        }
        let Some(slot) = extents.get_mut(u.value as usize) else {
            continue;
        };
        let e = extent_of(&dag[u.ordinal as usize]);
        if e.elems > slot.elems {
            slot.elems = e.elems;
        }
        if e.axis == RowAxis::Sorted {
            slot.axis = RowAxis::Sorted;
        } else if e.axis == RowAxis::Tail && slot.axis == RowAxis::Body {
            slot.axis = RowAxis::Tail;
        }
    }
    let mut elems = vec![0u64; coloring.colors_used as usize];
    for (value, extent) in extents.iter().enumerate() {
        let Some(Some(colour)) = coloring.color.get(value) else {
            continue;
        };
        let n = match extent.axis {
            RowAxis::Body => rows,
            RowAxis::Tail => head_rows,
            RowAxis::Sorted => sorted,
        };
        let need = u64::from(n) * u64::from(extent.elems);
        let slot = &mut elems[*colour as usize];
        *slot = (*slot).max(need);
    }
    for e in &mut elems {
        if *e == 0 {
            *e = fallback_elems;
        }
    }
    elems
}
