//! The integer arithmetic, with the C++ conversions spelled out.
//!
//! # Why these are functions and not expressions
//!
//! `ceil_div(x, y)` is `(x + y - 1) / y` in `flashinfer/utils.cuh`, and that is
//! not the whole specification: it is a template whose **return type is the
//! first argument's**, so `ceil_div(int32_t, uint32_t)` computes in `unsigned`
//! (usual arithmetic conversions) and then truncates back to `int32_t`, while
//! `ceil_div(int64_t, uint32_t)` computes in `int64_t` and does not. Two call
//! sites in `scheduler.cuh` differ by exactly that, and the difference decides
//! how many KV chunks a request is split into.
//!
//! Rust has no usual arithmetic conversions, which is a feature everywhere
//! except here. So each C++ instantiation the scheduler actually uses gets a
//! named function whose body is the widened computation, and the name says
//! which instantiation it is. A caller that reaches for `x.div_ceil(y)` instead
//! is writing a different function — `div_ceil` cannot overflow and this one
//! can, and upstream *relies* on the wrap in at least one place
//! ([`ceil_div_u32`] with `y = 0` is upstream UB, but `x + y - 1` wrapping is
//! reachable with large page counts).
//!
//! # Wrapping is the port, not a shortcut
//!
//! Every `wrapping_*` here mirrors an unsigned C++ expression, where wrapping
//! is defined behaviour and upstream depends on it (`total_num_rows -
//! batch_size + 1` in the CUDA-graph branch of the prefill splitter is
//! evaluated in `uint32_t` and wraps for an empty batch). Rust's default
//! arithmetic would panic in debug and wrap in release, which is the one
//! outcome worse than either: a planner that agrees with the C++ in the
//! profile-guided build and aborts in the debug build.

/// `ceil_div<uint32_t, uint32_t>` — the plain unsigned case.
///
/// `(x + y - 1) / y` in `uint32_t`, wrap included. Panics on `y == 0` where
/// C++ is undefined; no call site in the port can reach it, because every
/// divisor is a page count, tile size or chunk size that upstream has already
/// clamped to at least 1.
#[must_use]
pub const fn ceil_div_u32(x: u32, y: u32) -> u32 {
    x.wrapping_add(y).wrapping_sub(1) / y
}

/// `ceil_div<int32_t, uint32_t>` — computed in `unsigned`, truncated back.
///
/// This is `ceil_div(elem, mid)` inside
/// `PartitionPagedKVCacheBinarySearchMinNumPagePerBatch`, where `elem` is an
/// `IdType` page count and `mid` is a `uint32_t` chunk size. The C++ promotes
/// `elem` to `unsigned`, divides, and converts the `unsigned` result back to
/// the template's `T1 = int32_t`. A negative page count therefore becomes an
/// enormous quotient rather than a negative one — which is upstream's
/// behaviour, and the reason a negative span is refused before it gets here.
#[must_use]
pub const fn ceil_div_i32_in_u32(x: i32, y: u32) -> i32 {
    ((x as u32).wrapping_add(y).wrapping_sub(1) / y) as i32
}

/// `ceil_div<int64_t, ...>` — the widened case.
///
/// Used wherever upstream has already promoted to `int64_t`: packed QO lengths,
/// effective KV lengths, chunk counts. Signed division truncates toward zero in
/// both languages, so negatives agree without a cast.
#[must_use]
pub const fn ceil_div_i64(x: i64, y: i64) -> i64 {
    x.wrapping_add(y).wrapping_sub(1) / y
}

/// `ceil_div<int, int>` — the 32-bit signed case the SM90 and MLA schedulers
/// use for tile counts.
#[must_use]
pub const fn ceil_div_i32(x: i32, y: i32) -> i32 {
    x.wrapping_add(y).wrapping_sub(1) / y
}

/// `FA2DetermineCtaTileQ` — the tile width every prefill count is derived from.
///
/// The one place a planner reads compute capability, and it reads it only on
/// the `head_dim < 256`, `avg_packed_qo_len <= 64` path: pre-Ampere returns 64
/// unconditionally, Ampere-or-newer splits 16/64 at an average packed QO
/// length of 16. Getting this wrong does not fail — it changes `cta_tile_q`,
/// which changes `total_num_tiles_q`, `padded_batch_size`, the size of the
/// partial-output carve, and every tile index in the plan.
///
/// `cc_major` replaces `GetCudaComputeCapability().first`; the minor version is
/// not read by this function or any other in the file.
#[must_use]
pub const fn fa2_determine_cta_tile_q(avg_packed_qo_len: i64, head_dim: u32, cc_major: i32) -> u32 {
    if head_dim >= 512 {
        // decode / short-q (incl. speculative decode): lean CTA16
        if avg_packed_qo_len <= 32 {
            return 16;
        }
        // Long-q prefill uses CTA_TILE_Q=32
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
///
/// `2 * float(qo_len) + kv_len`, in `f32`, in that order. The order matters and
/// the type matters: these values accumulate into a per-CTA total that decides
/// heap ordering, and past 2^24 the additions round. Both languages round to
/// nearest-even and neither contracts this into an FMA (the multiply is by a
/// power of two and therefore exact), so the accumulated totals agree bit for
/// bit — which is what makes the CTA assignment agree.
#[must_use]
pub fn cost_function(qo_len: i32, kv_len: i32) -> f32 {
    2.0 * (qo_len as f32) + (kv_len as f32)
}

/// `packed_causal_kv_end` — how much KV a causal QO tile actually reads.
///
/// The last tile reads everything; an earlier tile reads up to the diagonal,
/// right-aligned (`kv_len - qo_len`) plus its own rows. Everything is `int`
/// here, including the `ceil_div`, exactly as upstream — a widening would
/// change nothing for real sequences and would be a lie about what runs.
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
    // right aligned
    let kv_len_init = kv_len - qo_len;
    let end = kv_len_init
        .wrapping_add(ceil_div_i32((qo_tile_idx + 1).wrapping_mul(cluster_tile_q), group_size));
    let clamped = if end < kv_len { end } else { kv_len };
    if clamped > 0 { clamped } else { 0 }
}

/// `max(uint32_t, int32_t)` — CUDA's overload, not `std::max`.
///
/// `PartitionPagedKVCacheBinarySearchMinNumPagePerBatch` writes
/// `high = max(high, elem)` with `high` a `uint32_t` and `elem` an `IdType`.
/// That does not resolve to `std::max` at all (the template arguments differ);
/// it resolves to CUDA's `unsigned int max(unsigned int, int)` from
/// `crt/math_functions.hpp`, which **converts the int to unsigned first**. A
/// negative page count therefore wins the maximum rather than losing it. We do
/// not rely on that — a negative span is refused earlier — but the port says
/// what runs.
#[must_use]
pub const fn cuda_max_u32_i32(a: u32, b: i32) -> u32 {
    let b = b as u32;
    if a > b { a } else { b }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three `ceil_div` instantiations disagree, which is why there are
    /// three of them.
    #[test]
    fn ceil_div_conversions_are_not_interchangeable() {
        assert_eq!(ceil_div_u32(9, 4), 3);
        assert_eq!(ceil_div_u32(8, 4), 2);
        assert_eq!(ceil_div_u32(0, 4), 0);
        // -1 page: the unsigned instantiation makes it a huge quotient, the
        // signed one makes it zero. Same C++ source text, different answers.
        assert_eq!(ceil_div_i32_in_u32(-1, 4), 0);
        assert_eq!(ceil_div_i64(-1, 4), 0);
        // -8 pages: 0xFFFFFFF8 rounded up to a multiple of 4, back into i32.
        assert_eq!(ceil_div_i32_in_u32(-8, 4), 1_073_741_822);
        assert_eq!(ceil_div_i64(-8, 4), -1);
    }

    /// The compute-capability branch is the only device fact in the file, and
    /// it moves the tile width by 4x.
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
