//! The top-k cut `pivot_threshold(x, rank_le(k))` keeps, shared by every
//! host evaluator of it.

use alloc::vec::Vec;

/// The top-k cut of `row`: calls `keep(i)` for each `i` where `row[i]` is not
/// NaN and fewer than `k` non-NaN elements are strictly greater than it. NaN
/// is never kept; ties at the cut are all kept.
///
/// With `t` the `min(k, m)`-th largest of the `m` non-NaN values, an element
/// `v` has fewer than `k` elements above it iff `v >= t`: the elements above
/// `v` are then all above `t`, of which there are fewer than `min(k, m)`;
/// and a `v < t` has the `min(k, m)` largest above it, `k` of them when
/// `k <= m`, and when `k > m` every non-NaN `v >= t`. One selection decides
/// the row, O(V) expected, instead of a count per element.
pub fn rank_cut(row: &[f32], k: usize, mut keep: impl FnMut(usize)) {
    if k == 0 {
        return;
    }
    let mut live: Vec<f32> = row.iter().copied().filter(|v| !v.is_nan()).collect();
    let Some(at) = k.min(live.len()).checked_sub(1) else {
        return;
    };
    let (_, &mut t, _) = live.select_nth_unstable_by(at, |a, b| {
        b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal)
    });
    for (i, &v) in row.iter().enumerate() {
        if v >= t {
            keep(i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::rank_cut;
    use alloc::vec;
    use alloc::vec::Vec;

    /// The per-element count `rank_cut` replaced, its reference.
    fn rank_cut_count(row: &[f32], k: usize) -> Vec<bool> {
        let mut keep = vec![false; row.len()];
        for (i, &x) in row.iter().enumerate() {
            if x.is_nan() {
                continue;
            }
            let greater = row.iter().filter(|&&y| !y.is_nan() && y > x).count();
            keep[i] = greater < k;
        }
        keep
    }

    /// The selected top-k cut keeps exactly what the per-element count did:
    /// random rows with ties, NaN, ±inf and ±0 (a small value set, or every
    /// special), k at 0, 1, mid, len-1, len, past len and around the
    /// non-NaN count.
    #[test]
    fn the_selected_rank_cut_keeps_what_the_count_did() {
        let mut state = 0xD1B5_4A32_D192_ED03u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let small = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            1.0,
            2.0,
            -1.0,
        ];
        let mut rows: Vec<Vec<f32>> = vec![
            vec![],
            vec![f32::NAN; 5],
            vec![f32::INFINITY; 4],
            vec![f32::NEG_INFINITY, f32::NAN, f32::NEG_INFINITY],
            vec![-0.0, 0.0, -0.0, 0.0],
        ];
        for len in [1usize, 2, 3, 7, 16, 64, 129, 300] {
            for _ in 0..60 {
                rows.push(
                    (0..len)
                        .map(|_| small[usize::try_from(next() % 8).expect("below 8")])
                        .collect(),
                );
                rows.push(
                    (0..len)
                        .map(|_| match next() % 16 {
                            0 => f32::NAN,
                            1 => f32::INFINITY,
                            2 => f32::NEG_INFINITY,
                            3 => 0.0,
                            4 => -0.0,
                            5..=9 => (next() % 5) as f32 - 2.0,
                            _ => {
                                f32::from_bits(
                                    u32::try_from(next() >> 32).expect("32 bits") & 0x7f7f_ffff,
                                ) * if next() % 2 == 0 { 1.0 } else { -1.0 }
                            }
                        })
                        .collect(),
                );
            }
        }
        for row in &rows {
            let len = row.len();
            let live = row.iter().filter(|v| !v.is_nan()).count();
            let mut ks = vec![0, 1, len / 2, len.saturating_sub(1), len, len + 3];
            ks.extend([live.saturating_sub(1), live, live + 1]);
            for k in ks {
                let mut keep = vec![false; len];
                rank_cut(row, k, |i| keep[i] = true);
                assert_eq!(keep, rank_cut_count(row, k), "k={k} row={row:?}");
            }
        }
    }
}
