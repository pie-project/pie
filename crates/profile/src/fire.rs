//! What a device profile states for choosing a load's top rung of fires:
//! the device's rates (`[rates]`), which a fire's time is estimated from.
//! They live in the backend's device file (or the user's layer above it);
//! a higher layer replaces a lower one's key by key. `pie tune --rates`
//! measures them through the backend's own paths, at the shapes
//! `engine::rates` states; a vendor's peak, where one is cited, is a bound
//! in a comment, not a rate.
//!
//! ```toml
//! [rates]
//! # bf16 matmul flops a ns at m rows, n columns and depth k: every (n, k)
//! # stated at every row
//! bf16_flops_per_ns_by_rows_n_k = [
//!   [16, 1024, 1024, 600.0], [8192, 1024, 1024, 15000.0],
//!   [16, 4096, 4096, 1900.0], [8192, 4096, 4096, 26000.0],
//! ]
//! # prefill attention flops a ns by keys and head width: a full grid
//! attn_flops_per_ns_by_keys_and_dim = [[512, 128, 6000.0], [16384, 128, 11000.0]]
//! scan_elements_per_ns_by_tokens = [[512, 900.0], [16384, 1500.0]]  # scan + conv by tokens
//! bytes_per_ns = 726.0             # device memory bytes a ns (GB/s)
//! launch_ns = 3500                 # a dispatch's cost beside its work
//! ```

use std::fmt::Write as _;

/// A device's rates, each where a layer states it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Rates {
    /// Dense bf16 matmul flops a nanosecond by a matmul's rows, columns and
    /// depth: `(m, n, k, rate)`, every stated `(n, k)` at every stated row,
    /// ordered by `(k, n, m)` ([`gemm_grid`], [`Rates::flops_at`]).
    pub bf16_flops_per_ns_by_rows_n_k: Option<Vec<(u32, u32, u32, f64)>>,
    /// Attention flops a nanosecond by the keys a row attends and the head
    /// width: `(keys, d, rate)` over a full grid of keys by widths, ordered
    /// by width then keys ([`grid`]; the count the estimate's attention
    /// term takes, over a prefill's rows and keys).
    pub attn_flops_per_ns_by_keys_and_dim: Option<Vec<(u32, u32, f64)>>,
    /// Recurrent state elements a nanosecond by a prompt's tokens: `(tokens,
    /// rate)`, non-empty and strictly ascending (the scan's state elements a
    /// token, the conv beside it folded in).
    pub scan_elements_per_ns_by_tokens: Option<Vec<(u32, f64)>>,
    /// Device memory bytes a nanosecond.
    pub bytes_per_ns: Option<f64>,
    /// What a launch costs beside its work, in nanoseconds.
    pub launch_ns: Option<u64>,
}

impl Rates {
    /// `self` laid over `below`: each key `self` states replaces `below`'s.
    #[must_use]
    pub fn over(self, below: Rates) -> Rates {
        Rates {
            bf16_flops_per_ns_by_rows_n_k: self
                .bf16_flops_per_ns_by_rows_n_k
                .or(below.bf16_flops_per_ns_by_rows_n_k),
            attn_flops_per_ns_by_keys_and_dim: self
                .attn_flops_per_ns_by_keys_and_dim
                .or(below.attn_flops_per_ns_by_keys_and_dim),
            scan_elements_per_ns_by_tokens: self
                .scan_elements_per_ns_by_tokens
                .or(below.scan_elements_per_ns_by_tokens),
            bytes_per_ns: self.bytes_per_ns.or(below.bytes_per_ns),
            launch_ns: self.launch_ns.or(below.launch_ns),
        }
    }

    fn is_empty(&self) -> bool {
        *self == Rates::default()
    }

    /// The matmul rate at `rows`, `n` columns and depth `k` ([`gemm_at`]).
    /// `None` where no grid is stated.
    #[must_use]
    pub fn flops_at(&self, rows: u32, n: u32, k: u32) -> Option<f64> {
        gemm_at(self.bf16_flops_per_ns_by_rows_n_k.as_deref()?, rows, n, k)
    }

    /// The attention rate at `keys` and head width `d` ([`attn_flops_at`]:
    /// at `keys` exactly, in `log d` between widths). `None` where no grid
    /// is stated, or it states no `keys`.
    #[must_use]
    pub fn attn_at(&self, keys: u32, d: u32) -> Option<f64> {
        attn_flops_at(self.attn_flops_per_ns_by_keys_and_dim.as_deref()?, keys, d)
    }

    /// The scan rate at a prompt of `tokens` ([`interpolate`]). `None` where
    /// no curve is stated.
    #[must_use]
    pub fn scan_at(&self, tokens: u32) -> Option<f64> {
        interpolate(self.scan_elements_per_ns_by_tokens.as_deref()?, tokens)
    }

    /// The grid's fastest point: the rate a compute-bound matmul reaches.
    #[must_use]
    pub fn peak_flops(&self) -> Option<f64> {
        self.bf16_flops_per_ns_by_rows_n_k
            .as_deref()?
            .iter()
            .map(|&(_, _, _, r)| r)
            .filter(|r| r.is_finite())
            .reduce(f64::max)
    }

    /// `[rates]` as profile TOML.
    #[must_use]
    pub fn toml(&self) -> String {
        let mut out = String::from("[rates]\n");
        if let Some(grid) = &self.bf16_flops_per_ns_by_rows_n_k {
            let _ = writeln!(out, "bf16_flops_per_ns_by_rows_n_k = [");
            for (m, n, k, r) in grid {
                let _ = writeln!(out, "  [{m}, {n}, {k}, {r:?}],");
            }
            let _ = writeln!(out, "]");
        }
        if let Some(grid) = &self.attn_flops_per_ns_by_keys_and_dim {
            let _ = writeln!(out, "attn_flops_per_ns_by_keys_and_dim = [");
            for (keys, d, r) in grid {
                let _ = writeln!(out, "  [{keys}, {d}, {r:?}],");
            }
            let _ = writeln!(out, "]");
        }
        if let Some(curve) = &self.scan_elements_per_ns_by_tokens {
            let points: Vec<String> = curve.iter().map(|(t, r)| format!("[{t}, {r:?}]")).collect();
            let _ = writeln!(
                out,
                "scan_elements_per_ns_by_tokens = [{}]",
                points.join(", ")
            );
        }
        if let Some(v) = self.bytes_per_ns {
            let _ = writeln!(out, "bytes_per_ns = {v:?}");
        }
        if let Some(v) = self.launch_ns {
            let _ = writeln!(out, "launch_ns = {v}");
        }
        out
    }
}

/// `grid` (`(m, k, rate)`, a full grid ordered by depth then rows) checked
/// and ordered: `None` where it is empty, a point is not positive, or a
/// depth lacks a row another states (or states one twice).
#[must_use]
pub fn grid(mut points: Vec<(u32, u32, f64)>) -> Option<Vec<(u32, u32, f64)>> {
    if points.is_empty()
        || points
            .iter()
            .any(|&(m, k, r)| m == 0 || k == 0 || !(r > 0.0))
    {
        return None;
    }
    points.sort_by_key(|&(m, k, _)| (k, m));
    let mut rows: Vec<u32> = points.iter().map(|&(m, _, _)| m).collect();
    rows.sort_unstable();
    rows.dedup();
    let mut depths: Vec<u32> = points.iter().map(|&(_, k, _)| k).collect();
    depths.dedup();
    let full = points.len() == rows.len() * depths.len()
        && points
            .iter()
            .enumerate()
            .all(|(i, &(m, k, _))| m == rows[i % rows.len()] && k == depths[i / rows.len()]);
    full.then_some(points)
}

/// The rate of `grid` (checked by [`grid`]) at `rows` and `depth`:
/// interpolated linearly in `log(rows)` along each depth's row curve, then
/// in `log(depth)` between the two depths around `depth`, and clamped to
/// the grid's edges past them. `None` for an empty grid.
#[must_use]
pub fn flops_at(grid: &[(u32, u32, f64)], rows: u32, depth: u32) -> Option<f64> {
    let mut depths: Vec<u32> = grid.iter().map(|&(_, k, _)| k).collect();
    depths.dedup();
    let along = |k: u32| -> Option<f64> {
        let curve: Vec<(u32, f64)> = grid
            .iter()
            .filter(|&&(_, kk, _)| kk == k)
            .map(|&(m, _, r)| (m, r))
            .collect();
        interpolate(&curve, rows)
    };
    let (&k0, &k1) = (depths.first()?, depths.last()?);
    if depth <= k0 {
        return along(k0);
    }
    if depth >= k1 {
        return along(k1);
    }
    let at = depths.partition_point(|&k| k <= depth);
    let (lo, hi) = (depths[at - 1], depths[at]);
    Some(lerp_log((lo, along(lo)?), (hi, along(hi)?), depth))
}

/// The attention rate of `grid` (`(keys, d, rate)`, grouped by `d`
/// ascending) at `keys` and head width `d`: each width's point at `keys`
/// exactly (the keys are the rung lattice the step timed, a kernel's wave
/// quantization moving the rate between them, so none is read between),
/// then linear in `log d` between widths and clamped past them. `None`
/// where a width states no `keys`.
#[must_use]
pub fn attn_flops_at(grid: &[(u32, u32, f64)], keys: u32, d: u32) -> Option<f64> {
    let mut widths: Vec<u32> = grid.iter().map(|&(_, w, _)| w).collect();
    widths.sort_unstable();
    widths.dedup();
    let along = |w: u32| -> Option<f64> {
        grid.iter()
            .find(|&&(k, ww, _)| k == keys && ww == w)
            .map(|&(_, _, r)| r)
    };
    let (&w0, &w1) = (widths.first()?, widths.last()?);
    if d <= w0 {
        return along(w0);
    }
    if d >= w1 {
        return along(w1);
    }
    let at = widths.partition_point(|&w| w <= d);
    let (lo, hi) = (widths[at - 1], widths[at]);
    Some(lerp_log((lo, along(lo)?), (hi, along(hi)?), d))
}

/// The keys an attention grid states, ascending.
#[must_use]
pub fn attn_keys(grid: &[(u32, u32, f64)]) -> Vec<u32> {
    let mut keys: Vec<u32> = grid.iter().map(|&(k, ..)| k).collect();
    keys.sort_unstable();
    keys.dedup();
    keys
}

/// `points` (`(m, n, k, rate)`) checked and ordered by `(k, n, m)`: `None`
/// where it is empty, a point is not positive, or a stated `(n, k)` lacks a
/// row another states (or states one twice). The `(n, k)` pairs are the
/// file's own: a full product of columns by depths, or (a file measured
/// square) its diagonal.
#[must_use]
pub fn gemm_grid(mut points: Vec<(u32, u32, u32, f64)>) -> Option<Vec<(u32, u32, u32, f64)>> {
    if points.is_empty()
        || points
            .iter()
            .any(|&(m, n, k, r)| m == 0 || n == 0 || k == 0 || !(r > 0.0))
    {
        return None;
    }
    points.sort_by_key(|&(m, n, k, _)| (k, n, m));
    let mut rows: Vec<u32> = points.iter().map(|&(m, ..)| m).collect();
    rows.sort_unstable();
    rows.dedup();
    let mut pairs: Vec<(u32, u32)> = points.iter().map(|&(_, n, k, _)| (k, n)).collect();
    pairs.dedup();
    let full = points.len() == rows.len() * pairs.len()
        && points
            .iter()
            .enumerate()
            .all(|(i, &(m, n, k, _))| m == rows[i % rows.len()] && (k, n) == pairs[i / rows.len()]);
    full.then_some(points)
}

/// The rows `grid` states, ascending: the rung lattice it was timed at.
#[must_use]
pub fn gemm_rows(grid: &[(u32, u32, u32, f64)]) -> Vec<u32> {
    let mut rows: Vec<u32> = grid.iter().map(|&(m, ..)| m).collect();
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// The rate of `grid` (checked by [`gemm_grid`]) at `rows`, `n` and `k`:
/// each stated `(n, k)` read at `rows` exactly (`None` for a row the grid
/// does not state: the rows are a rung lattice, not interpolated), then,
/// where the pairs are a full product of columns by depths,
/// interpolated in `log n` and `log k` between the four around `(n, k)`;
/// where they are not (a square grid's diagonal), in `log k` between the
/// depths around `k`. Clamped to the grid's edges past them. `None` for an
/// empty grid.
#[must_use]
pub fn gemm_at(grid: &[(u32, u32, u32, f64)], rows: u32, n: u32, k: u32) -> Option<f64> {
    // A row count is read where the grid states it, never between: the
    // rows are the rung lattice a load lays its fires at, each timed through
    // the kernel serving runs (its own padding and tile in the rate).
    let along = |nn: u32, kk: u32| -> Option<f64> {
        grid.iter()
            .find(|&&(m, gn, gk, _)| m == rows && gn == nn && gk == kk)
            .map(|&(_, _, _, r)| r)
    };
    let mut pairs: Vec<(u32, u32)> = grid.iter().map(|&(_, n, k, _)| (n, k)).collect();
    pairs.sort_unstable();
    pairs.dedup();
    let mut ns: Vec<u32> = pairs.iter().map(|&(n, _)| n).collect();
    ns.sort_unstable();
    ns.dedup();
    let mut ks: Vec<u32> = pairs.iter().map(|&(_, k)| k).collect();
    ks.sort_unstable();
    ks.dedup();
    // The two stated values around `x` (one where `x` is past an end).
    let around = |axis: &[u32], x: u32| -> Option<(u32, u32)> {
        let (&first, &last) = (axis.first()?, axis.last()?);
        if x <= first {
            return Some((first, first));
        }
        if x >= last {
            return Some((last, last));
        }
        let at = axis.partition_point(|&a| a <= x);
        Some((axis[at - 1], axis[at]))
    };
    let blend = |(lo, a): (u32, f64), (hi, b): (u32, f64), x: u32| {
        if lo == hi {
            a
        } else {
            lerp_log((lo, a), (hi, b), x.clamp(lo, hi))
        }
    };
    if pairs.len() == ns.len() * ks.len() {
        let (k0, k1) = around(&ks, k)?;
        let (n0, n1) = around(&ns, n)?;
        let at_k =
            |kk: u32| -> Option<f64> { Some(blend((n0, along(n0, kk)?), (n1, along(n1, kk)?), n)) };
        return Some(blend((k0, at_k(k0)?), (k1, at_k(k1)?), k));
    }
    // Not a full product: each depth at its stated columns nearest `n` in
    // log, then in `log k`.
    let nearest = |kk: u32| -> Option<f64> {
        let ln = |x: u32| f64::from(x.max(1)).ln();
        let nn = pairs
            .iter()
            .filter(|&&(_, pk)| pk == kk)
            .map(|&(pn, _)| pn)
            .min_by(|&a, &b| (ln(a) - ln(n)).abs().total_cmp(&(ln(b) - ln(n)).abs()))?;
        along(nn, kk)
    };
    let (k0, k1) = around(&ks, k)?;
    Some(blend((k0, nearest(k0)?), (k1, nearest(k1)?), k))
}

/// `curve` (`(x, rate)`, ascending in `x`) at `x`, linear in `log x`,
/// clamped at its ends. `None` for an empty curve.
#[must_use]
pub fn interpolate(curve: &[(u32, f64)], x: u32) -> Option<f64> {
    let (&(x0, r0), &(x1, r1)) = (curve.first()?, curve.last()?);
    if x <= x0 {
        return Some(r0);
    }
    if x >= x1 {
        return Some(r1);
    }
    let at = curve.partition_point(|&(m, _)| m <= x);
    Some(lerp_log(curve[at - 1], curve[at], x))
}

fn lerp_log((lo, a): (u32, f64), (hi, b): (u32, f64), x: u32) -> f64 {
    let ln = |m: u32| f64::from(m.max(1)).ln();
    let t = (ln(x) - ln(lo)) / (ln(hi) - ln(lo));
    if a.is_infinite() || b.is_infinite() {
        // A gap-bound point (no work at that size): the time a unit of work
        // takes, not the rate, between the two.
        let per = (1.0 / a) + ((1.0 / b) - (1.0 / a)) * t;
        return if per > 0.0 { 1.0 / per } else { f64::INFINITY };
    }
    a + (b - a) * t
}

/// `(tokens, ns)` of a fire at each rung width, ascending in tokens: what a
/// check of the estimate measures. Never a profile's.
pub type FireTable = Vec<(u32, u64)>;

/// What a profile states beside its decisions: its rates.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Tuned {
    pub rates: Rates,
    /// The rates were measured on the device by this load (a browser's,
    /// whose masked adapter no profile names), not read from a layer: they
    /// are this load's alone and are never written.
    pub measured_this_load: bool,
}

impl Tuned {
    /// `upper` laid over `self`.
    pub fn lay(&mut self, upper: &Tuned) {
        self.rates = upper.rates.clone().over(std::mem::take(&mut self.rates));
        self.measured_this_load |= upper.measured_this_load;
    }

    /// `[rates]` as profile TOML, nothing where nothing is stated or the
    /// rates were measured this load.
    #[must_use]
    pub fn toml(&self) -> String {
        if self.rates.is_empty() || self.measured_this_load {
            return String::new();
        }
        format!("\n{}", self.rates.toml())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_upper_layer_replaces_key_by_key() {
        let mut low = Tuned {
            rates: Rates {
                bf16_flops_per_ns_by_rows_n_k: Some(vec![(4096, 4096, 4096, 1.0)]),
                bytes_per_ns: Some(2.0),
                ..Rates::default()
            },
            measured_this_load: false,
        };
        let up = Tuned {
            rates: Rates {
                bytes_per_ns: Some(3.0),
                launch_ns: Some(7),
                ..Rates::default()
            },
            measured_this_load: false,
        };
        low.lay(&up);
        assert_eq!(
            low.rates,
            Rates {
                bf16_flops_per_ns_by_rows_n_k: Some(vec![(4096, 4096, 4096, 1.0)]),
                bytes_per_ns: Some(3.0),
                launch_ns: Some(7),
                ..Rates::default()
            }
        );
    }

    #[test]
    fn the_gemm_grid_reads_its_rows_exactly_and_interpolates_columns_and_depth() {
        // A full product: n and k each 1024 and 4096, rows 16 and 256.
        let mut points = Vec::new();
        for (n, k, base) in [
            (1024, 1024, 100.0),
            (4096, 1024, 200.0),
            (1024, 4096, 300.0),
            (4096, 4096, 400.0),
        ] {
            points.push((16, n, k, base));
            points.push((256, n, k, 2.0 * base));
        }
        let rates = Rates {
            bf16_flops_per_ns_by_rows_n_k: gemm_grid(points),
            ..Rates::default()
        };
        assert_eq!(rates.flops_at(16, 1024, 1024), Some(100.0));
        assert_eq!(
            rates.flops_at(256, 1 << 20, 1 << 20),
            Some(800.0),
            "n, k clamped"
        );
        // A row the grid does not state is no rate: never interpolated.
        assert_eq!(rates.flops_at(64, 1024, 1024), None);
        assert_eq!(rates.flops_at(1, 1024, 1024), None);
        assert_eq!(
            gemm_rows(rates.bf16_flops_per_ns_by_rows_n_k.as_deref().unwrap()),
            vec![16, 256]
        );
        // n 2048 halfway from 1024 to 4096: 150 at k 1024; k 2048 the same.
        assert!((rates.flops_at(16, 2048, 1024).unwrap() - 150.0).abs() < 1e-9);
        assert!((rates.flops_at(16, 2048, 2048).unwrap() - 250.0).abs() < 1e-9);
        // An 8B gate_up (n 24576, k 4096) reads the widest columns stated.
        assert_eq!(rates.flops_at(256, 24576, 4096), Some(800.0));
        assert_eq!(rates.peak_flops(), Some(800.0));
        assert_eq!(Rates::default().flops_at(16, 1024, 1024), None);
    }

    #[test]
    fn the_attention_reads_its_keys_exactly_and_interpolates_the_width() {
        // Keys 2234 and 2436 (a lattice's rows) at widths 64 and 256.
        let grid = vec![
            (2234, 64, 100.0),
            (2436, 64, 120.0),
            (2234, 256, 400.0),
            (2436, 256, 480.0),
        ];
        assert_eq!(attn_flops_at(&grid, 2234, 64), Some(100.0));
        assert_eq!(attn_flops_at(&grid, 2436, 512), Some(480.0), "width clamped");
        // Width 128 halfway from 64 to 256 in log width.
        assert!((attn_flops_at(&grid, 2234, 128).unwrap() - 250.0).abs() < 1e-9);
        // Keys between the rows: no rate, never read between them.
        assert_eq!(attn_flops_at(&grid, 2300, 64), None);
        assert_eq!(attn_keys(&grid), vec![2234, 2436]);
    }

    #[test]
    fn a_gap_bound_point_reads_as_no_work() {
        // Depth 1024 gap-bound at 16 rows (its work unmeasurable under a
        // launch), 4096 at 100 a ns: 2048 halfway in log depth, half 4096's
        // time a unit.
        let grid = gemm_grid(vec![
            (16, 1024, 1024, f64::INFINITY),
            (16, 4096, 4096, 100.0),
        ])
        .unwrap();
        assert_eq!(gemm_at(&grid, 16, 1024, 1024), Some(f64::INFINITY));
        assert!((gemm_at(&grid, 16, 2048, 2048).unwrap() - 200.0).abs() < 1e-9);
        let rates = Rates {
            bf16_flops_per_ns_by_rows_n_k: Some(grid),
            ..Rates::default()
        };
        assert_eq!(rates.peak_flops(), Some(100.0));
        assert!(
            rates.toml().contains("[16, 1024, 1024, inf]"),
            "{}",
            rates.toml()
        );
    }

    #[test]
    fn a_square_grid_reads_its_diagonal_by_depth() {
        // A file measured n = k: (1024, 1024) and (4096, 4096) only.
        let grid = gemm_grid(vec![(16, 1024, 1024, 100.0), (16, 4096, 4096, 300.0)]).unwrap();
        assert_eq!(gemm_at(&grid, 16, 6144, 1024), Some(100.0), "k 1024's");
        assert!((gemm_at(&grid, 16, 4096, 2048).unwrap() - 200.0).abs() < 1e-9);
    }

    #[test]
    fn a_gemm_grid_states_every_row_at_each_of_its_pairs_or_is_refused() {
        assert_eq!(
            gemm_grid(vec![(256, 64, 64, 2.0), (16, 64, 64, 1.0)]),
            Some(vec![(16, 64, 64, 1.0), (256, 64, 64, 2.0)])
        );
        assert_eq!(gemm_grid(vec![]), None, "empty");
        assert_eq!(
            gemm_grid(vec![
                (16, 64, 64, 1.0),
                (256, 64, 64, 2.0),
                (16, 128, 64, 3.0)
            ]),
            None,
            "a pair short of a row"
        );
        assert_eq!(gemm_grid(vec![(16, 64, 64, 0.0)]), None, "not positive");
    }

    #[test]
    fn a_grid_is_whole_or_refused() {
        // Ordered by depth then rows, whatever order it came in.
        assert_eq!(
            grid(vec![
                (256, 4096, 4.0),
                (16, 4096, 3.0),
                (256, 1024, 2.0),
                (16, 1024, 1.0)
            ]),
            Some(vec![
                (16, 1024, 1.0),
                (256, 1024, 2.0),
                (16, 4096, 3.0),
                (256, 4096, 4.0)
            ])
        );
        assert_eq!(grid(vec![]), None, "empty");
        assert_eq!(
            grid(vec![(16, 1024, 1.0), (256, 1024, 2.0), (16, 4096, 3.0)]),
            None,
            "a hole"
        );
        assert_eq!(grid(vec![(16, 1024, 1.0), (16, 1024, 2.0)]), None, "twice");
        assert_eq!(grid(vec![(16, 1024, -1.0)]), None, "not positive");
        assert_eq!(
            grid(vec![(16, 1024, 1.0)]),
            Some(vec![(16, 1024, 1.0)]),
            "one point"
        );
    }
}
