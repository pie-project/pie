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
//! # bf16 matmul flops a ns at m rows and depth k (m x k x k): a full grid
//! bf16_flops_per_ns_by_rows_and_depth = [
//!   [16, 1024, 600.0], [8192, 1024, 15000.0],
//!   [16, 4096, 1900.0], [8192, 4096, 26000.0],
//! ]
//! attn_flops_per_ns_by_keys = [[512, 6000.0], [16384, 11000.0]]  # prefill attention by keys
//! scan_elements_per_ns = 1500.0    # recurrent state elements a ns (scan + conv)
//! bytes_per_ns = 726.0             # device memory bytes a ns (GB/s)
//! launch_ns = 3500                 # a dispatch's cost beside its work
//! ```

use std::fmt::Write as _;

/// A device's rates, each where a layer states it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Rates {
    /// Dense bf16 matmul flops a nanosecond by a matmul's rows and depth:
    /// `(m, k, rate)` over a full grid of rows by depths, ordered by depth
    /// then rows ([`Rates::flops_at`]).
    pub bf16_flops_per_ns_by_rows_and_depth: Option<Vec<(u32, u32, f64)>>,
    /// Attention flops a nanosecond by the keys a row attends: `(keys,
    /// rate)`, non-empty and strictly ascending in keys (the count the
    /// estimate's attention term takes, over a prefill's rows and keys).
    pub attn_flops_per_ns_by_keys: Option<Vec<(u32, f64)>>,
    /// Recurrent state elements a nanosecond: the scan's state elements a
    /// token, the conv beside it folded in.
    pub scan_elements_per_ns: Option<f64>,
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
            bf16_flops_per_ns_by_rows_and_depth: self
                .bf16_flops_per_ns_by_rows_and_depth
                .or(below.bf16_flops_per_ns_by_rows_and_depth),
            attn_flops_per_ns_by_keys: self
                .attn_flops_per_ns_by_keys
                .or(below.attn_flops_per_ns_by_keys),
            scan_elements_per_ns: self.scan_elements_per_ns.or(below.scan_elements_per_ns),
            bytes_per_ns: self.bytes_per_ns.or(below.bytes_per_ns),
            launch_ns: self.launch_ns.or(below.launch_ns),
        }
    }

    fn is_empty(&self) -> bool {
        *self == Rates::default()
    }

    /// The matmul rate at `rows` and depth `depth` ([`flops_at`]). `None`
    /// where no grid is stated.
    #[must_use]
    pub fn flops_at(&self, rows: u32, depth: u32) -> Option<f64> {
        flops_at(
            self.bf16_flops_per_ns_by_rows_and_depth.as_deref()?,
            rows,
            depth,
        )
    }

    /// The attention rate at `keys` ([`interpolate`]). `None` where no
    /// curve is stated.
    #[must_use]
    pub fn attn_at(&self, keys: u32) -> Option<f64> {
        interpolate(self.attn_flops_per_ns_by_keys.as_deref()?, keys)
    }

    /// The grid's fastest point: the rate a compute-bound matmul reaches.
    #[must_use]
    pub fn peak_flops(&self) -> Option<f64> {
        self.bf16_flops_per_ns_by_rows_and_depth
            .as_deref()?
            .iter()
            .map(|&(_, _, r)| r)
            .reduce(f64::max)
    }

    /// `[rates]` as profile TOML.
    #[must_use]
    pub fn toml(&self) -> String {
        let mut out = String::from("[rates]\n");
        if let Some(grid) = &self.bf16_flops_per_ns_by_rows_and_depth {
            let _ = writeln!(out, "bf16_flops_per_ns_by_rows_and_depth = [");
            for (m, k, r) in grid {
                let _ = writeln!(out, "  [{m}, {k}, {r:?}],");
            }
            let _ = writeln!(out, "]");
        }
        if let Some(curve) = &self.attn_flops_per_ns_by_keys {
            let points: Vec<String> = curve.iter().map(|(k, r)| format!("[{k}, {r:?}]")).collect();
            let _ = writeln!(out, "attn_flops_per_ns_by_keys = [{}]", points.join(", "));
        }
        if let Some(v) = self.scan_elements_per_ns {
            let _ = writeln!(out, "scan_elements_per_ns = {v:?}");
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
            .any(|&(m, k, r)| m == 0 || k == 0 || !(r.is_finite() && r > 0.0))
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
    a + (b - a) * (ln(x) - ln(lo)) / (ln(hi) - ln(lo))
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
                bf16_flops_per_ns_by_rows_and_depth: Some(vec![(4096, 4096, 1.0)]),
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
                bf16_flops_per_ns_by_rows_and_depth: Some(vec![(4096, 4096, 1.0)]),
                bytes_per_ns: Some(3.0),
                launch_ns: Some(7),
                ..Rates::default()
            }
        );
    }

    #[test]
    fn the_grid_interpolates_in_log_rows_and_log_depth_and_clamps_at_its_edges() {
        let rates = Rates {
            bf16_flops_per_ns_by_rows_and_depth: grid(vec![
                (16, 1024, 100.0),
                (256, 1024, 500.0),
                (16, 4096, 300.0),
                (256, 4096, 900.0),
            ]),
            ..Rates::default()
        };
        assert_eq!(rates.flops_at(16, 1024), Some(100.0));
        assert_eq!(rates.flops_at(1, 1), Some(100.0), "clamped below both");
        assert_eq!(
            rates.flops_at(8192, 8192),
            Some(900.0),
            "clamped above both"
        );
        // 64 is halfway from 16 to 256 in log rows; 2048 from 1024 to 4096
        // in log depth.
        assert!((rates.flops_at(64, 1024).unwrap() - 300.0).abs() < 1e-9);
        assert!((rates.flops_at(16, 2048).unwrap() - 200.0).abs() < 1e-9);
        assert!((rates.flops_at(64, 2048).unwrap() - 450.0).abs() < 1e-9);
        assert_eq!(rates.peak_flops(), Some(900.0));
        assert_eq!(Rates::default().flops_at(64, 1024), None);
    }

    #[test]
    fn the_attention_curve_interpolates_in_log_keys_and_clamps() {
        let rates = Rates {
            attn_flops_per_ns_by_keys: Some(vec![(512, 100.0), (2048, 300.0), (8192, 400.0)]),
            ..Rates::default()
        };
        assert_eq!(rates.attn_at(1), Some(100.0), "a decode lane's one key");
        assert!((rates.attn_at(1024).unwrap() - 200.0).abs() < 1e-9);
        assert_eq!(rates.attn_at(2048), Some(300.0));
        assert_eq!(rates.attn_at(1 << 20), Some(400.0));
        assert_eq!(Rates::default().attn_at(2048), None);
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
