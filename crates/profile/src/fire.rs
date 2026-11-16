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
//! # bf16 matmul flops a ns at m rows (m x 4096 x 4096), ascending in m
//! bf16_flops_per_ns_by_rows = [[16, 1900.0], [256, 21000.0], [8192, 26000.0]]
//! attn_flops_per_ns = 9000.0       # prefill attention flops a ns
//! scan_elements_per_ns = 1500.0    # recurrent state elements a ns (scan + conv)
//! bytes_per_ns = 726.0             # device memory bytes a ns (GB/s)
//! launch_ns = 3500                 # a dispatch's cost beside its work
//! ```

use std::fmt::Write as _;

/// A device's rates, each where a layer states it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Rates {
    /// Dense bf16 matmul flops a nanosecond at a matmul's rows: `(m, rate)`
    /// points, non-empty and strictly ascending in `m` ([`Rates::flops_at`]).
    pub bf16_flops_per_ns_by_rows: Option<Vec<(u32, f64)>>,
    /// Attention flops a nanosecond (the count the estimate's attention
    /// term takes, over a prefill's rows and keys).
    pub attn_flops_per_ns: Option<f64>,
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
            bf16_flops_per_ns_by_rows: self
                .bf16_flops_per_ns_by_rows
                .or(below.bf16_flops_per_ns_by_rows),
            attn_flops_per_ns: self.attn_flops_per_ns.or(below.attn_flops_per_ns),
            scan_elements_per_ns: self.scan_elements_per_ns.or(below.scan_elements_per_ns),
            bytes_per_ns: self.bytes_per_ns.or(below.bytes_per_ns),
            launch_ns: self.launch_ns.or(below.launch_ns),
        }
    }

    fn is_empty(&self) -> bool {
        *self == Rates::default()
    }

    /// The matmul rate at `rows`: the curve interpolated linearly in
    /// `log(rows)` between its two neighbouring points, and its end point's
    /// rate past either end. `None` where no curve is stated.
    #[must_use]
    pub fn flops_at(&self, rows: u32) -> Option<f64> {
        let curve = self.bf16_flops_per_ns_by_rows.as_deref()?;
        let (&(m0, r0), &(m1, r1)) = (curve.first()?, curve.last()?);
        if rows <= m0 {
            return Some(r0);
        }
        if rows >= m1 {
            return Some(r1);
        }
        let at = curve.partition_point(|&(m, _)| m <= rows);
        let ((lo, a), (hi, b)) = (curve[at - 1], curve[at]);
        let ln = |m: u32| f64::from(m.max(1)).ln();
        let t = (ln(rows) - ln(lo)) / (ln(hi) - ln(lo));
        Some(a + (b - a) * t)
    }

    /// The curve's fastest point: the rate a compute-bound matmul reaches.
    #[must_use]
    pub fn peak_flops(&self) -> Option<f64> {
        self.bf16_flops_per_ns_by_rows
            .as_deref()?
            .iter()
            .map(|&(_, r)| r)
            .reduce(f64::max)
    }

    /// `[rates]` as profile TOML.
    #[must_use]
    pub fn toml(&self) -> String {
        let mut out = String::from("[rates]\n");
        if let Some(curve) = &self.bf16_flops_per_ns_by_rows {
            let points: Vec<String> = curve.iter().map(|(m, r)| format!("[{m}, {r:?}]")).collect();
            let _ = writeln!(out, "bf16_flops_per_ns_by_rows = [{}]", points.join(", "));
        }
        if let Some(v) = self.attn_flops_per_ns {
            let _ = writeln!(out, "attn_flops_per_ns = {v:?}");
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
                bf16_flops_per_ns_by_rows: Some(vec![(4096, 1.0)]),
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
                bf16_flops_per_ns_by_rows: Some(vec![(4096, 1.0)]),
                bytes_per_ns: Some(3.0),
                launch_ns: Some(7),
                ..Rates::default()
            }
        );
    }

    #[test]
    fn the_curve_interpolates_in_log_rows_and_clamps_at_its_ends() {
        let rates = Rates {
            bf16_flops_per_ns_by_rows: Some(vec![(16, 100.0), (256, 500.0), (4096, 900.0)]),
            ..Rates::default()
        };
        assert_eq!(rates.flops_at(1), Some(100.0));
        assert_eq!(rates.flops_at(16), Some(100.0));
        assert_eq!(rates.flops_at(256), Some(500.0));
        // 64 is halfway from 16 to 256 in log rows.
        assert!((rates.flops_at(64).unwrap() - 300.0).abs() < 1e-9);
        assert_eq!(rates.flops_at(8192), Some(900.0));
        assert_eq!(rates.peak_flops(), Some(900.0));
        assert_eq!(Rates::default().flops_at(64), None);
    }
}
