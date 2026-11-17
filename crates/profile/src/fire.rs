//! What a device profile states for choosing a load's top rung of fires:
//! the device's rates (`[rates]`), a handful of measured constants and two
//! short curves a fire's time is estimated from (`engine::load::
//! FireEstimate`, over `engine::roofline`). They live in the backend's
//! device file (or the user's layer above it); a higher layer replaces a
//! lower one's key by key. `pie tune --rates` (or a load on a device whose
//! profile lacks them) measures them through the backend's own paths, at the
//! shapes `engine::rates` states; a vendor's peak, where one is cited, is a
//! bound in a comment, not a rate.
//!
//! ```toml
//! [rates]
//! launch_ns = 900          # what a dependent dispatch adds to its work
//! # bf16 matmul TFLOP/s by rows, n = k = 4096
//! gemm_tflops_by_rows = [[256, 90.0], [1024, 160.0], [4096, 200.0], [8192, 210.0]]
//! # prefill attention TFLOP/s by keys (one sequence attending itself, d 128)
//! attn_tflops_by_keys = [[512, 60.0], [2048, 200.0], [8192, 400.0]]
//! hbm_gbps = 650.0         # a streaming copy, read + write
//! elementwise_gbps = 450.0 # the per-row ops (rmsnorm, swiglu) as served
//! scan_gelems = 2700.0     # gated-delta scan state elements, G a second
//! ```

use std::fmt::Write as _;

/// A device's rates, each where a layer states it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Rates {
    /// What a dependent dispatch adds to its work, in nanoseconds.
    pub launch_ns: Option<u64>,
    /// Dense bf16 matmul TFLOP/s by rows, at `engine::rates::GEMM_WIDTH`
    /// columns and depth: `(rows, TFLOP/s)`, strictly ascending in rows.
    pub gemm_tflops_by_rows: Option<Vec<(u32, f64)>>,
    /// Prefill attention TFLOP/s (the full rectangle's flops, four a query
    /// element a key) by keys, one sequence attending itself: `(keys,
    /// TFLOP/s)`, strictly ascending in keys.
    pub attn_tflops_by_keys: Option<Vec<(u32, f64)>>,
    /// Device memory GB/s, achieved by a streaming copy (read plus write).
    pub hbm_gbps: Option<f64>,
    /// GB/s the per-row ops (a norm, the gated activation) achieve as a
    /// fire runs them.
    pub elementwise_gbps: Option<f64>,
    /// Gated-delta scan state elements a nanosecond (G a second), the conv
    /// before it folded in.
    pub scan_gelems: Option<f64>,
}

impl Rates {
    /// `self` laid over `below`: each key `self` states replaces `below`'s.
    #[must_use]
    pub fn over(self, below: Rates) -> Rates {
        Rates {
            launch_ns: self.launch_ns.or(below.launch_ns),
            gemm_tflops_by_rows: self.gemm_tflops_by_rows.or(below.gemm_tflops_by_rows),
            attn_tflops_by_keys: self.attn_tflops_by_keys.or(below.attn_tflops_by_keys),
            hbm_gbps: self.hbm_gbps.or(below.hbm_gbps),
            elementwise_gbps: self.elementwise_gbps.or(below.elementwise_gbps),
            scan_gelems: self.scan_gelems.or(below.scan_gelems),
        }
    }

    fn is_empty(&self) -> bool {
        *self == Rates::default()
    }

    /// Whether every key is stated.
    #[must_use]
    pub fn complete(&self) -> bool {
        self.launch_ns.is_some()
            && self.gemm_tflops_by_rows.is_some()
            && self.attn_tflops_by_keys.is_some()
            && self.hbm_gbps.is_some()
            && self.elementwise_gbps.is_some()
            && self.scan_gelems.is_some()
    }

    /// `[rates]` as profile TOML.
    #[must_use]
    pub fn toml(&self) -> String {
        let mut out = String::from("[rates]\n");
        if let Some(v) = self.launch_ns {
            let _ = writeln!(out, "launch_ns = {v}");
        }
        let mut curve = |key: &str, v: &Option<Vec<(u32, f64)>>| {
            if let Some(points) = v {
                let points: Vec<String> = points
                    .iter()
                    .map(|(x, r)| format!("[{x}, {r:?}]"))
                    .collect();
                let _ = writeln!(out, "{key} = [{}]", points.join(", "));
            }
        };
        curve("gemm_tflops_by_rows", &self.gemm_tflops_by_rows);
        curve("attn_tflops_by_keys", &self.attn_tflops_by_keys);
        let mut float = |key: &str, v: Option<f64>| {
            if let Some(v) = v {
                let _ = writeln!(out, "{key} = {v:?}");
            }
        };
        float("hbm_gbps", self.hbm_gbps);
        float("elementwise_gbps", self.elementwise_gbps);
        float("scan_gelems", self.scan_gelems);
        out
    }
}

/// `curve` (`(x, y)`, ascending in `x`) at `x`: linear in `log x` between
/// its points, clamped at its ends. `None` for an empty curve.
#[must_use]
pub fn interpolate(curve: &[(u32, f64)], x: f64) -> Option<f64> {
    let (&(x0, y0), &(x1, y1)) = (curve.first()?, curve.last()?);
    if x <= f64::from(x0) {
        return Some(y0);
    }
    if x >= f64::from(x1) {
        return Some(y1);
    }
    let at = curve.partition_point(|&(p, _)| f64::from(p) <= x);
    let ((lo, a), (hi, b)) = (curve[at - 1], curve[at]);
    let t = (x.ln() - f64::from(lo).ln()) / (f64::from(hi).ln() - f64::from(lo).ln());
    Some(a + (b - a) * t)
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
                gemm_tflops_by_rows: Some(vec![(256, 1.0)]),
                hbm_gbps: Some(2.0),
                ..Rates::default()
            },
            measured_this_load: false,
        };
        let up = Tuned {
            rates: Rates {
                hbm_gbps: Some(3.0),
                launch_ns: Some(7),
                ..Rates::default()
            },
            measured_this_load: false,
        };
        low.lay(&up);
        assert_eq!(
            low.rates,
            Rates {
                gemm_tflops_by_rows: Some(vec![(256, 1.0)]),
                hbm_gbps: Some(3.0),
                launch_ns: Some(7),
                ..Rates::default()
            }
        );
        assert!(!low.rates.complete());
    }

    #[test]
    fn the_block_is_a_few_lines() {
        let rates = Rates {
            launch_ns: Some(904),
            gemm_tflops_by_rows: Some(vec![(256, 90.5), (8192, 210.0)]),
            attn_tflops_by_keys: Some(vec![(512, 60.0), (8192, 400.0)]),
            hbm_gbps: Some(654.4),
            elementwise_gbps: Some(450.0),
            scan_gelems: Some(2684.3),
        };
        assert!(rates.complete());
        assert_eq!(
            rates.toml(),
            "[rates]\nlaunch_ns = 904\ngemm_tflops_by_rows = [[256, 90.5], [8192, 210.0]]\n\
             attn_tflops_by_keys = [[512, 60.0], [8192, 400.0]]\nhbm_gbps = 654.4\n\
             elementwise_gbps = 450.0\nscan_gelems = 2684.3\n"
        );
    }

    #[test]
    fn a_curve_is_read_in_log_x_and_clamped() {
        let c = [(256, 100.0), (1024, 200.0)];
        assert_eq!(interpolate(&c, 1.0), Some(100.0));
        assert_eq!(interpolate(&c, 4096.0), Some(200.0));
        assert!((interpolate(&c, 512.0).unwrap() - 150.0).abs() < 1e-9);
        assert_eq!(interpolate(&[], 1.0), None);
    }
}
