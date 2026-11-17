//! What a device profile states for choosing a load's top rung of fires:
//! the device's rates (`[rates]`), a roofline of a few measured constants
//! a fire's time is estimated from (`engine::load::FireEstimate`). They live
//! in the backend's device file (or the user's layer above it); a higher
//! layer replaces a lower one's key by key. `pie tune --rates` measures them
//! through the backend's own paths, at the large shapes `engine::rates`
//! states; a vendor's peak, where one is cited, is a bound in a comment, not
//! a rate. The device's shape (its SMs or CUs, its memory) is read from the
//! device at run time, never from the file.
//!
//! ```toml
//! [rates]
//! launch_ns = 900        # what a dependent dispatch adds to its work
//! gemm_tflops = 180.0    # bf16 matmul, achieved at a large square shape
//! attn_tflops = 120.0    # prefill attention, achieved at long keys
//! hbm_gbps = 700.0       # device memory, achieved streaming (read + write)
//! scan_gelems = 2600.0   # gated-delta scan state elements, G a second
//! ```

use std::fmt::Write as _;

/// A device's rates, each where a layer states it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Rates {
    /// What a dependent dispatch adds to its work, in nanoseconds.
    pub launch_ns: Option<u64>,
    /// Dense bf16 matmul TFLOP/s at a large square-ish shape, its tiles'
    /// wave quantization taken off.
    pub gemm_tflops: Option<f64>,
    /// Prefill attention TFLOP/s at long keys (the flops of the full
    /// rectangle, four a query element a key), its work-groups' wave
    /// quantization taken off.
    pub attn_tflops: Option<f64>,
    /// Device memory GB/s, achieved by a streaming copy (read plus write).
    pub hbm_gbps: Option<f64>,
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
            gemm_tflops: self.gemm_tflops.or(below.gemm_tflops),
            attn_tflops: self.attn_tflops.or(below.attn_tflops),
            hbm_gbps: self.hbm_gbps.or(below.hbm_gbps),
            scan_gelems: self.scan_gelems.or(below.scan_gelems),
        }
    }

    fn is_empty(&self) -> bool {
        *self == Rates::default()
    }

    /// The matmul rate in flops a nanosecond.
    #[must_use]
    pub fn gemm_flops_per_ns(&self) -> Option<f64> {
        self.gemm_tflops.map(|t| t * 1e3)
    }

    /// The attention rate in flops a nanosecond.
    #[must_use]
    pub fn attn_flops_per_ns(&self) -> Option<f64> {
        self.attn_tflops.map(|t| t * 1e3)
    }

    /// Device memory bytes a nanosecond.
    #[must_use]
    pub fn bytes_per_ns(&self) -> Option<f64> {
        self.hbm_gbps
    }

    /// `[rates]` as profile TOML.
    #[must_use]
    pub fn toml(&self) -> String {
        let mut out = String::from("[rates]\n");
        if let Some(v) = self.launch_ns {
            let _ = writeln!(out, "launch_ns = {v}");
        }
        let mut float = |key: &str, v: Option<f64>| {
            if let Some(v) = v {
                let _ = writeln!(out, "{key} = {v:?}");
            }
        };
        float("gemm_tflops", self.gemm_tflops);
        float("attn_tflops", self.attn_tflops);
        float("hbm_gbps", self.hbm_gbps);
        float("scan_gelems", self.scan_gelems);
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
                gemm_tflops: Some(1.0),
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
                gemm_tflops: Some(1.0),
                hbm_gbps: Some(3.0),
                launch_ns: Some(7),
                ..Rates::default()
            }
        );
        assert_eq!(low.rates.gemm_flops_per_ns(), Some(1000.0));
    }

    #[test]
    fn the_block_is_a_few_constants() {
        let rates = Rates {
            launch_ns: Some(904),
            gemm_tflops: Some(181.5),
            attn_tflops: Some(120.0),
            hbm_gbps: Some(654.4),
            scan_gelems: Some(2684.3),
        };
        assert_eq!(
            rates.toml(),
            "[rates]\nlaunch_ns = 904\ngemm_tflops = 181.5\nattn_tflops = 120.0\nhbm_gbps = \
             654.4\nscan_gelems = 2684.3\n"
        );
    }
}
