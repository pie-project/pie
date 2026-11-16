//! What a device profile states for choosing a load's top rung of fires:
//! the device's rates (`[rates]`), which a fire's time is estimated from.
//! They live in the backend's device file (or the user's layer above it);
//! a higher layer replaces a lower one's key by key. `pie tune --rates`
//! measures them through the backend's own paths; a vendor's peak, where
//! one is cited, is a bound in a comment, not a rate.
//!
//! ```toml
//! [rates]
//! bf16_flops_per_ns = 13800.0  # dense bf16 flops a ns (1 TFLOP/s = 1000)
//! bytes_per_ns = 726.0         # device memory bytes a ns (GB/s)
//! launch_ns = 183000           # a launch's cost beside its work
//! ```
//!
//! A backend that times its matmul at each row count states the curve
//! (`bf16_flops_per_ns_by_rows`, at every row count of [`BY_ROWS`]) in
//! place of the one rate, and its own attention and recurrent-scan kernels'
//! rates beside it:
//!
//! ```toml
//! [rates]
//! bf16_flops_per_ns_by_rows = [[16, 9100.0], [32, 18000.0], ..., [8192, 251000.0]]
//! attn_flops_per_ns = 160000.0   # causal prefill sdpa, 4·rows·keys·q_heads·d / 2
//! scan_elements_per_ns = 2900.0  # gated delta scan + its conv, tokens·heads·d_k·d_v
//! ```

use std::fmt::Write as _;

/// The matmul row counts `bf16_flops_per_ns_by_rows` states, in order: the
/// powers of two from 16 to 8192.
pub const BY_ROWS: [u32; 10] = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];

/// A device's rates, each where a layer states it.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Rates {
    /// Dense bf16 flops a nanosecond.
    pub bf16_flops_per_ns: Option<f64>,
    /// Dense bf16 flops a nanosecond of an `m × 4096 × 4096` matmul at each
    /// row count `m` of [`BY_ROWS`], in its order.
    pub bf16_flops_per_ns_by_rows: Option<[f64; BY_ROWS.len()]>,
    /// Causal prefill attention flops a nanosecond (`4·rows·keys·q_heads·d
    /// / 2`).
    pub attn_flops_per_ns: Option<f64>,
    /// Gated-delta chunked scan elements (`tokens·heads·d_k·d_v`) a
    /// nanosecond, its causal conv beside it.
    pub scan_elements_per_ns: Option<f64>,
    /// Device memory bytes a nanosecond.
    pub bytes_per_ns: Option<f64>,
    /// What a launch costs beside its work, in nanoseconds.
    pub launch_ns: Option<u64>,
}

impl Rates {
    /// The one dense bf16 rate: the curve's at 4096 rows (the `4096³`
    /// product the one rate is measured at) where it is stated, else the
    /// one rate.
    #[must_use]
    pub fn bf16_flops_per_ns(&self) -> Option<f64> {
        self.bf16_flops_per_ns_by_rows
            .map(|curve| curve[8])
            .or(self.bf16_flops_per_ns)
    }

    /// `self` laid over `below`: each key `self` states replaces `below`'s.
    #[must_use]
    pub fn over(self, below: Rates) -> Rates {
        Rates {
            // A layer stating the curve replaces the one rate below it too.
            bf16_flops_per_ns: self.bf16_flops_per_ns.or(below
                .bf16_flops_per_ns
                .filter(|_| self.bf16_flops_per_ns_by_rows.is_none())),
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

    /// `[rates]` as profile TOML.
    #[must_use]
    pub fn toml(&self) -> String {
        let mut out = String::from("[rates]\n");
        if let Some(v) = self.bf16_flops_per_ns {
            let _ = writeln!(out, "bf16_flops_per_ns = {v:?}");
        }
        if let Some(curve) = self.bf16_flops_per_ns_by_rows {
            let pairs: Vec<String> = BY_ROWS
                .iter()
                .zip(curve)
                .map(|(rows, r)| format!("[{rows}, {r:?}]"))
                .collect();
            let _ = writeln!(out, "bf16_flops_per_ns_by_rows = [{}]", pairs.join(", "));
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
        self.rates = upper.rates.over(self.rates);
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
                bf16_flops_per_ns: Some(1.0),
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
                bf16_flops_per_ns: Some(1.0),
                bytes_per_ns: Some(3.0),
                launch_ns: Some(7),
                ..Rates::default()
            }
        );
    }
}
