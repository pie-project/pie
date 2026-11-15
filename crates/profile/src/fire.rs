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

use std::fmt::Write as _;

/// A device's rates, each where a layer states it.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Rates {
    /// Dense bf16 flops a nanosecond.
    pub bf16_flops_per_ns: Option<f64>,
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
            bf16_flops_per_ns: self.bf16_flops_per_ns.or(below.bf16_flops_per_ns),
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
                launch_ns: None,
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
            }
        );
    }
}
