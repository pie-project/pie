//! What a device profile states for choosing a load's top rung of fires:
//! the device's rates (`[rates]`), which a fire's time is estimated from,
//! and the fire times `pie tune` measured of a model class on it
//! (`[fire_ns]`). Both live in the backend's device file (or the user's
//! layer above it); a higher layer replaces a lower one's rates key by key
//! and its fire times model by model.
//!
//! ```toml
//! [rates]
//! bf16_flops_per_ns = 383000.0   # dense bf16 flops a ns (1 TFLOP/s = 1000)
//! bytes_per_ns = 896.0           # device memory bytes a ns (GB/s)
//! launch_ns = 4000               # a fire's launch cost beside its work
//!
//! [fire_ns]
//! "qwen3-8b-bf16-kv-bf16" = [[1, 21000000], [2, 21100000]]
//! ```
//!
//! A model class is the deployment less any tensor-parallel suffix
//! (`engine::load::model_class`).

use std::collections::BTreeMap;
use std::fmt::Write as _;

/// A device's rates, each where a layer states it.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Rates {
    /// Dense bf16 flops a nanosecond.
    pub bf16_flops_per_ns: Option<f64>,
    /// Device memory bytes a nanosecond.
    pub bytes_per_ns: Option<f64>,
    /// What a fire's launches cost beside its work, in nanoseconds.
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
}

/// `(tokens, ns)` of a fire at each rung width, ascending in tokens.
pub type FireTable = Vec<(u32, u64)>;

/// What a profile states beside its decisions: rates and fire times.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Tuned {
    pub rates: Rates,
    /// By model class.
    pub fire_ns: BTreeMap<String, FireTable>,
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
        for (model, table) in &upper.fire_ns {
            self.fire_ns.insert(model.clone(), table.clone());
        }
    }

    /// The fire times of `model`, or none.
    #[must_use]
    pub fn fire_ns(&self, model: &str) -> &[(u32, u64)] {
        self.fire_ns.get(model).map_or(&[], Vec::as_slice)
    }

    /// `[rates]` and `[fire_ns]` as profile TOML, nothing where nothing is
    /// stated.
    #[must_use]
    pub fn toml(&self) -> String {
        let mut out = String::new();
        if !self.rates.is_empty() && !self.measured_this_load {
            let _ = writeln!(out, "\n[rates]");
            if let Some(v) = self.rates.bf16_flops_per_ns {
                let _ = writeln!(out, "bf16_flops_per_ns = {v:?}");
            }
            if let Some(v) = self.rates.bytes_per_ns {
                let _ = writeln!(out, "bytes_per_ns = {v:?}");
            }
            if let Some(v) = self.rates.launch_ns {
                let _ = writeln!(out, "launch_ns = {v}");
            }
        }
        if !self.fire_ns.is_empty() {
            let _ = writeln!(out, "\n[fire_ns]");
            for (model, table) in &self.fire_ns {
                let _ = writeln!(out, "{}", table_line(model, table));
            }
        }
        out
    }
}

/// One `[fire_ns]` line: `"model" = [[tokens, ns], …]`.
#[must_use]
pub fn table_line(model: &str, table: &[(u32, u64)]) -> String {
    let rows: Vec<String> = table
        .iter()
        .map(|(tokens, ns)| format!("[{tokens}, {ns}]"))
        .collect();
    format!("\"{model}\" = [{}]", rows.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_upper_layer_replaces_by_key_and_by_model() {
        let mut low = Tuned {
            rates: Rates {
                bf16_flops_per_ns: Some(1.0),
                bytes_per_ns: Some(2.0),
                launch_ns: None,
            },
            fire_ns: BTreeMap::from([
                ("a".to_string(), vec![(1, 10)]),
                ("b".to_string(), vec![(1, 20)]),
            ]),
            measured_this_load: false,
        };
        let up = Tuned {
            rates: Rates {
                bytes_per_ns: Some(3.0),
                launch_ns: Some(7),
                ..Rates::default()
            },
            fire_ns: BTreeMap::from([("b".to_string(), vec![(1, 30), (2, 40)])]),
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
        assert_eq!(low.fire_ns("a"), &[(1, 10)]);
        assert_eq!(low.fire_ns("b"), &[(1, 30), (2, 40)]);
        assert!(low.fire_ns("c").is_empty());
    }
}
