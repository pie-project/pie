//! The chip's measured decode programs (`measured/*.json`): each shipping model's partition,
//! levers and bare step. They are cost-model data, not choices: a plan matching one on its own
//! model is priced at its measurement, and the per-kind layer costs are fitted to them all.

use serde::Deserialize;

use crate::compile::kernels::KernelSet;
use crate::compile::plan::Preset;

/// Every measured program's file.
const FILES: &[&str] = &[
    include_str!("measured/0.8b.json"),
    include_str!("measured/4b.json"),
    include_str!("measured/9b.json"),
    include_str!("measured/e2b.json"),
    include_str!("measured/e4b.json"),
    include_str!("measured/full.json"),
    include_str!("measured/g12.json"),
    include_str!("measured/l3-8b.json"),
    include_str!("measured/q3-8b.json"),
];

/// A measured program: its kernel set, partition and levers (a [`Preset`] whose `measured_on` is
/// the model's shape signature) and the chip's bare step (ms).
#[derive(Clone, Debug, Deserialize)]
pub struct Measured {
    pub kernels: KernelSet,
    pub ms: f64,
    pub source: String,
    #[serde(flatten)]
    pub preset: Preset,
}

/// Every measured program, in file order.
#[must_use]
pub fn all() -> Vec<Measured> {
    FILES
        .iter()
        .map(|f| {
            serde_json::from_str(f)
                .unwrap_or_else(|e| panic!("a measured program does not parse: {e}"))
        })
        .collect()
}

/// The measured program named `name` (`wafer-compile plan --preset`, a debug override).
#[must_use]
pub fn named(name: &str) -> Option<Measured> {
    all().into_iter().find(|m| m.preset.name == name)
}

/// The program measured on the model of shape signature `sig`, if the chip ran one.
#[must_use]
pub fn on(sig: [u32; 3]) -> Option<Measured> {
    all()
        .into_iter()
        .find(|m| m.preset.measured_on == Some(sig))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every measured program parses, names a model of its own, and its measurement is a step.
    #[test]
    fn the_measured_programs_parse_one_a_model() {
        let all = all();
        assert_eq!(all.len(), FILES.len());
        let mut on: Vec<[u32; 3]> = all.iter().filter_map(|m| m.preset.measured_on).collect();
        assert_eq!(on.len(), all.len(), "a measured program names no model");
        on.sort_unstable();
        on.dedup();
        assert_eq!(on.len(), all.len(), "two measured programs of one model");
        assert!(all.iter().all(|m| m.ms > 0.1 && m.ms < 2.0));
    }
}
