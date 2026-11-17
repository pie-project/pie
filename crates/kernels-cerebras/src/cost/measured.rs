//! The chip's measured decode programs: each shipping model's partition and kernel levers as the
//! chip ran them, and the bare step it measured (`measured/*.json`). They are data of the cost
//! model, not choices: a plan whose partition and levers are one of these, on the model it was
//! measured on, is priced at its measurement; a layer's own cost is read off them
//! ([`super::time::layer`]); and the search starts from them ([`crate::search`]), any model's
//! from its kernel set's. Which plan a model gets is the search's.

use serde::Deserialize;

use crate::kernels::KernelSet;
use crate::plan::Preset;

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

/// A measured program: its kernel set, its partition and levers (a [`Preset`] whose
/// `measured_on` is the model's shape signature), the chip's bare step (ms) and where it was
/// measured.
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
