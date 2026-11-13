//! The pinned plans: the partition each shipping model's chip-validated program was built with
//! (`plans/*.json`, one a program: its kernel set, the keys its set's partition reads, the kernel
//! levers it was measured with, the shape signature of the model it was made for, the stacked
//! arrangement where it has one), and the simulator's tiny ones. A model with a pinned plan made
//! for its shape plans it; any other model plans by the search ([`crate::search`]), which starts
//! from its kernel set's pinned plans.

use serde::Deserialize;

use crate::kernels::KernelSet;
use crate::plan::Preset;

/// Every pinned plan's file.
const FILES: &[&str] = &[
    include_str!("../plans/0.8b.json"),
    include_str!("../plans/4b.json"),
    include_str!("../plans/9b.json"),
    include_str!("../plans/e2b.json"),
    include_str!("../plans/e4b-meta.json"),
    include_str!("../plans/e4b.json"),
    include_str!("../plans/full.json"),
    include_str!("../plans/g12.json"),
    include_str!("../plans/l3-8b.json"),
    include_str!("../plans/q3-8b.json"),
    include_str!("../plans/tiny-g12.json"),
    include_str!("../plans/tiny-gemma.json"),
    include_str!("../plans/tiny-gptoss.json"),
    include_str!("../plans/tiny-q3-gap.json"),
    include_str!("../plans/tiny-q3.json"),
    include_str!("../plans/tiny-t08.json"),
    include_str!("../plans/tiny-t4.json"),
];

#[derive(Deserialize)]
struct Pin {
    kernels: KernelSet,
    #[serde(flatten)]
    preset: Preset,
}

/// Every pinned plan with its kernel set, in file order.
#[must_use]
pub fn all() -> Vec<(KernelSet, Preset)> {
    FILES
        .iter()
        .map(|f| {
            let p: Pin = serde_json::from_str(f)
                .unwrap_or_else(|e| panic!("a pinned plan does not parse: {e}"));
            (p.kernels, p.preset)
        })
        .collect()
}

/// The pinned plans of `set`.
#[must_use]
pub fn of(set: KernelSet) -> Vec<Preset> {
    all()
        .into_iter()
        .filter(|(l, _)| *l == set)
        .map(|(_, p)| p)
        .collect()
}

/// The pinned plan named `name`, with its kernel set.
#[must_use]
pub fn named(name: &str) -> Option<(KernelSet, Preset)> {
    all().into_iter().find(|(_, p)| p.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every pinned plan parses, no two share a name, and no two of a kernel set are made for one
    /// shape (a model's plan is one pinned plan).
    #[test]
    fn the_pinned_plans_parse_and_name_one_plan_each() {
        let all = all();
        assert_eq!(all.len(), FILES.len());
        let mut names: Vec<&str> = all.iter().map(|(_, p)| p.name.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), all.len(), "two pinned plans share a name");
        let mut made: Vec<(KernelSet, [u32; 3])> = all
            .iter()
            .filter_map(|(l, p)| p.made_for.map(|m| (*l, m)))
            .collect();
        let n = made.len();
        made.sort_by_key(|(l, m)| (l.name(), *m));
        made.dedup();
        assert_eq!(made.len(), n, "two pinned plans made for one shape");
    }
}
