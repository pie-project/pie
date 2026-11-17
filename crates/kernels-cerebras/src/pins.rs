//! The named partitions a build may ask for by name (`wafer-compile plan --preset`, a debug
//! override of the search): the chip's measured programs ([`crate::cost::measured`]) and the
//! simulator's tiny ones and a variant (`plans/*.json`). None is picked for a model by itself: a
//! plan with no preset named is the search's ([`crate::search`]), which starts from these.

use serde::Deserialize;

use crate::kernels::KernelSet;
use crate::plan::Preset;

/// Every named plan's file past the measured programs'.
const FILES: &[&str] = &[
    include_str!("../plans/e4b-meta.json"),
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

/// Every named plan with its kernel set: the measured programs', then the files'.
#[must_use]
pub fn all() -> Vec<(KernelSet, Preset)> {
    crate::cost::measured::all()
        .into_iter()
        .map(|m| (m.kernels, m.preset))
        .chain(FILES.iter().map(|f| {
            let p: Pin = serde_json::from_str(f)
                .unwrap_or_else(|e| panic!("a named plan does not parse: {e}"));
            (p.kernels, p.preset)
        }))
        .collect()
}

/// The named plans of `set`.
#[must_use]
pub fn of(set: KernelSet) -> Vec<Preset> {
    all()
        .into_iter()
        .filter(|(l, _)| *l == set)
        .map(|(_, p)| p)
        .collect()
}

/// The named plan `name`, with its kernel set.
#[must_use]
pub fn named(name: &str) -> Option<(KernelSet, Preset)> {
    all().into_iter().find(|(_, p)| p.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every named plan parses, no two share a name, and no two of a kernel set were measured on
    /// one shape.
    #[test]
    fn the_named_plans_parse_and_name_one_plan_each() {
        let all = all();
        assert_eq!(all.len(), FILES.len() + crate::cost::measured::all().len());
        let mut names: Vec<&str> = all.iter().map(|(_, p)| p.name.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), all.len(), "two named plans share a name");
        let mut made: Vec<(KernelSet, [u32; 3])> = all
            .iter()
            .filter_map(|(l, p)| p.measured_on.map(|m| (*l, m)))
            .collect();
        let n = made.len();
        made.sort_by_key(|(l, m)| (l.name(), *m));
        made.dedup();
        assert_eq!(made.len(), n, "two named plans measured on one shape");
    }
}
