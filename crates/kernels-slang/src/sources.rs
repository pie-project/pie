//! The compiled instantiations, by key: an entry's name at the baseline,
//! `name.tier` above it.

use crate::encode::Tier;

mod generated {
    include!(concat!(env!("OUT_DIR"), "/census.rs"));
    include!(concat!(env!("OUT_DIR"), "/spirv.rs"));
    include!(concat!(env!("OUT_DIR"), "/wgsl.rs"));
}

pub use generated::{CENSUS, SPIRV, WGSL};

/// The key `name` is built under at `tier`, if it is instantiated there.
#[must_use]
pub fn key(name: &str, tier: Tier) -> Option<&'static str> {
    CENSUS
        .iter()
        .find(|(_, n, t, _)| *n == name && *t == tier.name())
        .map(|(key, ..)| *key)
}

/// The SPIR-V module built under `key`, if this build carries it.
#[must_use]
pub fn spirv(key: &str) -> Option<&'static [u8]> {
    SPIRV
        .iter()
        .find(|(k, _, _)| *k == key)
        .map(|(_, _, words)| *words)
}

/// The WGSL module built under `key`, if this build carries it.
#[must_use]
pub fn wgsl(key: &str) -> Option<&'static str> {
    WGSL.iter()
        .find(|(k, _, _)| *k == key)
        .map(|(_, _, text)| *text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_instantiation_names_its_unit() {
        assert!(!CENSUS.is_empty());
        for (key, _, _, unit) in CENSUS {
            assert!(unit.ends_with(".slang"), "{key} lives in {unit}");
        }
        let compiled = SPIRV
            .iter()
            .map(|(k, u, _)| (*k, *u))
            .chain(WGSL.iter().map(|(k, u, _)| (*k, *u)));
        for (key, unit) in compiled {
            assert!(
                CENSUS.iter().any(|(k, _, _, u)| *k == key && *u == unit),
                "{key} is compiled but not counted"
            );
        }
    }

    #[test]
    fn a_tier_keys_its_build_apart() {
        for (key, name, tier, _) in CENSUS {
            let tier = Tier::ALL
                .into_iter()
                .find(|t| t.name() == *tier)
                .unwrap_or_else(|| panic!("{key}: `{tier}` is no tier"));
            assert_eq!(super::key(name, tier), Some(*key));
        }
    }
}
