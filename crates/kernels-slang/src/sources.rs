//! The compiled instantiations, by key: an entry's name at the baseline,
//! `name.tier` above it.

use crate::encode::Tier;

mod generated {
    include!(concat!(env!("OUT_DIR"), "/census.rs"));
    include!(concat!(env!("OUT_DIR"), "/spirv.rs"));
    include!(concat!(env!("OUT_DIR"), "/wgsl.rs"));
}

pub use generated::{CENSUS, POOL_OWN_MAX, SPIRV, WGSL};

/// The key `name` is built under at `tier`, if it is instantiated there:
/// looked up in an index of the census built once (every dispatch an
/// encode lays asks, the arm walk's millions of them too).
#[must_use]
pub fn key(name: &str, tier: Tier) -> Option<&'static str> {
    type Index = std::collections::HashMap<&'static str, Vec<(&'static str, &'static str)>>;
    static INDEX: std::sync::OnceLock<Index> = std::sync::OnceLock::new();
    let tier = tier.name();
    INDEX
        .get_or_init(|| {
            let mut index = Index::with_capacity(CENSUS.len());
            for &(key, name, tier, _) in CENSUS {
                index.entry(name).or_default().push((tier, key));
            }
            index
        })
        .get(name)?
        .iter()
        .find(|(at, _)| *at == tier)
        .map(|&(_, key)| key)
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

/// The one-segment WGSL form of `key` (`<key>.s1`, `kernels/prim/pool.slang`'s
/// `PIE_POOL_SEG_COUNT=1`), where `key` binds the pool and this build
/// carries it.
#[must_use]
pub fn wgsl_one_segment(key: &str) -> Option<&'static str> {
    let one = format!("{key}.s1");
    WGSL.iter()
        .find(|(k, _, _)| *k == one)
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
        let compiled = SPIRV.iter().map(|(k, u, _)| (*k, *u)).chain(
            WGSL.iter()
                .map(|(k, u, _)| (k.strip_suffix(".s1").unwrap_or(k), *u)),
        );
        for (key, unit) in compiled {
            assert!(
                CENSUS.iter().any(|(k, _, _, u)| *k == key && *u == unit),
                "{key} is compiled but not counted"
            );
        }
    }

    /// Every pool form a WGSL build carries has its one-segment form beside
    /// it, which binds one segment (its others name it) and the kernel's own
    /// buffers as before.
    #[test]
    fn a_pool_form_has_its_one_segment_form() {
        for (key, _, text) in WGSL.iter().filter(|(k, _, _)| !k.ends_with(".s1")) {
            if !text.contains("pie_seg1") {
                continue;
            }
            let one = wgsl_one_segment(key).unwrap_or_else(|| panic!("{key} has no `.s1` form"));
            assert!(!one.contains("pie_seg1"), "{key}.s1 binds a second segment");
            assert_eq!(
                one.matches("var<storage").count() + 3,
                text.matches("var<storage").count(),
                "{key}.s1 binds three segments fewer"
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
