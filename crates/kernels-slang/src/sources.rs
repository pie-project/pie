//! The compiled instantiations, by key: an entry's name at the baseline,
//! `name.tier` above it.

use crate::encode::Tier;

mod generated {
    include!(concat!(env!("OUT_DIR"), "/census.rs"));
    include!(concat!(env!("OUT_DIR"), "/spirv.rs"));
    include!(concat!(env!("OUT_DIR"), "/wgsl.rs"));
}

pub use generated::{CENSUS, SPIRV, WGSL, WGSL_POOL_STORAGE_BUFFERS};

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

/// The key of `key`'s WGSL build for a pool of one segment, if it binds the pool.
#[must_use]
pub fn one_segment(key: &str) -> Option<&'static str> {
    WGSL.iter()
        .find(|(k, _, _)| k.strip_suffix(".seg1") == Some(key))
        .map(|(k, _, _)| *k)
}

/// The pool segments WGSL kernels bind on a device of `storage_buffers` a stage (0: no limit).
#[must_use]
pub fn wgsl_pool_segments(storage_buffers: u32) -> usize {
    if storage_buffers == 0 || storage_buffers >= WGSL_POOL_STORAGE_BUFFERS {
        crate::SEGMENTS
    } else {
        1
    }
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
            let key = key.strip_suffix(".seg1").unwrap_or(key);
            assert!(
                CENSUS.iter().any(|(k, _, _, u)| *k == key && *u == unit),
                "{key} is compiled but not counted"
            );
        }
    }

    #[test]
    fn a_one_segment_build_binds_one_segment() {
        let storage = |wgsl: &str| wgsl.matches("var<storage").count();
        for (key, _, text) in WGSL {
            let Some(base) = key.strip_suffix(".seg1") else {
                continue;
            };
            assert_eq!(one_segment(base), Some(*key));
            let four = wgsl(base).unwrap_or_else(|| panic!("{key} has no four-segment build"));
            assert_eq!(storage(text) + crate::SEGMENTS - 1, storage(four), "{key}");
            assert!(
                storage(four) <= WGSL_POOL_STORAGE_BUFFERS as usize,
                "{base}"
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
