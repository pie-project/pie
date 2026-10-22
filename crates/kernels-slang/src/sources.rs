//! The compiled instantiations, by entrypoint name.

mod generated {
    include!(concat!(env!("OUT_DIR"), "/census.rs"));
    include!(concat!(env!("OUT_DIR"), "/spirv.rs"));
    include!(concat!(env!("OUT_DIR"), "/wgsl.rs"));
}

pub use generated::{CENSUS, SPIRV, WGSL};

/// The SPIR-V module of `entrypoint`, if this build carries it.
#[must_use]
pub fn spirv(entrypoint: &str) -> Option<&'static [u8]> {
    SPIRV
        .iter()
        .find(|(name, _, _)| *name == entrypoint)
        .map(|(_, _, words)| *words)
}

/// The WGSL module of `entrypoint`, if this build carries it.
#[must_use]
pub fn wgsl(entrypoint: &str) -> Option<&'static str> {
    WGSL.iter()
        .find(|(name, _, _)| *name == entrypoint)
        .map(|(_, _, text)| *text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_instantiation_names_its_unit() {
        assert!(!CENSUS.is_empty());
        for (name, unit) in CENSUS {
            assert!(unit.ends_with(".slang"), "{name} lives in {unit}");
        }
        let compiled = SPIRV
            .iter()
            .map(|(n, u, _)| (*n, *u))
            .chain(WGSL.iter().map(|(n, u, _)| (*n, *u)));
        for (name, unit) in compiled {
            assert!(
                CENSUS.contains(&(name, unit)),
                "{name} is compiled but not counted"
            );
        }
    }
}
