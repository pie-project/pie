include!(concat!(env!("OUT_DIR"), "/modules.rs"));
include!(concat!(env!("OUT_DIR"), "/census.rs"));

#[must_use]
pub fn module(entrypoint: &str) -> Option<&'static [u8]> {
    MODULES
        .iter()
        .find(|(name, _, _)| *name == entrypoint)
        .map(|(_, _, spv)| *spv)
}

#[must_use]
pub fn tier_of(entrypoint: &str) -> Option<&'static str> {
    MODULES
        .iter()
        .find(|(name, _, _)| *name == entrypoint)
        .map(|(_, tier, _)| *tier)
}

#[must_use]
pub fn census() -> &'static [&'static str] {
    CENSUS
}
