include!(concat!(env!("OUT_DIR"), "/modules.rs"));
include!(concat!(env!("OUT_DIR"), "/census.rs"));

/// The entrypoint's module for a device whose subgroups run `lanes` wide:
/// its `_narrow` variant where one exists and the lanes are fewer than 32.
#[must_use]
pub fn module_for(entrypoint: &str, lanes: u32) -> Option<&'static [u8]> {
    let find = |name: &str| {
        MODULES
            .iter()
            .find(|(n, _, _)| *n == name)
            .map(|(_, _, spv)| *spv)
    };
    if lanes < 32
        && let Some(spv) = find(&format!("{entrypoint}_narrow"))
    {
        return Some(spv);
    }
    find(entrypoint)
}

#[must_use]
pub fn module(entrypoint: &str) -> Option<&'static [u8]> {
    module_for(entrypoint, 32)
}

#[must_use]
pub fn census() -> &'static [&'static str] {
    CENSUS
}
