use crate::Capability;

include!(concat!(env!("OUT_DIR"), "/modules.rs"));

include!(concat!(env!("OUT_DIR"), "/census.rs"));

#[must_use]
pub fn stem(stem: &str) -> Option<&'static [u8]> {
    MODULES
        .binary_search_by_key(&stem, |&(name, _)| name)
        .ok()
        .map(|i| MODULES[i].1)
}

#[must_use]
pub fn code(entrypoint: &str, tier: Capability) -> Option<&'static [u8]> {
    let name = tier.module(entrypoint);
    stem(
        name.strip_suffix(".spv")
            .expect("`Capability::module` names a `.spv`"),
    )
}

#[must_use]
pub fn embedded() -> bool {
    !MODULES.is_empty()
}

#[must_use]
pub fn path(entrypoint: &str, want: Capability) -> &'static str {
    let tier = Capability::PREFERENCE
        .iter()
        .skip_while(|&&c| c != want)
        .find(|&&c| stem(&c.module(entrypoint).replace(".spv", "")).is_some())
        .copied()
        .unwrap_or(Capability::Baseline);
    intern(&tier.module(entrypoint))
}

fn intern(name: &str) -> &'static str {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    static NAMES: OnceLock<Mutex<HashMap<String, &'static str>>> = OnceLock::new();
    let mut map = NAMES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(found) = map.get(name) {
        return found;
    }
    let fresh: &'static str = Box::leak(name.to_owned().into_boxed_str());
    map.insert(name.to_owned(), fresh);
    fresh
}

#[must_use]
pub fn at(file: &str) -> Option<&'static [u8]> {
    stem(file.strip_suffix(".spv").unwrap_or(file))
}
