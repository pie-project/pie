pub use ::tokenizer::contract::{Contract, Fault};

pub type ContractRow = (&'static str, &'static Contract);

#[must_use]
pub fn contracts() -> Vec<ContractRow> {
    crate::deployments()
        .map(|d| (d.name.as_str(), d.tokenizer))
        .collect()
}

#[must_use]
pub fn contract_of(name: &str) -> Option<&'static Contract> {
    crate::catalog::parse(name).map(|(entry, _)| entry.tokenizer)
}

/// The tokenizer contract a package names `name`.
#[must_use]
pub fn named(name: &str) -> Option<&'static Contract> {
    match name {
        "inkling" => Some(&crate::inkling::tokenizer::CONTRACT),
        "glm_5" => Some(&crate::glm_5::tokenizer::CONTRACT),
        "muse_glimmer" => Some(&crate::muse_glimmer::tokenizer::CONTRACT),
        _ => None,
    }
}
