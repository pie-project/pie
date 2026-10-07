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
