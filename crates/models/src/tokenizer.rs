//! Per-SKU tokenizer contract table: concatenates each family's table and answers a SKU lookup.

pub use ::tokenizer::contract::{Contract, Fault};

pub type ContractRow = (&'static str, &'static Contract);

#[must_use]
pub fn contracts() -> Vec<ContractRow> {
    crate::skus().map(|sku| (sku.name.as_str(), sku.tokenizer)).collect()
}

#[must_use]
pub fn contract_of(name: &str) -> Option<&'static Contract> {
    crate::sku(name).map(|sku| sku.tokenizer)
}
