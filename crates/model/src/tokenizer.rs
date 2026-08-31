//! The per-SKU tokenizer contract table — [`template`](crate::template)'s
//! twin, one layer down: which vocabulary demands each serving row makes.
//!
//! The contract LANGUAGE lives with the party that can check it
//! (`tokenizer::contract`, the way `checkpoint::contract` holds the weight
//! language); each family DECLARES in it under `<family>/tokenizer.rs`; this
//! file only concatenates the tables and answers a SKU lookup. The verify
//! call is the runtime's, at serve boot, beside the template lookup keyed by
//! the same string.

pub use ::tokenizer::contract::{Contract, Fault};

pub type ContractRow = (&'static str, &'static Contract);

#[must_use]
pub fn contracts() -> Vec<ContractRow> {
    [
        crate::deepseek_v4::TOKENIZERS,
        crate::gemma_4::TOKENIZERS,
        crate::glm_5::TOKENIZERS,
        crate::gpt_oss::TOKENIZERS,
        crate::kimi_k3::TOKENIZERS,
        crate::qwen_3::TOKENIZERS,
        crate::qwen_4::TOKENIZERS,
    ]
    .concat()
}

#[must_use]
pub fn contract_of(sku: &str) -> Option<&'static Contract> {
    contracts()
        .into_iter()
        .find(|(name, _)| *name == sku)
        .map(|(_, contract)| contract)
}
