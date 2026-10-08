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
        "gemma_4" => Some(&crate::gemma_4::tokenizer::CONTRACT),
        "minimax_h3" => Some(&crate::minimax_h3::tokenizer::CONTRACT),
        "hunyuan_image_3" => Some(&crate::hunyuan_image_3::tokenizer::CONTRACT),
        "kimi_k3" => Some(&crate::kimi_k3::tokenizer::CONTRACT),
        "mini_dit" => Some(&crate::mini_dit::tokenizer::CONTRACT),
        "gpt_oss" => Some(&crate::gpt_oss::tokenizer::CONTRACT),
        "inkling" => Some(&crate::inkling::tokenizer::CONTRACT),
        "glm_5" => Some(&crate::glm_5::tokenizer::CONTRACT),
        "qwen_3" => Some(&crate::qwen_3::tokenizer::CONTRACT),
        "muse_glimmer" => Some(&crate::muse_glimmer::tokenizer::CONTRACT),
        _ => None,
    }
}
