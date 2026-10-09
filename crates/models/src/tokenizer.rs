pub use ::tokenizer::contract::{Contract, Fault};

/// The tokenizer contract a package names `name`, for a deployment that
/// serves a vision tower or not: a tower's delimiters must be in the
/// vocabulary too.
#[must_use]
pub fn named(name: &str, vision: bool) -> Option<&'static Contract> {
    Some(match (name, vision) {
        ("qwen_3", false) => &crate::qwen_3::tokenizer::CONTRACT,
        ("qwen_3", true) => &crate::qwen_3::tokenizer::CONTRACT_VISION,
        ("qwen_3.38", false) => &crate::qwen_3::tokenizer::CONTRACT_38,
        ("qwen_3.38", true) => &crate::qwen_3::tokenizer::CONTRACT_38_VISION,
        ("gemma_4", false) => &crate::gemma_4::tokenizer::CONTRACT,
        ("gemma_4", true) => &crate::gemma_4::tokenizer::CONTRACT_VISION,
        ("glm_5_next", false) => &crate::glm_5_next::tokenizer::CONTRACT,
        ("glm_5_next", true) => &crate::glm_5_next::tokenizer::CONTRACT_VISION,
        ("kimi_k3.instruct3", _) => &crate::kimi_k3::tokenizer::CONTRACT3,
        (name, _) => match name {
            "deepseek_v4" => &crate::deepseek_v4::tokenizer::CONTRACT,
            "hunyuan_image_3" => &crate::hunyuan_image_3::tokenizer::CONTRACT,
            "kimi_k3" => &crate::kimi_k3::tokenizer::CONTRACT,
            "wan_2" => &crate::wan_2::tokenizer::CONTRACT,
            "gpt_oss" => &crate::gpt_oss::tokenizer::CONTRACT,
            "inkling" => &crate::inkling::tokenizer::CONTRACT,
            "glm_5" => &crate::glm_5::tokenizer::CONTRACT,
            "muse_glimmer" => &crate::muse_glimmer::tokenizer::CONTRACT,
            _ => return None,
        },
    })
}
