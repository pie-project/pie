pub mod decoders;
pub mod deepseek_v4;
pub mod gemma_4;
pub mod glm_5;
pub mod gpt_oss;
pub mod instruct;
pub mod kimi_k3;
pub mod qwen_3_5;

pub fn catalog() -> Vec<(&'static str, fn(model_dsl::Plane) -> model_dsl::Plan)> {
    [
        deepseek_v4::CATALOG,
        gemma_4::CATALOG,
        glm_5::CATALOG,
        gpt_oss::CATALOG,
        kimi_k3::CATALOG,
        qwen_3_5::CATALOG,
    ]
    .concat()
}
