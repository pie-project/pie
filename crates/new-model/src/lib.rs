//! The model catalog of the menlo stack (design §10): each model is a
//! declaration — weights, dims, cache spaces — plus a forward pass written in
//! `new-model-dsl`. All six shipping models are ported; checkpoint imports
//! and serving rows stay in the old `model` crate until the new loader lands.

pub mod deepseek_v4;
pub mod gemma_4;
pub mod glm_5;
pub mod gpt_oss;
pub mod kimi_k3;
pub mod qwen_3_5;

/// One catalog row: a SKU name, and the trace that renders its plan for a
/// plane.
pub type Row = (&'static str, new_model_dsl::TraceFn);

#[must_use]
pub fn catalog() -> Vec<Row> {
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

#[must_use]
pub fn trace_of(sku: &str) -> Option<new_model_dsl::TraceFn> {
    catalog()
        .into_iter()
        .find(|(n, _)| *n == sku)
        .map(|(_, f)| f)
}
