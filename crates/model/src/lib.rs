pub mod decoders;
pub mod deepseek_v4;
pub mod gemma_4;
pub mod glm_5;
pub mod gpt_oss;
pub mod instruct;
pub mod kimi_k3;
pub mod produce;
pub mod qwen_3_5;
/// The serving half — chat templates, multimodal decode, the per-id
/// `(layers, vocab)` a sampler is sized from, and the artifact metadata an
/// engine is handed. Moved here from `model-legacy`'s `chat` feature by the
/// R1 cutover; see its module doc.
///
/// NOTE, for whoever collapses this: there are TWO `Instruct` traits in this
/// crate right now. [`serve::instruct`] is the one the guest-facing WIT
/// contract is written against (`feed(&[u32]) -> ChatEvent`, `reset`, a tool
/// grammar) and the one `engine` links; [`instruct`] beside it is the thinner
/// `push(u32) -> Option<String>` sketch that arrived with the family takeover,
/// together with the six per-family `template.rs` files that implement it. The
/// sketch has no consumer and cannot answer the WIT surface as written. It is
/// left standing rather than deleted because collapsing the two is a decision
/// about the guest contract, not a consequence of cutting a dependency.
#[cfg(feature = "serve")]
pub mod serve;
pub mod snapshot;

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

/// Every shipping import point, across all families.
///
/// Shorter than [`catalog`] and always will be: a tensor-parallel row is the
/// same bytes cut a different way at load, so it names no import of its own.
pub fn imports() -> Vec<model_dsl::load::ImportRow> {
    [
        deepseek_v4::IMPORTS,
        gemma_4::IMPORTS,
        glm_5::IMPORTS,
        gpt_oss::IMPORTS,
        kimi_k3::IMPORTS,
        qwen_3_5::IMPORTS,
    ]
    .concat()
}

/// The trace fn for `sku`, or `None` if no row ships under that name.
pub fn trace_of(sku: &str) -> Option<fn(model_dsl::Plane) -> model_dsl::Plan> {
    catalog().into_iter().find(|(n, _)| *n == sku).map(|(_, f)| f)
}

/// The import table that builds `sku` from a `base`-flavored checkpoint.
///
/// Both halves of the key are required. Asking for a SKU alone would pick
/// whichever flavor was filed first, which for Gemma is a coin flip between
/// its safetensors release and its GGUF one.
pub fn import_of(sku: &str, base: &str) -> Option<model_dsl::load::Import> {
    imports()
        .into_iter()
        .find(|r| r.sku == sku && r.base == base)
        .map(|r| (r.make)())
}

/// Every checkpoint flavor `sku` can be built from, in table order.
pub fn bases_for(sku: &str) -> Vec<&'static str> {
    imports()
        .into_iter()
        .filter(|r| r.sku == sku)
        .map(|r| r.base)
        .collect()
}
