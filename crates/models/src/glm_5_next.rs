//! Catalog, import, template and tokenizer rows for GLM-5.3-Flash (`glm5_next`).

pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use model_dsl::Dtype;

/// Identification order: the first row whose import fits the checkpoint wins.
pub fn skus() -> Vec<crate::Sku> {
    crate::skus![
        // The drafting row first: it fits only a source that carries the
        // `layers.45` head (or an artifact with the `aux.` overlay); a plain
        // one falls through to the text row below.
        (
            "glm53-flash-mtp",
            1,
            [Dtype::U8g64, Dtype::U2g64, Dtype::U4g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::flash_mtp(Dtype::U8g64, Dtype::U2g64, Dtype::U4g64, Dtype::Bf16, tp),
        ),
        (
            "glm53-flash",
            1,
            [Dtype::U8g64, Dtype::U2g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::flash(Dtype::U8g64, Dtype::U2g64, Dtype::Bf16, tp),
        ),
    ]
}
