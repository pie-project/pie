//! Catalog, import, template, and tokenizer contract rows for the qwen38-flash models.

pub mod forward;
pub mod import;
pub mod model;

use model::{Mix, Model};
use model_dsl::Dtype;

use crate::qwen_3::{template, tokenizer};

/// Identification order: the first row whose import fits the checkpoint wins.
pub fn skus() -> Vec<crate::Sku> {
    crate::skus![
        (
            "qwen38-flash",
            1,
            [Dtype::U4g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::chatml_interleaved,
            &tokenizer::CONTRACT_38,
            |tp: u32| Model::flash(Dtype::U4g64, Dtype::Bf16, tp),
        ),
        (
            "qwen38-flash-full",
            1,
            [Dtype::U4g64, Dtype::U2g128],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::chatml_interleaved,
            &tokenizer::CONTRACT_38,
            |tp: u32| Model::flash_mix(Mix::MIXED_2BIT, Dtype::Bf16, tp),
        ),
        (
            "qwen38-flash",
            1,
            [Dtype::U4g64, Dtype::U2g128],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::chatml_interleaved,
            &tokenizer::CONTRACT_38,
            |tp: u32| Model::flash_mini(Mix::MIXED_2BIT, Dtype::Bf16, tp),
        ),
        (
            "qwen38-flash",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::chatml_interleaved,
            &tokenizer::CONTRACT_38,
            |tp: u32| Model::flash(Dtype::Bf16, Dtype::Bf16, tp),
        ),
    ]
}
