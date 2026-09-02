pub mod forward;
pub mod import;
pub mod media;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use model_dsl::Dtype;

/// Identification order: the first row whose import fits the checkpoint wins.
pub fn skus() -> Vec<crate::Sku> {
    crate::skus![
        (
            "gemma4-26b-a4b",
            1,
            [Dtype::U4g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::a4b(Dtype::U4g64, Dtype::Bf16, tp),
        ),
        (
            "gemma4-31b",
            1,
            [Dtype::U4g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::b31(Dtype::U4g64, Dtype::Bf16, tp),
        ),
        (
            "gemma4-e4b-eagle",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::e4b_eagle(Dtype::Bf16, Dtype::Bf16, tp),
        ),
        (
            "gemma4-e4b",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::e4b(Dtype::Bf16, Dtype::Bf16, tp),
        ),
        (
            "gemma4-31b",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::b31(Dtype::Bf16, Dtype::Bf16, tp),
        ),
        (
            "gemma4-31b",
            2,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::b31(Dtype::Bf16, Dtype::Bf16, tp),
        ),
        (
            "gemma4-e4b-vision",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT_VISION,
            |tp: u32| Model::e4b_vision(Dtype::Bf16, Dtype::Bf16, tp),
        ),
        (
            "gemma4-26b-a4b-vision",
            1,
            [Dtype::U4g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT_VISION,
            |tp: u32| Model::a4b_vision(Dtype::U4g64, Dtype::Bf16, tp),
        ),
        (
            "gemma4-31b-vision",
            1,
            [Dtype::U4g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT_VISION,
            |tp: u32| Model::b31_vision(Dtype::U4g64, Dtype::Bf16, tp),
        ),
    ]
}
