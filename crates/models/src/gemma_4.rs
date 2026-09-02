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
        // Before the plain mixture: the head is extra tensors a plain row
        // would ignore, so the row that needs them must be asked first.
        (
            "gemma4-26b-a4b-mtp",
            1,
            [Dtype::U4g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::a4b_mtp(Dtype::U4g64, Dtype::Bf16, tp),
        ),
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
        // Parity miniatures, after every real row so identification never
        // picks them; a gate names them by SKU.
        (
            "gemma4-e4b-mini-l1",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::e4b_mini(1, Dtype::Bf16, Dtype::Bf16, tp),
        ),
        (
            "gemma4-e4b-mini-l6",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::e4b_mini(6, Dtype::Bf16, Dtype::Bf16, tp),
        ),
        (
            "gemma4-e4b-mini-l24",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::e4b_mini(24, Dtype::Bf16, Dtype::Bf16, tp),
        ),
        (
            "gemma4-e4b-mini-l30",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::e4b_mini(30, Dtype::Bf16, Dtype::Bf16, tp),
        ),
        (
            "gemma4-e4b-mini-l36",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::e4b_mini(36, Dtype::Bf16, Dtype::Bf16, tp),
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
