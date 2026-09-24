pub mod forward;
pub mod import;
pub mod media;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use model_dsl::Dtype;

pub fn skus() -> Vec<crate::Sku> {
    crate::skus![
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
            "glm53-flash-mini",
            1,
            [Dtype::U4g64, Dtype::U4g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::flash_mini(8, 32, Dtype::U4g64, Dtype::U4g64, Dtype::Bf16, tp),
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
        (
            "glm53-flash-mtp-vision",
            1,
            [Dtype::U8g64, Dtype::U2g64, Dtype::U4g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT_VISION,
            |tp: u32| {
                Model::flash_mtp_vision(Dtype::U8g64, Dtype::U2g64, Dtype::U4g64, Dtype::Bf16, tp)
            },
        ),
        (
            "glm53-flash-vision",
            1,
            [Dtype::U8g64, Dtype::U2g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT_VISION,
            |tp: u32| Model::flash_vision(Dtype::U8g64, Dtype::U2g64, Dtype::Bf16, tp),
        ),
        (
            "glm53-flash-mtp-vision",
            1,
            [Dtype::U4g64, Dtype::U2g64, Dtype::U4g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT_VISION,
            |tp: u32| {
                Model::flash_mtp_vision(Dtype::U4g64, Dtype::U2g64, Dtype::U4g64, Dtype::Bf16, tp)
            },
        ),
    ]
}
