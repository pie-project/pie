pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::{Model, Routed};
use model_dsl::Dtype;

pub fn flash_u2g64(tp: u32) -> Model {
    Model::flash_mini(Dtype::U4g64, Routed::DQ_2BIT, Dtype::Bf16, Dtype::Bf16, tp)
}

pub fn flash_u2g64_mtp(tp: u32) -> Model {
    Model::flash_mini_mtp(Dtype::U4g64, Routed::DQ_2BIT, Dtype::Bf16, Dtype::Bf16, tp)
}

pub fn flash_u2g64_full_mtp(tp: u32) -> Model {
    Model::flash_mixed_mtp(
        Dtype::U4g64,
        Routed::DQ_2BIT_FULL,
        Dtype::Bf16,
        Dtype::Bf16,
        tp,
    )
}

pub fn flash_u2g64_full(tp: u32) -> Model {
    Model::flash_mixed(
        Dtype::U4g64,
        Routed::DQ_2BIT_FULL,
        Dtype::Bf16,
        Dtype::Bf16,
        tp,
    )
}

pub fn skus() -> Vec<crate::Sku> {
    crate::skus![
        (
            "dsv4-flash-full-mtp",
            1,
            [Dtype::U4g64, Dtype::U2g64, Dtype::Mxfp4],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::r1,
            &tokenizer::CONTRACT,
            |tp: u32| flash_u2g64_full_mtp(tp),
        ),
        (
            "dsv4-flash-mtp",
            1,
            [Dtype::U4g64, Dtype::U2g64, Dtype::Mxfp4],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::r1,
            &tokenizer::CONTRACT,
            |tp: u32| flash_u2g64_mtp(tp),
        ),
        (
            "dsv4-flash-full",
            1,
            [Dtype::U4g64, Dtype::U2g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::r1,
            &tokenizer::CONTRACT,
            |tp: u32| flash_u2g64_full(tp),
        ),
        (
            "dsv4-flash",
            1,
            [Dtype::U4g64, Dtype::U2g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::r1,
            &tokenizer::CONTRACT,
            |tp: u32| flash_u2g64(tp),
        ),
        (
            "dsv4-flash-mini",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::r1,
            &tokenizer::CONTRACT,
            |tp: u32| Model::flash_mini(
                Dtype::Bf16,
                Routed::uniform(Dtype::Bf16),
                Dtype::Bf16,
                Dtype::Bf16,
                tp
            ),
        ),
        (
            "dsv4-base",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::r1,
            &tokenizer::CONTRACT,
            |tp: u32| Model::base(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, tp),
        ),
        (
            "dsv4-base",
            2,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::r1,
            &tokenizer::CONTRACT,
            |tp: u32| Model::base(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, tp),
        ),
        (
            "dsv4-flash",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::r1,
            &tokenizer::CONTRACT,
            |tp: u32| Model::flash(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, tp),
        ),
    ]
}
