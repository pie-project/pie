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

/// DeepSeek-V4.1-Flash as `mlx_lm` publishes it: every projection and the
/// routed experts at MLX affine 4-bit, the Engram tables read out to bf16.
pub fn flash41_u4g64(tp: u32) -> Model {
    Model::flash41(
        Dtype::U4g64,
        Routed::split(Dtype::U4g64),
        Dtype::Bf16,
        Dtype::Bf16,
        tp,
    )
}

/// DeepSeek-V4.1-Flash with its fp8 planes read out to bf16 and its fp4
/// experts kept as MXFP4 codes (the released checkpoint's own expert format).
pub fn flash41_mxfp4(tp: u32) -> Model {
    Model::flash41(
        Dtype::Bf16,
        Routed::split(Dtype::Mxfp4),
        Dtype::Bf16,
        Dtype::Bf16,
        tp,
    )
}

pub fn flash41_mini_mxfp4(tp: u32) -> Model {
    Model::flash41_mini(
        Dtype::Bf16,
        Routed::split(Dtype::Mxfp4),
        Dtype::Bf16,
        Dtype::Bf16,
        tp,
    )
}

pub fn skus() -> Vec<crate::Sku> {
    crate::skus![
        (
            "dsv41-flash",
            1,
            [Dtype::U4g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::r1,
            &tokenizer::CONTRACT,
            |tp: u32| flash41_u4g64(tp),
        ),
        (
            "dsv41-flash",
            1,
            [Dtype::Bf16, Dtype::Mxfp4],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::r1,
            &tokenizer::CONTRACT,
            |tp: u32| flash41_mxfp4(tp),
        ),
        (
            "dsv41-flash-mini",
            1,
            [Dtype::Bf16, Dtype::Mxfp4],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::r1,
            &tokenizer::CONTRACT,
            |tp: u32| flash41_mini_mxfp4(tp),
        ),
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
