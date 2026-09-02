pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::{Model, Routed};
use model_dsl::Dtype;

/// Trunk (attention, shared expert, embedding, head) is 4-bit at group 64;
/// routed experts use the [`Routed::DQ_2BIT`] mix; norms, position embedding,
/// and router gate stay bf16.
pub fn flash_u2g64(tp: u32) -> Model {
    Model::flash_mini(Dtype::U4g64, Routed::DQ_2BIT, Dtype::Bf16, Dtype::Bf16, tp)
}

/// [`flash_u2g64`] with the draft head over it (`--aux` from
/// `scripts/dsv4_mtp_companion.py`'s overlay).
pub fn flash_u2g64_mtp(tp: u32) -> Model {
    Model::flash_mini_mtp(Dtype::U4g64, Routed::DQ_2BIT, Dtype::Bf16, Dtype::Bf16, tp)
}

/// [`flash_u2g64_full`] with the draft head over it.
pub fn flash_u2g64_full_mtp(tp: u32) -> Model {
    Model::flash_mixed_mtp(
        Dtype::U4g64,
        Routed::DQ_2BIT_FULL,
        Dtype::Bf16,
        Dtype::Bf16,
        tp,
    )
}

/// Same four dtypes as [`flash_u2g64`], over the full 43-layer, 256-expert
/// geometry ([`Routed::DQ_2BIT_FULL`]); the mini's group-32 gate exception at
/// layer 4 corresponds to layer 42 here.
pub fn flash_u2g64_full(tp: u32) -> Model {
    Model::flash_mixed(
        Dtype::U4g64,
        Routed::DQ_2BIT_FULL,
        Dtype::Bf16,
        Dtype::Bf16,
        tp,
    )
}

/// Identification order: the first row whose import fits the checkpoint wins.
pub fn skus() -> Vec<crate::Sku> {
    crate::skus![
        // The drafting rows first: they fit only an artifact that carries the
        // `aux.` overlay, and a plain one falls through to the rows below.
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
