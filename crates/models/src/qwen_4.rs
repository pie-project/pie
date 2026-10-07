pub mod forward;
pub mod import;
pub mod model;

use model::{Mix, Model};
use poem_dsl::Dtype;

use crate::qwen_3::{template, tokenizer};

pub const ARCH: &str = "qwen4_exp";

pub fn skus() -> Vec<crate::Sku> {
    crate::skus![
        (
            "qwen38-flash-next",
            1,
            [Dtype::U4g64],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::chatml_interleaved,
            &tokenizer::CONTRACT_38,
            || Model::flash(Dtype::U4g64, Dtype::Bf16),
        ),
        (
            "qwen38-flash-next-full-mtp",
            1,
            [Dtype::U4g64, Dtype::U2g128],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::chatml_interleaved,
            &tokenizer::CONTRACT_38,
            || Model::flash_mix_mtp(Mix::MIXED_2BIT, Dtype::Bf16),
        ),
        (
            "qwen38-flash-next-full",
            1,
            [Dtype::U4g64, Dtype::U2g128],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::chatml_interleaved,
            &tokenizer::CONTRACT_38,
            || Model::flash_mix(Mix::MIXED_2BIT, Dtype::Bf16),
        ),
        (
            "qwen38-flash-next",
            1,
            [Dtype::U4g64, Dtype::U2g128],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::chatml_interleaved,
            &tokenizer::CONTRACT_38,
            || Model::flash_mini(Mix::MIXED_2BIT, Dtype::Bf16),
        ),
        (
            "qwen38-flash-next",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::chatml_interleaved,
            &tokenizer::CONTRACT_38,
            || Model::flash(Dtype::Bf16, Dtype::Bf16),
        ),
        (
            "qwen38-flash-next-full-mtp-vision",
            1,
            [Dtype::U4g64, Dtype::U2g128],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::chatml_interleaved,
            &tokenizer::CONTRACT_38_VISION,
            || Model::flash_mix_mtp_vision(Mix::MIXED_2BIT, Dtype::Bf16),
        ),
        (
            "qwen38-flash-next-full-vision",
            1,
            [Dtype::U4g64, Dtype::U2g128],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::chatml_interleaved,
            &tokenizer::CONTRACT_38_VISION,
            || Model::flash_mix_vision(Mix::MIXED_2BIT, Dtype::Bf16),
        ),
    ]
}
