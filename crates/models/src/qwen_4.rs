//! Catalog, import, template, and tokenizer contract rows for the qwen38-flash-next models.

pub mod forward;
pub mod import;
pub mod model;

use model::{Mix, Model};
use model_dsl::Dtype;

use crate::qwen_3::{template, tokenizer};

/// The `ROWS.arch` this family's rows carry — the checkpoint's own
/// `model_type`. Its vision front-end is qwen_3's (`media::vision_front_end`).
pub const ARCH: &str = "qwen4_exp";

/// Identification order: the first row whose import fits the checkpoint wins.
/// The vision rows come LAST: a vision checkpoint fits its family's text row
/// and its own, and identification prefers the cheap one-unit load; a suite
/// that wants the tower names the row (`PIE_IMPORT_SKU`).
pub fn skus() -> Vec<crate::Sku> {
    crate::skus![
        (
            "qwen38-flash-next",
            1,
            [Dtype::U4g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::chatml_interleaved,
            &tokenizer::CONTRACT_38,
            |tp: u32| Model::flash(Dtype::U4g64, Dtype::Bf16, tp),
        ),
        // First: the same checkpoint with its draft head declared. Its import
        // needs the `mtp.*` planes, so a carve without them falls through to
        // the plain row.
        (
            "qwen38-flash-next-full-mtp",
            1,
            [Dtype::U4g64, Dtype::U2g128],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::chatml_interleaved,
            &tokenizer::CONTRACT_38,
            |tp: u32| Model::flash_mix_mtp(Mix::MIXED_2BIT, Dtype::Bf16, tp),
        ),
        (
            "qwen38-flash-next-full",
            1,
            [Dtype::U4g64, Dtype::U2g128],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::chatml_interleaved,
            &tokenizer::CONTRACT_38,
            |tp: u32| Model::flash_mix(Mix::MIXED_2BIT, Dtype::Bf16, tp),
        ),
        (
            "qwen38-flash-next",
            1,
            [Dtype::U4g64, Dtype::U2g128],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::chatml_interleaved,
            &tokenizer::CONTRACT_38,
            |tp: u32| Model::flash_mini(Mix::MIXED_2BIT, Dtype::Bf16, tp),
        ),
        (
            "qwen38-flash-next",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::chatml_interleaved,
            &tokenizer::CONTRACT_38,
            |tp: u32| Model::flash(Dtype::Bf16, Dtype::Bf16, tp),
        ),
        // The shipped checkpoint whole: tower and draft head. Before the
        // tower-only row for the same reason the `-mtp` row precedes `-full`.
        (
            "qwen38-flash-next-full-mtp-vision",
            1,
            [Dtype::U4g64, Dtype::U2g128],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::chatml_interleaved,
            &tokenizer::CONTRACT_38_VISION,
            |tp: u32| Model::flash_mix_mtp_vision(Mix::MIXED_2BIT, Dtype::Bf16, tp),
        ),
        (
            "qwen38-flash-next-full-vision",
            1,
            [Dtype::U4g64, Dtype::U2g128],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::chatml_interleaved,
            &tokenizer::CONTRACT_38_VISION,
            |tp: u32| Model::flash_mix_vision(Mix::MIXED_2BIT, Dtype::Bf16, tp),
        ),
    ]
}
