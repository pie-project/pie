//! Catalog rows for DiffusionGemma: Gemma 4's trunk run in two attention
//! modes (a causal encoder that writes the KV, a bidirectional denoiser that
//! only reads it) with a self-conditioning block on the denoiser's input.
//!
//! Step one of the bring-up: the trunk alone, imported from the diffusion
//! checkpoint's spelling and served as a plain causal model. The denoise
//! class and the self-conditioning block land on top of this row.

pub mod forward;
pub mod import;
pub mod model;

use model::Model;
use model_dsl::Dtype;

use crate::gemma_4::{template, tokenizer};

/// The `ROWS.arch` this family's rows carry — the checkpoint's own
/// `model_type`.
pub const ARCH: &str = "diffusion_gemma";

/// Identification order: the first row whose import fits the checkpoint wins.
/// Every row is a diffusion row: the canvas is a fact about the family.
pub fn skus() -> Vec<crate::Sku> {
    let mut rows = crate::skus![
        (
            "diffusiongemma-26b-a4b",
            1,
            [Dtype::U4g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::a4b(Dtype::U4g64, Dtype::Bf16, tp),
        ),
    ];
    for row in &mut rows {
        row.diffusion = Some(crate::Diffusion {
            canvas: model::CANVAS,
            hidden: model::HIDDEN,
            self_cond_taps: crate::gemma_4::model::SELF_COND_TAPS,
        });
    }
    rows
}
