//! GLM-5.3-Flash's vision front-end: Qwen2-VL's processor at GLM's numbers
//! (patch 14, merge 2, temporal 2, CLIP mean/std, `min_image_tokens` 16 to
//! `max_image_tokens` 8000), wrapped in `<|begin_of_image|><|image|>…
//! <|end_of_image|>`. The trunk carries no rotary at all (nope-only MLA over
//! KDA), so an image span takes sequential positions rather than M-RoPE's
//! 2-D layout: the sparse indexer's pooling reads positions as a token count.

use crate::media::{Budget, Delimiters, EncodedSpan, Resample, Result, Rgb8, VisionFrontEnd};
use crate::qwen_3::media::{Qwen35Vision, QwenVisionConfig};

/// The `arch` the runtime row states (`model_type`).
pub const ARCH: &str = "glm5_next";

pub const BEGIN_OF_IMAGE: &str = "<|begin_of_image|>";
pub const IMAGE: &str = "<|image|>";
pub const END_OF_IMAGE: &str = "<|end_of_image|>";

/// The triple an image span is wrapped in — prefix, placeholder, suffix.
pub const VISION_DELIMITERS: &[&str] = &[BEGIN_OF_IMAGE, IMAGE, END_OF_IMAGE];

/// The processor's constants (`processor_config.json`).
#[must_use]
pub fn config() -> QwenVisionConfig {
    let patch = 14;
    let merge = 2;
    let per_token = patch * patch * merge * merge;
    QwenVisionConfig {
        patch_size: patch,
        merge_size: merge,
        temporal_patch_size: 2,
        min_pixels: 16 * per_token,
        max_pixels: 8000 * per_token,
        // No learned position table: the tower is rotary-only. A nonzero
        // side keeps the (unread) tap arithmetic finite.
        num_grid_per_side: 32,
        mean: [0.481_454_66, 0.457_827_5, 0.408_210_73],
        std: [0.268_629_54, 0.261_302_58, 0.275_777_11],
    }
}

pub struct Glm5Vision {
    inner: Qwen35Vision,
}

impl Glm5Vision {
    #[must_use]
    pub fn new() -> Glm5Vision {
        Glm5Vision {
            inner: Qwen35Vision { config: config() },
        }
    }
}

impl Default for Glm5Vision {
    fn default() -> Self {
        Glm5Vision::new()
    }
}

impl VisionFrontEnd for Glm5Vision {
    fn arch(&self) -> &'static str {
        ARCH
    }

    fn delimiters(&self) -> Delimiters {
        Delimiters {
            prefix: BEGIN_OF_IMAGE,
            placeholder: IMAGE,
            suffix: END_OF_IMAGE,
        }
    }

    fn encode(&self, src: &Rgb8, budget: Budget, resample: Resample) -> Result<EncodedSpan> {
        let mut span = self.inner.encode(src, budget, resample)?;
        // Sequential positions: see the module note. No learned position
        // table, so no taps: the tower's geometry owes none.
        span.uses_mrope = false;
        span.position_span = span.token_count;
        span.embed_rows.clear();
        span.embed_weights.clear();
        Ok(span)
    }
}
