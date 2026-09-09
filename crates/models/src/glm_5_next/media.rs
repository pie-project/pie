use crate::media::{Budget, Delimiters, EncodedSpan, Resample, Result, Rgb8, VisionFrontEnd};
use crate::qwen_3::media::{Qwen35Vision, QwenVisionConfig};

pub const ARCH: &str = "glm5_next";

pub const BEGIN_OF_IMAGE: &str = "<|begin_of_image|>";
pub const IMAGE: &str = "<|image|>";
pub const END_OF_IMAGE: &str = "<|end_of_image|>";

pub const VISION_DELIMITERS: &[&str] = &[BEGIN_OF_IMAGE, IMAGE, END_OF_IMAGE];

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
        num_grid_per_side: 32,
        mean: [0.481_454_66, 0.457_827_5, 0.408_210_73],
        std: [0.268_629_54, 0.261_302_6, 0.275_777_1],
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
        span.uses_mrope = false;
        span.position_span = span.token_count;
        span.embed_rows.clear();
        span.embed_weights.clear();
        Ok(span)
    }
}
