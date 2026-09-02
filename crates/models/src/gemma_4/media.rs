//! Gemma-4 vision preprocessing: resize to a whole `k x k` block grid,
//! patchify in pool-block-major order, and emit one soft token per block.

use crate::media::{Budget, Delimiters, EncodedSpan, Fault, Grid, Resample, Result, Rgb8,
    VisionFrontEnd};

/// The `ROWS.arch` this front-end answers for.
pub const ARCH: &str = "gemma4";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmaVisionConfig {
    /// Pixels per patch side.
    pub patch_size: u32,
    /// The soft-token pool folds `k²` patch rows.
    pub pooling_kernel_size: u32,
    /// A still image's soft-token ceiling.
    pub max_soft_tokens: u32,
    /// A video frame's soft-token ceiling (lower than a still image's).
    pub video_soft_tokens: u32,
    /// The separable position table's per-axis length — `[2, this, hidden]`.
    pub position_embedding_size: u32,
}

impl Default for GemmaVisionConfig {
    fn default() -> Self {
        GemmaVisionConfig {
            patch_size: 16,
            pooling_kernel_size: 3,
            max_soft_tokens: 280,
            video_soft_tokens: 70,
            position_embedding_size: 10240,
        }
    }
}

impl GemmaVisionConfig {
    /// `pooling_kernel_size * patch_size`; both resize sides round down to
    /// this multiple.
    #[must_use]
    pub const fn side_mult(&self) -> u32 {
        self.pooling_kernel_size * self.patch_size
    }

    /// One patch row's width: `3 * patch_size^2` (RGB, no temporal axis).
    #[must_use]
    pub const fn patch_width(&self) -> usize {
        3 * self.patch_size as usize * self.patch_size as usize
    }

    /// The soft-token ceiling this budget allows.
    #[must_use]
    pub const fn soft_tokens(&self, budget: Budget) -> u32 {
        match budget {
            Budget::Still => self.max_soft_tokens,
            Budget::VideoFrame => self.video_soft_tokens,
        }
    }

    /// `max_patches = max_soft_tokens * pooling_kernel_size^2`.
    #[must_use]
    pub const fn max_patches(&self, budget: Budget) -> u32 {
        self.soft_tokens(budget) * self.pooling_kernel_size * self.pooling_kernel_size
    }

    /// Rounds the target size down to `side_mult`, so every image's patch
    /// grid is a whole number of `k x k` blocks.
    ///
    /// # Errors
    ///
    /// [`Fault::Empty`] if both sides round to zero, or the rescued target
    /// exceeds the patch budget.
    pub fn aspect_ratio_preserving_size(
        &self,
        h: u32,
        w: u32,
        budget: Budget,
    ) -> Result<(u32, u32)> {
        let side_mult = self.side_mult();
        let max_patches = self.max_patches(budget);
        let total_px = f64::from(h) * f64::from(w);
        let target_px = f64::from(max_patches) * f64::from(self.patch_size).powi(2);
        let factor = (target_px / total_px).sqrt();

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let down = |ideal: f64| -> u32 { (ideal / f64::from(side_mult)).floor() as u32 * side_mult };

        let mut target_h = down(factor * f64::from(h));
        let mut target_w = down(factor * f64::from(w));

        if target_h == 0 && target_w == 0 {
            return Err(Fault::Empty(format!(
                "a {h} x {w} image resizes to 0 x 0: both sides round down to a multiple of \
                 `pooling_kernel_size · patch_size` = {side_mult} and neither reaches one"
            )));
        }

        // One side survived: give the other one block, cap the survivor by
        // aspect ratio.
        let max_side_length =
            (max_patches / (self.pooling_kernel_size * self.pooling_kernel_size)) * side_mult;
        if target_h == 0 {
            target_h = side_mult;
            target_w = (w / h * side_mult).min(max_side_length);
        } else if target_w == 0 {
            target_w = side_mult;
            target_h = (h / w * side_mult).min(max_side_length);
        }

        if f64::from(target_h) * f64::from(target_w) > target_px {
            return Err(Fault::Empty(format!(
                "resizing [{h}x{w}] to [{target_h}x{target_w}] exceeds {max_patches} patches at \
                 patch size {}",
                self.patch_size
            )));
        }
        Ok((target_h, target_w))
    }

    /// Emits patch rows in pool-block-major order (block-row, block-column,
    /// then row/column within the block) so pooling needs no geometry. Each
    /// row is HWC, values `2 · (v / 255 − 0.5)`: the processor rescales to
    /// `[0, 1]` and normalizes nothing (`do_normalize: false`), and the
    /// MODEL then centres — `Gemma4VisionPatchEmbedder.forward`'s "Gemma4
    /// applies no normalization and instead scales in model code",
    /// `2 * (pixel_values - 0.5)`, the same line in mlx_vlm's `_patchify`.
    /// The tower text has no such op, so the front-end folds it in here.
    /// Fed `[0, 1]`, the 31B tower called a red square "pink" and a blue one
    /// "purple"; mlx_vlm on the same 4-bit weights says red and blue.
    ///
    /// # Panics
    ///
    /// If `rgb` is shorter than `h * w * 3`.
    #[must_use]
    pub fn patchify(&self, rgb: &[u8], h: u32, w: u32) -> (Vec<f32>, Vec<u32>) {
        let p = self.patch_size as usize;
        let k = self.pooling_kernel_size as usize;
        let (h, w) = (h as usize, w as usize);
        assert!(
            rgb.len() >= h * w * 3,
            "patchify was handed {} bytes for a {h} x {w} RGB image, which needs {}",
            rgb.len(),
            h * w * 3
        );
        let (gh, gw) = (h / p, w / p);
        let (bh, bw) = (gh / k, gw / k);
        let n = gh * gw;
        let pd = self.patch_width();

        let mut pix = vec![0.0f32; n * pd];
        let mut pos = vec![0u32; n * 2];

        let mut out_idx = 0usize;
        for ib_r in 0..bh {
            for ib_c in 0..bw {
                for ir in 0..k {
                    for ic in 0..k {
                        let pr = ib_r * k + ir;
                        let pc = ib_c * k + ic;
                        #[allow(clippy::cast_possible_truncation)]
                        {
                            // `(x, y)`, the processor's `meshgrid(..,
                            // indexing="xy")` order: table 0 and the first
                            // rotary block are the COLUMN's.
                            pos[2 * out_idx] = pc as u32;
                            pos[2 * out_idx + 1] = pr as u32;
                        }
                        let base = out_idx * pd;
                        for r in 0..p {
                            for col in 0..p {
                                for ch in 0..3 {
                                    let src = ((pr * p + r) * w + (pc * p + col)) * 3 + ch;
                                    pix[base + (r * p + col) * 3 + ch] =
                                        2.0 * (f32::from(rgb[src]) / 255.0 - 0.5);
                                }
                            }
                        }
                        out_idx += 1;
                    }
                }
            }
        }
        (pix, pos)
    }

    /// Two gathers into the `[2, position_embedding_size, hidden]` position
    /// table, summed as two `embed_weighted` taps over the table flattened
    /// to `[2 * size, hidden]`.
    #[must_use]
    pub fn pos_embed_taps(&self, positions: &[u32]) -> (Vec<i32>, Vec<f32>) {
        let plane = self.position_embedding_size;
        let rows = positions.len() / 2;
        let mut ids = Vec::with_capacity(rows * 2);
        for row in positions.chunks_exact(2) {
            let (x, y) = (row[0], row[1]);
            #[allow(clippy::cast_possible_wrap)]
            {
                ids.push(x.min(plane - 1) as i32);
                ids.push((plane + y.min(plane - 1)) as i32);
            }
        }
        (ids, vec![1.0f32; rows * 2])
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Gemma4Vision {
    /// The processor's constants.
    pub config: GemmaVisionConfig,
}

impl Gemma4Vision {
    /// The front-end at the checkpoint's own numbers.
    #[must_use]
    pub fn new() -> Gemma4Vision {
        Gemma4Vision::default()
    }
}

impl VisionFrontEnd for Gemma4Vision {
    fn arch(&self) -> &'static str {
        ARCH
    }

    /// Gemma-4's own delimiters, not gemma-3's `<start_of_image>`.
    fn delimiters(&self) -> Delimiters {
        Delimiters {
            prefix: super::tokenizer::IMAGE_PREFIX,
            placeholder: super::tokenizer::IMAGE_PAD,
            suffix: super::tokenizer::IMAGE_SUFFIX,
        }
    }

    /// A video frame gets a smaller soft-token cap than a still image.
    fn encode(&self, src: &Rgb8, budget: Budget, resample: Resample) -> Result<EncodedSpan> {
        let c = self.config;
        let (target_h, target_w) = c.aspect_ratio_preserving_size(src.h, src.w, budget)?;
        let (gh, gw) = (target_h / c.patch_size, target_w / c.patch_size);
        let k2 = c.pooling_kernel_size * c.pooling_kernel_size;
        let token_count = gh * gw / k2;
        if token_count == 0 {
            return Err(Fault::Empty(format!(
                "a {} x {} image resized to a {gh} x {gw} patch grid, which is fewer than one \
                 {k} x {k} pooling block and so occupies no token rows",
                src.h,
                src.w,
                k = c.pooling_kernel_size
            )));
        }

        let resized = resample(src, target_h, target_w);
        let (payload, positions) = c.patchify(&resized.data, resized.h, resized.w);
        let (embed_rows, embed_weights) = c.pos_embed_taps(&positions);

        Ok(EncodedSpan {
            token_count,
            // No mrope: position advances by the rows occupied (1-D).
            position_span: token_count,
            grid: Grid::still(1, token_count),
            patch_grid: Grid::still(gh, gw),
            uses_mrope: false,
            payload,
            rows: gh * gw,
            positions,
            embed_rows,
            embed_weights,
            prefix: Vec::new(),
            placeholder: 0,
            suffix: Vec::new(),
        })
    }
}
