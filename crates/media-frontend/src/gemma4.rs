//! **GEMMA-4 VISION PREPROCESSING — the pooling tower's half of the statute.**
//!
//! Qwen merges by 2 and gemma pools by 3 (multimodal §8.4: "same rectangle,
//! same axis, same file, one row apart in what they do to the `k` rows they
//! read"), and everything that follows from that is the same sentence at
//! `k = 3`: the resize makes the grid a whole number of `k × k` blocks, the
//! patchifier emits POOL-BLOCK-MAJOR rows so `layout.pool_rows` reads no
//! geometry, and the placeholder run is one token per block.
//!
//! **THE CHECKPOINT'S OWN NUMBERS**, read off `Gemma4VisionConfig` and
//! `Gemma4ImageProcessor` (transformers v5.15.1) and the E4B snapshot:
//!
//! | fact | value | where it is stated |
//! |---|---|---|
//! | `patch_size` | 16 | `Gemma4VisionConfig.patch_size` |
//! | `pooling_kernel_size` | 3 | `Gemma4VisionConfig.pooling_kernel_size` |
//! | `max_soft_tokens` | 280 | `Gemma4ImageProcessor.max_soft_tokens` |
//! | `position_embedding_size` | 10240 | `Gemma4VisionConfig` — the `[2, 10240, 768]` table of multimodal §12.3 |
//! | rescale / normalize | `v / 255`, mean 0, std 1 | `Gemma4ImageProcessor.image_mean/std` — a plain rescale, unlike qwen's ±1 |
//! | resample | bicubic, antialias | `Gemma4ImageProcessor.resample` |
//!
//! **280 IS A BUDGET, NOT A LENGTH — and upstream says so in one line.**
//! `Gemma4Processor.replace_image_token` writes
//! `boi + image_token * num_soft_tokens + eoi`, and `num_soft_tokens` is
//! `patches.shape[0] // pooling_kernel_size**2`, which is at most
//! `max_soft_tokens` and equals it only when the resize landed on the budget
//! exactly. So the run this front-end spells is `gh · gw / 9`, capped by 280
//! through `max_patches = max_soft_tokens · k²` and not clamped to it.
//!
//! **AND THE PAYLOAD IS NOT PADDED.** `Gemma4ImageProcessor` pads every image's
//! patch rows out to `max_patches` with zeros and position `-1`, because it
//! stacks a ragged batch into one rectangle. Alto does not need that: the
//! submission's `rows` field says how many rows each image contributes and the
//! engine's own patch ladder does the padding, at the rung — §5.4's reasoning
//! about not paying for vectors nobody fills, and §7.4's "the only partial
//! block it tolerates is the RUNG TAIL, which is padding by construction". So
//! this front-end ships `gh · gw` real rows and no zeros.

use crate::decode;
use crate::{Budget, Delimiters, EncodedSpan, Fault, Grid, Result, VisionFrontEnd};

/// The `ROWS.arch` this front-end answers for.
pub const ARCH: &str = "gemma4";

/// **THE PROCESSOR'S CONSTANTS.**
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmaVisionConfig {
    /// Pixels per patch side.
    pub patch_size: u32,
    /// `pooling_kernel_size` — the soft-token pool folds `k²` patch rows.
    pub pooling_kernel_size: u32,
    /// A still image's soft-token ceiling.
    pub max_soft_tokens: u32,
    /// One video frame's, which is the whole reason [`Budget`] is an argument.
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
    /// `pooling_kernel_size · patch_size` — `side_mult` upstream, 48 here, and
    /// the multiple BOTH sides are rounded down to.
    #[must_use]
    pub const fn side_mult(&self) -> u32 {
        self.pooling_kernel_size * self.patch_size
    }

    /// How wide one patch row is: `C · P²`. No temporal axis — gemma's patch
    /// embedder is `[768, 768]` (multimodal §12.3), which is `3 · 16²`.
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

    /// `max_patches = max_soft_tokens · k²` — `Gemma4ImageProcessor._preprocess`'s
    /// own first line.
    #[must_use]
    pub const fn max_patches(&self, budget: Budget) -> u32 {
        self.soft_tokens(budget) * self.pooling_kernel_size * self.pooling_kernel_size
    }

    /// **`get_aspect_ratio_preserving_size`, TRANSCRIBED** from
    /// `transformers/models/gemma4/image_processing_gemma4.py` (v5.15.1),
    /// including both zero-side rescues and the final refusal.
    ///
    /// **THIS IS THE NO-EDGE PROOF, AND IT IS THE ROUNDING DIRECTION THAT
    /// CARRIES IT** (multimodal §7.4). The target is the largest aspect-
    /// preserving size that fits `max_patches` patches AND is divisible by
    /// `side_mult = k · patch` on both axes, and it is reached by rounding
    /// DOWN:
    ///
    /// ```text
    /// factor = sqrt(max_patches · patch² / (h · w))
    /// target = floor(factor · side / side_mult) · side_mult
    /// ```
    ///
    /// Rounding down is what makes every image's patch run a whole number of
    /// `k × k` blocks, which is what lets `layout.pool_rows` read no geometry:
    /// whole-block runs laid end to end never put a block across an image
    /// boundary, so the pool needs no indptr, no grid width and no atomics.
    /// Rounding UP would fit the patch budget's ceiling and break exactly that.
    ///
    /// And the edge case genuinely does not exist: upstream's
    /// `_avg_pool_by_positions` *raises* rather than rounding
    /// (`if k_squared * length != input_seq_len: raise`), so a non-divisible
    /// grid is not pooled by floor, ceil or pad — it is refused, and this
    /// resize is why it never arrives.
    ///
    /// # Errors
    ///
    /// [`Fault::Empty`] where upstream raises: an image whose aspect ratio is
    /// so extreme that both sides round to zero, and one whose rescued target
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

        // The two rescues, upstream's own: one side survived, so give the other
        // one block and let the survivor take the aspect ratio, capped.
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

    /// **THE POOL-BLOCK-MAJOR PATCH ORDER AND THE PER-PATCH VECTOR.**
    ///
    /// Two orderings, and only the second is upstream's verbatim:
    ///
    /// * **rows** come out block-row, block-column, then row and column INSIDE
    ///   the `k × k` block — the statute `layout.pool_rows` asks for
    ///   (multimodal §7.4). Upstream emits RASTER order and builds the pooling
    ///   as a one-hot matmul at runtime
    ///   (`Gemma4VisionPooler._avg_pool_by_positions`), which is a legible way
    ///   to write it in torch and an `O(patches²)` way to run it. Reordering
    ///   here turns the 2-D pool into a 1-D reduction that reads no geometry.
    ///   It is the same sentence qwen's `merge_size = 2` already writes.
    /// * **lanes within a row** are patch row, patch column, then CHANNEL —
    ///   `convert_image_to_patches`' `permute(1, 3, 2, 4, 0)`, i.e. HWC inside
    ///   the patch, which is the opposite of qwen's channel-major layout and is
    ///   what the `[768, 768]` patch embedder was trained against.
    ///
    /// Values are `v / 255`: `do_rescale` with `rescale_factor = 1/255` and
    /// `do_normalize = False` (`image_mean` 0, `image_std` 1).
    ///
    /// Answers the payload and each row's `(y, x)` in the patch grid.
    ///
    /// `rgb` is `h · w · 3` bytes, row-major HWC — a plain slice rather than
    /// this crate's decoded type so a golden can feed pixels it wrote by hand.
    ///
    /// # Panics
    ///
    /// If `rgb` is shorter than `h · w · 3`.
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
                            pos[2 * out_idx] = pr as u32;
                            pos[2 * out_idx + 1] = pc as u32;
                        }
                        let base = out_idx * pd;
                        for r in 0..p {
                            for col in 0..p {
                                for ch in 0..3 {
                                    let src = ((pr * p + r) * w + (pc * p + col)) * 3 + ch;
                                    pix[base + (r * p + col) * 3 + ch] = f32::from(rgb[src]) / 255.0;
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

    /// **THE SEPARABLE POSITION TABLE'S TWO TAPS, EACH AT WEIGHT ONE.**
    ///
    /// `Gemma4VisionEmbeddings.position_embedding_table` is
    /// `[2, position_embedding_size, hidden]` and
    /// `_position_embeddings` reads it as
    ///
    /// ```text
    /// x_emb = embedding(pixel_position_ids[..., 0], table[0])
    /// y_emb = embedding(pixel_position_ids[..., 1], table[1])
    /// position_embeddings = x_emb + y_emb
    /// ```
    ///
    /// — two gathers and a sum, where `[..., 0]` is the patch's COLUMN and
    /// `[..., 1]` is its ROW (the processor's `meshgrid(…, indexing="xy")`
    /// stacks `(x, y)`). A sum of two table rows is `embed_weighted` with two
    /// taps at weight 1, over the table flattened to `[2 · size, hidden]`
    /// — the leading axis of a `[2, size, hidden]` weight is "a `Slice` away"
    /// (multimodal §12.3), and flattened it is just the row offset below.
    ///
    /// So gemma resamples nothing and interpolates nothing: qwen's four taps
    /// carry a bilinear resample of a 48 × 48 grid, gemma's two carry a
    /// factorization. Same op, same stream, different arithmetic upstream of
    /// it — which is why [`EncodedSpan::embed_rows`] is a stream and not a
    /// resample.
    ///
    /// `positions` is [`patchify`](GemmaVisionConfig::patchify)'s `(y, x)`
    /// output; the taps swap them back to the table's `(x, y)` plane order.
    #[must_use]
    pub fn pos_embed_taps(&self, positions: &[u32]) -> (Vec<i32>, Vec<f32>) {
        let plane = self.position_embedding_size;
        let rows = positions.len() / 2;
        let mut ids = Vec::with_capacity(rows * 2);
        for row in positions.chunks_exact(2) {
            let (y, x) = (row[0], row[1]);
            #[allow(clippy::cast_possible_wrap)]
            {
                ids.push(x.min(plane - 1) as i32);
                ids.push((plane + y.min(plane - 1)) as i32);
            }
        }
        (ids, vec![1.0f32; rows * 2])
    }
}

/// **GEMMA-4'S VISION FRONT-END.**
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

    /// **GEMMA-4'S OWN DELIMITERS, AND THEY ARE NOT GEMMA-3'S.** This
    /// vocabulary spells its markers `<|x>` … `<x|>` with `<|x|>` for the
    /// standalone form — `<|turn>` / `<turn|>` is the pair
    /// `chat_template::gemma` already reads by name, and `<|audio>` /
    /// `<audio|>` is the pair the campaign's `audio_delimiters` pinned. Image
    /// is the third member of that family: `<|image>` opens,
    /// `<|image|>` is `image_token` (the reserved pad the run scan finds a
    /// span by), `<image|>` closes. `<start_of_image>` is gemma-3's spelling
    /// and is not in this checkpoint's vocabulary.
    ///
    /// The multimodal helper this crate promotes answered `("", "")` for
    /// gemma's delimiters — a gap left when only qwen had a served tower, and
    /// closed here.
    fn delimiters(&self) -> Delimiters {
        Delimiters {
            prefix: "<|image>",
            placeholder: "<|image|>",
            suffix: "<image|>",
        }
    }

    fn encode_image(&self, bytes: &[u8], budget: Budget) -> Result<EncodedSpan> {
        self.encode(&decode::decode(bytes)?, budget)
    }

    /// **A FRAME THAT IS ALREADY DECODED, THROUGH THE SAME ARITHMETIC** — see
    /// [`Qwen35Vision::encode_rgb8`](crate::qwen3_5::Qwen35Vision). Gemma DOES
    /// read the budget here: a video frame gets the smaller soft-token cap.
    fn encode_rgb8(
        &self,
        rgb8: &[u8],
        width: u32,
        height: u32,
        budget: Budget,
    ) -> Result<EncodedSpan> {
        self.encode(&decode::from_rgb8(rgb8, width, height)?, budget)
    }
}

impl Gemma4Vision {
    /// The whole encode past the decode — see [`VisionFrontEnd::encode_rgb8`].
    pub(crate) fn encode(&self, src: &decode::Rgb8, budget: Budget) -> Result<EncodedSpan> {
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

        let resized = decode::resize_exact(src, target_h, target_w);
        let (payload, positions) = c.patchify(&resized.data, resized.h, resized.w);
        let (embed_rows, embed_weights) = c.pos_embed_taps(&positions);

        Ok(EncodedSpan {
            token_count,
            // **1-D, AND THE GRID SAYS SO.** Gemma's trunk rotates scalar —
            // no mrope, no triple — so the span advances the cursor by exactly
            // the rows it occupies and its merged "grid" is a run.
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
