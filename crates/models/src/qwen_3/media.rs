//! Qwen3.5 / Qwen3.6 vision preprocessing: resize, patchify, position taps.

use crate::media::{Budget, Delimiters, EncodedSpan, Fault, Grid, Resample, Result, Rgb8,
    VisionFrontEnd};

/// The `ROWS.arch` this front-end answers for.
///
/// Shared by both qwen3.5 and qwen3.6; their preprocessing is identical even
/// though the towers differ in block count and width.
pub const ARCH: &str = "qwen3_5";

/// The processor's constants, as a field rather than a `const` so a future
/// SKU can vary them without forking the type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QwenVisionConfig {
    /// Pixels per patch side.
    pub patch_size: u32,
    /// `spatial_merge_size` — the merger folds `merge²` patch rows into one.
    pub merge_size: u32,
    /// `temporal_patch_size`. A still image repeats itself across this axis.
    pub temporal_patch_size: u32,
    /// Lower bound on `h̄ · w̄`; below it `smart_resize` scales UP.
    pub min_pixels: u32,
    /// Upper bound on `h̄ · w̄`; above it `smart_resize` scales DOWN.
    pub max_pixels: u32,
    /// `int(num_position_embeddings ** 0.5)` — the learned table's grid side.
    pub num_grid_per_side: u32,
    /// `image_mean` / `image_std`, per channel: qwen's are 0.5 (pixels land
    /// in `[-1, 1]`), GLM's tower keeps CLIP's.
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for QwenVisionConfig {
    fn default() -> Self {
        QwenVisionConfig {
            patch_size: 16,
            merge_size: 2,
            temporal_patch_size: 2,
            min_pixels: 65536,
            max_pixels: 16_777_216,
            num_grid_per_side: 48,
            mean: [0.5; 3],
            std: [0.5; 3],
        }
    }
}

impl QwenVisionConfig {
    /// `patch_size · spatial_merge_size` — what both sides round to.
    #[must_use]
    pub const fn factor(&self) -> u32 {
        self.patch_size * self.merge_size
    }

    /// How wide one patch row is: `C · T · P²`.
    #[must_use]
    pub const fn patch_width(&self) -> usize {
        3 * self.temporal_patch_size as usize * self.patch_size as usize * self.patch_size as usize
    }

    /// Rounds both sides to the nearest multiple of `factor`; if the product
    /// exceeds `max_pixels` scales down and floors to the factor, if under
    /// `min_pixels` scales up and ceils to it.
    ///
    /// The round branch must not clamp a side to `max(factor, ·)`: a zero
    /// side is the signal that lets the min-pixels branch rescue the image,
    /// and clamping it early can leave the product above `min_pixels` so
    /// neither branch fires.
    ///
    /// # Errors
    ///
    /// [`Fault::Empty`] for an absolute aspect ratio past 200.
    pub fn smart_resize(&self, h: u32, w: u32) -> Result<(u32, u32)> {
        let (hf, wf) = (f64::from(h), f64::from(w));
        let (long, short) = (hf.max(wf), hf.min(wf));
        if short <= 0.0 || long / short > 200.0 {
            return Err(Fault::Empty(format!(
                "a {h} x {w} image has an absolute aspect ratio of {:.4}, and this processor                  refuses anything past 200",
                long / short.max(1.0)
            )));
        }

        let factor = f64::from(self.factor());
        let round_f = |x: f64| (x / factor).round() * factor;
        let floor_f = |x: f64| (x / factor).floor() * factor;
        let ceil_f = |x: f64| (x / factor).ceil() * factor;

        let mut h_bar = round_f(hf);
        let mut w_bar = round_f(wf);

        if h_bar * w_bar > f64::from(self.max_pixels) {
            let beta = (hf * wf / f64::from(self.max_pixels)).sqrt();
            h_bar = floor_f(hf / beta).max(factor);
            w_bar = floor_f(wf / beta).max(factor);
        } else if h_bar * w_bar < f64::from(self.min_pixels) {
            let beta = (f64::from(self.min_pixels) / (hf * wf)).sqrt();
            h_bar = ceil_f(hf * beta);
            w_bar = ceil_f(wf * beta);
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Ok((h_bar as u32, w_bar as u32))
    }

    /// The pre-merge patch grid a source image of `(h, w)` resizes into.
    ///
    /// # Errors
    ///
    /// [`smart_resize`](QwenVisionConfig::smart_resize)'s.
    pub fn patch_grid(&self, h: u32, w: u32) -> Result<(u32, u32)> {
        let (h_bar, w_bar) = self.smart_resize(h, w)?;
        Ok((h_bar / self.patch_size, w_bar / self.patch_size))
    }

    /// The placeholder run's length: one token per merge block.
    #[must_use]
    pub const fn token_count(&self, gh: u32, gw: u32) -> u32 {
        gh * gw / (self.merge_size * self.merge_size)
    }

    /// How far the 1-D position cursor advances past the span; not the token
    /// count, but `max(merged_h, merged_w)`.
    #[must_use]
    pub const fn position_span(&self, gh: u32, gw: u32) -> u32 {
        let (hm, wm) = (gh / self.merge_size, gw / self.merge_size);
        if hm > wm { hm } else { wm }
    }

    /// Patchifies into merge-block-major order: block-row, block-column, then
    /// row and column inside the 2x2 block; lanes within a row are channel,
    /// then temporal, then patch row, then patch column.
    ///
    /// Normalization is `(v / 255 - 0.5) / 0.5` (SigLIP's mean/std, not
    /// CLIP's). `rgb` is `h * w * 3` bytes, row-major HWC.
    ///
    /// Also returns each row's `(y, x)` in the patch grid.
    ///
    /// # Panics
    ///
    /// If `rgb` is shorter than `h * w * 3`.
    #[must_use]
    pub fn patchify(&self, rgb: &[u8], h: u32, w: u32) -> (Vec<f32>, Vec<u32>) {
        let p = self.patch_size as usize;
        let m = self.merge_size as usize;
        let tp = self.temporal_patch_size as usize;
        let (h, w) = (h as usize, w as usize);
        assert!(
            rgb.len() >= h * w * 3,
            "patchify was handed {} bytes for a {h} x {w} RGB image, which needs {}",
            rgb.len(),
            h * w * 3
        );
        let (gh, gw) = (h / p, w / p);
        let (bh, bw) = (gh / m, gw / m);
        let n = gh * gw;
        let pd = self.patch_width();

        let mut pix = vec![0.0f32; n * pd];
        let mut pos = vec![0u32; n * 2];
        let norm = |v: u8, ch: usize| -> f32 { ((f32::from(v) / 255.0) - self.mean[ch]) / self.std[ch] };

        let mut out_idx = 0usize;
        for ih_blk in 0..bh {
            for iw_blk in 0..bw {
                for ih in 0..m {
                    for iw in 0..m {
                        let pr = ih_blk * m + ih;
                        let pc = iw_blk * m + iw;
                        #[allow(clippy::cast_possible_truncation)]
                        {
                            pos[2 * out_idx] = pr as u32;
                            pos[2 * out_idx + 1] = pc as u32;
                        }
                        let base = out_idx * pd;
                        for ch in 0..3 {
                            for t in 0..tp {
                                for r in 0..p {
                                    for col in 0..p {
                                        let off = ((ch * tp + t) * p + r) * p + col;
                                        let src = ((pr * p + r) * w + (pc * p + col)) * 3 + ch;
                                        pix[base + off] = norm(rgb[src], ch);
                                    }
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

    /// The learned position table's taps and weights, in [`patchify`]'s own
    /// row order. Always four taps per row; on the native grid, [`axis_taps`]
    /// puts weight 1 on the patch's own row and 0 on the rest.
    ///
    /// [`patchify`]: QwenVisionConfig::patchify
    #[must_use]
    pub fn pos_embed_taps(&self, gh: u32, gw: u32) -> (Vec<i32>, Vec<f32>) {
        let m = self.merge_size as usize;
        let side = self.num_grid_per_side as usize;
        let (gh, gw) = (gh as usize, gw as usize);
        let (bh, bw) = (gh / m, gw / m);

        let mut ids = Vec::with_capacity(gh * gw * 4);
        let mut weights = Vec::with_capacity(gh * gw * 4);
        for ih_blk in 0..bh {
            for iw_blk in 0..bw {
                for ih in 0..m {
                    for iw in 0..m {
                        let (tap, weight) =
                            interp(ih_blk * m + ih, iw_blk * m + iw, gh, gw, side);
                        ids.extend_from_slice(&tap);
                        weights.extend_from_slice(&weight);
                    }
                }
            }
        }
        (ids, weights)
    }
}

/// Bilinear interpolation, two taps, `align_corners = True`.
///
/// `index` is the target position on an axis of length `size`; `side` is the
/// stored table's. `max(1)` on the denominator guards `size == 1`.
#[must_use]
pub fn axis_taps(index: usize, size: usize, side: usize) -> ([usize; 2], [f32; 2]) {
    #[allow(clippy::cast_precision_loss)]
    let src = index as f32 * (side as f32 - 1.0) / (size.saturating_sub(1).max(1)) as f32;
    let floor = src.floor();
    let mut taps = [0usize; 2];
    let mut weights = [0f32; 2];
    for (t, offset) in [0f32, 1f32].into_iter().enumerate() {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let tap = (floor as i64 + offset as i64).clamp(0, side as i64 - 1) as usize;
        taps[t] = tap;
        // The linear hat kernel.
        weights[t] = (1.0 - (src - floor - offset).abs()).max(0.0);
    }
    (taps, weights)
}

/// The 2-D case: separable outer product of the two axes' taps and weights,
/// four per patch. `indices = h_taps * side + w_taps`.
#[must_use]
pub fn interp(
    row: usize,
    col: usize,
    grid_h: usize,
    grid_w: usize,
    side: usize,
) -> ([i32; 4], [f32; 4]) {
    let (h_taps, h_w) = axis_taps(row, grid_h, side);
    let (w_taps, w_w) = axis_taps(col, grid_w, side);
    let mut ids = [0i32; 4];
    let mut weights = [0f32; 4];
    for a in 0..2 {
        for b in 0..2 {
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            {
                ids[a * 2 + b] = (h_taps[a] * side + w_taps[b]) as i32;
            }
            weights[a * 2 + b] = h_w[a] * w_w[b];
        }
    }
    (ids, weights)
}

/// Qwen3.5 / Qwen3.6's vision front-end.
#[derive(Clone, Copy, Debug, Default)]
pub struct Qwen35Vision {
    /// The processor's constants.
    pub config: QwenVisionConfig,
}

impl Qwen35Vision {
    /// The front-end at the checkpoint's own numbers.
    #[must_use]
    pub fn new() -> Qwen35Vision {
        Qwen35Vision::default()
    }
}

impl VisionFrontEnd for Qwen35Vision {
    fn arch(&self) -> &'static str {
        ARCH
    }

    /// Names the three specials rather than their ids; the runtime resolves
    /// them through the tokenizer, so a checkpoint that renumbers them still
    /// works.
    fn delimiters(&self) -> Delimiters {
        Delimiters {
            prefix: super::tokenizer::VISION_START,
            placeholder: super::tokenizer::IMAGE_PAD,
            suffix: super::tokenizer::VISION_END,
        }
    }

    /// `budget` is ignored: qwen caps a span by pixels, not soft-token count,
    /// and a video frame uses the same ceiling as a still.
    fn encode(&self, src: &Rgb8, _budget: Budget, resample: Resample) -> Result<EncodedSpan> {
        let c = self.config;
        let (h_bar, w_bar) = c.smart_resize(src.h, src.w)?;
        let (gh, gw) = (h_bar / c.patch_size, w_bar / c.patch_size);
        let token_count = c.token_count(gh, gw);
        if token_count == 0 {
            return Err(Fault::Empty(format!(
                "a {} x {} image resized to a {gh} x {gw} patch grid, which is fewer than one \
                 {m} x {m} merge block and so occupies no token rows",
                src.h,
                src.w,
                m = c.merge_size
            )));
        }

        let resized = resample(src, h_bar, w_bar);
        let (payload, positions) = c.patchify(&resized.data, resized.h, resized.w);
        let (embed_rows, embed_weights) = c.pos_embed_taps(gh, gw);

        Ok(EncodedSpan {
            token_count,
            position_span: c.position_span(gh, gw),
            grid: Grid::still(gh / c.merge_size, gw / c.merge_size),
            patch_grid: Grid::still(gh, gw),
            uses_mrope: true,
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
