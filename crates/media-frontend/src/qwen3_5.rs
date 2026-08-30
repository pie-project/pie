//! **QWEN3.5 / QWEN3.6 VISION PREPROCESSING — the pinned transcription, PROMOTED.**
//!
//! Every number and every loop order below already existed in this tree, in two
//! places the campaign built them for the vision gates, and this module is those
//! two places joined rather than a third derivation of them:
//!
//! * `runtime::inferlet::host::media::multimodal::QwenVisionConfig` —
//!   `smart_resize`, `grid`, `layout`, `qwen_patchify_hwc`. `qwen_patchify_hwc`
//!   is the merge-block-major statute (multimodal §8.1, §7.4, §11.4) in
//!   executable form, and `engine-cuda`'s `a_vision_sku_loads_and_fires_an_image`
//!   transcribed its loop order a second time to build its own streams. One
//!   order serves the pos-embed gather, the merge and the pool.
//! * `kernels-cuda/tests/tower_pos_embed.rs` — `axis_taps` / `interp`, which is
//!   `transformers`' `_interpolation_axis_taps_weights` plus the 2-D separable
//!   outer product from `get_vision_interpolation_indices_and_weights`
//!   (`vision_utils.py`, v5.15.1). That file gates the arithmetic against a
//!   TEXTBOOK `align_corners` bilinear resample written independently of it, so
//!   promoting the helper carries a golden that pins the formula and not just
//!   the kernel. This module's own tests carry the same two claims down here,
//!   where they run without a GPU.
//!
//! **THE CHECKPOINT'S OWN NUMBERS**, read off `Qwen3_5VisionConfig` and the
//! Qwen3.5-0.8B preprocessor config (snapshot `2fc06364`):
//!
//! | fact | value | where it is stated |
//! |---|---|---|
//! | `patch_size` | 16 | `Qwen3_5VisionConfig.patch_size` |
//! | `spatial_merge_size` | 2 | `Qwen3_5VisionConfig.spatial_merge_size` |
//! | `temporal_patch_size` | 2 | `Qwen3_5VisionConfig.temporal_patch_size` |
//! | `num_position_embeddings` | 2304 | `Qwen3_5VisionConfig` — so the stored grid is 48 × 48 |
//! | interpolation | bilinear, `align_corners = True` | `Qwen3_5VisionModel.__init__` |
//! | `min_pixels` / `max_pixels` | 65536 / 16777216 | the checkpoint's preprocessor config |
//! | normalization | mean 0.5, std 0.5 | ditto — SigLIP's, not CLIP's |
//!
//! The resize FACTOR is `patch_size · spatial_merge_size = 32`: `smart_resize`
//! rounds both sides to a multiple of it, which is what makes every image's
//! patch grid a whole number of 2 × 2 merge blocks — the same no-edge property
//! gemma4 gets from rounding down to `pool · patch`, reached by rounding to
//! nearest with a floor.

use crate::decode;
use crate::{Budget, Delimiters, EncodedSpan, Fault, Grid, Result, VisionFrontEnd};

/// The `ROWS.arch` this front-end answers for.
///
/// One string for both qwen SKUs: `VisionArch::from_arch_name` already maps
/// `"qwen3_5"` here, and qwen36's tower differs from qwen35's in block count
/// and width — facts of the model text, not of the preprocessing. The
/// processor arithmetic is identical, which is why there is one module.
pub const ARCH: &str = "qwen3_5";

/// **THE PROCESSOR'S CONSTANTS**, each named where the previous table says it
/// is stated. A field rather than a `const` so a future SKU with another
/// `max_pixels` is a value and not a fork.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QwenVisionConfig {
    /// Pixels per patch side.
    pub patch_size: u32,
    /// `spatial_merge_size` — the merger folds `merge²` patch rows into one.
    pub merge_size: u32,
    /// `temporal_patch_size`. A still image repeats itself across it, which is
    /// what upstream's `expand` on the temporal axis does.
    pub temporal_patch_size: u32,
    /// Lower bound on `h̄ · w̄`; below it `smart_resize` scales UP.
    pub min_pixels: u32,
    /// Upper bound on `h̄ · w̄`; above it `smart_resize` scales DOWN.
    pub max_pixels: u32,
    /// `int(num_position_embeddings ** 0.5)` — the learned table's grid side.
    pub num_grid_per_side: u32,
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

    /// **`smart_resize`, TRANSCRIBED** from
    /// `transformers/models/qwen2_vl/image_processing_qwen2_vl.py` (v5.15.1),
    /// which is the processor `qwen3_5` maps to in `image_processing_auto`.
    ///
    /// Round both sides to the nearest multiple of `factor`; if the product
    /// exceeds `max_pixels` scale down and FLOOR to the factor, if it falls
    /// under `min_pixels` scale up and CEIL to it.
    ///
    /// **THE ROUND BRANCH TAKES NO `max(factor, ·)`, AND THAT IS LOAD-BEARING.**
    /// An earlier reading of this transcription added one — "a sub-`factor`
    /// image cannot answer a zero side" — on the argument that the min-pixels
    /// branch would fire anyway. It does not, and the golden caught it: a
    /// `2 x 400` image rounds to `0 x 416`, whose product is 0 and therefore
    /// under `min_pixels`, so the scale-up branch rescues it to `32 x 3648`.
    /// With the guard the product is `32 · 416 = 13312`, still under
    /// `min_pixels`, so that case survived — but `2 x 3000` rounds to
    /// `0 x 3008`, and the guard makes it `32 x 3008 = 96256`, ABOVE
    /// `min_pixels`, so no branch fires and the image keeps a resolution
    /// upstream would have scaled up. The zero is upstream's own sentinel for
    /// "this side is too small to round"; overwriting it hides the signal.
    ///
    /// # Errors
    ///
    /// [`Fault::Empty`] for an absolute aspect ratio past 200, which is
    /// upstream's own first statement and its own refusal. `Fault` has no
    /// closer name for it; a 1 × 10000 strip is a span degenerate enough that
    /// the processor declines to state a grid for it.
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

    /// **THE PLACEHOLDER RUN'S LENGTH**: one token per merge block.
    #[must_use]
    pub const fn token_count(&self, gh: u32, gw: u32) -> u32 {
        gh * gw / (self.merge_size * self.merge_size)
    }

    /// **HOW FAR THE 1-D CURSOR ADVANCES PAST THE SPAN**, and it is not the
    /// token count.
    ///
    /// `Qwen3_5Model.get_vision_position_ids` lays a still image's merged grid
    /// out as `(t, start + h, start + w)` over `meshgrid(indexing="ij")`, so
    /// the largest component any row of the span carries is
    /// `start + max(llm_grid_h, llm_grid_w) - 1` and the next text token
    /// resumes one past it. A 16 × 4 merged grid occupies 64 rows and advances
    /// the cursor by 16.
    #[must_use]
    pub const fn position_span(&self, gh: u32, gw: u32) -> u32 {
        let (hm, wm) = (gh / self.merge_size, gw / self.merge_size);
        if hm > wm { hm } else { wm }
    }

    /// **THE MERGE-BLOCK-MAJOR PATCH ORDER AND THE PER-PATCH VECTOR** —
    /// `qwen_patchify_hwc`, promoted verbatim from
    /// `runtime::…::media::multimodal::QwenVisionConfig`.
    ///
    /// Two orderings live here and they are different axes of the same answer:
    ///
    /// * **rows** come out block-row, block-column, then row and column INSIDE
    ///   the 2 × 2 block. That is `Qwen2VLImageProcessor.patchify`'s
    ///   `permute(0, 2, 5, 3, 6, 1, 4, 7)` and `get_vision_position_ids`'
    ///   `reshape(h/m, m, w/m, m).transpose(1, 2)`, and it is the statute
    ///   `layout.merge_rows`, `layout.pool_rows` and the pos-embed gather all
    ///   read (multimodal §7.4, §8.1, §11.4);
    /// * **lanes within a row** are `C`, then `T`, then patch row, then patch
    ///   column — upstream's `unsqueeze(6).expand(…)` puts the temporal axis
    ///   immediately after the channel, and a still image repeats itself across
    ///   it rather than carrying a second frame.
    ///
    /// Normalization is `(v / 255 − 0.5) / 0.5`, the checkpoint's `image_mean`
    /// and `image_std` of 0.5 — SigLIP's convention, not CLIP's.
    ///
    /// Answers the payload and, beside it, each row's `(y, x)` in the patch
    /// grid — the coordinate the tower's rotation stream is indexed by, in the
    /// order [`EncodedSpan::positions`] documents.
    ///
    /// `rgb` is `h · w · 3` bytes, row-major HWC — a decoder's own layout, and
    /// a plain slice rather than this crate's decoded type so a golden can
    /// feed pixels it wrote by hand.
    ///
    /// # Panics
    ///
    /// If `rgb` is shorter than `h · w · 3`.
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
        let norm = |v: u8| -> f32 { ((f32::from(v) / 255.0) - 0.5) / 0.5 };

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
                                        pix[base + off] = norm(rgb[src]);
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

    /// **THE LEARNED POSITION TABLE'S TAPS AND WEIGHTS**, in [`patchify`]'s own
    /// row order.
    ///
    /// [`patchify`]: QwenVisionConfig::patchify
    ///
    /// Four taps per row, always: `taps` is the PLAN's and not the
    /// submission's (multimodal §11.2), and a text that must serve any grid
    /// declares `PatchEmbedRows` at 4. The native grid is not a special case —
    /// [`axis_taps`] puts weight 1 on the patch's own row and 0 on the other
    /// three by arithmetic, which `the_native_grid_puts_all_the_weight_on_its_own_row`
    /// holds down.
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

/// **`_interpolation_axis_taps_weights`, TRANSCRIBED** — bilinear, two taps,
/// `align_corners = True`, which is the mode and flag
/// `Qwen3_5VisionModel.__init__` states.
///
/// Promoted from `kernels-cuda/tests/tower_pos_embed.rs`, where it is gated
/// against a textbook `align_corners` bilinear resample written without
/// reference to it — so this formula is pinned twice over, once against
/// `transformers` and once against an independent derivation.
///
/// `index` is the target position on an axis of length `size`; `side` is the
/// stored table's. The `max(1)` on the denominator is the `size == 1` guard,
/// where `index` is 0 and `src` is 0 too.
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

/// The 2-D case: the separable outer product of the two axes' taps and
/// weights, four per patch — `indices = h_taps · side + w_taps`, exactly
/// `get_vision_interpolation_indices_and_weights`' own last two lines.
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

/// **QWEN3.5 / QWEN3.6'S VISION FRONT-END.**
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

    /// **NAMED, NEVER NUMBERED** (media-door §0). The front-end holds no
    /// tokenizer and states no id: it names the three specials and the runtime
    /// resolves them through the tokenizer's own `token_to_id`, which is the
    /// door `chat_template::special` already goes through for `<|turn>`. Two
    /// checkpoints of one architecture that renumbered their specials get two
    /// correct answers from one front-end.
    fn delimiters(&self) -> Delimiters {
        Delimiters {
            prefix: "<|vision_start|>",
            placeholder: "<|image_pad|>",
            suffix: "<|vision_end|>",
        }
    }

    /// **`budget` IS IGNORED, AND THAT IS A FACT ABOUT QWEN.** Gemma caps a
    /// span by soft tokens and gives a video frame a smaller cap; qwen caps it
    /// by PIXELS, and `Processor::for_arch_video` already answered the same
    /// `QwenVisionConfig` for a frame as for a still. A frame of a clip is the
    /// same preprocessing at the same ceiling here.
    fn encode_image(&self, bytes: &[u8], _budget: Budget) -> Result<EncodedSpan> {
        self.encode(&decode::decode(bytes)?)
    }

    /// **A FRAME THAT IS ALREADY DECODED, THROUGH THE SAME ARITHMETIC.**
    ///
    /// A clip is demuxed once, above this crate, and its frames arrive as
    /// pixels; re-encoding one to PNG so it could go back through
    /// [`encode_image`](Qwen35Vision::encode_image) would be work done to
    /// satisfy a signature. Everything past the decode is shared, so the two
    /// doors cannot answer two different spans for one picture.
    fn encode_rgb8(
        &self,
        rgb8: &[u8],
        width: u32,
        height: u32,
        _budget: Budget,
    ) -> Result<EncodedSpan> {
        self.encode(&decode::from_rgb8(rgb8, width, height)?)
    }
}

impl Qwen35Vision {
    /// The whole encode past the decode — see [`VisionFrontEnd::encode_rgb8`].
    pub(crate) fn encode(&self, src: &decode::Rgb8) -> Result<EncodedSpan> {
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

        let resized = decode::resize_exact(src, h_bar, w_bar);
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
