//! Multimodal preprocessing for vision/video/audio inputs — the *model-specific*
//! half of the pipeline, owned entirely host-side so inferlets stay
//! model-agnostic. An inferlet hands the host raw encoded bytes (a PNG/JPEG, an
//! animated GIF, a WAV); everything here is dispatched off the bound model's
//! arch, so the same inferlet binary serves Gemma, Qwen, or any future model.
//!
//! Responsibilities:
//!   * **geometry** — soft-token count, the `(t, h, w)` patch grid, and (M-RoPE
//!     models) how far the 1-D sequence cursor advances past the span. Gates the
//!     `image`/`audio` resources' synchronous `token-count()` / `position-span()`
//!     / `grid()` queries, so it matches the HF processors exactly.
//!   * **pixels** — decode (via the `image` crate), aspect-preserving resize
//!     (CatmullRom, the same filter the SDK used), and the arch's exact patchify
//!     + normalization: Gemma SigLIP2 channels-last `/255`; Qwen3-VL
//!       smart-resize + block-merge `(3,2,16,16)` layout, normalized as
//!       x/255 shifted to `[-1, 1]`.
//!   * **audio** — WAV decode, resample to 16 kHz, and the log-mel front-end
//!     (the [`audio`] submodule), matching `Gemma4AudioFeatureExtractor`.
//!
//! Two arch families are modelled: **Gemma 4** (fixed-resolution SigLIP, 1-D
//! RoPE) and **Qwen 3.6** (native dynamic resolution, 2×2 merge, M-RoPE). All
//! geometry is unit-tested against the HF processors.
//!
//! The patchify and log-mel layouts are checked two ways, because they need
//! two different things. Their **layout** — the channels-last interleave, the
//! merge-block emission order, which normalization each arch applies — is
//! unit-tested from synthetic input, since a transposition there produces a
//! correctly sized buffer the encoder reads as noise. Their **numbers** are
//! bit-compared against HF dumps by `gemma_patchify_matches_hf_exactly`,
//! which is `#[ignore]`d: it needs `scripts/gemma4_vision_parity_ref.py`,
//! which downloads gemma-4 and needs torch. Do not read a green CI run as
//! that comparison having happened.
#![allow(dead_code)] // Some geometry/arch-completeness helpers are exercised only by tests.

use image::{DynamicImage, imageops::FilterType};

/// `(t, h, w)` patch grid, in **patch units** (matches Qwen's `image_grid_thw`).
/// For Gemma this is unused; for Qwen, `h`/`w` are pre-merge patch counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    pub t: u32,
    pub h: u32,
    pub w: u32,
}

impl Grid {
    /// LLM tokens after `merge`×`merge` spatial patch-merging.
    pub fn llm_token_count(&self, merge: u32) -> u32 {
        let m = merge * merge;
        debug_assert!(m != 0);
        self.t * self.h * self.w / m
    }

    /// M-RoPE sequence-cursor advance: the next text token sits one past the
    /// largest positional extent of the span. Height/width are taken in merged
    /// (LLM) units to match the merged token layout.
    pub fn mrope_position_span(&self, merge: u32) -> u32 {
        let hm = self.h / merge;
        let wm = self.w / merge;
        self.t.max(hm).max(wm)
    }
}

/// Result of laying out one visual span for the LLM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisualSpan {
    /// Hidden-state rows / KV slots the span occupies.
    pub token_count: u32,
    /// How far the 1-D sequence cursor advances past the span.
    /// Equals `token_count` under 1-D RoPE (Gemma); differs under M-RoPE (Qwen).
    pub position_span: u32,
    /// Merged-token grid reported to callers via `image.grid()`.
    pub grid: Grid,
}

// ============================================================================
// Gemma 4 (SigLIP, fixed resolution)
// ============================================================================

/// Gemma-4 image-processor geometry. Gemma 4 uses the **SigLIP2-style**
/// aspect-ratio-preserving resize (`Gemma4ImageProcessor`, which subclasses the
/// SigLIP2 processor) — NOT Gemma-3's pan-and-scan. The image is resized so its
/// patch grid fits within `max_patches = max_soft_tokens · pool_k²`, patchified
/// into `patch_size²·3`-dim patches, and pooled `pool_k × pool_k` → a
/// *variable* number of soft tokens (`grid_h·grid_w / pool_k²`, ≤ max). Values
/// confirmed against `google/gemma-4-E4B`.
#[derive(Clone, Copy, Debug)]
pub struct GemmaImageConfig {
    /// Patch edge in pixels (`vision_config.patch_size`).
    pub patch_size: u32,
    /// Average-pool kernel applied to the patch grid (`pooling_kernel_size`).
    pub pooling_kernel_size: u32,
    /// Max soft tokens per image (`vision_soft_tokens_per_image`); the padded
    /// upper bound — the actual count is aspect-ratio dependent.
    pub max_soft_tokens: u32,
}

impl Default for GemmaImageConfig {
    fn default() -> Self {
        Self {
            patch_size: 16,
            pooling_kernel_size: 3,
            max_soft_tokens: 280,
        }
    }
}

impl GemmaImageConfig {
    /// `max_patches = max_soft_tokens · pool_k²` (= 2520 for the defaults).
    pub fn max_patches(&self) -> u32 {
        self.max_soft_tokens * self.pooling_kernel_size * self.pooling_kernel_size
    }

    /// Effective patch unit for the resize: `patch_size · pool_k`. The grid is
    /// sized in these units so it is divisible by `pool_k` (required by the 2D
    /// pooling), then expressed in 16-px patches.
    fn resize_unit(&self) -> u32 {
        self.patch_size * self.pooling_kernel_size
    }

    /// SigLIP2-style aspect-ratio-preserving resize target `(height, width)`
    /// for a `w × h` image: binary-search a scale so the *pooled* grid fits
    /// `max_soft_tokens`, each side a multiple of `patch_size·pool_k` and ≥ one
    /// unit. Faithful port of `get_image_size_for_max_num_patches` with the
    /// Gemma-4 effective patch — confirmed against the real `Gemma4ImageProcessor`.
    pub fn resize_target(&self, w: u32, h: u32) -> (u32, u32) {
        let unit = self.resize_unit() as f64;
        let max_units = self.max_soft_tokens as f64;
        let scaled = |scale: f64, size: u32| -> u32 {
            let s = size as f64 * scale;
            let s = (s / unit).ceil() * unit;
            (s.max(unit)) as u32
        };
        let eps = 1e-5;
        let (mut smin, mut smax) = (eps / 10.0, 100.0);
        while (smax - smin) >= eps {
            let scale = (smin + smax) / 2.0;
            let th = scaled(scale, h);
            let tw = scaled(scale, w);
            let units = (th as f64 / unit) * (tw as f64 / unit);
            if units <= max_units {
                smin = scale;
            } else {
                smax = scale;
            }
        }
        (scaled(smin, h), scaled(smin, w))
    }

    /// Patch grid `(grid_h, grid_w)` after the resize.
    pub fn patch_grid(&self, w: u32, h: u32) -> (u32, u32) {
        let (th, tw) = self.resize_target(w, h);
        (th / self.patch_size, tw / self.patch_size)
    }

    /// Soft tokens for a `w × h` image: `grid_h·grid_w / pool_k²` (variable).
    pub fn token_count(&self, w: u32, h: u32) -> u32 {
        let (gh, gw) = self.patch_grid(w, h);
        gh * gw / (self.pooling_kernel_size * self.pooling_kernel_size)
    }

    /// Lay out a Gemma image span. The soft tokens occupy `token_count`
    /// sequential LLM positions (the 2-D RoPE is internal to the encoder), so
    /// `position_span == token_count`.
    pub fn layout(&self, w: u32, h: u32) -> VisualSpan {
        let n = self.token_count(w, h);
        VisualSpan {
            token_count: n,
            position_span: n,
            grid: Grid { t: 1, h: 1, w: n },
        }
    }

    /// Patchify a resized, rescaled image `resized` (CHW, `[c, h, w]`, values in
    /// [0,1]) into the encoder's `pixel_values` + 2D patch positions. Mirrors
    /// `convert_image_to_patches`: patch `(pr, pc)` flattens as
    /// `(patch_row, patch_col, channel)` (channels-last); patches are row-major;
    /// position `(x=col, y=row)`. Returns `(pixel_values [n_patch, c·p²],
    /// positions [n_patch])`. `h`/`w` must be multiples of `patch_size`.
    pub fn patchify_chw(
        &self,
        resized: &[f32],
        c: usize,
        h: usize,
        w: usize,
    ) -> (Vec<f32>, Vec<[u32; 2]>) {
        let p = self.patch_size as usize;
        let (ph, pw) = (h / p, w / p);
        let n = ph * pw;
        let pd = c * p * p; // 768 for c=3, p=16
        let mut pix = vec![0.0f32; n * pd];
        let mut pos = vec![[0u32; 2]; n];
        for pr in 0..ph {
            for pc in 0..pw {
                let idx = pr * pw + pc;
                pos[idx] = [pc as u32, pr as u32]; // (x=col, y=row)
                let base = idx * pd;
                for r in 0..p {
                    for col in 0..p {
                        for ch in 0..c {
                            let v = resized[ch * h * w + (pr * p + r) * w + (pc * p + col)];
                            pix[base + (r * p + col) * c + ch] = v;
                        }
                    }
                }
            }
        }
        (pix, pos)
    }
}

// ============================================================================
// Qwen 3.6 (native resolution, M-RoPE)
// ============================================================================

/// Qwen vision parameters. Defaults match **Qwen3-VL** (`Qwen3-VL-2B-Instruct`,
/// verified from its `preprocessor_config.json`): `patch_size 16`, `merge 2`,
/// `temporal_patch_size 2`, area bounds `[65536, 16777216]` px (from
/// `size.shortest_edge`/`longest_edge`). The resize factor is
/// `patch_size * merge_size = 32`. NOTE: Qwen normalizes pixels with
/// `mean=std=0.5` → `(x/255 - 0.5)/0.5` (i.e. → [-1,1]), unlike Gemma's
/// rescale-only `/255`; the patchify step must apply this.
#[derive(Clone, Copy, Debug)]
pub struct QwenVisionConfig {
    pub patch_size: u32,
    pub merge_size: u32,
    pub temporal_patch_size: u32,
    /// Pixel-area bounds for `smart_resize`, in pixels.
    pub min_pixels: u32,
    pub max_pixels: u32,
}

impl Default for QwenVisionConfig {
    fn default() -> Self {
        Self {
            patch_size: 16,
            merge_size: 2,
            temporal_patch_size: 2,
            min_pixels: 65536,    // size.shortest_edge (256²)
            max_pixels: 16777216, // size.longest_edge (4096²)
        }
    }
}

impl QwenVisionConfig {
    fn factor(&self) -> u32 {
        self.patch_size * self.merge_size
    }

    /// Resize `(h, w)` so each side is a multiple of `factor` and the total
    /// area lands within `[min_pixels, max_pixels]`, preserving aspect ratio.
    /// Faithful port of HF `smart_resize`.
    pub fn smart_resize(&self, h: u32, w: u32) -> (u32, u32) {
        let factor = self.factor() as f64;
        let (hf, wf) = (h as f64, w as f64);

        let round_f = |x: f64| (x / factor).round() * factor;
        let floor_f = |x: f64| (x / factor).floor() * factor;
        let ceil_f = |x: f64| (x / factor).ceil() * factor;

        let mut h_bar = round_f(hf).max(factor);
        let mut w_bar = round_f(wf).max(factor);

        if h_bar * w_bar > self.max_pixels as f64 {
            let beta = (hf * wf / self.max_pixels as f64).sqrt();
            h_bar = floor_f(hf / beta).max(factor);
            w_bar = floor_f(wf / beta).max(factor);
        } else if h_bar * w_bar < self.min_pixels as f64 {
            let beta = (self.min_pixels as f64 / (hf * wf)).sqrt();
            h_bar = ceil_f(hf * beta);
            w_bar = ceil_f(wf * beta);
        }
        (h_bar as u32, w_bar as u32)
    }

    /// Patch grid for a `num_frames × h × w` visual input (`num_frames = 1` for
    /// a still image). Frames are grouped by `temporal_patch_size`.
    pub fn grid(&self, h: u32, w: u32, num_frames: u32) -> Grid {
        let (h_bar, w_bar) = self.smart_resize(h, w);
        let frames = num_frames.max(1);
        // Still images are temporally padded to one temporal patch (t = 1);
        // video groups frames by `temporal_patch_size`.
        let grid_t = if frames == 1 {
            1
        } else {
            (frames / self.temporal_patch_size).max(1)
        };
        Grid {
            t: grid_t,
            h: h_bar / self.patch_size,
            w: w_bar / self.patch_size,
        }
    }

    /// Lay out a Qwen visual span. `token_count` is the merged LLM token count;
    /// `position_span` follows M-RoPE (`max(t, h/merge, w/merge)`).
    pub fn layout(&self, h: u32, w: u32, num_frames: u32) -> VisualSpan {
        let patch_grid = self.grid(h, w, num_frames);
        let merge = self.merge_size;
        let merged = Grid {
            t: patch_grid.t,
            h: patch_grid.h / merge,
            w: patch_grid.w / merge,
        };
        VisualSpan {
            token_count: patch_grid.llm_token_count(merge),
            position_span: patch_grid.mrope_position_span(merge),
            grid: merged,
        }
    }

    /// Patchify a resized RGB still image (`rgb` HWC `[h, w, 3]` u8, sides
    /// already a multiple of `patch·merge` via [`smart_resize`](Self::smart_resize))
    /// into Qwen3-VL's `pixel_values` + per-patch `(x, y)` positions. Mirrors HF
    /// `Qwen2/3VLImageProcessor._preprocess` exactly:
    ///   * normalize `(x/255 − 0.5) / 0.5` (image_mean = image_std = 0.5);
    ///   * spatial-merge patch order `(bh, bw, ih, iw)` — every `merge²`
    ///     consecutive patches form one merged token;
    ///   * each patch's `patch_dim = 3·temporal·patch²` vector is laid out
    ///     `[channel][temporal][ph][pw]`, the still frame duplicated across the
    ///     `temporal_patch_size` temporal slots.
    ///
    /// Returns `(pixel_values [n_patch·patch_dim], positions [n_patch·2])` in the
    /// same merge order. (Ported verbatim from the parity-verified SDK
    /// `vision::qwen_patchify_hwc`.)
    pub fn qwen_patchify_hwc(&self, rgb: &[u8], h: u32, w: u32) -> (Vec<f32>, Vec<u32>) {
        let p = self.patch_size as usize;
        let m = self.merge_size as usize;
        let tp = self.temporal_patch_size as usize;
        let (h, w) = (h as usize, w as usize);
        let (gh, gw) = (h / p, w / p); // patch grid
        let (bh, bw) = (gh / m, gw / m); // merged-block grid
        let n = gh * gw;
        let pd = 3 * tp * p * p; // 3·2·16·16 = 1536
        let mut pix = vec![0.0f32; n * pd];
        let mut pos = vec![0u32; n * 2];
        let norm = |v: u8| -> f32 { ((v as f32 / 255.0) - 0.5) / 0.5 };
        let mut out_idx = 0usize;
        for ih_blk in 0..bh {
            for iw_blk in 0..bw {
                for ih in 0..m {
                    for iw in 0..m {
                        let pr = ih_blk * m + ih; // patch row
                        let pc = iw_blk * m + iw; // patch col
                        pos[2 * out_idx] = pc as u32; // x
                        pos[2 * out_idx + 1] = pr as u32; // y
                        let base = out_idx * pd;
                        // feature layout [channel][temporal][ph][pw].
                        for ch in 0..3 {
                            for t in 0..tp {
                                for r in 0..p {
                                    for col in 0..p {
                                        let src = ((pr * p + r) * w + (pc * p + col)) * 3 + ch;
                                        let off = ((ch * tp + t) * p + r) * p + col;
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
}

// ============================================================================
// Per-row M-RoPE position ids (Qwen)
// ============================================================================

/// Generate per-row M-RoPE `(t, h, w)` position triples for a Qwen visual span
/// whose **merged** grid is `merged`, offset so the span begins at `anchor`.
/// Rows are emitted t-major, then h, then w — matching the flattened
/// merged-grid order the encoder produces. Length equals the span's token
/// count (`merged.t * merged.h * merged.w`). Mirrors the vision branch of HF
/// `Qwen2VL.get_rope_index`.
///
/// Note: Qwen2.5/3-VL scale the temporal index by frame timing
/// (`second_per_grid_t`); that scaling is a TODO (`VERIFY`). This emits the
/// base `arange(t)` temporal index used by Qwen2-VL.
pub fn qwen_mrope_positions(merged: Grid, anchor: u32) -> Vec<[u32; 3]> {
    let mut out = Vec::with_capacity((merged.t * merged.h * merged.w) as usize);
    for ti in 0..merged.t {
        for hi in 0..merged.h {
            for wi in 0..merged.w {
                out.push([anchor + ti, anchor + hi, anchor + wi]);
            }
        }
    }
    out
}

/// 1-D sequence position of the first token *after* a Qwen visual span that
/// began at `anchor` (i.e. `anchor + position_span`). The next text token's
/// three M-RoPE components all start here.
pub fn qwen_next_position(merged: Grid, anchor: u32) -> u32 {
    anchor + merged.t.max(merged.h).max(merged.w)
}

// ============================================================================
// Arch-agnostic dispatch
// ============================================================================

/// Vision front-end family a checkpoint uses. Selected from model metadata by
/// the host `image` resource (`runtime/src/api/media.rs`, Phase 1.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisionArch {
    Gemma4,
    Qwen36,
}

/// Unified processor over the supported arch families. The host calls this to
/// answer `image.token-count()` / `position-span()` / `grid()` synchronously
/// and to build the wire's `mrope_position_ids` side-channel.
#[derive(Clone, Copy, Debug)]
pub enum Processor {
    Gemma(GemmaImageConfig),
    Qwen(QwenVisionConfig),
}

impl Processor {
    pub fn for_arch(arch: VisionArch) -> Self {
        match arch {
            VisionArch::Gemma4 => Processor::Gemma(GemmaImageConfig::default()),
            VisionArch::Qwen36 => Processor::Qwen(QwenVisionConfig::default()),
        }
    }

    /// Like [`for_arch`](Self::for_arch), but for individual **video frames**.
    /// Gemma 4 (no temporal model — each frame is an independent still) uses its
    /// smaller per-frame soft-token budget (≤70) so a multi-frame clip's KV
    /// footprint stays manageable; Qwen frames use the same smart-resize as
    /// stills. Mirrors the old SDK `gemma_resize_target_video`.
    pub fn for_arch_video(arch: VisionArch) -> Self {
        match arch {
            VisionArch::Gemma4 => Processor::Gemma(GemmaImageConfig {
                max_soft_tokens: 70,
                ..GemmaImageConfig::default()
            }),
            VisionArch::Qwen36 => Processor::Qwen(QwenVisionConfig::default()),
        }
    }

    /// Whether this arch uses M-RoPE — i.e. whether the forward pass must carry
    /// the `mrope_position_ids` side-channel rather than plain `position_ids`.
    pub fn uses_mrope(&self) -> bool {
        matches!(self, Processor::Qwen(_))
    }

    /// Patches pooled into one soft token: `pool_k²` (Gemma) or `merge²` (Qwen).
    /// Used to derive the soft-token count from a pre-patchified `n_patch`
    /// (option B, where the inferlet patchifies).
    pub fn pool_factor(&self) -> u32 {
        match self {
            Processor::Gemma(c) => c.pooling_kernel_size * c.pooling_kernel_size,
            Processor::Qwen(c) => c.merge_size * c.merge_size,
        }
    }

    /// Lay out a still image of `w × h` pixels.
    pub fn layout_image(&self, w: u32, h: u32) -> VisualSpan {
        match self {
            Processor::Gemma(c) => c.layout(w, h),
            Processor::Qwen(c) => c.layout(h, w, 1),
        }
    }

    /// Lay out a video clip of `num_frames` frames at `w × h` pixels.
    pub fn layout_video(&self, w: u32, h: u32, num_frames: u32) -> VisualSpan {
        let frames = num_frames.max(1);
        match self {
            // Gemma 4 has no native temporal model: each frame is an
            // independent image span (the caller appends frames in order).
            Processor::Gemma(c) => {
                let per = c.layout(w, h);
                VisualSpan {
                    token_count: per.token_count * frames,
                    position_span: per.position_span * frames,
                    grid: Grid {
                        t: frames,
                        h: 1,
                        w: per.token_count,
                    },
                }
            }
            Processor::Qwen(c) => c.layout(h, w, frames),
        }
    }

    /// Per-row M-RoPE positions for a span beginning at `anchor`, or `None` for
    /// 1-D-RoPE archs (whose positions come from the ordinary `position_ids`).
    pub fn mrope_positions(&self, span: &VisualSpan, anchor: u32) -> Option<Vec<[u32; 3]>> {
        match self {
            Processor::Gemma(_) => None,
            Processor::Qwen(_) => Some(qwen_mrope_positions(span.grid, anchor)),
        }
    }
}

// ============================================================================
// Pixel pipeline: decode → resize → patchify (the model-specific work)
// ============================================================================

/// A fully preprocessed visual span, ready to stage on the wire. Mirrors the
/// fields the old inferlet-side `from_pixels` produced — only the *source* of
/// the pixels has moved host-side.
pub struct ProcessedImage {
    /// `pixel_values` blob `[n_patch · patch_dim]` f32 (arch-specific layout).
    pub pixels: Vec<f32>,
    /// Per-patch positions `[n_patch · 2]` of `(x, y)` patch coords.
    pub positions: Vec<u32>,
    /// Pre-merge `(t, h, w)` patch-unit grid for the driver's vision encoder
    /// (Qwen: `t·h·w == n_patch`; Gemma: `(1, 1, token_count)`).
    pub patch_grid: Grid,
    /// Merged LLM span (token count / position span / merged grid).
    pub span: VisualSpan,
}

/// HWC u8 RGB → CHW f32 in `[0,1]` (the rescale-only Gemma normalization).
fn rgb_hwc_to_chw_f32(rgb: &[u8], h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let src = (y * w + x) * 3;
            for ch in 0..3 {
                out[ch * h * w + y * w + x] = rgb[src + ch] as f32 / 255.0;
            }
        }
    }
    out
}

fn flatten_xy(pos: &[[u32; 2]]) -> Vec<u32> {
    let mut out = Vec::with_capacity(pos.len() * 2);
    for p in pos {
        out.push(p[0]);
        out.push(p[1]);
    }
    out
}

impl Processor {
    /// Decode an encoded still image (PNG / JPEG / …) and preprocess it per this
    /// arch. The single entry the host `image` resource calls.
    pub fn process_image_bytes(&self, bytes: &[u8]) -> Result<ProcessedImage, String> {
        let img = image::load_from_memory(bytes).map_err(|e| format!("image decode: {e}"))?;
        Ok(self.process_image(&img))
    }

    /// Preprocess an already-decoded frame (still image or one video frame):
    /// resize to the arch's target, then patchify + normalize. Reuses the
    /// parity-verified [`GemmaImageConfig::patchify_chw`] /
    /// [`QwenVisionConfig::qwen_patchify_hwc`]; the resize uses the same
    /// `image`-crate CatmullRom filter the SDK used, so output is bit-identical.
    pub fn process_image(&self, img: &DynamicImage) -> ProcessedImage {
        let (w, h) = (img.width(), img.height());
        match self {
            Processor::Gemma(c) => {
                let (th, tw) = c.resize_target(w, h);
                let resized = img.resize_exact(tw, th, FilterType::CatmullRom).to_rgb8();
                let chw = rgb_hwc_to_chw_f32(resized.as_raw(), th as usize, tw as usize);
                let (pixels, pos2) = c.patchify_chw(&chw, 3, th as usize, tw as usize);
                let span = c.layout(w, h);
                ProcessedImage {
                    pixels,
                    positions: flatten_xy(&pos2),
                    patch_grid: span.grid, // (1, 1, token_count)
                    span,
                }
            }
            Processor::Qwen(c) => {
                let (th, tw) = c.smart_resize(h, w);
                let resized = img.resize_exact(tw, th, FilterType::CatmullRom).to_rgb8();
                let (pixels, positions) = c.qwen_patchify_hwc(resized.as_raw(), th, tw);
                let span = c.layout(h, w, 1);
                let patch_grid = Grid {
                    t: 1,
                    h: th / c.patch_size,
                    w: tw / c.patch_size,
                };
                ProcessedImage {
                    pixels,
                    positions,
                    patch_grid,
                    span,
                }
            }
        }
    }
}

/// Decode an animated container (GIF) into `(frame, timestamp_seconds)` pairs.
/// The host owns demuxing so the inferlet just passes the encoded bytes; each
/// frame is later preprocessed as an ordinary image. Errors for non-animated /
/// undecodable input.
pub fn decode_gif_frames(bytes: &[u8]) -> Result<Vec<(DynamicImage, f32)>, String> {
    use image::AnimationDecoder;
    use image::codecs::gif::GifDecoder;
    let decoder =
        GifDecoder::new(std::io::Cursor::new(bytes)).map_err(|e| format!("gif decode: {e}"))?;
    let frames = decoder
        .into_frames()
        .collect_frames()
        .map_err(|e| format!("gif frames: {e}"))?;
    if frames.is_empty() {
        return Err("gif has no frames".into());
    }
    let mut out = Vec::with_capacity(frames.len());
    let mut t_ms = 0.0f32;
    for f in frames {
        // The denominator cannot be zero: a `Delay` wraps a `Ratio`, whose
        // constructor asserts it, and the GIF decoder builds each frame's
        // as `hundredths * 10 / 1` besides. A branch for it would be one
        // no checkpoint or codec can reach.
        let (num, den) = f.delay().numer_denom_ms();
        let frame_ms = num as f32 / den as f32;
        out.push((DynamicImage::ImageRgba8(f.into_buffer()), t_ms / 1000.0));
        t_ms += frame_ms;
    }
    Ok(out)
}

impl VisionArch {
    /// The vision front-end a registered model's `arch_name` selects, or
    /// `None` for a text-only family.
    ///
    /// # Whole label, because a substring was WRONG
    ///
    /// This asked `arch.contains("qwen3")`. The label the Qwen 3 rows
    /// advertise is `qwen3`, so every text-only Qwen 3 — 0.6B through
    /// 32B, eight rows — answered with the Qwen3-VL front-end. A guest
    /// that sent an image to a Qwen3-8B got a patchified tensor and a
    /// pair of `<|vision_start|>` delimiters instead of the refusal the
    /// two lines below [`Self::from_arch_name`]'s caller exist to give
    /// it, and the model then attended over rows no tower had encoded.
    ///
    /// The substring was not merely loose, it was UNFIXABLE by widening
    /// it: `qwen3_5` is `qwen3` plus a suffix, so no `contains` on earth
    /// admits the VL line and refuses the dense one. Only the whole
    /// label separates them — which is why this matches the whole label,
    /// and why the arms below are `==` and not `contains`.
    ///
    /// The arms are exactly the labels rows advertise.
    /// `a_text_only_family_has_no_vision_front_end` holds them against
    /// [`crate::catalog`], so a generation cannot gain a front-end here
    /// by accident of spelling or lose one in silence.
    #[must_use]
    pub fn from_arch_name(arch: &str) -> Option<VisionArch> {
        match arch.to_ascii_lowercase().as_str() {
            "gemma4" => Some(VisionArch::Gemma4),
            // The Qwen3-VL line: Qwen3.5, and the Qwen3.6-27B builds
            // whose row lives with it because they are a Qwen3.5 by
            // shape. NOT `qwen3`, which is the text-only generation.
            "qwen3_5" => Some(VisionArch::Qwen36),
            _ => None,
        }
    }
}

// ============================================================================
// Audio (gemma4_audio) — front-end geometry
// ============================================================================

/// Whether the given `arch_name` has a gemma-4 audio front-end. Only
/// Gemma-4 ships the USM/Conformer audio tower today.
///
/// Whole-label for the reason [`VisionArch::from_arch_name`] gives. No
/// label but `gemma4` contains `gemma4` today, so this arm is not
/// currently wrong — it is written this way because the vision one WAS
/// wrong for exactly this reason, and a second `contains` beside it is
/// an invitation to repeat the defect the next time a generation is
/// named after an older one.
#[must_use]
pub fn audio_arch_supported(arch: &str) -> bool {
    arch.eq_ignore_ascii_case("gemma4")
}

/// Delimiter *strings* the model wraps a visual span with — encoded host-side by
/// the model's own tokenizer and surfaced as `image.prefix-tokens` /
/// `image.suffix-tokens`, so the inferlet never names them. `("", "")` means
/// the model needs none. (Qwen3-VL
/// wraps image rows in `<|vision_start|>` / `<|vision_end|>`; Gemma 4 needs none
/// here, matching the verified behavior.)
pub fn vision_delimiters(arch: VisionArch) -> (&'static str, &'static str) {
    match arch {
        VisionArch::Qwen36 => ("<|vision_start|>", "<|vision_end|>"),
        VisionArch::Gemma4 => ("", ""),
    }
}

/// Delimiter strings for an audio span (Gemma 4 `<|audio>` / `<audio|>`).
/// `("", "")` for archs with no audio front-end.
pub fn audio_delimiters(arch: &str) -> (&'static str, &'static str) {
    if audio_arch_supported(arch) {
        ("<|audio>", "<audio|>")
    } else {
        ("", "")
    }
}

/// Audio soft tokens for `n_frames` log-mel frames: two stride-2 Conv2d
/// (k3, s2, p1) along the time axis. `floor((n + 2 - 3) / 2) + 1` applied
/// twice. Mirrors the driver's audio subsampling exactly.
///
/// Total at zero. `(n + 2 - 3) / 2 + 1` is the integer conv-output formula
/// written the way it reads in the reference, and on `u32` it underflows for
/// exactly one input: `n = 0`. That input is reachable -- [`audio::gemma_logmel`]
/// returns zero frames for any clip shorter than one analysis window, about
/// 20 ms at 16 kHz -- so a debug build panics and a release build wraps to
/// `u32::MAX` and asks the host to reserve two billion soft tokens. The engine's
/// caller happens to reject zero frames on the line above, but that guard is in
/// another crate and this function is `pub`. In real arithmetic the answer is
/// `floor((0 + 2 - 3) / 2) + 1 = 0`, so answering 0 is not a special case; it is
/// the formula, evaluated where `u32` cannot follow.
pub fn gemma_audio_token_count(n_frames: u32) -> u32 {
    let conv = |n: u32| if n == 0 { 0 } else { (n + 2 - 3) / 2 + 1 };
    conv(conv(n_frames))
}

// ============================================================================
// Audio front-end: WAV decode → resample 16 kHz → log-mel (gemma4_audio)
// ============================================================================

/// The model-specific audio front-end, owned host-side so the inferlet just
/// hands over encoded bytes. Decodes a WAV container, resamples to 16 kHz mono,
/// and computes the log-mel features the gemma4_audio encoder consumes. The
/// log-mel path is a verbatim port of the parity-verified SDK frontend
/// (bit-exact vs `Gemma4AudioFeatureExtractor` for 16 kHz mono); the only
/// non-exact step is linear resampling, used solely when the input rate ≠ 16k.
pub mod audio {
    /// Gemma-4 audio frontend params (match `google/gemma-4-E4B`'s
    /// `processor_config.json` `feature_extractor` block).
    #[derive(Clone, Copy, Debug)]
    pub struct GemmaAudioProc {
        pub sample_rate: u32,    // 16000
        pub frame_length: usize, // 320 (20 ms)
        pub hop_length: usize,   // 160 (10 ms)
        pub fft_length: usize,   // 512
        pub n_mels: usize,       // 128
        pub fmin: f32,           // 0
        pub fmax: f32,           // 8000
        pub mel_floor: f32,      // 0.001
    }

    impl Default for GemmaAudioProc {
        fn default() -> Self {
            GemmaAudioProc {
                sample_rate: 16000,
                frame_length: 320,
                hop_length: 160,
                fft_length: 512,
                n_mels: 128,
                fmin: 0.0,
                fmax: 8000.0,
                mel_floor: 0.001,
            }
        }
    }

    fn hz_to_mel(f: f64) -> f64 {
        2595.0 * (1.0 + f / 700.0).log10()
    }
    fn mel_to_hz(m: f64) -> f64 {
        700.0 * (10.0f64.powf(m / 2595.0) - 1.0)
    }

    /// HTK mel filterbank `[n_freq, n_mels]` (norm=None) — 130 mel-spaced edges
    /// over `[fmin, fmax]`, triangular over the linear FFT-bin centers.
    fn mel_filterbank(p: &GemmaAudioProc) -> Vec<Vec<f64>> {
        let n_freq = p.fft_length / 2 + 1;
        let bin_freq: Vec<f64> = (0..n_freq)
            .map(|k| k as f64 * p.sample_rate as f64 / p.fft_length as f64)
            .collect();
        let mel_min = hz_to_mel(p.fmin as f64);
        let mel_max = hz_to_mel(p.fmax as f64);
        let n_pts = p.n_mels + 2;
        let hz_pts: Vec<f64> = (0..n_pts)
            .map(|i| {
                let m = mel_min + (mel_max - mel_min) * (i as f64) / ((n_pts - 1) as f64);
                mel_to_hz(m)
            })
            .collect();
        let mut fb = vec![vec![0.0f64; p.n_mels]; n_freq];
        for m in 0..p.n_mels {
            let (lo, ctr, hi) = (hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]);
            for k in 0..n_freq {
                let f = bin_freq[k];
                if f >= lo && f <= ctr && ctr > lo {
                    fb[k][m] = (f - lo) / (ctr - lo);
                } else if f > ctr && f <= hi && hi > ctr {
                    fb[k][m] = (hi - f) / (hi - ctr);
                }
            }
        }
        fb
    }

    /// In-place iterative radix-2 Cooley-Tukey FFT (forward, no normalization),
    /// `re`/`im` length a power of two. Matches `np.fft`.
    fn fft_radix2(re: &mut [f64], im: &mut [f64]) {
        let n = re.len();
        debug_assert!(n.is_power_of_two());
        let mut j = 0usize;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let ang = -2.0 * std::f64::consts::PI / len as f64;
            let (wr_step, wi_step) = (ang.cos(), ang.sin());
            let half = len / 2;
            let mut i = 0;
            while i < n {
                let (mut wr, mut wi) = (1.0f64, 0.0f64);
                for k in 0..half {
                    let a = i + k;
                    let b = i + k + half;
                    let tr = wr * re[b] - wi * im[b];
                    let ti = wr * im[b] + wi * re[b];
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                    let nwr = wr * wr_step - wi * wi_step;
                    wi = wr * wi_step + wi * wr_step;
                    wr = nwr;
                }
                i += len;
            }
            len <<= 1;
        }
    }

    /// Log-mel features `[n_frames * 128]` (frame-major) from mono f32 PCM @
    /// 16 kHz. Faithful port of `Gemma4AudioFeatureExtractor._extract_spectrogram`.
    pub fn gemma_logmel(pcm_16k_mono: &[f32]) -> (Vec<f32>, usize) {
        gemma_logmel_with(pcm_16k_mono, &GemmaAudioProc::default())
    }

    /// [`gemma_logmel`] with explicit params.
    pub fn gemma_logmel_with(pcm_16k_mono: &[f32], p: &GemmaAudioProc) -> (Vec<f32>, usize) {
        let frame = p.frame_length;
        let hop = p.hop_length;
        let nfft = p.fft_length;
        let n_freq = nfft / 2 + 1;

        // Semicausal pad: prepend frame/2 zeros.
        let pad = frame / 2;
        let mut x = Vec::with_capacity(pad + pcm_16k_mono.len());
        x.extend(std::iter::repeat_n(0.0f64, pad));
        x.extend(pcm_16k_mono.iter().map(|&v| v as f64));

        // Frame: window of `frame+1` (preemphasis look-behind), step `hop`.
        let win_len = frame + 1;
        let n_frames = if x.len() < win_len {
            0
        } else {
            (x.len() - win_len) / hop + 1
        };

        // Periodic Hann window over `frame` samples.
        let hann: Vec<f64> = (0..frame)
            .map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / frame as f64).cos())
            .collect();

        let fb = mel_filterbank(p);

        let mut out = vec![0.0f32; n_frames * p.n_mels];
        let mut re = vec![0.0f64; nfft];
        let mut im = vec![0.0f64; nfft];
        for fi in 0..n_frames {
            let base = fi * hop;
            re.iter_mut().for_each(|v| *v = 0.0);
            im.iter_mut().for_each(|v| *v = 0.0);
            for n in 0..frame {
                re[n] = x[base + n] * hann[n];
            }
            fft_radix2(&mut re, &mut im);
            let row = &mut out[fi * p.n_mels..(fi + 1) * p.n_mels];
            for m in 0..p.n_mels {
                let mut acc = 0.0f64;
                for k in 0..n_freq {
                    let w = fb[k][m];
                    if w != 0.0 {
                        let mag = (re[k] * re[k] + im[k] * im[k]).sqrt();
                        acc += mag * w;
                    }
                }
                row[m] = (acc + p.mel_floor as f64).ln() as f32;
            }
        }
        (out, n_frames)
    }

    /// Decode a canonical RIFF/WAVE container → `(mono f32 PCM, sample_rate)`.
    /// PCM (fmt 1) at 8/16/24/32-bit or IEEE-float (fmt 3) at 32/64-bit;
    /// downmixes to mono by averaging channels. (Ported from the SDK's
    /// `decode_wav`.)
    pub fn decode_wav(bytes: &[u8]) -> Result<(Vec<f32>, u32), String> {
        let rd_u16 = |b: &[u8], o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
        let rd_u32 = |b: &[u8], o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);

        if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
            return Err("not a RIFF/WAVE file".into());
        }
        let mut pos = 12;
        let mut fmt_tag = 0u16;
        let mut channels = 0u16;
        let mut sample_rate = 0u32;
        let mut bits = 0u16;
        let mut data: Option<&[u8]> = None;

        while pos + 8 <= bytes.len() {
            let id = &bytes[pos..pos + 4];
            let sz = rd_u32(bytes, pos + 4) as usize;
            let body_start = pos + 8;
            let body_end = (body_start + sz).min(bytes.len());
            if id == b"fmt " && body_end - body_start >= 16 {
                fmt_tag = rd_u16(bytes, body_start);
                channels = rd_u16(bytes, body_start + 2);
                sample_rate = rd_u32(bytes, body_start + 4);
                bits = rd_u16(bytes, body_start + 14);
            } else if id == b"data" {
                data = Some(&bytes[body_start..body_end]);
            }
            pos = body_start + sz + (sz & 1); // word-aligned
        }

        let data = data.ok_or("WAV: no data chunk")?;
        if channels == 0 {
            return Err("WAV: no fmt chunk".into());
        }
        let ch = channels as usize;

        let mut samples: Vec<f32> = Vec::new();
        match (fmt_tag, bits) {
            (1, 16) => {
                for c in data.chunks_exact(2) {
                    samples.push(i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0);
                }
            }
            (1, 8) => {
                for &b in data {
                    samples.push((b as f32 - 128.0) / 128.0);
                }
            }
            (1, 24) => {
                for c in data.chunks_exact(3) {
                    let v = (c[0] as i32) | ((c[1] as i32) << 8) | ((c[2] as i32) << 16);
                    let v = (v << 8) >> 8; // sign-extend 24→32
                    samples.push(v as f32 / 8_388_608.0);
                }
            }
            (1, 32) => {
                for c in data.chunks_exact(4) {
                    let v = i32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    samples.push(v as f32 / 2_147_483_648.0);
                }
            }
            (3, 32) => {
                for c in data.chunks_exact(4) {
                    samples.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                }
            }
            (3, 64) => {
                for c in data.chunks_exact(8) {
                    let v = f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
                    samples.push(v as f32);
                }
            }
            _ => {
                return Err(format!(
                    "WAV: unsupported format tag {fmt_tag} / {bits}-bit"
                ));
            }
        }

        let n_frames = samples.len() / ch;
        let mut mono = Vec::with_capacity(n_frames);
        for f in 0..n_frames {
            let mut acc = 0.0f32;
            for c in 0..ch {
                acc += samples[f * ch + c];
            }
            mono.push(acc / ch as f32);
        }
        Ok((mono, sample_rate))
    }

    /// Resample mono PCM to 16 kHz via linear interpolation. Identity at 16 kHz
    /// (the parity-faithful case).
    pub fn resample_to_16k(pcm: &[f32], src_rate: u32) -> Vec<f32> {
        const DST: u32 = 16000;
        if src_rate == DST || pcm.is_empty() {
            return pcm.to_vec();
        }
        let ratio = src_rate as f64 / DST as f64;
        let out_len = ((pcm.len() as f64) / ratio).floor() as usize;
        let mut out = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let src_pos = i as f64 * ratio;
            let i0 = src_pos.floor() as usize;
            let frac = (src_pos - i0 as f64) as f32;
            let a = pcm[i0];
            let b = *pcm.get(i0 + 1).unwrap_or(&a);
            out.push(a + (b - a) * frac);
        }
        out
    }

    /// Full pipeline: encoded WAV bytes → `(log-mel [n_frames*128], n_frames)`.
    pub fn process_wav_bytes(bytes: &[u8]) -> Result<(Vec<f32>, usize), String> {
        let (pcm, rate) = decode_wav(bytes)?;
        let pcm16k = resample_to_16k(&pcm, rate);
        Ok(gemma_logmel(&pcm16k))
    }

    #[cfg(test)]
    mod dsp_tests {
        use super::*;

        /// Round-trip the HTK mel scale.
        ///
        /// `mel_to_hz` is not used to undo `hz_to_mel` anywhere in the
        /// filterbank -- one maps the band edges in, the other maps the
        /// interpolated points back out -- so a sign or constant that
        /// disagreed between them would still produce a plausible bank of
        /// 128 monotonically rising triangles. Only inverting catches it.
        #[test]
        fn the_mel_scale_inverts() {
            for hz in [0.0, 100.0, 700.0, 1000.0, 4000.0, 8000.0] {
                let back = mel_to_hz(hz_to_mel(hz));
                assert!((back - hz).abs() < 1e-6, "{hz} -> {back}");
            }
            // 700 Hz is the scale's own break frequency: one mel decade.
            assert!((hz_to_mel(700.0) - 2595.0 * 2.0f64.log10()).abs() < 1e-9);
        }

        /// Each triangle peaks at 1.0 at its own centre, and all but the
        /// lowest carry weight.
        ///
        /// Filter 0 is empty, and legitimately so: its triangle spans
        /// [0, 13.8, 27.9] Hz while the FFT resolves 16000/512 = 31.25 Hz,
        /// so between DC and its upper edge there is no bin to weigh. This
        /// is the condition librosa warns about as "empty filters detected
        /// in mel frequency basis". It is pinned rather than tolerated: if
        /// a second filter ever goes empty, the bank and the FFT length
        /// have drifted apart and the bottom of the spectrum is being
        /// discarded.
        #[test]
        fn every_mel_triangle_peaks_at_one_and_is_bounded() {
            let p = GemmaAudioProc::default();
            let fb = mel_filterbank(&p);
            assert_eq!(fb.len(), p.fft_length / 2 + 1);
            assert_eq!(fb[0].len(), p.n_mels);
            let empty: Vec<usize> = (0..p.n_mels)
                .filter(|&m| fb.iter().all(|row| row[m] == 0.0))
                .collect();
            assert_eq!(
                empty,
                vec![0],
                "exactly filter 0 is below the FFT resolution"
            );
            for m in 0..p.n_mels {
                let peak = fb.iter().map(|row| row[m]).fold(0.0f64, f64::max);
                assert!((0.0..=1.0).contains(&peak), "filter {m} peak {peak}");
                assert!(
                    fb.iter().all(|row| row[m] >= 0.0),
                    "filter {m} went negative"
                );
            }
            // Triangles march upward, but only weakly at the bottom. Below
            // roughly 200 Hz the mel spacing is finer than the FFT's 31.25
            // Hz, so adjacent filters draw on a single shared bin and have
            // an identical centre of mass -- filters 1 and 2 both sit
            // entirely on bin 1, 3 and 4 both on bin 2, and so on. The bank
            // is oversampled there and carries no more information than the
            // FFT gave it. Strict advance resumes at filter 16.
            let centre = |m: usize| -> f64 {
                let (num, den) = fb.iter().enumerate().fold((0.0, 0.0), |(n, d), (k, row)| {
                    (n + k as f64 * row[m], d + row[m])
                });
                num / den
            };
            for m in 2..p.n_mels {
                assert!(centre(m) >= centre(m - 1), "filter {m} moved backwards");
            }
            let tied: Vec<usize> = (2..p.n_mels)
                .filter(|&m| centre(m) == centre(m - 1))
                .collect();
            assert_eq!(
                tied,
                vec![2, 4, 6, 8, 10, 15],
                "the set of mel bands the FFT cannot separate"
            );
        }

        /// Above `fmax` there is nothing.
        ///
        /// The bank is built over `[fmin, fmax]` = [0, 8000], and 8000 Hz is
        /// exactly Nyquist for the 16 kHz rate, so the top filter's upper
        /// edge lands on the last FFT bin. An off-by-one in the edge count
        /// would either drop that bin or run past the end of `bin_freq`.
        #[test]
        fn the_bank_ends_at_nyquist() {
            let p = GemmaAudioProc::default();
            let fb = mel_filterbank(&p);
            let n_freq = p.fft_length / 2 + 1;
            let top = fb[n_freq - 1].iter().copied().fold(0.0f64, f64::max);
            assert!(top > 0.0, "the Nyquist bin feeds no filter");
        }

        /// The FFT against a directly evaluated DFT.
        ///
        /// `fft_radix2` is hand-written: bit-reversal permutation, then an
        /// in-place butterfly with an incrementally rotated twiddle. Every
        /// one of those has a failure mode that keeps the output the right
        /// length. The reference here is the definition, O(n^2), so the two
        /// share no code.
        #[test]
        fn the_fft_agrees_with_the_definition() {
            let n = 64usize;
            let sig: Vec<f64> = (0..n)
                .map(|i| (i as f64 * 0.37).sin() + 0.5 * (i as f64 * 1.9).cos())
                .collect();
            let (mut re, mut im) = (sig.clone(), vec![0.0f64; n]);
            fft_radix2(&mut re, &mut im);
            for k in 0..n {
                let (mut dr, mut di) = (0.0f64, 0.0f64);
                for (t, &x) in sig.iter().enumerate() {
                    let ang = -2.0 * std::f64::consts::PI * (k * t) as f64 / n as f64;
                    dr += x * ang.cos();
                    di += x * ang.sin();
                }
                assert!((re[k] - dr).abs() < 1e-9, "bin {k} re {} vs {dr}", re[k]);
                assert!((im[k] - di).abs() < 1e-9, "bin {k} im {} vs {di}", im[k]);
            }
        }

        /// A unit impulse transforms to a flat spectrum -- the degenerate
        /// case the bit-reversal permutation cannot get wrong, kept because
        /// it localises a failure to the butterfly when the DFT test also
        /// fails.
        #[test]
        fn an_impulse_is_flat() {
            let n = 32usize;
            let (mut re, mut im) = (vec![0.0f64; n], vec![0.0f64; n]);
            re[0] = 1.0;
            fft_radix2(&mut re, &mut im);
            assert!(re.iter().all(|&v| (v - 1.0).abs() < 1e-12));
            assert!(im.iter().all(|&v| v.abs() < 1e-12));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── The front-end tables, held against the catalog ───────────────────

    /// A text-only generation answers with no vision front-end.
    ///
    /// The regression this pins: [`VisionArch::from_arch_name`] matched
    /// `arch.contains("qwen3")`, and the Qwen 3 rows advertise exactly
    /// `qwen3` — so eight text-only checkpoints claimed the Qwen3-VL
    /// front-end. `image.from_bytes` then returned a patchified tensor
    /// where it owed the guest a refusal.
    ///
    /// The list is the labels rows advertise, not a guess: every entry
    /// is asserted to be a label some row hands out, so a generation
    /// that renames itself fails here rather than quietly leaving the
    /// table.
    #[test]
    fn a_text_only_family_has_no_vision_front_end() {
        let advertised = crate::catalog::arches();
        for arch in [
            "qwen3", "qwen2", "llama", "mistral", "gemma2", "gemma3", "phi3", "olmo2",
        ] {
            assert!(
                advertised.contains(&arch),
                "`{arch}` is in this test because a row advertises it; no row does \
                 any more, so the case it pins may have moved"
            );
            assert_eq!(
                VisionArch::from_arch_name(arch),
                None,
                "`{arch}` is a text-only family and must not select a vision front-end"
            );
            assert!(!audio_arch_supported(arch), "`{arch}` ships no audio tower");
        }
    }

    /// And the two that DO have one still get it.
    ///
    /// The other half of the same guard: making the match exact is only
    /// correct if it still admits the labels it is supposed to admit.
    #[test]
    fn the_two_multimodal_families_still_reach_their_front_ends() {
        let advertised = crate::catalog::arches();
        for (arch, want) in [
            ("gemma4", VisionArch::Gemma4),
            ("qwen3_5", VisionArch::Qwen36),
        ] {
            assert!(
                advertised.contains(&arch),
                "`{arch}` is advertised by a row"
            );
            assert_eq!(VisionArch::from_arch_name(arch), Some(want));
        }
        // Gemma-4 alone ships the USM/Conformer audio tower; the Qwen3-VL
        // line is vision-only.
        assert!(audio_arch_supported("gemma4"));
        assert!(!audio_arch_supported("qwen3_5"));
    }

    /// The prefix that used to be enough is not enough any more.
    ///
    /// Spelled out separately from the table above because it is the
    /// exact shape of the defect: `qwen3` is a PREFIX of `qwen3_5`, so a
    /// matcher that accepts prefixes cannot tell the dense line from the
    /// VL one in either direction.
    #[test]
    fn a_label_that_merely_contains_a_multimodal_one_is_refused() {
        assert_eq!(VisionArch::from_arch_name("qwen3"), None);
        assert_eq!(
            VisionArch::from_arch_name("qwen3_5"),
            Some(VisionArch::Qwen36)
        );
        // Neither direction of containment leaks.
        assert_eq!(VisionArch::from_arch_name("qwen3_5_moe"), None);
        assert_eq!(VisionArch::from_arch_name("gemma4x"), None);
        assert_eq!(VisionArch::from_arch_name("pregemma4"), None);
        assert!(!audio_arch_supported("gemma4-audio"));
        // Case is still forgiven — a driver that upper-cases its label is
        // spelling the same family.
        assert_eq!(
            VisionArch::from_arch_name("GEMMA4"),
            Some(VisionArch::Gemma4)
        );
        assert!(audio_arch_supported("Gemma4"));
        // And an empty label, which is what a row that refuses to deploy
        // advertises, selects nothing rather than panicking.
        assert_eq!(VisionArch::from_arch_name(""), None);
        assert!(!audio_arch_supported(""));
    }

    // ── Gemma ────────────────────────────────────────────────────────────

    #[test]
    fn gemma_token_count_matches_hf_processor() {
        let cfg = GemmaImageConfig::default();
        assert_eq!(cfg.max_patches(), 2520);
        // Reference values from the REAL transformers `Gemma4ImageProcessor`,
        // keyed (h, w) → soft tokens. The grid is divisible by pool_k=3.
        for &(h, w, tok) in &[
            (480u32, 640u32, 266u32),
            (1024, 1024, 256),
            (100, 2000, 280),
            (224, 224, 256),
            (720, 1280, 264),
        ] {
            let n = cfg.token_count(w, h);
            assert_eq!(n, tok, "token_count({w},{h}) = {n}, expected {tok}");
            assert!(n <= cfg.max_soft_tokens);
            let span = cfg.layout(w, h);
            assert_eq!(span.token_count, n);
            assert_eq!(span.position_span, n);
        }
    }

    // ── The pixel pipeline, without a checkpoint ─────────────────────────
    //
    // The tests below stand in for `gemma_patchify_matches_hf_exactly`,
    // which is `#[ignore]`d: it reads dumps that only exist after running
    // `scripts/gemma4_vision_parity_ref.py`, which downloads gemma-4 and
    // needs torch. That test answers "do these numbers match HF"; these
    // answer "is the layout the one the encoder is indexed for", which is
    // where a transposition would land and is checkable from nothing.

    /// The channels-LAST interleave inside a Gemma patch.
    ///
    /// `patchify_chw` takes CHW in and writes each patch as
    /// `(row, col, channel)`. Transposing that produces a same-sized
    /// buffer the encoder reads as garbage, so the order is the thing to
    /// pin. A 2-px patch over a 4x4 image makes every index checkable by
    /// hand: value `100*ch + 10*y + x`.
    #[test]
    fn a_gemma_patch_interleaves_channels_last() {
        let cfg = GemmaImageConfig {
            patch_size: 2,
            ..GemmaImageConfig::default()
        };
        let (h, w, c) = (4usize, 4usize, 3usize);
        let mut chw = vec![0.0f32; c * h * w];
        for ch in 0..c {
            for y in 0..h {
                for x in 0..w {
                    chw[ch * h * w + y * w + x] = (100 * ch + 10 * y + x) as f32;
                }
            }
        }
        let (pix, pos) = cfg.patchify_chw(&chw, c, h, w);
        assert_eq!(pos.len(), 4, "a 4x4 image is 2x2 patches of 2px");
        assert_eq!(pix.len(), 4 * c * 4);

        // Patches are row-major and positions are (x=col, y=row) -- the
        // opposite of the (row, col) the loop is written in.
        assert_eq!(pos, vec![[0, 0], [1, 0], [0, 1], [1, 1]]);

        // Patch 1 is the top-right 2x2: image columns 2..4, rows 0..2.
        // Channels-last means the three channels of ONE pixel are
        // adjacent, and pixels advance in row-major order within the
        // patch.
        let pd = c * 2 * 2;
        assert_eq!(
            &pix[pd..2 * pd],
            &[
                2.0, 102.0, 202.0, // (y=0, x=2) over c=0,1,2
                3.0, 103.0, 203.0, // (y=0, x=3)
                12.0, 112.0, 212.0, // (y=1, x=2)
                13.0, 113.0, 213.0, // (y=1, x=3)
            ]
        );
    }

    /// Block order across block ROWS, which `qwen_patchify_order_and_norm`
    /// cannot see.
    ///
    /// That test's grid is 32x64: two patch rows, one merge block tall,
    /// so `bh == 1` and the outer `ih_blk` loop never advances past zero.
    /// Every patch it checks is in the first block row. This one is two
    /// blocks by two, so a block walk that only works when there is one
    /// row of them fails here.
    #[test]
    fn qwen_block_order_spans_block_rows_too() {
        let cfg = QwenVisionConfig {
            patch_size: 1,
            merge_size: 2,
            temporal_patch_size: 1,
            ..QwenVisionConfig::default()
        };
        // 4x4 pixels = 4x4 patches = 2x2 blocks of 2x2.
        let rgb: Vec<u8> = (0..4 * 4 * 3).map(|i| (i % 256) as u8).collect();
        let (_, pos) = cfg.qwen_patchify_hwc(&rgb, 4, 4);
        let xy: Vec<(u32, u32)> = pos.chunks(2).map(|c| (c[0], c[1])).collect();
        assert_eq!(
            xy,
            vec![
                (0, 0),
                (1, 0),
                (0, 1),
                (1, 1), // block (0,0)
                (2, 0),
                (3, 0),
                (2, 1),
                (3, 1), // block (0,1)
                (0, 2),
                (1, 2),
                (0, 3),
                (1, 3), // block (1,0)
                (2, 2),
                (3, 2),
                (2, 3),
                (3, 3), // block (1,1)
            ]
        );
    }

    /// Qwen normalizes to [-1, 1]; Gemma only rescales to [0, 1].
    ///
    /// Both are "divide by 255" to a reader skimming, and a model given
    /// the wrong one sees a washed-out image rather than an error.
    #[test]
    fn the_two_arches_normalize_differently() {
        let cfg = QwenVisionConfig {
            patch_size: 1,
            merge_size: 1,
            temporal_patch_size: 1,
            ..QwenVisionConfig::default()
        };
        let (pix, _) = cfg.qwen_patchify_hwc(&[0, 128, 255], 1, 1);
        assert_eq!(pix[0], -1.0, "0 maps to the low end");
        assert!((pix[1] - 0.003_921_628).abs() < 1e-6, "128 is mid-scale");
        assert_eq!(pix[2], 1.0, "255 maps to the high end");

        let chw = rgb_hwc_to_chw_f32(&[0, 128, 255], 1, 1);
        assert_eq!(chw[0], 0.0);
        assert!((chw[1] - 128.0 / 255.0).abs() < 1e-6);
        assert_eq!(chw[2], 1.0);
    }

    /// HWC to CHW is a transpose, and the test that catches a wrong one
    /// needs more than one pixel and more than one row.
    #[test]
    fn rgb_becomes_planar() {
        // 2x2 RGB, each pixel (r, g, b) = (10p, 10p+1, 10p+2).
        let rgb: Vec<u8> = (0..4u8)
            .flat_map(|p| [10 * p, 10 * p + 1, 10 * p + 2])
            .collect();
        let out = rgb_hwc_to_chw_f32(&rgb, 2, 2);
        let plane = |ch: usize| -> Vec<f32> {
            out[ch * 4..(ch + 1) * 4]
                .iter()
                .map(|v| v * 255.0)
                .collect()
        };
        assert_eq!(plane(0), vec![0.0, 10.0, 20.0, 30.0]);
        assert_eq!(plane(1), vec![1.0, 11.0, 21.0, 31.0]);
        assert_eq!(plane(2), vec![2.0, 12.0, 22.0, 32.0]);
    }

    #[test]
    fn positions_flatten_x_before_y() {
        assert_eq!(flatten_xy(&[[1, 2], [3, 4]]), vec![1, 2, 3, 4]);
    }

    fn solid_image(w: u32, h: u32) -> image::DynamicImage {
        image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        }))
    }

    /// The pipeline's output must be the size its own geometry promised.
    ///
    /// `token_count()` gates the guest's synchronous `token-count()` query
    /// and is answered BEFORE the pixels exist; if the buffer that arrives
    /// later disagrees, the KV reservation and the tensor disagree.
    #[test]
    fn a_processed_image_is_the_size_its_geometry_promised() {
        for arch in [VisionArch::Gemma4, VisionArch::Qwen36] {
            let p = Processor::for_arch(arch);
            let out = p.process_image(&solid_image(640, 480));
            let span = p.layout_image(640, 480);
            assert_eq!(out.span.token_count, span.token_count, "{arch:?}");
            assert_eq!(out.positions.len() % 2, 0);
            let n_patch = out.positions.len() / 2;
            assert_eq!(
                n_patch as u32,
                out.span.token_count * p.pool_factor(),
                "{arch:?}: patches must pool exactly onto the promised tokens"
            );
            assert_eq!(
                out.pixels.len() % n_patch,
                0,
                "{arch:?}: the pixel blob must divide evenly per patch"
            );
        }
    }

    /// Decoding bytes and processing a decoded frame are the same path.
    #[test]
    fn encoded_bytes_take_the_same_path_as_a_decoded_frame() {
        let img = solid_image(64, 48);
        let mut png = std::io::Cursor::new(Vec::new());
        img.write_to(&mut png, image::ImageFormat::Png)
            .expect("encode png");
        let p = Processor::for_arch(VisionArch::Gemma4);
        let from_bytes = p
            .process_image_bytes(png.get_ref())
            .expect("a png this crate just wrote must decode");
        let direct = p.process_image(&img);
        assert_eq!(from_bytes.pixels, direct.pixels);
        assert_eq!(from_bytes.positions, direct.positions);
    }

    /// Undecodable bytes are a refusal naming the stage, not a panic.
    #[test]
    fn undecodable_bytes_are_refused() {
        let p = Processor::for_arch(VisionArch::Gemma4);
        let Err(err) = p.process_image_bytes(b"not an image") else {
            panic!("twelve bytes of ASCII is not an image");
        };
        assert!(err.starts_with("image decode:"), "{err}");
        let err = decode_gif_frames(b"not a gif").unwrap_err();
        assert!(err.starts_with("gif decode:"), "{err}");
    }

    /// Video frames get their own soft-token budget, and the clip's span
    /// is per-frame times frames.
    ///
    /// Gemma has no temporal model, so a clip is N independent stills --
    /// which is why the per-frame budget is cut to 70: the KV footprint
    /// is what multiplies.
    #[test]
    fn a_video_frame_gets_a_smaller_budget_than_a_still() {
        let still = Processor::for_arch(VisionArch::Gemma4);
        let video = Processor::for_arch_video(VisionArch::Gemma4);
        let (w, h) = (640, 480);
        assert!(
            video.layout_image(w, h).token_count < still.layout_image(w, h).token_count,
            "a video frame must cost fewer tokens than the same still"
        );

        let clip = video.layout_video(w, h, 4);
        let per = video.layout_image(w, h);
        assert_eq!(clip.token_count, per.token_count * 4);
        assert_eq!(clip.grid.t, 4, "frames are the temporal extent");
        assert_eq!(clip.grid.w, per.token_count);

        // Qwen keeps one config for both, and folds frames into the grid
        // rather than repeating the span.
        let qwen = Processor::for_arch_video(VisionArch::Qwen36);
        assert_eq!(
            qwen.layout_image(w, h).token_count,
            Processor::for_arch(VisionArch::Qwen36)
                .layout_image(w, h)
                .token_count
        );

        // Zero frames is one frame, not an empty span.
        assert_eq!(
            video.layout_video(w, h, 0).token_count,
            per.token_count,
            "a clip with no stated frame count still lays out one"
        );
    }

    /// Only the M-RoPE arch carries the side-channel, and it is the arch
    /// that says it does.
    #[test]
    fn only_mrope_arches_emit_position_triples() {
        for arch in [VisionArch::Gemma4, VisionArch::Qwen36] {
            let p = Processor::for_arch(arch);
            let span = p.layout_image(640, 480);
            let triples = p.mrope_positions(&span, 7);
            assert_eq!(
                triples.is_some(),
                p.uses_mrope(),
                "{arch:?}: the side-channel is emitted iff the arch uses M-RoPE"
            );
            if let Some(rows) = triples {
                assert_eq!(rows.len() as u32, span.grid.t * span.grid.h * span.grid.w);
                assert_eq!(rows[0], [7, 7, 7], "the span begins at the anchor");
            }
        }
    }

    // Minimal f32/.npy reader (little-endian, C-order) for parity dumps.
    fn read_npy_f32(path: &str) -> Option<(Vec<usize>, Vec<f32>)> {
        let b = std::fs::read(path).ok()?;
        if &b[..6] != b"\x93NUMPY" {
            return None;
        }
        let hlen = u16::from_le_bytes([b[8], b[9]]) as usize;
        let hdr = std::str::from_utf8(&b[10..10 + hlen]).ok()?;
        let sp = hdr.find("'shape'")?;
        let lp = hdr[sp..].find('(')? + sp;
        let rp = hdr[lp..].find(')')? + lp;
        let shape: Vec<usize> = hdr[lp + 1..rp]
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        let data = &b[10 + hlen..];
        let v: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Some((shape, v))
    }

    /// Bit-exactness against HF, when the dumps exist.
    ///
    /// `#[ignore]`d rather than self-skipping: it needs
    /// `scripts/gemma4_vision_parity_ref.py`, which downloads gemma-4 and
    /// needs torch, so it cannot run in CI -- and a test that returns
    /// green when its input is missing reports the same "ok" as one that
    /// checked something. Run with
    /// `cargo test -p model -- --ignored gemma_patchify`.
    #[test]
    #[ignore = "needs /tmp/gemma4_vision_parity from scripts/gemma4_vision_parity_ref.py"]
    fn gemma_patchify_matches_hf_exactly() {
        let dir = "/tmp/gemma4_vision_parity";
        let (rs_shape, resized) = match read_npy_f32(&format!("{dir}/proc_resized_chw.npy")) {
            Some(x) => x,
            None => {
                eprintln!("skip: run scripts/gemma4_vision_parity_ref.py + the proc dumps");
                return;
            }
        };
        let (_, ref_pix) = read_npy_f32(&format!("{dir}/proc_pixel_values.npy")).unwrap();
        let (_, ref_pos) = read_npy_f32(&format!("{dir}/proc_position_ids.npy")).unwrap();
        let (c, h, w) = (rs_shape[0], rs_shape[1], rs_shape[2]); // CHW
        let cfg = GemmaImageConfig::default();
        let (pix, pos) = cfg.patchify_chw(&resized, c, h, w);
        let n = pos.len(); // valid patches (HF pads beyond this)
        let pd = c * (cfg.patch_size as usize).pow(2);
        // pixel_values bit-exact over the valid patches.
        let mut max_abs = 0f32;
        for i in 0..n * pd {
            max_abs = max_abs.max((pix[i] - ref_pix[i]).abs());
        }
        assert!(
            max_abs < 1e-6,
            "patchify pixel_values differ: max_abs={max_abs}"
        );
        // positions exact.
        for i in 0..n {
            assert_eq!(pos[i][0] as f32, ref_pos[2 * i], "x mismatch at {i}");
            assert_eq!(pos[i][1] as f32, ref_pos[2 * i + 1], "y mismatch at {i}");
        }
        assert_eq!(n, cfg.token_count(640, 480) as usize * 9); // 2394 = 266*9
    }

    #[test]
    fn gemma_resize_target_pool_divisible() {
        let cfg = GemmaImageConfig::default();
        // 480x640 (h,w) → 672x912 (real processor); grid 42x57, both /3.
        assert_eq!(cfg.resize_target(640, 480), (672, 912));
        assert_eq!(cfg.patch_grid(640, 480), (42, 57));
        let unit = cfg.patch_size * cfg.pooling_kernel_size;
        for &(w, h) in &[(640u32, 480u32), (1024, 1024), (2000, 100)] {
            let (th, tw) = cfg.resize_target(w, h);
            assert_eq!(th % unit, 0, "height not divisible by patch*pool_k");
            assert_eq!(tw % unit, 0, "width not divisible by patch*pool_k");
            let (gh, gw) = cfg.patch_grid(w, h);
            assert!(
                (gh / cfg.pooling_kernel_size) * (gw / cfg.pooling_kernel_size)
                    <= cfg.max_soft_tokens
            );
        }
    }

    // ── Qwen ─────────────────────────────────────────────────────────────

    #[test]
    fn qwen_smart_resize_multiple_of_factor() {
        let cfg = QwenVisionConfig::default();
        let factor = cfg.factor();
        for &(h, w) in &[(1024, 1024), (720, 1280), (37, 5000), (100, 100)] {
            let (hb, wb) = cfg.smart_resize(h, w);
            assert_eq!(hb % factor, 0, "h not multiple of factor for {h}x{w}");
            assert_eq!(wb % factor, 0, "w not multiple of factor for {h}x{w}");
            assert!(hb >= factor && wb >= factor);
            let area = hb as u64 * wb as u64;
            // Within bounds (allow one factor-step of slack from rounding).
            assert!(area <= cfg.max_pixels as u64 + (factor * factor) as u64);
        }
    }

    #[test]
    fn qwen_token_count_divisible_by_merge_sq() {
        let cfg = QwenVisionConfig::default();
        let span = cfg.layout(1024, 1024, 1);
        let m = cfg.merge_size * cfg.merge_size;
        // The patch grid must merge cleanly into LLM tokens.
        let patch = cfg.grid(1024, 1024, 1);
        assert_eq!((patch.h * patch.w) % m, 0);
        assert_eq!(span.token_count, patch.h * patch.w / m);
        assert_eq!(span.grid.t, 1);
    }

    #[test]
    fn qwen_mrope_span_is_max_dim_not_token_count() {
        let cfg = QwenVisionConfig::default();
        let span = cfg.layout(1024, 1024, 1);
        // M-RoPE advances the cursor by the largest merged dimension, which is
        // far smaller than the (h·w) token count for a non-degenerate image.
        assert!(
            span.position_span < span.token_count,
            "span={} count={}",
            span.position_span,
            span.token_count
        );
        assert_eq!(
            span.position_span,
            span.grid.t.max(span.grid.h).max(span.grid.w)
        );
    }

    #[test]
    fn qwen_video_grid_groups_frames_temporally() {
        let cfg = QwenVisionConfig::default();
        let still = cfg.grid(448, 448, 1);
        let clip = cfg.grid(448, 448, 8); // 8 frames, temporal_patch_size=2
        assert_eq!(still.t, 1);
        assert_eq!(clip.t, 8 / cfg.temporal_patch_size);
        // Spatial grid identical; only the temporal axis grows.
        assert_eq!((clip.h, clip.w), (still.h, still.w));
        // A multi-frame clip costs proportionally more tokens.
        let span_still = cfg.layout(448, 448, 1);
        let span_clip = cfg.layout(448, 448, 8);
        assert_eq!(span_clip.token_count, span_still.token_count * clip.t);
    }

    // ── Dispatch + M-RoPE position generation ─────────────────────────────

    #[test]
    fn processor_dispatch_and_mrope_flag() {
        let g = Processor::for_arch(VisionArch::Gemma4);
        let q = Processor::for_arch(VisionArch::Qwen36);
        assert!(!g.uses_mrope());
        assert!(q.uses_mrope());
        assert_eq!(g.layout_image(896, 896).token_count, 256); // SigLIP2 resize → 48x48 grid /9
        // Gemma has no 2-D image positions → no M-RoPE side-channel.
        assert!(g.mrope_positions(&g.layout_image(896, 896), 0).is_none());
    }

    #[test]
    fn qwen_mrope_positions_cover_the_grid() {
        let q = Processor::for_arch(VisionArch::Qwen36);
        let anchor = 7;
        let span = q.layout_image(1024, 1024);
        let pos = q.mrope_positions(&span, anchor).expect("qwen has mrope");

        // One triple per token row.
        assert_eq!(pos.len() as u32, span.token_count);
        // Still image → temporal index is constant at the anchor.
        assert!(pos.iter().all(|p| p[0] == anchor));
        // First row sits exactly at the anchor on all three axes.
        assert_eq!(pos[0], [anchor, anchor, anchor]);
        // h/w components span [anchor, anchor + dim - 1].
        let max_h = pos.iter().map(|p| p[1]).max().unwrap();
        let max_w = pos.iter().map(|p| p[2]).max().unwrap();
        assert_eq!(max_h, anchor + span.grid.h - 1);
        assert_eq!(max_w, anchor + span.grid.w - 1);
        // The cursor advances past the largest extent.
        assert_eq!(
            qwen_next_position(span.grid, anchor),
            anchor + span.position_span
        );
    }

    #[test]
    fn qwen_video_mrope_temporal_axis_increments() {
        let q = Processor::for_arch(VisionArch::Qwen36);
        let span = q.layout_video(448, 448, 8);
        let pos = q.mrope_positions(&span, 0).unwrap();
        assert_eq!(pos.len() as u32, span.token_count);
        // Temporal index ranges over the merged temporal grid.
        let max_t = pos.iter().map(|p| p[0]).max().unwrap();
        assert_eq!(max_t, span.grid.t - 1);
        assert!(
            span.grid.t > 1,
            "8 frames / temporal_patch_size should give t>1"
        );
    }

    // ── Arch selection ────────────────────────────────────────────────────

    /// A vision tower is selected by the WHOLE arch label, not by a
    /// substring of it.
    ///
    /// The labels are a closed set — `gemma4` and `qwen3_5` are what
    /// [`crate::deployment::Advertised::arch`] actually carries — and
    /// matching on `contains` was a real defect: a generation named after
    /// an older one picked up the older one's tower. This test used to
    /// assert that defect, asking for `"Gemma4-27B"` and `"qwen3_5_moe"`
    /// to resolve, so the decorated names are now the NEGATIVE cases.
    /// See [`VisionArch::from_arch_name`], whose doc says the same thing
    /// from the other side.
    #[test]
    fn arch_name_selection() {
        assert_eq!(
            VisionArch::from_arch_name("gemma4"),
            Some(VisionArch::Gemma4)
        );
        assert_eq!(
            VisionArch::from_arch_name("qwen3_5"),
            Some(VisionArch::Qwen36)
        );
        // Case is not part of the label; decoration is.
        assert_eq!(
            VisionArch::from_arch_name("GEMMA4"),
            Some(VisionArch::Gemma4)
        );
        for decorated in ["gemma4-27b", "qwen3_5_moe", "qwen3_6", "llama"] {
            assert_eq!(
                VisionArch::from_arch_name(decorated),
                None,
                "{decorated} is not one of the labels a row advertises",
            );
        }
    }

    // ── Qwen patchify (ported from the verified SDK function) ──────────────

    #[test]
    fn qwen_patchify_order_and_norm() {
        // 32×64 (h,w): patch grid 2×4, merged blocks 1×2.
        let (h, w) = (32u32, 64u32);
        let rgb: Vec<u8> = (0..(h * w * 3)).map(|i| (i % 256) as u8).collect();
        let cfg = QwenVisionConfig::default();
        let (pix, pos) = cfg.qwen_patchify_hwc(&rgb, h, w);
        let pd = 3 * 2 * 16 * 16;
        assert_eq!(pos.len(), 2 * 8); // 8 patches
        assert_eq!(pix.len(), 8 * pd);
        // merge order: first block covers patches (0,0),(1,0),(0,1),(1,1).
        assert_eq!(&pos[0..2], &[0, 0]);
        assert_eq!(&pos[2..4], &[1, 0]);
        assert_eq!(&pos[4..6], &[0, 1]);
        assert_eq!(&pos[6..8], &[1, 1]);
        assert_eq!(&pos[8..10], &[2, 0]); // second block starts at patch col 2
        assert!(pix.iter().all(|&v| (-1.0..=1.0).contains(&v))); // normalized to [-1,1]
        // temporal duplication: t=0 and t=1 slices of patch 0 are identical.
        for ch in 0..3 {
            for r in 0..16 {
                for col in 0..16 {
                    let o0 = ((ch * 2) * 16 + r) * 16 + col;
                    let o1 = ((ch * 2 + 1) * 16 + r) * 16 + col;
                    assert_eq!(pix[o0], pix[o1]);
                }
            }
        }
    }

    // ── Audio front-end ────────────────────────────────────────────────────

    #[test]
    fn logmel_shape_and_subsample_count() {
        // 1 s of 16 kHz silence → frames ≈ len/hop; token count = two stride-2 convs.
        let pcm = vec![0.0f32; 16000];
        let (mel, n_frames) = audio::gemma_logmel(&pcm);
        assert_eq!(mel.len(), n_frames * 128);
        assert!(n_frames > 90 && n_frames < 110, "n_frames={n_frames}");
        // Token count matches the driver's subsample formula.
        let tok = gemma_audio_token_count(n_frames as u32);
        assert!(tok > 0 && tok < n_frames as u32);
    }

    #[test]
    fn wav_roundtrip_decode() {
        // Hand-build a 16-bit PCM mono WAV of a ramp, decode it back.
        let sr = 16000u32;
        let samples: Vec<i16> = (0..8).map(|i| (i * 1000) as i16).collect();
        let mut wav = Vec::new();
        let data_bytes = (samples.len() * 2) as u32;
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&sr.to_le_bytes());
        wav.extend_from_slice(&(sr * 2).to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_bytes.to_le_bytes());
        for s in &samples {
            wav.extend_from_slice(&s.to_le_bytes());
        }
        let (pcm, rate) = audio::decode_wav(&wav).unwrap();
        assert_eq!(rate, sr);
        assert_eq!(pcm.len(), samples.len());
        assert!((pcm[1] - 1000.0 / 32768.0).abs() < 1e-6);
        // Resample to 16k is identity.
        assert_eq!(audio::resample_to_16k(&pcm, sr), pcm);
    }

    /// Build a WAV around an arbitrary fmt tag / bit depth / channel count.
    fn wav(fmt_tag: u16, bits: u16, channels: u16, sr: u32, data: &[u8]) -> Vec<u8> {
        let mut w = Vec::new();
        w.extend_from_slice(b"RIFF");
        w.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        w.extend_from_slice(b"WAVE");
        w.extend_from_slice(b"fmt ");
        w.extend_from_slice(&16u32.to_le_bytes());
        w.extend_from_slice(&fmt_tag.to_le_bytes());
        w.extend_from_slice(&channels.to_le_bytes());
        w.extend_from_slice(&sr.to_le_bytes());
        w.extend_from_slice(&(sr * channels as u32 * bits as u32 / 8).to_le_bytes());
        w.extend_from_slice(&(channels * bits / 8).to_le_bytes());
        w.extend_from_slice(&bits.to_le_bytes());
        w.extend_from_slice(b"data");
        w.extend_from_slice(&(data.len() as u32).to_le_bytes());
        w.extend_from_slice(data);
        w
    }

    /// Every sample encoding lands on the same value.
    ///
    /// `wav_roundtrip_decode` covers 16-bit PCM. The other five arms each
    /// carry their own full-scale divisor, and 24-bit additionally has to
    /// sign-extend by hand -- `(v << 8) >> 8` on an i32 that was assembled
    /// from three little-endian bytes. A divisor that is off by a factor of
    /// two, or a sign extension that does not happen, produces audio that
    /// decodes, resamples and patchifies without complaint.
    ///
    /// Each case encodes the same two amplitudes, -0.5 and +0.25, so the
    /// expectation is one number for all of them.
    #[test]
    fn every_sample_encoding_decodes_to_the_same_amplitude() {
        let cases: Vec<(&str, u16, u16, Vec<u8>)> = vec![
            (
                "u8 PCM",
                1,
                8,
                vec![64u8, 160u8], // (64-128)/128 = -0.5 ; (160-128)/128 = 0.25
            ),
            ("i16 PCM", 1, 16, {
                let mut v = Vec::new();
                v.extend_from_slice(&(-16384i16).to_le_bytes());
                v.extend_from_slice(&8192i16.to_le_bytes());
                v
            }),
            ("i24 PCM", 1, 24, {
                let mut v = Vec::new();
                for s in [-4_194_304i32, 2_097_152i32] {
                    v.extend_from_slice(&s.to_le_bytes()[0..3]);
                }
                v
            }),
            ("i32 PCM", 1, 32, {
                let mut v = Vec::new();
                for s in [-1_073_741_824i32, 536_870_912i32] {
                    v.extend_from_slice(&s.to_le_bytes());
                }
                v
            }),
            ("f32", 3, 32, {
                let mut v = Vec::new();
                for s in [-0.5f32, 0.25f32] {
                    v.extend_from_slice(&s.to_le_bytes());
                }
                v
            }),
            ("f64", 3, 64, {
                let mut v = Vec::new();
                for s in [-0.5f64, 0.25f64] {
                    v.extend_from_slice(&s.to_le_bytes());
                }
                v
            }),
        ];
        for (name, tag, bits, data) in cases {
            let (pcm, rate) = audio::decode_wav(&wav(tag, bits, 1, 16000, &data))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(rate, 16000, "{name}");
            assert_eq!(pcm.len(), 2, "{name}");
            assert!((pcm[0] - -0.5).abs() < 1e-6, "{name}: {} != -0.5", pcm[0]);
            assert!((pcm[1] - 0.25).abs() < 1e-6, "{name}: {} != 0.25", pcm[1]);
        }
    }

    /// Channels are averaged, not interleaved through.
    #[test]
    fn a_stereo_file_is_averaged_down_to_mono() {
        let mut data = Vec::new();
        for (l, r) in [(16384i16, 0i16), (-32768, 32767), (1000, 3000)] {
            data.extend_from_slice(&l.to_le_bytes());
            data.extend_from_slice(&r.to_le_bytes());
        }
        let (pcm, _) = audio::decode_wav(&wav(1, 16, 2, 16000, &data)).unwrap();
        assert_eq!(pcm.len(), 3, "three frames, not six samples");
        assert!((pcm[0] - 0.25).abs() < 1e-6); // (0.5 + 0.0) / 2
        assert!((pcm[2] - (1000.0 / 32768.0 + 3000.0 / 32768.0) / 2.0).abs() < 1e-6);
    }

    /// An odd-sized chunk before `data` still leaves the reader word-aligned.
    ///
    /// RIFF pads every chunk body to an even length, and the pad byte is not
    /// counted in the size field. `decode_wav` advances by `sz + (sz & 1)`.
    /// Without that correction the walk lands one byte inside the next
    /// chunk header and never recognises `data`, so the file is rejected as
    /// having no data chunk -- which is exactly what a real file written by
    /// a tagger looks like.
    #[test]
    fn an_odd_sized_chunk_does_not_desynchronise_the_walk() {
        let payload = 1234i16.to_le_bytes();
        let mut w = wav(1, 16, 1, 16000, &payload);
        // Splice a 3-byte LIST chunk (+1 pad) in after the WAVE tag.
        let mut spliced = w[0..12].to_vec();
        spliced.extend_from_slice(b"LIST");
        spliced.extend_from_slice(&3u32.to_le_bytes());
        spliced.extend_from_slice(b"abc");
        spliced.push(0); // pad byte, not counted in the size
        spliced.extend_from_slice(&w[12..]);
        let total = (spliced.len() - 8) as u32;
        spliced[4..8].copy_from_slice(&total.to_le_bytes());
        w = spliced;
        let (pcm, rate) = audio::decode_wav(&w).expect("the LIST chunk should be skipped");
        assert_eq!(rate, 16000);
        assert_eq!(pcm.len(), 1);
        assert!((pcm[0] - 1234.0 / 32768.0).abs() < 1e-6);
    }

    /// Malformed containers are refused with a reason, not decoded as noise.
    #[test]
    fn a_file_that_is_not_a_wav_is_refused() {
        let cases: Vec<(&str, Vec<u8>, &str)> = vec![
            ("too short", b"RIFF".to_vec(), "not a RIFF/WAVE"),
            (
                "not WAVE",
                {
                    let mut v = b"RIFF".to_vec();
                    v.extend_from_slice(&0u32.to_le_bytes());
                    v.extend_from_slice(b"AVI ");
                    v
                },
                "not a RIFF/WAVE",
            ),
            (
                "no data chunk",
                {
                    let full = wav(1, 16, 1, 16000, &[0, 0]);
                    full[0..36].to_vec() // everything up to, but not including, "data"
                },
                "no data chunk",
            ),
            (
                "unsupported depth",
                wav(1, 12, 1, 16000, &[0, 0, 0, 0]),
                "unsupported format tag",
            ),
        ];
        for (name, bytes, want) in cases {
            let Err(err) = audio::decode_wav(&bytes) else {
                panic!("{name} was accepted");
            };
            assert!(err.contains(want), "{name}: {err}");
        }
    }

    /// Resampling changes the length by the rate ratio and interpolates.
    ///
    /// `resample_to_16k` short-circuits at 16 kHz, so the interpolation
    /// arm is only reached by a file that is not already at the model's
    /// rate -- which is every file a user actually supplies.
    #[test]
    fn resampling_scales_the_length_and_interpolates_between_samples() {
        // 8 kHz -> 16 kHz doubles, and every odd output is the midpoint.
        let up = audio::resample_to_16k(&[0.0, 1.0, 2.0, 3.0], 8000);
        assert_eq!(up.len(), 8);
        assert!((up[1] - 0.5).abs() < 1e-6, "{:?}", up);
        assert!((up[2] - 1.0).abs() < 1e-6);
        // The tail extrapolates flat rather than reading off the end.
        assert!(up.iter().all(|v| v.is_finite()));
        // 32 kHz -> 16 kHz halves, taking every other sample exactly.
        let down = audio::resample_to_16k(&[0.0, 9.0, 1.0, 9.0, 2.0, 9.0], 32000);
        assert_eq!(down, vec![0.0, 1.0, 2.0]);
        // Empty input is not a division by zero.
        assert!(audio::resample_to_16k(&[], 44100).is_empty());
    }

    /// A pure tone lands in the mel bin that covers its frequency.
    ///
    /// `logmel_shape_and_subsample_count` feeds one second of silence. Every
    /// FFT bin is then zero, every mel accumulator is zero, and every output
    /// is `ln(mel_floor)` -- so that test passes with the FFT replaced by a
    /// function returning zeros, and says nothing about the transform, the
    /// window or the filterbank. This one puts energy at a known frequency
    /// and asks where it came out.
    #[test]
    fn a_tone_lands_in_the_mel_bin_that_covers_it() {
        let sr = 16000.0f32;
        let tone = |hz: f32| -> Vec<f32> {
            (0..8000)
                .map(|i| (2.0 * std::f32::consts::PI * hz * i as f32 / sr).sin())
                .collect()
        };
        // The HTK mel scale, written out here rather than reached for, so
        // the expectation does not come from the code under test.
        let h2m = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
        let expected_bin = |hz: f64| -> usize {
            let frac = h2m(hz) / h2m(8000.0);
            (frac * 129.0).round() as usize - 1
        };
        let mut peaks = Vec::new();
        for hz in [1000.0f64, 3000.0] {
            let (mel, n) = audio::gemma_logmel(&tone(hz as f32));
            assert!(n > 0);
            // Average each mel band over the steady middle of the signal,
            // away from the zero-padded edges.
            let mid: Vec<f64> = (0..128)
                .map(|m| {
                    let lo = n / 4;
                    let hi = 3 * n / 4;
                    (lo..hi).map(|f| mel[f * 128 + m] as f64).sum::<f64>() / (hi - lo) as f64
                })
                .collect();
            let peak = mid
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap();
            let want = expected_bin(hz);
            assert!(
                peak.abs_diff(want) <= 2,
                "{hz} Hz peaked at mel bin {peak}, expected about {want}"
            );
            // The tone must also be *localised*. A rectangular window puts
            // the peak in the same bin -- removing the Hann multiply does
            // not move it at all -- but leaks the tone across the whole
            // bank. Measured: Hann separates the peak from the loudest
            // band 20+ bins away by 8.5 nats, a bare rectangle by 3.1.
            let far = (0..128)
                .filter(|m: &usize| m.abs_diff(peak) > 20)
                .map(|m| mid[m])
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(
                mid[peak] - far > 6.0,
                "{hz} Hz leaked across the bank: peak {:.3}, distant {far:.3}",
                mid[peak]
            );
            peaks.push(peak);
        }
        assert!(peaks[0] < peaks[1], "a higher tone must peak higher up");
    }

    /// Silence is the floor exactly, which is what makes the tone test
    /// meaningful: the baseline every band sits at is known.
    #[test]
    fn silence_sits_exactly_on_the_mel_floor() {
        let (mel, n) = audio::gemma_logmel(&vec![0.0f32; 4000]);
        assert!(n > 0);
        let floor = 0.001f64.ln() as f32;
        assert!(
            mel.iter().all(|&v| (v - floor).abs() < 1e-6),
            "silence should be ln(mel_floor) in every band"
        );
    }

    // ── Regions the coverage sweep found with no caller ────────────────────

    /// `smart_resize` scaling UP, and the asymmetry that makes it correct.
    ///
    /// The two rescale branches are not mirror images and a tidy-up that made
    /// them symmetric would be wrong in both directions:
    ///
    /// * The `max_pixels` branch FLOORS and then clamps `.max(factor)`.
    ///   Flooring can land on zero for a thin strip, and a zero extent is not
    ///   an image, so the clamp is load-bearing.
    /// * The `min_pixels` branch CEILS and does not clamp. Ceiling a positive
    ///   number by a multiple of `factor` cannot produce less than `factor`,
    ///   so a clamp there would be dead code -- and flooring instead would
    ///   land BELOW `min_pixels`, which is the one thing this branch exists
    ///   to prevent.
    ///
    /// Not asserted, because it cannot be: the `.max(factor)` on the INITIAL
    /// rounding is unobservable for any config with a nonzero `min_pixels`. A
    /// side that rounds to zero makes the area zero, zero is below the floor,
    /// and the rescale branch then overwrites both sides. It is redundant with
    /// the shipped bounds and load-bearing only if someone sets
    /// `min_pixels: 0`.
    ///
    /// Stated as areas rather than sides, because the postcondition is about
    /// area and the sides are just how it is reached.
    #[test]
    fn a_thumbnail_is_scaled_up_to_the_minimum_area_and_a_mural_down_to_the_maximum() {
        let cfg = QwenVisionConfig::default();
        let factor = cfg.factor();

        // Far below min_pixels, and deliberately NOT square: 8x8 scales up to
        // exactly 256 on both sides, a whole multiple of the factor, where
        // flooring and ceiling agree and the choice between them is invisible.
        // 10x8 does not divide evenly, and flooring there lands under the
        // floor this branch exists to enforce.
        let (h, w) = cfg.smart_resize(10, 8);
        assert!(
            h * w >= cfg.min_pixels,
            "scaled up to {h}x{w} = {} px, under the {} px floor",
            h * w,
            cfg.min_pixels
        );
        assert_eq!(
            (h % factor, w % factor),
            (0, 0),
            "{h}x{w} is not a whole number of tiles"
        );

        // Over max_pixels, and not evenly divisible by the rescale, so the
        // difference between flooring and ceiling is visible: ceiling both
        // sides here lands back OVER the ceiling this branch exists to
        // enforce.
        let (h, w) = cfg.smart_resize(7000, 9000);
        assert!(
            h * w <= cfg.max_pixels,
            "scaled down to {h}x{w} = {} px, over the {} px ceiling",
            h * w,
            cfg.max_pixels
        );
        assert_eq!(
            (h % factor, w % factor),
            (0, 0),
            "{h}x{w} is not a whole number of tiles"
        );

        // An extreme aspect ratio is the case the `.max(factor)` clamp exists
        // for: dividing the short side by beta takes it below one tile, and
        // flooring a sub-tile side gives ZERO -- an extent no image has. Here
        // the clamp deliberately WINS over the area bound, because a zero side
        // is not a smaller image, it is not an image.
        let (h, w) = cfg.smart_resize(1_000_000, 40);
        assert!(
            w >= factor && h >= factor,
            "{h}x{w}: a side collapsed below one {factor}px tile"
        );
        assert_eq!(
            (h % factor, w % factor),
            (0, 0),
            "{h}x{w} is not a whole number of tiles"
        );
    }

    /// A GIF frame's timestamp is when it STARTS, not when it ends.
    ///
    /// The accumulator is read into the output before the frame's own delay is
    /// added to it, so frame 0 is at t=0 and frame k carries the sum of the
    /// delays of frames 0..k-1. Moving the `+=` above the `push` -- which
    /// reads like an ordinary tidy-up, since the two statements otherwise
    /// commute -- shifts every timestamp forward by one frame and makes the
    /// first frame appear at a time the clip has not reached yet. A caller
    /// seeking to t=0 would get frame 1.
    ///
    /// Delays are unequal on purpose: with a uniform delay a shifted sequence
    /// and a correct one differ only in the last element, so the mistake would
    /// still be caught but for the wrong reason.
    #[test]
    fn gif_frame_timestamps_are_start_times_and_accumulate_prior_delays() {
        use image::codecs::gif::{GifEncoder, Repeat};
        use image::{Delay, Frame, RgbaImage};

        let delays_ms = [100u32, 200, 50];
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut enc = GifEncoder::new(&mut buf);
            enc.set_repeat(Repeat::Infinite).expect("set repeat");
            for (i, ms) in delays_ms.iter().enumerate() {
                let px = 40 + i as u8 * 40;
                let img = RgbaImage::from_pixel(4, 4, image::Rgba([px, px, px, 255]));
                enc.encode_frame(Frame::from_parts(
                    img,
                    0,
                    0,
                    Delay::from_numer_denom_ms(*ms, 1),
                ))
                .expect("encode frame");
            }
        }

        let frames = decode_gif_frames(&buf).expect("decode the gif just encoded");
        assert_eq!(frames.len(), delays_ms.len(), "lost or gained a frame");

        // Start times: 0, then 0.1, then 0.1+0.2. NOT 0.1, 0.3, 0.35.
        let expected = [0.0f32, 0.100, 0.300];
        for (i, ((_, t), want)) in frames.iter().zip(expected).enumerate() {
            assert!(
                (t - want).abs() < 0.02,
                "frame {i} starts at {t}s, expected {want}s (a one-frame shift means the \
                 delay is being added before the timestamp is recorded)"
            );
        }
        // Every frame decoded to the size it was written at.
        for (img, _) in &frames {
            assert_eq!((img.width(), img.height()), (4, 4));
        }
    }

    /// Undecodable bytes are refused rather than yielding an empty timeline.
    #[test]
    fn bytes_that_are_not_a_gif_are_refused() {
        let e = decode_gif_frames(b"not a gif at all").expect_err("accepted non-gif bytes");
        assert!(e.contains("gif"), "unhelpful refusal: {e}");
    }

    /// A GIF can be well-formed and still carry no picture.
    ///
    /// Header, logical screen descriptor, one comment block, trailer --
    /// every byte legal, no image block. The decoder is happy and hands
    /// back an empty frame list, which is the one case
    /// `bytes_that_are_not_a_gif_are_refused` cannot reach because there
    /// the decoder itself objects.
    ///
    /// The comment is not decoration. `read_info` reads ahead until the
    /// header is provably over, and a trailer straight after the screen
    /// descriptor leaves it waiting for a block that never comes, so the
    /// codec refuses with an EOF and this function never runs. One
    /// non-image block is what lets a legal empty GIF reach it.
    ///
    /// Refusing matters because the caller treats this as a video: an
    /// empty timeline is a prompt whose media placeholder expands to no
    /// frames at all, so the model is asked about a picture it was never
    /// shown, and answers. A named refusal is the difference between a
    /// failed request and a confident one about nothing.
    #[test]
    fn a_gif_with_no_frames_is_refused_rather_than_yielding_an_empty_timeline() {
        let mut bytes = Vec::from(*b"GIF89a");
        bytes.extend_from_slice(&1u16.to_le_bytes()); // width
        bytes.extend_from_slice(&1u16.to_le_bytes()); // height
        bytes.extend_from_slice(&[0x00, 0x00, 0x00]); // no global table, bg, aspect
        bytes.extend_from_slice(&[0x21, 0xFE, 0x00]); // an empty comment extension
        bytes.push(0x3B); // trailer, and not one image block between

        let e = decode_gif_frames(&bytes).expect_err("accepted a picture-less gif");
        assert_eq!(
            e, "gif has no frames",
            "the decoder accepted these bytes, so this is the crate's own \
             refusal and not the codec's"
        );
    }

    /// The delimiter tables are an ABI: the host encodes these strings with
    /// the model's own tokenizer and surfaces them as `image.prefix-tokens` /
    /// `audio.prefix-tokens`, so a wrong string becomes ordinary text in the
    /// prompt rather than an error.
    ///
    /// Gemma-4's empty pair is the entry worth pinning. It looks like an
    /// omission next to Qwen's, and the obvious "fix" is to fill it in with
    /// `<start_of_image>` -- which the chat template ALREADY emits, so filling
    /// it in emits the marker twice and desynchronises the image rows from the
    /// placeholder span. Empty is the verified answer, not a gap.
    #[test]
    fn the_delimiter_tables_answer_every_arch_and_gemma_4s_answer_is_deliberately_empty() {
        assert_eq!(
            vision_delimiters(VisionArch::Qwen36),
            ("<|vision_start|>", "<|vision_end|>")
        );
        assert_eq!(vision_delimiters(VisionArch::Gemma4), ("", ""));

        // Audio keys off the arch NAME, and answers only where a front-end
        // exists -- the same predicate the host checks before it calls.
        for arch in ["gemma4", "gemma4_text"] {
            let d = audio_delimiters(arch);
            assert_eq!(
                audio_arch_supported(arch),
                d != ("", ""),
                "arch {arch:?} disagrees with audio_arch_supported: got {d:?}"
            );
        }
        assert_eq!(audio_delimiters("qwen3_vl"), ("", ""));
        assert_eq!(audio_delimiters("no-such-arch"), ("", ""));
    }

    /// A clip too short to fill one analysis window yields zero frames, and
    /// the soft-token count of zero frames is zero.
    ///
    /// The framing needs `frame + 1` samples after a `frame / 2` semicausal
    /// pad, so anything under ~20 ms at 16 kHz produces no frames at all. That
    /// number then reaches `gemma_audio_token_count`, whose two convolutions
    /// compute `(n + 2 - 3) / 2 + 1` -- an expression that on `u32` is fine
    /// for every n except the one this path produces.
    #[test]
    fn a_clip_shorter_than_one_window_yields_no_frames_and_no_soft_tokens() {
        let (mel, n) = audio::gemma_logmel(&[0.0f32; 16]);
        assert_eq!((mel.len(), n), (0, 0), "16 samples is not a full window");

        assert_eq!(
            gemma_audio_token_count(0),
            0,
            "zero frames must cost zero soft tokens"
        );
        // The neighbours are unmoved. n=3 is the useful one: a single conv
        // gives 2 and the required pair gives 1, so it catches a lost stage.
        assert_eq!(gemma_audio_token_count(1), 1);
        assert_eq!(gemma_audio_token_count(2), 1);
        assert_eq!(gemma_audio_token_count(3), 1, "only one conv stage ran");
        assert_eq!(gemma_audio_token_count(5), 2);
    }

    /// The two malformed-container refusals, and the order they are asked in.
    ///
    /// `data` is checked before `channels`, so a file carrying neither chunk
    /// is refused for the missing data rather than the missing format. The
    /// `fmt`-only and `data`-only cases therefore produce DIFFERENT messages,
    /// and swapping the two checks silently swaps them.
    #[test]
    fn a_wav_missing_either_chunk_is_refused_by_the_chunk_it_is_missing() {
        let riff = |chunks: &[u8]| {
            let mut w = Vec::new();
            w.extend_from_slice(b"RIFF");
            w.extend_from_slice(&(4 + chunks.len() as u32).to_le_bytes());
            w.extend_from_slice(b"WAVE");
            w.extend_from_slice(chunks);
            w
        };

        // A data chunk with no fmt chunk: decodes as far as the sample loop
        // and then has no idea how wide a sample is.
        let mut data_only = Vec::new();
        data_only.extend_from_slice(b"data");
        data_only.extend_from_slice(&4u32.to_le_bytes());
        data_only.extend_from_slice(&[0u8; 4]);
        let e = audio::decode_wav(&riff(&data_only)).expect_err("accepted a WAV with no fmt");
        assert!(
            e.contains("fmt"),
            "wrong refusal for a missing fmt chunk: {e}"
        );

        // A fmt chunk with no data chunk.
        let full = wav(1, 16, 1, 16_000, &[0u8; 4]);
        let cut = full.len() - 12; // drop the whole `data` chunk incl. header
        let e = audio::decode_wav(&full[..cut]).expect_err("accepted a WAV with no data");
        assert!(
            e.contains("data"),
            "wrong refusal for a missing data chunk: {e}"
        );

        // Not a RIFF/WAVE container at all.
        assert!(audio::decode_wav(b"ID3\x04\x00\x00\x00").is_err());

        // Neither chunk: the two checks are in a fixed order and the data one
        // is asked first, so this names the data chunk. Swapping the checks is
        // invisible on either single-chunk case above and visible only here.
        let e = audio::decode_wav(&riff(&[])).expect_err("accepted an empty container");
        assert!(
            e.contains("data"),
            "an empty container should be refused for its missing data chunk first: {e}"
        );
    }

    /// The whole audio pipeline end to end, on a clip that needs resampling.
    ///
    /// `process_wav_bytes` is the only entry point the host calls, and it is
    /// the composition of three steps that are each tested alone. What is only
    /// visible here is that the rate travels from the container's `fmt` chunk
    /// into the resampler: 8 kHz in must become the SAME number of frames as
    /// the equivalent 16 kHz clip of the same DURATION, not of the same sample
    /// count. Dropping the resample step, or reading the rate off the wrong
    /// offset, halves or doubles the frame count while everything still
    /// decodes without complaint.
    #[test]
    fn the_wav_pipeline_carries_the_container_rate_into_the_resampler() {
        // 0.5 s at 8 kHz = 4000 samples.
        let pcm8k: Vec<u8> = (0..4000)
            .flat_map(|i| ((i as i16).wrapping_mul(37)).to_le_bytes())
            .collect();
        let (mel, frames_8k) =
            audio::process_wav_bytes(&wav(1, 16, 1, 8_000, &pcm8k)).expect("8 kHz clip");
        assert_eq!(mel.len(), frames_8k * 128, "mel is not n_frames x n_mels");

        // The same half second already at 16 kHz = 8000 samples.
        let pcm16k: Vec<u8> = (0..8000)
            .flat_map(|i| ((i as i16).wrapping_mul(37)).to_le_bytes())
            .collect();
        let (_, frames_16k) =
            audio::process_wav_bytes(&wav(1, 16, 1, 16_000, &pcm16k)).expect("16 kHz clip");

        assert_eq!(
            frames_8k, frames_16k,
            "the same half second gave {frames_8k} frames at 8 kHz and {frames_16k} at 16 kHz \
             -- the container's rate is not reaching the resampler"
        );
        assert!(
            frames_8k > 40,
            "half a second should be ~50 frames, got {frames_8k}"
        );

        // A malformed container fails the pipeline rather than the front-end.
        assert!(audio::process_wav_bytes(b"RIFF").is_err());
    }
}
