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
//!     `smart_resize` + block-merge `(3,2,16,16)` layout with `(x/255-0.5)/0.5`.
//!   * **audio** — WAV decode, resample to 16 kHz, and the log-mel front-end
//!     (the [`audio`] submodule), matching `Gemma4AudioFeatureExtractor`.
//!
//! Two arch families are modelled: **Gemma 4** (fixed-resolution SigLIP, 1-D
//! RoPE) and **Qwen 3.6** (native dynamic resolution, 2×2 merge, M-RoPE). All
//! geometry is unit-tested against the HF processors; the patchify / log-mel
//! layouts are parity-verified against the reference dumps.
#![allow(dead_code)] // Some geometry/arch-completeness helpers are exercised only by tests.

use image::{imageops::FilterType, DynamicImage};

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
    pub fn patchify_chw(&self, resized: &[f32], c: usize, h: usize, w: usize)
        -> (Vec<f32>, Vec<[u32; 2]>)
    {
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
                    grid: Grid { t: frames, h: 1, w: per.token_count },
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
    use image::codecs::gif::GifDecoder;
    use image::AnimationDecoder;
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
        let (num, den) = f.delay().numer_denom_ms();
        let frame_ms = if den != 0 { num as f32 / den as f32 } else { num as f32 };
        out.push((DynamicImage::ImageRgba8(f.into_buffer()), t_ms / 1000.0));
        t_ms += frame_ms;
    }
    Ok(out)
}

impl VisionArch {
    /// Map a registered model's `arch_name` to its vision front-end, or `None`
    /// for text-only archs. Covers the multimodal checkpoints Pie targets first.
    pub fn from_arch_name(arch: &str) -> Option<VisionArch> {
        let a = arch.to_ascii_lowercase();
        if a.contains("gemma4") || a.contains("gemma-4") {
            Some(VisionArch::Gemma4)
        } else if a.contains("qwen3") {
            // qwen3_5 / qwen3_6 — the Qwen3-VL line.
            Some(VisionArch::Qwen36)
        } else {
            None
        }
    }
}

// ============================================================================
// Audio (gemma4_audio) — front-end geometry
// ============================================================================

/// Whether the given `arch_name` has a gemma4 audio front-end. Only Gemma-4
/// ships the USM/Conformer audio tower today.
pub fn audio_arch_supported(arch: &str) -> bool {
    let a = arch.to_ascii_lowercase();
    a.contains("gemma4") || a.contains("gemma-4")
}

/// Delimiter *strings* the model wraps a visual span with — encoded host-side by
/// the model's own tokenizer and applied by the SDK's `append-image`, so the
/// inferlet never names them. `("", "")` means the model needs none. (Qwen3-VL
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
/// twice. Mirrors the driver's `gemma4_audio_subsampled_len` exactly.
pub fn gemma_audio_token_count(n_frames: u32) -> u32 {
    let conv = |n: u32| (n + 2 - 3) / 2 + 1;
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
        x.extend(std::iter::repeat(0.0f64).take(pad));
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
                    let v =
                        f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
                    samples.push(v as f32);
                }
            }
            _ => return Err(format!("WAV: unsupported format tag {fmt_tag} / {bits}-bit")),
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
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // Minimal f32/.npy reader (little-endian, C-order) for parity dumps.
    fn read_npy_f32(path: &str) -> Option<(Vec<usize>, Vec<f32>)> {
        let b = std::fs::read(path).ok()?;
        if &b[..6] != b"\x93NUMPY" { return None; }
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

    #[test]
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
        assert!(max_abs < 1e-6, "patchify pixel_values differ: max_abs={max_abs}");
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
            assert!((gh / cfg.pooling_kernel_size) * (gw / cfg.pooling_kernel_size) <= cfg.max_soft_tokens);
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
        assert_eq!(span.position_span, span.grid.t.max(span.grid.h).max(span.grid.w));
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
        assert_eq!(qwen_next_position(span.grid, anchor), anchor + span.position_span);
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
        assert!(span.grid.t > 1, "8 frames / temporal_patch_size should give t>1");
    }

    // ── Arch selection ────────────────────────────────────────────────────

    #[test]
    fn arch_name_selection() {
        assert_eq!(VisionArch::from_arch_name("gemma4"), Some(VisionArch::Gemma4));
        assert_eq!(VisionArch::from_arch_name("Gemma4-27B"), Some(VisionArch::Gemma4));
        assert_eq!(VisionArch::from_arch_name("qwen3_6"), Some(VisionArch::Qwen36));
        assert_eq!(VisionArch::from_arch_name("qwen3_5_moe"), Some(VisionArch::Qwen36));
        assert_eq!(VisionArch::from_arch_name("llama"), None);
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
}
