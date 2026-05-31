//! Multimodal preprocessing geometry for vision/video inputs.
//!
//! This module owns the *geometry* half of the image processor: how many
//! hidden-state rows (LLM tokens) a still image or video clip expands to, the
//! `(t, h, w)` patch grid, and — for M-RoPE models — how far the 1-D sequence
//! cursor advances past the span. It deliberately does **not** touch pixels
//! (resize/normalize/patchify); that heavier per-pixel work lands in a later
//! slice. The geometry is what gates the `image` resource's synchronous
//! `token-count()` / `position-span()` / `grid()` queries (see `MULTIMODAL.md`),
//! so it must match the HF processors exactly and is unit-tested in isolation.
//!
//! Two arch families are modelled:
//!   * **Gemma 4** — fixed-resolution SigLIP, `tokens_per_image` soft tokens per
//!     crop, optional pan-and-scan, standard 1-D RoPE.
//!   * **Qwen 3.6** — native dynamic resolution via `smart_resize`, 2×2 patch
//!     merge, M-RoPE positions.
//!
//! Constants marked `VERIFY` should be confirmed against the shipped HF config
//! for the exact checkpoint before parity is claimed.
#![allow(dead_code)] // Scaffolding: wired into the build ahead of the encoder + WIT surface.

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

/// Audio soft tokens for `n_frames` log-mel frames: two stride-2 Conv2d
/// (k3, s2, p1) along the time axis. `floor((n + 2 - 3) / 2) + 1` applied
/// twice. Mirrors the driver's `gemma4_audio_subsampled_len` exactly.
pub fn gemma_audio_token_count(n_frames: u32) -> u32 {
    let conv = |n: u32| (n + 2 - 3) / 2 + 1;
    conv(conv(n_frames))
}

// ============================================================================
// Image header parsing (dependency-free)
// ============================================================================

/// Pixel `(width, height)` of an encoded PNG or JPEG, read from its header
/// without a full decode (so the host can answer `image.token-count()`
/// synchronously without pulling in an image-codec dependency). Returns `None`
/// for unrecognized/truncated data. Full pixel decode for the encoder is a
/// later slice (Phase 2).
pub fn image_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    png_dimensions(bytes).or_else(|| jpeg_dimensions(bytes))
}

fn png_dimensions(b: &[u8]) -> Option<(u32, u32)> {
    const SIG: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    if b.len() < 24 || b[..8] != SIG || &b[12..16] != b"IHDR" {
        return None;
    }
    // IHDR is required to be the first chunk: width @16, height @20 (big-endian).
    let w = u32::from_be_bytes([b[16], b[17], b[18], b[19]]);
    let h = u32::from_be_bytes([b[20], b[21], b[22], b[23]]);
    Some((w, h))
}

fn jpeg_dimensions(b: &[u8]) -> Option<(u32, u32)> {
    if b.len() < 4 || b[0] != 0xFF || b[1] != 0xD8 {
        return None;
    }
    let mut i = 2usize;
    while i + 1 < b.len() {
        if b[i] != 0xFF {
            return None;
        }
        let marker = b[i + 1];
        // Standalone markers carry no length: RSTn/SOI/EOI (D0–D9), TEM (01).
        if matches!(marker, 0xD0..=0xD9 | 0x01) {
            i += 2;
            continue;
        }
        if i + 3 >= b.len() {
            return None;
        }
        let seglen = u16::from_be_bytes([b[i + 2], b[i + 3]]) as usize;
        if seglen < 2 {
            return None;
        }
        // Start-of-frame markers carry frame geometry (excludes DHT/DAC/JPG).
        if matches!(marker, 0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF) {
            if i + 9 > b.len() {
                return None;
            }
            let h = u16::from_be_bytes([b[i + 5], b[i + 6]]) as u32;
            let w = u16::from_be_bytes([b[i + 7], b[i + 8]]) as u32;
            return Some((w, h));
        }
        i += 2 + seglen;
    }
    None
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

    // ── Arch selection + header parsing ───────────────────────────────────

    #[test]
    fn arch_name_selection() {
        assert_eq!(VisionArch::from_arch_name("gemma4"), Some(VisionArch::Gemma4));
        assert_eq!(VisionArch::from_arch_name("Gemma4-27B"), Some(VisionArch::Gemma4));
        assert_eq!(VisionArch::from_arch_name("qwen3_6"), Some(VisionArch::Qwen36));
        assert_eq!(VisionArch::from_arch_name("qwen3_5_moe"), Some(VisionArch::Qwen36));
        assert_eq!(VisionArch::from_arch_name("llama"), None);
    }

    #[test]
    fn png_header_dimensions() {
        let mut png = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
        png.extend_from_slice(&[0, 0, 0, 13]); // IHDR length
        png.extend_from_slice(b"IHDR");
        png.extend_from_slice(&640u32.to_be_bytes());
        png.extend_from_slice(&480u32.to_be_bytes());
        png.extend_from_slice(&[8, 2, 0, 0, 0]); // bit depth, color type, …
        assert_eq!(image_dimensions(&png), Some((640, 480)));
    }

    #[test]
    fn jpeg_header_dimensions() {
        let mut jpg = vec![0xFF, 0xD8]; // SOI
        jpg.extend_from_slice(&[0xFF, 0xE0, 0x00, 0x04, 0x00, 0x00]); // APP0, len 4
        jpg.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x11, 0x08]); // SOF0, len 17, precision 8
        jpg.extend_from_slice(&300u16.to_be_bytes()); // height
        jpg.extend_from_slice(&400u16.to_be_bytes()); // width
        jpg.extend_from_slice(&[0u8; 10]); // component spec padding
        assert_eq!(image_dimensions(&jpg), Some((400, 300)));
    }

    #[test]
    fn unrecognized_bytes_yield_none() {
        assert_eq!(image_dimensions(b"not an image"), None);
        assert_eq!(image_dimensions(&[]), None);
    }
}
