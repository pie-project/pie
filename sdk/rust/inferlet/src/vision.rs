//! Inferlet-side image preprocessing for vision models (option B: the inferlet
//! decodes + resizes + patchifies into the `pixel_values` blob the engine's
//! vision encoder consumes). The deterministic geometry here mirrors the
//! host-side `runtime::multimodal` (verified bit-exact vs the HF
//! `Gemma4ImageProcessor`); the inferlet supplies the decode + resize (e.g. via
//! the `image` crate) and calls [`gemma_patchify_hwc`].
//!
//! Flow in an inferlet:
//! ```ignore
//! let img = image::load_from_memory(&bytes)?;                 // decode (image crate)
//! let (th, tw) = vision::gemma_resize_target(img.width(), img.height());
//! let resized = img.resize_exact(tw, th, FilterType::CatmullRom).to_rgb8();
//! let (pixels, positions) = vision::gemma_patchify_hwc(&resized, th, tw);
//! let image = Image::from_pixels(&model, &pixels, &positions)?;
//! ctx.append_image(&image).await?;
//! ```

/// Gemma-4 processor params (defaults match `google/gemma-4-E4B`).
#[derive(Clone, Copy, Debug)]
pub struct GemmaProc {
    pub patch_size: u32,
    pub pooling_kernel_size: u32,
    pub max_soft_tokens: u32,
}
impl Default for GemmaProc {
    fn default() -> Self {
        Self { patch_size: 16, pooling_kernel_size: 3, max_soft_tokens: 280 }
    }
}

/// SigLIP2-style aspect-ratio-preserving resize target `(height, width)` for a
/// `w × h` image — sides multiple of `patch·pool_k`, pooled grid ≤
/// `max_soft_tokens`. Mirrors `runtime::multimodal::resize_target`.
fn resize_target_for(max_soft_tokens: u32, w: u32, h: u32) -> (u32, u32) {
    let c = GemmaProc::default();
    let unit = (c.patch_size * c.pooling_kernel_size) as f64;
    let max_units = max_soft_tokens as f64;
    let scaled = |scale: f64, size: u32| -> u32 {
        let s = (size as f64 * scale / unit).ceil() * unit;
        s.max(unit) as u32
    };
    let eps = 1e-5;
    let (mut lo, mut hi) = (eps / 10.0, 100.0);
    while (hi - lo) >= eps {
        let s = (lo + hi) / 2.0;
        let (th, tw) = (scaled(s, h), scaled(s, w));
        if (th as f64 / unit) * (tw as f64 / unit) <= max_units {
            lo = s;
        } else {
            hi = s;
        }
    }
    (scaled(lo, h), scaled(lo, w))
}

/// Resize target for a still image (≤ 280 soft tokens, `Gemma4ImageProcessor`).
pub fn gemma_resize_target(w: u32, h: u32) -> (u32, u32) {
    resize_target_for(GemmaProc::default().max_soft_tokens, w, h)
}

/// Resize target for a *video frame* (≤ 70 soft tokens, `Gemma4VideoProcessor`).
///
/// Gemma 4 treats video as a sequence of independently-patchified frames
/// through the same vision tower (no temporal patching) — the only processor
/// difference vs still images is the smaller per-frame soft-token budget, which
/// keeps a multi-frame clip's KV footprint manageable. Patch geometry is
/// identical, so [`gemma_patchify_hwc`] is reused unchanged.
pub fn gemma_resize_target_video(w: u32, h: u32) -> (u32, u32) {
    resize_target_for(70, w, h)
}

/// Patchify a resized RGB image (`rgb` HWC, `[h, w, 3]` u8, already resized to a
/// patch-multiple via [`gemma_resize_target`]) into the encoder's
/// `pixel_values` + 2D positions. Rescales /255 and lays out each patch as
/// `(patch_row, patch_col, channel)` (channels-last), patches row-major,
/// position `(x=col, y=row)`. Returns `(pixel_values [n_patch·768],
/// positions [n_patch·2])`. Matches HF `convert_image_to_patches` exactly.
pub fn gemma_patchify_hwc(rgb: &[u8], h: u32, w: u32) -> (Vec<f32>, Vec<u32>) {
    let p = GemmaProc::default().patch_size as usize;
    let (h, w) = (h as usize, w as usize);
    let (ph, pw) = (h / p, w / p);
    let n = ph * pw;
    let pd = 3 * p * p; // 768
    let mut pix = vec![0.0f32; n * pd];
    let mut pos = vec![0u32; n * 2];
    for pr in 0..ph {
        for pc in 0..pw {
            let idx = pr * pw + pc;
            pos[2 * idx] = pc as u32; // x = col
            pos[2 * idx + 1] = pr as u32; // y = row
            let base = idx * pd;
            for r in 0..p {
                for col in 0..p {
                    let src = ((pr * p + r) * w + (pc * p + col)) * 3;
                    for ch in 0..3 {
                        pix[base + (r * p + col) * 3 + ch] = rgb[src + ch] as f32 / 255.0;
                    }
                }
            }
        }
    }
    (pix, pos)
}

// ============================================================================
// Qwen3-VL (native dynamic resolution, 2×2 merge, M-RoPE)
// ============================================================================

/// Qwen3-VL processor params (defaults match `Qwen/Qwen3-VL-2B-Instruct`).
#[derive(Clone, Copy, Debug)]
pub struct QwenProc {
    pub patch_size: u32,
    pub merge_size: u32,
    pub temporal_patch_size: u32,
    pub min_pixels: u32,
    pub max_pixels: u32,
}
impl Default for QwenProc {
    fn default() -> Self {
        Self {
            patch_size: 16,
            merge_size: 2,
            temporal_patch_size: 2,
            min_pixels: 65536,    // 256²
            max_pixels: 16_777_216, // 4096²
        }
    }
}

/// HF `smart_resize`: resize `(w, h)` so each side is a multiple of
/// `patch·merge` and the area lands in `[min_pixels, max_pixels]`, preserving
/// aspect ratio. Returns `(target_height, target_width)`. Mirrors
/// `runtime::multimodal::QwenVisionConfig::smart_resize`.
pub fn qwen_resize_target(w: u32, h: u32) -> (u32, u32) {
    let c = QwenProc::default();
    let factor = (c.patch_size * c.merge_size) as f64;
    let (hf, wf) = (h as f64, w as f64);
    let round_f = |x: f64| (x / factor).round() * factor;
    let floor_f = |x: f64| (x / factor).floor() * factor;
    let ceil_f = |x: f64| (x / factor).ceil() * factor;
    let mut h_bar = round_f(hf).max(factor);
    let mut w_bar = round_f(wf).max(factor);
    if h_bar * w_bar > c.max_pixels as f64 {
        let beta = (hf * wf / c.max_pixels as f64).sqrt();
        h_bar = floor_f(hf / beta).max(factor);
        w_bar = floor_f(wf / beta).max(factor);
    } else if h_bar * w_bar < c.min_pixels as f64 {
        let beta = (c.min_pixels as f64 / (hf * wf)).sqrt();
        h_bar = ceil_f(hf * beta);
        w_bar = ceil_f(wf * beta);
    }
    (h_bar as u32, w_bar as u32)
}

/// Patchify a resized RGB still image (`rgb` HWC `[h, w, 3]` u8, sides already a
/// multiple of `patch·merge` via [`qwen_resize_target`]) into Qwen3-VL's
/// `pixel_values` + per-patch positions.
///
/// Mirrors HF `Qwen2/3VLImageProcessor._preprocess` exactly:
///   * normalize `(x/255 − 0.5) / 0.5` (image_mean = image_std = 0.5);
///   * patch order is the spatial-merge order `(bh, bw, ih, iw)` (block-major):
///     every `merge²` consecutive patches form one merged token;
///   * each patch's `patch_dim = 3·temporal·patch²` feature vector is laid out
///     `[channel][temporal][ph][pw]`, with the still frame duplicated across
///     the `temporal_patch_size` temporal slots.
///
/// Returns `(pixel_values [n_patch·patch_dim], positions [n_patch·2])` where
/// `positions[2k] = patch col (x)`, `positions[2k+1] = patch row (y)` in the
/// same merge order. `n_patch = (h/patch)·(w/patch)`.
pub fn qwen_patchify_hwc(rgb: &[u8], h: u32, w: u32) -> (Vec<f32>, Vec<u32>) {
    let c = QwenProc::default();
    let p = c.patch_size as usize;
    let m = c.merge_size as usize;
    let tp = c.temporal_patch_size as usize;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_patchify_order_and_norm() {
        // 32×64 (h,w): patch grid 2×4, merged blocks 1×2.
        let (h, w) = (32u32, 64u32);
        let rgb: Vec<u8> = (0..(h * w * 3)).map(|i| (i % 256) as u8).collect();
        let (pix, pos) = qwen_patchify_hwc(&rgb, h, w);
        let pd = 3 * 2 * 16 * 16;
        assert_eq!(pos.len(), 2 * 8); // 8 patches
        assert_eq!(pix.len(), 8 * pd);
        // merge order: first block (bh=0,bw=0) covers patches (0,0),(1,0),(0,1),(1,1)
        // i.e. positions (x,y): (0,0),(1,0),(0,1),(1,1).
        assert_eq!(&pos[0..2], &[0, 0]);
        assert_eq!(&pos[2..4], &[1, 0]);
        assert_eq!(&pos[4..6], &[0, 1]);
        assert_eq!(&pos[6..8], &[1, 1]);
        // second block starts at patch col 2.
        assert_eq!(&pos[8..10], &[2, 0]);
        // normalized into [-1, 1].
        assert!(pix.iter().all(|&v| (-1.0..=1.0).contains(&v)));
        // temporal duplication: t=0 and t=1 slices of patch 0 are identical.
        let half = pd / 2; // not the temporal split; check explicit offsets
        let _ = half;
        for ch in 0..3 {
            for r in 0..16 {
                for col in 0..16 {
                    let o0 = ((ch * 2 + 0) * 16 + r) * 16 + col;
                    let o1 = ((ch * 2 + 1) * 16 + r) * 16 + col;
                    assert_eq!(pix[o0], pix[o1]);
                }
            }
        }
    }

    #[test]
    fn resize_target_pool_divisible() {
        // 480x640 (h,w) → 672x912, grid 42x57 (both /3). Matches HF.
        assert_eq!(gemma_resize_target(640, 480), (672, 912));
        let unit = 16 * 3;
        let (th, tw) = gemma_resize_target(1280, 720);
        assert_eq!(th % unit, 0);
        assert_eq!(tw % unit, 0);
    }

    #[test]
    fn video_resize_target_caps_at_70_soft_tokens() {
        let unit = 16 * 3; // patch · pool_k
        // A frame's pooled grid must be ≤ 70 soft tokens, sides /48.
        for (w, h) in [(640u32, 480u32), (1280, 720), (1920, 1080), (320, 240)] {
            let (th, tw) = gemma_resize_target_video(w, h);
            assert_eq!(th % unit, 0, "{w}x{h} h not /48");
            assert_eq!(tw % unit, 0, "{w}x{h} w not /48");
            let soft = (th / unit) * (tw / unit);
            assert!(soft <= 70, "{w}x{h} -> {soft} soft tokens > 70");
            assert!(soft > 0);
        }
        // A video frame is smaller than the still-image target for the same input.
        let (vh, vw) = gemma_resize_target_video(640, 480);
        let (ih, iw) = gemma_resize_target(640, 480);
        assert!((vh / unit) * (vw / unit) < (ih / unit) * (iw / unit));
    }

    #[test]
    fn patchify_layout_and_positions() {
        // 32x48 image (h=32,w=48): 2x3 patches of 16.
        let (h, w) = (32u32, 48u32);
        let rgb: Vec<u8> = (0..(h * w * 3)).map(|i| (i % 256) as u8).collect();
        let (pix, pos) = gemma_patchify_hwc(&rgb, h, w);
        assert_eq!(pos.len(), 2 * 6); // 6 patches
        assert_eq!(pix.len(), 6 * 768);
        // patch order row-major; positions (x=col, y=row).
        assert_eq!(&pos[0..2], &[0, 0]); // patch 0 → (0,0)
        assert_eq!(&pos[2..4], &[1, 0]); // patch 1 → (1,0)
        assert_eq!(&pos[6..8], &[0, 1]); // patch 3 → (0,1)  (row 1)
        // pixel[0] = rgb[0]/255 = 0; values in [0,1].
        assert!(pix.iter().all(|&v| (0.0..=1.0).contains(&v)));
        assert_eq!(pix[0], 0.0);
    }
}
