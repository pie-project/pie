use crate::front::{
    Budget, Delimiters, EncodedSpan, Fault, Grid, Resample, Result, Rgb8, VisionFrontEnd,
};

pub const ARCH: &str = "qwen3_5";
/// Qwen 3.8 Flash Next's tower reads what Qwen 3.5's reads.
pub const ARCH_NEXT: &str = "qwen4_exp";

pub const VISION_START: &str = "<|vision_start|>";
pub const IMAGE_PAD: &str = "<|image_pad|>";
pub const VISION_END: &str = "<|vision_end|>";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QwenVisionConfig {
    pub patch_size: u32,
    pub merge_size: u32,
    pub temporal_patch_size: u32,
    pub min_pixels: u32,
    pub max_pixels: u32,
    pub num_grid_per_side: u32,
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
    #[must_use]
    pub const fn factor(&self) -> u32 {
        self.patch_size * self.merge_size
    }

    #[must_use]
    pub const fn patch_width(&self) -> usize {
        3 * self.temporal_patch_size as usize * self.patch_size as usize * self.patch_size as usize
    }

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

    pub fn patch_grid(&self, h: u32, w: u32) -> Result<(u32, u32)> {
        let (h_bar, w_bar) = self.smart_resize(h, w)?;
        Ok((h_bar / self.patch_size, w_bar / self.patch_size))
    }

    #[must_use]
    pub const fn token_count(&self, gh: u32, gw: u32) -> u32 {
        gh * gw / (self.merge_size * self.merge_size)
    }

    #[must_use]
    pub const fn position_span(&self, gh: u32, gw: u32) -> u32 {
        let (hm, wm) = (gh / self.merge_size, gw / self.merge_size);
        if hm > wm { hm } else { wm }
    }

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
        let norm =
            |v: u8, ch: usize| -> f32 { ((f32::from(v) / 255.0) - self.mean[ch]) / self.std[ch] };

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
                        let (tap, weight) = interp(ih_blk * m + ih, iw_blk * m + iw, gh, gw, side);
                        ids.extend_from_slice(&tap);
                        weights.extend_from_slice(&weight);
                    }
                }
            }
        }
        (ids, weights)
    }
}

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
        weights[t] = (1.0 - (src - floor - offset).abs()).max(0.0);
    }
    (taps, weights)
}

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

#[derive(Clone, Copy, Debug, Default)]
pub struct Qwen35Vision {
    pub config: QwenVisionConfig,
}

impl Qwen35Vision {
    #[must_use]
    pub fn new() -> Qwen35Vision {
        Qwen35Vision::default()
    }
}

impl VisionFrontEnd for Qwen35Vision {
    fn arch(&self) -> &'static str {
        ARCH
    }

    fn delimiters(&self) -> Delimiters {
        Delimiters {
            prefix: VISION_START,
            placeholder: IMAGE_PAD,
            suffix: VISION_END,
        }
    }

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
