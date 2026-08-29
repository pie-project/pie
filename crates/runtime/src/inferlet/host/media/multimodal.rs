//! Host-side image, video and audio preprocessing — the arithmetic that turns
//! encoded bytes into the patches, positions and log-mel frames a model was
//! trained on.
//!
//! IT CAME FROM `model::serve::multimodal`, and it is here because M18 ruled
//! that module out of `model`: that crate says what a model family *is*,
//! backend-blind, and a `CatmullRom` resize behind twenty image crates is not
//! a family fact. It was already the one part of `serve` behind a feature
//! flag, for the same reason stated the other way round — an engine links
//! `model` for its catalog and must not link a JPEG decoder to get one.
//!
//! So it lands beside its one consumer. `media.rs` is the only caller in the
//! tree, the `image` dependency is this crate's now, and the flag is gone:
//! a runtime that serves inferlets decodes media, and there is no build of it
//! that does not.
//!
//! Dispatch is off the served model's `arch_name` — [`VisionArch::from_arch_name`]
//! and [`audio_arch_supported`] — which is a serving row's column, so the
//! processor and the chat template are chosen from the same fact.

use image::{DynamicImage, imageops::FilterType};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    pub t: u32,
    pub h: u32,
    pub w: u32,
}

impl Grid {
    pub fn llm_token_count(&self, merge: u32) -> u32 {
        let m = merge * merge;
        debug_assert!(m != 0);
        self.t * self.h * self.w / m
    }

    pub fn mrope_position_span(&self, merge: u32) -> u32 {
        let hm = self.h / merge;
        let wm = self.w / merge;
        self.t * hm.max(wm)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisualSpan {
    pub token_count: u32,

    pub position_span: u32,

    pub grid: Grid,
}

#[derive(Clone, Copy, Debug)]
pub struct GemmaImageConfig {
    pub patch_size: u32,

    pub pooling_kernel_size: u32,

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
    fn resize_unit(&self) -> u32 {
        self.patch_size * self.pooling_kernel_size
    }

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

    pub fn patch_grid(&self, w: u32, h: u32) -> (u32, u32) {
        let (th, tw) = self.resize_target(w, h);
        (th / self.patch_size, tw / self.patch_size)
    }

    pub fn token_count(&self, w: u32, h: u32) -> u32 {
        let (gh, gw) = self.patch_grid(w, h);
        gh * gw / (self.pooling_kernel_size * self.pooling_kernel_size)
    }

    pub fn layout(&self, w: u32, h: u32) -> VisualSpan {
        let n = self.token_count(w, h);
        VisualSpan {
            token_count: n,
            position_span: n,
            grid: Grid { t: 1, h: 1, w: n },
        }
    }

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
        let pd = c * p * p;
        let mut pix = vec![0.0f32; n * pd];
        let mut pos = vec![[0u32; 2]; n];
        for pr in 0..ph {
            for pc in 0..pw {
                let idx = pr * pw + pc;
                pos[idx] = [pc as u32, pr as u32];
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

#[derive(Clone, Copy, Debug)]
pub struct QwenVisionConfig {
    pub patch_size: u32,
    pub merge_size: u32,
    pub temporal_patch_size: u32,

    pub min_pixels: u32,
    pub max_pixels: u32,
}

impl Default for QwenVisionConfig {
    fn default() -> Self {
        Self {
            patch_size: 16,
            merge_size: 2,
            temporal_patch_size: 2,
            min_pixels: 65536,
            max_pixels: 16777216,
        }
    }
}

impl QwenVisionConfig {
    fn factor(&self) -> u32 {
        self.patch_size * self.merge_size
    }

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

    pub fn grid(&self, h: u32, w: u32, num_frames: u32) -> Grid {
        let (h_bar, w_bar) = self.smart_resize(h, w);
        let frames = num_frames.max(1);

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

    pub fn qwen_patchify_hwc(&self, rgb: &[u8], h: u32, w: u32) -> (Vec<f32>, Vec<u32>) {
        let p = self.patch_size as usize;
        let m = self.merge_size as usize;
        let tp = self.temporal_patch_size as usize;
        let (h, w) = (h as usize, w as usize);
        let (gh, gw) = (h / p, w / p);
        let (bh, bw) = (gh / m, gw / m);
        let n = gh * gw;
        let pd = 3 * tp * p * p;
        let mut pix = vec![0.0f32; n * pd];
        let mut pos = vec![0u32; n * 2];
        let norm = |v: u8| -> f32 { ((v as f32 / 255.0) - 0.5) / 0.5 };
        let mut out_idx = 0usize;
        for ih_blk in 0..bh {
            for iw_blk in 0..bw {
                for ih in 0..m {
                    for iw in 0..m {
                        let pr = ih_blk * m + ih;
                        let pc = iw_blk * m + iw;
                        pos[2 * out_idx] = pc as u32;
                        pos[2 * out_idx + 1] = pr as u32;
                        let base = out_idx * pd;

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

/// Qwen's 3-D rope positions for one visual span: `[t, h, w]` per merged
/// patch, the temporal axis advancing by the wider of the two spatial ones.
///
/// THE MROPE LEG'S ONE PUBLIC ENTRY, and it has no caller in this tree.
/// `Processor::mrope_positions` was the wrapper that reached it and had none
/// either; what carries mrope today is the `uses_mrope` BOOLEAN the runtime
/// stamps on an `Image` for a host->engine path that has not landed
/// (`runtime/src/inferlet/host/media.rs:13` says so). This is the arithmetic
/// that path will ask for, so it stays as the record of it rather than as a
/// reader count.
pub fn qwen_mrope_positions(merged: Grid, anchor: u32) -> Vec<[u32; 3]> {
    let mut out = Vec::with_capacity((merged.t * merged.h * merged.w) as usize);
    let advance = merged.h.max(merged.w);
    for ti in 0..merged.t {
        let base = anchor + ti * advance;
        for hi in 0..merged.h {
            for wi in 0..merged.w {
                out.push([base, base + hi, base + wi]);
            }
        }
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisionArch {
    Gemma4,
    Qwen36,
}

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

    pub fn for_arch_video(arch: VisionArch) -> Self {
        match arch {
            VisionArch::Gemma4 => Processor::Gemma(GemmaImageConfig {
                max_soft_tokens: 70,
                ..GemmaImageConfig::default()
            }),
            VisionArch::Qwen36 => Processor::Qwen(QwenVisionConfig::default()),
        }
    }

    pub fn uses_mrope(&self) -> bool {
        matches!(self, Processor::Qwen(_))
    }
}

pub struct ProcessedImage {
    pub pixels: Vec<f32>,

    pub positions: Vec<u32>,

    pub patch_grid: Grid,

    pub span: VisualSpan,
}

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
    pub fn process_image_bytes(&self, bytes: &[u8]) -> Result<ProcessedImage, String> {
        let img = image::load_from_memory(bytes).map_err(|e| format!("image decode: {e}"))?;
        Ok(self.process_image(&img))
    }

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
                    patch_grid: span.grid,
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
        let (num, den) = f.delay().numer_denom_ms();
        let frame_ms = num as f32 / den as f32;
        out.push((DynamicImage::ImageRgba8(f.into_buffer()), t_ms / 1000.0));
        t_ms += frame_ms;
    }
    Ok(out)
}

impl VisionArch {
    #[must_use]
    pub fn from_arch_name(arch: &str) -> Option<VisionArch> {
        match arch.to_ascii_lowercase().as_str() {
            "gemma4" => Some(VisionArch::Gemma4),

            "qwen3_5" => Some(VisionArch::Qwen36),
            _ => None,
        }
    }
}

#[must_use]
pub fn audio_arch_supported(arch: &str) -> bool {
    arch.eq_ignore_ascii_case("gemma4")
}

pub fn vision_delimiters(arch: VisionArch) -> (&'static str, &'static str) {
    match arch {
        VisionArch::Qwen36 => ("<|vision_start|>", "<|vision_end|>"),
        VisionArch::Gemma4 => ("", ""),
    }
}

pub fn audio_delimiters(arch: &str) -> (&'static str, &'static str) {
    if audio_arch_supported(arch) {
        ("<|audio>", "<audio|>")
    } else {
        ("", "")
    }
}

pub fn gemma_audio_token_count(n_frames: u32) -> u32 {
    let conv = |n: u32| if n == 0 { 0 } else { (n + 2 - 3) / 2 + 1 };
    conv(conv(n_frames))
}

pub mod audio {

    #[derive(Clone, Copy, Debug)]
    pub struct GemmaAudioProc {
        pub sample_rate: u32,
        pub frame_length: usize,
        pub hop_length: usize,
        pub fft_length: usize,
        pub n_mels: usize,
        pub fmin: f32,
        pub fmax: f32,
        pub mel_floor: f32,
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

    pub fn gemma_logmel(pcm_16k_mono: &[f32]) -> (Vec<f32>, usize) {
        gemma_logmel_with(pcm_16k_mono, &GemmaAudioProc::default())
    }

    pub fn gemma_logmel_with(pcm_16k_mono: &[f32], p: &GemmaAudioProc) -> (Vec<f32>, usize) {
        let frame = p.frame_length;
        let hop = p.hop_length;
        let nfft = p.fft_length;
        let n_freq = nfft / 2 + 1;

        let pad = frame / 2;
        let mut x = Vec::with_capacity(pad + pcm_16k_mono.len());
        x.extend(std::iter::repeat_n(0.0f64, pad));
        x.extend(pcm_16k_mono.iter().map(|&v| v as f64));

        let win_len = frame + 1;
        let n_frames = if x.len() < win_len {
            0
        } else {
            (x.len() - win_len) / hop + 1
        };

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
            pos = body_start + sz + (sz & 1);
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
                    let v = (v << 8) >> 8;
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

    pub fn process_wav_bytes(bytes: &[u8]) -> Result<(Vec<f32>, usize), String> {
        let (pcm, rate) = decode_wav(bytes)?;
        let pcm16k = resample_to_16k(&pcm, rate);
        Ok(gemma_logmel(&pcm16k))
    }
}
