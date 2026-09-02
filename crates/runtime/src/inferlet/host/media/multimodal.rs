//! Host-side image, video and audio preprocessing — the arithmetic that
//! turns encoded bytes into the patches, positions and log-mel frames a
//! model was trained on. `media.rs` is the only caller in this crate.
//!
//! Image preprocessing lives in `models::media` (with goldens pinning it
//! against the reference processors). What is left here is what the
//! front-ends do not do: GIF demuxing, and gemma's audio front-end.
//! Dispatch is off the served model's `arch_name` —
//! `models::media::vision_front_end` and [`audio_arch_supported`].

use image::DynamicImage;

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

#[must_use]
pub fn audio_arch_supported(arch: &str) -> bool {
    arch.eq_ignore_ascii_case("gemma4")
}

/// The same, for an audio span.
#[must_use]
pub fn audio_placeholder() -> &'static str {
    "<audio_soft_token>"
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
