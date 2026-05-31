//! Inferlet-side audio preprocessing (option B): mono f32 PCM @ 16 kHz →
//! log-mel features, matching `Gemma4AudioFeatureExtractor`. Mirrors `vision.rs`.
//!
//! The heavy, model-specific, deterministic front-end (STFT + mel filterbank)
//! runs here in the inferlet; only the compact log-mel blob crosses into the
//! engine, where the `gemma4_audio` encoder + projector turn it into soft-token
//! KV. See `audio_frontend.md`.
//!
//! The pipeline (verified bit-exact against `/tmp/gemma4_audio_parity/
//! input_features.npy` to cosine 1.0000, rel-rms 1.4e-3 — the residual is
//! bf16 rounding in the reference):
//!   1. semicausal pad (prepend `frame_length/2` zeros),
//!   2. frame (win `frame_length`, hop `hop_length`),
//!   3. periodic Hann window,
//!   4. real FFT length `fft_length` (512), magnitude,
//!   5. HTK mel filterbank `[257, 128]` (norm=None), `log(mel + 1e-3)`.
//!
//! The FFT is a self-contained radix-2 Cooley-Tukey (512 is a power of two), so
//! the SDK needs no external FFT crate (none in the offline cargo cache).

use crate::media::Audio;
use crate::model::Model;
use crate::Result;

/// Gemma-4 audio frontend params (defaults match `google/gemma-4-E4B`'s
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

/// HTK mel filterbank `[n_freq, n_mels]` (norm=None) — 130 mel-spaced edges over
/// `[fmin, fmax]`, triangular over the linear FFT-bin centers.
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
    // fb[k][m]
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

/// In-place iterative radix-2 Cooley-Tukey FFT. `re`/`im` length must be a
/// power of two. Forward transform (no normalization), matching `np.fft`.
fn fft_radix2(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());
    // Bit-reversal permutation.
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
    // Butterflies.
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

/// Compute log-mel features `[n_frames * 128]` (frame-major) from mono f32 PCM
/// @ 16 kHz. Returns `(features, n_frames)`. Faithful port of
/// `Gemma4AudioFeatureExtractor._extract_spectrogram` (preemphasis 0, dither 0,
/// per-bin norm off; the single non-exact step is FP summation order in the
/// FFT, absorbed downstream).
pub fn gemma_logmel(pcm_16k_mono: &[f32]) -> (Vec<f32>, usize) {
    gemma_logmel_with(pcm_16k_mono, &GemmaAudioProc::default())
}

/// `gemma_logmel` with explicit params.
pub fn gemma_logmel_with(pcm_16k_mono: &[f32], p: &GemmaAudioProc) -> (Vec<f32>, usize) {
    let frame = p.frame_length;
    let hop = p.hop_length;
    let nfft = p.fft_length;
    let n_freq = nfft / 2 + 1;

    // 1. Semicausal pad: prepend frame/2 zeros.
    let pad = frame / 2;
    let mut x = Vec::with_capacity(pad + pcm_16k_mono.len());
    x.extend(std::iter::repeat(0.0f64).take(pad));
    x.extend(pcm_16k_mono.iter().map(|&v| v as f64));

    // 2. Frame: window of `frame+1` (preemphasis look-behind), step `hop`.
    //    With preemphasis 0 we use the first `frame` samples of each window.
    let win_len = frame + 1;
    let n_frames = if x.len() < win_len {
        0
    } else {
        (x.len() - win_len) / hop + 1
    };

    // 3. Periodic Hann window over `frame` samples.
    let hann: Vec<f64> = (0..frame)
        .map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / frame as f64).cos())
        .collect();

    let fb = mel_filterbank(p);

    let mut out = vec![0.0f32; n_frames * p.n_mels];
    let mut re = vec![0.0f64; nfft];
    let mut im = vec![0.0f64; nfft];
    for fi in 0..n_frames {
        let base = fi * hop;
        // Window into the FFT real buffer (zero-padded 320 → 512).
        re.iter_mut().for_each(|v| *v = 0.0);
        im.iter_mut().for_each(|v| *v = 0.0);
        for n in 0..frame {
            re[n] = x[base + n] * hann[n];
        }
        // 4. FFT + magnitude over the first n_freq bins.
        fft_radix2(&mut re, &mut im);
        // 5. Mel project + log.
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

/// Audio soft tokens for `n_frames` mel frames (two stride-2 Conv2d along time).
/// Mirrors the driver's `gemma4_audio_subsampled_len`.
pub fn gemma_audio_token_count(n_frames: usize) -> usize {
    let c = |n: usize| (n + 2 - 3) / 2 + 1;
    c(c(n_frames))
}

impl Audio {
    /// Build from raw mono PCM @ 16 kHz: runs `gemma_logmel` then `from_mel`.
    pub fn from_pcm(model: &Model, pcm_16k_mono: &[f32]) -> Result<Audio> {
        let (mel, n_frames) = gemma_logmel(pcm_16k_mono);
        Audio::from_mel(model, &mel, n_frames as u32)
    }
}

// =============================================================================
// Native audio OUTPUT (CSM-1B + Mimi) — pie:core/audio-out. See AUDIO_OUTPUT.md.
// =============================================================================

/// Mimi / CSM output sample rate (24 kHz mono).
pub const CSM_SAMPLE_RATE: u32 = 24_000;
/// PCM samples per Mimi frame (12.5 Hz frame rate, ×1920 SEANet upsample = 80 ms).
pub const CSM_SAMPLES_PER_FRAME: usize = 1920;

/// Generate speech from a tokenized CSM prompt. `prompt_tokens` is the
/// "[speaker]text" prompt already encoded by the model's tokenizer; `max_frames`
/// caps the number of 12.5 Hz Mimi frames. Returns 24 kHz mono f32 PCM in
/// `[-1, 1]`. Errors unless the model is a CSM checkpoint.
///
/// Drives the engine's whole frame-stepped loop in one host call (the depth
/// decoder + RVQ sampler and the Mimi decoder are parity-verified — see
/// driver/cuda/tests/csm_depth_decoder_parity.cu / mimi_decoder_full_parity.cu).
pub async fn generate_speech(
    model: &Model,
    prompt_tokens: &[u32],
    max_frames: u32,
) -> Result<Vec<f32>> {
    // The guest WIT binding is synchronous (the host blocks on the driver's
    // one-shot generate_audio cold-path call); the `async` signature is kept so
    // callers can `.await` it uniformly alongside the other context helpers.
    crate::pie::core::audio_out::generate(model, prompt_tokens, max_frames)
}

/// Write mono f32 PCM (`[-1, 1]`) as a canonical 16-bit PCM WAV container.
/// Self-contained (no external crate): 44-byte RIFF/WAVE header + little-endian
/// i16 samples.
pub fn write_wav(pcm: &[f32], sample_rate: u32) -> Vec<u8> {
    let n = pcm.len();
    let data_bytes = (n * 2) as u32; // 16-bit mono
    let byte_rate = sample_rate * 2;
    let mut out = Vec::with_capacity(44 + data_bytes as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());
    for &s in pcm {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}
