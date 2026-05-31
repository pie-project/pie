# Audio frontend (log-mel) — design spec

> **SUPERSEDED (2026-05): the frontend moved host-side.** To make inferlets
> model-agnostic, the log-mel front-end no longer runs in the SDK. The inferlet
> now hands the host raw WAV bytes via `Audio::from_bytes`, and the host runs
> the bound model's front-end in **`runtime::multimodal::audio`** (a verbatim
> port of the pipeline below: `decode_wav` → `resample_to_16k` → `gemma_logmel`).
> The §1 log-mel math is still exactly what runs — only its *location* changed
> (WASM → host), so codecs stay sandboxed and the inferlet branches on nothing.
> The old SDK `audio.rs` input frontend (`gemma_logmel`/`from_pcm`/`from_mel`)
> and `vision.rs` have been deleted. Kept for the parity-exact spec; read all
> "in the inferlet" wording below as historical ("in the host").

**Original status (historical):** the inferlet-side option-B path — the inferlet
does the CPU front-end (decode → log-mel features) and hands the engine a feature
blob; the engine's `gemma4_audio` encoder + projector turn that into soft-token
KV. See MULTIMODAL.md ("encoded input becomes ordinary context KV").

---

## 0. Why option B (front-end in the inferlet)

The same reasoning as vision: the heavy, model-specific, deterministic
preprocessing (here the STFT + mel filterbank) runs in the inferlet, and only a
compact, encoder-ready blob crosses into the engine. The engine never learns
about waveforms or FFTs — it receives `log-mel [n_frames, 128]` exactly as the
driver encoder's `run_gemma4_audio(features, n_frames, n_mel, …)` expects, and
the SSCP subsampling + Conformer stack runs on the GPU. This keeps the audio
divergence quarantined in (a) this inferlet front-end and (b) the per-arch
driver graph, never in WIT/wire/runtime hot path.

The frontend must be **parity-exact** vs `Gemma4AudioFeatureExtractor`
(`transformers/models/gemma4/feature_extraction_gemma4.py`), the same way
`vision::gemma_patchify_hwc` is bit-exact vs `Gemma4ImageProcessor`. The single
non-exact step is allowed to be the FFT's floating-point summation order
(absorbed downstream like vision's resize interpolation; verify cosine > 0.999
against `/tmp/gemma4_audio_parity/input_features.npy`).

---

## 1. The exact log-mel pipeline (match `Gemma4AudioFeatureExtractor`)

Constants from `gemma-4-E4B`'s `processor_config.json` `feature_extractor`
block (confirmed in `scripts/gemma4_audio_parity_ref.py`):

| param | value |
|---|---|
| `sampling_rate` | 16000 Hz |
| `frame_length` | 320 samples (20 ms) |
| `hop_length` | 160 samples (10 ms) |
| `fft_length` | 512 (= `2^ceil(log2(320))`) |
| `feature_size` (mel bins) | 128 |
| `min_frequency` / `max_frequency` | 0 / 8000 Hz |
| `mel_floor` | 0.001 |
| `mel_scale` | **HTK** |
| mel filter `norm` | **None** (no Slaney area-norm) |
| `preemphasis` | 0.0 (htk_flavor true — moot at 0) |
| `dither` | 0.0 |
| `input_scale_factor` | 1.0 |
| `fft_overdrive` | false |
| window | **periodic Hann** over `frame_length` |
| `audio_ms_per_token` | 40 (→ encoder 4× subsamples the 10 ms frames) |
| `audio_seq_length` (max tokens) | 750 |

Per-step algorithm (faithful port of `_extract_spectrogram`; **input is `f32`
PCM in `[-1, 1]`, mono, already resampled to 16 kHz** — resampling is the
inferlet's job, see §4):

1. **Semicausal pad:** prepend `frame_length // 2 = 160` zeros to the waveform
   (so the first frame is centered at t=0). Pad the validity mask the same way.
2. **Frame:** unfold into frames of size `frame_length + 1 = 321`, step
   `hop_length = 160`. `num_frames = (padded_len - 321) // 160 + 1`.
   (The `+1` sample is the preemphasis look-behind; with preemphasis 0 you take
   `frames[..., :frame_length]` = the first 320 samples of each 321 window.)
3. **Preemphasis:** `0.0` here → no-op; keep the branch for completeness
   (htk flavor: `first = x[0]*(1-p)`, `rest[i] = x[i] - p*x[i-1]`).
4. **Window:** multiply each 320-sample frame by the **periodic Hann** window
   `w[n] = 0.5 - 0.5*cos(2*pi*n / frame_length)`, `n in [0, 320)`.
   (Periodic, NOT symmetric — divisor is `frame_length`, not `frame_length-1`.)
5. **RFFT:** real FFT of length `fft_length = 512` (frames right-padded with
   zeros from 320 → 512). Output `512/2 + 1 = 257` complex bins.
6. **Magnitude:** `|stft|` (abs of complex, NOT power) → `[n_frames, 257]`.
7. **Mel project:** `magnitude @ mel_filters`, where `mel_filters` is
   `[257, 128]` from an **HTK** mel filterbank (`mel_filter_bank`,
   `num_frequency_bins=257`, `num_mel_filters=128`, fmin 0, fmax 8000,
   `norm=None`). The uppermost triangular filter is all-zero (harmless — it
   falls between bins); reproduce it as zeros.
8. **Log:** `log(mel + mel_floor)` with `mel_floor = 0.001` (natural log).
9. **Per-bin norm:** `per_bin_mean` / `per_bin_stddev` are **null** → skip.
10. **Mask:** a mel frame is valid only if every sample in its analysis window
    `[i*hop, i*hop + 320]` is real audio; frame end index
    `i*hop + 321 - 1`. Then `features *= mask[..., None]` (zero out padding
    frames). For a fully-valid clip (no padding), all frames are valid.

Result: `input_features [n_frames, 128]` `f32`. This is the blob the engine
encoder consumes. `n_frames` for a clean `T`-second clip is
`floor((T*16000 + 160 - 320) / 160) + 1` ≈ `T*100` (one frame per 10 ms).

> The HF extractor's batch `pad(... pad_to_multiple_of=128, max_length=480000,
> truncation=True)` is a **batching** convenience, not part of the math; a
> single-clip inferlet skips it. Just cap at `audio_seq_length` tokens
> (= 750 → ~3000 mel frames ≈ 30 s) and surface a clear error past that.

### Mel filterbank (HTK), to implement once

`mel(f) = 2595 * log10(1 + f/700)` and its inverse. Build 128 triangular
filters over `[0, 8000] Hz` on 257 linear FFT-bin centers
(`bin_freq[k] = k * sr / fft_length = k * 16000/512 = k * 31.25 Hz`), 130 mel
edge points equally spaced in mel between `mel(0)` and `mel(8000)`, `norm=None`
(no `2/(f[i+2]-f[i])` area scaling). Cross-check the produced `[257,128]` matrix
against `weights/audio.…`? No — the mel matrix is not a checkpoint weight;
cross-check the **final** `input_features` against
`/tmp/gemma4_audio_parity/input_features.npy` (the parity ref already dumped it).

---

## 2. FFT crate choice (researched for wasm32-wasip2)

**Primary recommendation: [`realfft`](https://docs.rs/realfft) (on top of
[`rustfft`](https://docs.rs/rustfft)).** Both are **pure Rust** and build for
`wasm32` targets. RustFFT auto-uses Neon on aarch64 and can use WASM SIMD when
the `wasm_simd` crate feature is enabled — but because wasm has no runtime
feature detection it uses that path unconditionally, so for portability across
`wasm32-wasip2` runtimes **leave `wasm_simd` off** (scalar path; the front-end
is tiny relative to the GPU encode). `realfft` exposes an `RealFftPlanner` whose
`plan_fft_forward(512)` gives exactly the 257-bin real FFT step 5 needs and is
~2× faster than a full complex FFT. This matches HF's `np.fft.rfft(n=512)`.
Versions: `rustfft = "6"`, `realfft = "3"` (track latest; both maintained).

**no_std / alloc-free alternative: [`microfft`](https://docs.rs/microfft).** An
in-place radix-2 RFFT requiring no allocations — good if the inferlet must avoid
`alloc`. Caveat: radix-2 needs a power-of-two length, which 512 already is, so
it fits; but its API is fixed-size and less ergonomic than `realfft`. Prefer
`realfft` unless the no_std constraint is real.

> Do NOT pull in `librosa`-style C/numpy deps; the whole point of option B is a
> self-contained wasm front-end. Hann window + mel matrix are a few lines of
> Rust; only the FFT is worth a crate.

Sources:
- RustFFT — https://docs.rs/rustfft/latest/rustfft/ , https://github.com/ejmahler/RustFFT
- RealFFT (feature flags map 1:1 to RustFFT) — https://lib.rs/crates/realfft/features
- microfft (no_std, in-place RFFT) — https://docs.rs/microfft

Add to `sdk/rust/inferlet/Cargo.toml`:
```toml
realfft = "3"   # pulls rustfft (pure Rust); do NOT enable wasm_simd for portability
# microfft = "0.6"   # alternative iff no_std/alloc-free is required
```

---

## 3. Proposed API (mirrors `media::Image` + `Context::append_image`)

### 3.1 A host `audio` resource (WIT — the teammate's `media.wit` edit)

Mirror the `image` resource (MULTIMODAL.md §3). The inferlet computes log-mel,
so the analogue of `from-pixels` is `from-mel`; `from-pcm` is a convenience that
runs §1 in the inferlet then calls `from-mel`.

```wit
// proposed addition to interface media (media.wit) — teammate-owned edit
resource audio {
    // Preprocessed log-mel features [n_frames * 128] f32, row-major
    // (frame-major). The inferlet computed these via the §1 pipeline.
    from-mel: static func(model: borrow<model>, mel: list<f32>, n-frames: u32)
        -> result<audio, error>;

    // Hidden-state rows / KV slots this clip occupies == audio soft tokens ==
    // gemma4_audio_subsampled_len(n_frames) (two stride-2 convs: ~n_frames/4).
    token-count: func() -> u32;

    // 1-D position advance (== token-count for Gemma's 1-D RoPE).
    position-span: func() -> u32;
}
```

The host `HostAudio for InstanceState` (in `runtime/src/api/media.rs`) stores
the mel blob + derives `token_count` from `subsampled_len(n_frames)` (the exact
SSCP downsample: `conv(conv(n_frames))`, `conv(n) = (n-1)/2 + 1`), exactly as
`HostImage::from_pixels` derives soft-token count. Text-only models return a
clean error.

### 3.2 `forward-pass.input-audio` (inference.wit — teammate-owned edit)

```wit
// mirror input-image: splice an encoded audio span at `anchor`.
input-audio: func(audio: borrow<audio>, anchor: u32);
```

The driver runs `scatter_gemma4_audio` (already scaffolded) over the staged mel
features and overwrites `hidden[anchor .. anchor + token_count]` with the
projected `[n_audio_tok, 2560]`, exactly like `scatter_gemma4_vision`.

### 3.3 SDK guest surface (the new `sdk/rust/inferlet/src/audio.rs`)

```rust
//! Inferlet-side audio preprocessing (option B): decode/resample → log-mel
//! features, matching Gemma4AudioFeatureExtractor. Mirrors `vision.rs`.

/// Gemma-4 audio frontend params (defaults match `google/gemma-4-E4B`).
pub struct GemmaAudioProc {
    pub sample_rate: u32,     // 16000
    pub frame_length: usize,  // 320
    pub hop_length: usize,    // 160
    pub fft_length: usize,    // 512
    pub n_mels: usize,        // 128
    pub fmin: f32, pub fmax: f32, // 0, 8000
    pub mel_floor: f32,       // 0.001
}

/// Compute log-mel features [n_frames * 128] from mono f32 PCM @ 16 kHz.
/// Faithful port of §1. (n_frames returned alongside.)
pub fn gemma_logmel(pcm_16k_mono: &[f32]) -> (Vec<f32>, usize /*n_frames*/);

/// Audio soft tokens for n_frames mel frames (two stride-2 convs).
pub fn gemma_audio_token_count(n_frames: usize) -> usize {
    let c = |n: usize| (n - 1) / 2 + 1;
    c(c(n_frames))
}

// Re-exported handle (mirrors `inferlet::media::Image`):
//   pub use crate::pie::core::media::Audio;  // when the WIT resource lands

impl Audio {
    /// Build from raw mono PCM @ 16 kHz: runs `gemma_logmel` then `from_mel`.
    pub fn from_pcm(model: &Model, pcm_16k_mono: &[f32]) -> Result<Audio>;
    /// Build directly from precomputed log-mel features.
    pub fn from_mel(model: &Model, mel: &[f32], n_frames: u32) -> Result<Audio>;
}
```

### 3.4 `Context::append_audio` (the new high-level method — teammate-owned
edit to `context.rs`, shape only)

Byte-for-byte the same page/cursor flow as `append_image` (`context.rs:456`):

```rust
pub async fn append_audio(&mut self, audio: &crate::media::Audio) -> Result<()> {
    self.flush().await?;                       // commit pending text first
    let num_tokens = audio.token_count();
    if num_tokens == 0 { return Ok(()); }
    let span = audio.position_span();
    // … identical reserve_working_pages / ForwardPass::input_audio(audio, seq_len)
    //   / execute_async / commit_working_pages / seq_len += span as append_image …
}
```

And a low-level `Forward::input_audio(&audio, anchor)` builder mirroring
`forward.rs:241`'s `input_image`.

---

## 4. Decode + resample (inferlet's responsibility)

Like vision's decode (the `image` crate inside the inferlet), audio acquisition
+ decode + resample to 16 kHz mono live in the inferlet:

- **Decode** container/codec (wav/flac/mp3/ogg) with a wasm-friendly crate
  (e.g. `symphonia`, pure Rust; or `hound` for plain WAV). Produce interleaved
  or mono `f32`/`i16` PCM.
- **Downmix** to mono (average channels) and **resample** to exactly 16000 Hz
  (e.g. `rubato` sinc resampler, pure Rust; or a simple polyphase for fixed
  ratios). Resampling is the one inexact step, analogous to vision's resize —
  small spectral differences are absorbed by the encoder (target cosine
  > 0.999 vs the parity ref on a clip generated at native 16 kHz, where
  resampling is identity).
- Then call `Audio::from_pcm`.

A `inferlets/audio-qa/` example (mirroring `inferlets/image-qa/`) would: fetch a
clip → decode/resample → `Audio::from_pcm` → `ctx.append_audio` (wrapped by the
`<|audio>` / `<audio|>` open/close markers — see the instruct edit in the
integration checklist) → `ctx.user(question).cue()` → generate.

---

## 5. Parity check (before trusting the front-end)

Run `scripts/gemma4_audio_parity_ref.py` (already RUN: 751 tensors, missing=0).
It dumps `/tmp/gemma4_audio_parity/input_waveform.npy` (the 2 s chirp),
`input_features.npy` (the ground-truth log-mel), and the downstream encoder
stages. The Rust `gemma_logmel` must reproduce `input_features.npy` from
`input_waveform.npy` to cosine > 0.999 (bf16-vs-bf16 rel-rms, NOT max_abs — see
MULTIMODAL.md §11). The driver encoder is checked separately against
`sscp_out` / `layer{0,5,11}` / `encoder_out` / `projected`.
