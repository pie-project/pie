//! Inferlet-side audio helpers.
//!
//! Audio **input** preprocessing (decode → resample → log-mel) is not done here:
//! the inferlet hands the host raw encoded bytes via
//! [`Audio::from_bytes`](crate::media::Audio) and the host runs the bound model's
//! front-end (see `runtime::multimodal::audio`).
//!
//! Audio **output** is model-agnostic in the same spirit: the inferlet expresses
//! intent via [`speak`] — the text to say, a [`Voice`], and an optional
//! target duration — and the host applies the bound model's own prompt framing
//! (CSM: `"[speaker]text"` + BOS/EOS) and returns a self-describing [`Speech`]
//! clip. No CSM constant or special-token id lives in the inferlet.

use crate::Result;
use std::time::Duration;

/// Which voice to synthesize in. Re-exported from the host binding so callers
/// construct it directly: `Voice::Speaker(0)`.
pub use crate::pie::core::audio_out::Voice;

/// A generated audio clip. **Self-describing** — it carries its own sample rate
/// and channel count, so callers never hardcode a model constant like 24 kHz.
/// Thin wrapper over the host `speech` resource; PCM is materialized on demand.
pub struct Speech {
    inner: crate::pie::core::audio_out::Speech,
}

impl Speech {
    /// Output sample rate in Hz (CSM: 24 000).
    pub fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }

    /// Channel count (CSM: 1, mono).
    pub fn channels(&self) -> u32 {
        self.inner.channels()
    }

    /// Duration of the generated clip.
    pub fn duration(&self) -> Duration {
        Duration::from_millis(self.inner.duration_ms() as u64)
    }

    /// Decoded PCM samples in `[-1, 1]` (mono => one sample per frame).
    pub fn pcm(&self) -> Vec<f32> {
        self.inner.pcm()
    }

    /// Encode the clip as a canonical 16-bit PCM WAV container, using the clip's
    /// own [`sample_rate`](Self::sample_rate). Pure; no model knowledge.
    pub fn to_wav(&self) -> Vec<u8> {
        write_wav(&self.pcm(), self.sample_rate())
    }
}

/// Builder for a speech-synthesis request. Created via [`speak`]; finish
/// with [`generate`](Self::generate).
#[must_use = "a SpeechBuilder does nothing until `.generate().await` is called"]
pub struct SpeechBuilder {
    text: String,
    voice: Voice,
    max_duration: Option<Duration>,
}

impl SpeechBuilder {
    /// Set the voice. Defaults to `Voice::Speaker(0)`.
    pub fn voice(mut self, voice: Voice) -> Self {
        self.voice = voice;
        self
    }

    /// Convenience for [`voice`](Self::voice)`(Voice::Speaker(id))`.
    pub fn speaker(mut self, id: u32) -> Self {
        self.voice = Voice::Speaker(id);
        self
    }

    /// Cap the generated audio length. Omit for the model's default cap
    /// (generation still stops early at the model's end-of-speech signal).
    pub fn max_duration(mut self, dur: Duration) -> Self {
        self.max_duration = Some(dur);
        self
    }

    /// Synthesize. The host applies the bound model's prompt framing and returns
    /// a self-describing [`Speech`]. Errors if the model has no audio-output
    /// front-end (i.e. is not a CSM checkpoint).
    pub async fn generate(self) -> Result<Speech> {
        let req = crate::pie::core::audio_out::SpeechRequest {
            text: self.text,
            voice: self.voice,
            max_duration_ms: self
                .max_duration
                .map(|d| d.as_millis().min(u32::MAX as u128) as u32),
        };
        let inner = crate::pie::core::audio_out::Speech::generate(&req)?;
        Ok(Speech { inner })
    }
}

/// Begin a model-agnostic speech-synthesis request. The inferlet supplies
/// only intent; the host owns all model-specific framing.
///
/// ```ignore
/// use std::time::Duration;
/// let speech = inferlet::audio::speak("Hello, this is a test.")
///     .speaker(0)
///     .max_duration(Duration::from_secs(20))
///     .generate()
///     .await?;
/// let wav = speech.to_wav(); // uses speech.sample_rate(), not a constant
/// ```
pub fn speak(text: impl Into<String>) -> SpeechBuilder {
    SpeechBuilder {
        text: text.into(),
        voice: Voice::Speaker(0),
        max_duration: None,
    }
}

/// Write mono f32 PCM (`[-1, 1]`) as a canonical 16-bit PCM WAV container.
/// Self-contained (no external crate): 44-byte RIFF/WAVE header + little-endian
/// i16 samples. Prefer [`Speech::to_wav`], which supplies the sample rate for you.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_header_is_canonical() {
        let pcm: Vec<f32> = (0..1920).map(|i| (i as f32 / 1920.0) - 0.5).collect();
        let wav = write_wav(&pcm, 24_000);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(&wav[36..40], b"data");
        assert_eq!(wav.len(), 44 + pcm.len() * 2);
        assert_eq!(u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]), 24_000);
        assert_eq!(
            u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]),
            (pcm.len() * 2) as u32
        );
    }
}
