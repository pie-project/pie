//! Text-to-speech inferlet — native audio OUTPUT (CSM-1B + Mimi).
//!
//! The inverse of `audio-qa`: instead of perception -> text, this is text ->
//! generated speech waveform. It is **model-agnostic** — it hands the host plain
//! intent (`model.speak(text).speaker(n)`) and the host applies the bound model's
//! own prompt framing (CSM: `"[speaker]text"` + BOS/EOS), runs the engine's
//! audio-output generation (backbone samples codebook 0; the depth decoder's
//! 31-step RVQ loop samples codebooks 1..31; the Mimi decoder turns each 32-code
//! frame into 1920 PCM samples @ 24 kHz), and returns a self-describing [`Speech`]
//! clip. This inferlet just wraps it in a WAV container and returns it
//! base64-encoded. No CSM constant or special token lives here.
//!
//! See AUDIO_OUTPUT.md. Requires the CSM model (`eustlb/csm-1b`).

use inferlet::{model::Model, runtime, Result};
use serde::Deserialize;
use std::time::Duration;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_text")]
    text: String,
    #[serde(default)]
    speaker: u32,
    /// Cap the generated audio length in seconds. Generation still stops early at
    /// the model's end-of-speech signal.
    #[serde(default = "default_max_seconds")]
    max_seconds: u32,
}

fn default_text() -> String {
    "Hello, this is a test.".into()
}
fn default_max_seconds() -> u32 {
    20
}

/// Base64 (standard alphabet, padded) — for returning the WAV bytes as a string
/// (the offline cargo cache has no base64 crate, so this is self-contained).
fn base64(bytes: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(A[(n >> 18 & 63) as usize] as char);
        out.push(A[(n >> 12 & 63) as usize] as char);
        out.push(if chunk.len() > 1 { A[(n >> 6 & 63) as usize] as char } else { '=' });
        out.push(if chunk.len() > 2 { A[(n & 63) as usize] as char } else { '=' });
    }
    out
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let models = runtime::models();
    let model = Model::load(models.first().ok_or("No models available")?)?;

    // Model-agnostic: hand the host plain intent. The host frames the CSM prompt
    // ("[speaker]text" + BOS/EOS), runs the full frame-stepped loop, and returns
    // a self-describing clip.
    let speech = model
        .speak(&input.text)
        .speaker(input.speaker)
        .max_duration(Duration::from_secs(input.max_seconds as u64))
        .generate()
        .await?;

    let wav = speech.to_wav();
    println!(
        "tts: {:?} (speaker {}) -> {:.2} s @ {} Hz mono, {} WAV bytes",
        input.text,
        input.speaker,
        speech.duration().as_secs_f32(),
        speech.sample_rate(),
        wav.len()
    );
    Ok(base64(&wav))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_roundtrip_known_vector() {
        assert_eq!(base64(b"Man"), "TWFu");
        assert_eq!(base64(b"Ma"), "TWE=");
        assert_eq!(base64(b"M"), "TQ==");
    }
}
