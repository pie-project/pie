//! Text-to-speech inferlet — native audio OUTPUT (CSM-1B + Mimi).
//!
//! The inverse of `audio-qa`: instead of perception -> text, this is text ->
//! generated speech waveform. It builds the CSM text prompt (`[speaker]text`),
//! drives the engine's audio-output generation (backbone samples codebook 0;
//! the depth decoder's 31-step RVQ loop samples codebooks 1..31; the Mimi
//! decoder turns each 32-code frame into 1920 PCM samples @ 24 kHz), then wraps
//! the accumulated PCM in a WAV container and returns it base64-encoded.
//!
//! See AUDIO_OUTPUT.md. Requires the CSM model (`eustlb/csm-1b`).
//!
//! Drives the `pie:core/audio-out` host import (`inferlet::audio::generate_speech`):
//! the engine runs the full CSM frame-stepped loop (backbone prefill -> per-frame
//! depth decoder RVQ sampler -> Mimi decode) and returns the 24 kHz PCM, which
//! this inferlet wraps in a WAV container and returns base64-encoded.

use inferlet::{model::Model, runtime, Result};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_text")]
    text: String,
    #[serde(default)]
    speaker: u32,
    #[serde(default = "default_max_frames")]
    max_frames: u32,
}

fn default_text() -> String {
    "Hello, this is a test.".into()
}
fn default_max_frames() -> u32 {
    256
}

/// Mimi / CSM output sample rate (24 kHz mono).
const SAMPLE_RATE: u32 = 24_000;

/// Write mono f32 PCM (`[-1, 1]`) as a canonical 16-bit PCM WAV container — the
/// inverse of `audio-qa`'s hand-written WAV *parser*. No external crate (none in
/// the offline cargo cache); emits the 44-byte RIFF/WAVE header + interleaved
/// little-endian i16 samples.
pub fn write_wav(pcm: &[f32], sample_rate: u32) -> Vec<u8> {
    let n = pcm.len();
    let data_bytes = (n * 2) as u32; // 16-bit mono
    let byte_rate = sample_rate * 2; // sample_rate * channels * bytes_per_sample
    let mut out = Vec::with_capacity(44 + data_bytes as usize);

    // RIFF chunk descriptor.
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_bytes).to_le_bytes()); // ChunkSize
    out.extend_from_slice(b"WAVE");
    // "fmt " sub-chunk (PCM).
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // Subchunk1Size (PCM)
    out.extend_from_slice(&1u16.to_le_bytes()); // AudioFormat = PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // NumChannels = mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes()); // BlockAlign = channels*bytes
    out.extend_from_slice(&16u16.to_le_bytes()); // BitsPerSample
    // "data" sub-chunk.
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());
    for &s in pcm {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
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

    // CSM prompt format: "[speaker_id]text", framed with the Llama-3 special
    // tokens the CSM processor adds (`add_special_tokens=True`): BOS
    // (`<|begin_of_text|>` = 128000) prepended, EOS (`<|end_of_text|>` = 128001)
    // appended. The backbone prefills these before emitting Mimi frames; without
    // the framing the model immediately emits the audio-EOS frame (no speech).
    const BOS: u32 = 128000;
    const EOS: u32 = 128001;
    let prompt = format!("[{}]{}", input.speaker, input.text);
    let tok = model.tokenizer();
    let mut ids = Vec::new();
    ids.push(BOS);
    ids.extend(tok.encode(&prompt));
    ids.push(EOS);
    println!(
        "tts: prompt {prompt:?} -> {} tokens (BOS+text+EOS) {:?}, speaker {}, max_frames {}",
        ids.len(),
        ids,
        input.speaker,
        input.max_frames
    );

    // Drive the engine's audio-output generation (pie:core/audio-out): the
    // backbone samples codebook 0, the depth decoder's 31-step RVQ loop samples
    // codebooks 1..31, and the Mimi decoder turns each 32-code frame into 1920
    // PCM samples @ 24 kHz. Returns the accumulated 24 kHz mono PCM.
    let pcm = inferlet::audio::generate_speech(&model, &ids, input.max_frames).await?;
    let wav = write_wav(&pcm, SAMPLE_RATE);
    println!(
        "tts: generated {} samples ({:.2} s) -> {} WAV bytes",
        pcm.len(),
        pcm.len() as f32 / SAMPLE_RATE as f32,
        wav.len()
    );
    Ok(base64(&wav))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_header_is_canonical() {
        // 1920 samples = one Mimi frame (80 ms @ 24 kHz).
        let pcm: Vec<f32> = (0..1920).map(|i| ((i as f32) / 1920.0) - 0.5).collect();
        let wav = write_wav(&pcm, SAMPLE_RATE);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(&wav[36..40], b"data");
        // 44-byte header + 2 bytes/sample.
        assert_eq!(wav.len(), 44 + pcm.len() * 2);
        // Sample rate field at offset 24.
        assert_eq!(
            u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]),
            SAMPLE_RATE
        );
        // data chunk size at offset 40.
        assert_eq!(
            u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]),
            (pcm.len() * 2) as u32
        );
    }

    #[test]
    fn base64_roundtrip_known_vector() {
        assert_eq!(base64(b"Man"), "TWFu");
        assert_eq!(base64(b"Ma"), "TWE=");
        assert_eq!(base64(b"M"), "TQ==");
    }
}
