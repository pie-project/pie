//! Audio question-answering inferlet (multimodal).
//!
//! Fetches a WAV clip over HTTP, decodes it to mono f32 PCM @ 16 kHz, computes
//! log-mel features (`inferlet::audio::gemma_logmel`, parity-verified against
//! `Gemma4AudioFeatureExtractor`), encodes them with the model's audio tower
//! (`media::Audio` → `Context::append_audio`, which runs the gemma4_audio
//! encoder driver-side and commits the soft-token KV), then answers a question
//! about the clip with ordinary text generation. See audio_frontend.md.
//!
//! Requires a multimodal model with an audio tower (e.g. `gemma-4-E4B`). The
//! audio span is wrapped in the `<|audio>` / `<audio|>` delimiters and spliced
//! into the user turn.

use inferlet::audio;
use inferlet::media::Audio;
use inferlet::wstd::http::{Client, Method, Request};
use inferlet::wstd::io::{empty, AsyncRead};
use inferlet::{chat, model::Model, runtime, sample::Sampler, Context, Result};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    /// URL of the WAV clip to ask about.
    #[serde(default = "default_url")]
    audio_url: String,
    /// Question to ask about the audio.
    #[serde(default = "default_question")]
    question: String,
    #[serde(default = "default_system")]
    system: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
}

fn default_url() -> String {
    // 16 kHz mono 16-bit PCM speech clip ("what's the weather like"), ~2 s.
    "https://github.com/Azure-Samples/cognitive-services-speech-sdk/raw/master/samples/cpp/windows/console/samples/whatstheweatherlike.wav".into()
}
fn default_question() -> String {
    "Transcribe this audio. What is being said?".into()
}
fn default_system() -> String {
    "You are a helpful audio assistant.".into()
}
fn default_max_tokens() -> usize {
    128
}
fn default_temperature() -> f32 {
    0.7
}

/// Resolve a redirect `Location` against the request URL.
fn resolve_redirect(base: &str, loc: &str) -> String {
    if loc.starts_with("http://") || loc.starts_with("https://") {
        loc.to_string()
    } else if let Some(rest) = loc.strip_prefix("//") {
        let scheme = base.split("://").next().unwrap_or("https");
        format!("{scheme}://{rest}")
    } else if loc.starts_with('/') {
        let (scheme, after) = base.split_once("://").unwrap_or(("https", base));
        let authority = after.split('/').next().unwrap_or(after);
        format!("{scheme}://{authority}{loc}")
    } else {
        match base.rsplit_once('/') {
            Some((prefix, _)) => format!("{prefix}/{loc}"),
            None => loc.to_string(),
        }
    }
}

/// Fetch raw bytes over HTTP, following up to 8 redirects (GitHub raw 302s to a
/// CDN, so this is required — `wstd::http::Client` does not follow on its own).
async fn fetch_bytes(url: &str) -> Result<Vec<u8>> {
    let client = Client::new();
    let mut current = url.to_string();
    for _ in 0..8 {
        let req = Request::builder()
            .uri(&current)
            .method(Method::GET)
            .body(empty())
            .map_err(|e| e.to_string())?;
        let resp = client.send(req).await.map_err(|e| e.to_string())?;
        let status = resp.status().as_u16();
        if (300..400).contains(&status) {
            let loc = resp
                .headers()
                .get("location")
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| format!("redirect {status} without Location ({current})"))?;
            current = resolve_redirect(&current, loc);
            continue;
        }
        if !(200..300).contains(&status) {
            return Err(format!("HTTP {status} fetching {current}"));
        }
        let mut body = resp.into_body();
        let mut buf = Vec::new();
        body.read_to_end(&mut buf).await.map_err(|e| e.to_string())?;
        return Ok(buf);
    }
    Err(format!("too many redirects fetching {url}"))
}

/// Minimal WAV decoder → `(mono f32 PCM, sample_rate)`. Handles canonical RIFF
/// WAVE with a PCM (format 1) or IEEE-float (format 3) `data` chunk at 8/16/24/
/// 32-bit; downmixes to mono by averaging channels. Skips unknown chunks.
fn decode_wav(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
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
        // Chunks are word-aligned (pad to even).
        pos = body_start + sz + (sz & 1);
    }

    let data = data.ok_or("WAV: no data chunk")?;
    if channels == 0 {
        return Err("WAV: no fmt chunk".into());
    }
    let ch = channels as usize;

    // Decode interleaved samples → per-sample f32 in [-1, 1].
    let mut samples: Vec<f32> = Vec::new();
    match (fmt_tag, bits) {
        (1, 16) => {
            for c in data.chunks_exact(2) {
                samples.push(i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0);
            }
        }
        (1, 8) => {
            // 8-bit PCM is unsigned, centered at 128.
            for &b in data {
                samples.push((b as f32 - 128.0) / 128.0);
            }
        }
        (1, 24) => {
            for c in data.chunks_exact(3) {
                let v = (c[0] as i32) | ((c[1] as i32) << 8) | ((c[2] as i32) << 16);
                let v = (v << 8) >> 8; // sign-extend 24→32
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
        _ => return Err(format!("WAV: unsupported format tag {fmt_tag} / {bits}-bit")),
    }

    // Downmix to mono by averaging channels.
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

/// Resample mono PCM to 16 kHz via linear interpolation. Identity when the
/// input is already 16 kHz (the parity-faithful case). Linear interp is the one
/// inexact step (analogous to vision's resize), absorbed by the encoder.
fn resample_to_16k(pcm: &[f32], src_rate: u32) -> Vec<f32> {
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

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let models = runtime::models();
    let model = Model::load(models.first().ok_or("No models available")?)?;

    // Fetch + decode + resample to 16 kHz mono in the inferlet (option B).
    let bytes = fetch_bytes(&input.audio_url).await?;
    let (pcm, rate) = decode_wav(&bytes)?;
    let pcm16k = resample_to_16k(&pcm, rate);
    println!(
        "audio: {} bytes -> {} samples @ {} Hz -> {} samples @ 16 kHz ({:.2} s)",
        bytes.len(),
        pcm.len(),
        rate,
        pcm16k.len(),
        pcm16k.len() as f32 / 16000.0
    );

    // Log-mel front-end (parity-verified) → audio handle.
    let (mel, n_frames) = audio::gemma_logmel(&pcm16k);
    let clip = Audio::from_mel(&model, &mel, n_frames as u32).map_err(|e| e.to_string())?;
    println!(
        "log-mel: {} frames -> {} audio soft tokens",
        n_frames,
        clip.token_count()
    );

    // Build the prompt: system → user(audio + question) → cue. The audio span
    // is wrapped in the gemma-4 <|audio> … <audio|> delimiters.
    let tok = model.tokenizer();
    let boa = tok.encode("<|audio>");
    let eoa = tok.encode("<audio|>");

    let mut ctx = Context::new(&model)?;
    ctx.system(&input.system).user("Here is an audio clip:");
    ctx.append(&boa);
    ctx.append_audio(&clip).await?;
    ctx.append(&eoa);
    ctx.user(&input.question).cue();

    let answer = ctx
        .generate(Sampler::TopP {
            temperature: input.temperature,
            p: 0.95,
        })
        .max_tokens(input.max_tokens)
        .stop(&chat::stop_tokens(&model))
        .collect_text()
        .await?;

    println!("Q: {}\nA: {}", input.question, answer);
    Ok(answer)
}
