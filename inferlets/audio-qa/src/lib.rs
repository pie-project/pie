//! Audio question-answering inferlet (multimodal).
//!
//! Fetches a WAV clip over HTTP and hands the raw bytes to the host via
//! [`Audio::from_bytes`](inferlet::media::Audio). The host decodes the
//! container, resamples, computes the bound model's log-mel features, runs the
//! audio encoder, commits the soft-token KV, and applies the model's own audio
//! span delimiters. The inferlet then answers a question with ordinary text
//! generation. See audio_frontend.md.
//!
//! Model-agnostic: no decode, resample, log-mel, or delimiter handling here.
//! Requires a model with an audio front-end (e.g. `gemma-4-E4B`).

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

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let models = runtime::models();
    let model = Model::load(models.first().ok_or("No models available")?)?;

    // Model-agnostic: hand the host the raw encoded audio bytes. It decodes the
    // container, resamples, computes the bound model's log-mel features, and
    // wraps the span in whatever delimiters that model needs.
    let bytes = fetch_bytes(&input.audio_url).await?;
    let clip = Audio::from_bytes(&model, &bytes).map_err(|e| e.to_string())?;
    println!(
        "audio: {} bytes -> {} audio soft tokens",
        bytes.len(),
        clip.token_count()
    );

    // Build the prompt: system → user(audio + question) → cue. `append_audio`
    // applies the model's own span delimiters.
    let mut ctx = Context::new(&model)?;
    ctx.system(&input.system).user("Here is an audio clip:");
    ctx.append_audio(&clip).await?;
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
