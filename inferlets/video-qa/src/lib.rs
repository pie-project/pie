//! Video question-answering inferlet (multimodal).
//!
//! Fetches an animated GIF over HTTP and hands the raw bytes to the host, which
//! demuxes, uniformly samples up to `max_frames` frames, and preprocesses each
//! at the bound model's per-frame budget. The frames are spliced into the
//! context via [`Context::append_video`] (each preceded by a generic `mm:ss`
//! timestamp marker), then a question is answered with ordinary text
//! generation. See MULTIMODAL.md §8.
//!
//! Model-agnostic: no decode, resize, or patchify here — the same binary serves
//! any vision model. GIF is the first-cut container; mp4 would need a host-side
//! demuxer. `max_frames` is the KV-budget knob (each frame ≈ tens of soft tokens).

use inferlet::media::Video;
use inferlet::wstd::http::{Client, Method, Request};
use inferlet::wstd::io::{AsyncRead, empty};
use inferlet::{Context, Result, chat, model::Model, runtime, sample::Sampler};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    /// URL of an animated GIF to ask about.
    #[serde(default = "default_url")]
    video_url: String,
    #[serde(default = "default_question")]
    question: String,
    /// Max frames to uniformly sample from the clip (KV-budget knob).
    #[serde(default = "default_max_frames")]
    max_frames: usize,
    #[serde(default = "default_system")]
    system: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
}

fn default_url() -> String {
    // A rotating-earth animation: clear motion for a video (not still) test.
    "https://upload.wikimedia.org/wikipedia/commons/2/2c/Rotating_earth_%28large%29.gif".into()
}
fn default_question() -> String {
    "Describe what happens in this video in one or two sentences.".into()
}
fn default_max_frames() -> usize {
    8
}
fn default_system() -> String {
    "You are a helpful visual assistant.".into()
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

/// Fetch raw bytes over HTTP, following up to 8 redirects.
async fn fetch_bytes(url: &str) -> Result<Vec<u8>> {
    let client = Client::new();
    let mut current = url.to_string();
    for _ in 0..8 {
        let req = Request::builder()
            .uri(&current)
            // Many media hosts (e.g. Wikimedia) 403 requests without a UA.
            .header("User-Agent", "pie-inferlet/0.1 (multimodal video-qa)")
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

    // Model-agnostic: hand the host the raw GIF bytes. It demuxes, uniformly
    // samples up to `max_frames`, and preprocesses each frame at the bound
    // model's per-frame budget — no decode or model-specific code here.
    let bytes = fetch_bytes(&input.video_url).await?;
    let video =
        Video::from_bytes(&model, &bytes, input.max_frames as u32).map_err(|e| e.to_string())?;
    let n = video.frame_count();
    if n == 0 {
        return Err("no frames sampled from video".into());
    }
    let per_frame = video
        .frame(0)
        .map_err(|e| e.to_string())?
        .token_count();
    println!(
        "video: sampled {} frames @ {} soft tokens/frame ({} total)",
        n,
        per_frame,
        per_frame * n,
    );

    // Prompt: system → "Here is a video:" → frames+timestamps → question → cue.
    let mut ctx = Context::new(&model)?;
    ctx.system(&input.system).user("Here is a video:");
    ctx.append_video(&video).await?;
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
