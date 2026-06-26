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
use inferlet::{Context, Result, chat, sample::Sampler};
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

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    // Model-agnostic: hand the host the raw GIF bytes. It demuxes, uniformly
    // samples up to `max_frames`, and preprocesses each frame at the bound
    // model's per-frame budget — no decode or model-specific code here.
    let bytes = inferlet::http::fetch(&input.video_url).await?;
    let video =
        Video::from_bytes(&bytes, input.max_frames as u32).map_err(|e| e.to_string())?;
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
    let mut ctx = Context::new()?;
    ctx.system(&input.system).user("Here is a video:");
    ctx.append_video(&video).await?;
    ctx.user(&input.question).cue();

    let answer = ctx
        .generate(Sampler::TopP {
            temperature: input.temperature,
            p: 0.95,
        })
        .max_tokens(input.max_tokens)
        .stop(&chat::stop_tokens())
        .collect_text()
        .await?;

    println!("Q: {}\nA: {}", input.question, answer);
    Ok(answer)
}
