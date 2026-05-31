//! Video question-answering inferlet (multimodal).
//!
//! Fetches an animated GIF over HTTP, samples up to `max_frames` frames,
//! patchifies each at the Gemma-4 video soft-token budget (≤70 tokens/frame),
//! splices them into the context via [`Context::append_video`] (each frame is
//! encoded by the same vision tower as a still image, preceded by an `mm:ss`
//! timestamp marker — Gemma 4 has no temporal patching), then answers a
//! question with ordinary text generation. See MULTIMODAL.md §8.
//!
//! Requires a multimodal model (e.g. `gemma-4-E4B`). GIF is the first-cut
//! container because the `image` crate decodes it offline in wasm; mp4 would
//! need a host-side decoder. Each frame ≈ up to 70 soft tokens, so an N-frame
//! clip costs ≈ N×70 KV slots — `max_frames` is the budget knob.

use image::codecs::gif::GifDecoder;
use image::{AnimationDecoder, DynamicImage, imageops::FilterType};
use inferlet::media::Image;
use inferlet::vision;
use inferlet::wstd::http::{Client, Method, Request};
use inferlet::wstd::io::{AsyncRead, empty};
use inferlet::{Context, Result, chat, model::Model, runtime, sample::Sampler};
use serde::Deserialize;
use std::io::Cursor;

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

    // 1. Fetch + decode the GIF into RGBA frames (Option B: decode in the inferlet).
    let bytes = fetch_bytes(&input.video_url).await?;
    let frames = GifDecoder::new(Cursor::new(bytes))
        .map_err(|e| format!("gif: {e}"))?
        .into_frames()
        .collect_frames()
        .map_err(|e| format!("frames: {e}"))?;
    if frames.is_empty() {
        return Err("no frames in GIF".into());
    }

    // Cumulative timestamp (seconds) at the start of each frame, from GIF delays.
    let mut cum_ms = 0.0f32;
    let timestamps_all: Vec<f32> = frames
        .iter()
        .map(|f| {
            let t = cum_ms / 1000.0;
            let (num, den) = f.delay().numer_denom_ms();
            cum_ms += num as f32 / den.max(1) as f32;
            t
        })
        .collect();

    // 2. Uniformly sample up to max_frames; patchify each at the video budget.
    let n = frames.len();
    let take = input.max_frames.clamp(1, n);
    let mut images = Vec::with_capacity(take);
    let mut timestamps = Vec::with_capacity(take);
    for k in 0..take {
        let idx = k * n / take;
        let rgba = frames[idx].buffer();
        let (w, h) = (rgba.width(), rgba.height());
        let (th, tw) = vision::gemma_resize_target_video(w, h);
        let resized = DynamicImage::ImageRgba8(rgba.clone())
            .resize_exact(tw, th, FilterType::CatmullRom)
            .to_rgb8();
        let (pixels, positions) = vision::gemma_patchify_hwc(resized.as_raw(), th, tw);
        images.push(Image::from_pixels(&model, &pixels, &positions).map_err(|e| e.to_string())?);
        timestamps.push(timestamps_all[idx]);
    }
    println!(
        "video: {} frames -> sampled {} @ {} soft tokens/frame ({} total)",
        n,
        take,
        images[0].token_count(),
        images.iter().map(|i| i.token_count()).sum::<u32>(),
    );

    // 3. Prompt: system → "Here is a video:" → frames+timestamps → question → cue.
    let mut ctx = Context::new(&model)?;
    ctx.system(&input.system).user("Here is a video:");
    ctx.append_video(&images, &timestamps).await?;
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
