//! Image question-answering inferlet (multimodal).
//!
//! Fetches an image over HTTP, encodes it with the model's vision tower
//! (`media::Image` → `Context::append_image`, which runs the encoder driver-side
//! and commits the soft-token KV pages), then answers a question about it with
//! ordinary text generation. See MULTIMODAL.md.
//!
//! Requires a multimodal model (e.g. `gemma-4-E4B`). The image is spliced into
//! the user turn; `append_image` handles the encode + KV commit.

use image::{imageops::FilterType, load_from_memory};
use inferlet::media::Image;
use inferlet::vision;
use inferlet::wstd::http::{Client, Method, Request};
use inferlet::wstd::io::{empty, AsyncRead};
use inferlet::{Context, Result, chat, model::Model, runtime, sample::Sampler};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    /// URL of the image to ask about.
    #[serde(default = "default_url")]
    image_url: String,
    /// Question to ask about the image.
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
    "https://www.ilankelman.org/stopsigns/australia.jpg".into()
}
fn default_question() -> String {
    "What is in this image? Answer in one sentence.".into()
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

/// Resolve a redirect `Location` against the request URL (absolute,
/// scheme-relative `//host/…`, root-relative `/path`, or path-relative).
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

/// Fetch raw (encoded PNG/JPEG) image bytes over HTTP, following up to 8
/// redirects. `wstd::http::Client` does not follow redirects itself, so many
/// hosts (CDNs that 302, hotlink shims) would otherwise hand back a redirect
/// page that fails to decode.
async fn fetch_image_bytes(url: &str) -> Result<Vec<u8>> {
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

    // Arch dispatch (option B: the inferlet patchifies). Qwen3-VL tokenizers
    // carry a `<|vision_start|>` special token; Gemma's do not. The patch
    // layout + resize differ per arch, and Qwen wraps the image rows in
    // `<|vision_start|> … <|vision_end|>` delimiters.
    let tok = model.tokenizer();
    let vision_start = tok.encode("<|vision_start|>");
    let vision_end = tok.encode("<|vision_end|>");
    let is_qwen = vision_start.len() == 1 && vision_end.len() == 1;

    // Option B: decode + resize + patchify in the inferlet, send the pixel blob.
    let bytes = fetch_image_bytes(&input.image_url).await?;
    let decoded = load_from_memory(&bytes).map_err(|e| format!("decode: {e}"))?;

    let (pixels, positions) = if is_qwen {
        // Qwen3-VL: smart-resize to a (patch·merge)-multiple, patchify into the
        // (3,2,16,16)-flattened pixel_values + merge-order (x,y) positions.
        let (th, tw) = vision::qwen_resize_target(decoded.width(), decoded.height());
        let resized = decoded
            .resize_exact(tw, th, FilterType::CatmullRom)
            .to_rgb8();
        vision::qwen_patchify_hwc(resized.as_raw(), th, tw)
    } else {
        // Gemma-4: SigLIP2-style aspect-ratio-preserving resize, channels-last patches.
        let (th, tw) = vision::gemma_resize_target(decoded.width(), decoded.height());
        let resized = decoded
            .resize_exact(tw, th, FilterType::CatmullRom)
            .to_rgb8();
        vision::gemma_patchify_hwc(resized.as_raw(), th, tw)
    };
    let image = Image::from_pixels(&model, &pixels, &positions).map_err(|e| e.to_string())?;
    println!(
        "image ({}): {}x{} -> {} soft tokens (grid {:?})",
        if is_qwen { "qwen3-vl" } else { "gemma4" },
        decoded.width(),
        decoded.height(),
        image.token_count(),
        image.grid()
    );

    // Build the prompt: system → user(image + question) → cue. For Qwen the
    // image span is wrapped in <|vision_start|> … <|vision_end|> delimiters.
    let mut ctx = Context::new(&model)?;
    ctx.system(&input.system).user("Here is an image:");
    if is_qwen {
        ctx.append(&vision_start);
        ctx.append_image(&image).await?;
        ctx.append(&vision_end);
    } else {
        ctx.append_image(&image).await?;
    }
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
