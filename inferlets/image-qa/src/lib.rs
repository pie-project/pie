//! Image question-answering inferlet (multimodal).
//!
//! Fetches an image over HTTP, encodes it with the model's vision tower
//! (`media::Image` → `Context::append_image`, which runs the encoder driver-side
//! and commits the soft-token KV pages), then answers a question about it with
//! ordinary text generation. See MULTIMODAL.md.
//!
//! Requires a multimodal model (e.g. `gemma-4-E4B`). The image is spliced into
//! the user turn; `append_image` handles the encode + KV commit.

use inferlet::media::Image;
use inferlet::{Context, Result, chat, sample::Sampler};
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

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    // Model-agnostic: hand the host the raw encoded image bytes. It decodes +
    // resizes + patchifies per the bound model (Gemma SigLIP2, Qwen smart-resize,
    // …) and wraps the span in whatever delimiters that model needs. This
    // inferlet branches on nothing and serves any vision model unchanged.
    let bytes = inferlet::http::fetch(&input.image_url).await?;
    let image = Image::from_bytes(&bytes).map_err(|e| e.to_string())?;
    println!(
        "image: {} bytes -> {} soft tokens (grid {:?})",
        bytes.len(),
        image.token_count(),
        image.grid()
    );

    // Build the prompt: system → user(image + question) → cue. `append_image`
    // applies the model's own span delimiters.
    let mut ctx = Context::new()?;
    ctx.system(&input.system).user("Here is an image:");
    ctx.append_image(&image).await?;
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
