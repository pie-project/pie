//! Multi-image-per-forward test: fetch one image, splice it TWICE into a single
//! forward via `Context::append_images`, and ask the model to describe what it
//! sees. Validates the driver's multi-image batching (block-diagonal encode +
//! per-image scatter). If multi-image is correct, the model should describe the
//! (duplicated) stop-sign scene; if broken, output is garbled.

use inferlet::media::Image;
use inferlet::{Context, Result, chat, model::Model, runtime, sample::Sampler};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_url")]
    image_url: String,
    #[serde(default = "default_count")]
    count: usize,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
}

fn default_url() -> String {
    "https://www.ilankelman.org/stopsigns/australia.jpg".into()
}
fn default_count() -> usize {
    2
}
fn default_max_tokens() -> usize {
    64
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let models = runtime::models();
    let model = Model::load(models.first().ok_or("No models available")?)?;

    let bytes = inferlet::http::fetch(&input.image_url).await?;
    let image = Image::from_bytes(&model, &bytes).map_err(|e| e.to_string())?;
    println!(
        "image: {} soft tokens (grid {:?}); packing {} copies into one forward",
        image.token_count(),
        image.grid(),
        input.count
    );

    let mut ctx = Context::new(&model)?;
    ctx.system("You are a helpful visual assistant.")
        .user("Here are some images:");
    let refs: Vec<&Image> = (0..input.count).map(|_| &image).collect();
    ctx.append_images(&refs).await?;
    ctx.user("Describe what you see in one sentence.").cue();

    let answer = ctx
        .generate(Sampler::TopP {
            temperature: 0.0,
            p: 1.0,
        })
        .max_tokens(input.max_tokens)
        .stop(&chat::stop_tokens(&model))
        .collect_text()
        .await?;

    println!("A: {}", answer);
    Ok(answer)
}
