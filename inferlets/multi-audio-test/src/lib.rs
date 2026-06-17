//! Multi-clip-per-forward audio test: fetch one clip, splice it `count` times
//! into a single forward via `Context::append_audios`, and ask the model to
//! transcribe. Validates the driver's N-separate-encodes audio path (each clip
//! encoded independently, scattered at its own anchor). If multi-clip is broken,
//! the transcription is garbled.

use inferlet::media::Audio;
use inferlet::{Context, Result, chat, model::Model, runtime, sample::Sampler};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_url")]
    audio_url: String,
    #[serde(default = "default_count")]
    count: usize,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
}

fn default_url() -> String {
    "https://github.com/Azure-Samples/cognitive-services-speech-sdk/raw/master/samples/cpp/windows/console/samples/whatstheweatherlike.wav".into()
}
fn default_count() -> usize {
    2
}
fn default_max_tokens() -> usize {
    48
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let models = runtime::models();
    let model = Model::load(models.first().ok_or("No models available")?)?;

    let bytes = inferlet::http::fetch(&input.audio_url).await?;
    let clip = Audio::from_bytes(&model, &bytes).map_err(|e| e.to_string())?;
    println!(
        "audio: {} soft tokens; packing {} clips into one forward",
        clip.token_count(),
        input.count
    );

    let mut ctx = Context::new(&model)?;
    ctx.system("You are a helpful audio assistant.")
        .user("Here is some audio:");
    let refs: Vec<&Audio> = (0..input.count).map(|_| &clip).collect();
    ctx.append_audios(&refs).await?;
    ctx.user("Transcribe the speech. What is said?").cue();

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
