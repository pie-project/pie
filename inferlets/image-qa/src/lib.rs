//! Image question-answering inferlet (multimodal) — **raw-WIT keep-core** rewrite.
//!
//! Fetches an image over HTTP, encodes it with the model's vision tower
//! (`media::Image` → `prefill::image`, which runs the encoder driver-side and
//! commits the soft-token KV pages), then answers a question about it with an
//! ordinary chat-EOS decode loop written directly on the low-level WIT surface
//! (In Gim's SDK-minimize directive): no `Context`/`Generator`/`Sampler` facade.
//! See MULTIMODAL.md.
//!
//! Requires a multimodal model (e.g. `gemma-4-E4B`). The image is spliced into
//! the user turn between the model's own span delimiters
//! (`image.prefix_tokens()` / `suffix_tokens()`, host-provided → model-agnostic).

use inferlet::inference::ForwardPass;
use inferlet::media::Image;
use inferlet::sampler::{self, LoweredSampler, SamplerSpec};
use inferlet::working_set::KvWorkingSet;
use inferlet::{carrier, chat, model, prefill, Result};
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

/// Read the sampled token off a finalized pass's single-`Token` output tensor.
async fn read_token(pass: ForwardPass) -> Result<u32> {
    let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("tensor read: {e:?}"))?;
    Ok(if bytes.len() >= 4 {
        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u32
    } else {
        0
    })
}

/// Chat-EOS depth-1 pipelined decode over the keep-core carrier, streaming the
/// text out through `chat::Decoder`. `prompt` is the trailing prompt tail (the
/// first sampling pass); prior context (media + lead text) is already prefilled
/// into `kv`. Speculate the next forward eagerly, roll an over-shot pass back
/// with `carrier::discard_pass` on a stop. See `ptir-pipelined-eos-rollback-spec`.
async fn decode_chat(
    kv: &KvWorkingSet,
    seq_len: &mut u32,
    fresh: &mut bool,
    s: &LoweredSampler,
    prompt: &[u32],
    max_tokens: usize,
) -> Result<String> {
    let stop = chat::stop_tokens();
    let mut dec = chat::Decoder::new();
    let mut text = String::new();
    if max_tokens == 0 {
        return Ok(text);
    }
    let prompt = if prompt.is_empty() { &[0u32][..] } else { prompt };
    let mut producer = carrier::submit_pass(kv, seq_len, fresh, s, prompt, true)?;
    let mut generated = 0usize;
    loop {
        let speculate = generated + 1 < max_tokens;
        let consumer = if speculate {
            Some(carrier::submit_pass(kv, seq_len, fresh, s, &[0u32], true)?)
        } else {
            None
        };
        let token = read_token(producer).await?;
        generated += 1;
        let mut done = stop.contains(&token);
        if !done {
            match dec.feed(&[token])? {
                chat::Event::Delta(t) => {
                    print!("{}", t);
                    text.push_str(&t);
                }
                chat::Event::Done(t) => {
                    text = t;
                    done = true;
                }
                _ => {}
            }
        }
        if generated >= max_tokens {
            done = true;
        }
        if done {
            if let Some(c) = consumer {
                carrier::discard_pass(c, seq_len).await;
            }
            break;
        }
        producer = consumer.expect("consumer speculated when not terminal");
    }
    Ok(text)
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

    // Build the prompt on the raw WIT surface: system + "Here is an image:"
    // (a deferred system folds into the first user turn via `system_user`,
    // mirroring `Context::user`) + the image's span prefix → prefill; then the
    // image soft tokens → prefill; then the span suffix + question + cue is the
    // trailing tail the first decode pass samples from.
    let kv = KvWorkingSet::new();
    let mut seq_len: u32 = 0;
    let mut fresh = true;

    let mut lead = chat::system_user(&input.system, "Here is an image:");
    lead.extend(image.prefix_tokens());
    prefill::tokens(&kv, &mut seq_len, &lead)?;
    prefill::image(&kv, &mut seq_len, &image)?;

    let mut tail = image.suffix_tokens();
    tail.extend(chat::user(&input.question));
    tail.extend(chat::cue());

    let s = sampler::sampler_program(
        SamplerSpec::TopP {
            temperature: input.temperature,
            p: 0.95,
        },
        model::output_vocab_size(),
    )?;

    let answer = decode_chat(&kv, &mut seq_len, &mut fresh, &s, &tail, input.max_tokens).await?;

    println!("Q: {}\nA: {}", input.question, answer);
    Ok(answer)
}
