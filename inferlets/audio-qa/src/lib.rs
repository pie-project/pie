//! Audio question-answering inferlet (multimodal) — **raw-WIT keep-core** rewrite.
//!
//! Fetches a WAV clip over HTTP and hands the raw bytes to the host via
//! [`Audio::from_bytes`](inferlet::media::Audio). The host decodes the
//! container, resamples, computes the bound model's log-mel features, runs the
//! audio encoder, commits the soft-token KV (`prefill::audio`), and applies the
//! model's own audio span delimiters. The inferlet then answers a question with
//! an ordinary chat-EOS decode loop on the low-level WIT surface (In Gim's
//! SDK-minimize directive) — no `Context`/`Generator`/`Sampler` facade. See
//! audio_frontend.md.
//!
//! Model-agnostic: no decode, resample, log-mel, or delimiter handling here.
//! Requires a model with an audio front-end (e.g. `gemma-4-E4B`).

use inferlet::inference::ForwardPass;
use inferlet::media::Audio;
use inferlet::sampler::{self, LoweredSampler, SamplerSpec};
use inferlet::working_set::KvWorkingSet;
use inferlet::{carrier, chat, model, prefill, Result};
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
/// first sampling pass); prior context (audio + lead text) is already prefilled
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
    // Model-agnostic: hand the host the raw encoded audio bytes. It decodes the
    // container, resamples, computes the bound model's log-mel features, and
    // wraps the span in whatever delimiters that model needs.
    let bytes = inferlet::http::fetch(&input.audio_url).await?;
    let clip = Audio::from_bytes(&bytes).map_err(|e| e.to_string())?;
    println!(
        "audio: {} bytes -> {} audio soft tokens",
        bytes.len(),
        clip.token_count()
    );

    // Build the prompt on the raw WIT surface: system + "Here is an audio clip:"
    // (a deferred system folds into the first user turn via `system_user`) + the
    // clip's span prefix → prefill; then the audio soft tokens → prefill; then
    // the span suffix + question + cue is the tail the first decode pass samples.
    let kv = KvWorkingSet::new();
    let mut seq_len: u32 = 0;
    let mut fresh = true;

    let mut lead = chat::system_user(&input.system, "Here is an audio clip:");
    lead.extend(clip.prefix_tokens());
    prefill::tokens(&kv, &mut seq_len, &lead)?;
    prefill::audio(&kv, &mut seq_len, &clip)?;

    let mut tail = clip.suffix_tokens();
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
