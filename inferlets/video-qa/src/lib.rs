//! Video question-answering inferlet (multimodal) — **raw-WIT keep-core** rewrite.
//!
//! Fetches an animated GIF over HTTP and hands the raw bytes to the host, which
//! demuxes, uniformly samples up to `max_frames` frames, and preprocesses each
//! at the bound model's per-frame budget. Each frame's soft tokens are spliced
//! into the context via `prefill::image` (preceded by a generic `mm:ss`
//! timestamp marker), then a question is answered with an ordinary chat-EOS
//! decode loop on the low-level WIT surface (In Gim's SDK-minimize directive) —
//! no `Context`/`Generator`/`Sampler` facade. See MULTIMODAL.md §8.
//!
//! Model-agnostic: no decode, resize, or patchify here — the same binary serves
//! any vision model. GIF is the first-cut container; mp4 would need a host-side
//! demuxer. `max_frames` is the KV-budget knob (each frame ≈ tens of soft tokens).

use inferlet::inference::ForwardPass;
use inferlet::media::Video;
use inferlet::sampler::{self, LoweredSampler, SamplerSpec};
use inferlet::working_set::KvWorkingSet;
use inferlet::{carrier, chat, model, prefill, Result};
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
/// first sampling pass); prior context (frames + lead text) is already prefilled
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
    let per_frame = video.frame(0).map_err(|e| e.to_string())?.token_count();
    println!(
        "video: sampled {} frames @ {} soft tokens/frame ({} total)",
        n,
        per_frame,
        per_frame * n,
    );

    // Build the prompt on the raw WIT surface: system + "Here is a video:"
    // (a deferred system folds into the first user turn via `system_user`) →
    // prefill; then each frame — a `mm:ss` timestamp marker + the frame's span
    // prefix → prefill, the frame soft tokens → prefill, its span suffix →
    // prefill (mirrors `Context::append_video`'s per-frame `append_image`);
    // then the question + cue is the tail the first decode pass samples from.
    let kv = KvWorkingSet::new();
    let mut seq_len: u32 = 0;
    let mut fresh = true;

    let lead = chat::system_user(&input.system, "Here is a video:");
    prefill::tokens(&kv, &mut seq_len, &lead)?;

    for i in 0..n {
        let secs = video.timestamp(i).max(0.0) as u32;
        let marker = format!(" {:02}:{:02} ", secs / 60, secs % 60);
        let frame = video
            .frame(i)
            .map_err(|e| format!("video frame {i}: {e}"))?;
        let mut pre = model::encode(&marker);
        pre.extend(frame.prefix_tokens());
        prefill::tokens(&kv, &mut seq_len, &pre)?;
        prefill::image(&kv, &mut seq_len, &frame)?;
        prefill::tokens(&kv, &mut seq_len, &frame.suffix_tokens())?;
    }

    let mut tail = chat::user(&input.question);
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
