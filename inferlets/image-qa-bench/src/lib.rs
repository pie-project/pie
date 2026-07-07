//! Image-QA inferlet for benchmarking — the multimodal sibling of
//! `text-completion-bench`.
//!
//! Differences from `image-qa`:
//!   * **No network fetch.** The image arrives as base64 in the input, so the
//!     timed path is decode + resize + patchify + vision-encode + prefill +
//!     decode — never HTTP. (The host still does all model-specific work; this
//!     inferlet stays model-agnostic.)
//!   * Returns `num_prompt_tokens` (text tokens + image soft-token rows, read
//!     from the context's sequence length) and `num_output_tokens` (authoritative,
//!     from the generator) so the harness can do apples-to-apples accounting
//!     against vLLM — exactly like `text-completion-bench` does for text.
//!   * `ignore_eos` forces every request to consume the full `max_tokens` budget,
//!     so cross-engine comparisons aren't biased by stop-token handling.
//!
//! Single mode: one `image_b64` + `question`. Batch mode: `images_b64` /
//! `questions` arrays driven in one process (used by `--single-process-batch`).

use inferlet::inference::ForwardPass;
use inferlet::sampler::{self, LoweredSampler, SamplerSpec};
use inferlet::working_set::KvWorkingSet;
use inferlet::{
    carrier, chat, media::Image, model, pie::core::session, prefill, Result,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    /// Base64-encoded image bytes (single mode).
    #[serde(default)]
    image_b64: String,
    /// Base64-encoded images (batch mode). Falls back to `image_b64` per slot.
    #[serde(default)]
    images_b64: Vec<String>,
    #[serde(default = "default_question")]
    question: String,
    /// Per-request questions (batch mode). Falls back to `question` per slot.
    #[serde(default)]
    questions: Vec<String>,
    #[serde(default = "default_system")]
    system: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    /// Drop chat-template stop tokens so generation runs to exactly `max_tokens`.
    #[serde(default)]
    ignore_eos: bool,
    /// Decode + return generated text. Off in throughput runs (counts suffice).
    #[serde(default = "default_return_text")]
    return_text: bool,
    /// Prepare the prompt, signal `ready`, then block until the harness sends
    /// `start` — lets throughput runs exclude launch/prefill-setup skew.
    #[serde(default)]
    wait_for_start: bool,
    /// Cap how many batch items run concurrently inside one process.
    #[serde(default)]
    batch_concurrency: Option<usize>,
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
fn default_top_p() -> f32 {
    1.0
}
fn default_return_text() -> bool {
    true
}

#[derive(Serialize)]
struct Output {
    /// Text tokens + image soft-token rows that fed the prefill (== context
    /// sequence length before the first generated token). Comparable to the
    /// `prompt_tokens` vLLM reports (text + expanded vision placeholders).
    num_prompt_tokens: usize,
    /// Tokens the sampler actually emitted. Authoritative.
    num_output_tokens: usize,
    text: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    request_prompt_tokens: Vec<usize>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    request_output_tokens: Vec<usize>,
}

/// Decode standard base64 (padded or not; whitespace tolerated). Self-contained
/// so the inferlet pulls in no extra crate.
fn b64_decode(s: &str) -> Result<Vec<u8>> {
    fn val(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let mut buf: u32 = 0;
    let mut bits: u32 = 0;
    for &c in s.as_bytes() {
        if c == b'=' || c == b'\n' || c == b'\r' || c == b' ' {
            continue;
        }
        let v = val(c).ok_or("invalid base64 in image_b64")? as u32;
        buf = (buf << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
        }
    }
    Ok(out)
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let stop_tokens: Vec<u32> = if input.ignore_eos {
        Vec::new()
    } else {
        chat::stop_tokens()
    };

    let batch_len = input.images_b64.len().max(input.questions.len());
    if batch_len > 0 {
        if input.wait_for_start {
            session::send("ready");
            let _ = session::receive().await;
        }
        let batch_concurrency = input
            .batch_concurrency
            .unwrap_or(batch_len)
            .clamp(1, batch_len);
        let mut request_prompt_tokens = Vec::with_capacity(batch_len);
        let mut request_output_tokens = Vec::with_capacity(batch_len);
        let mut first_text = String::new();
        let mut offset = 0usize;
        while offset < batch_len {
            let end = (offset + batch_concurrency).min(batch_len);
            let futures = (offset..end).map(|i| {
                let image_b64 = input
                    .images_b64
                    .get(i)
                    .filter(|s| !s.is_empty())
                    .unwrap_or(&input.image_b64);
                let question = input.questions.get(i).unwrap_or(&input.question);
                run_one(&input, image_b64, question, &stop_tokens, false)
            });
            for (j, result) in futures::future::join_all(futures).await.into_iter().enumerate() {
                let r = result?;
                if offset + j == 0 {
                    first_text = r.text;
                }
                request_prompt_tokens.push(r.num_prompt_tokens);
                request_output_tokens.push(r.num_output_tokens);
            }
            offset = end;
        }
        return Ok(Output {
            num_prompt_tokens: request_prompt_tokens.iter().sum(),
            num_output_tokens: request_output_tokens.iter().sum(),
            text: first_text,
            request_prompt_tokens,
            request_output_tokens,
        });
    }

    let r = run_one(
        &input,
        &input.image_b64,
        &input.question,
        &stop_tokens,
        true,
    )
    .await?;
    Ok(Output {
        num_prompt_tokens: r.num_prompt_tokens,
        num_output_tokens: r.num_output_tokens,
        text: r.text,
        request_prompt_tokens: Vec::new(),
        request_output_tokens: Vec::new(),
    })
}

struct RunResult {
    num_prompt_tokens: usize,
    num_output_tokens: usize,
    text: String,
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

/// Pipelined greedy/parametric decode collecting raw tokens (the bench counts
/// them; no chat detok). `stop` may be EMPTY (`ignore_eos` → run to exactly
/// `max_tokens`, the no-discard run-ahead path); when non-empty this is the
/// depth-1 EOS pipeline (speculate + `carrier::discard_pass` on a stop, dropping
/// the stop token). `prompt` is the trailing prompt tail; media + lead text are
/// already prefilled into `kv`.
async fn decode_tokens(
    kv: &KvWorkingSet,
    seq_len: &mut u32,
    fresh: &mut bool,
    s: &LoweredSampler,
    prompt: &[u32],
    max_tokens: usize,
    stop: &[u32],
) -> Result<Vec<u32>> {
    let mut out: Vec<u32> = Vec::with_capacity(max_tokens);
    if max_tokens == 0 {
        return Ok(out);
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
            out.push(token);
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
    Ok(out)
}

async fn run_one(
    input: &Input,
    image_b64: &str,
    question: &str,
    stop_tokens: &[u32],
    honor_wait_for_start: bool,
) -> Result<RunResult> {
    if image_b64.is_empty() {
        return Err("image_b64 is empty".into());
    }
    let bytes = b64_decode(image_b64)?;

    // Host-side: decode + resize + patchify per the bound model (timed).
    let image = Image::from_bytes(&bytes).map_err(|e| e.to_string())?;

    // Build the same prompt shape as `image-qa` on the raw WIT surface: system +
    // "Here is an image:" (deferred system → system_user) + image span prefix →
    // prefill; image soft tokens → prefill (the vision encoder runs driver-side,
    // timed); span suffix + question + cue = the trailing tail the first decode
    // pass samples from.
    let kv = KvWorkingSet::new();
    let mut seq_len: u32 = 0;
    let mut fresh = true;

    let mut lead = chat::system_user(&input.system, "Here is an image:");
    lead.extend(image.prefix_tokens());
    prefill::tokens(&kv, &mut seq_len, &lead)?;
    prefill::image(&kv, &mut seq_len, &image)?;

    let mut tail = image.suffix_tokens();
    tail.extend(chat::user(question));
    tail.extend(chat::cue());
    // Full prompt = committed prefill rows (text + image soft tokens) + the
    // trailing tail (question + cue) the first decode pass consumes.
    let num_prompt_tokens = seq_len as usize + tail.len();

    if honor_wait_for_start && input.wait_for_start {
        session::send("ready");
        let _ = session::receive().await;
    }

    let s = if input.temperature <= 0.0 {
        sampler::sampler_program(SamplerSpec::Argmax, model::output_vocab_size())?
    } else {
        sampler::sampler_program(
            SamplerSpec::TopP {
                temperature: input.temperature,
                p: input.top_p,
            },
            model::output_vocab_size(),
        )?
    };

    let tokens = decode_tokens(
        &kv,
        &mut seq_len,
        &mut fresh,
        &s,
        &tail,
        input.max_tokens,
        stop_tokens,
    )
    .await?;

    let text = if input.return_text {
        model::decode(&tokens).unwrap_or_default()
    } else {
        String::new()
    };
    Ok(RunResult {
        num_prompt_tokens,
        num_output_tokens: tokens.len(),
        text,
    })
}
