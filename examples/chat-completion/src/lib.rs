//! Chat completion with top-p and temperature sampling.
//!
//! The prompt is prefilled in one forward pass. A device-carried decode loop
//! then applies temperature scaling, nucleus filtering, and Gumbel-max sampling
//! in the ETA epilogue.
//!
//! The host reads each sampled token for streaming detokenization and stop-token
//! handling; the sampled token itself remains device-carried into the next pass.

use inferlet::chat;
use inferlet::eta::prelude::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    /// The user prompt to complete.
    prompt: String,
    /// Maximum number of tokens to generate.
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    /// System message for the assistant.
    #[serde(default = "default_system")]
    system: String,
    /// Sampling temperature.
    #[serde(default = "default_temperature")]
    temperature: f32,
    /// Top-p (nucleus) sampling threshold.
    #[serde(default = "default_top_p")]
    top_p: f32,
}

fn default_max_tokens() -> usize {
    256
}
fn default_system() -> String {
    "You are a helpful, respectful and honest assistant.".into()
}
fn default_temperature() -> f32 {
    0.6
}
fn default_top_p() -> f32 {
    0.95
}

/// In-graph top-p + temperature sampler over the read-out row logits `[1,vocab]`.
/// `r` is the taken `[2]` u32 rng state (`[key, ctr]`) driving the Gumbel noise.
/// Returns the sampled token `[1]` i32. Zero temperature is exact greedy
/// decoding; positive temperatures use nucleus sampling.
fn sample_token(r: &Tensor, temperature: f32, top_p: f32) -> Tensor {
    let logits = intrinsics::output(0, "logits"); // [1, vocab] f32 (read-out row)
    if temperature == 0.0 {
        return reduce_argmax(&logits);
    }
    let scaled = &logits / temperature.max(1e-4);
    nucleus_sample(&scaled, top_p, r)
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    if !(0.0..=1.0).contains(&input.top_p) || input.top_p == 0.0 {
        return Err("top_p must be greater than 0 and at most 1".into());
    }
    if !input.temperature.is_finite() {
        return Err("temperature must be finite".into());
    }

    let temperature = input.temperature;
    let top_p = input.top_p;

    // Chat-template the system and user messages, then add the assistant cue.
    // `cue` asks for a thinking answer, so the system turn is the one the
    // template writes for a thinking request (Qwen3.8 opens it with its
    // reasoning-effort instruction).
    let mut prompt_tokens =
        chat::system_thinking(Some(&input.system), chat::ReasoningEffort::Xhigh);
    prompt_tokens.extend(chat::user(&input.prompt));
    prompt_tokens.extend(chat::cue());
    let prompt: Vec<i32> = prompt_tokens.into_iter().map(|t| t as i32).collect();
    let (&last, head) = prompt.split_last().ok_or("empty prompt")?;
    let stop = chat::stop_tokens();

    let ws = WorkingSet::new();
    let pipe = Pipeline::new();
    if !head.is_empty() {
        let fill = ForwardPass::new();
        fill.embed(
            &Channel::from(head),
            &Channel::from([0u32, head.len() as u32]),
        )?;
        fill.state(&[&ws], StateGeometry::folding())?;
        fill.submit(&pipe).context("prefill submit")?;
    }

    let tok_in = Channel::from([last]);
    // No mask: causal is the default, and a bound mask costs the run-ahead (one fire per frame).
    let out = Channel::new([1], dtype::i32).capacity(channel_capacity() as u32);
    let rng = Channel::from([0x9e37_u32, 0]);
    let lane1 = Channel::from([0u32, 1u32]);

    let fwd = ForwardPass::new();
    fwd.embed(&tok_in, &lane1)?;
    fwd.state(&[&ws], StateGeometry::folding())?;
    fwd.epilogue(move || {
        // TAKES + compute first, PUTS last (value-id discipline).
        let r = rng.take();

        let tok = sample_token(&r, temperature, top_p); // [1] i32
        let r_next = &r + iota(2);

        tok_in.put(&tok);
        out.put(&tok);
        rng.put(&r_next);
    });

    let chat_dec = chat::Decoder::new();
    let mut text = String::new();
    run_ahead(&pipe, &fwd, input.max_tokens, async || {
        let t = out.take_host::<Vec<i32>>().await?;
        let token = *t.first().unwrap_or(&0) as u32;
        if stop.contains(&token) {
            return Ok(ControlFlow::Break(()));
        }
        match chat_dec.feed(&[token])? {
            chat::Event::Delta(s) => {
                print!("{s}");
                text.push_str(&s);
            }
            chat::Event::Done(s) => {
                text = s;
                return Ok(ControlFlow::Break(()));
            }
            _ => {}
        }
        Ok(ControlFlow::Continue(()))
    })
    .await?;
    // Any fire still in flight after an early stop is left untaken; close
    // releases the scheduler wait-set, reclaims them, and rejects further
    // submissions.
    pipe.close();

    Ok(text)
}
