//! Applies a greenlist watermark while generating text.
//!
//! Each previously generated token deterministically partitions the vocabulary
//! into a green and red list. Green-token logits receive a configurable bias
//! before Gumbel-max sampling.
//!
//! The partition is drawn on the device: the engine's counter-based rng keyed
//! by `[salt, previous token]` gives every vocabulary entry a uniform that is
//! a pure function of (previous token, entry), and `uniform < gamma` is the
//! greenlist. Hashing the vocabulary on the host cost five milliseconds a
//! token — most of the token — and pinned the loop to one fire in flight.

use inferlet::chat;
use inferlet::eta::prelude::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_gamma")]
    gamma: f32,
    #[serde(default = "default_delta")]
    delta: f32,
}

fn default_prompt() -> String {
    "Explain language-model decoding in simple terms.".into()
}

fn default_max_tokens() -> usize {
    256
}

fn default_gamma() -> f32 {
    0.5
}

fn default_delta() -> f32 {
    2.0
}

/// Keeps the greenlist's rng stream apart from the sampler's.
const GREEN_SALT: u32 = 0x67ee_7157;

/// The greenlist keyed on the previous token (a `[1]` u32): entry `t` is
/// green when its keyed uniform falls below `gamma`.
fn green_mask(previous: &Tensor, gamma: f32, vocab: u32) -> Tensor {
    let is_counter = eq(iota(2), broadcast(1u32, [2]));
    let state = select(
        &is_counter,
        broadcast(previous, [2]),
        broadcast(GREEN_SALT, [2]),
    );
    lt(rng(&state, [vocab]), broadcast(gamma, [vocab]))
}

fn watermarked_sample(
    logits: &Tensor,
    green: &Tensor,
    rng_state: &Tensor,
    delta: f32,
    vocab: u32,
) -> Tensor {
    let bias = select(green, broadcast(delta, [vocab]), broadcast(0.0f32, [vocab]));
    gumbel_max(logits + bias, rng_state)
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    if !(0.0..=1.0).contains(&input.gamma) {
        return Err("gamma must be between 0 and 1".into());
    }
    if !input.delta.is_finite() {
        return Err("delta must be finite".into());
    }

    let vocab = model::output_vocab_size();
    let ws = WorkingSet::new();
    let pipeline = Pipeline::new();

    let mut prompt = chat::system_user("You are a helpful assistant.", &input.prompt);
    prompt.extend(chat::cue());
    let prompt: Vec<i32> = prompt.into_iter().map(|t| t as i32).collect();
    let (&last, head) = prompt.split_last().ok_or("empty prompt")?;
    if !head.is_empty() {
        let fill = ForwardPass::new();
        fill.embed(
            &Channel::from(head),
            &Channel::from([0u32, head.len() as u32]),
        )?;
        fill.state(&[&ws], StateGeometry::folding())?;
        fill.submit(&pipeline).context("watermark prefill")?;
    }
    let stop_tokens = chat::stop_tokens();
    let delta = input.delta;
    let gamma = input.gamma;

    let token_in = Channel::from([last]);
    let rng = Channel::from([0x51ed_u32, 0]);
    let token_out = Channel::new([1], dtype::i32).capacity(channel_capacity() as u32);

    let decode = ForwardPass::new();
    decode.embed(&token_in, &Channel::from([0u32, 1]))?;
    decode.state(&[&ws], StateGeometry::folding())?;
    decode.epilogue(move || {
        // The token this step embedded is the previous output: the key.
        let previous = cast(token_in.take(), dtype::u32);
        let green = green_mask(&previous, gamma, vocab);
        let rng_value = rng.take();
        let token = watermarked_sample(
            &intrinsics::output(0, "logits"),
            &green,
            &rng_value,
            delta,
            vocab,
        );

        token_in.put(&token);
        token_out.put(&token);
        rng.put(&rng_value + iota(2));
    });

    let mut generated = Vec::with_capacity(input.max_tokens);
    run_ahead(&pipeline, &decode, input.max_tokens, async || {
        let token = token_out.take_host::<i32>().await? as u32;
        if stop_tokens.contains(&token) {
            return Ok(ControlFlow::Break(()));
        }
        generated.push(token);
        Ok(ControlFlow::Continue(()))
    })
    .await?;
    pipeline.close();

    model::decode(&generated)
}
