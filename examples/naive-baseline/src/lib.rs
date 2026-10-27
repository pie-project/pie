//! Naive text completion — the performance control for the algorithm inferlets.
//!
//! Structurally identical to every truncation/penalty inferlet in this
//! directory: an epilogue-free fill of the prompt head, then a device-carried
//! decode loop from the prompt's last token, driven by `eta::run_ahead`, which
//! keeps the runtime's run-ahead window (`model.channel-capacity()`) full ahead
//! of the host drain. The only
//! difference is the epilogue, which does nothing but temperature-scale the
//! logits and draw a Gumbel-max sample. Whatever an algorithm inferlet costs
//! above this number is the algorithm.
//!
//! `stats` adds the two extra `[1]`-shaped f32 drains that every algorithm
//! inferlet carries for its self-verification metrics, without adding any
//! algorithm compute. Running with and without it separates the cost of the
//! extra host round-trip channels from the cost of the algorithm itself.

use inferlet::eta::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_seed")]
    seed: u32,
    /// Drain the two instrumentation channels the algorithm inferlets carry.
    #[serde(default)]
    stats: bool,
}

fn default_prompt() -> String {
    "Write a short paragraph about naive sampling.".into()
}

fn default_temperature() -> f32 {
    1.0
}

fn default_max_tokens() -> usize {
    32
}

fn default_seed() -> u32 {
    0x7ce1
}

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    text: String,
    /// The generated ids, so an identity gate can find WHERE two runs part.
    tokens: Vec<u32>,
    count: usize,
    stats: bool,
}

/// One sampling step: temperature, then a Gumbel-max draw over the full vocab.
fn step(logits: Tensor, temperature: f32, rng_state: &Tensor) -> Tensor {
    let scaled = if temperature == 1.0 {
        logits
    } else {
        &logits / temperature
    };
    gumbel_max(scaled, rng_state)
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if !input.temperature.is_finite() || input.temperature <= 0.0 {
        return Err("temperature must be finite and greater than 0".into());
    }
    let max_tokens = input.max_tokens;
    let temperature = input.temperature;
    let want_stats = input.stats;
    let ws = WorkingSet::new();
    let pipe = Pipeline::new();

    // The model's opening (`<bos>` where it has one) before the raw text: a
    // gemma without it answers noise, and no sampler can be read against that.
    let prompt: Vec<i32> = inferlet::chat::prefix()
        .into_iter()
        .chain(model::encode(&input.prompt))
        .map(|t| t as i32)
        .collect();
    let (&last, head) = prompt.split_last().ok_or("empty prompt")?;
    // The runtime cuts every program's prompt at the same rows (`PIE_FIRE_ROWS` pins them).
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
    let rng = Channel::from([input.seed, 0]);
    let tok_out = Channel::new([1], dtype::i32).capacity(channel_capacity() as u32);
    let s1_out = Channel::new([1], dtype::f32).capacity(channel_capacity() as u32);
    let s2_out = Channel::new([1], dtype::f32).capacity(channel_capacity() as u32);

    let fwd = ForwardPass::new();
    fwd.embed(&tok_in, &Channel::from([0u32, 1]))?;
    fwd.state(&[&ws], StateGeometry::folding())?;
    fwd.epilogue(move || {
        let r = rng.take();
        let logits = intrinsics::output(0, "logits");
        let token = step(logits, temperature, &r);
        let r_next = &r + iota(2);
        tok_in.put(&token);
        tok_out.put(&token);
        if want_stats {
            let mirror = reshape(cast(&token, dtype::f32), [1]);
            s1_out.put(&mirror);
            s2_out.put(&mirror);
        }
        rng.put(&r_next);
    });

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
    run_ahead(&pipe, &fwd, max_tokens, async || {
        let t = tok_out
            .take_host::<i32>()
            .await
            .with_context(|| format!("@{}", generated.len()))?;
        if want_stats {
            s1_out.take_host::<Vec<f32>>().await?;
            s2_out.take_host::<Vec<f32>>().await?;
        }
        generated.push(t as u32);
        Ok(ControlFlow::Continue(()))
    })
    .await?;
    pipe.close();

    Ok(Output {
        sampler: "naive-baseline",
        text: model::decode(&generated)?,
        count: generated.len(),
        tokens: generated,
        stats: want_stats,
    })
}
