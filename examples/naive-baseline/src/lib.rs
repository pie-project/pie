//! Naive text completion — the performance control for the algorithm inferlets.
//!
//! Structurally identical to every truncation/penalty inferlet in this
//! directory: one N-wide prefill fire, then a device-carried decode loop
//! driven by `eta::run_ahead`, which keeps the runtime's run-ahead window
//! (`model.channel-capacity()`) full ahead of the host drain. The only
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
    /// STRUCTURAL v0: run only the first k transformer layers (the
    /// layerskip-draft / logit-lens class). None = the full model.
    #[serde(default)]
    max_layers: Option<u32>,
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

    if max_tokens == 0 {
        return Ok(Output {
            sampler: "naive-baseline",
            text: String::new(),
            tokens: Vec::new(),
            count: 0,
            stats: want_stats,
        });
    }

    // The model's opening (`<bos>` where it has one) before the raw text: a
    // gemma without it answers noise, and no sampler can be read against that.
    let mut prompt = inferlet::chat::prefix();
    prompt.extend(model::encode(&input.prompt));
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);

    // The runtime cuts every program's prompt at the same rows (`PIE_FIRE_ROWS` pins them).
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let pipe = Pipeline::new();

    let toks_p = Channel::from(&prompt_i32[..]).named("toks_p");
    let embed_indptr_p = Channel::from([0u32, n]).named("embed_indptr_p");
    let rng_p = Channel::from([input.seed, 0]).named("rng_p");
    let tok_out_p = Channel::new([1], dtype::i32).named("tok_out_p");
    let s1_out_p = Channel::new([1], dtype::f32).named("s1_out_p");
    let s2_out_p = Channel::new([1], dtype::f32).named("s2_out_p");

    let fwd_p = ForwardPass::new();
    if let Some(k) = input.max_layers {
        fwd_p.set_max_layers(k)?;
    }
    fwd_p.embed(&toks_p, &embed_indptr_p)?;
    fwd_p.state(&[&ws], StateGeometry::folding())?;
    fwd_p.epilogue(move || {
        let r = rng_p.take();
        let logits = intrinsics::logits();
        let token = step(logits, temperature, &r);
        let r_next = &r + iota(2);
        tok_out_p.put(&token);
        if want_stats {
            let mirror = reshape(cast(&token, dtype::f32), [1]);
            s1_out_p.put(&mirror);
            s2_out_p.put(&mirror);
        }
        rng_p.put(&r_next);
    });

    fwd_p.submit(&pipe).context("prefill submit")?;

    let g0 = tok_out_p
        .take_host::<i32>()
        .await
        .context("prefill drain")?;
    if want_stats {
        s1_out_p
            .take_host::<Vec<f32>>()
            .await
            .context("prefill drain")?;
        s2_out_p
            .take_host::<Vec<f32>>()
            .await
            .context("prefill drain")?;
    }
    generated.push(g0 as u32);

    // ── DECODE LOOP (1-wide, run-ahead). ──
    if generated.len() < max_tokens {
        let tok_in = Channel::from([g0]).named("tok_in");
        let rng = Channel::from([input.seed ^ 0x5bd1, 0]).named("rng");
        let tok_out = Channel::new([1], dtype::i32)
            .capacity(channel_capacity() as u32)
            .named("tok_out");
        let s1_out = Channel::new([1], dtype::f32)
            .capacity(channel_capacity() as u32)
            .named("s1_out");
        let s2_out = Channel::new([1], dtype::f32)
            .capacity(channel_capacity() as u32)
            .named("s2_out");
        let lane1 = Channel::from([0u32, 1u32]).named("embed_indptr");

        let fwd = ForwardPass::new();
        if let Some(k) = input.max_layers {
            fwd.set_max_layers(k)?;
        }
        fwd.embed(&tok_in, &lane1)?;
        fwd.state(&[&ws], StateGeometry::folding())?;
        fwd.epilogue(move || {
            let r = rng.take();
            let logits = intrinsics::logits();
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

        let budget = max_tokens - 1;
        run_ahead(&pipe, &fwd, budget, async || {
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
    }
    pipe.close();

    Ok(Output {
        sampler: "naive-baseline",
        text: model::decode(&generated)?,
        count: generated.len(),
        tokens: generated,
        stats: want_stats,
    })
}
