//! Top-a sampling — a truncation floor that scales with the *square* of the
//! peak probability.
//!
//! `keep(x)  iff  p(x) >= a · p_max²`.
//!
//! The quadratic term is the whole idea, and it is what separates top-a from
//! min-p's linear `a · p_max`. When the model is confident (`p_max` near 1) the
//! floor is high and the candidate set collapses toward greedy. When the model
//! is unsure (`p_max` small) the floor falls off *quadratically* — twice as
//! fast as min-p — so the set opens up much more aggressively exactly where the
//! model has no strong opinion.
//!
//! Community method with no canonical paper; this follows the reference
//! implementation in KoboldAI and text-generation-webui.
//!
//! ## Cost
//!
//! Unlike every other truncation sampler here, top-a needs no sort and no
//! candidate bound: one `ReduceMax`, one multiply, one comparison over the
//! vocabulary. It is `O(vocab)` with no `k_max` approximation anywhere, so the
//! mask it produces is exact.
//!
//! The most likely token always survives, because `p_max >= a · p_max²`
//! whenever `a · p_max <= 1`, which holds for every `a <= 1`.
//!
//! ## Source
//!
//! No canonical paper. Reference implementations: KoboldAI —
//! <https://github.com/KoboldAI/KoboldAI-Client> — and
//! oobabooga/text-generation-webui.
//!
//! Faithfulness: **Exact**.

use inferlet::eta::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_a")]
    a: f32,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_seed")]
    seed: u32,
}

fn default_prompt() -> String {
    "Write a short paragraph about top-a sampling.".into()
}

fn default_a() -> f32 {
    0.2
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
    count: usize,
    a: f32,
    /// Mean size of the retained set, in tokens. Should swing widely across a
    /// generation — that swing *is* the algorithm.
    mean_kept: f32,
    min_kept: u32,
    /// Mean probability mass retained.
    mean_mass: f32,
}

#[derive(Clone, Copy)]
struct Cfg {
    a: f32,
    temperature: f32,
}

/// The top-a keep-mask. Exact — no candidate bound, no sort.
///
/// Returns `(keep_mask, kept_count, kept_mass)`.
fn top_a_keep(logits: &Tensor, vocab: u32, a: f32) -> (Tensor, Tensor, Tensor) {
    let probs = softmax(logits);
    let p_max = reduce_max(&probs);
    let threshold = a * (&p_max * &p_max);
    let keep = ge(&probs, broadcast(&threshold, [vocab]));

    let zeros = broadcast(0.0f32, [vocab]);
    let kept_mass = reshape(reduce_sum(select(&keep, &probs, &zeros)), [1]);
    let kept = reshape(reduce_sum(cast(&keep, dtype::f32)), [1]);
    (keep, kept, kept_mass)
}

/// One sampling step: temperature, quadratic floor, Gumbel-max draw.
fn step(logits: Tensor, vocab: u32, cfg: Cfg, rng_state: &Tensor) -> (Tensor, Tensor, Tensor) {
    // Temperature first: `p_max` has to be measured on the distribution being
    // sampled from, or the floor tracks the wrong peak.
    let scaled = if cfg.temperature == 1.0 {
        logits
    } else {
        &logits / cfg.temperature
    };
    let (keep, kept, kept_mass) = top_a_keep(&scaled, vocab, cfg.a);
    let neg_inf = broadcast(f32::NEG_INFINITY, [vocab]);
    let masked = select(&keep, &scaled, &neg_inf);
    (gumbel_max(masked, rng_state), kept, kept_mass)
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if !input.a.is_finite() || input.a <= 0.0 || input.a > 1.0 {
        return Err("a must be finite and in (0, 1]".into());
    }
    if !input.temperature.is_finite() || input.temperature <= 0.0 {
        return Err("temperature must be finite and greater than 0".into());
    }
    let max_tokens = input.max_tokens;
    let vocab = model::output_vocab_size();
    let cfg = Cfg {
        a: input.a,
        temperature: input.temperature,
    };
    let ws = WorkingSet::new();
    let pipe = Pipeline::new();

    // The model's opening (`<bos>` where it has one) before the raw text: a
    // gemma without it answers noise.
    let prompt: Vec<i32> = inferlet::chat::prefix()
        .into_iter()
        .chain(model::encode(&input.prompt))
        .map(|t| t as i32)
        .collect();
    let (&last, head) = prompt.split_last().ok_or("empty prompt")?;
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
        let (token, a, b) = step(logits, vocab, cfg, &r);
        let r_next = &r + iota(2);
        tok_in.put(&token);
        tok_out.put(&token);
        s1_out.put(&a);
        s2_out.put(&b);
        rng.put(&r_next);
    });

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut s1: Vec<f32> = Vec::with_capacity(max_tokens);
    let mut s2: Vec<f32> = Vec::with_capacity(max_tokens);
    run_ahead(&pipe, &fwd, max_tokens, async || {
        let t = tok_out
            .take_host::<i32>()
            .await
            .with_context(|| format!("@{}", generated.len()))?;
        let a = s1_out.take_host::<f32>().await?;
        let b = s2_out.take_host::<f32>().await?;
        generated.push(t as u32);
        s1.push(a);
        s2.push(b);
        Ok(ControlFlow::Continue(()))
    })
    .await?;
    pipe.close();

    let mean = |v: &[f32]| v.iter().fold(0.0, |a, b| a + b) / v.len().max(1) as f32;
    let min_s1 = s1.iter().copied().reduce(f32::min).unwrap_or(0.0).max(0.0);
    if !s1.is_empty() && min_s1 == 0.0 {
        return Err("top-a keep-set was empty — the peak token was masked out".into());
    }

    Ok(Output {
        sampler: "top-a",
        text: model::decode(&generated)?,
        count: generated.len(),
        a: cfg.a,
        mean_kept: mean(&s1),
        min_kept: min_s1 as u32,
        mean_mass: mean(&s2),
    })
}
