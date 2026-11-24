//! TOVA — Token Omission Via Attention (Oren et al., 2024), <https://arxiv.org/abs/2401.06104>.
//!
//! Observes only: nothing is masked, so given the same prompt, temperature,
//! seed and `max_tokens` the text must match `naive-baseline`'s — it draws
//! from `naive-baseline`'s streams, `seed` for the prefill and `seed ^ 0x5bd1`
//! for the decode.

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
    /// KV positions TOVA would keep.
    #[serde(default = "default_cache_size")]
    cache_size: u32,
}

fn default_prompt() -> String {
    "The capital of France is".to_string()
}
fn default_temperature() -> f32 {
    1.0
}
fn default_max_tokens() -> usize {
    32
}
fn default_seed() -> u32 {
    0x70Au32
}
fn default_cache_size() -> u32 {
    16
}

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    text: String,
    count: usize,
    kv_max: u32,
    /// Live KV positions at the last observed step.
    kv_len: u32,
    cache_size: u32,
    /// Slots inside the live prefix that carry a finite, non-negative score.
    live_scored: usize,
    /// Slots past the live prefix. Every one of them must be exactly zero.
    tail_nonzero: usize,
    observed_live: usize,
    /// `(declared kv_len, observed live length)` for every drained fire.
    trace: Vec<(u32, usize)>,
    scores_nan: usize,
    score_mass: f32,
    /// TOVA's keep-set: the `cache_size` highest-scoring live positions.
    kept_positions: Vec<u32>,
    /// The single position TOVA would evict next (lowest score).
    evicted_first: Option<u32>,
    score_head: Vec<String>,
}

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
    if input.cache_size == 0 {
        return Err("cache_size must be at least 1".into());
    }
    let scores = model::lane_of(&[])?
        .outputs
        .into_iter()
        .find(|output| output.name == "attn.scores")
        .ok_or("this model reads out no attention scores")?;
    // One row a lane: the attention mass a key of the load's context,
    // summed over the layers and averaged over the query heads; the guest
    // folds nothing.
    let score_kv = scores.width;
    let max_tokens = input.max_tokens;
    let temperature = input.temperature;
    let ws = WorkingSet::new();
    let page_size = kv_page_size();

    if max_tokens == 0 {
        return Err("max_tokens must be at least 1".into());
    }

    // As `naive-baseline` opens, since the text is compared against it.
    let mut prompt = inferlet::chat::prefix();
    prompt.extend(model::encode(&input.prompt));
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let max_pages = (n + max_tokens as u32 + 1).div_ceil(page_size).max(1);
    let kv_max = max_pages * page_size;
    if kv_max > score_kv {
        return Err(format!(
            "prompt + max_tokens needs {kv_max} KV slots, past the model's \
             score ceiling of {score_kv}"
        ));
    }
    let cache_size = input.cache_size.min(kv_max);

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);

    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let pipe = Pipeline::new();

    let toks_p = Channel::from(&prompt_i32[..]);
    let embed_indptr_p = Channel::from([0u32, n]);
    // `naive-baseline`'s streams, since the text is compared against it.
    let rng_p = Channel::from([input.seed, 0]);
    let tok_out_p = Channel::new([1], dtype::i32);

    let fwd_p = ForwardPass::new();
    fwd_p.embed(&toks_p, &embed_indptr_p)?;
    fwd_p.state(&[&ws], StateGeometry::folding())?;
    fwd_p.epilogue(move || {
        let r = rng_p.take();
        let logits = intrinsics::output(0, "logits");
        let token = step(logits, temperature, &r);
        tok_out_p.put(&token);
        rng_p.put(&(&r + iota(2)));
    });

    fwd_p.submit(&pipe).context("prefill submit")?;

    let g0 = tok_out_p
        .take_host::<i32>()
        .await
        .context("prefill drain")?;
    generated.push(g0 as u32);

    let mut last_scores: Vec<f32> = Vec::new();
    let mut last_kv_len = n + 1;
    let mut trace: Vec<(u32, usize)> = Vec::new();

    // ── DECODE LOOP (1-wide, run-ahead), with the TOVA tap. ──
    if generated.len() < max_tokens {
        let tok_in = Channel::from([g0]);
        let rng = Channel::from([input.seed ^ 0x5bd1, 0]);
        let tok_out = Channel::new([1], dtype::i32).capacity(channel_capacity() as u32);
        let lane1 = Channel::from([0u32, 1u32]);
        let scores_out = Channel::new([kv_max], dtype::f32).capacity(channel_capacity() as u32);

        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, &lane1)?;
        fwd.state(&[&ws], StateGeometry::folding())?;

        fwd.epilogue(move || {
            let r = rng.take();
            let logits = intrinsics::output(0, "logits");
            let token = step(logits, temperature, &r);

            tok_in.put(&token);
            tok_out.put(&token);
            rng.put(&(&r + iota(2)));

            let folded = reshape(intrinsics::output(0, "attn.scores"), [score_kv]);
            scores_out.put(gather(&folded, iota(kv_max)));
        });

        let budget_n = max_tokens - 1;
        run_ahead(&pipe, &fwd, budget_n, async || {
            let t = tok_out
                .take_host::<i32>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            last_scores = scores_out
                .take_host::<Vec<f32>>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            // The prompt plus every token committed before this fire was live.
            last_kv_len = n + generated.len() as u32;
            trace.push((
                last_kv_len,
                last_scores
                    .iter()
                    .rposition(|s| *s != 0.0)
                    .map_or(0, |i| i + 1),
            ));
            generated.push(t as u32);
            Ok(ControlFlow::Continue(()))
        })
        .await?;
    }
    pipe.close();

    let live = (last_kv_len as usize).min(last_scores.len());
    // Ties go to the newer position, the paper's recency tie-break.
    let mut order: Vec<u32> = (0..live as u32).collect();
    order.sort_by(|&a, &b| {
        last_scores[b as usize]
            .partial_cmp(&last_scores[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.cmp(&a))
    });
    let evicted_first = order.last().copied();
    let mut kept: Vec<u32> = order.into_iter().take(cache_size as usize).collect();
    kept.sort_unstable();

    Ok(Output {
        sampler: "tova-attention",
        text: model::decode(&generated)?,
        count: generated.len(),
        kv_max,
        kv_len: last_kv_len,
        cache_size,
        live_scored: last_scores[..live]
            .iter()
            .filter(|s| s.is_finite() && **s >= 0.0)
            .count(),
        tail_nonzero: last_scores[live..].iter().filter(|s| **s != 0.0).count(),
        observed_live: last_scores
            .iter()
            .rposition(|s| *s != 0.0)
            .map_or(0, |i| i + 1),
        scores_nan: last_scores.iter().filter(|s| s.is_nan()).count(),
        score_mass: last_scores[..live].iter().sum(),
        kept_positions: kept,
        evicted_first,
        trace,
        score_head: last_scores[..live.min(8)]
            .iter()
            .map(|s| format!("{s:.5}"))
            .collect(),
    })
}
