//! SnapKV — prompt compression from the observation window (Li et al., 2024),
//! <https://arxiv.org/abs/2404.14469>.
//!
//! Deviations: heads averaged, layers summed, and the page pooled rather than a stride-1 max-pool.

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
    /// Prompt KV pages SnapKV keeps. Positions written after prefill are kept
    /// on top of this, unconditionally — including the tail of the last prompt
    /// page, which is where the decode writes.
    #[serde(default = "default_page_budget")]
    page_budget: u32,
    #[serde(default = "default_heads")]
    heads: u32,
    #[serde(default)]
    prompt_tokens: Option<Vec<u32>>,
    #[serde(default)]
    stream: bool,
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
    0x5A4Bu32
}
fn default_page_budget() -> u32 {
    4
}
/// Qwen3.5-0.8B: 8 query heads.
fn default_heads() -> u32 {
    8
}

/// Tie-break ramp: equal-mass pages go to the earlier one, the StreamingLLM prior.
const SEED_SCALE: f32 = 1e-4;

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    text: String,
    count: usize,
    kv_max: u32,
    /// Prompt length in tokens, i.e. the KV length the observed fire had.
    prompt_len: u32,
    page_budget: u32,
    /// Pages the prompt occupies. Only these compete for the budget.
    prompt_pages: u32,
    page_size: u32,
    layers_observed: u32,
    /// Slots inside the prompt that carry a finite, non-negative score.
    live_scored: usize,
    tail_nonzero: usize,
    observed_live: usize,
    scores_nan: usize,
    /// `Σ score` over the prompt: one distribution per layer, so it equals
    /// `layers_observed`.
    score_mass: f32,
    kept_pages: Vec<u32>,
    /// The prompt page SnapKV would drop first (lowest mass).
    evicted_first: Option<u32>,
    served_pages: u32,
    served_kv: u32,
    full_kv: u32,
    /// Per-page mass over the prompt.
    page_mass: Vec<String>,
    tail_page_share: f32,
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
    if input.page_budget == 0 {
        return Err("page_budget must be at least 1".into());
    }
    if input.heads == 0 {
        return Err("heads must be at least 1".into());
    }
    let scores = model::lane_of(&[])?
        .outputs
        .into_iter()
        .find(|output| output.name == "attn.scores")
        .ok_or("this model reads out no attention scores")?;
    let layers = scores.layers.max(1);
    let heads = input.heads;
    let score_kv = scores.width / heads;
    if score_kv * heads != scores.width {
        return Err(format!(
            "the {}-wide score row does not split into {heads} heads",
            scores.width
        ));
    }
    let Some(planes) = layers.checked_mul(heads) else {
        return Err("layers * heads overflows the plane count".into());
    };
    let heads_f = heads as f32;
    let max_tokens = input.max_tokens;
    let temperature = input.temperature;
    let ws = WorkingSet::new();
    let page_size = kv_page_size();

    if max_tokens == 0 {
        return Err("max_tokens must be at least 1".into());
    }

    // As `naive-baseline` opens, since the all-keep arm is compared against it.
    let mut prompt = if let Some(toks) = input.prompt_tokens {
        toks
    } else {
        let mut p = inferlet::chat::prefix();
        p.extend(model::encode(&input.prompt));
        p
    };
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let stream = input.stream;
    let max_pages = (n + max_tokens as u32 + 1).div_ceil(page_size).max(1);
    let kv_max = max_pages * page_size;
    if kv_max > score_kv {
        return Err(format!(
            "prompt + max_tokens needs {kv_max} KV slots, past the model's \
             score ceiling of {score_kv}"
        ));
    }
    let p_max = max_pages;
    let prompt_pages = n.div_ceil(page_size).min(p_max);
    let page_budget = input.page_budget.min(prompt_pages);

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);

    // The runtime keeps the observation window's rows in the last fire, the one that captures.
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let pipe = Pipeline::new();

    let toks_p = Channel::from(&prompt_i32[..]);
    let embed_indptr_p = Channel::from([0u32, n]);
    // `naive-baseline`'s salt, so the all-keep arm differs from it only by the mask.
    let rng_p = Channel::from([input.seed ^ 0x5bd1, 0]);
    let tok_out_p = Channel::new([1], dtype::i32);

    let scores_out = Channel::new([kv_max], dtype::f32);
    let page_mass_out = Channel::new([p_max], dtype::f32);

    let fwd_p = ForwardPass::new();
    fwd_p.embed(&toks_p, &embed_indptr_p)?;
    fwd_p.state(&[&ws], StateGeometry::folding())?;

    fwd_p.epilogue(move || {
        let r = rng_p.take();
        let logits = intrinsics::output(0, "logits");
        let token = step(logits, temperature, &r);
        tok_out_p.put(&token);
        rng_p.put(&(&r + iota(2)));

        // `[planes, kv]` layer-major; dividing by `heads` averages heads and sums layers.
        let rect = reshape(intrinsics::output(0, "attn.scores"), [planes, score_kv]);
        let folded = gather(&(&reduce_sum(transpose(&rect)) / heads_f), iota(kv_max));
        scores_out.put(&folded);

        // Exact: `kv_max = p_max * page_size`.
        page_mass_out.put(reduce_sum(reshape(&folded, [p_max, page_size])));
    });

    fwd_p.submit(&pipe).context("prefill submit")?;

    let g0 = tok_out_p.take_host::<i32>().await?;
    if stream {
        inferlet::session::send("t:1");
    }
    generated.push(g0 as u32);

    let prefill_scores = scores_out.take_host::<Vec<f32>>().await?;
    let layers_observed = layers;
    let device_page_mass = page_mass_out.take_host::<Vec<f32>>().await?;

    let ranked = |p: u32| -> f32 {
        device_page_mass.get(p as usize).copied().unwrap_or(0.0)
            + SEED_SCALE * (prompt_pages - p) as f32 / prompt_pages.max(1) as f32
    };
    let mut order: Vec<u32> = (0..prompt_pages).collect();
    order.sort_by(|&a, &b| {
        ranked(b)
            .partial_cmp(&ranked(a))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let evicted_first = order.last().copied();
    let mut kept: Vec<u32> = order.into_iter().take(page_budget as usize).collect();
    kept.sort_unstable();

    let keep_set: std::collections::BTreeSet<u32> = kept.iter().copied().collect();
    let keep_row: Vec<bool> = (0..kv_max)
        .map(|j| j >= n || keep_set.contains(&(j / page_size)))
        .collect();
    let served_kv = keep_row[..n as usize].iter().filter(|k| **k).count() as u32;

    // ── DECODE LOOP (1-wide, run-ahead), enforcing the fixed keep-set. ──
    if generated.len() < max_tokens {
        let tok_in = Channel::from([g0]);
        let rng = Channel::from([input.seed ^ 0x5bd1, 2]);
        let tok_out = Channel::new([1], dtype::i32).capacity(channel_capacity() as u32);
        let lane1 = Channel::from([0u32, 1u32]);
        let fill = Channel::from([n + 1]);

        // A bound mask replaces the causal bound: the keep-set alone would admit unwritten slots.
        let keep = Channel::from_shaped([1, kv_max], keep_row.clone());
        let mask = Channel::from_shaped(
            [1, kv_max],
            keep_row
                .iter()
                .enumerate()
                .map(|(j, k)| *k && (j as u32) <= n)
                .collect::<Vec<bool>>(),
        );

        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, &lane1)?;
        fwd.mask(&mask)?;
        fwd.state(&[&ws], StateGeometry::folding())?;

        fwd.epilogue(move || {
            let length = fill.take();
            let r = rng.take();
            let logits = intrinsics::output(0, "logits");
            let token = step(logits, temperature, &r);

            tok_in.put(&token);
            fill.put(&(&length + 1u32));
            tok_out.put(&token);
            rng.put(&(&r + iota(2)));

            // A loop-carried channel is `take`n before it is `put`.
            let k = keep.take();
            mask.put(and(causal_mask(&length, kv_max), &k));
            keep.put(&k);
        });

        let budget_n = max_tokens - 1;
        run_ahead(&pipe, &fwd, budget_n, async || {
            let t = tok_out
                .take_host::<i32>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            if stream {
                inferlet::session::send("t:1");
            }
            generated.push(t as u32);
            Ok(ControlFlow::Continue(()))
        })
        .await?;
    }
    pipe.close();

    let live = (n as usize).min(prefill_scores.len());

    let masses: Vec<f32> = (0..prompt_pages as usize)
        .map(|p| {
            let lo = p * page_size as usize;
            let hi = ((p + 1) * page_size as usize).min(prefill_scores.len());
            if lo >= prefill_scores.len() {
                0.0
            } else {
                prefill_scores[lo..hi].iter().sum()
            }
        })
        .collect();
    let total_mass: f32 = masses.iter().sum();
    let tail_page_share = if total_mass > 0.0 && prompt_pages > 0 {
        masses[(prompt_pages - 1) as usize] / total_mass
    } else {
        0.0
    };

    Ok(Output {
        sampler: "trackb-snapkv",
        text: model::decode(&generated)?,
        count: generated.len(),
        kv_max,
        prompt_len: n,
        page_budget,
        prompt_pages,
        page_size,
        layers_observed,
        live_scored: prefill_scores[..live]
            .iter()
            .filter(|s| s.is_finite() && **s >= 0.0)
            .count(),
        tail_nonzero: prefill_scores[live..].iter().filter(|s| **s != 0.0).count(),
        observed_live: prefill_scores
            .iter()
            .rposition(|s| *s != 0.0)
            .map_or(0, |i| i + 1),
        scores_nan: prefill_scores.iter().filter(|s| s.is_nan()).count(),
        score_mass: prefill_scores[..live].iter().sum(),
        served_pages: kept.len() as u32,
        served_kv,
        full_kv: n,
        kept_pages: kept,
        evicted_first,
        page_mass: masses.iter().map(|m| format!("{m:.5}")).collect(),
        tail_page_share,
        score_head: prefill_scores[..live.min(8)]
            .iter()
            .map(|s| format!("{s:.5}"))
            .collect(),
    })
}
