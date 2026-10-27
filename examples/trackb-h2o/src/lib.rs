//! H2O — Heavy-Hitter Oracle (Zhang et al., 2023), <https://arxiv.org/abs/2306.14048>.
//!
//! Deviation: the paper re-ranks every step; here the keep-set is chosen once.

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
    /// KV pages H2O keeps. Positions written after the keep-set was chosen are
    /// kept on top of this, so the effective local window is never zero.
    #[serde(default = "default_page_budget")]
    page_budget: u32,
    #[serde(default)]
    enforce_tokens: Option<usize>,
    #[serde(default = "default_heads")]
    heads: u32,
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
fn default_page_budget() -> u32 {
    4
}
/// Qwen3.5-0.8B: 8 query heads.
fn default_heads() -> u32 {
    8
}

/// Four orders of magnitude below one layer-fire's 1.0, so one observation dominates the ramp.
const SEED_SCALE: f32 = 1e-4;

/// What a slot may carry and still be explained by the seed alone. The ramp is
/// bounded by `SEED_SCALE`; anything above this is attention mass.
const SEED_CEILING: f32 = SEED_SCALE * 1.001;

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    text: String,
    count: usize,
    kv_max: u32,
    page_size: u32,
    /// Live KV positions at the last observed step.
    kv_len: u32,
    page_budget: u32,
    live_pages: u32,
    layers_observed: u32,
    layers_per_fire: u32,
    /// Slots inside the live prefix that carry a finite, non-negative score.
    live_scored: usize,
    tail_polluted: usize,
    tail_max: f32,
    observed_live: usize,
    /// `(declared kv_len, observed live length)` for every drained fire.
    trace: Vec<(u32, usize)>,
    scores_nan: usize,
    score_mass: f32,
    mass_trace: Vec<f32>,
    kept_pages: Vec<u32>,
    /// The page H2O would evict next (lowest mass).
    evicted_first: Option<u32>,
    observe_fires: usize,
    enforce_tokens: usize,
    selected_pages: u32,
    served_pages: u32,
    served_kv: u32,
    full_kv: u32,
    /// Per-page mass at the last fire, live pages only.
    page_mass: Vec<String>,
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

    // As `naive-baseline` opens, since the all-keep arm is compared against it.
    let prompt: Vec<i32> = inferlet::chat::prefix()
        .into_iter()
        .chain(model::encode(&input.prompt))
        .map(|t| t as i32)
        .collect();
    let (&last, head) = prompt.split_last().ok_or("empty prompt")?;
    let n = prompt.len() as u32;
    let max_pages = (n + max_tokens as u32 + 1).div_ceil(page_size).max(1);
    let kv_max = max_pages * page_size;
    if kv_max > score_kv {
        return Err(format!(
            "prompt + max_tokens needs {kv_max} KV slots, past the model's \
             score ceiling of {score_kv}"
        ));
    }
    let p_max = max_pages;
    let page_budget = input.page_budget.min(p_max);

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

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut last_scores: Vec<f32> = Vec::new();
    let mut last_page_mass: Vec<f32> = Vec::new();
    let mut layers_observed = 0u32;
    let layers_per_fire = layers;
    let mut last_kv_len = 0;
    let mut trace: Vec<(u32, usize)> = Vec::new();
    let mut mass_trace: Vec<f32> = Vec::new();

    // A masked fire captures no scores, so the decode observes unmasked first, then enforces.
    let enforce_tokens = input
        .enforce_tokens
        .unwrap_or(max_tokens / 2)
        .min(max_tokens);
    let observe_fires = max_tokens - enforce_tokens;

    // ── OBSERVATION LOOP (1-wide, run-ahead), with the score tap. ──
    let tok_in = Channel::from([last]);
    let rng = Channel::from([input.seed, 0]);
    let tok_out = Channel::new([1], dtype::i32).capacity(channel_capacity() as u32);
    let lane1 = Channel::from([0u32, 1u32]);

    let seed: Vec<f32> = (0..kv_max)
        .map(|i| SEED_SCALE * (kv_max - i) as f32 / kv_max as f32)
        .collect();
    let acc = Channel::from(seed);

    let scores_out = Channel::new([kv_max], dtype::f32).capacity(channel_capacity() as u32);
    let page_mass_out = Channel::new([p_max], dtype::f32).capacity(channel_capacity() as u32);

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

        // `[planes, kv]` layer-major; dividing by `heads` averages heads and sums layers.
        let rect = reshape(intrinsics::output(0, "attn.scores"), [planes, score_kv]);
        let fire_row = &reduce_sum(transpose(&rect)) / heads_f;
        let prev = acc.take();
        let folded = &prev + &gather(&fire_row, iota(kv_max));
        scores_out.put(&folded);
        acc.put(&folded);

        // Exact: `kv_max = p_max * page_size`.
        page_mass_out.put(reduce_sum(reshape(&folded, [p_max, page_size])));
    });

    run_ahead(&pipe, &fwd, observe_fires, async || {
        let t = tok_out
            .take_host::<i32>()
            .await
            .with_context(|| format!("@{}", generated.len()))?;
        last_page_mass = page_mass_out
            .take_host::<Vec<f32>>()
            .await
            .with_context(|| format!("page mass @{}", generated.len()))?;
        last_scores = scores_out
            .take_host::<Vec<f32>>()
            .await
            .with_context(|| format!("@{}", generated.len()))?;
        // The prompt plus every token committed before this fire was live.
        last_kv_len = n + generated.len() as u32;
        layers_observed += layers_per_fire;
        mass_trace.push(last_scores.iter().filter(|s| s.is_finite()).sum());
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

    let sel_len = n - 1 + generated.len() as u32;
    let sel_pages = sel_len.div_ceil(page_size).min(p_max);
    let mut sel_order: Vec<u32> = (0..sel_pages).collect();
    let mass_of = |p: u32| last_page_mass.get(p as usize).copied().unwrap_or(0.0);
    sel_order.sort_by(|&a, &b| {
        mass_of(b)
            .partial_cmp(&mass_of(a))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let evicted_first = sel_order.last().copied();
    let mut kept: Vec<u32> = sel_order.into_iter().take(page_budget as usize).collect();
    kept.sort_unstable();

    let keep_set: std::collections::BTreeSet<u32> = kept.iter().copied().collect();
    let keep_row: Vec<bool> = (0..kv_max)
        .map(|j| j >= sel_len || keep_set.contains(&(j / page_size)))
        .collect();
    let served_kv = keep_row[..(sel_len as usize).min(keep_row.len())]
        .iter()
        .filter(|k| **k)
        .count() as u32;

    // `run_ahead` closed the first pipeline when it spent its budget. With
    // no observation the fill is still in flight there, so the enforcement
    // runs on it.
    let pipe = if observe_fires > 0 {
        Pipeline::new()
    } else {
        pipe
    };
    if enforce_tokens > 0 {
        let at = sel_len;
        let tok_in = Channel::from([generated.last().map_or(last, |&t| t as i32)]);
        // Resume the Gumbel stream: each observation fire advanced `[s, k]` to `[s, k + 1]`.
        let rng = Channel::from([input.seed, observe_fires as u32]);
        let tok_out = Channel::new([1], dtype::i32).capacity(channel_capacity() as u32);
        let lane1 = Channel::from([0u32, 1u32]);
        let fill = Channel::from([at + 1]);

        // A bound mask replaces the causal bound: the keep-set alone would admit unwritten slots.
        let keep = Channel::from_shaped([1, kv_max], keep_row.clone());
        let mask = Channel::from_shaped(
            [1, kv_max],
            keep_row
                .iter()
                .enumerate()
                .map(|(j, k)| *k && (j as u32) <= at)
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
            fill.put(&length + 1u32);
            tok_out.put(&token);
            rng.put(&(&r + iota(2)));

            let k = keep.take();
            mask.put(and(causal_mask(&length, kv_max), &k));
            keep.put(&k);
        });

        run_ahead(&pipe, &fwd, enforce_tokens, async || {
            let t = tok_out
                .take_host::<i32>()
                .await
                .with_context(|| format!("enforce @{}", generated.len()))?;
            generated.push(t as u32);
            Ok(ControlFlow::Continue(()))
        })
        .await?;
    }
    pipe.close();

    let live = (last_kv_len as usize).min(last_scores.len());
    let live_pages = last_kv_len.div_ceil(page_size).min(p_max);

    // Clamped to the row in hand: with no observation fire there is none.
    let masses: Vec<f32> = (0..live_pages as usize)
        .map(|p| {
            let lo = (p * page_size as usize).min(last_scores.len());
            let hi = ((p + 1) * page_size as usize).min(last_scores.len());
            last_scores[lo..hi].iter().sum()
        })
        .collect();
    Ok(Output {
        sampler: "trackb-h2o",
        text: model::decode(&generated)?,
        count: generated.len(),
        kv_max,
        page_size,
        kv_len: last_kv_len,
        page_budget,
        live_pages,
        layers_observed,
        layers_per_fire,
        live_scored: last_scores[..live]
            .iter()
            .filter(|s| s.is_finite() && **s >= 0.0)
            .count(),
        tail_polluted: last_scores[live..]
            .iter()
            .filter(|s| **s > SEED_CEILING)
            .count(),
        tail_max: last_scores[live..].iter().copied().fold(0.0f32, f32::max),
        observed_live: last_scores
            .iter()
            .rposition(|s| *s != 0.0)
            .map_or(0, |i| i + 1),
        scores_nan: last_scores.iter().filter(|s| s.is_nan()).count(),
        score_mass: last_scores[..live].iter().sum(),
        mass_trace,
        observe_fires,
        enforce_tokens,
        selected_pages: sel_pages,
        served_pages: kept.len() as u32,
        served_kv,
        full_kv: sel_len,
        kept_pages: kept,
        evicted_first,
        page_mass: masses.iter().map(|m| format!("{m:.5}")).collect(),
        trace,
        score_head: last_scores[..live.min(8)]
            .iter()
            .map(|s| format!("{s:.5}"))
            .collect(),
    })
}
