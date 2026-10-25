//! SnapKV, executed: the prefill's attention picks a smaller KV cache the answer is served.

use inferlet::eta::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    /// The fact hidden in the haystack. Kept short and lexically unlike the
    /// filler, so "did the answer survive" is a substring check and not a
    /// judgement call.
    #[serde(default = "default_needle")]
    needle: String,
    /// The question whose answer is the needle.
    #[serde(default = "default_question")]
    question: String,
    /// Six: about the most Qwen3.5-0.8B recalls on the full cache, so the control arm answers.
    #[serde(default = "default_filler")]
    filler: usize,
    /// `0.0` front, `1.0` back; the default is a middle page, away from the two always kept.
    #[serde(default = "default_depth")]
    depth: f32,
    #[serde(default = "default_page_budget")]
    page_budget: u32,
    /// `false` runs the control arm: the same prompt, the same sampling, the
    /// whole prompt served. The gate compares the two.
    #[serde(default = "default_evict")]
    evict: bool,
    #[serde(default = "default_layers")]
    layers: u32,
    #[serde(default = "default_heads")]
    heads: u32,
}

fn default_needle() -> String {
    "Remember this: the secret code word is banana.".to_string()
}
fn default_question() -> String {
    " What is the secret code word? The secret code word is".to_string()
}
fn default_filler() -> usize {
    6
}
fn default_depth() -> f32 {
    0.6
}
fn default_page_budget() -> u32 {
    5
}
fn default_evict() -> bool {
    true
}
/// Qwen3.5-0.8B: 6 of its 24 layers are attention layers.
fn default_layers() -> u32 {
    6
}
/// Qwen3.5-0.8B: 8 query heads.
fn default_heads() -> u32 {
    8
}

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    /// The continuation, greedily decoded.
    text: String,
    /// Whether eviction ran at all — `false` is the control arm.
    evicted: bool,
    /// Prompt tokens.
    prompt_len: u32,
    /// KV pages the PREFILL was served — the whole prompt.
    prompt_pages: u32,
    served_pages: u32,
    /// Which prompt pages survived, ascending.
    kept_pages: Vec<u32>,
    served_kv: u32,
    full_kv: u32,
    /// Which page the needle's tokens landed in, computed host-side from the
    /// prompt's own token offsets. The gate checks it survived the cut.
    needle_pages: Vec<u32>,
    /// Per-page attention mass from the observation window, device-folded.
    page_mass: Vec<String>,
    /// Attention layers folded into the row (`planes / heads`, declared).
    layers_observed: u32,
    /// Total mass over the live prefix — one distribution per layer, so this
    /// must come out at `layers_observed`.
    score_mass: f32,
    count: usize,
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if input.layers == 0 || input.heads == 0 {
        return Err("layers and heads must both be at least 1".into());
    }
    if !(0.0..=1.0).contains(&input.depth) {
        return Err("depth must be between 0.0 and 1.0".into());
    }
    let layers = input.layers;
    let heads = input.heads;
    let Some(planes) = layers.checked_mul(heads) else {
        return Err("layers * heads overflows the plane count".into());
    };
    let heads_f = heads as f32;

    // Varied: one repeated sentence makes a small model predict the loop, not the content.
    const FILLER: [&str; 6] = [
        "The weather in the valley is mild and the roads are clear. ",
        "Deliveries arrive on Tuesdays and the depot closes at six. ",
        "The library keeps its old maps in the east reading room. ",
        "Rain is expected later in the week across the northern hills. ",
        "The bakery on the corner opens early and sells out by noon. ",
        "Two footbridges cross the river between the mill and the park. ",
    ];
    let at = ((input.filler as f32) * input.depth).round() as usize;
    let at = at.min(input.filler);
    let mut haystack = String::new();
    for i in 0..at {
        haystack.push_str(FILLER[i % FILLER.len()]);
    }
    let needle_at_chars = haystack.len();
    haystack.push_str(&input.needle);
    haystack.push(' ');
    for i in at..input.filler {
        haystack.push_str(FILLER[i % FILLER.len()]);
    }

    let before_needle = model::encode(&haystack[..needle_at_chars]);
    let through_needle = model::encode(&haystack[..needle_at_chars + input.needle.len()]);
    // The question ends the prompt, so it is the observation window.
    let question = model::encode(&input.question);
    let q = question.len().max(1) as u32;
    let mut prompt = model::encode(&haystack);
    prompt.extend(question.iter().copied());
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    // Where the question's rows begin — the rows the answering fire re-reads
    // through the cut.
    let ask_at = n - q;

    let ws = WorkingSet::new();
    let page_size = kv_page_size();
    // The prompt, the answer's slot and one past it.
    let max_pages = (n + 2).div_ceil(page_size);
    let kv_max = max_pages * page_size;
    if kv_max > intrinsics::attn_score_kv_max() {
        return Err(format!(
            "this haystack needs {kv_max} KV slots, past the published attn_score \
             ceiling of {}",
            intrinsics::attn_score_kv_max()
        )
        .into());
    }
    let p_max = max_pages;
    let last_prompt_page = (n - 1) / page_size;
    let prompt_pages = last_prompt_page + 1;
    let needle_lo = before_needle.len() as u32;
    let needle_hi = (through_needle.len() as u32).max(needle_lo + 1).min(n);
    let needle_pages: Vec<u32> = (needle_lo / page_size..=(needle_hi - 1) / page_size).collect();

    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let pipe = Pipeline::new();

    // Freezes the haystack but buffers the question's rows: the answer below writes them again.
    let toks_p = Channel::from(&prompt_i32[..]).named("toks_p");
    let embed_indptr_p = Channel::from([0u32, n]).named("embed_indptr_p");
    let fold_p = Channel::from([ask_at]).named("fold_p");
    let tok_out_p = Channel::new([1], dtype::i32).named("tok_out_p");
    let page_mass_out = Channel::new([p_max], dtype::f32).named("evict_page_mass");
    let total_out = Channel::new([1], dtype::f32).named("evict_total");

    let fwd_p = ForwardPass::new();
    fwd_p.embed(&toks_p, &embed_indptr_p)?;
    fwd_p.state(
        &[&ws],
        StateGeometry {
            fold: Fold::Count(&fold_p),
            ..StateGeometry::default()
        },
    )?;
    fwd_p.epilogue(move || {
        let logits = intrinsics::logits();
        tok_out_p.put(&reduce_argmax(logits));

        // `[planes, kv]` layer-major; dividing by `heads` averages heads and sums layers.
        let rect = intrinsics::attn_score(planes);
        let folded = gather(&(&reduce_sum(&transpose(&rect)) / heads_f), iota(kv_max));
        // Exact: `kv_max = p_max * page_size`.
        let per_page = reduce_sum(&reshape(&folded, [p_max, page_size]));
        page_mass_out.put(&per_page);
        total_out.put(&reshape(&reduce_sum(&per_page), [1]));
    });
    fwd_p.submit(&pipe).context("prefill submit")?;

    let _g0 = tok_out_p.take_host::<i32>().await?;
    let masses = page_mass_out.take_host::<Vec<f32>>().await?;
    let score_mass = total_out.take_host::<f32>().await?;

    let mut kept: Vec<u32> = if input.evict {
        let budget = input.page_budget.max(2).min(prompt_pages) as usize;
        let mut forced = vec![0u32, last_prompt_page];
        forced.sort_unstable();
        forced.dedup();
        let mut ranked: Vec<u32> = (0..prompt_pages)
            .filter(|page| !forced.contains(page))
            .collect();
        ranked.sort_by(|a, b| {
            let (ma, mb) = (
                masses.get(*a as usize).copied().unwrap_or(0.0),
                masses.get(*b as usize).copied().unwrap_or(0.0),
            );
            mb.partial_cmp(&ma)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(b))
        });
        let room = budget.saturating_sub(forced.len());
        let mut kept = forced;
        kept.extend(ranked.into_iter().take(room));
        kept.sort_unstable();
        kept
    } else {
        (0..prompt_pages).collect()
    };
    kept.dedup();
    let served_pages = kept.len() as u32;

    let keep_set: std::collections::BTreeSet<u32> = kept.iter().copied().collect();
    let keep_row: Vec<bool> = (0..kv_max)
        .map(|j| j >= n || keep_set.contains(&(j / page_size)))
        .collect();
    let served_kv = keep_row[..n as usize].iter().filter(|keep| **keep).count() as u32;

    let question_i32: Vec<i32> = question.iter().map(|&t| t as i32).collect();
    let keep = &keep_row;
    let rows: Vec<bool> = (0..q)
        .flat_map(|r| {
            let at = ask_at + r;
            (0..kv_max).map(move |j| {
                // A bound mask replaces the causal bound, so causality is restated here.
                j <= at && (j >= ask_at || keep[j as usize])
            })
        })
        .collect();

    let toks_q = Channel::from(&question_i32[..]).named("toks_q");
    let embed_indptr_q = Channel::from([0u32, q]).named("embed_indptr_q");
    let slots_q = Channel::from_iter(ask_at..n).named("slots_q");
    let mask_q = Channel::from_shaped([q, kv_max], rows).named("mask_q");
    let tok_out_q = Channel::new([1], dtype::i32).named("tok_out_q");

    let fwd_q = ForwardPass::new();
    fwd_q.embed(&toks_q, &embed_indptr_q)?;
    fwd_q.mask(&mask_q)?;
    fwd_q.state(
        &[&ws],
        StateGeometry {
            slots: Some(&slots_q),
            ..StateGeometry::folding()
        },
    )?;
    fwd_q.epilogue(move || {
        let logits = intrinsics::logits();
        tok_out_q.put(&reduce_argmax(logits));
    });
    fwd_q.submit(&pipe).context("answer submit")?;
    let answer = tok_out_q.take_host::<i32>().await.context("answer take")?;
    let generated: Vec<u32> = vec![answer as u32];

    pipe.close();

    Ok(Output {
        sampler: "snapkv-eviction",
        text: model::decode(&generated)?,
        evicted: input.evict,
        prompt_len: n,
        prompt_pages,
        served_pages,
        kept_pages: kept,
        served_kv,
        full_kv: n,
        needle_pages,
        page_mass: masses[..prompt_pages as usize]
            .iter()
            .map(|m| format!("{m:.5}"))
            .collect(),
        layers_observed: layers,
        score_mass,
        count: generated.len(),
    })
}
