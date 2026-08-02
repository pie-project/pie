//! TOVA — Token Omission Via Attention (Oren et al., 2024).
//!
//! <https://arxiv.org/abs/2401.06104>
//!
//! TOVA is the simplest possible KV-eviction policy, and its simplicity is the
//! point: at every decoding step, keep the `cache_size` KV positions that the
//! *current* query attended to most, and drop the rest. No accumulated history
//! (that is H2O), no observation window (that is SnapKV) — just the attention
//! distribution of the most recent token.
//!
//! This inferlet is the **observability half** of that policy, in the same
//! sense as `quest-attention`: it runs TOVA's exact decision quantity on real
//! hardware, per layer, against the live KV cache, and drains the resulting
//! scores back to the host so the keep-set can be checked against a reference.
//! It does not yet mask the evicted positions out of the attention kernel, so
//! it produces *bit-identical output to `naive-baseline`* — which is exactly
//! what makes it testable: any divergence is a bug in the tap, not a property
//! of the algorithm.
//!
//! ## What the backend does under `intrinsics::attn_score`
//!
//! The call lowers to `Op::IntrinsicVal{intr: AttnScore}` at the `on_attn`
//! stage. The CUDA driver serves it only when the model reports the
//! `has_attn_score` capability (llama-like family, no sliding window, native
//! bf16 pages, `tp == 1`, and a decode fire on the plain paged-decode path).
//! The scores are the softmax probabilities the decode kernel itself computed —
//! they are captured inside the attention kernel via a FlashInfer attention
//! variant, not recomputed — so they cannot drift from the attention the model
//! actually performed.
//!
//! ## Two documented deviations from the paper
//!
//! 1. **Heads are folded by the backend.** TOVA ranks per head. The paged KV
//!    layout here carries one page list per request, so eviction is inherently
//!    a per-request decision and a per-head keep-set has no representable
//!    consumer. The backend therefore returns the mean over query heads, which
//!    is the union-friendly collapse: a position that any head needs keeps a
//!    non-trivial share of the mass. `quest-attention` documents the identical
//!    collapse for the identical reason.
//!
//! 2. **Layers are folded by the program.** TOVA maintains a separate cache per
//!    layer. One page list per request means one keep-set per request, so this
//!    accumulates the per-layer rows and ranks the sum. Summation is the
//!    layer-uniform variant the TOVA paper itself evaluates (§4), and it is
//!    monotone-equivalent to the mean, so the ranking is unaffected by the
//!    layer count the program cannot observe.
//!
//! ## Why the score row is a distribution
//!
//! Each layer contributes a row that sums to 1 over the live prefix, and slots
//! past the live length are exactly `0.0` — a position that does not exist
//! received no attention, so it needs no sentinel to sort to the bottom. After
//! folding `L` layers the row sums to `L`, which is what `layers_observed`
//! lets the host check: the drained row is self-validating.

use inferlet::ptir::attention::prelude::*;
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
    /// Prefill chunk width, clamped to the driver's `max_embed_length()`.
    /// Defaults to that limit, i.e. the fewest chunks the driver allows.
    /// Forcing it down runs the multi-chunk path on a short prompt, which is
    /// the only way to test chunk equivalence without a 16K-token prompt.
    #[serde(default)]
    prefill_chunk: Option<u32>,
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
    /// The static ceiling the program declared for the score row.
    kv_max: u32,
    /// Live KV positions at the last observed step.
    kv_len: u32,
    cache_size: u32,
    /// Layers that reported a score row on the last fire.
    layers_observed: u32,
    /// Slots inside the live prefix that carry a finite, non-negative score.
    live_scored: usize,
    /// Slots past the live prefix. Every one of them must be exactly zero.
    tail_nonzero: usize,
    /// One past the highest slot carrying any attention mass, i.e. the live KV
    /// length as the DRIVER saw it. Reported next to `kv_len` (the length the
    /// program believes it declared) because a disagreement between the two is
    /// the whole failure mode: it means the row describes different positions
    /// than the program thinks it does.
    observed_live: usize,
    /// `(declared kv_len, observed live length)` for every drained fire.
    trace: Vec<(u32, usize)>,
    scores_nan: usize,
    /// `Σ score` over the live prefix. Each layer contributes a distribution,
    /// so this must come out at `layers_observed` within float tolerance.
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
    let max_tokens = input.max_tokens;
    let temperature = input.temperature;
    let ws = WorkingSet::new();
    let page_size = kv_page_size();

    if max_tokens == 0 {
        return Err("max_tokens must be at least 1".into());
    }

    let mut prompt = model::encode(&input.prompt);
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let max_pages = (n + max_tokens as u32 + 1).div_ceil(page_size).max(1);
    // The program cannot know the runtime KV length, so — exactly like Quest's
    // `p_max` — it declares a static ceiling and the backend refuses (rather
    // than truncates) a request that outgrows it. Sizing it off `max_pages`
    // rather than `n + max_tokens` keeps it an exact multiple of the page
    // geometry the driver derives its own length from.
    let kv_max = max_pages * page_size;
    let cache_size = input.cache_size.min(kv_max);
    ws.reserve(max_pages).context("reserve KV")?;

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);

    // ── PREFILL (chunked, C-wide): no TOVA tap. ──
    //
    // TOVA is defined on the decoding step: it ranks by the attention of the
    // most recent query token, and during prefill "most recent" is still
    // moving. The paper applies it from the first generated token onward.
    //
    // Split into `ceil(n / C)` chunks, `C = max_embed_length()`. A one-shot
    // fire cannot exceed the driver's per-launch token capacity, which capped
    // this policy at 8192 prompt tokens; chunk `i` attends over the whole
    // prefix written so far and writes only its own tokens, so the
    // concatenation equals the one-shot fire (section 17).
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    // The split is `prefill_chunks` (SDK), which spreads the remainder over the
    // FIRST chunks so the last one is never a sliver. Every inferlet in this
    // tree uses it, which is what makes their chunk boundaries identical: a
    // text difference above the ceiling is then a policy difference, not a
    // difference in the attention tile decomposition (§11.4).
    let spans = prefill_chunks(n, input.prefill_chunk);
    let pipe = Pipeline::new();

    let mut g0 = 0i32;
    for &(base, end) in &spans {
        let len = end - base;

        let toks_p = Channel::from(&prompt_i32[base as usize..end as usize]).named("toks_p");
        let embed_indptr_p = Channel::from([0u32, len]).named("embed_indptr_p");
        let positions_p = Channel::from_iter(base..end).named("positions_p");
        let pages_p = Channel::from_iter(0..max_pages).named("pages_p");
        let page_indptr_p = Channel::from([0u32, end.div_ceil(page_size)]).named("page_indptr_p");
        let w_slot_p = Channel::from_iter((base..end).map(|p| p / page_size)).named("w_slot_p");
        let w_off_p = Channel::from_iter((base..end).map(|p| p % page_size)).named("w_off_p");
        let kv_len_p = Channel::from([end]).named("kv_len_p");
        let rng_p = Channel::from([input.seed, 0]).named("rng_p");
        let tok_out_p = Channel::new([1], dtype::i32).named("tok_out_p");

        let fwd_p = ForwardPass::new();
        fwd_p.embed(&toks_p, &embed_indptr_p)?;
        fwd_p.attention(
            &ws,
            KvGeometry {
                readable_pages: ..,
                writable_pages: ..,
                kv_len: &kv_len_p,
                pages: &pages_p,
                page_indptr: &page_indptr_p,
                w_slot: &w_slot_p,
                w_off: &w_off_p,
                positions: &positions_p,
                mask: None,
            },
        )?;
        fwd_p.epilogue(move || {
            let r = rng_p.take();
            let logits = intrinsics::logits();
            let token = step(logits, temperature, &r);
            tok_out_p.put(&token);
            rng_p.put(&(&r + iota(2)));
        });

        fwd_p
            .submit(&pipe)
            .with_context(|| format!("prefill submit @{base}"))?;

        g0 = tok_out_p
            .take_host::<i32>()
            .await
            .with_context(|| format!("@{base}"))?;
    }
    generated.push(g0 as u32);

    let mut last_scores: Vec<f32> = Vec::new();
    let mut layers_observed = 0u32;
    let mut last_kv_len = n + 1;
    let mut trace: Vec<(u32, usize)> = Vec::new();

    // ── DECODE LOOP (1-wide, run-ahead), with the TOVA tap. ──
    if generated.len() < max_tokens {
        let tok_in = Channel::from([g0]).named("tok_in");
        let rng = Channel::from([input.seed ^ 0x70a, 0]).named("rng");
        let tok_out = Channel::new([1], dtype::i32)
            .capacity(channel_capacity() as u32)
            .named("tok_out");
        let lane1 = Channel::from([0u32, 1u32]).named("embed_indptr");
        let positions = Channel::from([n]).named("positions");
        let pages = Channel::from_iter(0..max_pages).named("pages");
        let page_indptr = Channel::from([0u32, (n + 1).div_ceil(page_size)]).named("page_indptr");
        let w_slot = Channel::from([n / page_size]).named("w_slot");
        let w_off = Channel::from([n % page_size]).named("w_off");
        let kv_len = Channel::from([n + 1]).named("kv_len");

        // The per-layer accumulator. `on_attn` fires once per layer and the
        // inferlet cannot ask the model how many layers there are, so the
        // layer loop folds into a device-carried channel and the epilogue —
        // which fires exactly once per fire — publishes the fold. The seed is
        // `0.0` (the identity of `+`) and a zero layer counter.
        //
        // Contract #1: a loop-carried channel must be `take`n before it is
        // `put`, or the dummy run that infers the geometry sees a port that is
        // written but never read.
        let acc = Channel::from(vec![0.0f32; kv_max as usize]).named("tova_acc");
        let layer_ct = Channel::from([0u32]).named("tova_layers");

        // Host drains.
        let scores_out = Channel::new([kv_max], dtype::f32)
            .capacity(channel_capacity() as u32)
            .named("tova_scores");
        let layers_out = Channel::new([1], dtype::u32)
            .capacity(channel_capacity() as u32)
            .named("tova_layer_count");

        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, &lane1)?;
        fwd.attention(
            &ws,
            KvGeometry {
                readable_pages: ..,
                writable_pages: (n / page_size)..,
                kv_len: &kv_len,
                pages: &pages,
                page_indptr: &page_indptr,
                w_slot: &w_slot,
                w_off: &w_off,
                positions: &positions,
                mask: None,
            },
        )?;

        // ── THE TAP. Fires once per layer, inside the fire, AFTER the layer's
        //    attention — which is what distinguishes `on_attn` from Quest's
        //    `on_attn_proj`: the scores do not exist until attention has run. ──
        fwd.on_attn(move || {
            let prev = acc.take();
            let ct = layer_ct.take();
            let scores = intrinsics::attn_score(kv_max);
            acc.put(&(&prev + &scores));
            layer_ct.put(&(&ct + 1u32));
        });

        fwd.epilogue(move || {
            let length = kv_len.take();
            let r = rng.take();
            let logits = intrinsics::logits();
            let token = step(logits, temperature, &r);

            let next_length = &length + 1u32;
            let page_count = next_length.div_ceil(page_size);

            tok_in.put(&token);
            kv_len.put(&next_length);
            positions.put(&length);
            w_slot.put(&length / page_size);
            w_off.put(&length % page_size);
            page_indptr.put(indptr(1, &page_count));
            tok_out.put(&token);
            rng.put(&(&r + iota(2)));

            // Publish the layer fold, then re-seed it for the next fire.
            let folded = acc.take();
            let layers = layer_ct.take();
            scores_out.put(&folded);
            layers_out.put(&layers);
            acc.put(&broadcast(0.0f32, [kv_max]));
            layer_ct.put(&reshape(&(&layers * 0u32), [1]));
        });

        let budget_n = max_tokens - 1;
        run_ahead(&pipe, &fwd, budget_n as usize, async || {
            let t = tok_out
                .take_host::<i32>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            last_scores = scores_out
                .take_host::<Vec<f32>>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            layers_observed = layers_out
                .take_host::<u32>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            // The fire that produced this row had `n + generated.len()` KV
            // positions live: the prompt plus every token committed before it.
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
    // The host-side mirror of TOVA's selection: keep the `cache_size` highest
    // scoring LIVE positions. Ties break toward the newer position, which is
    // the paper's own tie-break (recency wins).
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
        layers_observed,
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
