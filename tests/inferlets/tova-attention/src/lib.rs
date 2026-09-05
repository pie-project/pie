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
//! hardware, against the live KV cache, and drains the resulting scores back to
//! the host so the keep-set can be checked against a reference. It does not yet
//! mask the evicted positions out of the attention kernel, so it produces
//! *bit-identical output to `naive-baseline`* — which is exactly what makes it
//! testable: any divergence is a bug in the tap, not a property of the
//! algorithm.
//!
//! ## What the backend does under `intrinsics::attn_score`
//!
//! **The graph wrote it; the epilogue reads it** (`.wiki/alto/attn-score.md`
//! §4). The attention capture arm accumulates per-key mass into an arena
//! rectangle as it runs, one row per (exported attention layer, query head),
//! and the epilogue — a boundary that already exists — is handed the whole
//! thing at once as a device tensor:
//!
//! ```text
//! intrinsics::attn_score(planes) -> [planes, intrinsics::attn_score_kv_max()]
//! ```
//!
//! Rows run **layer-major, head-minor** (`layer * heads + head`), so a program
//! that declares fewer planes than the load exports reads a prefix of the
//! layers rather than a stripe of the heads. There is no per-layer stage, no
//! mid-forward tap, and no host in the loop; the scores are still exactly as
//! fresh as the fire that produced them. `Stage::OnAttn` does not admit this
//! intrinsic at all — a program that reads it there is refused at bind.
//!
//! The scores are the softmax probabilities the attention kernel itself
//! computed — captured as it ran, never recomputed — so they cannot drift from
//! the attention the model actually performed.
//!
//! ## The observation window is one row here, and that is TOVA's definition
//!
//! The backend folds the last `min(32, qo_len)` query rows of the request into
//! the row it publishes. This program taps only 1-row decode fires, so the
//! window is exactly one row: the distribution of the *current* query token,
//! which is precisely the quantity TOVA ranks. (SnapKV is the policy that wants
//! the 32-row form, and `trackb-snapkv` reads it at a prefill epilogue.)
//!
//! ## Two documented deviations from the paper
//!
//! 1. **Heads are folded by the PROGRAM.** TOVA ranks per head. The paged KV
//!    layout here carries one page list per request, so eviction is inherently
//!    a per-request decision and a per-head keep-set has no representable
//!    consumer. The rectangle is per-head because observability wants it that
//!    way (§4: "per-head is the better answer"), so this program takes the mean
//!    over its own heads — the union-friendly collapse: a position that any
//!    head needs keeps a non-trivial share of the mass. `quest-attention`
//!    documents the identical collapse for the identical reason.
//!
//! 2. **Layers are folded by the program.** TOVA maintains a separate cache per
//!    layer. One page list per request means one keep-set per request, so this
//!    sums the per-layer distributions and ranks the sum. Summation is the
//!    layer-uniform variant the TOVA paper itself evaluates (§4), and it is
//!    monotone-equivalent to the mean, so the ranking is unaffected by how many
//!    layers the program declared.
//!
//! Both folds are ONE in-graph reduction at the epilogue, on the device
//! (§4: "reduction stays on device — only decisions cross to the host"):
//!
//! ```text
//! folded = reduce_sum(transpose(rect)) / heads
//! ```
//!
//! `transpose` turns `[planes, kv_max]` into `[kv_max, planes]` so that
//! `reduce_sum` — which reduces the LAST axis — sums down the planes; dividing
//! by `heads` turns that plane-sum into (mean over heads, then sum over
//! layers), because `Σ_l (1/H) Σ_h row = (1/H) Σ_planes row`. One pass over the
//! rectangle computes both folds.
//!
//! ## Why the score row is a distribution
//!
//! Each (layer, head) row sums to 1 over the live prefix, and slots past the
//! live length are exactly `0.0`, rewritten every fire — a position that does
//! not exist received no attention, so it needs no sentinel to sort to the
//! bottom. The mean over heads is therefore one distribution per layer, and the
//! sum over `L` layers is a row of mass exactly `L`. That is what
//! `layers_observed` lets the host check: the drained row is self-validating.
//!
//! **`layers_observed` is now a declared number, not a counted one.** There is
//! no per-layer tap to count and no device counter channel; the layer count is
//! `planes / heads`, read straight off the shape the program declared.

use inferlet::eta::attention::prelude::*;
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
    /// Attention layers the load EXPORTS a score plane for. Declared, not
    /// derived: the plane count is not in the model profile and the SDK has no
    /// host call for it, so the program states it and the backend refuses a
    /// claim larger than the load exports (the `hidden(width)` deviation —
    /// a declared ceiling, checked by name).
    ///
    /// The default is `Qwen/Qwen3.5-0.8B`: `Model::d0_8b` in
    /// `crates/models/src/qwen_3/model.rs` is `layers: 24, attn_every: 4`, and
    /// the SKU is hybrid — `attn_at(l) = l % attn_every == attn_every - 1`
    /// puts an attention mixer on 6 of the 24 layers and a GDN mixer on the
    /// other 18. Only the attention layers export a plane.
    #[serde(default = "default_layers")]
    layers: u32,
    /// Query heads per exported attention layer. `Model::d0_8b`'s
    /// `q_heads: 8`, at `tp == 1`.
    #[serde(default = "default_heads")]
    heads: u32,
    /// Prefill chunk width, clamped to the engine's `max_embed_length()`.
    /// Defaults to that limit, i.e. the fewest chunks the engine allows.
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
/// qwen35-d0.8b: 24 layers, `attn_every: 4` → 6 exported attention layers.
fn default_layers() -> u32 {
    6
}
/// qwen35-d0.8b: `q_heads: 8` at `tp == 1`.
fn default_heads() -> u32 {
    8
}

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    text: String,
    count: usize,
    /// The program's own KV window into the published score row: the exact
    /// page geometry it reserved, `max_pages * page_size`. The ROW's width is
    /// not the program's to declare any more — it is the published constant
    /// `intrinsics::attn_score_kv_max()` — so this is the prefix of that row
    /// the program reads, and the program refuses up front if its geometry
    /// would outgrow the published one.
    kv_max: u32,
    /// Live KV positions at the last observed step.
    kv_len: u32,
    cache_size: u32,
    /// Attention layers folded into the row. DERIVED FROM THE DECLARED SHAPE
    /// (`planes / heads`), not counted by a device channel: there is no
    /// per-layer tap left to count, and the rectangle arrives whole.
    layers_observed: u32,
    /// Slots inside the live prefix that carry a finite, non-negative score.
    live_scored: usize,
    /// Slots past the live prefix. Every one of them must be exactly zero.
    tail_nonzero: usize,
    /// One past the highest slot carrying any attention mass, i.e. the live KV
    /// length as the ENGINE saw it. Reported next to `kv_len` (the length the
    /// program believes it declared) because a disagreement between the two is
    /// the whole failure mode: it means the row describes different positions
    /// than the program thinks it does.
    observed_live: usize,
    /// `(declared kv_len, observed live length)` for every drained fire.
    trace: Vec<(u32, usize)>,
    scores_nan: usize,
    /// `Σ score` over the live prefix. The mean over heads is one distribution
    /// per layer, so the layer-sum must come out at `layers_observed` within
    /// float tolerance.
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
    if input.layers == 0 || input.heads == 0 {
        return Err("layers and heads must both be at least 1".into());
    }
    // The declared rectangle. `planes = exported attention layers * query
    // heads`, layer-major; the backend refuses a claim larger than the load
    // exports, by name.
    let layers = input.layers;
    let heads = input.heads;
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

    // The model's opening (`<bos>` where it has one) before the raw text —
    // the opening `naive-baseline` puts there, which is what makes this
    // program's all-keep arm comparable to it, and what puts the attention
    // sink these policies rank at position 0.
    let mut prompt = inferlet::chat::prefix();
    prompt.extend(model::encode(&input.prompt));
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let max_pages = (n + max_tokens as u32 + 1).div_ceil(page_size).max(1);
    // The program cannot know the runtime KV length, so — exactly like Quest's
    // `p_max` — it declares a static ceiling and the backend refuses (rather
    // than truncates) a request that outgrows it. Sizing it off `max_pages`
    // rather than `n + max_tokens` keeps it an exact multiple of the page
    // geometry the engine derives its own length from.
    let kv_max = max_pages * page_size;
    // The score row's width is the backend's, not the program's: a slab pitch
    // cannot be a per-program number, so `attn_score_kv_max()` publishes the
    // one that was carved and the program reads a prefix of it. Refusing here
    // is the honest failure — a truncated read would produce a plausible
    // ranking over positions that are not the ones it names.
    if kv_max > intrinsics::attn_score_kv_max() {
        return Err(format!(
            "prompt + max_tokens needs {kv_max} KV slots, past the published \
             attn_score ceiling of {}",
            intrinsics::attn_score_kv_max()
        )
        .into());
    }
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
    // fire cannot exceed the engine's per-launch token capacity, which capped
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

        // NO ACCUMULATOR, AND NO LAYER COUNTER. Both were device-carried
        // channels that existed only to fold a per-layer tap across the layer
        // loop, and there is no layer loop to fold: the epilogue is handed
        // every exported layer's rows at once. The layer count that used to
        // ride `tova_layers` back to the host is `planes / heads`, a number
        // the program declared.
        //
        // Host drain.
        let scores_out = Channel::new([kv_max], dtype::f32)
            .capacity(channel_capacity() as u32)
            .named("tova_scores");

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

            // ── THE TAP. Once per fire, at the epilogue, over the whole
            //    rectangle: `[planes, attn_score_kv_max()]`, layer-major.
            //
            //    `transpose` puts the planes on the last axis so `reduce_sum`
            //    — which reduces the LAST axis — sums down them; `/ heads`
            //    turns that plane-sum into (mean over heads, then sum over
            //    layers), since `Σ_l (1/H) Σ_h row = (1/H) Σ_planes row`. The
            //    result is a row of mass exactly `layers`.
            //
            //    The `gather` narrows the published width to the program's own
            //    page geometry — `kv_max = max_pages * page_size` — so the
            //    drained row is the prefix this program reserved and nothing
            //    past it. Every step of this is in-graph: only the row itself
            //    crosses to the host, and only because the test reads it.
            //
            //    Nothing is carried between fires. TOVA ranks the CURRENT
            //    step's attention and throws the previous steps away; that is
            //    the one line separating it from `trackb-h2o`, and with the
            //    rectangle arriving whole it is now the absence of a channel
            //    rather than a re-seed.
            let rect = intrinsics::attn_score(planes);
            let folded = &reduce_sum(&transpose(&rect)) / heads_f;
            scores_out.put(&gather(&folded, iota(kv_max)));
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
        layers_observed: layers,
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
