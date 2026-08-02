//! H2O — Heavy-Hitter Oracle (Zhang et al., 2023).
//!
//! <https://arxiv.org/abs/2306.14048>
//!
//! H2O and TOVA read the *same* quantity — the attention distribution the
//! decode kernel just computed — and differ in exactly one line: TOVA ranks by
//! the current step's attention and throws the previous steps away, H2O
//! accumulates. A position is a "heavy hitter" if it has drawn a large share of
//! attention *over the whole generation so far*, which is a much more stable
//! statistic than any single step's.
//!
//! That one-line difference is the whole reason this inferlet exists as a
//! separate program rather than a flag: it is the sharpest available test that
//! the score tap is genuinely stateful across fires. `tova-attention`'s
//! accumulator is re-seeded in the epilogue; this one is not, so its score mass
//! must grow linearly in the number of fires. If the loop-carried channel were
//! silently re-seeded, or the tap re-read stale device memory, the mass would
//! be flat and the test would say so.
//!
//! ## Unlike `tova-attention`, this one ENFORCES
//!
//! `tova-attention` is an observer: it computes a keep-set and the model
//! attends over everything anyway, which is what makes it a clean parity test
//! against `naive-baseline`. That is not a policy, though — it is the cost of a
//! policy with none of the benefit. H2O here closes the loop through
//! `attn_page_mask`, the same sink Quest uses.
//!
//! **Eviction is page-granular, and that is not a shortcut.** The KV cache is
//! paged; a position-granular mask can stop attention from *reading* a position
//! but cannot hand its memory back, so the thing H2O exists to do — run in less
//! memory — is only expressible at page granularity. The program therefore
//! folds its `[kv_max]` position scores into `[p_max]` page masses (a page's
//! mass is the sum of its positions', which is the natural collapse for a
//! quantity that is already a probability mass) and ranks pages.
//!
//! ## The cold start, and why the seed is a descending ramp
//!
//! The first decode fire has to choose a keep-set having observed nothing:
//! prefill has no score capture (design doc §9.5), so the accumulator is at its
//! seed. An all-zero seed would make every page tie and the selection arbitrary
//! — evicting real context on the strength of no evidence.
//!
//! The seed is instead a small descending ramp over positions, so with no
//! attention observed yet the ranking degenerates to "keep the earliest
//! positions", and the backend independently force-keeps the request's last
//! page. That is precisely the Λ-shape (attention sink + local window) that
//! StreamingLLM shows is the right prior, and that H2O's own keep-sets converge
//! to. The ramp is scaled far below one layer-fire of attention mass, so a
//! single observation dominates it; it breaks ties and nothing more.
//!
//! Slots past the live prefix get the *smallest* prior, which matters: the
//! backend consults the mask only for pages the request actually has, but a
//! ranking that put unused pages on top would evict live ones to make room.
//!
//! ## What the backend does under `intrinsics::attn_score`
//!
//! The call lowers to `Op::IntrinsicVal{intr: AttnScore}` at the `on_attn`
//! stage. The CUDA driver serves it only when the model reports the
//! `has_attn_score` capability (llama-like family, no sliding window, native
//! bf16 pages, `tp == 1`, and a decode fire on the plain paged-decode path).
//! The scores are the softmax probabilities the decode kernel itself computed —
//! captured inside the attention kernel via a FlashInfer attention variant, not
//! recomputed — so they cannot drift from the attention the model performed.
//!
//! ## Deviations from the paper, as built
//!
//! 1. **Heads are folded by the backend.** H2O ranks per head; one page list
//!    per request means a per-head keep-set has no representable consumer, so
//!    the backend returns the mean over query heads. `quest-attention` and
//!    `tova-attention` document the identical collapse.
//! 2. **Layers are folded by the program.** Same reason: one page list per
//!    request is one keep-set per request. Summing layers is monotone-
//!    equivalent to averaging them, so the ranking does not depend on a layer
//!    count the program cannot observe.
//! 3. **Pages, not positions** — see above. This is a property of a paged KV
//!    cache, not of H2O.

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
    /// KV pages H2O keeps. The backend force-keeps the request's last page on
    /// top of this, so the effective local window is never zero.
    #[serde(default = "default_page_budget")]
    page_budget: u32,
    /// Drain the per-step score row and layer counter to the host. The tests
    /// assert on them, so it defaults on; a benchmark must turn it off, because
    /// a `[kv_max]` f32 readback per decode step is tens of kilobytes and a
    /// round-trip that the POLICY never performs. Timing it measures the
    /// harness. See §14.2 of the design document.
    #[serde(default = "default_report")]
    report: bool,
    /// Pins the reserved page count so it does not move with `max_tokens`.
    /// `p_max` sets the width of the score row folded and ranked every layer,
    /// so a benchmark that differences two `max_tokens` would otherwise be
    /// comparing two different per-step workloads and would charge the policy
    /// for the difference. Setting this to the larger of the two endpoints
    /// makes the per-step cost identical. Defaults to `max_tokens`.
    #[serde(default)]
    reserve_tokens: Option<usize>,
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
fn default_page_budget() -> u32 {
    4
}
fn default_report() -> bool {
    true
}

/// Largest entry of the cold-start ramp (see the module docs). Four orders of
/// magnitude below the 1.0 a single layer-fire contributes, so one observation
/// dominates it.
const SEED_SCALE: f32 = 1e-4;

/// What a slot may carry and still be explained by the seed alone. The ramp is
/// bounded by `SEED_SCALE`; anything above this is attention mass.
const SEED_CEILING: f32 = SEED_SCALE * 1.001;

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    text: String,
    count: usize,
    /// The static ceiling the program declared for the score row.
    kv_max: u32,
    /// Live KV positions at the last observed step.
    kv_len: u32,
    page_budget: u32,
    /// Pages the request actually had at the last observed fire.
    live_pages: u32,
    /// Layers that have reported a score row, CUMULATIVELY across every fire --
    /// H2O never resets its counter, for the same reason it never re-seeds its
    /// accumulator.
    layers_observed: u32,
    /// Layers per fire, i.e. the model's layer count, recovered as the step of
    /// the cumulative counter. This is the expected step of `mass_trace`.
    layers_per_fire: u32,
    /// Slots inside the live prefix that carry a finite, non-negative score.
    live_scored: usize,
    /// Slots past the live prefix carrying more than the seed ramp could
    /// account for. TOVA can assert the tail is exactly zero; H2O cannot,
    /// because its seed is a non-zero ramp that is never re-seeded away. What
    /// must still hold is that no ATTENTION mass ever lands out there -- a
    /// position that does not exist cannot have been attended to -- so the bar
    /// is the seed ceiling rather than zero.
    tail_polluted: usize,
    /// The largest value past the live prefix, for the same check by eye.
    tail_max: f32,
    /// One past the highest slot carrying any attention mass, i.e. the live KV
    /// length as the DRIVER saw it. Reported next to `kv_len` (the length the
    /// program believes it declared) because a disagreement between the two is
    /// the whole failure mode: it means the row describes different positions
    /// than the program thinks it does.
    observed_live: usize,
    /// `(declared kv_len, observed live length)` for every drained fire.
    trace: Vec<(u32, usize)>,
    scores_nan: usize,
    /// `Σ score` over the live prefix. Each layer of each fire contributes one
    /// distribution, so this must come out at `layers_observed` -- which for
    /// H2O is the CUMULATIVE layer count, not the per-fire one. A flat mass
    /// across fires is the signature of an accumulator that is being re-seeded
    /// behind the program's back.
    score_mass: f32,
    /// `score_mass` after each drained fire. Must be strictly increasing, by
    /// one fire's worth of layers each time. This is the observable that
    /// separates H2O from TOVA.
    mass_trace: Vec<f32>,
    /// H2O's page-level keep-set: the `page_budget` highest-mass live pages.
    kept_pages: Vec<u32>,
    /// The page H2O would evict next (lowest mass).
    evicted_first: Option<u32>,
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
    let reserve = input.reserve_tokens.unwrap_or(max_tokens).max(max_tokens);
    let max_pages = (n + reserve as u32 + 1).div_ceil(page_size).max(1);
    // The program cannot know the runtime KV length, so — exactly like Quest's
    // `p_max` — it declares a static ceiling and the backend refuses (rather
    // than truncates) a request that outgrows it. Sizing it off `max_pages`
    // rather than `n + max_tokens` keeps it an exact multiple of the page
    // geometry the driver derives its own length from.
    let kv_max = max_pages * page_size;
    let p_max = max_pages;
    let page_budget = input.page_budget.min(p_max);
    let report = input.report;
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
    let mut layers_per_fire = 0u32;
    let mut last_kv_len = n + 1;
    let mut trace: Vec<(u32, usize)> = Vec::new();
    let mut mass_trace: Vec<f32> = Vec::new();

    // ── DECODE LOOP (1-wide, run-ahead), with the TOVA tap. ──
    if generated.len() < max_tokens {
        let tok_in = Channel::from([g0]).named("tok_in");
        // Same salt as `naive-baseline`. The coherence test asks whether an
        // all-keep page mask changes what the model produces, and that question
        // is only answerable if every other input is held fixed -- including
        // the Gumbel stream. A different salt makes the two runs disagree for
        // reasons that have nothing to do with attention.
        let rng = Channel::from([input.seed ^ 0x5bd1, 0]).named("rng");
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

        // The heavy-hitter accumulator. `on_attn` fires once per layer and the
        // inferlet cannot ask the model how many layers there are, so the layer
        // loop folds into a device-carried channel. Unlike TOVA's, the epilogue
        // does NOT re-seed it: H2O's statistic is cumulative over the whole
        // generation, and that carry is the entire difference between the two
        // policies.
        //
        // Contract #1: a loop-carried channel must be `take`n before it is
        // `put`, or the dummy run that infers the geometry sees a port that is
        // written but never read.
        //
        // The seed is a descending ramp, not zeros -- see the module docs. Its
        // largest entry is `SEED_SCALE`, four orders of magnitude below the
        // 1.0 a single layer-fire contributes, so it decides ties on the first
        // fire and is irrelevant afterwards.
        let seed: Vec<f32> = (0..kv_max)
            .map(|i| SEED_SCALE * (kv_max - i) as f32 / kv_max as f32)
            .collect();
        let acc = Channel::from(seed).named("h2o_acc");
        let layer_ct = Channel::from([0u32]).named("h2o_layers");

        // Host drains. Absent entirely when `report` is off -- not merely
        // undrained -- so the epilogue never writes them and the geometry pass
        // never sees a port. `acc` and `layer_ct` stay unconditional: their
        // fold feeds `page_mass_epi`, which IS the policy.
        let scores_out = report.then(|| {
            Channel::new([kv_max], dtype::f32)
                .capacity(channel_capacity() as u32)
                .named("h2o_scores")
        });
        let layers_out = report.then(|| {
            Channel::new([1], dtype::u32)
                .capacity(channel_capacity() as u32)
                .named("h2o_layer_count")
        });

        // The page-mass row the NEXT fire's `on_attn_proj` ranks. Published by
        // the epilogue because that is the only stage that sees the completed
        // fold; consumed a fire later because `attn_score` is not readable
        // until attention has run, and the mask has to be in place before it
        // does. For H2O that lag is not a compromise -- the policy is defined
        // on the history, and the history is exactly what the previous fires
        // accumulated.
        let page_mass = Channel::from(
            (0..p_max)
                .map(|p| SEED_SCALE * (p_max - p) as f32 / p_max as f32)
                .collect::<Vec<f32>>(),
        )
        .named("h2o_page_mass");
        let page_mass_epi = page_mass.clone();

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

        // ── THE ENFORCEMENT. Fires once per layer, BEFORE the layer's
        //    attention, which is the only point at which the page list can
        //    still be narrowed. It ranks the page masses the previous fires
        //    accumulated -- H2O's statistic is the history, so reading a
        //    fire-old row is the policy, not a lag to apologise for.
        //
        //    `pivot_threshold(.., rank_le(budget))` is the top-`budget` keep
        //    predicate in one op; `top_k` would need a second reduce and is a
        //    library region (a fusion barrier). The backend force-keeps the
        //    request's last page whatever this says -- it holds the token this
        //    fire is writing -- which doubles as H2O's local window. ──
        fwd.on_attn_proj(move || {
            let mass = page_mass.take();
            intrinsics::kernel::attn_page_mask(&pivot_threshold(&mass, rank_le(page_budget)));
            page_mass.put(&mass);
        });

        // ── THE TAP. Fires once per layer, inside the fire, AFTER the layer's
        //    attention — which is what distinguishes `on_attn` from
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

            // Publish the fold and CARRY IT FORWARD. This is the one line that
            // separates H2O from TOVA: `tova-attention` re-seeds `acc` here, so
            // its row describes the last step; H2O's describes every step so
            // far. The layer counter is carried for the same reason, which is
            // what makes `score_mass == layers_observed` stay true as both grow.
            let folded = acc.take();
            let layers = layer_ct.take();
            if let Some(c) = scores_out.as_ref() {
                c.put(&folded);
            }
            if let Some(c) = layers_out.as_ref() {
                c.put(&layers);
            }
            acc.put(&folded);
            layer_ct.put(&layers);

            // Fold positions into pages for the next fire's mask. `reduce_sum`
            // reduces the last axis per row, and `kv_max = p_max * page_size`
            // exactly (it was derived from the page geometry), so the reshape
            // is a reinterpretation rather than a resize. Summing is the right
            // collapse for a probability mass: a page's share of the attention
            // is the sum of its positions' shares.
            page_mass_epi.take();
            page_mass_epi.put(&reduce_sum(&reshape(&folded, [p_max, page_size])));
        });

        let budget_n = max_tokens - 1;
        run_ahead(&pipe, &fwd, budget_n as usize, async || {
            let t = tok_out
                .take_host::<i32>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            if let (Some(sc), Some(lc)) = (scores_out.as_ref(), layers_out.as_ref()) {
                last_scores = sc
                    .take_host::<Vec<f32>>()
                    .await
                    .with_context(|| format!("@{}", generated.len()))?;
                let layers_before = layers_observed;
                layers_observed = lc
                    .take_host::<u32>()
                    .await
                    .with_context(|| format!("@{}", generated.len()))?;
                // The fire that produced this row had `n + generated.len()` KV
                // positions live: the prompt plus every token committed before it.
                last_kv_len = n + generated.len() as u32;
                layers_per_fire = layers_observed - layers_before;
                mass_trace.push(last_scores.iter().filter(|s| s.is_finite()).sum());
                trace.push((
                    last_kv_len,
                    last_scores
                        .iter()
                        .rposition(|s| *s != 0.0)
                        .map_or(0, |i| i + 1),
                ));
            }
            generated.push(t as u32);
            Ok(ControlFlow::Continue(()))
        })
        .await?;
    }
    pipe.close();

    let live = (last_kv_len as usize).min(last_scores.len());
    let live_pages = last_kv_len.div_ceil(page_size).min(p_max);

    // The host-side mirror of the selection the device just made: fold the
    // drained position scores into page masses and rank them. Recomputing it
    // here rather than draining the device's mask is deliberate -- it is an
    // independent second opinion, so a disagreement is visible.
    //
    // `live_pages` is derived from `kv_len`, which is known whether or not the
    // scores were drained, so it can outrun `last_scores` when `report` is off
    // and the row was never fetched. Clamp BOTH ends against the row actually
    // in hand rather than only `hi`: a page whose whole span sits past the end
    // must yield an empty slice, not a backwards one.
    let masses: Vec<f32> = (0..live_pages as usize)
        .map(|p| {
            let lo = (p * page_size as usize).min(last_scores.len());
            let hi = ((p + 1) * page_size as usize).min(last_scores.len());
            last_scores[lo..hi].iter().sum()
        })
        .collect();
    let mut order: Vec<u32> = (0..live_pages).collect();
    order.sort_by(|&a, &b| {
        masses[b as usize]
            .partial_cmp(&masses[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.cmp(&a))
    });
    let evicted_first = order.last().copied();
    let mut kept: Vec<u32> = order.into_iter().take(page_budget as usize).collect();
    kept.sort_unstable();

    Ok(Output {
        sampler: "trackb-h2o",
        text: model::decode(&generated)?,
        count: generated.len(),
        kv_max,
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
