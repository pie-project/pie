//! SnapKV — prompt compression from the observation window (Li et al., 2024).
//!
//! <https://arxiv.org/abs/2404.14469>
//!
//! ## What makes SnapKV structurally different from H2O and TOVA
//!
//! H2O and TOVA are *decode-time* policies: they watch the attention of each
//! generated token and continuously re-rank the cache. SnapKV is a *prefill-time*
//! policy. It looks at the attention that the **tail of the prompt** paid to the
//! rest of the prompt, selects a keep-set once, and then holds it for the whole
//! generation. Its premise is that the last few prompt tokens are the best
//! available proxy for what the not-yet-generated continuation will want.
//!
//! That premise makes SnapKV the only member of Track B that needs a **prefill**
//! score capture. `attn_score` at `on_attn` during a decode fire yields one query
//! row; during a prefill fire it yields the mean over the last `window` query
//! rows (32 by default, `PIE_ATTN_SCORE_WINDOW`). Design doc §12.
//!
//! So this program is the end-to-end test of a path nothing else exercises: the
//! FlashInfer prefill variant, the causal-aware normalisation (the prefill hook
//! sees dot products at positions the softmax later discards, because
//! `LogitsMask` runs after `LogitsTransform`), and the head+row fold.
//!
//! ## What it enforces, and how the budget is spent
//!
//! One selection, made after prefill, then fixed. Two groups of pages are kept:
//!
//!  1. The top `page_budget` **prompt** pages by observed attention mass.
//!  2. **Every page written after prefill**, unconditionally.
//!
//! (2) is not a hedge, it is SnapKV: the paper compresses the prompt and then
//! lets the cache grow normally from there. Evicting freshly generated tokens
//! would be a different (and worse) algorithm. It is expressed by giving those
//! slots a mass far above anything a prompt page can reach and widening the
//! rank predicate by the same count, so the selection among prompt pages is
//! still exactly `page_budget` wide.
//!
//! ## Deviations from the paper, as built
//!
//! 1. **Pooling is page-granular and non-overlapping.** SnapKV max-pools the
//!    position scores with kernel 7, stride 1 before top-k, so that a selected
//!    position drags its neighbours in and the kept set is contiguous rather
//!    than speckled. Here the pool is the page fold: kernel = stride =
//!    `page_size`. That is not an approximation of SnapKV's pooling, it is the
//!    *enforceable* version of it — a paged KV cache only returns memory a page
//!    at a time, so a sub-page keep-set cannot free anything. The clustering
//!    effect SnapKV's pooling exists to produce is what page granularity gives
//!    for free.
//! 2. **Heads are folded by the backend.** SnapKV selects per head; one page
//!    list per request means a per-head keep-set has no representable consumer,
//!    so the backend returns the mean over query heads. `quest-attention`,
//!    `tova-attention` and `trackb-h2o` document the identical collapse.
//! 3. **Layers are folded by the program.** Same reason. Summing layers is
//!    monotone-equivalent to averaging, so the ranking does not depend on a
//!    layer count the program cannot observe.
//!
//! ## Why the page mass takes a host round trip
//!
//! The selection is made in the prefill program and enforced in the decode
//! program, and those are two separately submitted PTIR programs. The
//! established idiom for carrying a value between them is a host drain and a
//! fresh channel — `attention-sink` carries its first token that way. The
//! *ranking* still happens on device, inside `on_attn_proj`, from a device
//! tensor; only the one-time hand-off is on the host, which is also what makes
//! the selection independently checkable in `Output`.

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
    /// Prompt KV pages SnapKV keeps. Pages written after prefill are kept on
    /// top of this, as is the request's last page (the backend force-keeps it).
    #[serde(default = "default_page_budget")]
    page_budget: u32,
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
    0x5A4Bu32
}
fn default_page_budget() -> u32 {
    4
}

/// The mass handed to pages that did not exist at prefill time, so they sort
/// above every prompt page. One page cannot exceed `layers` in observed mass
/// (each layer contributes one distribution, summing to 1 over the prefix), and
/// no model in range has 1e6 layers.
const ALWAYS_KEEP_MASS: f32 = 1.0e6;

/// Largest entry of the tie-break ramp given to prompt pages before anything has
/// been observed. Only reachable if the capture returns an all-zero row, which
/// would itself be a bug -- but a tie among all pages would then evict real
/// context arbitrarily, so the ramp degrades that to "keep the earliest pages",
/// the StreamingLLM prior.
const SEED_SCALE: f32 = 1e-4;

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    text: String,
    count: usize,
    /// The static ceiling the program declared for the score row.
    kv_max: u32,
    /// Prompt length in tokens, i.e. the KV length the observed fire had.
    prompt_len: u32,
    page_budget: u32,
    /// Pages the prompt occupies. Only these compete for the budget.
    prompt_pages: u32,
    /// How many fires the prefill was split into, and how many tokens the
    /// FINAL (observed) one carried. Reported because the observation window
    /// is the last `window` rows *of that chunk*: if `prefill_final` drops
    /// below the driver's window the observation is silently truncated, and a
    /// truncated window still produces a plausible-looking ranking. With even
    /// chunking `prefill_final == prompt_len / prefill_chunks`, the largest a
    /// last chunk can be.
    prefill_chunks: u32,
    prefill_final: u32,
    /// KV page size, reported so a consumer can convert the driver's
    /// observation window (a token count) into the page span it covers.
    page_size: u32,
    /// Layers that reported a score row during the single prefill fire. This is
    /// the model's layer count, and it is what `score_mass` must equal.
    layers_observed: u32,
    /// Slots inside the prompt that carry a finite, non-negative score.
    live_scored: usize,
    /// Slots past the prompt carrying anything at all. Unlike H2O there is no
    /// seed ramp in the captured row, so this must be exactly zero: the prefill
    /// capture writes one entry per LIVE kv position and nothing else.
    tail_nonzero: usize,
    /// One past the highest slot carrying attention mass, i.e. the prompt
    /// length as the DRIVER saw it. Reported next to `prompt_len` because a
    /// disagreement between the two means the row describes different positions
    /// than the program thinks it does.
    observed_live: usize,
    scores_nan: usize,
    /// `Σ score` over the prompt. Each layer contributes one distribution over
    /// the prefix -- the fold averages over heads and window rows but not over
    /// layers -- so this must come out at `layers_observed`.
    score_mass: f32,
    /// SnapKV's keep-set among the PROMPT pages: the `page_budget` highest-mass
    /// ones. Recomputed on the host from the drained row, so it is an
    /// independent second opinion on the device's selection.
    kept_pages: Vec<u32>,
    /// The prompt page SnapKV would drop first (lowest mass).
    evicted_first: Option<u32>,
    /// Per-page mass over the prompt.
    page_mass: Vec<String>,
    /// The mass of the page holding the END of the prompt, as a fraction of the
    /// total. SnapKV's whole premise is that the observation window attends
    /// heavily to nearby positions, so this should be a large share -- and if it
    /// were not, the capture would not be describing the window it claims to.
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
    // Like Quest's `p_max` and H2O's, the program declares a static ceiling for
    // the score row; the backend refuses (rather than truncates) a request that
    // outgrows it. Sized off `max_pages` so it is an exact multiple of the page
    // geometry the driver derives its own length from.
    let kv_max = max_pages * page_size;
    let p_max = max_pages;
    let prompt_pages = n.div_ceil(page_size).min(p_max);
    let page_budget = input.page_budget.min(prompt_pages);
    ws.reserve(max_pages).context("reserve KV")?;

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);

    // ── PREFILL (chunked), WITH the tap on the FINAL chunk. ──
    //
    // This is the inversion that defines SnapKV: every other Track B policy
    // taps the decode fire and leaves prefill alone. Here prefill is the only
    // fire that is observed at all -- which is exactly what makes chunking it
    // delicate, and why the tap is attached where it is.
    //
    // The capture records the last `window` query rows OF THE FIRE. For a
    // one-shot prefill that is the last `window` tokens of the prompt, which is
    // SnapKV's definition. Under chunking only the FINAL chunk's window is the
    // prompt's tail; an earlier chunk's window is a well-defined observation of
    // an earlier position, but it is not the quantity SnapKV selects on. So the
    // tap goes on the final chunk alone, and the earlier chunks run the plain
    // prefill -- which also means they do not pay the capture variant's cost.
    //
    // The split is `prefill_chunks`, whose docs explain why it spreads the
    // remainder over the FIRST chunks: a "C tokens at a time until the
    // remainder" split can leave a final chunk of one token, and a final chunk
    // shorter than `window` silently truncates SnapKV's observation to whatever
    // was left over. That produces a plausible ranking, which is worse than an
    // implausible one. Using the shared helper also guarantees this inferlet
    // picks the SAME chunk boundaries as the baseline it is compared against,
    // so a text difference above the ceiling means a policy difference and not
    // a difference in the attention tile decomposition (§11.4).
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let spans = prefill_chunks(n, input.prefill_chunk);
    let k = spans.len() as u32;
    let pipe = Pipeline::new();

    // Non-final chunks: plain prefill, no tap, sampled token discarded.
    for &(base, end) in &spans[..spans.len() - 1] {
        let chunk = end - base;
        let toks_c = Channel::from(&prompt_i32[base as usize..end as usize]).named("toks_c");
        let embed_indptr_c = Channel::from([0u32, chunk]).named("embed_indptr_c");
        let positions_c = Channel::from_iter(base..end).named("positions_c");
        let pages_c = Channel::from_iter(0..max_pages).named("pages_c");
        let page_indptr_c = Channel::from([0u32, end.div_ceil(page_size)]).named("page_indptr_c");
        let w_slot_c = Channel::from_iter((base..end).map(|p| p / page_size)).named("w_slot_c");
        let w_off_c = Channel::from_iter((base..end).map(|p| p % page_size)).named("w_off_c");
        let kv_len_c = Channel::from([end]).named("kv_len_c");
        let rng_c = Channel::from([input.seed ^ 0x5bd1, 0]).named("rng_c");
        let tok_out_c = Channel::new([1], dtype::i32).named("tok_out_c");

        let fwd_c = ForwardPass::new();
        fwd_c.embed(&toks_c, &embed_indptr_c)?;
        fwd_c.attention(
            &ws,
            KvGeometry {
                readable_pages: ..,
                writable_pages: ..,
                kv_len: &kv_len_c,
                pages: &pages_c,
                page_indptr: &page_indptr_c,
                w_slot: &w_slot_c,
                w_off: &w_off_c,
                positions: &positions_c,
                mask: None,
            },
        )?;
        fwd_c.epilogue(move || {
            let r = rng_c.take();
            let logits = intrinsics::logits();
            let token = step(logits, temperature, &r);
            tok_out_c.put(&token);
            rng_c.put(&(&r + iota(2)));
        });
        fwd_c
            .submit(&pipe)
            .with_context(|| format!("prefill chunk submit @{base}"))?;
        tok_out_c
            .take_host::<Vec<i32>>()
            .await
            .with_context(|| format!("prefill chunk take @{base}"))?;
    }

    // ── FINAL CHUNK `[base, n)`: the observed one. ──
    let base = spans[spans.len() - 1].0;
    let tail = n - base;
    let toks_p = Channel::from(&prompt_i32[base as usize..n as usize]).named("toks_p");
    let embed_indptr_p = Channel::from([0u32, tail]).named("embed_indptr_p");
    let positions_p = Channel::from_iter(base..n).named("positions_p");
    let pages_p = Channel::from_iter(0..max_pages).named("pages_p");
    let page_indptr_p = Channel::from([0u32, n.div_ceil(page_size)]).named("page_indptr_p");
    let w_slot_p = Channel::from_iter((base..n).map(|p| p / page_size)).named("w_slot_p");
    let w_off_p = Channel::from_iter((base..n).map(|p| p % page_size)).named("w_off_p");
    let kv_len_p = Channel::from([n]).named("kv_len_p");
    // Same salt as `naive-baseline`. The coherence test asks whether a keep-set
    // that keeps everything changes what the model produces, and that question
    // is only answerable if every other input is held fixed -- including the
    // Gumbel stream.
    let rng_p = Channel::from([input.seed ^ 0x5bd1, 0]).named("rng_p");
    let tok_out_p = Channel::new([1], dtype::i32).named("tok_out_p");

    // The observation accumulator. `on_attn` fires once per layer and the
    // inferlet cannot ask the model how many layers there are, so the layer
    // loop folds into a device-carried channel.
    //
    // Contract #1: a loop-carried channel must be `take`n before it is `put`,
    // or the dummy run that infers the geometry sees a port written but never
    // read. The seed is zeros -- unlike H2O's ramp -- because this accumulator
    // lives for exactly one fire and every slot it will be ranked on is written
    // by that fire's capture.
    let acc = Channel::from(vec![0.0f32; kv_max as usize]).named("snapkv_acc");
    let layer_ct = Channel::from([0u32]).named("snapkv_layers");
    let scores_out = Channel::new([kv_max], dtype::f32).named("snapkv_scores");
    let layers_out = Channel::new([1], dtype::u32).named("snapkv_layer_count");
    // The device's own fold, drained alongside the position row so the host can
    // check the two against each other rather than trusting either.
    let page_mass_out = Channel::new([p_max], dtype::f32).named("snapkv_page_mass");

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

    // ── THE TAP. Fires once per layer, AFTER that layer's attention. During a
    //    prefill fire the row is the mean over the observation window's query
    //    rows, which is precisely the quantity SnapKV selects on. ──
    fwd_p.on_attn(move || {
        let prev = acc.take();
        let ct = layer_ct.take();
        let scores = intrinsics::attn_score(kv_max);
        acc.put(&(&prev + &scores));
        layer_ct.put(&(&ct + 1u32));
    });

    fwd_p.epilogue(move || {
        let r = rng_p.take();
        let logits = intrinsics::logits();
        let token = step(logits, temperature, &r);
        tok_out_p.put(&token);
        rng_p.put(&(&r + iota(2)));

        // Fold positions into pages. `reduce_sum` reduces the last axis per row
        // and `kv_max = p_max * page_size` exactly (it was derived from the page
        // geometry), so the reshape is a reinterpretation rather than a resize.
        // Summing is the right collapse for a probability mass: a page's share
        // of the attention is the sum of its positions' shares.
        let folded = acc.take();
        let layers = layer_ct.take();
        scores_out.put(&folded);
        layers_out.put(&layers);
        page_mass_out.put(&reduce_sum(&reshape(&folded, [p_max, page_size])));
        acc.put(&folded);
        layer_ct.put(&layers);
    });

    fwd_p
        .submit(&pipe)
        .with_context(|| format!("prefill submit @{base}"))?;

    let g0 = tok_out_p.take_host::<i32>().await?;
    generated.push(g0 as u32);

    let prefill_scores = scores_out.take_host::<Vec<f32>>().await?;
    let layers_observed = layers_out.take_host::<u32>().await?;
    let device_page_mass = page_mass_out.take_host::<Vec<f32>>().await?;

    // ── DECODE LOOP (1-wide, run-ahead), enforcing the fixed keep-set. ──
    if generated.len() < max_tokens {
        let tok_in = Channel::from([g0]).named("tok_in");
        let rng = Channel::from([input.seed ^ 0x5bd1, 2]).named("rng");
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

        // The frozen selection. Prompt pages carry the observed mass (with the
        // tie-break ramp folded in far below it); everything past the prompt
        // carries `ALWAYS_KEEP_MASS`, so the rank predicate below spends exactly
        // `page_budget` of its width on prompt pages and the rest on pages that
        // did not exist when the selection was made.
        let mask_row: Vec<f32> = (0..p_max)
            .map(|p| {
                if p >= prompt_pages {
                    ALWAYS_KEEP_MASS
                } else {
                    device_page_mass.get(p as usize).copied().unwrap_or(0.0)
                        + SEED_SCALE * (prompt_pages - p) as f32 / prompt_pages as f32
                }
            })
            .collect();
        let page_mass = Channel::from(mask_row).named("snapkv_mask_row");
        let effective_budget = page_budget + (p_max - prompt_pages);

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

        // ── THE ENFORCEMENT. Fires once per layer, BEFORE that layer's
        //    attention, which is the only point at which the page list can
        //    still be narrowed. The row never changes: SnapKV selects once.
        //
        //    `pivot_threshold(.., rank_le(k))` is the top-k keep predicate in
        //    one op; `top_k` would need a second reduce and is a library region
        //    (a fusion barrier). The backend force-keeps the request's last page
        //    whatever this says -- it holds the token this fire is writing. ──
        fwd.on_attn_proj(move || {
            let mass = page_mass.take();
            intrinsics::kernel::attn_page_mask(&pivot_threshold(&mass, rank_le(effective_budget)));
            page_mass.put(&mass);
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
        });

        let budget_n = max_tokens - 1;
        run_ahead(&pipe, &fwd, budget_n as usize, async || {
            let t = tok_out
                .take_host::<i32>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            generated.push(t as u32);
            Ok(ControlFlow::Continue(()))
        })
        .await?;
    }
    pipe.close();

    let live = (n as usize).min(prefill_scores.len());

    // Host-side mirror of the fold the device just did. Recomputed rather than
    // taken on trust, so a disagreement between the two is visible in `Output`
    // instead of silently deciding the keep-set.
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
    let mut order: Vec<u32> = (0..prompt_pages).collect();
    order.sort_by(|&a, &b| {
        masses[b as usize]
            .partial_cmp(&masses[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.cmp(&a))
    });
    let evicted_first = order.last().copied();
    let mut kept: Vec<u32> = order.into_iter().take(page_budget as usize).collect();
    kept.sort_unstable();

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
        prefill_chunks: k,
        prefill_final: tail,
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
