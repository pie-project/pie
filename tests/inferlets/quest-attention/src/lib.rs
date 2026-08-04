//! Quest — query-aware sparsity via page criticality (Tang et al., 2024).
//!
//! <https://arxiv.org/abs/2406.10774>
//!
//! Quest observes, at every layer and every decode step, an upper bound on how
//! much each KV *page* could contribute to attention, and keeps only the
//! top-`page_budget` pages. The bound is exact and cheap: for a page holding
//! keys `K`, precompute the elementwise envelope `kmin = min(K, axis=pos)` and
//! `kmax = max(K, axis=pos)`; then for a query `q`,
//!
//! ```text
//!     criticality(page) = Σ_d max(q_d * kmin_d, q_d * kmax_d)  ≥  max_k q·k
//! ```
//!
//! This inferlet is the **observability half** of that algorithm: it runs the
//! real bound on real hardware, per layer, against the live KV cache, and
//! drains the per-page scores back to the host so the selection can be
//! verified against a reference. It does not yet elide the unselected pages
//! from the attention kernel, so it produces *bit-identical output to
//! `naive-baseline`* — which is precisely what makes it testable: any
//! divergence is a bug in the tap, not in the algorithm.
//!
//! ## What the backend does under `intrinsics::kernel::envelope_dot`
//!
//! The call lowers to `Op::KernelCall{name:"envelope_dot"}`, which the CUDA
//! driver binds to a second-party kernel only when the model reports the
//! `has_kv_envelopes` capability (native bf16 NHD pages, a post-rope query
//! family, and `tp == 1`). The kernel reads the *runtime* post-rope query row
//! for the lane, so the `p_max` argument declares only the score width; there
//! is no query operand to shape.
//!
//! ## Two documented deviations from the paper
//!
//! 1. **Union over heads.** Quest selects top-K pages *per head*. The paged
//!    attention layout here carries one page list per request, so the kernel
//!    emits `score[p] = max over kv_heads (Σ_{qh in group} Σ_d max(q·kmin, q·kmax))`
//!    — the union of the per-head selections. Quality ≥ Quest, sparsity ≤ Quest.
//!
//! 2. **The in-flight page is always kept.** The observation hook fires
//!    *before* the layer's KV append, so pages the current fire is still
//!    filling hold envelopes that predate their contents. Those pages score
//!    `+inf` and are unconditionally retained. This is fail-safe, and it
//!    coincides with Quest's own rule that the local window is never evicted.
//!    Pages outside the request's page list score `-inf`.
//!
//!    The request's page list is resolved by the kernel from the *device* page
//!    CSR. Under decode envelopes — the path this program runs on, since its
//!    page table is composed on device — the host CSR is only an upper bound
//!    (the declared `pages` channel capacity), so a `p_max` sized for the whole
//!    generation leaves genuinely-absent slots at every earlier step. Those
//!    slots read `-inf`, and `pages_absent` counts them.

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
    /// Pages Quest would keep per layer.
    #[serde(default = "default_page_budget")]
    page_budget: u32,
    /// Drain the per-page scores, layer count and kv_len back to the host each
    /// step. Every test here needs them; nothing about the POLICY does, since
    /// the ranking, the threshold and the mask are all computed and consumed
    /// on device. They are separable because they cost real time -- a
    /// per-layer fold over `p_max`, plus three device-to-host channel drains
    /// per step that a runahead pipeline has to wait on -- so leaving them on
    /// while benchmarking measures the instrumentation, not Quest.
    #[serde(default = "default_report")]
    report: bool,
    /// Size the page channel for this many generated tokens instead of
    /// `max_tokens`. `p_max` is what `envelope_dot` scores per layer, so a
    /// benchmark that differences two `max_tokens` would otherwise be
    /// comparing two different per-step workloads and would charge Quest for
    /// the difference. Setting this to the larger of the two endpoints makes
    /// the per-step cost identical. Defaults to `max_tokens`.
    #[serde(default)]
    reserve_tokens: Option<usize>,
    /// Prefill chunk width, clamped to the driver's `max_embed_length()`.
    /// Defaults to that limit, i.e. the fewest chunks the driver allows.
    ///
    /// Concatenating the chunks has to equal the one-shot fire, and the only
    /// decisive way to check that is to run the same prompt at two chunk
    /// widths and compare the text. That is what this exists for -- a prompt
    /// short enough to finish in seconds still exercises the multi-chunk path
    /// when the width is forced down, so the property is testable without a
    /// 16K-token prompt.
    #[serde(default)]
    prefill_chunk: Option<u32>,
}

fn default_report() -> bool {
    true
}

fn default_prompt() -> String {
    "The capital of France is".into()
}

fn default_temperature() -> f32 {
    1.0
}

fn default_max_tokens() -> usize {
    32
}

fn default_seed() -> u32 {
    0x9e57
}

fn default_page_budget() -> u32 {
    4
}

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    text: String,
    count: usize,
    /// Pages the program was allowed to score.
    max_pages: u32,
    /// KV page size, so a test can convert token counts into page counts.
    page_size: u32,
    /// The fire's `kv_len` at the last decode step — i.e. the KV length the
    /// driver derives its slot boundaries from. Reported so a test can predict
    /// `pages_finite` / `pages_pinned` / `pages_absent` by independent host
    /// arithmetic instead of asserting on whatever the driver happened to say.
    kv_len_last: u32,
    page_budget: u32,
    /// Layers whose envelope tap actually fired, counted on-device.
    layers_observed: u32,
    /// Per-page criticality, max-reduced over layers, from the last step.
    /// Non-finite bounds are meaningful here — `+inf` is an in-flight page and
    /// `-inf` is a page outside the request — and JSON has no encoding for
    /// them, so they are rendered as strings.
    page_scores: Vec<String>,
    /// How the last step's bounds broke down. `nan` is always a bug.
    pages_finite: usize,
    pages_pinned: usize,
    pages_absent: usize,
    pages_nan: usize,
    /// The pages Quest would have kept at the last step.
    kept_pages: Vec<u32>,
    /// True when at least one page carried a finite (i.e. genuinely scored)
    /// bound — the signal that the tap saw real envelopes, not just the
    /// fail-safe `+inf` of an in-flight page.
    scored_any_page: bool,
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
    if input.page_budget == 0 {
        return Err("page_budget must be at least 1".into());
    }
    let max_tokens = input.max_tokens;
    let temperature = input.temperature;
    let report = input.report;
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
    let p_max = max_pages;
    let budget = input.page_budget.min(p_max);
    ws.reserve(max_pages).context("reserve KV")?;

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);

    // ── PREFILL (chunked, C-wide): no Quest tap. ──
    //
    // Every page of a fresh prefill is in-flight by construction (`kv_before`
    // is zero), so the bound is `+inf` everywhere and carries no information.
    //
    // The prompt is split into `ceil(n / C)` chunks, `C = max_embed_length()`
    // -- the driver's structural per-launch token capacity. Quest exists for
    // long contexts, so a one-shot prefill is the wrong shape for it: on this
    // CUDA driver C is 8192, which put a hard ceiling on the context Quest
    // could ever be run at, well below the range where it pays for itself.
    //
    // Chunk `i` attends over the whole prefix written so far (`kv_len` is
    // cumulative, `page_indptr` covers every page up to the chunk's end) and
    // writes only its own tokens, which is what makes the concatenation of the
    // chunks equal to the one-shot fire. When the prompt fits in one chunk the
    // loop runs once and builds exactly the pass this used to build.
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

        // Every chunk samples; only the last chunk's token is the prompt's
        // continuation. The intermediate takes are not waste that can be
        // skipped -- the epilogue put has to be drained or the channel fills.
        g0 = tok_out_p
            .take_host::<i32>()
            .await
            .with_context(|| format!("@{base}"))?;
    }
    generated.push(g0 as u32);

    let mut last_scores: Vec<f32> = Vec::new();
    let mut layers_observed = 0u32;
    let mut kv_len_last = n;

    // ── DECODE LOOP (1-wide, run-ahead), with the Quest tap. ──
    if generated.len() < max_tokens {
        let tok_in = Channel::from([g0]).named("tok_in");
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

        // The per-layer accumulator. `OnAttnProj` fires once per layer and the
        // inferlet has no way to ask the model how many layers there are, so
        // the layer loop folds into a device-carried channel and the epilogue
        // — which fires exactly once per fire — publishes the fold. The seed
        // is `-inf` (the identity of `max`) and a zero layer counter.
        //
        // Report-only: nothing in the policy reads it, so it is absent when the
        // caller did not ask for a report.
        let acc = report
            .then(|| Channel::from(vec![f32::NEG_INFINITY; p_max as usize]).named("quest_acc"));
        let layer_ct = report.then(|| Channel::from([0u32]).named("quest_layers"));

        // Host drains. Present only when the caller asked to be told what
        // happened; see `Input::report`.
        let scores_out = report.then(|| {
            Channel::new([p_max], dtype::f32)
                .capacity(channel_capacity() as u32)
                .named("quest_scores")
        });
        let layers_out = report.then(|| {
            Channel::new([1], dtype::u32)
                .capacity(channel_capacity() as u32)
                .named("quest_layer_count")
        });
        let kvlen_out = report.then(|| {
            Channel::new([1], dtype::u32)
                .capacity(channel_capacity() as u32)
                .named("quest_kv_len")
        });

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

        // ── THE TAP. Fires once per layer, inside the fire, after the layer's
        //    Q/K/V projection and rope but before its KV append. ──
        fwd.on_attn_proj(move || {
            let scores = intrinsics::kernel::envelope_dot(p_max);

            // ── THE ENFORCEMENT. `pivot_threshold(.., rank_le(budget))` is the
            //    top-`budget` keep predicate over the page scores, and
            //    `attn_page_mask` is what makes this layer's attention read
            //    only those pages. Without this call the program is an
            //    OBSERVER: it computes a perfectly good ranking and the
            //    backend attends over everything anyway, which is why a
            //    Quest run has to be checked against `naive-baseline` for
            //    DIFFERENCE, not just for coherence.
            //
            //    The backend keeps the request's last page whatever this says
            //    -- it holds the token this fire is writing -- which is also
            //    Quest's own rule that the local window is never evicted.
            intrinsics::kernel::attn_page_mask(&pivot_threshold(&scores, rank_le(budget)));

            // Everything below is the report, not the policy: the mask above
            // has already been applied to this layer.
            if let (Some(acc), Some(layer_ct)) = (acc.as_ref(), layer_ct.as_ref()) {
                let prev = acc.take();
                let ct = layer_ct.take();
                acc.put(&max_elem(&prev, &scores));
                layer_ct.put(&(&ct + 1u32));
            }
        });

        fwd.epilogue(move || {
            let length = kv_len.take();
            if let Some(kvlen_out) = kvlen_out.as_ref() {
                kvlen_out.put(&length);
            }
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
            if let (Some(acc), Some(layer_ct), Some(scores_out), Some(layers_out)) = (
                acc.as_ref(),
                layer_ct.as_ref(),
                scores_out.as_ref(),
                layers_out.as_ref(),
            ) {
                let folded = acc.take();
                let layers = layer_ct.take();
                scores_out.put(&folded);
                layers_out.put(&layers);
                acc.put(&broadcast(f32::NEG_INFINITY, [p_max]));
                layer_ct.put(&reshape(&(&layers * 0u32), [1]));
            }
        });

        let budget_n = max_tokens - 1;
        run_ahead(&pipe, &fwd, budget_n as usize, async || {
            let t = tok_out
                .take_host::<i32>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            if let Some(ch) = scores_out.as_ref() {
                last_scores = ch
                    .take_host::<Vec<f32>>()
                    .await
                    .with_context(|| format!("@{}", generated.len()))?;
            }
            layers_observed = match layers_out.as_ref() {
                Some(ch) => ch
                    .take_host::<u32>()
                    .await
                    .with_context(|| format!("@{}", generated.len()))?,
                None => layers_observed,
            };
            kv_len_last = match kvlen_out.as_ref() {
                Some(ch) => ch
                    .take_host::<u32>()
                    .await
                    .with_context(|| format!("@{}", generated.len()))?,
                None => kv_len_last,
            };
            generated.push(t as u32);
            Ok(ControlFlow::Continue(()))
        })
        .await?;
    }
    pipe.close();

    // The host-side mirror of Quest's selection: keep the `budget` highest
    // bounds. `+inf` pages (in-flight, always kept) sort to the front for free.
    let mut order: Vec<u32> = (0..last_scores.len() as u32).collect();
    order.sort_by(|&a, &b| {
        last_scores[b as usize]
            .partial_cmp(&last_scores[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut kept: Vec<u32> = order
        .into_iter()
        .filter(|&p| last_scores[p as usize] != f32::NEG_INFINITY)
        .take(budget as usize)
        .collect();
    kept.sort_unstable();

    Ok(Output {
        sampler: "quest-attention",
        text: model::decode(&generated)?,
        count: generated.len(),
        max_pages,
        page_size,
        kv_len_last,
        page_budget: budget,
        layers_observed,
        scored_any_page: last_scores.iter().any(|s| s.is_finite()),
        pages_finite: last_scores.iter().filter(|s| s.is_finite()).count(),
        pages_pinned: last_scores.iter().filter(|s| **s == f32::INFINITY).count(),
        pages_absent: last_scores
            .iter()
            .filter(|s| **s == f32::NEG_INFINITY)
            .count(),
        pages_nan: last_scores.iter().filter(|s| s.is_nan()).count(),
        page_scores: last_scores
            .iter()
            .map(|s| {
                if s.is_nan() {
                    "nan".to_string()
                } else if *s == f32::INFINITY {
                    "inf".to_string()
                } else if *s == f32::NEG_INFINITY {
                    "-inf".to_string()
                } else {
                    format!("{s:.4}")
                }
            })
            .collect(),
        kept_pages: kept,
    })
}
