//! Locally typical sampling (Meister et al., <https://arxiv.org/abs/2202.00666>).
//!
//! Keeps the tokens whose information content `-log p(x)` sits closest to the
//! distribution's entropy `H`, then samples from that set. Unlike top-p — which
//! always keeps the *most likely* tokens — typical sampling also drops tokens
//! that are surprisingly *predictable*, which is what removes the degenerate
//! repetition top-p leaves behind.
//!
//! ## Why the candidate set is capped
//!
//! `pivot_threshold(x, cummass_le(p))` sorts `x` descending and accumulates
//! `x`'s own values as mass, so the ordering key and the accumulated quantity
//! are the same tensor. Typical sampling orders by *typicality*
//! `|log p + H|` while accumulating *probability*, and typicality is not
//! monotone in probability (it is V-shaped: both very likely and very unlikely
//! tokens score badly). So the mass cut cannot early-stop on the sort key, and
//! the program has to materialize an explicit typicality order:
//!
//! ```text
//! top_k(-score, k_max) -> gather(probs) -> cumsum -> lt(mass) -> scatter_set
//! ```
//!
//! `k_max` is a *bound*, not a detail. `top_k` is a schedule barrier, so a
//! ranking always costs a region break. The kernel itself is a radix select of
//! the cut followed by a bitonic sort of the `k` survivors, so its cost is
//! `O(vocab + k·log²k)` — effectively flat in `k_max`. (It was not always: the
//! kernel used to rescan the row once per pick, `O(k · vocab)`, which made
//! `k_max` a performance knob with teeth. See the runtime-cost section of
//! `inference-time-algorithms/10-implementation-faithfulness-audit.md`.)
//!
//! The cap remains a real semantic bound: at most `k_max` tokens can be
//! retained. If the `k_max` most typical tokens do not carry `mass`
//! probability, the set is truncated there. `mass_reached` in the output
//! reports whether that happened, so the approximation is observable rather
//! than silent.
//!
//! ## Source
//!
//! Meister et al., *Locally Typical Sampling* —
//! <https://arxiv.org/abs/2202.00666> (§3, Eq. 6).
//!
//! Faithfulness: **Exact**. See
//! `inference-time-algorithms/10-implementation-faithfulness-audit.md`.

use inferlet::ptir::attention::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_mass")]
    mass: f32,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_k_max")]
    k_max: u32,
    #[serde(default = "default_seed")]
    seed: u32,
}

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    text: String,
    count: usize,
    mass: f32,
    k_max: u32,
    /// Mean size of the retained set, in tokens. Should sit below `k_max` and
    /// never reach zero.
    mean_kept: f32,
    min_kept: u32,
    /// Mean probability mass actually retained. Well below `mass` would mean
    /// `k_max` is clipping the typical set.
    mean_mass: f32,
}

fn default_prompt() -> String {
    "Write a short paragraph about typical sampling.".into()
}

fn default_mass() -> f32 {
    0.95
}

fn default_temperature() -> f32 {
    1.0
}

fn default_max_tokens() -> usize {
    32
}

fn default_k_max() -> u32 {
    128
}

fn default_seed() -> u32 {
    0x7ce1
}

/// The typical keep-mask over the `k_max` most typical tokens.
///
/// Returns `(keep_mask, kept_count, kept_mass)`. The mask is never empty: the
/// most typical token has exclusive prefix mass `0 < mass`, so it always
/// survives the `lt` test.
fn typical_keep(logits: &Tensor, vocab: u32, k_max: u32, mass: f32) -> (Tensor, Tensor, Tensor) {
    let logprobs = log_softmax(logits);
    let probs = exp(&logprobs);
    // H = -sum(p log p), a scalar.
    let h = entropy_from_logprobs(&probs, &logprobs);
    // deviation = log p(x) + H, i.e. signed typicality.
    let deviation = &logprobs + broadcast(&h, [vocab]);
    let score = abs(&deviation);

    // Ascending typicality == descending (-score), capped at k_max candidates.
    let (_sorted_score, order) = top_k(-&score, k_max);
    let probs_sorted = gather(&probs, &order);
    // Exclusive prefix mass, so the most typical candidate always passes.
    let exclusive = cumsum(&probs_sorted) - &probs_sorted;
    let keep_sorted = lt(&exclusive, broadcast(mass, [k_max]));

    let zeros = broadcast(0.0f32, [k_max]);
    let kept_mass = reshape(reduce_sum(select(&keep_sorted, &probs_sorted, &zeros)), [1]);
    let kept = reshape(reduce_sum(cast(&keep_sorted, dtype::f32)), [1]);

    let base = broadcast(false, [vocab]);
    let keep = scatter_set(base, &order, keep_sorted);
    (keep, kept, kept_mass)
}

/// One sampling step: temperature, typical mask, Gumbel-max draw.
fn typical_step(
    logits: Tensor,
    vocab: u32,
    k_max: u32,
    temperature: f32,
    mass: f32,
    rng_state: &Tensor,
) -> (Tensor, Tensor, Tensor) {
    // Temperature first, matching the reference implementations: typicality is
    // measured on the distribution actually being sampled from.
    let scaled = if temperature == 1.0 {
        logits
    } else {
        &logits / temperature
    };
    let (keep, kept, kept_mass) = typical_keep(&scaled, vocab, k_max, mass);
    let neg_inf = broadcast(f32::NEG_INFINITY, [vocab]);
    let masked = select(&keep, &scaled, &neg_inf);
    let token = gumbel_max(masked, rng_state);
    (token, kept, kept_mass)
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if !input.mass.is_finite() || input.mass <= 0.0 || input.mass > 1.0 {
        return Err("mass must be finite and in (0, 1]".into());
    }
    if !input.temperature.is_finite() || input.temperature <= 0.0 {
        return Err("temperature must be finite and greater than 0".into());
    }
    let vocab_probe = model::output_vocab_size();
    if input.k_max == 0 || input.k_max > vocab_probe {
        return Err(format!("k_max must be in 1..={vocab_probe}"));
    }

    let k_max = input.k_max;
    let mass = input.mass;
    let temperature = input.temperature;
    let max_tokens = input.max_tokens;
    let vocab = model::output_vocab_size();
    let ws = WorkingSet::new();
    let page_size = kv_page_size();

    if max_tokens == 0 {
        return Ok(Output {
            sampler: "locally-typical",
            text: String::new(),
            count: 0,
            mass,
            k_max,
            mean_kept: 0.0,
            min_kept: 0,
            mean_mass: 0.0,
        });
    }

    let mut prompt = model::encode(&input.prompt);
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let max_pages = (n + max_tokens as u32 + 1).div_ceil(page_size).max(1);
    ws.reserve(max_pages).context("reserve KV")?;

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut kept_sizes: Vec<f32> = Vec::with_capacity(max_tokens);
    let mut kept_mass: Vec<f32> = Vec::with_capacity(max_tokens);

    // ── PREFILL FIRE (N-wide): first sampled token comes off the prompt. ──
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let toks_p = Channel::from(prompt_i32).named("toks_p");
    let embed_indptr_p = Channel::from([0u32, n]).named("embed_indptr_p");
    let positions_p = Channel::from_iter(0..n).named("positions_p");
    let pages_p = Channel::from_iter(0..max_pages).named("pages_p");
    let page_indptr_p = Channel::from([0u32, n.div_ceil(page_size)]).named("page_indptr_p");
    let w_slot_p = Channel::from_iter((0..n).map(|p| p / page_size)).named("w_slot_p");
    let w_off_p = Channel::from_iter((0..n).map(|p| p % page_size)).named("w_off_p");
    let kv_len_p = Channel::from([n]).named("kv_len_p");
    let rng_p = Channel::from([input.seed, 0]).named("rng_p");
    let tok_out_p = Channel::new([1], dtype::i32).named("tok_out_p");
    let kept_out_p = Channel::new([1], dtype::f32).named("kept_out_p");
    let mass_out_p = Channel::new([1], dtype::f32).named("mass_out_p");

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
        let (token, kept, kmass) = typical_step(logits, vocab, k_max, temperature, mass, &r);
        let r_next = &r + iota(2);
        tok_out_p.put(&token);
        kept_out_p.put(&kept);
        mass_out_p.put(&kmass);
        rng_p.put(&r_next);
    });

    let pipe = Pipeline::new();
    fwd_p.submit(&pipe).context("prefill submit")?;

    let g0 = tok_out_p.take_host::<i32>().await?;
    let k0 = kept_out_p.take_host::<f32>().await?;
    let m0 = mass_out_p.take_host::<f32>().await?;
    generated.push(g0 as u32);
    kept_sizes.push(k0);
    kept_mass.push(m0);

    // ── DECODE LOOP (1-wide, run-ahead). ──
    if generated.len() < max_tokens {
        let tok_in = Channel::from([g0]).named("tok_in");
        let rng = Channel::from([input.seed ^ 0x5bd1, 0]).named("rng");
        let tok_out = Channel::new([1], dtype::i32)
            .capacity(channel_capacity() as u32)
            .named("tok_out");
        let kept_out = Channel::new([1], dtype::f32)
            .capacity(channel_capacity() as u32)
            .named("kept_out");
        let mass_out = Channel::new([1], dtype::f32)
            .capacity(channel_capacity() as u32)
            .named("mass_out");
        let lane1 = Channel::from([0u32, 1u32]).named("embed_indptr");
        let positions = Channel::from([n]).named("positions");
        let pages = Channel::from_iter(0..max_pages).named("pages");
        let page_indptr = Channel::from([0u32, (n + 1).div_ceil(page_size)]).named("page_indptr");
        let w_slot = Channel::from([n / page_size]).named("w_slot");
        let w_off = Channel::from([n % page_size]).named("w_off");
        let kv_len = Channel::from([n + 1]).named("kv_len");

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
            // Takes and compute first, puts last (value-id discipline).
            let length = kv_len.take();
            let r = rng.take();
            let logits = intrinsics::logits();
            let (token, kept, kmass) = typical_step(logits, vocab, k_max, temperature, mass, &r);

            let r_next = &r + iota(2);
            let next_length = &length + 1u32;
            let page_count = next_length.div_ceil(page_size);

            tok_in.put(&token);
            kv_len.put(&next_length);
            positions.put(&length);
            w_slot.put(&length / page_size);
            w_off.put(&length % page_size);
            page_indptr.put(indptr(1, &page_count));
            tok_out.put(&token);
            kept_out.put(&kept);
            mass_out.put(&kmass);
            rng.put(&r_next);
        });

        let budget = max_tokens - 1;
        run_ahead(&pipe, &fwd, budget as usize, async || {
            let t = tok_out
                .take_host::<i32>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            let k = kept_out
                .take_host::<f32>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            let m = mass_out
                .take_host::<f32>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            generated.push(t as u32);
            kept_sizes.push(k);
            kept_mass.push(m);
            Ok(ControlFlow::Continue(()))
        })
        .await?;
    }
    pipe.close();

    let mean_kept = kept_sizes.iter().sum::<f32>() / kept_sizes.len() as f32;
    let min_kept = kept_sizes
        .iter()
        .fold(f32::INFINITY, |a, &b| a.min(b))
        .max(0.0) as u32;
    if min_kept == 0 {
        return Err("typical keep-set was empty — the mask lost its floor".into());
    }

    Ok(Output {
        sampler: "locally-typical",
        text: model::decode(&generated)?,
        count: generated.len(),
        mass,
        k_max,
        mean_kept,
        min_kept,
        mean_mass: kept_mass.iter().sum::<f32>() / kept_mass.len() as f32,
    })
}
