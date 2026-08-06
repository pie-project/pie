//! Top-a sampling — a truncation floor that scales with the *square* of the
//! peak probability.
//!
//! `keep(x)  iff  p(x) >= a · p_max²`.
//!
//! The quadratic term is the whole idea, and it is what separates top-a from
//! min-p's linear `a · p_max`. When the model is confident (`p_max` near 1) the
//! floor is high and the candidate set collapses toward greedy. When the model
//! is unsure (`p_max` small) the floor falls off *quadratically* — twice as
//! fast as min-p — so the set opens up much more aggressively exactly where the
//! model has no strong opinion.
//!
//! Community method with no canonical paper; this follows the reference
//! implementation in KoboldAI and text-generation-webui.
//!
//! ## Cost
//!
//! Unlike every other truncation sampler here, top-a needs no sort and no
//! candidate bound: one `ReduceMax`, one multiply, one comparison over the
//! vocabulary. It is `O(vocab)` with no `k_max` approximation anywhere, so the
//! mask it produces is exact.
//!
//! The most likely token always survives, because `p_max >= a · p_max²`
//! whenever `a · p_max <= 1`, which holds for every `a <= 1`.
//!
//! ## Source
//!
//! No canonical paper. Reference implementations: KoboldAI —
//! <https://github.com/KoboldAI/KoboldAI-Client> — and
//! oobabooga/text-generation-webui.
//!
//! Faithfulness: **Exact**. See
//! `inference-time-algorithms/10-implementation-faithfulness-audit.md`.

use inferlet::ptir::attention::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_a")]
    a: f32,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_seed")]
    seed: u32,
}

fn default_prompt() -> String {
    "Write a short paragraph about top-a sampling.".into()
}

fn default_a() -> f32 {
    0.2
}

fn default_temperature() -> f32 {
    1.0
}

fn default_max_tokens() -> usize {
    32
}

fn default_seed() -> u32 {
    0x7ce1
}

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    text: String,
    count: usize,
    a: f32,
    /// Mean size of the retained set, in tokens. Should swing widely across a
    /// generation — that swing *is* the algorithm.
    mean_kept: f32,
    min_kept: u32,
    /// Mean probability mass retained.
    mean_mass: f32,
}

#[derive(Clone, Copy)]
struct Cfg {
    a: f32,
    temperature: f32,
}

/// The top-a keep-mask. Exact — no candidate bound, no sort.
///
/// Returns `(keep_mask, kept_count, kept_mass)`.
fn top_a_keep(logits: &Tensor, vocab: u32, a: f32) -> (Tensor, Tensor, Tensor) {
    let probs = softmax(logits);
    let p_max = reduce_max(&probs);
    let threshold = a * (&p_max * &p_max);
    let keep = ge(&probs, broadcast(&threshold, [vocab]));

    let zeros = broadcast(0.0f32, [vocab]);
    let kept_mass = reshape(reduce_sum(select(&keep, &probs, &zeros)), [1]);
    let kept = reshape(reduce_sum(cast(&keep, dtype::f32)), [1]);
    (keep, kept, kept_mass)
}

/// One sampling step: temperature, quadratic floor, Gumbel-max draw.
fn step(logits: Tensor, vocab: u32, cfg: Cfg, rng_state: &Tensor) -> (Tensor, Tensor, Tensor) {
    // Temperature first: `p_max` has to be measured on the distribution being
    // sampled from, or the floor tracks the wrong peak.
    let scaled = if cfg.temperature == 1.0 {
        logits
    } else {
        &logits / cfg.temperature
    };
    let (keep, kept, kept_mass) = top_a_keep(&scaled, vocab, cfg.a);
    let neg_inf = broadcast(f32::NEG_INFINITY, [vocab]);
    let masked = select(&keep, &scaled, &neg_inf);
    (gumbel_max(masked, rng_state), kept, kept_mass)
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if !input.a.is_finite() || input.a <= 0.0 || input.a > 1.0 {
        return Err("a must be finite and in (0, 1]".into());
    }
    if !input.temperature.is_finite() || input.temperature <= 0.0 {
        return Err("temperature must be finite and greater than 0".into());
    }
    let max_tokens = input.max_tokens;
    let vocab = model::output_vocab_size();
    let cfg = Cfg {
        a: input.a,
        temperature: input.temperature,
    };
    let ws = WorkingSet::new();
    let page_size = kv_page_size();

    if max_tokens == 0 {
        return Ok(Output {
            sampler: "top-a",
            text: String::new(),
            count: 0,
            a: cfg.a,
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
    let mut s1: Vec<f32> = Vec::with_capacity(max_tokens);
    let mut s2: Vec<f32> = Vec::with_capacity(max_tokens);

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
    let s1_out_p = Channel::new([1], dtype::f32).named("s1_out_p");
    let s2_out_p = Channel::new([1], dtype::f32).named("s2_out_p");

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
        let (token, a, b) = step(logits, vocab, cfg, &r);
        let r_next = &r + iota(2);
        tok_out_p.put(&token);
        s1_out_p.put(&a);
        s2_out_p.put(&b);
        rng_p.put(&r_next);
    });

    let pipe = Pipeline::new();
    fwd_p.submit(&pipe).context("prefill submit")?;

    let g0 = tok_out_p.take_host::<i32>().await?;
    let a0 = s1_out_p.take_host::<f32>().await?;
    let b0 = s2_out_p.take_host::<f32>().await?;
    generated.push(g0 as u32);
    s1.push(a0);
    s2.push(b0);

    // ── DECODE LOOP (1-wide, run-ahead). ──
    if generated.len() < max_tokens {
        let tok_in = Channel::from([g0]).named("tok_in");
        let rng = Channel::from([input.seed ^ 0x5bd1, 0]).named("rng");
        let tok_out = Channel::new([1], dtype::i32)
            .capacity(channel_capacity() as u32)
            .named("tok_out");
        let s1_out = Channel::new([1], dtype::f32)
            .capacity(channel_capacity() as u32)
            .named("s1_out");
        let s2_out = Channel::new([1], dtype::f32)
            .capacity(channel_capacity() as u32)
            .named("s2_out");
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
            let (token, a, b) = step(logits, vocab, cfg, &r);

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
            s1_out.put(&a);
            s2_out.put(&b);
            rng.put(&r_next);
        });

        let budget = max_tokens - 1;
        run_ahead(&pipe, &fwd, budget as usize, async || {
            let t = tok_out
                .take_host::<i32>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            let a = s1_out
                .take_host::<f32>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            let b = s2_out
                .take_host::<f32>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            generated.push(t as u32);
            s1.push(a);
            s2.push(b);
            Ok(ControlFlow::Continue(()))
        })
        .await?;
    }
    pipe.close();

    let mean_s1 = s1.iter().sum::<f32>() / s1.len() as f32;
    let mean_s2 = s2.iter().sum::<f32>() / s2.len() as f32;
    let min_s1 = s1.iter().fold(f32::INFINITY, |a, &b| a.min(b)).max(0.0);
    if min_s1 == 0.0 {
        return Err("top-a keep-set was empty — the peak token was masked out".into());
    }

    Ok(Output {
        sampler: "top-a",
        text: model::decode(&generated)?,
        count: generated.len(),
        a: cfg.a,
        mean_kept: mean_s1,
        min_kept: min_s1 as u32,
        mean_mass: mean_s2,
    })
}
