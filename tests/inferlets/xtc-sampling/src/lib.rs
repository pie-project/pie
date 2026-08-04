//! XTC — "exclude top choices".
//!
//! Every other truncation sampler removes the *tail*. XTC removes the *head*:
//! with probability `probability`, it drops every token whose probability is at
//! least `threshold` **except the least likely one among them**. The obvious
//! continuation is deliberately taken off the table.
//!
//! That makes it a creativity knob rather than a quality filter. It has to be
//! stacked underneath a normal truncation sampler, not used in place of one: on
//! its own it leaves the entire low-probability tail intact, so removing the
//! head just hands the step to the tail and the output degenerates into noise.
//! Every shipping implementation runs it inside a sampler chain for exactly
//! this reason, so this inferlet chains it with a min-p floor
//! (`p >= min_p · p_max`, the standard companion). Set `min_p = 0` to see the
//! unchained behaviour.
//!
//! The two stages compose in one direction only: min-p first, XTC second. Min-p
//! is defined against `p_max`, and XTC's whole purpose is to delete the token
//! that *is* `p_max`, so running it the other way would let the floor collapse
//! onto whatever survived.
//!
//! Community method (p-e-w), shipped in llama.cpp and text-generation-webui.
//!
//! ## The rule is self-limiting
//!
//! If exactly one token clears `threshold`, it is by definition the least
//! likely token that does, so `p > min_above` is false for it and nothing is
//! dropped. If none clear it, the same holds vacuously. No explicit "at least
//! two candidates" guard is needed — the comparison encodes it.
//!
//! ## Two independent random draws
//!
//! The Bernoulli gate must not correlate with the token draw, or the sampler
//! would preferentially fire on particular continuations. The gate therefore
//! runs on a state offset by the golden-ratio constant rather than on the
//! step's own `[key, ctr]`.
//!
//! ## Source
//!
//! p-e-w, *Exclude Top Choices (XTC)* — oobabooga/text-generation-webui
//! PR #6335, <https://github.com/oobabooga/text-generation-webui/pull/6335>.
//! Also in llama.cpp `src/llama-sampler.cpp`.
//!
//! Faithfulness: **Faithful with a bounded deviation** — the reference aborts
//! on newline and EOS tokens, which needs a tokenizer-specific token-ID table
//! this inferlet does not carry. See
//! `inference-time-algorithms/10-implementation-faithfulness-audit.md`.

use inferlet::ptir::attention::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_threshold")]
    threshold: f32,
    #[serde(default = "default_min_p")]
    min_p: f32,
    #[serde(default = "default_probability")]
    probability: f32,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_seed")]
    seed: u32,
}

fn default_prompt() -> String {
    "Write a short paragraph about unusual sampling strategies.".into()
}

fn default_threshold() -> f32 {
    0.1
}

fn default_min_p() -> f32 {
    0.02
}

fn default_probability() -> f32 {
    0.5
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
    threshold: f32,
    min_p: f32,
    probability: f32,
    /// Mean number of top choices removed per step, counting steps where the
    /// gate did not fire as zero.
    mean_dropped: f32,
    min_dropped: u32,
    /// Observed fire rate. Should converge on `probability`; a large gap means
    /// the gate is correlated with something it should not be.
    fire_rate: f32,
}

/// Decorrelates the Bernoulli gate from the token draw. Any odd constant
/// works; this is the 32-bit golden-ratio conjugate.
const GATE_OFFSET: u32 = 0x9e37_79b9;

#[derive(Clone, Copy)]
struct Cfg {
    threshold: f32,
    min_p: f32,
    probability: f32,
    temperature: f32,
}

/// The XTC keep-mask.
///
/// Returns `(keep_mask, dropped_count, fired)` where `fired` is 1.0 on steps
/// where the gate opened and 0.0 otherwise.
fn xtc_mask(logits: &Tensor, vocab: u32, cfg: Cfg, rng_state: &Tensor) -> (Tensor, Tensor, Tensor) {
    let probs = softmax(logits);
    let p_max = reduce_max(&probs);

    // Stage 1 — min-p floor. Inert when min_p == 0, since every probability is
    // >= 0.
    let floor = cfg.min_p * &p_max;
    let kept_floor = ge(&probs, broadcast(&floor, [vocab]));

    // Stage 2 — XTC, over the tokens that survived the floor.
    let above = and(&kept_floor, ge(&probs, broadcast(cfg.threshold, [vocab])));

    // Least likely token that still clears the threshold. Tokens below it are
    // sent to +inf so the reduction ignores them.
    let inf = broadcast(f32::INFINITY, [vocab]);
    let min_above = reduce_min(select(&above, &probs, &inf));

    // Bernoulli gate on a decorrelated state. `reduce_min` collapses the
    // one-element draw to a scalar.
    let gate_state = rng_state + broadcast(GATE_OFFSET, [2]);
    let u = reduce_min(rng(&gate_state, [1]));
    let fired = lt(&u, cfg.probability);

    let drop = and(
        broadcast(&fired, [vocab]),
        and(&above, gt(&probs, broadcast(&min_above, [vocab]))),
    );
    let dropped = reshape(reduce_sum(cast(&drop, dtype::f32)), [1]);
    let keep = and(&kept_floor, not(&drop));
    (keep, dropped, reshape(cast(&fired, dtype::f32), [1]))
}

/// One sampling step: temperature, head removal, Gumbel-max draw.
fn step(logits: Tensor, vocab: u32, cfg: Cfg, rng_state: &Tensor) -> (Tensor, Tensor, Tensor) {
    let scaled = if cfg.temperature == 1.0 {
        logits
    } else {
        &logits / cfg.temperature
    };
    let (keep, dropped, fired) = xtc_mask(&scaled, vocab, cfg, rng_state);
    let neg_inf = broadcast(f32::NEG_INFINITY, [vocab]);
    let masked = select(&keep, &scaled, &neg_inf);
    (gumbel_max(masked, rng_state), dropped, fired)
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if !input.threshold.is_finite() || input.threshold <= 0.0 || input.threshold > 1.0 {
        return Err("threshold must be finite and in (0, 1]".into());
    }
    if !input.min_p.is_finite() || !(0.0..1.0).contains(&input.min_p) {
        return Err("min_p must be finite and in [0, 1)".into());
    }
    if !input.probability.is_finite() || !(0.0..=1.0).contains(&input.probability) {
        return Err("probability must be finite and in [0, 1]".into());
    }
    if !input.temperature.is_finite() || input.temperature <= 0.0 {
        return Err("temperature must be finite and greater than 0".into());
    }
    let max_tokens = input.max_tokens;
    let vocab = model::output_vocab_size();
    let cfg = Cfg {
        threshold: input.threshold,
        min_p: input.min_p,
        probability: input.probability,
        temperature: input.temperature,
    };
    let ws = WorkingSet::new();
    let page_size = kv_page_size();

    if max_tokens == 0 {
        return Ok(Output {
            sampler: "xtc",
            text: String::new(),
            count: 0,
            threshold: cfg.threshold,
            min_p: cfg.min_p,
            probability: cfg.probability,
            mean_dropped: 0.0,
            min_dropped: 0,
            fire_rate: 0.0,
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

    Ok(Output {
        sampler: "xtc",
        text: model::decode(&generated)?,
        count: generated.len(),
        threshold: cfg.threshold,
        min_p: cfg.min_p,
        probability: cfg.probability,
        mean_dropped: mean_s1,
        min_dropped: min_s1 as u32,
        fire_rate: mean_s2,
    })
}
