//! Tail-free sampling — cut the tail where the sorted-probability curve
//! flattens out.
//!
//! Top-p asks *how much mass* to keep. Tail-free asks *where the distribution
//! stops changing*: it takes the second difference of the descending
//! probability curve — its discrete curvature — normalizes that into a
//! distribution of its own, and keeps the prefix holding `z` of the total
//! curvature. The head of the curve is where probabilities are still falling
//! steeply, so that is what survives; the flat tail contributes almost no
//! curvature and is dropped no matter how much mass it collectively holds.
//!
//! That is the property top-p cannot express. A long flat tail of near-equal
//! low probabilities can carry a large fraction of the mass, and top-p will
//! dutifully keep it. Tail-free sees a curvature of ~0 there and cuts.
//!
//! Community method with no canonical paper; this follows the reference
//! implementation shipped in llama.cpp and text-generation-webui.
//!
//! ## Why `k_max`
//!
//! The same bound as `locally-typical-sampling`: `top_k` is a schedule barrier,
//! so a ranking costs a region break regardless of `k`. The kernel itself is a
//! radix select plus a bitonic sort of the survivors, so it is effectively flat
//! in `k_max`. The curvature cut is computed over the `k_max` most likely
//! tokens — which is where essentially all of the curvature lives, since the
//! tail is flat by construction.
//!
//! ## Source
//!
//! Trenton Bricken, *Tail Free Sampling* (2019) —
//! <https://github.com/TrentBrick/TailFreeSampling>. Reference implementations:
//! llama.cpp `src/llama-sampler.cpp` and oobabooga/text-generation-webui
//! `modules/sampler_hijack.py`.
//!
//! Faithfulness: **Exact (equivalent form)**. This keeps ranks `0..j`, the
//! oobabooga convention, rather than the original gist's `0..j+1`; the gist
//! also indexes unsorted logits with a sorted rank, which is a plain bug. See
//! `inference-time-algorithms/10-implementation-faithfulness-audit.md`.

use inferlet::ptir::attention::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_z")]
    z: f32,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_k_max")]
    k_max: u32,
    #[serde(default = "default_seed")]
    seed: u32,
}

fn default_prompt() -> String {
    "Write a short paragraph about tail-free sampling.".into()
}

fn default_z() -> f32 {
    0.95
}

fn default_k_max() -> u32 {
    128
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
    z: f32,
    k_max: u32,
    /// Mean size of the retained set, in tokens.
    mean_kept: f32,
    min_kept: u32,
    /// Mean probability mass retained. Unlike top-p this is an *outcome*, not
    /// an input — a flat distribution can cut well below `z` of the mass.
    mean_mass: f32,
}

#[derive(Clone, Copy)]
struct Cfg {
    z: f32,
    k_max: u32,
    temperature: f32,
}

/// `x[i + 1]`, with the final element repeated so the difference vanishes at
/// the end of the candidate window instead of wrapping.
fn shift1(x: &Tensor, k: u32) -> Tensor {
    let idx = min_elem(iota(k) + 1u32, broadcast(k - 1, [k]));
    gather(x, &idx)
}

/// The tail-free keep-mask over the `k_max` most likely tokens.
///
/// Returns `(keep_mask, kept_count, kept_mass)`. The mask is never empty: the
/// most likely token has exclusive prefix curvature `0 < z`, so it always
/// survives the `lt` test.
fn tail_free_keep(logits: &Tensor, vocab: u32, k_max: u32, z: f32) -> (Tensor, Tensor, Tensor) {
    let probs = softmax(logits);
    // Probability is monotone in logits, so ranking by logit gives the
    // descending probability curve directly.
    let (_sorted, order) = top_k(logits, k_max);
    let p = gather(&probs, &order);

    let d1 = &p - shift1(&p, k_max);
    let d2 = &d1 - shift1(&d1, k_max);
    // |d2| without an abs op.
    let curvature = max_elem(&d2, -&d2);

    // A perfectly flat curve has zero curvature everywhere; floor the total so
    // the normalization cannot divide by zero.
    let total = max_elem(reduce_sum(&curvature), 1e-9f32);
    let norm = &curvature / broadcast(&total, [k_max]);
    // Exclusive prefix, so the most likely candidate always passes.
    let exclusive = cumsum(&norm) - &norm;
    let keep_sorted = lt(&exclusive, broadcast(z, [k_max]));

    let zeros = broadcast(0.0f32, [k_max]);
    let kept_mass = reshape(reduce_sum(select(&keep_sorted, &p, &zeros)), [1]);
    let kept = reshape(reduce_sum(cast(&keep_sorted, dtype::f32)), [1]);

    let base = broadcast(false, [vocab]);
    let keep = scatter_set(base, &order, keep_sorted);
    (keep, kept, kept_mass)
}

/// One sampling step: temperature, curvature mask, Gumbel-max draw.
fn step(logits: Tensor, vocab: u32, cfg: Cfg, rng_state: &Tensor) -> (Tensor, Tensor, Tensor) {
    // Temperature first: the curve whose curvature we measure has to be the
    // curve we actually sample from.
    let scaled = if cfg.temperature == 1.0 {
        logits
    } else {
        &logits / cfg.temperature
    };
    let (keep, kept, kept_mass) = tail_free_keep(&scaled, vocab, cfg.k_max, cfg.z);
    let neg_inf = broadcast(f32::NEG_INFINITY, [vocab]);
    let masked = select(&keep, &scaled, &neg_inf);
    (gumbel_max(masked, rng_state), kept, kept_mass)
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if !input.z.is_finite() || input.z <= 0.0 || input.z > 1.0 {
        return Err("z must be finite and in (0, 1]".into());
    }
    if !input.temperature.is_finite() || input.temperature <= 0.0 {
        return Err("temperature must be finite and greater than 0".into());
    }
    let vocab_probe = model::output_vocab_size();
    // Three points are the minimum for a second difference.
    if input.k_max < 3 || input.k_max > vocab_probe {
        return Err(format!("k_max must be in 3..={vocab_probe}"));
    }
    let max_tokens = input.max_tokens;
    let vocab = model::output_vocab_size();
    let cfg = Cfg {
        z: input.z,
        k_max: input.k_max,
        temperature: input.temperature,
    };
    let ws = WorkingSet::new();
    let page_size = kv_page_size();

    if max_tokens == 0 {
        return Ok(Output {
            sampler: "tail-free",
            text: String::new(),
            count: 0,
            z: cfg.z,
            k_max: cfg.k_max,
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
        return Err("tail-free keep-set was empty — the mask lost its floor".into());
    }

    Ok(Output {
        sampler: "tail-free",
        text: model::decode(&generated)?,
        count: generated.len(),
        z: cfg.z,
        k_max: cfg.k_max,
        mean_kept: mean_s1,
        min_kept: min_s1 as u32,
        mean_mass: mean_s2,
    })
}
