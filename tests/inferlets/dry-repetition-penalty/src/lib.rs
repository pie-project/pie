//! DRY — "Don't Repeat Yourself", the repetition penalty that penalizes
//! *sequences* rather than tokens.
//!
//! A plain repetition penalty charges a token for having appeared. That is a
//! blunt instrument: it equally punishes the `the` in a fluent sentence and the
//! fourth verbatim replay of a whole clause. DRY instead asks a sharper
//! question — *if I emit this token, how long a verbatim repeat would I be
//! extending?* — and charges only for that.
//!
//! ```text
//! n(t)     = length of the longest suffix of the history that, followed by t,
//!            has already occurred earlier in the history
//! penalty  = multiplier * base^(n - allowed_length)     when n >= allowed_length
//!          = 0                                          otherwise
//! ```
//!
//! The exponential is the whole design: continuing a 2-gram is nearly free,
//! continuing an 8-gram is unaffordable. Originally proposed by p-e-w for
//! text-generation-webui and since adopted by llama.cpp and KoboldCpp.
//!
//! ## Why this one needs the device
//!
//! Of everything in this directory, DRY has the heaviest per-step state. It is
//! not a histogram — it is the *entire token history*, re-scanned every step,
//! with a suffix match run against every position at once. A host-side
//! implementation must therefore ship the logits out and a decision back on
//! every single token, which is exactly what collapses run-ahead decoding.
//!
//! Here the history is an `[L]` i32 channel that the epilogue extends itself:
//!
//! ```text
//! hist_next = scatter_set(hist, hlen, sampled_token)
//! hlen_next = hlen + 1
//! ```
//!
//! and the suffix match is a `max_ngram`-deep unrolled comparison over all `L`
//! positions in parallel. The scan that a CPU writes as a nested loop becomes
//! `max_ngram` vector ops, and nothing crosses to the host.
//!
//! ## How the parallel suffix match works
//!
//! Let `last = hlen - 1` be the tail of the history. For every position `j` at
//! once, the loop maintains `alive[j]`, "the `d` tokens ending at `j` equal the
//! `d` tokens ending at `last`", and accumulates `m[j] = sum_d alive[j]`. After
//! `max_ngram` rounds `m[j]` is the common-suffix length at `j`, capped.
//!
//! `j` is restricted to `j < last`, so a position is only considered when
//! `hist[j + 1]` is a token that actually followed it. That `hist[j + 1]` is
//! the token DRY wants to charge, so the per-token penalty falls out of one
//! `scatter_add` per repeat length:
//!
//! ```text
//! cnt_n     = scatter_add(0, hist[j + 1], [m[j] >= n])
//! penalty   = select(cnt_n > 0, multiplier * base^(n - allowed), penalty)
//! ```
//!
//! run for `n` **ascending**, so the last write — the longest match — wins,
//! which is the maximum DRY asks for. `scatter_add(...) > 0` rather than
//! `scatter_set` because several positions can vote for the same token and
//! duplicate-index `scatter_set` has no defined winner.
//!
//! ## Cost
//!
//! `max_ngram` rounds of ~8 ops over `[L]`, then `max_ngram - allowed + 1`
//! scatters over `[vocab]`. `L` is a few hundred, so the scan is free; the
//! scatters dominate and are the reason `max_ngram` is capped at 16.
//!
//! ## Source
//!
//! p-e-w, *DRY* — first shipped in oobabooga/text-generation-webui, since
//! adopted by llama.cpp `src/llama-sampler.cpp` —
//! <https://github.com/ggml-org/llama.cpp> — and KoboldCpp.
//!
//! Faithfulness: **Faithful with two bounded deviations** — no sequence
//! breakers (they are strings, and mapping them to token ids is
//! tokenizer-dependent), and `max_ngram` caps the match length, so the penalty
//! saturates at `multiplier · base^(max_ngram − allowed_length)`. See
//! `inference-time-algorithms/10-implementation-faithfulness-audit.md`.

use inferlet::ptir::attention::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    /// Penalty scale. 0.0 disables DRY entirely.
    #[serde(default = "default_multiplier")]
    multiplier: f32,
    /// Exponential growth rate per token of repeat length.
    #[serde(default = "default_base")]
    base: f32,
    /// Repeats shorter than this are free.
    #[serde(default = "default_allowed_length")]
    allowed_length: u32,
    /// Longest repeat the scan can see; caps the penalty.
    #[serde(default = "default_max_ngram")]
    max_ngram: u32,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_seed")]
    seed: u32,
}

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    text: String,
    count: usize,
    multiplier: f32,
    base: f32,
    allowed_length: u32,
    max_ngram: u32,
    /// Mean number of vocabulary entries carrying a non-zero DRY penalty per
    /// step. Zero means no repeat ever reached `allowed_length`.
    mean_penalized: f32,
    /// Largest penalty applied to any token at any step. Grows as
    /// `base^(n - allowed_length)`, so this reports the longest repeat the
    /// sampler was asked to extend.
    peak_penalty: f32,
    /// Longest verbatim n-gram that still made it into the output. This is the
    /// number DRY exists to hold down.
    longest_repeat: usize,
}

fn default_prompt() -> String {
    "List three interesting facts about the ocean.".into()
}

fn default_multiplier() -> f32 {
    0.8
}

fn default_base() -> f32 {
    1.75
}

fn default_allowed_length() -> u32 {
    2
}

fn default_max_ngram() -> u32 {
    8
}

fn default_temperature() -> f32 {
    1.0
}

fn default_max_tokens() -> usize {
    32
}

fn default_seed() -> u32 {
    0x4d2b
}

#[derive(Clone, Copy)]
struct Cfg {
    multiplier: f32,
    base: f32,
    allowed_length: u32,
    max_ngram: u32,
    temperature: f32,
    /// History buffer length: prompt + everything that can still be generated.
    capacity: u32,
}

/// Common-suffix length at every history position, computed in parallel.
///
/// Returns `m` of shape `[L]`, where `m[j]` counts how many trailing tokens
/// ending at `j` match the trailing tokens ending at `hlen - 1`, capped at
/// `max_ngram`. Positions that cannot host a match — `j >= hlen - 1`, or a
/// window that would run off either end of the buffer — are held at zero by
/// `alive`, which never resurrects once cleared.
fn suffix_match(hist: &Tensor, hlen: &Tensor, cfg: Cfg) -> Tensor {
    let l = cfg.capacity;
    let pos = cast(iota(l), dtype::i32);
    let zero_i = broadcast(0i32, [l]);
    let last = cast(hlen, dtype::i32) - 1i32;

    // A position is a candidate only if the token after it was actually
    // written, which rules out the tail itself.
    let mut alive = lt(&pos, broadcast(&last, [l]));
    let mut m = broadcast(0.0f32, [l]);

    for d in 0..cfg.max_ngram {
        let d_i = d as i32;
        let d_row = broadcast(d_i, [l]);
        // Indices are clamped rather than guarded, so the gathers stay in
        // bounds; the corresponding `alive` terms discard the clamped lanes.
        let src = max_elem(&pos - &d_row, &zero_i);
        let tail = &last - d as i32;
        let window = gather(hist, &src);
        let target = gather(hist, max_elem(&tail, 0i32));

        let matches = eq(&window, broadcast(&target, [l]));
        let in_window = ge(&pos, &d_row);
        let tail_ok = broadcast(ge(&tail, 0i32), [l]);

        alive = and(and(alive, matches), and(in_window, tail_ok));
        m = &m + cast(&alive, dtype::f32);
    }
    m
}

/// Per-token DRY penalty over the whole vocabulary.
///
/// Returns `(penalty, penalized_count, peak_penalty)`.
fn dry_penalty(hist: &Tensor, hlen: &Tensor, vocab: u32, cfg: Cfg) -> (Tensor, Tensor, Tensor) {
    let l = cfg.capacity;
    let m = suffix_match(hist, hlen, cfg);

    // The token each candidate position would have us emit. The clamp keeps the
    // scatter in bounds over the buffer's unwritten tail, whose lanes carry a
    // zero vote anyway.
    let next_idx = min_elem(
        cast(iota(l), dtype::i32) + 1i32,
        broadcast(l as i32 - 1, [l]),
    );
    let next_tok = max_elem(gather(hist, &next_idx), broadcast(0i32, [l]));

    let vocab_zero = broadcast(0.0f32, [vocab]);
    let mut penalty = broadcast(0.0f32, [vocab]);
    // Ascending, so the longest match is written last and wins.
    for n in cfg.allowed_length..=cfg.max_ngram {
        let hit = ge(&m, broadcast(n as f32, [l]));
        let votes = scatter_add(&vocab_zero, &next_tok, cast(&hit, dtype::f32));
        let charge = cfg.multiplier * cfg.base.powi((n - cfg.allowed_length) as i32);
        penalty = select(
            gt(&votes, &vocab_zero),
            broadcast(charge, [vocab]),
            &penalty,
        );
    }

    let count = reshape(
        reduce_sum(cast(&gt(&penalty, &vocab_zero), dtype::f32)),
        [1],
    );
    let peak = reshape(reduce_max(&penalty), [1]);
    (penalty, count, peak)
}

/// One sampling step. Returns `(token, hist_next, penalized_count, peak)`.
fn step(
    logits: Tensor,
    vocab: u32,
    cfg: Cfg,
    hist: &Tensor,
    hlen: &Tensor,
    rng_state: &Tensor,
) -> (Tensor, Tensor, Tensor, Tensor) {
    let (penalty, count, peak) = dry_penalty(hist, hlen, vocab, cfg);
    // DRY is defined as a subtraction on raw logits, so temperature comes after
    // it — scaling first would make the penalty's strength depend on it.
    let penalized = &logits - &penalty;
    let scaled = if cfg.temperature == 1.0 {
        penalized
    } else {
        &penalized / cfg.temperature
    };
    let token = gumbel_max(scaled, rng_state);
    // `gumbel_max` reduces the row away, so `token` is rank-0 — which is
    // exactly the scalar-broadcast form `scatter_set` accepts for `vals`.
    let hist_next = scatter_set(hist, hlen, &token);
    (token, hist_next, count, peak)
}

/// Longest verbatim n-gram repeated anywhere in `toks`, measured on the host so
/// the device statistic can be checked against ground truth.
fn longest_repeat(toks: &[u32]) -> usize {
    let mut best = 0usize;
    for i in 0..toks.len() {
        for j in (i + 1)..toks.len() {
            let mut k = 0usize;
            while j + k < toks.len() && toks[i + k] == toks[j + k] {
                k += 1;
            }
            best = best.max(k);
        }
    }
    best
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if !input.multiplier.is_finite() || input.multiplier < 0.0 {
        return Err("multiplier must be finite and non-negative".into());
    }
    if !input.base.is_finite() || input.base <= 0.0 {
        return Err("base must be finite and greater than 0".into());
    }
    if input.allowed_length == 0 {
        return Err("allowed_length must be at least 1".into());
    }
    if input.max_ngram < input.allowed_length || input.max_ngram > 16 {
        return Err("max_ngram must satisfy allowed_length <= max_ngram <= 16".into());
    }
    if !input.temperature.is_finite() || input.temperature <= 0.0 {
        return Err("temperature must be finite and greater than 0".into());
    }

    let max_tokens = input.max_tokens;
    let vocab = model::output_vocab_size();
    let ws = WorkingSet::new();
    let page_size = kv_page_size();

    if max_tokens == 0 {
        return Ok(Output {
            sampler: "dry",
            text: String::new(),
            count: 0,
            multiplier: input.multiplier,
            base: input.base,
            allowed_length: input.allowed_length,
            max_ngram: input.max_ngram,
            mean_penalized: 0.0,
            peak_penalty: 0.0,
            longest_repeat: 0,
        });
    }

    let mut prompt = model::encode(&input.prompt);
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let max_pages = (n + max_tokens as u32 + 1).div_ceil(page_size).max(1);
    ws.reserve(max_pages).context("reserve KV")?;

    let cfg = Cfg {
        multiplier: input.multiplier,
        base: input.base,
        allowed_length: input.allowed_length,
        max_ngram: input.max_ngram,
        temperature: input.temperature,
        capacity: n + max_tokens as u32,
    };

    // The history buffer is the prompt padded out with a sentinel no token can
    // equal, so unwritten lanes never register a match.
    let mut history: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    history.resize(cfg.capacity as usize, -1);

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut counts: Vec<f32> = Vec::with_capacity(max_tokens);
    let mut peaks: Vec<f32> = Vec::with_capacity(max_tokens);

    // ── PREFILL FIRE (N-wide): first sampled token comes off the prompt. ──
    let toks_p = Channel::from_iter(history.iter().take(n as usize).copied()).named("toks_p");
    let embed_indptr_p = Channel::from([0u32, n]).named("embed_indptr_p");
    let positions_p = Channel::from_iter(0..n).named("positions_p");
    let pages_p = Channel::from_iter(0..max_pages).named("pages_p");
    let page_indptr_p = Channel::from([0u32, n.div_ceil(page_size)]).named("page_indptr_p");
    let w_slot_p = Channel::from_iter((0..n).map(|p| p / page_size)).named("w_slot_p");
    let w_off_p = Channel::from_iter((0..n).map(|p| p % page_size)).named("w_off_p");
    let kv_len_p = Channel::from([n]).named("kv_len_p");
    let rng_p = Channel::from([input.seed, 0]).named("rng_p");
    let hist_p = Channel::from(history.clone()).named("hist_p");
    let hlen_p = Channel::from([n]).named("hlen_p");
    let tok_out_p = Channel::new([1], dtype::i32).named("tok_out_p");
    let cnt_out_p = Channel::new([1], dtype::f32).named("cnt_out_p");
    let peak_out_p = Channel::new([1], dtype::f32).named("peak_out_p");

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
        let hist = hist_p.take();
        let hlen = hlen_p.take();
        let logits = intrinsics::logits();
        let (token, hist_next, count, peak) = step(logits, vocab, cfg, &hist, &hlen, &r);
        let r_next = &r + iota(2);
        tok_out_p.put(&token);
        cnt_out_p.put(&count);
        peak_out_p.put(&peak);
        hist_p.put(&hist_next);
        hlen_p.put(&hlen + 1u32);
        rng_p.put(&r_next);
    });

    let pipe = Pipeline::new();
    fwd_p.submit(&pipe).context("prefill submit")?;

    let g0 = tok_out_p.take_host::<i32>().await?;
    let c0 = cnt_out_p.take_host::<f32>().await?;
    let k0 = peak_out_p.take_host::<f32>().await?;
    generated.push(g0 as u32);
    counts.push(c0);
    peaks.push(k0);

    // ── DECODE LOOP (1-wide, run-ahead). ──
    if generated.len() < max_tokens {
        // The decode pass owns fresh channels, so the history has to be handed
        // over explicitly, already carrying the prefill's token.
        history[n as usize] = g0;

        let tok_in = Channel::from([g0]).named("tok_in");
        let rng = Channel::from([input.seed ^ 0x5bd1, 0]).named("rng");
        let hist_c = Channel::from(history.clone()).named("hist");
        let hlen_c = Channel::from([n + 1]).named("hlen");
        let tok_out = Channel::new([1], dtype::i32)
            .capacity(channel_capacity() as u32)
            .named("tok_out");
        let cnt_out = Channel::new([1], dtype::f32)
            .capacity(channel_capacity() as u32)
            .named("cnt_out");
        let peak_out = Channel::new([1], dtype::f32)
            .capacity(channel_capacity() as u32)
            .named("peak_out");
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
            let hist = hist_c.take();
            let hlen = hlen_c.take();
            let logits = intrinsics::logits();
            let (token, hist_next, count, peak) = step(logits, vocab, cfg, &hist, &hlen, &r);

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
            cnt_out.put(&count);
            peak_out.put(&peak);
            hist_c.put(&hist_next);
            hlen_c.put(&hlen + 1u32);
            rng.put(&r_next);
        });

        let budget = max_tokens - 1;
        run_ahead(&pipe, &fwd, budget as usize, async || {
            let t = tok_out
                .take_host::<i32>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            let c = cnt_out
                .take_host::<f32>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            let k = peak_out
                .take_host::<f32>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            generated.push(t as u32);
            counts.push(c);
            peaks.push(k);
            Ok(ControlFlow::Continue(()))
        })
        .await?;
    }
    pipe.close();

    // The history is device state, so prove the scan actually saw it: with a
    // stuck `hist` channel no suffix could ever grow and the penalty would stay
    // pinned at its prompt-only value forever.
    let peak = peaks.iter().fold(0.0f32, |a, &b| a.max(b));
    if cfg.multiplier > 0.0
        && generated.len() > 8
        && peak == 0.0
        && counts.iter().all(|&c| c == 0.0)
    {
        return Err(
            "no token was ever penalized — the history channel is not advancing".to_string(),
        );
    }

    let mut full = prompt.clone();
    full.extend_from_slice(&generated);
    Ok(Output {
        sampler: "dry",
        text: model::decode(&generated)?,
        count: generated.len(),
        multiplier: cfg.multiplier,
        base: cfg.base,
        allowed_length: cfg.allowed_length,
        max_ngram: cfg.max_ngram,
        mean_penalized: counts.iter().sum::<f32>() / counts.len() as f32,
        peak_penalty: peak,
        longest_repeat: longest_repeat(&full),
    })
}
