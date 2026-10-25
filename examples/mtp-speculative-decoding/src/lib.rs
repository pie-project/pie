//! **THE SPECULATIVE LOOP, ON THE DEVICE.** Draft `k` tokens from the model's
//! own head, verify them against the trunk's argmax in the same fire, commit
//! the accepted prefix and the correction, and build the next window — all of
//! it in the epilogue. The host drains committed tokens and watches for a
//! stop; it never learns the accepted count between one fire and the next.
//!
//! ```text
//! window = [x, d_1 .. d_k]           the correction, then the head's chain
//! truth  = argmax(logits)            what the trunk says follows each row
//! m      = |longest prefix with d_{i+1} == truth_i|
//! commit = window[1 ..= m], truth_m  the accepted run, then the correction
//! next   = [truth_m, chain_m[0..k]]  the head's chain at the accepted row
//! ```
//!
//! **WHY THE HEAD'S CHAIN AT ROW `m` IS THE RIGHT ONE.** The head is fed
//! `(hidden_i, argmax_i)` and drafts what follows — the model text's
//! contract, `mtp_drafts` — so row `m`'s chain is conditioned on exactly the
//! token the next window starts with. That is what lets one fire both verify
//! this window and draft the next; a head fed the row's own token could not
//! (see `.wiki/big/dsv4.md` §3.4).
//!
//! **THE ANSWER IS THE ANSWER, BY CONSTRUCTION.** Verification keeps the
//! trunk's argmax at every position, so the text equals greedy decoding
//! whatever the head proposes; `k = 0` runs the same loop with a one-row
//! window and nothing drafted — the controlled A/B `test_eagle.py` reads.
//! Acceptance is reported, not asserted: it is a property of the head.
//!
//! **ROLLBACK IS OVERWRITE.** The next fire writes again the slots past the accepted run.

use inferlet::chat;
use inferlet::eta::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    /// Draft tokens per round, capped at the model's `mtp_depth`. **`k = 0`
    /// IS THE CONTROLLED A/B**: the window is one row, nothing is drafted or
    /// verified, and the loop is sequential greedy decoding through the same
    /// geometry, fire and commit arithmetic.
    #[serde(default = "default_k")]
    k: u32,
    /// DIAGNOSTIC: extra lanes carrying the correction token again, verified
    /// and rejected like any wrong draft, so a run with no drafts can be
    /// given the fire SHAPE of one with `pad` of them — separating "a wider
    /// window moves the answer" from "the draft head does".
    #[serde(default)]
    pad: u32,
    /// DIAGNOSTIC: record the first rounds' proposals and answers
    /// (`proposed_trace`/`truth_trace`) instead of the per-round margin — a
    /// pass binds at most twelve channels, and this is the twelfth.
    #[serde(default)]
    trace: bool,
    /// DIAGNOSTIC: record lane 0's top-1/top-2 logit margin every round
    /// (`margin_trace`). Opt-in because it is a `top_k` over the whole
    /// vocabulary in the epilogue.
    #[serde(default)]
    margin: bool,
}

fn default_prompt() -> String {
    "The quick brown fox jumps over".into()
}

fn default_max_tokens() -> usize {
    64
}

fn default_k() -> u32 {
    4
}

/// What the gate reads: the text is the claim, the counters say how much of
/// the mechanism ran.
#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    text: String,
    tokens: Vec<u32>,
    count: usize,
    /// Verify fires after the prefill.
    rounds: usize,
    /// Draft tokens proposed, `k` per round.
    drafted: usize,
    /// Draft tokens the trunk's own argmax agreed with.
    accepted: usize,
    /// `accepted / drafted`, or zero when nothing was drafted.
    acceptance_rate: f64,
    /// The draft-window length this run used (the input's, capped at depth).
    k: u32,
    /// The model's draft depth.
    depth: u32,
    prompt_tokens: u32,
    /// Per round, lane 0's top-1 minus top-2 logit — whether a divergence
    /// between two widths is a near-tie (the bf16 floor: a one-row and a
    /// many-row fire take different kernel paths and differ in the last
    /// bits) or a real gap (a mechanism fault).
    margin_trace: Vec<f32>,
    /// Per round, how many tokens were committed (the accepted run plus the
    /// correction) — what maps a token index back to its round.
    commits_trace: Vec<u32>,
    /// DIAGNOSTIC, the first rounds only: what the window proposed (its
    /// `k` drafts) and what the trunk said at each of those rows — so a head
    /// that proposes nonsense can be told from a verify that compares the
    /// wrong rows.
    proposed_trace: Vec<Vec<u32>>,
    truth_trace: Vec<Vec<u32>>,
}

impl Output {
    fn empty(k: u32, depth: u32) -> Output {
        Output {
            sampler: "mtp-speculative-decoding",
            text: String::new(),
            tokens: Vec::new(),
            count: 0,
            rounds: 0,
            drafted: 0,
            accepted: 0,
            acceptance_rate: 0.0,
            k,
            depth,
            prompt_tokens: 0,
            margin_trace: Vec::new(),
            commits_trace: Vec::new(),
            proposed_trace: Vec::new(),
            truth_trace: Vec::new(),
        }
    }
}

/// The token slots of a window that is not there: `-1` embeds nothing.
const NONE: i32 = -1;

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let depth = model::mtp_depth();
    let k = input.k.min(depth);
    if input.max_tokens == 0 {
        return Ok(Output::empty(k, depth));
    }
    let w = k + 1 + input.pad;

    // The raw encoding, so the identity against `naive-baseline` compares the
    // same context.
    // The model's opening (`<bos>` where it has one) before the raw text: a
    // gemma without it answers noise, and no sampler can be read against that.
    let mut prompt = chat::prefix();
    prompt.extend(model::encode(&input.prompt));
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let stop_tokens = chat::stop_tokens();

    let ws = WorkingSet::new();
    // The device picks the slots, so they are reserved up front; `2 * w` because it may run a
    // window past `max_tokens` and lay out the one after it.
    let max_extent = n + input.max_tokens as u32 + 2 * w;
    ws.reserve(max_extent)
        .context("reserve the decode's slots")?;
    let pipe = Pipeline::new();

    // The `mtp_drafts` pass is not cut by the runtime, so it is no wider than a fire reads out.
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let tail = u32::try_from(max_readout_rows(model::ReadoutKind::Logits)).unwrap_or(u32::MAX);
    let spans = if k > 0 && n > tail {
        vec![(0, n - tail), (n - tail, n)]
    } else {
        vec![(0, n)]
    };
    let last_span = spans.len() - 1;
    let mut seed: Vec<i32> = Vec::new();
    for (at, &(base, end)) in spans.iter().enumerate() {
        let len = end - base;
        let toks_p = Channel::from(&prompt_i32[base as usize..end as usize]).named("toks_p");
        let indptr_p = Channel::from([0u32, len]).named("embed_indptr_p");
        // The last row only: the first window is seeded off it.
        let readout_p = Channel::from([len - 1]).named("readout_p");
        let seed_p = Channel::new([w], dtype::i32).named("seed_p");
        let drafting = at == last_span && k > 0;

        let fwd_p = ForwardPass::new();
        fwd_p.embed(&toks_p, &indptr_p)?;
        fwd_p.readout(&readout_p)?;
        fwd_p.state(&[&ws], StateGeometry::folding())?;
        fwd_p.epilogue(move || {
            // One readout row: `logits()` is `[vocab]`, its argmax a scalar.
            let t = reshape(reduce_argmax(intrinsics::logits()), [1]);
            let i = iota(w);
            let first = eq(&i, Tensor::constant(0u32));
            let window = if drafting {
                // The head's chain at the readout row, `[depth]`; the window
                // takes its first `k`, and any pad lane carries `t` again.
                let chain = intrinsics::mtp_drafts(depth);
                let j = min_elem(
                    max_elem(&i, Tensor::constant(1u32)) - Tensor::constant(1u32),
                    Tensor::constant(depth - 1),
                );
                let drafted = and(not(&first), le(&i, Tensor::constant(k)));
                select(&drafted, gather(&chain, &j), broadcast(&t, [w]))
            } else {
                broadcast(&t, [w])
            };
            seed_p.put(&window);
        });
        fwd_p
            .submit(&pipe)
            .with_context(|| format!("prefill submit @{base}"))?;
        seed = seed_p
            .take_host::<Vec<i32>>()
            .await
            .with_context(|| format!("@{base}"))?;
    }

    let mut generated: Vec<u32> = Vec::with_capacity(input.max_tokens);
    let x = seed[0] as u32;
    generated.push(x);
    let (mut rounds, mut drafted, mut accepted) = (0usize, 0usize, 0usize);
    let mut margin_trace: Vec<f32> = Vec::new();
    let mut commits_trace: Vec<u32> = Vec::new();
    let mut proposed_trace: Vec<Vec<u32>> = Vec::new();
    let mut truth_trace: Vec<Vec<u32>> = Vec::new();
    let mut stopped = stop_tokens.contains(&x) || generated.len() >= input.max_tokens;

    // ── DECODE: one window per fire, loop-carried on the device ──────────
    //
    // **THE WINDOW IS `w` LANES OF ONE TOKEN, NOT ONE LANE OF `w` TOKENS.**
    // The runtime admits one token per lane; every lane's slot is written before any lane
    // attends, so lane `i + 1` sees lane `i`'s.
    if !stopped {
        let win = Channel::from(seed.as_slice()).named("win");
        let base = Channel::from([n]).named("base");
        let indptr_d = Channel::from_iter(0..=w).named("embed_indptr");
        let slots = Channel::from_iter(n..n + w).named("slots");
        let kv_len = Channel::from_iter((n..n + w).map(|p| p + 1)).named("kv_len");
        // Lane 0 carries the fold count for the shared working set; the others fold nothing more.
        let fold = Channel::from(vec![0u32; w as usize]).named("fold");
        let out = Channel::new([w], dtype::i32)
            .capacity(channel_capacity() as u32)
            .named("out");
        let tracing = input.trace;
        let margin = input.margin;
        // One diagnostic channel, the margin or the trace: a declared channel
        // takes a reader cell whether or not a stage touches it, and the pass
        // has room for twelve. f32 for both — a token id is exact in f32.
        let aux = Channel::new([if tracing { 2 * w } else { 1 }], dtype::f32)
            .capacity(channel_capacity() as u32)
            .named("aux");

        let fwd = ForwardPass::new();
        fwd.embed(&win, &indptr_d)?;
        let lanes: Vec<&WorkingSet> = (0..w).map(|_| &ws).collect();
        fwd.state(
            &lanes,
            StateGeometry {
                slots: Some(&slots),
                kv_len: Some(&kv_len),
                fold: Fold::Count(&fold),
                ..StateGeometry::default()
            },
        )?;
        fwd.epilogue(move || {
            let window = win.take(); // [w] i32
            let b = base.take(); // [1] u32
            // One readout row per lane: `[w, vocab]`, or `[vocab]` at w = 1.
            let logits = intrinsics::logits();
            let truth = reshape(reduce_argmax(&logits), [w]); // [w] i32
            if tracing {
                let j = iota(2 * w);
                let lower = lt(&j, Tensor::constant(w));
                let at = &j % Tensor::constant(w);
                aux.put(&cast(
                    select(&lower, gather(&window, &at), gather(&truth, &at)),
                    dtype::f32,
                ));
            } else if margin {
                let vocab = intrinsics::vocab();
                let row0 = reshape(gather(reshape(&logits, [w, vocab]), iota(1)), [vocab]);
                let (top, _) = top_k(&row0, 2);
                aux.put(&reshape(
                    gather(&top, iota(1)) - gather(&top, iota(1) + Tensor::constant(1u32)),
                    [1],
                ));
            } else {
                // A declared channel takes a reader cell every fire.
                aux.put(&broadcast(Tensor::constant(0f32), [1]));
            }
            let i = iota(w); // [w] u32
            let one = Tensor::constant(1u32);
            let none = broadcast(Tensor::constant(NONE), [w]);

            // ── verify: the longest prefix of drafts the trunk agreed with
            let m = if k > 0 {
                let proposed = gather(&window, iota(k) + &one); // window[1..=k]
                let said = gather(&truth, iota(k)); // truth[0..k]
                let hit = cast(eq(&proposed, &said), dtype::u32);
                reshape(reduce_sum(cumprod(&hit)), [1])
            } else {
                &b - &b
            };
            let mb = broadcast(&m, [w]);

            // ── commit: window[1..=m], then truth[m]; -1 past that
            let next_of = gather(&window, min_elem(&i + &one, Tensor::constant(w - 1)));
            let correction = broadcast(gather(&truth, &m), [w]);
            let committed = select(
                lt(&i, &mb),
                &next_of,
                select(eq(&i, &mb), &correction, &none),
            );
            out.put(&committed);

            // ── the next window: the correction, then row m's chain
            let first = eq(&i, Tensor::constant(0u32));
            let next = if k > 0 {
                let chains = intrinsics::mtp_drafts(w * depth); // [w·depth]
                let j = min_elem(max_elem(&i, &one) - &one, Tensor::constant(depth - 1));
                let at = &mb * Tensor::constant(depth) + &j;
                let drafted = and(not(&first), le(&i, Tensor::constant(k)));
                select(&drafted, gather(&chains, &at), &correction)
            } else {
                correction
            };
            win.put(&next);

            // ── the geometry of that window, from the new committed length:
            //    lane i at slot nb + i, reading nb + i + 1 slots; the next
            //    fire freezes the m + 1 this one accepted.
            let nb = &b + &m + &one; // [1]
            base.put(&nb);
            let p = broadcast(&nb, [w]) + &i; // [w]
            slots.put(&p);
            kv_len.put(&p + &one);
            let none_more = broadcast(Tensor::constant(0u32), [w]);
            fold.put(select(&first, broadcast(&m + &one, [w]), &none_more));
        });

        let budget = input.max_tokens;
        run_ahead(&pipe, &fwd, budget, async || {
            let committed = out.take_host::<Vec<i32>>().await?;
            if tracing {
                let traced = aux.take_host::<Vec<f32>>().await?;
                if proposed_trace.len() < 12 {
                    let (fired, answered) = traced.split_at(w as usize);
                    proposed_trace.push(fired.iter().skip(1).map(|&t| t as u32).collect());
                    truth_trace.push(answered.iter().map(|&t| t as u32).collect());
                }
            } else {
                let value = aux.take_host::<f32>().await?;
                if margin {
                    margin_trace.push(value);
                }
            }
            rounds += 1;
            drafted += k as usize;
            let live: Vec<u32> = committed
                .iter()
                .filter(|&&t| t != NONE)
                .map(|&t| t as u32)
                .collect();
            // Everything before the correction was a draft the trunk kept.
            accepted += live.len().saturating_sub(1);
            commits_trace.push(live.len() as u32);
            for t in live {
                generated.push(t);
                if stop_tokens.contains(&t) || generated.len() >= input.max_tokens {
                    stopped = true;
                    break;
                }
            }
            Ok(if stopped {
                std::ops::ControlFlow::Break(())
            } else {
                std::ops::ControlFlow::Continue(())
            })
        })
        .await?;
    }
    pipe.close();

    // A stop token ends the text; anything the last window committed past
    // it is not the answer.
    if let Some(at) = generated.iter().position(|t| stop_tokens.contains(t)) {
        generated.truncate(at + 1);
    }
    generated.truncate(input.max_tokens);
    let acceptance_rate = if drafted == 0 {
        0.0
    } else {
        accepted as f64 / drafted as f64
    };
    Ok(Output {
        sampler: "mtp-speculative-decoding",
        text: model::decode(&generated)?,
        count: generated.len(),
        tokens: generated,
        rounds,
        drafted,
        accepted,
        acceptance_rate,
        k,
        depth,
        prompt_tokens: n,
        margin_trace,
        commits_trace,
        proposed_trace,
        truth_trace,
    })
}
