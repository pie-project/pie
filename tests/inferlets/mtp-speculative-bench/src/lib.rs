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
//! **ATTENTION-ONLY.** Rejected rows leave KV cells above the committed
//! length that the next fire overwrites (`nb + i`), so nothing is discarded
//! explicitly. A hybrid (recurrent-state) text binds the same `m` as its
//! `fold_len` — the fold-commit contract `model.wit` states — and is the RS
//! half's business, not this file's.

use inferlet::{chat, session};
use inferlet::eta::attention::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    /// Pre-tokenized prompt (the harness's, template applied); wins over
    /// `prompt` when given.
    #[serde(default)]
    prompt_tokens: Option<Vec<u32>>,
    #[serde(default = "default_system")]
    system: String,
    /// Drop the template's stop tokens so the loop runs to `max_tokens`.
    #[serde(default)]
    ignore_eos: bool,
    /// Announce `ready` and wait for the harness's `start` before the clock.
    #[serde(default)]
    wait_for_start: bool,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    /// Draft tokens per round, capped at the model's `mtp_depth`. **`k = 0`
    /// IS THE CONTROLLED A/B**: the window is one row, nothing is drafted or
    /// verified, and the loop is sequential greedy decoding through the same
    /// geometry, fire and commit arithmetic.
    #[serde(default = "default_k")]
    k: u32,
    /// DIAGNOSTIC: at `k = 0` the accepted count is a constant and the
    /// geometry folds on the host; this makes it device-decided (a zero read
    /// off the logits) so the one-row loop takes the pool-owned
    /// device-geometry class the wider windows take — separating "that class
    /// moves the answer" from "the drafting window does".
    #[serde(default)]
    device_geometry: bool,
    /// DIAGNOSTIC: extra lanes carrying the correction token again, verified
    /// and rejected like any wrong draft, so a run with no drafts can be
    /// given the fire SHAPE of one with `pad` of them — separating "a wider
    /// window moves the answer" from "the draft head does".
    #[serde(default)]
    pad: u32,
    /// DIAGNOSTIC: run only the first `max_layers` layers (the layerskip
    /// door), to bisect a divergence by depth.
    #[serde(default)]
    max_layers: Option<u32>,
    /// Prefill chunk cap in tokens. A streamed-expert artifact seats a bounded
    /// number of distinct experts per segment, and a long prompt routes past
    /// it; chunking the prefill keeps every fire under the seat count.
    #[serde(default)]
    prefill_chunk: Option<u32>,
    /// DIAGNOSTIC: record the first rounds' proposals and answers
    /// (`proposed_trace`/`truth_trace`) instead of the per-round margin — a
    /// pass binds at most twelve channels, and this is the twelfth.
    #[serde(default)]
    trace: bool,
    /// DIAGNOSTIC: record lane 0's top-1/top-2 logit margin every round
    /// (`margin_trace`). Opt-in because it is a `top_k` over the whole
    /// vocabulary in the epilogue, ~170 ms a fire on the full dsv4 — a third
    /// of the fire — and a gate's tie-judging is the only reader.
    #[serde(default)]
    margin: bool,
}

fn default_prompt() -> String {
    "The quick brown fox jumps over".into()
}

fn default_max_tokens() -> usize {
    64
}

fn default_system() -> String {
    "You are a helpful benchmarking assistant.".into()
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
    /// `text-completion-bench`'s envelope, for `benches/pie_bench.py`.
    num_prompt_tokens: usize,
    num_output_tokens: usize,
    token_ids: Vec<u32>,
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
    /// Where the generation started (prompt tokens) and the KV page size —
    /// what a divergence between two widths is read against.
    prompt_tokens: u32,
    page_size: u32,
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
            sampler: "mtp-speculative-bench",
            text: String::new(),
            tokens: Vec::new(),
            count: 0,
            num_prompt_tokens: 0,
            num_output_tokens: 0,
            token_ids: Vec::new(),
            rounds: 0,
            drafted: 0,
            accepted: 0,
            acceptance_rate: 0.0,
            k,
            depth,
            prompt_tokens: 0,
            page_size: 0,
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
    let device_geometry = input.device_geometry;
    if input.max_tokens == 0 {
        return Ok(Output::empty(k, depth));
    }
    let w = k + 1 + input.pad;
    let page_size = kv_page_size();

    // The raw encoding, so the identity against `naive-baseline` compares the
    // same context.
    // The model's opening (`<bos>` where it has one) before the raw text: a
    // gemma without it answers noise, and no sampler can be read against that.
    // The harness's tokens when it sent them; else the chat template the
    // plain bench applies (system + user + the assistant cue).
    let mut prompt: Vec<u32> = match &input.prompt_tokens {
        Some(tokens) => tokens.clone(),
        None => {
            let mut p = chat::system_user(&input.system, &input.prompt);
            p.extend(chat::cue());
            p
        }
    };
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let stop_tokens: Vec<u32> = if input.ignore_eos { Vec::new() } else { chat::stop_tokens() };
    if input.wait_for_start {
        session::send("ready");
        let _ = session::receive().await;
    }

    // One working set for the generation: the prompt's KV, every token the
    // host may keep, and one window of rejected cells above the committed
    // length that the next fire writes over.
    let ws = WorkingSet::new();
    // A round commits up to `w` tokens and the loop stops on the host, so
    // the device may have advanced one window past `max_tokens` and laid out
    // the window after that; a page of slack keeps every `pages` cut inside
    // the run every lane carries.
    let max_extent = n + input.max_tokens as u32 + 2 * w + page_size;
    let max_pages = max_extent.div_ceil(page_size);
    ws.reserve(max_pages).context("reserve KV")?;
    let pipe = Pipeline::new();

    // ── PREFILL: the prompt, chunked; the last chunk seeds the loop ──────
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let spans = prefill_chunks(n, input.prefill_chunk);
    let last_span = spans.len() - 1;
    let mut seed: Vec<i32> = Vec::new();
    for (at, &(base, end)) in spans.iter().enumerate() {
        let len = end - base;
        let toks_p = Channel::from(&prompt_i32[base as usize..end as usize]).named("toks_p");
        let indptr_p = Channel::from([0u32, len]).named("embed_indptr_p");
        let positions_p = Channel::from_iter(base..end).named("positions_p");
        let pages_p = Channel::from_iter(0..max_pages).named("pages_p");
        let page_indptr_p = Channel::from([0u32, end.div_ceil(page_size)]).named("page_indptr_p");
        let w_slot_p = Channel::from_iter((base..end).map(|p| p / page_size)).named("w_slot_p");
        let w_off_p = Channel::from_iter((base..end).map(|p| p % page_size)).named("w_off_p");
        let kv_len_p = Channel::from([end]).named("kv_len_p");
        // The last row only: the first window is seeded off it.
        let readout_p = Channel::from([len - 1]).named("readout_p");
        let seed_p = Channel::new([w], dtype::i32).named("seed_p");
        let drafting = at == last_span && k > 0;

        let fwd_p = ForwardPass::new();
        if let Some(layers) = input.max_layers {
            fwd_p.set_max_layers(layers)?;
        }
        fwd_p.embed(&toks_p, &indptr_p)?;
        fwd_p.readout(&readout_p)?;
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
        // Every chunk's put is drained; only the last chunk's seeds the loop.
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
    // The runtime's device-geometry class (the decode envelope) admits one
    // token per lane, and that shape is also the staircase without a mask:
    // lane `i` holds the window's `i`-th token at position `nb + i` and
    // reads `nb + i + 1` keys — its prefix, itself, and nothing drafted
    // after it — over the pages every lane shares (beam-search's shape,
    // with a per-lane `kv_len` where beam search has a mask). Every lane's
    // key is appended before any lane attends, so lane `i + 1` sees lane
    // `i`'s.
    if !stopped {
        let win = Channel::from(seed.as_slice()).named("win");
        let base = Channel::from([n]).named("base");
        let indptr_d = Channel::from_iter(0..=w).named("embed_indptr");
        let positions = Channel::from_iter(n..n + w).named("positions");
        let page_count0 = (n + w).div_ceil(page_size);
        let pages = Channel::from_iter((0..w * max_pages).map(|j| j % page_count0)).named("pages");
        let page_indptr =
            Channel::from_iter((0..=w).map(|lane| lane * page_count0)).named("page_indptr");
        let w_slot = Channel::from_iter((n..n + w).map(|p| p / page_size)).named("w_slot");
        let w_off = Channel::from_iter((n..n + w).map(|p| p % page_size)).named("w_off");
        let kv_len = Channel::from_iter((n..n + w).map(|p| p + 1)).named("kv_len");
        // The take-side rings a full frame of margin ABOVE the advertised
        // capacity, as `text-completion-bench` sizes its own: sized at exactly
        // `channel_capacity()` the runtime's ticket check skips continuations
        // that land inside its staging margin, the run-ahead collapses, and
        // every round's host turnaround lands on the critical path.
        let out = Channel::new([w], dtype::i32)
            .capacity((channel_capacity() + 7 * live_slots()) as u32)
            .named("out");
        // The twelfth channel is either the margin or the trace: a declared
        // channel is bound whether or not a stage touches it, and the pass
        // has room for twelve.
        let tracing = input.trace;
        let margin = input.margin;
        // One diagnostic channel, the margin or the trace: a declared channel
        // takes a reader cell whether or not a stage touches it, and the pass
        // has room for exactly this many. f32 for both — a token id is exact
        // in f32.
        let aux = Channel::new([if tracing { 2 * w } else { 1 }], dtype::f32)
            .capacity((channel_capacity() + 7 * live_slots()) as u32)
            .named("aux");

        let fwd = ForwardPass::new();
        if let Some(layers) = input.max_layers {
            fwd.set_max_layers(layers)?;
        }
        fwd.embed(&win, &indptr_d)?;
        fwd.attention(
            &ws,
            KvGeometry {
                readable_pages: ..,
                writable_pages: ..,
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
            } else if device_geometry {
                // Zero, but read off the logits so the host cannot fold it.
                let t0 = gather(&truth, iota(1));
                cast(ne(&t0, &t0), dtype::u32)
            } else {
                &b - &b
            };
            let mb = broadcast(&m, [w]);

            // ── commit: window[1..=m], then truth[m]; -1 past that
            let next_of = gather(&window, min_elem(&i + &one, Tensor::constant(w - 1)));
            let correction = broadcast(gather(&truth, &m), [w]);
            let committed = select(lt(&i, &mb), &next_of, select(eq(&i, &mb), &correction, &none));
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
            //    lane i at nb + i, reading nb + i + 1 keys, every lane over
            //    the same page run.
            let nb = &b + &m + &one; // [1]
            base.put(&nb);
            let p = broadcast(&nb, [w]) + &i; // [w]
            positions.put(&p);
            w_slot.put(&p / page_size);
            w_off.put(&p % page_size);
            kv_len.put(&p + &one);
            let page_count = (&nb + Tensor::constant(w)).div_ceil(page_size); // [1]
            pages.put(gather(
                iota(max_pages),
                iota(w * max_pages) % broadcast(&page_count, [w * max_pages]),
            ));
            page_indptr.put(iota(w + 1) * broadcast(&page_count, [w + 1]));
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
        sampler: "mtp-speculative-bench",
        text: model::decode(&generated)?,
        count: generated.len(),
        num_prompt_tokens: n as usize,
        num_output_tokens: generated.len(),
        token_ids: generated.clone(),
        tokens: generated,
        rounds,
        drafted,
        accepted,
        acceptance_rate,
        k,
        depth,
        prompt_tokens: n,
        page_size,
        margin_trace,
        commits_trace,
        proposed_trace,
        truth_trace,
    })
}
