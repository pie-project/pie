//! **THE SPECULATIVE LOOP ON THE DEVICE, RECURRENT STATE INCLUDED.**
//!
//! `mtp-speculative-decoding`'s loop — draft from the model's own head,
//! verify against the trunk's argmax in the same fire, commit and re-draft
//! in the epilogue — on a HYBRID model, where three layers in four are a
//! gated-delta recurrence. The host drains committed tokens and watches for
//! a stop; it never learns the accepted count.
//!
//! ```text
//! window = [x, d_1 .. d_k]           the correction, then the head's chain
//! truth  = argmax(logits)            what the trunk says follows each row
//! m      = |longest prefix with d_{i+1} == truth_i|
//! commit = window[1 ..= m], truth_m  the accepted run, then the correction
//! next   = [truth_m, chain_m[0..k]]  the head's chain at the accepted row
//! fold   = m + 1                     the rows whose state was real
//! ```
//!
//! **ONE LANE OF `k + 1` ROWS, NOT `k + 1` LANES OF ONE ROW.** The attention
//! twin spreads the window over lanes because the decode envelope admitted one
//! token a lane. A recurrent state cannot take that shape: its scan runs a
//! lane's rows IN ORDER, and a window spread over lanes has no order. So the
//! window here is one lane whose row count is a host-known constant — a
//! seeded `[0, w]` split the pool-owned device-geometry class carries
//! (`lease::detect_pooled_device_geometry`) — and the staircase is the
//! prefill arm's own causal bound: row `i` at position `nb + i` reads its
//! prefix, itself, and nothing drafted after it. Every descriptor port is
//! re-published by the epilogue, the pages included, because the committed
//! length `nb` is the device's and no host could fold the geometry off it.
//!
//! **THE STATE IS DRIVEN THROUGH THE WINDOW VERB.** The working set's buffer
//! is two runs the runtime alternates; a fire scatters its `w` rows into one
//! and replays the first `fold` tokens of the other — the rows the previous
//! round accepted, plus its correction — persisting the bank exactly after
//! them (`RsVerb::Window`, `rs-window-decode`). `fold` is `rs-geometry.fold-len`,
//! a channel the epilogue puts: the runtime reads it off the device port
//! every fire and never learns the number. The rejected rows' state was
//! computed and never folded, which is the whole discipline.
//!
//! **THE ANSWER IS THE ANSWER, BY CONSTRUCTION.** Verification keeps the
//! trunk's argmax at every position, so the text equals greedy decoding
//! whatever the head proposes; `k = 0` runs the same loop with a one-row
//! window — the controlled A/B against `text-completion`. Acceptance is
//! reported, not asserted.

use inferlet::{chat, session};
use inferlet::eta::hybrid::prelude::*;
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
    /// IS THE CONTROLLED A/B**: a one-row window, nothing drafted, the same
    /// geometry, fire and commit arithmetic.
    #[serde(default = "default_k")]
    k: u32,
}

fn default_prompt() -> String {
    "The quick brown fox jumps over".into()
}

fn default_max_tokens() -> usize {
    64
}

fn default_k() -> u32 {
    1
}

fn default_system() -> String {
    "You are a helpful benchmarking assistant.".into()
}

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
    /// Per round, how many tokens were committed (the accepted run plus the
    /// correction) — what maps a token index back to its round.
    commits_trace: Vec<u32>,
}

impl Output {
    fn empty(k: u32, depth: u32) -> Output {
        Output {
            sampler: "rs-mtp-speculative-bench",
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
            commits_trace: Vec::new(),
        }
    }
}

/// The token slots of a window that is not there: `-1` embeds nothing.
const NONE: i32 = -1;

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if model::pass_kind() != model::ForwardKind::Hybrid {
        return Err("rs-mtp-speculative-decoding drives a recurrent state; this model has none \
                    (use mtp-speculative-decoding)"
            .into());
    }
    let depth = model::mtp_depth();
    if input.k > 0 && depth == 0 {
        return Err("this SKU ships no draft head (mtp_depth = 0); run with k = 0".into());
    }
    let k = input.k.min(depth);
    if input.max_tokens == 0 {
        return Ok(Output::empty(k, depth));
    }
    let w = k + 1;
    let page_size = kv_page_size();
    let rs_page = model::rs_buffer_page_size().max(1);

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

    // The prompt's KV, every token the host may keep, and one window of
    // rejected cells above the committed length that the next fire writes over.
    let ws = WorkingSet::new();
    let max_extent = n + input.max_tokens as u32 + w;
    let max_pages = max_extent.div_ceil(page_size);
    ws.reserve(max_pages).context("reserve KV")?;
    // TWO runs of one window each: the runtime alternates them per fire.
    let rs = RsWorkingSet::new();
    let run_pages = w.div_ceil(rs_page).max(1);
    rs.alloc_buffer(2 * run_pages).map_err(|why| format!("alloc rs window runs: {why}"))?;
    let rs_set = vec![rs];
    let pipe = Pipeline::new();

    // ── PREFILL: folds everything in the forward, chunked; the head runs over
    //    every prompt row (so its own cache row is complete) and the last
    //    chunk seeds the first window off its last row.
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let spans = prefill_chunks(n, None);
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
        let readout_p = Channel::from([len - 1]).named("readout_p");
        let seed_p = Channel::new([w], dtype::i32).named("seed_p");
        let last = at == last_span;

        let fwd_p = ForwardPass::new();
        fwd_p.embed(&toks_p, &indptr_p)?;
        fwd_p.readout(&readout_p)?;
        fwd_p.attention(
            Some(KvBinding {
                working_set: &ws,
                geometry: KvGeometry {
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
            }),
            &rs_set,
            RsGeometry {
                fold_len: None,
                buffer: 0..0,
            },
        )?;
        fwd_p.epilogue(move || {
            let t = reshape(reduce_argmax(intrinsics::logits()), [1]);
            let window = if k > 0 {
                // Reading the head's chain is what runs the head over this
                // chunk's rows (`Lane::drafts`); the window takes its first
                // `k` on the last chunk and every chunk writes its cache row.
                let chain = intrinsics::mtp_drafts(depth);
                let i = iota(w);
                let first = eq(&i, Tensor::constant(0u32));
                let j = min_elem(
                    max_elem(&i, Tensor::constant(1u32)) - Tensor::constant(1u32),
                    Tensor::constant(depth - 1),
                );
                select(&not(&first), gather(&chain, &j), broadcast(&t, [w]))
            } else {
                broadcast(&t, [w])
            };
            seed_p.put(&window);
        });
        fwd_p
            .submit(&pipe)
            .with_context(|| format!("prefill submit @{base}"))?;
        let got = seed_p
            .take_host::<Vec<i32>>()
            .await
            .with_context(|| format!("@{base}"))?;
        if last {
            seed = got;
        }
    }

    let mut generated: Vec<u32> = Vec::with_capacity(input.max_tokens);
    let x = seed[0] as u32;
    generated.push(x);
    let (mut rounds, mut drafted, mut accepted) = (0usize, 0usize, 0usize);
    let mut commits_trace: Vec<u32> = Vec::new();
    let mut stopped = stop_tokens.contains(&x) || generated.len() >= input.max_tokens;

    // ── DECODE: one window per fire, loop-carried on the device ──────────
    if !stopped {
        let win = Channel::from(seed.as_slice()).named("win");
        let base = Channel::from([n]).named("base");
        // ONE lane of `w` rows: a seeded split the classifier reads as such.
        let indptr_d = Channel::from([0u32, w]).named("embed_indptr");
        let readout_d = Channel::from_iter(0..w).named("readout");
        let positions = Channel::from_iter(n..n + w).named("positions");
        let pages = Channel::from_iter(0..max_pages).named("pages");
        let page_indptr = Channel::from([0u32, (n + w).div_ceil(page_size)]).named("page_indptr");
        let w_slot = Channel::from_iter((n..n + w).map(|p| p / page_size)).named("w_slot");
        let w_off = Channel::from_iter((n..n + w).map(|p| p % page_size)).named("w_off");
        let kv_len = Channel::from([n + w]).named("kv_len");
        // The fold: the device's number. The prefill folded everything, so
        // the first window replays nothing; every epilogue puts the next.
        let fold_len = Channel::from([0u32]).named("fold_len");
        let out = Channel::new([w], dtype::i32)
            .capacity((channel_capacity() + 7 * frame_size()) as u32)
            .named("out");

        let fwd = ForwardPass::new();
        fwd.embed(&win, &indptr_d)?;
        fwd.readout(&readout_d)?;
        fwd.attention(
            Some(KvBinding {
                working_set: &ws,
                geometry: KvGeometry {
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
            }),
            &rs_set,
            RsGeometry {
                fold_len: Some(&fold_len),
                buffer: 0..run_pages,
            },
        )?;
        fwd.epilogue(move || {
            let window = win.take(); // [w] i32
            let b = base.take(); // [1] u32
            // One readout row per window row: `[w, vocab]`, or `[vocab]` at w = 1.
            let truth = reshape(reduce_argmax(intrinsics::logits()), [w]); // [w] i32
            let i = iota(w); // [w] u32
            let one = Tensor::constant(1u32);
            let none = broadcast(Tensor::constant(NONE), [w]);

            // ── verify: the longest prefix of drafts the trunk agreed with.
            //    At k = 0 it is zero, read off the logits so it stays the
            //    device's number like every other cell this loop carries.
            let m = if k > 0 {
                let proposed = gather(&window, iota(k) + &one); // window[1..=k]
                let said = gather(&truth, iota(k)); // truth[0..k]
                let hit = cast(eq(&proposed, &said), dtype::u32);
                reshape(reduce_sum(cumprod(&hit)), [1])
            } else {
                let t0 = gather(&truth, iota(1));
                reshape(cast(ne(&t0, &t0), dtype::u32), [1])
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
                select(&not(&first), gather(&chains, &at), &correction)
            } else {
                correction
            };
            win.put(&next);

            // ── the recurrent fold: the accepted rows and the correction's
            //    own row were real; the rest of this window never happened.
            fold_len.put(&(&m + &one));

            // ── the geometry of that window off the new committed length:
            //    rows at nb + i, the lane reading nb + w keys after its writes.
            let nb = &b + &m + &one; // [1]
            base.put(&nb);
            let p = broadcast(&nb, [w]) + &i; // [w]
            positions.put(&p);
            w_slot.put(&p / page_size);
            w_off.put(&p % page_size);
            let extent = &nb + Tensor::constant(w); // [1]
            kv_len.put(&extent);
            let page_count = extent.div_ceil(page_size); // [1]
            page_indptr.put(indptr(1, &page_count));
            // The page run is the whole reservation every round; re-published
            // so the pass states its WHOLE geometry in-graph — the pool-owned
            // device-geometry class, where the host leases and the device
            // resolves — rather than the envelope, whose host would have to
            // fold `nb` and cannot.
            pages.put(iota(max_pages));
        });

        let budget = input.max_tokens;
        run_ahead(&pipe, &fwd, budget, async || {
            let committed = out.take_host::<Vec<i32>>().await?;
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
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            })
        })
        .await?;
    }
    pipe.close();

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
        sampler: "rs-mtp-speculative-bench",
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
        commits_trace,
    })
}
