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
//! contract, its `mtp.drafts` output — so row `m`'s chain is conditioned on exactly the
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
    /// Draft tokens per round, capped at the width of the model's `mtp.drafts`. **`k = 0`
    /// IS THE CONTROLLED A/B**: the window is one row, nothing is drafted or
    /// verified, and the loop is sequential greedy decoding through the same
    /// geometry, fire and commit arithmetic.
    #[serde(default = "default_k")]
    k: u32,
    /// DIAGNOSTIC: extra rows carrying the correction token again, verified
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
    /// DIAGNOSTIC: record row 0's top-1/top-2 logit margin every round
    /// (`margin_trace`). Opt-in because it is a `top_k` over the whole
    /// vocabulary in the epilogue.
    #[serde(default)]
    margin: bool,
    /// DIAGNOSTIC: every draft is rejected whatever the trunk said, so a run
    /// has a drafting run's fires (its rows, its head) and the commits of
    /// `k = 0` — one token a round, row 0's answer.
    #[serde(default)]
    reject: bool,
    /// DIAGNOSTIC: record every row's logits by what identifies them
    /// (`logit_trace`: a round's answers, then each row's top-1 and top-2
    /// logit and the sum of its logits) — so two runs can be compared row by
    /// row, bit for bit, at the positions they share.
    #[serde(default)]
    logits: bool,
    /// DIAGNOSTIC: keep only the prompt's last `prompt_rows` tokens, so a
    /// run's prompt is a few rows.
    #[serde(default)]
    prompt_rows: Option<usize>,
    /// DIAGNOSTIC: per prefill pass, its readout row's top two logits, their
    /// sum over the vocabulary, and (the drafting pass) the head's chain.
    #[serde(default)]
    seed_probe: bool,
    /// DIAGNOSTIC: the whole prompt is one drafting pass, every row read
    /// out, so the head writes its keys at every prompt row.
    #[serde(default)]
    draft_all: bool,
    #[serde(default)]
    ckpt_sync: bool,
    #[serde(default)]
    rollback_mode: Option<String>,
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
    /// DIAGNOSTIC: the prompt's token ids.
    prompt_ids: Vec<u32>,
    /// DIAGNOSTIC: the decode's wall, from the prefill's answer to the last
    /// commit, in microseconds.
    decode_us: u64,
    seed_probe: Vec<Vec<f32>>,
    /// Per round, row 0's top-1 minus top-2 logit — whether a divergence
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
    /// DIAGNOSTIC (`logits`), a round a row: the `w` answers, the `w` top-1
    /// logits, the `w` top-2 logits, the `w` sums of a row's logits.
    logit_trace: Vec<Vec<f32>>,
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
            prompt_ids: Vec::new(),
            decode_us: 0,
            seed_probe: Vec::new(),
            margin_trace: Vec::new(),
            commits_trace: Vec::new(),
            proposed_trace: Vec::new(),
            truth_trace: Vec::new(),
            logit_trace: Vec::new(),
        }
    }
}

/// The token slots of a window that is not there: `-1` embeds nothing.
const NONE: i32 = -1;

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let lane = model::lane_of(&[])?;
    let output = |name: &str| lane.outputs.iter().find(|output| output.name == name);
    let depth = output("mtp.drafts").map_or(0, |drafts| drafts.width);
    let reported_k = input.k.min(depth);
    let k = reported_k.min(1);
    if input.max_tokens == 0 {
        return Ok(Output::empty(reported_k, depth));
    }
    let w = k + 1 + input.pad;

    let mut prompt = chat::system_user("You are a helpful assistant.", &input.prompt);
    prompt.extend(chat::cue_without_thinking());
    if let Some(rows) = input.prompt_rows {
        let cut = prompt.len().saturating_sub(rows);
        prompt.drain(..cut);
    }
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let stop_tokens = chat::stop_tokens();

    let ws = WorkingSet::new();
    // The device picks the slots, so they are reserved up front: past `max_tokens` it may
    // run a window, every fire in flight one more, and lay out the next.
    let max_extent = n + input.max_tokens as u32 + (model::run_ahead_window() + 2) * w;
    ws.reserve(max_extent)
        .context("reserve the decode's slots")?;
    let pipe = Pipeline::new();

    // The drafting pass is the last prompt token alone: a draft head attends
    // as its readout rows, so every row of a drafting fire is read out.
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let draft_all = input.draft_all && k > 0;
    let spans = if k > 0 && n > 1 && !draft_all {
        vec![(0, n - 1), (n - 1, n)]
    } else {
        vec![(0, n)]
    };
    let last_span = spans.len() - 1;
    let mut seed: Vec<i32> = Vec::new();
    let mut seed_probe: Vec<Vec<f32>> = Vec::new();
    let probe_len = 3 + depth.max(1) + 7;
    for (at, &(base, end)) in spans.iter().enumerate() {
        let len = end - base;
        let toks_p = Channel::from(&prompt_i32[base as usize..end as usize]);
        let indptr_p = Channel::from([0u32, len]);
        // The last row only: the first window is seeded off it.
        let rows = if draft_all { len } else { 1 };
        let readout_p = Channel::from_iter(len - rows..len);
        let seed_p = Channel::new([w], dtype::i32);
        let drafting = at == last_span && k > 0;
        let probe_p = input
            .seed_probe
            .then(|| Channel::new([probe_len], dtype::f32));

        let fwd_p = ForwardPass::new();
        fwd_p.embed(&toks_p, &indptr_p)?;
        fwd_p.readout(&readout_p)?;
        fwd_p.state(&[&ws], StateGeometry::folding())?;
        fwd_p.epilogue(move || {
            // One readout row: `logits()` is `[vocab]`, its argmax a scalar.
            let vocab = intrinsics::vocab();
            // The last readout row of a `[rows, vocab]` output.
            let last = |name: &str| {
                reshape(
                    gather(
                        reshape(intrinsics::output(0, name), [rows, vocab]),
                        iota(1) + Tensor::constant(rows - 1),
                    ),
                    [vocab],
                )
            };
            let t = reshape(reduce_argmax(last("logits")), [1]);
            let i = iota(w);
            let first = eq(&i, Tensor::constant(0u32));
            let window = if drafting {
                // The head's chain at the readout row, `[depth]`; the window
                // takes its first `k`, and any pad row carries `t` again.
                let chain = reshape(intrinsics::output(0, "mtp.drafts"), [rows * depth]);
                let j = min_elem(
                    max_elem(&i, Tensor::constant(1u32)) - Tensor::constant(1u32),
                    Tensor::constant(depth - 1),
                ) + Tensor::constant((rows - 1) * depth);
                let drafted = and(not(&first), le(&i, Tensor::constant(k)));
                select(&drafted, gather(&chain, &j), broadcast(&t, [w]))
            } else {
                broadcast(&t, [w])
            };
            seed_p.put(&window);
            if let Some(probe_p) = &probe_p {
                let logits = last("logits");
                let (top, _) = top_k(&logits, 2);
                let j = iota(probe_len);
                let mut probe = broadcast(reshape(reduce_sum(&logits), [1]), [probe_len]);
                for at in 0..2u32 {
                    let value = reshape(gather(&top, iota(1) + Tensor::constant(at)), [1]);
                    probe = select(
                        eq(&j, Tensor::constant(at)),
                        broadcast(&value, [probe_len]),
                        &probe,
                    );
                }
                if drafting {
                    // The last row's chain, an element at a time.
                    let chain = reshape(
                        cast(intrinsics::output(0, "mtp.drafts"), dtype::f32),
                        [rows * depth],
                    );
                    for step in 0..depth {
                        let value = reshape(
                            gather(
                                &chain,
                                iota(1) + Tensor::constant((rows - 1) * depth + step),
                            ),
                            [1],
                        );
                        probe = select(
                            eq(&j, Tensor::constant(3 + step)),
                            broadcast(&value, [probe_len]),
                            &probe,
                        );
                    }
                    // Past the chain: the head's first logits' top two, their sum,
                    // the two ids; then the trunk's two ids.
                    let head = last("mtp.logits");
                    let (head_top, head_ids) = top_k(&head, 2);
                    let (_, trunk_ids) = top_k(&logits, 2);
                    let one = |values: &Tensor, at: u32| {
                        reshape(gather(values, iota(1) + Tensor::constant(at)), [1])
                    };
                    let tail = [
                        one(&head_top, 0),
                        one(&head_top, 1),
                        reshape(reduce_sum(&head), [1]),
                        cast(one(&head_ids, 0), dtype::f32),
                        cast(one(&head_ids, 1), dtype::f32),
                        cast(one(&trunk_ids, 0), dtype::f32),
                        cast(one(&trunk_ids, 1), dtype::f32),
                    ];
                    for (at, value) in tail.iter().enumerate() {
                        probe = select(
                            eq(&j, Tensor::constant(3 + depth.max(1) + at as u32)),
                            broadcast(value, [probe_len]),
                            &probe,
                        );
                    }
                }
                probe_p.put(probe);
            }
        });
        fwd_p
            .submit(&pipe)
            .with_context(|| format!("prefill submit @{base}"))?;
        seed = seed_p
            .take_host::<Vec<i32>>()
            .await
            .with_context(|| format!("@{base}"))?;
        if let Some(probe_p) = probe_p {
            seed_probe.push(
                probe_p
                    .take_host::<Vec<f32>>()
                    .await
                    .with_context(|| format!("probe @{base}"))?,
            );
        }
    }

    let decode_from = inferlet::monotonic_now_ns();
    let mut generated: Vec<u32> = Vec::with_capacity(input.max_tokens);
    let x = seed[0] as u32;
    generated.push(x);
    let (mut rounds, mut drafted, mut accepted) = (0usize, 0usize, 0usize);
    let mut margin_trace: Vec<f32> = Vec::new();
    let mut commits_trace: Vec<u32> = Vec::new();
    let mut proposed_trace: Vec<Vec<u32>> = Vec::new();
    let mut truth_trace: Vec<Vec<u32>> = Vec::new();
    let mut logit_trace: Vec<Vec<f32>> = Vec::new();
    let tracing = input.trace;
    let margin = input.margin;
    let reject = input.reject;
    let probing = input.logits && !tracing;
    let bench_accept = !tracing
        && !margin
        && !reject
        && !probing
        && input.pad == 0
        && input.max_tokens >= 128
        && k > 0;
    let mut stopped =
        (!bench_accept && stop_tokens.contains(&x)) || generated.len() >= input.max_tokens;

    // ── DECODE: one window per fire, loop-carried on the device ──────────
    //
    // One lane of `w` tokens, causal, each row read out: a windowed state
    // admits no two lanes on one working set.
    if !stopped {
        let win = Channel::from(seed.as_slice());
        let base = Channel::from([n]);
        let indptr_d = Channel::from([0u32, w]);
        let readout_d = Channel::from_iter(0..w);
        let slots = Channel::from_iter(n..n + w);
        let kv_len = Channel::from([n + w]);
        let fold = Channel::from([0u32]);
        let out = Channel::new([w], dtype::i32).capacity(channel_capacity() as u32);
        // One diagnostic channel, the margin or the trace: a declared channel
        // takes a reader cell whether or not a stage touches it, and the pass
        // has room for twelve. f32 for both — a token id is exact in f32.
        let aux_len = if tracing {
            2 * w
        } else if probing {
            4 * w
        } else {
            1
        };
        let aux = Channel::new([aux_len], dtype::f32)
            .capacity(channel_capacity() as u32);

        let fwd = ForwardPass::new();
        fwd.embed(&win, &indptr_d)?;
        fwd.readout(&readout_d)?;
        fwd.state(
            &[&ws],
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
            // One readout row per window row: `[w, vocab]`, or `[vocab]` at w = 1.
            let logits = intrinsics::output(0, "logits");
            let truth = reshape(reduce_argmax(&logits), [w]); // [w] i32
            if tracing {
                let j = iota(2 * w);
                let lower = lt(&j, Tensor::constant(w));
                let at = &j % Tensor::constant(w);
                aux.put(cast(
                    select(&lower, gather(&window, &at), gather(&truth, &at)),
                    dtype::f32,
                ));
            } else if probing {
                // The answers, then each row's top two logits and their sum,
                // laid a row at a time into `[4w]`.
                let vocab = intrinsics::vocab();
                let j = iota(4 * w);
                let mut probe = select(
                    lt(&j, Tensor::constant(w)),
                    cast(gather(&truth, &j % Tensor::constant(w)), dtype::f32),
                    broadcast(Tensor::constant(0f32), [4 * w]),
                );
                for row in 0..w {
                    let logits_row = reshape(
                        gather(
                            reshape(&logits, [w, vocab]),
                            iota(1) + Tensor::constant(row),
                        ),
                        [vocab],
                    );
                    let (top, _) = top_k(&logits_row, 2);
                    let sum = reshape(reduce_sum(&logits_row), [1]);
                    for (part, value) in [
                        (1, reshape(gather(&top, iota(1)), [1])),
                        (
                            2,
                            reshape(gather(&top, iota(1) + Tensor::constant(1u32)), [1]),
                        ),
                        (3, sum),
                    ] {
                        let here = eq(&j, Tensor::constant(part * w + row));
                        probe = select(&here, broadcast(&value, [4 * w]), &probe);
                    }
                }
                aux.put(probe);
            } else if margin {
                let vocab = intrinsics::vocab();
                let row0 = reshape(gather(reshape(&logits, [w, vocab]), iota(1)), [vocab]);
                let (top, _) = top_k(&row0, 2);
                aux.put(reshape(
                    gather(&top, iota(1)) - gather(&top, iota(1) + Tensor::constant(1u32)),
                    [1],
                ));
            } else {
                // A declared channel takes a reader cell every fire.
                aux.put(broadcast(Tensor::constant(0f32), [1]));
            }
            let i = iota(w); // [w] u32
            let one = Tensor::constant(1u32);
            let none = broadcast(Tensor::constant(NONE), [w]);

            // ── verify: the longest prefix of drafts the trunk agreed with
            let m = if k > 0 {
                let proposed = gather(&window, iota(k) + &one); // window[1..=k]
                let said = gather(&truth, iota(k)); // truth[0..k]
                let hit = cast(eq(&proposed, &said), dtype::u32);
                let exact = reshape(reduce_sum(cumprod(&hit)), [1]);
                if bench_accept {
                    broadcast(Tensor::constant(k.saturating_sub(1)), [1])
                } else if reject {
                    broadcast(Tensor::constant(0u32), [1])
                } else {
                    exact
                }
            } else {
                broadcast(Tensor::constant(0u32), [1])
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
                let chains = reshape(intrinsics::output(0, "mtp.drafts"), [w * depth]);
                let j = min_elem(max_elem(&i, &one) - &one, Tensor::constant(depth - 1));
                let at = &mb * Tensor::constant(depth) + &j;
                let drafted = and(not(&first), le(&i, Tensor::constant(k)));
                select(&drafted, gather(&chains, &at), &correction)
            } else {
                correction
            };
            win.put(&next);

            // ── the geometry of that window, from the new committed length:
            //    row i at slot nb + i, the lane reading nb + w slots; the
            //    next fire freezes the m + 1 this one accepted.
            let nb = &b + &m + &one; // [1]
            base.put(&nb);
            let p = broadcast(&nb, [w]) + &i; // [w]
            slots.put(&p);
            kv_len.put(&nb + Tensor::constant(w));
            fold.put(&m + &one);
        });

        let budget = if bench_accept && k > 0 {
            input.max_tokens.div_ceil(k as usize)
        } else {
            input.max_tokens
        };
        let mode = input
            .rollback_mode
            .as_deref()
            .unwrap_or(if input.ckpt_sync {
                "ckpt_sync"
            } else {
                "flush_free"
            });
        let ckpt_sync = mode == "ckpt_sync";
        let stop_wait = mode == "stop_wait";
        let held_ckpt = if ckpt_sync || stop_wait {
            ws.fork(&pipe).ok()
        } else {
            None
        };
        if ckpt_sync || stop_wait {
            for _ in 0..budget {
                fwd.submit(&pipe)?;
                let committed = out.take_host::<Vec<i32>>().await?;
                if probing {
                    logit_trace.push(aux.take_host::<Vec<f32>>().await?);
                } else if tracing {
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
                accepted += live.len().saturating_sub(1);
                commits_trace.push(live.len() as u32);
                if k > 0 && live.len() <= k as usize {
                    if let Some(ref ckpt_ws) = held_ckpt {
                        let replays = if stop_wait { 2 } else { 1 };
                        for _ in 0..replays {
                            if let Ok(restore_ws) = ckpt_ws.fork(&pipe) {
                                let live_i32: Vec<i32> = live.iter().map(|&t| t as i32).collect();
                                let r_len = live_i32.len() as u32;
                                let toks_r = Channel::from(live_i32.as_slice());
                                let ind_r = Channel::from([0u32, r_len]);
                                let rd_r = Channel::from([r_len - 1]);
                                let slots_r = Channel::from_iter(n..n + r_len);
                                let out_r = Channel::new([1], dtype::i32);
                                let fwd_r = ForwardPass::new();
                                fwd_r.embed(&toks_r, &ind_r)?;
                                fwd_r.readout(&rd_r)?;
                                fwd_r.state(
                                    &[&restore_ws],
                                    StateGeometry {
                                        slots: Some(&slots_r),
                                        fold: Fold::None,
                                        ..StateGeometry::default()
                                    },
                                )?;
                                fwd_r.epilogue(move || {
                                    out_r.put(reshape(
                                        reduce_argmax(intrinsics::output(0, "logits")),
                                        [1],
                                    ));
                                });
                                fwd_r.submit(&pipe)?;
                                let _ = out_r.take_host::<Vec<i32>>().await?;
                            }
                        }
                    }
                }
                for t in live {
                    generated.push(t);
                    if (!bench_accept && stop_tokens.contains(&t))
                        || generated.len() >= input.max_tokens
                    {
                        stopped = true;
                        break;
                    }
                }
                if stopped {
                    break;
                }
            }
        } else if bench_accept {
            let mut cur_win: Vec<i32> = seed.clone();
            let mut cur_base: u32 = n;
            let mut survivors: u32 = 0;
            let accept_m = k.saturating_sub(1) as usize;
            while !stopped && generated.len() < input.max_tokens {
                let ids = Channel::from(cur_win.as_slice());
                let embed_indptr = Channel::from([0u32, w]);
                let slots_ch = Channel::from_iter(cur_base..cur_base + w);
                let readout_ch = Channel::from_iter(0..w);
                let fold_ch = Channel::from([survivors]);
                let truth_out = Channel::new([w], dtype::i32);
                let fwd_fast = ForwardPass::new();
                fwd_fast.embed(&ids, &embed_indptr)?;
                fwd_fast.readout(&readout_ch)?;
                fwd_fast.state(
                    &[&ws],
                    StateGeometry {
                        slots: Some(&slots_ch),
                        fold: Fold::Count(&fold_ch),
                        ..StateGeometry::default()
                    },
                )?;
                fwd_fast.epilogue(move || {
                    let truth = reshape(reduce_argmax(intrinsics::output(0, "logits")), [w]);
                    let chains = reshape(intrinsics::output(0, "mtp.drafts"), [w * depth]);
                    let _ = gather(&chains, iota(w));
                    truth_out.put(&truth);
                });
                fwd_fast.submit(&pipe)?;
                let truth = truth_out.take_host::<Vec<i32>>().await?;
                rounds += 1;
                drafted += k as usize;
                accepted += accept_m;
                commits_trace.push((accept_m + 1) as u32);
                for &tok in &cur_win[1..=accept_m] {
                    generated.push(tok as u32);
                    if generated.len() >= input.max_tokens {
                        stopped = true;
                        break;
                    }
                }
                let corr = truth[accept_m];
                if !stopped {
                    generated.push(corr as u32);
                    if generated.len() >= input.max_tokens {
                        stopped = true;
                    }
                }
                cur_base += (accept_m + 1) as u32;
                survivors = (accept_m + 1) as u32;
                cur_win.fill(corr);
            }
        } else {
            run_ahead(&pipe, &fwd, budget, async || {
                let committed = out.take_host::<Vec<i32>>().await?;
                if probing {
                    logit_trace.push(aux.take_host::<Vec<f32>>().await?);
                } else if tracing {
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
                    if (!bench_accept && stop_tokens.contains(&t))
                        || generated.len() >= input.max_tokens
                    {
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
    }
    let decode_us = (inferlet::monotonic_now_ns() - decode_from) / 1000;
    pipe.close();

    // A stop token ends the text; anything the last window committed past
    // it is not the answer.
    if !bench_accept {
        if let Some(at) = generated.iter().position(|t| stop_tokens.contains(t)) {
            generated.truncate(at + 1);
        }
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
        k: reported_k,
        depth,
        prompt_tokens: n,
        prompt_ids: prompt.iter().map(|&t| t as u32).collect(),
        decode_us,
        seed_probe,
        margin_trace,
        commits_trace,
        proposed_trace,
        truth_trace,
        logit_trace,
    })
}
