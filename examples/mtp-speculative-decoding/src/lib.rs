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
#[serde(deny_unknown_fields)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    /// Draft tokens per round, capped at the model's draft depth
    /// (`model::draft_depth`, the chain its head drafts). **`k = 0`
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
    #[serde(default)]
    ckpt_sync: bool,
    #[serde(default)]
    rollback_mode: Option<String>,
}

/// The prompt's prefill spans: with drafts on, runs of at most `limit` rows
/// each read out whole, so the head runs over every prompt row; without, the
/// whole prompt as one span.
fn prefill_spans(n: u32, drafting: bool, limit: u32) -> Vec<(u32, u32)> {
    if !drafting {
        return vec![(0, n)];
    }
    let limit = limit.max(1);
    (0..n.div_ceil(limit))
        .map(|at| (at * limit, ((at + 1) * limit).min(n)))
        .collect()
}

/// The most rows one prefill fire may read out: what its pass may bind and
/// what each output it reads holds a pass.
fn readout_rows(lane: &model::LaneDecl) -> u32 {
    lane.outputs
        .iter()
        .filter(|o| o.name == "logits" || o.name == "mtp.drafts")
        .map(|o| o.max_rows)
        .fold(model::max_pass_rows(), u32::min)
}

/// The window drafts what the caller asked for, up to the model's draft width.
fn working_k(asked: u32, depth: u32) -> u32 {
    asked.min(depth)
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
    /// The draft-window length the request asked for.
    asked: u32,
    /// The draft-window length this run used: `asked`, capped at `depth`.
    k: u32,
    /// The model's draft depth, the chain its text states it drafts
    /// (`model::draft_depth`), not the width of `mtp.drafts`.
    depth: u32,
    prompt_tokens: u32,
    /// Per round, row 0's top-1 minus top-2 logit — whether a divergence
    /// between two widths is a near-tie (the bf16 floor: a one-row and a
    /// many-row fire take different kernel paths and differ in the last
    /// bits) or a real gap (a mechanism fault).
    margin_trace: Vec<f32>,
    /// Per round, row 0's top-1 logit beside its margin: what sets the bf16
    /// step (`2^(exponent - 7)`) the margin is read against.
    top_trace: Vec<f32>,
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
    fn empty(asked: u32, k: u32, depth: u32) -> Output {
        Output {
            sampler: "mtp-speculative-decoding",
            text: String::new(),
            tokens: Vec::new(),
            count: 0,
            rounds: 0,
            drafted: 0,
            accepted: 0,
            acceptance_rate: 0.0,
            asked,
            k,
            depth,
            prompt_tokens: 0,
            margin_trace: Vec::new(),
            top_trace: Vec::new(),
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
    let lane = model::lane_of(&[])?;
    let output = |name: &str| lane.outputs.iter().find(|output| output.name == name);
    // The output's width is how the chains lie in memory; the depth is how
    // many of each chain's entries the head actually chained.
    let width = output("mtp.drafts").map_or(0, |drafts| drafts.width);
    let depth = model::draft_depth().min(width);
    let k = working_k(input.k, depth);
    if input.max_tokens == 0 {
        return Ok(Output::empty(input.k, k, depth));
    }
    let w = k + 1 + input.pad;

    let mut prompt = chat::system_user("You are a helpful assistant.", &input.prompt);
    prompt.extend(chat::cue_without_thinking());
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

    // A drafting fire runs the head over its readout rows and writes the
    // head's KV at their slots, so with drafts on the prompt is read out at
    // EVERY row: a prompt row the head never ran over would leave a slot it
    // later attends unwritten, read as whatever an earlier tenant left there.
    // The spans are as long as a pass may read out; without drafts the prompt
    // is one span and only its last row is read.
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let spans = prefill_spans(n, k > 0, readout_rows(&lane));
    let mut seed: Vec<i32> = Vec::new();
    for &(base, end) in &spans {
        let len = end - base;
        let toks_p = Channel::from(&prompt_i32[base as usize..end as usize]);
        let indptr_p = Channel::from([0u32, len]);
        let drafting = k > 0;
        let read = if drafting { 0 } else { len - 1 };
        let rows = len - read;
        let readout_p = Channel::from_iter(read..len);
        let seed_p = Channel::new([w], dtype::i32);

        let fwd_p = ForwardPass::new();
        fwd_p.embed(&toks_p, &indptr_p)?;
        fwd_p.readout(&readout_p)?;
        fwd_p.state(&[&ws], StateGeometry::folding())?;
        fwd_p.epilogue(move || {
            // The span's last readout row seeds the first window.
            let last = broadcast(Tensor::constant(rows - 1), [1]);
            let t = gather(&reshape(reduce_argmax(intrinsics::output(0, "logits")), [rows]), &last);
            let i = iota(w);
            let first = eq(&i, Tensor::constant(0u32));
            let window = if drafting {
                // The head's chain at that row, `[width]`; the window takes
                // its first `k`, and any pad row carries `t` again.
                let chains = reshape(intrinsics::output(0, "mtp.drafts"), [rows * width]);
                let j = min_elem(
                    max_elem(&i, Tensor::constant(1u32)) - Tensor::constant(1u32),
                    Tensor::constant(depth - 1),
                );
                let at = broadcast(Tensor::constant((rows - 1) * width), [w]) + &j;
                let drafted = and(not(&first), le(&i, Tensor::constant(k)));
                select(&drafted, gather(&chains, &at), broadcast(&t, [w]))
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
    let mut top_trace: Vec<f32> = Vec::new();
    let mut commits_trace: Vec<u32> = Vec::new();
    let mut proposed_trace: Vec<Vec<u32>> = Vec::new();
    let mut truth_trace: Vec<Vec<u32>> = Vec::new();
    let tracing = input.trace;
    let margin = input.margin;
    let mut stopped = stop_tokens.contains(&x) || generated.len() >= input.max_tokens;

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
        let aux = Channel::new([if tracing { 2 * w } else if margin { 2 } else { 1 }], dtype::f32)
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
            } else if margin {
                let vocab = intrinsics::vocab();
                let row0 = reshape(gather(reshape(&logits, [w, vocab]), iota(1)), [vocab]);
                let (top, _) = top_k(&row0, 2);
                // [top-1 - top-2, top-1]
                let first = gather(&top, broadcast(Tensor::constant(0u32), [2]));
                let second = gather(&top, broadcast(Tensor::constant(1u32), [2]));
                let lead = eq(iota(2), Tensor::constant(0u32));
                aux.put(select(&lead, &first - &second, &first));
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
                reshape(reduce_sum(cumprod(&hit)), [1])
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
                let chains = reshape(intrinsics::output(0, "mtp.drafts"), [w * width]);
                let j = min_elem(max_elem(&i, &one) - &one, Tensor::constant(depth - 1));
                let at = &mb * Tensor::constant(width) + &j;
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

        let budget = input.max_tokens;
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
                if tracing {
                    let traced = aux.take_host::<Vec<f32>>().await?;
                    if proposed_trace.len() < 12 {
                        let (fired, answered) = traced.split_at(w as usize);
                        proposed_trace.push(fired.iter().skip(1).map(|&t| t as u32).collect());
                        truth_trace.push(answered.iter().map(|&t| t as u32).collect());
                    }
                } else {
                    if margin {
                        let value = aux.take_host::<Vec<f32>>().await?;
                        margin_trace.push(value[0]);
                        top_trace.push(value[1]);
                    } else {
                        let _ = aux.take_host::<f32>().await?;
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
                    if stop_tokens.contains(&t) || generated.len() >= input.max_tokens
                    {
                        stopped = true;
                        break;
                    }
                }
                if stopped {
                    break;
                }
            }
        } else {
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
                    if margin {
                        let value = aux.take_host::<Vec<f32>>().await?;
                        margin_trace.push(value[0]);
                        top_trace.push(value[1]);
                    } else {
                        let _ = aux.take_host::<f32>().await?;
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
                    if stop_tokens.contains(&t) || generated.len() >= input.max_tokens
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
        asked: input.k,
        k,
        depth,
        prompt_tokens: n,
        margin_trace,
        top_trace,
        commits_trace,
        proposed_trace,
        truth_trace,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A key the request does not name is refused by name, not ignored: a
    /// removed option (`bench_accept`) cannot quietly run something else.
    #[test]
    fn an_unknown_key_is_refused() {
        let why = serde_json::from_str::<Input>(r#"{"bench_accept": true}"#)
            .err()
            .expect("an unknown key is refused")
            .to_string();
        assert!(why.contains("unknown field `bench_accept`"), "{why}");
    }

    /// With drafts on, every prompt row is in a span the head runs over.
    #[test]
    fn a_drafting_prefill_covers_every_prompt_row() {
        assert_eq!(prefill_spans(10, true, 4), [(0, 4), (4, 8), (8, 10)]);
        assert_eq!(prefill_spans(3, true, 64), [(0, 3)]);
        assert_eq!(prefill_spans(10, false, 4), [(0, 10)]);
        for (n, limit) in [(1, 1), (37, 8), (512, 512), (513, 512)] {
            let spans = prefill_spans(n, true, limit);
            assert_eq!(spans.first().unwrap().0, 0);
            assert_eq!(spans.last().unwrap().1, n);
            assert!(spans.windows(2).all(|p| p[0].1 == p[1].0));
            assert!(spans.iter().all(|&(a, b)| b > a && b - a <= limit));
        }
    }

    /// `k = 4` on a head that chains four is a window of four drafts, and
    /// on a head of one unchained layer a window of one.
    #[test]
    fn the_window_drafts_what_was_asked_up_to_the_model() {
        assert_eq!(working_k(4, 1), 1);
        assert_eq!(working_k(4, 4), 4);
        assert_eq!(working_k(2, 4), 2);
        assert_eq!(working_k(8, 4), 4);
        assert_eq!(working_k(4, 0), 0);
    }
}
