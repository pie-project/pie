//! **A BLOCK DRAFTER'S SPECULATIVE LOOP, ON THE DEVICE.** DeepSeek-V4.1's
//! DSpark head (or any block drafter whose anchor row proposes) drafts a
//! block, the trunk verifies it, and the next block is laid out — all of it
//! in the two passes' epilogues. The host drains committed tokens and
//! watches for a stop; it never learns the accepted count between rounds.
//!
//! ```text
//! draft    [a, NOISE x B-1] at b'..b'+B   B lanes, every key to the block's end
//!          -> d[0..B]                     the head's picks (`mtp.drafts`)
//! verify   [a, d[0..B]] at b'..b'+B+1     B+1 lanes, the trunk's staircase
//!          -> t[i] = argmax(row i)
//! keep     m = |longest prefix with d[i] == t[i]|, commit d[0..m], t[m]
//! next     a = t[m], b'' = b' + m + 1
//! ```
//!
//! A round is the two passes, draft then verify, and the runtime is kept a
//! window of them ahead (`run_ahead`'s shape over two passes). Each pass is
//! ONE lane of a host-constant row count (`rs-mtp-speculative-bench`'s
//! pool-owned device geometry): the draft's `B` rows read every key up to
//! the block's end (its `kv_len`), the head finding each row's depth in its
//! block from its position against that end; the verify's `B + 1` rows are
//! the prefill arm's causal staircase.
//!
//! **THE RECURRENCE IS DRIVEN THROUGH THE WINDOW VERB** on the verify (V4.1's
//! Engram n-gram state, its replay floor): two buffer runs the runtime
//! alternates, the epilogue putting the fold — the accepted rows and the
//! correction — so a rejected row's state is never folded. The draft's
//! rows are the head's alone and touch no state: it binds the plain fold.
//!
//! **THE ANSWER IS THE ANSWER, BY CONSTRUCTION.** Verification keeps the
//! trunk's argmax at every position, so the text equals greedy decoding
//! whatever the head proposes. Rejected rows leave KV cells above the
//! committed length that the next round writes over (the block's keys over
//! the head's windows, the verify's over the trunk's), so nothing is
//! discarded explicitly. Attention-only texts: a recurrence's fold is not
//! this file's.

use inferlet::eta::hybrid::prelude::*;
use inferlet::{chat, session};
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
    #[serde(default)]
    no_think: bool,
    /// Drop the template's stop tokens so the loop runs to `max_tokens`.
    #[serde(default)]
    ignore_eos: bool,
    /// Announce `ready` and wait for the harness's `start` before the clock.
    #[serde(default)]
    wait_for_start: bool,
    /// Send `t0` once the first token is known (the harness's TTFT).
    #[serde(default)]
    report_timing: bool,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    /// Proposals verified a round, at most the block (all of them, absent).
    #[serde(default)]
    k: Option<u32>,
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

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    text: String,
    count: usize,
    num_prompt_tokens: usize,
    num_output_tokens: usize,
    token_ids: Vec<u32>,
    /// Verify fires after the prefill.
    rounds: usize,
    /// Proposals verified, `k` a round.
    drafted: usize,
    /// Proposals the trunk's own argmax agreed with.
    accepted: usize,
    acceptance_rate: f64,
    k: u32,
    block: u32,
}

/// The token slots of a window that is not there: `-1` embeds nothing.
const NONE: i32 = -1;

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let Some(head) = model::draft_block() else {
        return Err("this deployment ships no block drafter".into());
    };
    if head.proposals_from != 0 {
        return Err("the device loop reads a head whose anchor row proposes (DSpark's)".into());
    }
    let block = head.rows;
    let k = input.k.unwrap_or(block).clamp(1, block);
    // The verify window: the anchor and `k` proposals.
    let w = k + 1;
    let noise = head.mask_token as i32;
    let page_size = kv_page_size();
    let rs_page = model::rs_buffer_page_size().max(1);

    let mut prompt: Vec<u32> = match &input.prompt_tokens {
        Some(tokens) => tokens.clone(),
        None => {
            let mut p = chat::system_user(&input.system, &input.prompt);
            p.extend(chat::cue_thinking(!input.no_think));
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

    // The prompt's KV, every token the host may keep, and the cells a round
    // writes above the committed length (a block, or a window) that the
    // next round writes over; the device may run a window of rounds past
    // the host's stop, and a page of slack keeps every cut inside the run.
    let ws = WorkingSet::new();
    let reach = block.max(w);
    let max_extent = n + input.max_tokens as u32 + 2 * reach + page_size;
    let max_pages = max_extent.div_ceil(page_size);
    ws.reserve(max_pages).context("reserve KV")?;
    // A hybrid text's recurrence: two runs of one verify window each, the
    // runtime alternating them per verify fire.
    let rs_set: Vec<RsWorkingSet> = match model::pass_kind() {
        model::ForwardKind::Attention => Vec::new(),
        model::ForwardKind::Hybrid => {
            let rs = RsWorkingSet::new();
            rs.alloc_buffer(2 * w.div_ceil(rs_page).max(1))
                .map_err(|why| format!("alloc rs window runs: {why}"))?;
            vec![rs]
        }
        _ => return Err("a block drafter's loop reads attention kv".into()),
    };
    let run_pages = if rs_set.is_empty() { 0 } else { w.div_ceil(rs_page).max(1) };
    let pipe = Pipeline::new();

    // ── the draft's inputs: one lane of the block's rows, every one put by
    //    a pass (the prefill's last chunk lays out the first block, each
    //    verify the next) — a seeded channel attaches to one pass alone.
    let ring = (channel_capacity() + 7 * frame_size()) as u32;
    let lane = |name: &str, len: u32, dt| Channel::new([len], dt).capacity(ring).named(name);
    let blk = lane("blk", block, dtype::i32);
    let base_d = lane("base_d", 1, dtype::u32).queued();
    let positions_d = lane("positions_d", block, dtype::u32);
    // The pages a lane reads and its extent are ports the draft peeks: a
    // queue it takes each value from, the verify's put appending.
    let pages_d = lane("pages_d", max_pages, dtype::u32).queued();
    let page_indptr_d = lane("page_indptr_d", 2, dtype::u32).queued();
    let w_slot_d = lane("w_slot_d", block, dtype::u32);
    let w_off_d = lane("w_off_d", block, dtype::u32);
    // Every row reads every key up to the block's end.
    let kv_len_d = lane("kv_len_d", 1, dtype::u32).queued();
    // Lays out the block anchored at `a` (`[1]` i32) at `b` (`[1]` u32).
    let lay_block = move |a: &Tensor, b: &Tensor| {
        let j = iota(block);
        blk.put(&select(
            eq(&j, Tensor::constant(0u32)),
            broadcast(a, [block]),
            broadcast(Tensor::constant(noise), [block]),
        ));
        base_d.put(b);
        let p = broadcast(b, [block]) + &j;
        positions_d.put(&p);
        w_slot_d.put(&p / page_size);
        w_off_d.put(&p % page_size);
        let end = b + Tensor::constant(block); // [1]
        kv_len_d.put(&end);
        page_indptr_d.put(indptr(1, &end.div_ceil(page_size)));
        pages_d.put(iota(max_pages));
    };

    // The passes are built before anything is submitted: a channel's host
    // role is read off every pass that claims it, and the prefill's last
    // chunk puts the draft's inputs that only the draft consumes.
    // ── the draft pass: one lane of the block's rows ─────────────────
    let indptr_d = Channel::from([0u32, block]).named("embed_indptr_d");
    let readout_d = Channel::from_iter(0..block).named("readout_d");

    // ── the verify pass: one lane of `w` rows, laid out by the draft ──
    let win = Channel::new([w], dtype::i32).capacity(ring).named("win");
    let base_v = Channel::new([1], dtype::u32).capacity(ring).named("base_v").queued();
    let indptr_v = Channel::from([0u32, w]).named("embed_indptr_v");
    let readout_v = Channel::from_iter(0..w).named("readout_v");
    let positions_v = Channel::new([w], dtype::u32).capacity(ring).named("positions_v");
    let pages_v = Channel::new([max_pages], dtype::u32).capacity(ring).named("pages_v").queued();
    let page_indptr_v =
        Channel::new([2], dtype::u32).capacity(ring).named("page_indptr_v").queued();
    let w_slot_v = Channel::new([w], dtype::u32).capacity(ring).named("w_slot_v");
    let w_off_v = Channel::new([w], dtype::u32).capacity(ring).named("w_off_v");
    let kv_len_v = Channel::new([1], dtype::u32).capacity(ring).named("kv_len_v").queued();
    // The fold: the prefill folded everything, so the first window
    // replays nothing; every verify's epilogue puts the next.
    let fold_len = Channel::from([0u32]).named("fold_len");
    let out = Channel::new([w], dtype::i32).capacity(ring).named("out");

    let draft = ForwardPass::new();
    draft
        .set_drafting_block(true)
        .map_err(|why| format!("stating the draft block: {why}"))?;
    draft.embed(&blk, &indptr_d)?;
    draft.readout(&readout_d)?;
    draft.attention(
        Some(KvBinding {
            working_set: &ws,
            geometry: KvGeometry {
                readable_pages: ..,
                writable_pages: (n / page_size)..,
                kv_len: &kv_len_d,
                pages: &pages_d,
                page_indptr: &page_indptr_d,
                w_slot: &w_slot_d,
                w_off: &w_off_d,
                positions: &positions_d,
                mask: None,
            },
        }),
        &rs_set,
        RsGeometry {
            fold_len: None,
            buffer: 0..0,
        },
    )?;
    draft.epilogue(move || {
        let tokens = blk.take(); // [block] i32: the anchor, then noise
        // The peeked geometry, taken: the verify laid out the next one.
        let _ = (kv_len_d.take(), page_indptr_d.take(), pages_d.take());
        let b = base_d.take(); // [1] u32: the anchor's position
        let picks = reshape(intrinsics::mtp_drafts(block), [block]); // [block] i32
        let i = iota(w);
        let one = Tensor::constant(1u32);
        let first = eq(&i, Tensor::constant(0u32));
        // The window: the anchor, then the first `k` picks.
        let at = min_elem(max_elem(&i, &one) - &one, Tensor::constant(block - 1));
        let a = broadcast(gather(&tokens, iota(1)), [w]);
        win.put(&select(&first, &a, gather(&picks, &at)));
        base_v.put(&b);
        let p = broadcast(&b, [w]) + &i;
        positions_v.put(&p);
        w_slot_v.put(&p / page_size);
        w_off_v.put(&p % page_size);
        let extent = &b + Tensor::constant(w); // [1]
        kv_len_v.put(&extent);
        page_indptr_v.put(indptr(1, &extent.div_ceil(page_size)));
        pages_v.put(iota(max_pages));
    });

    let verify = ForwardPass::new();
    verify.embed(&win, &indptr_v)?;
    verify.readout(&readout_v)?;
    verify.attention(
        Some(KvBinding {
            working_set: &ws,
            geometry: KvGeometry {
                readable_pages: ..,
                writable_pages: (n / page_size)..,
                kv_len: &kv_len_v,
                pages: &pages_v,
                page_indptr: &page_indptr_v,
                w_slot: &w_slot_v,
                w_off: &w_off_v,
                positions: &positions_v,
                mask: None,
            },
        }),
        &rs_set,
        if rs_set.is_empty() {
            RsGeometry {
                fold_len: None,
                buffer: 0..0,
            }
        } else {
            RsGeometry {
                fold_len: Some(&fold_len),
                buffer: 0..run_pages,
            }
        },
    )?;
    let folds = !rs_set.is_empty();
    verify.epilogue(move || {
        let window = win.take(); // [w] i32
        let _ = (kv_len_v.take(), page_indptr_v.take(), pages_v.take());
        let b = base_v.take(); // [1] u32
        let truth = reshape(reduce_argmax(intrinsics::logits()), [w]); // [w] i32
        let i = iota(w);
        let one = Tensor::constant(1u32);
        let none = broadcast(Tensor::constant(NONE), [w]);

        // ── keep: the longest prefix of proposals the trunk agreed with
        let proposed = gather(&window, iota(k) + &one); // window[1..=k]
        let said = gather(&truth, iota(k)); // truth[0..k]
        let hit = cast(eq(&proposed, &said), dtype::u32);
        let m = reshape(reduce_sum(cumprod(&hit)), [1]);
        let mb = broadcast(&m, [w]);

        // ── commit: window[1..=m], then truth[m]; -1 past that
        let next_of = gather(&window, min_elem(&i + &one, Tensor::constant(w - 1)));
        let correction = gather(&truth, &m); // [1]
        let committed = select(
            lt(&i, &mb),
            &next_of,
            select(eq(&i, &mb), broadcast(&correction, [w]), &none),
        );
        out.put(&committed);
        if folds {
            // The accepted rows and the correction's own were real.
            fold_len.put(&(&m + &one));
        }

        // ── the next block: the correction, then noise, at b + m + 1
        lay_block(&correction, &(&b + &m + &one));
    });

    // ── PREFILL: the prompt, chunked; the last chunk's pick anchors ──────
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let mut anchor: i32 = 0;
    let spans = prefill_chunks(n, None);
    let last_span = spans.len() - 1;
    for (at, &(base, end)) in spans.iter().enumerate() {
        let len = end - base;
        let toks = Channel::from(&prompt_i32[base as usize..end as usize]).named("toks_p");
        let indptr = Channel::from([0u32, len]).named("embed_indptr_p");
        let positions = Channel::from_iter(base..end).named("positions_p");
        let pages = Channel::from_iter(0..max_pages).named("pages_p");
        let page_indptr = Channel::from([0u32, end.div_ceil(page_size)]).named("page_indptr_p");
        let w_slot = Channel::from_iter((base..end).map(|p| p / page_size)).named("w_slot_p");
        let w_off = Channel::from_iter((base..end).map(|p| p % page_size)).named("w_off_p");
        let kv_len = Channel::from([end]).named("kv_len_p");
        let readout = Channel::from([len - 1]).named("readout_p");
        let next = Channel::new([1], dtype::i32).named("next_p");

        let fwd = ForwardPass::new();
        fwd.embed(&toks, &indptr)?;
        fwd.readout(&readout)?;
        fwd.attention(
            Some(KvBinding {
                working_set: &ws,
                geometry: KvGeometry {
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
            }),
            &rs_set,
            RsGeometry {
                fold_len: None,
                buffer: 0..0,
            },
        )?;
        let last = at == last_span;
        fwd.epilogue(move || {
            let t = reshape(reduce_argmax(intrinsics::logits()), [1]);
            if last {
                lay_block(&t, &broadcast(Tensor::constant(n), [1]));
            }
            next.put(&t);
        });
        fwd.submit(&pipe).context("prefill submit")?;
        anchor = next.take_host::<Vec<i32>>().await.context("prefill readback")?[0];
    }
    if input.report_timing {
        session::send("t0");
    }

    let mut generated: Vec<u32> = vec![anchor as u32];
    let (mut rounds, mut accepted) = (0usize, 0usize);
    let mut stopped = stop_tokens.contains(&(anchor as u32)) || generated.len() >= input.max_tokens;

    if !stopped {
        // ── the rounds: draft then verify, a window of them ahead ────────
        // A recurrent state holds one slot a posted frame: a pass a frame.
        let pair = rs_set.is_empty() && frame_size() >= 2;
        let rounds_ahead = (model::run_ahead_window() as usize / 2).max(1);
        let budget = input.max_tokens;
        let submit_round = || -> std::result::Result<(), String> {
            if pair {
                submit_frame(&pipe, &[Some(&draft), Some(&verify)])
            } else {
                submit_frame(&pipe, &[Some(&draft)])?;
                submit_frame(&pipe, &[Some(&verify)])
            }
        };
        let (mut submitted, mut consumed) = (0usize, 0usize);
        while submitted < rounds_ahead.min(budget) {
            submit_round()?;
            submitted += 1;
        }
        while consumed < submitted && !stopped {
            let committed = out.take_host::<Vec<i32>>().await?;
            consumed += 1;
            rounds += 1;
            let live: Vec<u32> = committed
                .iter()
                .filter(|&&t| t != NONE)
                .map(|&t| t as u32)
                .collect();
            accepted += live.len().saturating_sub(1);
            for t in live {
                generated.push(t);
                if stop_tokens.contains(&t) || generated.len() >= input.max_tokens {
                    stopped = true;
                    break;
                }
            }
            // A round commits at least one token: `budget` rounds always
            // reach the stop.
            if !stopped && submitted < budget {
                submit_round()?;
                submitted += 1;
            }
        }
    }
    pipe.close();

    if let Some(at) = generated.iter().position(|t| stop_tokens.contains(t)) {
        generated.truncate(at + 1);
    }
    generated.truncate(input.max_tokens);
    let drafted = rounds * k as usize;
    Ok(Output {
        sampler: "dspark-speculative-bench",
        text: model::decode(&generated)?,
        count: generated.len(),
        num_prompt_tokens: n as usize,
        num_output_tokens: generated.len(),
        token_ids: generated,
        rounds,
        drafted,
        accepted,
        acceptance_rate: if drafted == 0 { 0.0 } else { accepted as f64 / drafted as f64 },
        k,
        block,
    })
}
