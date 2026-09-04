//! **A BLOCK DRAFTER'S SPECULATIVE LOOP: ONE DRAFT FIRE, ONE VERIFY FIRE.**
//!
//! `dflash-block-acceptance` measures how much of a block the target keeps
//! and pays `block` ordinary decodes to learn it. This spends that number:
//! a round is TWO fires, and it emits every token the target agreed with
//! plus the one it corrected.
//!
//! ```text
//! draft    [anchor, MASK x block-1]      one fire, `set-drafting-block`
//!          -> d[0..block-1]              row i proposes position held+i
//! verify   [anchor, d[0], .., d[b-2]]    one fire, the TRUNK, plainly causal
//!          -> t[i] = argmax(row i)       the target's own token for held+i+1
//! keep     the longest prefix where d[j] == t[j], then t at the break
//! ```
//!
//! The break's token is the target's, so a round emits `kept + 1` tokens and
//! is never wrong: the sequence this writes is the sequence a one-token-a-
//! fire decode would have written. That is the whole claim a speculative
//! loop makes, and `--baseline` runs the other loop over the same prompt so
//! the ratio is measured rather than argued.
//!
//! # What it measured
//!
//! `qwen36-27b-dflash` on an M4 Pro, greedy, the same 192 tokens either way
//! (a 64-token run subtracted from a 256-token one, so the load is out of
//! the number) and the two roads token for token identical:
//!
//! ```text
//! counting  "1, 2, 3, ..."     14.00 tok/round   62.1 tok/s   vs 15.0   4.1x
//! code      "write a function" 6.14 tok/round    28.4 tok/s   vs 15.1   1.9x
//! ```
//!
//! **THE KV ROLLBACK IS THE SUBTLE PART.** The verify fire writes a row's
//! keys at `held + i` for all `block` rows, and the rows past the break
//! carried tokens the target rejected — so the next round states
//! `kv_len = held + kept + 1` and the stale rows are simply never read.
//! The drafter's own context rides the same rows (its context arm runs on
//! every trunk fire), so one length rolls both streams back.

use inferlet::eta::hybrid::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    /// Decode one token a fire instead — the same sequence by a different
    /// road, and the denominator of the speedup.
    #[serde(default)]
    baseline: bool,
    /// **HOW MANY OF THE BLOCK'S ROWS TO VERIFY**, the anchor's included.
    /// Zero (the default) verifies all of them.
    ///
    /// The drafter is trained at one block width and proposes fifteen
    /// whatever this says — a block diffusion model is out of distribution
    /// at any other width — but the TARGET need not read them all, and a
    /// verify fire is priced by its rows: sixteen cost 2.82 one-row fires on
    /// this box, eight cost 1.83. A workload whose accepted prefix rarely
    /// reaches eight pays for ten rows it was never going to keep. Which
    /// rows are worth a fire is the guest's call, not the engine's.
    #[serde(default)]
    verify_rows: u32,
    /// Let the drafter's own logit margin choose the width per round, rather
    /// than verifying the whole block — see `wide_enough`. Off by default,
    /// because on three workloads it is a WASH.
    #[serde(default)]
    margin_width: bool,
}

fn default_prompt() -> String {
    "The quick brown fox jumps over".into()
}

fn default_max_tokens() -> usize {
    64
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
    /// Draft+verify fire pairs after the prefill; zero under `--baseline`.
    rounds: usize,
    /// Proposals made, `block - 1` a round.
    drafted: usize,
    /// Proposals the target's own argmax agreed with.
    accepted: usize,
    /// `accepted / drafted`, or zero when nothing was drafted.
    acceptance_rate: f64,
    /// Tokens emitted per round, the number the economics turns on: a round
    /// costs two fires whatever this is.
    tokens_per_round: f64,
    /// The block width this run used.
    block: u32,
    /// The widths the rounds chose, in order.
    verify: Vec<u32>,
    /// Rows replayed through the buffer read path, summed over the rounds —
    /// a run whose `replayed` is zero exercised none of the fold-commit path.
    replayed: usize,
    /// Rows the buffer forgot because the target rejected them.
    discarded: usize,
    /// The recurrent buffer's page width, for reading the grant arithmetic.
    rs_page: u32,
}

/// **IS THIS ROUND WORTH A WIDE VERIFY? THE DRAFTER ALREADY SAID.**
///
/// The signal is the LOGIT MARGIN, top-1 less top-2, because that is what
/// comes free: `top_k` reads the logits plane once and returns both, where a
/// probability would want a softmax over the whole vocabulary and a second
/// pass over the plane.
///
/// A verify fire's price is a staircase — the tile point pads a fire's rows up
/// to a row block, so a round costs 2.10 one-row fires at eight rows and 3.09
/// at sixteen — and the question is whether the prefix will reach eight. The
/// drafter's own top-1 probability answers it: over sixty anchors on five
/// prompts, a position INSIDE the accepted prefix carries 0.936 and one
/// outside 0.383, and the first seven positions' mean reads 0.959 where the
/// prefix reached eight against 0.553 where it did not. At this threshold
/// that calls the wide round on 28 of 30 long anchors and on 2 of 30 short
/// ones.
///
/// This is the signal Dspark spends a trained confidence head on. A block
/// drafter that reads out through the target's own `lm_head` has it for one
/// reduction over a plane the fire already computed.
fn wide_enough(confidence: &[f32]) -> bool {
    const THRESHOLD: f64 = MARGIN_THRESHOLD;
    let head = &confidence[..confidence.len().min(NARROW as usize - 1)];
    let mean = head.iter().map(|c| f64::from(*c)).sum::<f64>() / head.len().max(1) as f64;
    mean >= THRESHOLD
}

/// The narrow rung, the only other width whose price differs (twelve rows
/// cost what sixteen do).
const NARROW: u32 = 8;

/// Where the margin separates a round that will reach eight from one that
/// will not — fitted offline, `scratchpad/dflash_ref/confidence.py`.
const MARGIN_THRESHOLD: f64 = 2.0;

/// The pages the recurrent buffer must hold for one fire: the survivors (at
/// most a page's worth of head offset before them), plus the window.
/// A fold releases only the WHOLE head pages its prefix covers and rebases
/// `buffer_head` by the remainder, so a run whose folds are not page-aligned
/// keeps a head offset the grant has to sit above — hence the extra page on
/// top of the survivors and the window.
fn buffer_pages_for(survivors: u32, window: u32, page: u32) -> u32 {
    (page.saturating_sub(1) + survivors + window)
        .div_ceil(page.max(1))
        .max(1)
        + 1
}

/// Rows one draft block carries — the model text's `DFLASH_BLOCK`. A guest
/// cannot read it off the load yet (`mtp_depth` advertises the seam's DEPTH,
/// which is one: a block drafter plants one proposal a ROW), so it is stated
/// in one place until the load advertises it.
const BLOCK_ROWS: u32 = 16;

/// The drafter's own mask token, `dflash_config.mask_token_id`. Stated here
/// for the same reason and with the same caveat.
const MASK_TOKEN: i32 = 248_070;

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if model::pass_kind() != model::ForwardKind::Hybrid {
        return Err("this inferlet drives a hybrid model's recurrent state".into());
    }
    if model::mtp_depth() == 0 {
        return Err("this SKU ships no draft head".into());
    }
    let block = BLOCK_ROWS;
    let pinned = (input.verify_rows != 0).then(|| input.verify_rows.clamp(2, block));
    let page_size = kv_page_size();
    let rs_page = model::rs_buffer_page_size().max(1);
    let mut prompt = model::encode(&input.prompt);
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();

    // The prompt, every token the loop commits, and one block of speculative
    // rows above the committed length that the next fire writes over.
    let ws = WorkingSet::new();
    let max_extent = n + input.max_tokens as u32 + 2 * block;
    let max_pages = max_extent.div_ceil(page_size);
    ws.reserve(max_pages).context("reserve KV")?;
    let pool = max_pages * page_size;
    let rs = RsWorkingSet::new();
    let rs_set = vec![rs];

    let pipe = Pipeline::new();

    // ── PREFILL: the prompt, chunked. Every chunk leaves the drafter's
    //    context behind it, because the context arm rides every trunk fire.
    let mut anchor: i32 = 0;
    for &(base, end) in &prefill_chunks(n, None) {
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
            RsGeometry { fold_len: None, buffer: 0..0 },
        )?;
        fwd.epilogue(move || {
            next.put(&reshape(reduce_argmax(intrinsics::logits()), [1]));
        });
        fwd.submit(&pipe).context("prefill submit")?;
        anchor = next.take_host::<Vec<i32>>().await.context("prefill readback")?[0];
    }

    // ── THE LOOP ────────────────────────────────────────────────────────
    // `held` is the committed length: positions `0..held` are in the cache
    // and `anchor` is the token AT position `held`, not yet written.
    let mut held = n;
    let mut generated: Vec<u32> = vec![anchor as u32];
    let mut rounds = 0usize;
    let mut drafted = 0usize;
    let mut accepted = 0usize;
    let mut replayed = 0usize;
    let mut discarded = 0usize;
    // Tokens sitting in the recurrent buffer unfolded. The prefill folded
    // everything and the anchor is the first window's own row, so the first
    // round replays nothing.
    let mut survivors: u32 = 0;
    let mut widths: Vec<u32> = Vec::new();

    while generated.len() < input.max_tokens {
        if input.baseline {
            // One token a fire, the road the speedup is measured against.
            let toks = Channel::from([anchor]).named("toks_b");
            let indptr = Channel::from([0u32, 1]).named("embed_indptr_b");
            let positions = Channel::from([held]).named("positions_b");
            let pages = Channel::from_iter(0..max_pages).named("pages_b");
            let page_indptr =
                Channel::from([0u32, (held + 1).div_ceil(page_size)]).named("page_indptr_b");
            let w_slot = Channel::from([held / page_size]).named("w_slot_b");
            let w_off = Channel::from([held % page_size]).named("w_off_b");
            let kv_len = Channel::from([held + 1]).named("kv_len_b");
            let next = Channel::new([1], dtype::i32).named("next_b");

            let fwd = ForwardPass::new();
            fwd.embed(&toks, &indptr)?;
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
                RsGeometry { fold_len: None, buffer: 0..0 },
            )?;
            fwd.epilogue(move || {
                next.put(&reshape(reduce_argmax(intrinsics::logits()), [1]));
            });
            fwd.submit(&pipe).context("baseline submit")?;
            anchor = next.take_host::<Vec<i32>>().await.context("baseline readback")?[0];
            held += 1;
            generated.push(anchor as u32);
            continue;
        }

        // The width is chosen AFTER the draft fire, from what the drafter
        // itself says — see `wide_enough`. Pinned, it is stated here.
        let mut verify = pinned.unwrap_or(block);
        // The buffer must hold the survivors and this window; the grant is
        // the guest's one allocation decision.
        let buffer_pages = buffer_pages_for(survivors, block, rs_page);
        let have = rs_set[0].buffer_size();
        if have < buffer_pages {
            rs_set[0]
                .alloc_buffer(buffer_pages - have)
                .map_err(|why| format!("alloc {} rs buffer page(s): {why}", buffer_pages - have))?;
        }
        let fold_none = Channel::from([0u32]).named("fold_none");
        let fold_len = Channel::from([survivors]).named("fold_len_v");

        // ── the draft: ONE pass over `[anchor, MASK x block-1]` ──────────
        let mut ids = vec![MASK_TOKEN; block as usize];
        ids[0] = anchor;
        let toks = Channel::from(ids.as_slice()).named("toks_d");
        let indptr = Channel::from([0u32, block]).named("embed_indptr_d");
        let positions = Channel::from_iter(held..held + block).named("positions_d");
        let pages = Channel::from_iter(0..max_pages).named("pages_d");
        let page_indptr =
            Channel::from([0u32, (held + block).div_ceil(page_size)]).named("page_indptr_d");
        let w_slot = Channel::from_iter((held..held + block).map(|p| p / page_size)).named("w_slot_d");
        let w_off = Channel::from_iter((held..held + block).map(|p| p % page_size)).named("w_off_d");
        let kv_len = Channel::from([held + block]).named("kv_len_d");
        // The drafter's full-attention layer is BIDIRECTIONAL over the
        // block, which only a stated mask says; every key up to the block's
        // end is visible to every block row.
        let visible: Vec<bool> = (0..block)
            .flat_map(|_| (0..pool).map(move |j| j < held + block))
            .collect();
        let mask = Channel::from_shaped([block, pool], visible).named("mask_d");
        // **EVERY BLOCK ROW READS OUT**, and the proposals are the fire's own
        // logits: a block drafter's rows go through the TARGET's one
        // `lm_head`, so there is no separate draft plane to read.
        let readout = Channel::from_iter(0..block).named("readout_d");
        let out = Channel::new([block * 2], dtype::i32).named("drafts_d");
        // **THE DRAFTER'S OWN CONFIDENCE, OFF THE SAME LOGITS.** One more
        // reduction over a plane the fire already computed, read back beside
        // the proposals in the same round trip.
        let conf = Channel::new([block * 2], dtype::f32).named("conf_d");

        let fwd = ForwardPass::new();
        fwd.set_drafting_block(true)
            .map_err(|why| format!("stating the draft block: {why}"))?;
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
                    mask: Some(&mask),
                },
            }),
            &rs_set,
            // **FOLDS NOTHING, AND ITS OWN ROWS ARE DISCARDED BELOW.** A
            // fire binds one recurrent working set per request and is
            // charged its rows in the buffer, but the trunk that owns the
            // recurrence runs over NONE of a draft fire's rows — they are
            // the drafter's. Folding "everything" here would fold the
            // previous round's accepted prefix at the wrong instant and the
            // draft rows on top of it; folding nothing and forgetting the
            // rows again leaves the buffer exactly as this fire found it.
            RsGeometry { fold_len: Some(&fold_none), buffer: 0..buffer_pages },
        )?;
        {
            let out = out.clone();
            let conf = conf.clone();
            fwd.epilogue(move || {
                // **ONE OP, ONE CONSUMER, BOTH ANSWERS.** A second reduction
                // over the logits plane costs about a second a round even
                // with the plane bound once (3.0 s to 7.4 s over a 64-token
                // run; 46 s with a softmax on it) — two consumers take it off
                // the device. `top_k` reads it once and returns the values
                // beside the indices, so the proposal and the margin that
                // says how sure of it the drafter is come out together.
                let (value, index) = top_k(intrinsics::logits(), 2);
                out.put(&reshape(cast(index, dtype::i32), [block * 2]));
                conf.put(&reshape(value, [block * 2]));
            });
        }
        fwd.submit(&pipe).context("draft submit")?;
        // **ROW 0 IS THE ANCHOR, NOT A PREDICTION.** A block diffusion model
        // denoises each mask into the token AT ITS OWN POSITION, so row `i`
        // proposes position `held + i` and the anchor's row proposes nothing
        // new. The proposals are rows `1..block`.
        let top = out.take_host::<Vec<i32>>().await.context("draft readback")?;
        let value = conf.take_host::<Vec<f32>>().await.context("margin readback")?;
        // Row `r` occupies `[2r, 2r + 1]`: the proposal and its runner-up.
        let proposals: Vec<i32> = (1..block as usize).map(|r| top[2 * r]).collect();
        let margin: Vec<f32> = (1..block as usize)
            .map(|r| value[2 * r] - value[2 * r + 1])
            .collect();
        let proposals = proposals.as_slice();
        if pinned.is_none() && input.margin_width {
            verify = if wide_enough(&margin) { block } else { NARROW };
        }
        widths.push(verify);
        // The buffer is back to the accepted prefix the verify is about to
        // fold — see the draft fire's geometry.
        rs_set[0]
            .discard_buffered(block)
            .map_err(|why| format!("forget the draft fire's {block} row(s): {why}"))?;

        // ── the verify: ONE trunk fire over `[anchor, proposals]` ────────
        let mut fed = vec![anchor];
        fed.extend_from_slice(&proposals[..verify as usize - 1]);
        let toks = Channel::from(fed.as_slice()).named("toks_v");
        let indptr = Channel::from([0u32, verify]).named("embed_indptr_v");
        let positions = Channel::from_iter(held..held + verify).named("positions_v");
        let pages = Channel::from_iter(0..max_pages).named("pages_v");
        let page_indptr =
            Channel::from([0u32, (held + verify).div_ceil(page_size)]).named("page_indptr_v");
        let w_slot = Channel::from_iter((held..held + verify).map(|p| p / page_size)).named("w_slot_v");
        let w_off = Channel::from_iter((held..held + verify).map(|p| p % page_size)).named("w_off_v");
        let kv_len = Channel::from([held + verify]).named("kv_len_v");
        let readout = Channel::from_iter(0..verify).named("readout_v");
        let truth = Channel::new([verify], dtype::i32).named("truth_v");

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
                    // No mask: the verify is an ordinary causal prefill of
                    // the rows the drafter guessed.
                    mask: None,
                },
            }),
            &rs_set,
            RsGeometry {
                fold_len: Some(&fold_len),
                buffer: 0..buffer_pages,
            },
        )?;
        {
            let truth = truth.clone();
            fwd.epilogue(move || {
                truth.put(&reshape(reduce_argmax(intrinsics::logits()), [verify]));
            });
        }
        fwd.submit(&pipe).context("verify submit")?;
        let truth = truth.take_host::<Vec<i32>>().await.context("verify readback")?;

        // ── what the target kept ────────────────────────────────────────
        // Row `i` of the verify predicts position `held + i + 1`, which is
        // what proposal `i` claims — so the prefix is read off one zip.
        let kept = proposals[..verify as usize - 1]
            .iter()
            .zip(&truth)
            .take_while(|(p, t)| p == t)
            .count();
        rounds += 1;
        drafted += verify as usize - 1;
        accepted += kept;
        replayed += survivors as usize;
        // The rejected tail never happened: forget it before the next fire,
        // whose fold reaches exactly the accepted prefix.
        let rejected = (verify as usize - 1 - kept) as u32;
        if rejected > 0 {
            rs_set[0]
                .discard_buffered(rejected)
                .map_err(|why| format!("discard {rejected} rejected row(s): {why}"))?;
            discarded += rejected as usize;
        }
        survivors = kept as u32 + 1;
        for tok in proposals[..kept].iter() {
            generated.push(*tok as u32);
        }
        // The break's token is the TARGET's, so the round is never wrong.
        anchor = truth[kept];
        generated.push(anchor as u32);
        // Positions `held ..= held + kept` carried tokens the target agreed
        // with and stay; everything above them is stale and never read.
        held += kept as u32 + 1;
    }

    generated.truncate(input.max_tokens);
    let count = generated.len();
    Ok(Output {
        sampler: "dflash-speculative-bench",
        text: model::decode(&generated)?,
        tokens: generated.clone(),
        count,
        num_prompt_tokens: prompt.len(),
        num_output_tokens: count,
        token_ids: generated,
        rounds,
        drafted,
        accepted,
        acceptance_rate: if drafted == 0 {
            0.0
        } else {
            accepted as f64 / drafted as f64
        },
        tokens_per_round: if rounds == 0 {
            0.0
        } else {
            (accepted + rounds) as f64 / rounds as f64
        },
        block,
        verify: widths,
        replayed,
        discarded,
        rs_page,
    })
}
