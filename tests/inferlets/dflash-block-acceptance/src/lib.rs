//! **HOW MUCH OF A DFLASH BLOCK THE TARGET ACTUALLY KEEPS.**
//!
//! A block drafter proposes `block` tokens in ONE pass over
//! `[anchor, MASK x block-1]`, and a speculative round keeps the longest
//! PREFIX of them the target agrees with. That prefix is the whole economics
//! of the technique — it is what multiplies a verify fire's cost into
//! tokens — and it cannot be read from the engine's own harness, because a
//! host readout seat hands back one row per lane and a truncated block is
//! not a window onto the full one (a block DIFFUSION model trained at one
//! width is out of distribution at every other; see
//! `engine-metal/tests/the_block_drafters_proposals_are_measured.rs`).
//!
//! So it is measured here, where an epilogue can read all `block` proposals
//! off the `mtp.drafts` seam on the device:
//!
//! ```text
//! draft   [anchor, MASK x block-1]   one pass, `set-drafting-block`, block mask
//!         -> mtp.drafts              `block` proposals, one a row
//! truth   the target's own next `block` tokens, decoded greedily
//! kept    |longest prefix where proposal_i == truth_i|
//! ```
//!
//! This measures; it does not go fast. Each round pays one draft pass and
//! `block` ordinary decodes, and the host is in the loop between them — the
//! point is the number, not the throughput. A round that turned the same
//! machinery into speed would verify the block in ONE fire instead of
//! decoding it, which is the loop `rs-mtp-speculative-decoding` runs for the
//! chained heads.

use inferlet::chat;
use inferlet::eta::hybrid::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_rounds")]
    rounds: usize,
    /// Diagnostic: state a CAUSAL block mask instead of the all-visible one
    /// the drafter's full-attention layer wants. If the proposals do not
    /// move, the stated mask is not loosening anything and the block is
    /// causal at every layer whatever this says.
    #[serde(default)]
    causal_block: bool,
    /// Diagnostic: hide every key from this index on. `-1` hides nothing.
    /// Hiding a PREFIX of the context and watching the drafts move is what
    /// says a mask is read per key rather than all-or-nothing.
    #[serde(default = "no_hide")]
    hide_from: i32,
    /// Diagnostic: hide every key for block ROWS from this index on. `-1`
    /// hides nothing. `hide_from` varies the mask by KEY and is uniform
    /// across rows; this varies it by ROW, which is what a causal triangle
    /// needs and what a shader reading one mask row for every query would
    /// lose.
    #[serde(default = "no_hide")]
    hide_rows_from: i32,
    /// Diagnostic: hide every key BELOW this index — the context, leaving
    /// the block's own rows. If the drafts barely move, the block's kv is
    /// not what the rows are reading.
    #[serde(default = "no_hide")]
    hide_before: i32,
    /// Diagnostic: bind no mask at all. A plan whose drafter states no
    /// masked arm refuses a lane that carries one, which is what a bisect
    /// that skips the drafter's layers runs into.
    #[serde(default)]
    no_mask: bool,
    /// Diagnostic: hide the whole context, leaving the block its own rows.
    /// The reference can be given a zero-length context, so the two are
    /// comparable and the drafter's LAYERS are what is being compared.
    #[serde(default)]
    block_only: bool,
    /// Diagnostic: read the prefill out at every row, not only the last.
    #[serde(default)]
    readout_all: bool,
}

fn no_hide() -> i32 {
    -1
}

fn default_prompt() -> String {
    "The quick brown fox jumps over".into()
}

fn default_rounds() -> usize {
    4
}

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    /// The block width the model's draft seam advertises.
    block: u32,
    /// The accepted prefix of each round's block, in order.
    kept: Vec<u32>,
    /// Their mean — the number a round's economics turns on.
    mean_kept: f64,
    /// Every position's hit rate across rounds, so a prefix that stops early
    /// can be told from a drafter that is wrong everywhere.
    hits_by_position: Vec<u32>,
    /// What the target actually produced, for eyeballing.
    first_anchor: i32,
    prefill_plane: Vec<i32>,
    truth: Vec<u32>,
    /// What the drafter proposed, round by round.
    drafts: Vec<Vec<u32>>,
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
    let page_size = kv_page_size();
    let rs_page = model::rs_buffer_page_size().max(1);
    let mut prompt = model::encode(&input.prompt);
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();

    // The prompt, every token the rounds decode, and one block of proposal
    // rows above the committed length that the next fire writes over.
    let ws = WorkingSet::new();
    let max_extent = n + (input.rounds as u32 + 1) * block + block;
    let max_pages = max_extent.div_ceil(page_size);
    ws.reserve(max_pages).context("reserve KV")?;
    let pool = max_pages * page_size;
    let rs = RsWorkingSet::new();
    rs.alloc_buffer(2 * block.div_ceil(rs_page).max(1))
        .map_err(|why| format!("alloc rs runs: {why}"))?;
    let rs_set = vec![rs];
    let pipe = Pipeline::new();

    // ── PREFILL: the prompt, chunked. Every chunk leaves the drafter's
    //    context behind it, because the context arm rides every trunk fire.
    let spans = prefill_chunks(n, None);
    let mut anchor: i32 = 0;
    let mut prefill_plane: Vec<i32> = Vec::new();
    for &(base, end) in &spans {
        let len = end - base;
        let toks = Channel::from(&prompt_i32[base as usize..end as usize]).named("toks_p");
        let indptr = Channel::from([0u32, len]).named("embed_indptr_p");
        let positions = Channel::from_iter(base..end).named("positions_p");
        let pages = Channel::from_iter(0..max_pages).named("pages_p");
        let page_indptr = Channel::from([0u32, end.div_ceil(page_size)]).named("page_indptr_p");
        let w_slot = Channel::from_iter((base..end).map(|p| p / page_size)).named("w_slot_p");
        let w_off = Channel::from_iter((base..end).map(|p| p % page_size)).named("w_off_p");
        let kv_len = Channel::from([end]).named("kv_len_p");
        // Diagnostic: read out EVERY prompt row, so a bisect can line the
        // whole plane up against a reference rather than one token.
        let wide = input.readout_all;
        let readout = if wide {
            Channel::from_iter(0..len).named("readout_p")
        } else {
            Channel::from([len - 1]).named("readout_p")
        };
        let out_rows = if wide { len } else { 1 };
        let next = Channel::new([out_rows], dtype::i32).named("next_p");

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
        fwd.epilogue(move || {
            next.put(&reshape(reduce_argmax(intrinsics::logits()), [out_rows]));
        });
        fwd.submit(&pipe).context("prefill submit")?;
        let read = next.take_host::<Vec<i32>>().await.context("prefill readback")?;
        if wide {
            prefill_plane = read.clone();
        }
        anchor = *read.last().expect("a prefill readout");
    }

    // ── ROUNDS ──────────────────────────────────────────────────────────
    let first_anchor = anchor;
    let mut held = n;
    let mut kept = Vec::new();
    let mut drafts_all = Vec::new();
    let mut truth_all = Vec::new();
    let mut hits_by_position = vec![0u32; block as usize - 1];

    for round in 0..input.rounds {
        // ── the draft: ONE pass over `[anchor, MASK x block-1]`, the trunk
        //    guarded away from its rows, the whole extent visible (the
        //    drafter's last layer is full attention with no causality).
        let mut ids = vec![MASK_TOKEN; block as usize];
        ids[0] = anchor;
        let toks = Channel::from(ids.as_slice()).named("toks_d");
        let indptr = Channel::from([0u32, block]).named("embed_indptr_d");
        let positions = Channel::from_iter(held..held + block).named("positions_d");
        let pages = Channel::from_iter(0..max_pages).named("pages_d");
        let page_indptr =
            Channel::from([0u32, (held + block).div_ceil(page_size)]).named("page_indptr_d");
        let w_slot =
            Channel::from_iter((held..held + block).map(|p| p / page_size)).named("w_slot_d");
        let w_off =
            Channel::from_iter((held..held + block).map(|p| p % page_size)).named("w_off_d");
        let kv_len = Channel::from([held + block]).named("kv_len_d");
        let causal = input.causal_block;
        let hide_from = input.hide_from;
        let hide_rows_from = input.hide_rows_from;
        let hide_before = input.hide_before;
        let block_only = input.block_only;
        let visible: Vec<bool> = (0..block)
            .flat_map(|i| {
                (0..pool).map(move |j| {
                    // `causal_block` doubles as the inert-mask probe: at its
                    // extreme the mask hides EVERYTHING, so drafts that do
                    // not move are drafts a mask never touched.
                    j < held + block
                        && (!causal || j <= held + i)
                        && (hide_from < 0 || (j as i32) < hide_from)
                        && (hide_rows_from < 0 || (i as i32) < hide_rows_from)
                        && (hide_before < 0 || (j as i32) >= hide_before)
                        && (!block_only || j >= held)
                })
            })
            .collect();
        let mask = Channel::from_shaped([block, pool], visible).named("mask_d");
        let bound_mask = if input.no_mask { None } else { Some(&mask) };
        // **EVERY BLOCK ROW READS OUT.** The drafts plane is cut to the rows
        // the readout names, so a pass that leaves it at the default gets a
        // one-row plane and asking it for `block` values is a geometry
        // mismatch — which is exactly how this first failed.
        let readout = Channel::from_iter(0..block).named("readout_d");
        let out = Channel::new([block], dtype::i32).named("drafts_d");

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
                    mask: bound_mask,
                },
            }),
            &rs_set,
            RsGeometry {
                fold_len: None,
                buffer: 0..0,
            },
        )?;
        fwd.epilogue(move || {
            // **THE PROPOSALS ARE THE LOGITS, NOT THE DRAFTS SEAM.** A
            // chained head's draft logits are a plane of their own, which is
            // what `mtp.drafts` is for; a BLOCK drafter's rows go through
            // the target's one `lm_head` beside the trunk's, so the fire's
            // own readout over the block rows already IS the drafter's. The
            // seam is bound one row wide at the readout row
            // (`serve.rs::bind_intrinsic`), so asking it for `block` values
            // is a geometry mismatch — which is how this was found.
            out.put(&reshape(reduce_argmax(intrinsics::logits()), [block]));
        });
        fwd.submit(&pipe)
            .with_context(|| format!("draft submit @round {round}"))?;
        let proposals = out
            .take_host::<Vec<i32>>()
            .await
            .with_context(|| format!("draft readback @round {round}"))?;

        // ── the truth: the target's own next `block` tokens, greedily.
        let mut truth = Vec::with_capacity(block as usize);
        let mut fed = anchor;
        for step in 0..block {
            let at = held + step;
            let toks = Channel::from([fed]).named("toks_t");
            let indptr = Channel::from([0u32, 1]).named("embed_indptr_t");
            let positions = Channel::from([at]).named("positions_t");
            let pages = Channel::from_iter(0..max_pages).named("pages_t");
            let page_indptr =
                Channel::from([0u32, (at + 1).div_ceil(page_size)]).named("page_indptr_t");
            let w_slot = Channel::from([at / page_size]).named("w_slot_t");
            let w_off = Channel::from([at % page_size]).named("w_off_t");
            let kv_len = Channel::from([at + 1]).named("kv_len_t");
            let next = Channel::new([1], dtype::i32).named("next_t");

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
                RsGeometry {
                    fold_len: None,
                    buffer: 0..0,
                },
            )?;
            fwd.epilogue(move || {
                next.put(&reshape(reduce_argmax(intrinsics::logits()), [1]));
            });
            fwd.submit(&pipe)
                .with_context(|| format!("truth submit @round {round} step {step}"))?;
            fed = next.take_host::<Vec<i32>>().await.context("truth readback")?[0];
            truth.push(fed);
        }

        // ── what the target kept ────────────────────────────────────────
        // **ROW 0 IS THE ANCHOR, NOT A PREDICTION.** The block is
        // `[anchor, MASK x block-1]` and a block diffusion model DENOISES
        // each mask into the token AT ITS OWN POSITION — so row `i` proposes
        // position `held + i`, and the anchor's own row proposes nothing new.
        // The drafts are rows `1..block`, which is why the checkpoint's
        // README runs it at `num_speculative_tokens: 15` against a block of
        // sixteen.
        let proposals: Vec<i32> = proposals[1..].to_vec();
        let mut prefix = 0u32;
        for (at, (p, t)) in proposals.iter().zip(&truth).enumerate() {
            if p == t {
                hits_by_position[at] += 1;
                if prefix as usize == at {
                    prefix += 1;
                }
            }
        }
        kept.push(prefix);
        drafts_all.push(proposals.iter().map(|&t| t as u32).collect::<Vec<u32>>());
        truth_all.extend(truth.iter().map(|&t| t as u32));
        anchor = *truth.last().expect("a decoded token");
        held += block;
    }

    let mean_kept = if kept.is_empty() {
        0.0
    } else {
        kept.iter().map(|k| f64::from(*k)).sum::<f64>() / kept.len() as f64
    };
    Ok(Output {
        sampler: "dflash-block-acceptance",
        block,
        kept,
        mean_kept,
        hits_by_position,
        first_anchor,
        prefill_plane,
        truth: truth_all,
        drafts: drafts_all,
    })
}
