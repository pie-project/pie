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
//! # Which head, of the three that ship
//!
//! `scratchpad/dflash_ref/heads.py` runs all three through ONE runtime
//! (`mlx_dspark`, which carries both DFlash architectures and DSpark's),
//! each against ITS OWN target, and keeps EVERY round's accepted prefix
//! rather than its mean. The index beside each cell is tokens a round over
//! fires a round at the width that prices it best, against this box's fire
//! staircase (`a_fire_is_priced_by_its_width`, one row = 66.54 ms: 1.04 /
//! 1.16 / 1.43 / 1.77 / 1.83 / 2.78 at 2 / 3 / 4 / 6 / 8 / 16 rows, plus
//! 0.27 for the draft fire — the same constant for every head). 192 tokens
//! a prompt, greedy, chat template:
//!
//! ```text
//!          block   counting        code        recall       prose        json     mean
//! DFlash     16  4.24 w8 1.80  6.66 w8 2.53  2.95 w3 1.64  3.34 w3 1.63  6.09 w8 2.38  1.99
//! DFlash2     8  6.62 w8 3.15  5.08 w8 2.42  3.78 w8 1.80  2.85 w3 1.50  5.19 w8 2.47  2.27
//! DSpark     16  8.29 w16 2.72 5.18 w8 2.05  2.98 w3 1.48  2.37 w3 1.36  5.05 w8 2.01  1.92
//! ```
//!
//! (Corrected 2026-09-05: the reference's `accept_lengths` is `len(committed)`,
//! the accepted prefix plus the bonus — already tokens a round — and an earlier
//! table here added one more, reading 2.33 / 2.58 / 2.29. Order unchanged.)
//!
//! **THE EARLIER TABLE HERE PINNED THE WIDTH, AND THAT DECIDED IT.** It read
//! DFlash 2.69 against DFlash2 2.76 — a tie — by pricing DFlash at sixteen
//! rows and DFlash2 at its native eight, which hands one head the cheaper
//! rung; and it reported DFlash2's prose as 0.93, "a round that loses to
//! plain decode". Neither survives. `E[min(kept, w)]` is not a function of
//! `E[kept]`, so a mean cannot be re-priced at another width — the
//! distribution has to be kept, and once it is, **no head loses on any
//! workload**: the smallest cell above is 1.36.
//!
//! **DFlash2 leads, by 14% on the mean**, and DSpark is NOT the outlier the
//! old note made it (it read "an index of about 1.4" from published means at
//! width-sixteen prices; measured here it is 1.92, within 4% of DFlash).
//! What the old note got right is the shape: DFlash2's shorter block is
//! stronger where prefixes are short (counting 3.15, recall 1.80) and DFlash
//! is stronger on code (2.53).
//!
//! **The head matters about as much as the width.** The spread between heads
//! is 18% (1.92 to 2.27); between a pinned width and the per-prompt best it is
//! 13% (DFlash w4 1.76 against 1.99). Pinned means for the record: DFlash w4
//! 1.76 / w8 1.90, DFlash2 w4 1.85 / w8 2.24, DSpark w4 1.65 / w8 1.81.
//!
//! **The confound that cannot be removed**: DFlash runs on Qwen3.6-27B and
//! the other two on Qwen3.8-27B, because that is what each was trained
//! against — and a head off its own target is not a control but a different
//! measurement (the old note measured that too: DFlash on the 3.8 checkpoint
//! loses half its counting prefix). So the 11% is a head-and-target
//! difference, and porting DFlash2 would also mean porting to a new target.
//!
//! This measures; it does not go fast. Each round pays one draft pass and
//! `block` ordinary decodes, and the host is in the loop between them — the
//! point is the number, not the throughput. A round that turned the same
//! machinery into speed would verify the block in ONE fire instead of
//! decoding it, which is the loop `rs-mtp-speculative-decoding` runs for the
//! chained heads.

use inferlet::eta::adapter::{Site, mm};
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
    /// The block the head was trained at: sixteen for the v1 head
    /// (`qwen36-27b-dflash`), eight for DFlash2 (`qwen38-27b-dflash2`). The
    /// load does not advertise it yet, so the caller states it; absent, v1's.
    #[serde(default)]
    block: Option<u32>,
    /// Diagnostic: bind a zero-`B` adapter (`lora-probe`'s shape) on every
    /// pass, the draft fire included. The correction is exactly zero, so the
    /// proposals must not move; what is under test is that a lane which bound
    /// an adapter may FIRE a draft block at all — its rows run no trunk
    /// layer, so there is nothing to correct and nothing to refuse.
    #[serde(default)]
    adapter: bool,
    /// The adapter bank's geometry, the trunk's `[layers, rank, hidden]`;
    /// `lora-probe`'s defaults (qwen35-d0.8b) when absent.
    #[serde(default)]
    adapter_layers: Option<u32>,
    #[serde(default)]
    adapter_hidden: Option<u32>,
    #[serde(default)]
    adapter_rank: Option<u32>,
}

/// A zero-`B` adapter in `lora-probe`'s orientations: `A` `[layers, rank,
/// hidden]` with a deterministic pattern, `B` `[layers, hidden, rank]` all
/// zero, so `B(Ax)` is exactly zero and the pass's answer is the base model's.
struct ZeroAdapter {
    a: Vec<f32>,
    b: Vec<f32>,
    layers: u32,
    hidden: u32,
    rank: u32,
}

impl ZeroAdapter {
    fn build(layers: u32, hidden: u32, rank: u32) -> ZeroAdapter {
        let a = (0..layers * rank * hidden)
            .map(|i| {
                let mut h = i ^ 0x0a0a_a0a0;
                h ^= h >> 16;
                h = h.wrapping_mul(0x7feb_352d);
                h ^= h >> 15;
                ((h % 10_000) as f32 / 10_000.0 - 0.5) * 0.1
            })
            .collect();
        let b = vec![0.0f32; (layers * hidden * rank) as usize];
        ZeroAdapter { a, b, layers, hidden, rank }
    }

    /// Fresh channels a fire: a channel's seed is put once, so one bound to an
    /// earlier fire cannot be re-stated on the next (`lora-probe` rebuilds its
    /// weights a phase for the same reason).
    fn attach(&self, fwd: &ForwardPass) -> Result<()> {
        let a = Channel::from_shaped([self.layers, self.rank, self.hidden], self.a.clone()).named("lora_a");
        let b = Channel::from_shaped([self.layers, self.hidden, self.rank], self.b.clone()).named("lora_b");
        fwd.adapter(Site::O, move |x, y| y + mm(&b, mm(&a, x)))
            .map_err(|e| e.into())
    }
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
    if model::mtp_depth() == 0 {
        return Err("this SKU ships no draft head".into());
    }
    // The head's facts off the load (`model::draft_block`); the inputs
    // override for a diagnostic, the constants are the last resort.
    let advertised = model::draft_block();
    let block = input
        .block
        .or(advertised.map(|d| d.rows))
        .unwrap_or(BLOCK_ROWS)
        .max(2);
    let mask_token = advertised.map_or(MASK_TOKEN, |d| d.mask_token as i32);
    let no_mask = input.no_mask || advertised.is_some_and(|d| !d.bidirectional);
    let from = advertised.map_or(1, |d| d.proposals_from) as usize;
    let page_size = kv_page_size();
    let rs_page = model::rs_buffer_page_size().max(1);
    // The model's opening (`<bos>` where it has one) before the raw text: a
    // gemma without it answers noise.
    let mut prompt = inferlet::chat::prefix();
    prompt.extend(model::encode(&input.prompt));
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
    // A hybrid text (qwen's GDN layers) needs its recurrent state bound; an
    // attention-only one (gemma) binds none — the drafter is the same either way.
    let rs_set = match model::pass_kind() {
        model::ForwardKind::Attention => Vec::new(),
        model::ForwardKind::Recurrent => {
            return Err("a block drafter reads attention kv; a recurrent-only text has none".into());
        }
        model::ForwardKind::Diffusion => {
            return Err("this program decodes a token at a time; a diffusion model wants a canvas loop".into());
        }
        model::ForwardKind::Hybrid => {
            let rs = RsWorkingSet::new();
            rs.alloc_buffer(2 * block.div_ceil(rs_page).max(1))
                .map_err(|why| format!("alloc rs runs: {why}"))?;
            vec![rs]
        }
    };
    let pipe = Pipeline::new();
    let adapter = input.adapter.then(|| {
        ZeroAdapter::build(
            input.adapter_layers.unwrap_or(24),
            input.adapter_hidden.unwrap_or(1024),
            input.adapter_rank.unwrap_or(16),
        )
    });

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
        if let Some(adapter) = &adapter {
            adapter.attach(&fwd)?;
        }
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
    let mut hits_by_position = vec![0u32; block as usize - from];

    for round in 0..input.rounds {
        // ── the draft: ONE pass over `[anchor, MASK x block-1]`, the trunk
        //    guarded away from its rows, the whole extent visible (the
        //    drafter's last layer is full attention with no causality).
        let mut ids = vec![mask_token; block as usize];
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
        let bound_mask = if no_mask { None } else { Some(&mask) };
        // **EVERY BLOCK ROW READS OUT.** The drafts plane is cut to the rows
        // the readout names, so a pass that leaves it at the default gets a
        // one-row plane and asking it for `block` values is a geometry
        // mismatch — which is exactly how this first failed.
        let readout = Channel::from_iter(0..block).named("readout_d");
        let out = Channel::new([block], dtype::i32).named("drafts_d");

        let fwd = ForwardPass::new();
        if let Some(adapter) = &adapter {
            adapter.attach(&fwd)?;
        }
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
            // **THE PROPOSALS ARE THE HEAD'S READOUT, OFF THE `mtp.drafts`
            // SEAM**, one id per readout row: a v1 head plants its per-slot
            // argmax there and DFlash2 its selector's walk, so this measures
            // whichever head the load carries without re-deriving either.
            out.put(&reshape(intrinsics::mtp_drafts(block), [block]));
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
        if let Some(adapter) = &adapter {
            adapter.attach(&fwd)?;
        }
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
        // DSpark's rows all propose (row `i` predicts `held + i + 1`), which
        // the load says with `proposals_from == 0`.
        let proposals: Vec<i32> = proposals[from..].to_vec();
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
