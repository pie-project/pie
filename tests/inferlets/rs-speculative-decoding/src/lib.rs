//! **SPECULATIVE DECODING WITH A BUFFERED RECURRENCE** — the fold-commit
//! programming model, driven end to end.
//!
//! A recurrence is a FOLD, not an addressed cell: once a token's activations
//! are folded into the state there is no cell to overwrite when a verifier
//! rejects the token. `mtp-speculative-decoding`'s header says why that
//! makes it RED on a hybrid SKU. This program is the shape that is not: the
//! verify window is BUFFERED (`rs-geometry.fold-len` leaves it unfolded), the
//! rejected tail is forgotten on the host (`rs-working-set.discard-buffered`),
//! and the NEXT window's fire folds the accepted prefix while buffering its
//! own rows — one fire per round, and the folded state only ever sees tokens
//! the target model itself confirmed.
//!
//! ```text
//! round r (buffer holds the b = m_{r-1} + 1 accepted tokens of round r-1)
//!   window   = [x, d_1 .. d_k]                      the pending token, then drafts
//!   fire     rows at positions base .. base + w
//!            rs-geometry { fold-len = b, buffer = grant }
//!              -> the recurrence REPLAYS the b buffered tokens ahead of the
//!                 window (the buffer read path), persists its state after
//!                 them, and buffers the window's own rows unfolded
//!   truth    = argmax(logits at each row)           row i is the truth after window[..=i]
//!   m        = |{ i : d_{i+1} == truth_i, every draft before it matched }|
//!   commit   window[1 ..= m], then truth_m
//!   discard-buffered(k - m)                         the rejected tail never happened
//!   base    += m + 1;  b = m + 1
//! ```
//!
//! Only `m + 1` tokens advance the KV length (the rejected cells sit above it
//! and the next round writes over them — `mtp-speculative-decoding`'s
//! argument, unchanged), and only `m + 1` tokens ever reach the folded state.
//!
//! # What the drafts are
//!
//! `draft = "ngram"` (default) is prompt-lookup: the longest suffix of the
//! committed text that recurs earlier in it proposes what followed it
//! (`cacheback-speculative-decoding`'s drafter). It needs nothing of the model
//! and is what makes this program runnable on a SKU that ships no draft head.
//! `draft = "mtp"` reads the model's own `mtp_logits` in the verify fire's
//! epilogue, exactly as `mtp-speculative-decoding` does; the runtime refuses
//! the bind on a model without the head.
//!
//! # What this is a gate for
//!
//! `k = 0` is the sequential control through the SAME geometry: one row a
//! fire, buffered then folded on the next. The greedy text at `k = 4` must
//! equal the greedy text at `k = 0` token for token — verification keeps the
//! target's own argmax — and the counters say how much speculation actually
//! ran. `replayed` is the buffer read path's own witness: every round but the
//! first replays the previous round's accepted prefix, so a run whose
//! `replayed` is zero exercised none of it.

use inferlet::chat;
use inferlet::eta::hybrid::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    /// Draft-window length. `0` is the sequential control, not a disabled
    /// feature: one-row windows through the same fire, the same buffer and
    /// the same commit arithmetic.
    #[serde(default = "default_k")]
    k: u32,
    /// `ngram` or `mtp`.
    #[serde(default = "default_draft")]
    draft: String,
    #[serde(default = "default_max_ngram")]
    max_ngram: usize,
    /// `none` binds no mask and trusts the engine's causal bound over the
    /// window's rows (what a chunked prefill relies on anyway); `causal`
    /// states the per-row staircase explicitly, `mtp-speculative-decoding`'s
    /// discriminator.
    #[serde(default = "default_mask_mode")]
    mask_mode: String,
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

fn default_draft() -> String {
    "ngram".into()
}

fn default_max_ngram() -> usize {
    3
}

fn default_mask_mode() -> String {
    "none".into()
}

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    text: String,
    /// The generated ids — what an identity gate compares, since distinct
    /// sequences can render to one string.
    tokens: Vec<u32>,
    count: usize,
    /// Verify fires after the prefill.
    rounds: usize,
    /// Draft tokens proposed.
    drafted: usize,
    /// Draft tokens the target's own argmax agreed with.
    accepted: usize,
    /// `accepted / drafted`, or zero when nothing was drafted.
    acceptance_rate: f64,
    /// Buffered tokens replayed ahead of a window, summed over the rounds —
    /// the buffer read path's witness.
    replayed: usize,
    /// Window rows buffered unfolded, summed over the rounds.
    buffered: usize,
    /// Buffered tokens discarded as rejected, summed over the rounds.
    discarded: usize,
    k: u32,
    draft: String,
    /// The first round's proposals and the truths they were judged against.
    draft_sample: Vec<u32>,
    truth_sample: Vec<u32>,
}

/// Prompt-lookup drafting: the longest suffix (up to `max_ngram`) of
/// `tokens` that appears earlier proposes the `k` tokens that followed it.
fn draft_from_cache(tokens: &[u32], k: usize, max_ngram: usize) -> Vec<u32> {
    if k == 0 || tokens.len() < 2 {
        return Vec::new();
    }
    let max_match = max_ngram.min(tokens.len() - 1);
    for width in (1..=max_match).rev() {
        let suffix = &tokens[tokens.len() - width..];
        for start in (0..tokens.len() - width).rev() {
            if &tokens[start..start + width] != suffix {
                continue;
            }
            let from = start + width;
            let to = (from + k).min(tokens.len());
            if from < to {
                return tokens[from..to].to_vec();
            }
        }
    }
    Vec::new()
}

/// The pages the recurrent buffer must hold for one fire: the survivors
/// (at most a page's worth of head offset before them), plus the window.
fn buffer_pages_for(survivors: u32, window: u32, page: u32) -> u32 {
    (page.saturating_sub(1) + survivors + window).div_ceil(page.max(1)).max(1)
}

/// One fire over `rows` token ids at positions `base ..`, folding `fold`
/// buffered tokens ahead of them and buffering the rows themselves;
/// `fold == None` folds everything in the forward (the prefill's shape).
/// Answers the target's argmax at every row and, when asked, the head's drafts.
#[allow(clippy::too_many_arguments)]
async fn fire(
    ws: &WorkingSet,
    rs: &[RsWorkingSet],
    pipeline: &Pipeline,
    tokens: &[u32],
    base: u32,
    fold: Option<u32>,
    buffer_pages: u32,
    k: u32,
    page_size: u32,
    max_pages: u32,
    mask_mode: &str,
    drafts_wanted: bool,
) -> Result<(Vec<u32>, Vec<u32>)> {
    let rows = tokens.len() as u32;
    let total = base + rows;
    let pages = total.div_ceil(page_size);
    let pool = max_pages * page_size;

    let ids = Channel::from_iter(tokens.iter().map(|&t| t as i32));
    let embed_indptr = Channel::from([0u32, rows]).named("embed_indptr");
    let positions = Channel::from_iter(base..total).named("positions");
    let page_list = Channel::from_iter(0..pages).named("pages");
    let page_indptr = Channel::from([0u32, pages]).named("page_indptr");
    let w_slot = Channel::from_iter((base..total).map(|p| p / page_size)).named("w_slot");
    let w_off = Channel::from_iter((base..total).map(|p| p % page_size)).named("w_off");
    let kv_len = Channel::from([total]).named("kv_len");
    // Row `i` sits at position `base + i` and may see key `j` exactly when
    // `j <= base + i`: its own prefix, itself, and nothing drafted after it.
    let bound: Vec<bool> = (base..total)
        .flat_map(|p| (0..pool).map(move |j| j <= p))
        .collect();
    let mask = Channel::from_shaped([rows, pool], bound).named("verify_mask");
    let readout = Channel::from_iter(0..rows).named("readout");
    let truth_out = Channel::new([rows], dtype::i32).named("truth");
    let drafts_out = Channel::new([k.max(1)], dtype::i32).named("drafts");
    // Where the folded boundary lands, counted over `[buffer | rows]`: the
    // survivors alone (`Some`), or everything (`None`, the fire-invariant
    // `u32::MAX`).
    let fold_len = fold.map(|n| Channel::from([n]).named("fold_len"));

    let fwd = ForwardPass::new();
    fwd.embed(&ids, &embed_indptr)?;
    fwd.readout(&readout)?;
    fwd.attention(
        Some(KvBinding {
            working_set: ws,
            geometry: KvGeometry {
                readable_pages: ..,
                writable_pages: ..,
                kv_len: &kv_len,
                pages: &page_list,
                page_indptr: &page_indptr,
                w_slot: &w_slot,
                w_off: &w_off,
                positions: &positions,
                mask: (mask_mode == "causal").then_some(&mask),
            },
        }),
        rs,
        RsGeometry {
            fold_len: fold_len.as_ref(),
            buffer: 0..buffer_pages,
        },
    )?;
    fwd.epilogue(move || {
        truth_out.put(reduce_argmax(intrinsics::logits()));
        if drafts_wanted {
            drafts_out.put(reduce_argmax(intrinsics::mtp_logits(k.max(1))));
        }
    });
    fwd.submit(pipeline).context("verify-and-extend")?;

    let truth = truth_out
        .take_host::<Vec<i32>>()
        .await?
        .into_iter()
        .map(|t| t as u32)
        .collect();
    let drafts = if drafts_wanted {
        drafts_out
            .take_host::<Vec<i32>>()
            .await?
            .into_iter()
            .map(|t| t as u32)
            .collect()
    } else {
        Vec::new()
    };
    Ok((truth, drafts))
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let k = input.k;
    if k > 32 {
        return Err("k must be at most 32".into());
    }
    if !matches!(input.draft.as_str(), "ngram" | "mtp") {
        return Err(format!("unknown draft source: {}", input.draft).into());
    }
    if !matches!(input.mask_mode.as_str(), "none" | "causal") {
        return Err(format!("unknown mask_mode: {}", input.mask_mode).into());
    }
    if model::pass_kind() == model::ForwardKind::Attention {
        return Err("this model folds no recurrent state; use mtp-speculative-decoding or \
                    cacheback-speculative-decoding, which bind no rs-working-set"
            .into());
    }
    let mtp = input.draft == "mtp";
    let w_max = k + 1;
    let page_size = kv_page_size();
    let rs_page = model::rs_buffer_page_size().max(1);

    let empty = |k: u32, draft: &str| Output {
        sampler: "rs-speculative-decoding",
        text: String::new(),
        tokens: Vec::new(),
        count: 0,
        rounds: 0,
        drafted: 0,
        accepted: 0,
        acceptance_rate: 0.0,
        replayed: 0,
        buffered: 0,
        discarded: 0,
        k,
        draft: draft.to_string(),
        draft_sample: Vec::new(),
        truth_sample: Vec::new(),
    };
    if input.max_tokens == 0 {
        return Ok(empty(k, &input.draft));
    }

    // The raw encoding, so the identity gate compares one context with the
    // sequential control and not two conversations.
    let mut prompt = model::encode(&input.prompt);
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let stop_tokens = chat::stop_tokens();

    // One KV working set and one recurrent working set for the whole
    // generation. The KV lease covers the prompt, every token the host may
    // keep, and a window whose drafts are all rejected.
    let ws = WorkingSet::new();
    let max_pages = (n + input.max_tokens as u32 + w_max).div_ceil(page_size).max(1);
    ws.reserve(max_pages).context("reserve KV")?;
    let rs = vec![RsWorkingSet::new()];
    let pipeline = Pipeline::new();

    // ── Prefill: folds everything, buffers nothing, and seeds the first window.
    let mut first = 0u32;
    let mut pending: Vec<u32> = Vec::new();
    let chunks = prefill_chunks(n, None);
    for (at, &(from, to)) in chunks.iter().enumerate() {
        let last = at + 1 == chunks.len();
        let (truth, drafts) = fire(
            &ws,
            &rs,
            &pipeline,
            &prompt[from as usize..to as usize],
            from,
            None,
            0,
            k,
            page_size,
            max_pages,
            &input.mask_mode,
            mtp && last,
        )
        .await?;
        if last {
            first = *truth.last().expect("a prefill answers one row per token");
            pending = drafts;
        }
    }

    let mut committed: Vec<u32> = prompt.clone();
    let mut generated: Vec<u32> = Vec::with_capacity(input.max_tokens);
    let (mut rounds, mut drafted, mut accepted) = (0usize, 0usize, 0usize);
    let (mut replayed, mut buffered, mut discarded) = (0usize, 0usize, 0usize);
    let (mut draft_sample, mut truth_sample) = (Vec::new(), Vec::new());

    // The seed token is the first thing the model said: nothing drafted it,
    // so nothing can reject it.
    let mut x = first;
    committed.push(x);
    generated.push(x);
    let mut stopped = stop_tokens.contains(&x) || generated.len() == input.max_tokens;
    let mut base = n;
    // How many tokens survive in the recurrent buffer, unfolded: the seed is
    // the prefill's own fold, so the first round replays nothing.
    let mut survivors: u32 = 0;

    while !stopped {
        // ── The window: the pending correct token, then the drafts.
        let drafts: Vec<u32> = if mtp {
            pending.iter().copied().take(k as usize).collect()
        } else {
            draft_from_cache(&committed, k as usize, input.max_ngram)
        };
        let mut window = Vec::with_capacity(w_max as usize);
        window.push(x);
        window.extend(drafts.iter().copied());
        let w = window.len() as u32;

        // ── The buffer must hold the survivors and the window; the grant is
        //    the guest's one allocation decision.
        let need = buffer_pages_for(survivors, w, rs_page);
        let have = rs[0].buffer_size();
        if have < need {
            rs[0]
                .alloc_buffer(need - have)
                .map_err(|why| format!("alloc {} rs buffer page(s): {why}", need - have))?;
        }

        let (truth, next) = fire(
            &ws,
            &rs,
            &pipeline,
            &window,
            base,
            Some(survivors),
            need,
            k,
            page_size,
            max_pages,
            &input.mask_mode,
            mtp,
        )
        .await?;
        rounds += 1;
        replayed += survivors as usize;
        buffered += w as usize;
        if rounds == 1 {
            draft_sample = window[1..].to_vec();
            truth_sample = truth.clone();
        }
        let proposed = window.len() - 1;
        drafted += proposed;

        // ── Verify: the longest matching prefix, and nothing after it.
        let mut m = 0usize;
        while m < proposed && window[m + 1] == truth[m] {
            m += 1;
        }
        accepted += m;

        // ── The rejected tail never happened: forget it in the buffer before
        //    the next fire, whose fold reaches exactly the accepted prefix.
        let rejected = (proposed - m) as u32;
        if rejected > 0 {
            rs[0]
                .discard_buffered(rejected)
                .map_err(|why| format!("discard {rejected} rejected token(s): {why}"))?;
            discarded += rejected as usize;
        }

        // ── Commit `window[1 ..= m]` — `window[0]` was committed last round as
        //    the correction that produced it — then the new correction.
        for &token in &window[1..=m] {
            committed.push(token);
            generated.push(token);
            if stop_tokens.contains(&token) || generated.len() == input.max_tokens {
                stopped = true;
                break;
            }
        }
        if stopped {
            break;
        }
        let correction = truth[m];
        committed.push(correction);
        generated.push(correction);
        if stop_tokens.contains(&correction) || generated.len() == input.max_tokens {
            break;
        }

        // Only the accepted run advances the length and survives in the
        // buffer; the next fire folds it ahead of its own rows.
        base += (m + 1) as u32;
        survivors = (m + 1) as u32;
        x = correction;
        pending = next;
    }
    pipeline.close();

    let acceptance_rate = if drafted == 0 {
        0.0
    } else {
        accepted as f64 / drafted as f64
    };
    Ok(Output {
        sampler: "rs-speculative-decoding",
        text: model::decode(&generated)?,
        count: generated.len(),
        rounds,
        drafted,
        accepted,
        acceptance_rate,
        replayed,
        buffered,
        discarded,
        k,
        draft: input.draft,
        draft_sample,
        truth_sample,
        tokens: generated,
    })
}
