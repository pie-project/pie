//! CacheBack-style speculative decoding with a prompt-lookup draft cache.
//!
//! The drafter searches the committed token history for the longest matching
//! suffix and reuses the tokens that followed its previous occurrence. One
//! target-model forward per round verifies the whole window — the pending
//! correct token, then the drafts — and supplies the next correction.
//!
//! ## The correctness property, and how to test it
//!
//! Verification is greedy (`reduce_argmax`), and a draft token is kept only
//! when it equals what the target model would itself have produced. So
//! speculation here is a pure latency optimization: it must change *how many
//! forward passes* run and nothing else about the output.
//!
//! `draft_length = 0` makes that testable without a second inferlet.
//! `draft_from_cache` returns empty immediately, every window is one row, and
//! the loop degenerates to sequential greedy decoding — through the same
//! prompt, the same stop tokens and the same fire. Setting it to 0 versus 4
//! is therefore a controlled A/B in which the token sequence must come out
//! **identical**. If a rejected draft ever leaked into later state, the two
//! would diverge.
//!
//! ## How the state is kept — and why nothing is rebuilt
//!
//! One KV working set lives for the whole generation. A rejected draft's KV
//! cells are addressed, so the next round simply overwrites them: its rows
//! land at the positions the rejected tail occupied, and `kv_len` never
//! counts past the committed prefix plus the window in flight.
//!
//! On a hybrid model the recurrence is a fold, not an addressed cell, so the
//! window is *buffered* rather than folded (`forward-hybrid.wit`,
//! `rs-geometry`): each fire folds the previously accepted run ahead of its
//! own rows, replays those rows over the recurrent buffer, and the rejected
//! tail is dropped from the buffer with `discard_buffered` before the next
//! fire. The state that persists is exactly the state of the committed
//! prefix — the same shape `rs-speculative-decoding` documents — and the
//! per-window rebuild the previous revision paid for (O(n) per round) is gone.
//! On a pure-attention model the recurrent binding is empty and the same
//! loop is plain KV speculation.

use inferlet::chat;
use inferlet::eta::hybrid::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_draft_length")]
    draft_length: usize,
    #[serde(default = "default_max_ngram")]
    max_ngram: usize,
}

fn default_prompt() -> String {
    "Repeat this pattern: red green blue, red green blue, red green".into()
}

fn default_max_tokens() -> usize {
    256
}

fn default_draft_length() -> usize {
    4
}

fn default_max_ngram() -> usize {
    8
}

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    text: String,
    /// Generated token ids. Reported so an equivalence test can compare the
    /// exact sequence rather than its detokenization, since distinct token
    /// sequences can render to the same string.
    tokens: Vec<u32>,
    prompt_tokens: usize,
    count: usize,
    draft_length: usize,
    /// Target-model forward passes actually run past the prefill. At
    /// `draft_length = 0` this is `count + stopped as usize`, which is
    /// precisely what makes that setting a sequential greedy control: same
    /// prompt, same stop tokens, same fire, no speculation. Below `count`
    /// means speculation is paying off.
    ///
    /// **THE `+ stopped` IS THE STOP TOKEN'S OWN PASS.** A run the model ends
    /// itself spends one forward pass producing the stop token, and that token
    /// is not part of the answer so it is not in `count`. A run truncated by
    /// `max_tokens` never pays it. Reading the invariant as a bare
    /// `verification_steps == count` therefore holds only on prompts long
    /// enough to run out the budget — which is a property of the model and the
    /// prompt, not of this loop.
    verification_steps: usize,
    /// Whether the model ended the run with a stop token rather than the
    /// caller's `max_tokens` doing it.
    stopped: bool,
    drafted: usize,
    accepted: usize,
    acceptance_rate: f64,
}

fn draft_from_cache(tokens: &[u32], draft_length: usize, max_ngram: usize) -> Vec<u32> {
    if draft_length == 0 || tokens.len() < 2 {
        return Vec::new();
    }

    let max_match = max_ngram.min(tokens.len() - 1);
    for width in (1..=max_match).rev() {
        let suffix = &tokens[tokens.len() - width..];
        for start in (0..tokens.len() - width).rev() {
            if &tokens[start..start + width] != suffix {
                continue;
            }
            let continuation = start + width;
            let end = (continuation + draft_length).min(tokens.len());
            if continuation < end {
                return tokens[continuation..end].to_vec();
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
/// `fold == None` folds everything in the forward (the prefill's shape, and
/// the only shape on a pure-attention model, where `rs` is empty).
/// Answers the target's argmax at every row.
#[allow(clippy::too_many_arguments)]
async fn fire(
    ws: &WorkingSet,
    rs: &[RsWorkingSet],
    pipeline: &Pipeline,
    tokens: &[u32],
    base: u32,
    fold: Option<u32>,
    buffer_pages: u32,
    page_size: u32,
) -> Result<Vec<u32>> {
    let rows = tokens.len() as u32;
    let total = base + rows;
    let pages = total.div_ceil(page_size);

    let ids = Channel::from_iter(tokens.iter().map(|&t| t as i32));
    let embed_indptr = Channel::from([0u32, rows]).named("embed_indptr");
    let positions = Channel::from_iter(base..total).named("positions");
    let page_list = Channel::from_iter(0..pages).named("pages");
    let page_indptr = Channel::from([0u32, pages]).named("page_indptr");
    let w_slot = Channel::from_iter((base..total).map(|p| p / page_size)).named("w_slot");
    let w_off = Channel::from_iter((base..total).map(|p| p % page_size)).named("w_off");
    let kv_len = Channel::from([total]).named("kv_len");
    let readout = Channel::from_iter(0..rows).named("readout");
    let truth_out = Channel::new([rows], dtype::i32).named("truth");
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
                mask: None,
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
    });
    fwd.submit(pipeline).context("verify-and-extend")?;

    Ok(truth_out
        .take_host::<Vec<i32>>()
        .await?
        .into_iter()
        .map(|t| t as u32)
        .collect())
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let empty = || Output {
        sampler: "cacheback-speculative",
        text: String::new(),
        tokens: Vec::new(),
        prompt_tokens: 0,
        count: 0,
        draft_length: input.draft_length,
        verification_steps: 0,
        stopped: false,
        drafted: 0,
        accepted: 0,
        acceptance_rate: 0.0,
    };
    if input.max_tokens == 0 {
        return Ok(empty());
    }
    let hybrid = match model::pass_kind() {
        model::ForwardKind::Attention => false,
        model::ForwardKind::Hybrid => true,
        other => {
            return Err(format!(
                "this program verifies drafts against a KV cache; the model's forward is {other:?}"
            )
            .into())
        }
    };

    let k = input.draft_length as u32;
    let w_max = k + 1;
    let page_size = kv_page_size();
    let rs_page = model::rs_buffer_page_size().max(1);

    let mut prompt = chat::system_user("Continue the requested text.", &input.prompt);
    prompt.extend(chat::cue());
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let stop_tokens = chat::stop_tokens();

    // One KV working set (and on a hybrid model one recurrent working set)
    // for the whole generation. The KV lease covers the prompt, every token
    // the host may keep, and a window whose drafts are all rejected.
    let ws = WorkingSet::new();
    let max_pages = (n + input.max_tokens as u32 + w_max).div_ceil(page_size).max(1);
    ws.reserve(max_pages).context("reserve KV")?;
    let rs: Vec<RsWorkingSet> = if hybrid { vec![RsWorkingSet::new()] } else { Vec::new() };
    let pipeline = Pipeline::new();

    // ── Prefill: folds everything, buffers nothing, and seeds the first window.
    let mut first = 0u32;
    let chunks = prefill_chunks(n, None);
    for (at, &(from, to)) in chunks.iter().enumerate() {
        let truth = fire(
            &ws,
            &rs,
            &pipeline,
            &prompt[from as usize..to as usize],
            from,
            None,
            0,
            page_size,
        )
        .await?;
        if at + 1 == chunks.len() {
            first = *truth.last().expect("a prefill answers one row per token");
        }
    }

    let mut committed: Vec<u32> = prompt.clone();
    let mut generated: Vec<u32> = Vec::with_capacity(input.max_tokens);
    let (mut verification_steps, mut drafted, mut accepted) = (1usize, 0usize, 0usize);

    // The seed token is the first thing the model said: nothing drafted it,
    // so nothing can reject it.
    let mut x = first;
    let mut stopped = stop_tokens.contains(&x);
    if !stopped {
        committed.push(x);
        generated.push(x);
    }
    let mut base = n;
    // How many tokens survive in the recurrent buffer, unfolded: the seed is
    // the prefill's own fold, so the first round replays nothing.
    let mut survivors: u32 = 0;

    while !stopped && generated.len() < input.max_tokens {
        // ── The window: the pending correct token, then the drafts.
        let drafts = draft_from_cache(&committed, input.draft_length, input.max_ngram);
        let mut window = Vec::with_capacity(w_max as usize);
        window.push(x);
        window.extend(drafts.iter().copied());
        let w = window.len() as u32;

        // ── On a hybrid model the buffer must hold the survivors and the
        //    window; the grant is the guest's one allocation decision.
        let need = if hybrid { buffer_pages_for(survivors, w, rs_page) } else { 0 };
        if hybrid {
            let have = rs[0].buffer_size();
            if have < need {
                rs[0]
                    .alloc_buffer(need - have)
                    .map_err(|why| format!("alloc {} rs buffer page(s): {why}", need - have))?;
            }
        }

        let truth = fire(
            &ws,
            &rs,
            &pipeline,
            &window,
            base,
            hybrid.then_some(survivors),
            need,
            page_size,
        )
        .await?;
        verification_steps += 1;
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
        //    Its KV cells are overwritten by the next window's rows.
        let rejected = (proposed - m) as u32;
        if hybrid && rejected > 0 {
            rs[0]
                .discard_buffered(rejected)
                .map_err(|why| format!("discard {rejected} rejected token(s): {why}"))?;
        }

        // ── Commit `window[1 ..= m]` — `window[0]` was committed last round as
        //    the correction that produced it — then the new correction.
        for &token in &window[1..=m] {
            if stop_tokens.contains(&token) {
                stopped = true;
                break;
            }
            committed.push(token);
            generated.push(token);
            if generated.len() == input.max_tokens {
                break;
            }
        }
        if stopped || generated.len() == input.max_tokens {
            break;
        }
        let correction = truth[m];
        if stop_tokens.contains(&correction) {
            stopped = true;
            break;
        }
        committed.push(correction);
        generated.push(correction);

        // Only the accepted run advances the length and survives in the
        // buffer; the next fire folds it ahead of its own rows.
        base += (m + 1) as u32;
        survivors = (m + 1) as u32;
        x = correction;
    }

    let acceptance_rate = if drafted == 0 {
        0.0
    } else {
        accepted as f64 / drafted as f64
    };
    Ok(Output {
        sampler: "cacheback-speculative",
        text: model::decode(&generated)?,
        tokens: generated.clone(),
        prompt_tokens: prompt.len(),
        count: generated.len(),
        draft_length: input.draft_length,
        verification_steps,
        stopped,
        drafted,
        accepted,
        acceptance_rate,
    })
}
