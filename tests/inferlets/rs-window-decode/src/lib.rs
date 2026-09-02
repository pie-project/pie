//! **THE RECURRENT HALF OF A DEVICE-RESIDENT SPECULATIVE LOOP, ALONE.**
//!
//! `naive-baseline`'s decode loop — one forward pass built once, every
//! geometry channel re-put by the epilogue, `run_ahead` resubmitting — on a
//! hybrid model, with the recurrent state driven the way a speculative
//! verifier will drive it: each step's row is BUFFERED (not folded), and the
//! next step folds it off a length the epilogue computed on the device
//! (`rs-geometry.fold-len`). The host takes tokens out and nothing else.
//!
//! This is the `RsVerb::Window` verb end to end: the working set's buffer is
//! two page runs the runtime alternates; a fire writes its rows into one run
//! and replays the first `fold-len` tokens of the other, persisting the bank
//! exactly after them. Here the window is one row and `fold-len` is 1 every
//! step, so the decode must answer token for token what `text-completion`
//! answers folding in the forward — that identity is what this program is
//! measured against. The speculative loop (k drafts, `m` accepted, `fold-len
//! = m + 1`) is the same program with a wider window and a verifier in the
//! epilogue.

use std::ops::ControlFlow;

use inferlet::eta::hybrid::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
}

fn default_prompt() -> String {
    "The capital of France is".into()
}

fn default_max_tokens() -> usize {
    8
}

#[derive(Serialize)]
struct Output {
    text: String,
    count: usize,
    tokens: Vec<u32>,
}

fn greedy(logits: Tensor) -> Tensor {
    reshape(reduce_argmax(&logits), [1])
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if model::pass_kind() != model::ForwardKind::Hybrid {
        return Err("rs-window-decode drives a recurrent state; this model has none".into());
    }
    let max_tokens = input.max_tokens;
    if max_tokens == 0 {
        return Ok(Output {
            text: String::new(),
            count: 0,
            tokens: Vec::new(),
        });
    }
    let page_size = kv_page_size();
    let rs_page = model::rs_buffer_page_size().max(1);

    let mut prompt = model::encode(&input.prompt);
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let max_pages = (n + max_tokens as u32 + 1).div_ceil(page_size).max(1);

    let ws = WorkingSet::new();
    ws.reserve(max_pages).context("reserve KV")?;
    let rs = RsWorkingSet::new();
    // TWO runs of one window each: the runtime alternates them per fire.
    // A window here is one row, so a run is one page.
    let run_pages = 1u32.div_ceil(rs_page).max(1);
    rs.alloc_buffer(2 * run_pages).map_err(|why| format!("alloc rs window runs: {why}"))?;
    let rs_set = vec![rs];
    let pipe = Pipeline::new();

    // The decode pass's fold length. Bound to a channel, it is the DEVICE's
    // number: the runtime resolves it off the `rs_fold_len` port every fire
    // and plans the window verb, so the host's seed (0: nothing to replay on
    // the first step) is just the first cell the port reads; every decode
    // epilogue puts the next (1: the row it just buffered).
    let fold_len = Channel::from([0u32]).named("fold_len");

    // ── PREFILL: folds everything in the forward, chunked as every inferlet.
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let chunks = prefill_chunks(n, None);
    let mut first = 0i32;
    for &(base, end) in &chunks {
        let len = end - base;
        let toks = Channel::from(&prompt_i32[base as usize..end as usize]).named("toks_p");
        let embed_indptr = Channel::from([0u32, len]).named("embed_indptr_p");
        let positions = Channel::from_iter(base..end).named("positions_p");
        let pages = Channel::from_iter(0..max_pages).named("pages_p");
        let page_indptr = Channel::from([0u32, end.div_ceil(page_size)]).named("page_indptr_p");
        let w_slot = Channel::from_iter((base..end).map(|p| p / page_size)).named("w_slot_p");
        let w_off = Channel::from_iter((base..end).map(|p| p % page_size)).named("w_off_p");
        let kv_len = Channel::from([end]).named("kv_len_p");
        let tok_out = Channel::new([1], dtype::i32).named("tok_out_p");

        let fwd = ForwardPass::new();
        fwd.embed(&toks, &embed_indptr)?;
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
            tok_out.put(&greedy(intrinsics::logits()));
        });
        fwd.submit(&pipe).with_context(|| format!("prefill submit @{base}"))?;
        first = tok_out
            .take_host::<i32>()
            .await
            .with_context(|| format!("prefill drain @{base}"))?;
    }
    let mut generated: Vec<u32> = vec![first as u32];

    // ── DECODE: one pass, device-resident; each step buffers its row and
    //    the next folds it.
    if generated.len() < max_tokens {
        let tok_in = Channel::from([first]).named("tok_in");
        let tok_out = Channel::new([1], dtype::i32)
            .capacity(channel_capacity() as u32)
            .named("tok_out");
        let lane1 = Channel::from([0u32, 1u32]).named("embed_indptr");
        let positions = Channel::from([n]).named("positions");
        let pages = Channel::from_iter(0..max_pages).named("pages");
        let page_indptr = Channel::from([0u32, (n + 1).div_ceil(page_size)]).named("page_indptr");
        let w_slot = Channel::from([n / page_size]).named("w_slot");
        let w_off = Channel::from([n % page_size]).named("w_off");
        let kv_len = Channel::from([n + 1]).named("kv_len");

        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, &lane1)?;
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
            let length = kv_len.take();
            let token = greedy(intrinsics::logits());
            let next_length = &length + 1u32;
            let page_count = next_length.div_ceil(page_size);

            tok_in.put(&token);
            kv_len.put(&next_length);
            positions.put(&length);
            w_slot.put(&length / page_size);
            w_off.put(&length % page_size);
            page_indptr.put(indptr(1, &page_count));
            // The row this step buffered is the whole window, and it is
            // accepted (greedy: nothing to reject): the next step folds one.
            // `rs_fold_len` is a CONSUMING port (the fire that folds on it
            // takes the cell), so the epilogue only puts the next one.
            let one = reshape(reduce_sum(cast(eq(&token, &token), dtype::u32)), [1]);
            fold_len.put(&one);
            tok_out.put(&token);
        });

        let budget = max_tokens - 1;
        run_ahead(&pipe, &fwd, budget, async || {
            let t = tok_out
                .take_host::<i32>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            generated.push(t as u32);
            Ok(ControlFlow::Continue(()))
        })
        .await?;
    }
    pipe.close();

    Ok(Output {
        text: model::decode(&generated)?,
        count: generated.len(),
        tokens: generated,
    })
}
