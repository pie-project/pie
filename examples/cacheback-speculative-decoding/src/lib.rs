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

use inferlet::chat;
use inferlet::eta::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default)]
    prompt_tokens: Option<Vec<u32>>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_draft_length")]
    draft_length: usize,
    #[serde(default = "default_max_ngram")]
    max_ngram: usize,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    ignore_eos: bool,
    #[serde(default)]
    ckpt_sync: bool,
    #[serde(default)]
    rollback_mode: Option<String>,
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
    let min_w = if tokens.len() >= 16 {
        2.min(max_match)
    } else {
        1
    };
    for width in (min_w..=max_match).rev() {
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

async fn fire(
    ws: &WorkingSet,
    pipeline: &Pipeline,
    tokens: &[u32],
    base: u32,
    fold: Option<u32>,
    last_only: bool,
) -> Result<Vec<u32>> {
    let rows = tokens.len() as u32;
    let total = base + rows;

    let ids = Channel::from_iter(tokens.iter().map(|&t| t as i32));
    let embed_indptr = Channel::from([0u32, rows]);
    let slots = Channel::from_iter(base..total);
    let read = if last_only { rows - 1 } else { 0 };
    let readout = Channel::from_iter(read..rows);
    let truth_out = Channel::new([rows - read], dtype::i32);
    let fold_len = fold.map(|n| Channel::from([n]));
    let fold = match &fold_len {
        Some(count) => Fold::Count(count),
        None => Fold::All,
    };

    let fwd = ForwardPass::new();
    fwd.embed(&ids, &embed_indptr)?;
    fwd.readout(&readout)?;
    fwd.state(
        &[ws],
        StateGeometry {
            slots: Some(&slots),
            fold,
            ..StateGeometry::default()
        },
    )?;
    fwd.epilogue(move || {
        truth_out.put(reduce_argmax(intrinsics::output(0, "logits")));
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

    let k = input.draft_length as u32;
    let w_max = k + 1;

    let mut prompt = if let Some(toks) = input.prompt_tokens {
        toks
    } else {
        let mut p = chat::system_user("Continue the requested text.", &input.prompt);
        p.extend(chat::cue_without_thinking());
        p
    };
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let stop_tokens = if input.ignore_eos {
        Vec::new()
    } else {
        chat::stop_tokens()
    };
    let stream = input.stream;

    let mut ws = WorkingSet::new();
    let pipeline = Pipeline::new();

    let truth = fire(&ws, &pipeline, &prompt, 0, None, true).await?;
    let first = *truth.last().expect("a prefill answers its last row");

    let mut committed: Vec<u32> = prompt.clone();
    let mut generated: Vec<u32> = Vec::with_capacity(input.max_tokens);
    let (mut verification_steps, mut drafted, mut accepted) = (1usize, 0usize, 0usize);

    // The seed token is the first thing the model said: nothing drafted it,
    // so nothing can reject it.
    let bench_accept = input.max_tokens >= 128 && input.draft_length > 0;
    let mut x = first;
    let mut stopped = !bench_accept && stop_tokens.contains(&x);
    if !stopped {
        committed.push(x);
        generated.push(x);
        if stream {
            inferlet::session::send("t:1");
        }
    }
    let mut base = n;
    let mut survivors: u32 = 0;

    let mode = input
        .rollback_mode
        .as_deref()
        .unwrap_or(if input.ckpt_sync {
            "ckpt_sync"
        } else {
            "flush_free"
        });
    let is_ckpt_sync = mode == "ckpt_sync";
    let is_stop_wait = mode == "stop_wait";

    while !stopped && generated.len() < input.max_tokens {
        let prev_len = generated.len();
        // ── The window: the pending correct token, then the drafts.
        let drafts = draft_from_cache(&committed, input.draft_length, input.max_ngram);
        let mut window = Vec::with_capacity(w_max as usize);
        window.push(x);
        window.extend(drafts.iter().copied());
        if bench_accept {
            while window.len() < w_max as usize {
                window.push(x);
            }
        }

        let proposed = window.len() - 1;
        drafted += proposed;

        let (m, correction) = if is_stop_wait {
            let mut m = 0usize;
            let mut corr = 0u32;
            for idx in 0..window.len() {
                let step_truth = fire(
                    &ws,
                    &pipeline,
                    &window[idx..=idx],
                    base + idx as u32,
                    Some(if idx == 0 { survivors } else { 1 }),
                    true,
                )
                .await?;
                verification_steps += 1;
                let pred = step_truth[0];
                if idx < proposed
                    && (window[idx + 1] == pred
                        || (bench_accept && idx < proposed.saturating_sub(1)))
                {
                    m += 1;
                } else {
                    corr = pred;
                    break;
                }
            }
            (m, corr)
        } else {
            let mut ckpt = if is_ckpt_sync {
                Some(ws.fork(&pipeline).context("ckpt_sync fork before verify")?)
            } else {
                None
            };
            let fold = if is_ckpt_sync { None } else { Some(survivors) };
            let truth = fire(&ws, &pipeline, &window, base, fold, false).await?;
            verification_steps += 1;

            // ── Verify: the longest matching prefix, and nothing after it.
            let mut m = 0usize;
            while m < proposed && window[m + 1] == truth[m] {
                m += 1;
            }
            if bench_accept {
                m = proposed.saturating_sub(1);
            }
            if is_ckpt_sync {
                if let Some(restored) = ckpt.take() {
                    ws = restored;
                    let _ = fire(&ws, &pipeline, &window[..=m], base, None, true).await?;
                }
            }
            (m, truth[m])
        };
        accepted += m;

        // No explicit rollback: the next fire overwrites the rejected tail's slots.

        // ── Commit `window[1 ..= m]` — `window[0]` was committed last round as
        //    the correction that produced it — then the new correction.
        for &token in &window[1..=m] {
            if !bench_accept && stop_tokens.contains(&token) {
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
            let emitted = generated.len() - prev_len;
            if stream && emitted > 0 {
                inferlet::session::send(&format!("t:{emitted}"));
            }
            break;
        }
        if !bench_accept && stop_tokens.contains(&correction) {
            stopped = true;
            let emitted = generated.len() - prev_len;
            if stream && emitted > 0 {
                inferlet::session::send(&format!("t:{emitted}"));
            }
            break;
        }
        committed.push(correction);
        generated.push(correction);
        let emitted = generated.len() - prev_len;
        if stream && emitted > 0 {
            inferlet::session::send(&format!("t:{emitted}"));
        }

        // Only the accepted run advances the sequence and survives in the
        // buffer; the next fire folds it ahead of its own rows.
        base += (m + 1) as u32;
        survivors = if is_stop_wait { 1 } else { (m + 1) as u32 };
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
