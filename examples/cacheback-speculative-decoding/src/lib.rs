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

async fn fire(
    ws: &WorkingSet,
    pipeline: &Pipeline,
    tokens: &[u32],
    base: u32,
    fold: u32,
) -> Result<Vec<u32>> {
    let rows = tokens.len() as u32;
    let total = base + rows;

    let ids = Channel::from_iter(tokens.iter().map(|&t| t as i32));
    let embed_indptr = Channel::from([0u32, rows]);
    let slots = Channel::from_iter(base..total);
    let readout = Channel::from_iter(0..rows);
    let truth_out = Channel::new([rows], dtype::i32);
    let fold = Channel::from([fold]);

    let fwd = ForwardPass::new();
    fwd.embed(&ids, &embed_indptr)?;
    fwd.readout(&readout)?;
    fwd.state(
        &[ws],
        StateGeometry {
            slots: Some(&slots),
            fold: Fold::Count(&fold),
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
    let k = input.draft_length as u32;
    let w_max = k + 1;

    let mut committed = chat::system_user("Continue the requested text.", &input.prompt);
    committed.extend(chat::cue());
    let prompt_tokens = committed.len();
    let (&last, head) = committed.split_last().ok_or("empty prompt")?;
    let stop_tokens = chat::stop_tokens();

    let ws = WorkingSet::new();
    let pipeline = Pipeline::new();
    if !head.is_empty() {
        let fill = ForwardPass::new();
        fill.embed(
            &Channel::from_iter(head.iter().map(|&t| t as i32)),
            &Channel::from([0u32, head.len() as u32]),
        )?;
        fill.state(&[&ws], StateGeometry::folding())?;
        fill.submit(&pipeline).context("prefill submit")?;
    }

    let mut generated: Vec<u32> = Vec::with_capacity(input.max_tokens);
    let (mut verification_steps, mut drafted, mut accepted) = (0usize, 0usize, 0usize);
    let mut x = last;
    let mut stopped = false;
    let mut base = head.len() as u32;
    let mut survivors: u32 = 0;

    while !stopped && generated.len() < input.max_tokens {
        // ── The window: the pending correct token, then the drafts.
        let drafts = draft_from_cache(&committed, input.draft_length, input.max_ngram);
        let mut window = Vec::with_capacity(w_max as usize);
        window.push(x);
        window.extend(drafts.iter().copied());

        let truth = fire(&ws, &pipeline, &window, base, survivors).await?;
        verification_steps += 1;
        let proposed = window.len() - 1;
        drafted += proposed;

        // ── Verify: the longest matching prefix, and nothing after it.
        let mut m = 0usize;
        while m < proposed && window[m + 1] == truth[m] {
            m += 1;
        }
        accepted += m;

        // No explicit rollback: the next fire overwrites the rejected tail's slots.

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

        // Only the accepted run advances the sequence and survives in the
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
        prompt_tokens,
        count: generated.len(),
        draft_length: input.draft_length,
        verification_steps,
        stopped,
        drafted,
        accepted,
        acceptance_rate,
    })
}
