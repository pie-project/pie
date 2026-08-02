//! CacheBack-style speculative decoding with a prompt-lookup draft cache.
//!
//! The drafter searches the committed token history for the longest matching
//! suffix and reuses the tokens that followed its previous occurrence. One
//! target-model forward verifies the whole draft and supplies a correction or
//! bonus token. This reference implementation rebuilds the target KV for each
//! verification window so rejected draft state can never leak into later steps.
//!
//! ## The correctness property, and how to test it
//!
//! Verification is greedy (`reduce_argmax`), and a draft token is kept only
//! when it equals what the target model would itself have produced. So
//! speculation here is a pure latency optimization: it must change *how many
//! forward passes* run and nothing else about the output.
//!
//! `draft_length = 0` makes that testable without a second inferlet.
//! `draft_from_cache` returns empty immediately, `verify` reduces to a
//! one-row readout, and the loop degenerates to sequential greedy decoding —
//! through the same prompt, the same stop tokens and the same `verify()`.
//! Setting it to 0 versus 4 is therefore a controlled A/B in which the token
//! sequence must come out **identical**. If the per-window KV rebuild ever
//! stopped isolating rejected drafts, the two would diverge.

use inferlet::chat;
use inferlet::ptir::attention::prelude::*;
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
    /// Target-model forward passes actually run. At `draft_length = 0` this
    /// equals `count`, which is precisely what makes that setting a sequential
    /// greedy control: same prompt, same stop tokens, same `verify()`, no
    /// speculation. Below `count` means speculation is paying off.
    verification_steps: usize,
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

async fn verify(committed: &[u32], draft: &[u32], page_size: u32) -> Result<Vec<u32>> {
    if committed.is_empty() {
        return Err("cannot verify from an empty committed sequence".into());
    }

    let mut input = committed.to_vec();
    input.extend_from_slice(draft);
    let total = input.len() as u32;
    let rows = draft.len() as u32 + 1;
    let readout_start = committed.len() as u32 - 1;
    let readout = (readout_start..readout_start + rows).collect::<Vec<_>>();

    let ws = WorkingSet::new();
    let max_pages = total.div_ceil(page_size);
    ws.reserve(max_pages).context("reserve verification KV")?;
    let tokens = Channel::from_iter(input.iter().map(|&token| token as i32));
    let embed_indptr = Channel::from([0u32, total]).named("embed_indptr");
    let positions = Channel::from_iter(0..total).named("positions");
    let pages = Channel::from_iter(0..max_pages).named("pages");
    let page_indptr = Channel::from([0u32, max_pages]).named("page_indptr");
    let w_slot = Channel::from_iter((0..total).map(|p| p / page_size)).named("w_slot");
    let w_off = Channel::from_iter((0..total).map(|p| p % page_size)).named("w_off");
    let kv_len = Channel::from([total]).named("kv_len");
    let target_out = Channel::new([rows], dtype::i32).named("target_tokens");
    let readout = Channel::from(readout).named("readout");

    let fwd = ForwardPass::new();
    fwd.embed(&tokens, &embed_indptr)?;
    fwd.readout(&readout)?;
    fwd.attention(
        &ws,
        KvGeometry {
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
    )?;
    fwd.epilogue(move || {
        target_out.put(reduce_argmax(intrinsics::logits()));
    });

    let pipeline = Pipeline::new();
    fwd.submit(&pipeline).context("verify cached draft")?;
    let target = target_out
        .take_host::<Vec<i32>>()
        .await?
        .into_iter()
        .map(|token| token as u32)
        .collect();
    pipeline.close();
    Ok(target)
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if input.max_tokens == 0 {
        return Ok(Output {
            sampler: "cacheback-speculative",
            text: String::new(),
            tokens: Vec::new(),
            prompt_tokens: 0,
            count: 0,
            draft_length: input.draft_length,
            verification_steps: 0,
            drafted: 0,
            accepted: 0,
            acceptance_rate: 0.0,
        });
    }

    let page_size = kv_page_size();

    let mut committed = chat::system_user("Continue the requested text.", &input.prompt);
    committed.extend(chat::cue());
    if committed.is_empty() {
        committed.push(0);
    }
    let prompt_len = committed.len();
    let stop_tokens = chat::stop_tokens();
    let mut generated = Vec::with_capacity(input.max_tokens);
    let mut total_drafted = 0usize;
    let mut total_accepted = 0usize;
    let mut verification_steps = 0usize;
    let mut stopped = false;

    while generated.len() < input.max_tokens && !stopped {
        let draft = draft_from_cache(&committed, input.draft_length, input.max_ngram);
        total_drafted += draft.len();
        verification_steps += 1;

        let target = verify(&committed, &draft, page_size).await?;
        if target.len() != draft.len() + 1 {
            return Err(format!(
                "verification returned {} tokens for a {}-token draft",
                target.len(),
                draft.len()
            ));
        }

        let mut accepted = Vec::new();
        let mut rejected = false;
        for (index, &draft_token) in draft.iter().enumerate() {
            if target[index] == draft_token {
                accepted.push(draft_token);
                total_accepted += 1;
            } else {
                accepted.push(target[index]);
                rejected = true;
                break;
            }
        }
        if !rejected {
            accepted.push(target[draft.len()]);
        }

        for token in accepted {
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
    }

    let acceptance_rate = if total_drafted == 0 {
        0.0
    } else {
        total_accepted as f64 / total_drafted as f64
    };
    Ok(Output {
        sampler: "cacheback-speculative",
        text: model::decode(&generated)?,
        prompt_tokens: prompt_len,
        count: generated.len(),
        draft_length: input.draft_length,
        verification_steps,
        drafted: total_drafted,
        accepted: total_accepted,
        acceptance_rate,
        tokens: generated,
    })
}
