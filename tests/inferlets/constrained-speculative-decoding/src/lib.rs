//! Grammar-constrained decoding composed with speculative decoding.
//!
//! Two mechanisms that are usually implemented in isolation, run together on
//! one token stream: a JSON-Schema grammar decides which tokens are legal at
//! each position, and a prompt-lookup drafter proposes several tokens at once
//! so one target forward can commit more than one of them.
//!
//! ## Why the composition is not free
//!
//! Constrained decoding is *positional*: the legal set at position `t` depends
//! on every token accepted before `t`. Speculative decoding proposes tokens for
//! positions `t .. t+k` and verifies them in a single forward. So the verifier
//! needs `k+1` different masks in that one pass — the grammar state has to be
//! run forward over the draft *before* the draft is known to be correct, and
//! then wound back to whatever the verifier actually accepted.
//!
//! That is exactly `matcher.fork()` and `matcher.rollback(n)`. Without them a
//! guest can only `reset()` and replay the entire accepted prefix token by
//! token on every rejection, which costs more than speculation saves.
//!
//! Two consequences fall out of the composition and are implemented here:
//!
//! - **Draft truncation.** A draft token the grammar forbids can never be
//!   accepted, because the verifier's argmax is taken over the masked logits
//!   and the mask excludes it. Such a draft is truncated at that point instead
//!   of spending a readout row on a foregone conclusion.
//! - **Correction tokens are always legal.** The verifier's replacement token
//!   is a masked argmax, so a rejection cannot push the grammar into an invalid
//!   state. Constraint satisfaction is preserved by construction, not by a
//!   post-hoc check.
//!
//! ## The correctness property, and how to test it
//!
//! Verification is greedy and grammar-masked at every row, and a draft token is
//! kept only when it equals what the target model would itself have produced
//! under the same mask. Speculation is therefore a pure latency optimization on
//! top of constrained decoding: it must change *how many forward passes run*
//! and nothing else about the output.
//!
//! `draft_length = 0` makes that testable without a second inferlet. The
//! drafter returns empty, `verify` reduces to a one-row masked readout, and the
//! loop degenerates to sequential constrained greedy decoding — through the
//! same prompt, the same schema and the same `verify()`. Setting it to 0 versus
//! 4 is a controlled A/B in which the token sequence must come out
//! **identical**.
//!
//! Each step additionally asserts an invariant that the ABI must satisfy for
//! any of this to be sound: after the grammar has been walked forward over a
//! `k`-token draft and rolled back by `k`, its mask must equal that of a fork
//! taken before the walk. A rollback that did not fully restore state would
//! silently corrupt the constraint instead of failing, so it is checked here
//! rather than assumed.

use inferlet::chat;
use inferlet::grammar::{Grammar, Matcher};
use inferlet::mask::unpack_mask;
use inferlet::ptir::attention::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_schema")]
    schema: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    /// Tokens proposed per verification. `0` disables speculation and makes
    /// this a sequential constrained decoder — the control arm of the A/B.
    #[serde(default = "default_draft_length")]
    draft_length: usize,
    #[serde(default = "default_max_ngram")]
    max_ngram: usize,
}

fn default_prompt() -> String {
    "Generate a profile for a fictional software engineer named Alice.".into()
}

fn default_schema() -> String {
    r#"{
        "type": "object",
        "properties": {
            "name": { "type": "string", "minLength": 1 },
            "age": { "type": "integer", "minimum": 0, "maximum": 150 },
            "skills": {
                "type": "array",
                "items": { "type": "string" },
                "minItems": 1
            }
        },
        "required": ["name", "age", "skills"],
        "additionalProperties": false
    }"#
    .into()
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
    /// Generated token ids. Reported so an equivalence test compares the exact
    /// sequence rather than its detokenization, since distinct token sequences
    /// can render to the same string.
    tokens: Vec<u32>,
    count: usize,
    draft_length: usize,
    /// Target-model forward passes actually run. At `draft_length = 0` this
    /// equals `count`. Below `count` means speculation is paying off.
    verification_steps: usize,
    drafted: usize,
    accepted: usize,
    /// Draft tokens dropped before verification because the grammar already
    /// forbade them. These cost no forward-pass row at all, which is a saving
    /// unique to the composition.
    grammar_pruned: usize,
    acceptance_rate: f64,
    /// How many times the fork/rollback invariant was checked and held.
    rollback_checks: usize,
}

/// Longest-suffix prompt lookup over the committed history (CacheBack-style).
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

/// One target forward over `committed ++ draft`, reading out `draft.len() + 1`
/// rows and taking a grammar-masked argmax at each.
///
/// `masks[i]` is the allowed set for the position that row `i` predicts, i.e.
/// the grammar state after `committed ++ draft[..i]`. The KV working set is
/// rebuilt per call, so a rejected draft can never leak into a later window.
async fn verify(
    committed: &[u32],
    draft: &[u32],
    masks: &[Vec<bool>],
    vocab: u32,
    page_size: u32,
) -> Result<Vec<u32>> {
    if committed.is_empty() {
        return Err("cannot verify from an empty committed sequence".into());
    }
    let rows = draft.len() as u32 + 1;
    if masks.len() != rows as usize {
        return Err(format!(
            "verification needs one mask per row: got {} for {rows} rows",
            masks.len()
        ));
    }

    let mut input = committed.to_vec();
    input.extend_from_slice(draft);
    let total = input.len() as u32;
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
    let readout = Channel::from(readout).named("readout");
    // One row of the allowed-token mask per readout row. The SDK reshapes a
    // single-row `logits` to `[vocab]`, so the mask has to follow suit or the
    // `Select` inside `masked_argmax` sees mismatched ranks.
    let allowed = if rows == 1 {
        Channel::new([vocab], dtype::bool)
    } else {
        Channel::new([rows, vocab], dtype::bool)
    }
    .named("allowed");
    let target_out = Channel::new([rows], dtype::i32).named("target_tokens");

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
        target_out.put(masked_argmax(intrinsics::logits(), allowed.take()));
    });

    allowed.put(masks.concat());
    let pipeline = Pipeline::new();
    fwd.submit(&pipeline).context("verify constrained draft")?;
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
        return Err("max_tokens must be at least 1".into());
    }

    let vocab = model::output_vocab_size();
    let page_size = kv_page_size();
    let constraint = Matcher::new(&Grammar::from_json_schema(&input.schema)?);

    let mut committed = chat::system_user(
        "Generate only the requested JSON value, with no markdown or explanation.",
        &input.prompt,
    );
    committed.extend(chat::cue());
    if committed.is_empty() {
        committed.push(0);
    }

    let mut generated: Vec<u32> = Vec::with_capacity(input.max_tokens);
    let mut total_drafted = 0usize;
    let mut total_accepted = 0usize;
    let mut grammar_pruned = 0usize;
    let mut verification_steps = 0usize;
    let mut rollback_checks = 0usize;

    while generated.len() < input.max_tokens && !constraint.is_terminated() {
        // Never draft past the token budget: an accepted draft token still
        // has to fit.
        let room = input.max_tokens - generated.len() - 1;
        let want = input.draft_length.min(room);
        let proposal = draft_from_cache(&committed, want, input.max_ngram);

        // Walk the grammar forward over the proposal, collecting one mask per
        // readout row, and truncate at the first token the grammar forbids.
        let checkpoint = constraint.fork();
        let mut masks = vec![unpack_mask(&constraint.mask(), vocab)];
        let mut draft: Vec<u32> = Vec::with_capacity(proposal.len());
        for &token in &proposal {
            if !masks[draft.len()][token as usize] {
                grammar_pruned += 1;
                break;
            }
            constraint
                .accept_tokens(&[token])
                .context("grammar rejected a drafted token")?;
            draft.push(token);
            if constraint.is_terminated() {
                // Nothing follows a completed document; the bonus row would
                // be predicting past the end of the grammar.
                break;
            }
            masks.push(unpack_mask(&constraint.mask(), vocab));
        }
        // The bonus row exists only when the grammar is still open.
        draft.truncate(masks.len() - 1);

        // Wind the speculative walk all the way back, then prove it landed
        // exactly where it started before committing anything.
        let walked = draft.len() as u32;
        if walked > constraint.rollback_capacity() {
            return Err(format!(
                "draft of {walked} exceeds rollback capacity {}",
                constraint.rollback_capacity()
            ));
        }
        constraint.rollback(walked);
        if constraint.mask() != checkpoint.mask() {
            return Err(format!(
                "rollback({walked}) did not restore the grammar state; \
                 the constraint would now be silently wrong"
            ));
        }
        rollback_checks += 1;

        total_drafted += draft.len();
        verification_steps += 1;
        let target = verify(&committed, &draft, &masks, vocab, page_size).await?;
        if target.len() != draft.len() + 1 {
            return Err(format!(
                "verification returned {} tokens for a {}-token draft",
                target.len(),
                draft.len()
            ));
        }

        // Greedy acceptance: keep a draft token only where it matches the
        // target's own masked argmax; on the first mismatch take the target's
        // token, which is legal by construction, and stop.
        let mut accepted = Vec::new();
        let mut rejected = false;
        for (index, &drafted) in draft.iter().enumerate() {
            if target[index] == drafted {
                accepted.push(drafted);
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
            constraint
                .accept_tokens(&[token])
                .context("accepted token violates the grammar")?;
            committed.push(token);
            generated.push(token);
            if constraint.is_terminated() || generated.len() == input.max_tokens {
                break;
            }
        }
    }

    if !constraint.is_terminated() {
        return Err(format!(
            "JSON generation did not terminate within {} tokens",
            input.max_tokens
        ));
    }

    let text = model::decode(&generated)?;
    serde_json::from_str::<Value>(&text)
        .map_err(|e| format!("constraint terminated with invalid JSON: {e}; output={text:?}"))?;
    let acceptance_rate = if total_drafted == 0 {
        0.0
    } else {
        total_accepted as f64 / total_drafted as f64
    };
    Ok(Output {
        sampler: "constrained-speculative",
        text,
        count: generated.len(),
        draft_length: input.draft_length,
        verification_steps,
        drafted: total_drafted,
        accepted: total_accepted,
        grammar_pruned,
        acceptance_rate,
        rollback_checks,
        tokens: generated,
    })
}
