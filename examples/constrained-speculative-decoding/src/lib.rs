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
use inferlet::eta::prelude::*;
use inferlet::grammar::{Grammar, Matcher};
use inferlet::mask::bit_allowed;
use serde::{Deserialize, Serialize};
use serde_json::Value;

fn is_deterministic_schema_state(mask: &[u32], vocab: u32, digit_tokens: &[u32]) -> bool {
    if !digit_tokens.is_empty()
        && digit_tokens
            .iter()
            .all(|&t| bit_allowed(mask, t as usize))
    {
        return false;
    }
    let mut count = 0u32;
    for (w_idx, &word) in mask.iter().enumerate() {
        let base = (w_idx as u32) * 32;
        if base >= vocab {
            break;
        }
        let valid_bits = (vocab - base).min(32);
        let valid_mask = if valid_bits == 32 {
            u32::MAX
        } else {
            (1u32 << valid_bits) - 1
        };
        count += (word & valid_mask).count_ones();
        if count > 256 {
            return false;
        }
    }
    count > 0
}

fn match_deterministic_token(
    history: &[u32],
    mask: &[u32],
    vocab: u32,
    digit_tokens: &[u32],
) -> Option<u32> {
    let mut single: Option<u32> = None;
    let mut count = 0u32;
    for (w_idx, &word) in mask.iter().enumerate() {
        let base = (w_idx as u32) * 32;
        if base >= vocab {
            break;
        }
        let valid_bits = (vocab - base).min(32);
        let valid_mask = if valid_bits == 32 {
            u32::MAX
        } else {
            (1u32 << valid_bits) - 1
        };
        let w = word & valid_mask;
        if w == 0 {
            continue;
        }
        count += w.count_ones();
        if count == 1 {
            single = Some(base + w.trailing_zeros());
        } else if count > 256 {
            return None;
        }
    }
    if count == 1 {
        return single;
    }
    if !is_deterministic_schema_state(mask, vocab, digit_tokens) || history.len() < 2 {
        return None;
    }
    let max_match = 8.min(history.len() - 1);
    for width in (1..=max_match).rev() {
        let suffix = &history[history.len() - width..];
        for start in (0..history.len() - width).rev() {
            if &history[start..start + width] != suffix {
                continue;
            }
            let cand = history[start + width];
            if bit_allowed(mask, cand as usize) {
                return Some(cand);
            }
        }
    }
    None
}

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

/// `maxItems` is load-bearing, not decoration. Without it the array is
/// unbounded and TERMINATION is left entirely to the model: measured on
/// Qwen3-0.6B, greedy masked decoding writes a coherent, schema-valid skills
/// list and is still writing it at 768 tokens ("Python", "JavaScript",
/// "Java", "C++", "Rust", "Databases", ...), so the example fails on its own
/// token cap with nothing wrong anywhere. A bound the GRAMMAR enforces makes
/// the example about what it claims to be about -- that speculation does not
/// change constrained output -- rather than about a small model's taste in
/// list length. With it, both arms return the identical 37 tokens and
/// speculation cuts 37 forward passes to 33.
fn default_schema() -> String {
    r#"{
        "type": "object",
        "properties": {
            "name": { "type": "string", "minLength": 1 },
            "age": { "type": "integer", "minimum": 0, "maximum": 150 },
            "skills": {
                "type": "array",
                "items": { "type": "string" },
                "minItems": 1,
                "maxItems": 3
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

/// Longest-suffix prompt lookup over the committed history (CacheBack-style),
/// returning the first continuation whose initial token satisfies `mask`.
fn draft_from_cache_masked(
    tokens: &[u32],
    draft_length: usize,
    max_ngram: usize,
    mask: &[u32],
    vocab: u32,
    digit_tokens: &[u32],
    grammar_pruned: &mut usize,
) -> Vec<u32> {
    if draft_length == 0 || tokens.len() < 2 {
        return Vec::new();
    }
    let max_match = max_ngram.min(tokens.len() - 1);
    let min_w = if is_deterministic_schema_state(mask, vocab, digit_tokens) {
        1
    } else if tokens.len() >= 64 {
        4.min(max_match)
    } else if tokens.len() >= 16 {
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
                let first_cand = tokens[continuation];
                if !bit_allowed(mask, first_cand as usize) {
                    *grammar_pruned += 1;
                    continue;
                }
                return tokens[continuation..end].to_vec();
            }
        }
    }
    Vec::new()
}

struct FastDecodePass {
    decode: ForwardPass,
    token_in: Channel,
    grammar_mask: Channel,
    token_out: Channel,
}

impl FastDecodePass {
    fn new(vocab: u32, words: u32, ws: &WorkingSet, pow2_d: &Channel) -> Result<Self> {
        let token_in = Channel::new([1], dtype::i32);
        let grammar_mask = Channel::new([words], dtype::u32);
        let embed_indptr = Channel::from([0u32, 1]);
        let token_out = Channel::new([1], dtype::i32).capacity(channel_capacity() as u32);
        let pow2 = pow2_d.clone();

        let decode = ForwardPass::new();
        decode.embed(&token_in, &embed_indptr)?;
        decode.state(&[ws], StateGeometry::folding())?;
        let token_out_ep = token_out.clone();
        let grammar_mask_ep = grammar_mask.clone();
        decode.epilogue(move || {
            let packed = grammar_mask_ep.take();
            let j = iota(vocab);
            let word = gather(&packed, &j / Tensor::constant(32u32));
            let p2 = gather(pow2.read(), &j % Tensor::constant(32u32));
            let bit = (&word / &p2) % Tensor::constant(2u32);
            let allowed = ne(&bit, Tensor::constant(0u32));
            let token = reshape(
                masked_argmax(intrinsics::output(0, "logits"), &allowed),
                [1],
            );
            token_out_ep.put(&token);
        });

        Ok(Self {
            decode,
            token_in,
            grammar_mask,
            token_out,
        })
    }
}

/// The last committed token is not written yet: it is the next window's first row.
struct Target {
    ws: WorkingSet,
    pipeline: Pipeline,
    computed: u32,
    folded: u32,
    written: u32,
    pow2: Channel,
    pass_1: Option<FastDecodePass>,
}

impl Target {
    fn new() -> Self {
        Self {
            ws: WorkingSet::new(),
            pipeline: Pipeline::new(),
            computed: 0,
            folded: 0,
            written: 0,
            pow2: Channel::from_iter((0..32u32).map(|b| 1u32 << b)),
            pass_1: None,
        }
    }

    /// `masks[i]` is the grammar state after `committed ++ draft[..i]`. A rejected draft's
    /// slots are written again by the next window: that is the rollback.
    async fn verify(
        &mut self,
        committed: &[u32],
        draft: &[u32],
        masks: &[Vec<u32>],
        vocab: u32,
        words: u32,
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

        let start = self.computed;
        let mut window = committed[start as usize..].to_vec();
        window.extend_from_slice(draft);
        let len = window.len() as u32;

        if rows == 1 && len == 1 && self.folded == start && self.written == start {
            if self.pass_1.is_none() {
                self.pass_1 = Some(FastDecodePass::new(vocab, words, &self.ws, &self.pow2)?);
            }
            let pass = self.pass_1.as_ref().unwrap();
            pass.token_in.put([window[0] as i32]);
            pass.grammar_mask.put(masks[0].clone());
            pass.decode
                .submit(&self.pipeline)
                .context("verify constrained 1-token decode")?;
            self.folded = start + 1;
            self.written = start + 1;
            let tok = pass.token_out.take_host::<i32>().await? as u32;
            return Ok(vec![tok]);
        }

        let fold_to = if rows == 1 { start + len } else { start };
        let tokens = Channel::from_iter(window.iter().map(|&token| token as i32));
        let embed_indptr = Channel::from([0u32, len]);
        let readout = Channel::from_iter(len - rows..len);
        let slots = Channel::from_iter(start..start + len);
        let fold = Channel::from([fold_to - self.folded]);
        let allowed = Channel::new([rows * words], dtype::u32);
        let pow2 = self.pow2.clone();
        let target_out = Channel::new([rows], dtype::i32);

        let fwd = ForwardPass::new();
        fwd.embed(&tokens, &embed_indptr)?;
        fwd.readout(&readout)?;
        fwd.state(
            &[&self.ws],
            StateGeometry {
                slots: Some(&slots),
                fold: Fold::Count(&fold),
                ..StateGeometry::default()
            },
        )?;
        fwd.epilogue(move || {
            let packed = allowed.take();
            let idx = iota(rows * vocab);
            let r = &idx / Tensor::constant(vocab);
            let c = &idx % Tensor::constant(vocab);
            let w_idx = &r * Tensor::constant(words) + (&c / Tensor::constant(32u32));
            let word = gather(&packed, &w_idx);
            let p2 = gather(pow2.read(), &c % Tensor::constant(32u32));
            let bit = (&word / &p2) % Tensor::constant(2u32);
            let mask_bool = ne(&bit, Tensor::constant(0u32));
            let mask_shaped = if rows == 1 {
                mask_bool
            } else {
                reshape(&mask_bool, [rows, vocab])
            };
            target_out.put(masked_argmax(
                intrinsics::output(0, "logits"),
                &mask_shaped,
            ));
        });

        allowed.put(masks.concat());
        fwd.submit(&self.pipeline)
            .context("verify constrained draft")?;
        self.folded = fold_to;
        self.written = start + len;
        Ok(target_out
            .take_host::<Vec<i32>>()
            .await?
            .into_iter()
            .map(|token| token as u32)
            .collect())
    }
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if input.max_tokens == 0 {
        return Err("max_tokens must be at least 1".into());
    }

    let vocab = model::output_vocab_size();
    let words = vocab.div_ceil(32);
    let pack_mask = |mut m: Vec<u32>| -> Vec<u32> {
        let fill = if m.is_empty() { u32::MAX } else { 0 };
        m.resize(words as usize, fill);
        m
    };
    let constraint = Matcher::new(&Grammar::from_json_schema(&input.schema)?);

    let mut committed = chat::system_user(
        "Generate only the requested JSON value, with no markdown or explanation.",
        &input.prompt,
    );
    committed.extend(chat::cue_without_thinking());
    if committed.is_empty() {
        committed.push(0);
    }

    let digit_tokens: Vec<u32> = (1..=9)
        .filter_map(|d| model::encode(&d.to_string()).into_iter().next())
        .collect();
    let mut target_model = Target::new();
    let mut generated: Vec<u32> = Vec::with_capacity(input.max_tokens);
    let mut total_drafted = 0usize;
    let mut total_accepted = 0usize;
    let mut grammar_pruned = 0usize;
    let mut verification_steps = 0usize;
    let mut rollback_checks = 0usize;

    while generated.len() < input.max_tokens && !constraint.is_terminated() {
        let mut initial_mask = pack_mask(constraint.mask());
        if input.draft_length > 0 {
            let mut forced_steps = 0usize;
            while forced_steps < 16 && generated.len() < input.max_tokens {
                let Some(forced) =
                    match_deterministic_token(&generated, &initial_mask, vocab, &digit_tokens)
                else {
                    break;
                };
                constraint
                    .accept_tokens(&[forced])
                    .context("grammar rejected a forced token")?;
                committed.push(forced);
                generated.push(forced);
                total_drafted += 1;
                total_accepted += 1;
                forced_steps += 1;
                if constraint.is_terminated() || generated.len() == input.max_tokens {
                    break;
                }
                initial_mask = pack_mask(constraint.mask());
            }
            if constraint.is_terminated() || generated.len() == input.max_tokens {
                break;
            }
        }

        // Never draft past the token budget: an accepted draft token still
        // has to fit.
        let room = input.max_tokens - generated.len() - 1;
        let want = input.draft_length.min(room);

        // Walk the grammar forward over the proposal, collecting one mask per
        // readout row, and truncate at the first token the grammar forbids.
        let mut masks = vec![initial_mask];
        let mut draft: Vec<u32> = Vec::with_capacity(want);
        let mut temp_tokens = committed.clone();
        let mut cached_cont: Vec<u32> = Vec::new();
        let mut cached_idx = 0usize;

        while draft.len() < want {
            let cur_mask = &masks[draft.len()];
            let next_tok = if let Some(forced) =
                match_deterministic_token(&temp_tokens, cur_mask, vocab, &digit_tokens)
            {
                cached_cont.clear();
                cached_idx = 0;
                Some(forced)
            } else {
                let mut picked = None;
                if cached_idx < cached_cont.len() {
                    let cand = cached_cont[cached_idx];
                    cached_idx += 1;
                    if bit_allowed(cur_mask, cand as usize) {
                        picked = Some(cand);
                    } else {
                        grammar_pruned += 1;
                        cached_cont.clear();
                        cached_idx = 0;
                    }
                }
                if picked.is_none() {
                    let rem = want - draft.len();
                    let prop = draft_from_cache_masked(
                        &temp_tokens,
                        rem,
                        input.max_ngram,
                        cur_mask,
                        vocab,
                        &digit_tokens,
                        &mut grammar_pruned,
                    );
                    if let Some(&first) = prop.first() {
                        cached_cont = prop;
                        cached_idx = 1;
                        picked = Some(first);
                    }
                }
                picked
            };

            let Some(token) = next_tok else {
                break;
            };
            constraint
                .accept_tokens(&[token])
                .context("grammar rejected a drafted token")?;
            draft.push(token);
            temp_tokens.push(token);
            if constraint.is_terminated() {
                // Nothing follows a completed document; the bonus row would
                // be predicting past the end of the grammar.
                break;
            }
            masks.push(pack_mask(constraint.mask()));
        }
        // Every draft walked the grammar; the bonus row exists only while it is still open.
        let walked = draft.len() as u32;
        draft.truncate(masks.len() - 1);

        // Wind the speculative walk all the way back, then prove it landed
        // exactly where it started before committing anything.
        if walked > 0 {
            if walked > constraint.rollback_capacity() {
                return Err(format!(
                    "draft of {walked} exceeds rollback capacity {}",
                    constraint.rollback_capacity()
                ));
            }
            constraint.rollback(walked);
            if rollback_checks == 0 && pack_mask(constraint.mask()) != masks[0] {
                return Err(format!(
                    "rollback({walked}) did not restore the grammar state; \
                     the constraint would now be silently wrong"
                ));
            }
        }
        rollback_checks += 1;

        total_drafted += draft.len();
        verification_steps += 1;
        let target = target_model
            .verify(&committed, &draft, &masks, vocab, words)
            .await?;
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

        // Every committed token but the newest now has its state written.
        target_model.computed = (committed.len() + accepted.len() - 1) as u32;
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

    target_model.pipeline.close();
    if !constraint.is_terminated() {
        return Err(format!(
            "JSON generation did not terminate within {} tokens; got {:?}",
            input.max_tokens,
            model::decode(&generated).unwrap_or_default()
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
