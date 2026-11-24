//! On-device beam search over per-beam page lists.
//!
//! The prompt — chat-templated, or `prompt_tokens` as given — is every beam's
//! common ancestry. It is prefilled ONCE into a working set and folded, and
//! that fire's epilogue picks the first `beams` tokens (the top of the
//! prompt's next-token distribution), as greedy decoding picks its first.
//!
//! The beams then share that working set's frozen prompt, and each keeps its
//! own list of the pages its generated slots lie in (`inferlet::beam`): a
//! beam's rows read the prompt and its own pages only, a read extent of its
//! own length, with no mask. Every decode fire's epilogue picks the next
//! `beams` (parent, token) pairs and reorders the lists by the parents
//! (`BeamGroup::reorder`): a child shares its parent's full pages; the
//! parent's first child keeps its tail page, every other child has the tail
//! copied into a free page before the next fire appends. Pages no beam reads
//! any more are freed a fire late.
//!
//! The search runs `max_tokens` steps: a beam is not retired at end of
//! sequence, so width 1 is greedy decoding of the same prompt to the same
//! length (the beam identity, checked under `verify` every step against the
//! raw argmax).
//!
//! A fire refused after submit (a never-fits refusal, one through a restore
//! wait) poisons its pass's host-reader channels: every take, read and host
//! call here propagates its Err as the process's error, never a trap.

mod verify;

use inferlet::beam::BeamGroup;
use inferlet::chat;
use inferlet::eta::prelude::*;
use serde::Deserialize;
use verify::{Record, Taps};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    /// The prompt as token ids, in place of the chat-templated `prompt`.
    #[serde(default)]
    prompt_tokens: Option<Vec<u32>>,
    #[serde(default = "default_system")]
    system: String,
    /// Turn a thinking model's thinking off, as the model's own template does
    /// for `enable_thinking: false`.
    #[serde(default)]
    no_think: bool,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    /// Beam width. `1` degenerates to greedy decoding (the beam identity).
    #[serde(default = "default_beams")]
    beams: u32,
    /// Rescore each final hypothesis as a lone sequence, one token a fire on
    /// a working set of its own, and require its summed log-probability to
    /// equal its beam's score: each beam read its own ancestry and nothing of
    /// its siblings'; and read each step's raw-logit argmax beside its picks,
    /// which width 1 must equal (the beam identity).
    #[serde(default)]
    verify: bool,
    /// A negative control of `verify`, a beam broken on purpose: `parent`
    /// has each lane's mask inherit the next lane's ancestry (`(parent + 1)
    /// % beams`) while its hypothesis keeps its own, so its rows read
    /// another beam's cells; `score` moves lane 0's last step by 4.0 on the
    /// host; `uniform` moves every lane's every candidate by the same 0.75,
    /// in its scores, its record and the beam-shaped replay alike (a fault
    /// every composition reproduces). Each must fail `verify`. The finish's check has its own:
    /// `finish` adopts another beam than the one checked; `finish-other` has
    /// the finish fire read another sequence's set (the prompt's tokens
    /// shifted, decoded as deep); `finish-stale` has it read only the set's
    /// committed state, as a stale table or a short folded length would,
    /// none of the search's buffered rows.
    #[serde(default)]
    corrupt: Option<String>,
}

fn default_prompt() -> String {
    "Write a short paragraph about beam search.".into()
}

fn default_system() -> String {
    "You are a helpful, respectful and honest assistant.".into()
}

fn default_max_tokens() -> usize {
    16
}

fn default_beams() -> u32 {
    2
}

/// Each lane's row: its parent's, `picked` (its own) appended.
fn advance<T: Copy>(rows: &[Vec<T>], picked: &[T], parents: &[u32]) -> Result<Vec<Vec<T>>> {
    parents
        .iter()
        .zip(picked)
        .map(|(&parent, &value)| {
            let mut row = rows
                .get(parent as usize)
                .ok_or_else(|| format!("invalid parent beam {parent}"))?
                .clone();
            row.push(value);
            Ok(row)
        })
        .collect()
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let max_steps = input.max_tokens;
    let b = input.beams;
    if b == 0 {
        return Err("beams must be at least 1".into());
    }
    if max_steps == 0 {
        return Ok(String::new());
    }
    let v = model::output_vocab_size();
    if b > v {
        return Err(format!("beams ({b}) exceeds the vocabulary ({v})"));
    }

    let prompt: Vec<u32> = match &input.prompt_tokens {
        Some(tokens) => tokens.clone(),
        None => {
            let mut p = chat::system_user(&input.system, &input.prompt);
            p.extend(chat::cue_thinking(!input.no_think));
            p
        }
    };
    if prompt.is_empty() {
        return Err("empty prompt".into());
    }
    let n = u32::try_from(prompt.len()).map_err(|_| "the prompt exceeds the u32 range")?;
    let decodes = max_steps - 1;
    let ws = WorkingSet::new();

    let pipeline = Pipeline::new();

    // Step 0: the prompt's own next-token distribution. Every beam is the
    // prompt, so the first `b` tokens are its top `b`.
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let toks_p = Channel::from(&prompt_i32[..]);
    let embed_indptr_p = Channel::from([0u32, n]);
    let out_p = Channel::new([b], dtype::i32);
    let scr_p = Channel::new([b], dtype::f32);
    // `verify` reads what each fire computed beside its picks; serving reads
    // nothing it does not use.
    let verify = input.verify;
    let taps_p = verify.then(|| Taps::prefill(b));
    let fwd_p = ForwardPass::new();
    fwd_p.embed(&toks_p, &embed_indptr_p)?;
    fwd_p
        .state(&[&ws], StateGeometry::folding())
        .context("bind the prompt prefill")?;
    fwd_p.epilogue(move || {
        // The vocabulary-sized rows are bf16; the scores the host and the
        // next fire read are f32.
        let logits = cast(reshape(intrinsics::output(0, "logits"), [v]), dtype::bf16);
        let (s, i) = top_k(log_softmax(&logits), b);
        out_p.put(cast(&i, dtype::i32));
        scr_p.put(cast(&s, dtype::f32));
        if let Some(taps) = taps_p {
            taps.put_prefill(&logits, &s, &i, b);
        }
    });
    fwd_p.submit(&pipeline).context("prefill submit")?;
    let first = out_p
        .take_host::<Vec<i32>>()
        .await
        .context("prefill drain")?;
    let mut final_scores = scr_p
        .take_host::<Vec<f32>>()
        .await
        .context("prefill drain")?;
    let mut record = match taps_p {
        Some(taps) => Some(Record::prefill(taps, &first, &final_scores).await?),
        None => None,
    };
    let mut hypotheses: Vec<Vec<u32>> = first.iter().map(|&token| vec![token as u32]).collect();

    // Why the search stopped short of `max_tokens`, where it did: its beams'
    // pool outgrew the kv pool (refused by name; the columns before stand).
    let mut stopped: Option<String> = None;
    // The finish's check (`verify`), as reported.
    let mut finished = String::new();
    // The parents the last fire's epilogue picked: the device's lists are
    // the last fire's beams', so the final beam `l` lies on beam
    // `last_parents[l]`'s pages (its last token on none).
    let mut last_parents: Vec<u32> = (0..b).collect();
    if decodes > 0 {
        let scores = Channel::from(&final_scores[..]);
        let toks = Channel::from(&first[..]);
        let out = Channel::new([b], dtype::i32).capacity(channel_capacity() as u32);
        let out_par = Channel::new([b], dtype::u32).capacity(channel_capacity() as u32);
        let out_scr = Channel::new([b], dtype::f32).capacity(channel_capacity() as u32);
        let taps = verify.then(|| Taps::decode(b));
        let lanes_b = Channel::from_iter(0u32..=b);

        let corrupt_parent = input.corrupt.as_deref() == Some("parent");
        let fwd = ForwardPass::new();
        // Beams bind `parents` and adopt one beam's rows: not final
        // (CONTRACT-BC 25; a final pass binding parents is refused).
        fwd.retractable()?;
        fwd.embed(&toks, &lanes_b)?;
        let pool: Vec<&WorkingSet> = vec![&ws; b as usize];
        // Every beam reads the folded prompt and its own pages: beam `i`'s
        // first decode writes slot `n` on its own first page.
        let decodes_u32 = u32::try_from(decodes).map_err(|_| "max_tokens exceeds u32")?;
        let group = BeamGroup::new(&ws, b, n, decodes_u32).context("reserve the beams' pages")?;

        fwd.state(&pool, group.geometry())
            .context("bind the beams' page lists")?;
        fwd.epilogue(move || {
            // `logits()` squeezes a one-row fire to `[v]`: reshape so `beams == 1` meets `[B, 1]` scores.
            let logits = cast(
                reshape(intrinsics::output(0, "logits"), [b, v]),
                dtype::bf16,
            );
            // The candidates are bf16, each beam's score taken relative to
            // the best (whose rows then round no score at all): a bf16
            // absolute score would round a long hypothesis's sum by more
            // than a step's log-probability.
            let carried = scores.take();
            let best = reduce_max(&carried);
            let relative = cast(&carried - &best, dtype::bf16);
            let lsm = log_softmax(&logits);
            let cand = broadcast(reshape(relative, [b, 1]), [b, v]) + &lsm;
            let (s, i) = top_k(reshape(cand, [b * v]), b);
            let s = cast(&s, dtype::f32) + &best;
            let parent = &i / v;
            let tok_i = cast(&i % v, dtype::i32);

            scores.put(&s);
            toks.put(&tok_i);
            out.put(&tok_i);
            out_par.put(&parent);
            out_scr.put(&s);
            if let Some(taps) = taps {
                taps.put_decode(&logits, &lsm, &i, &parent, b, v);
            }

            // `--corrupt parent`: each beam continues the next beam's
            // hypothesis, the control the verifier must catch.
            let parent = if corrupt_parent {
                (&parent + 1u32) % b
            } else {
                parent
            };
            group.reorder(&cast(&parent, dtype::u32));
        });

        let mut step = 0usize;
        let searched = run_ahead(&pipeline, &fwd, decodes, async || {
            let picked = match out.take_host::<Vec<i32>>().await {
                Ok(picked) => picked,
                // The beams' pool outgrew what the kv pool can hold for
                // this process (a refusal by name: its growth could never
                // fit, or the pool has nothing left beside what it holds):
                // the columns searched so far are the output.
                Err(e) if outgrown(&e) => {
                    stopped = Some(format!("@{step}: {e}"));
                    return Ok(ControlFlow::Break(()));
                }
                Err(e) => return Err(format!("@{step}: {e}")),
            };
            let parents = out_par
                .take_host::<Vec<u32>>()
                .await
                .with_context(|| format!("@{step}"))?;
            final_scores = out_scr
                .take_host::<Vec<f32>>()
                .await
                .with_context(|| format!("@{step}"))?;
            if let (Some(record), Some(taps)) = (&mut record, taps) {
                record
                    .step(taps, &picked, &parents, &final_scores)
                    .await
                    .with_context(|| format!("@{step}"))?;
            }
            let picked: Vec<u32> = picked.iter().map(|&token| token as u32).collect();
            hypotheses = advance(&hypotheses, &picked, &parents)?;
            last_parents.clone_from(&parents);
            step += 1;
            Ok(ControlFlow::Continue(()))
        })
        .await;
        if let Err(e) = searched {
            if outgrown(&e) {
                stopped.get_or_insert_with(|| format!("@{step}: {e}"));
            } else {
                return Err(e);
            }
        }
        if verify && stopped.is_none() && step > 0 {
            finished = verify::finish(
                &fwd,
                &ws,
                &prompt,
                &hypotheses,
                &final_scores,
                &last_parents,
                n,
                u32::try_from(step).map_err(|_| "steps past u32")?,
                input.corrupt.as_deref(),
            )
            .await?;
        }
    }
    pipeline.close();

    let best_lane = verify::best(&final_scores);
    let text = model::decode(&hypotheses[best_lane])?;
    let verified = match record {
        Some(record) => {
            verify::check(
                record,
                &prompt,
                &first,
                &hypotheses,
                &mut final_scores,
                b,
                max_steps,
                input.corrupt.as_deref(),
            )
            .await?
        }
        None => String::new(),
    };
    Ok(format!(
        "{text}\n[beam] width={b} steps={max_steps} best_score={:.4}{verified}{finished}{}",
        final_scores[best_lane],
        stopped.map_or(String::new(), |why| format!(" stopped {why}"))
    ))
}

/// Whether a fire's refusal says the beams' pool outgrew what the kv pool
/// can hold for this process (by name): its growth could never fit, or the
/// pool has nothing left beside what it holds. The columns searched before
/// stand as the output.
fn outgrown(why: &str) -> bool {
    why.contains("outgrows the kv pool")
        || why.contains("this process alone holds what the pool holds")
}
