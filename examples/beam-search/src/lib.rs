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

use inferlet::beam::BeamGroup;
use inferlet::chat;
use inferlet::eta::prelude::*;
use serde::Deserialize;

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
    #[serde(default)]
    ckpt_sync: bool,
    #[serde(default)]
    rollback_mode: Option<String>,
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

fn advance_hypotheses(
    hypotheses: &[Vec<u32>],
    picked: &[i32],
    parents: &[u32],
) -> Result<Vec<Vec<u32>>> {
    let picked: Vec<u32> = picked.iter().map(|&token| token as u32).collect();
    advance(hypotheses, &picked, parents)
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
    // The raw-logit argmax beside each pick, a cross-check run only under
    // `verify`: serving reads no argmax it does not use.
    let verify = input.verify;
    let greedy_p = verify.then(|| Channel::new([1], dtype::i32));
    // Under `verify`, each pick's own log-probability, target logit and
    // largest logit, as this fire computed them: the beam's record of its
    // candidates, which its scores must carry exactly.
    let record_p = verify.then(|| {
        (
            Channel::new([b], dtype::f32),
            Channel::new([b], dtype::f32),
            Channel::new([b], dtype::f32),
        )
    });
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
        if let Some(greedy_p) = greedy_p {
            greedy_p.put(reshape(reduce_argmax(&logits), [1]));
        }
        if let Some((own, picked, largest)) = record_p {
            own.put(cast(&s, dtype::f32));
            picked.put(cast(gather(&logits, &i), dtype::f32));
            largest.put(broadcast(
                reshape(cast(reduce_max(&logits), dtype::f32), [1]),
                [b],
            ));
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
    let mut greedy_mismatches = 0;
    if let Some(greedy_p) = greedy_p {
        let greedy = greedy_p
            .take_host::<Vec<i32>>()
            .await
            .context("prefill drain")?;
        greedy_mismatches += usize::from(first.first() != greedy.first());
    }
    let mut hypotheses: Vec<Vec<u32>> = first.iter().map(|&token| vec![token as u32]).collect();
    // Each lane's score after each of its steps, its ancestry's first.
    let mut scored: Vec<Vec<f32>> = final_scores.iter().map(|&s| vec![s]).collect();
    // Under `verify`, each lane's own candidates along its ancestry
    // (log-probability, target logit, largest logit), and the best score
    // each decode fire took its candidates relative to.
    let mut record: [Vec<Vec<f32>>; 3] = Default::default();
    let mut bests: Vec<f32> = vec![0.0];
    // Under `verify`, each decode fire's picks (tokens, parents), from which
    // the beam's own fires are replayed ([`replay_beam`]).
    let mut trail: Vec<(Vec<i32>, Vec<u32>)> = Vec::new();
    if let Some((own, picked, largest)) = record_p {
        for (slot, channel) in record.iter_mut().zip([own, picked, largest]) {
            let got = channel
                .take_host::<Vec<f32>>()
                .await
                .context("prefill drain")?;
            *slot = got.iter().map(|&x| vec![x]).collect();
        }
    }

    // Why the search stopped short of `max_tokens`, where it did: its beams'
    // pool outgrew the kv pool (refused by name; the columns before stand).
    let mut stopped: Option<String> = None;
    // The finish's check (`verify`): how far the adopted set's next-token
    // score lay from an unbranched decode's, and its room.
    let mut finished: Option<(f32, f32, f32)> = None;
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
        let out_greedy =
            verify.then(|| Channel::new([b], dtype::i32).capacity(channel_capacity() as u32));
        let out_record = verify.then(|| {
            let ring = || Channel::new([b], dtype::f32).capacity(channel_capacity() as u32);
            (ring(), ring(), ring())
        });
        let lanes_b = Channel::from_iter(0u32..=b);

        let corrupt_parent = input.corrupt.as_deref() == Some("parent");
        let fwd = ForwardPass::new();
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
            // Cross-check: at `beams == 1` the raw-logit argmax must equal the beam pick every step.
            if let Some(out_greedy) = out_greedy {
                out_greedy.put(reshape(reduce_argmax(&logits), [b]));
            }
            // Each pick's own log-probability, its target logit, and its
            // parent row's largest logit.
            if let Some((own, picked, largest)) = out_record {
                let flat = reshape(&logits, [b * v]);
                own.put(cast(gather(reshape(&lsm, [b * v]), &i), dtype::f32));
                picked.put(cast(gather(&flat, &i), dtype::f32));
                let top = cast(reduce_argmax(&logits), dtype::u32);
                let rows = gather(&flat, iota(b) * v + &top);
                largest.put(cast(gather(&rows, &parent), dtype::f32));
            }

            // `--corrupt parent`: each beam continues the next beam's
            // hypothesis, the control the verifier must catch.
            let parent = if corrupt_parent { (&parent + 1u32) % b } else { parent };
            group.reorder(&cast(&parent, dtype::u32));
        });

        let mut step = 0usize;
        let mode = input
            .rollback_mode
            .as_deref()
            .unwrap_or(if input.ckpt_sync {
                "ckpt_sync"
            } else {
                "flush_free"
            });
        let ckpt_sync = mode == "ckpt_sync";
        let stop_wait = mode == "stop_wait";
        if ckpt_sync || stop_wait {
            for _ in 0..decodes {
                // The step's checkpoint, forked before its fire.
                let ckpt_ws = ws
                    .fork(&pipeline)
                    .with_context(|| format!("@{step}: checkpoint the pool"))?;
                fwd.submit(&pipeline).with_context(|| format!("@{step}"))?;
                let picked = out
                    .take_host::<Vec<i32>>()
                    .await
                    .with_context(|| format!("@{step}"))?;
                let parents = out_par
                    .take_host::<Vec<u32>>()
                    .await
                    .with_context(|| format!("@{step}"))?;
                let carried_best = best_of(&final_scores);
                final_scores = out_scr
                    .take_host::<Vec<f32>>()
                    .await
                    .with_context(|| format!("@{step}"))?;
                let greedy = match out_greedy {
                    Some(out_greedy) => out_greedy
                        .take_host::<Vec<i32>>()
                        .await
                        .with_context(|| format!("@{step}"))?,
                    None => picked.clone(),
                };
                if stop_wait || parents.iter().enumerate().any(|(i, &p)| p as usize != i) {
                    let restore_ws = ckpt_ws
                        .fork(&pipeline)
                        .with_context(|| format!("@{step}: restore the checkpoint"))?;
                    let toks_r = Channel::from(&picked[..1]);
                    let ind_r = Channel::from([0u32, 1]);
                    let out_r = Channel::new([1], dtype::i32);
                    let fwd_r = ForwardPass::new();
                    fwd_r.embed(&toks_r, &ind_r)?;
                    fwd_r.state(&[&restore_ws], StateGeometry::folding())?;
                    fwd_r.epilogue(move || {
                        out_r.put(reshape(reduce_argmax(intrinsics::output(0, "logits")), [1]));
                    });
                    fwd_r.submit(&pipeline)?;
                    let _ = out_r.take_host::<Vec<i32>>().await?;
                }
                if verify {
                    trail.push((picked.clone(), parents.clone()));
                }
                if let Some((own, picked, largest)) = out_record {
                    for (slot, channel) in record.iter_mut().zip([own, picked, largest]) {
                        let got = channel
                            .take_host::<Vec<f32>>()
                            .await
                            .with_context(|| format!("@{step}"))?;
                        *slot = advance(slot, &got, &parents)?;
                    }
                    bests.push(carried_best);
                }
                greedy_mismatches += picked.iter().zip(&greedy).filter(|(p, g)| p != g).count();
                hypotheses = advance_hypotheses(&hypotheses, &picked, &parents)?;
                last_parents.clone_from(&parents);
                scored = advance(&scored, &final_scores, &parents)?;
                step += 1;
            }
        } else {
            let searched = run_ahead(&pipeline, &fwd, decodes, async || {
                let picked = match out.take_host::<Vec<i32>>().await {
                    Ok(picked) => picked,
                    // The beams' pool outgrew what the kv pool can hold for
                    // this process (a refusal by name: its growth could never
                    // fit, or the pool has nothing left beside what it holds):
                    // the columns searched so far are the output.
                    Err(e) if outgrown(&format!("{e}")) => {
                        stopped = Some(format!("@{step}: {e}"));
                        return Ok(ControlFlow::Break(()));
                    }
                    Err(e) => return Err(format!("@{step}: {e}").into()),
                };
                let parents = out_par
                    .take_host::<Vec<u32>>()
                    .await
                    .with_context(|| format!("@{step}"))?;
                let carried_best = best_of(&final_scores);
                final_scores = out_scr
                    .take_host::<Vec<f32>>()
                    .await
                    .with_context(|| format!("@{step}"))?;
                let greedy = match out_greedy {
                    Some(out_greedy) => out_greedy
                        .take_host::<Vec<i32>>()
                        .await
                        .with_context(|| format!("@{step}"))?,
                    None => picked.clone(),
                };
                if verify {
                    trail.push((picked.clone(), parents.clone()));
                }
                if let Some((own, picked, largest)) = out_record {
                    for (slot, channel) in record.iter_mut().zip([own, picked, largest]) {
                        let got = channel
                            .take_host::<Vec<f32>>()
                            .await
                            .with_context(|| format!("@{step}"))?;
                        *slot = advance(slot, &got, &parents)?;
                    }
                    bests.push(carried_best);
                }
                greedy_mismatches += picked.iter().zip(&greedy).filter(|(p, g)| p != g).count();
                hypotheses = advance_hypotheses(&hypotheses, &picked, &parents)?;
                last_parents.clone_from(&parents);
                scored = advance(&scored, &final_scores, &parents)?;
                step += 1;
                Ok(ControlFlow::Continue(()))
            })
            .await;
            if let Err(e) = searched {
                let why = format!("{e}");
                if outgrown(&why) {
                    stopped.get_or_insert_with(|| format!("@{step}: {why}"));
                } else {
                    return Err(e);
                }
            }
            // The search's finish (BEAM-PAGE-LISTS.md §6.6), checked under
            // `verify`: the set holds the best beam's hypothesis in its own
            // order, and a plain fire on it scores the next token as an
            // unbranched decode of the same tokens does.
            if verify && stopped.is_none() && step > 0 {
                let best = final_scores
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, x)| a.total_cmp(x))
                    .map_or(0, |(index, _)| index);
                let fired = u32::try_from(step).map_err(|_| "steps past u32")?;
                // `BeamGroup::finish`, its group moved into the epilogue: the
                // set's slots `[n, n + fired)`.
                // `--corrupt finish`: the set adopts another beam than the one
                // checked, the control the finish's check must catch.
                // The final beam adopted (its parent's pages, the last fire's
                // beam); under the control, the one whose hypothesis parts
                // from the checked one's earliest.
                let adopted = if input.corrupt.as_deref() == Some("finish") {
                    let shared = |lane: usize| {
                        hypotheses[lane]
                            .iter()
                            .zip(&hypotheses[best])
                            .take_while(|(x, y)| x == y)
                            .count()
                    };
                    let lane = (0..b as usize)
                        .filter(|&lane| lane != best)
                        .min_by_key(|&lane| shared(lane))
                        .unwrap_or(best);
                    // A beam that parts from the checked one only at its last
                    // token (which no fire fed, so no row holds) holds the
                    // same rows: no control.
                    if shared(lane) + 1 >= hypotheses[best].len() {
                        return Err("the finish's control: every beam holds the checked \
                                    hypothesis's rows (they part at the last token at most), \
                                    so adopting another proves nothing"
                            .into());
                    }
                    lane
                } else {
                    best
                };
                let device_beam = last_parents.get(adopted).copied().unwrap_or(0);
                fwd.adopt_beam(device_beam, n + fired)
                    .context("finish the search")?;
                let read = match input.corrupt.as_deref() {
                    Some("finish-other") => FinishRead::Other,
                    Some("finish-stale") => FinishRead::Committed(n),
                    _ => FinishRead::Set,
                };
                finished = Some(
                    finish_check(&ws, &prompt, &hypotheses[best], n + fired, b, read).await?,
                );
            }
        }
    }
    pipeline.close();

    let best_lane = final_scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .ok_or("beam search produced no hypotheses")?;
    // The beam identity is only an identity at width 1; at B > 1 the top-B
    // block legitimately picks non-argmax continuations for the lower lanes.
    if b == 1 && greedy_mismatches != 0 {
        return Err(format!(
            "beam identity violated: width-1 beam search disagreed with greedy \
                 argmax on {greedy_mismatches} of {max_steps} steps"
        ));
    }
    let text = model::decode(&hypotheses[best_lane])?;
    if input.corrupt.as_deref() == Some("score")
        && let Some(last) = scored.first_mut().and_then(|lane| lane.last_mut())
    {
        *last += 4.0;
    }
    let verified = if input.verify {
        let mut worst = 0f32;
        // The most of its room any step came to, by check.
        let (mut closest_own, mut closest) = (0f32, 0f32);
        // The widest composition spread any step's rescore measured, the
        // shape that measured it, and the widest the other shapes measured.
        let mut widest = 0f32;
        let mut widest_by = "none";
        let mut unshaped = 0f32;
        // The prompt's slots, then one slot per beam per decode fire: the
        // pool the beam-shaped replay lays the search's fires over.
        let pool_len = u32::try_from(trail.len())
            .ok()
            .and_then(|fires| fires.checked_mul(b))
            .and_then(|cells| cells.checked_add(n))
            .ok_or("beams * max_tokens exceeds the u32 range")?;
        let mut shaped = replay_beam(&prompt, &first, &trail, b, pool_len).await?;
        // `--corrupt uniform`: every lane's every candidate computed the same
        // wrong amount off, as a kernel fault every composition of the
        // search's shape reproduces would: its scores, its own record and the
        // beam-shaped replay all moved alike, so the bookkeeping holds and a
        // spread-scaled room would absorb it. The absolute room must not.
        if input.corrupt.as_deref() == Some("uniform") {
            const OFF: f32 = 0.75;
            for lane in 0..hypotheses.len() {
                for (at, step) in record[0][lane].iter_mut().enumerate() {
                    *step += OFF;
                    scored[lane][at] += OFF * (at + 1) as f32;
                    shaped[lane][at].0 += OFF;
                }
                final_scores[lane] += OFF * record[0][lane].len() as f32;
            }
        }
        // The hypothesis's summed rescore against its final score, within
        // the sum of its steps' rooms and roundings: the worst ratio.
        let mut worst_total = 0f32;
        for (lane, hypothesis) in hypotheses.iter().enumerate() {
            let alone = rescore(&prompt, hypothesis, b, Some(&shaped[lane])).await?;
            let own = &record[0][lane];
            let mut before = 0f32;
            let mut bound = 0f32;
            for (at, &after) in scored[lane].iter().enumerate() {
                // Bookkeeping, exact: the step a lane's score took is the
                // candidate its own fire computed, within that candidate's
                // own rounding (the score relative to the best, the bf16
                // candidate, the f32 scores either side). A score, token or
                // parent cell read from another lane misses it.
                let took = finite(
                    &format!("beam {lane}'s score step at step {at}"),
                    after - before,
                )?;
                let mine = finite(
                    &format!("beam {lane}'s own candidate at step {at}"),
                    own[at],
                )?;
                let rounding = if at == 0 {
                    ulp_f32(after)
                } else {
                    let relative = before - bests[at];
                    ulp_bf16(relative)
                        + ulp_bf16(relative + mine)
                        + ulp_f32(after)
                        + ulp_f32(before)
                };
                let gap = (took - mine).abs();
                closest_own = closest_own.max(gap / rounding);
                if gap > rounding {
                    return Err(format!(
                        "beam {lane} took {took:.4} at step {at}, and its own fire's candidate \
                         {mine:.4} there: {gap:.4} apart, past its rounding {rounding:.4}; a \
                         score, token or parent cell read from another lane",
                    ));
                }
                // Numerics: the beam's own candidate against its hypothesis
                // rescored alone as one row, within an absolute room of bf16
                // gaps at the step's logit scale ([`Rescored::room`]),
                // whatever the compositions' measured spread (reported, never
                // a room). A recurrence or attention that read another beam's
                // cells, or a kernel fault every composition reproduces, moves
                // the beam's candidate off the one-row rescore.
                let r = &alone[at];
                let room = r.room(mine);
                bound += room + rounding;
                if r.spread > widest {
                    widest = r.spread;
                    widest_by = r.by;
                }
                unshaped = unshaped.max(r.unshaped);
                let apart = (mine - r.step).abs();
                closest = closest.max(apart / room);
                if apart > room {
                    return Err(format!(
                        "beam {lane}'s fire took {mine:.4} at step {at}, and its hypothesis alone \
                         {:.4} there: {apart:.4} apart, past {room:.4} ({STEP_ULPS} bf16 gaps at \
                         its logits' scale; composition spread {:.4}, widest from the {}); its \
                         fire computed another value than the hypothesis's own",
                        r.step, r.spread, r.by,
                    ));
                }
                before = after;
            }
            // The whole hypothesis: its rescored sum against its final score,
            // within its steps' rooms and roundings summed (each step's error
            // bound, accumulated over its length).
            let total: f32 = alone.iter().map(|s| s.step).sum();
            let apart = (total - final_scores[lane]).abs();
            worst = worst.max(apart);
            worst_total = worst_total.max(apart / bound.max(f32::MIN_POSITIVE));
            if !(apart <= bound) {
                return Err(format!(
                    "beam {lane}'s final score {:.4} and its hypothesis rescored alone {total:.4}: \
                     {apart:.4} apart, past {bound:.4}, its {} steps' rooms summed",
                    final_scores[lane],
                    alone.len()
                ));
            }
        }
        format!(
            " greedy_mismatches={greedy_mismatches} verified={} worst_apart={worst:.4} worst_total={worst_total:.3} worst_own={closest_own:.3} worst_step={closest:.3} max_spread={widest:.4} spread_by={widest_by} unshaped_spread={unshaped:.4}",
            hypotheses.len()
        )
    } else {
        String::new()
    };
    Ok(format!(
        "{text}\n[beam] width={b} steps={max_steps} best_score={:.4}{verified}{}{}",
        final_scores[best_lane],
        finished.map_or(String::new(), |(apart, room, got)| format!(
            " finished_apart={apart:.6} finished_room={room:.3} finished_got={got:.6}"
        )),
        stopped.map_or(String::new(), |why| format!(" stopped {why}"))
    ))
}

/// After a search's finish, one plain fire on its working set: the last
/// token of `hypothesis` (the one the search fed no fire) at slot `slot`,
/// folding, its second-ranked next token's log-probability against an
/// unbranched decode of `prompt` and `hypothesis` and that token
/// ([`rescore`]: its spread over the compositions the width and an unfolded
/// run take, as the set carries the search's unfolded rows into this fire),
/// within a step's room ([`Rescored::room`]). How far apart, the room, and its score; past
/// the room it fails by name.
async fn finish_check(
    ws: &WorkingSet,
    prompt: &[u32],
    hypothesis: &[u32],
    slot: u32,
    width: u32,
    read: FinishRead,
) -> Result<(f32, f32, f32)> {
    let v = model::output_vocab_size();
    let last = *hypothesis.last().ok_or("an empty hypothesis")?;
    let pipeline = Pipeline::new();
    // `FinishRead::Other`: another sequence as deep, prefilled and folded on
    // a set of its own: the prompt's tokens shifted by one, and the
    // hypothesis's but its last.
    let other = WorkingSet::new();
    let ws = if let FinishRead::Other = read {
        let fed: Vec<i32> = prompt
            .iter()
            .map(|&t| (t + 1) as i32)
            .chain(hypothesis[..hypothesis.len() - 1].iter().map(|&t| (t + 1) as i32))
            .collect();
        let rows = u32::try_from(fed.len()).map_err(|_| "a sequence past u32 rows")?;
        let toks = Channel::from(&fed[..]);
        let indptr = Channel::from([0u32, rows]);
        let fwd = ForwardPass::new();
        fwd.embed(&toks, &indptr)?;
        fwd.state(&[&other], StateGeometry::folding())
            .context("bind the other sequence")?;
        fwd.submit(&pipeline).context("other-sequence submit")?;
        &other
    } else {
        ws
    };
    let toks = Channel::from([last as i32]);
    let indptr = Channel::from([0u32, 1]);
    let positions = Channel::from([slot]);
    // `FinishRead::Committed(c)`: the row lands at slot `c`, so it reads the
    // committed state alone and none of the buffered rows past it.
    let slots = Channel::from([match read {
        FinishRead::Committed(committed) => committed,
        _ => slot,
    }]);
    let out = Channel::new([1], dtype::f32);
    let next = Channel::new([1], dtype::i32);
    let fwd = ForwardPass::new();
    fwd.embed(&toks, &indptr)?;
    fwd.state(
        &[ws],
        StateGeometry {
            positions: Some(&positions),
            slots: Some(&slots),
            ..StateGeometry::folding()
        },
    )
    .context("bind the finished set")?;
    fwd.epilogue(move || {
        // The second-ranked token: its log-probability moves with the context
        // even where the first is near certain (log-probability near 0 under
        // any context, which no check could tell apart).
        let logits = cast(reshape(intrinsics::output(0, "logits"), [v]), dtype::bf16);
        let (s, i) = top_k(log_softmax(&logits), 2);
        out.put(cast(gather(&s, iota(1) + 1u32), dtype::f32));
        next.put(cast(gather(&i, iota(1) + 1u32), dtype::i32));
    });
    fwd.submit(&pipeline).context("finished-set submit")?;
    let got = out.take_host::<Vec<f32>>().await.context("finished-set drain")?;
    let token = next.take_host::<Vec<i32>>().await.context("finished-set drain")?;
    pipeline.close();
    let (got, token) = match (got.first(), token.first()) {
        (Some(&got), Some(&token)) => (got, token as u32),
        _ => return Err("the finished set's fire returned nothing".into()),
    };
    let mut forced = hypothesis.to_vec();
    forced.push(token);
    let rescored = rescore(prompt, &forced, width, None).await?;
    let want = rescored.last().ok_or("the unbranched decode returned nothing")?;
    // The set's fire rounds its logits and log-probability as the beam's
    // fires do, and takes no candidate: a step's room with nothing carried.
    let room = want.room(got);
    let apart = (got - want.step).abs();
    if apart > room {
        return Err(format!(
            "after its finish the set scored the next token {got:.4} and an unbranched decode \
             {:.4}: {apart:.4} apart, room {room:.4} ({STEP_ULPS} bf16 gaps at its logits' scale): \
             {:.2} times its room; its pages are not its \
             beam's",
            want.step,
            apart / room
        )
        .into());
    }
    Ok((apart, room, got))
}

/// A value the verifier compares, refused by name where it is not finite: a
/// NaN fails no `>` test, so it would pass every room unseen.
fn finite(what: &str, x: f32) -> Result<f32> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(format!("{what} came out {x}: not a finite number"))
    }
}

/// The gap between `x` and the next bf16 away from zero.
fn ulp_bf16(x: f32) -> f32 {
    let exponent = ((x.abs().to_bits() >> 23) & 0xff) as i32 - 127;
    2f32.powi(exponent.max(-126) - 7)
}

/// The gap between `x` and the next f32 away from zero.
fn ulp_f32(x: f32) -> f32 {
    let exponent = ((x.abs().to_bits() >> 23) & 0xff) as i32 - 127;
    2f32.powi(exponent.max(-126) - 23)
}

fn best_of(scores: &[f32]) -> f32 {
    scores.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// What the finish's fire reads: the finished set as it stands, or (the
/// finish's negative controls) another sequence's set, or only the set's
/// first `c` slots, the committed state.
#[derive(Clone, Copy)]
enum FinishRead {
    Set,
    Other,
    Committed(u32),
}

/// Whether a fire's refusal says the beams' pool outgrew what the kv pool
/// can hold for this process (by name): its growth could never fit, or the
/// pool has nothing left beside what it holds. The columns searched before
/// stand as the output.
fn outgrown(why: &str) -> bool {
    why.contains("outgrows the kv pool") || why.contains("this process alone holds what the pool holds")
}

/// One step of a hypothesis rescored alone: its log-probability, target and
/// largest logit as a 1-row fire, and how far its log-probability moved
/// across the compositions a beam's fire may run in: the widest, the shape
/// that measured it, and the widest the shapes other than the search's own
/// ([`replay_beam`]) measured.
struct Rescored {
    step: f32,
    picked: f32,
    largest: f32,
    spread: f32,
    by: &'static str,
    unshaped: f32,
}

/// The bf16 gaps one honest log-probability may sit from another of the
/// same token at the same context: each side rounds the target logit, the
/// log-sum-exp's leading (largest) logit and the log-probability once to
/// bf16, half a gap each (1.5 a side, 3 for the two), and one gap more for
/// the logits' own drift between compositions (no batch invariance).
const STEP_ULPS: f32 = 4.0;

impl Rescored {
    /// The absolute room a fire's log-probability `took` has from this
    /// rescore: [`STEP_ULPS`] bf16 gaps at the step's logit scale, the
    /// largest magnitude among its target logit, its largest logit and the
    /// two log-probabilities. It holds no measured composition spread, so a
    /// fault every composition reproduces alike widens nothing.
    fn room(&self, took: f32) -> f32 {
        let scale = self
            .picked
            .abs()
            .max(self.largest.abs())
            .max(self.step.abs())
            .max(took.abs());
        STEP_ULPS * ulp_bf16(scale)
    }
}

/// Each step of `hypothesis` after `prompt`, teacher-forced one token a fire
/// on working sets of its own ([`teacher_forced`]): as a 1-row fire, what a
/// beam that read only its own ancestry takes there; and how far that moves
/// as the beam's own fires run it, at the beam's width (the hypothesis and
/// copies of itself, nothing folded past the prompt, as a beam's rows run
/// unfolded), at width 1 unfolded ([`replay`]: one pool read through its
/// mask), and, where given, in the search's own fire shape (`shaped`, from
/// [`replay_beam`]: every lane of every fire at its width, depth and
/// parents, read through masks built on the host from the parents the search
/// reported, not through the page lists the device carried). Other rows and
/// another geometry
/// run other kernels and arms (other tiles, splits and reduction orders,
/// compounding through every layer and the cache rows written), which the
/// user allows (no batch invariance): the spread is the honest computation's
/// own, measured, which a read of another beam's cells cannot widen. It is
/// one sample of that variance: these fires share their frames with whatever
/// else the server runs then, as the beam's fires did with theirs, and the
/// two need not have met the same neighbours, so a composition the rescore
/// did not meet can still move the beam's candidate past it.
///
/// The beam-shaped spread is measured on the search's own fire shapes, so
/// it absorbs whatever those shapes compute deterministically: a kernel
/// fault that a W-wide fire reproduces on every run widens the room by
/// exactly what it moves, and passes as composition. The host-built mask
/// catches parent and bookkeeping faults, not those; a W-shape kernel fault
/// shows only against the width copies and the width-1 replay. The report
/// names the shape that set the widest spread (`spread_by`) and the widest
/// the other shapes measured (`unshaped_spread`), so a room held open by
/// the beam-shaped replay alone is visible.
async fn rescore(
    prompt: &[u32],
    hypothesis: &[u32],
    width: u32,
    shaped: Option<&[(f32, f32, f32)]>,
) -> Result<Vec<Rescored>> {
    let one = teacher_forced(prompt, hypothesis, 1, Fold::All).await?;
    let mut others = vec![
        teacher_forced(prompt, hypothesis, width, Fold::None).await?,
        replay(prompt, hypothesis).await?,
    ];
    if let Some(shaped) = shaped {
        if shaped.len() != one.len() {
            return Err(format!(
                "the beam-shaped replay holds {} steps of a {}-step hypothesis",
                shaped.len(),
                one.len()
            ));
        }
        others.push(shaped.to_vec());
    }
    const SHAPES: [&str; 3] = ["width-copies", "width-1-replay", "beam-shaped-replay"];
    let passes = std::iter::once(("1-row", &one)).chain(SHAPES.iter().copied().zip(&others));
    for (name, pass) in passes {
        for (at, &(step, picked, largest)) in pass.iter().enumerate() {
            for value in [step, picked, largest] {
                finite(&format!("the {name} rescore at step {at}"), value)?;
            }
        }
    }
    Ok(one
        .iter()
        .enumerate()
        .map(|(at, &(step, picked, largest))| {
            let moved: Vec<f32> = others
                .iter()
                .map(|other| (other[at].0 - step).abs())
                .collect();
            // The first shape to reach the widest, the others before the
            // search's own.
            let (by, spread) =
                SHAPES
                    .iter()
                    .zip(&moved)
                    .fold(("none", 0f32), |(by, widest), (&shape, &m)| {
                        if m > widest { (shape, m) } else { (by, widest) }
                    });
            Rescored {
                step,
                picked,
                largest,
                spread,
                by,
                unshaped: moved[0].max(moved[1]),
            }
        })
        .collect())
}

/// Each step of `hypothesis` after `prompt` as a beam of width 1 that took
/// exactly it: the prompt prefilled folding, then each token fired unfolded
/// at its own pool slot and depth, read through a mask of its own cells, as
/// the beam's epilogue lays its lanes; its log-probability, target logit and
/// largest logit at each step.
async fn replay(prompt: &[u32], hypothesis: &[u32]) -> Result<Vec<(f32, f32, f32)>> {
    let v = model::output_vocab_size();
    let n = u32::try_from(prompt.len()).map_err(|_| "a prompt past u32 rows")?;
    let cells =
        u32::try_from(hypothesis.len().max(1)).map_err(|_| "a hypothesis past u32 cells")?;
    let ws = WorkingSet::new();
    ws.reserve(n + cells).context("reserve the replay's pool")?;
    let pipeline = Pipeline::new();
    let mut steps = Vec::with_capacity(hypothesis.len());
    for (at, &target) in hypothesis.iter().enumerate() {
        let fwd = ForwardPass::new();
        let wanted = Channel::from([target]);
        let out = Channel::new([1], dtype::f32);
        let picked = Channel::new([1], dtype::f32);
        let largest = Channel::new([1], dtype::f32);
        if at == 0 {
            let fed: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
            fwd.embed(&Channel::from(&fed[..]), &Channel::from([0u32, n]))?;
            fwd.state(&[&ws], StateGeometry::folding())
                .context("bind the replay's prefill")?;
        } else {
            let depth = n + at as u32 - 1;
            fwd.embed(
                &Channel::from([hypothesis[at - 1] as i32]),
                &Channel::from([0u32, 1]),
            )?;
            let own: Vec<bool> = (0..cells).map(|c| c < at as u32).collect();
            fwd.mask(&Channel::from_shaped([1, cells], own))?;
            let slots = Channel::from([depth]);
            let pos = Channel::from([depth]);
            fwd.state(
                &[&ws],
                StateGeometry {
                    positions: Some(&pos),
                    slots: Some(&slots),
                    kv_len: None,
                    fold: Fold::None,
                    ..StateGeometry::default()
                },
            )
            .context("bind the replay's step")?;
        }
        fwd.epilogue(move || {
            let logits = cast(reshape(intrinsics::output(0, "logits"), [v]), dtype::bf16);
            let wanted = wanted.take();
            picked.put(cast(gather(&logits, &wanted), dtype::f32));
            largest.put(cast(reshape(reduce_max(&logits), [1]), dtype::f32));
            out.put(cast(gather(log_softmax(&logits), &wanted), dtype::f32));
        });
        fwd.submit(&pipeline).context("replay submit")?;
        let first = |of: Vec<f32>| {
            of.first()
                .copied()
                .ok_or("the replay fire returned nothing")
        };
        let step = first(out.take_host::<Vec<f32>>().await.context("replay drain")?)?;
        let picked = first(
            picked
                .take_host::<Vec<f32>>()
                .await
                .context("replay drain")?,
        )?;
        let largest = first(
            largest
                .take_host::<Vec<f32>>()
                .await
                .context("replay drain")?,
        )?;
        steps.push((step, picked, largest));
    }
    pipeline.close();
    Ok(steps)
}

/// The search's own fires again, teacher-forced from the picks it reported:
/// the prompt prefilled folding on a pool of the search's size, then each
/// decode fire in the search's composition (every lane at its width and
/// depth, at slot `n + fire * width + lane` of one pool), each lane read
/// through a mask the host builds from the reported parents (its parent's
/// cells and its own), not through the page lists the device carried. Each
/// pick's log-probability, target logit and parent row's largest logit,
/// followed along each final lane's ancestry as the search's record is. A
/// fire whose rows moved only with its composition (its width, row count and
/// kernels) moves here alike; a lane that read cells outside its ancestry
/// does not, as this mask holds none of them.
async fn replay_beam(
    prompt: &[u32],
    first: &[i32],
    trail: &[(Vec<i32>, Vec<u32>)],
    b: u32,
    pool_len: u32,
) -> Result<Vec<Vec<(f32, f32, f32)>>> {
    let v = model::output_vocab_size();
    let n = u32::try_from(prompt.len()).map_err(|_| "a prompt past u32 rows")?;
    let lanes = b as usize;
    let cells = pool_len - n;
    let ws = WorkingSet::new();
    ws.reserve(pool_len)
        .context("reserve the beam replay's pool")?;
    let pipeline = Pipeline::new();
    let drain = async |c: Channel| -> Result<Vec<f32>> {
        let got = c
            .take_host::<Vec<f32>>()
            .await
            .context("beam replay drain")?;
        if got.len() == lanes {
            Ok(got)
        } else {
            Err(format!(
                "the beam replay returned {} of {lanes} rows",
                got.len()
            ))
        }
    };

    let fed: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let wanted = Channel::from(first);
    let (own, picked, largest) = (
        Channel::new([b], dtype::f32),
        Channel::new([b], dtype::f32),
        Channel::new([b], dtype::f32),
    );
    let fwd = ForwardPass::new();
    fwd.embed(&Channel::from(&fed[..]), &Channel::from([0u32, n]))?;
    fwd.state(&[&ws], StateGeometry::folding())
        .context("bind the beam replay's prefill")?;
    fwd.epilogue(move || {
        let logits = cast(reshape(intrinsics::output(0, "logits"), [v]), dtype::bf16);
        let wanted = wanted.take();
        own.put(cast(gather(log_softmax(&logits), &wanted), dtype::f32));
        picked.put(cast(gather(&logits, &wanted), dtype::f32));
        largest.put(broadcast(
            reshape(cast(reduce_max(&logits), dtype::f32), [1]),
            [b],
        ));
    });
    fwd.submit(&pipeline).context("beam replay submit")?;
    let (o, p, l) = (
        drain(own).await?,
        drain(picked).await?,
        drain(largest).await?,
    );
    let mut rows: Vec<Vec<(f32, f32, f32)>> =
        (0..lanes).map(|j| vec![(o[j], p[j], l[j])]).collect();

    let mut masks: Vec<Vec<bool>> = (0..b)
        .map(|lane| (0..cells).map(|c| c == lane).collect())
        .collect();
    let mut toks: Vec<i32> = first.to_vec();
    for (fire, (tokens, parents)) in trail.iter().enumerate() {
        if tokens.len() != lanes || parents.len() != lanes {
            return Err(format!(
                "fire {fire} reported {} picks of width {b}",
                tokens.len()
            ));
        }
        if let Some(&bad) = parents.iter().find(|&&p| p >= b) {
            return Err(format!("fire {fire} reported parent {bad} at width {b}"));
        }
        let fire_u = u32::try_from(fire).map_err(|_| "a search past u32 fires")?;
        let at: Vec<u32> = parents
            .iter()
            .zip(tokens)
            .map(|(&p, &t)| p * v + t as u32)
            .collect();
        let flat_mask: Vec<bool> = masks.iter().flatten().copied().collect();
        let mask = Channel::from_shaped([b, cells], flat_mask);
        let slots = Channel::from_iter((0..b).map(|lane| n + fire_u * b + lane));
        let pos = Channel::from(vec![n + fire_u; lanes]);
        let wanted = Channel::from(at);
        let parent = Channel::from(parents.clone());
        let (own, picked, largest) = (
            Channel::new([b], dtype::f32),
            Channel::new([b], dtype::f32),
            Channel::new([b], dtype::f32),
        );
        let fwd = ForwardPass::new();
        fwd.embed(&Channel::from(&toks[..]), &Channel::from_iter(0u32..=b))?;
        fwd.mask(&mask)?;
        fwd.state(
            &vec![&ws; lanes],
            StateGeometry {
                positions: Some(&pos),
                slots: Some(&slots),
                kv_len: None,
                fold: Fold::None,
                ..StateGeometry::default()
            },
        )
        .context("bind the beam replay's fire")?;
        fwd.epilogue(move || {
            let logits = cast(
                reshape(intrinsics::output(0, "logits"), [b, v]),
                dtype::bf16,
            );
            let lsm = log_softmax(&logits);
            let flat = reshape(&logits, [b * v]);
            let wanted = wanted.take();
            own.put(cast(gather(reshape(&lsm, [b * v]), &wanted), dtype::f32));
            picked.put(cast(gather(&flat, &wanted), dtype::f32));
            let top = cast(reduce_argmax(&logits), dtype::u32);
            let tops = gather(&flat, iota(b) * v + &top);
            largest.put(cast(gather(&tops, parent.take()), dtype::f32));
        });
        fwd.submit(&pipeline).context("beam replay submit")?;
        let (o, p, l) = (
            drain(own).await?,
            drain(picked).await?,
            drain(largest).await?,
        );
        let step: Vec<(f32, f32, f32)> = (0..lanes).map(|j| (o[j], p[j], l[j])).collect();
        rows = advance(&rows, &step, parents)?;
        // The next fire's lanes: each its parent's cells and its own new one.
        let next = (fire_u + 1) * b;
        masks = parents
            .iter()
            .enumerate()
            .map(|(j, &p)| {
                let mut m = masks[p as usize].clone();
                if let Some(cell) = m.get_mut((next + j as u32) as usize) {
                    *cell = true;
                }
                m
            })
            .collect();
        toks = tokens.clone();
    }
    pipeline.close();
    Ok(rows)
}

/// Each step's log-probability of `hypothesis` after `prompt`, its target
/// logit and its largest logit, teacher-forced one token a fire as `copies`
/// rows (the hypothesis repeated, each copy on a working set of its own,
/// the prompt's fire the first copy's alone), read off the first. The prompt is folded; each step after it folds by
/// `fold`.
async fn teacher_forced(
    prompt: &[u32],
    hypothesis: &[u32],
    copies: u32,
    fold: Fold<'static>,
) -> Result<Vec<(f32, f32, f32)>> {
    let v = model::output_vocab_size();
    // The prompt is prefilled once, on the first set; the other copies are
    // forks of it after that fire, sharing the prompt's pages, so the
    // rescore needs one prompt and each copy's own tail.
    let mut sets: Vec<WorkingSet> = vec![WorkingSet::new()];
    let pipeline = Pipeline::new();
    let mut steps = Vec::with_capacity(hypothesis.len());
    for (at, &target) in hypothesis.iter().enumerate() {
        if at == 1 {
            for _ in 1..copies {
                let copy = sets[0].fork(&pipeline).context("fork the prompt's set")?;
                sets.push(copy);
            }
        }
        let bound: Vec<&WorkingSet> = sets.iter().collect();
        let copies = u32::try_from(bound.len()).map_err(|_| "copies past u32")?;
        let fed: Vec<i32> = if at == 0 {
            prompt.iter().map(|&t| t as i32).collect()
        } else {
            vec![hypothesis[at - 1] as i32]
        };
        let rows = u32::try_from(fed.len()).map_err(|_| "a prompt past u32 rows")?;
        let all: Vec<i32> = (0..copies).flat_map(|_| fed.iter().copied()).collect();
        let toks = Channel::from(&all[..]);
        let indptr = Channel::from_iter((0..=copies).map(|c| c * rows));
        let wanted = Channel::from([target]);
        let out = Channel::new([1], dtype::f32);
        let picked = Channel::new([1], dtype::f32);
        let largest = Channel::new([1], dtype::f32);
        let fwd = ForwardPass::new();
        fwd.embed(&toks, &indptr)?;
        let geometry = StateGeometry {
            fold: if at == 0 { Fold::All } else { fold },
            ..StateGeometry::default()
        };
        fwd.state(&bound, geometry)
            .context("bind the rescoring sets")?;
        fwd.epilogue(move || {
            // The first copy's row, at the sampler's bf16 grain.
            let logits = cast(
                reshape(
                    gather(
                        reshape(intrinsics::output(0, "logits"), [copies, v]),
                        iota(1),
                    ),
                    [v],
                ),
                dtype::bf16,
            );
            let wanted = wanted.take();
            picked.put(cast(&gather(&logits, &wanted), dtype::f32));
            largest.put(cast(&reshape(reduce_max(&logits), [1]), dtype::f32));
            out.put(cast(gather(log_softmax(&logits), &wanted), dtype::f32));
        });
        fwd.submit(&pipeline).context("rescore submit")?;
        let got = out.take_host::<Vec<f32>>().await.context("rescore drain")?;
        let step = got
            .first()
            .copied()
            .ok_or("the rescoring fire returned nothing")?;
        let picked = picked
            .take_host::<Vec<f32>>()
            .await
            .context("rescore drain")?;
        let largest = largest
            .take_host::<Vec<f32>>()
            .await
            .context("rescore drain")?;
        let first = |of: &[f32]| {
            of.first()
                .copied()
                .ok_or("the rescoring fire returned nothing")
        };
        steps.push((step, first(&picked)?, first(&largest)?));
    }
    pipeline.close();
    Ok(steps)
}
