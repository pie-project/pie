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
    /// host. Either must fail `verify`. The finish's check has its own:
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
    // Each step's best score, the one the next fire's candidates are taken
    // relative to.
    let mut bests: Vec<f32> = vec![best_of(&final_scores)];

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
            let cand = broadcast(reshape(relative, [b, 1]), [b, v]) + log_softmax(&logits);
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
                greedy_mismatches += picked.iter().zip(&greedy).filter(|(p, g)| p != g).count();
                hypotheses = advance_hypotheses(&hypotheses, &picked, &parents)?;
                last_parents.clone_from(&parents);
                scored = advance(&scored, &final_scores, &parents)?;
                bests.push(best_of(&final_scores));
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
                greedy_mismatches += picked.iter().zip(&greedy).filter(|(p, g)| p != g).count();
                hypotheses = advance_hypotheses(&hypotheses, &picked, &parents)?;
                last_parents.clone_from(&parents);
                scored = advance(&scored, &final_scores, &parents)?;
                bests.push(best_of(&final_scores));
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
        // The step that came closest to its room (apart / room), and what
        // it was.
        let mut closest = 0f32;
        let mut closest_at = String::new();
        // The most any step's log-probability moved between its 1-row and
        // its beam-width rescore (d_t).
        let mut drift = 0f32;
        for (lane, hypothesis) in hypotheses.iter().enumerate() {
            let alone = rescore(&prompt, hypothesis, b).await?;
            // Each step's log-probability as the beam took it (its score's
            // step), against the hypothesis's own, within its room.
            drift = drift.max(alone.iter().map(|s| s.drift).fold(0.0, f32::max));
            let mut before = 0f32;
            for (at, (&after, rescored)) in scored[lane].iter().zip(&alone).enumerate() {
                let took = after - before;
                let step = rescored.step;
                let room = rescored.room(took, (at > 0).then(|| (before, after, bests[at - 1])));
                let apart = (took - step).abs();
                if apart / room.total() > closest {
                    closest = apart / room.total();
                    closest_at = format!(
                        "beam {lane} took {took:.4} at step {at}, and its hypothesis alone \
                         {step:.4} there: {apart:.4} apart, room {room}"
                    );
                }
                before = after;
            }
            let total: f32 = alone.iter().map(|s| s.step).sum();
            worst = worst.max((total - final_scores[lane]).abs());
        }
        if closest > 1.0 {
            return Err(format!(
                "{closest_at}: {closest:.2} times its room; its recurrence or attention read \
                 another beam's cells",
            ));
        }
        format!(
            " greedy_mismatches={greedy_mismatches} verified={} worst_apart={worst:.4} worst_step={closest:.3} max_drift={drift:.4} closest: {closest_at}",
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
/// ([`rescore`]: `width` rows run as the beam's fires run for the drift, as
/// the set carries the search's unfolded rows into this fire), within a
/// step's room ([`Rescored::room`]). How far apart, the room, and its score; past
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
    let rescored = rescore(prompt, &forced, width).await?;
    let want = rescored.last().ok_or("the unbranched decode returned nothing")?;
    // The set's fire rounds its logits and log-probability as the beam's
    // fires do, and takes no candidate: a step's room with nothing carried.
    let room = want.room(got, None);
    let apart = (got - want.step).abs();
    if apart > room.total() {
        return Err(format!(
            "after its finish the set scored the next token {got:.4} and an unbranched decode \
             {:.4}: {apart:.4} apart, room {room}: {:.2} times its room; its pages are not its \
             beam's",
            want.step,
            apart / room.total()
        )
        .into());
    }
    Ok((apart, room.total(), got))
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

/// The gap between `x` and the next value away from zero in bf16 (8
/// significant bits): the most a value rounded to bf16 at `x`'s magnitude
/// may sit from the value it stands for.
fn ulp_bf16(x: f32) -> f32 {
    ulp(x, 7)
}

/// The same gap in f32 (24 significant bits).
fn ulp_f32(x: f32) -> f32 {
    ulp(x, 23)
}

fn ulp(x: f32, fraction_bits: i32) -> f32 {
    let exponent = ((x.abs().to_bits() >> 23) & 0xff) as i32 - 127;
    2f32.powi(exponent.max(-126) - fraction_bits)
}

fn best_of(scores: &[f32]) -> f32 {
    scores.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// One step of a hypothesis rescored alone: its log-probability, its target
/// and largest logit, and its drift between row counts.
struct Rescored {
    step: f32,
    picked: f32,
    largest: f32,
    drift: f32,
}

impl Rescored {
    /// The room a beam's step `took` has from this rescore, each part a bf16
    /// (or f32) gap at the magnitude of the value it rounds:
    ///
    /// - the log-probability and the logits are rounded to bf16 by both
    ///   fires. Two roundings of values a drift apart land at most that
    ///   drift and one gap apart: one gap at the larger log-probability, and
    ///   one each at the target logit and the largest logit (which leads the
    ///   log-sum-exp). The beam's logits are not drained; they are the
    ///   rescore's within the drift, so the rescore's give the magnitudes;
    /// - a decode step (`carried`: the beam's score `before` it, `after` it,
    ///   and the best score `best` its fire took candidates relative to)
    ///   also rounds, on the beam alone, the relative score and the
    ///   candidate (relative plus log-probability) to bf16, and its scores
    ///   either side to f32;
    /// - a drift: the beam's fires run other kernels than a 1-row fire
    ///   folding every step (other tiles, splits and reduction orders for
    ///   their rows, a recurrence read through an unfolded run, compounding
    ///   through every layer and the cache rows written), which the user
    ///   allows (no batch invariance); `drift` is this step's own measure of
    ///   it, the log-probability moved between that 1-row fire and the
    ///   hypothesis run as the beam's fires run ([`rescore`]). Both sides may
    ///   carry one: twice the drift.
    fn room(&self, took: f32, carried: Option<(f32, f32, f32)>) -> Room {
        let candidate = carried.map_or(0.0, |(before, after, best)| {
            ulp_bf16(before - best) + ulp_bf16(after - best) + ulp_f32(before) + ulp_f32(after)
        });
        Room {
            log_prob: ulp_bf16(took.abs().max(self.step.abs())),
            logits: ulp_bf16(self.picked) + ulp_bf16(self.largest),
            candidate,
            drift: 2.0 * self.drift,
        }
    }
}

/// A step's room, by part ([`Rescored::room`]).
#[derive(Clone, Copy, Default)]
struct Room {
    log_prob: f32,
    logits: f32,
    candidate: f32,
    drift: f32,
}

impl Room {
    fn total(&self) -> f32 {
        self.log_prob + self.logits + self.candidate + self.drift
    }
}

impl std::fmt::Display for Room {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:.3} (log-probability {:.4}, logits {:.4}, candidate {:.4}, twice the drift {:.4})",
            self.total(),
            self.log_prob,
            self.logits,
            self.candidate,
            self.drift
        )
    }
}

/// Each step of `hypothesis` after `prompt`, teacher-forced one token a fire
/// on working sets of its own ([`teacher_forced`]): as a 1-row fire folding
/// every step, what a beam that read only its own ancestry takes there; and
/// as the beam's own fires run, `width` rows (the hypothesis and copies of
/// itself) with nothing folded past the prompt, how far the beam's kernels
/// drift from it. A recurrent layer read through a long unfolded run is
/// another computation from one folded every step, and the gap grows with
/// the run (Qwen3.5: up to 0.375 at log-probability -7 by step 255, with
/// row counts alone drifting 0 there).
async fn rescore(prompt: &[u32], hypothesis: &[u32], width: u32) -> Result<Vec<Rescored>> {
    let one = teacher_forced(prompt, hypothesis, 1, Fold::All).await?;
    let wide = teacher_forced(prompt, hypothesis, width, Fold::None).await?;
    Ok(one
        .iter()
        .zip(&wide)
        .map(|(&(step, picked, largest), &(paired, ..))| Rescored {
            step,
            picked,
            largest,
            drift: (step - paired).abs(),
        })
        .collect())
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
                    gather(reshape(intrinsics::output(0, "logits"), [copies, v]), iota(1)),
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
        let step = got.first().copied().ok_or("the rescoring fire returned nothing")?;
        let picked = picked.take_host::<Vec<f32>>().await.context("rescore drain")?;
        let largest = largest.take_host::<Vec<f32>>().await.context("rescore drain")?;
        let first = |of: &[f32]| of.first().copied().ok_or("the rescoring fire returned nothing");
        steps.push((step, first(&picked)?, first(&largest)?));
    }
    pipeline.close();
    Ok(steps)
}
