//! On-device beam search with logical ancestry masks.
//!
//! The prompt — chat-templated, or `prompt_tokens` as given — is every beam's
//! common ancestry. It is prefilled ONCE into the head of a working set, and
//! that fire's epilogue picks the first `beams` tokens (the top of the
//! prompt's next-token distribution), as greedy decoding picks its first.
//!
//! The beams then share that ONE working set, a pool repeated for every lane.
//! Each surviving beam appends its new token at the next free pool slot
//! (`slot = fill + lane`, computed on the device), and its ancestry is encoded
//! in the per-beam mask — inherit the parent's mask (`gather(mask, parent)`)
//! then set the one new cell (`eq(col, slot)`). That is the entire fork/prune
//! mechanism. The prompt is folded, so it is committed: every beam reads it
//! and no mask speaks of it. A mask's column `c` is pool slot `n + c`, the
//! cells past the prompt. A beam's lane reads the pool up to its own slot, as
//! every lane the host places does (it states no `kv_len`): its ancestors all
//! stand at earlier steps' slots, below this step's, and its mask hides the
//! other beams' cells among them.
//!
//! The search runs `max_tokens` steps: a beam is not retired at end of
//! sequence, so width 1 is greedy decoding of the same prompt to the same
//! length (the beam identity, checked under `verify` every step against the
//! raw argmax).

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
    /// `rollback_mode: "ckpt_sync"` when `rollback_mode` is left out.
    #[serde(default)]
    ckpt_sync: bool,
    /// `flush_free` (no checkpoint), `ckpt_sync` (each step checkpoints the
    /// pool and, when a beam changed parent, replays on a fork of it) or
    /// `stop_wait` (the replay every step). Any other name is refused.
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
    /// host. Either must fail `verify`.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rollback {
    FlushFree,
    CkptSync,
    StopWait,
}

/// The rollback the request names; a name this program does not run is
/// refused, never taken for the default.
fn rollback(input: &Input) -> Result<Rollback> {
    match input.rollback_mode.as_deref() {
        None if input.ckpt_sync => Ok(Rollback::CkptSync),
        None | Some("flush_free") => Ok(Rollback::FlushFree),
        Some("ckpt_sync") => Ok(Rollback::CkptSync),
        Some("stop_wait") => Ok(Rollback::StopWait),
        Some(other) => Err(format!(
            "unknown rollback_mode `{other}`: this program runs flush_free, ckpt_sync or stop_wait"
        )),
    }
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
    let mode = rollback(&input)?;
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
    // The prompt's slots, then one slot per beam per decode step.
    let decodes = max_steps - 1;
    let pool_len = u32::try_from(decodes)
        .ok()
        .and_then(|steps| steps.checked_mul(b))
        .and_then(|cells| cells.checked_add(n))
        .ok_or("beams * max_tokens exceeds the u32 range")?;
    let ws = WorkingSet::new();
    // The device chooses the beams' slots, so the pool is reserved up front.
    ws.reserve(pool_len).context("ws.reserve pool")?;

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

        // The first decode writes beam `i`'s token at slot `n + i`, column
        // `i` of the cells past the committed prompt; it sees the prompt and
        // its own cell.
        let cells = pool_len - n;
        let init_mask: Vec<bool> = (0..b)
            .flat_map(|lane| (0..cells).map(move |c| c == lane))
            .collect();
        let mask = Channel::from_shaped([b, cells], init_mask);
        let slots = Channel::from_iter(n..n + b);
        // A beam's RoPE position is its depth in the sequence, not its pool slot.
        let pos = Channel::from(vec![n; b as usize]);
        // Where the fire after this one writes its first beam.
        let fill = Channel::from([n + b]);
        fwd.mask(&mask)?;
        fwd.state(
            &pool,
            StateGeometry {
                positions: Some(&pos),
                slots: Some(&slots),
                kv_len: None,
                fold: Fold::None,
            },
        )
        .context("bind the beam pool")?;
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

            let base = fill.take(); // [1]
            let base_b = broadcast(reshape(&base, [1]), [b]); // [1] -> [B]
            let wpos = &base_b + iota(b); // [B]

            let inherited = if corrupt_parent {
                gather(mask.take(), (&parent + 1u32) % b)
            } else {
                gather(mask.take(), &parent)
            }; // bool [B,cells]
            // Column `c` is slot `n + c`.
            let col = broadcast(reshape(iota(cells) + n, [1, cells]), [b, cells]);
            let wpos_b = broadcast(reshape(&wpos, [b, 1]), [b, cells]);
            let newpos = eq(col, wpos_b); // bool [B,cells]
            mask.put(or(inherited, &newpos));

            // Device-resolved geometry is loop-carried: the host never drains
            // these rings, so every fire's values are re-put here.
            let filled = &base + b; // [1]
            slots.put(&wpos);
            pos.put(pos.take() + 1u32);
            fill.put(&filled);
        });

        let mut step = 0usize;
        let stop_wait = mode == Rollback::StopWait;
        if mode != Rollback::FlushFree {
            for _ in 0..decodes {
                let ckpt_ws = ws
                    .fork(&pipeline)
                    .with_context(|| format!("checkpoint @{step}"))?;
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
                        .with_context(|| format!("restore @{step}"))?;
                    let toks_r = Channel::from(&picked[..1]);
                    let ind_r = Channel::from([0u32, 1]);
                    let out_r = Channel::new([1], dtype::i32);
                    let fwd_r = ForwardPass::new();
                    fwd_r.embed(&toks_r, &ind_r)?;
                    fwd_r.state(&[&restore_ws], StateGeometry::folding())?;
                    fwd_r.epilogue(move || {
                        out_r.put(reshape(
                            reduce_argmax(intrinsics::output(0, "logits")),
                            [1],
                        ));
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
                scored = advance(&scored, &final_scores, &parents)?;
                step += 1;
            }
        } else {
            run_ahead(&pipeline, &fwd, decodes, async || {
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
                scored = advance(&scored, &final_scores, &parents)?;
                step += 1;
                Ok(ControlFlow::Continue(()))
            })
            .await?;
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
        let shaped = replay_beam(&prompt, &first, &trail, b, pool_len).await?;
        for (lane, hypothesis) in hypotheses.iter().enumerate() {
            let alone = rescore(&prompt, hypothesis, b, &shaped[lane]).await?;
            let own = &record[0][lane];
            let mut before = 0f32;
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
                // rescored alone, within one bf16 gap of each value the
                // log-probability is read from and the hypothesis's own
                // spread across compositions ([`rescore`]). A recurrence or
                // attention that read another beam's cells moves the beam's
                // candidate, not the hypothesis's rescores.
                let r = &alone[at];
                let room = ulp_bf16(mine.abs().max(r.step.abs()))
                    + ulp_bf16(r.picked)
                    + ulp_bf16(r.largest)
                    + r.spread;
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
                         {:.4} there: {apart:.4} apart, past {room:.3} (composition spread \
                         {:.4}, widest from the {}, {:.4} without the beam-shaped replay); its \
                         recurrence or attention read another beam's cells",
                        r.step, r.spread, r.by, r.unshaped,
                    ));
                }
                before = after;
            }
            let total: f32 = alone.iter().map(|s| s.step).sum();
            worst = worst.max((total - final_scores[lane]).abs());
        }
        format!(
            " greedy_mismatches={greedy_mismatches} verified={} worst_apart={worst:.4} worst_own={closest_own:.3} worst_step={closest:.3} max_spread={widest:.4} spread_by={widest_by} unshaped_spread={unshaped:.4}",
            hypotheses.len()
        )
    } else {
        String::new()
    };
    Ok(format!(
        "{text}\n[beam] width={b} steps={max_steps} best_score={:.4}{verified}",
        final_scores[best_lane]
    ))
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

/// Each step of `hypothesis` after `prompt`, teacher-forced one token a fire
/// on working sets of its own ([`teacher_forced`]): as a 1-row fire, what a
/// beam that read only its own ancestry takes there; and how far that moves
/// as the beam's own fires run it, at the beam's width (the hypothesis and
/// copies of itself), on the beam's own geometry at width 1 ([`replay`]: the
/// shared pool unfolded, read through its mask), and in the beam's own fire
/// shape (`shaped`, from [`replay_beam`]: every lane of every fire at its
/// width, slots and parents, its masks built on the host from the parents
/// the search reported). Other rows and another geometry
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
    shaped: &[(f32, f32, f32)],
) -> Result<Vec<Rescored>> {
    let one = teacher_forced(prompt, hypothesis, 1).await?;
    let others = [
        teacher_forced(prompt, hypothesis, width).await?,
        replay(prompt, hypothesis).await?,
        shaped.to_vec(),
    ];
    if shaped.len() != one.len() {
        return Err(format!(
            "the beam-shaped replay holds {} steps of a {}-step hypothesis",
            shaped.len(),
            one.len()
        ));
    }
    const SHAPES: [&str; 3] = ["width-copies", "width-1-replay", "beam-shaped-replay"];
    for (name, pass) in [
        ("1-row", &one),
        (SHAPES[0], &others[0]),
        (SHAPES[1], &others[1]),
        (SHAPES[2], &others[2]),
    ] {
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
/// decode fire as the search laid it (every lane at its width, its slot
/// `n + fire * width + lane` and depth), each lane read through a mask the
/// host builds from the reported parents (its parent's cells and its own),
/// not the one the device carried. Each pick's log-probability, target
/// logit and parent row's largest logit, followed along each final lane's
/// ancestry as the search's record is. A fire whose rows moved only with
/// its composition (its width, row count and kernels) moves here alike; a
/// lane that read cells outside its ancestry does not, as this mask holds
/// none of them.
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
/// rows (the hypothesis repeated, each copy on a working set of its own),
/// read off the first.
async fn teacher_forced(
    prompt: &[u32],
    hypothesis: &[u32],
    copies: u32,
) -> Result<Vec<(f32, f32, f32)>> {
    let v = model::output_vocab_size();
    let sets: Vec<WorkingSet> = (0..copies).map(|_| WorkingSet::new()).collect();
    let bound: Vec<&WorkingSet> = sets.iter().collect();
    let pipeline = Pipeline::new();
    let mut steps = Vec::with_capacity(hypothesis.len());
    for (at, &target) in hypothesis.iter().enumerate() {
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
        fwd.state(&bound, StateGeometry::folding())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn request(json: &str) -> Input {
        serde_json::from_str(json).unwrap()
    }

    /// A rollback the program does not run is refused by name, not flushed.
    #[test]
    fn an_unknown_rollback_mode_is_refused() {
        let err = rollback(&request(r#"{"rollback_mode": "ckpt"}"#)).unwrap_err();
        assert!(err.contains("unknown rollback_mode `ckpt`"), "{err}");
        assert_eq!(rollback(&request("{}")), Ok(Rollback::FlushFree));
        assert_eq!(rollback(&request(r#"{"ckpt_sync": true}"#)), Ok(Rollback::CkptSync));
        assert_eq!(
            rollback(&request(r#"{"rollback_mode": "stop_wait"}"#)),
            Ok(Rollback::StopWait)
        );
    }
}
