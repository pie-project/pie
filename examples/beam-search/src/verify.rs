//! `verify`: the search checked against its hypotheses rescored alone, its
//! finish against a fresh prefill, and width 1 against greedy decoding; and
//! the negative controls (`corrupt`) each of those checks must catch.

use super::advance;
use inferlet::eta::prelude::*;

/// What a fire computed beside its picks, read only under `verify`: the
/// raw-logit argmax, and each pick's own log-probability, target logit and
/// largest logit, the beam's record of its candidates, which its scores must
/// carry exactly.
#[derive(Clone, Copy)]
pub(crate) struct Taps {
    greedy: Channel,
    own: Channel,
    picked: Channel,
    largest: Channel,
}

impl Taps {
    /// The prefill's: one argmax beside `b` picks.
    pub(crate) fn prefill(b: u32) -> Taps {
        Taps {
            greedy: Channel::new([1], dtype::i32),
            own: Channel::new([b], dtype::f32),
            picked: Channel::new([b], dtype::f32),
            largest: Channel::new([b], dtype::f32),
        }
    }

    /// A decode fire's, ringed as the search's own outputs are.
    pub(crate) fn decode(b: u32) -> Taps {
        let ring = |dtype| Channel::new([b], dtype).capacity(channel_capacity() as u32);
        Taps {
            greedy: ring(dtype::i32),
            own: ring(dtype::f32),
            picked: ring(dtype::f32),
            largest: ring(dtype::f32),
        }
    }

    /// In the prefill's epilogue: its bf16 `logits` `[v]`, and the top `b`
    /// log-probabilities `s` at `i`.
    pub(crate) fn put_prefill(&self, logits: &Tensor, s: &Tensor, i: &Tensor, b: u32) {
        self.greedy.put(reshape(reduce_argmax(logits), [1]));
        self.own.put(cast(s, dtype::f32));
        self.picked.put(cast(gather(logits, i), dtype::f32));
        self.largest.put(broadcast(
            reshape(cast(reduce_max(logits), dtype::f32), [1]),
            [b],
        ));
    }

    /// In a decode fire's epilogue: its bf16 `logits` `[b, v]`, their
    /// log-softmax `lsm`, the picks `i` (flat over `b * v`) and their
    /// `parent` rows. At `beams == 1` the raw-logit argmax must equal the
    /// pick every step.
    pub(crate) fn put_decode(
        &self,
        logits: &Tensor,
        lsm: &Tensor,
        i: &Tensor,
        parent: &Tensor,
        b: u32,
        v: u32,
    ) {
        self.greedy.put(reshape(reduce_argmax(logits), [b]));
        let flat = reshape(logits, [b * v]);
        self.own
            .put(cast(gather(reshape(lsm, [b * v]), i), dtype::f32));
        self.picked.put(cast(gather(&flat, i), dtype::f32));
        let top = cast(reduce_argmax(logits), dtype::u32);
        let rows = gather(&flat, iota(b) * v + &top);
        self.largest.put(cast(gather(&rows, parent), dtype::f32));
    }
}

/// What the search reported, as `verify` reads it back.
pub(crate) struct Record {
    /// Picks that were not their row's raw-logit argmax.
    greedy_mismatches: usize,
    /// Each lane's score after each of its steps, its ancestry's first.
    scored: Vec<Vec<f32>>,
    /// Each lane's own candidates along its ancestry (log-probability,
    /// target logit, largest logit).
    record: [Vec<Vec<f32>>; 3],
    /// The best score each decode fire took its candidates relative to.
    bests: Vec<f32>,
    /// Each decode fire's picks (tokens, parents), from which the beam's own
    /// fires are replayed ([`replay_beam`]).
    trail: Vec<(Vec<i32>, Vec<u32>)>,
}

impl Record {
    /// The prefill's taps, beside its picks `first` and their `scores`.
    pub(crate) async fn prefill(taps: Taps, first: &[i32], scores: &[f32]) -> Result<Record> {
        let greedy = taps
            .greedy
            .take_host::<Vec<i32>>()
            .await
            .context("prefill drain")?;
        let mut record: [Vec<Vec<f32>>; 3] = Default::default();
        for (slot, channel) in record.iter_mut().zip([taps.own, taps.picked, taps.largest]) {
            let got = channel
                .take_host::<Vec<f32>>()
                .await
                .context("prefill drain")?;
            *slot = got.iter().map(|&x| vec![x]).collect();
        }
        Ok(Record {
            greedy_mismatches: usize::from(first.first() != greedy.first()),
            scored: scores.iter().map(|&s| vec![s]).collect(),
            record,
            bests: vec![0.0],
            trail: Vec::new(),
        })
    }

    /// One decode fire's taps, beside its `picked` tokens, their `parents`
    /// and the `scores` it left each lane with.
    pub(crate) async fn step(
        &mut self,
        taps: Taps,
        picked: &[i32],
        parents: &[u32],
        scores: &[f32],
    ) -> Result<()> {
        let carried_best = best_of(self.scored.iter().filter_map(|lane| lane.last().copied()));
        let greedy = taps.greedy.take_host::<Vec<i32>>().await?;
        self.trail.push((picked.to_vec(), parents.to_vec()));
        for (slot, channel) in self
            .record
            .iter_mut()
            .zip([taps.own, taps.picked, taps.largest])
        {
            let got = channel.take_host::<Vec<f32>>().await?;
            *slot = advance(slot, &got, parents)?;
        }
        self.bests.push(carried_best);
        self.greedy_mismatches += picked.iter().zip(&greedy).filter(|(p, g)| p != g).count();
        self.scored = advance(&self.scored, scores, parents)?;
        Ok(())
    }
}

/// The lane with the best score (the last of equals).
pub(crate) fn best(scores: &[f32]) -> usize {
    scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, x)| a.total_cmp(x))
        .map_or(0, |(index, _)| index)
}

fn best_of(scores: impl Iterator<Item = f32>) -> f32 {
    scores.fold(f32::NEG_INFINITY, f32::max)
}

/// The search's finish (BEAM-PAGE-LISTS.md §6.6), checked: the set adopts
/// the best beam's pages, its `fired` slots from `n` in its own order, and a plain fire on
/// it scores the next token as a fresh prefill of the same tokens does
/// ([`finish_check`]). The report's fragment.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn finish(
    fwd: &ForwardPass,
    ws: &WorkingSet,
    prompt: &[u32],
    hypotheses: &[Vec<u32>],
    scores: &[f32],
    last_parents: &[u32],
    n: u32,
    fired: u32,
    corrupt: Option<&str>,
) -> Result<String> {
    let best = best(scores);
    let slot = n + fired;
    // `--corrupt finish`: the set adopts another beam than the one checked,
    // the control the finish's check must catch: the one whose hypothesis
    // parts from the checked one's earliest.
    let mut parted = None;
    let adopted = if corrupt == Some("finish") {
        let shared = |lane: usize| {
            hypotheses[lane]
                .iter()
                .zip(&hypotheses[best])
                .take_while(|(x, y)| x == y)
                .count()
        };
        let lane = (0..hypotheses.len())
            .filter(|&lane| lane != best)
            .min_by_key(|&lane| shared(lane))
            .unwrap_or(best);
        // A beam that parts from the checked one only near its end holds
        // nearly the same context, which no check can tell from rounding:
        // the control is formed only where the adopted beam parts at least
        // `CONTROL_PARTS_BEFORE` tokens before the end. A formed one is
        // caught where its context still moves the set's next-token head
        // past rounding; one whose context no longer moves it cannot be told
        // from correct output by any check on outputs.
        let (at, of) = (shared(lane), hypotheses[best].len());
        if at + CONTROL_PARTS_BEFORE > of {
            return Err(format!(
                "no control formed: every other beam shares {at} of the checked \
                 hypothesis's {of} tokens, parting later than \
                 {CONTROL_PARTS_BEFORE} before its end"
            ));
        }
        parted = Some((at, of));
        lane
    } else {
        best
    };
    // The final beam adopted lies on its parent's pages, the last fire's beam.
    let device_beam = last_parents.get(adopted).copied().unwrap_or(0);
    fwd.adopt_beam(device_beam, slot)
        .context("finish the search")?;
    let read = match corrupt {
        Some("finish-other") => FinishRead::Other,
        Some("finish-stale") => FinishRead::Committed(n),
        _ => FinishRead::Set,
    };
    let (apart, room, got) = finish_check(ws, prompt, &hypotheses[best], slot, read).await?;
    Ok(format!(
        " finished_apart={apart:.6} finished_room={room:.3} finished_got={got:.6}{}",
        parted.map_or(String::new(), |(at, of)| format!(
            " finished_parts_at={at}/{of}"
        ))
    ))
}

/// The whole check, after the search: width 1 against greedy decoding; each
/// lane's every score step against its own fire's candidate, and against
/// its hypothesis rescored alone; each final score against the rescored sum.
/// The report's fragment, or the first failure by name. The `score` and
/// `uniform` controls are applied here, to what the search reported.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn check(
    record: Record,
    prompt: &[u32],
    first: &[i32],
    hypotheses: &[Vec<u32>],
    final_scores: &mut [f32],
    b: u32,
    max_steps: usize,
    corrupt: Option<&str>,
) -> Result<String> {
    let Record {
        greedy_mismatches,
        mut scored,
        mut record,
        bests,
        trail,
    } = record;
    // The beam identity is only an identity at width 1; at B > 1 the top-B
    // block legitimately picks non-argmax continuations for the lower lanes.
    if b == 1 && greedy_mismatches != 0 {
        return Err(format!(
            "beam identity violated: width-1 beam search disagreed with greedy \
                 argmax on {greedy_mismatches} of {max_steps} steps"
        ));
    }
    if corrupt == Some("score")
        && let Some(last) = scored.first_mut().and_then(|lane| lane.last_mut())
    {
        *last += 4.0;
    }
    let n = u32::try_from(prompt.len()).map_err(|_| "a prompt past u32 rows")?;
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
    let mut shaped = replay_beam(prompt, first, &trail, b, pool_len).await?;
    // `--corrupt uniform`: every lane's every candidate computed the same
    // wrong amount off, as a kernel fault every composition of the
    // search's shape reproduces would: its scores, its own record and the
    // beam-shaped replay all moved alike, so the bookkeeping holds and a
    // spread-scaled room would absorb it. The absolute room must not.
    if corrupt == Some("uniform") {
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
        let alone = rescore(prompt, hypothesis, b, &shaped[lane]).await?;
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
                ulp_bf16(relative) + ulp_bf16(relative + mine) + ulp_f32(after) + ulp_f32(before)
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
        // A NaN on either side fails, as no comparison holds for it.
        if !matches!(
            apart.partial_cmp(&bound),
            Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
        ) {
            return Err(format!(
                "beam {lane}'s final score {:.4} and its hypothesis rescored alone {total:.4}: \
                 {apart:.4} apart, past {bound:.4}, its {} steps' rooms summed",
                final_scores[lane],
                alone.len()
            ));
        }
    }
    Ok(format!(
        " greedy_mismatches={greedy_mismatches} verified={} worst_apart={worst:.4} worst_total={worst_total:.3} worst_own={closest_own:.3} worst_step={closest:.3} max_spread={widest:.4} spread_by={widest_by} unshaped_spread={unshaped:.4}",
        hypotheses.len()
    ))
}

/// After a search's finish, one plain fire on its working set: the last
/// token of `hypothesis` (the one the search fed no fire) at slot `slot`,
/// folding, the log-probabilities of its [`FINISH_TOKENS`] likeliest next
/// tokens, each against a fresh prefill of `prompt` and `hypothesis`
/// scoring that token there ([`fresh_last`]), each within its own room
/// ([`Rescored::room`]). How far apart, the room and the score of the one
/// farthest from its room; past its room it fails, stating what it measured
/// (the rank, the values, how far past the room). It establishes only that
/// the set's next-token head differs from the fresh prefill's past
/// rounding, not what made it differ.
async fn finish_check(
    ws: &WorkingSet,
    prompt: &[u32],
    hypothesis: &[u32],
    slot: u32,
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
            .chain(
                hypothesis[..hypothesis.len() - 1]
                    .iter()
                    .map(|&t| (t + 1) as i32),
            )
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
    let out = Channel::new([FINISH_TOKENS], dtype::f32);
    let next = Channel::new([FINISH_TOKENS], dtype::i32);
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
        // The head of the next-token distribution: a set that holds another
        // context moves it, though one token's log-probability may move less
        // than its room.
        let logits = cast(reshape(intrinsics::output(0, "logits"), [v]), dtype::bf16);
        let (s, i) = top_k(log_softmax(&logits), FINISH_TOKENS);
        out.put(cast(&s, dtype::f32));
        next.put(cast(&i, dtype::i32));
    });
    fwd.submit(&pipeline).context("finished-set submit")?;
    let got = out
        .take_host::<Vec<f32>>()
        .await
        .context("finished-set drain")?;
    let token = next
        .take_host::<Vec<i32>>()
        .await
        .context("finished-set drain")?;
    pipeline.close();
    if got.len() != FINISH_TOKENS as usize || token.len() != got.len() {
        return Err("the finished set's fire returned nothing".into());
    }
    let tokens: Vec<u32> = token.iter().map(|&t| t as u32).collect();
    let wants = fresh_last(prompt, hypothesis, &tokens).await?;
    // The set's fire and the fresh prefill each round their logits and
    // log-probabilities once, as a step does, and take no candidate: each
    // token's room is a step's, with nothing carried. The farthest from its
    // room is the one reported.
    let mut farthest: Option<(usize, f32, f32)> = None;
    for (at, (&got, want)) in got.iter().zip(&wants).enumerate() {
        let got = finite(&format!("the finished set's {}-ranked token", at + 1), got)?;
        let room = want.room(got);
        let apart = (got - want.step).abs();
        if farthest.is_none_or(|(_, far, far_room)| apart / room > far / far_room) {
            farthest = Some((at, apart, room));
        }
    }
    let (at, apart, room) = farthest.ok_or("the finished set's fire returned nothing")?;
    let (got, want) = (got[at], &wants[at]);
    if apart > room {
        return Err(format!(
            "after its finish the set scored its {}-ranked next token ({}) {got:.4} and a \
             fresh prefill {:.4}: {apart:.4} apart, {:.2} times its room {room:.4} \
             ({STEP_ULPS} bf16 gaps at its logits' scale); past rounding, so the set's next-token \
             head is not the fresh prefill's",
            at + 1,
            tokens[at],
            want.step,
            apart / room,
        ));
    }
    Ok((apart, room, got))
}

/// The next tokens a finish's check reads ([`finish_check`]): the head of
/// the finished set's distribution, compared token by token.
const FINISH_TOKENS: u32 = 16;

/// How far before its end the beam `--corrupt finish` adopts parts from the
/// checked one, at least: a beam parting later holds nearly the checked
/// context, and the control is not formed.
const CONTROL_PARTS_BEFORE: usize = 16;

/// The reference a finish's check reads ([`finish_check`]): `prompt` and
/// `hypothesis` prefilled as one fire on a working set of its own, folding,
/// and at its last row, for each of `last`, its log-probability and logit,
/// beside the row's largest logit. A fire built so, like the set's own fire,
/// rounds its row once; a decode one token a fire (as [`teacher_forced`]
/// takes it) carries its own steps' drift on a hybrid model's recurrence,
/// which the set's check must not be charged with.
async fn fresh_last(prompt: &[u32], hypothesis: &[u32], last: &[u32]) -> Result<Vec<Rescored>> {
    let v = model::output_vocab_size();
    let set = WorkingSet::new();
    let pipeline = Pipeline::new();
    let fed: Vec<i32> = prompt.iter().chain(hypothesis).map(|&t| t as i32).collect();
    let rows = u32::try_from(fed.len()).map_err(|_| "a sequence past u32 rows")?;
    let count = u32::try_from(last.len()).map_err(|_| "targets past u32")?;
    let toks = Channel::from(&fed[..]);
    let indptr = Channel::from([0u32, rows]);
    let wanted = Channel::from(last);
    let out = Channel::new([count], dtype::f32);
    let picked = Channel::new([count], dtype::f32);
    let largest = Channel::new([1], dtype::f32);
    let fwd = ForwardPass::new();
    fwd.embed(&toks, &indptr)?;
    fwd.state(&[&set], StateGeometry::folding())
        .context("bind the fresh prefill")?;
    fwd.epilogue(move || {
        // The last row, at the sampler's bf16 grain.
        let logits = cast(reshape(intrinsics::output(0, "logits"), [v]), dtype::bf16);
        let wanted = wanted.take();
        picked.put(cast(gather(&logits, &wanted), dtype::f32));
        largest.put(cast(reshape(reduce_max(&logits), [1]), dtype::f32));
        out.put(cast(gather(log_softmax(&logits), &wanted), dtype::f32));
    });
    fwd.submit(&pipeline).context("fresh-prefill submit")?;
    let step = out
        .take_host::<Vec<f32>>()
        .await
        .context("fresh-prefill drain")?;
    let picked = picked
        .take_host::<Vec<f32>>()
        .await
        .context("fresh-prefill drain")?;
    let largest = largest
        .take_host::<Vec<f32>>()
        .await
        .context("fresh-prefill drain")?;
    pipeline.close();
    let largest = *largest
        .first()
        .ok_or("the fresh prefill returned nothing")?;
    if step.len() != last.len() || picked.len() != last.len() {
        return Err("the fresh prefill returned fewer targets than it was given".into());
    }
    step.iter()
        .zip(&picked)
        .enumerate()
        .map(|(at, (&step, &picked))| {
            Ok(Rescored {
                step: finite(&format!("the fresh prefill of target {at}"), step)?,
                picked: finite(&format!("the fresh prefill of target {at}"), picked)?,
                largest: finite("the fresh prefill's largest logit", largest)?,
                spread: 0.0,
                by: "fresh-prefill",
                unshaped: 0.0,
            })
        })
        .collect()
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

/// What the finish's fire reads: the finished set as it stands, or (the
/// finish's negative controls) another sequence's set, or only the set's
/// first `c` slots, the committed state.
#[derive(Clone, Copy)]
enum FinishRead {
    Set,
    Other,
    Committed(u32),
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
/// mask), and in the search's own fire shape (`shaped`, from
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
    shaped: &[(f32, f32, f32)],
) -> Result<Vec<Rescored>> {
    let one = teacher_forced(prompt, hypothesis, 1, Fold::All).await?;
    if shaped.len() != one.len() {
        return Err(format!(
            "the beam-shaped replay holds {} steps of a {}-step hypothesis",
            shaped.len(),
            one.len()
        ));
    }
    let others = [
        teacher_forced(prompt, hypothesis, width, Fold::None).await?,
        replay(prompt, hypothesis).await?,
        shaped.to_vec(),
    ];
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
/// the prompt's fire the first copy's alone), read off the first. The
/// prompt is folded; each step after it folds by `fold`.
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
            picked.put(cast(gather(&logits, &wanted), dtype::f32));
            largest.put(cast(reshape(reduce_max(&logits), [1]), dtype::f32));
            out.put(cast(gather(log_softmax(&logits), &wanted), dtype::f32));
        });
        fwd.submit(&pipeline).context("rescore submit")?;
        let got = out.take_host::<Vec<f32>>().await.context("rescore drain")?;
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
        steps.push((first(&got)?, first(&picked)?, first(&largest)?));
    }
    pipeline.close();
    Ok(steps)
}
