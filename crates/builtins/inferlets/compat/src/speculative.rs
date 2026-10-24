//! A block drafter's loop for the API inferlets: one draft fire and one
//! verify fire a round, where drafting pays.
//!
//! At temperature zero the verify keeps the proposals the target's argmax
//! agrees with. Above it the verify samples: the drafter names its tokens
//! without a distribution (it drafted from a point mass), so each verify row
//! lands the target's chance of its proposal and a draw with the proposal
//! struck out (`inferlet::speculative::point_rows`), and the host keeps a
//! proposal with that chance and stops on the struck-out draw
//! (`verify_point_rows`); the text is distributed as plain sampling's.

use inferlet::eta::prelude::*;
use inferlet::speculative::{SplitMix64, point_rows, verify_point_rows};
use std::ops::ControlFlow;

enum Round<'a> {
    Prompt(&'a [u32]),
    Draft,
    Verify(u32),
    /// One token, folding every buffered row.
    Plain,
}

/// How a non-draft pass picks the target's tokens above temperature zero.
#[derive(Clone, Copy)]
struct Pick {
    temperature: f32,
    vocab: u32,
    /// The Gumbel stream every pass's draws advance.
    noise: Channel,
}

/// What a pass landed: a token a row (the target's argmax, a draw past
/// the row's proposal, or the draft's picks), and under a `Pick` each
/// row's chance of its proposal.
struct Landed {
    tokens: Vec<i32>,
    chances: Vec<f32>,
}

async fn pass(
    pipe: &Pipeline,
    ws: &WorkingSet,
    tokens: &[i32],
    base: u32,
    round: Round<'_>,
    pick: Option<&Pick>,
) -> Result<Landed> {
    let count = tokens.len() as u32;
    let end = base + count;
    let drafting = matches!(round, Round::Draft);
    let toks = Channel::from(tokens).named("spec_tokens");
    let indptr = Channel::from([0u32, count]).named("spec_indptr");
    let slots = Channel::from_iter(base..end).named("spec_slots");
    let readouts = match round {
        Round::Prompt(_) => vec![count - 1],
        Round::Draft | Round::Verify(_) | Round::Plain => (0..count).collect(),
    };
    let shown = readouts.len() as u32;
    let readout = Channel::from(readouts).named("spec_readout");
    let fold = match round {
        Round::Prompt(_) | Round::Plain => None,
        Round::Draft => Some(0),
        Round::Verify(survivors) => Some(survivors),
    };
    let fold = fold.map(|n| Channel::from([n]).named("spec_fold"));
    let fold = match &fold {
        Some(count) => Fold::Count(count),
        None => Fold::All,
    };
    let out = Channel::new([shown], dtype::i32).named("spec_output");
    // Row `j` of a verify reads the token proposal `j` claims, the input
    // one past it; the last row (and a prompt's or plain fire's one) has
    // none.
    let sampled = pick.filter(|_| !drafting).map(|pick| {
        let mut proposals: Vec<u32> = match round {
            Round::Verify(_) => tokens[1..].iter().map(|&t| t as u32).collect(),
            _ => Vec::new(),
        };
        proposals.resize(shown as usize, 0);
        (
            *pick,
            Channel::from(proposals).named("spec_proposals"),
            Channel::new([shown], dtype::f32).named("spec_chances"),
        )
    });
    let chances = sampled.as_ref().map(|(_, _, chances)| *chances);
    let fwd = ForwardPass::new();
    if drafting {
        fwd.set_drafting_block(true).context("draft block")?;
    }
    fwd.embed(&toks, &indptr)?;
    fwd.readout(&readout)?;
    fwd.state(
        &[ws],
        StateGeometry {
            slots: Some(&slots),
            fold,
            ..StateGeometry::default()
        },
    )?;
    if let Round::Prompt(ends) = round {
        fwd.checkpoints(ends)?;
    }
    fwd.epilogue(move || {
        if drafting {
            out.put(reshape(intrinsics::mtp_drafts(shown), [shown]));
            return;
        }
        let Some((pick, proposals, chances)) = &sampled else {
            out.put(reshape(reduce_argmax(intrinsics::logits()), [shown]));
            return;
        };
        let logits = reshape(intrinsics::logits(), [shown, pick.vocab]);
        let r = pick.noise.take();
        let (chance, draws) = point_rows(
            &(&logits / pick.temperature),
            &proposals.read(),
            shown,
            pick.vocab,
            &r,
        );
        out.put(reshape(draws, [shown]));
        chances.put(reshape(chance, [shown]));
        pick.noise.put(&r + iota(2));
    });
    fwd.submit(pipe).context("speculative submit")?;
    let tokens = out
        .take_host::<Vec<i32>>()
        .await
        .context("speculative readback")?;
    let chances = match chances {
        Some(chances) => chances
            .take_host::<Vec<f32>>()
            .await
            .context("speculative chances")?,
        None => Vec::new(),
    };
    Ok(Landed { tokens, chances })
}

/// Fire prices, the median of the last few of each shape, so the loop drafts
/// only while a round returns more tokens a second than a plain fire.
struct Prices {
    draft: Vec<u64>,
    /// Indexed by verify width; `[1]` is the plain fire.
    verify: Vec<Vec<u64>>,
}

impl Prices {
    const KEEP: usize = 5;

    fn keep(bucket: &mut Vec<u64>, ns: u64) {
        bucket.push(ns);
        if bucket.len() > Self::KEEP {
            bucket.remove(0);
        }
    }

    fn median(bucket: &[u64]) -> Option<f64> {
        if bucket.is_empty() {
            return None;
        }
        let mut sorted = bucket.to_vec();
        sorted.sort_unstable();
        Some(sorted[sorted.len() / 2] as f64)
    }

    /// An unfired width prices on the line between the fired widths around
    /// it (the plain fire is width 1), so a narrower rung gets tried.
    fn round(&self, width: u32) -> Option<f64> {
        let draft = Self::median(&self.draft)?;
        let w = width as usize;
        let below = (1..=w)
            .rev()
            .find_map(|v| Self::median(&self.verify[v]).map(|p| (v, p)))?;
        let verify = match (w + 1..self.verify.len())
            .find_map(|v| Self::median(&self.verify[v]).map(|p| (v, p)))
        {
            _ if below.0 == w => below.1,
            Some(above) => {
                below.1 + (above.1 - below.1) * (w - below.0) as f64 / (above.0 - below.0) as f64
            }
            None => below.1,
        };
        Some(draft + verify)
    }
}

/// Per-position acceptance, decayed so a changed text shows within a few rounds.
struct Yield {
    hits: Vec<f64>,
    tries: Vec<f64>,
}

impl Yield {
    const DECAY: f64 = 0.9;

    /// A round verified `width - 1` proposals and kept the first `kept`.
    fn record(&mut self, kept: usize, width: u32) {
        for (h, t) in self.hits.iter_mut().zip(self.tries.iter_mut()) {
            *h *= Self::DECAY;
            *t *= Self::DECAY;
        }
        for i in 0..kept {
            self.hits[i] += 1.0;
            self.tries[i] += 1.0;
        }
        if kept + 1 < width as usize {
            self.tries[kept] += 1.0;
        }
    }

    /// Tokens a round at `width` is expected to emit, the target's own included.
    fn expect(&self, width: u32) -> f64 {
        let (mut chain, mut total, mut rate) = (1.0f64, 1.0f64, 1.0f64);
        for i in 0..width.saturating_sub(1) as usize {
            // An untried position takes the rate of the one before it.
            if self.tries[i] > 0.0 {
                rate = self.hits[i] / self.tries[i];
            }
            chain *= rate;
            total += chain;
        }
        total
    }
}

/// The widths a round may verify at: a fire's price is a staircase in rows.
fn rungs(widest: u32) -> Vec<u32> {
    let mut out: Vec<u32> = [2u32, 4, 8, 16]
        .into_iter()
        .filter(|w| *w < widest)
        .collect();
    out.push(widest);
    out
}

/// Generate as the target would, drafting where it pays: greedily, or with
/// `sampled`'s `(temperature, seed)` by speculative sampling.
pub async fn generate(
    prompt: &[u32],
    ends: &[u32],
    max_tokens: usize,
    draft: model::BlockDrafter,
    sampled: Option<(f32, u32)>,
    on_token: &mut dyn FnMut(u32) -> ControlFlow<()>,
) -> Result<(usize, usize)> {
    let block = draft.rows;
    // The first block row whose pick is a proposal (1 for DFlash, 0 for DSpark),
    // and the widest verify a block feeds: the anchor and every proposal.
    let from = draft.proposals_from.min(1) as usize;
    let widest = block + 1 - from as u32;
    let ws = WorkingSet::new();
    let pipe = Pipeline::new();
    let prompt: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let pick = sampled.map(|(temperature, seed)| Pick {
        temperature,
        vocab: model::output_vocab_size(),
        noise: Channel::from([seed, 0u32]).named("spec_noise"),
    });
    // The acceptance draws, a stream apart from the device's noise.
    let mut uniform = SplitMix64::new(u64::from(sampled.map_or(0, |(_, seed)| seed)) ^ 0x5bec);
    let pick = pick.as_ref();
    let mut anchor = pass(&pipe, &ws, &prompt, 0, Round::Prompt(ends), pick)
        .await?
        .tokens[0];
    let cached = ws.cached_tokens() as usize;
    let mut generated = 1;
    if on_token(anchor as u32).is_break() || generated == max_tokens {
        pipe.close();
        return Ok((generated, cached));
    }
    let mut held = prompt.len() as u32;
    let mut survivors = 0;
    let mut prices = Prices {
        draft: Vec::new(),
        verify: vec![Vec::new(); widest as usize + 1],
    };
    let mut yields = Yield {
        hits: vec![0.0; widest as usize],
        tries: vec![0.0; widest as usize],
    };
    let rungs = rungs(widest);
    // Plain fires left before drafting is tried again, and the gap the next
    // closure waits: a probe that loses doubles it.
    let (mut wait, mut backoff) = (0u32, 8u32);
    let mut rounds = 0u32;
    while generated < max_tokens {
        // Price the plain fire first; then draft at the widest rung until the
        // yield is known, then at whichever width (or none) earns the most.
        let best = rungs
            .iter()
            .filter_map(|&w| prices.round(w).map(|p| (w, yields.expect(w) / p)))
            .max_by(|a, b| a.1.total_cmp(&b.1));
        let width = match (Prices::median(&prices.verify[1]), best) {
            (None, _) => None,
            // Beside other requests a plain fire batches with theirs for
            // nearly nothing, and a round would cost all of them.
            _ if Pipeline::live() > 1 => None,
            (Some(_), _) if rounds < 2 => Some(widest),
            (Some(_), _) if wait > 1 => {
                wait -= 1;
                None
            }
            // The wait is over: one round probes whether drafting pays again.
            (Some(_), Some((w, _))) if wait == 1 => {
                wait = 0;
                Some(w)
            }
            (Some(plain), Some((w, rate))) if rate * plain >= 1.0 => {
                backoff = 8;
                Some(w)
            }
            _ => {
                wait = backoff;
                backoff = (backoff * 2).min(256);
                None
            }
        };
        let Some(width) = width else {
            // A plain fire folds whatever the last round buffered.
            let fired = inferlet::monotonic_now_ns();
            let truth = pass(&pipe, &ws, &[anchor], held, Round::Plain, pick)
                .await?
                .tokens;
            if survivors == 0 {
                Prices::keep(&mut prices.verify[1], inferlet::monotonic_now_ns() - fired);
            }
            survivors = 0;
            anchor = truth[0];
            held += 1;
            generated += 1;
            if on_token(anchor as u32).is_break() || generated == max_tokens {
                pipe.close();
                return Ok((generated, cached));
            }
            continue;
        };
        // The draft and its verify write the same slots: the verify replaces
        // the draft, and the next round's slots replace what it rejected.
        let mut input = vec![draft.mask_token as i32; block as usize];
        input[0] = anchor;
        let fired = inferlet::monotonic_now_ns();
        let picks = pass(&pipe, &ws, &input, held, Round::Draft, None)
            .await?
            .tokens;
        Prices::keep(&mut prices.draft, inferlet::monotonic_now_ns() - fired);
        let proposals = &picks[from..from + width as usize - 1];
        let mut verify = vec![anchor];
        verify.extend_from_slice(proposals);
        let fired = inferlet::monotonic_now_ns();
        let landed = pass(&pipe, &ws, &verify, held, Round::Verify(survivors), pick).await?;
        Prices::keep(
            &mut prices.verify[width as usize],
            inferlet::monotonic_now_ns() - fired,
        );
        let k = proposals.len();
        let (kept, next) = if pick.is_some() {
            let verdict = verify_point_rows(
                &landed.chances[..k],
                &landed.tokens[..k]
                    .iter()
                    .map(|&t| t as u32)
                    .collect::<Vec<_>>(),
                landed.tokens[k] as u32,
                &mut uniform,
            );
            (verdict.kept, verdict.next as i32)
        } else {
            let kept = proposals
                .iter()
                .zip(&landed.tokens)
                .take_while(|(p, t)| p == t)
                .count();
            (kept, landed.tokens[kept])
        };
        yields.record(kept, width);
        rounds += 1;
        survivors = kept as u32 + 1;
        anchor = next;
        held += survivors;
        for &token in proposals[..kept].iter().chain(std::iter::once(&anchor)) {
            generated += 1;
            if on_token(token as u32).is_break() || generated == max_tokens {
                pipe.close();
                return Ok((generated, cached));
            }
        }
    }
    pipe.close();
    Ok((generated, cached))
}
