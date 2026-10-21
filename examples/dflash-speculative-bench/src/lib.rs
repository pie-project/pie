//! A block drafter's speculative loop: one draft fire and one verify fire a
//! round.
//!
//! ```text
//! draft    [anchor, MASK x block-1]      one fire, `set-drafting-block`
//!          -> d[0..block-1]              d[j] proposes position held+j+1
//! verify   [anchor, d[0] .. d[w-2]]      one fire, the trunk, plainly causal
//!          -> t[i] = argmax(row i)       the target's own token for held+i+1
//! keep     the longest prefix where d[j] == t[j], then t at the break
//! ```
//!
//! The break's token is the target's, so a round emits `kept + 1` tokens and
//! the text is what one-token-a-fire greedy decode writes (up to the bf16
//! accumulation-order floor at a near-tie: a wide verify fire sums in a
//! different order than a one-row one). `baseline` runs that decode over the
//! same prompt, so the speedup is measured rather than argued.
//!
//! **ONE FRAME A ROUND.** The draft and the verify are submitted together:
//! the draft's epilogue lays out the verify's tokens (`[anchor, d[..]]`) on a
//! channel the verify embeds, so the verify runs on the device right behind
//! the draft and the host reads the round once, after the verify. The verify
//! also counts the agreeing prefix (`kept`), which the host would otherwise
//! need the proposals for. `chain: false` reads the draft back before the
//! verify instead, as does a round whose width the draft's margins choose.

use inferlet::eta::prelude::*;
use inferlet::{chat, monotonic_now_ns, session};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default)]
    prompt_tokens: Option<Vec<u32>>,
    #[serde(default = "default_system")]
    system: String,
    #[serde(default)]
    no_think: bool,
    /// Drop the template's stop tokens so the loop runs to `max_tokens`.
    #[serde(default)]
    ignore_eos: bool,
    /// Announce `ready` and wait for the harness's `start` before the clock.
    #[serde(default)]
    wait_for_start: bool,
    #[serde(default)]
    report_timing: bool,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    baseline: bool,
    #[serde(default)]
    verify_rows: u32,
    #[serde(default)]
    block: Option<u32>,
    /// Bind no mask to the draft fire. The runtime then stamps the whole
    /// extent on the block lane and reads it both ways, which is the mask
    /// this loop would build, without building it on the guest every round.
    #[serde(default)]
    no_mask: bool,
    #[serde(default)]
    margin_width: bool,
    #[serde(default)]
    min_tokens_per_round: Option<f64>,
    #[serde(default = "default_concurrency")]
    batch_concurrency: u32,
    #[serde(default = "default_true")]
    auto_width: bool,
    /// Off by default in a crowd, where a lane's clock prices the batch it waited for.
    #[serde(default)]
    priced: Option<bool>,
    #[serde(default = "default_drafter")]
    drafter: String,
    #[serde(default = "default_ngram_n")]
    ngram_n: u32,
    #[serde(default = "default_ngram_max")]
    ngram_max: u32,
    /// Submit the draft and the verify as one frame, the verify embedding the
    /// draft's picks on the device; off reads the draft back in between.
    #[serde(default = "default_true")]
    chain: bool,
}

fn default_drafter() -> String {
    "head".into()
}

fn default_ngram_n() -> u32 {
    3
}

fn default_ngram_max() -> u32 {
    7
}

/// Prompt-lookup: the tokens that followed the most recent earlier occurrence
/// of the context's last `n` tokens (falling back to shorter keys), at most
/// `max` of them. Empty when nothing repeats.
fn ngram_lookup(context: &[i32], n: u32, max: u32) -> Vec<i32> {
    let len = context.len();
    for k in (1..=n as usize).rev() {
        if len <= k {
            continue;
        }
        let key = &context[len - k..];
        // Most recent earlier occurrence: scan back from the end, skipping
        // the key's own position.
        let mut start = len - k;
        while start > 0 {
            start -= 1;
            if &context[start..start + k] == key {
                let from = start + k;
                let to = (from + max as usize).min(len);
                if to > from {
                    return context[from..to].to_vec();
                }
                break;
            }
        }
    }
    Vec::new()
}

fn default_floor(concurrency: u32) -> f64 {
    match concurrency {
        c if c >= 8 => f64::INFINITY,
        c if c > 1 => 5.0,
        _ => 0.0,
    }
}

fn default_concurrency() -> u32 {
    1
}

fn default_prompt() -> String {
    "The quick brown fox jumps over".into()
}

fn default_max_tokens() -> usize {
    64
}

fn default_true() -> bool {
    true
}

fn default_system() -> String {
    "You are a helpful benchmarking assistant.".into()
}

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    text: String,
    /// `text-completion-bench`'s envelope, for `scripts/bench/pie_bench.py`.
    num_prompt_tokens: usize,
    num_output_tokens: usize,
    token_ids: Vec<u32>,
    rounds: usize,
    drafted: usize,
    /// Proposals the target's own argmax agreed with.
    accepted: usize,
    /// `accepted / drafted`, or zero when nothing was drafted.
    acceptance_rate: f64,
    tokens_per_round: f64,
    /// The block width this run used.
    block: u32,
    /// The widths the rounds chose, in order.
    verify: Vec<u32>,
    replayed: usize,
    plain_fires: usize,
    /// Keyed by width: `0` is the draft fire, `1` the plain fire.
    prices_ms: Vec<(u32, f64)>,
    intertoken_us: Vec<u32>,
}

struct Prices {
    draft: Vec<u64>,
    /// Indexed by width; `[1]` is the plain fire.
    verify: Vec<Vec<u64>>,
}

impl Prices {
    const KEEP: usize = 5;

    fn new(block: u32) -> Self {
        Prices {
            draft: Vec::new(),
            verify: vec![Vec::new(); block as usize + 1],
        }
    }

    /// A shape's first fire compiles its plan, so its sample is dropped (a `0` placeholder).
    fn keep(bucket: &mut Vec<u64>, ns: u64) {
        if bucket.is_empty() && ns != 0 {
            bucket.push(0);
            return;
        }
        if bucket.first() == Some(&0) {
            bucket.remove(0);
        }
        bucket.push(ns);
        if bucket.len() > Self::KEEP {
            bucket.remove(0);
        }
    }

    fn median(bucket: &[u64]) -> Option<f64> {
        if bucket.is_empty() || bucket == [0] {
            return None;
        }
        let mut sorted = bucket.to_vec();
        sorted.sort_unstable();
        Some(sorted[sorted.len() / 2] as f64)
    }

    fn drafted(&mut self, ns: u64) {
        Self::keep(&mut self.draft, ns);
    }

    fn verified(&mut self, width: u32, ns: u64) {
        if let Some(bucket) = self.verify.get_mut(width as usize) {
            Self::keep(bucket, ns);
        }
    }

    fn plain(&self) -> Option<f64> {
        Self::median(&self.verify[1])
    }

    /// An unfired width prices as the nearest fired one below, so a new rung gets tried once.
    fn round(&self, width: u32) -> Option<f64> {
        let draft = Self::median(&self.draft)?;
        let verify = (1..=width as usize)
            .rev()
            .find_map(|w| Self::median(&self.verify[w]))?;
        Some(draft + verify)
    }

    fn report(&self) -> Vec<(u32, f64)> {
        let ms = |ns: f64| ns / 1e6;
        Self::median(&self.draft)
            .map(|d| (0, ms(d)))
            .into_iter()
            .chain(
                self.verify
                    .iter()
                    .enumerate()
                    .filter_map(|(w, b)| Self::median(b).map(|p| (w as u32, ms(p)))),
            )
            .collect()
    }
}

struct Gate {
    /// Tokens a round below which drafting stops. Zero never closes.
    floor: f64,
    /// `(tokens, width)`, newest last; a round that emitted its whole width is censored.
    recent: Vec<(u32, u32)>,
    last: u32,
    since: u32,
    probe_after: u32,
    /// While closed, `recent` holds only the probes fired since closing.
    closed: bool,
    reopen: usize,
    /// Whether the caller holds other requests open. A crowded gate backs its
    /// probe cadence off per probe, since each probe steps the lane out of
    /// its neighbours' batch.
    crowded: bool,
    /// Whether the verify width follows the yield. Off leaves it at `block`.
    auto: bool,
}

impl Gate {
    const WINDOW: usize = 8;
    const PROBE: u32 = 16;
    const PROBE_MAX: u32 = 128;
    const WARM: usize = 2;
    const REOPEN_ALONE: usize = 2;
    const REOPEN_CROWDED: usize = 4;
    /// Close below this fraction of a plain fire's rate; reopen only at the full rate.
    const HYSTERESIS: f64 = 0.9;
    /// A fire's price is a staircase in rows: a width between rungs pays the rung above.
    const RUNGS: [u32; 3] = [4, 8, 16];
    const WARM_TOP: u32 = 8;

    fn new(floor: f64, auto: bool, crowded: bool) -> Self {
        Gate {
            floor,
            recent: Vec::new(),
            last: 0,
            since: 0,
            probe_after: Self::PROBE,
            closed: false,
            reopen: if crowded {
                Self::REOPEN_CROWDED
            } else {
                Self::REOPEN_ALONE
            },
            crowded,
            auto,
        }
    }

    fn drafts(&mut self) -> bool {
        if self.floor <= 0.0 {
            return true;
        }
        if self.closed {
            if self.recent.len() >= self.reopen {
                if self.mean() >= self.floor {
                    self.closed = false;
                    self.since = 0;
                    self.probe_after = Self::PROBE;
                    return true;
                }
                self.recent.clear();
                self.probe_after = (self.probe_after * 2).min(Self::PROBE_MAX);
            }
            return self.probe(0).is_some();
        }
        if self.recent.len() < Self::WINDOW || self.mean() >= self.floor {
            return true;
        }
        self.closed = true;
        self.recent.clear();
        self.probe(0).is_some()
    }

    /// Strictly above the recent yield, so a saturated width steps up instead of pinning the loop.
    fn width(&self, block: u32) -> u32 {
        if !self.auto {
            return block;
        }
        if self.recent.len() < Self::WINDOW {
            return Self::WARM_TOP.min(block);
        }
        // The yield is censored at the width, so the mean alone would only ratchet down.
        let saturated = self.recent.iter().filter(|(t, w)| *w > 1 && t == w).count();
        let over = f64::from(self.last);
        let want = if saturated * 4 >= self.recent.len() * 3 {
            over
        } else {
            self.mean()
        };
        Self::RUNGS
            .iter()
            .copied()
            .find(|w| f64::from(*w) > want)
            .unwrap_or(block)
            .min(block)
    }

    fn mean(&self) -> f64 {
        self.recent.iter().map(|(t, _)| f64::from(*t)).sum::<f64>() / self.recent.len() as f64
    }

    /// What a drafting round emitted, the anchor's token included, and the
    /// width it was verified at.
    fn yielded(&mut self, tokens: u32, width: u32) {
        self.last = width;
        self.recent.push((tokens, width));
        if self.recent.len() > Self::WINDOW {
            self.recent.remove(0);
        }
    }

    fn expect(&self, width: u32) -> f64 {
        let slots = width.saturating_sub(1) as usize;
        let mut hits = vec![0u32; slots];
        let mut tries = vec![0u32; slots];
        for &(tokens, w) in &self.recent {
            let kept = tokens.saturating_sub(1) as usize;
            for i in 0..kept.min(slots) {
                hits[i] += 1;
                tries[i] += 1;
            }
            if kept + 1 < w as usize && kept < slots {
                tries[kept] += 1;
            }
        }
        // Pseudo-tries pulling each position's rate to the pooled one (an untried one takes it).
        const PRIOR: f64 = 2.0;
        let pooled = {
            let (h, t) = (hits.iter().sum::<u32>(), tries.iter().sum::<u32>());
            if t == 0 {
                1.0
            } else {
                f64::from(h) / f64::from(t)
            }
        };
        let mut chain = 1.0;
        let mut total = 1.0;
        for i in 0..slots {
            let rate = (f64::from(hits[i]) + PRIOR * pooled) / (f64::from(tries[i]) + PRIOR);
            chain *= rate;
            total += chain;
        }
        total
    }

    /// `None` is a plain fire.
    fn plan(&mut self, block: u32, prices: &Prices, pinned: Option<u32>) -> Option<u32> {
        let plain = prices.plain();
        // The window is emptied at closure, so stale rounds cannot outvote the probes.
        let enough = self.recent.len() >= if self.closed { self.reopen } else { Self::WARM };
        let measured = plain.is_some() && prices.round(1).is_some() && enough;
        if !self.drafts() {
            return None;
        }
        if !measured {
            if self.closed {
                return self.probe(pinned.unwrap_or(Self::RUNGS[0]).min(block));
            }
            return Some(pinned.unwrap_or_else(|| self.width(block)));
        }
        let plain = plain.expect("measured");
        let rungs: Vec<u32> = match pinned {
            Some(w) => vec![w.min(block)],
            None => Self::RUNGS
                .iter()
                .copied()
                .filter(|w| *w < block)
                .chain([block])
                .collect(),
        };
        let narrowest = rungs[0];
        let best = rungs
            .into_iter()
            .filter_map(|w| prices.round(w).map(|price| (w, self.expect(w) / price)))
            .max_by(|a, b| a.1.total_cmp(&b.1))?;
        let (width, rate) = best;
        // Probe at the narrowest rung: cheap, and a changed workload shows there first.
        let open = if self.closed {
            rate * plain >= 1.0
        } else {
            rate * plain >= Self::HYSTERESIS
        };
        if open {
            self.closed = false;
            self.since = 0;
            self.probe_after = Self::PROBE;
            return Some(width);
        }
        if !self.closed {
            self.closed = true;
        } else {
            self.probe_after = (self.probe_after * 2).min(Self::PROBE_MAX);
        }
        self.recent.clear();
        self.probe(narrowest)
    }

    fn probe(&mut self, width: u32) -> Option<u32> {
        if self.since >= self.probe_after {
            self.since = 0;
            if self.crowded {
                self.probe_after = (self.probe_after * 2).min(Self::PROBE_MAX);
            }
            return Some(width);
        }
        self.since += 1;
        None
    }
}

fn wide_enough(confidence: &[f32]) -> bool {
    let head = &confidence[..confidence.len().min(NARROW as usize - 1)];
    let mean = head.iter().map(|c| f64::from(*c)).sum::<f64>() / head.len().max(1) as f64;
    mean >= MARGIN_THRESHOLD
}

const NARROW: u32 = 8;

/// The mean margin at which a round is expected to reach `NARROW`.
const MARGIN_THRESHOLD: f64 = 2.0;

/// The DFlash v1 head's block, for a load that advertises none.
const BLOCK_ROWS: u32 = 16;
const MASK_TOKEN: i32 = 248_070;

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if model::mtp_depth() == 0 && input.drafter != "ngram" {
        return Err("this deployment ships no draft head".into());
    }
    let advertised = model::draft_block();
    let block = input
        .block
        .or(advertised.map(|d| d.rows))
        .unwrap_or(BLOCK_ROWS)
        .max(2);
    let mask_token = advertised.map_or(MASK_TOKEN, |d| d.mask_token as i32);
    let bind_mask = !input.no_mask && advertised.is_none_or(|d| d.bidirectional);
    // The first block row whose pick is a proposal: 1 when the anchor row
    // proposes nothing (DFlash), 0 when it proposes the next token (DSpark).
    let from = advertised.map_or(1, |d| d.proposals_from) as usize;
    // The widest verify a block feeds: the anchor and every proposal (a
    // head whose anchor row proposes, DSpark's, proposes a whole block).
    let widest = block + 1 - from.min(1) as u32;
    let pinned = (input.verify_rows != 0).then(|| input.verify_rows.clamp(2, widest));
    // The template `text-completion-bench` applies, so the two read the same prefill.
    let mut prompt: Vec<u32> = match &input.prompt_tokens {
        Some(tokens) => tokens.clone(),
        None => {
            let mut p = chat::system_user(&input.system, &input.prompt);
            p.extend(chat::cue_thinking(!input.no_think));
            p
        }
    };
    if prompt.is_empty() {
        prompt.push(0);
    }
    let stop_tokens: Vec<u32> = if input.ignore_eos {
        Vec::new()
    } else {
        chat::stop_tokens()
    };
    if input.wait_for_start {
        session::send("ready");
        let _ = session::receive().await;
    }
    let n = prompt.len() as u32;
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();

    let ws = WorkingSet::new();

    let pipe = Pipeline::new();

    let mut anchor: i32 = 0;
    for &(base, end) in &readout_spans(n, model::ReadoutKind::Logits) {
        let len = end - base;
        let toks = Channel::from(&prompt_i32[base as usize..end as usize]).named("toks_p");
        let indptr = Channel::from([0u32, len]).named("embed_indptr_p");
        let readout = Channel::from([len - 1]).named("readout_p");
        let next = Channel::new([1], dtype::i32).named("next_p");

        let fwd = ForwardPass::new();
        fwd.embed(&toks, &indptr)?;
        fwd.readout(&readout)?;
        fwd.state(&[&ws], StateGeometry::folding())?;
        fwd.epilogue(move || {
            next.put(&reshape(reduce_argmax(intrinsics::logits()), [1]));
        });
        fwd.submit(&pipe).context("prefill submit")?;
        anchor = next
            .take_host::<Vec<i32>>()
            .await
            .context("prefill readback")?[0];
    }

    if input.report_timing {
        session::send("t0");
    }

    // ── THE LOOP ────────────────────────────────────────────────────────
    // Slots `0..held` are written; `anchor` is the token at slot `held`, not yet written.
    let mut held = n;
    let mut generated: Vec<u32> = vec![anchor as u32];
    let mut emitted_ns: Vec<u64> = vec![monotonic_now_ns()];
    let mut rounds = 0usize;
    let mut drafted = 0usize;
    let mut accepted = 0usize;
    let mut replayed = 0usize;
    let mut survivors: u32 = 0;
    let mut widths: Vec<u32> = Vec::new();
    let crowded = input.batch_concurrency > 1;
    let floor = input
        .min_tokens_per_round
        .unwrap_or_else(|| default_floor(input.batch_concurrency));
    let priced = input.priced.unwrap_or(!crowded);
    let mut gate = Gate::new(floor, input.auto_width, crowded);
    let mut prices = Prices::new(block);
    let mut plain_fires: usize = 0;
    // Drafting rounds since the plain fire was last priced, and plain fires
    // still owed to the current re-pricing; see `REPRICE`.
    let mut since_priced: u32 = 0;
    let mut owed_plain: u32 = 0;
    // Re-pricing takes two plain fires: the first folds the survivors, the second is priced.
    const REPRICE: u32 = 32;

    // A round commits several tokens at once, so the stop is read off the
    // committed run after the fact rather than one token at a time.
    while generated.len() < input.max_tokens && !generated.iter().any(|t| stop_tokens.contains(t)) {
        if input.baseline {
            // One token a fire, the road the speedup is measured against.
            let toks = Channel::from([anchor]).named("toks_b");
            let indptr = Channel::from([0u32, 1]).named("embed_indptr_b");
            let next = Channel::new([1], dtype::i32)
                .capacity(channel_capacity() as u32)
                .named("next_b");

            let fwd = ForwardPass::new();
            fwd.embed(&toks, &indptr)?;
            fwd.state(&[&ws], StateGeometry::folding())?;
            fwd.epilogue(move || {
                next.put(&reshape(reduce_argmax(intrinsics::logits()), [1]));
            });
            let fired_at = monotonic_now_ns();
            fwd.submit(&pipe).context("baseline submit")?;
            anchor = next
                .take_host::<Vec<i32>>()
                .await
                .context("baseline readback")?[0];
            prices.verified(1, monotonic_now_ns().saturating_sub(fired_at));
            held += 1;
            generated.push(anchor as u32);
            emitted_ns.push(monotonic_now_ns());
            continue;
        }

        // Priced, the first two fires are plain: they price the one-row baseline the rungs face.
        let ngram = input.drafter == "ngram";
        let ngram_props: Vec<i32> = if ngram {
            let mut context: Vec<i32> = prompt_i32.clone();
            context.extend(generated.iter().map(|&t| t as i32));
            ngram_lookup(&context, input.ngram_n, input.ngram_max)
        } else {
            Vec::new()
        };
        let planned = if floor.is_infinite() {
            None
        } else if ngram {
            // A match drafts as many rows as it found; none is a plain fire.
            (!ngram_props.is_empty()).then(|| (ngram_props.len() as u32 + 1).min(block))
        } else if priced {
            if since_priced >= REPRICE {
                since_priced = 0;
                owed_plain = 2;
            }
            if prices.plain().is_none() || owed_plain > 0 {
                owed_plain = owed_plain.saturating_sub(1);
                None
            } else {
                since_priced += 1;
                gate.plan(block, &prices, pinned)
            }
        } else {
            gate.drafts().then(|| {
                if gate.closed {
                    pinned.unwrap_or(Gate::RUNGS[0]).min(widest)
                } else {
                    pinned.unwrap_or_else(|| gate.width(block))
                }
            })
        };
        let mut verify = planned.unwrap_or(1);

        let drafting = planned.is_some();
        let fold_len = Channel::from([survivors]).named("fold_len_v");
        let mut proposals_owned: Vec<i32> = Vec::new();
        // The verify's answer when the draft and the verify ran as one frame:
        // the target's picks, and how many proposals they agreed with.
        let mut judged: Option<(Vec<i32>, usize)> = None;
        let chained = input.chain && !input.margin_width && frame_size() >= 2;
        if !drafting {
            verify = 1;
            plain_fires += 1;
        } else if ngram {
            // No draft fire: the proposals are the lookup's.
            proposals_owned = ngram_props[..verify as usize - 1].to_vec();
        } else if chained {
            let started = monotonic_now_ns();
            let (truth, kept) = chained_round(&ChainedRound {
                pipe: &pipe,
                ws: &ws,
                anchor,
                held,
                block,
                verify,
                from: from as u32,
                mask_token,
                bind_mask,
                fold_len: &fold_len,
            })
            .await?;
            // The round is one price: the verify's bucket carries it whole.
            prices.drafted(1);
            prices.verified(verify, monotonic_now_ns().saturating_sub(started));
            judged = Some((truth, kept));
        } else {
            let mut ids = vec![mask_token; block as usize];
            ids[0] = anchor;
            let toks = Channel::from(ids.as_slice()).named("toks_d");
            let indptr = Channel::from([0u32, block]).named("embed_indptr_d");
            let slots = Channel::from_iter(held..held + block).named("slots_d");
            let kv_len = Channel::from([held + block]).named("kv_len_d");
            // The drafter attends bidirectionally over the block.
            let visible = vec![true; (block * (held + block)) as usize];
            let mask = Channel::from_shaped([block, held + block], visible).named("mask_d");
            // A DFlash2 head's selector picks one row per block row, so the
            // readout covers the block; the proposals are cut to the verify
            // width afterwards.
            let shown = block;
            let readout = Channel::from_iter(0..shown).named("readout_d");
            let out = Channel::new([shown], dtype::i32)
                .capacity(channel_capacity() as u32)
                .named("drafts_d");
            let conf = Channel::new([shown * 2], dtype::f32)
                .capacity(channel_capacity() as u32)
                .named("conf_d");
            let want_margin = input.margin_width;

            let fwd = ForwardPass::new();
            fwd.set_drafting_block(true)
                .map_err(|why| format!("stating the draft block: {why}"))?;
            fwd.embed(&toks, &indptr)?;
            fwd.readout(&readout)?;
            // Folds nothing: no trunk row runs here, and the verify writes these slots again.
            fwd.state(
                &[&ws],
                StateGeometry {
                    slots: Some(&slots),
                    kv_len: Some(&kv_len),
                    mask: if bind_mask { Some(&mask) } else { None },
                    ..StateGeometry::default()
                },
            )?;
            {
                let out = out.clone();
                let conf = conf.clone();
                fwd.epilogue(move || {
                    out.put(&reshape(intrinsics::mtp_drafts(shown), [shown]));
                    if want_margin {
                        let (value, _) = top_k(intrinsics::logits(), 2);
                        conf.put(&reshape(value, [shown * 2]));
                    }
                });
            }
            let drafted_at = monotonic_now_ns();
            fwd.submit(&pipe).context("draft submit")?;
            // DFlash's anchor row proposes nothing; DSpark's proposes the next token.
            let picks = out
                .take_host::<Vec<i32>>()
                .await
                .context("draft readback")?;
            prices.drafted(monotonic_now_ns().saturating_sub(drafted_at));
            proposals_owned = picks[from..shown as usize].to_vec();
            if want_margin {
                let value = conf
                    .take_host::<Vec<f32>>()
                    .await
                    .context("margin readback")?;
                // Row `r` occupies `[2r, 2r + 1]`: the best and its runner-up.
                let margin: Vec<f32> = (from..shown as usize)
                    .map(|r| value[2 * r] - value[2 * r + 1])
                    .collect();
                if pinned.is_none() {
                    verify = if wide_enough(&margin) { block } else { NARROW };
                }
            }
        }
        let proposals = proposals_owned.as_slice();
        widths.push(verify);

        let (truth, kept) = match judged {
            Some(judged) => judged,
            None => {
                verify_round(
                    &pipe,
                    &ws,
                    anchor,
                    proposals,
                    verify,
                    held,
                    drafting,
                    survivors,
                    &fold_len,
                    &mut prices,
                )
                .await?
            }
        };
        if drafting {
            rounds += 1;
            drafted += verify as usize - 1;
            accepted += kept;
            gate.yielded(kept as u32 + 1, verify);
        }
        replayed += survivors as usize;
        // A drafting round leaves its accepted prefix buffered for the next
        // fire to fold; the rejected tail is written over.
        survivors = if drafting { kept as u32 + 1 } else { 0 };
        let round_ns = monotonic_now_ns();
        // The kept proposals are the target's own picks up to the break.
        for tok in truth[..kept].iter() {
            generated.push(*tok as u32);
            emitted_ns.push(round_ns);
        }
        anchor = truth[kept];
        generated.push(anchor as u32);
        emitted_ns.push(round_ns);
        held += kept as u32 + 1;
    }
    finish(Finish {
        input: &input,
        prompt: &prompt,
        stop_tokens: &stop_tokens,
        generated,
        emitted_ns,
        rounds,
        drafted,
        accepted,
        block,
        widths,
        replayed,
        plain_fires,
        prices: &prices,
    })
}

/// One trunk fire over `[anchor, proposals]` from the host: the target's
/// pick per row, and the proposals it agreed with.
#[allow(clippy::too_many_arguments)]
async fn verify_round(
    pipe: &Pipeline,
    ws: &WorkingSet,
    anchor: i32,
    proposals: &[i32],
    verify: u32,
    held: u32,
    drafting: bool,
    survivors: u32,
    fold_len: &Channel,
    prices: &mut Prices,
) -> Result<(Vec<i32>, usize)> {
    // ── the verify: ONE trunk fire over `[anchor, proposals]` ────────
    let mut fed = vec![anchor];
    fed.extend_from_slice(&proposals[..verify as usize - 1]);
    let toks = Channel::from(fed.as_slice()).named("toks_v");
    let indptr = Channel::from([0u32, verify]).named("embed_indptr_v");
    let slots = Channel::from_iter(held..held + verify).named("slots_v");
    let readout = Channel::from_iter(0..verify).named("readout_v");
    let truth = Channel::new([verify], dtype::i32)
        .capacity(channel_capacity() as u32)
        .named("truth_v");

    let fwd = ForwardPass::new();
    fwd.embed(&toks, &indptr)?;
    fwd.readout(&readout)?;
    fwd.state(
        &[ws],
        StateGeometry {
            slots: Some(&slots),
            fold: if drafting {
                Fold::Count(fold_len)
            } else {
                Fold::All
            },
            ..StateGeometry::default()
        },
    )?;
    {
        let truth = truth.clone();
        fwd.epilogue(move || {
            truth.put(&reshape(reduce_argmax(intrinsics::logits()), [verify]));
        });
    }
    let verified_at = monotonic_now_ns();
    fwd.submit(pipe).context("verify submit")?;
    let truth = truth
        .take_host::<Vec<i32>>()
        .await
        .context("verify readback")?;
    // A plain fire is priced only with nothing buffered behind it, the baseline's geometry.
    if drafting || survivors == 0 {
        prices.verified(verify, monotonic_now_ns().saturating_sub(verified_at));
    }

    // ── what the target kept ────────────────────────────────────────
    // Row `i` of the verify predicts position `held + i + 1`, which is
    // what proposal `i` claims — so the prefix is read off one zip.
    let kept = proposals[..verify as usize - 1]
        .iter()
        .zip(&truth)
        .take_while(|(p, t)| p == t)
        .count();
    Ok((truth, kept))
}

/// What a chained round reads off the loop.
struct ChainedRound<'a> {
    pipe: &'a Pipeline,
    ws: &'a WorkingSet,
    anchor: i32,
    held: u32,
    block: u32,
    verify: u32,
    from: u32,
    mask_token: i32,
    bind_mask: bool,
    fold_len: &'a Channel,
}

/// The draft and the verify as ONE frame. The draft's epilogue lays out the
/// verify's window, `[anchor, d[from], .., d[from + verify - 2]]`, on a
/// channel the verify embeds, so the device runs the verify right behind the
/// draft; the verify's epilogue answers its picks and the length of the
/// prefix the window's proposals agreed with, the one read of the round.
async fn chained_round(round: &ChainedRound<'_>) -> Result<(Vec<i32>, usize)> {
    let &ChainedRound {
        pipe,
        ws,
        anchor,
        held,
        block,
        verify,
        from,
        mask_token,
        bind_mask,
        fold_len,
    } = round;

    // ── the draft: `[anchor, MASK x block-1]`, as the unchained round ────
    let mut ids = vec![mask_token; block as usize];
    ids[0] = anchor;
    let toks = Channel::from(ids.as_slice()).named("toks_d");
    let indptr = Channel::from([0u32, block]).named("embed_indptr_d");
    let slots = Channel::from_iter(held..held + block).named("slots_d");
    let kv_len = Channel::from([held + block]).named("kv_len_d");
    let visible = vec![true; (block * (held + block)) as usize];
    let mask = Channel::from_shaped([block, held + block], visible).named("mask_d");
    let readout = Channel::from_iter(0..block).named("readout_d");
    let first = Channel::from([anchor]).named("anchor_d");
    // The verify's window: put by the draft, embedded by the verify.
    let window = Channel::new([verify], dtype::i32).named("window_v");

    let draft = ForwardPass::new();
    draft
        .set_drafting_block(true)
        .map_err(|why| format!("stating the draft block: {why}"))?;
    draft.embed(&toks, &indptr)?;
    draft.readout(&readout)?;
    draft.state(
        &[ws],
        StateGeometry {
            slots: Some(&slots),
            kv_len: Some(&kv_len),
            mask: if bind_mask { Some(&mask) } else { None },
            ..StateGeometry::default()
        },
    )?;
    draft.epilogue(move || {
        let a = first.take(); // [1] i32
        let picks = reshape(intrinsics::mtp_drafts(block), [block]); // [block] i32
        let i = iota(verify);
        let one = Tensor::constant(1u32);
        // Row i > 0 of the window is proposal i - 1, the pick of block row
        // from + i - 1.
        let at = min_elem(
            max_elem(&i, &one) - &one + Tensor::constant(from),
            Tensor::constant(block - 1),
        );
        window.put(&select(
            eq(&i, Tensor::constant(0u32)),
            broadcast(&a, [verify]),
            gather(&picks, &at),
        ));
    });

    // ── the verify: ONE trunk fire over the window ───────────────────────
    let indptr = Channel::from([0u32, verify]).named("embed_indptr_v");
    let slots = Channel::from_iter(held..held + verify).named("slots_v");
    let readout = Channel::from_iter(0..verify).named("readout_v");
    // `[t[0..verify], kept]`: the target's picks, then the agreeing prefix.
    let judged = Channel::new([verify + 1], dtype::i32)
        .capacity(channel_capacity() as u32)
        .named("judged_v");

    let check = ForwardPass::new();
    check.embed(&window, &indptr)?;
    check.readout(&readout)?;
    check.state(
        &[ws],
        StateGeometry {
            slots: Some(&slots),
            fold: Fold::Count(fold_len),
            ..StateGeometry::default()
        },
    )?;
    {
        let judged = judged.clone();
        check.epilogue(move || {
            let window = window.take(); // [verify] i32
            let truth = reshape(reduce_argmax(intrinsics::logits()), [verify]); // [verify] i32
            let k = verify - 1;
            let one = Tensor::constant(1u32);
            // Row i of the verify predicts what proposal i (window row i + 1)
            // claims.
            let proposed = gather(&window, iota(k) + &one);
            let said = gather(&truth, iota(k));
            let hit = cast(eq(&proposed, &said), dtype::u32);
            let kept = cast(reshape(reduce_sum(cumprod(&hit)), [1]), dtype::i32);
            let i = iota(verify + 1);
            let last = Tensor::constant(verify - 1);
            judged.put(&select(
                lt(&i, Tensor::constant(verify)),
                gather(&truth, min_elem(&i, &last)),
                broadcast(&kept, [verify + 1]),
            ));
        });
    }
    submit_frame(pipe, &[Some(&draft), Some(&check)]).context("draft and verify submit")?;
    let mut truth = judged
        .take_host::<Vec<i32>>()
        .await
        .context("verify readback")?;
    let kept = truth.pop().map_or(0, |kept| kept.max(0) as usize);
    Ok((truth, kept.min(verify as usize - 1)))
}

/// What the loop leaves for the answer.
struct Finish<'a> {
    input: &'a Input,
    prompt: &'a [u32],
    stop_tokens: &'a [u32],
    generated: Vec<u32>,
    emitted_ns: Vec<u64>,
    rounds: usize,
    drafted: usize,
    accepted: usize,
    block: u32,
    widths: Vec<u32>,
    replayed: usize,
    plain_fires: usize,
    prices: &'a Prices,
}

fn finish(done: Finish<'_>) -> Result<Output> {
    let Finish {
        input,
        prompt,
        stop_tokens,
        mut generated,
        mut emitted_ns,
        rounds,
        drafted,
        accepted,
        block,
        widths,
        replayed,
        plain_fires,
        prices,
    } = done;
    // A stop token ends the text; anything the last block committed past it
    // is not the answer.
    if let Some(at) = generated.iter().position(|t| stop_tokens.contains(t)) {
        generated.truncate(at + 1);
    }
    generated.truncate(input.max_tokens);
    let count = generated.len();
    emitted_ns.truncate(count);
    let intertoken_us: Vec<u32> = if input.report_timing {
        emitted_ns
            .windows(2)
            .map(|w| (w[1].saturating_sub(w[0]) / 1_000) as u32)
            .collect()
    } else {
        Vec::new()
    };
    Ok(Output {
        sampler: "dflash-speculative-bench",
        text: model::decode(&generated)?,
        num_prompt_tokens: prompt.len(),
        num_output_tokens: count,
        token_ids: generated,
        rounds,
        drafted,
        accepted,
        acceptance_rate: if drafted == 0 {
            0.0
        } else {
            accepted as f64 / drafted as f64
        },
        tokens_per_round: if rounds == 0 {
            0.0
        } else {
            (accepted + rounds) as f64 / rounds as f64
        },
        block,
        verify: widths,
        replayed,
        plain_fires,
        prices_ms: prices.report(),
        intertoken_us,
    })
}
