//! The generation loop: a one-pass prefill, then a device-carried decode pass
//! whose epilogue samples the next token and feeds it straight back into the
//! channel `embed` reads. The host drains a mirror of the sampled tokens
//! behind the runtime's run-ahead window.
//!
//! The sampler is one ETA epilogue: the OpenAI penalties (presence,
//! frequency, and the multiplicative repetition penalty) over a vocabulary
//! histogram carried on the device, an optional grammar mask, temperature,
//! then min-p, top-k and top-p truncation and a Gumbel-max draw. Temperature
//! zero is exact greedy. Asked for logprobs, the epilogue also lands the
//! sampled token's log-probability under the model's raw logits and the
//! `top_logprobs` likeliest tokens', as vLLM's `raw_logprobs` reports them.
//!
//! Asked for prompt logprobs, the prefill reads out every prompt row but the
//! last and scores row `i` against token `i + 1`: its logprob, its rank and
//! the likeliest alternatives, in passes of at most the runtime's readout
//! rows. A pass that reads out rows is not mapped in from the prefix cache,
//! so the prompt is computed whole: the price of asking.
//!
//! With a grammar the loop is depth-1 — the mask for fire k+1 exists only
//! once fire k's token has advanced the matcher — so constrained generation
//! submits one fire at a time; unconstrained generation runs ahead. A
//! grammar that holds only after a trigger ([`crate::trigger`]) is a
//! guide too, so its free stretches go a fire at a time as well, unmasked.

use std::ops::ControlFlow;

use inferlet::eta::prelude::*;
use inferlet::grammar::Matcher;
use inferlet::mask::unpack_mask;
use inferlet::media;

use crate::prompt::Prompt;

/// Sampling parameters. The defaults are the OpenAI defaults: temperature 1,
/// no truncation, no penalties.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    /// Draft with the model's block drafter where it pays (greedy, or
    /// sampled under `speculative_sampled`; alone); on unless the request
    /// turns it off.
    pub speculative: bool,
    /// Draft above temperature zero too, verifying by speculative
    /// sampling's acceptance rule (temperature alone, no truncation). Off
    /// unless the request asks for speculation outright: which rounds draft
    /// follows the lane's clock, and with it which noise a seeded request's
    /// tokens draw, so a seed no longer pins the text.
    pub speculative_sampled: bool,
    pub temperature: f32,
    pub top_p: f32,
    /// 0 means off.
    pub top_k: u32,
    pub seed: u32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    /// 1.0 means off.
    pub repetition_penalty: f32,
    /// Keep tokens at least this fraction as likely as the likeliest after
    /// temperature; 0 means off.
    pub min_p: f32,
    /// Land each sampled token's logprob and this many likeliest
    /// alternatives' (`Some(0)`: the sampled token's alone).
    pub logprobs: Option<u32>,
    /// Score every prompt token after the first under the rows before it,
    /// with its rank and this many likeliest alternatives (vLLM's
    /// `prompt_logprobs`); a text prompt only.
    pub prompt_logprobs: Option<u32>,
}

/// The most alternatives a step reports, as OpenAI caps `top_logprobs`.
pub const MAX_TOP_LOGPROBS: u32 = 20;

/// A sampled token's log-probability under the model's raw logits, and the
/// likeliest tokens' with theirs, most likely first.
#[derive(Clone, Debug, PartialEq)]
pub struct Logprob {
    pub logprob: f32,
    pub top: Vec<(u32, f32)>,
    /// Where the token ranks among the vocabulary, 1 for the likeliest:
    /// counted for prompt tokens, which were not sampled.
    pub rank: Option<u32>,
}

/// What constrains generation fire by fire: a grammar's matcher, or a
/// grammar that holds only part of the way ([`crate::trigger`]).
pub trait Guide {
    /// The tokens allowed next as a packed bitmask, empty for any.
    fn mask(&mut self) -> Vec<u32>;
    /// Advance past a sampled `token`.
    fn accept(&mut self, token: u32) -> Result<()>;
    /// The grammar ended the turn.
    fn is_terminated(&self) -> bool;
}

impl Guide for Matcher {
    fn mask(&mut self) -> Vec<u32> {
        Matcher::mask(self)
    }

    fn accept(&mut self, token: u32) -> Result<()> {
        self.accept_tokens(&[token])
            .map_err(|e| format!("token {token} rejected: {e:?}"))
    }

    fn is_terminated(&self) -> bool {
        Matcher::is_terminated(self)
    }
}

/// What [`generate`] did: how many tokens it sampled, how many of the
/// prompt's were mapped in from the prefix cache, and, where asked, the
/// logprob of every prompt token after the first.
#[derive(Clone, Debug, Default)]
pub struct Generated {
    pub sampled: usize,
    pub cached: usize,
    pub prompt_logprobs: Vec<Logprob>,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            speculative: true,
            speculative_sampled: false,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0x51ed,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            min_p: 0.0,
            logprobs: None,
            prompt_logprobs: None,
        }
    }
}

impl Sampling {
    /// The API's 64-bit seed folded into the sampler's 32-bit state.
    pub fn set_seed(&mut self, seed: i64) {
        self.seed = (seed as u64 as u32) ^ ((seed as u64 >> 32) as u32);
    }

    /// Whether temperature alone shapes the distribution sampled from.
    fn untruncated(&self) -> bool {
        self.top_k == 0 && self.top_p == 1.0 && self.min_p == 0.0
    }

    fn penalized(&self) -> bool {
        self.presence_penalty != 0.0
            || self.frequency_penalty != 0.0
            || self.repetition_penalty != 1.0
    }

    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err("temperature must be a finite number >= 0".into());
        }
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err("top_p must be in (0, 1]".into());
        }
        if !self.presence_penalty.is_finite() || !self.frequency_penalty.is_finite() {
            return Err("penalties must be finite".into());
        }
        if !(self.repetition_penalty.is_finite() && self.repetition_penalty > 0.0) {
            return Err("repetition_penalty must be > 0".into());
        }
        if !(0.0..=1.0).contains(&self.min_p) {
            return Err("min_p must be in [0, 1]".into());
        }
        if [self.logprobs, self.prompt_logprobs]
            .iter()
            .any(|k| k.is_some_and(|k| k > MAX_TOP_LOGPROBS))
        {
            return Err(format!(
                "logprobs are at most {MAX_TOP_LOGPROBS} alternatives a token"
            ));
        }
        Ok(())
    }
}

/// The sampler's knob values as scalars. Which knobs exist is the program's
/// shape; their values ride in channels, so requests that differ only in
/// value share one program and fly as one batch.
struct Knobs {
    /// A program constant: a channel-fed divisor costs the fused sampler
    /// ~20% of a decode step, more than mixed temperatures cost by splitting.
    temperature: Option<f32>,
    top_p: Option<Tensor>,
    top_k: Option<Tensor>,
    min_p: Option<Tensor>,
    repetition: Option<Tensor>,
    frequency: Option<Tensor>,
    presence: Option<Tensor>,
    /// A program constant: how many alternatives a step lands, if any.
    logprobs: Option<u32>,
}

/// The channels a pass's knobs are seeded in.
struct KnobChannels {
    temperature: Option<f32>,
    top_p: Option<Channel>,
    top_k: Option<Channel>,
    min_p: Option<Channel>,
    repetition: Option<Channel>,
    frequency: Option<Channel>,
    presence: Option<Channel>,
    logprobs: Option<u32>,
}

impl KnobChannels {
    fn new(s: &Sampling, tag: &str) -> Self {
        let f = |on: bool, v: f32, name: &str| {
            on.then(|| Channel::from([v]).named(&format!("{name}{tag}")))
        };
        let sampled = s.temperature > 0.0;
        let penalized = s.penalized();
        Self {
            temperature: sampled.then(|| s.temperature.max(1e-4)),
            top_p: f(sampled && s.top_p < 1.0, s.top_p, "top_p"),
            top_k: (sampled && s.top_k > 0)
                .then(|| Channel::from([s.top_k]).named(&format!("top_k{tag}"))),
            min_p: f(sampled && s.min_p > 0.0, s.min_p, "min_p"),
            repetition: f(
                penalized && s.repetition_penalty != 1.0,
                s.repetition_penalty,
                "rep_pen",
            ),
            frequency: f(
                penalized && s.frequency_penalty != 0.0,
                s.frequency_penalty,
                "freq_pen",
            ),
            presence: f(
                penalized && s.presence_penalty != 0.0,
                s.presence_penalty,
                "pres_pen",
            ),
            logprobs: s.logprobs,
        }
    }

    /// Read without consuming: the seeded values stay for every fire.
    fn read(&self) -> Knobs {
        let t = |c: &Option<Channel>| c.as_ref().map(|c| reshape(c.read(), []));
        Knobs {
            temperature: self.temperature,
            top_p: t(&self.top_p),
            top_k: t(&self.top_k),
            min_p: t(&self.min_p),
            repetition: t(&self.repetition),
            frequency: t(&self.frequency),
            presence: t(&self.presence),
            logprobs: self.logprobs,
        }
    }
}

/// What a step lands beside its token when asked for logprobs: the sampled
/// token's logprob then the alternatives' (`[1 + top]` f32), and the
/// alternatives' ids (`[top]` i32, absent for none).
struct Landed {
    logprobs: Tensor,
    ids: Option<Tensor>,
}

/// The sampled token as a rank-0 i32, the next output-token histogram, and
/// the step's logprobs where the knobs ask. `histogram` is `(counts,
/// present)`: how often each token was generated, and whether it was in the
/// prompt.
fn sample(
    vocab: u32,
    k: &Knobs,
    r: &Tensor,
    histogram: Option<(&Tensor, &Tensor)>,
    mask: Option<&Tensor>,
) -> (Tensor, Option<Tensor>, Option<Landed>) {
    let mut logits = reshape(intrinsics::logits(), [vocab]);
    let raw = k.logprobs.map(|_| logits.clone());
    if let Some((counts, present)) = histogram {
        let zero = broadcast(0.0f32, [vocab]);
        let seen_out = gt(counts, &zero);
        let seen_prompt = gt(present, broadcast(0.5f32, [vocab]));
        let seen = or(&seen_out, &seen_prompt);
        if let Some(rep) = &k.repetition {
            let rp = broadcast(rep, [vocab]);
            let positive = gt(&logits, &zero);
            let repenalized = select(&positive, &logits / &rp, &logits * &rp);
            logits = select(&seen, &repenalized, &logits);
        }
        if let Some(freq) = &k.frequency {
            logits = &logits - broadcast(freq, [vocab]) * counts;
        }
        if let Some(pres) = &k.presence {
            logits = &logits - broadcast(pres, [vocab]) * cast(&seen_out, dtype::f32);
        }
    }
    if let Some(mask) = mask {
        logits = mask_apply(&logits, mask);
    }
    let token = match k.temperature {
        None => reduce_argmax(&logits),
        Some(temperature) => {
            let scaled = &logits / temperature;
            let mut keep: Option<Tensor> = None;
            if k.top_k.is_some() || k.top_p.is_some() || k.min_p.is_some() {
                let probs = softmax(&scaled);
                if let Some(min_p) = &k.min_p {
                    let floor = reduce_max(&probs) * min_p;
                    keep = Some(ge(&probs, broadcast(&floor, [vocab])));
                }
                if let Some(top_k) = &k.top_k {
                    let ranked = pivot_threshold(&probs, rank_le(top_k));
                    keep = Some(match keep {
                        Some(k) => and(&k, &ranked),
                        None => ranked,
                    });
                }
                if let Some(top_p) = &k.top_p {
                    let nucleus = pivot_threshold(&probs, cummass_le(top_p));
                    keep = Some(match keep {
                        Some(k) => and(&k, &nucleus),
                        None => nucleus,
                    });
                }
            }
            let truncated = match keep {
                Some(keep) => mask_apply(&scaled, keep),
                None => scaled,
            };
            gumbel_max(&truncated, r)
        }
    };
    let next = histogram.map(|(counts, _)| scatter_add(counts, &token, 1.0f32));
    let landed = raw.zip(k.logprobs).map(|(raw, top)| {
        let lp = log_softmax(&raw);
        let chosen = reshape(scalar_gather(&lp, reshape(&token, [1])), [1]);
        if top == 0 {
            return Landed {
                logprobs: chosen,
                ids: None,
            };
        }
        let (values, ids) = top_k(&lp, top);
        Landed {
            logprobs: concat_f32(&chosen, &values, top),
            ids: Some(cast(&ids, dtype::i32)),
        }
    });
    (token, next, landed)
}

/// `[1]` and `[n]` f32 rows as one `[1 + n]` row.
fn concat_f32(head: &Tensor, tail: &Tensor, n: u32) -> Tensor {
    let at = iota(n + 1);
    let first = eq(&at, broadcast(0u32, [n + 1]));
    let shifted = gather(tail, select(&first, broadcast(0u32, [n + 1]), &at - 1u32));
    select(&first, broadcast(reshape(head, []), [n + 1]), &shifted)
}

/// The host's read of a step's landed logprobs.
fn logprob_of(values: &[f32], ids: &[i32]) -> Logprob {
    Logprob {
        logprob: values.first().copied().unwrap_or(f32::NEG_INFINITY),
        top: ids
            .iter()
            .zip(values.iter().skip(1))
            .map(|(&id, &lp)| (id as u32, lp))
            .collect(),
        rank: None,
    }
}

/// The channels a pass lands its logprobs in, where asked.
fn logprob_channels(
    logprobs: Option<u32>,
    tag: &str,
    capacity: u32,
) -> Option<(Channel, Option<Channel>)> {
    logprobs.map(|top| {
        (
            Channel::new([1 + top], dtype::f32)
                .capacity(capacity)
                .named(&format!("lp{tag}")),
            (top > 0).then(|| {
                Channel::new([top], dtype::i32)
                    .capacity(capacity)
                    .named(&format!("lp_ids{tag}"))
            }),
        )
    })
}

/// A step's landed logprobs put in its channels, where asked.
fn land(landed: Option<Landed>, channels: Option<(Channel, Option<Channel>)>) {
    if let (Some(landed), Some((values, ids))) = (landed, channels) {
        values.put(&landed.logprobs);
        if let (Some(landed), Some(ids)) = (landed.ids, ids) {
            ids.put(&landed);
        }
    }
}

/// A step's logprobs off the host side of its channels.
async fn take_logprob(channels: Option<&(Channel, Option<Channel>)>) -> Result<Option<Logprob>> {
    let Some((values, ids)) = channels else {
        return Ok(None);
    };
    let values = values.take_host::<Vec<f32>>().await?;
    let ids = match ids {
        Some(ids) => ids.take_host::<Vec<i32>>().await?,
        None => Vec::new(),
    };
    Ok(Some(logprob_of(&values, &ids)))
}

/// The prompt's own logprobs: rows `0..prompt.len() - 1` prefilled and read
/// out, row `i` scored against token `i + 1` — its logprob under the raw
/// logits, its rank, and the `top` likeliest tokens' — in passes of at most
/// the runtime's readout rows. The last prompt token is left for the pass
/// that samples after it.
async fn score_prompt(
    ws: &WorkingSet,
    pipe: &Pipeline,
    prompt: &[i32],
    top: u32,
    ends: &[u32],
) -> Result<Vec<Logprob>> {
    let vocab = model::output_vocab_size();
    let rows = prompt.len().saturating_sub(1) as u32;
    let mut scored = Vec::with_capacity(rows as usize);
    for (at, &(base, end)) in readout_spans(rows, model::ReadoutKind::Logits)
        .iter()
        .enumerate()
    {
        let len = end - base;
        let (base, end) = (base as usize, end as usize);
        let toks = Channel::from(&prompt[base..end]).named(&format!("toks_s{at}"));
        let embed_indptr = Channel::from([0u32, len]).named(&format!("embed_indptr_s{at}"));
        let readout = Channel::from_iter(0..len).named(&format!("readout_s{at}"));
        let targets = Channel::from(&prompt[base + 1..end + 1]).named(&format!("targets_s{at}"));
        let chosen_out = Channel::new([len], dtype::f32).named(&format!("chosen_s{at}"));
        let rank_out = Channel::new([len], dtype::f32).named(&format!("rank_s{at}"));
        let top_out = (top > 0).then(|| {
            (
                Channel::new([len, top], dtype::f32).named(&format!("top_s{at}")),
                Channel::new([len, top], dtype::i32).named(&format!("top_ids_s{at}")),
            )
        });

        let fwd = ForwardPass::new();
        fwd.embed(&toks, &embed_indptr)?;
        fwd.state(&[ws], StateGeometry::folding())?;
        fwd.readout(&readout)?;
        fwd.epilogue(move || {
            let lp = log_softmax(reshape(intrinsics::logits(), [len, vocab]));
            let chosen = scalar_gather(&lp, targets.take());
            // Rank 1 is the likeliest: one more than the tokens likelier.
            let likelier = gt(&lp, broadcast(reshape(&chosen, [len, 1]), [len, vocab]));
            rank_out.put(reduce_sum(cast(&likelier, dtype::f32)) + 1.0f32);
            chosen_out.put(&chosen);
            if let Some((values, ids)) = top_out {
                let (top_values, top_ids) = top_k(&lp, top);
                values.put(&top_values);
                ids.put(cast(&top_ids, dtype::i32));
            }
        });
        fwd.checkpoints(ends)?;
        fwd.submit(pipe)
            .with_context(|| format!("prompt scoring submit @{base}"))?;
        let chosen = chosen_out.take_host::<Vec<f32>>().await?;
        let ranks = rank_out.take_host::<Vec<f32>>().await?;
        let (values, ids) = match top_out {
            Some((values, ids)) => (
                values.take_host::<Vec<f32>>().await?,
                ids.take_host::<Vec<i32>>().await?,
            ),
            None => (Vec::new(), Vec::new()),
        };
        scored.extend(prompt_rows(&chosen, &ranks, &values, &ids, top as usize));
    }
    Ok(scored)
}

/// The host's read of a scoring pass: each row's chosen logprob and rank,
/// and its `top` alternatives off the flat `[rows, top]` values and ids.
fn prompt_rows(
    chosen: &[f32],
    ranks: &[f32],
    values: &[f32],
    ids: &[i32],
    top: usize,
) -> Vec<Logprob> {
    chosen
        .iter()
        .zip(ranks)
        .enumerate()
        .map(|(row, (&logprob, &rank))| {
            let span = row * top..(row + 1) * top;
            Logprob {
                logprob,
                top: ids
                    .get(span.clone())
                    .unwrap_or_default()
                    .iter()
                    .zip(values.get(span).unwrap_or_default())
                    .map(|(&id, &lp)| (id as u32, lp))
                    .collect(),
                rank: Some(rank as u32),
            }
        })
        .collect()
}

/// Generate up to `max_tokens` after `prompt`, prefilled in one pass with
/// its pictures. Every sampled token — the stop token included, if one
/// comes — goes to `on_token`, which ends generation by returning `Break`. Where `sampling` asks for prompt
/// logprobs, `on_prompt` has them ([`Generated::prompt_logprobs`]) before
/// the first token comes.
///
/// With a `guide`, every fire's logits are masked to what it allows next,
/// and generation ends when it terminates.
pub async fn generate(
    prompt: &Prompt,
    sampling: Sampling,
    max_tokens: usize,
    mut guide: Option<&mut dyn Guide>,
    on_prompt: &mut dyn FnMut(&[Logprob]),
    on_token: &mut dyn FnMut(u32, Option<Logprob>) -> ControlFlow<()>,
) -> Result<Generated> {
    sampling.validate()?;
    let images = &prompt.images;
    let ends = &prompt.ends;
    let prompt = &prompt.tokens[..];
    if max_tokens == 0 {
        return Ok(Generated::default());
    }
    if sampling.prompt_logprobs.is_some() && !images.is_empty() {
        return Err("prompt logprobs score a text prompt; this one carries pictures".into());
    }
    let sampled = sampling.temperature > 0.0;
    if sampling.speculative
        && (!sampled || (sampling.speculative_sampled && sampling.untruncated()))
        && !sampling.penalized()
        && guide.is_none()
        && sampling.logprobs.is_none()
        && sampling.prompt_logprobs.is_none()
        && !prompt.is_empty()
        && images.is_empty()
        // Beside other requests the run-ahead decode below batches with
        // theirs; drafting would cost all of them.
        && Pipeline::live() == 0
        && let Some(draft) = model::draft_block()
        && draft.proposals_from <= 1
        && (2..=16).contains(&draft.rows)
    {
        let sampled = sampled.then_some((sampling.temperature, sampling.seed));
        let (sampled, cached) =
            crate::speculative::generate(prompt, ends, max_tokens, draft, sampled, &mut |t| {
                on_token(t, None)
            })
            .await?;
        return Ok(Generated {
            sampled,
            cached,
            prompt_logprobs: Vec::new(),
        });
    }
    let vocab = model::output_vocab_size();
    let n = prompt.len() as u32;

    let ws = WorkingSet::new();
    let pipe = Pipeline::new();

    let penalized = sampling.penalized();
    let present: Vec<f32> = if penalized {
        let mut v = vec![0.0f32; vocab as usize];
        for &t in prompt {
            if (t as usize) < v.len() {
                v[t as usize] = 1.0;
            }
        }
        v
    } else {
        Vec::new()
    };

    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    // Scored, every prompt row but the last is prefilled by the scoring
    // passes, and only the last token is left to sample after.
    let prompt_logprobs = match sampling.prompt_logprobs {
        Some(top) => score_prompt(&ws, &pipe, &prompt_i32, top, ends).await?,
        None => Vec::new(),
    };
    if sampling.prompt_logprobs.is_some() {
        on_prompt(&prompt_logprobs);
    }
    let base = prompt_logprobs.len();

    // Unconstrained, the prompt's last fire samples its token straight into
    // the decode pass's channels, as each decode fire does for the next: the
    // decode fires queue behind the prompt with no turn through the host,
    // which reads the prompt's token once they are on their way. A grammar
    // advances on the host before the next mask exists, so it waits; so do
    // logprobs, which the host reads beside each token.
    let carried_on = (guide.is_none() && sampling.logprobs.is_none()).then(|| {
        (
            Channel::new([1], dtype::i32).named("tok_in").queued(),
            penalized.then(|| {
                (
                    Channel::new([vocab], dtype::f32).named("counts").queued(),
                    Channel::new([vocab], dtype::f32).named("present").queued(),
                )
            }),
        )
    });
    let toks = Channel::from(&prompt_i32[base..]).named("toks_p");
    let embed_indptr = Channel::from([0u32, n - base as u32]).named("embed_indptr_p");
    let rng = Channel::from([sampling.seed, 0u32]).named("rng_p");
    let hist = penalized.then(|| {
        (
            Channel::from(vec![0.0f32; vocab as usize]).named("counts_p"),
            Channel::from(present.clone()).named("present_p"),
        )
    });
    let mask = guide
        .is_some()
        .then(|| Channel::new([vocab], dtype::bool).named("mask_p"));
    let g0_ch = Channel::new([1], dtype::i32).named("g0");
    let lp_ch = logprob_channels(sampling.logprobs, "_p", 1);
    let knobs = KnobChannels::new(&sampling, "_p");

    let fwd = ForwardPass::new();
    fwd.embed(&toks, &embed_indptr)?;
    if !images.is_empty() {
        let spans: Vec<media::Span> = images.iter().map(media::Span::Image).collect();
        fwd.media(0, &spans)?;
    }
    fwd.state(&[&ws], StateGeometry::folding())?;
    fwd.epilogue(move || {
        let r = rng.take();
        let taken = hist.map(|(c, p)| (c.take(), p.take()));
        let m = mask.map(|m| m.take());
        let (token, counts_next, landed) = sample(
            vocab,
            &knobs.read(),
            &r,
            taken.as_ref().map(|(c, p)| (c, p)),
            m.as_ref(),
        );
        let token = reshape(token, [1]);
        g0_ch.put(&token);
        land(landed, lp_ch);
        if let Some((tok_in, hist)) = carried_on {
            tok_in.put(&token);
            if let (Some((c, p)), Some(next), Some((_, present))) = (hist, counts_next, taken) {
                c.put(&next);
                p.put(&present);
            }
        }
    });
    if let (Some(m), Some(guide)) = (mask, guide.as_deref_mut()) {
        m.put(unpack_mask(&guide.mask(), vocab));
    }
    // Checkpoints only where messages end: each costs a state slot.
    fwd.checkpoints(ends)?;
    fwd.submit(&pipe).context("prefill submit")?;
    if max_tokens == 1 {
        // Nothing follows the prompt's token: the pipeline ends before
        // it is read, as a decode's does once its budget is submitted.
        pipe.close();
    }
    let (g0, mut g0_later, lp0) = match carried_on {
        Some(_) => (0, Some(g0_ch), None),
        None => (
            g0_ch.take_host::<i32>().await.context("prefill drain")?,
            None,
            take_logprob(lp_ch.as_ref()).await?,
        ),
    };
    let cached = ws.cached_tokens();

    let mut sampled = 0usize;
    if carried_on.is_none() {
        sampled = 1;
        if let Some(guide) = guide.as_deref_mut() {
            guide.accept(g0 as u32).context("advance grammar")?;
        }
        if on_token(g0 as u32, lp0).is_break()
            || max_tokens == 1
            || guide.as_deref().is_some_and(|g| g.is_terminated())
        {
            pipe.close();
            return Ok(Generated {
                sampled,
                cached: cached as usize,
                prompt_logprobs,
            });
        }
    }

    let (tok_in, carried_hist) = match carried_on {
        Some((tok_in, hist)) => (tok_in, Some(hist)),
        None => (Channel::from([g0]).named("tok_in"), None),
    };
    let out = Channel::new([1], dtype::i32)
        .capacity(channel_capacity() as u32)
        .named("out");
    let rng = Channel::from([sampling.seed ^ 0x9e37, 1u32]).named("rng");
    let knobs = KnobChannels::new(&sampling, "");
    let lp_ch = logprob_channels(sampling.logprobs, "", channel_capacity() as u32);
    let lane1 = Channel::from([0u32, 1u32]).named("embed_indptr");
    let hist = carried_hist.unwrap_or_else(|| {
        penalized.then(|| {
            let mut counts0 = vec![0.0f32; vocab as usize];
            if let Ok(g0) = usize::try_from(g0)
                && g0 < counts0.len()
            {
                counts0[g0] = 1.0;
            }
            (
                Channel::from(counts0).named("counts"),
                Channel::from(present.clone()).named("present"),
            )
        })
    });
    let mask = guide
        .is_some()
        .then(|| Channel::new([vocab], dtype::bool).named("mask"));

    let fwd = ForwardPass::new();
    fwd.embed(&tok_in, &lane1)?;
    fwd.state(&[&ws], StateGeometry::folding())?;
    fwd.epilogue(move || {
        let r = rng.take();
        let taken = hist.map(|(c, p)| (c.take(), p.take()));
        let m = mask.map(|m| m.take());
        let k = knobs.read();

        let (token, counts_next, landed) = sample(
            vocab,
            &k,
            &r,
            taken.as_ref().map(|(c, p)| (c, p)),
            m.as_ref(),
        );
        let token = reshape(token, [1]);
        land(landed, lp_ch);
        let r_next = &r + iota(2);

        tok_in.put(&token);
        out.put(&token);
        rng.put(&r_next);
        if let (Some((c, p)), Some(next), Some((_, present))) = (hist, counts_next, taken) {
            c.put(&next);
            p.put(&present);
        }
    });

    let budget = max_tokens - 1;
    match (guide, mask) {
        (Some(guide), Some(mask)) => {
            let mut fired = 0usize;
            while fired < budget {
                mask.put(unpack_mask(&guide.mask(), vocab));
                fwd.submit(&pipe).context("constrained decode submit")?;
                fired += 1;
                let t = out.take_host::<Vec<i32>>().await?;
                let token = *t.first().unwrap_or(&0) as u32;
                let lp = take_logprob(lp_ch.as_ref()).await?;
                sampled += 1;
                guide.accept(token).context("advance grammar")?;
                if on_token(token, lp).is_break() || guide.is_terminated() {
                    break;
                }
            }
        }
        _ => {
            // A sampled token off `ch` to `on_token`: the prompt's, read once
            // the decode fires are queued, then each decode fire's off `out`
            // (with its logprobs where asked: logprobs turn the carry off, so
            // a carried prompt token never has any).
            let mut take = async |ch: Channel,
                                  lp: Option<&(Channel, Option<Channel>)>|
                   -> std::result::Result<ControlFlow<()>, String> {
                let t = ch.take_host::<Vec<i32>>().await?;
                let token = *t.first().unwrap_or(&0) as u32;
                let lp = take_logprob(lp).await?;
                sampled += 1;
                Ok(on_token(token, lp))
            };
            if budget == 0 {
                // The prompt's token is the only one, whatever it says.
                if let Some(g0_ch) = g0_later.take() {
                    let _ = take(g0_ch, None).await?;
                }
            } else {
                run_ahead(&pipe, &fwd, budget, async || {
                    if let Some(g0_ch) = g0_later.take()
                        && take(g0_ch, None).await?.is_break()
                    {
                        return Ok(ControlFlow::Break(()));
                    }
                    take(out, lp_ch.as_ref()).await
                })
                .await?;
            }
        }
    }
    pipe.close();
    Ok(Generated {
        sampled,
        cached: cached as usize,
        prompt_logprobs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scored_row_reads_its_own_slice_of_the_flat_alternatives() {
        let rows = prompt_rows(
            &[-0.5, -2.0],
            &[1.0, 7.0],
            &[-0.5, -1.0, -0.1, -0.3],
            &[4, 9, 2, 3],
            2,
        );
        assert_eq!(
            rows,
            vec![
                Logprob {
                    logprob: -0.5,
                    top: vec![(4, -0.5), (9, -1.0)],
                    rank: Some(1),
                },
                Logprob {
                    logprob: -2.0,
                    top: vec![(2, -0.1), (3, -0.3)],
                    rank: Some(7),
                },
            ]
        );
    }

    #[test]
    fn a_row_scored_with_no_alternatives_has_none() {
        let rows = prompt_rows(&[-1.0], &[3.0], &[], &[], 0);
        assert_eq!(rows[0].top, Vec::new());
        assert_eq!(rows[0].rank, Some(3));
    }

    #[test]
    fn prompt_logprobs_past_the_cap_are_refused() {
        let sampling = Sampling {
            prompt_logprobs: Some(MAX_TOP_LOGPROBS + 1),
            ..Sampling::default()
        };
        assert!(sampling.validate().is_err());
    }
}
