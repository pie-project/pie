//! The SDK's standard sampler: the stage bodies the built-in inferlets'
//! generation loop runs (`inferlet::sampler`), here so a host can trace the
//! same programs. The OpenAI penalties over a vocabulary histogram carried
//! on the device, an optional grammar mask, temperature, then min-p, top-k
//! and top-p truncation and a Gumbel-max draw; temperature zero is exact
//! greedy. Asked for logprobs, a step also lands the sampled token's
//! log-probability under the model's raw logits and the likeliest tokens'.
//!
//! A runtime builds the guest code of [`STANDARD`]'s shapes before ready:
//! a program's regions are keyed by their text, which the knobs' values
//! (they ride in channels, or are constants the text does not hold) do not
//! change, so a request of one of these shapes builds nothing.

use crate::channel::Channel;
use crate::dtype;
use crate::intrinsics;
use crate::value::*;

/// Which knobs a sampler has: the program's shape. Their values are the
/// request's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape {
    /// A program constant; `None` samples greedily.
    pub temperature: Option<f32>,
    pub top_p: bool,
    pub top_k: bool,
    pub min_p: bool,
    pub repetition: bool,
    pub frequency: bool,
    pub presence: bool,
    /// A program constant: how many alternatives a step lands, if any.
    pub logprobs: Option<u32>,
}

impl Shape {
    pub const GREEDY: Shape = Shape {
        temperature: None,
        top_p: false,
        top_k: false,
        min_p: false,
        repetition: false,
        frequency: false,
        presence: false,
        logprobs: None,
    };
}

/// The shapes the built-in inferlets' requests take most ([`Values`]):
/// greedy, sampled, and greedy landing logprobs. The knobs' values here are
/// any request's.
pub const STANDARD: [(&str, Shape); 3] = [
    ("greedy", Shape::GREEDY),
    (
        "sampled",
        Shape {
            temperature: Some(1.0),
            top_p: true,
            top_k: true,
            min_p: true,
            ..Shape::GREEDY
        },
    ),
    (
        "top logprobs",
        Shape {
            logprobs: Some(MAX_TOP_LOGPROBS),
            ..Shape::GREEDY
        },
    ),
];

/// The most alternatives a step lands, as OpenAI caps `top_logprobs`: a
/// program landing logprobs always lands this many, and the host keeps the
/// ones a request asked for, so every logprobs request shares one program.
pub const MAX_TOP_LOGPROBS: u32 = 20;

/// A request's sampling as the built-in inferlets take it: temperature zero
/// is greedy, a top-k of zero is off.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Request {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub min_p: f32,
    pub repetition: f32,
    pub frequency: f32,
    pub presence: f32,
    /// Alternatives a step reports, where asked.
    pub logprobs: Option<u32>,
}

/// A request's values for every knob its kind carries: the program's shape
/// is which are present, and a request's unused knobs ride at their neutral
/// values (top-p 1, min-p 0, top-k the vocabulary; penalties 1 and 0), so
/// one program serves every request of its kind: greedy or sampled (at a
/// temperature, a program constant), penalized or not, landing logprobs or
/// not. Each present knob rides in a `[1]` channel the host seeds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Values {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub repetition: Option<f32>,
    pub frequency: Option<f32>,
    pub presence: Option<f32>,
    /// Always [`MAX_TOP_LOGPROBS`] where a request asks for any.
    pub logprobs: Option<u32>,
}

impl Request {
    pub fn penalized(&self) -> bool {
        self.repetition != 1.0 || self.frequency != 0.0 || self.presence != 0.0
    }

    /// The knobs this request's kind carries, at its values.
    pub fn values(&self, vocab: u32) -> Values {
        let sampled = self.temperature > 0.0;
        let penalized = self.penalized();
        let top_k = match self.top_k {
            0 => vocab,
            k => k.min(vocab),
        };
        Values {
            temperature: sampled.then(|| self.temperature.max(1e-4)),
            top_p: sampled.then_some(self.top_p),
            top_k: sampled.then_some(top_k),
            min_p: sampled.then_some(self.min_p),
            repetition: penalized.then_some(self.repetition),
            frequency: penalized.then_some(self.frequency),
            presence: penalized.then_some(self.presence),
            logprobs: self.logprobs.map(|_| MAX_TOP_LOGPROBS),
        }
    }
}

impl Values {
    /// Which knobs are present: the program's shape.
    pub fn shape(&self) -> Shape {
        Shape {
            temperature: self.temperature,
            top_p: self.top_p.is_some(),
            top_k: self.top_k.is_some(),
            min_p: self.min_p.is_some(),
            repetition: self.repetition.is_some(),
            frequency: self.frequency.is_some(),
            presence: self.presence.is_some(),
            logprobs: self.logprobs,
        }
    }
}

/// The sampler's knob values as scalars.
pub struct Knobs {
    /// A program constant: a channel-fed divisor costs the fused sampler
    /// ~20% of a decode step, more than mixed temperatures cost by splitting.
    pub temperature: Option<f32>,
    pub top_p: Option<Tensor>,
    pub top_k: Option<Tensor>,
    pub min_p: Option<Tensor>,
    pub repetition: Option<Tensor>,
    pub frequency: Option<Tensor>,
    pub presence: Option<Tensor>,
    pub logprobs: Option<u32>,
}

/// What a step lands beside its token when asked for logprobs: the sampled
/// token's logprob then the alternatives' (`[1 + top]` f32), and the
/// alternatives' ids (`[top]` i32, absent for none).
pub struct Landed {
    pub logprobs: Tensor,
    pub ids: Option<Tensor>,
}

/// The channels a step reads beside the logits: its rng, the histogram
/// (`(counts, present)`) where it penalizes, the grammar mask where it is
/// guided, and where it lands logprobs (`(values, ids)`).
pub struct Slots {
    pub rng: Channel,
    pub hist: Option<(Channel, Channel)>,
    pub mask: Option<Channel>,
    pub logprobs: Option<(Channel, Option<Channel>)>,
}

/// The prompt's last fire: samples its token into `g0`, and, `carried_on`,
/// into the decode pass's token channel (and histogram), so the decode
/// fires queue behind it with no turn through the host.
pub fn prefill(
    vocab: u32,
    knobs: &Knobs,
    slots: &Slots,
    g0: &Channel,
    carried_on: Option<&(Channel, Option<(Channel, Channel)>)>,
) {
    let r = slots.rng.take();
    let taken = slots.hist.as_ref().map(|(c, p)| (c.take(), p.take()));
    let m = slots.mask.as_ref().map(Channel::take);
    let (token, counts_next, landed) = sample(
        vocab,
        knobs,
        &r,
        taken.as_ref().map(|(c, p)| (c, p)),
        m.as_ref(),
    );
    let token = reshape(token, [1]);
    g0.put(&token);
    land(landed, slots.logprobs.as_ref());
    if let Some((tok_in, hist)) = carried_on {
        tok_in.put(&token);
        if let (Some((c, p)), Some(next), Some((_, present))) = (hist, counts_next, taken) {
            c.put(&next);
            p.put(&present);
        }
    }
}

/// A decode fire: samples the next token into `tok_in` (which the next
/// fire embeds) and `out` (which the host drains), and steps the rng and
/// the histogram.
pub fn decode(vocab: u32, knobs: &Knobs, slots: &Slots, tok_in: &Channel, out: &Channel) {
    let r = slots.rng.take();
    let taken = slots.hist.as_ref().map(|(c, p)| (c.take(), p.take()));
    let m = slots.mask.as_ref().map(Channel::take);
    let (token, counts_next, landed) = sample(
        vocab,
        knobs,
        &r,
        taken.as_ref().map(|(c, p)| (c, p)),
        m.as_ref(),
    );
    let token = reshape(token, [1]);
    land(landed, slots.logprobs.as_ref());
    let r_next = &r + iota(2);
    tok_in.put(&token);
    out.put(&token);
    slots.rng.put(&r_next);
    if let (Some((c, p)), Some(next), Some((_, present))) = (&slots.hist, counts_next, taken) {
        c.put(&next);
        p.put(&present);
    }
}

/// A step's landed logprobs put in its channels, where asked.
fn land(landed: Option<Landed>, channels: Option<&(Channel, Option<Channel>)>) {
    if let (Some(landed), Some((values, ids))) = (landed, channels) {
        values.put(&landed.logprobs);
        if let (Some(landed), Some(ids)) = (landed.ids, ids) {
            ids.put(&landed);
        }
    }
}

/// The sampled token as a rank-0 i32, the next output-token histogram, and
/// the step's logprobs where the knobs ask. `histogram` is `(counts,
/// present)`: how often each token was generated, and whether it was in the
/// prompt.
pub fn sample(
    vocab: u32,
    k: &Knobs,
    r: &Tensor,
    histogram: Option<(&Tensor, &Tensor)>,
    mask: Option<&Tensor>,
) -> (Tensor, Option<Tensor>, Option<Landed>) {
    let mut logits = reshape(intrinsics::output(0, "logits"), [vocab]);
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
