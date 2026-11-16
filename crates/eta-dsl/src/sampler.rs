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

/// The shapes the SDK's requests take most: greedy, a temperature draw
/// (Gumbel-max), top-p, top-k with top-p, and greedy landing its top
/// logprobs. The knobs' values here are any request's.
pub const STANDARD: [(&str, Shape); 5] = [
    ("greedy", Shape::GREEDY),
    (
        "gumbel-max",
        Shape {
            temperature: Some(1.0),
            ..Shape::GREEDY
        },
    ),
    (
        "top-p",
        Shape {
            temperature: Some(1.0),
            top_p: true,
            ..Shape::GREEDY
        },
    ),
    (
        "top-k + top-p",
        Shape {
            temperature: Some(1.0),
            top_p: true,
            top_k: true,
            ..Shape::GREEDY
        },
    ),
    (
        "top logprobs",
        Shape {
            logprobs: Some(5),
            ..Shape::GREEDY
        },
    ),
];

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
