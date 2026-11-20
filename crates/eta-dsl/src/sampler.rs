//! The SDK's standard sampler: the stage bodies the built-in inferlets'
//! generation loop runs (`inferlet::sampler`), here so a host can trace the
//! same programs. The OpenAI penalties over a vocabulary histogram carried
//! on the device, an optional grammar mask, temperature, then min-p, top-k
//! and top-p truncation and a Gumbel-max draw; temperature zero is exact
//! greedy. Asked for logprobs, a step also lands the sampled token's
//! log-probability under the model's raw logits and the likeliest tokens'.
//!
//! The vocabulary-sized values a step computes (the penalized and masked
//! logits, the probabilities its cuts read, the log-probabilities) are bf16
//! (`eta_ir::types`' rules: computed in f32, each rounded once on its
//! store, every reduction over them in f32). The draw is one
//! `gumbel_argmax` over the logits as they are, the model's f32 row where
//! no penalty or grammar mask changed them: temperature, the cut and the
//! noise are applied per lane in f32 inside the reduction, so no
//! perturbed row is rounded and no noise row is laid. Temperature has one
//! form, in the cut and the draw alike: a multiply by `1 / T`, rounded once
//! to f32 on the host ([`inverse_temperature`], `eta_ir::types`). A greedy
//! step that changes no logit reads the model's f32 logits whole.
//!
//! A runtime builds the guest code of [`standard`]'s shapes before ready:
//! every knob's value, the temperature's reciprocal among them, rides in a
//! channel the stage reads, so no value is in a program's text, its hash or
//! its launch package, by which the engines key what they compile, and a
//! request of one of these shapes builds nothing.

use crate::channel::Channel;
use crate::dtype;
use crate::intrinsics;
use crate::value::*;

/// Which knobs a sampler has: the program's shape. Their values are the
/// request's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape {
    /// A draw at a temperature; `false` samples greedily.
    pub temperature: bool,
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
        temperature: false,
        top_p: false,
        top_k: false,
        min_p: false,
        repetition: false,
        frequency: false,
        presence: false,
        logprobs: None,
    };
}

/// The draws the SDK's requests take most: greedy, a temperature draw
/// (Gumbel-max), top-p, and top-k with top-p.
pub const DRAWS: [(&str, Shape); 4] = [
    ("greedy", Shape::GREEDY),
    (
        "gumbel-max",
        Shape {
            temperature: true,
            ..Shape::GREEDY
        },
    ),
    (
        "top-p",
        Shape {
            temperature: true,
            top_p: true,
            ..Shape::GREEDY
        },
    ),
    (
        "top-k + top-p",
        Shape {
            temperature: true,
            top_p: true,
            top_k: true,
            ..Shape::GREEDY
        },
    ),
];

/// How many alternatives a standard shape that lands logprobs lands.
pub const STANDARD_TOP_LOGPROBS: u32 = 5;

/// The standard shapes: each of [`DRAWS`] as it is, penalized, landing its
/// top [`STANDARD_TOP_LOGPROBS`] logprobs, and both. A penalized request
/// carries all three penalties, the ones it does not ask for at their
/// neutral values (repetition 1, frequency and presence 0), so which it
/// asks for is data, as their values are. The knobs' values here are any
/// request's.
#[must_use]
pub fn standard() -> Vec<(String, Shape)> {
    let extras: [(&str, &str, bool, Option<u32>); 4] = [
        ("", "", false, None),
        ("penalized ", "", true, None),
        ("", " + top logprobs", false, Some(STANDARD_TOP_LOGPROBS)),
        ("penalized ", " + top logprobs", true, Some(STANDARD_TOP_LOGPROBS)),
    ];
    extras
        .iter()
        .flat_map(|&(before, after, penalized, logprobs)| {
            DRAWS.iter().map(move |(name, draw)| {
                (
                    format!("{before}{name}{after}"),
                    Shape {
                        repetition: penalized,
                        frequency: penalized,
                        presence: penalized,
                        logprobs,
                        ..*draw
                    },
                )
            })
        })
        .collect()
}

/// The sampler's knob values as scalars.
pub struct Knobs {
    /// The temperature's reciprocal ([`inverse_temperature`]), an f32 the
    /// cut and the draw multiply by; `None` samples greedily.
    pub temperature: Option<Tensor>,
    pub top_p: Option<Tensor>,
    pub top_k: Option<Tensor>,
    pub min_p: Option<Tensor>,
    pub repetition: Option<Tensor>,
    pub frequency: Option<Tensor>,
    pub presence: Option<Tensor>,
    pub logprobs: Option<u32>,
}

/// The value a request's temperature `t` rides in its channel as: `1 / t`,
/// rounded once to f32 on the host, which the cut and the draw multiply by
/// (a multiply is correctly rounded on every backend, a divide is not).
#[must_use]
pub fn inverse_temperature(t: f32) -> f32 {
    1.0 / t
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
    let model = reshape(intrinsics::output(0, "logits"), [vocab]);
    let raw = k.logprobs.map(|_| shifted(&model));
    // A step that changes the logits works on a bf16 copy of them; one
    // that does not (greedy, or a draw) reads the model's f32 row.
    let changes = histogram.is_some() || mask.is_some();
    let mut logits = if changes {
        cast(&model, dtype::bf16)
    } else {
        model
    };
    // A knob is one value, an f32 scalar every op mixes in unrounded.
    let knob = |t: &Tensor| {
        if t.shape().is_scalar() {
            t.clone()
        } else {
            reshape(t, [])
        }
    };
    if let Some((counts, present)) = histogram {
        let seen_out = gt(counts, 0.0f32);
        let seen_prompt = gt(present, 0.5f32);
        let seen = or(&seen_out, &seen_prompt);
        if let Some(rep) = &k.repetition {
            let rp = knob(rep);
            let positive = gt(&logits, 0.0f32);
            let repenalized = select(&positive, &logits / &rp, &logits * &rp);
            logits = select(&seen, &repenalized, &logits);
        }
        if let Some(freq) = &k.frequency {
            logits = &logits - cast(counts, dtype::bf16) * knob(freq);
        }
        if let Some(pres) = &k.presence {
            logits = &logits - cast(&seen_out, dtype::bf16) * knob(pres);
        }
    }
    if let Some(mask) = mask {
        logits = mask_apply(&logits, mask);
    }
    let token = match &k.temperature {
        None => reduce_argmax(&logits),
        Some(inverse) => {
            // The temperature's one form: a multiply by its reciprocal.
            let inverse = knob(inverse);
            let mut keep: Option<Tensor> = None;
            if k.top_k.is_some() || k.top_p.is_some() || k.min_p.is_some() {
                // The cuts read bf16 probabilities of the scaled row.
                let probs = cut_probabilities(&logits, &inverse);
                if let Some(min_p) = &k.min_p {
                    let floor = reduce_max(&probs) * knob(min_p);
                    keep = Some(ge(&probs, floor));
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
            match keep {
                Some(keep) => gumbel_argmax(&logits, keep, &inverse, r),
                None => gumbel_argmax(&logits, true, &inverse, r),
            }
        }
    };
    let next = histogram.map(|(counts, _)| scatter_add(counts, &token, 1.0f32));
    let landed = raw.zip(k.logprobs).map(|(raw, top)| {
        let lp = log_softmax(&raw);
        let chosen = cast(
            reshape(scalar_gather(&lp, reshape(&token, [1])), [1]),
            dtype::f32,
        );
        if top == 0 {
            return Landed {
                logprobs: chosen,
                ids: None,
            };
        }
        let (values, ids) = top_k(&lp, top);
        let values = cast(&values, dtype::f32);
        Landed {
            logprobs: concat_f32(&chosen, &values, top),
            ids: Some(cast(&ids, dtype::i32)),
        }
    });
    (token, next, landed)
}

/// `row` less its max rounded to bf16, as a bf16 row: a softmax and a
/// log-softmax do not see the shift, and the row's largest values, which
/// decide them, round near zero, where a bf16 is finest (a logit of 20
/// rounded alone is 1/16 off, its shifted value a few thousandths).
fn shifted(row: &Tensor) -> Tensor {
    row - cast(reduce_max(row), dtype::bf16)
}

/// The probabilities the cuts read, `softmax(logits * inverse)` with
/// `inverse` the temperature's reciprocal, as a bf16 row: the exponent is
/// the row less its max rounded to bf16, times `inverse`, so its largest
/// values round near zero and no lane overflows; the one max the row needs
/// is that shift.
fn cut_probabilities(logits: &Tensor, inverse: &Tensor) -> Tensor {
    let e = exp(shifted(logits) * inverse);
    &e / reduce_sum(&e)
}

/// `[1]` and `[n]` f32 rows as one `[1 + n]` row.
fn concat_f32(head: &Tensor, tail: &Tensor, n: u32) -> Tensor {
    let at = iota(n + 1);
    let first = eq(&at, broadcast(0u32, [n + 1]));
    let shifted = gather(tail, select(&first, broadcast(0u32, [n + 1]), &at - 1u32));
    select(&first, broadcast(reshape(head, []), [n + 1]), &shifted)
}
