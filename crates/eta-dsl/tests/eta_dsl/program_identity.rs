//! Program identity: a program kind registers ONE program whatever a
//! request's values. The runtime dedups programs by the hash of their whole
//! encoded container, so a request value that reaches the container (a knob
//! traced as a constant, a length traced as a shape) makes every request
//! register, plan and build its own program.
//!
//! Each case builds a kind's programs twice, for two requests that differ in
//! everything a request may vary (knob values, temperature where it is no
//! program constant, prompt length, `max_tokens`, widths within their max),
//! and asserts the same container hashes. The programs are traced through
//! eta-dsl as the SDK's guests build them: the SDK's `Channel::from(v)` is a
//! seeded declaration here (the value never enters the container), and the
//! stage bodies are the guests' own (`eta_dsl::sampler` for the built-in
//! inferlets) or follow them line for line. The guests themselves build for
//! wasm32-wasip3 against WIT imports, so they cannot be traced here: a change
//! to a guest's channels that this module does not follow is not caught by it.
//!
//! A kind not yet reusable is a `should_panic` case naming why, so the gate
//! shows the gap: a fix that makes it reusable fails the case until it is
//! turned into a plain one.

use eta_dsl::builder::Builder;
use eta_dsl::prelude::*;
use eta_dsl::sampler::{self, Knobs, Slots};
use eta_dsl::{Channel, Dtype, Traced};

const VOCAB: u32 = 151_936;
const PAGE: u32 = 32;
/// The run-ahead window's channel capacity.
const CAP: u32 = 8;
/// The most alternatives compat lands a step.
const MAX_TOP: u32 = sampler::MAX_TOP_LOGPROBS;

fn leak<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

/// A channel as the SDK's `Channel::from(v)` declares it: seeded, its value
/// staged outside the program.
fn from(v: impl IntoConst) -> &'static Channel {
    let data = v.into_const();
    leak(Channel::seeded(data.shape, data.dtype))
}

/// A channel the host takes.
fn taken(shape: impl IntoShape, dtype: Dtype, capacity: u32) -> &'static Channel {
    let channel = leak(Channel::new(shape, dtype).capacity(capacity));
    channel.note_host_take();
    channel
}

/// A channel the host puts each fire.
fn written(shape: impl IntoShape, dtype: Dtype) -> &'static Channel {
    let channel = leak(Channel::new(shape, dtype));
    channel.note_host_put();
    channel
}

/// A knob's value as a stage reads it off its `[1]` channel.
fn knob(channel: &Channel) -> Tensor {
    reshape(channel.read(), [])
}

/// The ports a pass binds beside its stage body.
struct Ports {
    tokens: &'static Channel,
    indptr: &'static Channel,
    fold: Option<&'static Channel>,
    slots: Option<&'static Channel>,
    readout: Option<&'static Channel>,
}

impl Ports {
    /// Tokens appending at the lane's tail and folding all they write.
    fn folding(tokens: &'static Channel, indptr: &'static Channel) -> Ports {
        Ports {
            tokens,
            indptr,
            fold: Some(from([u32::MAX])),
            slots: None,
            readout: None,
        }
    }
}

fn pass(ports: Ports, body: impl Fn() + 'static) -> Traced {
    let mut b = Builder::new(VOCAB, PAGE);
    b.outputs(crate::outputs::logits(VOCAB));
    b.bind_port(Port::EmbedTokens, ports.tokens);
    b.bind_port(Port::EmbedIndptr, ports.indptr);
    if let Some(fold) = ports.fold {
        b.bind_port(Port::Fold, fold);
    }
    if let Some(slots) = ports.slots {
        b.bind_port(Port::Slots, slots);
    }
    if let Some(readout) = ports.readout {
        b.bind_port(Port::Readout, readout);
    }
    b.stage(Stage::Epilogue, body);
    b.build().expect("traces")
}

/// The most rows one fire binds, as an engine states it
/// (`model.max-pass-rows()`): what a per-row channel registers at.
const ROWS: u32 = 8192;

/// The hashes the runtime registers `programs` by: each container with its
/// per-row channels declared at the row bound
/// (`TraceContainer::bind_row_extents`).
fn hashes(programs: &[Traced]) -> Vec<u64> {
    programs
        .iter()
        .map(|traced| {
            let mut container =
                eta_ir::container::decode(&traced.encode()).expect("the container decodes");
            container.bind_row_extents(ROWS);
            container.hash()
        })
        .collect()
}

/// Every request's programs hash alike.
fn one_identity<R>(kind: &str, requests: &[R], build: impl Fn(&R) -> Vec<Traced>) {
    let first = hashes(&build(&requests[0]));
    for (at, request) in requests.iter().enumerate().skip(1) {
        assert_eq!(
            first,
            hashes(&build(request)),
            "{kind}: request {at} has a program identity of its own"
        );
    }
}

/// A prompt's tokens as the SDK channels them.
fn prompt(len: usize) -> &'static Channel {
    from(vec![0i32; len])
}

// ── compat: the built-in inferlets' generation loop ─────────────────────

/// A compat request's sampling, as `compat::generate::Sampling` holds it.
#[derive(Clone, Copy)]
struct Sampling {
    temperature: f32,
    top_p: f32,
    top_k: u32,
    min_p: f32,
    repetition: f32,
    frequency: f32,
    presence: f32,
    logprobs: Option<u32>,
    guided: bool,
    seed: u32,
    prompt: usize,
}

const NEUTRAL: Sampling = Sampling {
    temperature: 0.0,
    top_p: 1.0,
    top_k: 0,
    min_p: 0.0,
    repetition: 1.0,
    frequency: 0.0,
    presence: 0.0,
    logprobs: None,
    guided: false,
    seed: 1,
    prompt: 12,
};

impl Sampling {
    fn penalized(&self) -> bool {
        self.repetition != 1.0 || self.frequency != 0.0 || self.presence != 0.0
    }
}

/// compat's `KnobChannels`: the knobs the request's kind carries, as
/// `eta_dsl::sampler::Request::values` decides them for compat, each seeded
/// in a `[1]` channel and read in the stage.
struct KnobChannels {
    values: sampler::Values,
    channels: [Option<&'static Channel>; 6],
    top_k: Option<&'static Channel>,
}

fn knobs(s: &Sampling) -> KnobChannels {
    let values = sampler::Request {
        temperature: s.temperature,
        top_p: s.top_p,
        top_k: s.top_k,
        min_p: s.min_p,
        repetition: s.repetition,
        frequency: s.frequency,
        presence: s.presence,
        logprobs: s.logprobs,
    }
    .values(VOCAB);
    let f = |v: Option<f32>| v.map(|v| from([v]));
    KnobChannels {
        values,
        channels: [
            f(values.top_p),
            f(values.min_p),
            f(values.repetition),
            f(values.frequency),
            f(values.presence),
            None,
        ],
        top_k: values.top_k.map(|k| from([k])),
    }
}

impl KnobChannels {
    fn read(&self) -> Knobs {
        let t = |c: Option<&'static Channel>| c.map(knob);
        let [top_p, min_p, repetition, frequency, presence, _] = self.channels;
        Knobs {
            temperature: self.values.temperature,
            top_p: t(top_p),
            top_k: t(self.top_k),
            min_p: t(min_p),
            repetition: t(repetition),
            frequency: t(frequency),
            presence: t(presence),
            logprobs: self.values.logprobs,
        }
    }
}

/// The channels compat lands a step's logprobs in: always the most asked.
fn logprob_channels(s: &Sampling, capacity: u32) -> Option<(Channel, Option<Channel>)> {
    s.logprobs.map(|_| {
        (
            taken([1 + MAX_TOP], dtype::f32, capacity).clone(),
            Some(taken([MAX_TOP], dtype::i32, capacity).clone()),
        )
    })
}

/// compat's prompt pass and decode pass for `s`.
fn compat(s: &Sampling) -> Vec<Traced> {
    let vocab = VOCAB;
    let penalized = s.penalized();
    let carried = !s.guided && s.logprobs.is_none();
    let tok_in = if carried {
        leak(Channel::new([1], dtype::i32).queued())
    } else {
        from([0i32])
    };
    let carried_hist = (carried && penalized).then(|| {
        (
            Channel::new([vocab], dtype::f32).queued(),
            Channel::new([vocab], dtype::f32).queued(),
        )
    });
    // The `(counts, present)` histogram, `[vocab]` rows the host seeds.
    let hist = || {
        penalized.then(|| {
            (
                Channel::seeded([vocab], dtype::f32),
                Channel::seeded([vocab], dtype::f32),
            )
        })
    };
    let mask = || s.guided.then(|| written([vocab], dtype::bool).clone());

    let g0 = taken([1], dtype::i32, 1);
    let slots = Slots {
        rng: from([s.seed, 0u32]).clone(),
        hist: hist(),
        mask: mask(),
        logprobs: logprob_channels(s, 1),
    };
    let k = knobs(s);
    let carried_on = carried.then(|| (tok_in.clone(), carried_hist.clone()));
    let prefill = pass(
        Ports::folding(prompt(s.prompt), from([0u32, s.prompt as u32])),
        move || sampler::prefill(vocab, &k.read(), &slots, g0, carried_on.as_ref()),
    );

    let out = taken([1], dtype::i32, CAP);
    let slots = Slots {
        rng: from([s.seed ^ 0x9e37, 1u32]).clone(),
        hist: carried_hist.or_else(hist),
        mask: mask(),
        logprobs: logprob_channels(s, CAP),
    };
    let k = knobs(s);
    let decode = pass(Ports::folding(tok_in, from([0u32, 1])), move || {
        sampler::decode(vocab, &k.read(), &slots, tok_in, out)
    });
    vec![prefill, decode]
}

/// The decode pass alone: what a request of a kind runs every token.
fn compat_decode(s: &Sampling) -> Vec<Traced> {
    compat(s).split_off(1)
}

/// Two requests of a kind, differing in every value the kind does not fix.
fn compat_requests(kind: Sampling) -> [Sampling; 2] {
    let sampled = kind.temperature > 0.0;
    let penalized = kind.penalized();
    let other = Sampling {
        top_p: if sampled { 0.5 } else { kind.top_p },
        top_k: if sampled { 40 } else { kind.top_k },
        min_p: if sampled { 0.05 } else { kind.min_p },
        repetition: if penalized { 1.3 } else { 1.0 },
        frequency: if penalized { 0.0 } else { kind.frequency },
        presence: if penalized { 0.8 } else { kind.presence },
        logprobs: kind.logprobs.map(|_| 3),
        seed: 99,
        ..kind
    };
    [kind, other]
}

const GREEDY: Sampling = NEUTRAL;
const SAMPLED: Sampling = Sampling {
    temperature: 0.7,
    top_p: 0.95,
    ..NEUTRAL
};
const PENALIZED: Sampling = Sampling {
    frequency: 0.5,
    ..SAMPLED
};
const GUIDED: Sampling = Sampling {
    guided: true,
    ..NEUTRAL
};
const LOGPROBS: Sampling = Sampling {
    logprobs: Some(5),
    ..NEUTRAL
};

#[test]
fn every_compat_kind_decodes_with_one_program() {
    for (name, kind) in [
        ("greedy", GREEDY),
        ("sampled", SAMPLED),
        ("penalized", PENALIZED),
        ("guided", GUIDED),
        ("logprobs", LOGPROBS),
        (
            "sampled logprobs",
            Sampling {
                logprobs: Some(1),
                ..SAMPLED
            },
        ),
    ] {
        one_identity(name, &compat_requests(kind), compat_decode);
    }
}

#[test]
fn every_compat_kind_prefills_one_length_with_one_program() {
    for (name, kind) in [
        ("greedy", GREEDY),
        ("sampled", SAMPLED),
        ("penalized", PENALIZED),
        ("guided", GUIDED),
        ("logprobs", LOGPROBS),
    ] {
        one_identity(name, &compat_requests(kind), compat);
    }
}

/// A sampled request's temperature is a program constant (a channel-fed
/// divisor costs the fused sampler ~20% of a decode step until the compiler
/// lowers rank-0 channels as constants): each temperature is a program.
#[test]
#[should_panic(expected = "program identity of its own")]
fn a_compat_temperature_is_still_a_program_constant() {
    let other = Sampling {
        temperature: 1.1,
        ..SAMPLED
    };
    one_identity("sampled temperatures", &[SAMPLED, other], compat_decode);
}

/// The prompt's tokens register at the row bound, whatever its length.
#[test]
fn compat_prompts_of_two_lengths_prefill_with_one_program() {
    let other = Sampling {
        prompt: 300,
        ..GREEDY
    };
    one_identity("greedy prefill", &[GREEDY, other], compat);
}


// ── speculative verifies ────────────────────────────────────────────────

/// A verify of `width` rows with explicit slots, readout and fold: compat's
/// speculative rungs, cacheback's verify and mtp's window share this layout.
fn verify(width: u32, body: impl Fn(&'static Channel, u32) + 'static) -> Traced {
    let out = taken([width], dtype::i32, CAP);
    pass(
        Ports {
            tokens: written([width], dtype::i32),
            indptr: from([0u32, width]),
            fold: Some(from([0u32])),
            slots: Some(written([width], dtype::u32)),
            readout: Some(from((0..width).collect::<Vec<u32>>())),
        },
        move || body(out, width),
    )
}

/// compat's speculative rungs: one pass a rung, fired with each round's
/// tokens, slots, fold and proposals; a sampled one reads its proposals off
/// a channel it is seeded with and the host sets.
fn compat_rung(round: &(u32, u32)) -> Vec<Traced> {
    let &(width, proposal) = round;
    let proposals = from(vec![proposal; width as usize]);
    let chances = taken([width], dtype::f32, 1);
    let noise = from([7u32, 0]);
    vec![verify(width, move |out, width| {
        let logits = reshape(intrinsics::output(0, "logits"), [width, VOCAB]);
        let r = noise.take();
        let draws = reduce_argmax(&(&logits / 0.7f32) + gumbel(&r, [width, VOCAB]));
        let hit = eq(cast(&draws, dtype::u32), proposals.read());
        out.put(reshape(draws, [width]));
        chances.put(cast(hit, dtype::f32));
        noise.put(&r + iota(2));
    })]
}

#[test]
fn a_speculative_rung_verifies_every_round_with_one_program() {
    for width in [2, 4, 8, 16] {
        one_identity("speculative rung", &[(width, 3), (width, 911)], compat_rung);
    }
}

/// cacheback's verify of `(draft_length, drafted)`: it fires at
/// `draft_length + 1` rows however few the cache drafted, the rows past the
/// drafts repeating the anchor.
fn cacheback_verify(round: &(u32, u32)) -> Vec<Traced> {
    let &(draft_length, _drafted) = round;
    vec![verify(draft_length + 1, |out, _| {
        out.put(reduce_argmax(intrinsics::output(0, "logits")));
    })]
}

#[test]
fn a_cacheback_verify_runs_one_program_whatever_was_drafted() {
    one_identity("cacheback verify", &[(4u32, 1u32), (4, 4)], cacheback_verify);
}

/// mtp's window: the model's widest, `k` read off a channel.
fn mtp_window(k: &u32) -> Vec<Traced> {
    const DEPTH: u32 = 3;
    let w = DEPTH + 1;
    let k_c = from([*k]);
    vec![verify(w, move |out, w| {
        let truth = reshape(reduce_argmax(intrinsics::output(0, "logits")), [w]);
        let asked = knob(k_c);
        let within = lt(iota(DEPTH), broadcast(&asked, [DEPTH]));
        let said = gather(&truth, iota(DEPTH));
        let hit = cast(and(eq(&said, &said), &within), dtype::u32);
        let m = reshape(reduce_sum(cumprod(&hit)), [1]);
        out.put(gather(&truth, broadcast(m, [w])));
    })]
}

#[test]
fn an_mtp_window_runs_one_program_for_every_k() {
    one_identity("mtp window", &[1u32, 2, 3], mtp_window);
}

// ── the sampler examples ────────────────────────────────────────────────

/// An example's decode pass: one token in, its rng, the stage `body` over
/// the logits and the taken rng state, the token out.
fn sampler_decode(body: impl Fn(Tensor, &Tensor) -> Tensor + 'static) -> Traced {
    let tok_in = from([0i32]);
    let rng = from([1u32, 0]);
    let out = taken([1], dtype::i32, CAP);
    pass(Ports::folding(tok_in, from([0u32, 1])), move || {
        let r = rng.take();
        let token = body(intrinsics::output(0, "logits"), &r);
        tok_in.put(&token);
        out.put(&token);
        rng.put(&r + iota(2));
    })
}

/// A request's knob values, the example's own.
type Values = [f32; 4];

fn masked(keep: &Tensor, scaled: &Tensor) -> Tensor {
    select(keep, scaled, broadcast(f32::NEG_INFINITY, [VOCAB]))
}

fn tail_free(v: &Values) -> Vec<Traced> {
    let (z, t) = (from([v[0]]), from([v[1]]));
    vec![sampler_decode(move |logits, r| {
        let scaled = &logits / knob(t);
        let k_max = 128;
        let (_, order) = top_k(&scaled, k_max);
        let p = gather(softmax(&scaled), &order);
        let exclusive = cumsum(&p) - &p;
        let keep_sorted = lt(&exclusive, broadcast(knob(z), [k_max]));
        let keep = scatter_set(broadcast(false, [VOCAB]), &order, keep_sorted);
        gumbel_max(masked(&keep, &scaled), r)
    })]
}

fn typical(v: &Values) -> Vec<Traced> {
    let (mass, t) = (from([v[0]]), from([v[1]]));
    vec![sampler_decode(move |logits, r| {
        let scaled = &logits / knob(t);
        let k_max = 64;
        let lp = log_softmax(&scaled);
        let probs = exp(&lp);
        let h = entropy_from_logprobs(&probs, &lp);
        let (_, order) = top_k(-abs(&lp + broadcast(&h, [VOCAB])), k_max);
        let sorted = gather(&probs, &order);
        let keep_sorted = lt(cumsum(&sorted) - &sorted, broadcast(knob(mass), [k_max]));
        let keep = scatter_set(broadcast(false, [VOCAB]), &order, keep_sorted);
        gumbel_max(masked(&keep, &scaled), r)
    })]
}

fn top_a(v: &Values) -> Vec<Traced> {
    let (a, t) = (from([v[0]]), from([v[1]]));
    vec![sampler_decode(move |logits, r| {
        let scaled = &logits / knob(t);
        let probs = softmax(&scaled);
        let p_max = reduce_max(&probs);
        let threshold = knob(a) * &(&p_max * &p_max);
        let keep = ge(&probs, broadcast(&threshold, [VOCAB]));
        gumbel_max(masked(&keep, &scaled), r)
    })]
}

fn xtc(v: &Values) -> Vec<Traced> {
    let (threshold, min_p, probability, t) =
        (from([v[0]]), from([v[1]]), from([v[2]]), from([v[3]]));
    vec![sampler_decode(move |logits, r| {
        let scaled = &logits / knob(t);
        let probs = softmax(&scaled);
        let floor = knob(min_p) * reduce_max(&probs);
        let kept = ge(&probs, broadcast(&floor, [VOCAB]));
        let above = and(&kept, ge(&probs, broadcast(knob(threshold), [VOCAB])));
        let u = reduce_min(rng(&(r + broadcast(0x9e37_79b9u32, [2])), [1]));
        let fired = lt(&u, knob(probability));
        let drop = and(broadcast(&fired, [VOCAB]), &above);
        gumbel_max(masked(&and(&kept, not(&drop)), &scaled), r)
    })]
}

fn eta_epsilon(v: &Values) -> Vec<Traced> {
    let (epsilon, t) = (from([v[0]]), from([v[1]]));
    vec![sampler_decode(move |logits, r| {
        let scaled = &logits / knob(t);
        let lp = log_softmax(&scaled);
        let probs = exp(&lp);
        let eps = knob(epsilon);
        let h = entropy_from_logprobs(&probs, &lp);
        let threshold = min_elem(&eps, sqrt(&eps) * exp(-&h));
        let keep = pivot_threshold(&probs, prob_ge(threshold));
        gumbel_max(masked(&keep, &scaled), r)
    })]
}

fn mirostat(v: &Values) -> Vec<Traced> {
    let (tau, lr, k_min) = (from([v[0]]), from([v[1]]), from([v[2] as u32]));
    let mu = from([5.0f32]);
    vec![sampler_decode(move |logits, r| {
        let mu_v = mu.take();
        let probs = softmax(&logits);
        let base = pivot_threshold(&probs, prob_ge(exp(-reshape(&mu_v, []))));
        let floor = pivot_threshold(&logits, rank_le(knob(k_min)));
        let token = gumbel_max(masked(&or(base, floor), &logits), r);
        let surprise = -log(scalar_gather(&probs, cast(&token, dtype::u32)));
        mu.put(&mu_v - &(&(&surprise - &knob(tau)) * &knob(lr)));
        token
    })]
}

fn repetition_penalty(v: &Values) -> Vec<Traced> {
    let (rep, freq, pres, t) = (from([v[0]]), from([v[1]]), from([v[2]]), from([v[3]]));
    let counts_c = leak(Channel::seeded([VOCAB], dtype::f32));
    let present_c = leak(Channel::seeded([VOCAB], dtype::f32));
    vec![sampler_decode(move |logits, r| {
        let counts = counts_c.take();
        let zero = broadcast(0.0f32, [VOCAB]);
        let out_seen = gt(&counts, &zero);
        let seen = or(&out_seen, gt(present_c.read(), broadcast(0.5f32, [VOCAB])));
        let rp = broadcast(knob(rep), [VOCAB]);
        let repenalized = select(gt(&logits, &zero), &logits / &rp, &logits * &rp);
        let l = select(&seen, &repenalized, &logits);
        let l = &l - &(broadcast(knob(freq), [VOCAB]) * &counts);
        let l = &l - &(broadcast(knob(pres), [VOCAB]) * cast(&out_seen, dtype::f32));
        let token = gumbel_max(&l / knob(t), r);
        counts_c.put(scatter_add(&counts, &token, 1.0f32));
        token
    })]
}

fn temperature_only(v: &Values) -> Vec<Traced> {
    let t = from([v[0]]);
    vec![sampler_decode(move |logits, r| gumbel_max(&logits / knob(t), r))]
}

fn nucleus(v: &Values) -> Vec<Traced> {
    let (t, top_p) = (from([v[0]]), from([v[1]]));
    vec![sampler_decode(move |logits, r| {
        nucleus_sample(&logits / knob(t), knob(top_p), r)
    })]
}

fn entropy_adaptive(v: &Values) -> Vec<Traced> {
    let knobs = from(*v);
    let at = move |i: u32| reshape(gather(knobs.read(), broadcast(Tensor::constant(i), [1])), []);
    vec![sampler_decode(move |logits, r| {
        let lp = log_softmax(&logits);
        let h = max_elem(entropy_from_logprobs(exp(&lp), &lp), 1e-6f32);
        let t = at(0) * exp(&(at(1) / &h) * at(2));
        let t = max_elem(&t, at(3));
        gumbel_max(&logits / broadcast(&t, [VOCAB]), r)
    })]
}

#[test]
fn every_sampler_example_decodes_with_one_program() {
    let requests = [[0.95, 1.0, 0.5, 0.7], [0.3, 0.6, 0.1, 1.3]];
    for (name, build) in [
        ("tail-free", tail_free as fn(&Values) -> Vec<Traced>),
        ("locally-typical", typical),
        ("top-a", top_a),
        ("xtc", xtc),
        ("eta-epsilon", eta_epsilon),
        ("mirostat-v2", mirostat),
        ("repetition-penalty", repetition_penalty),
        ("naive-baseline", temperature_only),
        ("chat-completion / consensus", nucleus),
        ("entropy-adaptive-temperature", entropy_adaptive),
    ] {
        one_identity(name, &requests, build);
    }
}

/// diffusion-baseline's denoise step: the schedule and the stopping knobs
/// off channels, through the SDK's helpers (taken here as the SDK spells
/// them: `inferlet::eta::diffusion`).
fn denoise(v: &Values) -> Vec<Traced> {
    const CANVAS: u32 = 16;
    let (t_max, t_min, bound, confidence) =
        (from([v[0]]), from([v[1]]), from([v[2]]), from([v[3]]));
    let steps = from([48u32]);
    let step_index = from([0u32]);
    let out = taken([CANVAS + 1], dtype::f32, 1);
    vec![pass(
        Ports {
            tokens: from(vec![0i32; CANVAS as usize]),
            indptr: from([0u32, CANVAS]),
            fold: None,
            slots: Some(from((0..CANVAS).collect::<Vec<u32>>())),
            readout: Some(from((0..CANVAS).collect::<Vec<u32>>())),
        },
        move || {
            let i = step_index.take();
            let steps = reshape(steps.read(), [1]);
            let (hi, lo) = (reshape(t_max.read(), [1]), reshape(t_min.read(), [1]));
            let remaining = cast(&(&steps - &i), dtype::f32);
            let t = reshape(
                &(&(&remaining / &cast(&steps, dtype::f32)) * &(&hi - &lo)) + &lo,
                [],
            );
            step_index.put(&i + 1u32);
            let scaled = &intrinsics::output(0, "logits") / &t;
            let h = entropy(softmax(&scaled));
            let n = CANVAS;
            let (neg, order) = sort_desc(-&h);
            let sorted = -&neg;
            let below = le(&(&cumsum(&sorted) - &sorted), broadcast(knob(bound), [n]));
            let accept = scatter_set(lt(iota(n), 0u32), &order, &below);
            let mean = &reduce_sum(&h) / (n as f32);
            let done = lt(&mean, knob(confidence));
            let word = select(&accept, broadcast(1.0f32, [n]), broadcast(0.0f32, [n]));
            let word = scatter_set(
                broadcast(0f32, [n + 1]),
                iota(n),
                &word,
            );
            out.put(scatter_set(&word, &(&iota(1) + n), cast(reshape(&done, [1]), dtype::f32)));
        },
    )]
}

#[test]
fn a_denoise_step_runs_one_program_for_every_schedule() {
    one_identity("diffusion denoise", &[[0.8, 0.4, 0.1, 0.005], [1.0, 0.2, 0.3, 0.01]], denoise);
}

// ── kinds not yet reusable ──────────────────────────────────────────────

/// A prefill of `len` prompt tokens, as every example and the prefill
/// builtin channel them.
fn prefill(len: &usize) -> Vec<Traced> {
    let out = taken([1], dtype::i32, 1);
    vec![pass(Ports::folding(prompt(*len), from([0u32, *len as u32])), move || {
        out.put(reshape(reduce_argmax(intrinsics::output(0, "logits")), [1]));
    })]
}

#[test]
fn prompts_of_two_lengths_prefill_with_one_program() {
    one_identity("prefill", &[7usize, 300], prefill);
}

/// json-schema's decode embeds the forced chunk, 1 to 16 tokens, through a
/// channel the host writes: its length is the chunk's rows, registered at
/// the row bound.
fn json_chunk(len: &u32) -> Vec<Traced> {
    let out = taken([1], dtype::i32, CAP);
    let mask = written([VOCAB.div_ceil(32)], dtype::u32);
    vec![pass(Ports::folding(written([*len], dtype::i32), from([0u32, *len])), move || {
        let packed = mask.take();
        let allowed = ne(gather(&packed, &iota(VOCAB) / Tensor::constant(32u32)), 0u32);
        out.put(reshape(masked_argmax(intrinsics::output(0, "logits"), &allowed), [1]));
    })]
}

#[test]
fn json_schema_chunks_of_two_lengths_decode_with_one_program() {
    one_identity("json-schema decode", &[1u32, 5], json_chunk);
}

/// constrained-speculative's verify spans the committed tail and the draft:
/// its tokens and slots register at the row bound, but its readout rows and
/// its `[window]` output are what the stage computes on, one program a
/// window until it reads out at the widest window.
#[test]
#[should_panic(expected = "program identity of its own")]
fn constrained_speculative_windows_verify_with_one_program() {
    one_identity("constrained verify", &[3u32, 6], |&w| {
        vec![verify(w, |out, _| {
            out.put(reduce_argmax(intrinsics::output(0, "logits")));
        })]
    });
}

/// A program whose channel is shaped by a request's length: beam search's
/// mask over its key pool (`pool_len` = prompt + decodes), naive-masked's
/// key span, the score readers' `kv_max` rows (snapkv, tova, h2o), and the
/// history rings of DRY, gumbel-watermark and synthid (prompt +
/// `max_tokens`). Reusable only with dynamic key lengths (step 4).
fn keyed(len: &u32) -> Vec<Traced> {
    let keys = leak(Channel::seeded([*len], dtype::f32));
    let out = taken([1], dtype::i32, CAP);
    vec![pass(Ports::folding(from([0i32]), from([0u32, 1])), move || {
        let logits = intrinsics::output(0, "logits");
        let k = keys.take();
        keys.put(&k);
        out.put(reshape(reduce_argmax(&logits), [1]));
    })]
}

#[test]
#[should_panic(expected = "program identity of its own")]
fn beam_search_pools_of_two_lengths_run_one_program() {
    one_identity("beam pool", &[128u32, 160], keyed);
}

#[test]
#[should_panic(expected = "program identity of its own")]
fn naive_masked_spans_of_two_lengths_run_one_program() {
    one_identity("naive-masked span", &[40u32, 41], keyed);
}

#[test]
#[should_panic(expected = "program identity of its own")]
fn score_readers_of_two_lengths_run_one_program() {
    one_identity("snapkv / tova / h2o kv_max", &[256u32, 512], keyed);
}

#[test]
#[should_panic(expected = "program identity of its own")]
fn history_rings_of_two_lengths_run_one_program() {
    one_identity("DRY / watermark history", &[64u32, 96], keyed);
}
