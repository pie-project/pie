//! Text completion for benchmarking: exact token counts, no per-token streaming.

use inferlet::eta::prelude::*;
use inferlet::{chat, sampler, session};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    prompt_tokens: Option<Vec<u32>>,
    #[serde(default)]
    prompts: Vec<String>,
    #[serde(default)]
    prompt_tokens_batch: Vec<Vec<u32>>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_system")]
    system: String,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    /// Top-k truncation where sampled (0: none), as compat's.
    #[serde(default)]
    top_k: u32,
    /// When true, drop the chat-template stop tokens so the generator runs to
    /// `max_tokens` regardless of model emit.
    #[serde(default)]
    ignore_eos: bool,
    /// Turn a thinking model's thinking off, as the model's own template does
    /// for `enable_thinking: false`.
    #[serde(default)]
    no_think: bool,
    /// Per-token inferlet compute, in microseconds: after each drained token
    /// the guest spins on the monotonic clock this long (it reads the clock;
    /// it never waits on a timer, which the host would read as outside input
    /// and park the lane for). The lane stays awaited throughout, as a guest
    /// computing between fires does. Off by default (0): each delayed lane
    /// burns one CPU while it spins.
    #[serde(default)]
    wasm_delay_us: u64,
    /// Decode and return generated text (the throughput benchmark only needs
    /// counts unless it dumps a sample).
    #[serde(default = "default_return_text")]
    return_text: bool,
    #[serde(default)]
    wait_for_start: bool,
    #[serde(default)]
    batch_concurrency: Option<usize>,
    /// Report first-token and inter-token timing: send a `t0` session
    /// message when the first sampled token reaches guest code (the client
    /// stamps its arrival for launch-inclusive TTFT) and return per-token
    /// drain gaps in the envelope. Single-request mode only.
    #[serde(default)]
    report_timing: bool,
    /// Return guest-drain stamps from the shared host monotonic clock without
    /// the live `t0` message.
    #[serde(default)]
    report_arrivals: bool,
    /// Frames of decode run-ahead each lane keeps queued in the runtime.
    /// Leave unset to use the runtime's own sizing (`channel-capacity()`),
    /// which is what every other inferlet does; set it only to sweep the
    /// depth against a measurement. The wave quorum waits for every lane, so
    /// a lane that drops to one queued frame puts its whole guest turnaround
    /// on the fleet's critical path; deeper queues buy slack for that
    /// turnaround.
    #[serde(default)]
    run_ahead_frames: Option<usize>,
    /// Decode fires each lane keeps submitted past the last token it took,
    /// one a frame: a cap on run-ahead counted in fires, not frames (a
    /// sweep of a hybrid model's cost of a short window). Overrides
    /// `run_ahead_frames`.
    #[serde(default)]
    run_ahead_fires: Option<usize>,
    /// Rows the working set leaves unfolded behind each lane's last row:
    /// the prompt folds all but this many and each decode folds one, so
    /// every decode carries this many buffered rows (a hybrid model's
    /// replay at a steady lag). 0 folds all it writes.
    #[serde(default)]
    fold_lag: u32,
    /// Whether the prefill and the decode are final, the SDK's default: the
    /// bench never takes a row back, so they are unless asked not to
    /// (`false`: `ForwardPass::retractable`, the lagged fold, for an A/B
    /// against it) or a `fold_lag` holds rows unfolded on purpose.
    #[serde(default = "stated_final")]
    final_rows: bool,
    /// For a plane whose channel rings belong to the pass instance, not the channel.
    #[serde(default)]
    host_first_token: bool,
    /// Ask the runtime to keep the prompt's state at its end (`checkpoints`),
    /// so a later pass sharing the prompt resumes there.
    #[serde(default)]
    checkpoint_prompt_end: bool,
}

fn default_max_tokens() -> usize {
    256
}
fn default_system() -> String {
    "You are a helpful, respectful and honest assistant.".into()
}
fn default_temperature() -> f32 {
    0.6
}
fn default_top_p() -> f32 {
    0.95
}
fn default_return_text() -> bool {
    true
}

#[derive(Serialize)]
struct Output {
    /// Tokens in the chat-templated prompt that fed the prefill.
    num_prompt_tokens: usize,
    /// Tokens the sampler actually emitted — authoritative, no harness-side
    /// re-tokenisation.
    num_output_tokens: usize,
    /// Decoded text — for spot-checking output quality.
    text: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    token_ids: Vec<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    request_prompt_tokens: Vec<usize>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    request_output_tokens: Vec<usize>,
    /// A batch's lanes' emitted tokens, each lane's (`token_ids` is the
    /// first's).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    request_token_ids: Vec<Vec<u32>>,
    /// Guest-clock first-token latency from run start (report_timing only).
    /// The client-stamped `t0` message is the launch-inclusive number; this
    /// one excludes instantiation and locates where the time went.
    #[serde(skip_serializing_if = "Option::is_none")]
    ttft_us: Option<u64>,
    /// Gaps between successive accepted-token drains (timing/arrivals only).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    intertoken_us: Vec<u32>,
    /// Absolute shared-host `CLOCK_MONOTONIC` marks for accepted tokens.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    token_monotonic_ns: Vec<u64>,
    /// Guest prologue step durations in µs, ordered
    /// [main_pre (configure+stop_tokens), tokenize, setup (vocab/rng/ws),
    /// reserve, build_submit] (report_timing only) — locates pre-bind time.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    prologue_us: Vec<u64>,
}

/// The standard sampler's knobs for `temperature` and `top_p`, as the
/// built-in inferlets' generation loop seeds them: argmax at `temperature <=
/// 0`, otherwise temperature-scaled Gumbel-max, nucleus-truncated below
/// `top_p < 1` (`top_p >= 1` keeps every token, so it is no knob). The same
/// shapes the runtime's prewarm builds, so a request builds nothing.
struct SampleKnobs {
    /// The temperature's reciprocal (`sampler::inverse_temperature`), data
    /// the stage reads, so every temperature runs one program.
    temperature: Option<Channel>,
    top_p: Option<Channel>,
    top_k: Option<Channel>,
}

impl SampleKnobs {
    fn new(temperature: f32, top_p: f32, top_k: u32) -> Self {
        let sampled = temperature > 0.0;
        Self {
            temperature: sampled
                .then(|| Channel::from([sampler::inverse_temperature(temperature)])),
            top_p: (sampled && top_p < 1.0).then(|| Channel::from([top_p])),
            top_k: (sampled && top_k > 0).then(|| Channel::from([top_k])),
        }
    }

    /// Read without consuming: the seeded value stays for every fire.
    fn read(&self) -> sampler::Knobs {
        sampler::Knobs {
            temperature: self.temperature.as_ref().map(|c| reshape(c.read(), [])),
            top_p: self.top_p.as_ref().map(|c| reshape(c.read(), [])),
            top_k: self.top_k.as_ref().map(|c| reshape(c.read(), [])),
            min_p: None,
            repetition: None,
            frequency: None,
            presence: None,
            logprobs: None,
        }
    }
}

/// A pass's rng as the standard sampler's stage bodies take it.
fn slots(rng: &Channel) -> sampler::Slots {
    sampler::Slots {
        rng: rng.traced(),
        hist: None,
        mask: None,
        logprobs: None,
    }
}

/// A batch's lanes step through their prologue together: every lane binds
/// before any opens its pipeline, and every lane of a wave opens its pipeline,
/// then submits its first frame, before any goes on. Each lane records the
/// last step it reached, and a lane that returns, `Ok` or `Err`, is past every
/// step, so a lane that fails never strands the others at a step it will not
/// reach.
///
/// The calls between the steps are synchronous, so a step is a yield, not a
/// sleep: a lane at a step returns to `drive_until`, which polls the other
/// lanes up to it within the same poll of the task. No lane registers a waker
/// at a step; the task may sleep only once every lane is past its prologue and
/// parked on a host future (a channel take). A lane left at a step when the
/// task sleeps has nothing that would wake it, which wit-bindgen refuses
/// ("cannot sleep waiting only on Rust-originating events").
struct Lockstep {
    reached: Vec<std::cell::Cell<u8>>,
}

const BOUND: u8 = 1;
const PIPED: u8 = 2;
const SUBMITTED: u8 = 3;
const FINISHED: u8 = u8::MAX;

impl Lockstep {
    fn new(lanes: usize) -> Self {
        Self {
            reached: (0..lanes).map(|_| std::cell::Cell::new(0)).collect(),
        }
    }

    fn all(&self, lanes: std::ops::Range<usize>, step: u8) -> bool {
        self.reached[lanes].iter().all(|r| r.get() >= step)
    }

    /// Records `lane` at `step`, then yields until every lane of `group` has
    /// reached it.
    async fn arrive(&self, lane: usize, step: u8, group: std::ops::Range<usize>) {
        self.reached[lane].set(step);
        let mut yielded = false;
        std::future::poll_fn(|_| {
            if yielded && self.all(group.clone(), step) {
                std::task::Poll::Ready(())
            } else {
                yielded = true;
                std::task::Poll::Pending
            }
        })
        .await;
    }
}

/// Marks a lane past every step when its `run_one` returns.
struct Finished<'a>(&'a Lockstep, usize);

impl Drop for Finished<'_> {
    fn drop(&mut self) {
        self.0.reached[self.1].set(FINISHED);
    }
}

/// A batch lane's place in the lockstep: the lane and its wave.
struct LaneStep<'a> {
    lockstep: &'a Lockstep,
    lane: usize,
    wave: std::ops::Range<usize>,
}

type Lane<'a> = (
    std::pin::Pin<Box<dyn std::future::Future<Output = Result<RunResult>> + 'a>>,
    Option<Result<RunResult>>,
);

fn failed(lanes: &[Lane<'_>]) -> bool {
    lanes.iter().any(|(_, res)| matches!(res, Some(Err(_))))
}

/// Polls `lanes` until `done()` holds or a lane fails. Each lane reaches its
/// next step within one poll, and a returned lane is past every step, so this
/// ends.
async fn drive_until(lanes: &mut [Lane<'_>], done: impl Fn() -> bool) {
    std::future::poll_fn(|cx| {
        while !done() && !failed(lanes) {
            for (fut, res) in lanes.iter_mut() {
                if done() {
                    break;
                }
                if res.is_none()
                    && let std::task::Poll::Ready(r) = fut.as_mut().poll(cx)
                {
                    *res = Some(r);
                }
            }
        }
        std::task::Poll::Ready(())
    })
    .await;
}

struct RunResult {
    num_prompt_tokens: usize,
    num_output_tokens: usize,
    tokens: Vec<u32>,
    ttft_us: Option<u64>,
    intertoken_us: Vec<u32>,
    token_monotonic_ns: Vec<u64>,
    prologue_us: Vec<u64>,
}

async fn run_one(
    input: &Input,
    prompt: &str,
    prompt_tokens: Option<&[u32]>,
    stop_tokens: &[u32],
    honor_wait_for_start: bool,
    rng_seed: u32,
    main_pre_us: Option<u64>,
    lockstep: Option<LaneStep<'_>>,
) -> Result<RunResult> {
    let _finished = lockstep.as_ref().map(|l| Finished(l.lockstep, l.lane));
    let mut prologue_us: Vec<u64> = Vec::new();
    if input.report_timing {
        prologue_us.push(main_pre_us.unwrap_or(0)); // [0] main_pre
    }
    let mut step_clock = std::time::Instant::now();
    let mut step = |v: &mut Vec<u64>, on: bool| {
        if on {
            v.push(step_clock.elapsed().as_micros() as u64);
        }
        step_clock = std::time::Instant::now();
    };
    // Prompt: explicit pre-tokenized tokens (the batch path) or chat-template
    // system+user+cue.
    let prompt_vec: Vec<u32> = if let Some(tokens) = prompt_tokens {
        tokens.to_vec()
    } else {
        let mut p = chat::system_user(&input.system, prompt);
        p.extend(chat::cue_thinking(!input.no_think));
        p
    };
    step(&mut prologue_us, input.report_timing); // [1] tokenize
    let num_prompt_tokens = prompt_vec.len();
    if prompt_vec.is_empty() {
        return Err("empty prompt".into());
    }

    if honor_wait_for_start && input.wait_for_start {
        session::send("ready");
        let _ = session::receive().await;
    }
    // Timing origin: after the optional start handshake, mirroring the
    // harness's client-side clock reset on `start`.
    let run_start = std::time::Instant::now();

    let vocab = model::output_vocab_size();
    // A greedy pass steps its rng too, unread: the standard shape's.
    let sampled_rng = Channel::from([rng_seed, 0u32]);
    let prefill_rng = Channel::from([rng_seed ^ 0x9e37_79b9, 0u32]);
    let (temperature, top_p, top_k) = (input.temperature, input.top_p, input.top_k);

    // A bench's prompts share their prefixes across its requests.
    let ws = WorkingSet::shared(ShareScope::Tenant)?;
    let n = prompt_vec.len() as u32;
    step(&mut prologue_us, input.report_timing); // [2] setup (vocab/rng/ws)
    // Nothing to reserve; the empty step keeps `prologue_us`'s shape.
    step(&mut prologue_us, input.report_timing); // [3] reserve

    let prompt_i32: Vec<i32> = prompt_vec.iter().map(|&t| t as i32).collect();

    let toks_p = Channel::from(&prompt_i32[..]);
    let embed_indptr_p = Channel::from([0u32, n]);
    // Decode fires: the prefill sample already spends one of the
    // `max_tokens` sampler activations.
    let budget = decode_budget(input.max_tokens)?;
    // A host-visible channel cannot attach to two passes, so this one is device-only.
    let tok_in = (!input.host_first_token).then(|| Channel::new([1], dtype::i32));
    let g0_ch = Channel::new([1], dtype::i32);
    let live_slots = frame_size();

    // A frame of margin above the window (`7 * live_slots` covers every frame size): the ticket
    // check silently skips a continuation that lands inside the staging margin.
    // When `ignore_eos` is active and per-token host work is zero, keep a
    // deeper window (`16` frames) so 256-lane high-concurrency waves do not
    // serialize on host-side frame-seal wakes between fast 2-step GPU batches.
    let (window_fires, out_capacity) = match input.run_ahead_frames {
        _ if let Some(f) = input.run_ahead_fires => {
            let w = f.max(1);
            (w, w + 1 + 7 * live_slots)
        }
        Some(r) => {
            let w = r.max(1) * live_slots;
            (w, w + 1 + 7 * live_slots)
        }
        None if input.ignore_eos && input.wasm_delay_us == 0 => {
            let w = (12 * live_slots).max(channel_capacity() - 1);
            (w, w + 1 + 7 * live_slots)
        }
        None => {
            let cap = channel_capacity();
            (cap - 1, cap + 7 * live_slots)
        }
    };
    let out = Channel::new([1], dtype::i32).capacity(out_capacity as u32);

    let fwd_p = ForwardPass::new();
    fwd_p.embed(&toks_p, &embed_indptr_p)?;
    let lag = input.fold_lag.min(u32::try_from(num_prompt_tokens).unwrap_or(0));
    let prefill_fold = Channel::from([u32::try_from(num_prompt_tokens).unwrap_or(0) - lag]);
    let decode_fold = Channel::from([1u32]);
    fn geometry(lag: u32, fold: &Channel) -> StateGeometry<'_> {
        match lag {
            0 => StateGeometry::folding(),
            _ => StateGeometry {
                fold: Fold::Count(fold),
                ..StateGeometry::default()
            },
        }
    }
    fwd_p
        .state(&[&ws], geometry(lag, &prefill_fold))
        .context("bind prefill state")?;
    // The bench never takes a row back, the prompt's included: a prefill
    // final as its decodes are leaves no row for the first decode to carry,
    // so no steady step catches one buffered (CONTRACT-BC 25). Final is the
    // SDK's default; the lagged A/B and a fold lag opt out.
    let final_rows = input.final_rows && lag == 0;
    if !final_rows {
        fwd_p
            .retractable()
            .context("state the prefill retractable")?;
    }
    if input.checkpoint_prompt_end {
        let end = u32::try_from(num_prompt_tokens).map_err(|_| "a prompt past u32 rows")?;
        fwd_p.checkpoints(&[end]).context("ask the prompt end's checkpoint")?;
    }
    let prefill_knobs = SampleKnobs::new(temperature, top_p, top_k);
    fwd_p.epilogue(move || {
        let carried_on = tok_in.map(|tok_in| (tok_in.traced(), None));
        sampler::prefill(
            vocab,
            &prefill_knobs.read(),
            &slots(&prefill_rng),
            &g0_ch.traced(),
            carried_on.as_ref(),
        );
    });

    // Built before the prefill submits, so its `tok_in` claim precedes the prefill's build.
    let ws_ref = &ws;
    let build_decode = move |tok_in: Channel| -> std::result::Result<Option<ForwardPass>, String> {
        Ok(if budget > 0 {
            let fwd = ForwardPass::new();
            let embed_indptr = Channel::from([0u32, 1u32]);
            fwd.embed(&tok_in, &embed_indptr)?;
            fwd.state(&[ws_ref], geometry(lag, &decode_fold))
                .context("bind decode state")?;
            if !final_rows {
                fwd.retractable().context("state the decode retractable")?;
            }
            let knobs = SampleKnobs::new(temperature, top_p, top_k);
            fwd.epilogue(move || {
                sampler::decode(
                    vocab,
                    &knobs.read(),
                    &slots(&sampled_rng),
                    &tok_in.traced(),
                    &out.traced(),
                );
            });
            Some(fwd)
        } else {
            None
        })
    };

    let mut build_decode = Some(build_decode);
    let mut fwd_d: Option<ForwardPass> = match tok_in {
        Some(tok_in) => build_decode
            .take()
            .expect("the decode builder is called once")(tok_in)?,
        None => None,
    };

    // First frame: the prefill in slot 0, then up to live_slots-1
    // decode slots. At live_slots = 1 this is a bare prefill submit
    // (`submit` IS a single-slot frame); trailing slots pad to no-ops.
    // Under `host_first_token` there is no decode pass yet, so the
    // frame is one slot wide — the same shape live_slots = 1 already
    // submits.
    // The prefill goes alone, and the decode window is staged only after its
    // token has reached the host (below): a window staged ahead held the
    // first token's drain behind up to three decode fires, which the client
    // saw as time to first token (Gemma-4-31B on an M5 Max: 754 ms of
    // prefill, 866 ms to the first token). Throughput at 32 lanes is the same.
    let first_decodes = if !honor_wait_for_start
        && !input.report_timing
        && !input.report_arrivals
        && !input.host_first_token
        && fwd_d.is_some()
    {
        live_slots.saturating_sub(1).min(budget).min(window_fires)
    } else {
        0
    };
    let mut first_slots: Vec<Option<&ForwardPass>> = Vec::with_capacity(live_slots);
    first_slots.push(Some(&fwd_p));
    for _ in 0..first_decodes {
        first_slots.push(Some(fwd_d.as_ref().expect("decode pass exists")));
    }
    if let Some(l) = &lockstep {
        fwd_p.bind().context("bind prefill program")?;
        if let Some(fwd) = &fwd_d {
            fwd.bind().context("bind decode program")?;
        }
        let batch = 0..l.lockstep.reached.len();
        l.lockstep.arrive(l.lane, BOUND, batch).await;
    }
    let pipe = Pipeline::new();
    if let Some(l) = &lockstep {
        l.lockstep.arrive(l.lane, PIPED, l.wave.clone()).await;
    }
    submit_frame(&pipe, &first_slots).context("first frame submit")?;
    if let Some(l) = &lockstep {
        l.lockstep.arrive(l.lane, SUBMITTED, l.wave.clone()).await;
    }
    let mut submitted = first_decodes;

    // The pass is an argument: under `host_first_token` it is built after this closure.
    let submit_ahead = |fwd: &ForwardPass,
                        mut submitted: usize,
                        drained: usize|
     -> std::result::Result<usize, String> {
        let burst_limit = if drained == 0 {
            (5 * live_slots).min(window_fires)
        } else {
            (2 * live_slots).min(window_fires)
        };
        let mut burst = 0usize;
        while submitted < budget && burst < burst_limit {
            let s = (budget - submitted).min(live_slots).min(window_fires);
            if submitted - drained + s > window_fires {
                break;
            }
            let slots: Vec<Option<&ForwardPass>> = (0..s).map(|_| Some(fwd)).collect();
            submit_frame(&pipe, &slots).context("decode frame submit")?;
            submitted += s;
            burst += s;
        }
        Ok(submitted)
    };

    if !honor_wait_for_start
        && build_decode.is_none()
        && let Some(fwd) = fwd_d.as_ref()
    {
        submitted = submit_ahead(fwd, submitted, 0)?;
    }

    step(&mut prologue_us, input.report_timing); // [4] build_submit (both passes)

    // ── HOST DRAIN (off the critical path — the decode burst is already in
    // the runtime while this first take waits on the prefill).
    let g0 = g0_ch.take_host::<i32>().await?;

    let ttft_us = input
        .report_timing
        .then(|| run_start.elapsed().as_micros() as u64);
    if input.report_timing && honor_wait_for_start {
        session::send("t0");
    }
    // The device arm stages its decode window now, with the first token out.
    if honor_wait_for_start
        && build_decode.is_none()
        && let Some(fwd) = fwd_d.as_ref()
    {
        submitted = submit_ahead(fwd, submitted, 0)?;
    }
    if let Some(build_decode) = build_decode.take() {
        fwd_d = build_decode(Channel::from([g0]))?;
        if let Some(fwd) = fwd_d.as_ref() {
            submitted = submit_ahead(fwd, submitted, 0)?;
        }
    }
    let mut intertoken_us: Vec<u32> = Vec::new();
    let mut last_tick = run_start.elapsed();
    let mut token_monotonic_ns: Vec<u64> = Vec::new();
    let first_token_monotonic_ns = input.report_arrivals.then(inferlet::monotonic_now_ns);

    let wasm_delay = std::time::Duration::from_micros(input.wasm_delay_us);
    let mut generated: Vec<u32> = Vec::with_capacity(input.max_tokens);
    let mut emitted = 0usize; // sampler emissions counted (stop excluded)

    let mut stopped = stop_tokens.contains(&(g0 as u32));
    if !stopped {
        emitted += 1;
        generated.push(g0 as u32);
        if let Some(now_ns) = first_token_monotonic_ns {
            token_monotonic_ns.push(now_ns);
        }
    }

    // Close as soon as all `budget` fires are submitted so the other lanes'
    // last frame seals do not wait on this lane's final host take.
    let mut closed = false;
    if submitted >= budget {
        pipe.close();
        closed = true;
    }
    let mut taken = 0usize;
    while taken < submitted {
        let t = out.take_host::<Vec<i32>>().await?;
        taken += 1;
        let Some(&t0) = t.first() else {
            return Err("out.take: empty tensor".into());
        };
        if stopped {
            continue; // a fire that settled before the close landed
        }
        if stop_tokens.contains(&(t0 as u32)) {
            stopped = true;
            if !closed {
                pipe.close();
                closed = true;
            }
            continue;
        }
        emitted += 1;
        generated.push(t0 as u32);
        if input.report_arrivals {
            let now_ns = inferlet::monotonic_now_ns();
            let previous_ns = *token_monotonic_ns
                .last()
                .expect("first accepted token has a monotonic stamp");
            intertoken_us.push(((now_ns - previous_ns) / 1_000) as u32);
            token_monotonic_ns.push(now_ns);
        } else if input.report_timing {
            let now = run_start.elapsed();
            intertoken_us.push((now - last_tick).as_micros() as u32);
            last_tick = now;
        }
        if input.wasm_delay_us > 0 {
            let until = std::time::Instant::now() + wasm_delay;
            while std::time::Instant::now() < until {
                std::hint::spin_loop();
            }
        }
        // Refill the window after draining an accepted token. A stopped lane
        // `continue`s above and never reaches here, so it never submits more.
        if submitted < budget && submitted - taken + live_slots.min(window_fires) <= window_fires {
            submitted = submit_ahead(
                fwd_d.as_ref().expect("decode pass exists while budget > 0"),
                submitted,
                taken,
            )?;
            if !closed && submitted >= budget {
                pipe.close();
                closed = true;
            }
        }
    }
    if !closed {
        pipe.close();
    }
    drop(fwd_d);
    drop(fwd_p);
    drop(ws);

    Ok(RunResult {
        num_prompt_tokens,
        num_output_tokens: emitted,
        tokens: generated,
        ttft_us,
        intertoken_us,
        token_monotonic_ns,
        prologue_us,
    })
}

/// The bench's decode only appends: its rows are final unless asked.
fn stated_final() -> bool {
    true
}

/// The decode fires after the prefill's sample, which is the first token:
/// a request for no tokens has nothing to sample and is refused.
fn decode_budget(max_tokens: usize) -> Result<usize> {
    max_tokens
        .checked_sub(1)
        .ok_or_else(|| "max_tokens must be at least 1: the prefill samples the first token".into())
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let main_start = std::time::Instant::now();
    decode_budget(input.max_tokens)?;

    // This program is pure: it reads its input, generates, and returns the
    // token counts in one envelope. Nothing is streamed and nothing outside
    // the process is touched, so a re-run is indistinguishable from the
    // first run and the planner may reclaim our KV pages by restarting us
    // instead of failing the request.
    //
    // The `wait_for_start` handshake is the one exception: it hands "ready"
    // to the harness and waits to be released, and the harness releases each
    // process exactly once. A restart there would wait forever, so that mode
    // keeps the fail-loud contract.
    if !input.wait_for_start {
        inferlet::runtime::declare_restartable();
    }

    let stop_tokens: Vec<u32> = if input.ignore_eos {
        Vec::new()
    } else {
        chat::stop_tokens()
    };

    let batch_len = input.prompt_tokens_batch.len().max(input.prompts.len());
    if batch_len > 0 {
        let mut prepared_prompt_tokens: Vec<Vec<u32>> = Vec::new();
        if input.wait_for_start && input.prompt_tokens_batch.is_empty() {
            prepared_prompt_tokens.reserve(batch_len);
            for i in 0..batch_len {
                let prompt = input
                    .prompts
                    .get(i)
                    .map(String::as_str)
                    .unwrap_or(input.prompt.as_str());
                let mut pt = chat::system_user(&input.system, prompt);
                pt.extend(chat::cue_thinking(!input.no_think));
                prepared_prompt_tokens.push(pt);
            }
        }
        let batch_concurrency = input
            .batch_concurrency
            .unwrap_or(batch_len)
            .clamp(1, batch_len);
        let lockstep = Lockstep::new(batch_len);
        let waves: Vec<std::ops::Range<usize>> = (0..batch_len)
            .step_by(batch_concurrency)
            .map(|offset| offset..(offset + batch_concurrency).min(batch_len))
            .collect();
        let mut lanes: Vec<Lane<'_>> = waves
            .iter()
            .flat_map(|wave| wave.clone().map(move |i| (i, wave.clone())))
            .map(|(i, wave)| {
                let prompt = input
                    .prompts
                    .get(i)
                    .map(String::as_str)
                    .unwrap_or(input.prompt.as_str());
                let prompt_tokens = if !prepared_prompt_tokens.is_empty() {
                    prepared_prompt_tokens.get(i).map(Vec::as_slice)
                } else {
                    input.prompt_tokens_batch.get(i).map(Vec::as_slice)
                };
                let lane: Lane<'_> = (
                    Box::pin(run_one(
                        &input,
                        prompt,
                        prompt_tokens,
                        &stop_tokens,
                        false,
                        0x51ed_0001u32.wrapping_add(i as u32),
                        None,
                        Some(LaneStep {
                            lockstep: &lockstep,
                            lane: i,
                            wave,
                        }),
                    )),
                    None,
                );
                lane
            })
            .collect();
        drive_until(&mut lanes, || lockstep.all(0..batch_len, BOUND)).await;
        if input.wait_for_start {
            session::send("ready");
            let _ = session::receive().await;
        }
        let mut request_prompt_tokens = Vec::with_capacity(batch_len);
        let mut request_output_tokens = Vec::with_capacity(batch_len);
        let mut first_tokens: Vec<u32> = Vec::new();
        // A failed lane fails the batch at once: the lanes still running are
        // dropped, which closes their pipelines and frees their pages, and
        // none is left waiting at a step for lanes that will not come.
        for wave in &waves {
            if failed(&lanes) {
                break;
            }
            let wave_lanes = &mut lanes[wave.clone()];
            drive_until(wave_lanes, || lockstep.all(wave.clone(), PIPED)).await;
            drive_until(wave_lanes, || lockstep.all(wave.clone(), SUBMITTED)).await;
            if failed(wave_lanes) {
                break;
            }
            // Every lane of the wave is past its prologue: what is left
            // waits on channel takes, host futures that wake the task.
            std::future::poll_fn(|cx| {
                let mut all_done = true;
                for (fut, res) in wave_lanes.iter_mut() {
                    if res.is_none() {
                        if let std::task::Poll::Ready(r) = fut.as_mut().poll(cx) {
                            *res = Some(r);
                        } else {
                            all_done = false;
                        }
                    }
                }
                if all_done || failed(wave_lanes) {
                    std::task::Poll::Ready(())
                } else {
                    std::task::Poll::Pending
                }
            })
            .await;
        }
        if let Some(Some(Err(error))) = lanes
            .iter_mut()
            .map(|(_, res)| res)
            .find(|res| matches!(res, Some(Err(_))))
            .map(Option::take)
        {
            return Err(error);
        }
        let mut request_token_ids = Vec::with_capacity(batch_len);
        for (i, (_, res)) in lanes.into_iter().enumerate() {
            let result = res.expect("lane completed")?;
            if i == 0 {
                first_tokens.clone_from(&result.tokens);
            }
            request_token_ids.push(result.tokens);
            request_prompt_tokens.push(result.num_prompt_tokens);
            request_output_tokens.push(result.num_output_tokens);
        }
        let text = if input.return_text {
            model::decode(&first_tokens).unwrap_or_default()
        } else {
            String::new()
        };
        return Ok(Output {
            num_prompt_tokens: request_prompt_tokens.iter().sum(),
            num_output_tokens: request_output_tokens.iter().sum(),
            text,
            token_ids: first_tokens,
            request_prompt_tokens,
            request_output_tokens,
            request_token_ids,
            ttft_us: None,
            intertoken_us: Vec::new(),
            token_monotonic_ns: Vec::new(),
            prologue_us: Vec::new(),
        });
    }

    let result = run_one(
        &input,
        &input.prompt,
        input.prompt_tokens.as_deref(),
        &stop_tokens,
        true,
        0x51ed_0000,
        Some(main_start.elapsed().as_micros() as u64),
        None,
    )
    .await?;
    let text = if input.return_text {
        model::decode(&result.tokens).unwrap_or_default()
    } else {
        String::new()
    };

    Ok(Output {
        num_prompt_tokens: result.num_prompt_tokens,
        num_output_tokens: result.num_output_tokens,
        text,
        token_ids: result.tokens,
        request_prompt_tokens: Vec::new(),
        request_token_ids: Vec::new(),
        request_output_tokens: Vec::new(),
        ttft_us: result.ttft_us,
        intertoken_us: result.intertoken_us,
        token_monotonic_ns: result.token_monotonic_ns,
        prologue_us: result.prologue_us,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `max_tokens: 0` is refused by name, not wrapped to a run past `usize::MAX`.
    #[test]
    fn no_tokens_is_refused() {
        let err = decode_budget(0).unwrap_err();
        assert!(err.contains("max_tokens must be at least 1"), "{err}");
        assert_eq!(decode_budget(1), Ok(0));
        assert_eq!(decode_budget(256), Ok(255));
    }
}
