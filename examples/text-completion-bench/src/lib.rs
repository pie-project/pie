//! Text completion for benchmarking: exact token counts, no per-token streaming.

use inferlet::eta::prelude::*;
use inferlet::{chat, session};
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
    /// When true, drop the chat-template stop tokens so the generator runs to
    /// `max_tokens` regardless of model emit.
    #[serde(default)]
    ignore_eos: bool,
    /// Turn a thinking model's thinking off, as the model's own template does
    /// for `enable_thinking: false`.
    #[serde(default)]
    no_think: bool,
    /// Busy-wait inside WASM for this many microseconds per drained token —
    /// simulates per-token inferlet work.
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
    /// For a plane whose channel rings belong to the pass instance, not the channel.
    #[serde(default)]
    host_first_token: bool,
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

/// In-graph sample from `logits`: argmax at `temperature <= 0`, otherwise
/// temperature-scaled TopP Gumbel-max (`keep = pivot_threshold(softmax(l/T),
/// cummass_le(p)); argmax(select(keep, l/T, -inf) + gumbel(r))`).
fn sample(
    logits: Tensor,
    vocab: u32,
    temperature: f32,
    top_p: f32,
    rng: Option<Channel>,
) -> Tensor {
    match rng {
        None => reduce_argmax(logits),
        Some(rng) => {
            let scaled = logits / temperature;
            let probs = softmax(&scaled);
            let keep = pivot_threshold(probs, cummass_le(top_p));
            let neg_inf = broadcast(f32::NEG_INFINITY, [vocab]);
            let masked = select(&keep, &scaled, &neg_inf);
            let r = rng.take();
            let g = gumbel(&r, [vocab]);
            let r_next = &r + iota(2);
            rng.put(&r_next);
            reduce_argmax(masked + g)
        }
    }
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
) -> Result<RunResult> {
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
    let sampled_rng = (input.temperature > 0.0).then(|| Channel::from([rng_seed, 0u32]));
    let prefill_rng =
        (input.temperature > 0.0).then(|| Channel::from([rng_seed ^ 0x9e37_79b9, 0u32]));
    let temperature = input.temperature;
    let top_p = input.top_p;

    let ws = WorkingSet::new();
    let n = prompt_vec.len() as u32;
    step(&mut prologue_us, input.report_timing); // [2] setup (vocab/rng/ws)
    // Nothing to reserve; the empty step keeps `prologue_us`'s shape.
    step(&mut prologue_us, input.report_timing); // [3] reserve

    let prompt_i32: Vec<i32> = prompt_vec.iter().map(|&t| t as i32).collect();

    let toks_p = Channel::from(&prompt_i32[..]);
    let embed_indptr_p = Channel::from([0u32, n]);
    // Decode fires: the prefill sample already spends one of the
    // `max_tokens` sampler activations.
    let budget = input.max_tokens - 1;
    // A host-visible channel cannot attach to two passes, so this one is device-only.
    let tok_in = (!input.host_first_token).then(|| Channel::new([1], dtype::i32));
    let g0_ch = Channel::new([1], dtype::i32);
    let live_slots = frame_size();

    // A frame of margin above the window (`7 * live_slots` covers every frame size): the ticket
    // check silently skips a continuation that lands inside the staging margin.
    let (window_fires, out_capacity) = match input.run_ahead_frames {
        Some(r) => {
            let w = r.max(1) * live_slots;
            (w, w + 1 + 7 * live_slots)
        }
        None => {
            let cap = channel_capacity();
            (cap - 1, cap + 7 * live_slots)
        }
    };
    let out = Channel::new([1], dtype::i32).capacity(out_capacity as u32);
    let pipe = Pipeline::new();

    let fwd_p = ForwardPass::new();
    fwd_p.embed(&toks_p, &embed_indptr_p)?;
    fwd_p
        .state(&[&ws], StateGeometry::folding())
        .context("bind prefill state")?;
    fwd_p.epilogue(move || {
        let t = reshape(
            sample(
                intrinsics::output(0, "logits"),
                vocab,
                temperature,
                top_p,
                prefill_rng,
            ),
            [1],
        );
        if let Some(tok_in) = tok_in {
            tok_in.put(&t);
        }
        g0_ch.put(&t);
    });

    // Built before the prefill submits, so its `tok_in` claim precedes the prefill's build.
    let ws_ref = &ws;
    let build_decode = move |tok_in: Channel| -> std::result::Result<Option<ForwardPass>, String> {
        Ok(if budget > 0 {
            let fwd = ForwardPass::new();
            let embed_indptr = Channel::from([0u32, 1u32]);
            fwd.embed(&tok_in, &embed_indptr)?;
            fwd.state(&[ws_ref], StateGeometry::folding())
                .context("bind decode state")?;
            fwd.epilogue(move || {
                let t = reshape(
                    sample(
                        intrinsics::output(0, "logits"),
                        vocab,
                        temperature,
                        top_p,
                        sampled_rng,
                    ),
                    [1],
                );
                tok_in.put(&t);
                out.put(&t);
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
    let first_decodes = 0;
    let mut first_slots: Vec<Option<&ForwardPass>> = Vec::with_capacity(live_slots);
    first_slots.push(Some(&fwd_p));
    for _ in 0..first_decodes {
        first_slots.push(Some(fwd_d.as_ref().expect("decode pass exists")));
    }
    submit_frame(&pipe, &first_slots).context("first frame submit")?;
    let mut submitted = first_decodes;

    // The pass is an argument: under `host_first_token` it is built after this closure.
    let submit_ahead = |fwd: &ForwardPass,
                        mut submitted: usize,
                        drained: usize|
     -> std::result::Result<usize, String> {
        while submitted < budget {
            let s = (budget - submitted).min(live_slots);
            if submitted - drained + s > window_fires {
                break;
            }
            let slots: Vec<Option<&ForwardPass>> = (0..s).map(|_| Some(fwd)).collect();
            submit_frame(&pipe, &slots).context("decode frame submit")?;
            submitted += s;
        }
        Ok(submitted)
    };

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
    if build_decode.is_none()
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

    // Close early: the other lanes' frame seals stop waiting on this one.
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
            std::thread::sleep(wasm_delay);
        }
        // Refill the window after draining an accepted token. A stopped lane
        // `continue`s above and never reaches here, so it never submits more.
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
    if !closed {
        pipe.close();
    }

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

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let main_start = std::time::Instant::now();

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
        if input.wait_for_start {
            prepared_prompt_tokens.reserve(batch_len);
            if input.prompt_tokens_batch.is_empty() {
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
            session::send("ready");
            let _ = session::receive().await;
        }
        let mut request_prompt_tokens = Vec::with_capacity(batch_len);
        let mut request_output_tokens = Vec::with_capacity(batch_len);
        let mut first_tokens: Vec<u32> = Vec::new();
        let batch_concurrency = input
            .batch_concurrency
            .unwrap_or(batch_len)
            .clamp(1, batch_len);
        let mut offset = 0usize;
        while offset < batch_len {
            let end = (offset + batch_concurrency).min(batch_len);
            let futures = (offset..end).map(|i| {
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
                run_one(
                    &input,
                    prompt,
                    prompt_tokens,
                    &stop_tokens,
                    false,
                    0x51ed_0001u32.wrapping_add(i as u32),
                    None,
                )
            });
            for (j, result) in futures::future::join_all(futures)
                .await
                .into_iter()
                .enumerate()
            {
                let result = result?;
                if offset + j == 0 {
                    first_tokens = result.tokens.clone();
                }
                request_prompt_tokens.push(result.num_prompt_tokens);
                request_output_tokens.push(result.num_output_tokens);
            }
            offset = end;
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
        request_output_tokens: Vec::new(),
        ttft_us: result.ttft_us,
        intertoken_us: result.intertoken_us,
        token_monotonic_ns: result.token_monotonic_ns,
        prologue_us: result.prologue_us,
    })
}
