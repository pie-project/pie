//! Text completion inferlet for benchmarking — PTIR bridge rewrite.
//!
//! Same harness contract as the classic version (exact token counts in the
//! final `Return` envelope, no per-token streaming, `ignore_eos` forces the
//! full `max_tokens` budget), expressed in the current architecture: an
//! N-wide prompt-prefill fire, then ONE decode pass whose sampled token is
//! device loop-carried, submitted `DEFAULT_RUNAHEAD_DEPTH` fires ahead of the
//! host drain — the sampler runs in-graph (argmax at temperature 0, otherwise
//! temperature-scaled TopP Gumbel-max), so the host never round-trips per
//! token.
//!
//! With stop tokens active (`ignore_eos = false`), up to depth-1 fires past
//! the stop are already submitted when the host observes it. They drain and
//! their outputs are ignored; token counts remain unchanged.
//!
//! Lifecycle (R4-4 single-pipeline): prefill and decode are ONE sequential
//! stream on ONE pipeline. The prefill epilogue hands its sampled token to
//! the decode pass on-device (`tok_in`, a device-only channel attached to
//! both passes) and mirrors it to the host on `g0`, so decode fires are
//! submitted immediately after the prefill submit — the g0 host round-trip
//! runs in parallel, off the critical path. `close()` ends the stream after
//! the submitted tail is drained.

use inferlet::ptir::prelude::*;
use inferlet::{Result, chat, model as wit_model, session};
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
    /// Accepted for input-compat; speculation is driver-side now.
    #[serde(default)]
    #[allow(dead_code)]
    system_speculation: Option<bool>,
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
            let scaled = div(logits, temperature);
            let probs = softmax(&scaled);
            let keep = pivot_threshold(probs, cummass_le(top_p));
            let neg_inf = broadcast(Tensor::constant(f32::NEG_INFINITY), [vocab]);
            let masked = select(&keep, &scaled, &neg_inf);
            let r = rng.take();
            let g = gumbel(&r, [vocab]);
            let r_next = add(&r, iota(2));
            rng.put(&r_next);
            reduce_argmax(add(masked, g))
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
        p.extend(chat::cue());
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

    let vocab = wit_model::output_vocab_size();
    let sampled_rng =
        (input.temperature > 0.0).then(|| Channel::from(vec![rng_seed, 0u32]).named("rng"));
    let prefill_rng = (input.temperature > 0.0)
        .then(|| Channel::from(vec![rng_seed ^ 0x9e37_79b9, 0u32]).named("rng_p"));
    let temperature = input.temperature;
    let top_p = input.top_p;

    let ws = WorkingSet::new();
    let page_size = ws.page_size();
    let n = prompt_vec.len() as u32;
    let max_pages = (n + input.max_tokens as u32 + 1).div_ceil(page_size);
    let reserve_to_tokens = |tokens: u32| -> std::result::Result<(), String> {
        let target = tokens.div_ceil(page_size).saturating_add(1).min(max_pages);
        let current = ws.page_len();
        if current < target {
            ws.reserve(target - current)?;
        }
        Ok(())
    };
    step(&mut prologue_us, input.report_timing); // [2] setup (vocab/rng/ws)
    reserve_to_tokens(n.max(1)).map_err(|e| format!("ws.reserve prompt: {e}"))?;
    step(&mut prologue_us, input.report_timing); // [3] reserve

    // ── ONE PIPELINE, ONE STREAM (R4-4): prefill then decode, in order.
    // The prefill epilogue seeds the decode's loop-carried `tok_in`
    // on-device and mirrors the token to the host via `out`, so the decode
    // fires ride the queue immediately behind the prefill — the host drain
    // runs in parallel off the critical path.
    let prompt_i32: Vec<i32> = prompt_vec.iter().map(|&t| t as i32).collect();
    let toks_p = Channel::from(prompt_i32).named("toks_p");
    let embed_indptr_p = Channel::from(vec![0u32, n]).named("embed_indptr_p");
    let positions_p = Channel::from((0..n).collect::<Vec<_>>()).named("positions_p");
    let pages_p = Channel::from((0..max_pages).collect::<Vec<_>>()).named("pages_p");
    let page_indptr_p = Channel::from(vec![0u32, n.div_ceil(page_size)]).named("page_indptr_p");
    let w_slot_p =
        Channel::from((0..n).map(|p| p / page_size).collect::<Vec<_>>()).named("w_slot_p");
    let w_off_p = Channel::from((0..n).map(|p| p % page_size).collect::<Vec<_>>()).named("w_off_p");
    let depth = DEFAULT_RUNAHEAD_DEPTH;
    // Decode fires: the prefill sample already spends one of the
    // `max_tokens` sampler activations.
    let budget = input.max_tokens - 1;
    // Device-carried handoff: unseeded, DEVICE-ONLY, attached to both
    // passes — the prefill's epilogue put is what fills it (full/empty bits
    // order the first decode read behind it). A host-visible channel cannot
    // attach to two passes, so the host copies ride separate per-pass
    // channels: `g0` (prefill) and `out` (decode). No annotation (F8): the
    // decode pass is CONSTRUCTED before the prefill submits, so its eager
    // embed claim on `tok_in` is visible to the prefill's build and the
    // channel derives device-only naturally.
    let tok_in = Channel::new([1], dtype::i32).named("tok_in");
    let g0_ch = Channel::new([1], dtype::i32).named("g0");
    let out = Channel::new([1], dtype::i32)
        .capacity(depth as u32)
        .named("out");

    let fwd_p = ForwardPass::new();
    fwd_p.embed(&toks_p, &embed_indptr_p)?;
    let kv_len_p = Channel::from(vec![n]).named("kv_len_p");
    fwd_p.attention(
        &ws,
        ..,
        ..,
        &kv_len_p,
        &pages_p,
        &page_indptr_p,
        &w_slot_p,
        &w_off_p,
        &positions_p,
        None,
    )?;
    fwd_p.epilogue(move || {
        let t = reshape(
            sample(intrinsics::logits(), vocab, temperature, top_p, prefill_rng),
            [1],
        );
        tok_in.put(&t);
        g0_ch.put(&t);
    });

    // Decode pass (1-wide, device loop-carried), built BEFORE the prefill
    // submits (its eager `tok_in` claim must precede the prefill's build)
    // and submitted run-ahead before the prefill has executed — its
    // geometry needs only the author-carried KvLen (from n+1, +1/fire), not
    // the prefill's token value, which flows through `tok_in`.
    let fwd_d: Option<ForwardPass> = if budget > 0 {
        let fwd = ForwardPass::new();
        let embed_indptr = Channel::from(vec![0u32, 1u32]).named("embed_indptr");
        let positions = Channel::from(vec![n]).named("positions");
        let pages = Channel::from((0..max_pages).collect::<Vec<_>>()).named("pages");
        let page_indptr =
            Channel::from(vec![0u32, (n + 1).div_ceil(page_size)]).named("page_indptr");
        let w_slot = Channel::from(vec![n / page_size]).named("w_slot");
        let w_off = Channel::from(vec![n % page_size]).named("w_off");
        fwd.embed(&tok_in, &embed_indptr)?;
        let kv_len = Channel::from(vec![n + 1]).named("kv_len");
        fwd.attention(
            &ws,
            ..,
            (n / page_size)..,
            &kv_len,
            &pages,
            &page_indptr,
            &w_slot,
            &w_off,
            &positions,
            None,
        )?;
        fwd.epilogue(move || {
            let length = kv_len.take().tensor();
            let t = reshape(
                sample(intrinsics::logits(), vocab, temperature, top_p, sampled_rng),
                [1],
            );
            let next_length = add(&length, 1u32);
            let page_count = div(add(&next_length, page_size - 1), page_size);
            tok_in.put(&t);
            kv_len.put(&next_length);
            positions.put(&length);
            w_slot.put(div(&length, page_size));
            w_off.put(rem(&length, page_size));
            page_indptr.take();
            page_indptr.put(mul(iota(2), broadcast(&page_count, [2])));
            out.put(&t);
        });
        Some(fwd)
    } else {
        None
    };

    let pipe = Pipeline::new();
    fwd_p
        .submit(&pipe)
        .map_err(|e| format!("prefill submit: {e}"))?;

    let mut submitted = 0usize;
    if let Some(fwd) = &fwd_d {
        while submitted < depth.min(budget) {
            reserve_to_tokens(n + submitted as u32 + 1)
                .map_err(|e| format!("reserve decode burst: {e}"))?;
            fwd.submit(&pipe)
                .map_err(|e| format!("decode submit: {e}"))?;
            submitted += 1;
        }
    }
    step(&mut prologue_us, input.report_timing); // [4] build_submit (both passes)

    // ── HOST DRAIN (off the critical path — the decode burst is already in
    // the engine while this first take waits on the prefill).
    let g0 = g0_ch
        .take()
        .get::<i32>()
        .await
        .map_err(|e| format!("g0 take: {e}"))?[0];

    let ttft_us = input
        .report_timing
        .then(|| run_start.elapsed().as_micros() as u64);
    if input.report_timing && honor_wait_for_start {
        session::send("t0");
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

    let mut taken = 0usize;
    while taken < submitted {
        let t = out
            .take()
            .get::<i32>()
            .await
            .map_err(|e| format!("out.take: {e}"))?;
        taken += 1;
        let Some(&t0) = t.first() else {
            return Err("out.take: empty tensor".into());
        };
        if stopped {
            continue; // a fire that settled before the close landed
        }
        if stop_tokens.contains(&(t0 as u32)) {
            stopped = true;
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
        if submitted < budget {
            let fwd = fwd_d.as_ref().expect("decode pass exists while budget > 0");
            reserve_to_tokens(n + submitted as u32 + 1)
                .map_err(|e| format!("reserve decode continuation: {e}"))?;
            fwd.submit(&pipe)
                .map_err(|e| format!("decode submit: {e}"))?;
            submitted += 1;
        }
    }
    pipe.close();

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
                    pt.extend(chat::cue());
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
            wit_model::decode(&first_tokens).unwrap_or_default()
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
        wit_model::decode(&result.tokens).unwrap_or_default()
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
