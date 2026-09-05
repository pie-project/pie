//! Text completion inferlet for benchmarking — ETA bridge rewrite.
//!
//! Same harness contract as the classic version (exact token counts in the
//! final `Return` envelope, no per-token streaming, `ignore_eos` forces the
//! full `max_tokens` budget), expressed in the current architecture: an
//! N-wide prompt-prefill fire, then ONE decode pass whose sampled token is
//! device loop-carried, kept two fires ahead of the host drain by a single
//! frame-quantized run-ahead discipline (`top_up` — the same window rule at
//! every frame size k) — the sampler runs in-graph (argmax at temperature 0,
//! otherwise temperature-scaled TopP Gumbel-max), so the host never
//! round-trips per token.
//!
//! With stop tokens active (`ignore_eos = false`), the fires already staged
//! ahead when the host observes the stop drain and their outputs are ignored;
//! token counts remain unchanged.
//!
//! Lifecycle (R4-4 single-pipeline): prefill and decode are ONE sequential
//! stream on ONE pipeline. The prefill epilogue hands its sampled token to
//! the decode pass on-device (`tok_in`, a device-only channel attached to
//! both passes) and mirrors it to the host on `g0`, so decode fires are
//! submitted immediately after the prefill submit — the g0 host round-trip
//! runs in parallel, off the critical path. `close()` ends the stream after
//! the submitted tail is drained.
//!
//! `host_first_token` is the one arm that departs from that: the first token
//! travels out on `g0` and back as the decode's SEED, and the window is staged
//! behind that drain rather than in front of it. Everything after the first
//! token is device loop-carried either way. It exists for a plane whose
//! channel rings belong to the INSTANCE and not to the channel — see the input
//! field.

use inferlet::eta::attention::prelude::*;
use inferlet::{chat, session};
use serde::{Deserialize, Serialize};
use std::ops::RangeBounds;

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
    /// Attach a per-attention-layer `on_attn` counter tap to the decode pass
    /// (a genuine hook program, numerics untouched) and return the final
    /// count — layers-that-fired x fires. On hybrid models this is the direct
    /// probe that the engine executes hook stages on attention layers only.
    #[serde(default)]
    hook_fold: bool,
    /// Accepted for input-compat; speculation is engine-side now.
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
    /// Frames of decode run-ahead each lane keeps queued in the runtime.
    /// Leave unset to use the runtime's own sizing (`channel-capacity()`),
    /// which is what every other inferlet does; set it only to sweep the
    /// depth against a measurement. The wave quorum waits for every lane, so
    /// a lane that drops to one queued frame puts its whole guest turnaround
    /// on the fleet's critical path; deeper queues buy slack for that
    /// turnaround.
    #[serde(default)]
    run_ahead_frames: Option<usize>,
    /// **Send the FIRST token through the host instead of across the device.**
    ///
    /// Default `false`, which is the shape this file was written for and the
    /// one design §5 promises: `tok_in` is a DEVICE-ONLY channel attached to
    /// both passes, the prefill epilogue's put fills it, and the pipeline FIFO
    /// orders the first decode read behind that put — so the whole decode
    /// window can be submitted before the prefill has even run.
    ///
    /// That promise needs ONE DEVICE RING for the two instances to meet on.
    /// `engine-cuda` gave the ring to the channel in `27de300fa` (gate:
    /// `tests/gpu/tests/cuda_shared_device_ring.rs`); `engine-metal` has NOT
    /// mirrored it — `api::register_channel` is `Unsupported` there and every
    /// instance's rings are carved inside `Session::bind`, so a channel two
    /// passes share is TWO rings: the prefill's put lands in one and the
    /// decode's `embed` reads the other, forever empty. Every request died at
    /// the first decode frame in ~30 ms, with `serve::prepare` refusing
    /// "channel 0 is empty and its program takes from it".
    ///
    /// Set it and the first token goes out through `g0` and comes back as the
    /// decode's `tok_in` SEED, which is a shape this plane does serve
    /// (`hook_fold_acc` is the same one: seeded, one pass, taken and put by
    /// its own stages). Every token after the first still stays on the device,
    /// because the decode's own epilogue puts into its own instance's ring.
    /// What it costs is one host hop per REQUEST, on the TTFT path, and the
    /// decode run-ahead window moves behind that hop rather than in front of
    /// it.
    ///
    /// It is a knob and not the default because the default is what CUDA
    /// measures, and moving the window behind the first drain would change
    /// every published number on the plane that does not need it. The knob
    /// retires the day `engine-metal` mirrors `27de300fa`.
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
    /// Final `on_attn` tap count (hook_fold only): attention-layer stage
    /// executions summed over every settled decode fire.
    #[serde(skip_serializing_if = "Option::is_none")]
    hook_fold_final: Option<u32>,
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
    hook_fold_final: Option<u32>,
}

/// `run_one`, generated once per forward-pass kind.
///
/// `pie:inferlet` now exposes three forward interfaces, so `ForwardPass` is
/// three unrelated types and this benchmark -- which drives BOTH a
/// pure-attention model and a hybrid GDN model -- can no longer be one
/// function. The runtime `pass_kind()` branch that used to decide whether to
/// bind recurrent state has moved UP to a `model::pass_kind()` branch over two
/// monomorphisations. The body is written once here and expanded twice, so the
/// two versions cannot drift.
/// The state binding this benchmark needs, over the two forward interfaces it
/// runs on. `pie:inferlet` gives them separate `attention` signatures so that
/// an attention-only algorithm cannot name a folded recurrent state; saying
/// what the two have IN COMMON for THIS algorithm is the guest's job.
trait BindState {
    fn bind_state<R, W>(
        &self,
        ws: &WorkingSet,
        geom: KvGeometry<'_, R, W>,
        rs: &[RsWorkingSet],
    ) -> ::std::result::Result<(), String>
    where
        R: RangeBounds<u32>,
        W: RangeBounds<u32>;
}

impl BindState for inferlet::eta::attention::ForwardPass {
    fn bind_state<R, W>(
        &self,
        ws: &WorkingSet,
        geom: KvGeometry<'_, R, W>,
        rs: &[RsWorkingSet],
    ) -> ::std::result::Result<(), String>
    where
        R: RangeBounds<u32>,
        W: RangeBounds<u32>,
    {
        // A pure-attention model has no recurrent state, and the type system
        // now proves this set can only be empty.
        debug_assert!(rs.is_empty());
        self.attention(ws, geom)
    }
}

impl BindState for inferlet::eta::hybrid::ForwardPass {
    fn bind_state<R, W>(
        &self,
        ws: &WorkingSet,
        geom: KvGeometry<'_, R, W>,
        rs: &[RsWorkingSet],
    ) -> ::std::result::Result<(), String>
    where
        R: RangeBounds<u32>,
        W: RangeBounds<u32>,
    {
        // The benchmark never buffers: every fire folds straight into the
        // recurrence.
        self.attention(
            Some(KvBinding {
                working_set: ws,
                geometry: geom,
            }),
            rs,
            RsGeometry {
                fold_len: None,
                buffer: 0..0,
            },
        )
    }
}

macro_rules! define_run_one {
    ($name:ident, $kind:ident) => {
        async fn $name(
            input: &Input,
            prompt: &str,
            prompt_tokens: Option<&[u32]>,
            stop_tokens: &[u32],
            honor_wait_for_start: bool,
            rng_seed: u32,
            main_pre_us: Option<u64>,
        ) -> Result<RunResult> {
            use inferlet::eta::$kind::{ForwardPass, submit_frame};

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

            let vocab = model::output_vocab_size();
            let sampled_rng =
                (input.temperature > 0.0).then(|| Channel::from([rng_seed, 0u32]).named("rng"));
            let prefill_rng = (input.temperature > 0.0)
                .then(|| Channel::from([rng_seed ^ 0x9e37_79b9, 0u32]).named("rng_p"));
            let temperature = input.temperature;
            let top_p = input.top_p;

            let ws = WorkingSet::new();
            let page_size = kv_page_size();
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
            reserve_to_tokens(n.max(1)).context("ws.reserve prompt")?;
            step(&mut prologue_us, input.report_timing); // [3] reserve

            // ── ONE PIPELINE, ONE STREAM (R4-4): prefill then decode, in order.
            // The prefill epilogue seeds the decode's loop-carried `tok_in`
            // on-device and mirrors the token to the host via `out`, so the decode
            // fires ride the queue immediately behind the prefill — the host drain
            // runs in parallel off the critical path.
            let prompt_i32: Vec<i32> = prompt_vec.iter().map(|&t| t as i32).collect();

            // Prefill is chunked against the engine's per-launch token capacity
            // `max_embed_length()` (C). A single pass over the whole prompt is
            // rejected outright once L > C —
            //     "forward request has 1047 forward tokens, exceeding engine limit 1024"
            // — and C is derived from the memory plan, so on a model whose weights
            // leave a small workspace (Qwen3.6-35B-A3B: 71.9 GB on an 80 GB card)
            // C stays at 1024 and no `--gpu-mem-util` / `--memory-profile` setting
            // raises it. Chunking is guest-side by design (see model.wit
            // `max-embed-length`), and `prefill_chunks` is the same SDK splitter
            // naive-baseline and the trackb/quest/tova inferlets use, so chunk
            // boundaries match theirs.
            //
            // Only the LAST chunk carries the sampling epilogue that seeds the
            // decode's loop-carried `tok_in`; earlier chunks just extend the KV
            // and are submitted straight to the pipeline. When the prompt fits in
            // one chunk the loop body runs zero times and the pass built below is
            // byte-for-byte what this inferlet built before.
            let spans = prefill_chunks(n, None);
            let (last_base, last_end) = *spans.last().expect("prefill_chunks is non-empty");

            let toks_p = Channel::from(&prompt_i32[last_base as usize..last_end as usize])
                .named("toks_p");
            let embed_indptr_p = Channel::from([0u32, last_end - last_base]).named("embed_indptr_p");
            let positions_p = Channel::from_iter(last_base..last_end).named("positions_p");
            let pages_p = Channel::from_iter(0..max_pages).named("pages_p");
            let page_indptr_p = Channel::from([0u32, n.div_ceil(page_size)]).named("page_indptr_p");
            let w_slot_p =
                Channel::from_iter((last_base..last_end).map(|p| p / page_size)).named("w_slot_p");
            let w_off_p =
                Channel::from_iter((last_base..last_end).map(|p| p % page_size)).named("w_off_p");
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
            //
            // **UNLESS `host_first_token` SAYS OTHERWISE**, in which case the
            // channel is not built here at all: it is seeded from `g0` below,
            // after the drain, and attaches to the decode pass alone. See the
            // input field for the engine that needs that and why.
            let k = frame_size();
            let device_handoff = !input.host_first_token;
            let tok_in = device_handoff.then(|| Channel::new([1], dtype::i32).named("tok_in"));
            let g0_ch = Channel::new([1], dtype::i32).named("g0");
            // Live slots per frame. Always k now: an RS mapping publishes at prepare,
            // in slot order, so a linear model fills a frame exactly like a dense one.
            let live_slots = live_slots();

            // Run-ahead window in FIRES and the take-side ring that has to hold it.
            //
            // The runtime owns this sizing: `channel-capacity()` is the ring size in
            // cells with the staging margin already baked in, so the window is
            // exactly one less. The window is a whole number of frames, and the
            // top-up below adds a whole frame at a time starting from below the
            // window, so `submitted - taken` peaks at exactly `window_fires` — one
            // cell short of the ring even when every outstanding fire settles
            // before the host takes. Undersizing the ring does not merely throttle,
            // it POISONS the channel ("work item completion published Failed
            // terminal outcome") once a publish finds no free cell.
            //
            // `run_ahead_frames` overrides the depth for sweeps only; it is stated
            // in frames, and the ring grows with it.
            //
            // The ring is sized a full frame of margin ABOVE the advertised
            // capacity (`7 * live_slots` covers every frame size this bench
            // runs). `channel-capacity()` bakes in a staging margin, but the
            // runtime's ticket check is more conservative still, and a
            // continuation that lands inside that margin is silently skipped at
            // the reader-cell validation rather than refused. Sized at exactly
            // `cap`, the measured continuation engaged on 12% of frames — the
            // run-ahead the whole pipeline depends on, lost with no error
            // anywhere. Chaining collapses and every frame's guest turnaround
            // lands on the critical path: c256 23,478 -> 19,321 tok/s, S 30,277
            // -> 21,506, X 5,429 -> 4,039 (analysis.md 10.30).
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
            let out = Channel::new([1], dtype::i32)
                .capacity(out_capacity as u32)
                .named("out");
            // The tap's running count lives in a stage-only ring (`acc` —
            // SPSC forbids a host take on a stage-taken channel), so the
            // epilogue mirrors it into a host-Reader channel drained once
            // per fire alongside `out`.
            let hook_fold_chans = input.hook_fold.then(|| {
                (
                    Channel::from([0u32]).named("hook_fold_acc"),
                    Channel::new([1], dtype::u32)
                        .capacity(out_capacity as u32)
                        .named("hook_fold_out"),
                )
            });

            // Hybrid/recurrent architectures (Qwen3.6's GDN layers, Mamba2) carry a
            // folded recurrent state per request alongside the KV pages, and the
            // engine rejects a forward whose rs-working-set count doesn't match its
            // request rows. One row per fire here, so one set — reused by BOTH passes
            // so the decode continues the prefill's state rather than starting cold.
            // `rs_state_size() == 0` on pure-attention models, which bind none.
            // `pass_kind() != Attention` is the class predicate: true iff the model
            // folds a recurrent state, which is exactly when the engine requires one
            // rs-working-set per request row.
            let rs_ws: Vec<RsWorkingSet> = if model::pass_kind() != model::ForwardKind::Attention {
                vec![RsWorkingSet::new()]
            } else {
                Vec::new()
            };

            // One pipeline for the whole prefill+decode program, created here so
            // the leading prefill chunks (below) and the first frame share it and
            // stay ordered.
            let pipe = Pipeline::new();

            // Leading prefill chunks: everything except the last span. They only
            // need to extend the KV, but a pass with no epilogue is rejected at
            // registration (`pie_cuda_register_program failed with status -1`), so
            // each one samples into a throwaway channel that is drained right
            // after — the same shape naive-baseline / quest / tova use. Only the
            // last span's token continues the prompt; these are discarded. The
            // loop body runs zero times whenever the prompt fits in one chunk.
            for &(base, end) in &spans[..spans.len() - 1] {
                let toks_c =
                    Channel::from(&prompt_i32[base as usize..end as usize]).named("toks_p");
                let embed_indptr_c = Channel::from([0u32, end - base]).named("embed_indptr_p");
                let positions_c = Channel::from_iter(base..end).named("positions_p");
                let pages_c = Channel::from_iter(0..max_pages).named("pages_p");
                let page_indptr_c =
                    Channel::from([0u32, end.div_ceil(page_size)]).named("page_indptr_p");
                let w_slot_c =
                    Channel::from_iter((base..end).map(|p| p / page_size)).named("w_slot_p");
                let w_off_c =
                    Channel::from_iter((base..end).map(|p| p % page_size)).named("w_off_p");
                let kv_len_c = Channel::from([end]).named("kv_len_p");
                let fwd_c = ForwardPass::new();
                fwd_c.embed(&toks_c, &embed_indptr_c)?;
                fwd_c
                    .bind_state(
                        &ws,
                        KvGeometry {
                            readable_pages: ..,
                            writable_pages: ..,
                            kv_len: &kv_len_c,
                            pages: &pages_c,
                            page_indptr: &page_indptr_c,
                            w_slot: &w_slot_c,
                            w_off: &w_off_c,
                            positions: &positions_c,
                            mask: None,
                        },
                        &rs_ws,
                    )
                    .with_context(|| format!("bind prefill chunk @{base}"))?;
                let drop_tok_c = Channel::new([1], dtype::i32).named("drop_tok_c");
                let drop_sink = drop_tok_c.clone();
                fwd_c.epilogue(move || {
                    let t = reshape(
                        sample(intrinsics::logits(), vocab, temperature, top_p, prefill_rng),
                        [1],
                    );
                    drop_sink.put(&t);
                });
                fwd_c
                    .submit(&pipe)
                    .with_context(|| format!("prefill chunk submit @{base}"))?;
                // Must be drained: an epilogue put that is never taken fills the
                // channel and stalls the lane.
                drop_tok_c
                    .take_host::<i32>()
                    .await
                    .with_context(|| format!("drain prefill chunk @{base}"))?;
            }

            let fwd_p = ForwardPass::new();
            fwd_p.embed(&toks_p, &embed_indptr_p)?;
            let kv_len_p = Channel::from([n]).named("kv_len_p");
            fwd_p
                .bind_state(
                    &ws,
                    KvGeometry {
                        readable_pages: ..,
                        writable_pages: ..,
                        kv_len: &kv_len_p,
                        pages: &pages_p,
                        page_indptr: &page_indptr_p,
                        w_slot: &w_slot_p,
                        w_off: &w_off_p,
                        positions: &positions_p,
                        mask: None,
                    },
                    &rs_ws,
                )
                .context("bind prefill state")?;
            fwd_p.epilogue(move || {
                let t = reshape(
                    sample(intrinsics::logits(), vocab, temperature, top_p, prefill_rng),
                    [1],
                );
                if let Some(tok_in) = tok_in {
                    tok_in.put(&t);
                }
                g0_ch.put(&t);
            });

            // Decode pass (1-wide, device loop-carried), built BEFORE the prefill
            // submits (its eager `tok_in` claim must precede the prefill's build)
            // and submitted run-ahead before the prefill has executed — its
            // geometry needs only the author-carried KvLen (from n+1, +1/fire), not
            // the prefill's token value, which flows through `tok_in`.
            //
            // **A BUILDER RATHER THAN A BLOCK, BECAUSE THE OTHER ARM BUILDS IT
            // LATER.** Under `host_first_token` the channel it embeds is seeded
            // with the token the prefill produced, so the pass cannot exist
            // until that token has been drained. Everything it moves — `out`,
            // `sampled_rng`, the fold channels — is moved ONCE, which is what
            // makes this `FnOnce` and what makes calling it in exactly one of
            // the two places below a fact the compiler checks.
            // `ws` and `rs_ws` are the PREFILL's too, so the `move` below takes
            // references to them and not the values.
            let ws_ref = &ws;
            let rs_ws_ref = &rs_ws;
            let build_decode = move |tok_in: Channel| -> std::result::Result<Option<ForwardPass>, String> {
            Ok(if budget > 0 {
                let fwd = ForwardPass::new();
                let embed_indptr = Channel::from([0u32, 1u32]).named("embed_indptr");
                let positions = Channel::from([n]).named("positions");
                let pages = Channel::from_iter(0..max_pages).named("pages");
                let page_indptr =
                    Channel::from([0u32, (n + 1).div_ceil(page_size)]).named("page_indptr");
                let w_slot = Channel::from([n / page_size]).named("w_slot");
                let w_off = Channel::from([n % page_size]).named("w_off");
                fwd.embed(&tok_in, &embed_indptr)?;
                let kv_len = Channel::from([n + 1]).named("kv_len");
                fwd.bind_state(
                    ws_ref,
                    KvGeometry {
                        readable_pages: ..,
                        writable_pages: (n / page_size)..,
                        kv_len: &kv_len,
                        pages: &pages,
                        page_indptr: &page_indptr,
                        w_slot: &w_slot,
                        w_off: &w_off,
                        positions: &positions,
                        mask: None,
                    },
                    rs_ws_ref,
                )
                .context("bind decode state")?;
                if let Some((acc, _)) = hook_fold_chans {
                    // Take/put per attention layer: the fold keeps the stage
                    // classifiable (a publish-only body reads as Unknown) and
                    // the running count is the evidence the stage EXECUTED —
                    // attention layers x fires.
                    fwd.on_attn(move || {
                        let prev = acc.take();
                        acc.put(&prev + 1u32);
                    });
                }
                fwd.epilogue(move || {
                    let length = kv_len.take();
                    let t = reshape(
                        sample(intrinsics::logits(), vocab, temperature, top_p, sampled_rng),
                        [1],
                    );
                    let next_length = &length + 1u32;
                    let page_count = next_length.div_ceil(page_size);
                    tok_in.put(&t);
                    kv_len.put(&next_length);
                    positions.put(&length);
                    w_slot.put(&length / page_size);
                    w_off.put(&length % page_size);
                    page_indptr.put(indptr(1, &page_count));
                    out.put(&t);
                    if let Some((acc, fold_out)) = hook_fold_chans {
                        let count = acc.take();
                        acc.put(&count);
                        fold_out.put(&count);
                    }
                });
                Some(fwd)
            } else {
                None
            })
            };

            // (`pipe` was created before the leading prefill chunks so they and
            // this frame share one pipeline and stay ordered.)

            let mut build_decode = Some(build_decode);
            let mut fwd_d: Option<ForwardPass> = match tok_in {
                Some(tok_in) => build_decode
                    .take()
                    .expect("the decode builder is called once")(tok_in)?,
                None => None,
            };

            // First frame: the prefill chunk in slot 0, then up to live_slots-1
            // decode slots. At live_slots = 1 this is a bare prefill submit
            // (`submit` IS a single-slot frame); trailing slots pad to no-ops.
            // Under `host_first_token` there is no decode pass yet, so the
            // frame is one slot wide — the same shape live_slots = 1 already
            // submits.
            let first_decodes = if fwd_d.is_some() {
                budget.min(live_slots - 1)
            } else {
                0
            };
            reserve_to_tokens(n + first_decodes as u32 + 1).context("reserve first frame")?;
            let mut first_slots: Vec<Option<&ForwardPass>> = Vec::with_capacity(k);
            first_slots.push(Some(&fwd_p));
            for _ in 0..first_decodes {
                first_slots.push(Some(fwd_d.as_ref().expect("decode pass exists")));
            }
            submit_frame(&pipe, &first_slots).context("first frame submit")?;
            let mut submitted = first_decodes;

            // Unified run-ahead discipline (ONE rule for every k): submit frames of
            // min(live_slots, remaining) decode slots until the window (submitted
            // minus drained) is full or the budget is spent, returning the new
            // submitted count. Decode geometry is device-carried (`tok_in`), so a
            // successor never waits on a sampled token.
            //
            // `window_fires` comes from the runtime (see the `channel_capacity()`
            // computation above). What it has to buy is COVER: the frame seal is
            // wait-for-ALL, so it holds until every awaited lane's oldest queued
            // frame is arrival-complete and the binding term is the SLOWEST of
            // `concurrency` lanes' resubmit. Cover (`window_fires - live_slots`) is
            // what gives that straggler time, and it is counted in WAVES, not
            // frames — a 2-frame window gives only k waves, so k = 1 got a single
            // ~7 ms wave and a cohort that fell behind never caught back up
            // (see §20.10). That is why the runtime's `HOST_TURNAROUND_WAVES` is
            // k-independent and the frame count is derived from it.
            //
            // Measured, Qwen3-0.6B / L40S / conc 256, median of 6-8 trials:
            //   k=1 cover 1 (old 2*k rule)  multimodal 18.8k / 22k / 28.5k
            //   k=1 cover 2                 7/8 at 28.6-29.5k, 1/8 still collapsed
            //   k=1 cover 3                 6/6 at 28.7-29.5k, median 29044
            //   k=2 cover 2 (old rule)      median 28497   <- already saturated
            //   k=2 cover 3                 median 28654   <- within noise of it
            // **THE PASS IS AN ARGUMENT AND NOT A CAPTURE**, because under
            // `host_first_token` it is assigned after this closure is defined.
            let submit_ahead = |fwd: &ForwardPass,
                                mut submitted: usize,
                                drained: usize|
             -> std::result::Result<usize, String> {
                    // Top up only while a WHOLE frame still fits inside the window.
                    // Testing `submitted - drained < window_fires` instead would let
                    // a frame start from `window_fires - 1` and overshoot to
                    // `window_fires + live_slots - 1`, which is past the ring and is
                    // rejected outright at k >= 3 ("frame would need 8 reader cells,
                    // capacity 7"). This is the same discipline `eta::run_ahead`
                    // uses, and it is what makes `channel_capacity()`'s single spare
                    // cell the correct margin.
                    while submitted < budget {
                        let s = (budget - submitted).min(live_slots);
                        if submitted - drained + s > window_fires {
                            break;
                        }
                        reserve_to_tokens(n + (submitted + s) as u32 + 1)
                            .context("reserve decode frame")?;
                        let slots: Vec<Option<&ForwardPass>> = (0..s).map(|_| Some(fwd)).collect();
                        submit_frame(&pipe, &slots).context("decode frame submit")?;
                        submitted += s;
                    }
                    Ok(submitted)
                };

            // Stage the window before the prefill token even arrives — the decode
            // successors ride the queue behind the prefill, off the g0 critical path.
            // The other arm has no pass to stage yet and does it after the drain.
            if let Some(fwd) = fwd_d.as_ref() {
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

            // **THE HOST ARM'S HANDOFF, AND THE WHOLE OF IT.** The token is
            // here; the decode pass can be built on it now, `tok_in` SEEDED
            // rather than device-filled, and the run-ahead window staged
            // behind this drain instead of in front of it. The measurement it
            // reads after this point is the same one the other arm reads —
            // every later token is device loop-carried either way.
            if let Some(build_decode) = build_decode.take() {
                fwd_d = build_decode(Channel::from([g0]).named("tok_in"))?;
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

            // Leaving the wait-set is an END-OF-STREAM statement, not a
            // teardown step: from the moment this lane will not submit again,
            // every other lane in the fleet is holding its frame seal for a
            // guest that has nothing left to contribute. The runtime cannot
            // infer this -- an empty queue looks identical between a guest
            // that is finished and one that is mid-decode -- so the guest,
            // which does know, has to say it. Closing here rather than after
            // the drain is what makes that statement on time; the remaining
            // takes below are unaffected (close never waits for an unsettled
            // fire). `closed` is not redundant with the runtime's `first_close`
            // latch -- that latch only suppresses the wait-set notify, while
            // each `close()` still crosses into the host and drains settled
            // entries -- so the trailing call stays correct but is worth
            // skipping.
            let mut closed = false;
            if submitted >= budget {
                pipe.close();
                closed = true;
            }
            let mut taken = 0usize;
            let mut hook_fold_final: Option<u32> = None;
            while taken < submitted {
                let t = out.take_host::<Vec<i32>>().await?;
                if let Some((_, fold_out)) = hook_fold_chans {
                    hook_fold_final =
                        Some(fold_out.take_host::<u32>().await.context("fold drain")?);
                }
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
                hook_fold_final,
            })
        }
    };
}

define_run_one!(run_one_attention, attention);
define_run_one!(run_one_hybrid, hybrid);

async fn run_one(
    input: &Input,
    prompt: &str,
    prompt_tokens: Option<&[u32]>,
    stop_tokens: &[u32],
    honor_wait_for_start: bool,
    rng_seed: u32,
    main_pre_us: Option<u64>,
) -> Result<RunResult> {
    match model::pass_kind() {
        model::ForwardKind::Attention => {
            run_one_attention(
                input,
                prompt,
                prompt_tokens,
                stop_tokens,
                honor_wait_for_start,
                rng_seed,
                main_pre_us,
            )
            .await
        }
        model::ForwardKind::Hybrid => {
            run_one_hybrid(
                input,
                prompt,
                prompt_tokens,
                stop_tokens,
                honor_wait_for_start,
                rng_seed,
                main_pre_us,
            )
            .await
        }
        model::ForwardKind::Recurrent => Err(
            "this benchmark has no recurrent-only path (no registered model reports that kind)"
                .to_string()
                .into(),
        ),
        model::ForwardKind::Diffusion => Err(
            "this benchmark decodes a token at a time; a diffusion model wants a canvas loop"
                .to_string()
                .into(),
        ),
    }
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
            hook_fold_final: None,
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
        hook_fold_final: result.hook_fold_final,
    })
}
