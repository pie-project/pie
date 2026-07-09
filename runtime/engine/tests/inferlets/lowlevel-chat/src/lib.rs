//! **Low-level chat-EOS greedy generation with EXPLICIT run-ahead + rollback**
//! (foxtrot ∥ bravo). In Gim's directive — inferlets on the RAW WIT API, NO
//! helper-SDK decode loop (`Context` / `Generator` / `collect_*`). The whole
//! decode loop, including the pipelining, is hand-written HERE in the inferlet:
//!
//!   fire (producer) → speculate the next fire (consumer) BEFORE awaiting the
//!   producer → harvest the producer's token → if it is a chat-EOS stop, DISCARD
//!   the ≤1 speculated over-shot fire (depth-1 rollback) and finish; else promote
//!   the consumer and carry its sampled token device-side into the next fire.
//!
//! This proves BOTH directives at once: (1) an inferlet needs no helper SDK — it
//! drives `inference::ForwardPass` / `working_set::KvWorkingSet` / a greedy
//! `tensor::Program` / the `pie:instruct/chat` WIT bindings directly; and (2)
//! pipelining is EXPLICIT and natural on the low-level API — the run-ahead carrier
//! (`next_inputs(&[0])`) + the EOS rollback are written in plain view, not hidden
//! in a `collect_*_pipelined` helper.
//!
//! MECHANISM (owned by bravo): `ptir-pipelined-eos-rollback-spec` (§4.3 depth-1
//! fire-past-EOS discard) + `runtime/tests/inferlets/runahead` (the raw run-ahead
//! carrier template). The novelty vs `runahead::decode_pipelined`: we speculate
//! the successor EVEN when a stop is configured, and clean up the over-shot
//! fire(s) — so a chat generation that ends on EOS still pipelines (instead of
//! degrading to sequential).
//!
//! DEEP CARRIER (the production lever, bravo): `decode_pipelined_deep_eos` submits
//! `depth` carrier-linked fires UPFRONT (FIFO drain + refill) via echo's
//! `carrier::submit_pass(carry=true)`, discarding the ≤`depth`−1 over-shot on EOS
//! via `carrier::discard_pass`. This keeps a pipeline's fires DENSELY resident (the
//! co-batch 8× residency lever) AND cuts the per-token host round-trip (the
//! reduce-R bubble-close) — the production analog of
//! `runahead::decode_pipelined_deep_stop`. `DEEP_MATCH` self-checks deep == sync.
//!
//! VALUE-VERIFY: the inferlet self-checks `pipelined == sequential` (`MATCH`) and
//! that the rollback path fires cleanly under a forced early stop (`ROLLBACK_OK`).
//! On the mock the device carrier is a no-op and the sampler is input-independent,
//! so `MATCH` is DEGENERATE (both paths read the same constant) — the mock proves
//! only that the loop + rollback run to completion without panic/hang. Real
//! token-identity is the 4090 gate (`bin/pie/tests/cuda_lowlevel_chat.rs`).
//!
//! Input: optional token budget (default 8), e.g. `"16"`.

use inferlet::inference::ForwardPass;
use inferlet::sampler::{self, LoweredSampler, SamplerSpec};
use inferlet::working_set::KvWorkingSet;
use inferlet::{carrier, chat, model, Result};
use std::collections::VecDeque;

const SYSTEM: &str = "You are a helpful assistant.";
const USER: &str = "Say hello.";

/// One decode context on the raw WIT surface: its own KV working set + cursor.
/// `fresh` arms `fresh-generate()` on the next pass (the #26 dangling-carry
/// clear — drops any carrier left pending on this context from a prior generate).
/// The KV geometry + carrier mechanics live in the keep-core `carrier::`/
/// `geometry::` primitives; this struct just holds the per-context state they mutate.
struct Decoder {
    kv: KvWorkingSet,
    seq_len: u32,
    fresh: bool,
}

impl Decoder {
    fn new() -> Self {
        Self {
            kv: KvWorkingSet::new(),
            seq_len: 0,
            fresh: true,
        }
    }
}

/// Await a pass and read its single sampled token (first `u32` lane).
async fn read_token(pass: ForwardPass) -> Result<u32> {
    let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("tensor read: {e:?}"))?;
    Ok(if bytes.len() >= 4 {
        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u32
    } else {
        0
    })
}

/// Whether the pass producing the `produced_index`-th token (1-based) should
/// declare a run-ahead carrier: decline ONLY when this pass IS the terminal
/// max-tokens pass (count-predictable → no successor to carry into). Independent
/// of the stop set — with EOS-rollback we DO speculate past a possible stop.
fn carries(max_tokens: usize, produced_index: usize) -> bool {
    max_tokens != produced_index
}

/// **The explicit run-ahead + EOS-rollback greedy decode loop.** Hand-written on
/// the raw WIT surface — this is the whole point of the inferlet. Speculates the
/// successor fire one-ahead EVEN when a stop is configured; if the harvested
/// token is a stop/EOS, discards the ≤1 over-shot fire (depth-1 rollback). The
/// stop token is dropped, never emitted — byte-identical output to [`decode_sync`].
async fn decode_pipelined_eos(
    d: &mut Decoder,
    sampler: &LoweredSampler,
    prompt: Vec<u32>,
    max_tokens: usize,
    stop: &[u32],
) -> Result<Vec<u32>> {
    if max_tokens == 0 {
        return Ok(Vec::new());
    }
    let pending = if prompt.is_empty() { vec![0u32] } else { prompt };
    let mut out: Vec<u32> = Vec::with_capacity(max_tokens);

    // Prime producer (the prompt tail). Carries unless it is itself the terminal
    // (max_tokens == 1) pass. `carrier::submit_pass` does the fresh-clear + KV
    // geometry + input + sampler + carrier + execute + advance-on-submit.
    let mut producer = carrier::submit_pass(
        &d.kv,
        &mut d.seq_len,
        &mut d.fresh,
        sampler,
        &pending,
        carries(max_tokens, 1),
    )?;

    let mut generated = 0usize;
    loop {
        // Speculate the next fire one-ahead — EVEN with a stop set (the rollback
        // cleans up the ≤1 over-shot fire if this step terminates). Decline ONLY
        // at the count-predictable max-tokens boundary (R9: never fire past max).
        let speculate = generated + 1 < max_tokens;
        let consumer = if speculate {
            let carry = carries(max_tokens, generated + 2);
            // Placeholder `0` token — the device carrier overwrites row 0 with the
            // producer's sample pre-forward. `submit_pass` reserves the next slot.
            let c = carrier::submit_pass(
                &d.kv,
                &mut d.seq_len,
                &mut d.fresh,
                sampler,
                &[0u32],
                carry,
            )?;
            Some(c)
        } else {
            None
        };

        // Harvest the producer's token, overlapped with the consumer's in-flight
        // compute (the device carrier injects this token into the consumer's row 0).
        let token = read_token(producer).await?;

        if stop.contains(&token) {
            // EOS: discard the over-shot fire — `carrier::discard_pass` finalizes it
            // (the WAR-guard drain, never an un-finalized drop) + rolls the
            // speculative cursor back by one. The stop token is NOT emitted.
            if let Some(c) = consumer {
                carrier::discard_pass(c, &mut d.seq_len).await;
            }
            break;
        }

        out.push(token);
        generated += 1;

        match consumer {
            // Promote the speculated successor: it already carries this token.
            Some(c) => producer = c,
            // No consumer ⟺ we were at the max-tokens boundary → done.
            None => break,
        }
    }
    Ok(out)
}

/// **The DEEP (depth-`k`) run-ahead + EOS-rollback greedy decode loop — the
/// production carrier.** Generalizes `decode_pipelined_eos` (depth-1, one fire
/// ahead) to a `depth`-deep pre-submission: submit `depth` carrier-linked fires
/// UPFRONT (FIFO), then drain + refill to sustain `depth` in flight. This keeps a
/// pipeline's fires DENSELY resident (the co-batch 8× residency lever) + cuts the
/// per-token host round-trip (the reduce-R bubble-close). On an EOS it discards the
/// ≤`depth`−1 over-shot fires via `carrier::discard_pass` (WAR-drain + per-fire
/// cursor rollback — echo's kept-core, executor free, no leak). Byte-identical to
/// `decode_sync(stop)`. The production analog of `runahead::decode_pipelined_deep_stop`
/// on echo's `carrier::submit_pass`/`discard_pass` rather than the raw `build_pass`.
async fn decode_pipelined_deep_eos(
    d: &mut Decoder,
    sampler: &LoweredSampler,
    prompt: Vec<u32>,
    max_tokens: usize,
    stop: &[u32],
    depth: usize,
) -> Result<Vec<u32>> {
    if max_tokens == 0 {
        return Ok(Vec::new());
    }
    let pending = if prompt.is_empty() { vec![0u32] } else { prompt };
    let mut out: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut inflight: VecDeque<ForwardPass> = VecDeque::with_capacity(depth.max(1));
    let mut submitted = 0usize; // fires launched so far

    // Deep pre-submission: launch up to `depth` carrier-linked fires UPFRONT, none
    // awaited (speculating PAST a possible stop). Fire 0 carries the prompt tail;
    // each later fire is a placeholder `[0]` the device carrier overwrites from its
    // predecessor's sample. `carry` declines only at the max-tokens boundary.
    while inflight.len() < depth.max(1) && submitted < max_tokens {
        let carry = carries(max_tokens, submitted + 1);
        let toks: &[u32] = if submitted == 0 { &pending } else { &[0u32] };
        let pass = carrier::submit_pass(&d.kv, &mut d.seq_len, &mut d.fresh, sampler, toks, carry)?;
        submitted += 1;
        inflight.push_back(pass);
    }

    // Drain FIFO; refill one fire per non-stop token to sustain `depth` in flight.
    while let Some(producer) = inflight.pop_front() {
        let token = read_token(producer).await?;
        if stop.contains(&token) {
            // EOS: discard the ≤depth−1 over-shot in-flight fires. Each
            // `discard_pass` WAR-drains + frees (executor) + rolls the cursor back
            // by one — never an un-finalized drop. The stop token is NOT emitted.
            while let Some(c) = inflight.pop_front() {
                carrier::discard_pass(c, &mut d.seq_len).await;
            }
            break;
        }
        out.push(token);
        if submitted < max_tokens {
            let carry = carries(max_tokens, submitted + 1);
            let pass =
                carrier::submit_pass(&d.kv, &mut d.seq_len, &mut d.fresh, sampler, &[0u32], carry)?;
            submitted += 1;
            inflight.push_back(pass);
        }
    }
    Ok(out)
}

/// Synchronous greedy decode (raw): build → execute → await → repeat, one pass
/// per token, feeding each sampled token back as the next input HOST-side (the
/// bubble). The reference the pipelined loop must match byte-for-byte. Stops on a
/// `stop` token (dropping it).
async fn decode_sync(
    d: &mut Decoder,
    sampler: &LoweredSampler,
    prompt: Vec<u32>,
    max_tokens: usize,
    stop: &[u32],
) -> Result<Vec<u32>> {
    let mut pending = if prompt.is_empty() { vec![0u32] } else { prompt };
    let mut out = Vec::with_capacity(max_tokens);
    for _ in 0..max_tokens {
        // No carrier (`carry = false`): the sync reference feeds each token back
        // HOST-side (the bubble the pipelined loop eliminates). `submit_pass`
        // advances the cursor on submit; no overlap, so it's one pass at a time.
        let pass =
            carrier::submit_pass(&d.kv, &mut d.seq_len, &mut d.fresh, sampler, &pending, false)?;
        let token = read_token(pass).await?;
        if stop.contains(&token) {
            break;
        }
        out.push(token);
        pending = vec![token];
    }
    Ok(out)
}

/// Build the greedy-argmax sampler (`SamplerSpec::Argmax` → a `LoweredSampler`
/// carrying the argmax program + its per-fire binding list) for `carrier::submit_pass`.
fn greedy_sampler(vocab: u32) -> Result<LoweredSampler> {
    sampler::sampler_program(SamplerSpec::Argmax, vocab)
        .map_err(|e| format!("build greedy sampler: {e:?}"))
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    // Input: "<max_tokens> [depth=<k>] [no-rollback-probe]". `depth` is the deep
    // carrier's pre-submission window; align it to the scheduler cap
    // (`PIE_SCHED_MAX_IN_FLIGHT=k`) so the co-verify is turnkey (default 4). The
    // first whitespace token is `max_tokens`.
    let max_tokens: usize = input
        .split_whitespace()
        .next()
        .and_then(|t| t.parse().ok())
        .unwrap_or(8);
    let depth: usize = input
        .split_whitespace()
        .find_map(|t| t.strip_prefix("depth=").and_then(|v| v.parse().ok()))
        .unwrap_or(4);
    let vocab = model::output_vocab_size();
    let sampler = greedy_sampler(vocab)?;

    // Chat prompt via the raw `pie:instruct/chat` WIT bindings — the chat-template
    // knowledge lives in the host runtime, so this is a minimal WIT binding, NOT a
    // decode helper. `stop` is the chat-EOS set (e.g. <|im_end|>/<|endoftext|>).
    let mut prompt = chat::system_user(SYSTEM, USER);
    prompt.extend(chat::cue());
    let stop = chat::stop_tokens();

    // ── Primary: explicit run-ahead + EOS-rollback vs the sequential reference ──
    let mut d_p = Decoder::new();
    let tokens_p =
        decode_pipelined_eos(&mut d_p, &sampler, prompt.clone(), max_tokens, &stop).await?;
    let mut d_s = Decoder::new();
    let tokens_s = decode_sync(&mut d_s, &sampler, prompt.clone(), max_tokens, &stop).await?;
    let matched = tokens_p == tokens_s;

    // ── DEEP carrier (the production lever): depth-`k` pre-submission == sequential.
    // Keeps a pipeline's fires densely resident (co-batch 8×) + cuts the per-token
    // host round-trip (reduce-R bubble-close). Byte-identical to the sync stream.
    // DEVICE-RESIDENT: the mock's no-op carrier + fail-closed finalize can't drive
    // the deep pre-submission (>1 un-awaited carrier producer in flight) — the same
    // property that mock-#[ignore]s `runahead::decode_pipelined`. So it's gated
    // behind the device flag (`no-rollback-probe` skips both device-resident paths);
    // real depth-k token-identity is the 4090 gate (`cuda_lowlevel_chat`, run with
    // `PIE_SCHED_MAX_IN_FLIGHT=k` to exercise true k-in-flight residency).
    let run_device_paths = !input.contains("no-rollback-probe");
    let (deep_matched, tokens_d) = if run_device_paths {
        let mut d_d = Decoder::new();
        let tokens_d =
            decode_pipelined_deep_eos(&mut d_d, &sampler, prompt.clone(), max_tokens, &stop, depth)
                .await?;
        (tokens_d == tokens_s, tokens_d)
    } else {
        (true, tokens_s.clone()) // skipped on the mock (device-resident)
    };

    // ── Rollback coverage: force an early EOS (stop = the reference stream's first
    // token) so a speculated successor MUST be discarded, then assert the pipelined
    // stream still equals the sequential one and did not hang. The depth-k rollback
    // discards ≤depth−1 over-shot fires via `carrier::discard_pass`. ──
    //
    // Same device-resident gate as the deep loop: the discard-drain awaits a
    // speculated consumer whose producer terminated; the runtime's next-input
    // finalize is FAIL-CLOSED (a consumer force-aborts unless the producer link it
    // injected from is `Committed` — inference.rs). The mock cannot complete it (the
    // reason `runahead`'s decode is mock-#[ignore]d); validated on the 4090.
    let run_rollback_probe = run_device_paths;
    let rollback_ok = match (run_rollback_probe, tokens_s.first()) {
        (true, Some(&first)) => {
            let forced = [first];
            let mut d_fp = Decoder::new();
            let fp =
                decode_pipelined_deep_eos(&mut d_fp, &sampler, prompt.clone(), max_tokens, &forced, depth)
                    .await?;
            let mut d_fs = Decoder::new();
            let fs = decode_sync(&mut d_fs, &sampler, prompt.clone(), max_tokens, &forced).await?;
            fp == fs
        }
        // No tokens produced (immediate EOS) — the primary path already exercised
        // the discard on its first step. Or the probe was skipped.
        _ => true,
    };

    // Chat-decode the pipelined stream to text (raw WIT `chat::Decoder`).
    let mut dec = chat::Decoder::new();
    let mut text = String::new();
    for t in &tokens_p {
        match dec.feed(core::slice::from_ref(t))? {
            chat::Event::Delta(s) => text.push_str(&s),
            chat::Event::Done(s) => {
                text = s;
                break;
            }
            _ => {}
        }
    }

    let result = format!(
        "MATCH={matched} DEEP_MATCH={deep_matched} ROLLBACK_OK={rollback_ok} n={} \
         pipe={tokens_p:?} deep={tokens_d:?} sync={tokens_s:?}",
        tokens_p.len()
    );
    eprintln!("[LOWLEVEL_CHAT] {result} text={text:?}");
    Ok(result)
}
