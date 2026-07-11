//! **Low-level chat-EOS greedy generation with EXPLICIT run-ahead + rollback**,
//! rewritten on the PTIR surface (`inferlet::ptir`). The whole decode loop,
//! including the pipelining, is still hand-written HERE in the inferlet:
//!
//!   submit (producer fire) → speculate the next fire (consumer submit) BEFORE
//!   harvesting the producer's token off the `out` channel → if the harvested
//!   token is a chat-EOS stop, DISCARD the ≤1 speculated over-shot fire's
//!   output (depth-1 rollback) and finish; else keep draining. The sampled
//!   token is carried DEVICE-side between fires (the loop-carried `tok_in`
//!   channel put in the epilogue), so a speculated fire needs no host input.
//!
//! MIGRATION MAP (classic → PTIR):
//!   - `carrier::submit_pass(carry=true)`  → `pipeline.submit(&fwd)` where the
//!     epilogue puts the sampled token into `tok_in` (the device carrier) and
//!     evolves the KV geometry + causal mask in-graph (the text-completion /
//!     ptir-prefill-e2e wire form).
//!   - `SamplerSpec::Argmax` / `LoweredSampler` → the in-graph greedy program
//!     `reduce_argmax(intrinsics::logits())` in the epilogue.
//!   - `KvWorkingSet` + host cursor → `WorkingSet` page pool + seeded geometry
//!     channels (`pos`/`fill`/`klen`/`w_slot`/`w_off`/`mask`), advanced by the
//!     epilogue each fire.
//!   - `carrier::discard_pass` (WAR-drain + cursor rollback) → take-and-drop
//!     the over-shot fire's `out` value. A discarded fire has already run and
//!     written its (unused) KV cell; the stream ends there, so the cell is
//!     inert and the pool is torn down with the stream.
//!   - prompt prefill: the classic path fed the prompt as the first pass's
//!     input tokens; here it is ONE N-wide prefill fire (multi-query causal
//!     mask + N-cell KV write) whose read-out row yields the first token g0.
//!
//! NOT PRESERVED (and why): speculation ACROSS the prefill boundary. The
//! classic loop speculated the successor of the prompt pass itself; the PTIR
//! templates keep the prefill host-gated (read g0, then seed the decode
//! channels from it), so run-ahead starts at the first decode fire. The
//! rollback probe therefore forces the stop on the SECOND generated token
//! (the first DECODE-fire token) so the deep loop still speculates past a
//! stop and discards its over-shot fires.
//!
//! DEEP CARRIER: `Mode::Deep(depth)` submits `depth` fires UPFRONT (FIFO
//! drain + refill) over the ptir run-ahead FIFO, discarding the ≤`depth`−1
//! over-shot outputs on EOS. `DEEP_MATCH` self-checks deep == sync.
//!
//! VALUE-VERIFY: the inferlet self-checks `pipelined == sequential` (`MATCH`)
//! and that the forced-early-stop rollback path drains cleanly
//! (`ROLLBACK_OK`). On the mock the dummy driver's deterministic logits vary
//! per fire AND per pass instance, so the two streams need not be equal:
//! the mock smoke (`lowlevel_chat_mock`) asserts only that the loops run to
//! completion with the full `n=max_tokens` budget and a well-formed report.
//! Token identity is the 4090 gate (`bin/pie/tests/cuda_lowlevel_chat.rs`).
//! The mock run passes `no-rollback-probe`, which skips the deep + forced-stop
//! device paths exactly like the classic version did.
//!
//! Input: `"<max_tokens> [depth=<k>] [no-rollback-probe]"` (default 8 / 4).

use inferlet::ptir::prelude::*;
use inferlet::{chat, model as wit_model, Result};

const SYSTEM: &str = "You are a helpful assistant.";
const USER: &str = "Say hello.";

const PAGE_T: u32 = 16; // tokens per KV pool page (mock and Qwen3 page size)
const NUM_LAYERS: u32 = 28; // Qwen3-0.6B (inert for epilogue-only passes)

fn bx<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

/// How the decode fires are driven. All three run the SAME device-carried
/// pass; they differ only in submission discipline (the point of this
/// inferlet: pipelining is explicit host-side code, not a helper).
#[derive(Clone, Copy)]
enum Mode {
    /// One fire at a time: submit → harvest → submit (the host round-trip
    /// bubble per token). The reference stream.
    Sync,
    /// Depth-1 explicit run-ahead: speculate the successor fire BEFORE
    /// harvesting the producer, even when a stop is configured; discard the
    /// ≤1 over-shot output on EOS.
    Pipelined,
    /// Depth-k pre-submission (the production carrier): k fires in flight,
    /// FIFO drain + refill; discard the ≤k−1 over-shot outputs on EOS.
    Deep(usize),
}

/// One decode context: its own KV page pool + the device loop-carried decode
/// pass. Replaces the classic `Decoder { KvWorkingSet, seq_len, fresh }`; the
/// cursor now lives in the seeded geometry channels the epilogue advances.
struct Stream {
    pipeline: Pipeline,
    decode: &'static ForwardPass<'static>,
    out: &'static Channel,
    pool_ids_ch: &'static Channel,
    pool_ids: &'static Vec<u32>,
}

impl Stream {
    /// Fire the decode pass once (never blocks; ordering rides the channels).
    /// The physical page ids are per-instance data, so they ride a host-put
    /// channel each fire (D2), exactly like the templates.
    fn submit(&self) -> Result<()> {
        self.pool_ids_ch.put(self.pool_ids.clone());
        self.pipeline
            .submit(self.decode)
            .map_err(|e| format!("decode submit: {e}"))
    }

    /// Harvest one fire's sampled token off the `out` channel (blocks by
    /// awaiting the in-flight fire; poison surfaces as `Err`).
    fn take_token(&self) -> Result<u32> {
        let v = self
            .out
            .take()
            .get::<i32>()
            .map_err(|e| format!("out take: {e}"))?;
        Ok(v.first().copied().unwrap_or(0) as u32)
    }
}

/// Run the N-wide prompt prefill fire, then build the device loop-carried
/// decode pass seeded from (n, g0). Returns the first generated token g0 and
/// the armed decode stream. `budget` bounds the run-ahead window, so `out`
/// gets `budget + 1` ring cells (every fire's token stays takeable even with
/// the deep loop's k un-harvested fires in flight).
fn start_stream(prompt: &[u32], budget: usize) -> Result<(u32, Stream)> {
    let n = prompt.len() as u32;
    let pool_pages = (n + budget as u32 + 2).div_ceil(PAGE_T);
    let pool = pool_pages * PAGE_T;

    let ws: &'static WorkingSet = bx(WorkingSet::new());
    let grant = ws.alloc(pool_pages).map_err(|e| format!("ws.alloc: {e}"))?;
    let pool_ids: &'static Vec<u32> = bx(grant.ids().to_vec());

    // ── 1. PREFILL FIRE (N-wide): causal [N, POOL] mask, N-cell KV write,
    //       read-out row N−1, greedy argmax → g0. ──
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let toks_p = bx(Channel::from(prompt_i32).named("toks_p"));
    let embed_indptr_p = Tensor::constant(vec![0u32, n]);

    let w_slot_pv: Vec<u32> = (0..n).map(|c| pool_ids[(c / PAGE_T) as usize]).collect();
    let w_off_pv: Vec<u32> = (0..n).map(|c| c % PAGE_T).collect();
    let w_slot_p = bx(Channel::from(w_slot_pv).named("w_slot_p"));
    let w_off_p = bx(Channel::from(w_off_pv).named("w_off_p"));
    let klen_p = bx(Channel::from(vec![n; 1]).named("klen_p"));
    let pages_p = bx(Channel::from(pool_ids.clone()).named("pages_p"));
    let page_indptr_p = bx(Channel::from_shaped([2], vec![0u32, pool_pages]).named("pidx_p"));
    let mask_pv: Vec<bool> = (0..n).flat_map(|i| (0..pool).map(move |j| j <= i)).collect();
    let mask_p = bx(Channel::from_shaped([n, pool], mask_pv).named("mask_p"));
    let g0_ch = bx(Channel::new([1], dtype::i32).named("g0"));

    let fwd_p: &'static ForwardPass<'static> = bx(ForwardPass::new());
    fwd_p.embed(toks_p, embed_indptr_p);
    fwd_p.attn_working_set(ws, klen_p);
    fwd_p.port_channel(Port::Pages, pages_p);
    fwd_p.port_channel(Port::PageIndptr, page_indptr_p);
    fwd_p.port_channel(Port::WSlot, w_slot_p);
    fwd_p.port_channel(Port::WOff, w_off_p);
    fwd_p.attn_mask(mask_p);
    fwd_p.epilogue(move || {
        let tok = reduce_argmax(intrinsics::logits()); // greedy over row N−1
        g0_ch.put(&tok);
    });

    let prefill = Pipeline::new();
    prefill
        .submit(fwd_p)
        .map_err(|e| format!("prefill submit: {e}"))?;
    let g0 = g0_ch.take().get::<i32>().map_err(|e| format!("g0 take: {e}"))?;
    let g0 = g0.first().copied().unwrap_or(0) as u32;
    prefill.close();

    // ── 2. DECODE PASS (1-wide, device loop-carried): the epilogue carries
    //       the sampled token into the next fire's embed (the run-ahead
    //       carrier) and advances geometry + mask in-graph. ──
    let phys_n = pool_ids[(n / PAGE_T) as usize];
    let tok_in = bx(Channel::from(vec![g0 as i32; 1]).named("tok_in"));
    let pos = bx(Channel::from(vec![n; 1]).named("pos"));
    let fill = bx(Channel::from(vec![n + 1; 1]).named("fill"));
    let klen = bx(Channel::from(vec![n + 1; 1]).named("klen"));
    let w_slot = bx(Channel::from(vec![phys_n; 1]).named("w_slot"));
    let w_off = bx(Channel::from(vec![n % PAGE_T; 1]).named("w_off"));
    let seed_mask: Vec<bool> = (0..pool).map(|j| j <= n).collect();
    let mask = bx(Channel::from_shaped([1, pool], seed_mask).named("mask"));
    let pages = bx(Channel::from(pool_ids.clone()).named("pages"));
    let page_indptr = bx(Channel::from_shaped([2], vec![0u32, pool_pages]).named("page_indptr"));
    let pool_ids_ch = bx(Channel::new([pool_pages], dtype::u32).named("pool_ids"));
    let out = bx(
        Channel::new([1], dtype::i32)
            .capacity(budget as u32 + 1)
            .named("out"),
    );
    let lane1 = Tensor::constant(vec![0u32, 1u32]);

    let fwd: &'static ForwardPass<'static> = bx(ForwardPass::new());
    fwd.embed(tok_in, lane1);
    fwd.positions(pos);
    fwd.attn_working_set(ws, klen);
    fwd.port_channel(Port::Pages, pages);
    fwd.port_channel(Port::PageIndptr, page_indptr);
    fwd.port_channel(Port::WSlot, w_slot);
    fwd.port_channel(Port::WOff, w_off);
    fwd.attn_mask(mask);
    fwd.epilogue(move || {
        // Takes + compute first, puts last (value-id discipline).
        let base = fill.take().tensor(); // [1] u32: position this fire writes
        let pids = pool_ids_ch.take();

        let tok = reduce_argmax(intrinsics::logits()); // [1] i32, greedy

        // Full causal mask for the query at `base`: attend all j <= base.
        let col = iota(pool);
        let base_b = broadcast(reshape(&base, [1]), [pool]);
        let new_mask = reshape(le(&col, &base_b), [1, pool]);

        let logical_slot = div(&base, PAGE_T);
        let w_slot_v = gather(&pids, &logical_slot);
        let w_off_v = rem(&base, PAGE_T);
        let klen_v = add(&base, 1u32);
        let next_free = add(&base, 1u32);
        let pages_v = reshape(&pids, [pool_pages]);
        let pidx_v = mul(&iota(2), pool_pages);

        tok_in.put(&tok); // the device carrier: next fire embeds this token
        out.put(&tok);
        mask.put(&new_mask);
        w_slot.put(&w_slot_v);
        w_off.put(&w_off_v);
        klen.put(&klen_v);
        pos.put(&base);
        fill.put(&next_free);
        pages.put(&pages_v);
        page_indptr.put(&pidx_v);
    });

    Ok((
        g0,
        Stream {
            pipeline: Pipeline::new(),
            decode: fwd,
            out,
            pool_ids_ch,
            pool_ids,
        },
    ))
}

/// **The explicit depth-1 run-ahead + EOS-rollback loop.** Speculates the
/// successor fire BEFORE harvesting the producer, even with a stop configured
/// (declining only at the count-predictable max-tokens boundary, R9); on a
/// stop, drains + drops the ≤1 over-shot output (the rollback).
fn decode_pipelined_eos(s: &Stream, remaining: usize, stop: &[u32]) -> Result<Vec<u32>> {
    let mut out = Vec::with_capacity(remaining);
    if remaining == 0 {
        return Ok(out);
    }
    s.submit()?; // prime the producer
    let mut submitted = 1usize;
    let mut harvested = 0usize;
    loop {
        // Speculate one ahead of the fire we are about to harvest.
        if submitted < remaining {
            s.submit()?;
            submitted += 1;
        }
        let token = s.take_token()?;
        harvested += 1;
        if stop.contains(&token) {
            // EOS: drain + drop the over-shot speculated output. The stop
            // token is NOT emitted.
            while harvested < submitted {
                let _ = s.take_token()?;
                harvested += 1;
            }
            break;
        }
        out.push(token);
        if harvested == submitted {
            break; // max-tokens boundary: nothing speculated past it
        }
    }
    Ok(out)
}

/// **The DEEP (depth-k) run-ahead + EOS-rollback loop.** Submits `depth`
/// fires upfront (none harvested, speculating PAST a possible stop), then
/// drains FIFO + refills one fire per non-stop token. On EOS, drains + drops
/// the ≤depth−1 over-shot outputs.
fn decode_pipelined_deep_eos(
    s: &Stream,
    remaining: usize,
    stop: &[u32],
    depth: usize,
) -> Result<Vec<u32>> {
    let mut out = Vec::with_capacity(remaining);
    let mut submitted = 0usize;
    let mut harvested = 0usize;
    while submitted < depth.max(1).min(remaining) {
        s.submit()?;
        submitted += 1;
    }
    while harvested < submitted {
        let token = s.take_token()?;
        harvested += 1;
        if stop.contains(&token) {
            while harvested < submitted {
                let _ = s.take_token()?;
                harvested += 1;
            }
            break;
        }
        out.push(token);
        if submitted < remaining {
            s.submit()?;
            submitted += 1;
        }
    }
    Ok(out)
}

/// Synchronous reference: one fire at a time, the host harvesting each token
/// before submitting the next fire (the per-token round-trip bubble the
/// pipelined loops close). Stops on a `stop` token (dropping it).
fn decode_sync(s: &Stream, remaining: usize, stop: &[u32]) -> Result<Vec<u32>> {
    let mut out = Vec::with_capacity(remaining);
    for _ in 0..remaining {
        s.submit()?;
        let token = s.take_token()?;
        if stop.contains(&token) {
            break;
        }
        out.push(token);
    }
    Ok(out)
}

/// One full greedy chat generation on a FRESH decode context (own KV pool),
/// like the classic per-`Decoder` runs: prefill → g0 → decode loop per `mode`.
fn generate(prompt: &[u32], max_tokens: usize, stop: &[u32], mode: Mode) -> Result<Vec<u32>> {
    if max_tokens == 0 {
        return Ok(Vec::new());
    }
    let (g0, stream) = start_stream(prompt, max_tokens)?;
    if stop.contains(&g0) {
        stream.pipeline.close();
        return Ok(Vec::new());
    }
    let mut tokens = vec![g0];
    let rest = match mode {
        Mode::Sync => decode_sync(&stream, max_tokens - 1, stop)?,
        Mode::Pipelined => decode_pipelined_eos(&stream, max_tokens - 1, stop)?,
        Mode::Deep(depth) => decode_pipelined_deep_eos(&stream, max_tokens - 1, stop, depth)?,
    };
    tokens.extend(rest);
    stream.pipeline.close();
    Ok(tokens)
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    // Input: "<max_tokens> [depth=<k>] [no-rollback-probe]". `depth` is the
    // deep carrier's pre-submission window; align it to the scheduler cap
    // (`PIE_SCHED_MAX_IN_FLIGHT=k`) so the co-verify is turnkey (default 4).
    let max_tokens: usize = input
        .split_whitespace()
        .next()
        .and_then(|t| t.parse().ok())
        .unwrap_or(8);
    let depth: usize = input
        .split_whitespace()
        .find_map(|t| t.strip_prefix("depth=").and_then(|v| v.parse().ok()))
        .unwrap_or(4);

    let vocab = wit_model::output_vocab_size();
    model::configure(vocab, PAGE_T, NUM_LAYERS);

    // Chat prompt via the thin `pie:inferlet/chat` bindings (the template
    // knowledge lives in the host runtime). `stop` is the chat-EOS set.
    let mut prompt = chat::system_user(SYSTEM, USER);
    prompt.extend(chat::cue());
    if prompt.is_empty() {
        prompt.push(0); // tiny test tokenizers may drop every template char
    }
    let stop = chat::stop_tokens();

    // ── Primary: explicit run-ahead + EOS-rollback vs the sequential reference ──
    let tokens_p = generate(&prompt, max_tokens, &stop, Mode::Pipelined)?;
    let tokens_s = generate(&prompt, max_tokens, &stop, Mode::Sync)?;
    let matched = tokens_p == tokens_s;

    // ── DEEP carrier: depth-k pre-submission == sequential. Device-gated
    // (`no-rollback-probe` skips it, as on the classic path): token identity
    // needs the real model; the mock's deterministic logits differ per pass
    // instance, so only the 4090 gate asserts DEEP_MATCH. ──
    let run_device_paths = !input.contains("no-rollback-probe");
    let (deep_matched, tokens_d) = if run_device_paths {
        let d = generate(&prompt, max_tokens, &stop, Mode::Deep(depth))?;
        (d == tokens_s, d)
    } else {
        (true, tokens_s.clone()) // skipped (device-gated)
    };

    // ── Rollback coverage: force an early EOS on the SECOND token (the first
    // DECODE fire; the prefill boundary is host-gated on ptir, see module
    // docs) so the deep loop MUST discard speculated over-shot fires, then
    // assert the forced streams still agree and nothing hung. ──
    let rollback_ok = match (run_device_paths, tokens_s.get(1)) {
        (true, Some(&second)) => {
            let forced = [second];
            let fp = generate(&prompt, max_tokens, &forced, Mode::Deep(depth))?;
            let fs = generate(&prompt, max_tokens, &forced, Mode::Sync)?;
            fp == fs
        }
        // Probe skipped, or the stream was too short to force a mid-stream
        // stop (the primary path already exercised its boundary).
        _ => true,
    };

    // Chat-decode the pipelined stream to text (thin WIT `chat::Decoder`).
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
