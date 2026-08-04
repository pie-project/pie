//! **Pipelined ① copy of `tempgen`** — pure-temperature (`Multinomial`-shaped)
//! decode with SUBMIT-AHEAD depth, on the `inferlet::ptir` bridge. Where
//! `tempgen`'s decode loop is synchronous (submit → take per fire), this copy
//! keeps `DEPTH` fires in flight: the loop-carried `tok_in`/`rng` channels are
//! device-side (each fire's epilogue puts the values the next fire takes — no
//! host round-trip), so the host submits fire `t+1` BEFORE draining fire `t`'s
//! token. The host-facing `out` ring is widened to `DEPTH` cells to absorb the
//! run-ahead (the `runahead` idiom).
//!
//! A COPY (the original `tempgen` is a load-bearing temperature-dispatch
//! baseline): this validates that the temperature epilogue shape survives
//! run-ahead submission, not just the synchronous drain.

use inferlet::ptir::attention::prelude::*;
use inferlet::{Result, model as wit_model};

const TEMPERATURE: f32 = 0.8;
const MAX_TOKENS: usize = 8;
/// Submit-ahead window: fire `t+DEPTH-1` is submitted before fire `t`'s token
/// is drained. Also the host-read `out` ring capacity.
const DEPTH: usize = 2;

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    let vocab = wit_model::output_vocab_size();
    let ws = WorkingSet::new();
    let page_size = kv_page_size();

    let prompt_tokens = wit_model::encode("hello world");
    let prompt: Vec<u32> = if prompt_tokens.is_empty() {
        vec![0]
    } else {
        prompt_tokens
    };
    let n = prompt.len() as u32;
    let max_pages = (n + MAX_TOKENS as u32 + 1).div_ceil(page_size);
    ws.reserve(max_pages).context("ws.reserve")?;

    // ──────────────── ONE PIPELINE, ONE STREAM (R4-4) ───────────────────────
    // Prefill then decode on a single pipeline: the prefill epilogue seeds
    // the decode's loop-carried `tok_in` ON-DEVICE (no g0 host round-trip)
    // and mirrors the token to the host via `out`. `close` releases the
    // scheduler wait-set after the fixed submission budget is enqueued.
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let toks_p = Channel::from(prompt_i32).named("toks_p");
    let embed_indptr_p = Channel::from([0u32, n]).named("embed_indptr_p");
    let positions_p = Channel::from_iter(0..n).named("positions_p");
    let pages_p = Channel::from_iter(0..max_pages).named("pages_p");
    let page_indptr_p = Channel::from([0u32, n.div_ceil(page_size)]).named("page_indptr_p");
    let w_slot_p = Channel::from_iter((0..n).map(|p| p / page_size)).named("w_slot_p");
    let w_off_p = Channel::from_iter((0..n).map(|p| p % page_size)).named("w_off_p");
    let rng_p = Channel::from([0x9e37_u32, 0]).named("rng_p");
    // Device-carried handoff: unseeded, DEVICE-ONLY, attached to both
    // passes — the prefill's put fills it. Host copies ride separate
    // per-pass channels (`g0` for the prefill, `out` for the decode): a
    // host-visible channel cannot attach to two passes.
    let tok_in = Channel::new([1], dtype::i32).named("tok_in");
    let g0_ch = Channel::new([1], dtype::i32).named("g0");
    let rng = Channel::from([0x51ed_u32, 0]).named("rng");
    // Host-read ring widened to the submit-ahead window.
    let out = Channel::new([1], dtype::i32)
        .capacity(DEPTH as u32)
        .named("out");

    let fwd_p = ForwardPass::new();
    fwd_p.embed(&toks_p, &embed_indptr_p)?;
    let kv_len_p = Channel::from([n]).named("kv_len_p");
    fwd_p.attention(
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
    )?;
    fwd_p.epilogue(move || {
        let r = rng_p.take(); // [2] u32 rng state (key, ctr)
        let logits = intrinsics::logits(); // [vocab] f32
        let scaled = logits / TEMPERATURE;
        let g = gumbel(&r, [vocab]);
        let t = reduce_argmax(scaled + g); // [1] i32 categorical draw
        let r_next = &r + iota(2); // advance ctr: [key, ctr+1]
        tok_in.put(&t);
        g0_ch.put(&t);
        rng_p.put(&r_next);
    });

    // ──────────────── PIPELINED DECODE (1-wide, depth-DEPTH) ────────────────
    // Built BEFORE the prefill submits (its eager `tok_in` claim must precede
    // the prefill's build, F8) and submitted run-ahead immediately behind it: its geometry needs
    // only KvLen (from n+1, +1/fire); the token value rides `tok_in`.
    let lane1 = Channel::from([0u32, 1u32]).named("embed_indptr");
    let positions = Channel::from([n]).named("positions");
    let pages = Channel::from_iter(0..max_pages).named("pages");
    let page_indptr = Channel::from([0u32, (n + 1).div_ceil(page_size)]).named("page_indptr");
    let w_slot = Channel::from([n / page_size]).named("w_slot");
    let w_off = Channel::from([n % page_size]).named("w_off");
    let fwd = ForwardPass::new();
    fwd.embed(&tok_in, &lane1)?;
    let kv_len = Channel::from([n + 1]).named("kv_len");
    fwd.attention(
        &ws,
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
    )?;
    fwd.epilogue(move || {
        let length = kv_len.take();
        let r = rng.take(); // [2] u32 rng state
        let logits = intrinsics::logits(); // [vocab] f32 (single read-out row)
        let scaled = logits / TEMPERATURE;
        let g = gumbel(&r, [vocab]);
        let t = reduce_argmax(scaled + g); // [1] i32 categorical draw

        let r_next = &r + iota(2);
        let next_length = &length + 1u32;
        let page_count = next_length.div_ceil(page_size);

        tok_in.put(&t);
        kv_len.put(&next_length);
        positions.put(&length);
        w_slot.put(&length / page_size);
        w_off.put(&length % page_size);
        page_indptr.put(indptr(1, &page_count));
        out.put(&t);
        rng.put(&r_next);
    });

    let pipe = Pipeline::new();
    let budget = MAX_TOKENS - 1;
    fwd_p.submit(&pipe).context("prefill submit")?;

    // Prime + fill: up to DEPTH chain-linked fires upfront (none awaited);
    // finish() right after the last budget submit (F7); then FIFO drain +
    // refill one fire per drained decode token.
    let mut submitted = 0usize;
    while submitted < DEPTH.min(budget) {
        fwd.submit(&pipe)
            .with_context(|| format!("decode submit @{submitted}"))?;
        submitted += 1;
    }

    // Host drain: the g0 copy first (the decode burst is already in the
    // engine while this take waits), then the decode ring with refill.
    let g0 = g0_ch.take_host::<i32>().await?;
    let mut generated: Vec<u32> = Vec::with_capacity(MAX_TOKENS);
    eprintln!("[TEMPGEN_PIPELINED] got token: {g0}");
    generated.push(g0 as u32);

    let mut taken = 0usize;
    while taken < submitted {
        let t = out
            .take_host::<Vec<i32>>()
            .await
            .with_context(|| format!("@{}", generated.len()))?;
        taken += 1;
        let Some(&t0) = t.first() else {
            return Err(format!("out.take @{}: empty tensor", generated.len()));
        };
        eprintln!("[TEMPGEN_PIPELINED] got token: {t0}");
        generated.push(t0 as u32);
        if submitted < budget {
            fwd.submit(&pipe)
                .with_context(|| format!("decode submit @{submitted}"))?;
            submitted += 1;
        }
    }
    pipe.close();

    let text = wit_model::decode(&generated);
    eprintln!(
        "[TEMPGEN_PIPELINED] generated {} tokens: {:?}",
        generated.len(),
        text
    );
    Ok(format!("{{\"tokens\": {generated:?}, \"text\": {text:?}}}"))
}
