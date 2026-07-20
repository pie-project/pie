//! Speculative decoding with a model's native multi-token-prediction heads,
//! expressed as a single loop-carried pass with in-band `-1` acceptance.
//!
//! One prefill pass seeds the pipeline; after that a single verify-and-extend
//! pass fires repeatedly over a FIXED `k + 1` token envelope. Each fire embeds
//! `[x, d_1 .. d_k]` (the pending correct token plus the previous round's
//! drafts), verifies the drafts against its own logits in-stage, and emits:
//!
//! - `committed`: the round's accepted tokens, `-1`-padded to the envelope —
//!   the host recovers them with `unpad_tokens` and never sees geometry;
//! - the next window `[x', d'_1 .. d'_k]`, drafted from the MTP heads;
//! - the loop-carried KV length, advanced by ONLY the accepted count.
//!
//! Rejected drafts are never retracted: their KV cells simply sit above the
//! advanced length and are overwritten by the next fire (shape decides slots,
//! `-1` decides existence, loop-carry decides position). Once a stop token is
//! committed the pass keeps firing all-`-1` windows — nothing embeds, nothing
//! appends, the length freezes — so in-flight runahead fires ride out as
//! no-ops instead of corrupting state.
//!
//! MTP drafts are anchored at the model's draft row (the window tail), so the
//! round after a partial acceptance drafts from a stale tail and typically
//! re-corrects in one round; correctness always comes from verification.

use inferlet::ptir::prelude::*;
use inferlet::{Result, chat, model as wit_model};
use serde::Deserialize;

const PAGE_T: u32 = 16;
const NUM_LAYERS: u32 = 28;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_k")]
    k: u32,
}

fn default_prompt() -> String {
    "The quick brown fox jumps over".into()
}

fn default_max_tokens() -> usize {
    64
}

fn default_k() -> u32 {
    4
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    if input.max_tokens == 0 {
        return Ok(String::new());
    }
    if !(1..=32).contains(&input.k) {
        return Err("k must be between 1 and 32".into());
    }

    let vocab = wit_model::output_vocab_size();
    model::configure(vocab, PAGE_T, NUM_LAYERS);
    model::configure_gates(true, false);

    let k = input.k;
    let w = k + 1;

    let mut prompt = chat::system_user("Continue the requested text.", &input.prompt);
    prompt.extend(chat::cue());
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let stop_tokens = chat::stop_tokens();

    // One working set carries the whole generation. The lease covers the
    // prompt, every token the host may keep, and the transient overshoot of
    // in-flight windows (each fire may write up to `w` slots above the
    // committed length before a rejection rolls the length back over them).
    let ws = WorkingSet::new();
    let max_extent = n + input.max_tokens as u32 + (DEFAULT_RUNAHEAD_DEPTH as u32 + 2) * w;
    let max_pages = max_extent.div_ceil(PAGE_T);
    ws.reserve(max_pages)
        .map_err(|e| format!("reserve KV: {e}"))?;

    let pipeline = Pipeline::new();

    // ── Prefill: host-known prompt, one fire ────────────────────────────
    let prompt_tokens = Channel::from(prompt.iter().map(|&token| token as i32).collect::<Vec<_>>());
    let seed_out = Channel::new([1], dtype::i32).named("seed");
    let drafts_out = Channel::new([k], dtype::i32).named("drafts");
    let prefill = ForwardPass::new();
    prefill.embed(&prompt_tokens, Tensor::constant(vec![0u32, n]));
    let prefill_kv_len = Channel::from(vec![n]).named("prefill_kv_len");
    prefill.port_channel(Port::KvLen, &prefill_kv_len);
    prefill.attn_working_set(&ws, .., ..)?;
    prefill.derive_dense_geometry();
    prefill.epilogue(move || {
        seed_out.put(reshape(reduce_argmax(intrinsics::logits()), [1]));
        drafts_out.put(reduce_argmax(intrinsics::mtp_logits(k)));
    });
    prefill
        .submit(&pipeline)
        .map_err(|e| format!("prefill: {e}"))?;

    let seed = seed_out
        .take()
        .get::<i32>()
        .await
        .map_err(|e| format!("read prefill seed: {e}"))?[0] as u32;
    let seed_drafts = drafts_out
        .take()
        .get::<i32>()
        .await
        .map_err(|e| format!("read prefill drafts: {e}"))?
        .into_iter()
        .map(|token| token as u32)
        .collect::<Vec<_>>();

    // ── Loop-carried verify-and-extend pass ─────────────────────────────
    let mut window0 = vec![seed];
    window0.extend_from_slice(&seed_drafts);
    let window = Channel::from(pad_tokens(&window0, w as usize)).named("window");
    let len = Channel::from(vec![n]).named("len");
    let kv_len = Channel::from((1..=w).map(|row| n + row).collect::<Vec<_>>()).named("kv_len");
    let stopped = Channel::from_shaped([1u32], vec![false]).named("stopped");
    let committed_out = Channel::new([w], dtype::i32)
        .capacity(DEFAULT_RUNAHEAD_DEPTH as u32)
        .named("committed");

    let fwd = ForwardPass::new();
    fwd.embed(&window, Tensor::constant((0..=w).collect::<Vec<_>>()));
    fwd.port_channel(Port::KvLen, &kv_len);
    fwd.attn_working_set(&ws, .., (n / ws.page_size())..)?;
    fwd.derive_dense_geometry();
    fwd.readout(&Tensor::constant((0..w).collect::<Vec<u32>>()));
    let stage_stop_tokens = stop_tokens.clone();
    fwd.epilogue(move || {
        let win = window.take().tensor();
        let base = len.take().tensor();
        kv_len.take();
        let stop_prev = stopped.take().tensor();
        let neg1_w = broadcast(Tensor::constant(TOKEN_PAD), [w]);

        // Verify: row i holds the truth for window slot i + 1. A draft is
        // accepted while every draft before it matched (cumprod prefix).
        let targets = reduce_argmax(intrinsics::logits());
        let drafts = gather(&win, add(iota(k), 1u32));
        let truth = gather(&targets, iota(k));
        let acc = cumprod(cast(eq(&drafts, &truth), DType::F32));
        let m = reshape(cast(reduce_sum(&acc), DType::U32), [1]);

        // Committed this round: slot 0 plus the accepted prefix, `-1`-padded
        // to the envelope; all `-1` on post-stop fires.
        let live = le(iota(w), broadcast(&m, [w]));
        let stop_prev_w = broadcast(&stop_prev, [w]);
        let committed = select(&stop_prev_w, &neg1_w, select(&live, &win, &neg1_w));
        committed_out.put(&committed);

        // Stop once any committed token is a stop token: later fires carry
        // all-`-1` windows and the frozen length below.
        let mut eos_hit = broadcast(Tensor::constant(false), [w]);
        for &stop in &stage_stop_tokens {
            eos_hit = or(&eos_hit, eq(&committed, Tensor::constant(stop as i32)));
        }
        let eos_any = ne(
            reduce_max(cast(&eos_hit, DType::U32)),
            Tensor::constant(0u32),
        );
        let stop_next = or(&stop_prev, reshape(eos_any, [1]));
        stopped.put(&stop_next);

        // Loop-carry: the length advances by the accepted count only —
        // rejected drafts stay above it and are overwritten next fire.
        let advance = select(
            &stop_prev,
            broadcast(Tensor::constant(0u32), [1]),
            add(&m, 1u32),
        );
        let next_base = add(&base, &advance);

        // Next window: the correction/bonus token followed by fresh MTP
        // drafts, or all `-1` once stopped.
        let x_next = scalar_gather(&targets, &m);
        let drafts_next = reduce_argmax(intrinsics::mtp_logits(k));
        let win_next = scatter_set(
            scatter_set(&neg1_w, Tensor::constant(vec![0u32]), &x_next),
            add(iota(k), 1u32),
            &drafts_next,
        );
        let stop_next_w = broadcast(&stop_next, [w]);
        let next_window = select(&stop_next_w, &neg1_w, &win_next);
        let next_live = ne(&next_window, Tensor::constant(TOKEN_PAD));
        let next_live_u32 = cast(&next_live, DType::U32);
        let next_live_f32 = cast(&next_live, DType::F32);
        let next_rank = cast(sub(cumsum(&next_live_f32), &next_live_f32), DType::U32);
        let next_positions = add(broadcast(&next_base, [w]), &next_rank);

        len.put(&next_base);
        kv_len.put(add(&next_positions, &next_live_u32));
        window.put(&next_window);
    });

    let mut generated = Vec::with_capacity(input.max_tokens);
    let mut drafted = 0usize;
    let mut accepted = 0usize;
    let mut rounds = 0usize;
    let mut done = false;
    let mut in_flight = 0usize;

    while in_flight < DEFAULT_RUNAHEAD_DEPTH {
        fwd.submit(&pipeline)
            .map_err(|e| format!("submit verify round: {e}"))?;
        in_flight += 1;
    }
    while !done || in_flight > 0 {
        let round = committed_out
            .take()
            .get::<i32>()
            .await
            .map_err(|e| format!("read committed round: {e}"))?;
        in_flight -= 1;
        let live = unpad_tokens(&round);
        if !done && !live.is_empty() {
            rounds += 1;
            drafted += k as usize;
            accepted += live.len() - 1;
            for token in live {
                if stop_tokens.contains(&token) {
                    done = true;
                    break;
                }
                generated.push(token);
                if generated.len() == input.max_tokens {
                    done = true;
                    break;
                }
            }
        }
        if !done {
            fwd.submit(&pipeline)
                .map_err(|e| format!("submit verify round: {e}"))?;
            in_flight += 1;
        }
    }
    pipeline.close();

    let acceptance_rate = if drafted == 0 {
        0.0
    } else {
        accepted as f64 / drafted as f64
    };
    eprintln!(
        "mtp-speculative-decoding: rounds={rounds} drafted={drafted} accepted={accepted} \
         acceptance_rate={acceptance_rate:.3}"
    );
    wit_model::decode(&generated)
}
