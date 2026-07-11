//! Demo: KV state persists across requests via `snapshot::save` / `open`.
//!
//! Uniquely-Pie demo: a chat session's token log lives in the engine under a
//! user-chosen name and survives across inferlet invocations. Turn 2 reopens
//! the snapshot and only pays *accounted* prefill cost for the new user
//! message — a naive client-server pattern (the way most apps work today on
//! vLLM/TGI) has to rebuild and re-account the full conversation history every
//! turn.
//!
//! Two strategies, same conversation:
//!
//! - **BASELINE** — turn 1 runs a fresh `Ctx`. Turn 2 throws that context away
//!   and rebuilds another fresh one with system + user1 + assistant1 + user2 in
//!   its buffer. Pays prefill on the entire history every time.
//! - **RESUMED** — turn 1 saves the post-generation context under a name. Turn
//!   2 calls `snapshot::open(name)`, replays the saved token log, appends
//!   user2, and generates. Only the new user2 wrapper is accounted as turn-2
//!   prefill.
//!
//! `mode = plain | smart | both` (default `both`).
//!
//! **PTIR rewrite** (classic `forward-pass` / `carrier` / `prefill` keep-core
//! retirement). Each generation is the text-completion wire form: the full
//! token log + buffered prompt tail is materialized in ONE N-wide prefill fire
//! (multi-query custom-mask pack + N-cell KV write) that also samples token 1,
//! followed by a 1-wide device-loop-carried decode loop with the in-graph
//! top-p sampler. For a snapshot-restored `Ctx` that single prefill fire IS
//! the token-log REPLAY, fused with the new turn's prompt (the ptir prefill is
//! position-0 anchored; an *incremental* offset prefill is not a proven ptir
//! wire form, and every `Ctx` here generates exactly once, so fusing loses
//! nothing). The `snapshot::` module stays the thin data + wasi:filesystem
//! keep-core it already was.

use std::io::{self, Write};
use std::time::{Duration, Instant};

use inferlet::ptir::prelude::*;
use inferlet::ptir::Taken;
use inferlet::{chat, model as wit_model, snapshot, Result};
use serde::Deserialize;

const PAGE_T: u32 = 16; // tokens per pool page
const NUM_LAYERS: u32 = 28; // Qwen3-0.6B

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_mode")]
    mode: String,

    #[serde(default = "default_turn1")]
    turn1: String,

    #[serde(default = "default_turn2")]
    turn2: String,

    #[serde(default = "default_max_tokens")]
    max_tokens: usize,

    #[serde(default = "default_system")]
    system: String,

    #[serde(default = "default_snapshot")]
    snapshot: String,

    #[serde(default)]
    delay: u64,
}

fn default_system() -> String {
    "You are a concise assistant. Answer in one or two short sentences. \
     Plain ASCII, no markdown."
        .into()
}

fn default_mode() -> String {
    "both".into()
}
fn default_turn1() -> String {
    "What is Python? Answer in one sentence.".into()
}
fn default_turn2() -> String {
    "Name three things it is most commonly used for.".into()
}
fn default_max_tokens() -> usize {
    400
}
fn default_snapshot() -> String {
    "demo-persistent-kv-conv".into()
}

// ── ANSI helpers ───────────────────────────────────────────────────────
const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const YELLOW: &str = "\x1b[33m";
const GREEN: &str = "\x1b[32m";
const CYAN: &str = "\x1b[36m";
const MAGENTA: &str = "\x1b[35m";

fn bx<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

// ── In-inferlet decode context (ptir keep-core, no `Context` facade) ─────
//
// A minimal visible bundle: the materialized token log (`tokens`, = the
// facade's `history`) so a snapshot can be serialized, a `buffer` of prompt
// tokens not yet materialized, and the deferred system prompt. Chat templating
// is the kept thin `chat::` bindings. There is no live device handle here: the
// working set + fires are built inside `decode_stream` (each `Ctx` generates
// exactly once), and the single N-wide prefill fire there materializes
// `tokens ++ buffer` — for a snapshot-restored ctx that is the token-log
// replay, fused with the new turn's prompt.
struct Ctx {
    /// Tokens accounted as already materialized (the replayable log length).
    seq_len: u32,
    /// The materialized token log a snapshot serializes / a restore replays.
    tokens: Vec<u32>,
    /// Prompt tokens not yet materialized (the facade's unflushed buffer).
    buffer: Vec<u32>,
    /// A deferred `chat::system` prompt not yet folded into the buffer.
    pending_system: Option<String>,
}

impl Ctx {
    fn new() -> Self {
        Self {
            seq_len: 0,
            tokens: Vec::new(),
            buffer: Vec::new(),
            pending_system: None,
        }
    }

    /// Rebuild from a snapshot manifest. The token-log REPLAY is deferred into
    /// the next `decode_stream`'s single N-wide prefill fire (fused with the
    /// new turn's prompt); the log itself is accounted as already paid, so the
    /// turn-2 prefill number stays "new tokens only" — the classic accounting.
    fn from_snapshot(snap: snapshot::SnapshotData) -> Result<Self> {
        Ok(Self {
            seq_len: snap.tokens.len() as u32,
            tokens: snap.tokens,
            buffer: snap.buffer,
            pending_system: snap.pending_system,
        })
    }

    /// The manifest to serialize: the materialized log + unflushed tail + geometry.
    fn to_snapshot(&self) -> snapshot::SnapshotData {
        snapshot::SnapshotData {
            version: snapshot::SNAPSHOT_VERSION,
            page_size: PAGE_T,
            seq_len: self.seq_len,
            tokens: self.tokens.clone(),
            buffer: self.buffer.clone(),
            pending_system: self.pending_system.clone(),
            cas_hashes: Vec::new(),
        }
    }

    fn flush_pending_system(&mut self) {
        if let Some(system) = self.pending_system.take() {
            self.buffer.extend(chat::system(&system));
        }
    }

    fn is_first_chat_fill(&self) -> bool {
        self.seq_len == 0 && self.buffer.is_empty()
    }

    fn system(&mut self, message: &str) {
        self.flush_pending_system();
        self.pending_system = Some(message.to_string());
    }

    fn user(&mut self, message: &str) {
        let tokens = match self.pending_system.take() {
            Some(system) => chat::system_user(&system, message),
            None if self.is_first_chat_fill() => chat::first_user(message),
            None => chat::user(message),
        };
        self.buffer.extend(tokens);
    }

    fn assistant(&mut self, message: &str) {
        self.flush_pending_system();
        self.buffer.extend(chat::assistant(message));
    }

    fn cue(&mut self) {
        self.flush_pending_system();
        self.buffer.extend(chat::cue());
    }

    /// Fold the buffer into the token log. Mirrors `Context::flush`: after this
    /// the tokens are part of the replayable log a snapshot serializes (their
    /// physical materialization happens in the restore replay / the next
    /// generation's prefill fire — each `Ctx` here generates at most once, so
    /// no separate flush fire is needed).
    fn flush(&mut self) -> Result<()> {
        self.flush_pending_system();
        if self.buffer.is_empty() {
            return Ok(());
        }
        let tokens = std::mem::take(&mut self.buffer);
        self.tokens.extend(tokens);
        self.seq_len = self.tokens.len() as u32;
        Ok(())
    }

    /// Tokens that will be prefilled on the next decode / flush (= the facade's
    /// `buffer().len()`), for the prefill-cost accounting.
    fn pending_prefill(&self) -> u32 {
        self.buffer.len() as u32
    }
}

/// In-graph top-p + temperature sampler over the read-out row logits. `r` is
/// the taken `[2]` u32 rng state (`[key, ctr]`) driving the Gumbel noise.
fn sample_token(r: &Taken, temperature: f32, top_p: f32, vocab: u32) -> Tensor {
    let logits = intrinsics::logits();
    let scaled = div(&logits, temperature.max(1e-4));
    let probs = softmax(&scaled);
    let keep = pivot_threshold(&probs, cummass_le(top_p));
    let masked = mask_apply(&scaled, &keep);
    let g = gumbel(r, [vocab]);
    reduce_argmax(add(&masked, &g)) // [1] i32
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let mode = input.mode.to_lowercase();

    let model_name = wit_model::name();
    model::configure(wit_model::output_vocab_size(), PAGE_T, NUM_LAYERS);

    // Best-effort: clear stale snapshots before modes that create a new one.
    if !matches!(mode.as_str(), "open" | "turn2" | "resume-turn2") {
        let _ = snapshot::delete(&input.snapshot);
    }

    match mode.as_str() {
        "baseline" | "plain" => {
            run_baseline(&model_name, &input).await?;
        }
        "resumed" | "smart" => {
            run_resumed(&model_name, &input).await?;
            let _ = snapshot::delete(&input.snapshot);
        }
        "save" | "turn1" | "save-turn1" => {
            run_save_turn1(&model_name, &input).await?;
        }
        "open" | "turn2" | "resume-turn2" => {
            run_open_turn2(&model_name, &input).await?;
            let _ = snapshot::delete(&input.snapshot);
        }
        "both" | "" => {
            let b = run_baseline(&model_name, &input).await?;
            println!();
            // Reset snapshot before resumed run.
            let _ = snapshot::delete(&input.snapshot);
            let r = run_resumed(&model_name, &input).await?;
            println!();
            comparison(&b, &r);
            let _ = snapshot::delete(&input.snapshot);
        }
        other => {
            return Err(format!(
                "unknown mode '{}': expected 'baseline', 'resumed', 'save', 'open', or 'both'",
                other
            ));
        }
    }

    Ok(String::new())
}

#[derive(Default, Clone)]
struct ModeResult {
    turn1_prefill: u32,
    turn2_prefill: u32,
    turn1_elapsed: Duration,
    turn2_elapsed: Duration,
    answer1: String,
    answer2: String,
}

// ── BASELINE: rebuild the context from scratch every turn ─────────────
async fn run_baseline(model_name: &str, input: &Input) -> Result<ModeResult> {
    print_header(
        "BASELINE",
        YELLOW,
        "stateless client — every turn re-prefills the full history",
        model_name,
    );

    // ── TURN 1 ──
    println!(
        "  {}{}turn 1{}  user: {}",
        BOLD,
        YELLOW,
        RESET,
        oneline(&input.turn1)
    );
    let mut ctx = Ctx::new();
    ctx.system(&input.system);
    ctx.user(&format!("{} /no_think", input.turn1.trim()));
    ctx.cue();
    let turn1_prefill = ctx.seq_len + ctx.pending_prefill();
    let t = Instant::now();
    let answer1 = decode_stream(&mut ctx, input.max_tokens, input.delay).await?;
    let turn1_elapsed = t.elapsed();
    println!(
        "  {}prefill {} tokens, decode {:?}{}",
        DIM, turn1_prefill, turn1_elapsed, RESET
    );
    println!();

    // ── TURN 2: rebuild from scratch, replay the conversation ──
    println!(
        "  {}{}turn 2{}  user: {}",
        BOLD,
        YELLOW,
        RESET,
        oneline(&input.turn2)
    );
    let mut ctx2 = Ctx::new();
    ctx2.system(&input.system);
    ctx2.user(&input.turn1);
    ctx2.assistant(answer1.trim());
    ctx2.user(&format!("{} /no_think", input.turn2.trim()));
    ctx2.cue();
    let turn2_prefill = ctx2.seq_len + ctx2.pending_prefill();
    let t = Instant::now();
    let answer2 = decode_stream(&mut ctx2, input.max_tokens, input.delay).await?;
    let turn2_elapsed = t.elapsed();
    println!(
        "  {}prefill {} tokens, decode {:?}{}",
        DIM, turn2_prefill, turn2_elapsed, RESET
    );

    print_footer(
        "BASELINE",
        YELLOW,
        turn1_prefill,
        turn2_prefill,
        turn1_elapsed,
        turn2_elapsed,
    );
    Ok(ModeResult {
        turn1_prefill,
        turn2_prefill,
        turn1_elapsed,
        turn2_elapsed,
        answer1,
        answer2,
    })
}

// ── RESUMED: save after turn 1, open before turn 2 ────────────────────
async fn run_resumed(model_name: &str, input: &Input) -> Result<ModeResult> {
    print_header(
        "RESUMED",
        GREEN,
        "snapshot::save() after turn 1; snapshot::open() + replay before turn 2",
        model_name,
    );

    // ── TURN 1 ──
    println!(
        "  {}{}turn 1{}  user: {}",
        BOLD,
        GREEN,
        RESET,
        oneline(&input.turn1)
    );
    let mut ctx = Ctx::new();
    ctx.system(&input.system);
    ctx.user(&format!("{} /no_think", input.turn1.trim()));
    ctx.cue();
    let turn1_prefill = ctx.seq_len + ctx.pending_prefill();
    let t = Instant::now();
    let answer1 = decode_stream(&mut ctx, input.max_tokens, input.delay).await?;
    let turn1_elapsed = t.elapsed();
    // Fold any residual into the log before saving so the snapshot includes
    // the assistant's reply in full.
    ctx.flush()?;
    snapshot::save(&input.snapshot, &ctx.to_snapshot())?;
    println!(
        "  {}prefill {} tokens, decode {:?}, then save() snapshot ✓{}",
        DIM, turn1_prefill, turn1_elapsed, RESET
    );
    drop(ctx);
    println!();

    // ── TURN 2: open the snapshot, append only the new user message ──
    println!(
        "  {}{}turn 2{}  user: {}",
        BOLD,
        GREEN,
        RESET,
        oneline(&input.turn2)
    );
    let mut ctx2 = Ctx::from_snapshot(snapshot::open(&input.snapshot)?)?;
    let pre_open_seq = ctx2.seq_len;
    ctx2.user(&format!("{} /no_think", input.turn2.trim()));
    ctx2.cue();
    let turn2_prefill = (ctx2.seq_len + ctx2.pending_prefill()) - pre_open_seq;
    let t = Instant::now();
    let answer2 = decode_stream(&mut ctx2, input.max_tokens, input.delay).await?;
    let turn2_elapsed = t.elapsed();
    println!(
        "  {}new prefill {} tokens (history reused from snapshot), decode {:?}{}",
        DIM, turn2_prefill, turn2_elapsed, RESET
    );

    print_footer(
        "RESUMED",
        GREEN,
        turn1_prefill,
        turn2_prefill,
        turn1_elapsed,
        turn2_elapsed,
    );
    Ok(ModeResult {
        turn1_prefill,
        turn2_prefill,
        turn1_elapsed,
        turn2_elapsed,
        answer1,
        answer2,
    })
}

// ── SAVE mode: run turn 1 and leave the snapshot for a later invocation ──
async fn run_save_turn1(model_name: &str, input: &Input) -> Result<()> {
    print_header(
        "SAVE",
        GREEN,
        "run turn 1 and leave a named KV snapshot",
        model_name,
    );

    println!(
        "  {}{}turn 1{}  user: {}",
        BOLD,
        GREEN,
        RESET,
        oneline(&input.turn1)
    );
    let mut ctx = Ctx::new();
    ctx.system(&input.system);
    ctx.user(&format!("{} /no_think", input.turn1.trim()));
    ctx.cue();
    let turn1_prefill = ctx.seq_len + ctx.pending_prefill();
    let t = Instant::now();
    let _answer1 = decode_stream(&mut ctx, input.max_tokens, input.delay).await?;
    let turn1_elapsed = t.elapsed();
    ctx.flush()?;
    snapshot::save(&input.snapshot, &ctx.to_snapshot())?;
    println!(
        "  {}saved snapshot {:?} after {} prefill tokens and {:?} decode{}",
        DIM, input.snapshot, turn1_prefill, turn1_elapsed, RESET
    );
    Ok(())
}

// ── OPEN mode: separate invocation that resumes from SAVE mode ─────────
async fn run_open_turn2(model_name: &str, input: &Input) -> Result<()> {
    print_header(
        "OPEN",
        GREEN,
        "open a named KV snapshot and append only turn 2",
        model_name,
    );

    println!("  {}opening snapshot {:?}{}", DIM, input.snapshot, RESET);
    let mut ctx = Ctx::from_snapshot(snapshot::open(&input.snapshot)?)?;
    let pre_open_seq = ctx.seq_len;
    println!(
        "  {}{}turn 2{}  user: {}",
        BOLD,
        GREEN,
        RESET,
        oneline(&input.turn2)
    );
    ctx.user(&format!("{} /no_think", input.turn2.trim()));
    ctx.cue();
    let turn2_prefill = (ctx.seq_len + ctx.pending_prefill()) - pre_open_seq;
    let t = Instant::now();
    let _answer2 = decode_stream(&mut ctx, input.max_tokens, input.delay).await?;
    let turn2_elapsed = t.elapsed();
    println!(
        "  {}new prefill {} tokens (history reused), decode {:?}{}",
        DIM, turn2_prefill, turn2_elapsed, RESET
    );
    Ok(())
}

// ── Stream one turn on the ptir keep-core, return the decoded text ────────
//
// The text-completion wire form: ONE N-wide prefill fire materializes the full
// token log + buffered tail (for a snapshot-restored ctx this IS the token-log
// replay, fused with the new turn's prompt) and samples generation token 1;
// then a 1-wide decode loop whose fires carry the sampled token device-side
// (`tok_in`) and evolve geometry + mask in-graph. Every token embedded by a
// fire is recorded in `ctx.tokens` (it is materialized KV); a sampled token
// that never gets a follow-up fire is either a dropped stop token or — at the
// max-tokens terminal — parked in `ctx.buffer` as the residual, exactly the
// classic facade's stop-truncation / auto-buffer semantics.
async fn decode_stream(ctx: &mut Ctx, max_tokens: usize, delay_ms: u64) -> Result<String> {
    print!("  {}>{} ", CYAN, RESET);
    let _ = io::stdout().flush();

    let vocab = wit_model::output_vocab_size();
    let temperature = 0.6f32;
    let top_p = 0.95f32;
    let stop = chat::stop_tokens();
    let mut decoder = chat::Decoder::new();
    let mut stripper = ThinkStripper::new();
    let mut text = String::new();

    if max_tokens == 0 {
        // Nothing to decode: fold the buffered prompt into the log so history
        // stays faithful (the facade's `generate` flushes the buffer first).
        ctx.flush()?;
        println!();
        return Ok(strip_think_blocks(&text));
    }

    // The full materialization for this generation: log (replay) + buffered tail.
    ctx.flush_pending_system();
    let head = std::mem::take(&mut ctx.buffer);
    ctx.tokens.extend_from_slice(&head);
    if ctx.tokens.is_empty() {
        ctx.tokens.push(0);
    }
    let n = ctx.tokens.len() as u32;
    ctx.seq_len = n;

    // Shared physical page pool: history + decode headroom, page-rounded.
    let pool_pages = (n + max_tokens as u32 + 2 + PAGE_T - 1) / PAGE_T;
    let pool = pool_pages * PAGE_T;
    let ws: &'static WorkingSet = bx(WorkingSet::new());
    let slots = ws.alloc(pool_pages).map_err(|e| format!("ws.alloc: {e}"))?;
    let pool_ids: &'static Vec<u32> = bx(slots.ids().to_vec());

    // ───────────────────────── 1. PREFILL FIRE (N-wide) ─────────────────────
    let prefill_i32: Vec<i32> = ctx.tokens.iter().map(|&t| t as i32).collect();
    let toks_p = bx(Channel::from(prefill_i32).named("toks_p"));
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
    let rng_p = bx(Channel::from(vec![0x51ed_u32, 0]).named("rng_p"));
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
        let r = rng_p.take();
        let tok = sample_token(&r, temperature, top_p, vocab);
        let r_next = add(&r, iota(2));
        g0_ch.put(&tok);
        rng_p.put(&r_next);
    });

    let prefill = Pipeline::new();
    prefill.submit(fwd_p).map_err(|e| format!("prefill submit: {e}"))?;
    let g0 = g0_ch.take().get::<i32>().map_err(|e| format!("g0 take: {e}"))?[0];
    prefill.close();

    // ───────────────────────── 2. DECODE LOOP (1-wide) ──────────────────────
    let phys_n = pool_ids[(n / PAGE_T) as usize];
    let tok_in = bx(Channel::from(vec![g0; 1]).named("tok_in"));
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
    let out = bx(Channel::new([1], dtype::i32).named("out"));
    let rng = bx(Channel::from(vec![0x9e37_u32, 0]).named("rng"));
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
        // TAKES + compute first, PUTS last (value-id discipline).
        let base = fill.take().tensor();
        let pids = pool_ids_ch.take();
        let r = rng.take();

        let tok = sample_token(&r, temperature, top_p, vocab);
        let r_next = add(&r, iota(2));

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

        tok_in.put(&tok);
        out.put(&tok);
        mask.put(&new_mask);
        w_slot.put(&w_slot_v);
        w_off.put(&w_off_v);
        klen.put(&klen_v);
        pos.put(&base);
        fill.put(&next_free);
        pages.put(&pages_v);
        page_indptr.put(&pidx_v);
        rng.put(&r_next);
    });

    let decode = Pipeline::new();
    let mut generated = 0usize;
    let mut residual: Option<u32> = None; // sampled but never materialized
    let mut next_tok = g0 as u32;
    loop {
        // Consume `next_tok` — sampled by the previous fire, not yet in KV.
        if stop.contains(&next_tok) {
            // Stop token → drop it (never emitted, never materialized),
            // matching the facade's stop-truncation semantics.
            break;
        }
        generated += 1;
        let mut done = false;
        match decoder.feed(&[next_tok])? {
            chat::Event::Delta(sd) => {
                text.push_str(&sd);
                let visible = stripper.process(&sd);
                if !visible.is_empty() {
                    let rendered = visible.replace('\n', "\n    ");
                    print!("{}", rendered);
                    let _ = io::stdout().flush();
                    if delay_ms > 0 {
                        inferlet::sleep(std::time::Duration::from_millis(delay_ms)).await;
                    }
                }
            }
            chat::Event::Done(sd) => {
                text = sd;
                done = true; // accepted, but no follow-up fire → not in the log
            }
            _ => {}
        }
        if done {
            break;
        }
        if generated >= max_tokens {
            // Max-tokens terminal: this token never gets a fire, so it is NOT
            // in the KV. Park it as the residual (the facade auto-buffers the
            // last token); a following `flush()`/`save()` folds it into the log.
            residual = Some(next_tok);
            break;
        }
        // The next fire embeds `next_tok` (device loop-carried) → it becomes
        // materialized KV → record it in the log so a snapshot captures it.
        ctx.tokens.push(next_tok);
        ctx.seq_len += 1;
        pool_ids_ch.put(pool_ids.clone());
        decode.submit(fwd).map_err(|e| format!("decode submit: {e}"))?;
        let t = out.take().get::<i32>().map_err(|e| format!("out.take: {e}"))?;
        next_tok = *t.first().unwrap_or(&0) as u32;
    }
    decode.close();
    if let Some(t) = residual {
        ctx.buffer.push(t);
    }

    println!();
    // Strip any think blocks from the captured assistant text so callers can
    // replay it cleanly.
    Ok(strip_think_blocks(&text))
}

fn strip_think_blocks(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    loop {
        match rest.find("<think>") {
            None => {
                out.push_str(rest);
                break;
            }
            Some(start) => {
                out.push_str(&rest[..start]);
                let after_open = &rest[start + "<think>".len()..];
                match after_open.find("</think>") {
                    Some(end) => rest = &after_open[end + "</think>".len()..],
                    None => break,
                }
            }
        }
    }
    out
}

// ── ThinkStripper for streaming output ────────────────────────────────
struct ThinkStripper {
    in_think: bool,
    pending: String,
}

impl ThinkStripper {
    fn new() -> Self {
        Self {
            in_think: false,
            pending: String::new(),
        }
    }

    fn process(&mut self, delta: &str) -> String {
        self.pending.push_str(delta);
        let mut out = String::new();
        loop {
            if self.in_think {
                if let Some(idx) = self.pending.find("</think>") {
                    self.pending = self.pending.split_off(idx + "</think>".len());
                    self.in_think = false;
                    continue;
                }
                let len = self.pending.len();
                if len > 7 {
                    self.pending = self.pending.split_off(len - 7);
                }
                break;
            } else {
                if let Some(idx) = self.pending.find("<think>") {
                    out.push_str(&self.pending[..idx]);
                    self.pending = self.pending.split_off(idx + "<think>".len());
                    self.in_think = true;
                    continue;
                }
                let len = self.pending.len();
                let safe = len.saturating_sub(6);
                if safe > 0 {
                    let head: String = self.pending.drain(..safe).collect();
                    out.push_str(&head);
                }
                break;
            }
        }
        out
    }
}

// ── TUI helpers ────────────────────────────────────────────────────────
fn print_header(mode: &str, color: &str, tagline: &str, model_name: &str) {
    let bar = "═".repeat(64);
    println!("{}{}{}{}", BOLD, color, bar, RESET);
    println!(
        "{}{}  PERSISTENT-KV DEMO   ▸   mode: {}{}  ({}){}",
        BOLD, color, mode, RESET, tagline, RESET
    );
    println!("  {}model {}{}", DIM, model_name, RESET);
    println!("{}{}{}{}", BOLD, color, bar, RESET);
    println!();
}

fn print_footer(
    mode: &str,
    color: &str,
    t1_prefill: u32,
    t2_prefill: u32,
    t1_elapsed: Duration,
    t2_elapsed: Duration,
) {
    let bar = "═".repeat(64);
    let total = t1_elapsed + t2_elapsed;
    println!();
    println!("{}{}{}{}", BOLD, color, bar, RESET);
    println!(
        "{}{}  RESULT  •  {}  •  prefill turn1={} turn2={} (total {})  •  total wall {:?}{}",
        BOLD,
        color,
        mode,
        t1_prefill,
        t2_prefill,
        t1_prefill + t2_prefill,
        total,
        RESET
    );
    println!("{}{}{}{}", BOLD, color, bar, RESET);
}

fn comparison(b: &ModeResult, r: &ModeResult) {
    let b_total = b.turn1_prefill + b.turn2_prefill;
    let r_total = r.turn1_prefill + r.turn2_prefill;
    let prefill_ratio = if r_total > 0 {
        b_total as f64 / r_total as f64
    } else {
        0.0
    };
    let t2_ratio = if r.turn2_prefill > 0 {
        b.turn2_prefill as f64 / r.turn2_prefill as f64
    } else {
        0.0
    };
    let total_b = b.turn1_elapsed + b.turn2_elapsed;
    let total_r = r.turn1_elapsed + r.turn2_elapsed;
    let speedup = if total_r.as_secs_f64() > 0.0 {
        total_b.as_secs_f64() / total_r.as_secs_f64()
    } else {
        0.0
    };
    println!(
        "{}{}BASELINE prefill {}+{}={}, wall {:?}   ▸   RESUMED prefill {}+{}={}, wall {:?}   ▸   {:.2}× total prefill, {:.2}× turn-2 prefill, {:.2}× wall-time{}",
        BOLD,
        MAGENTA,
        b.turn1_prefill,
        b.turn2_prefill,
        b_total,
        total_b,
        r.turn1_prefill,
        r.turn2_prefill,
        r_total,
        total_r,
        prefill_ratio,
        t2_ratio,
        speedup,
        RESET
    );
    let _ = b.answer1.len();
    let _ = b.answer2.len();
    let _ = r.answer1.len();
    let _ = r.answer2.len();
}

fn oneline(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}
