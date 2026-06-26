//! Demo: KV state persists across requests via `Context::save` / `open`.
//!
//! Uniquely-Pie demo: a chat session's KV cache lives inside the engine
//! and survives across inferlet invocations under a user-chosen name.
//! Turn 2 reopens the snapshot and only pays the prefill cost for the
//! new user message — the prior turn's pages are reused as-is. A naive
//! client-server pattern (the way most apps work today on vLLM/TGI)
//! has to replay the full conversation history every turn, paying the
//! full prefill bill again and again.
//!
//! Two strategies, same conversation:
//!
//! - **BASELINE** — turn 1 runs a fresh `Context::new()`. Turn 2 throws
//!   that context away and rebuilds another fresh one with
//!   system + user1 + assistant1 + user2 in its buffer. Pays prefill on
//!   the entire history every time.
//! - **RESUMED** — turn 1 saves the post-generation context under a
//!   name. Turn 2 calls `Context::open(name)` which forks from the
//!   saved snapshot, appends user2, and generates. Only the new user2
//!   wrapper costs prefill on turn 2.
//!
//! `mode = plain | smart | both` (default `both`).

use std::io::{self, Write};
use std::time::{Duration, Instant};

use inferlet::{Context, Result, chat, sample::Sampler};
use serde::Deserialize;

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

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let mode = input.mode.to_lowercase();

    let model_name = inferlet::model::name();

    // Best-effort: clear stale snapshots before modes that create a new one.
    if !matches!(mode.as_str(), "open" | "turn2" | "resume-turn2") {
        let _ = Context::delete(&input.snapshot);
    }

    match mode.as_str() {
        "baseline" | "plain" => {
            run_baseline(&model_name, &input).await?;
        }
        "resumed" | "smart" => {
            run_resumed(&model_name, &input).await?;
            let _ = Context::delete(&input.snapshot);
        }
        "save" | "turn1" | "save-turn1" => {
            run_save_turn1(&model_name, &input).await?;
        }
        "open" | "turn2" | "resume-turn2" => {
            run_open_turn2(&model_name, &input).await?;
            let _ = Context::delete(&input.snapshot);
        }
        "both" | "" => {
            let b = run_baseline(&model_name, &input).await?;
            println!();
            // Reset snapshot before resumed run.
            let _ = Context::delete(&input.snapshot);
            let r = run_resumed(&model_name, &input).await?;
            println!();
            comparison(&b, &r);
            let _ = Context::delete(&input.snapshot);
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
    let mut ctx = Context::new()?;
    ctx.system(&input.system);
    ctx.user(&format!("{} /no_think", input.turn1.trim()));
    ctx.cue();
    let turn1_prefill = ctx.seq_len() + ctx.buffer().len() as u32;
    let t = Instant::now();
    let answer1 = stream_attempt(&mut ctx, input.max_tokens, input.delay).await?;
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
    let mut ctx2 = Context::new()?;
    ctx2.system(&input.system);
    ctx2.user(&input.turn1);
    ctx2.assistant(answer1.trim());
    ctx2.user(&format!("{} /no_think", input.turn2.trim()));
    ctx2.cue();
    let turn2_prefill = ctx2.seq_len() + ctx2.buffer().len() as u32;
    let t = Instant::now();
    let answer2 = stream_attempt(&mut ctx2, input.max_tokens, input.delay).await?;
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
        "Context::save() after turn 1; Context::open() before turn 2",
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
    let mut ctx = Context::new()?;
    ctx.system(&input.system);
    ctx.user(&format!("{} /no_think", input.turn1.trim()));
    ctx.cue();
    let turn1_prefill = ctx.seq_len() + ctx.buffer().len() as u32;
    let t = Instant::now();
    let answer1 = stream_attempt(&mut ctx, input.max_tokens, input.delay).await?;
    let turn1_elapsed = t.elapsed();
    // Commit any working pages before saving so the snapshot includes
    // the assistant's reply.
    ctx.flush().await?;
    ctx.save(&input.snapshot)?;
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
    let mut ctx2 = Context::open(&input.snapshot)?;
    let pre_open_seq = ctx2.seq_len();
    ctx2.user(&format!("{} /no_think", input.turn2.trim()));
    ctx2.cue();
    let turn2_prefill = (ctx2.seq_len() + ctx2.buffer().len() as u32) - pre_open_seq;
    let t = Instant::now();
    let answer2 = stream_attempt(&mut ctx2, input.max_tokens, input.delay).await?;
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
    let mut ctx = Context::new()?;
    ctx.system(&input.system);
    ctx.user(&format!("{} /no_think", input.turn1.trim()));
    ctx.cue();
    let turn1_prefill = ctx.seq_len() + ctx.buffer().len() as u32;
    let t = Instant::now();
    let _answer1 = stream_attempt(&mut ctx, input.max_tokens, input.delay).await?;
    let turn1_elapsed = t.elapsed();
    ctx.flush().await?;
    ctx.save(&input.snapshot)?;
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
    let mut ctx = Context::open(&input.snapshot)?;
    let pre_open_seq = ctx.seq_len();
    println!(
        "  {}{}turn 2{}  user: {}",
        BOLD,
        GREEN,
        RESET,
        oneline(&input.turn2)
    );
    ctx.user(&format!("{} /no_think", input.turn2.trim()));
    ctx.cue();
    let turn2_prefill = (ctx.seq_len() + ctx.buffer().len() as u32) - pre_open_seq;
    let t = Instant::now();
    let _answer2 = stream_attempt(&mut ctx, input.max_tokens, input.delay).await?;
    let turn2_elapsed = t.elapsed();
    println!(
        "  {}new prefill {} tokens (history reused), decode {:?}{}",
        DIM, turn2_prefill, turn2_elapsed, RESET
    );
    Ok(())
}

// ── Stream one turn, return the decoded text ──────────────────────────
async fn stream_attempt(
    ctx: &mut Context,
    max_tokens: usize,
    delay_ms: u64,
) -> Result<String> {
    print!("  {}>{} ", CYAN, RESET);
    let _ = io::stdout().flush();

    let mut g = ctx
        .generate(Sampler::TopP {
            temperature: 0.6,
            p: 0.95,
        })
        .max_tokens(max_tokens)
        .stop(&chat::stop_tokens());
    let mut decoder = chat::Decoder::new();
    let mut stripper = ThinkStripper::new();
    let mut text = String::new();

    while let Some(step) = g.next()? {
        let out = step.execute().await?;
        if out.tokens.is_empty() {
            continue;
        }
        match decoder.feed(&out.tokens)? {
            chat::Event::Delta(s) => {
                text.push_str(&s);
                let visible = stripper.process(&s);
                if !visible.is_empty() {
                    let rendered = visible.replace('\n', "\n    ");
                    print!("{}", rendered);
                    let _ = io::stdout().flush();
                    if delay_ms > 0 {
                        inferlet::sleep(std::time::Duration::from_millis(delay_ms)).await;
                    }
                }
            }
            chat::Event::Done(s) => {
                text = s;
                break;
            }
            _ => {}
        }
    }
    println!();
    // Strip any think blocks from the captured assistant text so callers
    // can replay it cleanly.
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
