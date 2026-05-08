//! Demo: best-of-N via `Context::fork()` shared-prefix decoding.
//!
//! Uniquely-Pie demo: the inferlet builds **one** system+user context,
//! `flush()`es it so the prefill is committed in a single batched forward
//! pass, then `Context::fork()`s **N** times and runs N concurrent
//! generate streams. All N forks share the same KV pages for the prompt,
//! so the engine pays the prefill cost exactly once. A naive baseline
//! issues N independent requests — each one re-prefills the full prompt
//! — which is what client-only patterns on vLLM/TGI/sglang force.
//!
//! Two strategies, same problem:
//!
//! - **BASELINE** — N independent `Context::new()`s, each with its own
//!   `system → user → cue` prefill. Fired concurrently via
//!   `future::join_all`, but the prefill compute is duplicated N times.
//! - **FORKED** — one prefill, then `fork()` ×N, decoded concurrently.
//!   Same wall-time win on long prompts, plus N× less redundant compute.
//!
//! `mode = plain | smart | both` (default `both`). For side-by-side
//! recording, run `--mode plain` on one pane and `--mode smart` on the
//! other.

use std::cell::RefCell;
use std::collections::HashMap;
use std::io::{self, Write};
use std::rc::Rc;
use std::time::{Duration, Instant};

use futures::future;
use inferlet::{Context, Result, chat, model::Model, runtime, sample::Sampler, wstd};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_mode")]
    mode: String,

    #[serde(default = "default_question")]
    question: String,

    #[serde(default = "default_expected")]
    expected: String,

    #[serde(default = "default_num_forks")]
    num_forks: usize,

    #[serde(default = "default_max_tokens")]
    max_tokens: usize,

    #[serde(default = "default_temperature")]
    temperature: f32,

    #[serde(default = "default_system")]
    system: String,

    #[serde(default)]
    delay: u64,
}

fn default_system() -> String {
    "You are a careful math tutor. Solve the problem step by step. Use plain \
     ASCII (no LaTeX, no markdown). Keep your reasoning under 8 short lines. \
     End with exactly one line: Final Answer: <value>"
        .into()
}

fn default_mode() -> String {
    "both".into()
}
fn default_question() -> String {
    // Plain arithmetic with one "right" answer that several reasoning
    // paths can converge on. Sampling at T=0.7 gives genuine diversity in
    // wording; the consensus pick is the mode-vote of extracted numbers.
    "What is 17 * 24 + 13? Show your reasoning step by step, then give \
     the final number on its own line as: Final Answer: <number>"
        .into()
}
fn default_expected() -> String {
    // 17 * 24 = 408; 408 + 13 = 421.
    "421".into()
}
fn default_num_forks() -> usize {
    4
}
fn default_max_tokens() -> usize {
    220
}
fn default_temperature() -> f32 {
    0.7
}

// ── ANSI helpers ───────────────────────────────────────────────────────
const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const YELLOW: &str = "\x1b[33m";
const GREEN: &str = "\x1b[32m";
const MAGENTA: &str = "\x1b[35m";
const RED: &str = "\x1b[31m";

// Per-fork colors (cycled if num_forks > FORK_COLORS.len()).
const FORK_COLORS: &[&str] = &[
    "\x1b[36m", // cyan
    "\x1b[35m", // magenta
    "\x1b[33m", // yellow
    "\x1b[32m", // green
    "\x1b[34m", // blue
    "\x1b[31m", // red
];
const FORK_LABELS: &[char] = &['A', 'B', 'C', 'D', 'E', 'F'];

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let mode = input.mode.to_lowercase();
    let n = input.num_forks.max(2);

    let model_name = runtime::models()
        .first()
        .cloned()
        .ok_or("No models available")?;
    let model = Model::load(&model_name)?;

    match mode.as_str() {
        "baseline" | "plain" => {
            run_baseline(&model, &model_name, &input, n).await?;
        }
        "forked" | "smart" => {
            run_forked(&model, &model_name, &input, n).await?;
        }
        "both" | "" => {
            let p = run_baseline(&model, &model_name, &input, n).await?;
            println!();
            let s = run_forked(&model, &model_name, &input, n).await?;
            println!();
            comparison(&p, &s, &input.expected);
        }
        other => {
            return Err(format!(
                "unknown mode '{}': expected 'baseline', 'forked', or 'both'",
                other
            ));
        }
    }

    Ok(String::new())
}

#[derive(Default, Clone)]
struct ModeResult {
    answers: Vec<Option<String>>,
    consensus: Option<String>,
    elapsed: Duration,
    prefill_tokens: u32,
    decode_tokens: usize,
    n: usize,
}

// ── BASELINE: N independent prefills, decoded concurrently ────────────
async fn run_baseline(
    model: &Model,
    model_name: &str,
    input: &Input,
    n: usize,
) -> Result<ModeResult> {
    print_header(
        "BASELINE",
        YELLOW,
        "N independent contexts — prompt prefilled N times",
        model_name,
        &input.question,
        n,
    );

    // Build N fully-independent contexts. Each one will prefill the full
    // system+user+cue from scratch.
    let mut ctxs = Vec::with_capacity(n);
    let mut per_ctx_prefill: u32 = 0;
    for _ in 0..n {
        let mut ctx = Context::new(model)?;
        ctx.system(&input.system);
        ctx.user(&format!("{} /no_think", input.question));
        ctx.cue();
        per_ctx_prefill = ctx.seq_len() + ctx.buffer().len() as u32;
        ctxs.push(ctx);
    }
    let total_prefill = per_ctx_prefill * n as u32;
    println!(
        "  {}prefill computed: {} tokens × {} contexts = {} total{}",
        DIM, per_ctx_prefill, n, total_prefill, RESET
    );
    println!();

    let printer = Rc::new(RefCell::new(Printer::new(n)));
    let start = Instant::now();
    let answers = run_streams(model, ctxs, &input, n, &printer).await?;
    let elapsed = start.elapsed();
    let decode_tokens = printer.borrow().decode_tokens();

    let consensus = mode_vote(&answers);
    print_footer(
        "BASELINE",
        YELLOW,
        &consensus,
        &input.expected,
        elapsed,
        total_prefill,
        decode_tokens,
    );
    Ok(ModeResult {
        answers,
        consensus,
        elapsed,
        prefill_tokens: total_prefill,
        decode_tokens,
        n,
    })
}

// ── FORKED: 1 prefill, fork() ×N, decoded concurrently ────────────────
async fn run_forked(
    model: &Model,
    model_name: &str,
    input: &Input,
    n: usize,
) -> Result<ModeResult> {
    print_header(
        "FORKED",
        GREEN,
        "1 prefill, Context::fork() ×N — KV pages shared",
        model_name,
        &input.question,
        n,
    );

    // Build one context, system+user, flush so the shared prefix commits
    // in a single batched forward pass. Then fork N times. The cue is
    // added per-fork so each fork has fresh tokens for its first
    // generate step (forks inherit committed KV but start with an empty
    // buffer; an empty buffer makes the first forward have zero inputs).
    let mut base = Context::new(model)?;
    base.system(&input.system);
    base.user(&format!("{} /no_think", input.question));
    let prefill = base.seq_len() + base.buffer().len() as u32;
    base.flush().await?;

    println!(
        "  {}prefill computed: {} tokens × 1 context (shared by all forks){}",
        DIM, prefill, RESET
    );
    println!();

    let mut ctxs = Vec::with_capacity(n);
    for _ in 0..n {
        let mut c = base.fork()?;
        c.cue();
        ctxs.push(c);
    }

    let printer = Rc::new(RefCell::new(Printer::new(n)));
    let start = Instant::now();
    let answers = run_streams(model, ctxs, &input, n, &printer).await?;
    let elapsed = start.elapsed();
    let decode_tokens = printer.borrow().decode_tokens();

    let consensus = mode_vote(&answers);
    print_footer(
        "FORKED",
        GREEN,
        &consensus,
        &input.expected,
        elapsed,
        prefill,
        decode_tokens,
    );
    Ok(ModeResult {
        answers,
        consensus,
        elapsed,
        prefill_tokens: prefill,
        decode_tokens,
        n,
    })
}

// ── Run N concurrent streams; return per-fork answers ─────────────────
async fn run_streams(
    model: &Model,
    ctxs: Vec<Context>,
    input: &Input,
    n: usize,
    printer: &Rc<RefCell<Printer>>,
) -> Result<Vec<Option<String>>> {
    let temperature = input.temperature;
    let max_tokens = input.max_tokens;
    let delay = input.delay;
    let stop = chat::stop_tokens(model);

    let futs: Vec<_> = ctxs
        .into_iter()
        .enumerate()
        .map(|(idx, mut ctx)| {
            let stop = stop.clone();
            let printer = printer.clone();
            let model = model;
            async move {
                let sampler = if temperature <= 0.0 {
                    Sampler::Argmax
                } else {
                    Sampler::TopP {
                        temperature,
                        p: 0.95,
                    }
                };
                let mut g = ctx.generate(sampler).max_tokens(max_tokens).stop(&stop);
                let mut decoder = chat::Decoder::new(model);
                let mut text = String::new();
                while let Some(step) = g.next()? {
                    let out = step.execute().await?;
                    if out.tokens.is_empty() {
                        continue;
                    }
                    printer.borrow_mut().count(idx, out.tokens.len());
                    match decoder.feed(&out.tokens)? {
                        chat::Event::Delta(s) => {
                            text.push_str(&s);
                            printer.borrow_mut().emit(idx, &s);
                            if delay > 0 {
                                wstd::task::sleep(wstd::time::Duration::from_millis(delay)).await;
                            }
                        }
                        chat::Event::Done(s) => {
                            text = s;
                            break;
                        }
                        _ => {}
                    }
                }
                Ok::<_, String>((idx, extract_answer(&text)))
            }
        })
        .collect();

    let results: Vec<_> = future::join_all(futs).await;
    printer.borrow_mut().finish();

    let mut answers = vec![None; n];
    for r in results {
        let (idx, ans) = r?;
        answers[idx] = ans;
    }
    Ok(answers)
}

// ── Interleaved per-fork stream printer ───────────────────────────────
//
// Each fork prints to stdout with a colored `[A] ` / `[B] ` … prefix.
// Switching between forks always starts on a fresh line so the visual
// stays scannable; trailing partial lines from prior fork are flushed
// with a newline before the next fork's prefix.
struct Printer {
    last: Option<usize>,
    at_line_start: bool,
    decode_tokens: Vec<usize>,
}

impl Printer {
    fn new(n: usize) -> Self {
        Self {
            last: None,
            at_line_start: true,
            decode_tokens: vec![0; n],
        }
    }

    fn count(&mut self, idx: usize, n: usize) {
        if idx < self.decode_tokens.len() {
            self.decode_tokens[idx] += n;
        }
    }

    fn decode_tokens(&self) -> usize {
        self.decode_tokens.iter().sum()
    }

    fn emit(&mut self, idx: usize, delta: &str) {
        if delta.is_empty() {
            return;
        }
        let switched = self.last.map(|p| p != idx).unwrap_or(true);
        if switched && !self.at_line_start {
            print!("\n");
            self.at_line_start = true;
        }
        for ch in delta.chars() {
            if self.at_line_start {
                self.print_prefix(idx);
                self.at_line_start = false;
            }
            if ch == '\n' {
                print!("\n");
                self.at_line_start = true;
            } else {
                print!("{}", ch);
            }
        }
        let _ = io::stdout().flush();
        self.last = Some(idx);
    }

    fn print_prefix(&self, idx: usize) {
        let label = FORK_LABELS.get(idx).copied().unwrap_or('?');
        let color = FORK_COLORS
            .get(idx % FORK_COLORS.len())
            .copied()
            .unwrap_or(RESET);
        print!("  {}{}[{}]{}  ", BOLD, color, label, RESET);
    }

    fn finish(&mut self) {
        if !self.at_line_start {
            println!();
            self.at_line_start = true;
        }
    }
}

// ── Final-answer extraction + mode vote ───────────────────────────────
fn extract_answer(text: &str) -> Option<String> {
    let lower = text.to_lowercase();
    let pos = lower.rfind("final answer")?;
    let tail = &text[pos..];
    let after_colon = tail.find(':').map(|i| &tail[i + 1..]).unwrap_or(tail);
    let line = after_colon
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .trim_end_matches(|c: char| matches!(c, '.' | ',' | ';' | '!' | '?' | '*' | '_'))
        .trim();
    if line.is_empty() {
        return None;
    }
    Some(normalize_answer(line))
}

fn normalize_answer(s: &str) -> String {
    let cleaned: String = s
        .chars()
        .filter(|c| !matches!(c, '*' | '_' | '`' | '"' | '\''))
        .collect();
    cleaned
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_uppercase()
}

fn mode_vote(answers: &[Option<String>]) -> Option<String> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for a in answers.iter().flatten() {
        *counts.entry(a.clone()).or_insert(0) += 1;
    }
    counts.into_iter().max_by_key(|(_, c)| *c).map(|(k, _)| k)
}

// ── TUI helpers ────────────────────────────────────────────────────────
fn print_header(
    mode: &str,
    color: &str,
    tagline: &str,
    model_name: &str,
    question: &str,
    n: usize,
) {
    let bar = "═".repeat(64);
    println!("{}{}{}{}", BOLD, color, bar, RESET);
    println!(
        "{}{}  PARALLEL FORK DEMO   ▸   mode: {}{}  ({}){}",
        BOLD, color, mode, RESET, tagline, RESET
    );
    println!("  {}model {}  •  N={} forks{}", DIM, model_name, n, RESET);
    println!("{}{}{}{}", BOLD, color, bar, RESET);
    println!();
    println!("  {}Q:{} {}", BOLD, RESET, oneline(question));
    println!();
}

fn print_footer(
    mode: &str,
    color: &str,
    consensus: &Option<String>,
    expected: &str,
    elapsed: Duration,
    prefill_tokens: u32,
    decode_tokens: usize,
) {
    let bar = "═".repeat(64);
    let pick = consensus.as_deref().unwrap_or("(no consensus)");
    let ok = consensus
        .as_deref()
        .map(|p| p.eq_ignore_ascii_case(&normalize_answer(expected)))
        .unwrap_or(false);
    let verdict = if ok {
        format!(" {}{}✓ matches expected{}", BOLD, GREEN, RESET)
    } else {
        format!(" {}{}✗ expected {}{}", BOLD, RED, expected, RESET)
    };
    println!();
    println!("{}{}{}{}", BOLD, color, bar, RESET);
    println!(
        "{}{}  RESULT  •  {}  •  consensus {}{}{}{}  •  {:?}{}{}",
        BOLD, color, mode, BOLD, pick, RESET, color, elapsed, verdict, RESET
    );
    println!(
        "  {}prefill {} tokens  •  decode {} tokens{}",
        DIM, prefill_tokens, decode_tokens, RESET
    );
    println!("{}{}{}{}", BOLD, color, bar, RESET);
}

fn comparison(b: &ModeResult, f: &ModeResult, expected: &str) {
    let prefill_ratio = if f.prefill_tokens == 0 {
        0.0
    } else {
        b.prefill_tokens as f64 / f.prefill_tokens as f64
    };
    let speedup = if f.elapsed.as_secs_f64() > 0.0 {
        b.elapsed.as_secs_f64() / f.elapsed.as_secs_f64()
    } else {
        0.0
    };
    println!(
        "{}{}BASELINE {:?}, prefill {} tokens   ▸   FORKED {:?}, prefill {} tokens   ▸   {:.2}× less prefill, {:.2}× wall-time{}",
        BOLD,
        MAGENTA,
        b.elapsed,
        b.prefill_tokens,
        f.elapsed,
        f.prefill_tokens,
        prefill_ratio,
        speedup,
        RESET
    );
    let exp = normalize_answer(expected);
    let b_ok = b
        .consensus
        .as_deref()
        .map(|s| s.eq_ignore_ascii_case(&exp))
        .unwrap_or(false);
    let f_ok = f
        .consensus
        .as_deref()
        .map(|s| s.eq_ignore_ascii_case(&exp))
        .unwrap_or(false);
    match (b_ok, f_ok) {
        (true, true) => println!("{}both consensus answers matched expected.{}", DIM, RESET),
        (false, true) => println!(
            "{}{}FORKED converged on the right answer where BASELINE didn't.{}",
            BOLD, GREEN, RESET
        ),
        (true, false) => println!(
            "{}{}BASELINE got it but FORKED's consensus drifted this run.{}",
            BOLD, YELLOW, RESET
        ),
        (false, false) => println!(
            "{}{}neither consensus matched expected ({}).{}",
            BOLD, RED, expected, RESET
        ),
    }
    let _ = (b.n, f.n, b.decode_tokens, f.decode_tokens);
    let _ = b.answers.len();
    let _ = f.answers.len();
}

fn oneline(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}
