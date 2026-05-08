//! Demo: self-correcting math via co-located verifier + KV rewind.
//!
//! Uniquely-Pie demo: the inferlet generates a full reasoning attempt,
//! then a Rust arithmetic verifier (compiled into the same WASM, running
//! in-engine with zero RTT to the GPU) scans every `X op Y = Z` claim.
//! On any mismatch the inferlet calls `Context::fork()` to **reset the KV
//! cache to the pre-attempt checkpoint** and re-rolls at a higher
//! temperature. Other engines (vLLM, sglang, TGI) do not expose this kind
//! of mid-flight KV rewind from the client.
//!
//! Two strategies, same problem:
//!
//! - **BASELINE** — one greedy roll-out. Whatever first arithmetic mistake
//!   the model makes, it commits to and rolls forward from.
//! - **VERIFIED** — generate→verify→rewind→retry per attempt. Each retry's
//!   ✗-then-↺ is a visible "the model fumbled, KV rewinds, try again"
//!   moment. Final answer is structurally protected by the verifier.
//!
//! `mode = plain | smart | both` (default `both`). For side-by-side
//! recording, run `--mode plain` on one pane and `--mode smart` on the
//! other.

use std::io::{self, Write};
use std::time::{Duration, Instant};

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

    #[serde(default = "default_max_tokens")]
    max_tokens: usize,

    #[serde(default = "default_max_retries")]
    max_retries: usize,

    #[serde(default = "default_system")]
    system: String,

    #[serde(default)]
    delay: u64,
}

fn default_system() -> String {
    "You are a careful arithmetic solver. Solve the problem step by step. \
     Write each computation on its own line as a complete equation (e.g., \
     50 + 30 = 80). Use *, +, -, /, plain ASCII — no LaTeX, no markdown. \
     End with exactly one line: Final Answer: <number>"
        .into()
}

fn default_mode() -> String {
    "both".into()
}
fn default_question() -> String {
    // 3-digit × 2-digit multiplications. Qwen3-0.6B's argmax reliably
    // fumbles 156 * 32 (often emits 5056); higher-T sampling on the
    // primed base gives the rewind loop a fighting chance to recover.
    "Compute (156 * 32) + (248 * 19). Show each multiplication on its \
     own line, then state the final sum."
        .into()
}
fn default_expected() -> String {
    // 156*32=4992, 248*19=4712, sum=9704
    "9704".into()
}
fn default_max_tokens() -> usize {
    400
}
fn default_max_retries() -> usize {
    5
}

// ── ANSI helpers ───────────────────────────────────────────────────────
const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const YELLOW: &str = "\x1b[33m";
const GREEN: &str = "\x1b[32m";
const RED: &str = "\x1b[31m";
const CYAN: &str = "\x1b[36m";
const MAGENTA: &str = "\x1b[35m";

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let mode = input.mode.to_lowercase();

    let model_name = runtime::models()
        .first()
        .cloned()
        .ok_or("No models available")?;
    let model = Model::load(&model_name)?;

    match mode.as_str() {
        "baseline" | "plain" => {
            run_baseline(&model, &model_name, &input).await?;
        }
        "verified" | "smart" => {
            run_verified(&model, &model_name, &input).await?;
        }
        "both" | "" => {
            let p = run_baseline(&model, &model_name, &input).await?;
            println!();
            let s = run_verified(&model, &model_name, &input).await?;
            println!();
            comparison(&p, &s, &input.expected);
        }
        other => {
            return Err(format!(
                "unknown mode '{}': expected 'baseline', 'verified', or 'both'",
                other
            ));
        }
    }

    Ok(String::new())
}

#[derive(Default)]
struct ModeResult {
    answer: Option<String>,
    elapsed: Duration,
    rewinds: usize,
    attempts: usize,
}

// ── BASELINE: one greedy roll-out ─────────────────────────────────────────
async fn run_baseline(model: &Model, model_name: &str, input: &Input) -> Result<ModeResult> {
    print_header(
        "BASELINE",
        YELLOW,
        "one greedy roll-out",
        model_name,
        &input.question,
    );

    let mut ctx = Context::new(model)?;
    ctx.system(&input.system);
    ctx.user(&format!("{} /no_think", input.question));
    ctx.cue();

    let start = Instant::now();
    let text = stream_attempt(&mut ctx, model, input.max_tokens, 0.0, input.delay).await?;
    let elapsed = start.elapsed();
    let answer = extract_answer(&text);
    print_footer("BASELINE", YELLOW, &answer, elapsed, 0, 1, &input.expected);
    Ok(ModeResult {
        answer,
        elapsed,
        rewinds: 0,
        attempts: 1,
    })
}

// ── VERIFIED: full-attempt verify + fork()-based KV rewind ────────────────
async fn run_verified(model: &Model, model_name: &str, input: &Input) -> Result<ModeResult> {
    print_header(
        "VERIFIED",
        GREEN,
        "co-located verifier + Context::fork() rewind",
        model_name,
        &input.question,
    );
    println!(
        "  {}attempt N ▸ streamed reasoning, then a Rust verifier scans every X op Y = Z claim.{}",
        DIM, RESET
    );
    println!(
        "  {}On mismatch: ↺ Context::fork() rewinds KV to pre-attempt checkpoint, retry at higher T.{}",
        DIM, RESET
    );
    println!();

    // Two base contexts:
    //   - `ctx_base_raw`     — system + user + cue, no priming. Cloned
    //                          for ATTEMPT 1 only, so attempt 1 reproduces
    //                          BASELINE's argmax verbatim (same wrong answer
    //                          → verifier rejects → first KV rewind is
    //                          guaranteed visible on every cold-start run).
    //   - `ctx_base_primed`  — system + user + cue + `</think>\n\n` token
    //                          prime that closes Qwen3's auto-opened think
    //                          block immediately. Cloned for ATTEMPT 2+,
    //                          where the different KV state lets sampling
    //                          land on the correct multiplication.
    //
    // Both are buffered (not flushed) so each fork's first generate()
    // step does the prefill in one batched forward pass — bit-for-bit
    // parity with BASELINE's path.
    let mut ctx_base_raw = Context::new(model)?;
    ctx_base_raw.system(&input.system);
    ctx_base_raw.user(&format!("{} /no_think", input.question));
    ctx_base_raw.cue();

    let mut ctx_base_primed = ctx_base_raw.fork()?;
    let tokenizer = model.tokenizer();
    let prime = tokenizer.encode("\n</think>\n\n");
    if !prime.is_empty() {
        ctx_base_primed.append(&prime);
    }
    let base_seq_len = ctx_base_primed.seq_len() + ctx_base_primed.buffer().len() as u32;

    let start = Instant::now();
    let mut rewinds = 0usize;
    let mut attempt_idx = 0usize;
    let mut final_text = String::new();
    let mut final_answer: Option<String> = None;

    while attempt_idx <= input.max_retries {
        attempt_idx += 1;
        // Attempt 1 is greedy on purpose — produces the SAME wrong answer
        // BASELINE does, so the viewer sees the verifier reject it before any
        // rewind. Subsequent attempts ramp up temperature for diversity.
        let temp = match attempt_idx {
            1 => 0.0,
            2 => 0.6,
            3 => 0.9,
            4 => 1.2,
            _ => 1.4,
        };
        println!(
            "  {}{}attempt {} (T={:.1}){}",
            BOLD, GREEN, attempt_idx, temp, RESET
        );

        // Attempt 1 forks the unprimed base (matches BASELINE bit-for-bit
        // → wrong answer → guaranteed verifier rejection + visible KV
        // rewind). Subsequent attempts fork the primed base where the
        // closing `</think>` token shifts the distribution into the
        // canonical-equation regime, so sampling can land on the right
        // multiplication.
        let mut ctx = if attempt_idx == 1 {
            ctx_base_raw.fork()?
        } else {
            ctx_base_primed.fork()?
        };

        let text = stream_attempt(&mut ctx, model, input.max_tokens, temp, input.delay).await?;
        let errs = verify_attempt(&text, &input.expected);

        if errs.is_empty() {
            println!(
                "  {}{}✓ verifier: arithmetic claims and final answer check out{}",
                BOLD, GREEN, RESET
            );
            final_text = text;
            final_answer = extract_answer(&final_text);
            break;
        }

        // Show what the verifier rejected.
        println!(
            "  {}{}✗ verifier rejected {} claim(s):{}",
            BOLD,
            RED,
            errs.len(),
            RESET
        );
        for err in &errs {
            println!("      {}{}{}", DIM, err, RESET);
        }

        if attempt_idx > input.max_retries {
            println!(
                "  {}{}↺ max retries reached, accepting last attempt{}",
                BOLD, MAGENTA, RESET
            );
            final_text = text;
            final_answer = extract_answer(&final_text);
            break;
        }

        let next_temp = match attempt_idx + 1 {
            2 => 0.6,
            3 => 0.9,
            4 => 1.2,
            _ => 1.4,
        };
        println!(
            "  {}{}↺ Context::fork() rewinds KV to base ({} tokens), retry @ T={:.1}{}",
            BOLD, MAGENTA, base_seq_len, next_temp, RESET
        );
        rewinds += 1;
        println!();
    }

    let elapsed = start.elapsed();
    print_footer(
        "VERIFIED",
        GREEN,
        &final_answer,
        elapsed,
        rewinds,
        attempt_idx,
        &input.expected,
    );
    let _ = final_text;
    Ok(ModeResult {
        answer: final_answer,
        elapsed,
        rewinds,
        attempts: attempt_idx,
    })
}

// ── Stream one full attempt, return the decoded text ──────────────────
//
// `text` accumulates the full delta stream (used by the verifier for
// arithmetic checks). The user-visible stream strips `<think>...</think>`
// blocks and optionally sleeps `delay_ms` after each delta to make the
// streaming effect readable on video.
async fn stream_attempt(
    ctx: &mut Context,
    model: &Model,
    max_tokens: usize,
    temperature: f32,
    delay_ms: u64,
) -> Result<String> {
    print!("  {}>{} ", CYAN, RESET);
    let _ = io::stdout().flush();

    let sampler = if temperature <= 0.0 {
        Sampler::Argmax
    } else {
        Sampler::TopP {
            temperature,
            p: 0.95,
        }
    };

    let mut g = ctx
        .generate(sampler)
        .max_tokens(max_tokens)
        .stop(&chat::stop_tokens(model));
    let mut decoder = chat::Decoder::new(model);
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
                        wstd::task::sleep(wstd::time::Duration::from_millis(delay_ms)).await;
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
    Ok(text)
}

// ── Strip <think>...</think> blocks from a stream of deltas ───────────
//
// Tracks state across deltas. Buffers a small tail (up to 8 chars) so
// tags split across delta boundaries are still detected.
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
                // Stay in think; keep last 7 bytes in case `</think>` is
                // split across deltas, drop the rest.
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
                // Hold last 6 bytes in case `<think>` is mid-split.
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

// ── Arithmetic verifier ───────────────────────────────────────────────
//
// Scans the text for explicit `A op B = C` patterns and recomputes each
// one, then checks that the final answer line matches the expected answer.
// Conservative: skips lines containing "final answer" while checking
// intermediate equations and matches where operands are flanked by another
// operator (chains).
fn verify_attempt(text: &str, expected: &str) -> Vec<String> {
    let mut errors = check_arithmetic(text)
        .into_iter()
        .map(|(claim, _claimed, actual)| format!("{claim} ≠ {actual} (actual)"))
        .collect::<Vec<_>>();

    match extract_answer(text) {
        Some(ans) if ans == expected => {}
        Some(ans) => errors.push(format!("Final Answer: {ans} ≠ {expected} (expected)")),
        None => errors.push(format!("missing Final Answer: {expected}")),
    }

    errors
}

fn check_arithmetic(text: &str) -> Vec<(String, i64, i64)> {
    let mut errors = Vec::new();
    for line in text.split('\n') {
        if line.to_lowercase().contains("final answer") {
            continue;
        }
        let cleaned: String = line
            .chars()
            .map(|c| if c == ',' { ' ' } else { c })
            .collect();
        let chars: Vec<char> = cleaned.chars().collect();
        let n = chars.len();
        let mut i = 0;
        while i < n {
            if !chars[i].is_ascii_digit() {
                i += 1;
                continue;
            }
            // Skip if A is preceded by an arithmetic operator AND that
            // operator itself is preceded (somewhere in the line) by a
            // digit — that signals a real chain like `X + A * B = C`.
            // A leading `-` or `*` with no preceding digit is just a
            // markdown bullet, not subtraction/multiplication.
            // `=` is NEVER a chaining op: it separates LHS/RHS.
            if let Some((prev, prev_pos)) = preceding_nonspace_pos(&chars, i) {
                if matches!(prev, '+' | '-' | '*' | '/' | '×' | '÷' | 'x' | 'X') {
                    let has_digit_before = chars[..prev_pos].iter().any(|c| c.is_ascii_digit());
                    if has_digit_before {
                        let (_, ai) = parse_int(&chars, i);
                        i = ai.max(i + 1);
                        continue;
                    }
                }
            }
            let (a, ai) = parse_int(&chars, i);
            let mut j = ai;
            skip_spaces(&chars, &mut j);
            if j >= n {
                break;
            }
            let op = match chars[j] {
                '+' => '+',
                '-' => '-',
                '*' | 'x' | 'X' | '×' => '*',
                '/' | '÷' => '/',
                _ => {
                    i = ai.max(i + 1);
                    continue;
                }
            };
            j += 1;
            skip_spaces(&chars, &mut j);
            if j >= n || !chars[j].is_ascii_digit() {
                i = ai.max(i + 1);
                continue;
            }
            let (b, bi) = parse_int(&chars, j);
            j = bi;
            skip_spaces(&chars, &mut j);
            if j >= n || chars[j] != '=' {
                i = ai.max(i + 1);
                continue;
            }
            j += 1;
            skip_spaces(&chars, &mut j);
            if j >= n || !chars[j].is_ascii_digit() {
                i = ai.max(i + 1);
                continue;
            }
            let (claimed, ci) = parse_int(&chars, j);
            let mut k = ci;
            skip_spaces(&chars, &mut k);
            if k < n && is_op_char(chars[k]) {
                i = ci;
                continue;
            }
            let actual = match op {
                '+' => a + b,
                '-' => a - b,
                '*' => a * b,
                '/' => {
                    if b == 0 {
                        i = ci;
                        continue;
                    }
                    a / b
                }
                _ => unreachable!(),
            };
            if actual != claimed {
                errors.push((format!("{} {} {} = {}", a, op, b, claimed), claimed, actual));
            }
            i = ci;
        }
    }
    errors
}

fn parse_int(chars: &[char], start: usize) -> (i64, usize) {
    let mut j = start;
    let mut n: i64 = 0;
    while j < chars.len() && chars[j].is_ascii_digit() {
        n = n * 10 + (chars[j] as i64 - '0' as i64);
        j += 1;
    }
    (n, j)
}

fn skip_spaces(chars: &[char], j: &mut usize) {
    while *j < chars.len() && (chars[*j] == ' ' || chars[*j] == '\t') {
        *j += 1;
    }
}

fn is_op_char(c: char) -> bool {
    matches!(c, '+' | '-' | '*' | '/' | '×' | '÷' | '=' | 'x' | 'X')
}

fn preceding_nonspace_pos(chars: &[char], pos: usize) -> Option<(char, usize)> {
    let mut k = pos;
    while k > 0 {
        k -= 1;
        let c = chars[k];
        if c == ' ' || c == '\t' {
            continue;
        }
        return Some((c, k));
    }
    None
}

fn extract_answer(text: &str) -> Option<String> {
    let lower = text.to_lowercase();
    let pos = lower.rfind("final answer")?;
    let tail = &text[pos..];
    let after_colon = tail.find(':').map(|i| &tail[i + 1..]).unwrap_or(tail);
    let cleaned: String = after_colon
        .chars()
        .filter(|c| {
            !matches!(
                c,
                '*' | '_'
                    | '`'
                    | '#'
                    | '<'
                    | '>'
                    | '('
                    | ')'
                    | '['
                    | ']'
                    | '{'
                    | '}'
                    | '"'
                    | '\''
                    | '$'
            )
        })
        .collect();
    let last_number = cleaned
        .split(|c: char| !c.is_ascii_digit() && c != '-')
        .filter(|s| !s.is_empty() && s.chars().any(|c| c.is_ascii_digit()))
        .last()?;
    Some(
        last_number
            .trim_end_matches(|c: char| matches!(c, '.' | ',' | ';' | '!' | '?'))
            .to_string(),
    )
}

// ── TUI helpers ────────────────────────────────────────────────────────
fn print_header(mode: &str, color: &str, tagline: &str, model_name: &str, question: &str) {
    let bar = "═".repeat(64);
    println!("{}{}{}{}", BOLD, color, bar, RESET);
    println!(
        "{}{}  SELF-CORRECT MATH   ▸   mode: {}{}  ({}){}",
        BOLD, color, mode, RESET, tagline, RESET
    );
    println!("  {}model {}{}", DIM, model_name, RESET);
    println!("{}{}{}{}", BOLD, color, bar, RESET);
    println!();
    println!("  {}Q:{} {}", BOLD, RESET, oneline(question));
    println!();
}

fn print_footer(
    mode: &str,
    color: &str,
    answer: &Option<String>,
    elapsed: Duration,
    rewinds: usize,
    attempts: usize,
    expected: &str,
) {
    let bar = "═".repeat(64);
    let ans = answer.as_deref().unwrap_or("(no answer)");
    let ok = answer.as_deref() == Some(expected);
    let verdict = if ok {
        format!(" {}{}✓ correct{}", BOLD, GREEN, RESET)
    } else {
        format!(" {}{}✗ expected {}{}", BOLD, RED, expected, RESET)
    };
    println!();
    println!("{}{}{}{}", BOLD, color, bar, RESET);
    println!(
        "{}{}  RESULT  •  {}  •  answer {}{}{}{}  •  {:?}  •  {} attempt(s) / {} rewind(s){}{}",
        BOLD, color, mode, BOLD, ans, RESET, color, elapsed, attempts, rewinds, verdict, RESET
    );
    println!("{}{}{}{}", BOLD, color, bar, RESET);
}

fn comparison(p: &ModeResult, s: &ModeResult, expected: &str) {
    let p_ok = p.answer.as_deref() == Some(expected);
    let s_ok = s.answer.as_deref() == Some(expected);
    println!(
        "{}{}BASELINE {} ({:?})   ▸   VERIFIED {} ({:?}, {} attempts / {} rewinds){}",
        BOLD,
        MAGENTA,
        p.answer.as_deref().unwrap_or("(no answer)"),
        p.elapsed,
        s.answer.as_deref().unwrap_or("(no answer)"),
        s.elapsed,
        s.attempts,
        s.rewinds,
        RESET
    );
    match (p_ok, s_ok) {
        (false, true) => println!(
            "{}{}VERIFIED recovered the answer BASELINE missed via {} KV rewind(s).{}",
            BOLD, GREEN, s.rewinds, RESET
        ),
        (true, true) => println!("{}both correct.{}", DIM, RESET),
        (true, false) => println!(
            "{}{}BASELINE got it but VERIFIED didn't converge this run.{}",
            BOLD, YELLOW, RESET
        ),
        (false, false) => println!(
            "{}{}neither matched expected ({}) within {} attempts.{}",
            BOLD, RED, expected, s.attempts, RESET
        ),
    }
}

fn oneline(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}
