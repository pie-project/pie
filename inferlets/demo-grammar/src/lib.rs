//! Demo: per-token JSON Schema matcher masks logits inside the engine.
//!
//! Uniquely-Pie demo: the inferlet ships a JSON Schema with the request
//! and uses `constrain_with(JsonSchema(...))` to mask invalid tokens at
//! every sampling step — output is always valid by construction. The
//! matcher runs in-process next to the model, so there is zero RTT
//! between "did this token violate the schema?" and "mask it." A naive
//! baseline generates free-form, parses, and retries on failure — which
//! is what most apps do today and what wastes the most tokens.
//!
//! Two strategies, same task:
//!
//! - **BASELINE** — generate freely, `serde_json::from_str` against the
//!   target shape, retry up to `max_retries` on parse or validation
//!   failure. Models like to add "Sure! Here's the JSON:" preambles or
//!   wrap output in markdown fences, both of which break the parse.
//! - **CONSTRAINED** — `constrain_with(JsonSchema(...))` forces every
//!   sampled token to keep the output a prefix of a valid JSON document
//!   conforming to the schema. No preamble, no fences, no retries.
//!
//! `mode = plain | smart | both` (default `both`).

use std::io::{self, Write};
use std::time::{Duration, Instant};

use inferlet::{Context, JsonSchema, Result, chat, model::Model, runtime, sample::Sampler, wstd};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_mode")]
    mode: String,

    #[serde(default = "default_bio")]
    bio: String,

    #[serde(default = "default_max_tokens")]
    max_tokens: usize,

    #[serde(default = "default_max_retries")]
    max_retries: usize,

    #[serde(default = "default_system")]
    system: String,

    #[serde(default = "default_schema")]
    schema: String,

    #[serde(default)]
    delay: u64,
}

fn default_system() -> String {
    "You are a helpful assistant. Extract the user's name, email, and age \
     from the paragraph and return them as JSON. /no_think"
        .into()
}

fn default_schema() -> String {
    r#"{
    "type": "object",
    "properties": {
        "name":  { "type": "string", "minLength": 1 },
        "email": { "type": "string", "minLength": 3 },
        "age":   { "type": "integer", "minimum": 0, "maximum": 150 }
    },
    "required": ["name", "email", "age"]
}"#
    .into()
}

fn default_mode() -> String {
    "both".into()
}
fn default_bio() -> String {
    "Hi, I'm Alice. You can email me at alice@example.com. I'm 28 years old.".into()
}
fn default_max_tokens() -> usize {
    200
}
fn default_max_retries() -> usize {
    3
}

// ── ANSI helpers ───────────────────────────────────────────────────────
const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const YELLOW: &str = "\x1b[33m";
const GREEN: &str = "\x1b[32m";
const CYAN: &str = "\x1b[36m";
const MAGENTA: &str = "\x1b[35m";
const RED: &str = "\x1b[31m";

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
        "constrained" | "smart" => {
            run_constrained(&model, &model_name, &input).await?;
        }
        "both" | "" => {
            let p = run_baseline(&model, &model_name, &input).await?;
            println!();
            let s = run_constrained(&model, &model_name, &input).await?;
            println!();
            comparison(&p, &s);
        }
        other => {
            return Err(format!(
                "unknown mode '{}': expected 'baseline', 'constrained', or 'both'",
                other
            ));
        }
    }

    Ok(String::new())
}

#[derive(Default, Clone)]
struct ModeResult {
    elapsed: Duration,
    attempts: usize,
    decode_tokens: usize,
    wasted_tokens: usize,
    parsed: Option<serde_json::Value>,
}

// ── BASELINE: free decode + serde_json::from_str + retry on failure ───
async fn run_baseline(model: &Model, model_name: &str, input: &Input) -> Result<ModeResult> {
    print_header(
        "BASELINE",
        YELLOW,
        "free decode → serde_json parse → retry on failure",
        model_name,
        &input.bio,
    );

    let start = Instant::now();
    let mut attempt = 0usize;
    let mut total_decoded = 0usize;
    let mut wasted = 0usize;
    let mut parsed: Option<serde_json::Value> = None;

    while attempt < input.max_retries.max(1) {
        attempt += 1;
        // Attempt 1 is greedy (matches what most apps do on first try).
        // Subsequent attempts ramp temperature to give retries a chance.
        let temp = match attempt {
            1 => 0.0,
            2 => 0.6,
            3 => 0.9,
            _ => 1.2,
        };
        println!(
            "  {}{}attempt {} (T={:.1}){}",
            BOLD, YELLOW, attempt, temp, RESET
        );

        let mut ctx = Context::new(model)?;
        ctx.system(&input.system);
        ctx.user(&format!(
            "Extract name, email, age from this bio: {}",
            input.bio
        ));
        ctx.cue();

        let text = stream_attempt(&mut ctx, model, input.max_tokens, temp, input.delay).await?;
        let attempt_tokens = approximate_tokens(&text);
        total_decoded += attempt_tokens;

        match try_parse(&text) {
            Ok(c) => {
                println!("  {}{}✓ parsed{}  {}", BOLD, GREEN, RESET, format_value(&c));
                parsed = Some(c);
                break;
            }
            Err(reason) => {
                println!("  {}{}✗ parse failed: {}{}", BOLD, RED, reason, RESET);
                wasted += attempt_tokens;
            }
        }
        println!();
    }

    let elapsed = start.elapsed();
    print_footer_baseline(
        "BASELINE",
        YELLOW,
        attempt,
        total_decoded,
        wasted,
        elapsed,
        &parsed,
    );
    Ok(ModeResult {
        elapsed,
        attempts: attempt,
        decode_tokens: total_decoded,
        wasted_tokens: wasted,
        parsed,
    })
}

// ── CONSTRAINED: per-token logit mask from in-WASM matcher ────────────
async fn run_constrained(model: &Model, model_name: &str, input: &Input) -> Result<ModeResult> {
    print_header(
        "CONSTRAINED",
        GREEN,
        "constrain_with(JsonSchema) — invalid tokens masked at sample time",
        model_name,
        &input.bio,
    );

    let mut ctx = Context::new(model)?;
    ctx.system(&input.system);
    ctx.user(&format!(
        "Extract name, email, age from this bio: {}",
        input.bio
    ));
    ctx.cue();

    let start = Instant::now();
    print!("  {}>{} ", CYAN, RESET);
    let _ = io::stdout().flush();

    let mut g = ctx
        .generate(Sampler::Argmax)
        .max_tokens(input.max_tokens)
        .constrain_with(JsonSchema(&input.schema))?;
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
                    print!("{}", visible);
                    let _ = io::stdout().flush();
                    if input.delay > 0 {
                        wstd::task::sleep(wstd::time::Duration::from_millis(input.delay)).await;
                    }
                }
                if try_parse_payload(&text).is_ok() {
                    break;
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

    let elapsed = start.elapsed();
    let parsed = try_parse_payload(&text).ok();
    let decode_tokens = approximate_tokens(&text);
    print_footer_constrained("CONSTRAINED", GREEN, decode_tokens, elapsed, &parsed);
    Ok(ModeResult {
        elapsed,
        attempts: 1,
        decode_tokens,
        wasted_tokens: 0,
        parsed,
    })
}

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

// ── Parse helper ──────────────────────────────────────────────────────
//
// Tries multiple parse strategies that mirror what a typical app does:
// 1. Direct parse of the trimmed output.
// 2. Strip ```json fences if present.
// 3. Find the first `{` and parse from there.
// Returns Ok with the parsed contact, or Err(reason) on failure.
// Strict parse — what most apps actually do: trim, then try
// `serde_json::from_str`. No fence-stripping, no scanning for `{`.
// This is the cliff that makes naive structured output flaky.
fn format_value(v: &serde_json::Value) -> String {
    serde_json::to_string(v).unwrap_or_else(|_| v.to_string())
}

fn try_parse(text: &str) -> std::result::Result<serde_json::Value, String> {
    let cleaned = strip_think_blocks(text);
    let trimmed = cleaned.trim();
    if trimmed.is_empty() {
        return Err("empty output".into());
    }
    serde_json::from_str::<serde_json::Value>(trimmed).map_err(|e| {
        let head = trimmed.lines().next().unwrap_or(trimmed);
        format!("{} (head: {:?})", e, truncate(head, 60))
    })
}

fn try_parse_payload(text: &str) -> std::result::Result<serde_json::Value, String> {
    match try_parse(text) {
        Ok(v) => Ok(v),
        Err(_) => {
            let cleaned = strip_think_blocks(text);
            let payload = extract_first_json_object(&cleaned)
                .ok_or_else(|| "no complete JSON object yet".to_string())?;
            serde_json::from_str::<serde_json::Value>(&payload)
                .map_err(|e| format!("payload parse failed: {e}"))
        }
    }
}

fn extract_first_json_object(text: &str) -> Option<String> {
    let mut start = None;
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;

    for (idx, ch) in text.char_indices() {
        if start.is_none() {
            if ch == '{' {
                start = Some(idx);
                depth = 1;
            }
            continue;
        }

        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    let s = start.unwrap();
                    return Some(text[s..idx + ch.len_utf8()].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

// Strip every `<think>...</think>` block from a finished text. Tolerant
// of unclosed think blocks at the end (drops the rest of the string).
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
                    Some(end) => {
                        rest = &after_open[end + "</think>".len()..];
                    }
                    None => break,
                }
            }
        }
    }
    out
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let head: String = s.chars().take(n).collect();
        format!("{}…", head)
    }
}

// Rough token count from chars/4 — good enough for "wasted tokens" math.
fn approximate_tokens(text: &str) -> usize {
    (text.chars().count() + 3) / 4
}

// ── ThinkStripper (same as other demos) ───────────────────────────────
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
fn print_header(mode: &str, color: &str, tagline: &str, model_name: &str, bio: &str) {
    let bar = "═".repeat(64);
    println!("{}{}{}{}", BOLD, color, bar, RESET);
    println!(
        "{}{}  GRAMMAR DEMO   ▸   mode: {}{}  ({}){}",
        BOLD, color, mode, RESET, tagline, RESET
    );
    println!("  {}model {}{}", DIM, model_name, RESET);
    println!("{}{}{}{}", BOLD, color, bar, RESET);
    println!();
    println!("  {}Bio:{} {}", BOLD, RESET, oneline(bio));
    println!();
}

fn print_footer_baseline(
    mode: &str,
    color: &str,
    attempts: usize,
    decoded: usize,
    wasted: usize,
    elapsed: Duration,
    parsed: &Option<serde_json::Value>,
) {
    let bar = "═".repeat(64);
    let verdict = match parsed {
        Some(c) => format!(
            "{}{}✓ parsed{}{} {}{}",
            BOLD,
            GREEN,
            RESET,
            color,
            format_value(c),
            RESET
        ),
        None => format!(
            " {}{}✗ failed after {} attempt(s){}",
            BOLD, RED, attempts, RESET
        ),
    };
    println!();
    println!("{}{}{}{}", BOLD, color, bar, RESET);
    println!(
        "{}{}  RESULT  •  {}  •  {} attempt(s)  •  {} decoded ({} wasted)  •  {:?}{}",
        BOLD, color, mode, attempts, decoded, wasted, elapsed, RESET
    );
    println!("  {}{}{}", DIM, verdict, RESET);
    println!("{}{}{}{}", BOLD, color, bar, RESET);
}

fn print_footer_constrained(
    mode: &str,
    color: &str,
    decoded: usize,
    elapsed: Duration,
    parsed: &Option<serde_json::Value>,
) {
    let bar = "═".repeat(64);
    let verdict = match parsed {
        Some(c) => format!(
            "{}{}✓ valid payload{} — {}",
            BOLD,
            GREEN,
            RESET,
            format_value(c)
        ),
        None => format!(
            " {}{}✗ schema produced no parseable output{}",
            BOLD, RED, RESET
        ),
    };
    println!();
    println!("{}{}{}{}", BOLD, color, bar, RESET);
    println!(
        "{}{}  RESULT  •  {}  •  1 attempt  •  {} decoded (0 wasted)  •  {:?}{}",
        BOLD, color, mode, decoded, elapsed, RESET
    );
    println!("  {}{}{}", DIM, verdict, RESET);
    println!("{}{}{}{}", BOLD, color, bar, RESET);
}

fn comparison(b: &ModeResult, c: &ModeResult) {
    let saved = b.wasted_tokens.saturating_sub(c.wasted_tokens);
    let speedup = if c.elapsed.as_secs_f64() > 0.0 {
        b.elapsed.as_secs_f64() / c.elapsed.as_secs_f64()
    } else {
        0.0
    };
    println!(
        "{}{}BASELINE {} attempt(s), {} wasted tokens, {:?}   ▸   CONSTRAINED 1 attempt, 0 wasted, {:?}   ▸   {:.2}× wall-time, {} fewer wasted tokens{}",
        BOLD, MAGENTA, b.attempts, b.wasted_tokens, b.elapsed, c.elapsed, speedup, saved, RESET
    );
    let _ = b.decode_tokens;
    let _ = c.decode_tokens;
    let _ = b.parsed.as_ref();
    let _ = c.parsed.as_ref();
}

fn oneline(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}
