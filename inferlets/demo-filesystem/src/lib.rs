//! Demo: `/scratch` as an inline persistent disk — no sidecar.
//!
//! Uniquely-Pie demo: the engine pre-opens a per-process `/scratch`
//! directory under WASI. The inferlet uses `std::fs` directly to
//! memoize an expensive model call. The first invocation is a cache
//! miss (full generate); the second is a cache hit (file read, no
//! model call). A naive client-driven flow either re-asks the model
//! every time or stands up Redis / a database to persist between
//! requests.
//!
//! Two strategies, same task asked twice:
//!
//! - **BASELINE** — no inline storage. The "client" always re-asks the
//!   model, paying the full decode each time. Simulates not having
//!   `/scratch` available.
//! - **CACHED** — use `/scratch/answers.json` as a key/value store.
//!   First call: miss, generate, write. Second call: hit, read, return.
//!
//! `mode = plain | smart | both` (default `both`).
//!
//! Note: this inferlet requires `allow_fs = true` in the engine config
//! (the default disables `/scratch`).

use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use inferlet::{Context, Result, chat, sample::Sampler};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_mode")]
    mode: String,

    #[serde(default = "default_question")]
    question: String,

    #[serde(default = "default_max_tokens")]
    max_tokens: usize,

    #[serde(default = "default_system")]
    system: String,

    #[serde(default = "default_scratch_file")]
    scratch_file: String,

    #[serde(default)]
    delay: u64,
}

fn default_system() -> String {
    "You are a brisk reference assistant. Reply in two short lines max. \
     Plain ASCII, no markdown."
        .into()
}

fn default_scratch_file() -> String {
    "/scratch/answers.json".into()
}

fn default_mode() -> String {
    "both".into()
}
fn default_question() -> String {
    "What is the Pythagorean theorem? Give the formula and one short \
     example. Plain ASCII, two short lines."
        .into()
}
fn default_max_tokens() -> usize {
    120
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

    // Reset the scratch file so reruns inside the same engine don't
    // confuse the cache-miss branch.
    let _ = fs::remove_file(&input.scratch_file);

    match mode.as_str() {
        "baseline" | "plain" => {
            run_baseline(&model_name, &input).await?;
        }
        "cached" | "smart" => {
            run_cached(&model_name, &input).await?;
        }
        "both" | "" => {
            let b = run_baseline(&model_name, &input).await?;
            println!();
            // Reset cache before the cached run.
            let _ = fs::remove_file(&input.scratch_file);
            let s = run_cached(&model_name, &input).await?;
            println!();
            comparison(&b, &s);
        }
        other => {
            return Err(format!(
                "unknown mode '{}': expected 'baseline', 'cached', or 'both'",
                other
            ));
        }
    }

    let _ = fs::remove_file(&input.scratch_file);
    Ok(String::new())
}

#[derive(Default, Clone)]
struct ModeResult {
    call1_elapsed: Duration,
    call2_elapsed: Duration,
    call1_was_hit: bool,
    call2_was_hit: bool,
}

// ── BASELINE: regenerate every call ──────────────────────────────────
async fn run_baseline(model_name: &str, input: &Input) -> Result<ModeResult> {
    print_header(
        "BASELINE",
        YELLOW,
        "no inline storage — every call re-asks the model",
        model_name,
    );
    println!(
        "  {}{}call 1{}  cache miss (no cache available)",
        BOLD, YELLOW, RESET
    );
    let t = Instant::now();
    generate_answer(input).await?;
    let call1 = t.elapsed();
    println!(
        "  {}generated, {} ms, no persistence{}",
        DIM,
        call1.as_millis(),
        RESET
    );
    println!();
    println!(
        "  {}{}call 2{}  same question, but no place to remember",
        BOLD, YELLOW, RESET
    );
    let t = Instant::now();
    generate_answer(input).await?;
    let call2 = t.elapsed();
    println!(
        "  {}regenerated again, {} ms{}",
        DIM,
        call2.as_millis(),
        RESET
    );
    print_footer("BASELINE", YELLOW, false, false, call1, call2);
    Ok(ModeResult {
        call1_elapsed: call1,
        call2_elapsed: call2,
        call1_was_hit: false,
        call2_was_hit: false,
    })
}

// ── CACHED: memoize via /scratch/answers.json ────────────────────────
async fn run_cached(model_name: &str, input: &Input) -> Result<ModeResult> {
    print_header(
        "CACHED",
        GREEN,
        "memoize via /scratch — second call is a file read",
        model_name,
    );

    println!(
        "  {}{}call 1{}  lookup /scratch/answers.json",
        BOLD, GREEN, RESET
    );
    let t = Instant::now();
    let (text1, hit1) = lookup_or_generate(input).await?;
    let call1 = t.elapsed();
    let _ = text1;
    println!(
        "  {}{} {} ms{}",
        DIM,
        if hit1 {
            "✓ hit"
        } else {
            "✗ miss → generate → write to /scratch"
        },
        call1.as_millis(),
        RESET
    );
    println!();
    println!(
        "  {}{}call 2{}  same question, lookup /scratch/answers.json",
        BOLD, GREEN, RESET
    );
    let t = Instant::now();
    let (text2, hit2) = lookup_or_generate(input).await?;
    let call2 = t.elapsed();
    let _ = text2;
    println!(
        "  {}{} {} ms{}",
        DIM,
        if hit2 {
            "✓ hit — answered from disk, no model call"
        } else {
            "✗ miss → generate"
        },
        call2.as_millis(),
        RESET
    );
    if !hit2 {
        return Err(
            "cache contract failed: second cached lookup missed; is allow_fs enabled?".into(),
        );
    }

    print_footer("CACHED", GREEN, hit1, hit2, call1, call2);
    Ok(ModeResult {
        call1_elapsed: call1,
        call2_elapsed: call2,
        call1_was_hit: hit1,
        call2_was_hit: hit2,
    })
}

async fn lookup_or_generate(input: &Input) -> Result<(String, bool)> {
    let key = cache_key(&input.question);
    let cache = read_cache(&input.scratch_file);
    if let Some(text) = cache.get(&key).cloned() {
        // Render from disk without involving the model.
        print!("  {}>{} ", CYAN, RESET);
        let _ = io::stdout().flush();
        for line in text.split_inclusive('\n') {
            print!("{}", line);
            let _ = io::stdout().flush();
            if input.delay > 0 {
                inferlet::sleep(std::time::Duration::from_millis(input.delay)).await;
            }
        }
        println!();
        return Ok((text, true));
    }
    // Cache miss — generate and write back.
    let text = generate_answer(input).await?;
    let mut cache = read_cache(&input.scratch_file);
    cache.insert(key, text.clone());
    write_cache(&input.scratch_file, &cache)?;
    Ok((text, false))
}

fn cache_key(q: &str) -> String {
    // Stable normalization — whitespace-collapsed lowercase. Plenty for
    // the demo; in production you'd hash this.
    q.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn read_cache(path: &str) -> HashMap<String, String> {
    if !Path::new(path).exists() {
        return HashMap::new();
    }
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(_) => return HashMap::new(),
    };
    serde_json::from_slice(&bytes).unwrap_or_default()
}

fn write_cache(path: &str, cache: &HashMap<String, String>) -> std::result::Result<(), String> {
    let bytes = serde_json::to_vec_pretty(cache).map_err(|e| e.to_string())?;
    fs::write(path, bytes).map_err(|e| e.to_string())
}

// ── Generate + stream the answer ──────────────────────────────────────
async fn generate_answer(input: &Input) -> Result<String> {
    let mut ctx = Context::new()?;
    ctx.system(&input.system);
    ctx.user(&format!("{} /no_think", input.question.trim()));
    ctx.cue();

    print!("  {}>{} ", CYAN, RESET);
    let _ = io::stdout().flush();

    let mut g = ctx
        .generate(Sampler::Argmax)
        .max_tokens(input.max_tokens)
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
                    if input.delay > 0 {
                        inferlet::sleep(std::time::Duration::from_millis(input.delay)).await;
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
    Ok(strip_think_blocks(&text).trim().to_string())
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
        "{}{}  FILESYSTEM DEMO   ▸   mode: {}{}  ({}){}",
        BOLD, color, mode, RESET, tagline, RESET
    );
    println!("  {}model {}{}", DIM, model_name, RESET);
    println!("{}{}{}{}", BOLD, color, bar, RESET);
    println!();
}

fn print_footer(mode: &str, color: &str, hit1: bool, hit2: bool, e1: Duration, e2: Duration) {
    let bar = "═".repeat(64);
    let total = e1 + e2;
    println!();
    println!("{}{}{}{}", BOLD, color, bar, RESET);
    println!(
        "{}{}  RESULT  •  {}  •  call 1: {} ms ({})  •  call 2: {} ms ({})  •  total {:?}{}",
        BOLD,
        color,
        mode,
        e1.as_millis(),
        if hit1 { "hit" } else { "miss" },
        e2.as_millis(),
        if hit2 { "hit" } else { "miss" },
        total,
        RESET
    );
    println!("{}{}{}{}", BOLD, color, bar, RESET);
}

fn comparison(b: &ModeResult, c: &ModeResult) {
    let total_b = b.call1_elapsed + b.call2_elapsed;
    let total_c = c.call1_elapsed + c.call2_elapsed;
    let speedup = if total_c.as_secs_f64() > 0.0 {
        total_b.as_secs_f64() / total_c.as_secs_f64()
    } else {
        0.0
    };
    let call2_speedup = if c.call2_elapsed.as_secs_f64() > 0.0 {
        b.call2_elapsed.as_secs_f64() / c.call2_elapsed.as_secs_f64()
    } else {
        0.0
    };
    println!(
        "{}{}BASELINE total {:?} (regen × 2)   ▸   CACHED total {:?} (miss + hit)   ▸   {:.2}× total, {:.0}× call-2{}",
        BOLD, MAGENTA, total_b, total_c, speedup, call2_speedup, RESET
    );
    let _ = (
        b.call1_was_hit,
        b.call2_was_hit,
        c.call1_was_hit,
        c.call2_was_hit,
    );
}
