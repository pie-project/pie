//! Demo: per-token watermark via `Probe::Distribution` + in-WASM bias.
//!
//! Uniquely-Pie demo: every token is sampled from a custom distribution
//! that biases the "green list" — a pseudorandom partition of the
//! vocabulary seeded by the previous token. A co-located detector
//! verifies the watermark by counting green-list hits and reporting a
//! z-score. A naive serving engine has no surface for this: there's
//! nowhere to read the distribution, transform it, sample from it, and
//! stay inside the inference loop.
//!
//! Two strategies, same prompt:
//!
//! - **BASELINE** — pull the distribution every step and sample without
//!   bias. Same control flow as WATERMARKED so wall-time is comparable.
//!   The detector should land near chance (z ≈ 0).
//! - **WATERMARKED** — apply +`delta` to the green-list probabilities
//!   before sampling. Output reads naturally, but the detector confirms
//!   a strong signal (z >> 0).
//!
//! `mode = plain | smart | both` (default `both`). The trick of this
//! demo is that BOTH outputs look fluent; the difference is only
//! visible to the detector.

use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{self, Write};
use std::time::{Duration, Instant};

use inferlet::{Context, Result, chat, model::Model, runtime, sample::Distribution, wstd};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_mode")]
    mode: String,

    #[serde(default = "default_prompt")]
    prompt: String,

    #[serde(default = "default_max_tokens")]
    max_tokens: usize,

    #[serde(default = "default_gamma")]
    gamma: f32,

    #[serde(default = "default_delta")]
    delta: f32,

    #[serde(default = "default_system")]
    system: String,

    #[serde(default)]
    delay: u64,
}

fn default_system() -> String {
    "You are a brisk technical writer. Reply in two or three short \
     sentences. Plain ASCII, no markdown."
        .into()
}

fn default_mode() -> String {
    "both".into()
}
fn default_prompt() -> String {
    "Explain in three sentences how LLMs decode tokens. Plain ASCII, \
     no markdown."
        .into()
}
fn default_max_tokens() -> usize {
    80
}
fn default_gamma() -> f32 {
    0.5
}
fn default_delta() -> f32 {
    8.0
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
            run_one(&model, &model_name, &input, false).await?;
        }
        "watermarked" | "smart" => {
            run_one(&model, &model_name, &input, true).await?;
        }
        "both" | "" => {
            let b = run_one(&model, &model_name, &input, false).await?;
            println!();
            let w = run_one(&model, &model_name, &input, true).await?;
            println!();
            comparison(&b, &w);
        }
        other => {
            return Err(format!(
                "unknown mode '{}': expected 'baseline', 'watermarked', or 'both'",
                other
            ));
        }
    }
    Ok(String::new())
}

#[derive(Default, Clone)]
struct ModeResult {
    elapsed: Duration,
    n_tokens: usize,
    green_hits: usize,
    z_score: f32,
}

// ── Run one mode (watermark on or off) ────────────────────────────────
async fn run_one(
    model: &Model,
    model_name: &str,
    input: &Input,
    watermark_on: bool,
) -> Result<ModeResult> {
    let (label, color, tagline) = if watermark_on {
        (
            "WATERMARKED",
            GREEN,
            "green-list bias applied per token; detector verifies the signal",
        )
    } else {
        (
            "BASELINE",
            YELLOW,
            "no bias — argmax sample from the distribution; detector should see chance",
        )
    };
    print_header(label, color, tagline, model_name, &input.prompt);

    // Build the prompt up front so the first forward pass prefills it.
    // Subsequent passes feed one chosen token at a time.
    let mut ctx = Context::new(model)?;
    let mut pending: Vec<u32> = Vec::new();
    pending.extend(chat::system(model, &input.system));
    pending.extend(chat::user(
        model,
        &format!("{} /no_think", input.prompt.trim()),
    ));
    pending.extend(chat::cue(model));

    let stop_tokens = chat::stop_tokens(model);
    let mut watermark = WatermarkState::new(input.gamma, input.delta);
    let mut detector = Detector::new(input.gamma);
    let mut chat_dec = chat::Decoder::new(model);
    let mut stripper = ThinkStripper::new();
    let mut generated: Vec<u32> = Vec::new();

    print!("  {}>{} ", CYAN, RESET);
    let _ = io::stdout().flush();

    let start = Instant::now();
    for _ in 0..input.max_tokens {
        let mut pass = ctx.forward();
        pass.input(&pending);
        let last_idx = (pending.len() - 1) as u32;
        // temperature=1.0 returns the model's natural softmax (not a
        // one-hot) so that adding +delta to green-list log-probs can
        // actually change which token wins on the biased argmax.
        // k=0 returns the full vocabulary.
        let h = pass.probe(
            last_idx,
            Distribution {
                temperature: 1.0,
                k: 0,
            },
        );
        let out = pass.execute().await?;
        let (ids, probs) = match out.distribution(h) {
            Some(d) => d,
            None => break,
        };

        let chosen = if watermark_on {
            watermark.sample_biased(ids, probs)
        } else {
            argmax_unbiased(ids, probs)
        };

        // Detector counts green-list hits using the same seeding rule.
        // It runs in either mode — for BASELINE we expect z ≈ 0.
        detector.feed(chosen, ids);

        if stop_tokens.contains(&chosen) {
            break;
        }
        generated.push(chosen);

        // Render visible deltas through the chat decoder + think stripper.
        if let chat::Event::Delta(s) = chat_dec.feed(&[chosen])? {
            let visible = stripper.process(&s);
            if !visible.is_empty() {
                let rendered = visible.replace('\n', "\n    ");
                print!("{}", rendered);
                let _ = io::stdout().flush();
                if input.delay > 0 {
                    wstd::task::sleep(wstd::time::Duration::from_millis(input.delay)).await;
                }
            }
        }

        pending = vec![chosen];
    }
    let elapsed = start.elapsed();
    println!();

    let (n, hits, z) = detector.report();
    print_footer(label, color, n, hits, z, elapsed);
    Ok(ModeResult {
        elapsed,
        n_tokens: n,
        green_hits: hits,
        z_score: z,
    })
}

// ── Watermark sampler: green-list bias + argmax of biased distribution
struct WatermarkState {
    gamma: f32,
    delta: f32,
    previous_token: Option<u32>,
}

impl WatermarkState {
    fn new(gamma: f32, delta: f32) -> Self {
        Self {
            gamma,
            delta,
            previous_token: None,
        }
    }

    fn sample_biased(&mut self, ids: &[u32], probs: &[f32]) -> u32 {
        if ids.is_empty() {
            return 0;
        }
        let seed = seed_from(self.previous_token);
        let green = green_list(ids, self.gamma, seed);
        let exp_delta = self.delta.exp();
        // Apply bias.
        let mut biased: Vec<f32> = probs
            .iter()
            .enumerate()
            .map(|(i, &p)| if green[i] { p * exp_delta } else { p })
            .collect();
        // Normalise — not strictly required for argmax, but keeps the
        // numbers interpretable if we ever switch to multinomial.
        let sum: f32 = biased.iter().sum();
        if sum > 0.0 {
            for p in &mut biased {
                *p /= sum;
            }
        }
        // Pick the argmax of the biased distribution. With delta=2 the
        // green-list winner usually beats the unbiased argmax.
        let mut best_i = 0usize;
        let mut best_p = f32::MIN;
        for (i, &p) in biased.iter().enumerate() {
            if p > best_p {
                best_p = p;
                best_i = i;
            }
        }
        let chosen = ids[best_i];
        self.previous_token = Some(chosen);
        chosen
    }
}

fn argmax_unbiased(ids: &[u32], probs: &[f32]) -> u32 {
    if ids.is_empty() {
        return 0;
    }
    let mut best_i = 0usize;
    let mut best_p = f32::MIN;
    for (i, &p) in probs.iter().enumerate() {
        if p > best_p {
            best_p = p;
            best_i = i;
        }
    }
    ids[best_i]
}

// ── Detector: scan generated tokens, count green-list hits, z-score ──
struct Detector {
    gamma: f32,
    n: usize,
    hits: usize,
    previous_token: Option<u32>,
}

impl Detector {
    fn new(gamma: f32) -> Self {
        Self {
            gamma,
            n: 0,
            hits: 0,
            previous_token: None,
        }
    }

    fn feed(&mut self, chosen: u32, ids: &[u32]) {
        if !ids.is_empty() {
            let seed = seed_from(self.previous_token);
            let green = green_list(ids, self.gamma, seed);
            // Find the index of `chosen` in `ids` to look up green status.
            if let Some(idx) = ids.iter().position(|&id| id == chosen) {
                if green[idx] {
                    self.hits += 1;
                }
                self.n += 1;
            }
        }
        self.previous_token = Some(chosen);
    }

    fn report(&self) -> (usize, usize, f32) {
        if self.n == 0 {
            return (0, 0, 0.0);
        }
        let n = self.n as f32;
        let g = self.gamma;
        // z = (hits - n*gamma) / sqrt(n * gamma * (1 - gamma))
        let denom = (n * g * (1.0 - g)).sqrt();
        let z = if denom > 0.0 {
            (self.hits as f32 - n * g) / denom
        } else {
            0.0
        };
        (self.n, self.hits, z)
    }
}

fn seed_from(prev: Option<u32>) -> u64 {
    match prev {
        Some(token) => {
            let mut hasher = DefaultHasher::new();
            token.hash(&mut hasher);
            hasher.finish()
        }
        None => 0,
    }
}

fn green_list(ids: &[u32], gamma: f32, seed: u64) -> Vec<bool> {
    let mut indices: Vec<usize> = (0..ids.len()).collect();
    deterministic_shuffle(&mut indices, seed);
    let green_size = (ids.len() as f32 * gamma).round() as usize;
    let mut is_green = vec![false; ids.len()];
    for &idx in &indices[..green_size.min(indices.len())] {
        is_green[idx] = true;
    }
    is_green
}

fn deterministic_shuffle(indices: &mut [usize], mut seed: u64) {
    for i in (1..indices.len()).rev() {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let j = (seed as usize) % (i + 1);
        indices.swap(i, j);
    }
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
fn print_header(mode: &str, color: &str, tagline: &str, model_name: &str, prompt: &str) {
    let bar = "═".repeat(64);
    println!("{}{}{}{}", BOLD, color, bar, RESET);
    println!(
        "{}{}  WATERMARK DEMO   ▸   mode: {}{}  ({}){}",
        BOLD, color, mode, RESET, tagline, RESET
    );
    println!("  {}model {}{}", DIM, model_name, RESET);
    println!("{}{}{}{}", BOLD, color, bar, RESET);
    println!();
    println!("  {}Q:{} {}", BOLD, RESET, oneline(prompt));
    println!();
}

fn print_footer(
    mode: &str,
    color: &str,
    n_tokens: usize,
    green_hits: usize,
    z: f32,
    elapsed: Duration,
) {
    let bar = "═".repeat(64);
    let pct = if n_tokens > 0 {
        100.0 * green_hits as f32 / n_tokens as f32
    } else {
        0.0
    };
    let verdict = if z >= 3.0 {
        format!("{}{}✓ watermark detected{}", BOLD, GREEN, RESET)
    } else if z >= 2.0 {
        format!("{}{}~ weak signal{}", BOLD, YELLOW, RESET)
    } else {
        format!("{}{}— no watermark detected{}", BOLD, RED, RESET)
    };
    println!();
    println!("{}{}{}{}", BOLD, color, bar, RESET);
    println!(
        "{}{}  RESULT  •  {}  •  {} tokens  •  green-list hits {}/{} ({:.0}%)  •  z = {:.2}  •  {:?}{}",
        BOLD, color, mode, n_tokens, green_hits, n_tokens, pct, z, elapsed, RESET
    );
    println!("  {}detector verdict: {}{}", DIM, verdict, RESET);
    println!("{}{}{}{}", BOLD, color, bar, RESET);
}

fn comparison(b: &ModeResult, w: &ModeResult) {
    let b_pct = if b.n_tokens > 0 {
        100.0 * b.green_hits as f32 / b.n_tokens as f32
    } else {
        0.0
    };
    let w_pct = if w.n_tokens > 0 {
        100.0 * w.green_hits as f32 / w.n_tokens as f32
    } else {
        0.0
    };
    println!(
        "{}{}BASELINE  green {:.0}%, z={:.2}  ({:?})   ▸   WATERMARKED  green {:.0}%, z={:.2}  ({:?}){}",
        BOLD, MAGENTA, b_pct, b.z_score, b.elapsed, w_pct, w.z_score, w.elapsed, RESET
    );
    println!(
        "{}both outputs read naturally; only the watermarked one carries a verifiable signal.{}",
        DIM, RESET
    );
}

fn oneline(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}
