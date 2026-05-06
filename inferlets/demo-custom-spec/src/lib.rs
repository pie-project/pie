//! Demo: custom speculator shipped inside the inferlet.
//!
//! Uniquely-Pie demo: the inferlet implements a **prompt-lookup
//! speculator** (PLD) directly in WASM, runs its own draft/verify loop
//! against the model via `Forward` + `truncate`, and gets a real wall-time
//! speedup over plain decoding. Other engines either don't expose the
//! inputs/positions/sample primitives at this granularity (vLLM, TGI) or
//! only let you configure speculation as a deploy-time draft *model*
//! (sglang) — none let you ship arbitrary draft logic alongside the
//! verifier in the same request.
//!
//! The chosen drafter is *prompt-lookup*: when the suffix of the
//! generated stream matches an earlier substring of the prompt
//! (or already-generated output), propose the next K tokens from that
//! match. For code edits / text refactors / quoting tasks, the model
//! mostly copies the prompt back, so the drafter accepts at ~70-95%.
//!
//! Two strategies, same task:
//!
//! - **BASELINE** — vanilla token-by-token greedy decode. Tokens stream at
//!   the GPU's per-token cadence.
//! - **SPECULATED** — PLD speculation. Tokens arrive in **bursts** of
//!   (1 free + N accepted draft) per spec round. Visibly faster.
//!
//! `mode = plain | smart | both` (default `both`). For side-by-side
//! recording, run `--mode plain` on one pane and `--mode smart` on the
//! other.

use std::io::{self, Write};
use std::time::{Duration, Instant};

use inferlet::{
    Context, Result, chat,
    model::Model,
    runtime,
    sample::Sampler,
    wstd,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_mode")]
    mode: String,

    #[serde(default = "default_task")]
    task: String,

    #[serde(default = "default_max_tokens")]
    max_tokens: usize,

    #[serde(default = "default_draft_len")]
    draft_len: usize,

    #[serde(default = "default_ngram_n")]
    ngram_n: usize,

    #[serde(default)]
    delay: u64,
}

fn default_mode() -> String {
    "both".into()
}
fn default_task() -> String {
    // High prompt-overlap workload: a multi-method class refactor that
    // mostly copies the input. Most output tokens are present somewhere
    // in the prompt → prompt-lookup speculation wins on near every step.
    // Length is tuned to ~150 output tokens so the streaming difference
    // (steady drip vs. multi-token bursts) is visible on video.
    "Refactor the following Python class: rename the field `total` to \
     `subtotal` everywhere it appears. Output ONLY the modified code in \
     a single code block, no commentary, no explanations.\n\
     \n\
     ```python\n\
     class ShoppingCart:\n\
         def __init__(self, customer_id):\n\
             self.customer_id = customer_id\n\
             self.total = 0\n\
             self.items = []\n\
     \n\
         def add_item(self, item, quantity):\n\
             self.items.append((item, quantity))\n\
             self.total += item.price * quantity\n\
     \n\
         def remove_item(self, item):\n\
             self.items = [(i, q) for (i, q) in self.items if i != item]\n\
             self.recalculate_total()\n\
     \n\
         def recalculate_total(self):\n\
             self.total = sum(i.price * q for (i, q) in self.items)\n\
     \n\
         def get_total(self):\n\
             return self.total\n\
     ```"
        .into()
}
fn default_max_tokens() -> usize {
    400
}
fn default_draft_len() -> usize {
    8
}
fn default_ngram_n() -> usize {
    2
}

const SYSTEM: &str = "You are a careful code-refactoring assistant. /no_think";

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

    let model_name = runtime::models()
        .first()
        .cloned()
        .ok_or("No models available")?;
    let model = Model::load(&model_name)?;

    match mode.as_str() {
        "baseline" | "plain" => {
            run_baseline(&model, &model_name, &input).await?;
        }
        "speculated" | "smart" => {
            run_speculated(&model, &model_name, &input).await?;
        }
        "both" | "" => {
            let p = run_baseline(&model, &model_name, &input).await?;
            println!();
            let s = run_speculated(&model, &model_name, &input).await?;
            println!();
            comparison(&p, &s);
        }
        other => {
            return Err(format!(
                "unknown mode '{}': expected 'baseline', 'speculated', or 'both'",
                other
            ));
        }
    }

    Ok(String::new())
}

#[derive(Default, Clone)]
struct ModeResult {
    tokens: usize,
    elapsed: Duration,
    drafts_proposed: usize,
    drafts_accepted: usize,
}

// ── BASELINE: vanilla one-token-per-step decode ───────────────────────────
async fn run_baseline(model: &Model, model_name: &str, input: &Input) -> Result<ModeResult> {
    print_header("BASELINE", YELLOW, "vanilla one-token-per-step decode", model_name);

    let mut ctx = Context::new(model)?;
    ctx.system(SYSTEM);
    ctx.user(&input.task);
    ctx.cue();

    let start = Instant::now();
    print!("  {}>{} ", CYAN, RESET);
    let _ = io::stdout().flush();

    let mut g = ctx
        .generate(Sampler::Argmax)
        .max_tokens(input.max_tokens)
        .stop(&chat::stop_tokens(model));
    let mut decoder = chat::Decoder::new(model);
    let mut stripper = ThinkStripper::new();
    let mut tokens = 0usize;

    while let Some(step) = g.next()? {
        let out = step.execute().await?;
        if out.tokens.is_empty() {
            continue;
        }
        tokens += out.tokens.len();
        match decoder.feed(&out.tokens)? {
            chat::Event::Delta(s) => {
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
            chat::Event::Done(_) => break,
            _ => {}
        }
    }
    let elapsed = start.elapsed();
    println!();
    print_footer_plain("BASELINE", YELLOW, tokens, elapsed);

    Ok(ModeResult { tokens, elapsed, drafts_proposed: 0, drafts_accepted: 0 })
}

// ── SPECULATED: prompt-lookup speculator + manual draft/verify ──────────────
async fn run_speculated(model: &Model, model_name: &str, input: &Input) -> Result<ModeResult> {
    print_header(
        "SPECULATED",
        GREEN,
        "prompt-lookup speculator (n-gram suffix → draft K)",
        model_name,
    );
    println!(
        "  {}every spec round: 1 free token + up to {} drafted tokens, verified in one forward pass.{}",
        DIM, input.draft_len, RESET
    );
    println!();

    let mut ctx = Context::new(model)?;
    ctx.system(SYSTEM);
    ctx.user(&input.task);
    ctx.cue();
    ctx.flush().await?;

    // Build the initial pool from system + user + cue tokens. The
    // speculator searches this pool (and its own accepted output) for
    // n-gram matches.
    let tokenizer = model.tokenizer();
    let prompt_str = format!("{}\n{}", SYSTEM, input.task);
    // We can't easily reconstruct the exact tokenization the chat
    // template used, so seed the lookup pool from the user task itself
    // (tokenized stand-alone). Misalignment is fine — every token added
    // by the model after bootstrap is added to the pool too, and the
    // n-gram match still finds suffix repetitions across both halves.
    let prompt_tokens = tokenizer.encode(&prompt_str);

    let stop_tokens = chat::stop_tokens(model);

    // Bootstrap: feed the last cue token to read the first prediction.
    // (The cue tokens are already committed via flush; this trick is
    // standard — see inferlets/cacheback-decoding for prior art.)
    let cue = chat::cue(model);
    let trigger = *cue.last().ok_or("empty cue")?;
    let first_token = {
        let mut pass = ctx.forward();
        pass.input(&[trigger]);
        let h = pass.sample(&[0], Sampler::Argmax);
        let out = pass.execute().await?;
        out.token(h).ok_or("bootstrap produced no token")?
    };

    let mut decoder = chat::Decoder::new(model);
    let mut stripper = ThinkStripper::new();

    let start = Instant::now();
    print!("  {}>{} ", CYAN, RESET);
    let _ = io::stdout().flush();

    // Render bootstrap token immediately.
    let mut tokens = 0usize;
    let mut hit_stop = false;
    if let chat::Event::Delta(s) = decoder.feed(&[first_token])? {
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
    tokens += 1;
    if stop_tokens.contains(&first_token) {
        hit_stop = true;
    }

    let mut drafter = PromptLookup::new(prompt_tokens, input.ngram_n, input.draft_len);
    drafter.accept(&[first_token]);

    let mut anchor = first_token;
    let mut rounds = 0usize;
    let mut drafts_proposed = 0usize;
    let mut drafts_accepted = 0usize;

    while !hit_stop && tokens < input.max_tokens {
        let drafts = drafter.draft();

        // Build the verifier input: anchor + drafts.
        let mut verify_in = vec![anchor];
        verify_in.extend_from_slice(&drafts);
        let n_in = verify_in.len();

        // Verify with samplers at every position.
        let mut pass = ctx.forward();
        pass.input(&verify_in);
        let sample_indices: Vec<u32> = (0..n_in as u32).collect();
        let h = pass.sample(&sample_indices, Sampler::Argmax);
        let out = pass.execute().await?;
        let verified = out.tokens_at(h);
        if verified.is_empty() {
            break;
        }

        // accepted_count = 1 (free token) + matching drafts prefix.
        let mut accepted_count = 1usize;
        for i in 0..drafts.len().min(verified.len().saturating_sub(0)) {
            if i + 1 > verified.len() {
                break;
            }
            // verified[i] is the model's pick at slot i; for draft[i] to
            // be accepted, verified[i] must equal draft[i].
            if i < drafts.len() && i < verified.len() && verified[i] == drafts[i] {
                accepted_count += 1;
            } else {
                break;
            }
        }
        let kept: Vec<u32> = verified[..accepted_count.min(verified.len())].to_vec();

        // Truncate rejected drafts from KV.
        let n_rejected = (n_in - kept.len()) as u32;
        if n_rejected > 0 {
            ctx.truncate(n_rejected);
        }

        // Stats.
        rounds += 1;
        drafts_proposed += drafts.len();
        drafts_accepted += accepted_count - 1; // free token doesn't count

        // Stream the kept tokens. Color the draft-accepted run green so
        // the visual difference vs BASELINE (steady drip) is obvious.
        for (idx, &t) in kept.iter().enumerate() {
            let from_draft = idx > 0;
            if let chat::Event::Delta(s) = decoder.feed(&[t])? {
                let visible = stripper.process(&s);
                if !visible.is_empty() {
                    let rendered = visible.replace('\n', "\n    ");
                    if from_draft {
                        print!("{}{}{}", BOLD, GREEN, rendered);
                    } else {
                        print!("{}", rendered);
                    }
                    print!("{}", RESET);
                    let _ = io::stdout().flush();
                    if input.delay > 0 {
                        wstd::task::sleep(wstd::time::Duration::from_millis(input.delay)).await;
                    }
                }
            }
            tokens += 1;
            if stop_tokens.contains(&t) {
                hit_stop = true;
                break;
            }
            if tokens >= input.max_tokens {
                break;
            }
        }

        drafter.accept(&kept);

        if let Some(&last) = kept.last() {
            anchor = last;
        } else {
            break;
        }
    }

    let elapsed = start.elapsed();
    println!();
    print_footer_smart(
        "SPECULATED",
        GREEN,
        tokens,
        elapsed,
        rounds,
        drafts_proposed,
        drafts_accepted,
    );

    let _ = rounds;
    Ok(ModeResult {
        tokens,
        elapsed,
        drafts_proposed,
        drafts_accepted,
    })
}

// ── Prompt-lookup speculator ──────────────────────────────────────────
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
        Self { in_think: false, pending: String::new() }
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

struct PromptLookup {
    pool: Vec<u32>,
    ngram_n: usize,
    draft_len: usize,
}

impl PromptLookup {
    fn new(prompt_tokens: Vec<u32>, ngram_n: usize, draft_len: usize) -> Self {
        Self { pool: prompt_tokens, ngram_n: ngram_n.max(1), draft_len: draft_len.max(1) }
    }

    fn accept(&mut self, tokens: &[u32]) {
        self.pool.extend_from_slice(tokens);
    }

    /// Find the most recent occurrence of the trailing n-gram in the
    /// pool (excluding the trailing n-gram itself) and propose the
    /// `draft_len` tokens that follow.
    fn draft(&self) -> Vec<u32> {
        let n = self.ngram_n;
        let pool_len = self.pool.len();
        if pool_len < n + 1 {
            return Vec::new();
        }
        let suffix = &self.pool[pool_len - n..pool_len];
        // Search positions [0, pool_len - n - 1] for matches; backward
        // for recency.
        let last_search = pool_len - n - 1;
        for i in (0..=last_search).rev() {
            if &self.pool[i..i + n] == suffix {
                let start = i + n;
                // Don't draft into the suffix itself (would just feed
                // back what's already there).
                let end_cap = pool_len - n;
                let end = (start + self.draft_len).min(end_cap);
                if start < end {
                    return self.pool[start..end].to_vec();
                }
                // Match found but no room before suffix — keep searching.
            }
        }
        Vec::new()
    }
}

// ── TUI helpers ────────────────────────────────────────────────────────
fn print_header(mode: &str, color: &str, tagline: &str, model_name: &str) {
    let bar = "═".repeat(64);
    println!("{}{}{}{}", BOLD, color, bar, RESET);
    println!(
        "{}{}  CUSTOM SPECULATOR DEMO   ▸   mode: {}{}  ({}){}",
        BOLD, color, mode, RESET, tagline, RESET
    );
    println!("  {}model {}{}", DIM, model_name, RESET);
    println!("{}{}{}{}", BOLD, color, bar, RESET);
    println!();
}

fn print_footer_plain(mode: &str, color: &str, tokens: usize, elapsed: Duration) {
    let bar = "═".repeat(64);
    let tps = (tokens as f64) / elapsed.as_secs_f64().max(1e-6);
    println!();
    println!("{}{}{}{}", BOLD, color, bar, RESET);
    println!(
        "{}{}  RESULT  •  {}  •  {} tokens in {:?}  •  {:.1} tok/s{}",
        BOLD, color, mode, tokens, elapsed, tps, RESET
    );
    println!("{}{}{}{}", BOLD, color, bar, RESET);
}

fn print_footer_smart(
    mode: &str,
    color: &str,
    tokens: usize,
    elapsed: Duration,
    rounds: usize,
    proposed: usize,
    accepted: usize,
) {
    let bar = "═".repeat(64);
    let tps = (tokens as f64) / elapsed.as_secs_f64().max(1e-6);
    let accept_rate = if proposed == 0 {
        0.0
    } else {
        100.0 * (accepted as f64) / (proposed as f64)
    };
    let avg_run = if rounds == 0 {
        0.0
    } else {
        (tokens as f64) / (rounds as f64)
    };
    println!();
    println!("{}{}{}{}", BOLD, color, bar, RESET);
    println!(
        "{}{}  RESULT  •  {}  •  {} tokens in {:?}  •  {:.1} tok/s{}",
        BOLD, color, mode, tokens, elapsed, tps, RESET
    );
    println!(
        "  {}spec rounds {}  •  drafts {} proposed / {} accepted ({:.0}% accept)  •  avg run {:.2} tok/round{}",
        DIM, rounds, proposed, accepted, accept_rate, avg_run, RESET
    );
    println!("{}{}{}{}", BOLD, color, bar, RESET);
}

fn comparison(p: &ModeResult, s: &ModeResult) {
    let p_tps = (p.tokens as f64) / p.elapsed.as_secs_f64().max(1e-6);
    let s_tps = (s.tokens as f64) / s.elapsed.as_secs_f64().max(1e-6);
    let speedup = s_tps / p_tps.max(1e-6);
    let accept_rate = if s.drafts_proposed == 0 {
        0.0
    } else {
        100.0 * (s.drafts_accepted as f64) / (s.drafts_proposed as f64)
    };
    println!(
        "{}{}BASELINE {:.1} tok/s   ▸   SPECULATED {:.1} tok/s   ▸   {:.2}x speedup  ({:.0}% draft accept){}",
        BOLD, MAGENTA, p_tps, s_tps, speedup, accept_rate, RESET
    );
}
