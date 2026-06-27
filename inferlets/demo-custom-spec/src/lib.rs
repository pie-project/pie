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
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use inferlet::{Context, Result, Speculator, chat, sample::Sampler};
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

    #[serde(default = "default_system")]
    system: String,

    #[serde(default)]
    delay: u64,
}

fn default_system() -> String {
    "You are a careful code-refactoring assistant. /no_think".into()
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

    match mode.as_str() {
        "baseline" | "plain" => {
            run_baseline(&model_name, &input).await?;
        }
        "speculated" | "smart" => {
            run_speculated(&model_name, &input).await?;
        }
        "both" | "" => {
            let p = run_baseline(&model_name, &input).await?;
            println!();
            let s = run_speculated(&model_name, &input).await?;
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
async fn run_baseline(model_name: &str, input: &Input) -> Result<ModeResult> {
    print_header(
        "BASELINE",
        YELLOW,
        "vanilla one-token-per-step decode",
        model_name,
    );

    let mut ctx = Context::new()?;
    ctx.system(&input.system);
    ctx.user(&input.task);
    ctx.cue();

    let start = Instant::now();
    print!("  {}>{} ", CYAN, RESET);
    let _ = io::stdout().flush();

    let mut g = ctx
        .generate(Sampler::Argmax)
        .max_tokens(input.max_tokens)
        .stop(&chat::stop_tokens());
    let mut decoder = chat::Decoder::new();
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
                        inferlet::sleep(std::time::Duration::from_millis(input.delay)).await;
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

    Ok(ModeResult {
        tokens,
        elapsed,
        drafts_proposed: 0,
        drafts_accepted: 0,
    })
}

// ── SPECULATED: prompt-lookup speculator via Generator::speculator ────────
async fn run_speculated(model_name: &str, input: &Input) -> Result<ModeResult> {
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

    let mut ctx = Context::new()?;
    ctx.system(&input.system);
    ctx.user(&input.task);
    ctx.cue();
    // Prefill the prompt as a standalone forward, then bootstrap the
    // first model token, then enter the speculator loop. Letting the
    // Generator fuse prompt prefill with the first verify pass works on
    // some backends but not others: greedy-argmax is sensitive to small
    // numerical differences between "prefill alone" and "prefill +
    // multi-position samplers" kernel shapes, and on cuda 0.1.0 the
    // fused shape produces logits that drift off the greedy trajectory.
    // The split-pass pattern matches inferlets/cacheback-decoding.
    ctx.flush().await?;

    let prompt_str = format!("{}\n{}", input.system, input.task);
    // Pool seeded from the user task tokenized stand-alone. Chat
    // template alignment isn't critical: accepted tokens grow the pool
    // as generation progresses, so n-gram suffix matches still find
    // prompt-repetition hits across both halves.
    let prompt_tokens = inferlet::model::encode(&prompt_str);
    let stop_tokens = chat::stop_tokens();

    // Bootstrap: re-feed the last cue token at the next free slot and
    // sample the model's first response token. The cue is already in
    // KV from flush(); duplicating its last token kicks the decoder
    // out of prefill into a normal sampling step.
    let cue = chat::cue();
    let trigger = *cue.last().ok_or("empty cue")?;
    let first_token = {
        let mut pass = ctx.forward();
        pass.input(&[trigger]);
        let h = pass.sample(&[0], Sampler::Argmax);
        let out = pass.execute().await?;
        out.token(h).ok_or("bootstrap produced no token")?
    };
    // Stage first_token as iter 1's pending. The Generator drains
    // buffer into pending on each `next()`, so iter 1 forwards it at
    // position seq_len; drafts then start at seq_len + 1.
    ctx.append(&[first_token]);

    let stats = Arc::new(Mutex::new(SpecStats::default()));
    let cursor = ctx.seq_len() + ctx.buffer().len() as u32;
    // Pre-seed the pool with first_token so iter 1's n-gram lookup
    // searches suffix [last_prompt_tok, first_token]. Without this seed
    // iter 1 would draft against a model-token-free pool and the very
    // first lookup would miss.
    let mut pool = prompt_tokens;
    pool.push(first_token);
    let drafter = PromptLookup::new(
        pool,
        input.ngram_n,
        input.draft_len,
        cursor,
        Arc::clone(&stats),
    );

    let mut decoder = chat::Decoder::new();
    let mut stripper = ThinkStripper::new();

    let start = Instant::now();
    print!("  {}>{} ", CYAN, RESET);
    let _ = io::stdout().flush();

    let mut tokens = 0usize;
    let mut rounds = 0usize;

    // Render the bootstrap token (sampled outside the Generator loop).
    if let chat::Event::Delta(s) = decoder.feed(&[first_token])? {
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
    tokens += 1;

    // If the very first sample is a stop token, skip the loop entirely.
    let bootstrap_done = stop_tokens.contains(&first_token) || tokens >= input.max_tokens;

    let mut g = ctx
        .generate(Sampler::Argmax)
        .max_tokens(input.max_tokens.saturating_sub(tokens))
        .stop(&stop_tokens)
        .speculator(drafter);

    if !bootstrap_done {
        while let Some(step) = g.next()? {
            let out = step.execute().await?;
            if out.tokens.is_empty() {
                continue;
            }
            rounds += 1;
            // out.tokens shape: [free_pick, ...accepted_drafts...]. Render
            // the free pick in normal cyan stream color; color draft-
            // accepted tokens green/bold so the burst is visible against
            // BASELINE's steady drip.
            for (idx, &t) in out.tokens.iter().enumerate() {
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
                            inferlet::sleep(std::time::Duration::from_millis(input.delay)).await;
                        }
                    }
                }
                tokens += 1;
            }
        }
    }

    let elapsed = start.elapsed();
    println!();
    let (drafts_proposed, drafts_accepted) = {
        let s = stats.lock().unwrap();
        (s.proposed, s.accepted)
    };
    print_footer_smart(
        "SPECULATED",
        GREEN,
        tokens,
        elapsed,
        rounds,
        drafts_proposed,
        drafts_accepted,
    );

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

#[derive(Default)]
struct SpecStats {
    proposed: usize,
    accepted: usize,
}

struct PromptLookup {
    pool: Vec<u32>,
    ngram_n: usize,
    draft_len: usize,
    /// Position the next draft would occupy in the model's KV.
    cursor: u32,
    /// Drafts proposed in the most recent `draft()` call. Read by
    /// `accept()` to attribute hits.
    last_proposed: usize,
    stats: Arc<Mutex<SpecStats>>,
}

impl PromptLookup {
    fn new(
        prompt_tokens: Vec<u32>,
        ngram_n: usize,
        draft_len: usize,
        cursor: u32,
        stats: Arc<Mutex<SpecStats>>,
    ) -> Self {
        Self {
            pool: prompt_tokens,
            ngram_n: ngram_n.max(1),
            draft_len: draft_len.max(1),
            cursor,
            last_proposed: 0,
            stats,
        }
    }

    /// Find the most recent occurrence of the trailing n-gram in the
    /// pool (excluding the trailing n-gram itself) and propose the
    /// `draft_len` tokens that follow.
    fn lookup(&self) -> Vec<u32> {
        let n = self.ngram_n;
        let pool_len = self.pool.len();
        if pool_len < n + 1 {
            return Vec::new();
        }
        let suffix = &self.pool[pool_len - n..pool_len];
        let last_search = pool_len - n - 1;
        for i in (0..=last_search).rev() {
            if &self.pool[i..i + n] == suffix {
                let start = i + n;
                // Don't draft into the suffix itself.
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

impl Speculator for PromptLookup {
    fn draft(&mut self) -> (Vec<u32>, Vec<u32>) {
        let drafts = self.lookup();
        self.last_proposed = drafts.len();
        let positions: Vec<u32> = (self.cursor..self.cursor + drafts.len() as u32).collect();
        (drafts, positions)
    }

    fn accept(&mut self, accepted: &[u32]) {
        // accepted[0] is the anchor's free pick (not from drafts).
        // The remaining tokens, up to last_proposed, are draft hits.
        let n_drafts_hit = accepted.len().saturating_sub(1).min(self.last_proposed);
        {
            let mut s = self.stats.lock().unwrap();
            s.proposed += self.last_proposed;
            s.accepted += n_drafts_hit;
        }
        self.last_proposed = 0;
        self.pool.extend_from_slice(accepted);
        self.cursor += accepted.len() as u32;
    }

    // rollback: pool only grows via accept(), so there's nothing to undo
    // on draft rejection. Default no-op fits.
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
