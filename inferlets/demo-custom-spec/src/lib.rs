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

use inferlet::inference::ForwardPass;
use inferlet::working_set::KvWorkingSet;
use inferlet::{chat, geometry, model, sampler, Result};
use serde::Deserialize;

/// Raw n-token KV rollback (the keep-core equivalent of `Context::truncate`):
/// drop the trailing `n` materialized tokens and free the page slots that no
/// longer hold a live token. `n` is clamped to `*seq_len`. Byte-identical to
/// the jacobi/cacheback helper.
fn kv_truncate(kv: &KvWorkingSet, seq_len: &mut u32, n: u32) {
    let n = n.min(*seq_len);
    if n == 0 {
        return;
    }
    *seq_len -= n;
    let page = kv.page_size();
    let live_pages = seq_len.div_ceil(page);
    let have = kv.size();
    if have > live_pages {
        // Best-effort: a stale trailing page is harmless — the next forward
        // overwrites it. Matches the facade's non-fatal free.
        let drop: Vec<u32> = (live_pages..have).collect();
        let _ = kv.free(&drop);
    }
}

/// Read a Token output tensor as `u32` ids.
async fn read_tokens(pass: ForwardPass) -> Result<Vec<u32>> {
    let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("read: {e:?}"))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32)
        .collect())
}

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

    let stop_tokens = chat::stop_tokens();
    let vocab = model::output_vocab_size();
    let mut prompt = chat::system_user(&input.system, &input.task);
    prompt.extend(chat::cue());

    let kv = KvWorkingSet::new();
    let page = kv.page_size();
    let mut seq_len: u32 = 0;
    let greedy = sampler::sampler_program(sampler::SamplerSpec::Argmax, vocab)?;

    let start = Instant::now();
    print!("  {}>{} ", CYAN, RESET);
    let _ = io::stdout().flush();

    let mut decoder = chat::Decoder::new();
    let mut stripper = ThinkStripper::new();
    let mut tokens = 0usize;

    // Bootstrap: fire the full prompt, argmax at the last position → the
    // first response token (the fused prompt-prefill + first greedy sample).
    let mut next = {
        let n = prompt.len() as u32;
        let pass = ForwardPass::new();
        pass.fresh_generate();
        let geom = geometry::ensure_pages(&kv, geometry::kv_write_geometry(seq_len, n, page))?;
        geometry::attach_kv_write(&pass, &kv, &geom);
        let positions: Vec<u32> = (seq_len..seq_len + n).collect();
        pass.input_tokens(&prompt, &positions);
        pass.sampler(&greedy.program, greedy.bindings(seq_len + n - 1)?);
        pass.execute();
        seq_len += n;
        *read_tokens(pass)
            .await?
            .first()
            .ok_or("bootstrap produced no token")?
    };

    // Vanilla one-token-per-step greedy decode. Stop tokens break the loop
    // without being rendered (mirrors the facade's `stop` + empty-out drain).
    loop {
        if stop_tokens.contains(&next) || tokens >= input.max_tokens {
            break;
        }
        if let chat::Event::Delta(s) = decoder.feed(&[next])? {
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

        let pass = ForwardPass::new();
        let geom = geometry::ensure_pages(&kv, geometry::kv_write_geometry(seq_len, 1, page))?;
        geometry::attach_kv_write(&pass, &kv, &geom);
        pass.input_tokens(&[next], &[seq_len]);
        pass.sampler(&greedy.program, greedy.bindings(seq_len)?);
        pass.execute();
        seq_len += 1;
        next = match read_tokens(pass).await?.first() {
            Some(&t) => t,
            None => break,
        };
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

// ── SPECULATED: prompt-lookup drafter, manual draft/verify/accept loop ────
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

    let stop_tokens = chat::stop_tokens();
    let vocab = model::output_vocab_size();
    let mut prompt = chat::system_user(&input.system, &input.task);
    prompt.extend(chat::cue());

    let kv = KvWorkingSet::new();
    let page = kv.page_size();
    let mut seq_len: u32 = 0;
    let greedy = sampler::sampler_program(sampler::SamplerSpec::Argmax, vocab)?;

    // Bootstrap: one prompt fire with a greedy sampler at the last position →
    // the first response token. The canonical single-pass bootstrap (as in
    // jacobi/cacheback) keeps the speculated greedy trajectory identical to
    // BASELINE — the correctness guarantee of speculative decoding — and avoids
    // the facade-era fused-vs-split first-token drift the old split worked
    // around.
    let first_token = {
        let n = prompt.len() as u32;
        let pass = ForwardPass::new();
        pass.fresh_generate();
        let geom = geometry::ensure_pages(&kv, geometry::kv_write_geometry(seq_len, n, page))?;
        geometry::attach_kv_write(&pass, &kv, &geom);
        let positions: Vec<u32> = (seq_len..seq_len + n).collect();
        pass.input_tokens(&prompt, &positions);
        pass.sampler(&greedy.program, greedy.bindings(seq_len + n - 1)?);
        pass.execute();
        seq_len += n;
        *read_tokens(pass)
            .await?
            .first()
            .ok_or("bootstrap produced no token")?
    };

    let stats = Arc::new(Mutex::new(SpecStats::default()));
    // Pool seeded from the prompt (tokenized stand-alone) plus first_token so
    // iter 1's n-gram lookup searches suffix [.., first_token].
    let prompt_str = format!("{}\n{}", input.system, input.task);
    let mut pool = inferlet::model::encode(&prompt_str);
    pool.push(first_token);
    let mut drafter = PromptLookup::new(
        pool,
        input.ngram_n,
        input.draft_len,
        seq_len,
        Arc::clone(&stats),
    );

    let mut decoder = chat::Decoder::new();
    let mut stripper = ThinkStripper::new();

    let start = Instant::now();
    print!("  {}>{} ", CYAN, RESET);
    let _ = io::stdout().flush();

    let mut tokens = 0usize;
    let mut rounds = 0usize;

    // Render the bootstrap token (the first free pick).
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

    let mut anchor = first_token;
    let mut done = stop_tokens.contains(&first_token) || tokens >= input.max_tokens;

    // Manual draft/verify/accept loop (jacobi/cacheback pattern with the guest
    // PromptLookup drafter). Each round: draft off the n-gram pool, verify
    // [anchor] + drafts in ONE fire (argmax at every position), accept the
    // longest converged prefix, roll back the rejected suffix.
    while !done {
        let (draft_tokens, _positions) = drafter.draft();

        let mut verify_input = vec![anchor];
        verify_input.extend_from_slice(&draft_tokens);
        let n = verify_input.len() as u32;
        let verify = sampler::argmax_matrix_program(vocab, n)?;

        let pass = ForwardPass::new();
        let geom = geometry::ensure_pages(&kv, geometry::kv_write_geometry(seq_len, n, page))?;
        geometry::attach_kv_write(&pass, &kv, &geom);
        let positions: Vec<u32> = (seq_len..seq_len + n).collect();
        pass.input_tokens(&verify_input, &positions);
        pass.sampler(&verify.program, verify.bindings(&positions)?);
        pass.execute();
        let verified = read_tokens(pass).await?;
        if verified.is_empty() {
            break;
        }

        // Accepted = anchor's own prediction + each matching draft.
        let mut accepted_count = 1usize;
        for i in 1..verified.len().min(draft_tokens.len() + 1) {
            if i - 1 < draft_tokens.len() && verified[i - 1] == draft_tokens[i - 1] {
                accepted_count += 1;
            } else {
                break;
            }
        }
        let newly_accepted: Vec<u32> = verified[..accepted_count.min(verified.len())].to_vec();

        // Roll back the rejected suffix — keep `accepted_count` of the `n`
        // written tokens.
        kv_truncate(&kv, &mut seq_len, n - accepted_count as u32);

        // Grow the drafter pool + attribute draft hits (accept() reads
        // last_proposed for the accept-rate stats).
        drafter.accept(&newly_accepted);
        rounds += 1;

        // Render the burst: idx 0 = free pick (normal cyan), idx > 0 =
        // draft-accepted (green/bold). Stop tokens break without rendering.
        for (idx, &t) in newly_accepted.iter().enumerate() {
            if stop_tokens.contains(&t) {
                done = true;
                break;
            }
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
            if tokens >= input.max_tokens {
                done = true;
                break;
            }
        }

        anchor = *newly_accepted.last().unwrap_or(&anchor);
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

impl PromptLookup {
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
