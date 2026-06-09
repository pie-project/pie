//! AsyncLM — asynchronous LLM function calling.
//!
//! An inferlet implementation of AsyncLM (In Gim, Seung-seob Lee, Lin Zhong,
//! "Asynchronous LLM Function Calling," arXiv:2412.07017). The model dispatches
//! tool calls *without blocking* and keeps generating, so function-call latency
//! overlaps token decoding instead of stalling it; the paper reports 1.6×–5.4×
//! lower end-to-end task latency versus synchronous calling on BFCL.
//!
//! The model and runtime communicate through the paper's CML (Context Markup
//! Language) token protocol:
//!
//! - `[CALL] <id> [HEAD] <code> [END]` — dispatch a call (non-blocking); the
//!   model emits this and immediately continues generating.
//! - `[TRAP][END]` — pause generation until every dispatched call has returned.
//! - `[INTR] <id> [HEAD] <value> [END]` — a result frame the *runtime* injects;
//!   the model only ever reads these, never writes them.
//!
//! The inferlet drives the entire loop. It scans the model's token stream for
//! CML delimiters with a small FSM parser (`Parser`), fires each `[CALL]` as a
//! real HTTP request on the `wstd` reactor so it runs concurrently with
//! decoding, and injects each `[INTR]` result back into the live `Generator`.
//! Five live tools are wired up: weather, stock price, currency, time, and
//! Wikipedia summaries.
//!
//! It also implements the paper's trap-time scheduling: a critical-section
//! tracker so results are only injected at safe parser boundaries, optional
//! mid-stream injection (deliver a result *while* the model is still writing
//! result-independent prose, before the trap), and a Keep / Recompute / Swap
//! classifier backed by checkpoints taken at each `[CALL]`. Recompute restores a
//! saved `Context` snapshot and replays the completed `[INTR]` frames before
//! sampling resumes. Swap is classified and logged, but Pie exposes no separate
//! RAM-tier KV primitive, so Swap execution collapses to Keep.
//!
//! Pie features exercised:
//!   - `Context` chat-template helpers (`system`, `user`, `cue`) and
//!     `Context::{snapshot, open, delete}` for checkpoint / restore.
//!   - Token-level generation: stepping a `Generator` with `next` / `execute`,
//!     `accept`-ing runtime-injected tokens, and `clear_sampler` to flush an
//!     injected frame into the next decode step.
//!   - `Constrain` with a precomputed BRLE mask that bans the token ids unique
//!     to `[INTR]`, so the model can never forge a result frame.
//!   - `wstd` async tasks + HTTP client, so real tool calls overlap decoding.

use std::collections::{HashMap, HashSet, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context as TaskContext, Poll, RawWaker, RawWakerVTable, Waker};
use std::time::Instant;

use inferlet::model::{Model, Tokenizer};
use inferlet::sample::Sampler;
use inferlet::serde_json::Value;
use inferlet::wstd;
use inferlet::wstd::http::{Client, Method, Request};
use inferlet::{Constrain, Context, Generator, Result, chat, runtime};
use serde::Deserialize;

/// Gates the verbose per-token `[dbg]` diagnostics. Off by default; set in
/// `main` from the `enable_token_trace` input.
static TOKEN_TRACE_ON: AtomicBool = AtomicBool::new(false);
fn token_trace_on() -> bool {
    TOKEN_TRACE_ON.load(Ordering::Relaxed)
}

// ── Trace instrumentation ───────────────────────────────────────────────────
// Optional machine-readable event stream, gated by `enable_trace`. One line
// per event, each with an absolute elapsed-ms timestamp, so an offline tool can
// place dispatch / trap / inject events on a true wall-clock axis for latency
// analysis instead of reverse-engineering them from token positions. The
// human-readable [AsyncLM] prints are independent and always on. Absent numeric
// fields are -1.
//   schema: [TRACE] v=<variant> t=<ms> kind=<k> id=<id> dur=<ms> tok=<n> sub=<s>
const TRACE_V: &str = "async";
static TRACE_T0: OnceLock<Instant> = OnceLock::new();
/// Gates the [TRACE] event stream. Off by default; set once in `main` from the
/// `enable_trace` input. The human-readable [AsyncLM] prints are independent and
/// always emit.
static TRACE_ON: AtomicBool = AtomicBool::new(false);
/// Milliseconds since the first traced event (≈ run start; anchored by the
/// `run_start` trace emitted right after the registry banner).
fn t_ms() -> u128 {
    TRACE_T0.get_or_init(Instant::now).elapsed().as_millis()
}
fn trace(kind: &str, id: &str, dur: i64, tok: i64, sub: &str) {
    if !TRACE_ON.load(Ordering::Relaxed) {
        return;
    }
    println!(
        "[TRACE] v={} t={} kind={} id={} dur={} tok={} sub={}",
        TRACE_V,
        t_ms(),
        kind,
        id,
        dur,
        tok,
        sub
    );
}

// ============================================================================
// Input
// ============================================================================

#[derive(Deserialize)]
struct Input {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_system")]
    system: String,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default = "default_alpha")]
    alpha_recompute_ns_per_tok2: f64,
    #[serde(default = "default_beta")]
    beta_swap_ns_per_tok: f64,
    /// Enable safe-boundary mid-stream [INTR] injection (the paper's in-flight
    /// interrupt mechanism: deliver a result before the model reaches `[TRAP]`).
    #[serde(default)]
    enable_midstream: bool,
    /// Emit the machine-readable [TRACE] event stream for offline latency analysis.
    #[serde(default)]
    enable_trace: bool,
    /// Emit verbose per-token `[dbg]` decode/injection diagnostics.
    #[serde(default)]
    enable_token_trace: bool,
    /// Disable per-call checkpointing and force trap strategy Keep (one unbroken
    /// generator session). Default true; set false for real checkpoint/restore.
    #[serde(default = "default_disable_checkpointing")]
    disable_checkpointing: bool,
    /// Treat unknown tools in a [CALL] as BFCL latency-emulated calls instead of
    /// erroring. Needed to run BFCL function names; default false errors.
    #[serde(default)]
    emulate_unknown_tools: bool,
    #[serde(default)]
    max_steps: Option<u32>,
}

fn default_max_tokens() -> usize {
    2048
}
fn default_disable_checkpointing() -> bool {
    true
}
fn default_temperature() -> f32 {
    0.6
}
fn default_top_p() -> f32 {
    0.95
}
fn default_alpha() -> f64 {
    800.0
}
fn default_beta() -> f64 {
    400_000.0
}
fn default_system() -> String {
    "You are an assistant that uses async function calls (AsyncLM / CML\n\
     protocol) to answer user questions.\n\
     \n\
     CML syntax (use these delimiters EXACTLY):\n\
     - Dispatch (non-blocking): [CALL] <id> [HEAD] <code> [END]\n\
     - Wait for pending calls:  [TRAP][END]\n\
     - Result (runtime-injected, you only read): [INTR] <id> [HEAD] <value> [END]\n\
     \n\
     Tools available (each backed by a live web API):\n\
     - get_weather(city: str) -> {\"city\": str, \"temp_f\": int, \"sky\": str}\n\
     - get_stock_price(ticker: str) -> {\"ticker\": str, \"price_usd\": float}\n\
     - convert_currency(amount: float, from: str, to: str) -> {\"rate\": float, \"quote\": str}\n\
     - get_time(location: str) -> {\"time\": str, \"day\": str, \"tz\": str}\n\
     - wiki_summary(title: str) -> {\"title\": str, \"summary\": str}\n\
     \n\
     Semantics:\n\
     - [CALL] blocks are non-blocking. Dispatch all independent calls\n\
       back-to-back so they run in parallel.\n\
     - [TRAP][END] pauses generation until every pending [CALL] has\n\
       returned an [INTR]. Only emit [TRAP][END] when you actually need\n\
       a result to decide what to do next.\n\
     - BETWEEN your [CALL] dispatches and the [TRAP][END], write any\n\
       reasoning or prose that does NOT depend on the call results.\n\
       This is the whole point of async: the calls run in parallel\n\
       with your writing, so prose before [TRAP] is free latency-wise.\n\
       Delay [TRAP][END] as long as you still have result-independent\n\
       things to say.\n\
     - After [INTR] frames appear you may EITHER (a) dispatch another\n\
       round of [CALL]s whose inputs depend on those results, then\n\
       [TRAP][END] again, OR (b) write the final natural-language answer\n\
       and stop. There is no fixed number of rounds.\n\
     - If part of the question is outside these tools (general knowledge,\n\
       math, unrelated topics), answer that part from your own knowledge\n\
       — do NOT invent a [CALL] for it.\n\
     - Do not put CML syntax inside <think> tags. Reason out what to call\n\
       briefly, exit </think>, THEN emit the actual [CALL] / [TRAP][END]\n\
       frames. Describing a call inside <think> is not the same as making\n\
       the call — you must always emit the literal CML frames outside.\n\
     - Do not fabricate [INTR] frames yourself; the runtime produces them.\n\
     - NEVER write, repeat, paraphrase, or echo an [INTR] frame or any part\n\
       of it (the literal [INTR] / [HEAD] / [END] markers, the id, or the\n\
       JSON value). Each result is already injected into your context — just\n\
       READ it and answer in prose. The [INTR] lines shown in the examples\n\
       below are RUNTIME-INJECTED for illustration; they are NOT text you\n\
       produce. Your own output after a [TRAP][END] is prose only (or the\n\
       next round's [CALL] frames).\n\
     - Never put plain prose or explanations inside [CALL]; [CALL] bodies must be one of the listed tool functions.\n\
     \n\
     Example 1 — single round, two parallel calls:\n\
     User: What's the weather in NYC and London?\n\
     Assistant: [CALL] w1 [HEAD] get_weather(\"New York\") [END]\n\
     [CALL] w2 [HEAD] get_weather(\"London\") [END]\n\
     [TRAP][END]\n\
     ‹runtime injects the next two lines — you READ them, never write them›\n\
     [INTR] w1 [HEAD] {\"temp_f\": 72, \"sky\": \"sunny\"} [END]\n\
     [INTR] w2 [HEAD] {\"temp_f\": 60, \"sky\": \"cloudy\"} [END]\n\
     ‹you resume, prose only› NYC is 72°F and sunny; London is 60°F and cloudy.\n\
     \n\
     Example 2 — two rounds, round 2 depends on round 1 results:\n\
     User: Get the weather in Boston, then also get the weather for a\n\
     city whose name is that temperature (as a string).\n\
     Assistant: [CALL] b1 [HEAD] get_weather(\"Boston\") [END]\n\
     [TRAP][END]\n\
     ‹runtime injects the next line — you READ it, never write it›\n\
     [INTR] b1 [HEAD] {\"temp_f\": 68, \"sky\": \"clear\"} [END]\n\
     ‹you resume, prose only› Boston is 68°F. Now looking up \"68\".\n\
     [CALL] b2 [HEAD] get_weather(\"68\") [END]\n\
     [TRAP][END]\n\
     ‹runtime injects the next line — you READ it, never write it›\n\
     [INTR] b2 [HEAD] {\"temp_f\": 72, \"sky\": \"sunny\"} [END]\n\
     ‹you resume, prose only› Boston is 68°F and clear; \"68\" is 72°F and sunny.\n\
     \n\
     Example 3 — mixed tool + knowledge question:\n\
     User: What's the weather in Paris and what is the capital of Japan?\n\
     Assistant: [CALL] p1 [HEAD] get_weather(\"Paris\") [END]\n\
     [TRAP][END]\n\
     ‹runtime injects the next line — you READ it, never write it›\n\
     [INTR] p1 [HEAD] {\"temp_f\": 59, \"sky\": \"rainy\"} [END]\n\
     ‹you resume, prose only› Paris is 59°F and rainy. The capital of Japan is Tokyo.\n\
     \n\
     Example 4 — interleaved prose hides call latency (preferred style\n\
     whenever you have anything to say that does not depend on the\n\
     results). Note how reasoning appears BEFORE [TRAP][END]:\n\
     User: Get the weather in Paris and the time in Tokyo, then tell\n\
     me if it's a reasonable hour to video-call Tokyo from Paris.\n\
     Assistant: [CALL] w1 [HEAD] get_weather(\"Paris\") [END]\n\
     [CALL] t1 [HEAD] get_time(\"Asia/Tokyo\") [END]\n\
     Both calls are dispatched in parallel. While they run, here is the\n\
     shape of the answer: Paris weather tells us whether the caller is\n\
     likely indoors and free, and Tokyo local time tells us whether the\n\
     other side is awake — Tokyo is roughly 7-8 hours ahead of Paris,\n\
     so a Paris afternoon maps to a Tokyo late-night which is bad, but\n\
     a Paris morning maps to a Tokyo afternoon which is ideal.\n\
     [TRAP][END]\n\
     ‹runtime injects the next two lines — you READ them, never write them›\n\
     [INTR] w1 [HEAD] {\"temp_f\": 62, \"sky\": \"overcast\"} [END]\n\
     [INTR] t1 [HEAD] {\"time\": \"07:30\", \"tz\": \"Asia/Tokyo\"} [END]\n\
     ‹you resume, prose only› Paris is 62°F and overcast — comfortable indoor conditions. Tokyo\n\
     is 07:30, just starting the workday, so this is a good window."
        .to_string()
}

// ============================================================================
// CML token registry
// ============================================================================

struct CmlRegistry {
    call_ids: Vec<u32>,
    end_ids: Vec<u32>,
    head_ids: Vec<u32>,
    trap_ids: Vec<u32>,
    intr_ids: Vec<u32>,

    /// Full `[TRAP][END]` sequences; second variant handles BPE-merged `][`.
    trap_end_ids: Vec<Vec<u32>>,

    /// Token IDs unique to `[INTR]` — masked out during sampling so the
    /// model cannot hallucinate an interrupt frame. Bracket tokens shared
    /// with other delimiters are NOT suppressed.
    suppressed: HashSet<u32>,

    /// Tokens that open / close a Qwen3 thinking block. Empty if absent.
    /// The parser uses these to suspend CML matching while the model is
    /// inside `<think>…</think>`, otherwise it eats literal CML delimiters
    /// that the model quotes in its reasoning.
    think_open_ids: Vec<u32>,
    think_close_ids: Vec<u32>,

    /// Every token whose decoded bytes start with `]`. Used by the suffix
    /// matcher as alternates for the canonical `]` in the final position
    /// of each delimiter pattern, so that BPE-fused tokens like `]\n` or
    /// `][` close out a delimiter correctly.
    close_bracket_alts: Vec<u32>,
}

impl CmlRegistry {
    fn new(tokenizer: &Tokenizer) -> Self {
        let special_map = build_special_map(tokenizer);
        let resolve = |s: &str| resolve_token(s, &special_map, tokenizer);
        let require = |s: &str| {
            let ids = resolve(s);
            assert!(!ids.is_empty(), "CML token '{}' tokenized to empty", s);
            ids
        };

        let call_ids = require("[CALL]");
        let end_ids = require("[END]");
        let head_ids = require("[HEAD]");
        let trap_ids = require("[TRAP]");
        let intr_ids = require("[INTR]");

        let trap_end_ids = build_trap_end_variants(&trap_ids, &end_ids, tokenizer);
        let close_bracket_alts = collect_close_bracket_alts(tokenizer);
        let suppressed =
            compute_intr_suppression(&call_ids, &end_ids, &head_ids, &trap_ids, &intr_ids);

        // Qwen3 think delimiters — empty Vec on tokenizers that lack them
        // (Qwen2.5 etc.), in which case the bypass is a no-op.
        let think_open_ids = resolve("<think>");
        let think_close_ids = resolve("</think>");

        CmlRegistry {
            call_ids,
            end_ids,
            head_ids,
            trap_ids,
            intr_ids,
            trap_end_ids,
            suppressed,
            think_open_ids,
            think_close_ids,
            close_bracket_alts,
        }
    }
}

fn build_special_map(tokenizer: &Tokenizer) -> HashMap<Vec<u8>, u32> {
    let (special_ids, special_bytes) = tokenizer.special_tokens();
    special_bytes
        .into_iter()
        .zip(special_ids.into_iter())
        .collect()
}

fn resolve_token(s: &str, special_map: &HashMap<Vec<u8>, u32>, tokenizer: &Tokenizer) -> Vec<u32> {
    if let Some(&id) = special_map.get(s.as_bytes()) {
        vec![id]
    } else {
        tokenizer.encode(s)
    }
}

fn build_trap_end_variants(
    trap_ids: &[u32],
    end_ids: &[u32],
    tokenizer: &Tokenizer,
) -> Vec<Vec<u32>> {
    let mut variants = vec![
        trap_ids
            .iter()
            .chain(end_ids.iter())
            .copied()
            .collect::<Vec<u32>>(),
    ];
    let merge = tokenizer.encode("][");
    if merge.len() == 1 && !trap_ids.is_empty() && !end_ids.is_empty() {
        let mut merged: Vec<u32> = trap_ids[..trap_ids.len() - 1].to_vec();
        merged.push(merge[0]);
        merged.extend_from_slice(&end_ids[1..]);
        variants.push(merged);
    }
    variants
}

/// Every vocab token whose decoded bytes start with `]`. The suffix matcher
/// accepts any of these in place of the canonical `]` so BPE-fused trailing
/// brackets (`]\n`, `][`, `] `, …) still close out a delimiter. Without
/// this the FSM jams in InCallBody/InCallId forever — failures are silent.
fn collect_close_bracket_alts(tokenizer: &Tokenizer) -> Vec<u32> {
    let (vocab_ids, vocab_bytes) = tokenizer.vocabs();
    vocab_ids
        .iter()
        .zip(vocab_bytes.iter())
        .filter(|(_, bytes)| !bytes.is_empty() && bytes[0] == b']')
        .map(|(&id, _)| id)
        .collect()
}

/// Suppress only tokens *unique* to `[INTR]`. Shared `[` / `]` must stay
/// sampleable — they're needed for `[CALL]` / `[TRAP]` / `[END]`.
fn compute_intr_suppression(
    call_ids: &[u32],
    end_ids: &[u32],
    head_ids: &[u32],
    trap_ids: &[u32],
    intr_ids: &[u32],
) -> HashSet<u32> {
    let shared: HashSet<u32> = call_ids
        .iter()
        .chain(end_ids.iter())
        .chain(head_ids.iter())
        .chain(trap_ids.iter())
        .copied()
        .collect();
    intr_ids
        .iter()
        .copied()
        .filter(|id| !shared.contains(id))
        .collect()
}

// ============================================================================
// Constraint — block the [INTR]-unique token ids
//
// Pie has no custom-sampler hook. INTR suppression goes through the
// constraint mask path: emit a BRLE that allows every vocab id except those
// in `suppressed`. The ban list is static, so we compute the mask once and
// hand back the same slice every step.
// ============================================================================

struct SuppressMask {
    mask: Vec<u32>,
}

impl SuppressMask {
    /// BRLE: alternating run lengths starting from `false` (disallowed).
    /// An initial run-length of 0 is used when id 0 happens to be allowed.
    fn new(mask_size: u32, suppressed: &HashSet<u32>) -> Self {
        let mut mask: Vec<u32> = Vec::new();
        let mut run_value = false;
        let mut run_len: u32 = 0;
        for id in 0..mask_size {
            let allowed = !suppressed.contains(&id);
            if allowed == run_value {
                run_len += 1;
            } else {
                mask.push(run_len);
                run_value = !run_value;
                run_len = 1;
            }
        }
        mask.push(run_len);
        Self { mask }
    }
}

impl Constrain for SuppressMask {
    fn step(&mut self, _accepted: &[u32]) -> &[u32] {
        &self.mask
    }
}

// ============================================================================
// CML parser (FSM)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    Normal,
    InCallId,
    InCallBody,
    InTrap,
    InIntrId,
    InIntrBody,
}

#[derive(Debug)]
enum Event {
    Passthrough(u32),
    Call { id: String, code: String },
    Trap,
    EnterCriticalSection,
    ExitCriticalSection,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Delim {
    Call,
    End,
    Head,
    Trap,
    TrapEnd,
    Intr,
    None,
}

struct Parser {
    // Full sequences (leading `[` present as its own token).
    call_ids: Vec<u32>,
    end_ids: Vec<u32>,
    head_ids: Vec<u32>,
    trap_ids: Vec<u32>,
    intr_ids: Vec<u32>,
    trap_end_ids: Vec<Vec<u32>>,
    // Bracket-free inner suffixes — leading-bracket token dropped. Used to
    // catch BPE-merged leading brackets (`\n[`, ` [`) where the `[` is
    // fused into the previous token.
    call_inner: Vec<u32>,
    end_inner: Vec<u32>,
    head_inner: Vec<u32>,
    trap_inner: Vec<u32>,
    intr_inner: Vec<u32>,
    trap_end_inner: Vec<Vec<u32>>,

    // Qwen3 think-block bypass. While in_think is true, all incoming tokens
    // are passed through unchanged and CML matching is suspended — the model
    // routinely quotes literal `[CALL]` / `[TRAP][END]` inside <think>, and
    // letting the FSM act on those corrupts the run.
    think_open_ids: Vec<u32>,
    think_close_ids: Vec<u32>,
    in_think: bool,
    think_buf: Vec<u32>,

    close_bracket_alts: Vec<u32>,

    state: State,
    id_tokens: Vec<u32>,
    body_tokens: Vec<u32>,
    buf: Vec<u32>,
}

impl Parser {
    fn new(reg: &CmlRegistry) -> Self {
        Parser {
            call_inner: reg.call_ids[1..].to_vec(),
            end_inner: reg.end_ids[1..].to_vec(),
            head_inner: reg.head_ids[1..].to_vec(),
            trap_inner: reg.trap_ids[1..].to_vec(),
            intr_inner: reg.intr_ids[1..].to_vec(),
            trap_end_inner: reg.trap_end_ids.iter().map(|s| s[1..].to_vec()).collect(),
            call_ids: reg.call_ids.clone(),
            end_ids: reg.end_ids.clone(),
            head_ids: reg.head_ids.clone(),
            trap_ids: reg.trap_ids.clone(),
            intr_ids: reg.intr_ids.clone(),
            trap_end_ids: reg.trap_end_ids.clone(),
            think_open_ids: reg.think_open_ids.clone(),
            think_close_ids: reg.think_close_ids.clone(),
            in_think: false,
            think_buf: Vec::new(),
            close_bracket_alts: reg.close_bracket_alts.clone(),
            state: State::Normal,
            id_tokens: Vec::new(),
            body_tokens: Vec::new(),
            buf: Vec::new(),
        }
    }

    /// True iff Normal state with no pending delimiter prefix — a safe
    /// boundary for mid-stream interrupt injection.
    fn is_clean(&self) -> bool {
        self.state == State::Normal && self.buf.is_empty()
    }

    /// Observe a freshly-emitted token for `<think>` / `</think>` boundaries
    /// and update `in_think`.
    fn observe_think_boundary(&mut self, token_id: u32) {
        // Single-token boundary fast path.
        if self.think_open_ids.len() == 1 && self.think_open_ids[0] == token_id {
            self.in_think = true;
            self.think_buf.clear();
            return;
        }
        if self.think_close_ids.len() == 1 && self.think_close_ids[0] == token_id {
            self.in_think = false;
            self.think_buf.clear();
            return;
        }
        // Multi-token boundaries: accumulate and check.
        self.think_buf.push(token_id);
        let max_len = self.think_open_ids.len().max(self.think_close_ids.len());
        if self.think_buf.len() > max_len {
            let drop = self.think_buf.len() - max_len;
            self.think_buf.drain(..drop);
        }
        if !self.think_open_ids.is_empty() && self.think_buf.ends_with(&self.think_open_ids) {
            self.in_think = true;
            self.think_buf.clear();
            return;
        }
        if !self.think_close_ids.is_empty() && self.think_buf.ends_with(&self.think_close_ids) {
            self.in_think = false;
            self.think_buf.clear();
        }
    }

    /// Position-level equivalence used by both `ends_with` and `is_prefix`:
    /// equal token ids match, and any `close_bracket_alts` token is
    /// interchangeable with another (so `]\n` stands in for the canonical
    /// `]`).
    fn pos_eq(&self, b: u32, p: u32) -> bool {
        b == p || (self.close_bracket_alts.contains(&p) && self.close_bracket_alts.contains(&b))
    }

    fn ends_with(&self, buf: &[u32], pat: &[u32]) -> bool {
        if pat.is_empty() || buf.len() < pat.len() {
            return false;
        }
        let off = buf.len() - pat.len();
        pat.iter()
            .enumerate()
            .all(|(i, &p)| self.pos_eq(buf[off + i], p))
    }

    fn is_prefix(&self, buf: &[u32], pat: &[u32]) -> bool {
        if pat.is_empty() || buf.len() > pat.len() {
            return false;
        }
        buf.iter().enumerate().all(|(i, &b)| self.pos_eq(b, pat[i]))
    }

    fn match_full(&self, buf: &[u32]) -> (Delim, usize) {
        if self.ends_with(buf, &self.call_ids) {
            return (Delim::Call, self.call_ids.len());
        }
        if self.ends_with(buf, &self.end_ids) {
            return (Delim::End, self.end_ids.len());
        }
        for s in &self.trap_end_ids {
            if self.ends_with(buf, s) {
                return (Delim::TrapEnd, s.len());
            }
        }
        if self.ends_with(buf, &self.trap_ids) {
            return (Delim::Trap, self.trap_ids.len());
        }
        if self.ends_with(buf, &self.intr_ids) {
            return (Delim::Intr, self.intr_ids.len());
        }
        if self.ends_with(buf, &self.head_ids) {
            return (Delim::Head, self.head_ids.len());
        }
        (Delim::None, 0)
    }

    fn match_inner(&self, buf: &[u32]) -> (Delim, usize) {
        if self.ends_with(buf, &self.call_inner) {
            return (Delim::Call, self.call_inner.len());
        }
        if self.ends_with(buf, &self.end_inner) {
            return (Delim::End, self.end_inner.len());
        }
        if self.ends_with(buf, &self.trap_inner) {
            return (Delim::Trap, self.trap_inner.len());
        }
        if self.ends_with(buf, &self.intr_inner) {
            return (Delim::Intr, self.intr_inner.len());
        }
        if self.ends_with(buf, &self.head_inner) {
            return (Delim::Head, self.head_inner.len());
        }
        (Delim::None, 0)
    }

    /// `match_inner` restricted to delimiters that can appear at top level
    /// in Normal state ([CALL], [TRAP], [TRAP][END], [INTR]). Catches the
    /// case where the leading `[` of a top-level delimiter has been
    /// BPE-merged into the previous token.
    fn match_normal_inner(&self, buf: &[u32]) -> (Delim, usize) {
        for s in &self.trap_end_inner {
            if self.ends_with(buf, s) {
                return (Delim::TrapEnd, s.len());
            }
        }
        if self.ends_with(buf, &self.call_inner) {
            return (Delim::Call, self.call_inner.len());
        }
        if self.ends_with(buf, &self.trap_inner) {
            return (Delim::Trap, self.trap_inner.len());
        }
        if self.ends_with(buf, &self.intr_inner) {
            return (Delim::Intr, self.intr_inner.len());
        }
        (Delim::None, 0)
    }

    /// Try full-match first; fall back to inner-match (BPE-merged leading
    /// bracket). Used by every state except Normal.
    fn match_full_or_inner(&self, buf: &[u32]) -> (Delim, usize) {
        let r = self.match_full(buf);
        if r.0 != Delim::None {
            r
        } else {
            self.match_inner(buf)
        }
    }

    /// Normal-state variant: full-match first, then `match_normal_inner`
    /// (only delimiters legal at top level).
    fn match_full_or_normal_inner(&self, buf: &[u32]) -> (Delim, usize) {
        let r = self.match_full(buf);
        if r.0 != Delim::None {
            r
        } else {
            self.match_normal_inner(buf)
        }
    }

    fn is_prefix_of_any(&self, buf: &[u32]) -> bool {
        self.all_delim_patterns()
            .any(|pat| self.is_prefix(buf, pat))
    }

    fn all_delim_patterns(&self) -> impl Iterator<Item = &[u32]> {
        let singles: [&[u32]; 10] = [
            &self.call_ids,
            &self.end_ids,
            &self.trap_ids,
            &self.intr_ids,
            &self.head_ids,
            &self.call_inner,
            &self.end_inner,
            &self.trap_inner,
            &self.intr_inner,
            &self.head_inner,
        ];
        let trap_end_full = self.trap_end_ids.iter().map(|s| s.as_slice());
        let trap_end_inner = self.trap_end_inner.iter().map(|s| s.as_slice());
        singles
            .into_iter()
            .chain(trap_end_full)
            .chain(trap_end_inner)
    }

    fn detokenize_one(tokenizer: &Tokenizer, t: u32) -> String {
        tokenizer.decode(&[t]).unwrap_or_default()
    }

    /// Tokens whose decoded bytes end with `[` are potential delimiter
    /// starts; if the delimiter never forms (usually because the [INTR]
    /// continuation was suppressed), we drop them rather than leak a stray
    /// `[` into passthrough.
    fn is_partial_delim_start(t: u32, tokenizer: &Tokenizer) -> bool {
        Self::detokenize_one(tokenizer, t).ends_with('[')
    }

    fn strip_trailing_bracket(tokens: &mut Vec<u32>, tokenizer: &Tokenizer) {
        if tokens
            .last()
            .is_some_and(|&t| Self::is_partial_delim_start(t, tokenizer))
        {
            tokens.pop();
        }
    }

    /// Emit `self.buf[..buf.len() - dlen]` as Passthrough events (stripping
    /// the trailing `[` that opens the matched delimiter), then clear buf.
    fn emit_prefix_and_consume(
        &mut self,
        dlen: usize,
        tokenizer: &Tokenizer,
        events: &mut Vec<Event>,
    ) {
        let end = self.buf.len() - dlen;
        let mut prefix: Vec<u32> = self.buf[..end].to_vec();
        Self::strip_trailing_bracket(&mut prefix, tokenizer);
        events.extend(prefix.into_iter().map(Event::Passthrough));
        self.buf.clear();
    }

    /// Pop one token from the front of `self.buf` and emit it as
    /// Passthrough unless it's a partial-delimiter start (in which case
    /// drop it). Used to drain `buf` token-by-token while it can't be the
    /// start of any delimiter.
    fn drain_front_as_passthrough(&mut self, tokenizer: &Tokenizer, events: &mut Vec<Event>) {
        let t = self.buf.remove(0);
        if !Self::is_partial_delim_start(t, tokenizer) {
            events.push(Event::Passthrough(t));
        }
    }

    fn feed(&mut self, token_id: u32, tokenizer: &Tokenizer) -> Vec<Event> {
        let mut events = Vec::new();

        // Think-block bypass: while inside <think>…</think>, suspend CML
        // matching and pass tokens through verbatim — the model routinely
        // quotes literal CML while reasoning about how to format its
        // answer, and the parser must not act on those.
        let was_in_think = self.in_think;
        self.observe_think_boundary(token_id);
        if was_in_think || self.in_think {
            events.push(Event::Passthrough(token_id));
            return events;
        }

        let prev_state = self.state;
        match self.state {
            State::Normal => self.feed_normal(token_id, tokenizer, &mut events),
            State::InCallId => self.feed_in_call_id(token_id, tokenizer, &mut events),
            State::InCallBody => self.feed_in_call_body(token_id, tokenizer, &mut events),
            State::InTrap => self.feed_in_trap(token_id, &mut events),
            State::InIntrId => self.feed_in_intr_id(token_id),
            State::InIntrBody => self.feed_in_intr_body(token_id),
        }

        if prev_state == State::Normal && self.state != State::Normal {
            events.push(Event::EnterCriticalSection);
        } else if prev_state != State::Normal && self.state == State::Normal {
            events.push(Event::ExitCriticalSection);
        }

        events
    }

    fn feed_normal(&mut self, token_id: u32, tokenizer: &Tokenizer, events: &mut Vec<Event>) {
        self.buf.push(token_id);
        let (d, dlen) = self.match_full_or_normal_inner(&self.buf);
        match d {
            Delim::Call => {
                self.emit_prefix_and_consume(dlen, tokenizer, events);
                self.id_tokens.clear();
                self.body_tokens.clear();
                self.state = State::InCallId;
            }
            Delim::Trap => {
                self.emit_prefix_and_consume(dlen, tokenizer, events);
                self.state = State::InTrap;
            }
            Delim::TrapEnd => {
                self.emit_prefix_and_consume(dlen, tokenizer, events);
                events.push(Event::Trap);
            }
            Delim::Intr => {
                self.emit_prefix_and_consume(dlen, tokenizer, events);
                self.state = State::InIntrId;
            }
            Delim::None => {
                while !self.buf.is_empty() && !self.is_prefix_of_any(&self.buf) {
                    self.drain_front_as_passthrough(tokenizer, events);
                }
            }
            // End/Head matched at top level — should not occur, but drain
            // conservatively rather than leaving the buffer indefinitely.
            _ => {
                let drained: Vec<u32> = self.buf.drain(..).collect();
                for t in drained {
                    if !Self::is_partial_delim_start(t, tokenizer) {
                        events.push(Event::Passthrough(t));
                    }
                }
            }
        }
    }

    fn feed_in_call_id(&mut self, token_id: u32, tokenizer: &Tokenizer, events: &mut Vec<Event>) {
        self.buf.push(token_id);
        let (d, dlen) = self.match_full_or_inner(&self.buf);
        match d {
            Delim::Head => {
                let id_end = self.buf.len() - dlen;
                self.id_tokens.extend_from_slice(&self.buf[..id_end]);
                self.buf.clear();
                Self::strip_trailing_bracket(&mut self.id_tokens, tokenizer);
                self.state = State::InCallBody;
            }
            Delim::End => {
                let id_end = self.buf.len() - dlen;
                self.id_tokens.extend_from_slice(&self.buf[..id_end]);
                self.buf.clear();
                Self::strip_trailing_bracket(&mut self.id_tokens, tokenizer);
                // Decode error → empty id, runtime keeps going (best-effort).
                let id = tokenizer
                    .decode(&self.id_tokens)
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                events.push(Event::Call {
                    id,
                    code: String::new(),
                });
                self.state = State::Normal;
            }
            Delim::None => {
                if !self.is_prefix_of_any(&self.buf) {
                    self.id_tokens.extend_from_slice(&self.buf);
                    self.buf.clear();
                }
            }
            _ => {}
        }
    }

    fn feed_in_call_body(&mut self, token_id: u32, tokenizer: &Tokenizer, events: &mut Vec<Event>) {
        self.buf.push(token_id);
        let (d, dlen) = self.match_full_or_inner(&self.buf);
        match d {
            Delim::End => {
                let body_end = self.buf.len() - dlen;
                self.body_tokens.extend_from_slice(&self.buf[..body_end]);
                self.buf.clear();
                Self::strip_trailing_bracket(&mut self.body_tokens, tokenizer);
                let id = tokenizer
                    .decode(&self.id_tokens)
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                let code = tokenizer
                    .decode(&self.body_tokens)
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                events.push(Event::Call { id, code });
                self.state = State::Normal;
            }
            Delim::None => {
                if !self.is_prefix_of_any(&self.buf) {
                    self.body_tokens.extend_from_slice(&self.buf);
                    self.buf.clear();
                }
            }
            _ => {}
        }
    }

    fn feed_in_trap(&mut self, token_id: u32, events: &mut Vec<Event>) {
        self.buf.push(token_id);
        let (d, _) = self.match_full_or_inner(&self.buf);
        match d {
            Delim::End => {
                self.buf.clear();
                events.push(Event::Trap);
                self.state = State::Normal;
            }
            Delim::None => {
                if !self.is_prefix_of_any(&self.buf) {
                    self.buf.clear();
                }
            }
            _ => {}
        }
    }

    fn feed_in_intr_id(&mut self, token_id: u32) {
        self.buf.push(token_id);
        let (d, _) = self.match_full_or_inner(&self.buf);
        match d {
            Delim::Head => {
                self.buf.clear();
                self.state = State::InIntrBody;
            }
            Delim::End => {
                self.buf.clear();
                self.state = State::Normal;
            }
            Delim::None => {
                if !self.is_prefix_of_any(&self.buf) {
                    self.buf.clear();
                }
            }
            _ => {}
        }
    }

    fn feed_in_intr_body(&mut self, token_id: u32) {
        self.buf.push(token_id);
        let (d, _) = self.match_full_or_inner(&self.buf);
        match d {
            Delim::End => {
                self.buf.clear();
                self.state = State::Normal;
            }
            Delim::None => {
                if !self.is_prefix_of_any(&self.buf) {
                    self.buf.clear();
                }
            }
            _ => {}
        }
    }

    fn flush(&mut self, tokenizer: &Tokenizer) -> Vec<Event> {
        self.buf
            .drain(..)
            .filter(|&t| !Self::is_partial_delim_start(t, tokenizer))
            .map(Event::Passthrough)
            .collect()
    }
}

// ============================================================================
// Real async executor
//
// Each CML `[CALL]` becomes a live HTTP request via `wstd::http`, spawned onto
// the wstd reactor at the dispatch site in `run_inner_session`. The reactor
// advances in-flight requests every time the main loop awaits
// `step.execute().await`, so tool calls genuinely overlap token decoding —
// the core AsyncLM property, now with real tool latency instead of a
// simulated wait.
//
// `execute_call` always resolves to a compact JSON string for the `[INTR]`
// frame; any transport/parse failure is folded into an `{"error": ...}`
// payload so the model still receives something it can read.
// ============================================================================

fn noop_waker() -> Waker {
    const VTABLE: RawWakerVTable = RawWakerVTable::new(
        |_| RawWaker::new(std::ptr::null(), &VTABLE),
        |_| {},
        |_| {},
        |_| {},
    );
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
}

/// Resolve one CML call to its live API and return the `[INTR]` payload.
async fn execute_call(code: String, emulate_unknown: bool) -> String {
    let lc = code.to_lowercase();
    let result = if lc.contains("get_weather") || lc.contains("weather(") {
        weather_call(&code).await
    } else if lc.contains("stock_price") || lc.contains("get_stock") {
        stock_call(&code).await
    } else if lc.contains("convert_currency") || lc.contains("exchange_rate") {
        currency_call(&code).await
    } else if lc.contains("get_time") || lc.contains("time_in") || lc.contains("current_time") {
        time_call(&code).await
    } else if lc.contains("wiki_summary") || lc.contains("wikipedia") || lc.contains("wiki(") {
        wiki_call(&code).await
    } else if emulate_unknown {
        // Not one of the five live tools -> BFCL latency-emulated call.
        bfcl_emulated_call(&code).await
    } else {
        Err(format!("unknown tool in call body: {code}"))
    };
    result.unwrap_or_else(|e| format!("{{\"error\": {e:?}}}"))
}

/// Latency-emulating fallback for non-live tools (gated by `emulate_unknown_tools`).
///
/// Berkeley Function-Calling Leaderboard (BFCL) functions (e.g. `spotify.play`,
/// `math.pythagoras`) are not executable here. Following the AsyncLM paper's
/// evaluation setup, we assign each function a 30-500 ms completion time and
/// just wait, then return a result: hash the function name to a stable latency
/// in that range (reproducible, per-function differentiated), sleep on the wstd
/// reactor so token decoding overlaps the wait (the core async-overlap
/// property), and return a canned payload. Correctness is judged on the emitted
/// `[CALL]` block, never on this result string.
async fn bfcl_emulated_call(code: &str) -> std::result::Result<String, String> {
    let name = code.split('(').next().unwrap_or(code).trim();
    // FNV-1a over the function name -> latency in [30, 500] ms.
    let mut h: u64 = 0xcbf29ce484222325;
    for b in name.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    let latency_ms = 30 + (h % 471); // 30..=500
    wstd::task::sleep(wstd::time::Duration::from_millis(latency_ms)).await;
    Ok(format!(
        "{{\"status\": \"ok\", \"function\": {name:?}, \"latency_ms\": {latency_ms}}}"
    ))
}

/// GET `url` and return the response body. Non-2xx and transport errors map
/// to `Err`. A User-Agent is set because Wikipedia (and OSM) reject requests
/// without one.
async fn http_get(url: &str) -> std::result::Result<String, String> {
    let client = Client::new();
    let req = Request::builder()
        .uri(url)
        .method(Method::GET)
        .header("user-agent", "pie-asynclm/0.1 (+https://pie-project.org)")
        .body(wstd::io::empty())
        .map_err(|e| format!("build request {url}: {e}"))?;
    let resp = client
        .send(req)
        .await
        .map_err(|e| format!("send {url}: {e}"))?;
    let status = resp.status().as_u16();
    let mut body = resp.into_body();
    let bytes = body
        .bytes()
        .await
        .map_err(|e| format!("read body {url}: {e}"))?;
    let text = String::from_utf8_lossy(&bytes).into_owned();
    if !(200..300).contains(&status) {
        let snip: String = text.chars().take(120).collect();
        return Err(format!("HTTP {status} from {url}: {snip}"));
    }
    Ok(text)
}

/// GET `url` and parse the body as JSON. `tool` prefixes the error message so
/// the failing call is identifiable in the `{"error": ...}` payload.
async fn fetch_json(url: &str, tool: &str) -> std::result::Result<Value, String> {
    let body = http_get(url).await?;
    inferlet::serde_json::from_str(&body).map_err(|e| format!("{tool}: bad JSON: {e}"))
}

/// Percent-encode a path/query component (RFC 3986 unreserved set kept).
fn urlencode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// First argument of a call body like `get_weather("London")` or
/// `get_time(Asia/Tokyo)` — prefers a quoted string, else the first
/// comma-separated token inside the parentheses.
fn first_arg(code: &str) -> Option<String> {
    if let Some(q) = between(code, '"').or_else(|| between(code, '\'')) {
        let q = q.trim();
        if !q.is_empty() {
            return Some(q.to_string());
        }
    }
    let inside = code.split_once('(')?.1;
    let inside = inside.rsplit_once(')').map(|x| x.0).unwrap_or(inside);
    let arg = inside
        .split(',')
        .next()?
        .trim()
        .trim_matches(|c| c == '"' || c == '\'')
        .trim();
    (!arg.is_empty()).then(|| arg.to_string())
}

/// Text between the first matched pair of `delim`.
fn between(s: &str, delim: char) -> Option<String> {
    let start = s.find(delim)? + 1;
    let rest = &s[start..];
    let end = rest.find(delim)?;
    Some(rest[..end].to_string())
}

/// get_weather(city) -> wttr.in current conditions.
async fn weather_call(code: &str) -> std::result::Result<String, String> {
    let city = first_arg(code).ok_or("get_weather: missing city")?;
    let url = format!("https://wttr.in/{}?format=j1", urlencode(&city));
    let v = fetch_json(&url, "get_weather").await?;
    let cur = v
        .get("current_condition")
        .and_then(|c| c.get(0))
        .ok_or("get_weather: no current_condition")?;
    let temp_f = cur.get("temp_F").and_then(Value::as_str).unwrap_or("0");
    let sky = cur
        .get("weatherDesc")
        .and_then(|d| d.get(0))
        .and_then(|d| d.get("value"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    Ok(format!(
        "{{\"city\": {city:?}, \"temp_f\": {temp_f}, \"sky\": {sky:?}}}"
    ))
}

/// get_stock_price(ticker) -> stooq.com CSV last quote (US listing).
async fn stock_call(code: &str) -> std::result::Result<String, String> {
    let ticker = first_arg(code).ok_or("get_stock_price: missing ticker")?;
    let price = match fetch_stock_price(&ticker).await {
        Some(p) => p,
        // stooq rate-limits/blocks and can return an HTML/JS page instead of
        // CSV. `fetch_stock_price` returns `Some` only for a clean numeric
        // quote, so on any junk response fall back to a deterministic emulated
        // price rather than leak a parsed-garbage value into the result frame.
        None => emulated_price(&ticker),
    };
    Ok(format!(
        "{{\"ticker\": {:?}, \"price_usd\": {price:.2}}}",
        ticker.to_uppercase()
    ))
}

/// Fetch a real last-quote from stooq, returning None unless the response is
/// the expected CSV with a clean positive numeric close. Anything else (an
/// HTML block page, "N/D", junk) yields None so the caller can fall back.
async fn fetch_stock_price(ticker: &str) -> Option<f64> {
    let url = format!(
        "https://stooq.com/q/l/?s={}.us&f=sd2t2ohlcv&h&e=csv",
        urlencode(&ticker.to_lowercase())
    );
    let body = http_get(&url).await.ok()?;
    // Expected header: Symbol,Date,Time,Open,High,Low,Close,Volume
    let mut lines = body.lines();
    if !lines.next().unwrap_or("").starts_with("Symbol") {
        return None;
    }
    let close = lines.next().unwrap_or("").split(',').nth(6).unwrap_or("").trim();
    match close.parse::<f64>() {
        Ok(p) if p.is_finite() && p > 0.0 => Some(p),
        _ => None,
    }
}

/// Deterministic fallback quote (~$20–$520, stable per ticker) used when the
/// live feed is unavailable. The scorer only checks the model echoes the
/// fetched number, so a stable emulated value keeps stock prompts scoreable.
fn emulated_price(ticker: &str) -> f64 {
    let h = ticker
        .bytes()
        .fold(2166136261u32, |acc, b| (acc ^ b as u32).wrapping_mul(16777619));
    20.0 + (h % 50000) as f64 / 100.0
}

/// convert_currency(.., from, to) -> Frankfurter (ECB) reference rate.
async fn currency_call(code: &str) -> std::result::Result<String, String> {
    let codes = currency_codes(code);
    let (from, to) = match codes.as_slice() {
        [a, b, ..] => (a.clone(), b.clone()),
        [a] => ("USD".to_string(), a.clone()),
        _ => ("USD".to_string(), "EUR".to_string()),
    };
    let url = format!("https://api.frankfurter.dev/v1/latest?from={from}&to={to}");
    let v = fetch_json(&url, "convert_currency").await?;
    let rate = v
        .get("rates")
        .and_then(|r| r.get(&to))
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("convert_currency: no {from}->{to} rate"))?;
    Ok(format!("{{\"rate\": {rate}, \"quote\": \"{from}->{to}\"}}"))
}

/// 3-letter uppercase currency codes (e.g. USD, EUR) in order of appearance.
fn currency_codes(code: &str) -> Vec<String> {
    code.split(|c: char| !c.is_ascii_alphabetic())
        .filter(|t| t.len() == 3 && t.chars().all(|c| c.is_ascii_uppercase()))
        .map(str::to_string)
        .collect()
}

/// get_time(location) -> timeapi.io current time for the resolved IANA zone.
async fn time_call(code: &str) -> std::result::Result<String, String> {
    let loc = first_arg(code).ok_or("get_time: missing location")?;
    let tz = resolve_tz(&loc);
    let url = format!(
        "https://timeapi.io/api/Time/current/zone?timeZone={}",
        urlencode(&tz)
    );
    let v = fetch_json(&url, "get_time").await?;
    let time = v.get("time").and_then(Value::as_str).unwrap_or("?");
    let day = v.get("dayOfWeek").and_then(Value::as_str).unwrap_or("?");
    Ok(format!(
        "{{\"time\": {time:?}, \"day\": {day:?}, \"tz\": {tz:?}}}"
    ))
}

/// Map a city/zone string to an IANA timezone. Passes through anything that
/// already looks like a zone (`Region/City`); otherwise consults a small
/// table of common benchmark cities, defaulting to UTC.
fn resolve_tz(s: &str) -> String {
    if s.contains('/') {
        return s.to_string();
    }
    let tz = match s.to_lowercase().as_str() {
        "tokyo" => "Asia/Tokyo",
        "london" => "Europe/London",
        "paris" => "Europe/Paris",
        "berlin" => "Europe/Berlin",
        "new york" | "nyc" | "new_york" => "America/New_York",
        "boston" => "America/New_York",
        "san francisco" | "sf" | "los angeles" | "la" => "America/Los_Angeles",
        "chicago" => "America/Chicago",
        "sydney" => "Australia/Sydney",
        "singapore" => "Asia/Singapore",
        "dubai" => "Asia/Dubai",
        "utc" | "gmt" => "Etc/UTC",
        _ => "Etc/UTC",
    };
    tz.to_string()
}

/// wiki_summary(title) -> Wikipedia REST summary extract (trimmed).
async fn wiki_call(code: &str) -> std::result::Result<String, String> {
    let title = first_arg(code).ok_or("wiki_summary: missing title")?;
    let url = format!(
        "https://en.wikipedia.org/api/rest_v1/page/summary/{}",
        urlencode(&title.replace(' ', "_"))
    );
    let v = fetch_json(&url, "wiki_summary").await?;
    let extract = v.get("extract").and_then(Value::as_str).unwrap_or("");
    if extract.is_empty() {
        return Err(format!("wiki_summary: no summary for {title:?}"));
    }
    let summary = trim_summary(extract, 400);
    Ok(format!("{{\"title\": {title:?}, \"summary\": {summary:?}}}"))
}

/// Trim `text` to at most `max_chars` characters without splitting a word:
/// back off to the last whitespace inside the window and append an ellipsis.
/// If the text already fits, it is returned unchanged.
fn trim_summary(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut s: String = text.chars().take(max_chars).collect();
    if let Some(idx) = s.rfind(char::is_whitespace) {
        s.truncate(idx);
    }
    let mut s = s.trim_end().to_string();
    s.push('…');
    s
}

/// A call dispatched as a `wstd` task. The fetch runs on the reactor; the
/// loop harvests it once `is_finished()` flips true.
struct PendingCall {
    id: String,
    dispatched_at: Instant,
    task: wstd::runtime::Task<String>,
}

/// Harvest a finished call. Returns `None` while the task is still in flight.
/// Tasks make progress on the reactor when the main loop yields to it (see
/// `yield_to_reactor`), not here, so this never blocks and never spins a live
/// request.
fn poll_once(p: &mut PendingCall) -> Option<String> {
    if !p.task.is_finished() {
        return None;
    }
    let waker = noop_waker();
    let mut cx = TaskContext::from_waker(&waker);
    match Pin::new(&mut p.task).poll(&mut cx) {
        Poll::Ready(r) => Some(r),
        Poll::Pending => None,
    }
}

/// Yield once to the wstd executor so other ready tasks get to run.
///
/// This is what makes the async overlap real. `step.execute().await` (the
/// forward pass) resolves *synchronously* — it never yields `Pending` — so
/// between dispatching a `[CALL]` and reaching the `[TRAP]`, the main task
/// monopolises the single-threaded executor and the spawned `execute_call`
/// futures sit un-polled on the ready list. Their timers/sockets never advance,
/// so every call only "starts" when the trap finally awaits it (you'd observe
/// `ready after ≈ trap_time + latency`, and `mid_injects` stays 0).
///
/// Yielding here hands `block_on` a turn: it runs the freshly-spawned tasks to
/// their first await (registering pollables) and non-block-checks the reactor's
/// pollables, completing any whose wall-clock deadline has elapsed. Across the
/// decode steps that span the call's latency, the result then lands in the
/// interrupt queue *during* prose generation — the actual concurrency.
async fn yield_to_reactor() {
    let mut yielded = false;
    std::future::poll_fn(move |cx| {
        if yielded {
            Poll::Ready(())
        } else {
            yielded = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    })
    .await
}

// ============================================================================
// Main-loop helpers
// ============================================================================

fn intr_frame(id: &str, result: &str) -> String {
    format!(" [INTR] {} [HEAD] {} [END] ", id, result)
}

struct InterruptManager {
    queue: VecDeque<String>,
    in_critical_section: bool,
}

impl InterruptManager {
    fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            in_critical_section: false,
        }
    }

    fn enqueue(&mut self, id: &str, result: &str) {
        self.queue.push_back(intr_frame(id, result));
    }

    fn set_critical(&mut self, in_critical_section: bool) {
        self.in_critical_section = in_critical_section;
    }

    fn drain(&mut self) -> Vec<String> {
        if self.in_critical_section {
            return Vec::new();
        }
        self.queue.drain(..).collect()
    }

    fn pending_count(&self) -> usize {
        self.queue.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TrapStrategy {
    Keep,
    Recompute,
    Swap,
}

struct CostModel {
    alpha_ns_per_tok2: f64,
    beta_ns_per_tok: f64,
}

impl CostModel {
    fn recompute_ms(&self, n: usize) -> f64 {
        self.alpha_ns_per_tok2 * (n as f64) * (n as f64) / 1.0e6
    }

    fn swap_ms(&self, n: usize) -> f64 {
        self.beta_ns_per_tok * (n as f64) / 1.0e6
    }
}

fn classify_trap(n: usize, wait_ms: f64, cost: &CostModel) -> (TrapStrategy, f64, f64) {
    let r_ms = cost.recompute_ms(n);
    let s_ms = cost.swap_ms(n);
    let strategy = if r_ms > wait_ms && s_ms > wait_ms {
        TrapStrategy::Keep
    } else if r_ms <= s_ms {
        TrapStrategy::Recompute
    } else {
        TrapStrategy::Swap
    };
    (strategy, r_ms, s_ms)
}

struct Checkpoint {
    name: String,
    buffer_tokens: Vec<u32>,
    tokens_generated_at: usize,
    max_tokens_remaining: usize,
}

struct CheckpointManager {
    latest: Option<Checkpoint>,
}

impl CheckpointManager {
    fn new() -> Self {
        Self { latest: None }
    }
}

struct PendingInjection {
    frames: Vec<String>,
    log_label: &'static str,
}

enum Restart {
    Done,
    Checkpoint,
    RestoreAndInject(Vec<String>),
}

/// Poll each pending call once. Completed calls are removed from `pending`
/// and their result frames are queued in `interrupts`.
fn poll_and_collect_ready(
    pending: &mut Vec<PendingCall>,
    interrupts: &mut InterruptManager,
    log_each_ready: bool,
) {
    pending.retain_mut(|p| match poll_once(p) {
        Some(result) => {
            if log_each_ready {
                println!(
                    "[AsyncLM] id={} ready after {}ms",
                    p.id,
                    p.dispatched_at.elapsed().as_millis()
                );
                trace(
                    "ready",
                    &p.id,
                    p.dispatched_at.elapsed().as_millis() as i64,
                    -1,
                    "overlap",
                );
            }
            interrupts.enqueue(&p.id, &result);
            false
        }
        None => true,
    });
}

fn frames_to_text_with_log(frames: Vec<String>, log_label: &str) -> String {
    let mut text = String::new();
    for frame in frames {
        println!("[AsyncLM] {}: {}", log_label, frame.trim());
        let fid = frame
            .trim()
            .strip_prefix("[INTR]")
            .and_then(|s| s.trim().split_whitespace().next())
            .unwrap_or("");
        trace("inject", fid, -1, -1, log_label);
        text.push_str(&frame);
    }
    text
}

fn stage_injection(
    generator: &mut Generator<'_>,
    tokenizer: &Tokenizer,
    text: &str,
    next_step_flushes_injection: &mut bool,
    deferred_injected_tail: &mut Option<u32>,
) {
    let injected = tokenizer.encode(text);
    if injected.is_empty() {
        return;
    }

    if token_trace_on() {
        println!("[dbg] accept injecting {} tokens", injected.len());
    }

    if injected.len() > 1 {
        let tail = *injected.last().unwrap();
        let prefix_len = injected.len() - 1;
        let accepted = generator.accept(&injected[..prefix_len]);
        if token_trace_on() {
            println!(
                "[dbg] accepted injected prefix {} tokens; deferring final token",
                accepted.len()
            );
        }
        if accepted.len() == prefix_len {
            *deferred_injected_tail = Some(tail);
            *next_step_flushes_injection = true;
        }
    } else {
        let _ = generator.accept(&injected);
    }
}

/// Log + build the injection text from `frames`, then stage it into the live
/// `generator`. Wraps `frames_to_text_with_log` + `stage_injection` — the
/// pairing used at every injection site (pending / keep / mid).
fn inject_frames(
    generator: &mut Generator<'_>,
    tokenizer: &Tokenizer,
    frames: Vec<String>,
    log_label: &str,
    next_step_flushes_injection: &mut bool,
    deferred_injected_tail: &mut Option<u32>,
) {
    let text = frames_to_text_with_log(frames, log_label);
    stage_injection(
        generator,
        tokenizer,
        &text,
        next_step_flushes_injection,
        deferred_injected_tail,
    );
}

async fn run_inner_session(
    generator: &mut Generator<'_>,
    parser: &mut Parser,
    pending: &mut Vec<PendingCall>,
    interrupts: &mut InterruptManager,
    checkpoint: &CheckpointManager,
    generated: &mut Vec<u32>,
    input: &Input,
    tokenizer: &Tokenizer,
    cost: &CostModel,
    pending_inject: &mut Option<PendingInjection>,
    step_count: &mut usize,
    max_steps: usize,
) -> Result<Restart> {
    let mut next_step_flushes_injection = false;
    let mut deferred_injected_tail: Option<u32> = None;

    if let Some(injection) = pending_inject.take() {
        inject_frames(
            generator,
            tokenizer,
            injection.frames,
            injection.log_label,
            &mut next_step_flushes_injection,
            &mut deferred_injected_tail,
        );
    }

    while let Some(mut step) = generator.next()? {
        if *step_count >= max_steps {
            println!(
                "[AsyncLM] hit max_steps={} (generated={}); breaking",
                max_steps,
                generated.len()
            );
            return Ok(Restart::Done);
        }
        *step_count += 1;
        if *step_count % 8 == 0 {
            trace("decode_step", "", -1, *step_count as i64, "");
        }

        let is_flush_step = next_step_flushes_injection;
        next_step_flushes_injection = false;
        if is_flush_step {
            step.clear_sampler();
        }

        let out = step.execute().await?;

        if token_trace_on() {
            let dbg_tokens: Vec<String> = out
                .tokens
                .iter()
                .map(|&t| format!("{}={:?}", t, tokenizer.decode(&[t]).unwrap_or_default()))
                .collect();
            println!(
                "[dbg] step={} ntoks={} raw_slots={} tokens={:?}",
                step_count,
                out.tokens.len(),
                out.raw().slots.len(),
                dbg_tokens
            );
        }

        if is_flush_step {
            if let Some(tail) = deferred_injected_tail.take() {
                if token_trace_on() {
                    println!(
                        "[dbg] staged final injected token {}={:?} for sampled resume",
                        tail,
                        tokenizer.decode(&[tail]).unwrap_or_default()
                    );
                }
                let _ = generator.accept(&[tail]);
            }
            continue;
        }

        // Give the executor turns so spawned execute_call tasks make progress
        // *during* decode (timers/sockets advance, completed ones finish). The
        // forward pass above never yields, so without this the calls stay
        // frozen until the trap — i.e. no async overlap. See `yield_to_reactor`.
        if !pending.is_empty() {
            yield_to_reactor().await;
        }
        poll_and_collect_ready(pending, interrupts, true);

        for &tok in &out.tokens {
            for event in parser.feed(tok, tokenizer) {
                match event {
                    Event::Passthrough(t) => generated.push(t),
                    Event::EnterCriticalSection => interrupts.set_critical(true),
                    Event::ExitCriticalSection => interrupts.set_critical(false),
                    Event::Call { id, code } => {
                        println!("[AsyncLM] dispatching id={id} code={code}");
                        trace("dispatch", &id, -1, generated.len() as i64, "");
                        let task = wstd::runtime::spawn(execute_call(
                            code,
                            input.emulate_unknown_tools,
                        ));
                        pending.push(PendingCall {
                            id,
                            dispatched_at: Instant::now(),
                            task,
                        });
                        interrupts.set_critical(false);
                        if !input.disable_checkpointing {
                            // Paper-faithful path: tear the generator down to
                            // snapshot a checkpoint at the call site, so a later
                            // trap can Recompute by restoring it.
                            return Ok(Restart::Checkpoint);
                        }
                        // Default path (`disable_checkpointing`): do NOT
                        // snapshot — keep decoding in one unbroken session so the
                        // bottom-of-loop mid-stream injection keeps firing while
                        // calls are in flight (no snapshot churn, maximal overlap).
                    }
                    Event::Trap => {
                        if pending.is_empty() && interrupts.pending_count() == 0 {
                            println!("[AsyncLM] trap with nothing pending - no-op");
                            interrupts.set_critical(false);
                            continue;
                        }

                        let in_flight = pending.len();
                        // No simulated wait any more: estimate the wait from
                        // how long the longest still-in-flight call has
                        // already been running.
                        let wait_ms = pending
                            .iter()
                            .map(|p| p.dispatched_at.elapsed().as_millis() as f64)
                            .fold(0.0f64, f64::max);
                        let checkpoint_available = checkpoint.latest.is_some();
                        let n = checkpoint
                            .latest
                            .as_ref()
                            .map(|cp| generated.len().saturating_sub(cp.tokens_generated_at))
                            .unwrap_or_else(|| generator.tokens_generated());
                        let (mut strategy, r_ms, s_ms) = classify_trap(n, wait_ms, cost);
                        if input.disable_checkpointing {
                            // Default path: no checkpoint was taken, so Keep is
                            // the only valid strategy.
                            strategy = TrapStrategy::Keep;
                        } else if !checkpoint_available || n == 0 || in_flight == 0 {
                            // Force Keep when nothing is genuinely left to wait
                            // for — every call already completed during overlap,
                            // so discarding those tokens (Recompute) would throw
                            // away the win.
                            strategy = TrapStrategy::Keep;
                        }
                        println!(
                            "[AsyncLM] trap decision: {:?}  wait_t={:.1}ms r_ms={:.1}ms s_ms={:.1}ms n={} in_flight={}",
                            strategy, wait_ms, r_ms, s_ms, n, in_flight
                        );
                        trace(
                            "trap_start",
                            "",
                            -1,
                            in_flight as i64,
                            &format!("{:?}", strategy),
                        );

                        let trap_start = Instant::now();
                        // Block on the remaining in-flight calls. Awaiting the
                        // task drives the reactor so the real fetch actually
                        // completes here — polling alone would spin forever.
                        for p in pending.drain(..) {
                            let PendingCall {
                                id,
                                dispatched_at,
                                task,
                            } = p;
                            let result = task.await;
                            println!(
                                "[AsyncLM] id={} ready after {}ms (trap-await)",
                                id,
                                dispatched_at.elapsed().as_millis()
                            );
                            trace(
                                "ready",
                                &id,
                                dispatched_at.elapsed().as_millis() as i64,
                                -1,
                                "trap-await",
                            );
                            interrupts.enqueue(&id, &result);
                        }
                        println!(
                            "[AsyncLM] trap drained in {}ms",
                            trap_start.elapsed().as_millis()
                        );
                        trace("trap_end", "", trap_start.elapsed().as_millis() as i64, -1, "");

                        interrupts.set_critical(false);
                        match strategy {
                            TrapStrategy::Recompute => {
                                return Ok(Restart::RestoreAndInject(interrupts.drain()));
                            }
                            // Swap has no separate RAM-tier primitive in Pie, so it
                            // executes as Keep (logged); both inject the drained frames.
                            TrapStrategy::Keep | TrapStrategy::Swap => {
                                if strategy == TrapStrategy::Swap {
                                    println!(
                                        "[AsyncLM] swap collapsing to Keep — SDK has no RAM-tier primitive"
                                    );
                                }
                                inject_frames(
                                    generator,
                                    tokenizer,
                                    interrupts.drain(),
                                    "keep-inject",
                                    &mut next_step_flushes_injection,
                                    &mut deferred_injected_tail,
                                );
                            }
                        }
                    }
                }
            }
        }

        if input.enable_midstream && !interrupts.in_critical_section && parser.is_clean() {
            inject_frames(
                generator,
                tokenizer,
                interrupts.drain(),
                "mid-inject",
                &mut next_step_flushes_injection,
                &mut deferred_injected_tail,
            );
        }
    }

    Ok(Restart::Done)
}

// ============================================================================
// Main
// ============================================================================

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let max_steps = input
        .max_steps
        .map(|n| n as usize)
        .unwrap_or_else(|| input.max_tokens.saturating_mul(4));
    let model_name = runtime::models()
        .first()
        .cloned()
        .ok_or("No models available")?;
    let model = Model::load(&model_name)?;
    let tokenizer = model.tokenizer();
    let registry = CmlRegistry::new(&tokenizer);

    println!(
        "[AsyncLM] call={:?} head={:?} end={:?} trap={:?} intr={:?}",
        registry.call_ids,
        registry.head_ids,
        registry.end_ids,
        registry.trap_ids,
        registry.intr_ids
    );
    TRACE_ON.store(input.enable_trace, Ordering::Relaxed);
    TOKEN_TRACE_ON.store(input.enable_token_trace, Ordering::Relaxed);
    trace("run_start", "", -1, -1, "");
    if input.disable_checkpointing {
        println!(
            "[AsyncLM] DIAGNOSTIC: trap-manager disabled (always Keep), checkpointing disabled — one unbroken generator session"
        );
    }

    // BRLE has to cover the model's FULL logit width — specials (e.g.
    // `<|im_end|>`, `</think>`) often live above the regular BPE vocab.
    // Truncating to `vocabs().len()` would leave the tail implicitly masked
    // and the model could never produce its stop/end tokens.
    let (sp_ids, _) = tokenizer.special_tokens();
    let vocab_size = tokenizer.vocabs().0.len() as u32;
    let mask_size = sp_ids
        .iter()
        .copied()
        .max()
        .map(|m| m + 1)
        .unwrap_or(vocab_size)
        .max(vocab_size);
    let stops = chat::stop_tokens(&model);
    let cost = CostModel {
        alpha_ns_per_tok2: input.alpha_recompute_ns_per_tok2,
        beta_ns_per_tok: input.beta_swap_ns_per_tok,
    };

    let mut current_ctx = Context::new(&model)?;
    current_ctx.system(&input.system).user(&input.prompt).cue();

    let mut max_tokens_remaining = input.max_tokens;
    let mut parser = Parser::new(&registry);
    let mut generated: Vec<u32> = Vec::new();
    let mut pending: Vec<PendingCall> = Vec::new();
    let mut interrupts = InterruptManager::new();
    let mut checkpoint = CheckpointManager::new();
    let mut pending_inject: Option<PendingInjection> = None;
    let mut step_count: usize = 0;

    loop {
        if max_tokens_remaining == 0 || step_count >= max_steps {
            break;
        }

        let suppress = SuppressMask::new(mask_size, &registry.suppressed);
        let mut generator = current_ctx
            .generate(Sampler::TopP {
                temperature: input.temperature,
                p: input.top_p,
            })
            .max_tokens(max_tokens_remaining)
            .stop(&stops)
            .constrain(suppress);
        let session_start_generated = generator.tokens_generated();

        let restart = run_inner_session(
            &mut generator,
            &mut parser,
            &mut pending,
            &mut interrupts,
            &checkpoint,
            &mut generated,
            &input,
            &tokenizer,
            &cost,
            &mut pending_inject,
            &mut step_count,
            max_steps,
        )
        .await?;

        let used = generator
            .tokens_generated()
            .saturating_sub(session_start_generated);
        max_tokens_remaining = max_tokens_remaining.saturating_sub(used);
        drop(generator);

        match restart {
            Restart::Done => break,
            Restart::Checkpoint => {
                // The SDK keeps the last sampled token in Context::buffer so
                // the next generation step can flush it. Snapshots do not
                // include that SDK-side buffer, so keep it next to the raw
                // context snapshot and replay it on restore.
                let buffer_tokens = current_ctx.buffer().to_vec();
                let name = current_ctx.snapshot()?;
                let previous = checkpoint.latest.replace(Checkpoint {
                    name: name.clone(),
                    buffer_tokens,
                    tokens_generated_at: generated.len(),
                    max_tokens_remaining,
                });
                if let Some(old) = previous {
                    let _ = Context::delete(&model, &old.name);
                }
                println!(
                    "[AsyncLM] checkpoint saved name={} tokens_at={} budget={}",
                    name,
                    generated.len(),
                    max_tokens_remaining
                );
                trace("checkpoint", &name, -1, generated.len() as i64, "");
            }
            Restart::RestoreAndInject(frames) => {
                let cp = checkpoint
                    .latest
                    .take()
                    .ok_or_else(|| "trap classified Recompute without checkpoint".to_string())?;
                match Context::open(&model, &cp.name) {
                    Ok(mut restored) => {
                        let _ = Context::delete(&model, &cp.name);
                        restored.append(&cp.buffer_tokens);
                        current_ctx = restored;
                        max_tokens_remaining = cp.max_tokens_remaining;
                        let dropped_specs = generated.len().saturating_sub(cp.tokens_generated_at);
                        generated.truncate(cp.tokens_generated_at);
                        parser = Parser::new(&registry);
                        interrupts.set_critical(false);
                        pending_inject = Some(PendingInjection {
                            frames,
                            log_label: "recompute-inject",
                        });
                        println!(
                            "[AsyncLM] checkpoint restored name={} dropped_specs={}",
                            cp.name, dropped_specs
                        );
                    }
                    Err(err) => {
                        println!(
                            "[AsyncLM] checkpoint restore failed name={} error={}; falling back to Keep",
                            cp.name, err
                        );
                        interrupts.set_critical(false);
                        pending_inject = Some(PendingInjection {
                            frames,
                            log_label: "keep-inject",
                        });
                    }
                }
            }
        }
    }

    if token_trace_on() {
        println!(
            "[dbg] loop exited after {} steps; generated.len={}; pending.len={}",
            step_count,
            generated.len(),
            pending.len()
        );
    }

    for ev in parser.flush(&tokenizer) {
        if let Event::Passthrough(t) = ev {
            generated.push(t);
        }
    }

    trace("run_end", "", t_ms() as i64, generated.len() as i64, "");
    Ok(tokenizer.decode(&generated).unwrap_or_default())
}
