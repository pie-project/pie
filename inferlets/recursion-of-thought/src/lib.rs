//! Recursion-of-Thought (RoT): solve a problem by recursively dividing it
//! into independent subproblems whose answers compose into the final
//! answer.
//!
//! **Low-level ① rewrite (grammar, SEQUENTIAL + fork).** Off the
//! `Context`/`Generator`/`Sampler`/`constrain_with` facade onto the keep-core
//! (`ptir-grammar-tranche-conversion-spec`):
//!   - `KvWorkingSet::fork()` (COW-shared prefix) — every recursive call branches
//!     off a shared prefix, so the system prompt + chat template are KV-cached once;
//!   - `Ebnf(grammar).build_constraint()` → a kept `constraint::GrammarConstraint`
//!     (host `Matcher`: `advance`/`mask`/`is_terminated`) — the hand-rolled grammar
//!     forbids the whitespace stalls JSON-Schema's auto-grammar permits;
//!   - `sampler::grammar_program` — the masked GREEDY sampler `argmax(mask_apply(
//!     logits, mask))` (Argmax). The shipped facade computed-but-DROPPED the mask
//!     (Stage-1); this conversion now ENFORCES the grammar. One `grammar_program`
//!     serves both grammars (the program is grammar-agnostic; the mask supplies
//!     the grammar per step);
//!   - `futures::future::join` over forked contexts — sibling subtasks run
//!     concurrently and the runtime batches their forward passes.
//!
//! **Grammar decode is SEQUENTIAL** — the next mask depends on this token, so the
//! run-ahead carrier does NOT apply; each step is a per-fire `grammar_fire`.

use futures::future;
use inferlet::inference::ForwardPass;
use inferlet::working_set::KvWorkingSet;
use inferlet::{chat, geometry, model, prefill, sampler, Constrain, Ebnf, Result, Schema};
use serde::Deserialize;
use serde_json::Value;

use std::future::Future;
use std::pin::Pin;
use std::time::Instant;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_question")]
    question: String,
    #[serde(default = "default_max_depth")]
    max_depth: usize,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    verbose: bool,
}

fn default_question() -> String {
    "Compute the sum 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8.".to_string()
}
fn default_max_depth() -> usize { 2 }
fn default_max_tokens() -> usize { 256 }

/// EBNF grammar for the divide step. We hand-roll it instead of using
/// JSON-Schema because the auto-generated JSON grammar permits arbitrary
/// whitespace between `:` and the value, which under any sampler tends to
/// trap small models in a "emit space until budget runs out" state. This
/// grammar admits no whitespace; control characters that slip into the
/// string body are stripped on the parse side (`sanitize_json`).
const PLAN_GRAMMAR: &str = r#"
root      ::= leaf | branch
leaf      ::= "{\"mode\":\"leaf\",\"answer\":\"" chars "\"}"
branch    ::= "{\"mode\":\"branch\",\"subtasks\":[\"" chars "\",\"" chars "\"]}"
chars     ::= char chars | char
char      ::= [^"\\] | "\\\""
"#;

/// EBNF for the merge step: a single non-empty answer field.
const MERGE_GRAMMAR: &str = r#"
root   ::= "{\"answer\":\"" chars "\"}"
chars  ::= char chars | char
char   ::= [^"\\] | "\\\""
"#;

/// Strip ASCII control characters (besides space and printable ASCII)
/// from raw model output before handing it to `serde_json`. Small models
/// occasionally emit a literal newline / NUL inside the string body; the
/// EBNF lets them, but `serde_json` rejects unescaped control chars in
/// JSON strings. We treat them as noise rather than parse failures.
fn sanitize_json(raw: &str) -> String {
    raw.chars()
        .filter(|c| !c.is_ascii_control() || *c == '\t')
        .collect()
}

const PLAN_SYSTEM: &str = "\
You are a careful problem solver. For each problem, decide if you can solve \
it directly with a short, confident answer (mode=\"leaf\"), or if it should \
be split into exactly two simpler, independent subproblems (mode=\"branch\"). \
Prefer branching whenever the problem has two or more clearly separable \
sub-expressions, products, or distinct quantities. Each subproblem must be \
self-contained: solvable without seeing the other subproblem's answer.

When you pick leaf, write the full computation in `answer` — show the \
expression and the final value, e.g. \"12 + 5 = 17\". Always state the \
final value at the end so the merge step can read it.

Two short examples:

  User: Compute 12 + 5.
  Assistant: {\"mode\":\"leaf\",\"answer\":\"12 + 5 = 17\"}

  User: Compute (12 + 5) * (6 - 2).
  Assistant: {\"mode\":\"branch\",\"subtasks\":[\"Compute 12 + 5\",\"Compute 6 - 2\"]}

Output JSON only — no explanation, no markdown, no extra whitespace.";

const MERGE_SYSTEM: &str = "\
You will be given the answers to two independent subproblems and asked to \
combine them into the final answer. Be concise. Output JSON only.";

/// Macro: `println!` only when verbose.
macro_rules! vlog {
    ($v:expr, $($arg:tt)*) => { if $v { println!($($arg)*) } };
}

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

/// In-inferlet decode context (raw-WIT keep-core, no `Context` facade): a KV
/// working set + sequence cursor, forkable with COW-shared prefix. The greedy
/// grammar decode is sequential (no run-ahead carrier: the next mask depends on
/// this token).
struct Ctx {
    kv: KvWorkingSet,
    seq_len: u32,
    fresh: bool,
}

impl Ctx {
    fn new() -> Self {
        Self { kv: KvWorkingSet::new(), seq_len: 0, fresh: true }
    }

    /// COW-shared-prefix fork (raw `KvWorkingSet::fork`) + cursor copy. The fork
    /// starts a new generation, so its first pass is `fresh` (#26 clear).
    fn fork(&self) -> Result<Self> {
        Ok(Self {
            kv: self.kv.fork().map_err(|e| format!("fork: {e}"))?,
            seq_len: self.seq_len,
            fresh: true,
        })
    }

    /// Non-sampling prefill of `tokens` (the facade's `flush`).
    fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        prefill::tokens(&self.kv, &mut self.seq_len, tokens)
    }

    /// One SEQUENTIAL grammar fire: geometry + input + the masked GREEDY grammar
    /// sampler + execute, advancing the cursor. Fires `fresh_generate` once.
    fn grammar_fire(
        &mut self,
        g: &sampler::LoweredGrammar,
        tokens: &[u32],
        packed_mask: &[u32],
    ) -> Result<ForwardPass> {
        let n = tokens.len() as u32;
        let pass = ForwardPass::new();
        if self.fresh {
            pass.fresh_generate();
            self.fresh = false;
        }
        let geom = geometry::ensure_pages(
            &self.kv,
            geometry::kv_write_geometry(self.seq_len, n, self.kv.page_size()),
        )?;
        geometry::attach_kv_write(&pass, &self.kv, &geom);
        let positions: Vec<u32> = (self.seq_len..self.seq_len + n).collect();
        pass.input_tokens(tokens, &positions);
        let decode_pos = self.seq_len + n - 1;
        pass.sampler(&g.program, g.bindings(decode_pos, packed_mask)?);
        pass.execute();
        self.seq_len += n;
        Ok(pass)
    }

    /// Prefill `tail`, then sequentially grammar-decode under `matcher` until it
    /// TERMINATES, a stop token fires, or `max_tokens` is hit. Returns the decoded
    /// text. This context is dropped after the parse, so no residual is preserved.
    async fn grammar_decode(
        &mut self,
        g: &sampler::LoweredGrammar,
        mut matcher: inferlet::GrammarConstraint,
        tail: &[u32],
        max_tokens: usize,
        stop: &[u32],
    ) -> Result<String> {
        let mut decoder = chat::Decoder::new();
        let mut text = String::new();
        let mut pending = tail.to_vec();
        if pending.is_empty() {
            pending = vec![0u32];
        }
        let mut generated = 0usize;

        loop {
            let m = matcher.mask();
            let packed: Vec<u32> = if m.is_empty() {
                vec![u32::MAX; g.mask_words]
            } else {
                m
            };

            let pass = self.grammar_fire(g, &pending, &packed)?;
            let token = read_token(pass).await?;

            if stop.contains(&token) {
                return Ok(text);
            }

            generated += 1;
            match decoder.feed(&[token])? {
                chat::Event::Delta(sd) => text.push_str(&sd),
                chat::Event::Done(sd) => return Ok(sd),
                _ => {}
            }
            matcher.advance(&[token]);
            pending = vec![token];

            if matcher.is_terminated() || generated >= max_tokens {
                return Ok(text);
            }
        }
    }
}

/// Read the sampled token off a finalized pass's single-`Token` output tensor.
async fn read_token(pass: ForwardPass) -> Result<u32> {
    let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("tensor read: {e:?}"))?;
    Ok(if bytes.len() >= 4 {
        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u32
    } else {
        0
    })
}

#[derive(Debug)]
enum Plan {
    Leaf(String),
    Branch(String, String),
}

/// Parse the model's plan-step JSON into a Plan.
fn parse_plan(json: &str) -> std::result::Result<Plan, String> {
    let v: Value = serde_json::from_str(&sanitize_json(json))
        .map_err(|e| format!("JSON parse: {e}"))?;
    let mode = v.get("mode").and_then(Value::as_str).ok_or("missing mode")?;
    match mode {
        "leaf" => {
            let answer = v.get("answer")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .ok_or("leaf with empty answer")?;
            Ok(Plan::Leaf(answer.to_string()))
        }
        "branch" => {
            let subtasks = v.get("subtasks")
                .and_then(Value::as_array)
                .ok_or("missing subtasks")?;
            if subtasks.len() != 2 {
                return Err(format!("branch needs 2 subtasks, got {}", subtasks.len()));
            }
            let t1 = subtasks[0].as_str().ok_or("subtask 1 not a string")?.trim();
            let t2 = subtasks[1].as_str().ok_or("subtask 2 not a string")?.trim();
            Ok(Plan::Branch(t1.to_string(), t2.to_string()))
        }
        m => Err(format!("unknown mode: {m}")),
    }
}

fn parse_answer(json: &str) -> std::result::Result<String, String> {
    let v: Value = serde_json::from_str(&sanitize_json(json))
        .map_err(|e| format!("JSON parse: {e}"))?;
    v.get("answer")
        .and_then(Value::as_str)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "missing answer".to_string())
}

/// Recursive divide-solve-merge. Branches run concurrently when not verbose.
#[allow(clippy::too_many_arguments)]
fn solve<'a>(
    plan_root: &'a Ctx,
    merge_root: &'a Ctx,
    g: &'a sampler::LoweredGrammar,
    stop: &'a [u32],
    question: &'a str,
    path: String,
    max_depth: usize,
    max_tokens: usize,
    verbose: bool,
) -> BoxFuture<'a, Result<String>> {
    Box::pin(async move {
        // Force a leaf at max depth.
        let force_leaf = path.len() >= max_depth;

        vlog!(verbose, "[{path}] {} {question}",
              if force_leaf { "leaf (max depth):" } else { "plan:" });

        let mut ctx = plan_root.fork()?;
        let prompt = if force_leaf {
            format!("Solve this directly. Set mode=\"leaf\" and put the answer in \
                     `answer`. Leave `subtasks` empty. Problem: {question}")
        } else {
            format!("Decide leaf vs branch for this problem. Problem: {question}")
        };
        let mut tail = chat::user(&prompt);
        tail.extend(chat::cue());

        let matcher = Ebnf(PLAN_GRAMMAR).build_constraint()?;
        let json = ctx
            .grammar_decode(g, matcher, &tail, max_tokens, stop)
            .await?;

        let plan = parse_plan(&json)
            .map_err(|e| format!("plan parse at {path:?}: {e} (raw: {json})"))?;

        match plan {
            Plan::Leaf(answer) => {
                vlog!(verbose, "[{path}] -> {answer}");
                Ok(answer)
            }
            Plan::Branch(t1, t2) => {
                vlog!(verbose, "[{path}] split:\n  L: {t1}\n  R: {t2}");

                let f1 = solve(plan_root, merge_root, g, stop, &t1,
                               format!("{path}l"), max_depth, max_tokens, verbose);
                let f2 = solve(plan_root, merge_root, g, stop, &t2,
                               format!("{path}r"), max_depth, max_tokens, verbose);
                let (a1, a2) = if verbose {
                    (f1.await?, f2.await?)
                } else {
                    let (r1, r2) = future::join(f1, f2).await;
                    (r1?, r2?)
                };

                let mut mctx = merge_root.fork()?;
                let mut mtail = chat::user(&format!(
                    "Original problem: {question}\n\
                     Subproblem 1 answer: {a1}\n\
                     Subproblem 2 answer: {a2}\n\
                     Compose the final answer to the original problem."
                ));
                mtail.extend(chat::cue());
                let mmatcher = Ebnf(MERGE_GRAMMAR).build_constraint()?;
                let merged = mctx
                    .grammar_decode(g, mmatcher, &mtail, max_tokens, stop)
                    .await?;
                let answer = parse_answer(&merged)
                    .map_err(|e| format!("merge parse at {path:?}: {e}"))?;
                vlog!(verbose, "[{path}] merge -> {answer}");
                Ok(answer)
            }
        }
    })
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let start = Instant::now();
    let vocab = model::output_vocab_size();
    let g = sampler::grammar_program(vocab)?;
    let stop = chat::stop_tokens();

    // One context per system prompt — the divide step and merge step want
    // different framing. Both share a clean fork point so each call gets
    // its own KV without leaking state across siblings.
    let mut plan_root = Ctx::new();
    plan_root.prefill(&chat::system(PLAN_SYSTEM))?;

    let mut merge_root = Ctx::new();
    merge_root.prefill(&chat::system(MERGE_SYSTEM))?;

    println!("--- Recursion-of-Thought (max_depth={}, max_tokens={}) ---",
             input.max_depth, input.max_tokens);
    println!("Question: {}", input.question);

    let answer = solve(
        &plan_root, &merge_root, &g, &stop,
        &input.question,
        String::new(),
        input.max_depth,
        input.max_tokens,
        input.verbose,
    ).await?;

    println!("\n--- ✅ RoT Complete in {:?} ---", start.elapsed());
    println!("Final solution: {answer}");
    Ok(String::new())
}
