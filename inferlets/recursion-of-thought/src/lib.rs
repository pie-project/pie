//! Recursion-of-Thought (RoT): solve a problem by recursively dividing it
//! into independent subproblems whose answers compose into the final
//! answer.
//!
//! Pie features exercised:
//!   - `Context::fork()` — every recursive call branches off a shared
//!     prefix, so the system prompt + chat template are KV-cached once.
//!   - `Generator::constrain_with(Ebnf)` — the divide/merge steps emit a
//!     small JSON shape under a hand-rolled grammar that forbids the
//!     whitespace stalls JSON-Schema's auto-grammar permits.
//!   - `futures::future::join` over forked contexts — sibling subtasks
//!     run concurrently and the runtime batches their forward passes.

use futures::future;
use inferlet::{
    Context, Result, sample::Sampler,
};
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
fn solve<'a>(
    plan_root: &'a Context,
    merge_root: &'a Context,
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
        ctx.user(&prompt);
        ctx.cue();

        let json = ctx
            .generate(Sampler::Argmax)
            .max_tokens(max_tokens)
            .constrain_with(inferlet::Ebnf(PLAN_GRAMMAR))?
            .collect_text()
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

                let f1 = solve(plan_root, merge_root, &t1,
                               format!("{path}l"), max_depth, max_tokens, verbose);
                let f2 = solve(plan_root, merge_root, &t2,
                               format!("{path}r"), max_depth, max_tokens, verbose);
                let (a1, a2) = if verbose {
                    (f1.await?, f2.await?)
                } else {
                    let (r1, r2) = future::join(f1, f2).await;
                    (r1?, r2?)
                };

                let mut mctx = merge_root.fork()?;
                mctx.user(&format!(
                    "Original problem: {question}\n\
                     Subproblem 1 answer: {a1}\n\
                     Subproblem 2 answer: {a2}\n\
                     Compose the final answer to the original problem."
                ));
                mctx.cue();
                let merged = mctx
                    .generate(Sampler::Argmax)
                    .max_tokens(max_tokens)
                    .constrain_with(inferlet::Ebnf(MERGE_GRAMMAR))?
                    .collect_text()
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
    // One context per system prompt — the divide step and merge step want
    // different framing. Both share a clean fork point so each call gets
    // its own KV without leaking state across siblings.
    let mut plan_root = Context::new()?;
    plan_root.system(PLAN_SYSTEM);
    plan_root.flush().await?;

    let mut merge_root = Context::new()?;
    merge_root.system(MERGE_SYSTEM);
    merge_root.flush().await?;

    println!("--- Recursion-of-Thought (max_depth={}, max_tokens={}) ---",
             input.max_depth, input.max_tokens);
    println!("Question: {}", input.question);

    let answer = solve(
        &plan_root, &merge_root,
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
