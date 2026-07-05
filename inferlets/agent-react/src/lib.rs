//! ReAct-style agent on top of Pie. Each step the model emits a
//! `{thought, tool, input}` JSON object under a JSON-Schema constraint;
//! the host runs the tool and feeds the observation back as the next
//! user turn.
//!
//! **Low-level ① rewrite (grammar, SEQUENTIAL).** Off the
//! `Context`/`Generator`/`Sampler`/`constrain_with` facade onto the keep-core
//! (`ptir-grammar-tranche-conversion-spec`): `chat::` fillers + an in-inferlet
//! `Ctx`, `JsonSchema(schema).build_constraint()` (host `Matcher`:
//! `advance`/`mask`/`is_terminated`), and `sampler::grammar_program_sampled` (the
//! masked SAMPLED grammar sampler — the shipped facade DROPPED the grammar mask,
//! so this now ENFORCES the schema while keeping the `Multinomial` diversity).
//!   - **Per-step grammar switching:** the schema starts as the full tool menu
//!     and is swapped to a `FinalAnswer`-only variant on the last iteration — the
//!     new constraint is a fresh `Matcher` built for that step's decode, so
//!     termination is guaranteed without post-hoc parsing.
//!
//! **Grammar decode is SEQUENTIAL** — the next mask depends on this token, so the
//! run-ahead carrier does NOT apply; each step is a per-fire `grammar_fire`.

use chrono::{NaiveDate, Utc};
use evalexpr::eval;
use inferlet::inference::ForwardPass;
use inferlet::working_set::KvWorkingSet;
use inferlet::{chat, geometry, model, sampler, Constrain, JsonSchema, Result, Schema};
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_max_steps")]
    max_steps: u32,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_question")]
    question: String,
}

fn default_max_steps() -> u32 { 5 }
fn default_max_tokens() -> usize { 256 }
fn default_question() -> String {
    "What is 17 multiplied by 24, plus 13?".to_string()
}

const SYSTEM_PROMPT: &str = "\
You are a tool-using assistant. At every step output one JSON object \
{thought, tool, input} that picks a single tool call:

  - Calculator   input is a single arithmetic expression with operators \
                 +, -, *, /, e.g. \"17 * 24\", \"408 + 13\".
  - CurrentDate  input is ignored. Returns today's date YYYY-MM-DD.
  - DaysBetween  input is two ISO dates joined by ONE comma, e.g. \
                 \"2026-05-02,2030-12-31\".
  - FinalAnswer  input is the final answer to the user.

As soon as the most recent Observation IS the value the user is asking \
for, your next action MUST be FinalAnswer with that observation as \
`input`. Do not run extra calculations once the answer has appeared.

Worked example:

  User: Question: What is 5 + 3 - 1?
  Assistant: {\"thought\":\"Compute 5 + 3.\",\"tool\":\"Calculator\",\"input\":\"5 + 3\"}
  User: Observation: 8
  Assistant: {\"thought\":\"Subtract 1.\",\"tool\":\"Calculator\",\"input\":\"8 - 1\"}
  User: Observation: 7
  Assistant: {\"thought\":\"7 is the answer.\",\"tool\":\"FinalAnswer\",\"input\":\"7\"}";

const ACTION_SCHEMA: &str = r#"{
    "type": "object",
    "properties": {
        "thought": { "type": "string", "minLength": 1 },
        "tool":    {
            "type": "string",
            "enum": ["Calculator", "CurrentDate", "DaysBetween", "FinalAnswer"]
        },
        "input":   { "type": "string" }
    },
    "required": ["thought", "tool", "input"],
    "additionalProperties": false
}"#;

/// Same shape as `ACTION_SCHEMA` but with `tool` locked to FinalAnswer.
/// We swap to this on the final iteration so the loop is guaranteed to
/// terminate even when the model would otherwise keep proposing more
/// Calculator calls. Demonstrates Pie's per-decode grammar switching.
const FINAL_SCHEMA: &str = r#"{
    "type": "object",
    "properties": {
        "thought": { "type": "string", "minLength": 1 },
        "tool":    { "type": "string", "const": "FinalAnswer" },
        "input":   { "type": "string", "minLength": 1 }
    },
    "required": ["thought", "tool", "input"],
    "additionalProperties": false
}"#;

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let vocab = model::output_vocab_size();
    let g = sampler::grammar_program_sampled(vocab)?;
    let stop = chat::stop_tokens();

    let mut ctx = Ctx::new();
    ctx.system(SYSTEM_PROMPT);
    ctx.user(&format!("Question: {q}\nWhat is the next action?", q = input.question));
    ctx.cue();

    let mut final_answer: Option<String> = None;

    for step in 1..=input.max_steps {
        // Last iteration: swap the schema to force a FinalAnswer turn.
        let schema = if step == input.max_steps { FINAL_SCHEMA } else { ACTION_SCHEMA };

        let raw = ctx
            .generate_step(&g, schema, input.max_tokens, &stop)
            .await
            .map_err(|e| format!("step {step}: generate: {e}"))?;

        // The grammar enforces a structurally valid prefix at every
        // step. Parse can still fail when `max_tokens` cuts mid-string
        // — bias on real models points away from open-ended thought
        // text, but a backend with no convergence pressure (e.g. the
        // dummy driver) can run the `thought` field past the budget.
        // Treat that as "step skipped" instead of aborting the loop.
        let action = match serde_json::from_str::<Value>(&raw) {
            Ok(v) => v,
            Err(e) => {
                println!(
                    "\n[step {step}] grammar truncated at max_tokens \
                     ({e}). Skipping to next step."
                );
                ctx.user(&format!(
                    "Observation: (no action — generator hit max_tokens \
                     mid-grammar)\nQuestion (reminder): {q}\nWhat is \
                     the next action?",
                    q = input.question
                ));
                ctx.cue();
                continue;
            }
        };
        let thought = action.get("thought").and_then(Value::as_str).unwrap_or("");
        let tool = action.get("tool").and_then(Value::as_str).unwrap_or("");
        let arg = action.get("input").and_then(Value::as_str).unwrap_or("");

        println!("\n[step {step}] thought: {thought}");
        println!("[step {step}] {tool}({arg:?})");

        let observation = match tool {
            "Calculator"  => calculator(arg),
            "CurrentDate" => current_date(),
            "DaysBetween" => days_between(arg),
            "FinalAnswer" => {
                final_answer = Some(arg.trim().to_string());
                break;
            }
            other => format!("Error: unknown tool {other}"),
        };

        println!("[step {step}] observation: {observation}");
        ctx.user(&format!(
            "Observation: {observation}\nQuestion (reminder): {q}\nWhat is \
             the next action?",
            q = input.question
        ));
        ctx.cue();
    }

    match final_answer {
        Some(a) => println!("\nFinal answer: {a}"),
        None => println!("\nNo final answer found within the iteration limit."),
    }
    Ok(String::new())
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

/// In-inferlet decode context (raw-WIT keep-core, no `Context` facade): its own
/// KV working set + sequence cursor + a `buffer` of chat-template tokens not yet
/// prefilled. Chat templating is the kept thin `chat::` bindings.
struct Ctx {
    kv: KvWorkingSet,
    seq_len: u32,
    fresh: bool,
    buffer: Vec<u32>,
    pending_system: Option<String>,
}

impl Ctx {
    fn new() -> Self {
        Self {
            kv: KvWorkingSet::new(),
            seq_len: 0,
            fresh: true,
            buffer: Vec::new(),
            pending_system: None,
        }
    }

    fn flush_pending_system(&mut self) {
        if let Some(system) = self.pending_system.take() {
            self.buffer.extend(chat::system(&system));
        }
    }

    fn is_first_chat_fill(&self) -> bool {
        self.seq_len == 0 && self.buffer.is_empty()
    }

    fn system(&mut self, message: &str) {
        self.flush_pending_system();
        self.pending_system = Some(message.to_string());
    }

    fn user(&mut self, message: &str) {
        let tokens = match self.pending_system.take() {
            Some(system) => chat::system_user(&system, message),
            None if self.is_first_chat_fill() => chat::first_user(message),
            None => chat::user(message),
        };
        self.buffer.extend(tokens);
    }

    fn cue(&mut self) {
        self.flush_pending_system();
        self.buffer.extend(chat::cue());
    }

    /// One SEQUENTIAL grammar fire: geometry + input + the masked SAMPLED grammar
    /// sampler + execute, advancing the cursor. No run-ahead carrier (the next
    /// mask depends on this token). Fires `fresh_generate` once per session.
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

    /// Generate one grammar-constrained step: prefill the buffered prompt +
    /// sequentially decode under a fresh `JsonSchema` matcher until it TERMINATES
    /// (JSON object complete), a stop token fires, or `max_tokens` is hit. Returns
    /// the decoded text. The final content token is parked as a residual in
    /// `buffer` so the next turn's prefill materializes it.
    async fn generate_step(
        &mut self,
        g: &sampler::LoweredGrammar,
        schema: &str,
        max_tokens: usize,
        stop: &[u32],
    ) -> Result<String> {
        let mut matcher = JsonSchema(schema).build_constraint()?;
        let mut decoder = chat::Decoder::new();
        let mut text = String::new();

        self.flush_pending_system();
        let mut pending = std::mem::take(&mut self.buffer);
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
                chat::Event::Done(sd) => {
                    text = sd;
                    self.buffer.push(token);
                    return Ok(text);
                }
                _ => {}
            }
            matcher.advance(&[token]);
            pending = vec![token];

            if matcher.is_terminated() || generated >= max_tokens {
                self.buffer.push(token);
                return Ok(text);
            }
        }
    }
}

fn calculator(expr: &str) -> String {
    match eval(expr) {
        Ok(v) => format!("{v}"),
        Err(e) => format!("Error evaluating {expr:?}: {e}"),
    }
}

fn current_date() -> String {
    Utc::now().date_naive().format("%Y-%m-%d").to_string()
}

fn days_between(arg: &str) -> String {
    let parts: Vec<&str> = arg.split(',').map(str::trim).collect();
    if parts.len() != 2 {
        return format!("Error: DaysBetween needs YYYY-MM-DD,YYYY-MM-DD (got {arg:?})");
    }
    let parse = |s: &str| NaiveDate::parse_from_str(s, "%Y-%m-%d");
    match (parse(parts[0]), parse(parts[1])) {
        (Ok(a), Ok(b)) => (b - a).num_days().to_string(),
        (Err(e), _) | (_, Err(e)) => format!("Error parsing date: {e}"),
    }
}
