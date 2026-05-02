//! ReAct-style agent on top of Pie. Each step the model emits a
//! `{thought, tool, input}` JSON object under a JSON-Schema constraint;
//! the host runs the tool and feeds the observation back as the next
//! user turn.
//!
//! Pie features exercised:
//!   - `Context` chat-template helpers (`system`, `user`, `cue`).
//!   - `Generator::constrain_with(JsonSchema)` for guaranteed-shape output.
//!   - **Per-step grammar switching:** the schema starts as the full tool
//!     menu and is swapped to a `FinalAnswer`-only variant on the last
//!     iteration. Pie applies the new constraint to the very next decode
//!     so termination is guaranteed without any post-hoc parsing or
//!     output rewriting.
//!   - `Sampler::Multinomial` — small temperature breaks Argmax loops on
//!     identical contexts.

use chrono::{NaiveDate, Utc};
use evalexpr::eval;
use inferlet::{
    Context, Result, sample::Sampler, model::Model, runtime,
};
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
    let model_name = runtime::models()
        .first()
        .cloned()
        .ok_or("No models available")?;
    let model = Model::load(&model_name)?;

    let mut ctx = Context::new(&model)?;
    ctx.system(SYSTEM_PROMPT);
    ctx.user(&format!("Question: {q}\nWhat is the next action?", q = input.question));
    ctx.cue();

    let mut final_answer: Option<String> = None;

    for step in 1..=input.max_steps {
        // Last iteration: swap the schema to force a FinalAnswer turn.
        let schema = if step == input.max_steps { FINAL_SCHEMA } else { ACTION_SCHEMA };

        let raw = ctx
            .generate(Sampler::Multinomial { temperature: 0.7, draws: 0 })
            .max_tokens(input.max_tokens)
            .constrain_with(inferlet::JsonSchema(schema))?
            .collect_text()
            .await?;

        let action = serde_json::from_str::<Value>(&raw)
            .map_err(|e| format!("step {step}: JSON parse: {e} (raw: {raw})"))?;
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
