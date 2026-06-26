//! CodeACT agent on top of Pie. Each step the model emits a
//! `{thought, kind, code, answer}` JSON object under a JSON-Schema
//! constraint; the host either runs `code` (Boa-evaluated JavaScript) and
//! feeds the value back as the next observation, or accepts `answer` as
//! the terminal output.
//!
//! Pie features exercised:
//!   - `Context` chat-template helpers (`system`, `user`, `cue`).
//!   - `Generator::constrain_with(JsonSchema)` for guaranteed-shape output.
//!   - `Sampler::Multinomial` — Argmax tends to lock identical contexts
//!     into the same continuation every step, so a small temperature is
//!     standard for agent loops even with grammar constraints.

use inferlet::{sample::Sampler, Context, Result};
use serde::Deserialize;
use serde_json::Value;

mod js;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_max_steps")]
    max_steps: u32,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_task")]
    task: String,
}

fn default_max_steps() -> u32 { 5 }
fn default_max_tokens() -> usize { 384 }
fn default_task() -> String {
    "Compute 25 squared, then add 17 to it.".to_string()
}

const SYSTEM_PROMPT: &str = "\
You are CodeACT, a problem-solver that thinks one short JavaScript step at \
a time. Each turn output one JSON object {thought, kind, code, answer}:

  - kind=\"code\":  put a JS snippet in `code` and leave `answer` empty. \
                  The host evaluates the snippet; the LAST expression's \
                  value comes back as the next Observation. Each \
                  evaluation is stateless — re-define helpers each turn.
  - kind=\"final\": you are done. Put the final answer in `answer` (the \
                  value from the latest Observation) and leave `code` empty.

Worked example:

  User: Task: Compute (12 * 7) + 5.
  Assistant: {\"thought\":\"Multiply.\",\"kind\":\"code\",\"code\":\"12 * 7\",\"answer\":\"\"}
  User: Observation: 84
  Assistant: {\"thought\":\"Add 5.\",\"kind\":\"code\",\"code\":\"84 + 5\",\"answer\":\"\"}
  User: Observation: 89
  Assistant: {\"thought\":\"Done.\",\"kind\":\"final\",\"code\":\"\",\"answer\":\"89\"}

Each `code` block is a single self-contained JS expression or short \
statement list — do not fence with backticks, just put the source in \
the `code` field.";

const STEP_SCHEMA: &str = r#"{
    "type": "object",
    "properties": {
        "thought": { "type": "string", "minLength": 1 },
        "kind":    { "type": "string", "enum": ["code", "final"] },
        "code":    { "type": "string" },
        "answer":  { "type": "string" }
    },
    "required": ["thought", "kind", "code", "answer"],
    "additionalProperties": false
}"#;

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let mut ctx = Context::new()?;
    ctx.system(SYSTEM_PROMPT);
    ctx.user(&format!("Task: {t}\nWhat is the next step?", t = input.task));
    ctx.cue();

    let mut final_answer: Option<String> = None;

    for step in 1..=input.max_steps {
        let raw = ctx
            .generate(Sampler::Multinomial { temperature: 0.7, draws: 0 })
            .max_tokens(input.max_tokens)
            .constrain_with(inferlet::JsonSchema(STEP_SCHEMA))?
            .collect_text()
            .await?;

        let v = serde_json::from_str::<Value>(&raw)
            .map_err(|e| format!("step {step}: JSON parse: {e} (raw: {raw})"))?;
        let thought = v.get("thought").and_then(Value::as_str).unwrap_or("");
        let kind = v.get("kind").and_then(Value::as_str).unwrap_or("");
        let code = v.get("code").and_then(Value::as_str).unwrap_or("");
        let answer = v.get("answer").and_then(Value::as_str).unwrap_or("");

        println!("\n[step {step}] thought: {thought}");

        match kind {
            "code" => {
                println!("[step {step}] code: {code}");
                let observation = js::eval(code);
                println!("[step {step}] observation: {observation}");
                ctx.user(&format!(
                    "Observation: {observation}\nTask (reminder): {t}\nWhat is \
                     the next step?",
                    t = input.task
                ));
                ctx.cue();
            }
            "final" => {
                let answer = answer.trim().to_string();
                println!("[step {step}] final: {answer}");
                final_answer = Some(answer);
                break;
            }
            _ => unreachable!("schema enforces kind ∈ {{code, final}}"),
        }
    }

    match final_answer {
        Some(a) => println!("\nFinal answer: {a}"),
        None => println!("\nNo final answer found within the iteration limit."),
    }
    Ok(String::new())
}
