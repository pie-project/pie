//! CodeACT agent: solves problems by generating and executing JavaScript.
//!
//! The agent loops {generate → run code → feed result back} until the
//! model emits a `Final Answer:` line.

use inferlet::{sample::Sampler, model::Model, runtime, Context, Result};
use serde::Deserialize;

mod js;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_steps")]
    num_function_calls: u32,
    #[serde(default = "default_tokens")]
    tokens_between_calls: usize,
}

fn default_steps() -> u32 { 5 }
fn default_tokens() -> usize { 512 }

const SYSTEM_PROMPT: &str = "\
You are CodeACT, a highly intelligent AI assistant that solves problems by writing \
and executing JavaScript code step by step.

## Interaction Format

You will be given a task to solve, and you need to respond with the code that carries out \
the next step to solve the task. You may also receive a history of previous steps and their \
execution results reported by the user.

If you receive a history of previous steps and their execution results, it will be \
formatted as follows:
Code execution result: [Execution result here]

If you don't receive a history of previous steps and their execution results, it means that \
the conversation has just started. You must generate the code for the first step to solve \
the task.

You must generate the code for the NEXT STEP ONLY. Do not repeat previous steps or generate \
multiple code blocks at once. Respond with the following format:

Thought: Your reasoning about what to do next based on the history.
```javascript
// JavaScript code for this step only
```

When you have the final answer and no more code needs to be executed, respond with:

Thought: I have the answer.
Final Answer: [Your final answer here]

Important Notes:

- Each code execution is stateless - you cannot reference variables from previous executions.
- If you need helper functions, you must redefine them in each code block.
- The last expression in your code block will be returned as the result.
- Keep each code block focused on a single step of your solution.

Reminder: You must respond with the code for the NEXT STEP ONLY. Do not repeat previous \
steps or generate multiple code blocks at once.";

const TASK: &str = "Calculate the sum of the first 10 prime numbers.";

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let model_name = runtime::models().first().cloned().ok_or("No models available")?;
    let model = Model::load(&model_name)?;

    let mut ctx = Context::new(&model)?;
    ctx.system(SYSTEM_PROMPT);
    ctx.user(&format!("{TASK}\n\nWhat is the first step?"));
    ctx.cue();

    for _ in 0..input.num_function_calls {
        let response = ctx
            .generate(Sampler::Argmax)
            .max_tokens(input.tokens_between_calls)
            .collect_text()
            .await?;

        match js::try_eval_block(&response) {
            Some(result) => {
                ctx.user(&format!(
                    "Code execution result: {result}\n\nWhat is the next step?"
                ));
                ctx.cue();
            }
            None => {
                println!("Final answer: {}", final_answer(&response));
                return Ok(String::new());
            }
        }
    }

    println!("No final answer found within the iteration limit.");
    Ok(String::new())
}

/// Extracts the last `Final Answer:` line, falling back to the last
/// non-empty line of the response.
fn final_answer(text: &str) -> String {
    text.lines()
        .rev()
        .find_map(|l| l.trim().strip_prefix("Final Answer:").map(str::trim))
        .or_else(|| text.lines().rev().map(str::trim).find(|l| !l.is_empty()))
        .unwrap_or("Unknown")
        .to_string()
}
