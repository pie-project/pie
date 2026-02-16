//! Demonstrates CodeACT-style agentic workflow with code execution.
//!
//! This example implements a CodeACT agent that generates and executes
//! JavaScript code to solve problems, using the Boa engine for execution.
//! The agent iteratively writes code, receives execution results, and
//! continues until it arrives at the final answer.

use boa_engine::{Context as JsContext, Source};
use inferlet::{
    context::Context, inference::Sampler, model::Model,
    runtime, ContextExt, InstructExt, Result,
};

/// Result of parsing the assistant's response.
enum CodeResult {
    /// JavaScript code was found and executed (result returned).
    Code(String),
    /// No code block was found, indicating the model is providing a final answer.
    FinalAnswer,
}

const HELP: &str = "\
Usage: agent-codeact [OPTIONS]

An example of CodeACT-style agentic workflows with JavaScript code execution.

Options:
  -f, --num-function-calls <N>    Number of sequential code execution cycles [default: 5]
  -t, --tokens-between-calls <N>  Max tokens for each generation step [default: 512]
  -h, --help                      Prints help information";

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

const USER_PROMPT: &str = "Calculate the sum of the first 10 prime numbers.";

#[inferlet::main]
async fn main(args: Vec<String>) -> Result<String> {
    let mut args = inferlet::parse_args(args);

    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(String::new());
    }

    let num_function_calls: u32 = args
        .value_from_str(["-f", "--num-function-calls"])
        .unwrap_or(5);
    let tokens_between_calls: usize = args
        .value_from_str(["-t", "--tokens-between-calls"])
        .unwrap_or(512);

    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?;
    let model = Model::load(model_name)?;

    let ctx = Context::new(&model)?;
    ctx.system(SYSTEM_PROMPT);
    ctx.user(&format!("{}\n\n{}", USER_PROMPT, "What is the first step?"));
    ctx.cue();

    let mut final_answer = None;

    for _ in 0..num_function_calls {
        let response = ctx
            .generate(Sampler::TopP((0.0, 1.0)))
            .with_max_tokens(tokens_between_calls)
            .collect_text()
            .await?;

        // Parse and execute any JavaScript code in the response
        let code_result = parse_and_execute_code(&response);

        match code_result {
            CodeResult::Code(observation) => {
                ctx.user(&format!(
                    "Code execution result: {observation}\n\nWhat is the next step?"
                ));
                ctx.cue();
            }
            CodeResult::FinalAnswer => {
                final_answer = Some(extract_final_answer(&response));
                break;
            }
        }
    }

    if let Some(answer) = final_answer {
        println!("Final answer: {}", answer);
    } else {
        println!("No final answer found within the iteration limit.");
    }

    Ok(String::new())
}

/// Parses the assistant's response for JavaScript code blocks and executes them.
fn parse_and_execute_code(text: &str) -> CodeResult {
    if let Some(js_code) = extract_js_code(text) {
        let result = execute_js_code(&js_code);
        CodeResult::Code(result)
    } else {
        CodeResult::FinalAnswer
    }
}

/// Extracts JavaScript code from a markdown block.
/// Scans backwards to find the last code block, which is important for thinking models
/// that may generate multiple code patterns during their reasoning process.
fn extract_js_code(text: &str) -> Option<String> {
    let start_marker = "```javascript";
    let end_marker = "```";
    // Search backwards to find the last code block
    if let Some(start) = text.rfind(start_marker) {
        let code_start = start + start_marker.len();
        if let Some(end) = text[code_start..].find(end_marker) {
            let code = &text[code_start..code_start + end];
            return Some(code.trim().to_string());
        }
    }
    None
}

/// Extracts the final answer from the response text.
/// Scans backwards to find the last "Final Answer:" pattern, which is important for thinking
/// models that may generate multiple answer patterns during their reasoning process.
fn extract_final_answer(text: &str) -> String {
    // Scan backwards through lines to find the last "Final Answer:" pattern
    for line in text.lines().rev() {
        let line = line.trim();
        if let Some(answer) = line.strip_prefix("Final Answer:") {
            return answer.trim().to_string();
        }
    }
    // If no explicit final answer marker, return the last non-empty line
    text.lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("Unknown")
        .trim()
        .to_string()
}

/// Executes the given JavaScript code using the Boa engine.
fn execute_js_code(code: &str) -> String {
    let mut context = JsContext::default();
    match context.eval(Source::from_bytes(code)) {
        Ok(res) => res
            .to_string(&mut context)
            .unwrap_or_else(|_| "undefined".into())
            .to_std_string()
            .unwrap(),
        Err(e) => format!("Execution Error: {}", e),
    }
}
