//! Demonstrates ReAct-style (Reasoning + Acting) agent workflow.
//!
//! This example implements a ReAct agent that performs sequential
//! Thought/Action/Observation cycles with actual tool execution.

use chrono::{NaiveDate, Utc};
use evalexpr::eval;
use inferlet::stop_condition::{self, StopCondition};
use inferlet::{Args, Result, Sampler};

/// Result of parsing and executing an action from the assistant's response.
enum ActionResult {
    /// A tool was executed successfully (observation returned).
    Tool(String),
    /// A final answer was provided (observation returned).
    FinalAnswer(String),
    /// No valid action was found in the response.
    NotFound,
}

const HELP: &str = "\
Usage: agent-react [OPTIONS]

A benchmark for ReAct-style function calling scenarios.

Options:
  -f, --num-function-calls <N>    Number of sequential function calls
                                  (i.e. Thought/Action/Observation cycles) [default: 5]
  -t, --tokens-between-calls <N>  Max tokens for each Thought/Action step [default: 512]
  -h, --help                      Prints help information";

const SYSTEM_PROMPT: &str = "
You are a helpful assistant that understands how to break down a complex question into \
a series of steps. \
The following tools are available:

- `Calculator[expression]`: Evaluates a mathematical expression \
                            (e.g., \"15 * 30\", \"100 / 2.5\", \"5 + 3 * 2\").
- `CurrentDate[]`: Returns today's date in YYYY-MM-DD format.
- `DaysBetween[YYYY-MM-DD, YYYY-MM-DD]`: Calculates the number of days between \
                                         two dates (from first date to second date).
- `FinalAnswer[answer]`: Reports your final answer to the user's question.

Please respond with one tool use at a time, and don't nest tool calls.

The user's question might be complicated, so it may require multiple steps to answer. \
You will receive a history of the interactions with the tools so far. Use this history \
to reason about the next action to take. If you don't see a history, it means that the \
conversation has just started.

You need to answer next action to take, you must output your thoughts and the action \
to take. The format should be:

Thought: Your reasoning for the next action.
Action: The tool to use, in the format `ToolName[input]`.

When possible, please prefer using the tools that are available.

In the interaction history, you will see the results of the previous tool calls with \
the following format:

Observation: The result of the tool call.

As a reminder, you must respond only the next action to take and end the conversation, \
and use only one tool call at a time.";

const USER_PROMPT: &str = "\
If I save $12.50 per day starting today, how much money \
will I have saved by the end of the year 2030?";

#[inferlet::main]
async fn main(mut args: Args) -> Result<()> {
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    let num_function_calls: u32 = args
        .value_from_str(["-f", "--num-function-calls"])
        .unwrap_or(5);
    let tokens_between_calls: usize = args
        .value_from_str(["-t", "--tokens-between-calls"])
        .unwrap_or(512);

    let model = inferlet::get_auto_model();
    let eos_tokens = model.eos_tokens();
    let mut ctx = model.create_context();

    ctx.fill_system(SYSTEM_PROMPT);
    ctx.fill_user(&format!(
        "{USER_PROMPT} What is the next step to solve this problem?"
    ));

    let stop_condition =
        stop_condition::max_len(tokens_between_calls).or(stop_condition::ends_with_any(eos_tokens));

    let mut final_answer = None;

    for _ in 0..num_function_calls {
        let response = ctx
            .generate(Sampler::greedy(), stop_condition.clone())
            .await;

        // Parse and execute any action in the response
        let action_result = parse_and_execute_action(&response);

        let observation = match &action_result {
            ActionResult::Tool(obs) => obs.clone(),
            ActionResult::FinalAnswer(obs) => {
                final_answer = Some(obs.clone());
                break;
            }
            ActionResult::NotFound => {
                "No action detected. Please use the format: Action: ToolName[input]".to_string()
            }
        };

        ctx.fill_user(&format!(
            "Observation: {observation}\n What is the next step to solve this problem?"
        ));
    }

    println!("Full context: {}", ctx.get_text());

    if let Some(final_answer) = final_answer {
        println!("Final answer: {}", final_answer);
    } else {
        println!("No final answer found.");
    }

    Ok(())
}

/// Parses the assistant's response looking for an Action and executes it.
/// Scans backwards to find the last action, which is important for thinking models
/// that may generate multiple action patterns during their reasoning process.
fn parse_and_execute_action(text: &str) -> ActionResult {
    // Scan backwards through lines to find the last action
    for line in text.lines().rev() {
        let line = line.trim();
        if let Some(action_part) = line.strip_prefix("Action:") {
            let action_part = action_part.trim();

            // Try to parse Calculator[expression]
            if let Some(inner) = extract_tool_input(action_part, "Calculator") {
                return ActionResult::Tool(execute_calculator(&inner));
            }

            // Try to parse CurrentDate[]
            if let Some(inner) = extract_tool_input(action_part, "CurrentDate") {
                return ActionResult::Tool(execute_current_date(&inner));
            }

            // Try to parse DaysBetween[date_from, date_until]
            if let Some(inner) = extract_tool_input(action_part, "DaysBetween") {
                return ActionResult::Tool(execute_days_between(&inner));
            }

            // Try to parse FinalAnswer[answer]
            if let Some(inner) = extract_tool_input(action_part, "FinalAnswer") {
                return ActionResult::FinalAnswer(execute_final_answer(&inner));
            }
        }
    }

    ActionResult::NotFound
}

/// Extracts the input from "ToolName[input]" format.
fn extract_tool_input(text: &str, tool_name: &str) -> Option<String> {
    let text = text.trim();
    if text.contains(tool_name) {
        if let Some(start) = text.find('[') {
            if let Some(end) = text.rfind(']') {
                if start < end {
                    return Some(text[start + 1..end].to_string());
                }
            }
        }
    }
    None
}

/// Executes a calculator expression.
fn execute_calculator(expression: &str) -> String {
    match eval(expression) {
        Ok(result) => format!("The result is: {}", result),
        Err(e) => format!("Error evaluating expression: {}", e),
    }
}

/// Returns the current date.
fn execute_current_date(_input: &str) -> String {
    let today = Utc::now().date_naive();
    format!("Today's date is: {}", today.format("%Y-%m-%d"))
}

/// Calculates days between two dates.
fn execute_days_between(input: &str) -> String {
    // Parse input expecting "date_from, date_until"
    let parts: Vec<&str> = input.split(',').map(|s| s.trim()).collect();

    if parts.len() != 2 {
        return format!(
            "Error: DaysBetween requires exactly 2 dates separated by comma. \
            Expected format: DaysBetween[YYYY-MM-DD, YYYY-MM-DD]"
        );
    }

    let date_from_str = parts[0];
    let date_until_str = parts[1];

    let date_from = match NaiveDate::parse_from_str(date_from_str, "%Y-%m-%d") {
        Ok(date) => date,
        Err(e) => {
            return format!(
                "Error parsing start date '{}': {}. Expected format: YYYY-MM-DD",
                date_from_str, e
            );
        }
    };

    let date_until = match NaiveDate::parse_from_str(date_until_str, "%Y-%m-%d") {
        Ok(date) => date,
        Err(e) => {
            return format!(
                "Error parsing end date '{}': {}. Expected format: YYYY-MM-DD",
                date_until_str, e
            );
        }
    };

    let days = (date_until - date_from).num_days();

    if days > 0 {
        format!(
            "There are {} days from {} to {}",
            days, date_from_str, date_until_str
        )
    } else if days == 0 {
        format!("Both dates are the same: {}", date_from_str)
    } else {
        format!(
            "The date {} is {} days before {}",
            date_until_str, -days, date_from_str
        )
    }
}

/// Reports the final answer.
fn execute_final_answer(answer: &str) -> String {
    format!("Task completed. Final answer: {}", answer.trim())
}
