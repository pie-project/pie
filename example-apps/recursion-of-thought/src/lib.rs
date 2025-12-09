//! Demonstrates Recursion-of-Thought (RoT) for divide-and-conquer problem solving.
//!
//! The model recursively decides whether to solve a problem directly (leaf node)
//! or divide it into two independent subtasks (branch node). Solutions from
//! subtasks are merged to produce the final answer.

use futures::future;
use inferlet::stop_condition::{self, StopCondition};
use inferlet::{Args, Context, Result, Sampler};

use std::future::Future;
use std::pin::Pin;
use std::time::Instant;

const HELP: &str = "\
Usage: rot [OPTIONS]

A program to solve a problem using a recursive divide-and-conquer strategy (Recursion-of-Thought).

Options:
  -q, --question <TEXT>   The initial question to solve [default: \"What is the sum of 123456789 and 987654321?\"]
  -d, --max-depth <NUM>   The maximum recursion depth [default: 5]
  -t, --max-tokens <NUM>  The max tokens to generate at each step [default: 128]
  -v, --verbose           Print intermediate messages during recursion
  -h, --help              Prints this help message";

/// Prints a message only if verbose mode is enabled.
macro_rules! verbose_println {
    ($verbose:expr, $($arg:tt)*) => {
        if $verbose {
            println!($($arg)*);
        }
    };
}

const DIVIDE_PROMPT_TEMPLATE: &str = "\
Your task is to analyze the given problem and decide whether it can be solved directly or needs \
to be divided into smaller subproblems. If the problem is simple and can be solved immediately, \
provide the solution wrapped in `<leaf>THE ANSWER</leaf>`. If not, divide the problem into \
exactly two independent subtasks such that solving these subtasks and combining their solutions \
will lead to the solution of the original problem. Present the subtasks wrapped in \
`<branch>SUBTASK 1</branch>` and `<branch>SUBTASK 2</branch>`. Be concise and ensure the \
subtasks are distinct and solvable. Please also ensure that the description of the subtasks is \
clear and self-contained, that is, each subtask should be able to be solved independently of the \
other. One subtask should not depend on the result of the other subtask. Problem: {}";

const SOLVE_PROMPT: &str =
    "Now, please solve the problem. Reason step-by-step. Make your response short.";

const MERGE_PROMPT: &str =
    "Now, please merge the two solutions into one. Make your response short.";

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

/// Parses the model's response to extract either a leaf answer or two branch subtasks.
fn parse_response(response: &str) -> Result<(Option<String>, Option<(String, String)>), String> {
    if let Some(start) = response.find("<leaf>") {
        if let Some(end) = response.find("</leaf>") {
            let answer = response[start + 6..end].trim().to_string();
            return Ok((Some(answer), None));
        }
    }

    let branches: Vec<String> = response
        .match_indices("<branch>")
        .zip(response.match_indices("</branch>"))
        .map(|((start, _), (end, _))| response[start + 8..end].trim().to_string())
        .collect();

    if branches.len() == 2 {
        Ok((None, Some((branches[0].clone(), branches[1].clone()))))
    } else {
        Err(format!(
            "Error: Expected a <leaf> tag or exactly two <branch> tags, but found {} branches.",
            branches.len()
        ))
    }
}

/// Recursively divides a problem, solves sub-problems, and merges the solutions.
fn divide_and_conquer<'a>(
    ctx: Context,
    question: &'a str,
    eos_tokens: &'a Vec<Vec<u32>>,
    path: String,
    max_depth: usize,
    max_tokens: usize,
    verbose: bool,
) -> BoxFuture<'a, String> {
    Box::pin(async move {
        // Base Case: If max depth is reached, solve the problem directly.
        if path.len() >= max_depth {
            let mut solve_ctx = ctx;
            let solve_prompt = format!("{} {}", SOLVE_PROMPT, question);
            solve_ctx.fill_user(&solve_prompt);

            let stop_condition = stop_condition::max_len(max_tokens)
                .or(stop_condition::ends_with_any(eos_tokens.clone()));
            let response = solve_ctx.generate(Sampler::greedy(), stop_condition).await;

            verbose_println!(verbose, "Reached max depth at path {:?}", path);
            verbose_println!(verbose, "Response: {}", response.trim());
            verbose_println!(verbose, "");
            return response;
        }

        // Recursive Step: Try to divide the problem.
        verbose_println!(verbose, "Analysing problem at path {:?}", path);
        let mut divide_ctx = ctx.fork();
        let divide_prompt = format!("{}", DIVIDE_PROMPT_TEMPLATE.replace("{}", question));
        divide_ctx.fill_user(&divide_prompt);
        let stop_condition = stop_condition::max_len(max_tokens)
            .or(stop_condition::ends_with_any(eos_tokens.clone()));
        let response = divide_ctx.generate(Sampler::greedy(), stop_condition).await;

        verbose_println!(verbose, "Response: {}", response.trim());

        match parse_response(&response) {
            // Case 1: The model provided a direct answer (leaf node).
            Ok((Some(answer), None)) => {
                verbose_println!(verbose, "Leaf node found at path {:?}", path);
                verbose_println!(verbose, "Response: {}", answer.trim());
                verbose_println!(verbose, "");
                answer
            }
            // Case 2: The model divided the problem into two subtasks (branch node).
            Ok((None, Some((task1, task2)))) => {
                verbose_println!(verbose, "Branch node found at path {:?}", path);
                verbose_println!(verbose, "Subtask 1: {}", task1.trim());
                verbose_println!(verbose, "Subtask 2: {}", task2.trim());
                verbose_println!(verbose, "");

                let solution1_future = divide_and_conquer(
                    ctx.fork(),
                    &task1,
                    eos_tokens,
                    format!("{}l", path),
                    max_depth,
                    max_tokens,
                    verbose,
                );
                let solution2_future = divide_and_conquer(
                    ctx.fork(),
                    &task2,
                    eos_tokens,
                    format!("{}r", path),
                    max_depth,
                    max_tokens,
                    verbose,
                );

                let solution1;
                let solution2;

                // If verbose mode is enabled, run the subtask solutions sequentially, so that
                // the output is not interleaved.
                if verbose {
                    solution1 = solution1_future.await;
                    solution2 = solution2_future.await;
                } else {
                    (solution1, solution2) = future::join(solution1_future, solution2_future).await;
                }

                verbose_println!(verbose, "Merging solutions at path {:?}", path);
                let mut merge_ctx = ctx;
                let merge_prompt = format!(
                    "Subtask 1 solution: {}\nSubtask 2 solution: {}\n{}",
                    solution1, solution2, MERGE_PROMPT
                );
                merge_ctx.fill_user(&merge_prompt);
                let stop_condition = stop_condition::max_len(max_tokens)
                    .or(stop_condition::ends_with_any(eos_tokens.clone()));
                let response = merge_ctx.generate(Sampler::greedy(), stop_condition).await;

                verbose_println!(verbose, "Response: {}", response.trim());
                verbose_println!(verbose, "");
                response
            }
            // Case 3: Error in parsing the response.
            Err(e) => format!("Parsing Error: {}", e),
            // Case 4: Invalid response format.
            _ => "Error: Invalid response format from model.".to_string(),
        }
    })
}

#[inferlet::main]
async fn main(mut args: Args) -> Result<()> {
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    let question: String = args
        .value_from_str(["-q", "--question"])
        .unwrap_or_else(|_| "Please calculate the expression (42 + 3) * 5 / 15.".to_string());

    let max_depth: usize = args.value_from_str(["-d", "--max-depth"]).unwrap_or(5);

    let max_tokens: usize = args.value_from_str(["-t", "--max-tokens"]).unwrap_or(128);

    let verbose: bool = args.contains(["-v", "--verbose"]);

    let start_time = Instant::now();
    println!("--- Initializing Model and Context ---");
    let model = inferlet::get_auto_model();
    let eos_tokens = model.eos_tokens();
    let mut ctx = model.create_context();

    ctx.fill_system("You are a helpful, respectful and honest assistant.");
    ctx.flush().await;

    println!("--- Starting Recursion-of-Thought (RoT) ---");
    println!("Question: {}", question);
    println!("Max Depth: {}, Max Tokens: {}", max_depth, max_tokens);

    let solution = divide_and_conquer(
        ctx,
        &question,
        &eos_tokens,
        "".to_string(),
        max_depth,
        max_tokens,
        verbose,
    )
    .await;

    println!("\n--- ✅ RoT Complete in {:?} ---", start_time.elapsed());
    println!("\nFinal solution: {}", solution);

    Ok(())
}
