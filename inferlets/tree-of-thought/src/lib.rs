//! Demonstrates Tree-of-Thought (ToT) for multi-branch reasoning.
//!
//! This example performs a 3-level tree search (Propose, Execute, Reflect) where
//! each level spawns multiple branches. All branches are explored concurrently,
//! leveraging KV cache sharing from common prefixes.

use futures::future;
use inferlet::{
    context::Context, inference::Sampler, model::Model,
    runtime, ContextExt, InstructExt, Result,
};
use std::time::Instant;

const HELP: &str = "\
Usage: tree_of_thought [OPTIONS]

A program to perform a 3-level tree of thought (Propose, Execute, Reflect) search.

Options:
  -q, --question <TEXT>        The question to solve [default: Calculate (42 + 3) * 5 / 15.]
  -b, --num-branches <INT>     Number of branches at each level of the tree [default: 2]
  -t, --max-tokens <INT>       Max new tokens to generate at each step [default: 512]
  -h, --help                   Prints this help message";

const PROPOSE_PROMPT_TEMPLATE: &str = "\
Please generate a high-level plan for solving the following question. \
First, just state the method you will use. Do not do the actual calculation. \
Keep your response concise and within 80 words. Question: ";

const EXECUTE_PROMPT: &str = "\
The plan looks good! Now, use real numbers and do the calculation. \
Please solve the question step-by-step according to the plan. \
Give me the final answer. Make your response short.";

const REFLECT_PROMPT: &str = "\
Okay. Now, evaluate your own solution and give it a score on a scale of 1 to 5. \
Please rigorously check the correctness of the calculations and the final answer.";


#[inferlet::main]
async fn main(args: Vec<String>) -> Result<String> {
    let mut args = inferlet::parse_args(args);

    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(String::new());
    }

    let question: String = args
        .value_from_str(["-q", "--question"])
        .unwrap_or_else(|_| "Calculate (42 + 3) * 5 / 15.".to_string());

    let num_branches: usize = args.value_from_str(["-b", "--num-branches"]).unwrap_or(2);
    let max_tokens_per_step: usize = args.value_from_str(["-t", "--max-tokens"]).unwrap_or(512);

    let total_leaves = num_branches.pow(3);
    println!(
        "--- Starting Tree of Thought (Branches={}, Leaves={}, MaxTokens/Step={}) ---",
        num_branches, total_leaves, max_tokens_per_step
    );
    let start = Instant::now();

    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?;
    let model = Model::load(model_name)?;

    let ctx_root = Context::new(&model)?;
    ctx_root.system(
        "You are a helpful, respectful, and honest assistant that excels at \
        mathematical reasoning. Please follow the user's instructions precisely.",
    );
    ctx_root.flush().await?;

    // Build and execute tree in parallel
    let level1_futures = (0..num_branches)
        .map(|_| {
            let propose_ctx = ctx_root.fork()?;
            let question_ = question.clone();
            Ok(async move {
                // Level 1: Propose Plan
                let propose_prompt = format!("{}{}", PROPOSE_PROMPT_TEMPLATE, question_);
                propose_ctx.user(&propose_prompt);
                propose_ctx.cue();

                propose_ctx
                    .generate(Sampler::TopP((0.6, 0.95)))
                    .with_max_tokens(max_tokens_per_step)
                    .collect_text()
                    .await?;

                // Level 2: Execute Plan
                propose_ctx.user(EXECUTE_PROMPT);
                propose_ctx.flush().await?;

                let level2_futures = (0..num_branches)
                    .map(|_| {
                        let execute_ctx = propose_ctx.fork()?;
                        Ok(async move {
                            execute_ctx.cue();
                            execute_ctx
                                .generate(Sampler::TopP((0.6, 0.95)))
                                .with_max_tokens(max_tokens_per_step)
                                .collect_text()
                                .await?;

                            // Level 3: Reflect on Solution
                            execute_ctx.user(REFLECT_PROMPT);
                            execute_ctx.flush().await?;

                            let level3_futures = (0..num_branches)
                                .map(|_| {
                                    let reflect_ctx = execute_ctx.fork()?;
                                    Ok(async move {
                                        reflect_ctx.cue();
                                        reflect_ctx
                                            .generate(Sampler::TopP((0.6, 0.95)))
                                            .with_max_tokens(max_tokens_per_step)
                                            .collect_text()
                                            .await?;
                                        Ok::<_, String>(())
                                    })
                                })
                                .collect::<Result<Vec<_>>>()?;
                            let results = future::join_all(level3_futures).await;
                            for r in results {
                                r?;
                            }
                            Ok::<_, String>(())
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let results = future::join_all(level2_futures).await;
                for r in results {
                    r?;
                }
                Ok::<_, String>(())
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let results = future::join_all(level1_futures).await;
    for r in results {
        r?;
    }

    println!(
        "\n--- All leaf nodes generated in {:?} ---",
        start.elapsed()
    );

    Ok(String::new())
}
