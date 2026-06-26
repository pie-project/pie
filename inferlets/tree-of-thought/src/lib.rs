//! Demonstrates Tree-of-Thought (ToT) for multi-branch reasoning.
//!
//! This example performs a 3-level tree search (Propose, Execute, Reflect) where
//! each level spawns multiple branches. All branches are explored concurrently,
//! leveraging KV cache sharing from common prefixes.

use futures::future;
use inferlet::{
    Context, sample::Sampler,
    Result,
};
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_question")]
    question: String,
    #[serde(default = "default_num_branches")]
    num_branches: usize,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
}

fn default_question() -> String { "Calculate (42 + 3) * 5 / 15.".to_string() }
fn default_num_branches() -> usize { 2 }
fn default_max_tokens() -> usize { 512 }

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
async fn main(input: Input) -> Result<String> {
    let question = input.question;
    let num_branches = input.num_branches;
    let max_tokens_per_step = input.max_tokens;

    let total_leaves = num_branches.pow(3);
    println!(
        "--- Starting Tree of Thought (Branches={}, Leaves={}, MaxTokens/Step={}) ---",
        num_branches, total_leaves, max_tokens_per_step
    );
    let start = Instant::now();

    let mut ctx_root = Context::new()?;
    ctx_root.system(
        "You are a helpful, respectful, and honest assistant that excels at \
        mathematical reasoning. Please follow the user's instructions precisely.",
    );
    ctx_root.flush().await?;

    // Build and execute tree in parallel
    let level1_futures = (0..num_branches)
        .map(|_| {
            let mut propose_ctx = ctx_root.fork()?;
            let question_ = question.clone();
            Ok(async move {
                // Level 1: Propose Plan
                let propose_prompt = format!("{}{}", PROPOSE_PROMPT_TEMPLATE, question_);
                propose_ctx.user(&propose_prompt);
                propose_ctx.cue();

                propose_ctx
                    .generate(Sampler::TopP { temperature: 0.6, p: 0.95 })
                    .max_tokens(max_tokens_per_step)
                    .collect_text()
                    .await?;

                // Level 2: Execute Plan
                propose_ctx.user(EXECUTE_PROMPT);
                propose_ctx.flush().await?;

                let level2_futures = (0..num_branches)
                    .map(|_| {
                        let mut execute_ctx = propose_ctx.fork()?;
                        Ok(async move {
                            execute_ctx.cue();
                            execute_ctx
                                .generate(Sampler::TopP { temperature: 0.6, p: 0.95 })
                                .max_tokens(max_tokens_per_step)
                                .collect_text()
                                .await?;

                            // Level 3: Reflect on Solution
                            execute_ctx.user(REFLECT_PROMPT);
                            execute_ctx.flush().await?;

                            let level3_futures = (0..num_branches)
                                .map(|_| {
                                    let mut reflect_ctx = execute_ctx.fork()?;
                                    Ok(async move {
                                        reflect_ctx.cue();
                                        reflect_ctx
                                            .generate(Sampler::TopP { temperature: 0.6, p: 0.95 })
                                            .max_tokens(max_tokens_per_step)
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
