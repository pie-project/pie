//! Demonstrates Tree-of-Thought (ToT) for multi-branch reasoning.
//!
//! This example performs a 3-level tree search (Propose, Execute, Reflect) where
//! each level spawns multiple branches. All branches are explored concurrently,
//! leveraging KV cache sharing from common prefixes.

use futures::future;
use inferlet::{
    Context, sample::Sampler, model::Model,
    runtime, Result,
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

    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?;
    let model = Model::load(model_name)?;

    let mut ctx_root = Context::new(&model)?;
    ctx_root.system(
        "You are a helpful, respectful, and honest assistant that excels at \
        mathematical reasoning. Please follow the user's instructions precisely.",
    );
    ctx_root.flush().await?;

    // Build and execute tree in parallel
    let level1_futures = (0..num_branches)
        .map(|_| {
            let question_ = question.clone();
            let model_name_1 = model_name.clone();
            Ok(async move {
                let sys_prompt = "You are a helpful, respectful, and honest assistant that excels at \
                    mathematical reasoning. Please follow the user\'s instructions precisely.";
                // Level 1: Propose Plan (no fork - fresh context)
                let model1 = Model::load(&model_name_1)?;
                let mut propose_ctx = Context::new(&model1)?;
                propose_ctx.system(sys_prompt);
                let propose_prompt = format!("{}{}", PROPOSE_PROMPT_TEMPLATE, question_);
                propose_ctx.user(&propose_prompt);
                propose_ctx.cue();

                let propose_text = propose_ctx
                    .generate(Sampler::TopP { temperature: 0.6, p: 0.95 })
                    .max_tokens(max_tokens_per_step)
                    .collect_text()
                    .await?;
                println!("\n[1] PROPOSE: {}", propose_text);

                let level2_futures = (0..num_branches)
                    .map(|_| {
                        let model_name_2 = model_name_1.clone();
                        let propose_prompt_ = propose_prompt.clone();
                        let propose_text_ = propose_text.clone();
                        Ok(async move {
                            // Level 2: Execute Plan (no fork - fresh context, replay as text)
                            let model2 = Model::load(&model_name_2)?;
                            let mut execute_ctx = Context::new(&model2)?;
                            execute_ctx.system(sys_prompt);
                            execute_ctx.user(&propose_prompt_);
                            execute_ctx.assistant(&propose_text_);
                            execute_ctx.user(EXECUTE_PROMPT);
                            execute_ctx.cue();

                            let execute_text = execute_ctx
                                .generate(Sampler::TopP { temperature: 0.6, p: 0.95 })
                                .max_tokens(max_tokens_per_step)
                                .collect_text()
                                .await?;
                            println!("\nEXECUTE: {}", execute_text);

                            let level3_futures = (0..num_branches)
                                .map(|_| {
                                    let model_name_3 = model_name_2.clone();
                                    let propose_prompt_3 = propose_prompt_.clone();
                                    let propose_text_3 = propose_text_.clone();
                                    let execute_text_3 = execute_text.clone();
                                    Ok(async move {
                                        // Level 3: Reflect (no fork - fresh context, replay as text)
                                        let model3 = Model::load(&model_name_3)?;
                                        let mut reflect_ctx = Context::new(&model3)?;
                                        reflect_ctx.system(sys_prompt);
                                        reflect_ctx.user(&propose_prompt_3);
                                        reflect_ctx.assistant(&propose_text_3);
                                        reflect_ctx.user(EXECUTE_PROMPT);
                                        reflect_ctx.assistant(&execute_text_3);
                                        reflect_ctx.user(REFLECT_PROMPT);
                                        reflect_ctx.cue();

                                        let reflect_text = reflect_ctx
                                            .generate(Sampler::TopP { temperature: 0.6, p: 0.95 })
                                            .max_tokens(max_tokens_per_step)
                                            .collect_text()
                                            .await?;
                                        println!("\nREFLECT: {}", reflect_text);
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
