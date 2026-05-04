//! Demonstrates Skeleton-of-Thought (SoT) for parallel elaboration.
//!
//! This example first generates a high-level plan (skeleton) with key points,
//! then elaborates on each point concurrently. This approach can reduce latency
//! by parallelizing the detailed generation phase.

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
    #[serde(default = "default_num_points")]
    num_points: usize,
    #[serde(default = "default_plan_tokens")]
    plan_tokens: usize,
    #[serde(default = "default_elab_tokens")]
    elab_tokens: usize,
}

fn default_question() -> String { "What are the defining characteristics of Rome?".to_string() }
fn default_num_points() -> usize { 3 }
fn default_plan_tokens() -> usize { 256 }
fn default_elab_tokens() -> usize { 256 }


/// Generates a high-level plan and elaborates on each point in parallel.
async fn plan_and_generate_parallel(
    ctx: &mut Context,
    question: &str,
    max_points: usize,
    plan_max_tokens: usize,
    elab_max_tokens: usize,
) -> Result<Vec<String>> {
    // 1. Fork a context for generating the plan.
    let mut plan_ctx = ctx.fork()?;
    let plan_prompt = format!(
        "Generate up to {} key points that outline the answer to the following question: {}. \
        Each point must be enclosed between the <point> and </point> tags.",
        max_points, question
    );
    plan_ctx.user(&plan_prompt);
    plan_ctx.cue();

    let output = plan_ctx
        .generate(Sampler::TopP { temperature: 0.6, p: 0.95 })
        .max_tokens(plan_max_tokens)
        .collect_text()
        .await?;

    // 2. Robustly parse points from the output.
    let points: Vec<String> = output
        .split("<point>")
        .skip(1)
        .filter_map(|s| s.split("</point>").next())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    if points.is_empty() {
        return Ok(Vec::new());
    }

    // 3. Fork from the original base context for a clean state for each elaboration.
    let leaf_futures = points
        .into_iter()
        .map(|point| {
            let mut elab_ctx = ctx.fork()?;
            let complete_prompt = format!(
                "Elaborate on the following point: {}. \
                Your response should be complete and only concerned with this point.",
                point
            );
            elab_ctx.user(&complete_prompt);
            elab_ctx.cue();

            Ok(async move {
                elab_ctx
                    .generate(Sampler::TopP { temperature: 0.6, p: 0.95 })
                    .max_tokens(elab_max_tokens)
                    .collect_text()
                    .await
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let results = future::join_all(leaf_futures).await;
    results.into_iter().collect::<Result<Vec<_>>>()
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let question = input.question;
    let num_points = input.num_points;
    let plan_max_tokens = input.plan_tokens;
    let elab_max_tokens = input.elab_tokens;

    let start = Instant::now();

    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?;
    let model = Model::load(model_name)?;

    let mut ctx = Context::new(&model)?;
    ctx.system("You are a helpful, respectful and honest assistant.");
    ctx.flush().await?;

    println!(
        "--- Starting plan and generate (plan: {} points, {} tokens; elab: {} tokens) ---",
        num_points, plan_max_tokens, elab_max_tokens
    );

    let elaborations = plan_and_generate_parallel(
        &mut ctx,
        &question,
        num_points,
        plan_max_tokens,
        elab_max_tokens,
    )
    .await?;

    println!("\n--- Completed in {:?} ---\n", start.elapsed());

    if elaborations.is_empty() {
        println!("No points were generated or elaborated upon.");
    } else {
        for (i, elaboration) in elaborations.iter().enumerate() {
            println!("Elaboration {}:\n{}\n", i + 1, elaboration);
        }
    }

    Ok(String::new())
}
