//! Demonstrates Skeleton-of-Thought (SoT) for parallel elaboration.
//!
//! This example first generates a high-level plan (skeleton) with key points,
//! then elaborates on each point concurrently. This approach can reduce latency
//! by parallelizing the detailed generation phase.

use futures::future;
use inferlet::{
    context::Context, inference::Sampler, model::Model,
    runtime, ContextExt, InstructExt, Result,
};
use std::time::Instant;

const HELP: &str = "\
Usage: skeleton-of-thought [OPTIONS]

A program that first generates a plan (a list of key points) for a question,
and then elaborates on each point concurrently.

Options:
  -q, --question <TEXT>        The question to answer [default: What are the defining characteristics of Rome?]
  -p, --num-points <POINTS>    Sets the maximum number of key points to generate in the plan [default: 3]
  -t, --plan-tokens <TOKENS>   Sets the max tokens for the planning generation [default: 256]
  -e, --elab-tokens <TOKENS>   Sets the max tokens for each elaboration generation [default: 256]
  -h, --help                   Prints this help message";


/// Generates a high-level plan and elaborates on each point in parallel.
async fn plan_and_generate_parallel(
    ctx: &Context,
    question: &str,
    max_points: usize,
    plan_max_tokens: usize,
    elab_max_tokens: usize,
) -> Result<Vec<String>> {
    // 1. Fork a context for generating the plan.
    let plan_ctx = ctx.fork()?;
    let plan_prompt = format!(
        "Generate up to {} key points that outline the answer to the following question: {}. \
        Each point must be enclosed between the <point> and </point> tags.",
        max_points, question
    );
    plan_ctx.user(&plan_prompt);
    plan_ctx.cue();

    let output = plan_ctx
        .generate(Sampler::TopP((0.6, 0.95)))
        .with_max_tokens(plan_max_tokens)
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
            let elab_ctx = ctx.fork()?;
            let complete_prompt = format!(
                "Elaborate on the following point: {}. \
                Your response should be complete and only concerned with this point.",
                point
            );
            elab_ctx.user(&complete_prompt);
            elab_ctx.cue();

            Ok(async move {
                elab_ctx
                    .generate(Sampler::TopP((0.6, 0.95)))
                    .with_max_tokens(elab_max_tokens)
                    .collect_text()
                    .await
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let results = future::join_all(leaf_futures).await;
    results.into_iter().collect::<Result<Vec<_>>>()
}

#[inferlet::main]
async fn main(args: Vec<String>) -> Result<String> {
    let mut args = inferlet::parse_args(args);

    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(String::new());
    }

    let question: String = args
        .value_from_str(["-q", "--question"])
        .unwrap_or_else(|_| "What are the defining characteristics of Rome?".to_string());

    let num_points: usize = args.value_from_str(["-p", "--num-points"]).unwrap_or(3);
    let plan_max_tokens: usize = args.value_from_str(["-t", "--plan-tokens"]).unwrap_or(256);
    let elab_max_tokens: usize = args.value_from_str(["-e", "--elab-tokens"]).unwrap_or(256);

    let start = Instant::now();

    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?;
    let model = Model::load(model_name)?;

    let ctx = Context::new(&model)?;
    ctx.system("You are a helpful, respectful and honest assistant.");
    ctx.flush().await?;

    println!(
        "--- Starting plan and generate (plan: {} points, {} tokens; elab: {} tokens) ---",
        num_points, plan_max_tokens, elab_max_tokens
    );

    let elaborations = plan_and_generate_parallel(
        &ctx,
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
