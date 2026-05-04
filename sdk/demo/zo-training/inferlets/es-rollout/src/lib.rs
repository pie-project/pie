//! Perform ES rollouts — generate text with a perturbed adapter.
//!
//! Accepts a list of rollout tasks (each with a uid, problem, and ZO seed),
//! generates text for each using the adapter with the given seed perturbation,
//! and returns the generated texts as a JSON array.

use inferlet::{
    Context,
    adapter::Adapter,
    inference::Sampler,
    model::Model,
    runtime,
    Result,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    /// Adapter name.
    name: String,
    /// System prompt for generation.
    system_prompt: String,
    /// Maximum tokens to generate per rollout.
    max_num_outputs: usize,
    /// Rollout tasks.
    rollouts: Vec<Rollout>,
    /// If false, skip the adapter (no perturbation). Used for A/B latency
    /// comparisons. Defaults to true to preserve existing behaviour.
    #[serde(default = "default_true")]
    use_adapter: bool,
}

fn default_true() -> bool { true }

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct Rollout {
    uid: String,
    task: String,
    seed: i64,
}

#[inferlet::main]
async fn main(input: Input) -> Result<Vec<String>> {
    let model_name = runtime::models().into_iter().next()
        .ok_or("No models available")?;
    let model = Model::load(&model_name)?;

    let adapter = if input.use_adapter {
        Some(
            Adapter::open(&model, &input.name)
                .ok_or_else(|| format!("Adapter '{}' not found", input.name))?,
        )
    } else {
        None
    };

    let sampler = Sampler::TopP((0.6, 0.95));
    let mut results: Vec<String> = Vec::with_capacity(input.rollouts.len());

    for rollout in &input.rollouts {
        let mut ctx = Context::new(&model)?;
        ctx.system(&input.system_prompt);
        ctx.user(&rollout.task);
        ctx.cue();

        let mut stream = ctx
            .generate(sampler.clone())
            .with_max_tokens(input.max_num_outputs);
        if let Some(ref a) = adapter {
            stream = stream.with_adapter(a).with_zo_seed(rollout.seed);
        }
        let text = stream.collect_text().await?;

        results.push(text);
    }

    Ok(results)
}
