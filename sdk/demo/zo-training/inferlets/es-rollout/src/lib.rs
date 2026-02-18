use inferlet::prelude::*;
use inferlet::{adapter::Adapter, parse_args, runtime, ContextExt, InstructExt, Result};
use inferlet::inference::Sampler;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct Rollout {
    uid: String,
    task: String,
    seed: i64,
}

#[inferlet::main]
async fn main(args: Vec<String>) -> Result<String> {
    let mut args = parse_args(args);
    let name: String = args.value_from_str("--name").map_err(|e| e.to_string())?;
    let system_prompt: String = args.value_from_str("--system-prompt").map_err(|e| e.to_string())?;
    let max_num_outputs: usize = args.value_from_str("--max-num-outputs").map_err(|e| e.to_string())?;
    let rollouts_str: String = args.value_from_str("--rollouts").map_err(|e| e.to_string())?;
    let rollouts: Vec<Rollout> = serde_json::from_str(&rollouts_str)
        .map_err(|e| e.to_string())?;

    // Load model and adapter.
    let model_name = runtime::models().into_iter().next()
        .ok_or_else(|| "No models available".to_string())?;
    let model = Model::load(&model_name)?;
    let adapter = Adapter::open(&model, &name)
        .ok_or_else(|| format!("Adapter '{}' not found", name))?;

    let sampler = Sampler::TopP((0.6, 0.95));

    // Run rollouts and collect results.
    // Each rollout creates a context, fills chat template, and generates.
    let mut results: Vec<String> = Vec::with_capacity(rollouts.len());

    for rollout in &rollouts {
        let ctx = <Context as ContextExt>::new(&model)?;
        ctx.system(&system_prompt);
        ctx.user(&rollout.task);
        ctx.cue();

        let tokens = ctx.generate(sampler.clone())
            .with_adapter(&adapter)
            .with_zo_seed(rollout.seed)
            .with_max_tokens(max_num_outputs)
            .collect_tokens()
            .await?;

        let text = model.tokenizer().decode(&tokens)?;
        results.push(text);
    }

    serde_json::to_string(&results).map_err(|e| e.to_string())
}
