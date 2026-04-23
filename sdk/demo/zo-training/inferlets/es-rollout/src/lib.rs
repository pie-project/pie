use futures::future::join_all;
use inferlet::stop_condition::{StopCondition, ends_with_any, max_len};
use inferlet::{self, Adapter, Args, Result, Sampler, bail, store_delete, store_get, store_set};
use serde::Deserialize;
use std::future::Future;
use std::pin::Pin;

#[derive(Deserialize, Debug)]
struct Rollout {
    uid: String,
    task: String,
    seed: i64,
}

#[inferlet::main]
async fn main(mut args: Args) -> Result<String> {
    // Parse required arguments.
    let name: String = args.value_from_str("--name")?;
    let system_prompt: String = args.value_from_str("--system-prompt")?;
    let max_num_outputs: usize = args.value_from_str("--max-num-outputs")?;

    // Parse the tasks from a JSON string.
    let rollouts_str: String = args.value_from_str("--rollouts")?;
    let rollouts: Vec<Rollout> = serde_json::from_str(&rollouts_str)?;

    // The futures vector for a single-threaded (Wasm) environment does not need the `Send` bound.
    let mut futures: Vec<Pin<Box<dyn Future<Output = String>>>> = vec![];

    // import the main adapter
    let model = inferlet::get_auto_model();
    let queue = model.create_queue();

    let es_adapter = queue.import_adapter(&name);

    let stop_cond = max_len(max_num_outputs).or(ends_with_any(model.eos_tokens()));

    // println!("🚀 Starting parallel rollout...");
    for rollout in rollouts {
        let stop_cond_ = stop_cond.clone();
        let sampler = Sampler::top_p(0.6, 0.95);

        let mut ctx = model.create_context();
        ctx.set_adapter(es_adapter);
        ctx.set_adapter_random_seed(rollout.seed);
        ctx.fill_system(&system_prompt);
        ctx.fill_user(&rollout.task);

        let generation_future = async move {
            let mut generated_token_ids = Vec::new();

            loop {
                let next_token_id = ctx.decode_step_and_fill(&sampler).await;
                generated_token_ids.push(next_token_id);
                if stop_cond_.check(&generated_token_ids) {
                    break;
                }

            }

            // clear up the cache
            // store_delete(&rollout.uid);

            ctx.tokenizer.detokenize(&generated_token_ids)
        };

        futures.push(Box::pin(generation_future));
    }

    // println!("⏳ Waiting for {} tasks to complete...", futures.len());
    let results: Vec<String> = join_all(futures).await;

    // Serialize the collected text outputs into a JSON string array.
    let response_json = serde_json::to_string(&results)?;

    Ok(response_json)   
}
