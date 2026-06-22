//! Generate test inferlet — exercises the full forward pass pipeline.
//!
//! This inferlet tests: append → flush → generate (step loop).
//! Skips chat template rendering to work with mock backends.

use inferlet::{
    Context,
    model::Model,
    runtime,
    sample::Sampler,
    Result,
};

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    // Load the first available model
    let models = runtime::models();
    let model = Model::load(&models[0])?;
    let tokenizer = model.tokenizer();

    // Create a context — skip chat template, fill tokens directly
    let mut context = Context::new(&model)?;

    // Encode a test prompt and append it directly (no chat template).
    let prompt_tokens = tokenizer.encode("hello world");
    eprintln!("[GENERATE] encoded prompt: {} tokens", prompt_tokens.len());
    context.append(&prompt_tokens);

    // Generate tokens with a small limit. Each step.execute() flushes
    // any pending tokens and runs one forward pass.
    let max_tokens: usize = 5;
    let mut g = context
        .generate(Sampler::TopK { temperature: 0.0, k: 1 })
        .max_tokens(max_tokens);

    let mut generated = Vec::new();
    let mut checkpointed_mid_generation = false;
    while let Some(step) = g.next()? {
        let out = step.execute().await?;
        if out.tokens.is_empty() {
            continue;
        }
        if !checkpointed_mid_generation {
            let buffered_after_step = g.context().buffer().len();
            let checkpoint = g.fork()?;
            eprintln!(
                "[GENERATE] forked checkpoint mid-generation with {} buffered tokens",
                buffered_after_step
            );
            drop(checkpoint);
            checkpointed_mid_generation = true;
        }
        eprintln!("[GENERATE] got tokens: {:?}", out.tokens);
        generated.extend(out.tokens.iter().copied());
    }

    let result = format!("generated {} tokens: {:?}", generated.len(), generated);
    eprintln!("[GENERATE] {}", result);
    Ok(result)
}
