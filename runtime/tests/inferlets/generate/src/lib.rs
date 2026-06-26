//! Generate test inferlet — exercises the full forward pass pipeline.
//!
//! This inferlet tests: append → flush → generate (step loop).
//! Skips chat template rendering to work with mock backends.

use inferlet::{Context, model, sample::Sampler, Result};

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    // The engine serves exactly one model — no handle to load.
    // Create a context — skip chat template, fill tokens directly.
    let mut context = Context::new()?;

    // Encode a test prompt and append it directly (no chat template).
    let prompt_tokens = model::encode("hello world");
    eprintln!("[GENERATE] encoded prompt: {} tokens", prompt_tokens.len());
    context.append(&prompt_tokens);

    // Generate tokens with a small limit. Each step.execute() flushes
    // any pending tokens and runs one forward pass.
    let max_tokens: usize = 5;
    let mut g = context
        .generate(Sampler::TopK { temperature: 0.0, k: 1 })
        .max_tokens(max_tokens);

    let mut generated = Vec::new();
    while let Some(step) = g.next()? {
        let out = step.execute().await?;
        if out.tokens.is_empty() {
            continue;
        }
        eprintln!("[GENERATE] got tokens: {:?}", out.tokens);
        generated.extend(out.tokens.iter().copied());
    }

    let result = format!("generated {} tokens: {:?}", generated.len(), generated);
    eprintln!("[GENERATE] {}", result);
    Ok(result)
}
