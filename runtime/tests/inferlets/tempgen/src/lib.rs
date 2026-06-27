//! Temperature-sampling test inferlet (echo, Task #4 verify).
//!
//! Drives `Sampler::Multinomial { temperature, draws: 1 }`, which lowers to
//! `Spec::Multinomial { temperature }` → per-row params `(T>0, top_k=0,
//! top_p=1, min_p=0)` → the host recognizer classifies it **Temperature** →
//! the BakedIR de-hardwiring path (`PIE_DEHARDWIRE_STD_SAMPLERS`). Used by the
//! 4090 executor-integration verify to prove temp fires route to the baked IR
//! `standard_sampler_program(Temperature, V)` and produce valid tokens.

use inferlet::{Context, model::Model, runtime, sample::Sampler, Result};

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    let models = runtime::models();
    let model = Model::load(&models[0])?;
    let tokenizer = model.tokenizer();

    let mut context = Context::new(&model)?;
    let prompt_tokens = tokenizer.encode("hello world");
    eprintln!("[TEMPGEN] encoded prompt: {} tokens", prompt_tokens.len());
    context.append(&prompt_tokens);

    let max_tokens: usize = 8;
    let mut g = context
        .generate(Sampler::Multinomial { temperature: 0.8, draws: 1 })
        .max_tokens(max_tokens);

    let mut generated = Vec::new();
    while let Some(step) = g.next()? {
        let out = step.execute().await?;
        if out.tokens.is_empty() {
            continue;
        }
        eprintln!("[TEMPGEN] got tokens: {:?}", out.tokens);
        generated.extend(out.tokens.iter().copied());
    }

    let text = tokenizer.decode(&generated);
    eprintln!("[TEMPGEN] generated {} tokens: {:?}", generated.len(), text);
    Ok(format!("{{\"tokens\": {generated:?}, \"text\": {text:?}}}"))
}
