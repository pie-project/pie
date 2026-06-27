//! Multi-sampler #7 per-kind coverage inferlet (echo).
//!
//! Generates a few tokens with each standard DedicatedKernel sampler kind in
//! sequence — TopK, TopP, MinP, TopKTopP — so one run exercises the executor's
//! recognizer→dispatch path across every kind that routes to a dedicated kernel
//! (FlashInfer top-k/p/joint + the temperature group). Used by the #7 cutover
//! per-kind `gate-on≡gate-off` verify: run with `PIE_RECOGNIZER_DISPATCH=1`
//! (recognizer drives the flag-set) vs unset (legacy flags), assert identical
//! tokens. Each `generate` continues the same context, so the kinds appear as
//! distinct decode fires.

use inferlet::{Context, model::Model, runtime, sample::Sampler, Result};

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    let models = runtime::models();
    let model = Model::load(&models[0])?;
    let tokenizer = model.tokenizer();

    let mut context = Context::new(&model)?;
    let prompt_tokens = tokenizer.encode("hello world");
    context.append(&prompt_tokens);

    let samplers: [(&str, Sampler); 4] = [
        ("topk", Sampler::TopK { temperature: 0.8, k: 40 }),
        ("topp", Sampler::TopP { temperature: 0.8, p: 0.9 }),
        ("minp", Sampler::MinP { temperature: 0.8, p: 0.05 }),
        ("joint", Sampler::TopKTopP { temperature: 0.8, k: 40, p: 0.9 }),
    ];

    let mut all = Vec::new();
    for (name, s) in samplers {
        let mut g = context.generate(s).max_tokens(4);
        let mut got = Vec::new();
        while let Some(step) = g.next()? {
            let out = step.execute().await?;
            got.extend(out.tokens.iter().copied());
        }
        eprintln!("[MULTISAMP] {name} tokens: {got:?}");
        all.extend(got);
    }

    Ok(format!("{{\"tokens\": {all:?}}}"))
}
