//! Callee inferlet for the `runtime::launch` E2E test.
//!
//! Accepts a raw string prompt, runs a tiny model generation, returns the
//! decoded text. Used by `launch-caller` via `inferlet::call`.

use inferlet::{Context, Result, chat, model::Model, runtime, sample::Sampler};

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    println!("[callee] received prompt: {}", input);

    let model_name = runtime::models()
        .first()
        .cloned()
        .ok_or("no models available")?;
    let model = Model::load(&model_name)?;

    let mut ctx = Context::new(&model)?;
    ctx.system("Answer in one short sentence.")
        .user(&format!("{} /no_think", input))
        .cue();

    let mut decoder = chat::Decoder::new(&model);
    let mut text = String::new();

    let mut g = ctx
        .generate(Sampler::Argmax)
        .max_tokens(32)
        .stop(&chat::stop_tokens(&model));

    while let Some(step) = g.next()? {
        let out = step.execute().await?;
        if out.tokens.is_empty() {
            continue;
        }
        match decoder.feed(&out.tokens)? {
            chat::Event::Delta(s) => text.push_str(&s),
            chat::Event::Done(s) => {
                text = s;
                break;
            }
            _ => {}
        }
    }

    let trimmed = strip_think_blocks(&text).trim().to_string();
    println!("[callee] returning: {}", trimmed);
    Ok(trimmed)
}

fn strip_think_blocks(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    loop {
        match rest.find("<think>") {
            None => {
                out.push_str(rest);
                break;
            }
            Some(start) => {
                out.push_str(&rest[..start]);
                let after_open = &rest[start + "<think>".len()..];
                match after_open.find("</think>") {
                    Some(end) => rest = &after_open[end + "</think>".len()..],
                    None => break,
                }
            }
        }
    }
    out
}
