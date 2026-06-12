//! Context test inferlet — exercises model, context, and tokenizer host APIs.

use inferlet::{Context, Result, model::Model, runtime};

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    // Load the first available model
    let models = runtime::models();
    let model = Model::load(&models[0])?;

    // Get tokenizer and encode a test string
    let tokenizer = model.tokenizer();
    let encoded = tokenizer.encode("hello world");

    // Create a context
    let mut ctx = Context::new(&model)?;

    // Stage some buffered tokens
    ctx.append(&encoded);
    let buffered = ctx.buffer();

    // Query page info
    let page_size = ctx.page_size();

    Ok(format!(
        "encoded:{} buffered:{} page_size:{}",
        encoded.len(),
        buffered.len(),
        page_size
    ))
}
