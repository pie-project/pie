//! Context test inferlet — exercises model, context, and tokenizer host APIs.

use inferlet::{
    context::Context,
    model::Model,
    runtime,
    Result,
};

#[inferlet::main]
async fn main(_args: Vec<String>) -> Result<String> {
    // Load the first available model
    let models = runtime::models();
    let model = Model::load(&models[0])?;

    // Get tokenizer and encode a test string
    let tokenizer = model.tokenizer();
    let encoded = tokenizer.encode("hello world");

    // Create a context
    let ctx = Context::create(&model)?;

    // Stage some buffered tokens
    ctx.set_buffered_tokens(&encoded);
    let buffered = ctx.buffered_tokens();

    // Query page info
    let page_size = ctx.tokens_per_page();

    Ok(format!(
        "encoded:{} buffered:{} page_size:{}",
        encoded.len(),
        buffered.len(),
        page_size
    ))
}
