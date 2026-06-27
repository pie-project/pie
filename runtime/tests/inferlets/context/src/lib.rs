//! Context test inferlet — exercises model, context, and tokenizer host APIs.

use inferlet::{Context, model, Result};

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    // The engine serves exactly one model — encode directly.
    let encoded = model::encode("hello world");

    // Create a context
    let mut ctx = Context::new()?;

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
