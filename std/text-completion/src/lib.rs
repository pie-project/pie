//! Simple text completion inferlet using chat-style prompting.
//!
//! Demonstrates text generation with a system prompt and user message
//! using the ContextExt high-level API.

use inferlet::{
    context::Context,
    inference::Sampler,
    model::Model,
    runtime,
    ContextExt,
    Result,
};

#[inferlet::main]
async fn main(args: Vec<String>) -> Result<String> {
    let mut args = inferlet::parse_args(args);

    let prompt: String = args.value_from_str(["-p", "--prompt"])
        .map_err(|e| format!("--prompt: {e}"))?;
    let max_tokens: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(256);
    let system_message: String = args.value_from_str(["-s", "--system"])
        .unwrap_or_else(|_| "You are a helpful, respectful and honest assistant.".to_string());
    let temperature: f32 = args.value_from_str(["-t", "--temperature"]).unwrap_or(0.6);
    let top_p: f32 = args.value_from_str("--top-p").unwrap_or(0.95);

    // Load model
    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?;
    let model = Model::load(model_name)?;

    // Create context and fill with messages
    let context = Context::new(&model)?;
    context
        .fill_sys(&system_message)
        .fill_user(&prompt)
        .flush()
        .await?;

    eprintln!("[INFERLET] flush done, starting generate (max_tokens={})", max_tokens);

    // Generate output stream
    let mut stream = context
        .generate(Sampler::TopP((temperature, top_p)))
        .with_max_tokens(max_tokens);
    
    let mut all_tokens = Vec::new();
    let tokenizer = model.tokenizer();

    // Stream tokens
    use std::io::Write;
    while let Some(token) = stream.next().await? {
        all_tokens.push(token);
        
        // Decode and print (simple partial decoding for now)
        // In real app we'd handle accumulation properly
        let text = tokenizer.decode(&[token])?;
        print!("{}", text);
        std::io::stdout().flush().ok();
        
        eprintln!("[INFERLET] token: {} {:?}", token, text);
    }
    
    let generated = tokenizer.decode(&all_tokens)?;
    eprintln!("[INFERLET] generated {} bytes", generated.len());
    
    Ok(generated)
}
