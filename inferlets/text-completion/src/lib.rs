//! Simple text completion inferlet.
//!
//! Demonstrates chat-style generation using the `InstructExt` +
//! `EventStream` high-level API.

use inferlet::{
    context::Context,
    inference::Sampler,
    model::Model,
    runtime,
    ContextExt, Event, InstructExt,
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

    // Create context and fill with instruct messages
    let ctx = Context::new(&model)?;
    ctx.system(&system_message);
    ctx.user(&prompt);
    ctx.cue();

    // Generate
    let mut events = ctx
        .generate(Sampler::TopP((temperature, top_p)))
        .with_max_tokens(max_tokens)
        .decode()
        .with_reasoning();

    let mut output = String::new();

    while let Some(event) = events.next().await? {
        match event {
            Event::Thinking(s) => {
                print!("{}", s);
            }
            Event::ThinkingDone(_) => {
            }
            Event::Text(s) => {
                print!("{}", s);
            }
            Event::Done(s) => {
                output = s;
                break;
            }
            _ => {}
        }
    }

    Ok(output)
}
