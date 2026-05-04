//! Smoke test for the #[tool] macro + Context::equip flow.
//!
//! - declares two tools with different param shapes
//! - registers them via `ctx.equip`
//! - exercises both `call(json)` and `call_typed(args)` invocation paths
//! - prints what would be the agent-loop dispatch for a hand-crafted call

use inferlet::{Context, Result, Tool, model::Model, runtime, tool};

/// Search the web for current information.
#[tool]
async fn web_search(query: String) -> Result<String> {
    Ok(format!("(stub: searched for `{query}`)"))
}

/// Add two integers and return the sum as a string.
#[tool]
async fn add(a: i64, b: i64) -> Result<String> {
    Ok((a + b).to_string())
}

#[inferlet::main]
async fn main(_prompt: String) -> Result<String> {
    let model = Model::load(runtime::models().first().ok_or("no models")?)?;
    let mut ctx = Context::new(&model)?;

    // ── Trait metadata ──
    let tools: &[&dyn Tool] = &[&web_search, &add];
    for t in tools {
        println!("name={} desc={} schema={}", t.name(), t.description(), t.schema());
    }

    // ── equip splices the chat-template tool block into ctx.buffer ──
    let buf_before = ctx.buffer().len();
    ctx.system("Use tools when helpful.").equip(tools)?;
    let buf_after = ctx.buffer().len();
    println!("equip prefix tokens: {}", buf_after - buf_before);

    // ── Direct invocation paths ──
    let typed = web_search::call_typed("rust async traits".into()).await?;
    println!("call_typed -> {typed}");

    let json = web_search::call(r#"{"query":"rust async traits"}"#).await?;
    println!("call(json)  -> {json}");

    let sum_json = add::call(r#"{"a":2,"b":40}"#).await?;
    println!("add(json)   -> {sum_json}");

    // ── Bad JSON path: serde error wraps the tool name ──
    let bad = add::call(r#"{"a":2}"#).await;
    println!("add(missing b) -> {:?}", bad);

    Ok("ok".into())
}
