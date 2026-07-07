//! Smoke test for the #[tool] macro + keep-core equip flow.
//!
//! - declares two tools with different param shapes
//! - splices their schemas into a raw token buffer via `tools::equip_prefix`
//!   (off the `Context` facade — the keep-core equip path)
//! - exercises both `call(json)` and `call_typed(args)` invocation paths
//! - prints what would be the agent-loop dispatch for a hand-crafted call

use inferlet::{Result, Tool, chat, serde_json, tool, tools};

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
    // ── Trait metadata ──
    let tool_list: &[&dyn Tool] = &[&web_search, &add];
    for t in tool_list {
        println!("name={} desc={} schema={}", t.name(), t.description(), t.schema());
    }

    // ── Keep-core equip: build the tool envelopes + splice the equip prefix into
    //    a raw token buffer (mirrors the retired `Context::equip`, no facade). The
    //    system block + equip prefix are the same tokens the facade produced. ──
    let mut buffer: Vec<u32> = chat::system("Use tools when helpful.");
    let buf_before = buffer.len();
    let envelopes: Vec<String> = tool_list
        .iter()
        .map(|t| {
            let parsed: serde_json::Value = serde_json::from_str(t.schema())
                .map_err(|e| format!("tool `{}`: invalid schema: {e}", t.name()))?;
            Ok(serde_json::json!({
                "name": t.name(),
                "description": t.description(),
                "parameters": parsed,
            })
            .to_string())
        })
        .collect::<Result<_>>()?;
    buffer.extend(tools::equip_prefix(&envelopes)?);
    let buf_after = buffer.len();
    println!("equip prefix tokens: {}", buf_after - buf_before);

    // ── Direct invocation paths (unchanged — #[tool] macro, keep-core) ──
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
