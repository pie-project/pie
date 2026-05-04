//! Connects to an MCP server registered on the client session, lists its
//! tools, calls one, and demonstrates the canonical pattern for handling
//! the raw JSON responses returned by `pie:mcp/client`.

use inferlet::{Result, mcp};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_server")]
    server: String,
    #[serde(default = "default_tool")]
    tool: String,
    #[serde(default = "default_text")]
    text: String,
}

fn default_server() -> String { "demo".into() }
fn default_tool() -> String { "echo".into() }
fn default_text() -> String { "hello".into() }

#[derive(Serialize)]
struct Output {
    servers: Vec<String>,
    tools_json: String,
    result: String,
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let servers = mcp::client::available_servers();
    println!("available servers: {:?}", servers);

    if !servers.contains(&input.server) {
        return Err(format!(
            "MCP server '{}' is not registered (have: {:?})",
            input.server, servers
        ));
    }

    let session = mcp::client::connect(&input.server)
        .map_err(|e| format!("connect('{}') failed: {}", input.server, e.message))?;

    // tools/list response is opaque JSON: `{"tools": [{"name": ..., ...}, ...]}`.
    let tools_json = session.list_tools()
        .map_err(|e| format!("list_tools failed: {}", e.message))?;
    println!("tools: {}", tools_json);

    // tools/call: arguments must be JSON. The response carries
    // `content[]` and `isError`; we have to inspect both.
    let args = serde_json::json!({ "text": input.text }).to_string();
    let raw = session.call_tool(&input.tool, &args)
        .map_err(|e| format!("call_tool('{}') failed: {}", input.tool, e.message))?;

    let result = extract_tool_result(&input.tool, &raw)?;
    println!("result: {}", result);

    Ok(Output { servers, tools_json, result })
}

/// Pull the user-visible string out of a `tools/call` response.
///
/// Honors `isError: true` by surfacing it as an inferlet error. For
/// success, takes the text of the first text content item.
fn extract_tool_result(tool: &str, raw: &str) -> Result<String> {
    let v: Value = serde_json::from_str(raw)
        .map_err(|e| format!("call_tool('{}') returned non-JSON: {}", tool, e))?;

    let text_of_first_text_item = || -> String {
        v.get("content")
            .and_then(Value::as_array)
            .and_then(|arr| arr.iter().find_map(|item| {
                if item.get("type").and_then(Value::as_str) == Some("text") {
                    item.get("text").and_then(Value::as_str).map(str::to_string)
                } else {
                    None
                }
            }))
            .unwrap_or_else(|| "<no text content>".to_string())
    };

    if v.get("isError").and_then(Value::as_bool).unwrap_or(false) {
        return Err(format!(
            "tool '{}' reported failure: {}",
            tool,
            text_of_first_text_item(),
        ));
    }
    Ok(text_of_first_text_item())
}
