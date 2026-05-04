//! pie:mcp/client - MCP client session management
//!
//! Pure relay: the runtime never inspects MCP response bodies. Inferlets
//! receive opaque JSON strings and parse them in their SDK of choice.

use crate::api::pie;
use crate::instance::InstanceState;
use crate::process::{self, ProcessId};
use crate::server;
use anyhow::{Result, anyhow};
use serde_json::{Value, json};
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

/// An MCP session resource, representing a connection to a named MCP server.
#[derive(Debug)]
pub struct Session {
    pub server_name: String,
    pub client_id: server::ClientId,
    pub process_id: ProcessId,
}

impl pie::mcp::client::Host for InstanceState {
    async fn available_servers(&mut self) -> Result<Vec<String>> {
        let client_id = process::get_client_id(self.id()).await?
            .ok_or_else(|| anyhow!("No client session for process {}", self.id()))?;
        Ok(server::get_mcp_servers(client_id))
    }

    async fn connect(&mut self, server_name: String) -> Result<Result<Resource<Session>, pie::mcp::types::Error>> {
        let client_id = process::get_client_id(self.id()).await?
            .ok_or_else(|| anyhow!("No client session for process {}", self.id()))?;

        let servers = server::get_mcp_servers(client_id);
        if !servers.contains(&server_name) {
            return Ok(Err(pie::mcp::types::Error {
                code: -1,
                message: format!("MCP server '{}' not registered", server_name),
                data: None,
            }));
        }

        let session = Session {
            server_name,
            client_id,
            process_id: self.id(),
        };
        Ok(Ok(self.ctx().table.push(session)?))
    }
}

impl pie::mcp::client::HostSession for InstanceState {
    async fn list_tools(&mut self, this: Resource<Session>) -> Result<Result<String, pie::mcp::types::Error>> {
        let session = self.ctx().table.get(&this)?;
        relay(session, "tools/list", "{}").await
    }

    async fn call_tool(
        &mut self,
        this: Resource<Session>,
        name: String,
        args: String,
    ) -> Result<Result<String, pie::mcp::types::Error>> {
        let session = self.ctx().table.get(&this)?;
        let args_value = match parse_args_json(&args) {
            Ok(v) => v,
            Err(e) => return Ok(Err(e)),
        };
        let params = json!({ "name": name, "arguments": args_value }).to_string();
        relay(session, "tools/call", &params).await
    }

    async fn list_resources(&mut self, this: Resource<Session>) -> Result<Result<String, pie::mcp::types::Error>> {
        let session = self.ctx().table.get(&this)?;
        relay(session, "resources/list", "{}").await
    }

    async fn read_resource(
        &mut self,
        this: Resource<Session>,
        uri: String,
    ) -> Result<Result<String, pie::mcp::types::Error>> {
        let session = self.ctx().table.get(&this)?;
        let params = json!({ "uri": uri }).to_string();
        relay(session, "resources/read", &params).await
    }

    async fn list_prompts(&mut self, this: Resource<Session>) -> Result<Result<String, pie::mcp::types::Error>> {
        let session = self.ctx().table.get(&this)?;
        relay(session, "prompts/list", "{}").await
    }

    async fn get_prompt(
        &mut self,
        this: Resource<Session>,
        name: String,
        args: String,
    ) -> Result<Result<String, pie::mcp::types::Error>> {
        let session = self.ctx().table.get(&this)?;
        let args_value = match parse_args_json(&args) {
            Ok(v) => v,
            Err(e) => return Ok(Err(e)),
        };
        let params = json!({ "name": name, "arguments": args_value }).to_string();
        relay(session, "prompts/get", &params).await
    }

    async fn drop(&mut self, this: Resource<Session>) -> Result<()> {
        let _ = self.ctx().table.delete(this);
        Ok(())
    }
}

/// Relay an MCP method call through the client's WebSocket connection.
async fn relay(
    session: &Session,
    method: &str,
    params: &str,
) -> Result<Result<String, pie::mcp::types::Error>> {
    match server::send_mcp_request(
        session.client_id,
        session.process_id,
        session.server_name.clone(),
        method.to_string(),
        params.to_string(),
    )
    .await
    {
        Ok(Ok(result)) => Ok(Ok(result)),
        Ok(Err(err_payload)) => Ok(Err(decode_error_payload(&err_payload))),
        Err(e) => Ok(Err(pie::mcp::types::Error {
            code: -32000,
            message: e.to_string(),
            data: None,
        })),
    }
}

/// Parse a tool-input JSON string. The MCP wire format requires structured
/// arguments; if the inferlet hands us malformed JSON, refuse the call
/// rather than silently substituting `{}`.
fn parse_args_json(s: &str) -> std::result::Result<Value, pie::mcp::types::Error> {
    serde_json::from_str(s).map_err(|e| pie::mcp::types::Error {
        code: -32602, // JSON-RPC "invalid params"
        message: format!("Invalid arguments JSON: {}", e),
        data: None,
    })
}

/// Decode the structured error payload that the client puts in
/// `McpResponse.result` when `ok=false`. The payload is a JSON object with
/// `{code, message, data?}`; on parse failure, treat the raw string as the
/// message and use a generic `-32000` code.
fn decode_error_payload(payload: &str) -> pie::mcp::types::Error {
    if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(payload) {
        let code = map
            .get("code")
            .and_then(Value::as_i64)
            .map(|c| c as i32)
            .unwrap_or(-32000);
        let message = map
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or(payload)
            .to_string();
        let data = map.get("data").map(|v| v.to_string());
        return pie::mcp::types::Error { code, message, data };
    }
    pie::mcp::types::Error {
        code: -32000,
        message: payload.to_string(),
        data: None,
    }
}
