//! pie:mcp/client - MCP client session management
//!
//! Relays WIT calls from inferlets through the client's WebSocket connection
//! to MCP servers registered on the client session.

use crate::api::pie;
use crate::linker::InstanceState;
use crate::process::{self, ProcessId};
use crate::server;
use anyhow::{Result, anyhow};
use serde_json::json;
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

        // Validate the server is registered
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
    ) -> Result<Result<Vec<pie::mcp::types::Content>, pie::mcp::types::Error>> {
        let session = self.ctx().table.get(&this)?;
        let params = json!({ "name": name, "arguments": serde_json::from_str::<serde_json::Value>(&args).unwrap_or_default() }).to_string();
        match relay(session, "tools/call", &params).await? {
            Ok(json) => Ok(Ok(parse_content_list(&json))),
            Err(e) => Ok(Err(e)),
        }
    }

    async fn list_resources(&mut self, this: Resource<Session>) -> Result<Result<String, pie::mcp::types::Error>> {
        let session = self.ctx().table.get(&this)?;
        relay(session, "resources/list", "{}").await
    }

    async fn read_resource(
        &mut self,
        this: Resource<Session>,
        uri: String,
    ) -> Result<Result<Vec<pie::mcp::types::Content>, pie::mcp::types::Error>> {
        let session = self.ctx().table.get(&this)?;
        let params = json!({ "uri": uri }).to_string();
        match relay(session, "resources/read", &params).await? {
            Ok(json) => Ok(Ok(parse_content_list(&json))),
            Err(e) => Ok(Err(e)),
        }
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
        let params = json!({ "name": name, "arguments": serde_json::from_str::<serde_json::Value>(&args).unwrap_or_default() }).to_string();
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
        Ok(result) => Ok(Ok(result)),
        Err(e) => Ok(Err(pie::mcp::types::Error {
            code: -32000,
            message: e.to_string(),
            data: None,
        })),
    }
}

/// Parse a JSON string into a list of MCP content items.
/// For now, treats the entire response as a single text content item.
fn parse_content_list(json: &str) -> Vec<pie::mcp::types::Content> {
    vec![pie::mcp::types::Content::Text(json.to_string())]
}
