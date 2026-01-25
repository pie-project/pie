//! pie:mcp/client - MCP client session management

use crate::api::pie;
use crate::api::mcp::types::FutureJsonString;
use crate::instance::InstanceState;
use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

#[derive(Debug)]
pub struct Session {
    pub server_name: String,
    // TODO: Add actual MCP session state
}

impl pie::mcp::client::Host for InstanceState {
    async fn available_servers(&mut self) -> Result<Vec<String>> {
        // TODO: Return list of available MCP servers
        Ok(vec![])
    }

    async fn connect(&mut self, server_name: String) -> Result<Result<Resource<Session>, pie::mcp::types::Error>> {
        // TODO: Establish MCP connection
        let session = Session { server_name };
        Ok(Ok(self.ctx().table.push(session)?))
    }
}

impl pie::mcp::client::HostSession for InstanceState {
    async fn list_tools(&mut self, _this: Resource<Session>) -> Result<Result<String, pie::mcp::types::Error>> {
        // TODO: Implement MCP tools/list
        Ok(Ok("[]".to_string()))
    }

    async fn list_resources(&mut self, _this: Resource<Session>) -> Result<Result<String, pie::mcp::types::Error>> {
        // TODO: Implement MCP resources/list
        Ok(Ok("[]".to_string()))
    }

    async fn list_prompts(&mut self, _this: Resource<Session>) -> Result<Result<String, pie::mcp::types::Error>> {
        // TODO: Implement MCP prompts/list
        Ok(Ok("[]".to_string()))
    }

    async fn call_tool(
        &mut self,
        _this: Resource<Session>,
        _name: String,
        _args: String,
    ) -> Result<Result<String, pie::mcp::types::Error>> {
        // TODO: Implement MCP tools/call
        Ok(Ok("{}".to_string()))
    }

    async fn read_resource(
        &mut self,
        _this: Resource<Session>,
        _uri: String,
    ) -> Result<Result<Vec<pie::mcp::types::Content>, pie::mcp::types::Error>> {
        // TODO: Implement MCP resources/read
        Ok(Ok(vec![]))
    }

    async fn get_prompt(
        &mut self,
        _this: Resource<Session>,
        _name: String,
        _args: String,
    ) -> Result<Result<Vec<pie::mcp::types::Content>, pie::mcp::types::Error>> {
        // TODO: Implement MCP prompts/get
        Ok(Ok(vec![]))
    }

    async fn receive_request(&mut self, _this: Resource<Session>) -> Result<Resource<FutureJsonString>> {
        // TODO: Implement receive request for server-initiated messages
        anyhow::bail!("Session::receive_request not yet implemented")
    }

    async fn send_response(&mut self, _this: Resource<Session>, _response: String) -> Result<()> {
        // TODO: Implement sending response
        Ok(())
    }

    async fn close(&mut self, this: Resource<Session>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }

    async fn drop(&mut self, this: Resource<Session>) -> Result<()> {
        let _ = self.ctx().table.delete(this);
        Ok(())
    }
}
