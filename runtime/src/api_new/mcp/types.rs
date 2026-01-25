//! mcp:core/types - MCP shared types

use crate::api_new::pie;
use crate::instance::InstanceState;
use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;
use wasmtime_wasi::p2::{DynPollable, Pollable, subscribe};

#[derive(Debug)]
pub struct FutureContent {
    receiver: oneshot::Receiver<Result<pie::mcp::types::Content, pie::mcp::types::Error>>,
    result: Option<Result<pie::mcp::types::Content, pie::mcp::types::Error>>,
    done: bool,
}

impl FutureContent {
    pub fn new(receiver: oneshot::Receiver<Result<pie::mcp::types::Content, pie::mcp::types::Error>>) -> Self {
        Self {
            receiver,
            result: None,
            done: false,
        }
    }
}

#[async_trait]
impl Pollable for FutureContent {
    async fn ready(&mut self) {
        if self.done {
            return;
        }
        if let Ok(res) = (&mut self.receiver).await {
            self.result = Some(res);
        }
        self.done = true;
    }
}

#[derive(Debug)]
pub struct FutureJsonString {
    receiver: oneshot::Receiver<Result<String, pie::mcp::types::Error>>,
    result: Option<Result<String, pie::mcp::types::Error>>,
    done: bool,
}

impl FutureJsonString {
    pub fn new(receiver: oneshot::Receiver<Result<String, pie::mcp::types::Error>>) -> Self {
        Self {
            receiver,
            result: None,
            done: false,
        }
    }
}

#[async_trait]
impl Pollable for FutureJsonString {
    async fn ready(&mut self) {
        if self.done {
            return;
        }
        if let Ok(res) = (&mut self.receiver).await {
            self.result = Some(res);
        }
        self.done = true;
    }
}

impl pie::mcp::types::Host for InstanceState {}

impl pie::mcp::types::HostFutureContent for InstanceState {
    async fn pollable(&mut self, this: Resource<FutureContent>) -> Result<Resource<DynPollable>> {
        subscribe(self.ctx().table, this)
    }

    async fn get(
        &mut self,
        this: Resource<FutureContent>,
    ) -> Result<Option<Result<pie::mcp::types::Content, pie::mcp::types::Error>>> {
        let result = self.ctx().table.get(&this)?;
        if result.done {
            Ok(result.result.clone())
        } else {
            Ok(None)
        }
    }

    async fn drop(&mut self, this: Resource<FutureContent>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl pie::mcp::types::HostFutureJsonString for InstanceState {
    async fn pollable(&mut self, this: Resource<FutureJsonString>) -> Result<Resource<DynPollable>> {
        subscribe(self.ctx().table, this)
    }

    async fn get(
        &mut self,
        this: Resource<FutureJsonString>,
    ) -> Result<Option<Result<String, pie::mcp::types::Error>>> {
        let result = self.ctx().table.get(&this)?;
        if result.done {
            Ok(result.result.clone())
        } else {
            Ok(None)
        }
    }

    async fn drop(&mut self, this: Resource<FutureJsonString>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
