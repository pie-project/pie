//! pie:core/types - FutureString, FutureBlob resources

use crate::api::pie;
use crate::linker::InstanceState;
use anyhow::Result;
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;
use wasmtime_wasi::async_trait;
use wasmtime_wasi::p2::{DynPollable, Pollable, subscribe};

/// Future for string results (e.g., spawn results, async messages)
#[derive(Debug)]
pub struct FutureString {
    receiver: oneshot::Receiver<String>,
    result: Option<String>,
    done: bool,
}

impl FutureString {
    pub fn new(receiver: oneshot::Receiver<String>) -> Self {
        Self {
            receiver,
            result: None,
            done: false,
        }
    }
}

#[async_trait]
impl Pollable for FutureString {
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

/// Future for blob (byte vector) results (e.g., file transfers)
#[derive(Debug)]
pub struct FutureBlob {
    receiver: oneshot::Receiver<Vec<u8>>,
    result: Option<Vec<u8>>,
    done: bool,
}

impl FutureBlob {
    pub fn new(receiver: oneshot::Receiver<Vec<u8>>) -> Self {
        Self {
            receiver,
            result: None,
            done: false,
        }
    }
}

#[async_trait]
impl Pollable for FutureBlob {
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

impl pie::core::types::Host for InstanceState {}

impl pie::core::types::HostFutureString for InstanceState {
    async fn pollable(&mut self, this: Resource<FutureString>) -> Result<Resource<DynPollable>> {
        subscribe(self.ctx().table, this)
    }

    async fn get(&mut self, this: Resource<FutureString>) -> Result<Option<String>> {
        let result = self.ctx().table.get(&this)?;
        if result.done {
            Ok(result.result.clone())
        } else {
            Ok(None)
        }
    }

    async fn drop(&mut self, this: Resource<FutureString>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl pie::core::types::HostFutureBlob for InstanceState {
    async fn pollable(&mut self, this: Resource<FutureBlob>) -> Result<Resource<DynPollable>> {
        subscribe(self.ctx().table, this)
    }

    async fn get(&mut self, this: Resource<FutureBlob>) -> Result<Option<Vec<u8>>> {
        let result = self.ctx().table.get(&this)?;
        if result.done {
            Ok(result.result.clone())
        } else {
            Ok(None)
        }
    }

    async fn drop(&mut self, this: Resource<FutureBlob>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
