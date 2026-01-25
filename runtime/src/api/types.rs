//! pie:core/types - Queue, FutureBool, FutureString resources

use crate::api::pie;
use crate::instance::InstanceState;
use crate::model::ModelInfo;
use anyhow::Result;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;
use wasmtime_wasi::async_trait;
use wasmtime_wasi::p2::{DynPollable, Pollable, subscribe};

static NEXT_QUEUE_ID: AtomicU32 = AtomicU32::new(0);

#[derive(Debug, Clone)]
pub struct Queue {
    pub service_id: usize,
    pub info: Arc<ModelInfo>,
    pub uid: u32,
}

#[derive(Debug)]
pub struct FutureBool {
    receiver: oneshot::Receiver<bool>,
    result: Option<bool>,
    done: bool,
}

impl FutureBool {
    pub fn new(receiver: oneshot::Receiver<bool>) -> Self {
        Self {
            receiver,
            result: None,
            done: false,
        }
    }
}

#[async_trait]
impl Pollable for FutureBool {
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

impl pie::core::types::Host for InstanceState {}

impl pie::core::types::HostQueue for InstanceState {
    async fn new(&mut self) -> Result<Resource<Queue>> {
        // TODO: Create queue - requires model context
        anyhow::bail!("Queue::new not yet implemented")
    }

    async fn synchronize(&mut self, _this: Resource<Queue>) -> Result<Resource<FutureBool>> {
        // TODO: Implement synchronization
        anyhow::bail!("Queue::synchronize not yet implemented")
    }

    async fn drop(&mut self, this: Resource<Queue>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl pie::core::types::HostFutureBool for InstanceState {
    async fn pollable(&mut self, this: Resource<FutureBool>) -> Result<Resource<DynPollable>> {
        subscribe(self.ctx().table, this)
    }

    async fn get(&mut self, this: Resource<FutureBool>) -> Result<Option<bool>> {
        let result = self.ctx().table.get(&this)?;
        if result.done {
            Ok(result.result)
        } else {
            Ok(None)
        }
    }

    async fn drop(&mut self, this: Resource<FutureBool>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

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
