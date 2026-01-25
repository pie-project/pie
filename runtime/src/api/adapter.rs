//! pie:core/adapter - Adapter resource for LoRA/fine-tuning weights

use crate::api::pie;
use crate::api::types::{Queue, FutureBool};
use crate::instance::InstanceState;
use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

#[derive(Debug)]
pub struct Adapter {
    pub name: String,
    pub ptr: u32,
}

impl pie::core::adapter::Host for InstanceState {}

impl pie::core::adapter::HostAdapter for InstanceState {
    async fn new(&mut self, name: String) -> Result<Resource<Adapter>> {
        // TODO: Allocate adapter resource
        let adapter = Adapter { name, ptr: 0 };
        Ok(self.ctx().table.push(adapter)?)
    }

    async fn get(&mut self, _name: String) -> Result<Option<Resource<Adapter>>> {
        // TODO: Look up existing adapter by name
        Ok(None)
    }

    async fn clone(&mut self, this: Resource<Adapter>, name: String) -> Result<Resource<Adapter>> {
        let _original = self.ctx().table.get(&this)?;
        // TODO: Clone adapter
        let adapter = Adapter { name, ptr: 0 };
        Ok(self.ctx().table.push(adapter)?)
    }

    async fn drop(&mut self, this: Resource<Adapter>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }

    async fn lock(&mut self, _this: Resource<Adapter>) -> Result<Resource<FutureBool>> {
        // TODO: Implement locking
        anyhow::bail!("Adapter::lock not yet implemented")
    }

    async fn unlock(&mut self, _this: Resource<Adapter>) -> Result<()> {
        // TODO: Implement unlocking
        Ok(())
    }

    async fn load(&mut self, _this: Resource<Adapter>, _queue: Resource<Queue>, _path: String) -> Result<Result<(), String>> {
        // TODO: Load adapter weights from path
        Ok(Ok(()))
    }

    async fn save(&mut self, _this: Resource<Adapter>, _queue: Resource<Queue>, _path: String) -> Result<Result<(), String>> {
        // TODO: Save adapter weights to path
        Ok(Ok(()))
    }
}
