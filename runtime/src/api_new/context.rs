//! pie:core/context - Context resource for KV cache management

use crate::api_new::pie;
use crate::api_new::types::FutureBool;
use crate::instance::InstanceState;
use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

#[derive(Debug)]
pub struct Context {
    pub name: String,
    // TODO: Add KV cache page pointers and state
}

impl pie::core::context::Host for InstanceState {}

impl pie::core::context::HostContext for InstanceState {
    async fn new(&mut self, name: String) -> Result<Resource<Context>> {
        let ctx = Context { name };
        Ok(self.ctx().table.push(ctx)?)
    }

    async fn get(&mut self, _name: String) -> Result<Option<Resource<Context>>> {
        // TODO: Look up existing context by name
        Ok(None)
    }

    async fn fork(&mut self, this: Resource<Context>, new_name: String) -> Result<Resource<Context>> {
        let _parent = self.ctx().table.get(&this)?;
        // TODO: Fork KV cache pages
        let forked = Context { name: new_name };
        Ok(self.ctx().table.push(forked)?)
    }

    async fn join(&mut self, _this: Resource<Context>, _other: Resource<Context>) -> Result<()> {
        // TODO: Merge contexts
        Ok(())
    }

    async fn drop(&mut self, this: Resource<Context>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }

    async fn lock(&mut self, _this: Resource<Context>) -> Result<Resource<FutureBool>> {
        // TODO: Implement locking
        anyhow::bail!("Context::lock not yet implemented")
    }

    async fn unlock(&mut self, _this: Resource<Context>) -> Result<()> {
        // TODO: Implement unlocking
        Ok(())
    }

    async fn grow(&mut self, _this: Resource<Context>, _size: u32) -> Result<()> {
        // TODO: Grow context capacity
        Ok(())
    }

    async fn shrink(&mut self, _this: Resource<Context>, _size: u32) -> Result<()> {
        // TODO: Shrink context capacity
        Ok(())
    }
}
