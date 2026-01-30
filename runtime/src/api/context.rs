//! pie:core/context - Context resource for KV cache management

use crate::api::pie;
use crate::api::model::Model;
use crate::api::types::FutureBool;
use crate::instance::InstanceState;
use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

#[derive(Debug)]
pub struct Context {
    pub name: String,
    pub pointer: u32,
    pub staged_tokens: Vec<u32>,
    // TODO: Add KV cache page pointers and state
}

impl pie::core::context::Host for InstanceState {}

impl pie::core::context::HostContext for InstanceState {
    async fn create(&mut self, _model: Resource<Model>, name: String, fill: Option<Vec<u32>>) -> Result<Result<Resource<Context>, String>> {
        let _ = fill;
        let ctx = Context { 
            name,
            pointer: 0,
            staged_tokens: vec![],
        };
        Ok(Ok(self.ctx().table.push(ctx)?))
    }

    async fn destroy(&mut self, this: Resource<Context>) -> Result<Result<(), String>> {
        self.ctx().table.delete(this)?;
        Ok(Ok(()))
    }

    async fn get(&mut self, _model: Resource<Model>, _name: String) -> Result<Option<Resource<Context>>> {
        // TODO: Look up existing context by name
        Ok(None)
    }

    async fn fork(&mut self, this: Resource<Context>, new_name: String) -> Result<Result<Resource<Context>, String>> {
        let parent = self.ctx().table.get(&this)?;
        let forked = Context { 
            name: new_name,
            pointer: parent.pointer,
            staged_tokens: parent.staged_tokens.clone(),
        };
        Ok(Ok(self.ctx().table.push(forked)?))
    }

    async fn join(&mut self, _this: Resource<Context>, _other: Resource<Context>) -> Result<Result<(), String>> {
        // TODO: Merge contexts
        Ok(Ok(()))
    }

    async fn drop(&mut self, this: Resource<Context>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }

    async fn lock(&mut self, _this: Resource<Context>) -> Result<Resource<FutureBool>> {
        // TODO: Implement locking
        anyhow::bail!("Context::lock not yet implemented")
    }

    async fn unlock(&mut self, _this: Resource<Context>) -> Result<Result<(), String>> {
        // TODO: Implement unlocking
        Ok(Ok(()))
    }

    async fn page_size(&mut self, _this: Resource<Context>) -> Result<u32> {
        Ok(256) // TODO: Get from model
    }

    async fn num_total_pages(&mut self, _this: Resource<Context>) -> Result<u32> {
        // TODO: Get number of committed pages
        Ok(0)
    }

    async fn num_total_tokens(&mut self, _this: Resource<Context>) -> Result<u32> {
        // TODO: Get total tokens
        Ok(0)
    }

    async fn commit_pages(&mut self, _this: Resource<Context>, _indices: Vec<u32>) -> Result<Result<(), String>> {
        // TODO: Commit KV pages to context
        Ok(Ok(()))
    }

    async fn allocate_pages(&mut self, _this: Resource<Context>, _num_pages: u32) -> Result<Result<(), String>> {
        // TODO: Allocate pages
        Ok(Ok(()))
    }

    async fn free_pages(&mut self, _this: Resource<Context>, _num_pages: u32) -> Result<Result<(), String>> {
        // TODO: Free pages
        Ok(Ok(()))
    }

    async fn pointer(&mut self, this: Resource<Context>) -> Result<u32> {
        let ctx = self.ctx().table.get(&this)?;
        Ok(ctx.pointer)
    }

    async fn set_pointer(&mut self, this: Resource<Context>, pointer: u32) -> Result<Result<(), String>> {
        let ctx = self.ctx().table.get_mut(&this)?;
        ctx.pointer = pointer;
        Ok(Ok(()))
    }

    async fn staged_tokens(&mut self, this: Resource<Context>) -> Result<Vec<u32>> {
        let ctx = self.ctx().table.get(&this)?;
        Ok(ctx.staged_tokens.clone())
    }

    async fn set_staged_tokens(&mut self, this: Resource<Context>, tokens: Vec<u32>) -> Result<Result<(), String>> {
        let ctx = self.ctx().table.get_mut(&this)?;
        ctx.staged_tokens = tokens;
        Ok(Ok(()))
    }


}
