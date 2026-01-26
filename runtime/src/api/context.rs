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
    // TODO: Add KV cache page pointers and state
}

impl pie::core::context::Host for InstanceState {
    async fn kv_page_size(&mut self, _model: Resource<Model>) -> Result<u32> {
        // TODO: Get actual page size from model
        Ok(256)
    }

    async fn allocate_pages(&mut self, _model: Resource<Model>, num_pages: u32) -> Result<Result<Vec<u32>, String>> {
        // TODO: Allocate KV pages from model's page pool
        let page_ids: Vec<u32> = (0..num_pages).collect();
        Ok(Ok(page_ids))
    }

    async fn free_pages(&mut self, _model: Resource<Model>, _page_ids: Vec<u32>) -> Result<Result<(), String>> {
        // TODO: Free KV pages back to model's page pool
        Ok(Ok(()))
    }
}

impl pie::core::context::HostContext for InstanceState {
    async fn create(&mut self, _model: Resource<Model>, name: String) -> Result<Result<Resource<Context>, String>> {
        let ctx = Context { name };
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
        let _parent = self.ctx().table.get(&this)?;
        // TODO: Fork KV cache pages
        let forked = Context { name: new_name };
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

    async fn commit_pages(&mut self, _this: Resource<Context>, _page_ids: Vec<u32>) -> Result<Result<(), String>> {
        // TODO: Commit KV pages to context
        Ok(Ok(()))
    }

    async fn trim_pages(&mut self, _this: Resource<Context>, _num_pages: u32) -> Result<Result<(), String>> {
        // TODO: Trim KV pages from context
        Ok(Ok(()))
    }
}
