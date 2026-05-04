//! pie:core/context - Context resource for KV cache management

use crate::api::model::Model;
use crate::api::pie;
use crate::context::{self, ContextId};
use crate::instance::InstanceState;
use crate::model::ModelId;
use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

/// Context resource - represents a KV cache context managed by the ContextManager.
#[derive(Debug)]
pub struct Context {
    /// The context ID assigned by the ContextManager.
    pub context_id: ContextId,
    /// The model ID (for routing to the correct ContextManager).
    pub model_id: ModelId,
}

impl pie::core::context::Host for InstanceState {}

impl pie::core::context::HostContext for InstanceState {
    async fn create(
        &mut self,
        model: Resource<Model>,
    ) -> Result<Result<Resource<Context>, String>> {
        let model = self.ctx().table.get(&model)?;
        let model_id = model.model_id;
        let process_id = self.id();

        match context::create(model_id, process_id).await {
            Ok(context_id) => {
                let ctx = Context { context_id, model_id };
                Ok(Ok(self.ctx().table.push(ctx)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn open(
        &mut self,
        model: Resource<Model>,
        name: String,
    ) -> Result<Result<Resource<Context>, String>> {
        let model = self.ctx().table.get(&model)?;
        let model_id = model.model_id;
        let username = self.get_username();
        let process_id = self.id();

        let snapshot_id = match context::lookup(model_id, username, name).await {
            Ok(id) => id,
            Err(e) => return Ok(Err(e.to_string())),
        };
        match context::fork(model_id, snapshot_id, process_id).await {
            Ok(context_id) => {
                let ctx = Context { context_id, model_id };
                Ok(Ok(self.ctx().table.push(ctx)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn take(
        &mut self,
        model: Resource<Model>,
        name: String,
    ) -> Result<Result<Resource<Context>, String>> {
        let model = self.ctx().table.get(&model)?;
        let model_id = model.model_id;
        let username = self.get_username();
        let process_id = self.id();

        match context::take(model_id, username, name, process_id).await {
            Ok(context_id) => {
                let ctx = Context { context_id, model_id };
                Ok(Ok(self.ctx().table.push(ctx)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn delete(
        &mut self,
        model: Resource<Model>,
        name: String,
    ) -> Result<Result<(), String>> {
        let model = self.ctx().table.get(&model)?;
        let model_id = model.model_id;
        let username = self.get_username();

        match context::delete(model_id, username, name).await {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn fork(
        &mut self,
        this: Resource<Context>,
    ) -> Result<Result<Resource<Context>, String>> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_id = ctx.model_id;
        let process_id = self.id();

        match context::fork(model_id, context_id, process_id).await {
            Ok(new_context_id) => {
                let new_ctx = Context { context_id: new_context_id, model_id };
                Ok(Ok(self.ctx().table.push(new_ctx)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn save(
        &mut self,
        this: Resource<Context>,
        name: String,
    ) -> Result<Result<(), String>> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_id = ctx.model_id;
        let username = self.get_username();

        match context::save(model_id, context_id, username, Some(name)).await {
            Ok(_) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn snapshot(
        &mut self,
        this: Resource<Context>,
    ) -> Result<Result<String, String>> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_id = ctx.model_id;
        let username = self.get_username();

        match context::save(model_id, context_id, username, None).await {
            Ok(Some(name)) => Ok(Ok(name)),
            Ok(None) => Ok(Err("snapshot returned no name".into())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn destroy(&mut self, this: Resource<Context>) -> Result<()> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_id = ctx.model_id;

        let _ = context::destroy(model_id, context_id).await;
        self.ctx().table.delete(this)?;
        Ok(())
    }

    async fn drop(&mut self, this: Resource<Context>) -> Result<()> {
        // Context cleanup is handled by DestroyAll on instance drop.
        // Individual handle drops just remove the resource table entry.
        self.ctx().table.delete(this)?;
        Ok(())
    }

    async fn tokens_per_page(&mut self, this: Resource<Context>) -> Result<u32> {
        let ctx = self.ctx().table.get(&this)?;
        Ok(context::tokens_per_page(ctx.model_id))
    }

    async fn model(&mut self, this: Resource<Context>) -> Result<Resource<Model>> {
        let ctx = self.ctx().table.get(&this)?;
        let model_id = ctx.model_id;

        if let Some(m) = crate::model::get_model(model_id) {
            let model = Model { model_id, model: m.clone() };
            return Ok(self.ctx().table.push(model)?);
        }

        anyhow::bail!("Model not found in cache")
    }

    async fn committed_page_count(&mut self, this: Resource<Context>) -> Result<u32> {
        let ctx = self.ctx().table.get(&this)?;
        Ok(context::committed_page_count(ctx.model_id, ctx.context_id))
    }

    async fn working_page_count(&mut self, this: Resource<Context>) -> Result<u32> {
        let ctx = self.ctx().table.get(&this)?;
        Ok(context::working_page_count(ctx.model_id, ctx.context_id))
    }

    async fn commit_working_pages(
        &mut self,
        this: Resource<Context>,
        num_pages: u32,
    ) -> Result<Result<(), String>> {
        let ctx = self.ctx().table.get(&this)?;
        match context::commit_working_pages(ctx.model_id, ctx.context_id, num_pages as usize).await {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn reserve_working_pages(
        &mut self,
        this: Resource<Context>,
        num_pages: u32,
    ) -> Result<Result<(), String>> {
        let ctx = self.ctx().table.get(&this)?;
        match context::reserve_working_pages(ctx.model_id, ctx.context_id, num_pages as usize).await {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn release_working_pages(&mut self, this: Resource<Context>, num_pages: u32) -> Result<()> {
        let ctx = self.ctx().table.get(&this)?;
        context::release_working_pages(ctx.model_id, ctx.context_id, num_pages as usize)?;
        Ok(())
    }

    async fn working_page_token_count(&mut self, this: Resource<Context>) -> Result<u32> {
        let ctx = self.ctx().table.get(&this)?;
        Ok(context::working_page_token_count(ctx.model_id, ctx.context_id))
    }

    async fn truncate_working_page_tokens(&mut self, this: Resource<Context>, num_tokens: u32) -> Result<()> {
        let ctx = self.ctx().table.get(&this)?;
        let current = context::working_page_token_count(ctx.model_id, ctx.context_id);
        let new_count = current.saturating_sub(num_tokens);
        context::truncate_working_page_tokens(ctx.model_id, ctx.context_id, new_count).await?;
        Ok(())
    }

    async fn suspend(
        &mut self,
        this: Resource<Context>,
    ) -> Result<Result<(), String>> {
        let ctx = self.ctx().table.get(&this)?;
        match context::suspend(ctx.model_id, ctx.context_id).await {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn bid(&mut self, this: Resource<Context>, value: f64) -> Result<()> {
        let ctx = self.ctx().table.get(&this)?;
        let _ = context::bid(ctx.model_id, ctx.context_id, value).await;
        Ok(())
    }
}
