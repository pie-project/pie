//! pie:core/context - Context resource for KV cache management

use crate::api::model::Model;
use crate::api::pie;
use crate::context::{self, ContextId};
use crate::linker::InstanceState;
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

        match context::create_owned(model_id, Some(process_id)).await {
            Ok(context_id) => {
                self.track_context(model_id, context_id);
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

        match context::open(model_id, username, name).await {
            Ok(context_id) => {
                // Opened (forked-from-snapshot) contexts are tracked for auto-cleanup
                self.track_context(model_id, context_id);
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

        match context::fork(model_id, context_id).await {
            Ok(new_context_id) => {
                self.track_context(model_id, new_context_id);
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

        match context::save(model_id, context_id, username, name).await {
            Ok(()) => Ok(Ok(())),
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

        match context::snapshot(model_id, context_id, username).await {
            Ok(name) => Ok(Ok(name)),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn destroy(&mut self, this: Resource<Context>) -> Result<()> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_id = ctx.model_id;

        self.untrack_context(model_id, context_id);
        let _ = context::destroy(model_id, context_id, false).await;
        self.ctx().table.delete(this)?;
        Ok(())
    }

    async fn drop(&mut self, this: Resource<Context>) -> Result<()> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_id = ctx.model_id;
        // Only destroy anonymous contexts (ones still tracked).
        // Named/saved contexts survive handle drop.
        if self.untrack_context(model_id, context_id) {
            let _ = context::destroy(model_id, context_id, true).await;
        }
        self.ctx().table.delete(this)?;
        Ok(())
    }

    async fn tokens_per_page(&mut self, this: Resource<Context>) -> Result<u32> {
        let ctx = self.ctx().table.get(&this)?;
        Ok(context::tokens_per_page(ctx.model_id, ctx.context_id))
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

    async fn uncommitted_page_count(&mut self, this: Resource<Context>) -> Result<u32> {
        let ctx = self.ctx().table.get(&this)?;
        let tokens = context::get_buffered_tokens(ctx.model_id, ctx.context_id);
        let page_size = context::tokens_per_page(ctx.model_id, ctx.context_id);
        Ok((tokens.len() as u32 + page_size - 1) / page_size)
    }

    async fn commit_pages(
        &mut self,
        this: Resource<Context>,
        page_indices: Vec<u32>,
    ) -> Result<Result<(), String>> {
        let ctx = self.ctx().table.get(&this)?;
        match context::commit_pages(ctx.model_id, ctx.context_id, page_indices).await {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn reserve_pages(
        &mut self,
        this: Resource<Context>,
        num_pages: u32,
    ) -> Result<Result<(), String>> {
        let ctx = self.ctx().table.get(&this)?;
        match context::reserve_pages(ctx.model_id, ctx.context_id, num_pages).await {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn release_pages(&mut self, this: Resource<Context>, num_pages: u32) -> Result<()> {
        let ctx = self.ctx().table.get(&this)?;
        context::release_pages(ctx.model_id, ctx.context_id, num_pages)?;
        Ok(())
    }

    async fn cursor(&mut self, this: Resource<Context>) -> Result<u32> {
        let ctx = self.ctx().table.get(&this)?;
        Ok(context::get_cursor(ctx.model_id, ctx.context_id))
    }

    async fn set_cursor(&mut self, this: Resource<Context>, cursor: u32) -> Result<()> {
        let ctx = self.ctx().table.get(&this)?;
        context::set_cursor(ctx.model_id, ctx.context_id, cursor)?;
        Ok(())
    }

    async fn buffered_tokens(&mut self, this: Resource<Context>) -> Result<Vec<u32>> {
        let ctx = self.ctx().table.get(&this)?;
        Ok(context::get_buffered_tokens(ctx.model_id, ctx.context_id))
    }

    async fn set_buffered_tokens(&mut self, this: Resource<Context>, tokens: Vec<u32>) -> Result<()> {
        let ctx = self.ctx().table.get(&this)?;
        context::set_buffered_tokens(ctx.model_id, ctx.context_id, tokens)?;
        Ok(())
    }

    async fn append_buffered_tokens(&mut self, this: Resource<Context>, tokens: Vec<u32>) -> Result<()> {
        let ctx = self.ctx().table.get(&this)?;
        context::append_buffered_tokens(ctx.model_id, ctx.context_id, tokens)?;
        Ok(())
    }

    async fn last_position(&mut self, this: Resource<Context>) -> Result<Option<u32>> {
        let ctx = self.ctx().table.get(&this)?;
        Ok(context::last_position(ctx.model_id, ctx.context_id))
    }
}
