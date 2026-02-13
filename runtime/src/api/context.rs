//! pie:core/context - Context resource for KV cache management

use crate::api::model::Model;
use crate::api::pie;
use crate::api::types::FutureBool;
use crate::context::{self, ContextId, LockId};
use crate::linker::InstanceState;
use crate::model::ModelId;
use anyhow::Result;
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

/// Context resource - represents a KV cache context managed by the ContextManager.
#[derive(Debug)]
pub struct Context {
    /// The context ID assigned by the ContextManager.
    pub context_id: ContextId,
    /// The model ID (for routing to the correct ContextManager).
    pub model_id: ModelId,
    /// Currently held lock ID (if any).
    pub lock_id: Option<LockId>,
}

impl pie::core::context::Host for InstanceState {}

impl pie::core::context::HostContext for InstanceState {
    async fn create(
        &mut self,
        model: Resource<Model>,
        name: String,
        fill: Option<Vec<u32>>,
    ) -> Result<Result<Resource<Context>, String>> {
        let model = self.ctx().table.get(&model)?;
        let model_id = model.model_id;
        let username = self.get_username();

        match context::create(model_id, username, name, fill).await {
            Ok(context_id) => {
                let ctx = Context { context_id, model_id, lock_id: None };
                Ok(Ok(self.ctx().table.push(ctx)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn destroy(&mut self, this: Resource<Context>) -> Result<()> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_id = ctx.model_id;
        let lock_id = ctx.lock_id.unwrap_or(0);

        let _ = context::destroy(model_id, context_id, lock_id).await;
        self.ctx().table.delete(this)?;
        Ok(())
    }

    async fn lookup(
        &mut self,
        model: Resource<Model>,
        name: String,
    ) -> Result<Option<Resource<Context>>> {
        let model = self.ctx().table.get(&model)?;
        let model_id = model.model_id;
        let username = self.get_username();

        match context::lookup(model_id, username, name).await {
            Some(context_id) => {
                let ctx = Context { context_id, model_id, lock_id: None };
                Ok(Some(self.ctx().table.push(ctx)?))
            }
            None => Ok(None),
        }
    }

    async fn fork(
        &mut self,
        this: Resource<Context>,
        new_name: String,
    ) -> Result<Result<Resource<Context>, String>> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_id = ctx.model_id;
        let username = self.get_username();

        match context::fork(model_id, context_id, username, new_name).await {
            Ok(new_context_id) => {
                let new_ctx = Context { context_id: new_context_id, model_id, lock_id: None };
                Ok(Ok(self.ctx().table.push(new_ctx)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn drop(&mut self, this: Resource<Context>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }

    async fn acquire_lock(&mut self, this: Resource<Context>) -> Result<Resource<FutureBool>> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_id = ctx.model_id;

        // Acquire the lock synchronously so we can store the lock_id
        let lock_id = context::acquire_lock(model_id, context_id);
        let success = lock_id != 0;

        if success {
            let ctx = self.ctx().table.get_mut(&this)?;
            ctx.lock_id = Some(lock_id);
        }

        // Return a pre-resolved future
        let (bool_tx, bool_rx) = oneshot::channel();
        let _ = bool_tx.send(success);

        let future_bool = FutureBool::new(bool_rx);
        Ok(self.ctx().table.push(future_bool)?)
    }

    async fn release_lock(&mut self, this: Resource<Context>) -> Result<Result<(), String>> {
        let ctx = self.ctx().table.get_mut(&this)?;
        let context_id = ctx.context_id;
        let model_id = ctx.model_id;
        let lock_id = ctx.lock_id.take().unwrap_or(0);

        context::release_lock(model_id, context_id, lock_id)?;
        Ok(Ok(()))
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
        let count = context::committed_page_count(ctx.model_id, ctx.context_id);
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/pie_api_debug.log") {
            use std::io::Write;
            let _ = writeln!(f, "[API] committed_page_count ctx={} -> {}", ctx.context_id, count);
        }
        Ok(count)
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
        match context::commit_pages(ctx.model_id, ctx.context_id, ctx.lock_id.unwrap_or(0), page_indices).await {
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
        match context::reserve_pages(ctx.model_id, ctx.context_id, ctx.lock_id.unwrap_or(0), num_pages).await {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn release_pages(&mut self, this: Resource<Context>, num_pages: u32) -> Result<()> {
        let ctx = self.ctx().table.get(&this)?;
        context::release_pages(ctx.model_id, ctx.context_id, ctx.lock_id.unwrap_or(0), num_pages)?;
        Ok(())
    }

    async fn cursor(&mut self, this: Resource<Context>) -> Result<u32> {
        let ctx = self.ctx().table.get(&this)?;
        Ok(context::get_cursor(ctx.model_id, ctx.context_id))
    }

    async fn set_cursor(&mut self, this: Resource<Context>, cursor: u32) -> Result<()> {
        let ctx = self.ctx().table.get(&this)?;
        context::set_cursor(ctx.model_id, ctx.context_id, ctx.lock_id.unwrap_or(0), cursor)?;
        Ok(())
    }

    async fn buffered_tokens(&mut self, this: Resource<Context>) -> Result<Vec<u32>> {
        let ctx = self.ctx().table.get(&this)?;
        Ok(context::get_buffered_tokens(ctx.model_id, ctx.context_id))
    }

    async fn set_buffered_tokens(&mut self, this: Resource<Context>, tokens: Vec<u32>) -> Result<()> {
        let ctx = self.ctx().table.get(&this)?;
        context::set_buffered_tokens(ctx.model_id, ctx.context_id, ctx.lock_id.unwrap_or(0), tokens)?;
        Ok(())
    }

    async fn append_buffered_tokens(&mut self, this: Resource<Context>, tokens: Vec<u32>) -> Result<()> {
        let ctx = self.ctx().table.get(&this)?;
        context::append_buffered_tokens(ctx.model_id, ctx.context_id, ctx.lock_id.unwrap_or(0), tokens)?;
        Ok(())
    }

    async fn last_position(&mut self, this: Resource<Context>) -> Result<Option<u32>> {
        let ctx = self.ctx().table.get(&this)?;
        Ok(context::last_position(ctx.model_id, ctx.context_id))
    }
}
