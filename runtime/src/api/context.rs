//! pie:core/context - Context resource for KV cache management

use crate::api::model::Model;
use crate::api::pie;
use crate::api::types::FutureBool;
use crate::context::{self, ContextId, LockId, Message};
use crate::instance::InstanceState;
use crate::model::ModelId;
use anyhow::Result;
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

/// Context resource - represents a KV cache context managed by the ContextManagerActor.
#[derive(Debug)]
pub struct Context {
    /// The context ID assigned by the ContextManagerActor
    pub context_id: ContextId,
    /// The model ID (for routing to the correct ContextManagerActor)
    pub model_id: ModelId,
    /// The user ID associated with this context
    pub user_id: u32,
    /// Currently held lock ID (if any)
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
        let user_id = 0u32; // TODO: Get from InstanceState

        let (tx, rx) = oneshot::channel();
        Message::Create {
            user_id,
            name,
            fill,
            response: tx,
        }
        .send(model_id)?;

        match rx.await? {
            Ok(context_id) => {
                let ctx = Context {
                    context_id,
                    model_id,
                    user_id,
                    lock_id: None,
                };
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

        let (tx, rx) = oneshot::channel();
        Message::Destroy {
            id: context_id,
            lock_id,
            response: tx,
        }
        .send(model_id)?;

        // Wait for response but ignore errors - destroy is void in WIT
        let _ = rx.await;
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
        let user_id = 0u32; // TODO: Get from InstanceState

        let (tx, rx) = oneshot::channel();
        Message::Lookup {
            user_id,
            name,
            response: tx,
        }
        .send(model_id)?;

        match rx.await? {
            Some(context_id) => {
                let ctx = Context {
                    context_id,
                    model_id,
                    user_id,
                    lock_id: None,
                };
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
        let user_id = ctx.user_id;

        let (tx, rx) = oneshot::channel();
        Message::Fork {
            id: context_id,
            user_id,
            new_name,
            response: tx,
        }
        .send(model_id)?;

        match rx.await? {
            Ok(new_context_id) => {
                let new_ctx = Context {
                    context_id: new_context_id,
                    model_id,
                    user_id,
                    lock_id: None,
                };
                Ok(Ok(self.ctx().table.push(new_ctx)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn drop(&mut self, this: Resource<Context>) -> Result<()> {
        // Note: This doesn't destroy the context in the actor, just the local resource
        self.ctx().table.delete(this)?;
        Ok(())
    }

    async fn acquire_lock(&mut self, this: Resource<Context>) -> Result<Resource<FutureBool>> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_id = ctx.model_id;

        let (tx, rx) = oneshot::channel();
        Message::AcquireLock {
            id: context_id,
            response: tx,
        }
        .send(model_id)?;

        // Create a channel to transform the LockId response into a bool
        let (bool_tx, bool_rx) = oneshot::channel();

        tokio::spawn(async move {
            if let Ok(lock_id) = rx.await {
                // Lock succeeded if lock_id != 0
                let _ = bool_tx.send(lock_id != 0);
            }
        });

        // Store the lock_id when the response comes back
        // Note: We need to update the Context resource with the lock_id separately
        // For now, we'll handle this in a simplified way
        let future_bool = FutureBool::new(bool_rx);
        Ok(self.ctx().table.push(future_bool)?)
    }

    async fn release_lock(&mut self, this: Resource<Context>) -> Result<Result<(), String>> {
        let ctx = self.ctx().table.get_mut(&this)?;
        let context_id = ctx.context_id;
        let model_id = ctx.model_id;
        let lock_id = ctx.lock_id.take().unwrap_or(0);

        Message::ReleaseLock {
            id: context_id,
            lock_id,
        }
        .send(model_id)?;

        Ok(Ok(()))
    }

    async fn tokens_per_page(&mut self, this: Resource<Context>) -> Result<u32> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_id = ctx.model_id;

        let (tx, rx) = oneshot::channel();
        Message::TokensPerPage {
            id: context_id,
            response: tx,
        }
        .send(model_id)?;

        Ok(rx.await?)
    }

    async fn model(&mut self, this: Resource<Context>) -> Result<Resource<Model>> {
        let ctx = self.ctx().table.get(&this)?;
        let model_id = ctx.model_id;

        // Get cached model directly - no message passing needed
        if let Some(m) = crate::model::get_model(model_id) {
            let model = Model {
                model_id: model_id,
                info: m.info,
                tokenizer: m.tokenizer,
            };
            return Ok(self.ctx().table.push(model)?);
        }

        anyhow::bail!("Model not found in cache")
    }

    async fn committed_page_count(&mut self, this: Resource<Context>) -> Result<u32> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_id = ctx.model_id;

        let (tx, rx) = oneshot::channel();
        Message::CommittedPageCount {
            id: context_id,
            response: tx,
        }
        .send(model_id)?;

        Ok(rx.await?)
    }

    async fn uncommitted_page_count(&mut self, this: Resource<Context>) -> Result<u32> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_id = ctx.model_id;
        let lock_id = ctx.lock_id.unwrap_or(0);

        let (tx, rx) = oneshot::channel();
        Message::GetBufferedTokens {
            id: context_id,
            lock_id,
            response: tx,
        }
        .send(model_id)?;

        let tokens = rx.await?;
        // Calculate page count from tokens
        // TODO: Get actual page size for proper calculation
        Ok((tokens.len() as u32 + 255) / 256) // Approximate page count
    }

    async fn commit_pages(
        &mut self,
        this: Resource<Context>,
        page_indices: Vec<u32>,
    ) -> Result<Result<(), String>> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_id = ctx.model_id;
        let lock_id = ctx.lock_id.unwrap_or(0);

        let (tx, rx) = oneshot::channel();
        Message::CommitPages {
            id: context_id,
            lock_id,
            page_indices,
            response: tx,
        }
        .send(model_id)?;

        match rx.await? {
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
        let context_id = ctx.context_id;
        let model_id = ctx.model_id;
        let lock_id = ctx.lock_id.unwrap_or(0);

        let (tx, rx) = oneshot::channel();
        Message::ReservePages {
            id: context_id,
            lock_id,
            num_pages,
            response: tx,
        }
        .send(model_id)?;

        match rx.await? {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn release_pages(
        &mut self,
        this: Resource<Context>,
        num_pages: u32,
    ) -> Result<()> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_id = ctx.model_id;
        let lock_id = ctx.lock_id.unwrap_or(0);

        Message::ReleasePages {
            id: context_id,
            lock_id,
            num_pages,
        }
        .send(model_id)?;

        Ok(())
    }

    async fn cursor(&mut self, this: Resource<Context>) -> Result<u32> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_id = ctx.model_id;
        let lock_id = ctx.lock_id.unwrap_or(0);

        let (tx, rx) = oneshot::channel();
        Message::GetCursor {
            id: context_id,
            lock_id,
            response: tx,
        }
        .send(model_id)?;

        Ok(rx.await?)
    }

    async fn set_cursor(
        &mut self,
        this: Resource<Context>,
        cursor: u32,
    ) -> Result<()> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_id = ctx.model_id;
        let lock_id = ctx.lock_id.unwrap_or(0);

        Message::SetCursor {
            id: context_id,
            lock_id,
            cursor,
        }
        .send(model_id)?;

        Ok(())
    }

    async fn buffered_tokens(&mut self, this: Resource<Context>) -> Result<Vec<u32>> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_id = ctx.model_id;
        let lock_id = ctx.lock_id.unwrap_or(0);

        let (tx, rx) = oneshot::channel();
        Message::GetBufferedTokens {
            id: context_id,
            lock_id,
            response: tx,
        }
        .send(model_id)?;

        Ok(rx.await?)
    }

    async fn set_buffered_tokens(
        &mut self,
        this: Resource<Context>,
        tokens: Vec<u32>,
    ) -> Result<()> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_id = ctx.model_id;
        let lock_id = ctx.lock_id.unwrap_or(0);

        Message::SetBufferedTokens {
            id: context_id,
            lock_id,
            tokens,
        }
        .send(model_id)?;

        Ok(())
    }

    async fn append_buffered_tokens(
        &mut self,
        this: Resource<Context>,
        tokens: Vec<u32>,
    ) -> Result<()> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_id = ctx.model_id;
        let lock_id = ctx.lock_id.unwrap_or(0);

        Message::AppendBufferedTokens {
            id: context_id,
            lock_id,
            tokens,
        }
        .send(model_id)?;

        Ok(())
    }
}
