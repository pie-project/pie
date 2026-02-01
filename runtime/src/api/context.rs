//! pie:core/context - Context resource for KV cache management

use crate::api::model::Model;
use crate::api::pie;
use crate::api::types::FutureBool;
use crate::context::{self, ContextId, LockId, Message};
use crate::instance::InstanceState;
use anyhow::Result;
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

/// Context resource - represents a KV cache context managed by the ContextActor.
#[derive(Debug)]
pub struct Context {
    /// The context ID assigned by the ContextActor
    pub context_id: ContextId,
    /// The model service index (for routing to the correct ContextActor)
    pub model_idx: usize,
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
        let model_idx = model.service_id;
        let user_id = 0u32; // TODO: Get from InstanceState

        let (tx, rx) = oneshot::channel();
        Message::Create {
            user_id,
            name,
            fill,
            response: tx,
        }
        .send(model_idx)?;

        match rx.await? {
            Ok(context_id) => {
                let ctx = Context {
                    context_id,
                    model_idx,
                    user_id,
                    lock_id: None,
                };
                Ok(Ok(self.ctx().table.push(ctx)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn destroy(&mut self, this: Resource<Context>) -> Result<Result<(), String>> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_idx = ctx.model_idx;
        let lock_id = ctx.lock_id.unwrap_or(0);

        let (tx, rx) = oneshot::channel();
        Message::Destroy {
            id: context_id,
            lock_id,
            response: tx,
        }
        .send(model_idx)?;

        match rx.await? {
            Ok(()) => {
                self.ctx().table.delete(this)?;
                Ok(Ok(()))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn get(
        &mut self,
        model: Resource<Model>,
        name: String,
    ) -> Result<Option<Resource<Context>>> {
        let model = self.ctx().table.get(&model)?;
        let model_idx = model.service_id;
        let user_id = 0u32; // TODO: Get from InstanceState

        let (tx, rx) = oneshot::channel();
        Message::Get {
            user_id,
            name,
            response: tx,
        }
        .send(model_idx)?;

        match rx.await? {
            Some(context_id) => {
                let ctx = Context {
                    context_id,
                    model_idx,
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
        let model_idx = ctx.model_idx;
        let user_id = ctx.user_id;

        let (tx, rx) = oneshot::channel();
        Message::Fork {
            id: context_id,
            user_id,
            new_name,
            response: tx,
        }
        .send(model_idx)?;

        match rx.await? {
            Ok(new_context_id) => {
                let new_ctx = Context {
                    context_id: new_context_id,
                    model_idx,
                    user_id,
                    lock_id: None,
                };
                Ok(Ok(self.ctx().table.push(new_ctx)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn join(
        &mut self,
        _this: Resource<Context>,
        _other: Resource<Context>,
    ) -> Result<Result<(), String>> {
        // TODO: Implement join - merge contexts
        Ok(Ok(()))
    }

    async fn drop(&mut self, this: Resource<Context>) -> Result<()> {
        // Note: This doesn't destroy the context in the actor, just the local resource
        self.ctx().table.delete(this)?;
        Ok(())
    }

    async fn lock(&mut self, this: Resource<Context>) -> Result<Resource<FutureBool>> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_idx = ctx.model_idx;

        let (tx, rx) = oneshot::channel();
        Message::Lock {
            id: context_id,
            response: tx,
        }
        .send(model_idx)?;

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

    async fn unlock(&mut self, this: Resource<Context>) -> Result<Result<(), String>> {
        let ctx = self.ctx().table.get_mut(&this)?;
        let context_id = ctx.context_id;
        let model_idx = ctx.model_idx;
        let lock_id = ctx.lock_id.take().unwrap_or(0);

        Message::Unlock {
            id: context_id,
            lock_id,
        }
        .send(model_idx)?;

        Ok(Ok(()))
    }

    async fn page_size(&mut self, this: Resource<Context>) -> Result<u32> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_idx = ctx.model_idx;

        let (tx, rx) = oneshot::channel();
        Message::PageSize {
            id: context_id,
            response: tx,
        }
        .send(model_idx)?;

        Ok(rx.await?)
    }

    async fn num_total_pages(&mut self, this: Resource<Context>) -> Result<u32> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_idx = ctx.model_idx;

        let (tx, rx) = oneshot::channel();
        Message::NumTotalPages {
            id: context_id,
            response: tx,
        }
        .send(model_idx)?;

        Ok(rx.await?)
    }

    async fn num_total_tokens(&mut self, this: Resource<Context>) -> Result<u32> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_idx = ctx.model_idx;
        let lock_id = ctx.lock_id.unwrap_or(0);

        let (tx, rx) = oneshot::channel();
        Message::NumTotalTokens {
            id: context_id,
            lock_id,
            response: tx,
        }
        .send(model_idx)?;

        Ok(rx.await?)
    }

    async fn commit_pages(
        &mut self,
        this: Resource<Context>,
        indices: Vec<u32>,
    ) -> Result<Result<(), String>> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_idx = ctx.model_idx;
        let lock_id = ctx.lock_id.unwrap_or(0);

        let (tx, rx) = oneshot::channel();
        Message::CommitPages {
            id: context_id,
            lock_id,
            indices,
            response: tx,
        }
        .send(model_idx)?;

        match rx.await? {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn allocate_pages(
        &mut self,
        this: Resource<Context>,
        num_pages: u32,
    ) -> Result<Result<(), String>> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_idx = ctx.model_idx;
        let lock_id = ctx.lock_id.unwrap_or(0);

        let (tx, rx) = oneshot::channel();
        Message::AllocatePages {
            id: context_id,
            lock_id,
            num_pages,
            response: tx,
        }
        .send(model_idx)?;

        match rx.await? {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn free_pages(
        &mut self,
        this: Resource<Context>,
        num_pages: u32,
    ) -> Result<Result<(), String>> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_idx = ctx.model_idx;
        let lock_id = ctx.lock_id.unwrap_or(0);

        let (tx, rx) = oneshot::channel();
        Message::FreePages {
            id: context_id,
            lock_id,
            num_pages,
            response: tx,
        }
        .send(model_idx)?;

        match rx.await? {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn pointer(&mut self, this: Resource<Context>) -> Result<u32> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_idx = ctx.model_idx;
        let lock_id = ctx.lock_id.unwrap_or(0);

        let (tx, rx) = oneshot::channel();
        Message::GetPointer {
            id: context_id,
            lock_id,
            response: tx,
        }
        .send(model_idx)?;

        Ok(rx.await?)
    }

    async fn set_pointer(
        &mut self,
        this: Resource<Context>,
        pointer: u32,
    ) -> Result<Result<(), String>> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_idx = ctx.model_idx;
        let lock_id = ctx.lock_id.unwrap_or(0);

        let (tx, rx) = oneshot::channel();
        Message::SetPointer {
            id: context_id,
            lock_id,
            pointer,
            response: tx,
        }
        .send(model_idx)?;

        match rx.await? {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn staged_tokens(&mut self, this: Resource<Context>) -> Result<Vec<u32>> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_idx = ctx.model_idx;
        let lock_id = ctx.lock_id.unwrap_or(0);

        let (tx, rx) = oneshot::channel();
        Message::GetUncommittedTokens {
            id: context_id,
            lock_id,
            response: tx,
        }
        .send(model_idx)?;

        let token_infos = rx.await?;
        // Extract just the token IDs from TokenInfo
        Ok(token_infos.into_iter().map(|t| t.token).collect())
    }

    async fn set_staged_tokens(
        &mut self,
        this: Resource<Context>,
        tokens: Vec<u32>,
    ) -> Result<Result<(), String>> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let model_idx = ctx.model_idx;
        let lock_id = ctx.lock_id.unwrap_or(0);

        // Convert tokens to TokenInfo (with default position/mask)
        let token_infos: Vec<context::TokenInfo> = tokens
            .into_iter()
            .enumerate()
            .map(|(i, token)| context::TokenInfo {
                token,
                position: i as u32,
                mask: crate::brle::Brle::new(0),
                adapter: None,
            })
            .collect();

        let (tx, rx) = oneshot::channel();
        Message::SetUncommittedTokens {
            id: context_id,
            lock_id,
            tokens: token_infos,
            response: tx,
        }
        .send(model_idx)?;

        match rx.await? {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }
}
