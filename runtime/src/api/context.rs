//! pie:core/context - Context resource for KV cache management

use crate::api::pie;
use crate::context::{self, ContextId};
use crate::inference::StagedBatch;
use crate::instance::InstanceState;
use anyhow::Result;
use std::sync::OnceLock;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

/// Context resource - represents a KV cache context managed by the ContextManager.
#[derive(Debug)]
pub struct Context {
    /// The context ID assigned by the ContextManager.
    pub context_id: ContextId,
    /// Lazily-populated speculator handle for this ctx's device.
    /// Populated on first `pass.context()` call from the inferlet so
    /// `execute()` skips the global REGISTRY lookup on every iteration.
    /// `Some(None)` means speculation is disabled; the OnceLock
    /// distinguishes that from "not yet initialized".
    pub spec: OnceLock<Option<StagedBatch>>,
}

impl pie::core::context::Host for InstanceState {}

impl pie::core::context::HostContext for InstanceState {
    async fn create(&mut self) -> Result<Result<Resource<Context>, String>> {
        let process_id = self.id();

        match context::create(process_id).await {
            Ok(context_id) => {
                let ctx = Context {
                    context_id,
                    spec: OnceLock::new(),
                };
                Ok(Ok(self.ctx().table.push(ctx)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn open(&mut self, name: String) -> Result<Result<Resource<Context>, String>> {
        let username = self.get_username();
        let process_id = self.id();

        let snapshot_id = match context::lookup(username, name).await {
            Ok(id) => id,
            Err(e) => return Ok(Err(e.to_string())),
        };
        match context::fork(snapshot_id, process_id).await {
            Ok(context_id) => {
                let ctx = Context {
                    context_id,
                    spec: OnceLock::new(),
                };
                Ok(Ok(self.ctx().table.push(ctx)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn take(&mut self, name: String) -> Result<Result<Resource<Context>, String>> {
        let username = self.get_username();
        let process_id = self.id();

        match context::take(username, name, process_id).await {
            Ok(context_id) => {
                let ctx = Context {
                    context_id,
                    spec: OnceLock::new(),
                };
                Ok(Ok(self.ctx().table.push(ctx)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn delete(&mut self, name: String) -> Result<Result<(), String>> {
        let username = self.get_username();

        match context::delete(username, name).await {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn fork(&mut self, this: Resource<Context>) -> Result<Result<Resource<Context>, String>> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let process_id = self.id();

        match context::fork(context_id, process_id).await {
            Ok(new_context_id) => {
                let new_ctx = Context {
                    context_id: new_context_id,
                    spec: OnceLock::new(),
                };
                Ok(Ok(self.ctx().table.push(new_ctx)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn save(&mut self, this: Resource<Context>, name: String) -> Result<Result<(), String>> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let username = self.get_username();

        match context::save(context_id, username, Some(name)).await {
            Ok(_) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn snapshot(&mut self, this: Resource<Context>) -> Result<Result<String, String>> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        let username = self.get_username();

        match context::save(context_id, username, None).await {
            Ok(Some(name)) => Ok(Ok(name)),
            Ok(None) => Ok(Err("snapshot returned no name".into())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn destroy(&mut self, this: Resource<Context>) -> Result<()> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;

        // Drop any speculation chain BEFORE the context store
        // releases the ctx. Order matters: the chain queue holds
        // ForwardPassOutput channels keyed by the ctx; we want them
        // dropped while the ctx still exists so any in-flight
        // BatchComplete handler that races us sees a coherent state
        // (chain gone → output silently discarded).
        crate::inference::invalidate_speculation_for_ctx(context_id);
        let _ = context::destroy(context_id).await;
        self.ctx().table.delete(this)?;
        Ok(())
    }

    async fn drop(&mut self, this: Resource<Context>) -> Result<()> {
        // Pull the ctx id BEFORE we release the table entry — the
        // resource handle is invalid after delete.
        let context_id = {
            let ctx = self.ctx().table.get(&this)?;
            ctx.context_id
        };
        // Drop the speculation chain associated with this ctx. The
        // ctx itself stays alive for the process's wider cleanup
        // path (InstanceState::drop → unregister_process); only the
        // speculative state attached to it goes away here. Idempotent
        // if no chain exists.
        crate::inference::invalidate_speculation_for_ctx(context_id);
        self.ctx().table.delete(this)?;
        Ok(())
    }

    async fn tokens_per_page(&mut self, _this: Resource<Context>) -> Result<u32> {
        Ok(context::tokens_per_page())
    }

    async fn committed_page_count(&mut self, this: Resource<Context>) -> Result<u32> {
        let ctx = self.ctx().table.get(&this)?;
        Ok(context::committed_page_count(ctx.context_id))
    }

    async fn working_page_count(&mut self, this: Resource<Context>) -> Result<u32> {
        let ctx = self.ctx().table.get(&this)?;
        Ok(context::working_page_count(ctx.context_id))
    }

    async fn commit_working_pages(
        &mut self,
        this: Resource<Context>,
        num_pages: u32,
    ) -> Result<Result<(), String>> {
        let ctx = self.ctx().table.get(&this)?;
        match context::commit_working_pages(ctx.context_id, num_pages as usize).await {
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
        match context::reserve_working_pages(ctx.context_id, num_pages as usize).await {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn release_working_pages(
        &mut self,
        this: Resource<Context>,
        num_pages: u32,
    ) -> Result<()> {
        let ctx = self.ctx().table.get(&this)?;
        context::release_working_pages(ctx.context_id, num_pages as usize)?;
        Ok(())
    }

    async fn working_page_token_count(&mut self, this: Resource<Context>) -> Result<u32> {
        let ctx = self.ctx().table.get(&this)?;
        Ok(context::working_page_token_count(ctx.context_id))
    }

    async fn truncate_working_page_tokens(
        &mut self,
        this: Resource<Context>,
        num_tokens: u32,
    ) -> Result<()> {
        let ctx = self.ctx().table.get(&this)?;
        context::truncate_working_page_tokens(ctx.context_id, num_tokens).await?;
        Ok(())
    }

    async fn suspend(&mut self, this: Resource<Context>) -> Result<Result<(), String>> {
        let ctx = self.ctx().table.get(&this)?;
        let context_id = ctx.context_id;
        // Eviction safety: drop the speculation chain BEFORE the
        // context store releases the ctx's GPU pages. Chain entries
        // hold fingerprints referencing KV positions that won't exist
        // after the page reclaim; the next inferlet hit would read
        // from a stale slot. See SPECULATIVE_EXECUTION_DESIGN.md
        // §"Eviction safety".
        crate::inference::invalidate_speculation_for_ctx(context_id);
        match context::suspend(context_id).await {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }
}
