//! `pie:inferlet/working-set` — KV working-set host resource.
//!
//! The WASM resource type is [`crate::store::kv::working_set::KvWorkingSet`],
//! a thin handle (model, driver, WorkingSetId); every substantive operation
//! delegates to the per-(model, driver) [`KvStore`] resolved through
//! `store::registry`. Lock the store synchronously and release before any
//! await (see `store::registry` docs).
//!
//! reserve is purely logical; discard/fork/slice are ordered on a pipeline.
//! All pipelines share the single per-driver sequencer queue, so host-side
//! inline execution here IS their queue position relative to this instance's
//! submissions.

use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

use crate::inferlet::ProcessCtx;
use crate::inferlet::host::pie;
use crate::inferlet::host::pipeline::Pipeline;
use crate::store::kv::working_set::KvWorkingSet;
use crate::store::registry as store_registry;

type WitRange = pie::inferlet::working_set::PageRange;

fn scoped_working_set(
    ctx: &mut ProcessCtx,
    this: &Resource<KvWorkingSet>,
    on: &Resource<Pipeline>,
) -> Result<Result<KvWorkingSet, String>> {
    let (scope, failure) = {
        let pipeline = ctx.ctx().table.get(on)?;
        (pipeline.scope.clone(), pipeline.failure.clone())
    };
    if scope.is_closed() {
        return Ok(Err("pipeline is closed".to_string()));
    }
    let ws = ctx.ctx().table.get(this)?.clone();
    if let Err(owner) = ws.claim_pipeline_scope(&scope) {
        return Ok(Err(format!(
            "working set is scoped to pipeline {owner:#x}, not supplied pipeline {:#x}",
            scope.id()
        )));
    }
    if let Some(reason) = failure.lock().unwrap().clone() {
        return Ok(Err(format!("pipeline failed: {reason}")));
    }
    Ok(Ok(ws))
}

impl pie::inferlet::working_set::HostKvWorkingSet for ProcessCtx {
    async fn new(&mut self) -> Result<Resource<KvWorkingSet>> {
        crate::inferlet::process::preemption::honor(self).await?;
        // Single-model runtime: bind the one model (index 0), driver 0.
        let stores = store_registry::get(0, 0);
        let id = store_registry::with_kv_lock(&stores.kv, "host-working-set", |kv| {
            kv.create_working_set()
        });
        let ws = KvWorkingSet::new(0, 0, id, stores.kv_page_size);
        self.register_kv_working_set(0, 0, id);
        Ok(self.ctx().table.push(ws)?)
    }

    async fn page_size(&mut self, this: Resource<KvWorkingSet>) -> Result<u32> {
        crate::inferlet::process::preemption::honor(self).await?;
        Ok(self.ctx().table.get(&this)?.page_size)
    }

    async fn page_len(&mut self, this: Resource<KvWorkingSet>) -> Result<u32> {
        crate::inferlet::process::preemption::honor(self).await?;
        let ws = self.ctx().table.get(&this)?.clone();
        let stores = store_registry::get(ws.model, ws.driver as usize);
        let len =
            store_registry::with_kv_lock(&stores.kv, "host-working-set", |kv| kv.page_len(ws.id));
        Ok(len.map_err(anyhow::Error::from)? as u32)
    }

    async fn reserve(
        &mut self,
        this: Resource<KvWorkingSet>,
        pages: u32,
    ) -> Result<Result<WitRange, String>> {
        // Strict admission: even a logical page claim counts as pooled
        // demand the contention orchestrator reasons about.
        crate::inferlet::process::ensure_execution_admitted(self).await;
        crate::inferlet::process::preemption::honor(self).await?;
        let ws = self.ctx().table.get(&this)?.clone();
        let stores = store_registry::get(ws.model, ws.driver as usize);
        let range = store_registry::with_kv_lock(&stores.kv, "host-working-set", |kv| {
            kv.reserve(ws.id, pages as u64)
        });
        Ok(range
            .map(|r| WitRange {
                start: r.start as u32,
                len: (r.end - r.start) as u32,
            })
            .map_err(|e| e.to_string()))
    }

    async fn update_index(
        &mut self,
        this: Resource<KvWorkingSet>,
        key: Vec<u8>,
    ) -> Result<Result<(), String>> {
        crate::inferlet::process::preemption::honor(self).await?;
        let ws = self.ctx().table.get(&this)?.clone();
        if !ws.is_settled() {
            return Ok(Err(
                "working set cannot be indexed while an operation is in flight".to_string(),
            ));
        }
        let stores = store_registry::get(ws.model, ws.driver as usize);
        let result = store_registry::with_kv_lock(&stores.kv, "host-working-set", |kv| {
            kv.update_index(key, ws.id)
        });
        match result {
            Ok(freed) => {
                if freed != 0
                    && let Some(orchestrator) = crate::store::reclaim::contention()
                {
                    orchestrator.on_blocks_freed();
                }
                Ok(Ok(()))
            }
            Err(error) => Ok(Err(error.to_string())),
        }
    }

    async fn from_index(
        &mut self,
        key: Vec<u8>,
    ) -> Result<Result<Option<Resource<KvWorkingSet>>, String>> {
        crate::inferlet::process::preemption::honor(self).await?;
        let stores = store_registry::get(0, 0);
        let indexed =
            store_registry::with_kv_lock(&stores.kv, "host-working-set", |kv| kv.from_index(&key));
        match indexed {
            Ok(Some(id)) => {
                let ws = KvWorkingSet::new(0, 0, id, stores.kv_page_size);
                self.register_kv_working_set(0, 0, id);
                Ok(Ok(Some(self.ctx().table.push(ws)?)))
            }
            Ok(None) => Ok(Ok(None)),
            Err(error) => Ok(Err(error.to_string())),
        }
    }

    async fn remove_index(&mut self, key: Vec<u8>) -> Result<Result<bool, String>> {
        crate::inferlet::process::preemption::honor(self).await?;
        let stores = store_registry::get(0, 0);
        let removed = store_registry::with_kv_lock(&stores.kv, "host-working-set", |kv| {
            kv.remove_index(&key)
        });
        match removed {
            Ok((removed, freed)) => {
                if freed != 0
                    && let Some(orchestrator) = crate::store::reclaim::contention()
                {
                    orchestrator.on_blocks_freed();
                }
                Ok(Ok(removed))
            }
            Err(error) => Ok(Err(error.to_string())),
        }
    }

    async fn discard(
        &mut self,
        this: Resource<KvWorkingSet>,
        on: Resource<Pipeline>,
        ranges: Vec<WitRange>,
    ) -> Result<Result<(), String>> {
        crate::inferlet::process::preemption::contention_gate(self).await?;
        let ws = match scoped_working_set(self, &this, &on)? {
            Ok(ws) => ws,
            Err(error) => return Ok(Err(error)),
        };
        let ranges: Vec<std::ops::Range<u64>> = ranges
            .into_iter()
            .map(|r| r.start as u64..(r.start as u64 + r.len as u64))
            .collect();
        let stores = store_registry::get(ws.model, ws.driver as usize);
        let out = store_registry::with_kv_lock(&stores.kv, "host-working-set", |kv| {
            let epoch = kv.current_epoch();
            let out = kv.discard(ws.id, &ranges, epoch).map_err(|e| e.to_string());
            kv.retire_idle();
            out
        }); // store lock released before the contention drain re-locks pools.
        if out.is_ok() {
            if let Some(orchestrator) = crate::store::reclaim::contention() {
                orchestrator.on_blocks_freed();
            }
        }
        Ok(out)
    }

    async fn fork(
        &mut self,
        this: Resource<KvWorkingSet>,
        on: Resource<Pipeline>,
    ) -> Result<Result<Resource<KvWorkingSet>, String>> {
        crate::inferlet::process::preemption::contention_gate(self).await?;
        let ws = match scoped_working_set(self, &this, &on)? {
            Ok(ws) => ws,
            Err(error) => return Ok(Err(error)),
        };
        let stores = store_registry::get(ws.model, ws.driver as usize);
        let forked =
            store_registry::with_kv_lock(&stores.kv, "host-working-set", |kv| kv.fork(ws.id));
        match forked {
            Ok(id) => {
                // A distinct working-set id gets its OWN fresh lifecycle —
                // never a clone of the parent's, which would (a) share its
                // release-once guard for an unrelated id and (b) release the
                // WRONG working set when the last clone drops.
                let child = ws.forked(id);
                self.register_kv_working_set(ws.model, ws.driver, id);
                Ok(Ok(self.ctx().table.push(child)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn slice(
        &mut self,
        this: Resource<KvWorkingSet>,
        on: Resource<Pipeline>,
        range: WitRange,
    ) -> Result<Result<Resource<KvWorkingSet>, String>> {
        crate::inferlet::process::preemption::contention_gate(self).await?;
        let ws = match scoped_working_set(self, &this, &on)? {
            Ok(ws) => ws,
            Err(error) => return Ok(Err(error)),
        };
        let stores = store_registry::get(ws.model, ws.driver as usize);
        let sliced = store_registry::with_kv_lock(&stores.kv, "host-working-set", |kv| {
            kv.slice(
                ws.id,
                range.start as u64..(range.start as u64 + range.len as u64),
            )
        });
        match sliced {
            Ok(id) => {
                // See `fork`: a fresh id always gets a fresh lifecycle.
                let child = ws.forked(id);
                self.register_kv_working_set(ws.model, ws.driver, id);
                Ok(Ok(self.ctx().table.push(child)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn copy_into(
        &mut self,
        this: Resource<KvWorkingSet>,
        on: Resource<Pipeline>,
        dst_page_ids: Vec<u32>,
        dst_tok_idx: Vec<u32>,
        src_page_ids: Vec<u32>,
        src_tok_idx: Vec<u32>,
    ) -> Result<Result<(), String>> {
        crate::inferlet::process::preemption::contention_gate(self).await?;
        crate::pipeline::fire::working_set_copy_into(
            self,
            this,
            on,
            dst_page_ids,
            dst_tok_idx,
            src_page_ids,
            src_tok_idx,
        )
        .await
    }

    async fn drop(&mut self, this: Resource<KvWorkingSet>) -> Result<()> {
        crate::inferlet::process::preemption::honor(self).await?;
        // `release` performs the exact release/retire/contention-drain
        // sequence and marks the shared
        // lifecycle done; `ws`'s own drop just below (and the fallback
        // `KvLifecycle::drop` it would otherwise trigger) is then a no-op.
        let ws = self.ctx().table.delete(this)?;
        self.unregister_kv_working_set(ws.model, ws.driver, ws.id);
        ws.release();
        Ok(())
    }
}
