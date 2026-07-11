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

/// Auto-retained prefix-cache root cap (`PIE_KV_CACHE_ROOTS_MAX`, default
/// 256; `0` disables retention on release). The contention ladder's rung 1
/// reclaims retained roots the moment memory is needed, so the cap bounds
/// metadata, not pressure behavior.
fn cache_roots_max() -> usize {
    static MAX: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MAX.get_or_init(|| {
        std::env::var("PIE_KV_CACHE_ROOTS_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256)
    })
}

impl pie::inferlet::working_set::HostKvWorkingSet for ProcessCtx {
    async fn new(&mut self) -> Result<Resource<KvWorkingSet>> {
        // Single-model runtime: bind the one model (index 0), driver 0.
        let stores = store_registry::get(0, 0);
        let id = stores.kv.lock().unwrap().create_working_set();
        let ws = KvWorkingSet {
            model: 0,
            driver: 0,
            id,
            page_size: stores.kv_page_size,
        };
        Ok(self.ctx().table.push(ws)?)
    }

    async fn page_size(&mut self, this: Resource<KvWorkingSet>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.page_size)
    }

    async fn page_len(&mut self, this: Resource<KvWorkingSet>) -> Result<u32> {
        let ws = *self.ctx().table.get(&this)?;
        let stores = store_registry::get(ws.model, ws.driver as usize);
        let len = stores.kv.lock().unwrap().page_len(ws.id);
        Ok(len.map_err(anyhow::Error::from)? as u32)
    }

    async fn reserve(
        &mut self,
        this: Resource<KvWorkingSet>,
        pages: u32,
    ) -> Result<Result<WitRange, String>> {
        let ws = *self.ctx().table.get(&this)?;
        let stores = store_registry::get(ws.model, ws.driver as usize);
        let range = stores.kv.lock().unwrap().reserve(ws.id, pages as u64);
        Ok(range
            .map(|r| WitRange {
                start: r.start as u32,
                len: (r.end - r.start) as u32,
            })
            .map_err(|e| e.to_string()))
    }

    async fn discard(
        &mut self,
        this: Resource<KvWorkingSet>,
        _on: Resource<Pipeline>,
        ranges: Vec<WitRange>,
    ) -> Result<Result<(), String>> {
        let ws = *self.ctx().table.get(&this)?;
        let ranges: Vec<std::ops::Range<u64>> = ranges
            .into_iter()
            .map(|r| r.start as u64..(r.start as u64 + r.len as u64))
            .collect();
        let stores = store_registry::get(ws.model, ws.driver as usize);
        let out = {
            let mut kv = stores.kv.lock().unwrap();
            let epoch = kv.current_epoch();
            let out = kv.discard(ws.id, &ranges, epoch).map_err(|e| e.to_string());
            kv.retire_idle();
            out
        }; // store lock released before the contention drain re-locks pools.
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
        _on: Resource<Pipeline>,
    ) -> Result<Result<Resource<KvWorkingSet>, String>> {
        let ws = *self.ctx().table.get(&this)?;
        let stores = store_registry::get(ws.model, ws.driver as usize);
        let forked = stores.kv.lock().unwrap().fork(ws.id);
        match forked {
            Ok(id) => {
                let child = KvWorkingSet { id, ..ws };
                Ok(Ok(self.ctx().table.push(child)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn slice(
        &mut self,
        this: Resource<KvWorkingSet>,
        _on: Resource<Pipeline>,
        range: WitRange,
    ) -> Result<Result<Resource<KvWorkingSet>, String>> {
        let ws = *self.ctx().table.get(&this)?;
        let stores = store_registry::get(ws.model, ws.driver as usize);
        let sliced = stores.kv.lock().unwrap().slice(
            ws.id,
            range.start as u64..(range.start as u64 + range.len as u64),
        );
        match sliced {
            Ok(id) => {
                let child = KvWorkingSet { id, ..ws };
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
        let ws = self.ctx().table.delete(this)?;
        let stores = store_registry::get(ws.model, ws.driver as usize);
        {
            // Retain canonical paths as prefix-cache roots (bounded FIFO;
            // reclaimed by the contention ladder's rung 1 under pressure).
            let mut kv = stores.kv.lock().unwrap();
            let epoch = kv.current_epoch();
            kv.release_working_set_cached(ws.id, epoch, cache_roots_max());
            kv.retire_idle();
        } // store lock released before the contention drain re-locks pools.
        // Freed pool space may unblock a preempted inferlet.
        if let Some(orchestrator) = crate::store::reclaim::contention() {
            orchestrator.on_blocks_freed();
        }
        Ok(())
    }
}
