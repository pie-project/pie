//! pie:core/working-set - `rs-working-set` host resource (Lane D, Phase 3).
//!
//! Thin WIT binding over the RS runtime core (`crate::working_set::rs`). The
//! bindgen `with:` maps the WIT `rs-working-set` resource directly onto
//! [`RsWorkingSet`], so `echo`'s forward `execute()` and `delta`'s
//! `inference.fold` call the core's `resolve_buffer` / `cow_write_buffer` /
//! `prepare_fold` methods on the table object without an extra wrapper.
//!
//! The arena-touching methods (`alloc-buffer`, `free-buffer`, `reorder-buffer`,
//! `fork`, `constructor`, `drop`) reach the per-(model, driver) arena via
//! `arena::get(model_idx, driver_idx).lock()`. Those are `todo!()` here pending
//! the resource→model binding (the WIT constructor takes no model; binding to
//! the instance's model is being settled with alpha/echo). The pure accessors
//! are wired now so the resource is usable for read-side wiring immediately.

use crate::api::pie;
use crate::instance::InstanceState;
use crate::working_set::rs::{RsGeometry, RsWorkingSet};
use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

impl pie::core::working_set::HostRsWorkingSet for InstanceState {
    /// Fresh, empty RS working set bound to the single bound model (model 0).
    /// The WIT `constructor()` takes no handle (global-model runtime). Structural
    /// buffer ops before the first forward are arena-free. Geometry is read from
    /// the model's RS caps (0/0/1 for pure-attention models). v1: a folded
    /// recurrent state is a single arena slab.
    async fn new(&mut self) -> Result<Resource<RsWorkingSet>> {
        // Single-model bind: model 0 is THE bound model (spawn-order index,
        // lock-step with the arena/kv_cas registries). Caps from the registry.
        let model_id = 0;
        let m = crate::model::model();
        let caps = m.rs_caps();
        let geom = RsGeometry {
            state_size: caps.state_size,
            state_blocks: if caps.state_size > 0 { 1 } else { 0 },
            buffer_page_tokens: caps.buffer_page_size,
            fold_granularity: caps.fold_granularity,
        };
        let ws = RsWorkingSet::new(model_id, geom);
        Ok(self.ctx().table.push(ws)?)
    }

    async fn state_size(&mut self, this: Resource<RsWorkingSet>) -> Result<u64> {
        Ok(self.ctx().table.get(&this)?.state_size())
    }

    async fn buffer_size(&mut self, this: Resource<RsWorkingSet>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.buffer_size())
    }

    async fn buffer_page_size(&mut self, this: Resource<RsWorkingSet>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.buffer_page_size())
    }

    /// Append `n` fresh **reserved** buffered page slots. Arena-free — slots are
    /// materialized lazily on the first forward write, so no driver is needed.
    async fn alloc_buffer(
        &mut self,
        this: Resource<RsWorkingSet>,
        n: u32,
    ) -> Result<Result<pie::core::working_set::PageRange, String>> {
        let ws = self.ctx().table.get_mut(&this)?;
        Ok(ws.alloc_buffer(n).map(|r| pie::core::working_set::PageRange {
            start: r.start,
            len: r.len,
        }).map_err(|e| e.to_string()))
    }

    /// Remove buffered slots at `indices` and densely compact. Decrefs any
    /// materialized slabs (reserved slots are arena-free).
    async fn free_buffer(
        &mut self,
        this: Resource<RsWorkingSet>,
        indices: Vec<u32>,
    ) -> Result<Result<(), String>> {
        let (model_id, driver) = {
            let ws = self.ctx().table.get(&this)?;
            (ws.model(), ws.driver().unwrap_or(0))
        };
        // Sync lock, released before returning (never held across an await).
        let res = {
            let arena = crate::arena::get(model_id, driver);
            let mut guard = arena.lock().unwrap();
            let ws = self.ctx().table.get_mut(&this)?;
            ws.free_buffer(&mut guard, &indices).map_err(|e| e.to_string())
        };
        Ok(res)
    }

    /// Reorder the buffered slots by `perm` (no arena access — pure structural).
    async fn reorder_buffer(
        &mut self,
        this: Resource<RsWorkingSet>,
        perm: Vec<u32>,
    ) -> Result<Result<(), String>> {
        Ok(self
            .ctx()
            .table
            .get_mut(&this)?
            .reorder_buffer(&perm)
            .map_err(|e| e.to_string()))
    }

    /// Fork into a new RS working set sharing folded + buffered slabs (lazy CoW).
    async fn fork(
        &mut self,
        this: Resource<RsWorkingSet>,
    ) -> Result<Result<Resource<RsWorkingSet>, String>> {
        let (model_id, driver) = {
            let ws = self.ctx().table.get(&this)?;
            (ws.model(), ws.driver().unwrap_or(0))
        };
        let child = {
            let arena = crate::arena::get(model_id, driver);
            let mut guard = arena.lock().unwrap();
            let ws = self.ctx().table.get(&this)?;
            ws.fork(&mut guard)
        };
        match child {
            Ok(child) => Ok(Ok(self.ctx().table.push(child)?)),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    /// Release the working set's folded + buffered arena references, then remove
    /// the resource from the table.
    async fn drop(&mut self, this: Resource<RsWorkingSet>) -> Result<()> {
        let ws = self.ctx().table.delete(this)?;
        let driver = ws.driver().unwrap_or(0);
        let model_id = ws.model();
        let arena = crate::arena::get(model_id, driver);
        let mut guard = arena.lock().unwrap();
        // Best-effort release; a bad arena ref must not fail resource teardown.
        let _ = ws.release(&mut guard);
        Ok(())
    }
}
