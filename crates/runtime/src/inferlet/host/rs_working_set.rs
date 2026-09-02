//! `pie:inferlet/working-set` — RS working-set host resource.
//!
//! The WASM resource type is [`crate::store::rs::working_set::RsWorkingSet`],
//! a thin handle (model, engine, RsWorkingSetId, cached geometry); every
//! substantive operation delegates to the per-(model, engine) [`RsStore`]
//! resolved through `store::registry`.

use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

use crate::inferlet::ProcessCtx;
use crate::inferlet::host::pie;
use crate::inferlet::host::pipeline::Pipeline;
use crate::store::registry as store_registry;
use crate::store::rs::RsGeometry;
use crate::store::rs::working_set::RsWorkingSet;

type WitRange = pie::inferlet::working_set::PageRange;

impl pie::inferlet::working_set::HostRsWorkingSet for ProcessCtx {
    /// Fresh, empty RS working set bound to the single bound model (model 0),
    /// engine 0. Geometry comes from the model's RS caps (0/0/1 for
    /// pure-attention models).
    async fn new(&mut self) -> Result<Resource<RsWorkingSet>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let model = 0;
        let caps = crate::model::model().rs_caps();
        let geom = RsGeometry {
            state_size: caps.state_size,
            buffer_page_tokens: caps.buffer_page_size,
            fold_granularity: caps.fold_granularity,
        };
        let stores = store_registry::get(model, 0);
        let id = stores.rs.lock().unwrap().create_working_set(geom);
        let ws = RsWorkingSet::new(model, 0, id, geom);
        self.register_rs_working_set(model, 0, id);
        Ok(self.ctx().table.push(ws)?)
    }

    async fn buffer_size(&mut self, this: Resource<RsWorkingSet>) -> Result<u32> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let ws = self.ctx().table.get(&this)?.clone();
        let stores = store_registry::get(ws.model, ws.engine);
        let size = stores.rs.lock().unwrap().buffer_size(ws.id);
        size.map_err(anyhow::Error::from)
    }

    async fn alloc_buffer(
        &mut self,
        this: Resource<RsWorkingSet>,
        n: u32,
    ) -> Result<Result<WitRange, String>> {
        // Strict admission: RS buffer slots are scarce pooled resources.
        crate::inferlet::process::ensure_bind_admitted(self).await;
        crate::inferlet::process::gate::residency_gate(self).await?;
        let ws = self.ctx().table.get(&this)?.clone();
        let stores = store_registry::get(ws.model, ws.engine);
        let range = stores.rs.lock().unwrap().alloc_buffer(ws.id, n);
        Ok(range
            .map(|r| WitRange {
                start: r.start,
                len: r.len,
            })
            .map_err(|e| e.to_string()))
    }

    async fn free_buffer(
        &mut self,
        this: Resource<RsWorkingSet>,
        indices: Vec<u32>,
    ) -> Result<Result<(), String>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let ws = self.ctx().table.get(&this)?.clone();
        let stores = store_registry::get(ws.model, ws.engine);
        let mut rs = stores.rs.lock().unwrap();
        let epoch = rs.current_epoch();
        let out = rs
            .free_buffer(ws.id, &indices, epoch)
            .map_err(|e| e.to_string());
        rs.retire_idle();
        Ok(out)
    }

    async fn discard_buffered(
        &mut self,
        this: Resource<RsWorkingSet>,
        count: u32,
    ) -> Result<Result<(), String>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let ws = self.ctx().table.get(&this)?.clone();
        let stores = store_registry::get(ws.model, ws.engine);
        let out = stores
            .rs
            .lock()
            .unwrap()
            .discard_buffered(ws.id, count)
            .map_err(|e| e.to_string());
        Ok(out)
    }

    async fn reorder_buffer(
        &mut self,
        this: Resource<RsWorkingSet>,
        perm: Vec<u32>,
    ) -> Result<Result<(), String>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let ws = self.ctx().table.get(&this)?.clone();
        let stores = store_registry::get(ws.model, ws.engine);
        let out = stores
            .rs
            .lock()
            .unwrap()
            .reorder_buffer(ws.id, &perm)
            .map_err(|e| e.to_string());
        Ok(out)
    }

    async fn fork(
        &mut self,
        this: Resource<RsWorkingSet>,
        on: Resource<Pipeline>,
    ) -> Result<Result<Resource<RsWorkingSet>, String>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        // no drain: RS mappings publish at prepare, in submission order, so
        // the committed mapping already carries every fire submitted on
        // `on` before this call; a later CoW copy is issued behind the
        // fires that wrote the parent.
        let (failure, scope) = {
            let pipeline = self.ctx().table.get(&on)?;
            (pipeline.failure.clone(), pipeline.scope.clone())
        };
        let ws = self.ctx().table.get(&this)?.clone();
        if let Err(owner) = ws.claim_pipeline_scope(&scope) {
            return Ok(Err(format!(
                "rs working set fork: parent is scoped to pipeline {owner:#x}, \
                 not supplied pipeline {:#x}",
                scope.id()
            )));
        }
        if let Some(reason) = failure.lock().unwrap().clone() {
            return Ok(Err(format!(
                "rs working set fork: pipeline failed: {reason}"
            )));
        }

        let stores = store_registry::get(ws.model, ws.engine);
        let forked = stores.rs.lock().unwrap().fork(ws.id);
        match forked {
            Ok(id) => {
                // a distinct working-set id gets its own fresh lifecycle,
                // never a clone of the parent's.
                let child = RsWorkingSet::new(ws.model, ws.engine, id, ws.geom);
                self.register_rs_working_set(ws.model, ws.engine, id);
                Ok(Ok(self.ctx().table.push(child)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn drop(&mut self, this: Resource<RsWorkingSet>) -> Result<()> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        // `release` performs the `release_working_set`/`retire_idle`
        // sequence and marks the shared lifecycle done, so `ws`'s own drop
        // is a no-op.
        let ws = self.ctx().table.delete(this)?;
        self.unregister_rs_working_set(ws.model, ws.engine, ws.id);
        ws.release();
        Ok(())
    }
}
