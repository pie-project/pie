//! `pie:inferlet/working-set` — RS working-set host resource.
//!
//! The WASM resource type is [`crate::store::rs::working_set::RsWorkingSet`],
//! a thin handle (model, driver, RsWorkingSetId, cached geometry); every
//! substantive operation delegates to the per-(model, driver) [`RsStore`]
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
use pie_model;

type WitRange = pie::inferlet::working_set::PageRange;

impl pie::inferlet::working_set::HostRsWorkingSet for ProcessCtx {
    /// Fresh, empty RS working set bound to the single bound model (model 0),
    /// driver 0. Geometry comes from the model's RS caps (0/0/1 for
    /// pure-attention models).
    async fn new(&mut self) -> Result<Resource<RsWorkingSet>> {
        let model = 0;
        let caps = pie_model::model().rs_caps();
        let geom = RsGeometry {
            state_size: caps.state_size,
            buffer_page_tokens: caps.buffer_page_size,
            fold_granularity: caps.fold_granularity,
        };
        let stores = store_registry::get(model, 0);
        let id = stores.rs.lock().unwrap().create_working_set(geom);
        let ws = RsWorkingSet::new(model, 0, id, geom);
        Ok(self.ctx().table.push(ws)?)
    }

    async fn state_size(&mut self, this: Resource<RsWorkingSet>) -> Result<u64> {
        Ok(self.ctx().table.get(&this)?.geom.state_size)
    }

    async fn buffer_size(&mut self, this: Resource<RsWorkingSet>) -> Result<u32> {
        let ws = self.ctx().table.get(&this)?.clone();
        let stores = store_registry::get(ws.model, ws.driver as usize);
        let size = stores.rs.lock().unwrap().buffer_size(ws.id);
        Ok(size.map_err(anyhow::Error::from)?)
    }

    async fn buffer_page_size(&mut self, this: Resource<RsWorkingSet>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.geom.buffer_page_tokens)
    }

    async fn alloc_buffer(
        &mut self,
        this: Resource<RsWorkingSet>,
        n: u32,
    ) -> Result<Result<WitRange, String>> {
        let ws = self.ctx().table.get(&this)?.clone();
        let stores = store_registry::get(ws.model, ws.driver as usize);
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
        let ws = self.ctx().table.get(&this)?.clone();
        let stores = store_registry::get(ws.model, ws.driver as usize);
        let mut rs = stores.rs.lock().unwrap();
        let epoch = rs.current_epoch();
        let out = rs
            .free_buffer(ws.id, &indices, epoch)
            .map_err(|e| e.to_string());
        rs.retire_idle();
        Ok(out)
    }

    async fn reorder_buffer(
        &mut self,
        this: Resource<RsWorkingSet>,
        perm: Vec<u32>,
    ) -> Result<Result<(), String>> {
        let ws = self.ctx().table.get(&this)?.clone();
        let stores = store_registry::get(ws.model, ws.driver as usize);
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
        _on: Resource<Pipeline>,
    ) -> Result<Result<Resource<RsWorkingSet>, String>> {
        let ws = self.ctx().table.get(&this)?.clone();
        let stores = store_registry::get(ws.model, ws.driver as usize);
        let forked = stores.rs.lock().unwrap().fork(ws.id);
        match forked {
            Ok(id) => {
                // A distinct working-set id gets its OWN fresh lifecycle —
                // never a clone of the parent's (see the KV working-set
                // `fork`/`slice` for the identical rationale).
                let child = RsWorkingSet::new(ws.model, ws.driver, id, ws.geom);
                Ok(Ok(self.ctx().table.push(child)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn drop(&mut self, this: Resource<RsWorkingSet>) -> Result<()> {
        // `release` performs the exact `release_working_set` /
        // `retire_idle` sequence and marks the shared lifecycle done; `ws`'s
        // own drop just below (and the fallback `RsLifecycle::drop` it would
        // otherwise trigger) is then a no-op.
        let ws = self.ctx().table.delete(this)?;
        ws.release();
        Ok(())
    }
}
