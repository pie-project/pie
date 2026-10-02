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
        let out = rs.free_buffer(ws.id, &indices).map_err(|e| e.to_string());
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
                let child = RsWorkingSet::new(ws.model, ws.engine, id, ws.geom);
                self.register_rs_working_set(ws.model, ws.engine, id);
                Ok(Ok(self.ctx().table.push(child)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn update_index(
        &mut self,
        this: Resource<RsWorkingSet>,
        key: Vec<u8>,
    ) -> Result<Result<(), String>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let ws = self.ctx().table.get(&this)?.clone();
        let stores = store_registry::get(ws.model, ws.engine);
        let result = stores.rs.lock().unwrap().update_index(key, ws.id);
        Ok(result.map(slots_freed).map_err(|e| e.to_string()))
    }

    async fn from_index(
        &mut self,
        key: Vec<u8>,
    ) -> Result<Result<Option<Resource<RsWorkingSet>>, String>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let (model, engine) = (0, 0);
        let stores = store_registry::get(model, engine);
        let indexed = {
            let mut rs = stores.rs.lock().unwrap();
            rs.from_index(&key)
                .and_then(|id| id.map(|id| Ok((id, rs.geometry(id)?))).transpose())
        };
        match indexed {
            Ok(Some((id, geom))) => {
                let ws = RsWorkingSet::new(model, engine, id, geom);
                self.register_rs_working_set(model, engine, id);
                Ok(Ok(Some(self.ctx().table.push(ws)?)))
            }
            Ok(None) => Ok(Ok(None)),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn remove_index(&mut self, key: Vec<u8>) -> Result<Result<bool, String>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let stores = store_registry::get(0, 0);
        let removed = stores.rs.lock().unwrap().remove_index(&key);
        Ok(removed
            .map(|(removed, freed)| {
                slots_freed(freed);
                removed
            })
            .map_err(|e| e.to_string()))
    }

    async fn drop(&mut self, this: Resource<RsWorkingSet>) -> Result<()> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let ws = self.ctx().table.delete(this)?;
        self.unregister_rs_working_set(ws.model, ws.engine, ws.id);
        ws.release();
        Ok(())
    }
}

fn slots_freed(freed: usize) {
    if freed != 0
        && let Some(planner) = crate::planner::planner()
    {
        planner.pages_freed();
    }
}
