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

/// A prefix is shareable by its pages only when they are all of the context:
/// recurrent state and windowed rows would have to travel with them.
fn pages_alone_hold_kv(kv: &crate::store::kv::KvStore) -> bool {
    kv.window().is_none() && crate::model::model().rs_caps().state_size == 0
}

/// Published prefixes are shared only among processes of the same program,
/// since it is the program that vouches for what its pages hold.
fn prefix_scope(ctx: &ProcessCtx) -> crate::store::kv::hash::Hash256 {
    crate::store::kv::hash::cache_domain(ctx.program.as_bytes())
}

impl pie::inferlet::working_set::HostKvWorkingSet for ProcessCtx {
    async fn new(&mut self) -> Result<Resource<KvWorkingSet>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let stores = store_registry::get(0, 0);
        let prepared = crate::store::kv::PreparedWorkingSet::new();
        let id = store_registry::with_kv_lock(&stores.kv, "host-working-set", move |kv| {
            kv.install_working_set(prepared)
        });
        let ws = KvWorkingSet::new(0, 0, id, stores.kv_page_size);
        self.register_kv_working_set(&ws);
        Ok(self.ctx().table.push(ws)?)
    }

    async fn page_len(&mut self, this: Resource<KvWorkingSet>) -> Result<u32> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let len = self.ctx().table.get(&this)?.page_len();
        Ok(len.map_err(anyhow::Error::from)? as u32)
    }

    async fn reserve(
        &mut self,
        this: Resource<KvWorkingSet>,
        pages: u32,
    ) -> Result<Result<WitRange, String>> {
        crate::inferlet::process::ensure_bind_admitted(self).await;
        crate::inferlet::process::gate::residency_gate(self).await?;
        let ws = self.ctx().table.get(&this)?.clone();
        let stores = store_registry::get(ws.model, ws.engine);
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
        crate::inferlet::process::gate::residency_gate(self).await?;
        let ws = self.ctx().table.get(&this)?.clone();
        if !ws.is_settled() {
            return Ok(Err(
                "working set cannot be indexed while an operation is in flight".to_string(),
            ));
        }
        let stores = store_registry::get(ws.model, ws.engine);
        let result = store_registry::with_kv_lock(&stores.kv, "host-working-set", |kv| {
            kv.update_index(key, ws.id)
        });
        match result {
            Ok(freed) => {
                if freed != 0
                    && let Some(planner) = crate::planner::planner()
                {
                    planner.pages_freed();
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
        crate::inferlet::process::gate::residency_gate(self).await?;
        let stores = store_registry::get(0, 0);
        let prepared = crate::store::kv::PreparedWorkingSet::new();
        let indexed = store_registry::with_kv_lock(&stores.kv, "host-working-set", move |kv| {
            kv.from_index(&key, prepared)
        });
        match indexed {
            Ok(Some(id)) => {
                let ws = KvWorkingSet::new(0, 0, id, stores.kv_page_size);
                self.register_kv_working_set(&ws);
                Ok(Ok(Some(self.ctx().table.push(ws)?)))
            }
            Ok(None) => Ok(Ok(None)),
            Err(error) => Ok(Err(error.to_string())),
        }
    }

    async fn remove_index(&mut self, key: Vec<u8>) -> Result<Result<bool, String>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let stores = store_registry::get(0, 0);
        let removed = store_registry::with_kv_lock(&stores.kv, "host-working-set", |kv| {
            kv.remove_index(&key)
        });
        match removed {
            Ok((removed, freed)) => {
                if freed != 0
                    && let Some(planner) = crate::planner::planner()
                {
                    planner.pages_freed();
                }
                Ok(Ok(removed))
            }
            Err(error) => Ok(Err(error.to_string())),
        }
    }

    async fn publish_prefix(
        &mut self,
        this: Resource<KvWorkingSet>,
        tokens: Vec<u32>,
    ) -> Result<Result<u32, String>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let ws = self.ctx().table.get(&this)?.clone();
        let scope = prefix_scope(self);
        let stores = store_registry::get(ws.model, ws.engine);
        let published = store_registry::with_kv_lock(&stores.kv, "host-working-set", |kv| {
            if !pages_alone_hold_kv(kv) {
                return Ok(0);
            }
            kv.publish_prefix(ws.id, scope, &tokens, stores.kv_page_size)
        });
        Ok(published
            .map(|pages| pages as u32)
            .map_err(|e| e.to_string()))
    }

    async fn adopt_prefix(
        &mut self,
        this: Resource<KvWorkingSet>,
        tokens: Vec<u32>,
    ) -> Result<Result<u32, String>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let ws = self.ctx().table.get(&this)?.clone();
        let scope = prefix_scope(self);
        let stores = store_registry::get(ws.model, ws.engine);
        let page_size = stores.kv_page_size;
        let adopted = store_registry::with_kv_lock(&stores.kv, "host-working-set", |kv| {
            if !pages_alone_hold_kv(kv) {
                return Ok(None);
            }
            crate::pipeline::fire::kv::match_prefix(kv, ws.id, Some(scope), &tokens, page_size)
        });
        Ok(adopted
            .map(|pages| pages.unwrap_or(0) as u32 * page_size)
            .map_err(|e| e.to_string()))
    }

    async fn discard(
        &mut self,
        this: Resource<KvWorkingSet>,
        on: Resource<Pipeline>,
        ranges: Vec<WitRange>,
    ) -> Result<Result<(), String>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let ws = match scoped_working_set(self, &this, &on)? {
            Ok(ws) => ws,
            Err(error) => return Ok(Err(error)),
        };
        let ranges: Vec<std::ops::Range<u64>> = ranges
            .into_iter()
            .map(|r| r.start as u64..(r.start as u64 + r.len as u64))
            .collect();
        let stores = store_registry::get(ws.model, ws.engine as usize);
        let out = store_registry::with_kv_lock(&stores.kv, "host-working-set", |kv| {
            let epoch = kv.current_epoch();
            let out = kv.discard(ws.id, &ranges, epoch).map_err(|e| e.to_string());
            kv.retire_idle();
            out
        }); // store lock released before the planner's drain re-locks pools.
        if out.is_ok()
            && let Some(planner) = crate::planner::planner()
        {
            planner.pages_freed();
        }
        Ok(out)
    }

    async fn fork(
        &mut self,
        this: Resource<KvWorkingSet>,
        on: Resource<Pipeline>,
    ) -> Result<Result<Resource<KvWorkingSet>, String>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let ws = match scoped_working_set(self, &this, &on)? {
            Ok(ws) => ws,
            Err(error) => return Ok(Err(error)),
        };
        let stores = store_registry::get(ws.model, ws.engine as usize);
        let prepared = crate::store::kv::PreparedWorkingSet::new();
        let forked = store_registry::with_kv_lock(&stores.kv, "host-working-set", move |kv| {
            kv.fork(ws.id, prepared)
        });
        match forked {
            Ok(id) => {
                let child = ws.forked(id);
                self.register_kv_working_set(&child);
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
        crate::inferlet::process::gate::residency_gate(self).await?;
        let ws = match scoped_working_set(self, &this, &on)? {
            Ok(ws) => ws,
            Err(error) => return Ok(Err(error)),
        };
        let stores = store_registry::get(ws.model, ws.engine as usize);
        let prepared = crate::store::kv::PreparedWorkingSet::new();
        let sliced = store_registry::with_kv_lock(&stores.kv, "host-working-set", move |kv| {
            kv.slice(
                ws.id,
                range.start as u64..(range.start as u64 + range.len as u64),
                prepared,
            )
        });
        match sliced {
            Ok(id) => {
                let child = ws.forked(id);
                self.register_kv_working_set(&child);
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
        crate::inferlet::process::gate::residency_gate(self).await?;
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
        crate::inferlet::process::gate::residency_gate(self).await?;
        let ws = self.ctx().table.delete(this)?;
        self.unregister_kv_working_set(ws.model, ws.engine, ws.id);
        ws.release();
        Ok(())
    }
}
