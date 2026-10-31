//! The inferlet's working set: the KV working set and, where the model keeps
//! one, the recurrent one.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

use crate::inferlet::ProcessCtx;
use crate::inferlet::host::pie;
use crate::inferlet::host::pipeline::Pipeline;
use crate::store::kv::working_set::KvWorkingSet;
use crate::store::registry as store_registry;
use crate::store::rs::RsGeometry;
use crate::store::rs::working_set::RsWorkingSet;

#[derive(Clone, Debug)]
pub struct Cursor {
    pub folded: u32,
    /// False while a fold the host could not place at submit is on its
    /// way: `folded` is then a lower bound.
    pub folded_exact: bool,
    /// Fires in flight whose fold is placed when they settle; the last to
    /// settle makes `folded` exact again.
    pub unsettled: u32,
    pub tail: Option<u32>,
    pub reserved: u32,
    /// M-RoPE spans `(start_slot, end_slot, lag)`, `lag = token_count - position_span`.
    pub spans: Vec<(u32, u32, u32)>,
}

impl Cursor {
    /// The last write wins from `start` on.
    pub fn rewrite(&mut self, start: u32, spans: impl IntoIterator<Item = (u32, u32, u32)>) {
        self.spans.retain(|&(from, _, _)| from < start);
        self.spans.extend(spans);
    }
}

/// The summed lags of spans ending at or before `slot`: its position is `slot - lag`.
pub fn lag(spans: &[(u32, u32, u32)], slot: u32) -> u32 {
    spans
        .iter()
        .filter(|&&(_, end, _)| end <= slot)
        .map(|&(_, _, lag)| lag)
        .sum()
}

impl Default for Cursor {
    fn default() -> Self {
        Cursor {
            folded: 0,
            folded_exact: true,
            unsettled: 0,
            tail: Some(0),
            reserved: 0,
            spans: Vec::new(),
        }
    }
}

pub struct WorkingSet {
    pub kv: u32,
    pub rs: Option<u32>,
    pub cursor: Arc<Mutex<Cursor>>,
}

impl WorkingSet {
    pub fn cursor(&self) -> Cursor {
        self.cursor.lock().unwrap().clone()
    }
}

fn new_kv(ctx: &mut ProcessCtx) -> Result<u32> {
    let engine = crate::planner::engine_of(ctx.id());
    let stores = store_registry::get(0, engine);
    let id =
        store_registry::with_kv_lock(&stores.kv, "host-working-set", |kv| kv.create_working_set());
    let ws = KvWorkingSet::new(0, engine, id, stores.kv_page_size);
    ctx.register_kv_working_set(&ws);
    Ok(ctx.ctx().table.push(ws)?.rep())
}

fn new_rs(ctx: &mut ProcessCtx) -> Result<u32> {
    let caps = crate::model::model().rs_caps();
    let geom = RsGeometry {
        state_size: caps.state_size,
        buffer_page_tokens: caps.buffer_page_size,
    };
    let engine = crate::planner::engine_of(ctx.id());
    let stores = store_registry::get(0, engine);
    let id = stores.rs.lock().unwrap().create_working_set(geom);
    let ws = RsWorkingSet::new(0, engine, id, geom);
    ctx.register_rs_working_set(0, engine, id);
    Ok(ctx.ctx().table.push(ws)?.rep())
}

pub(crate) fn kv_of(ctx: &mut ProcessCtx, rep: u32) -> Result<KvWorkingSet> {
    let res: Resource<KvWorkingSet> = Resource::new_borrow(rep);
    Ok(ctx.ctx().table.get(&res)?.clone())
}

impl pie::inferlet::working_set::Host for ProcessCtx {}

impl pie::inferlet::working_set::HostWorkingSet for ProcessCtx {
    async fn new(&mut self) -> Result<Resource<WorkingSet>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let kv = new_kv(self)?;
        let rs = if crate::model::model().rs_caps().has_state() {
            Some(new_rs(self)?)
        } else {
            None
        };
        Ok(self.ctx().table.push(WorkingSet {
            kv,
            rs,
            cursor: Arc::new(Mutex::new(Cursor::default())),
        })?)
    }

    async fn reserve(
        &mut self,
        this: Resource<WorkingSet>,
        end: u32,
    ) -> Result<Result<(), String>> {
        crate::inferlet::process::ensure_bind_admitted(self).await;
        crate::inferlet::process::gate::residency_gate(self).await?;
        let (kv_rep, cursor) = {
            let ws = self.ctx().table.get(&this)?;
            (ws.kv, ws.cursor.clone())
        };
        let kv = kv_of(self, kv_rep)?;
        if let Err(error) = grow_pages(&kv, end) {
            return Ok(Err(error));
        }
        let mut cursor = cursor.lock().unwrap();
        cursor.reserved = cursor.reserved.max(end);
        Ok(Ok(()))
    }

    async fn folded(&mut self, this: Resource<WorkingSet>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.cursor().folded)
    }

    async fn cached_tokens(&mut self, this: Resource<WorkingSet>) -> Result<u32> {
        let kv_rep = self.ctx().table.get(&this)?.kv;
        let ws = kv_of(self, kv_rep)?;
        let stores = store_registry::get(ws.model, ws.engine);
        let tokens = store_registry::with_kv_lock(&stores.kv, "host-working-set", |kv| {
            kv.supplied_tokens(ws.id)
        });
        Ok(u32::try_from(tokens.unwrap_or(0)).unwrap_or(u32::MAX))
    }

    async fn fork(
        &mut self,
        this: Resource<WorkingSet>,
        on: Resource<Pipeline>,
    ) -> Result<Result<Resource<WorkingSet>, String>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let (scope, failure) = {
            let pipeline = self.ctx().table.get(&on)?;
            (pipeline.scope.clone(), pipeline.failure.clone())
        };
        if scope.is_closed() {
            return Ok(Err("pipeline is closed".to_string()));
        }
        let (kv_rep, rs_rep, cursor) = {
            let ws = self.ctx().table.get(&this)?;
            (ws.kv, ws.rs, ws.cursor())
        };
        let kv = kv_of(self, kv_rep)?;
        let rs = match rs_rep {
            Some(rep) => {
                let res: Resource<RsWorkingSet> = Resource::new_borrow(rep);
                Some(self.ctx().table.get(&res)?.clone())
            }
            None => None,
        };
        let claimed = kv.claim_pipeline_scope(&scope).and_then(|()| match &rs {
            Some(rs) => rs.claim_pipeline_scope(&scope),
            None => Ok(()),
        });
        if let Err(owner) = claimed {
            return Ok(Err(format!(
                "working set is scoped to pipeline {owner:#x}, not supplied pipeline {:#x}",
                scope.id()
            )));
        }
        if let Some(reason) = failure.lock().unwrap().clone() {
            return Ok(Err(format!("pipeline failed: {reason}")));
        }
        let stores = store_registry::get(kv.model, kv.engine);
        let kv_child =
            match store_registry::with_kv_lock(&stores.kv, "host-working-set", |s| s.fork(kv.id)) {
                Ok(id) => kv.forked(id),
                Err(error) => return Ok(Err(error.to_string())),
            };
        let rs_child = match rs {
            Some(rs) => {
                let forked = stores.rs.lock().unwrap().fork(rs.id);
                match forked {
                    Ok(id) => Some(RsWorkingSet::new(rs.model, rs.engine, id, rs.geom)),
                    Err(error) => {
                        kv_child.release();
                        return Ok(Err(error.to_string()));
                    }
                }
            }
            None => None,
        };
        self.register_kv_working_set(&kv_child);
        let kv_rep = self.ctx().table.push(kv_child)?.rep();
        let rs_rep = match rs_child {
            Some(child) => {
                self.register_rs_working_set(child.model, child.engine, child.id);
                Some(self.ctx().table.push(child)?.rep())
            }
            None => None,
        };
        Ok(Ok(self.ctx().table.push(WorkingSet {
            kv: kv_rep,
            rs: rs_rep,
            cursor: Arc::new(Mutex::new(cursor)),
        })?))
    }

    async fn forget(
        &mut self,
        this: Resource<WorkingSet>,
        start: u32,
        end: u32,
    ) -> Result<Result<u32, String>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let model = crate::model::model();
        if !model.eta_caps().rotates_by_table || model.rs_caps().has_state() {
            return Ok(Err(
                "this model cannot forget: its rows rotate by their slots, or a recurrent \
                 state has already folded them"
                    .to_string(),
            ));
        }
        let (kv_rep, cursor) = {
            let ws = self.ctx().table.get(&this)?;
            (ws.kv, ws.cursor())
        };
        if !cursor.folded_exact || end > cursor.folded {
            return Ok(Err(format!(
                "forget reaches {end}, past the frozen {} slot(s)",
                cursor.folded
            )));
        }
        let kv = kv_of(self, kv_rep)?;
        let stores = store_registry::get(kv.model, kv.engine);
        let page = stores.kv_page_size.max(1);
        let (first, last) = (start.div_ceil(page), end / page);
        // An empty span still gives up what waited on another reader.
        let (pages, from) = (u64::from(first)..u64::from(last), u64::from(first * page));
        Ok(match kv.forget(pages, from) {
            Ok(()) => Ok(last.saturating_sub(first) * page),
            Err(error) => Err(error.to_string()),
        })
    }

    async fn drop(&mut self, this: Resource<WorkingSet>) -> Result<()> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let ws = self.ctx().table.delete(this)?;
        let kv = self
            .ctx()
            .table
            .delete(Resource::<KvWorkingSet>::new_own(ws.kv))?;
        self.unregister_kv_working_set(kv.model, kv.engine, kv.id);
        kv.release();
        if let Some(rep) = ws.rs {
            let rs = self
                .ctx()
                .table
                .delete(Resource::<RsWorkingSet>::new_own(rep))?;
            self.unregister_rs_working_set(rs.model, rs.engine, rs.id);
            rs.release();
        }
        Ok(())
    }
}

pub(crate) fn grow_pages(kv: &KvWorkingSet, end: u32) -> Result<(), String> {
    let stores = store_registry::get(kv.model, kv.engine);
    let page = u64::from(stores.kv_page_size.max(1));
    let want = u64::from(end).div_ceil(page);
    store_registry::with_kv_lock(&stores.kv, "host-working-set", |s| {
        let have = s.page_len(kv.id)?;
        if want > have {
            s.reserve(kv.id, want - have)?;
        }
        Ok::<_, crate::store::kv::KvStoreError>(())
    })
    .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lag_every_case() {
        a_row_lags_the_spans_ending_before_it();
        a_write_drops_the_spans_from_where_it_starts();
    }

    fn a_row_lags_the_spans_ending_before_it() {
        let spans = [(10, 30, 16), (40, 50, 7)];
        assert_eq!(lag(&spans, 5), 0);
        assert_eq!(lag(&spans, 29), 0, "inside the span");
        assert_eq!(lag(&spans, 30), 16);
        assert_eq!(lag(&spans, 50), 23);
        assert_eq!(lag(&[], 100), 0, "no span, no lag");
    }

    fn a_write_drops_the_spans_from_where_it_starts() {
        let mut cursor = Cursor::default();
        cursor.rewrite(0, [(10, 30, 16), (40, 50, 7)]);
        cursor.rewrite(40, [(45, 55, 3)]);
        assert_eq!(cursor.spans, [(10, 30, 16), (45, 55, 3)]);
        cursor.rewrite(35, []);
        assert_eq!(cursor.spans, [(10, 30, 16)], "a text write past the span");
        cursor.rewrite(10, []);
        assert!(cursor.spans.is_empty(), "the span itself written over");
    }
}
