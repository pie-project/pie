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
/// Where a working set's rows end. What it folded is not here, nor whether
/// that is known: both are its KV working set's, one length the recurrent
/// state follows.
pub struct Cursor {
    pub tail: Option<u32>,
    pub reserved: u32,
    /// One past the highest slot a fire is known to have written: a mark
    /// that only rises, since a lane that rewinds leaves the rows it wrote
    /// further on in its buffer, where a mask may still admit them.
    pub written: u32,
    /// One past the slots the set holds written without a gap, by its own
    /// fires or in the prefix it shares: what a fire staged now may read
    /// ([`engine::KvDelta::readable`]). A page holds what its last owner
    /// left until the set writes it, so unlike [`Self::written`] it never
    /// counts what the set only reserved. Rows the host places count from
    /// their fire's submission (and stop counting if it fails); rows the
    /// device places, from where their fire reported they ended.
    pub readable: u32,
    /// A fire whose rows the device placed settled with no end told: what
    /// it wrote is known to its engine alone, which from then on holds the
    /// set's lanes to what the set wrote ([`engine::extent`]).
    pub untold: bool,
    /// The fires staged on the set whose rows land where the device says,
    /// and which have not settled: where they land is not known until they
    /// do. A fork shares them, as it does the rows they write.
    pub staged: Vec<Arc<StagedRows>>,
    /// M-RoPE spans `(start_slot, end_slot, lag)`, `lag = token_count - position_span`.
    pub spans: Vec<(u32, u32, u32)>,
}

/// The rows of one fire on a working set that land where the device says,
/// from its staging until it settles.
#[derive(Debug)]
pub struct StagedRows {
    rows: u32,
    settled: std::sync::OnceLock<Option<u32>>,
}

impl StagedRows {
    #[must_use]
    pub fn new(rows: u32) -> Arc<StagedRows> {
        Arc::new(StagedRows {
            rows,
            settled: std::sync::OnceLock::new(),
        })
    }

    /// The fire settled: its rows ended at `end`, one past the last slot
    /// they landed at, as its engine reported; `None` where none was
    /// reported (a fire that failed, or never ran).
    pub fn settle(&self, end: Option<u32>) {
        let _ = self.settled.set(end);
    }
}

impl Cursor {
    /// `fire` is staged on the set, its rows to land where the device says.
    pub fn stage(&mut self, fire: Arc<StagedRows>) {
        let (written, _) = self.standing();
        self.written = written;
        self.readable = self.readable();
        self.untold |= self
            .staged
            .iter()
            .any(|fire| matches!(fire.settled.get(), Some(None)));
        self.staged.retain(|fire| fire.settled.get().is_none());
        self.staged.push(fire);
    }

    /// The written end as far as settled fires tell it, and the rows of
    /// the fires still to settle. A fire that settled with no end reported
    /// wrote inside what the set reserved, as every such fire does.
    pub fn standing(&self) -> (u32, u32) {
        let (mut written, mut staged) = (self.written, 0u32);
        for fire in &self.staged {
            match fire.settled.get() {
                Some(end) => written = written.max(end.unwrap_or(self.reserved)),
                None => staged = staged.saturating_add(fire.rows),
            }
        }
        (written, staged)
    }

    /// The written end a fire staged now may read: [`Self::readable`] as
    /// far as the device-placed fires that settled reported it. A fire that
    /// reported no end moves it nowhere.
    pub fn readable(&self) -> u32 {
        self.staged
            .iter()
            .filter_map(|fire| fire.settled.get().copied().flatten())
            .fold(self.readable, u32::max)
    }

    /// Whether the host knows the set's written end: no device-placed fire
    /// is still to settle on it, and every one that settled told its end.
    pub fn readable_known(&self) -> bool {
        !self.untold
            && self
                .staged
                .iter()
                .all(|fire| fire.settled.get().is_some_and(Option::is_some))
    }

    /// The host placed rows ending at `end` on the set, in a fire just
    /// submitted.
    pub fn wrote(&mut self, end: u32) {
        self.readable = self.readable.max(end);
    }

    /// A fire that wrote from `start` failed: its rows may never have
    /// landed, so nothing from there on counts as written.
    pub fn unwrote(&mut self, start: u32) {
        self.readable = self.readable.min(start);
    }

    /// No more than this many rows stand unfolded in the set's buffer when
    /// a fire staged now runs, its recurrent state folded through `frozen`:
    /// those written past `frozen` by fires that settled, and every row of
    /// the fires still to settle ([`engine::fire::rung_rows`], device
    /// geometry).
    pub fn unfolded(&self, frozen: u32) -> u32 {
        let (written, staged) = self.standing();
        written.saturating_sub(frozen).saturating_add(staged)
    }

    /// The written end a snapshot states: what the set reserved while a
    /// fire is still to settle.
    pub fn written_end(&self) -> u32 {
        match self.standing() {
            (written, 0) => written,
            _ => self.reserved.max(self.written),
        }
    }

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
            tail: Some(0),
            reserved: 0,
            written: 0,
            readable: 0,
            untold: false,
            staged: Vec::new(),
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
    ctx.register_rs_working_set(&ws);
    Ok(ctx.ctx().table.push(ws)?.rep())
}

/// A working set made for `ctx` where it runs: what a handle no process has
/// touched yet becomes when one binds it.
pub(crate) fn materialize(ctx: &mut ProcessCtx) -> Result<u32> {
    let kv = new_kv(ctx)?;
    let rs = if crate::model::model().rs_caps().has_state() {
        Some(new_rs(ctx)?)
    } else {
        None
    };
    Ok(ctx
        .ctx()
        .table
        .push(WorkingSet {
            kv,
            rs,
            cursor: Arc::new(Mutex::new(Cursor::default())),
        })?
        .rep())
}

/// `ctx` takes a working set made elsewhere, as if it had made it.
pub(crate) fn adopt(
    ctx: &mut ProcessCtx,
    kv: KvWorkingSet,
    rs: Option<RsWorkingSet>,
    cursor: Cursor,
) -> Result<u32> {
    ctx.register_kv_working_set(&kv);
    let kv = ctx.ctx().table.push(kv)?.rep();
    let rs = match rs {
        Some(rs) => {
            ctx.register_rs_working_set(&rs);
            Some(ctx.ctx().table.push(rs)?.rep())
        }
        None => None,
    };
    Ok(ctx
        .ctx()
        .table
        .push(WorkingSet {
            kv,
            rs,
            cursor: Arc::new(Mutex::new(cursor)),
        })?
        .rep())
}

pub(crate) fn kv_of(ctx: &mut ProcessCtx, rep: u32) -> Result<KvWorkingSet> {
    let res: Resource<KvWorkingSet> = Resource::new_borrow(rep);
    Ok(ctx.ctx().table.get(&res)?.clone())
}

impl pie::inferlet::working_set::Host for ProcessCtx {}

impl pie::inferlet::working_set::HostWorkingSet for ProcessCtx {
    async fn new(&mut self) -> Result<Resource<WorkingSet>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        Ok(Resource::new_own(materialize(self)?))
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
        let kv_rep = self.ctx().table.get(&this)?.kv;
        let kv = kv_of(self, kv_rep)?;
        let stores = store_registry::get(kv.model, kv.engine);
        let folded =
            store_registry::with_kv_lock(&stores.kv, "host-working-set", |s| s.folded(kv.id))?;
        Ok(u32::try_from(folded).unwrap_or(u32::MAX))
    }

    async fn fork(
        &mut self,
        this: Resource<WorkingSet>,
        on: Resource<Pipeline>,
    ) -> Result<Result<Resource<WorkingSet>, String>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let (scope, failure, fires) = {
            let pipeline = self.ctx().table.get(&on)?;
            (
                pipeline.scope.clone(),
                pipeline.failure.clone(),
                pipeline.fires.clone(),
            )
        };
        if scope.is_closed() {
            return Ok(Err("pipeline is closed".to_string()));
        }
        let (kv_rep, rs_rep) = {
            let ws = self.ctx().table.get(&this)?;
            (ws.kv, ws.rs)
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
        match claimed {
            Ok(()) => {}
            Err(crate::store::ScopeRefusal::Held(owner)) => {
                return Ok(Err(format!(
                    "working set is scoped to pipeline {owner:#x}, not supplied pipeline {:#x}",
                    scope.id()
                )));
            }
            Err(failed) => return Ok(Err(format!("working set {failed}"))),
        }
        // The child starts from the frozen length the fires on their way
        // leave, or the fork is refused: never from a bound.
        let stores = store_registry::get(kv.model, kv.engine);
        let at_rest = {
            let cell = self.ctx().table.get(&this)?.cursor.clone();
            let rs = rs.as_ref().map(|rs| rs.id);
            crate::pipeline::fire::cursor_at_rest(self, &fires, &failure, &kv, rs, &cell).await?
        };
        let cursor = match at_rest {
            Ok(cursor) => cursor,
            Err(why) => return Ok(Err(why)),
        };
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
                self.register_rs_working_set(&child);
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

    /// What a set holds written, as a fire staged now may read it: never
    /// what it only reserved, nor what a failed fire may not have landed.
    #[test]
    fn readable_every_case() {
        let mut set = Cursor {
            reserved: 64,
            ..Cursor::default()
        };
        set.wrote(10);
        assert_eq!((set.readable(), set.readable_known()), (10, true));
        // A device-placed fire in flight: the host knows nothing past 10.
        let told = StagedRows::new(2);
        set.stage(told.clone());
        assert!(!set.readable_known());
        told.settle(Some(12));
        assert_eq!((set.readable(), set.readable_known()), (12, true));
        // One that told no end: its engine alone knows, and the set's lanes
        // are its engine's to hold from then on.
        let untold = StagedRows::new(2);
        set.stage(untold.clone());
        untold.settle(None);
        assert_eq!(set.readable(), 12, "nothing it only reserved");
        assert!(!set.readable_known());
        set.stage(StagedRows::new(1));
        assert!(set.untold && !set.readable_known(), "kept past the fire");
        // A failed host fire's rows count no more.
        set.wrote(20);
        set.unwrote(14);
        assert_eq!(set.readable(), 14);
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
