//! Handles on their way into a process, in it, and back.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use client_api::{HandleKind, HandleName, Handles};

use super::HandleError;
use super::held::{Held, Parts};
use super::table::{Landing, Slot};
use crate::inferlet::host::frames::{Frames, Pcm};
use crate::inferlet::host::grammar::{self, Matcher};
use crate::inferlet::host::working_set::{self, WorkingSet};
use crate::inferlet::process::ProcessResidency;
use crate::inferlet::{ProcessCtx, ProcessId};
use crate::planner::Demand;
use crate::server::ClientId;
use crate::store::kv::working_set::KvWorkingSet;
use crate::store::registry as store_registry;
use crate::store::rs::working_set::RsWorkingSet;

/// One handle a launch moves in.
#[derive(Debug)]
pub(super) struct Bound {
    /// The name its guest takes it by.
    pub(super) slot: String,
    /// The name its session knows it by.
    pub(super) name: HandleName,
    pub(super) handle: Slot,
}

/// A launch's handles on their way into its process. Dropped unbound, which
/// is a process that never ran, each goes back as it came.
#[derive(Debug)]
pub(crate) struct Lease {
    pub(super) session: ClientId,
    pub(super) bound: Vec<Bound>,
    /// Each name's holder id, which what lands on it rests under.
    pub(super) ids: HashMap<HandleName, ProcessId>,
    pub(super) returns: Handles,
}

impl Lease {
    /// The engine its handles are on, if one is made yet.
    pub(crate) fn engine(&self) -> Option<usize> {
        self.parked().next().map(|held| held.parts().kv.engine)
    }

    fn parked(&self) -> impl Iterator<Item = &Held> {
        self.bound.iter().filter_map(|bound| match &bound.handle {
            Slot::Parked(held) => Some(held.as_ref()),
            _ => None,
        })
    }

    fn back(&self) -> HashSet<HandleName> {
        self.bound.iter().map(|bound| bound.name).collect()
    }

    /// Does what can fail on what a client wrote before anything moves: a
    /// grammar that does not compile, or bytes that decode to nothing, is
    /// lost for that reason, and the rest go back as they came.
    pub(crate) fn prepare(&mut self) -> Result<(), String> {
        for Bound { slot, handle, .. } in &mut self.bound {
            let made = match handle {
                Slot::Unborn(HandleKind::Matcher(source)) => grammar::compile_matcher(source)
                    .map(|matcher| Slot::Matcher(Box::new(Mutex::new(matcher))))
                    .map_err(|error| error.to_string()),
                Slot::Unborn(HandleKind::Frames(bytes)) => {
                    Frames::decode(bytes).map(|frames| Slot::Frames(Box::new(frames)))
                }
                _ => continue,
            };
            match made {
                Ok(made) => *handle = made,
                Err(why) => {
                    *handle = Slot::Dead(format!("binding `{slot}`: {why}"));
                    return Err(why);
                }
            }
        }
        Ok(())
    }

    /// Whether what it moves in can ever share the device.
    fn fits(&self) -> Result<(), HandleError> {
        let sets: HashSet<_> = self.parked().map(|held| held.parts().kv.id).collect();
        let Some(first) = self.parked().next() else {
            return Ok(());
        };
        let kv = &first.parts().kv;
        let stores = store_registry::get(kv.model, kv.engine);
        store_registry::with_kv_lock(&stores.kv, "handle-fits", |store| {
            // One count over all of them: forks share pages.
            let need = store.held_page_count(&sets).unwrap_or(0);
            let capacity = store.capacity_pages() as usize;
            if need > capacity {
                return Err(HandleError::quota(format!(
                    "these handles map {need} pages and the device holds {capacity}"
                )));
            }
            Ok(())
        })
    }

    /// Readies each handle to leave its holder.
    pub(crate) async fn claim(&self) -> Result<(), String> {
        self.fits()?;
        for held in self.parked() {
            held.claim().await?;
        }
        Ok(())
    }

    /// Puts each handle in `ctx`, under the name its guest takes it by. One
    /// that fails stops here: it is lost for that reason, and the rest go
    /// back as they came.
    pub(crate) fn bind(mut self, ctx: &mut ProcessCtx) -> (Bag, anyhow::Result<()>) {
        let mut bag = Bag {
            session: self.session,
            taken: HashMap::new(),
            lent: Vec::new(),
            lost: Vec::new(),
            given: Vec::new(),
            returns: std::mem::take(&mut self.returns),
            ids: std::mem::take(&mut self.ids),
        };
        while let Some(Bound { slot, name, handle }) = self.bound.pop() {
            match bind_one(ctx, handle) {
                Ok(moved) => {
                    bag.taken.insert(slot, (moved, name));
                    bag.lent.push(name);
                }
                Err(error) => {
                    bag.lost.push((name, format!("binding `{slot}`: {error}")));
                    // What is left goes back as it came, and lands no other way.
                    let back = self.back();
                    bag.returns.retain(|_, name| !back.contains(name));
                    return (bag, Err(error));
                }
            }
        }
        (bag, Ok(()))
    }
}

/// Brings what `ctx` was bound to the device, as one ask of the planner: a
/// process that holds some of it while waiting for the rest would wedge, and a
/// failure here is the process's own, ending it like any restore's.
pub(crate) async fn enter(process: ProcessId) -> Result<(), String> {
    let Some(planner) = crate::planner::planner_of(process) else {
        return Ok(());
    };
    let (model, engine) = planner.locus();
    use crate::inferlet::process::residency;
    let sets = residency::kv_working_set_ids(process, model, engine);
    let states = residency::rs_working_set_ids(process, model, engine);
    let stores = store_registry::get(model, engine);
    // A move ending (kv or state) is the event it waits on: no clock, no
    // deadline.
    let (kv_ended, rs_ended) = (
        store_registry::with_kv_lock(&stores.kv, "handle-enter", |store| store.moves_ended()),
        stores.rs.lock().unwrap().moves_ended(),
    );
    let demand = loop {
        let kv_moved = kv_ended.notified();
        let rs_moved = rs_ended.notified();
        let mut kv_moved = std::pin::pin!(kv_moved);
        let mut rs_moved = std::pin::pin!(rs_moved);
        kv_moved.as_mut().enable();
        rs_moved.as_mut().enable();
        // Reading what is owed and waking the sets is one step: running
        // readers, they cannot lose another page to a peer's eviction.
        let kv_pages = store_registry::with_kv_lock(&stores.kv, "handle-enter", |store| {
            if store.in_move(&sets)? {
                return Ok(None);
            }
            for &set in &sets {
                store.set_parked(set, false);
            }
            Ok(Some(store.swapped_page_count(&sets)? as u32))
        })
        .map_err(|error: crate::store::kv::KvStoreError| format!("its handles: {error}"))?;
        let rs = {
            let store = stores.rs.lock().unwrap();
            (!states.iter().any(|&id| store.in_move(id))).then(|| store.swapped_units(&states))
        };
        if let (Some(kv_pages), Some(rs)) = (kv_pages, rs) {
            break Demand { kv_pages, rs };
        }
        tokio::select! {
            () = &mut kv_moved => {}
            () = &mut rs_moved => {}
        }
    };
    if demand.is_zero() {
        // An eviction that found no page left its fence up.
        for handle in residency::kv_suspend_handles(process, model, engine) {
            handle.unfence();
        }
        return Ok(());
    }
    planner.owe(process, demand);
    planner
        .wait_resident(process)
        .await
        .map_err(|error| format!("restoring its handles: {error}"))
}

fn bind_one(ctx: &mut ProcessCtx, handle: Slot) -> anyhow::Result<Moved> {
    use wasmtime_wasi::WasiView;
    let (kind, rep) = match handle {
        Slot::Parked(held) => {
            let Parts { kv, rs, cursor } = held.into_parts();
            (Kind::WorkingSet, working_set::adopt(ctx, kv, rs, cursor)?)
        }
        Slot::Matcher(matcher) => {
            let matcher = matcher.into_inner().unwrap();
            (Kind::Matcher, ctx.ctx().table.push(matcher)?.rep())
        }
        Slot::Frames(frames) => (Kind::Frames, ctx.ctx().table.push(*frames)?.rep()),
        Slot::Pcm(pcm) => (Kind::Pcm, ctx.ctx().table.push(*pcm)?.rep()),
        Slot::Unborn(HandleKind::WorkingSet) => (Kind::WorkingSet, working_set::materialize(ctx)?),
        // `lend` and `prepare` stand between a client and each of these.
        Slot::Unborn(_) | Slot::Lent | Slot::Dead(_) => {
            anyhow::bail!("a handle that cannot bind reached a process")
        }
    };
    Ok(Moved { kind, rep })
}

impl Drop for Lease {
    fn drop(&mut self) {
        if self.bound.is_empty() && self.returns.is_empty() {
            return;
        }
        self.parked().for_each(Held::unclaim);
        let back = self.back();
        let mut settled: Vec<(HandleName, Landing)> = self
            .returns
            .values()
            .filter(|name| !back.contains(name))
            .map(|&name| (name, Landing::Gone))
            .collect();
        settled.extend(
            self.bound
                .drain(..)
                .map(|bound| (bound.name, Landing::At(bound.handle))),
        );
        crate::server::handles_settled(self.session, settled);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    WorkingSet,
    Matcher,
    Frames,
    Pcm,
}

/// A handle in a process's table, by its table rep.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Moved {
    pub(crate) kind: Kind,
    pub(crate) rep: u32,
}

/// A running process's handles.
#[derive(Debug)]
pub(crate) struct Bag {
    session: ClientId,
    /// Not taken by the guest yet, by name.
    taken: HashMap<String, (Moved, HandleName)>,
    lent: Vec<HandleName>,
    /// What binding lost, and why.
    lost: Vec<(HandleName, String)>,
    given: Vec<(String, Moved)>,
    returns: Handles,
    ids: HashMap<HandleName, ProcessId>,
}

impl Bag {
    /// Teardown could not finish: nothing the process held comes back.
    pub(crate) fn abandon(self) {
        let why = || Landing::At(Slot::Dead("its process could not be torn down".to_string()));
        let settled = self
            .lent
            .iter()
            .chain(self.returns.values())
            .map(|&name| (name, why()))
            .collect();
        crate::server::handles_settled(self.session, settled);
    }

    /// The table rep of what the launch moved in under `name`, once.
    pub(crate) fn take(&mut self, name: &str, kind: Kind) -> Option<u32> {
        let (moved, _) = self.taken.get(name)?;
        if moved.kind != kind {
            return None;
        }
        let rep = moved.rep;
        self.taken.remove(name);
        Some(rep)
    }

    /// Whether `name` is one the launch said to expect.
    pub(crate) fn expects(&self, name: &str) -> bool {
        self.returns.contains_key(name) && !self.given.iter().any(|(given, _)| given == name)
    }

    pub(crate) fn give(&mut self, name: String, moved: Moved) {
        self.given.push((name, moved));
    }

    /// Takes what the guest gave out of `resources`, and tells the session
    /// where everything landed. Runs once every fire has settled.
    pub(crate) fn settle(
        self,
        resources: &mut wasmtime::component::ResourceTable,
        residency: &Mutex<ProcessResidency>,
    ) {
        let mut settled: HashMap<HandleName, Landing> = self
            .lent
            .iter()
            .map(|&name| (name, Landing::Gone))
            .collect();
        for (slot, &name) in &self.returns {
            settled.insert(
                name,
                Landing::At(Slot::Dead(format!(
                    "the process did not give `{slot}` back"
                ))),
            );
        }
        for (name, why) in self.lost {
            settled.insert(name, Landing::At(Slot::Dead(why)));
        }
        for (slot, moved) in self.given {
            let Some(&name) = self.returns.get(&slot) else {
                continue;
            };
            let landed = match take_out(resources, residency, moved, self.ids[&name]) {
                Ok(landed) => landed,
                Err(error) => Slot::Dead(format!("taking `{slot}` from its process: {error}")),
            };
            settled.insert(name, Landing::At(landed));
        }
        // What the guest never took is as it came, whatever became of the guest.
        for (slot, (moved, name)) in self.taken {
            let landed = match take_out(resources, residency, moved, self.ids[&name]) {
                Ok(landed) => landed,
                Err(error) => Slot::Dead(format!("taking `{slot}` from its process: {error}")),
            };
            settled.insert(name, Landing::At(landed));
        }
        crate::server::handles_settled(self.session, settled.into_iter().collect());
    }
}

fn take_out(
    resources: &mut wasmtime::component::ResourceTable,
    residency: &Mutex<ProcessResidency>,
    moved: Moved,
    holder: ProcessId,
) -> anyhow::Result<Slot> {
    use wasmtime::component::Resource;
    let rep = moved.rep;
    match moved.kind {
        Kind::Matcher => {
            let matcher = resources.delete(Resource::<Matcher>::new_own(rep))?;
            return Ok(Slot::Matcher(Box::new(Mutex::new(matcher))));
        }
        Kind::Frames => {
            let frames = resources.delete(Resource::<Frames>::new_own(rep))?;
            return Ok(Slot::Frames(Box::new(frames)));
        }
        Kind::Pcm => {
            let pcm = resources.delete(Resource::<Pcm>::new_own(rep))?;
            return Ok(Slot::Pcm(Box::new(pcm)));
        }
        Kind::WorkingSet => {}
    }
    let ws = resources.delete(Resource::<WorkingSet>::new_own(rep))?;
    let cursor = ws.cursor();
    let kv = resources.delete(Resource::<KvWorkingSet>::new_own(ws.kv))?;
    let rs = match ws.rs {
        Some(rep) => Some(resources.delete(Resource::<RsWorkingSet>::new_own(rep))?),
        None => None,
    };
    {
        let mut held = residency.lock().unwrap();
        held.kv_working_sets.remove(&(kv.model, kv.engine, kv.id));
        if let Some(rs) = &rs {
            held.rs_working_sets.remove(&(rs.model, rs.engine, rs.id));
        }
    }
    // A fire that failed left the cursor ahead of what was written.
    if let Some(why) = kv.pipeline_failure() {
        kv.release();
        if let Some(rs) = rs {
            rs.release();
        }
        return Ok(Slot::Dead(format!("its pipeline failed: {why}")));
    }
    Ok(Slot::Parked(Box::new(Held::park(holder, kv, rs, cursor))))
}
