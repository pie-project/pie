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

    /// Brings its handles back to the device together or not at all: each
    /// stays there until the process has it, so one past what the device
    /// holds would wait on pages only its siblings have.
    fn fits(&self) -> Result<(), HandleError> {
        let Some(first) = self.parked().next() else {
            return Ok(());
        };
        let kv = &first.parts().kv;
        let stores = store_registry::get(kv.model, kv.engine);
        store_registry::with_kv_lock(&stores.kv, "handle-fits", |store| {
            let need: usize = self.parked().map(|held| held.page_count(store)).sum();
            let capacity = store.capacity_pages() as usize;
            if need > capacity {
                return Err(HandleError::quota(format!(
                    "these handles map {need} pages and the device holds {capacity}"
                )));
            }
            Ok(())
        })
    }

    pub(crate) async fn ready(&self, process: ProcessId) -> Result<(), String> {
        self.fits()?;
        for held in self.parked() {
            held.ready(process).await?;
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
        };
        while let Some(Bound { slot, name, handle }) = self.bound.pop() {
            match bind_one(ctx, handle) {
                Ok(moved) => {
                    bag.taken.insert(slot, moved);
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
        self.parked().for_each(Held::rest);
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
    taken: HashMap<String, Moved>,
    lent: Vec<HandleName>,
    /// What binding lost, and why.
    lost: Vec<(HandleName, String)>,
    given: Vec<(String, Moved)>,
    returns: Handles,
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
        let moved = self.taken.get(name)?;
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
            let landed = match take_out(resources, residency, moved) {
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
    Ok(Slot::Parked(Box::new(Held::park(kv, rs, cursor))))
}
