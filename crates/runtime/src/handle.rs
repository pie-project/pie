//! Handles: state a session owns between processes.
//!
//! A process is short and its state long: a working set a client names lives
//! in the session, moves into a process for a launch, and comes back when the
//! process gives it. It has one owner at a time, so nothing here shares a
//! mutable set; a client that wants to keep one forks it first.
//!
//! The planner schedules residency, not programs: each handle out of a
//! process stands in it as a [holder](Held), idle from the start, and the
//! eviction and restore every process gets are the ones it gets.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use client_api::{HandleKind, HandleName, HandleOp, Handles};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::inferlet::host::grammar::Matcher;
use crate::inferlet::host::working_set::{Cursor, WorkingSet};
use crate::inferlet::process::ProcessResidency;
use crate::inferlet::process::residency::{register_residency, unregister_residency};
use crate::inferlet::{ProcessCtx, ProcessId};
use crate::planner::Demand;
use crate::server::ClientId;
use crate::store::kv::working_set::KvWorkingSet;
use crate::store::kv::{KvStore, KvStoreError};
use crate::store::registry as store_registry;
use crate::store::rs::working_set::RsWorkingSet;

const MOVE_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

struct Parts {
    kv: KvWorkingSet,
    rs: Option<RsWorkingSet>,
    cursor: Cursor,
}

/// A working set no process owns.
pub(crate) struct Held {
    holder: ProcessId,
    residency: Arc<Mutex<ProcessResidency>>,
    parts: Option<Parts>,
}

impl std::fmt::Debug for Held {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Held")
            .field("holder", &self.holder)
            .finish()
    }
}

impl Held {
    /// Takes `kv` and `rs` out of their process: no fire may be in flight.
    pub(crate) fn park(kv: KvWorkingSet, rs: Option<RsWorkingSet>, cursor: Cursor) -> Self {
        kv.leave_owner();
        let holder = Uuid::new_v4();
        let residency = Arc::new(Mutex::new(ProcessResidency::default()));
        {
            let mut held = residency.lock().unwrap();
            held.kv_working_sets
                .insert((kv.model, kv.engine, kv.id), kv.suspend_handle());
            if let Some(rs) = &rs {
                held.rs_working_sets.insert((rs.model, rs.engine, rs.id));
            }
        }
        register_residency(holder, Arc::downgrade(&residency));
        crate::planner::place(holder, kv.engine);
        if let Some(planner) = crate::planner::planner_for(kv.model, kv.engine) {
            let stores = store_registry::get(kv.model, kv.engine);
            let owes = store_registry::with_kv_lock(&stores.kv, "handle-park", |store| {
                // Parked, it is no running reader: a pair of forks at rest
                // cannot keep the pages they share on the device.
                store.set_parked(kv.id, true);
                owed(store, &kv, rs.as_ref()).unwrap_or(None)
            });
            planner.register_holder(holder, owes);
        }
        Held {
            holder,
            residency,
            parts: Some(Parts { kv, rs, cursor }),
        }
    }

    fn parts(&self) -> &Parts {
        self.parts.as_ref().expect("taken only by consuming self")
    }

    /// Back on the device, and no one's victim until it has a process.
    ///
    /// Its set is parked from the moment it came here, so a peer's eviction
    /// may have taken pages it shares: what it owes is read here, not known
    /// from whether its own holder was evicted.
    async fn ready(&self, process: ProcessId) -> Result<(), String> {
        let Parts { kv, rs, .. } = self.parts();
        let owed = settled(kv, "handle-ready", |store| {
            let owed = owed(store, kv, rs.as_ref())?;
            if owed.is_none() {
                store.set_parked(kv.id, false);
            }
            Ok(owed)
        })
        .await?;
        if owed.is_none() {
            // An eviction that found no page left its fence up.
            kv.suspend_handle().unfence();
        }
        let Some(planner) = crate::planner::planner_of(self.holder) else {
            return Ok(());
        };
        planner.recall_holder(self.holder, process, owed);
        planner
            .wait_resident(self.holder)
            .await
            .map_err(|error| format!("restoring a handle: {error}"))
    }

    async fn fork(&self) -> Result<Held, String> {
        let Parts { kv, rs, cursor } = self.parts();
        let stores = store_registry::get(kv.model, kv.engine);
        let child = settled(kv, "handle-fork", |store| {
            let child = store.fork(kv.id)?;
            if let Some(owner) = store.owner_of(kv.id) {
                store.set_owner(child, owner)?;
            }
            Ok(child)
        })
        .await?;
        let kv_child = kv.forked(child);
        let rs_child = match rs {
            Some(rs) => {
                let forked = stores.rs.lock().unwrap().fork(rs.id);
                match forked {
                    Ok(id) => Some(RsWorkingSet::new(rs.model, rs.engine, id, rs.geom)),
                    Err(error) => {
                        kv_child.release();
                        return Err(error.to_string());
                    }
                }
            }
            None => None,
        };
        Ok(Held::park(kv_child, rs_child, cursor.clone()))
    }

    fn leave_planner(&self) {
        self.residency.lock().unwrap().kv_working_sets.clear();
        self.residency.lock().unwrap().rs_working_sets.clear();
        if let Some(planner) = crate::planner::planner_of(self.holder) {
            planner.unregister(self.holder);
        }
        crate::planner::release(self.holder);
        unregister_residency(self.holder);
    }

    fn into_parts(mut self) -> Parts {
        self.leave_planner();
        self.parts.take().expect("taken only by consuming self")
    }

    fn stat(&self) -> Value {
        let Parts { kv, cursor, .. } = self.parts();
        let stores = store_registry::get(kv.model, kv.engine);
        let swapped = store_registry::with_kv_lock(&stores.kv, "handle-stat", |store| {
            store
                .swapped_page_count(&HashSet::from([kv.id]))
                .unwrap_or(0)
        });
        json!({
            "state": "parked",
            "tokens": cursor.tail.unwrap_or(cursor.folded),
            "folded": cursor.folded,
            "engine": kv.engine,
            "resident": swapped == 0,
        })
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        if let Some(Parts { kv, rs, .. }) = self.parts.take() {
            self.leave_planner();
            kv.release();
            if let Some(rs) = rs {
                rs.release();
            }
        }
    }
}

/// Runs `f` on the store once no move holds a page `kv` maps: what a set
/// names under a move is not where the flat map says.
async fn settled<T>(
    kv: &KvWorkingSet,
    label: &'static str,
    mut f: impl FnMut(&mut KvStore) -> Result<T, KvStoreError>,
) -> Result<T, String> {
    let stores = store_registry::get(kv.model, kv.engine);
    let set = HashSet::from([kv.id]);
    let deadline = crate::rt::time::Instant::now() + MOVE_WAIT;
    loop {
        let ended = store_registry::with_kv_lock(&stores.kv, label, |store| store.moves_ended());
        let notified = ended.notified();
        let mut notified = std::pin::pin!(notified);
        notified.as_mut().enable();
        let done = store_registry::with_kv_lock(&stores.kv, label, |store| {
            if store.in_move(&set)? {
                return Ok(None);
            }
            f(store).map(Some)
        })
        .map_err(|error| error.to_string())?;
        if let Some(done) = done {
            return Ok(done);
        }
        let now = crate::rt::time::Instant::now();
        if now >= deadline {
            return Err("its pages are still moving".to_string());
        }
        let _ = crate::rt::time::timeout(deadline - now, notified).await;
    }
}

/// What the set has off the device.
fn owed(
    store: &KvStore,
    kv: &KvWorkingSet,
    rs: Option<&RsWorkingSet>,
) -> Result<Option<Demand>, KvStoreError> {
    let kv_pages = store.swapped_page_count(&HashSet::from([kv.id]))? as u32;
    let rs = rs.map_or_else(Default::default, |rs| {
        let stores = store_registry::get(rs.model, rs.engine);
        stores
            .rs
            .lock()
            .unwrap()
            .swapped_units(&HashSet::from([rs.id]))
    });
    let demand = Demand { kv_pages, rs };
    Ok((!demand.is_zero()).then_some(demand))
}

#[derive(Debug)]
pub(crate) enum Slot {
    /// Named, and nothing else yet: a process makes it, on its own engine.
    Unborn(HandleKind),
    Parked(Box<Held>),
    /// Host memory and nothing else: it moves as it is. The lock is for
    /// `Sync`, which the actor holding the table needs and it alone reaches.
    Matcher(Mutex<Matcher>),
    /// In a process, or promised by one: frames naming it wait.
    Lent,
    /// Gone for a reason the next use hears.
    Dead(String),
}

/// A session's handles. An actor owns it, so names resolve in the order the
/// client sent them.
#[derive(Debug, Default)]
pub(crate) struct Table {
    slots: HashMap<HandleName, Slot>,
}

impl Table {
    pub(crate) fn lent(&self, name: HandleName) -> bool {
        matches!(self.slots.get(&name), Some(Slot::Lent))
    }

    pub(crate) async fn apply(&mut self, ops: Vec<HandleOp>) -> Vec<Value> {
        let mut answers = Vec::with_capacity(ops.len());
        for op in ops {
            answers.push(match self.op(op).await {
                Ok(answer) => answer,
                Err(error) => json!({ "error": error }),
            });
        }
        answers
    }

    async fn op(&mut self, op: HandleOp) -> Result<Value, String> {
        match op {
            HandleOp::New { name, kind } => {
                self.claim(name)?;
                self.slots.insert(name, Slot::Unborn(kind));
                Ok(json!({ "ok": true }))
            }
            HandleOp::Fork { name, from } => {
                self.claim(name)?;
                let child = match self.slots.get(&from) {
                    Some(Slot::Unborn(kind)) => Slot::Unborn(kind.clone()),
                    Some(Slot::Parked(held)) => Slot::Parked(Box::new(held.fork().await?)),
                    Some(Slot::Matcher(matcher)) => {
                        Slot::Matcher(Mutex::new(matcher.lock().unwrap().fork()))
                    }
                    other => return Err(refusal(from, other)),
                };
                self.slots.insert(name, child);
                Ok(json!({ "ok": true }))
            }
            HandleOp::Close { name } => match self.slots.remove(&name) {
                Some(_) => Ok(json!({ "ok": true })),
                None => Err(refusal(name, None)),
            },
            HandleOp::Stat { name } => match self.slots.get(&name) {
                Some(Slot::Unborn(HandleKind::WorkingSet)) => {
                    Ok(json!({ "state": "unborn", "tokens": 0, "folded": 0, "engine": null }))
                }
                Some(Slot::Unborn(_)) => Ok(json!({ "state": "unborn" })),
                Some(Slot::Parked(held)) => Ok(held.stat()),
                Some(Slot::Matcher(matcher)) => {
                    let terminated = matcher.lock().unwrap().is_terminated();
                    Ok(json!({ "state": "parked", "terminated": terminated }))
                }
                other => Err(refusal(name, other)),
            },
        }
    }

    fn claim(&self, name: HandleName) -> Result<(), String> {
        match self.slots.get(&name) {
            None | Some(Slot::Dead(_)) => Ok(()),
            Some(_) => Err(format!("handle {name} is already in use")),
        }
    }

    /// Moves what a launch names out of the table. Nothing moves if it fails.
    pub(crate) fn lend(
        &mut self,
        session: ClientId,
        handles: Handles,
        returns: Handles,
        state_room: Option<u32>,
    ) -> Result<Option<Lease>, String> {
        if handles.is_empty() && returns.is_empty() {
            return Ok(None);
        }
        let mut seen = HashSet::new();
        for (slot, &name) in &handles {
            if !seen.insert(name) {
                return Err(format!(
                    "handle {name} is bound twice, the second as `{slot}`"
                ));
            }
            let slot = self.slots.get(&name);
            if matches!(slot, None | Some(Slot::Dead(_) | Slot::Lent)) {
                return Err(refusal(name, slot));
            }
        }
        let mut engines = handles
            .values()
            .filter_map(|name| match self.slots.get(name) {
                Some(Slot::Parked(held)) => Some(held.parts().kv.engine),
                _ => None,
            });
        if let Some(first) = engines.next()
            && engines.any(|engine| engine != first)
        {
            return Err("its handles are on different engines".to_string());
        }
        let unborn = handles
            .values()
            .filter(|name| {
                matches!(
                    self.slots.get(name),
                    Some(Slot::Unborn(HandleKind::WorkingSet))
                )
            })
            .count();
        if state_room.is_some_and(|room| (room as usize) < unborn) {
            return Err(format!(
                "quota: {unborn} new working set(s) would leave more recurrent state at rest \
                 than the host holds; close a handle, or raise `[engine] max_state_slots`"
            ));
        }
        let mut landing = HashSet::new();
        for (slot, &name) in &returns {
            if !landing.insert(name) {
                return Err(format!(
                    "handle {name} is returned twice, the second as `{slot}`"
                ));
            }
            if !seen.contains(&name) {
                self.claim(name)?;
            }
        }
        let bound = handles
            .into_iter()
            .map(|(slot, name)| {
                let held = self.slots.insert(name, Slot::Lent);
                (slot, name, held.expect("checked above"))
            })
            .collect();
        for &name in returns.values() {
            self.slots.insert(name, Slot::Lent);
        }
        Ok(Some(Lease {
            session,
            bound,
            returns,
        }))
    }

    /// What a process's end settled: a name with no slot here is gone.
    pub(crate) fn settle(&mut self, settled: Vec<(HandleName, Option<Slot>)>) {
        for (name, slot) in settled {
            match slot {
                Some(slot) => self.slots.insert(name, slot),
                None => self.slots.remove(&name),
            };
        }
    }
}

/// How many more sets can come to rest on the most open engine, where it has
/// recurrent state to hold. That state has no disk tier and a running lane
/// may want every device slot, so what rests must fit the host: past that, a
/// fire would starve on state nothing can move.
pub(crate) fn state_room() -> Option<u32> {
    store_registry::all_for_model(0)
        .iter()
        .filter_map(|stores| {
            let rs = stores.rs.lock().unwrap();
            let device = rs.capacity().slots;
            (device > 0).then(|| {
                rs.available()
                    .slots
                    .saturating_add(rs.host_available().slots)
                    .saturating_sub(device)
            })
        })
        .max()
}

fn refusal(name: HandleName, slot: Option<&Slot>) -> String {
    match slot {
        None => format!("unknown: no handle {name} in this session"),
        Some(Slot::Dead(why)) => format!("dead: handle {name}: {why}"),
        Some(Slot::Lent) => format!("handle {name} is in a process"),
        Some(_) => format!("handle {name} cannot do that"),
    }
}

/// A launch's handles on their way into its process. Dropped unbound, which
/// is a process that never ran, each goes back as it came.
#[derive(Debug)]
pub(crate) struct Lease {
    session: ClientId,
    bound: Vec<(String, HandleName, Slot)>,
    returns: Handles,
}

impl Lease {
    /// The engine its handles are on, if one is made yet.
    pub(crate) fn engine(&self) -> Option<usize> {
        self.bound.iter().find_map(|(_, _, slot)| match slot {
            Slot::Parked(held) => Some(held.parts().kv.engine),
            _ => None,
        })
    }

    pub(crate) async fn ready(&self, process: ProcessId) -> Result<(), String> {
        for (_, _, slot) in &self.bound {
            if let Slot::Parked(held) = slot {
                held.ready(process).await?;
            }
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
        while let Some((slot, name, handle)) = self.bound.pop() {
            match bind_one(ctx, handle) {
                Ok(moved) => {
                    bag.taken.insert(slot, moved);
                    bag.lent.push(name);
                }
                Err(error) => {
                    bag.lost.push((name, format!("binding `{slot}`: {error}")));
                    return (bag, Err(error));
                }
            }
        }
        (bag, Ok(()))
    }
}

fn bind_one(ctx: &mut ProcessCtx, handle: Slot) -> anyhow::Result<Moved> {
    use crate::inferlet::host::{grammar, working_set};
    use wasmtime_wasi::WasiView;
    Ok(match handle {
        Slot::Parked(held) => {
            let Parts { kv, rs, cursor } = held.into_parts();
            Moved::WorkingSet(working_set::adopt(ctx, kv, rs, cursor)?)
        }
        Slot::Matcher(matcher) => {
            let matcher = matcher.into_inner().unwrap();
            Moved::Matcher(ctx.ctx().table.push(matcher)?.rep())
        }
        Slot::Unborn(HandleKind::Matcher(source)) => {
            let matcher = grammar::compile_matcher(&source)?;
            Moved::Matcher(ctx.ctx().table.push(matcher)?.rep())
        }
        _ => Moved::WorkingSet(working_set::materialize(ctx)?),
    })
}

impl Drop for Lease {
    fn drop(&mut self) {
        if self.bound.is_empty() && self.returns.is_empty() {
            return;
        }
        let back: HashSet<HandleName> = self.bound.iter().map(|(_, name, _)| *name).collect();
        let mut settled: Vec<(HandleName, Option<Slot>)> = self
            .returns
            .values()
            .filter(|name| !back.contains(name))
            .map(|&name| (name, None))
            .collect();
        settled.extend(
            self.bound
                .drain(..)
                .map(|(_, name, slot)| (name, Some(slot))),
        );
        crate::server::handles_settled(self.session, settled);
    }
}

/// A handle in a process's table, by its table rep.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Moved {
    WorkingSet(u32),
    Matcher(u32),
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
        let why = || Some(Slot::Dead("its process could not be torn down".to_string()));
        let settled = self
            .lent
            .iter()
            .chain(self.returns.values())
            .map(|&name| (name, why()))
            .collect();
        crate::server::handles_settled(self.session, settled);
    }

    pub(crate) fn take_working_set(&mut self, name: &str) -> Option<u32> {
        match self.taken.get(name)? {
            &Moved::WorkingSet(rep) => {
                self.taken.remove(name);
                Some(rep)
            }
            Moved::Matcher(_) => None,
        }
    }

    pub(crate) fn take_matcher(&mut self, name: &str) -> Option<u32> {
        match self.taken.get(name)? {
            &Moved::Matcher(rep) => {
                self.taken.remove(name);
                Some(rep)
            }
            Moved::WorkingSet(_) => None,
        }
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
        let mut settled: HashMap<HandleName, Option<Slot>> =
            self.lent.iter().map(|&name| (name, None)).collect();
        for (slot, &name) in &self.returns {
            settled.insert(
                name,
                Some(Slot::Dead(format!(
                    "the process did not give `{slot}` back"
                ))),
            );
        }
        for (name, why) in self.lost {
            settled.insert(name, Some(Slot::Dead(why)));
        }
        for (slot, moved) in self.given {
            let Some(&name) = self.returns.get(&slot) else {
                continue;
            };
            let landed = match take_out(resources, residency, moved) {
                Ok(landed) => landed,
                Err(error) => Slot::Dead(format!("taking `{slot}` from its process: {error}")),
            };
            settled.insert(name, Some(landed));
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
    let rep = match moved {
        Moved::Matcher(rep) => {
            return Ok(Slot::Matcher(Mutex::new(resources.delete(Resource::<
                Matcher,
            >::new_own(
                rep
            ))?)));
        }
        Moved::WorkingSet(rep) => rep,
    };
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
    Ok(Slot::Parked(Box::new(Held::park(kv, rs, cursor))))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(pairs: &[(&str, HandleName)]) -> Handles {
        pairs
            .iter()
            .map(|&(slot, name)| (slot.into(), name))
            .collect()
    }

    fn new(name: HandleName) -> HandleOp {
        HandleOp::New {
            name,
            kind: HandleKind::WorkingSet,
        }
    }

    async fn answer(table: &mut Table, op: HandleOp) -> Value {
        table.apply(vec![op]).await.remove(0)
    }

    /// A process that ended: nobody hears, so these tests read the table.
    fn spent(lease: Lease) {
        let mut lease = lease;
        lease.bound.clear();
        lease.returns.clear();
    }

    #[tokio::test]
    async fn handle_every_case() {
        an_unborn_handle_is_a_name_and_nothing_else().await;
        a_name_is_taken_once().await;
        one_bad_op_does_not_stop_the_rest().await;
        a_launch_moves_all_it_names_or_none().await;
        a_name_a_launch_promises_waits_like_one_it_took().await;
        a_settled_end_lands_each_name().await;
    }

    async fn an_unborn_handle_is_a_name_and_nothing_else() {
        let mut table = Table::default();
        answer(&mut table, new(1)).await;
        let stat = answer(&mut table, HandleOp::Stat { name: 1 }).await;
        assert_eq!(stat["state"], "unborn");
        assert_eq!(stat["engine"], Value::Null);
        answer(&mut table, HandleOp::Fork { name: 2, from: 1 }).await;
        let child = answer(&mut table, HandleOp::Stat { name: 2 }).await;
        assert_eq!(child["state"], "unborn", "there is nothing to share yet");
        answer(&mut table, HandleOp::Close { name: 1 }).await;
        let gone = answer(&mut table, HandleOp::Stat { name: 1 }).await;
        assert!(gone["error"].as_str().unwrap().starts_with("unknown"));
    }

    async fn a_name_is_taken_once() {
        let mut table = Table::default();
        answer(&mut table, new(1)).await;
        let again = answer(&mut table, new(1)).await;
        assert!(again["error"].as_str().unwrap().contains("in use"));
        let fork = answer(&mut table, HandleOp::Fork { name: 1, from: 1 }).await;
        assert!(fork["error"].as_str().unwrap().contains("in use"));
    }

    async fn one_bad_op_does_not_stop_the_rest() {
        let mut table = Table::default();
        let answers = table.apply(vec![HandleOp::Close { name: 9 }, new(1)]).await;
        assert!(answers[0].get("error").is_some());
        assert_eq!(answers[1]["ok"], true);
    }

    async fn a_launch_moves_all_it_names_or_none() {
        let mut table = Table::default();
        answer(&mut table, new(1)).await;
        let refused = table.lend(7, names(&[("a", 1), ("b", 2)]), Handles::new(), None);
        assert!(refused.unwrap_err().starts_with("unknown"));
        assert!(!table.lent(1), "the known one stays");
        let twice = table.lend(7, names(&[("a", 1), ("b", 1)]), Handles::new(), None);
        assert!(twice.unwrap_err().contains("bound twice"));
        assert!(!table.lent(1));
        let full = table.lend(7, names(&[("a", 1)]), Handles::new(), Some(0));
        assert!(full.unwrap_err().starts_with("quota"));
        assert!(!table.lent(1), "a full host moves nothing");
        assert!(
            table
                .lend(7, Handles::new(), Handles::new(), None)
                .unwrap()
                .is_none()
        );
    }

    async fn a_name_a_launch_promises_waits_like_one_it_took() {
        let mut table = Table::default();
        answer(&mut table, new(1)).await;
        let lease = table
            .lend(
                7,
                names(&[("ws", 1)]),
                names(&[("ws", 1), ("child", 2)]),
                None,
            )
            .unwrap()
            .unwrap();
        assert!(table.lent(1) && table.lent(2));
        assert_eq!(lease.engine(), None, "an unborn handle is on no engine");
        let in_use = table.lend(7, names(&[("ws", 1)]), Handles::new(), None);
        assert!(in_use.unwrap_err().contains("in a process"));
        spent(lease);
    }

    async fn a_settled_end_lands_each_name() {
        let mut table = Table::default();
        answer(&mut table, new(1)).await;
        answer(&mut table, new(2)).await;
        table.settle(vec![
            (1, None),
            (
                2,
                Some(Slot::Dead("the process did not give `ws` back".into())),
            ),
        ]);
        let gone = answer(&mut table, HandleOp::Stat { name: 1 }).await;
        assert!(gone["error"].as_str().unwrap().starts_with("unknown"));
        let dead = answer(&mut table, HandleOp::Stat { name: 2 }).await;
        assert!(dead["error"].as_str().unwrap().contains("did not give"));
        let refused = table.lend(7, names(&[("ws", 2)]), Handles::new(), None);
        assert!(refused.unwrap_err().starts_with("dead"), "the reason stays");
        answer(&mut table, new(2)).await;
        assert!(
            table
                .lend(7, names(&[("ws", 2)]), Handles::new(), None)
                .is_ok(),
            "a dead name is free"
        );
    }
}
