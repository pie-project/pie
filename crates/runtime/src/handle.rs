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

use crate::inferlet::host::frames::{Frames, Pcm};
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
use crate::store::rs::{RsStore, RsUnits};

const MOVE_WAIT: std::time::Duration = std::time::Duration::from_secs(10);
const RS_MOVE_POLL: std::time::Duration = std::time::Duration::from_millis(5);

/// Frames a session keeps waiting behind a lent handle. They are what a client
/// has sent and nothing has answered, so they need a bound of their own.
pub(crate) const MAX_WAITING_FRAMES: usize = 256;

/// The most handles one session names at once: a name costs almost nothing,
/// but not nothing, and a client that never closes one should hear so.
const MAX_HANDLES: usize = 4096;

/// The most pictures and sound one session holds, by what they decode to.
/// Nothing moves these off the host, so this is all that bounds them.
const MAX_MEDIA_BYTES: usize = 8 << 30;

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
            let swapped = store_registry::with_kv_lock(&stores.kv, "handle-park", |store| {
                // Parked, it is no running reader: a pair of forks at rest
                // cannot keep the pages they share on the device.
                store.set_parked(kv.id, true);
                swapped_pages(store, &kv).unwrap_or(0)
            });
            planner.register_holder(holder, demand(swapped, rs.as_ref()));
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
        if let Some(rs) = rs {
            rs_settled(rs, |_| ()).await?;
        }
        let rs_home = rs.as_ref().is_none_or(|rs| rs_units(rs).is_zero());
        let kv_pages = settled(kv, "handle-ready", |store| {
            let pages = swapped_pages(store, kv)?;
            if pages == 0 && rs_home {
                store.set_parked(kv.id, false);
            }
            Ok(pages)
        })
        .await?;
        let owed = demand(kv_pages, rs.as_ref());
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

    /// A recall that came to nothing: it is at rest again, as if never asked.
    fn rest(&self) {
        let Parts { kv, .. } = self.parts();
        let stores = store_registry::get(kv.model, kv.engine);
        store_registry::with_kv_lock(&stores.kv, "handle-rest", |store| {
            store.set_parked(kv.id, true);
        });
        if let Some(planner) = crate::planner::planner_of(self.holder) {
            planner.rest_holder(self.holder);
        }
    }

    /// The pages this set maps, wherever they are.
    fn page_count(&self, store: &KvStore) -> usize {
        let set = HashSet::from([self.parts().kv.id]);
        store.held_page_count(&set).unwrap_or(0) + store.swapped_page_count(&set).unwrap_or(0)
    }

    /// Each store forks under its own lock, with no move of its own under
    /// way: a move of the other's cannot touch what this one holds.
    async fn fork(&self) -> Result<Held, String> {
        let Parts { kv, rs, cursor } = self.parts();
        let rs_child = match rs {
            Some(rs) => {
                let id = rs_settled(rs, |store| store.fork(rs.id))
                    .await?
                    .map_err(|error| error.to_string())?;
                Some(RsWorkingSet::new(rs.model, rs.engine, id, rs.geom))
            }
            None => None,
        };
        let child = settled(kv, "handle-fork", |store| {
            let child = store.fork(kv.id)?;
            if let Some(owner) = store.owner_of(kv.id) {
                store.set_owner(child, owner)?;
            }
            Ok(child)
        })
        .await;
        match child {
            Ok(child) => Ok(Held::park(kv.forked(child), rs_child, cursor.clone())),
            Err(why) => {
                if let Some(rs) = rs_child {
                    rs.release();
                }
                Err(why)
            }
        }
    }

    /// Whether its next write takes state it does not hold yet.
    fn shares_state(&self) -> bool {
        self.parts().rs.as_ref().is_some_and(|rs| {
            let stores = store_registry::get(rs.model, rs.engine);
            stores.rs.lock().unwrap().shares_state(rs.id)
        })
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

/// Runs `f` on the store once no move holds a page `kv` maps: what a set names
/// under a move is not where its map says.
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

/// [`settled`] for recurrent state, whose moves end with no signal.
async fn rs_settled<T>(
    rs: &RsWorkingSet,
    mut f: impl FnMut(&mut RsStore) -> T,
) -> Result<T, String> {
    let stores = store_registry::get(rs.model, rs.engine);
    let deadline = crate::rt::time::Instant::now() + MOVE_WAIT;
    loop {
        {
            let mut store = stores.rs.lock().unwrap();
            if !store.in_move(rs.id) {
                return Ok(f(&mut store));
            }
        }
        if crate::rt::time::Instant::now() >= deadline {
            return Err("its state is still moving".to_string());
        }
        let _ = crate::rt::time::timeout(RS_MOVE_POLL, std::future::pending::<()>()).await;
    }
}

fn swapped_pages(store: &KvStore, kv: &KvWorkingSet) -> Result<u32, KvStoreError> {
    Ok(store.swapped_page_count(&HashSet::from([kv.id]))? as u32)
}

fn rs_units(rs: &RsWorkingSet) -> RsUnits {
    let stores = store_registry::get(rs.model, rs.engine);
    let held = stores.rs.lock().unwrap();
    held.swapped_units(&HashSet::from([rs.id]))
}

/// What a set owes the device, given the pages it has off it.
fn demand(kv_pages: u32, rs: Option<&RsWorkingSet>) -> Option<Demand> {
    let demand = Demand {
        kv_pages,
        rs: rs.map(rs_units).unwrap_or_default(),
    };
    (!demand.is_zero()).then_some(demand)
}

#[derive(Debug)]
pub(crate) enum Slot {
    /// Named, and nothing else yet: a process makes it, on its own engine.
    Unborn(HandleKind),
    Parked(Box<Held>),
    /// Host memory and nothing else: it moves as it is. The lock is for
    /// `Sync`, which the actor holding the table needs and it alone reaches.
    Matcher(Box<Mutex<Matcher>>),
    /// Host memory never written once made: a fork shares it.
    Frames(Box<Frames>),
    Pcm(Box<Pcm>),
    /// In a process, or promised by one: frames naming it wait.
    Lent,
    /// Gone for a reason the next use hears.
    Dead(String),
}

impl Slot {
    /// What tells a buffer shared between forks from another.
    fn media_key(&self) -> Option<usize> {
        match self {
            Slot::Frames(frames) => frames.buffer(),
            Slot::Pcm(pcm) => Some(pcm.samples.as_ptr() as usize),
            _ => None,
        }
    }

    fn media_bytes(&self) -> usize {
        match self {
            Slot::Unborn(HandleKind::Frames(bytes)) => bytes.len(),
            Slot::Frames(frames) => frames.rgb8().map_or(0, <[u8]>::len),
            Slot::Pcm(pcm) => pcm.samples.len() * size_of::<f32>(),
            _ => 0,
        }
    }
}

/// Encoding is slow and the session has other frames to serve.
pub(crate) type Encode = Box<dyn FnOnce() -> Result<Vec<u8>, String> + Send>;

/// A session's handles. An actor owns it, so names resolve in the order the
/// client sent them.
#[derive(Debug)]
pub(crate) struct Table {
    slots: HashMap<HandleName, Slot>,
    max_media_bytes: usize,
}

impl Default for Table {
    fn default() -> Self {
        Table {
            slots: HashMap::new(),
            max_media_bytes: MAX_MEDIA_BYTES,
        }
    }
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
                let slot = Slot::Unborn(kind);
                self.room_for(&slot)?;
                self.slots.insert(name, slot);
                Ok(json!({ "ok": true }))
            }
            HandleOp::Fork { name, from } => {
                self.claim(name)?;
                let child = match self.slots.get(&from) {
                    Some(Slot::Unborn(kind)) => {
                        let child = Slot::Unborn(kind.clone());
                        self.room_for(&child)?;
                        child
                    }
                    Some(Slot::Parked(held)) => Slot::Parked(Box::new(held.fork().await?)),
                    Some(Slot::Matcher(matcher)) => {
                        Slot::Matcher(Box::new(Mutex::new(matcher.lock().unwrap().fork())))
                    }
                    Some(Slot::Frames(frames)) => Slot::Frames(frames.clone()),
                    Some(Slot::Pcm(pcm)) => Slot::Pcm(pcm.clone()),
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
                Some(Slot::Frames(frames)) => Ok(json!({
                    "state": "parked",
                    "width": frames.width,
                    "height": frames.height,
                    "count": frames.count,
                    "fps": frames.fps,
                    "bytes": frames.rgb8().map_or(0, <[u8]>::len),
                })),
                Some(Slot::Pcm(pcm)) => Ok(json!({
                    "state": "parked",
                    "rate": pcm.rate,
                    "channels": pcm.channels,
                    "samples": pcm.samples.len(),
                })),
                Some(Slot::Parked(held)) => Ok(held.stat()),
                Some(Slot::Matcher(matcher)) => {
                    let terminated = matcher.lock().unwrap().is_terminated();
                    Ok(json!({ "state": "parked", "terminated": terminated }))
                }
                other => Err(refusal(name, other)),
            },
        }
    }

    /// The media `slot` would add within what the session may hold.
    fn room_for(&self, slot: &Slot) -> Result<(), String> {
        // A fork shares what it forked: what is shared counts once.
        let mut seen = HashSet::new();
        let held: usize = self
            .slots
            .values()
            .chain(std::iter::once(slot))
            .filter(|slot| slot.media_key().is_none_or(|key| seen.insert(key)))
            .map(Slot::media_bytes)
            .sum();
        if held > self.max_media_bytes {
            return Err(format!(
                "quota: this would hold {held} bytes of media; close a handle"
            ));
        }
        Ok(())
    }

    /// What encodes a handle for a client, and the name of the file. It owns
    /// what it reads, which is never written, so it runs on while the table
    /// moves on.
    pub(crate) fn read(&self, name: HandleName, format: &str) -> Result<(Encode, String), String> {
        use crate::inferlet::host::frames::{
            audio_extension, audio_format, image_extension, image_format,
        };
        match self.slots.get(&name) {
            Some(Slot::Frames(frames)) => {
                let format =
                    image_format(format).ok_or_else(|| format!("no image format `{format}`"))?;
                let frames = frames.clone();
                Ok((
                    Box::new(move || frames.encode(format)),
                    format!("{name}.{}", image_extension(format)),
                ))
            }
            Some(Slot::Pcm(pcm)) => {
                let format =
                    audio_format(format).ok_or_else(|| format!("no audio format `{format}`"))?;
                let pcm = pcm.clone();
                Ok((
                    Box::new(move || Ok(pcm.encode(format))),
                    format!("{name}.{}", audio_extension(format)),
                ))
            }
            other => Err(refusal(name, other)),
        }
    }

    fn claim(&self, name: HandleName) -> Result<(), String> {
        match self.slots.get(&name) {
            None if self.slots.len() >= MAX_HANDLES => Err(format!(
                "quota: this session names {MAX_HANDLES} handles; close one"
            )),
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
        let growing = handles
            .values()
            .filter(|name| match self.slots.get(name) {
                Some(Slot::Unborn(HandleKind::WorkingSet)) => true,
                Some(Slot::Parked(held)) => held.shares_state(),
                _ => false,
            })
            .count();
        if state_room.is_some_and(|room| (room as usize) < growing) {
            return Err(format!(
                "quota: {growing} working set(s) taking new state would leave more recurrent \
                 state at rest than the host holds; close a handle, or raise \
                 `[engine] max_state_slots`"
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
            self.slots.remove(&name);
            let Some(slot) = slot else {
                continue;
            };
            let slot = match self.room_for(&slot) {
                Ok(()) => slot,
                Err(why) => Slot::Dead(why),
            };
            self.slots.insert(name, slot);
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

    /// Does what can fail on what a client wrote before anything moves: a
    /// grammar that does not compile, or bytes that decode to nothing, is
    /// lost for that reason, and the rest go back as they came.
    pub(crate) fn prepare(&mut self) -> Result<(), String> {
        use crate::inferlet::host::grammar;
        for (slot, _, handle) in &mut self.bound {
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
    fn fits(&self) -> Result<(), String> {
        let held = || {
            self.bound.iter().filter_map(|(_, _, slot)| match slot {
                Slot::Parked(held) => Some(held),
                _ => None,
            })
        };
        let Some(first) = held().next() else {
            return Ok(());
        };
        let kv = &first.parts().kv;
        let stores = store_registry::get(kv.model, kv.engine);
        store_registry::with_kv_lock(&stores.kv, "handle-fits", |store| {
            let need: usize = held().map(|held| held.page_count(store)).sum();
            let capacity = store.capacity_pages() as usize;
            if need > capacity {
                return Err(format!(
                    "quota: these handles map {need} pages and the device holds {capacity}"
                ));
            }
            Ok(())
        })
    }

    pub(crate) async fn ready(&self, process: ProcessId) -> Result<(), String> {
        self.fits()?;
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
                    // What is left goes back as it came, and lands no other way.
                    let back: HashSet<HandleName> =
                        self.bound.iter().map(|(_, name, _)| *name).collect();
                    bag.returns.retain(|_, name| !back.contains(name));
                    return (bag, Err(error));
                }
            }
        }
        (bag, Ok(()))
    }
}

fn bind_one(ctx: &mut ProcessCtx, handle: Slot) -> anyhow::Result<Moved> {
    use crate::inferlet::host::working_set;
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
        Slot::Frames(frames) => Moved::Frames(ctx.ctx().table.push(*frames)?.rep()),
        Slot::Pcm(pcm) => Moved::Pcm(ctx.ctx().table.push(*pcm)?.rep()),
        Slot::Unborn(HandleKind::WorkingSet) => Moved::WorkingSet(working_set::materialize(ctx)?),
        // `lend` and `prepare` stand between a client and each of these.
        Slot::Unborn(_) | Slot::Lent | Slot::Dead(_) => {
            anyhow::bail!("a handle that cannot bind reached a process")
        }
    })
}

impl Drop for Lease {
    fn drop(&mut self) {
        if self.bound.is_empty() && self.returns.is_empty() {
            return;
        }
        for (_, _, slot) in &self.bound {
            if let Slot::Parked(held) = slot {
                held.rest();
            }
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
    Frames(u32),
    Pcm(u32),
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
            &Moved::WorkingSet(rep) => self.taken.remove(name).map(|_| rep),
            _ => None,
        }
    }

    pub(crate) fn take_matcher(&mut self, name: &str) -> Option<u32> {
        match self.taken.get(name)? {
            &Moved::Matcher(rep) => self.taken.remove(name).map(|_| rep),
            _ => None,
        }
    }

    pub(crate) fn take_frames(&mut self, name: &str) -> Option<u32> {
        match self.taken.get(name)? {
            &Moved::Frames(rep) => self.taken.remove(name).map(|_| rep),
            _ => None,
        }
    }

    pub(crate) fn take_pcm(&mut self, name: &str) -> Option<u32> {
        match self.taken.get(name)? {
            &Moved::Pcm(rep) => self.taken.remove(name).map(|_| rep),
            _ => None,
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
            let matcher = resources.delete(Resource::<Matcher>::new_own(rep))?;
            return Ok(Slot::Matcher(Box::new(Mutex::new(matcher))));
        }
        Moved::Frames(rep) => {
            return Ok(Slot::Frames(Box::new(resources.delete(Resource::<
                Frames,
            >::new_own(
                rep
            ))?)));
        }
        Moved::Pcm(rep) => {
            return Ok(Slot::Pcm(Box::new(
                resources.delete(Resource::<Pcm>::new_own(rep))?,
            )));
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
        a_session_names_only_so_many().await;
        media_past_the_session_limit_is_refused_or_dead().await;
        a_fork_counts_against_the_limit_as_what_it_shares().await;
        a_name_is_taken_once().await;
        one_bad_op_does_not_stop_the_rest().await;
        a_launch_moves_all_it_names_or_none().await;
        a_name_a_launch_promises_waits_like_one_it_took().await;
        a_settled_end_lands_each_name().await;
    }

    async fn a_session_names_only_so_many() {
        let mut table = Table::default();
        for name in 0..MAX_HANDLES as HandleName {
            table
                .slots
                .insert(name, Slot::Unborn(HandleKind::WorkingSet));
        }
        let over = answer(&mut table, new(MAX_HANDLES as HandleName)).await;
        assert!(over["error"].as_str().unwrap().starts_with("quota"));
        let fork = HandleOp::Fork {
            name: MAX_HANDLES as HandleName,
            from: 0,
        };
        assert!(
            answer(&mut table, fork).await["error"]
                .as_str()
                .unwrap()
                .starts_with("quota")
        );
        answer(&mut table, HandleOp::Close { name: 0 }).await;
        assert_eq!(
            answer(&mut table, new(MAX_HANDLES as HandleName)).await["ok"],
            true
        );
    }

    async fn media_past_the_session_limit_is_refused_or_dead() {
        let mut table = Table {
            max_media_bytes: 8,
            ..Table::default()
        };
        let upload = |name, len| HandleOp::New {
            name,
            kind: HandleKind::Frames(vec![0u8; len].into()),
        };
        assert_eq!(answer(&mut table, upload(1, 6)).await["ok"], true);
        let over = answer(&mut table, upload(2, 6)).await;
        assert!(over["error"].as_str().unwrap().starts_with("quota"));

        let frames = Frames::from_rgb8(vec![0; 12], 2, 2, 1, 0.0).unwrap();
        table.settle(vec![(3, Some(Slot::Frames(Box::new(frames))))]);
        let landed = answer(&mut table, HandleOp::Stat { name: 3 }).await;
        assert!(
            landed["error"].as_str().unwrap().contains("quota"),
            "what a process gave past the limit is lost, and says so: {landed}"
        );
    }

    async fn a_fork_counts_against_the_limit_as_what_it_shares() {
        let mut table = Table {
            max_media_bytes: 16,
            ..Table::default()
        };
        let upload = HandleOp::New {
            name: 1,
            kind: HandleKind::Frames(vec![0u8; 10].into()),
        };
        assert_eq!(answer(&mut table, upload).await["ok"], true);
        let copy = answer(&mut table, HandleOp::Fork { name: 2, from: 1 }).await;
        assert!(
            copy["error"].as_str().unwrap().starts_with("quota"),
            "an upload's fork copies its bytes: {copy}"
        );

        assert_eq!(
            answer(&mut table, HandleOp::Close { name: 1 }).await["ok"],
            true
        );
        let decoded = Frames::from_rgb8(vec![0; 12], 2, 2, 1, 0.0).unwrap();
        table.settle(vec![(3, Some(Slot::Frames(Box::new(decoded))))]);
        let fork = answer(&mut table, HandleOp::Fork { name: 4, from: 3 }).await;
        assert_eq!(fork["ok"], true, "decoded pixels are shared: {fork}");
        let held = Frames::from_rgb8(vec![0; 3], 1, 1, 1, 0.0).unwrap();
        table.settle(vec![(5, Some(Slot::Frames(Box::new(held))))]);
        assert_eq!(
            answer(&mut table, HandleOp::Stat { name: 5 }).await["state"],
            "parked",
            "12 shared twice and 3 is 15, not 27"
        );
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
