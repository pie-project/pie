//! A working set no process owns, and the planner's view of it.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use super::{FORK_WAIT, MOVE_WAIT, RS_MOVE_POLL};
use crate::inferlet::ProcessId;
use crate::inferlet::host::working_set::Cursor;
use crate::inferlet::process::ProcessResidency;
use crate::inferlet::process::residency::{register_residency, unregister_residency};
use crate::planner::Demand;
#[cfg(not(target_arch = "wasm32"))]
use crate::store::kv::page_table::ExportedSet;
use crate::store::kv::working_set::KvWorkingSet;
use crate::store::kv::{KvStore, KvStoreError};
use crate::store::registry as store_registry;
use crate::store::rs::working_set::RsWorkingSet;
use crate::store::rs::{RsStore, RsUnits};

pub(super) struct Parts {
    pub(super) kv: KvWorkingSet,
    pub(super) rs: Option<RsWorkingSet>,
    pub(super) cursor: Cursor,
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
    /// `holder` names it to the planner and the retention table, and what it
    /// leaves in the cache stays under that name.
    pub(crate) fn park(
        holder: ProcessId,
        kv: KvWorkingSet,
        rs: Option<RsWorkingSet>,
        cursor: Cursor,
    ) -> Self {
        kv.leave_owner();
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
                let _ = store.set_owner(kv.id, holder);
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

    pub(super) fn parts(&self) -> &Parts {
        self.parts.as_ref().expect("taken only by consuming self")
    }

    /// Keeps the planner off this set, and answers once it has left it: the
    /// next owner reads what it owes, and no move may be starting then.
    pub(super) async fn claim(&self) -> Result<(), String> {
        let Some(planner) = crate::planner::planner_of(self.holder) else {
            return Ok(());
        };
        planner.claim_holder(self.holder);
        let deadline = crate::rt::time::Instant::now() + MOVE_WAIT;
        while planner.holder_moving(self.holder) {
            if crate::rt::time::Instant::now() >= deadline {
                return Err("its pages are still moving".to_string());
            }
            let _ = crate::rt::time::timeout(RS_MOVE_POLL, std::future::pending::<()>()).await;
        }
        Ok(())
    }

    /// A claim that came to nothing: anyone's to evict again.
    pub(super) fn unclaim(&self) {
        if let Some(planner) = crate::planner::planner_of(self.holder) {
            planner.unclaim_holder(self.holder);
        }
    }

    /// Each store forks under its own lock, with no move of its own under
    /// way: a move of the other's cannot touch what this one holds.
    pub(super) async fn fork(&self, holder: ProcessId) -> Result<Held, String> {
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
        let child = settled(kv, "handle-fork", |store| store.fork(kv.id)).await;
        match child {
            Ok(child) => Ok(Held::park(
                holder,
                kv.forked(child),
                rs_child,
                cursor.clone(),
            )),
            Err(why) => {
                if let Some(rs) = rs_child {
                    rs.release();
                }
                Err(why)
            }
        }
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

    pub(super) fn into_parts(mut self) -> Parts {
        self.leave_planner();
        self.parts.take().expect("taken only by consuming self")
    }

    /// Its pages as a snapshot reads them, held where they stand until the
    /// pin goes back: no move holds one now.
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) async fn pinned(
        &self,
    ) -> Result<(crate::store::kv::ReaderPin, ExportedSet), String> {
        let kv = &self.parts().kv;
        settled(kv, "handle-export", |store| store.export(kv.id)).await
    }

    /// Its tokens, what of them are folded, and its recurrent slots.
    pub(super) fn measure(&self) -> (u64, u64, u32) {
        let Parts { rs, cursor, .. } = self.parts();
        let slots = rs.as_ref().map_or(0, |rs| {
            let stores = store_registry::get(rs.model, rs.engine);
            let held = stores.rs.lock().unwrap().held([rs.id]);
            held.slots
        });
        (
            u64::from(cursor.tail.unwrap_or(cursor.folded)),
            u64::from(cursor.folded),
            slots,
        )
    }

    pub(super) fn stat(&self) -> Value {
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
            "swapped_pages": swapped,
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
    let deadline = crate::rt::time::Instant::now() + FORK_WAIT;
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
    let deadline = crate::rt::time::Instant::now() + FORK_WAIT;
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
