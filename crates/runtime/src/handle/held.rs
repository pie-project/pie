//! A working set no process owns, and the planner's view of it.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

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
        if let Some(rs) = &rs {
            rs.leave_owner();
        }
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
        // Its eviction ending is the event: no clock, no deadline.
        planner.wait_not_moving(self.holder).await;
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
                // A fold the device counts settles first: the child takes
                // the frozen length it leaves, or the fork is refused as the
                // fire's failure.
                let stores = store_registry::get(rs.model, rs.engine);
                crate::store::rs::device_folds_settled(&stores.rs, &[rs.id]).await;
                let id = rs_settled(rs, |store| store.fork(rs.id))
                    .await?
                    .map_err(|error| error.to_string())?;
                Some(RsWorkingSet::new(rs.model, rs.engine, id, rs.geom))
            }
            None => None,
        };
        // Parked, the set has no fire on its way: a frozen length a fold
        // left unknown stays so, and the KV store refuses the fork rather
        // than have the child state the bound.
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
        let folded = self.folded();
        (cursor.tail.map_or(folded, u64::from), folded, slots)
    }

    /// What it has folded: its KV working set's folded length.
    fn folded(&self) -> u64 {
        let kv = &self.parts().kv;
        let stores = store_registry::get(kv.model, kv.engine);
        store_registry::with_kv_lock(&stores.kv, "handle-stat", |store| {
            store.folded(kv.id).unwrap_or(0)
        })
    }

    pub(super) fn stat(&self) -> Value {
        let Parts { kv, cursor, .. } = self.parts();
        let stores = store_registry::get(kv.model, kv.engine);
        let swapped = store_registry::with_kv_lock(&stores.kv, "handle-stat", |store| {
            store
                .swapped_page_count(&HashSet::from([kv.id]))
                .unwrap_or(0)
        });
        let folded = self.folded();
        json!({
            "state": "parked",
            "tokens": cursor.tail.map_or(folded, u64::from),
            "folded": folded,
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
/// under a move is not where its map says. It waits for the move to end with
/// no deadline: every move ends, landed or aborted, and each end wakes it
/// ([`KvStore::moves_ended`]). A store that refuses the set, then or after,
/// refuses it by name.
async fn settled<T>(
    kv: &KvWorkingSet,
    label: &'static str,
    mut f: impl FnMut(&mut KvStore) -> Result<T, KvStoreError>,
) -> Result<T, String> {
    let stores = store_registry::get(kv.model, kv.engine);
    let set = HashSet::from([kv.id]);
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
        notified.await;
    }
}

/// [`settled`] for recurrent state: woken when a move lets slots go
/// ([`RsStore::moves_ended`]), landed or aborted.
async fn rs_settled<T>(
    rs: &RsWorkingSet,
    mut f: impl FnMut(&mut RsStore) -> T,
) -> Result<T, String> {
    let stores = store_registry::get(rs.model, rs.engine);
    loop {
        let ended = stores.rs.lock().unwrap().moves_ended();
        let notified = ended.notified();
        let mut notified = std::pin::pin!(notified);
        notified.as_mut().enable();
        {
            let mut store = stores.rs.lock().unwrap();
            if !store.in_move(rs.id) {
                return Ok(f(&mut store));
            }
        }
        notified.await;
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

#[cfg(test)]
pub(super) mod tests {
    use futures::FutureExt;

    use super::*;
    use crate::store::rs::{RsGeometry, RsWorkingSetId};

    pub(in crate::handle) const DEVICE_LOST: &str = "pipeline: forward failed: device lost";

    /// A parked set whose recurrent state is frozen at 4, a fire that folds
    /// it by a count the device computes still on its way.
    pub(in crate::handle) fn folding() -> (Held, store_registry::Stores, RsWorkingSetId) {
        let model = store_registry::register_model(4, &[8], &[2]);
        let stores = store_registry::get(model, 0);
        let kv = stores.kv.lock().create_working_set();
        let geom = RsGeometry {
            state_size: 8,
            buffer_page_tokens: 4,
        };
        let rs = {
            let mut store = stores.rs.lock().unwrap();
            let ws = store.create_working_set(geom);
            let written = store
                .prepare_group(ws, true, false, None, None, None)
                .unwrap();
            let published = store.publish_prepared(written).unwrap();
            store.settle(published);
            store.fold(ws, 4);
            store.begin_device_fold(ws);
            ws
        };
        let held = Held::park(
            ProcessId::new_v4(),
            KvWorkingSet::new(model, 0, kv, 4),
            Some(RsWorkingSet::new(model, 0, rs, geom)),
            Cursor::default(),
        );
        (held, stores, rs)
    }

    /// A parked set that keeps no recurrent state, its cursor as `cursor`.
    pub(in crate::handle) fn parked(cursor: Cursor) -> Held {
        let model = store_registry::register_model(4, &[8], &[2]);
        let stores = store_registry::get(model, 0);
        let kv = stores.kv.lock().create_working_set();
        Held::park(
            ProcessId::new_v4(),
            KvWorkingSet::new(model, 0, kv, 4),
            None,
            cursor,
        )
    }

    /// A parked set whose frozen length a fold left unknown: the fire that
    /// was to place it settled and never told its count.
    pub(in crate::handle) fn unplaced() -> Held {
        let held = parked(Cursor::default());
        let kv = &held.parts().kv;
        let stores = store_registry::get(kv.model, kv.engine);
        let mut store = stores.kv.lock();
        store.unsettle(kv.id).unwrap();
        store.fold_settled(kv.id, false).unwrap();
        drop(store);
        held
    }

    /// A parked set has no fire on its way, so a frozen length a fold left
    /// unknown stays unknown: its fork is refused by name, never handed the
    /// bound. One at rest forks.
    #[tokio::test(flavor = "current_thread")]
    async fn a_fork_of_a_set_whose_frozen_length_is_not_known_is_refused_by_name() {
        let bound = unplaced();
        let refused = bound
            .fork(ProcessId::new_v4())
            .now_or_never()
            .expect("there is nothing to wait for")
            .expect_err("the child would state a bound");
        assert_eq!(refused, crate::store::kv::page_table::UNPLACED);
        let child = parked(Cursor::default()).fork(ProcessId::new_v4()).await;
        let child = child.expect("its frozen length is known");
        let kv = &child.parts().kv;
        let stores = store_registry::get(kv.model, kv.engine);
        assert!(stores.kv.lock().folded_exact(kv.id).unwrap());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_fork_waits_for_a_fold_the_device_counts() {
        let (held, stores, rs) = folding();
        let mut fork = Box::pin(held.fork(ProcessId::new_v4()));
        assert!(
            fork.as_mut().now_or_never().is_none(),
            "the frozen length is behind the record until the fold settles"
        );
        stores.rs.lock().unwrap().end_device_fold(rs, Some(3), None);
        let child = fork.await.expect("the fold settled");
        let child = child.parts().rs.as_ref().expect("its state forks").id;
        assert_eq!(
            stores.rs.lock().unwrap().frozen(child),
            Ok(7),
            "the child takes what the fold left"
        );
    }

    /// A fork of a set whose page is under a move waits for the move to
    /// end, however long it takes (it was refused at 250 ms), and goes on
    /// as the move lands or is called off.
    #[tokio::test(flavor = "current_thread")]
    async fn a_fork_waits_for_a_move_with_no_deadline_and_goes_on_as_it_ends() {
        use crate::store::kv::write::PageCommit;
        for lands in [true, false] {
            let model = store_registry::register_model_with_swap(
                4,
                &[8],
                &[8],
                &[0],
                &[store_registry::StatePool::default()],
                &[0],
            );
            let stores = store_registry::get(model, 0);
            let (kv, txn) = {
                let mut store = stores.kv.lock();
                let ws = store.create_working_set();
                store.reserve(ws, 1).unwrap();
                let need = store.write_demand(ws, &[0]).unwrap();
                let mut granted = store.reserve_device_pages(need).unwrap();
                let prepared = store
                    .prepare_write_reserved(ws, &[0], &mut granted)
                    .unwrap();
                let commit = PageCommit {
                    token_hashes: Vec::new(),
                    page_hash: None,
                };
                let seq = store.publish_prepared(prepared, &[commit]).unwrap();
                store.settle(seq);
                let floor = |_: Option<ProcessId>| crate::policy::retention::DEFAULT_KEY;
                let txn = store.prepare_suspend(&HashSet::from([ws]), floor).unwrap();
                (ws, txn.expect("its page moves"))
            };
            let held = Held::park(
                ProcessId::new_v4(),
                KvWorkingSet::new(model, 0, kv, 4),
                None,
                Cursor::default(),
            );
            let mut fork = Box::pin(held.fork(ProcessId::new_v4()));
            let waited =
                crate::rt::time::timeout(std::time::Duration::from_millis(300), fork.as_mut())
                    .await;
            assert!(waited.is_err(), "still waiting past 250 ms: {lands}");
            {
                let mut store = stores.kv.lock();
                if lands {
                    store.commit_suspend(txn).unwrap();
                } else {
                    store.abort_suspend(txn);
                }
            }
            let child = fork.await.expect("the move ended");
            assert!(stores.kv.lock().folded_exact(child.parts().kv.id).unwrap());
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_fork_is_refused_by_the_failure_of_the_fold_it_waited_for() {
        let (held, stores, rs) = folding();
        let mut fork = Box::pin(held.fork(ProcessId::new_v4()));
        assert!(fork.as_mut().now_or_never().is_none());
        stores
            .rs
            .lock()
            .unwrap()
            .end_device_fold(rs, None, Some(DEVICE_LOST));
        let refused = fork
            .await
            .expect_err("how far its record is folded is not known");
        assert!(refused.contains(DEVICE_LOST), "{refused}");
    }
}
