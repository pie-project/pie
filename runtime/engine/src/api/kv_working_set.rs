//! `pie:core/working-set` — KV working-set host resource (Lane C / Phase 2).
//!
//! The WASM resource type is [`crate::working_set::kv::KvWorkingSet`] (the real
//! dense-array core; mapped in the `bindgen!` `with:` block). This module is the
//! thin WIT shell: it implements [`HostKvWorkingSet`] (the inferlet-facing
//! `constructor`/`size`/`generation`/`page-size`/`alloc`/`free`/`reorder`/
//! `slice`/`append`/`fork` mutators) on [`InstanceState`].
//!
//! ## Physical state access
//! The structural mutators that touch physical pages (`free`/`slice`/`append`/
//! `fork`/`drop`) borrow the per-(model,driver) unified arena from bravo's
//! `arena::registry::get(..).lock()` and, for release, the per-(model,driver)
//! [`KvCas`](crate::working_set::kv::KvCas) from `kv_cas::get(..).lock()`.
//! **Lock order is always arena → cas**, locks are released before any `await`.
//! The `(model_idx, driver_idx)` comes from [`KvWorkingSet::device`] (v1
//! single-driver `(0, 0)`).
//!
//! ## Forward-pass contract (echo)
//! The resolve / CoW / seal methods echo drives from the forward-pass txn —
//! `resolve_read`, `resolve_write`, `cow_write_slot`, `seal`, plus `page_size`/
//! `size`/`generation`/`device` — are **inherent methods on `KvWorkingSet`** in
//! `crate::working_set::kv`; echo fetches the resource from the table and calls
//! them directly. They are not duplicated here.

use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

use crate::api::pie;
use crate::arena::registry as arena_registry;
use crate::instance::InstanceState;
use crate::working_set::kv::{KvWorkingSet, PageRange};
use crate::working_set::kv_cas;

/// Convert the core [`PageRange`] into the WIT-generated record.
fn wit_range(r: PageRange) -> pie::inferlet::working_set::PageRange {
    pie::inferlet::working_set::PageRange {
        start: r.start,
        len: r.len,
    }
}

// NOTE(working-set): the interface-level `impl pie::inferlet::working_set::Host for
// InstanceState {}` aggregates BOTH HostKvWorkingSet (here) + HostRsWorkingSet
// (delta). echo adds it (and the single `add_to_linker`) once both land, so it
// is intentionally omitted from this shell.

impl pie::inferlet::working_set::HostKvWorkingSet for InstanceState {
    async fn new(&mut self) -> Result<Resource<KvWorkingSet>> {
        // Single-model runtime: bind the one model (index 0). The driver is bound
        // lazily on the first forward write; page_size comes from its arena.
        let page_size = arena_registry::get(0, 0).lock().unwrap().block_size();
        let ws = KvWorkingSet::new(page_size, 0);
        Ok(self.ctx().table.push(ws)?)
    }

    async fn size(&mut self, this: Resource<KvWorkingSet>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.size())
    }

    async fn generation(&mut self, this: Resource<KvWorkingSet>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.generation())
    }

    async fn page_size(&mut self, this: Resource<KvWorkingSet>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.page_size())
    }

    async fn alloc(
        &mut self,
        this: Resource<KvWorkingSet>,
        n: u32,
    ) -> Result<std::result::Result<pie::inferlet::working_set::PageRange, String>> {
        // alloc reserves slots lazily — no arena access needed.
        match self.ctx().table.get_mut(&this)?.alloc(n) {
            Ok(range) => Ok(Ok(wit_range(range))),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn reorder(
        &mut self,
        this: Resource<KvWorkingSet>,
        perm: Vec<u32>,
    ) -> Result<std::result::Result<(), String>> {
        // reorder permutes slots only — no arena access needed.
        match self.ctx().table.get_mut(&this)?.reorder(&perm) {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn free(
        &mut self,
        this: Resource<KvWorkingSet>,
        indices: Vec<u32>,
    ) -> Result<std::result::Result<(), String>> {
        let (m, d) = self.ctx().table.get(&this)?.device();
        // Lock order: arena → cas.
        let arena_arc = arena_registry::get(m, d);
        let cas_arc = kv_cas::get(m, d);
        let mut arena = arena_arc.lock().unwrap();
        let mut cas = cas_arc.lock().unwrap();
        let ws = self.ctx().table.get_mut(&this)?;
        match ws.free(&indices, &mut arena, &mut cas) {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn alloc_slots(
        &mut self,
        this: Resource<KvWorkingSet>,
        n: u32,
    ) -> Result<std::result::Result<Vec<u32>, String>> {
        // alloc_slots reserves/recycles slots lazily — no arena access needed.
        match self.ctx().table.get_mut(&this)?.alloc_slots(n) {
            Ok(ids) => Ok(Ok(ids)),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn slice(
        &mut self,
        this: Resource<KvWorkingSet>,
        start: u32,
        len: u32,
    ) -> Result<std::result::Result<Resource<KvWorkingSet>, String>> {
        let (m, d) = self.ctx().table.get(&this)?.device();
        let arena_arc = arena_registry::get(m, d);
        let mut arena = arena_arc.lock().unwrap();
        let new_ws = match self.ctx().table.get(&this)?.slice(start, len, &mut arena) {
            Ok(ws) => ws,
            Err(e) => return Ok(Err(e.to_string())),
        };
        drop(arena);
        Ok(Ok(self.ctx().table.push(new_ws)?))
    }

    async fn append(
        &mut self,
        this: Resource<KvWorkingSet>,
        other: Resource<KvWorkingSet>,
    ) -> Result<std::result::Result<(), String>> {
        let (m, d) = self.ctx().table.get(&this)?.device();
        // Read `other`'s slots before taking the `&mut this` borrow (same table).
        let other_slots = self.ctx().table.get(&other)?.slot_objects();
        let arena_arc = arena_registry::get(m, d);
        let mut arena = arena_arc.lock().unwrap();
        let ws = self.ctx().table.get_mut(&this)?;
        match ws.append_shared(&other_slots, &mut arena) {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn fork(
        &mut self,
        this: Resource<KvWorkingSet>,
    ) -> Result<std::result::Result<Resource<KvWorkingSet>, String>> {
        let (m, d) = self.ctx().table.get(&this)?.device();
        let arena_arc = arena_registry::get(m, d);
        let mut arena = arena_arc.lock().unwrap();
        let new_ws = match self.ctx().table.get(&this)?.fork(&mut arena) {
            Ok(ws) => ws,
            Err(e) => return Ok(Err(e.to_string())),
        };
        drop(arena);
        Ok(Ok(self.ctx().table.push(new_ws)?))
    }

    async fn drop(&mut self, this: Resource<KvWorkingSet>) -> Result<()> {
        // Read the device first so the immutable borrow ends before `get_mut`.
        let dev = self.ctx().table.get(&this).ok().map(|ws| ws.device());
        if let Some((m, d)) = dev {
            let arena_arc = arena_registry::get(m, d);
            let cas_arc = kv_cas::get(m, d);
            {
                let mut arena = arena_arc.lock().unwrap();
                let mut cas = cas_arc.lock().unwrap();
                if let Ok(ws) = self.ctx().table.get_mut(&this) {
                    ws.destroy(&mut arena, &mut cas);
                }
            } // arena/cas locks released before the contention drain (Task-B: the
            //   orchestrator's drain re-locks the arena via pool_stats, so it must
            //   never run under these guards).
            // A completing/terminating process just freed its KV pool blocks —
            // wake FCFS contention waiters + restore suspended processes. No-op
            // (None) unless PIE_KV_CONTENTION=preempt wired the orchestrator.
            if let Some(o) = crate::inference::contention::contention() {
                o.on_blocks_freed();
            }
        }
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
