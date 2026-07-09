//! Atomic transaction helpers for the arena (brief §5 "Atomic Transaction").
//!
//! Forward and fold passes route their physical-memory mutation through a
//! transaction so a failed driver round-trip leaves no half-visible mapping
//! (W13). The transaction is **handle-based, not a scoped closure**: it spans
//! the async driver round-trip — the host's `execute()` runs `prepare` (alloc
//! missing objects, CoW write targets, pin) and gets back an owned [`ArenaTxn`]
//! handle; control returns to the guest inferlet across an `await`; later
//! `FutureOutput.get()` runs `commit` or `abort` on that same handle.
//!
//! ```text
//! prepare:  txn = arena.txn_begin();
//!           arena.txn_alloc / txn_cow / txn_pin / txn_mark_write  (build descriptors)
//!           -> hand (driver descriptors, txn) to the caller; txn lives across await
//! commit:   arena.txn_commit(txn) -> write targets to CAS-seal; mappings published
//! abort:    arena.txn_abort(txn)  -> write targets discarded; nothing sealed;
//!                                    prior mappings stay visible
//! ```
//!
//! The arena owns only the *physical* side (pins, staged allocations, CoW
//! copies, deferred refcount folds). CAS sealing of full KV pages is the KV
//! lane's concern; `commit` returns the write-target object ids so the caller
//! can host-hash and seal eligible full pages (reusing `compute_page_hashes`).

use crate::driver::DriverId;

use super::{Arena, ArenaError, ArenaHandle, ArenaKind, CowPlan, ObjectId};

/// Owned, movable transaction handle. Holds the pinned objects, CoW'd write
/// targets, and deferred refcount folds alive across the async driver
/// round-trip. Must be finished with [`Arena::txn_commit`] or
/// [`Arena::txn_abort`]; dropping it unfinished leaks the staged blocks.
#[derive(Debug)]
pub struct ArenaTxn {
    device: DriverId,
    /// Objects to release if the txn aborts (staged allocs + CoW copies).
    on_abort_free: Vec<ObjectId>,
    /// Originals to decref if the txn commits (the writer left their sharing
    /// group once its private CoW copy became the live mapping).
    on_commit_decref: Vec<ObjectId>,
    /// Objects pinned during prepare; released on both commit and abort.
    pins: Vec<ObjectId>,
    /// Objects the caller will write — returned by commit so eligible full KV
    /// pages can be CAS-sealed (host-hash).
    write_targets: Vec<ObjectId>,
    finished: bool,
}

impl ArenaTxn {
    fn new(device: DriverId) -> Self {
        ArenaTxn {
            device,
            on_abort_free: Vec::new(),
            on_commit_decref: Vec::new(),
            pins: Vec::new(),
            write_targets: Vec::new(),
            finished: false,
        }
    }

    /// Device this transaction targets.
    pub fn device(&self) -> DriverId {
        self.device
    }

    /// Objects designated as write targets so far.
    pub fn write_targets(&self) -> &[ObjectId] {
        &self.write_targets
    }
}

impl Drop for ArenaTxn {
    fn drop(&mut self) {
        if !self.finished {
            tracing::error!(
                device = self.device,
                staged = self.on_abort_free.len(),
                pins = self.pins.len(),
                "arena: ArenaTxn dropped without commit/abort — staged blocks leaked"
            );
        }
    }
}

/// Outcome of a committed transaction.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct TxnCommit {
    /// Objects the caller wrote. The commit succeeded, so the caller may now
    /// CAS-seal eligible full KV pages (host-hash via `compute_page_hashes`).
    pub write_targets: Vec<ObjectId>,
}

impl Arena {
    /// Begin a transaction against this arena's device.
    pub fn txn_begin(&mut self) -> ArenaTxn {
        ArenaTxn::new(self.device())
    }

    /// Allocate a new object inside the transaction. Freed automatically if the
    /// txn aborts; retained (refcount 1, owned by the caller) if it commits.
    pub fn txn_alloc(
        &mut self,
        txn: &mut ArenaTxn,
        kind: ArenaKind,
        size_blocks: u32,
    ) -> Result<ArenaHandle, ArenaError> {
        let handle = self.alloc(kind, size_blocks)?;
        txn.on_abort_free.push(handle.object_id);
        Ok(handle)
    }

    /// CoW a write target inside the transaction. On abort the (possibly fresh)
    /// copy is discarded and the original's mapping stays visible; on commit the
    /// original is decref'd (the writer left its sharing group). The resulting
    /// object is recorded as a write target for post-commit sealing.
    pub fn txn_cow(&mut self, txn: &mut ArenaTxn, id: ObjectId) -> Result<CowPlan, ArenaError> {
        match self.cow_internal(id)? {
            super::CowOutcome::InPlace => {
                txn.write_targets.push(id);
                Ok(CowPlan::InPlace {
                    handle: self.handle(id)?,
                })
            }
            super::CowOutcome::Copied { new_id, from, to } => {
                txn.on_abort_free.push(new_id);
                txn.on_commit_decref.push(id);
                txn.write_targets.push(new_id);
                Ok(CowPlan::Copy {
                    handle: self.handle(new_id)?,
                    from,
                    to,
                })
            }
        }
    }

    /// Pin an object for the duration of the transaction (cannot be offloaded /
    /// evicted while pinned). Released on commit and abort.
    pub fn txn_pin(&mut self, txn: &mut ArenaTxn, id: ObjectId) -> Result<(), ArenaError> {
        self.pin(id)?;
        txn.pins.push(id);
        Ok(())
    }

    /// Explicitly designate an object as a write target (for objects allocated
    /// fresh via [`Arena::txn_alloc`] that the caller will write and may seal).
    /// CoW'd objects are designated automatically.
    pub fn txn_mark_write(&mut self, txn: &mut ArenaTxn, id: ObjectId) -> Result<(), ArenaError> {
        // Validate the object exists so callers fail fast on bad ids.
        let _ = self.object(id)?;
        txn.write_targets.push(id);
        Ok(())
    }

    /// Commit the transaction: release pins, fold in deferred refcount drops on
    /// CoW originals (publishing the new mappings), and return the write targets
    /// so the caller can CAS-seal eligible full pages.
    pub fn txn_commit(&mut self, mut txn: ArenaTxn) -> Result<TxnCommit, ArenaError> {
        for id in txn.pins.drain(..) {
            self.unpin(id)?;
        }
        for id in txn.on_commit_decref.drain(..) {
            self.decref(id)?;
        }
        let write_targets = std::mem::take(&mut txn.write_targets);
        txn.finished = true;
        Ok(TxnCommit { write_targets })
    }

    /// Abort the transaction: release pins, discard staged allocations and CoW
    /// copies (write targets become invalid), and leave prior mappings visible
    /// (CoW originals are untouched; nothing is sealed).
    pub fn txn_abort(&mut self, mut txn: ArenaTxn) {
        for id in txn.pins.drain(..) {
            let _ = self.unpin(id);
        }
        for id in txn.on_abort_free.drain(..) {
            // Staged objects were created at refcount 1 and never shared.
            let _ = self.decref(id);
        }
        txn.finished = true;
    }
}
