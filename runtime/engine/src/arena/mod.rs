//! # Unified Typed Arena (runtime memory core, Phase 1)
//!
//! One arena per driver/device. The single physical-feasibility authority for
//! KV pages, RS slabs, scratch, CPU stash, and snapshot blobs (brief W12/W13/W14).
//!
//! ## Model
//! - The accounting unit is the **block**, sized to one KV page. A KV page is
//!   one block; an RS slab is an integer number of blocks; scratch / stash /
//!   blob objects are sized in blocks.
//! - An **object** ([`ArenaHandle`] / `ObjectId`) owns `size_blocks` physical
//!   block ids in whichever pool matches its current [`Residency`]. Objects are
//!   refcounted; sharing is `incref`, divergent writes go through copy-on-write
//!   ([`Arena::cow`]).
//! - "Unified" is at the handle / refcount / residency / transaction layer.
//!   The physical id-spaces stay separate per region because the driver
//!   consumes distinct namespaces (`kv_page_indices` vs `rs_slot_ids`); see
//!   [`pool::BlockPool`]. This matches W12's "may make RS slabs consume an
//!   integer number of KV-page arena blocks" without forcing one free-list.
//!
//! ## What this module owns (Seam 2)
//! The frozen handle + transaction surface that the KV lane (charlie) and the
//! RS lane (delta) allocate against. They must not churn these signatures.
//!
//! ## What it deliberately does NOT do (v1)
//! - No general compaction / size classes (brief W12 — add later if measured).
//! - No CAS content-addressing: `compute_page_hashes` stays a pure fn in
//!   `context::pagestore`; the KV lane reuses it for sealing.
//! - No driver I/O: offload / restore / CoW return *plans* (source/destination
//!   block ids); the caller issues the actual d2h / h2d / d2d driver copy.

pub mod pool;
pub mod registry;
mod txn;
use rustc_hash::FxHashMap;

use crate::driver::DriverId;
use pool::BlockPool;

pub use txn::ArenaTxn;

/// Bug#2 diagnostic (concurrent-decode KV-page churn): when `PTIR_ARENA_TRACE`
/// is set, log every KvPage alloc/free — the free with a short backtrace so the
/// churn's trigger (legitimate `KvWorkingSet::destroy` vs a per-fire release) is
/// visible on a real-HW `cuda_bubble` run. Zero-cost when unset (one atomic
/// load per alloc/free). Enable with `PTIR_ARENA_TRACE=1` on the repro.
static ARENA_TRACE: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var("PTIR_ARENA_TRACE").is_ok());

/// Synchronous lookup of the unified arena for `(model_idx, driver_idx)`,
/// re-exported at the module root so callers write `arena::get(..)` /
/// `arena::try_get(..)`. Lock the returned handle sync, release before any
/// await (see [`registry`]).
pub use registry::{get, try_get};

/// Stable identifier for an arena object. Recycled after free; callers must not
/// use a handle after dropping their last reference to it.
pub type ObjectId = u32;

/// Physical block id within a single pool's id-space (a KV page index, an RS
/// slot id, a CPU-stash slot, ...). Only meaningful relative to its pool.
pub type BlockId = u32;

/// A physical KV page id the driver consumes. v1: one KV page == one arena
/// `KvPage` block, so a `PhysicalPageId` is a `BlockId`. (Relocated here from
/// the retired `context::pagestore` in Phase 5.)
pub type PhysicalPageId = BlockId;

/// The typed role of an arena object. Determines which physical pool backs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArenaKind {
    /// A single KV cache page. Always exactly one block.
    KvPage,
    /// A recurrent-state slab (folded RS state). Integer number of blocks.
    RsSlab,
    /// Transient device scratch (e.g. staging during a forward).
    Scratch,
    /// CPU-side stash backing an offloaded object.
    CpuStash,
    /// A CPU-resident, non-evictable snapshot blob (inferlet-owned bytes).
    SnapshotBlob,
}

impl ArenaKind {
    /// Whether objects of this kind are device-resident when first allocated.
    fn allocates_on_gpu(self) -> bool {
        matches!(self, ArenaKind::KvPage | ArenaKind::RsSlab | ArenaKind::Scratch)
    }

    /// Whether objects of this kind may be evicted (offloaded / dropped to
    /// replay). Snapshot blobs are pinned in CPU memory (W14).
    fn default_evictable(self) -> bool {
        !matches!(self, ArenaKind::SnapshotBlob)
    }
}

/// Where an object's bytes physically live right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residency {
    /// Resident on the device, ready for the driver.
    Gpu,
    /// Offloaded to the CPU stash; must be restored before device use.
    Cpu,
    /// Not materialized anywhere, but reconstructible by replay (e.g. a GPU
    /// copy of snapshot-backed data evicted under contention).
    Replayable,
    /// Evicted and not reconstructible without external action (error state;
    /// RS parity with `rs_cache::RsState::Missing`).
    Missing,
}

impl Residency {
    /// Whether an object in this residency currently owns physical blocks.
    fn has_physical(self) -> bool {
        matches!(self, Residency::Gpu | Residency::Cpu)
    }
}

/// A lightweight value handle to an arena object. The authoritative state lives
/// in the [`Arena`]; the `residency` here is a snapshot — re-query via
/// [`Arena::handle`] / [`Arena::residency`] after any residency transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaHandle {
    pub kind: ArenaKind,
    pub device: DriverId,
    pub object_id: ObjectId,
    pub size_blocks: u32,
    pub residency: Residency,
}

/// Result of a copy-on-write request ([`Arena::cow`] / [`ArenaTxn`] CoW).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CowPlan {
    /// The object was uniquely owned (refcount 1); write in place, no copy.
    InPlace { handle: ArenaHandle },
    /// The object was shared; a private copy was allocated. The caller must
    /// copy `from` -> `to` on the device (driver d2d) before writing.
    Copy {
        handle: ArenaHandle,
        from: Vec<BlockId>,
        to: Vec<BlockId>,
    },
}

/// Source/destination block ids for a residency move. The caller issues the
/// driver copy (`from` on the old tier -> `to` on the new tier).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MovePlan {
    pub from: Vec<BlockId>,
    pub to: Vec<BlockId>,
}

/// Errors from arena operations. All are recoverable — the arena never traps.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ArenaError {
    #[error("arena {device}: out of {kind:?} blocks (requested {requested}, available {available})")]
    OutOfBlocks {
        device: DriverId,
        kind: ArenaKind,
        requested: u32,
        available: u32,
    },
    #[error("arena: unknown object {0}")]
    UnknownObject(ObjectId),
    #[error("arena: object {id} kind {kind:?} cannot be {op}")]
    InvalidKind {
        id: ObjectId,
        kind: ArenaKind,
        op: &'static str,
    },
    #[error("arena: object {id} has residency {residency:?}, expected {expected:?} for {op}")]
    InvalidResidency {
        id: ObjectId,
        residency: Residency,
        expected: Residency,
        op: &'static str,
    },
    #[error("arena: object {0} is pinned")]
    Pinned(ObjectId),
    #[error("arena: object {0} is non-evictable")]
    NotEvictable(ObjectId),
    #[error("arena: object {id} bad size_blocks {size} for kind {kind:?}")]
    BadSize {
        id: ObjectId,
        kind: ArenaKind,
        size: u32,
    },
}

/// Per-object bookkeeping. Private — callers hold [`ArenaHandle`] / `ObjectId`.
#[derive(Debug)]
struct Object {
    kind: ArenaKind,
    size_blocks: u32,
    /// Physical block ids in the pool matching `residency`. Empty when the
    /// object is `Replayable` / `Missing`. Length always equals `size_blocks`
    /// while the object has physical residency.
    blocks: Vec<BlockId>,
    residency: Residency,
    refcount: u32,
    pinned: u32,
    evictable: bool,
}

/// Construction parameters for an [`Arena`]. All capacities are in blocks.
#[derive(Debug, Clone, Copy)]
pub struct ArenaConfig {
    pub device: DriverId,
    /// Logical KV-page size (tokens/elements). Accounting block == this size.
    pub block_size: u32,
    /// Device KV-page pool capacity (blocks).
    pub kv_pages: u32,
    /// Device RS pool capacity (blocks).
    pub rs_blocks: u32,
    /// Device scratch pool capacity (blocks).
    pub scratch_blocks: u32,
    /// CPU stash capacity (blocks) shared by offloaded objects + snapshot blobs.
    pub cpu_blocks: u32,
}

/// One unified typed arena per driver/device.
#[derive(Debug)]
pub struct Arena {
    device: DriverId,
    block_size: u32,
    kv_gpu: BlockPool,
    rs_gpu: BlockPool,
    scratch_gpu: BlockPool,
    cpu: BlockPool,
    objects: FxHashMap<ObjectId, Object>,
    next_id: ObjectId,
    recycled_ids: Vec<ObjectId>,
}

impl Arena {
    /// Build an arena from explicit pool capacities.
    pub fn new(config: ArenaConfig) -> Self {
        Arena {
            device: config.device,
            block_size: config.block_size,
            kv_gpu: BlockPool::new(config.kv_pages),
            rs_gpu: BlockPool::new(config.rs_blocks),
            scratch_gpu: BlockPool::new(config.scratch_blocks),
            cpu: BlockPool::new(config.cpu_blocks),
            objects: FxHashMap::default(),
            next_id: 0,
            recycled_ids: Vec::new(),
        }
    }

    /// Device this arena serves.
    pub fn device(&self) -> DriverId {
        self.device
    }

    /// Logical KV-page / accounting block size.
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    // =========================================================================
    // Allocation / sharing / release
    // =========================================================================

    /// Allocate a new object of `kind` spanning `size_blocks` blocks (refcount
    /// 1). KV pages must be exactly one block. KV/RS/scratch land on the device;
    /// CPU-stash / snapshot-blob objects land in the CPU stash (snapshot blobs
    /// non-evictable).
    pub fn alloc(&mut self, kind: ArenaKind, size_blocks: u32) -> Result<ArenaHandle, ArenaError> {
        self.validate_size(kind, size_blocks)?;
        let (residency, blocks) = if kind.allocates_on_gpu() {
            let device = self.device;
            let pool = self.gpu_pool_mut(kind)?;
            let blocks = pool.alloc(size_blocks).ok_or(ArenaError::OutOfBlocks {
                device,
                kind,
                requested: size_blocks,
                available: pool.available(),
            })?;
            (Residency::Gpu, blocks)
        } else {
            let device = self.device;
            let blocks = self.cpu.alloc(size_blocks).ok_or(ArenaError::OutOfBlocks {
                device,
                kind,
                requested: size_blocks,
                available: self.cpu.available(),
            })?;
            (Residency::Cpu, blocks)
        };

        let id = self.fresh_id();
        let obj = Object {
            kind,
            size_blocks,
            blocks,
            residency,
            refcount: 1,
            pinned: 0,
            evictable: kind.default_evictable(),
        };
        if *ARENA_TRACE && matches!(kind, ArenaKind::KvPage) {
            eprintln!("[ARENA] alloc  id={:?} blocks={:?}", id, obj.blocks);
        }
        self.objects.insert(id, obj);
        self.handle(id)
    }

    /// Share an existing object: bump its refcount. Used by KV slice/append and
    /// RS fork to share page/slab objects by reference (CoW on first write).
    pub fn incref(&mut self, id: ObjectId) -> Result<(), ArenaError> {
        let obj = self.object_mut(id)?;
        obj.refcount += 1;
        Ok(())
    }

    /// Drop one reference. When the last reference goes away the physical blocks
    /// are reclaimed and the object id is recycled.
    pub fn decref(&mut self, id: ObjectId) -> Result<(), ArenaError> {
        let obj = self.object_mut(id)?;
        debug_assert!(obj.refcount > 0, "decref of object with refcount 0");
        obj.refcount -= 1;
        let reclaim = obj.refcount == 0 && obj.pinned == 0;
        // A pinned object is held by an in-flight forward that released the arena
        // lock across its async submit. Freeing it now (removing it from
        // `objects` + recycling its blocks) would make that pass's finalize
        // unpin/decref hit `UnknownObject` — the co-batched run-ahead corruption.
        // Defer the physical free to `unpin`, when the last in-flight reference
        // retires. Consistent with `offload`/`reserve_gpu` already refusing pins.
        if reclaim {
            self.free_object(id);
        }
        Ok(())
    }

    // =========================================================================
    // Copy-on-write
    // =========================================================================

    /// Copy-on-write a write target. If the object is uniquely owned the caller
    /// writes in place. If it is shared, a private device copy is allocated and
    /// this reference is moved to the copy (the original's other sharers keep
    /// their mapping, per W6); the caller must copy `from` -> `to` before
    /// writing. The write target must be device-resident.
    pub fn cow(&mut self, id: ObjectId) -> Result<CowPlan, ArenaError> {
        match self.cow_internal(id)? {
            CowOutcome::InPlace => Ok(CowPlan::InPlace {
                handle: self.handle(id)?,
            }),
            CowOutcome::Copied { new_id, from, to } => {
                // The writer leaves the sharing group of the original.
                self.decref(id)?;
                Ok(CowPlan::Copy {
                    handle: self.handle(new_id)?,
                    from,
                    to,
                })
            }
        }
    }

    /// Unconditionally allocate a private duplicate of `src` (same kind/size,
    /// device-resident) and return the new handle plus the block ids for the
    /// caller's device copy (driver `copy_d2d` / `copy_slot_d2d`). Refcounts are
    /// untouched — the caller's layer decides sharing.
    ///
    /// This is the physical primitive behind charlie's `PageBackend::copy`
    /// (the KV CAS layer manages refcounts itself) and delta's RS-fork device
    /// copy. For the forward write-target path that should only copy when
    /// shared, use [`Arena::cow`] / [`Arena::txn_cow`] instead.
    pub fn copy(&mut self, src: ObjectId) -> Result<(ArenaHandle, MovePlan), ArenaError> {
        let (kind, size_blocks, residency, from) = {
            let obj = self.object(src)?;
            (obj.kind, obj.size_blocks, obj.residency, obj.blocks.clone())
        };
        if residency != Residency::Gpu {
            return Err(ArenaError::InvalidResidency {
                id: src,
                residency,
                expected: Residency::Gpu,
                op: "copy",
            });
        }
        let dup = self.alloc(kind, size_blocks)?;
        let to = self.object(dup.object_id)?.blocks.clone();
        Ok((dup, MovePlan { from, to }))
    }

    /// Shared CoW core. Returns the decision without applying any decref of the
    /// original (callers/txn decide when to fold that in).
    fn cow_internal(&mut self, id: ObjectId) -> Result<CowOutcome, ArenaError> {
        let (kind, size_blocks, residency, refcount, from) = {
            let obj = self.object(id)?;
            (
                obj.kind,
                obj.size_blocks,
                obj.residency,
                obj.refcount,
                obj.blocks.clone(),
            )
        };
        if residency != Residency::Gpu {
            return Err(ArenaError::InvalidResidency {
                id,
                residency,
                expected: Residency::Gpu,
                op: "cow",
            });
        }
        if refcount <= 1 {
            return Ok(CowOutcome::InPlace);
        }
        let copy = self.alloc(kind, size_blocks)?;
        let to = self.object(copy.object_id)?.blocks.clone();
        Ok(CowOutcome::Copied {
            new_id: copy.object_id,
            from,
            to,
        })
    }

    // =========================================================================
    // Residency transitions (CPU offload / replay)
    // =========================================================================

    /// Offload a device-resident object to the CPU stash. Returns the block ids
    /// for the caller's d2h copy. Refuses pinned or non-evictable objects.
    pub fn offload(&mut self, id: ObjectId) -> Result<MovePlan, ArenaError> {
        let (kind, size_blocks, residency, pinned, evictable, from) = {
            let obj = self.object(id)?;
            (
                obj.kind,
                obj.size_blocks,
                obj.residency,
                obj.pinned,
                obj.evictable,
                obj.blocks.clone(),
            )
        };
        if residency != Residency::Gpu {
            return Err(ArenaError::InvalidResidency {
                id,
                residency,
                expected: Residency::Gpu,
                op: "offload",
            });
        }
        if pinned > 0 {
            return Err(ArenaError::Pinned(id));
        }
        if !evictable {
            return Err(ArenaError::NotEvictable(id));
        }
        let device = self.device;
        let to = self.cpu.alloc(size_blocks).ok_or(ArenaError::OutOfBlocks {
            device,
            kind: ArenaKind::CpuStash,
            requested: size_blocks,
            available: self.cpu.available(),
        })?;
        self.gpu_pool_mut(kind)?.free(&from);
        let obj = self.object_mut(id)?;
        obj.blocks = to.clone();
        obj.residency = Residency::Cpu;
        Ok(MovePlan { from, to })
    }

    /// STAGE an offload: validate + allocate the CPU destination, returning the
    /// D2H [`MovePlan`] (`from` = the still-resident GPU blocks, `to` = the fresh
    /// CPU blocks). The object is left UNCHANGED — it still owns its GPU blocks
    /// and reads [`Residency::Gpu`] — so the caller can safely issue the D2H copy
    /// BEFORE the GPU blocks are freed. This fixes the stash-free-before-copy race:
    /// [`offload`](Self::offload) frees the GPU block immediately, so a concurrent
    /// alloc could reuse it and overwrite the page mid-copy. Finish with
    /// [`offload_commit`](Self::offload_commit) after the copy, or
    /// [`offload_abort`](Self::offload_abort) to release the staged CPU blocks if
    /// the suspend is dropped.
    pub fn offload_stage(&mut self, id: ObjectId) -> Result<MovePlan, ArenaError> {
        let (_kind, size_blocks, residency, pinned, evictable, from) = {
            let obj = self.object(id)?;
            (
                obj.kind,
                obj.size_blocks,
                obj.residency,
                obj.pinned,
                obj.evictable,
                obj.blocks.clone(),
            )
        };
        if residency != Residency::Gpu {
            return Err(ArenaError::InvalidResidency {
                id,
                residency,
                expected: Residency::Gpu,
                op: "offload",
            });
        }
        if pinned > 0 {
            return Err(ArenaError::Pinned(id));
        }
        if !evictable {
            return Err(ArenaError::NotEvictable(id));
        }
        let device = self.device;
        let to = self.cpu.alloc(size_blocks).ok_or(ArenaError::OutOfBlocks {
            device,
            kind: ArenaKind::CpuStash,
            requested: size_blocks,
            available: self.cpu.available(),
        })?;
        // Object left UNCHANGED (still Gpu, still owns `from`) — the caller copies
        // `from`→`to`, then calls `offload_commit` to free `from` and repoint.
        Ok(MovePlan { from, to })
    }

    /// COMMIT a staged offload (call AFTER the D2H copy completes): free the GPU
    /// blocks and repoint the object to the staged CPU blocks `to`.
    pub fn offload_commit(&mut self, id: ObjectId, to: &[BlockId]) -> Result<(), ArenaError> {
        let (kind, from) = {
            let obj = self.object(id)?;
            (obj.kind, obj.blocks.clone())
        };
        self.gpu_pool_mut(kind)?.free(&from);
        let obj = self.object_mut(id)?;
        obj.blocks = to.to_vec();
        obj.residency = Residency::Cpu;
        Ok(())
    }

    /// ABORT a staged offload: release the staged CPU blocks. The object was never
    /// changed (still owns its GPU blocks), so this simply reclaims the CPU stash.
    pub fn offload_abort(&mut self, to: &[BlockId]) {
        self.cpu.free(to);
    }

    /// Restore a CPU-offloaded object back onto the device. Returns the block
    /// ids for the caller's h2d copy.
    pub fn restore(&mut self, id: ObjectId) -> Result<MovePlan, ArenaError> {
        let (kind, size_blocks, residency, from) = {
            let obj = self.object(id)?;
            (obj.kind, obj.size_blocks, obj.residency, obj.blocks.clone())
        };
        if residency != Residency::Cpu {
            return Err(ArenaError::InvalidResidency {
                id,
                residency,
                expected: Residency::Cpu,
                op: "restore",
            });
        }
        let device = self.device;
        let to = {
            let pool = self.gpu_pool_mut(kind)?;
            pool.alloc(size_blocks).ok_or(ArenaError::OutOfBlocks {
                device,
                kind,
                requested: size_blocks,
                available: pool.available(),
            })?
        };
        self.cpu.free(&from);
        let obj = self.object_mut(id)?;
        obj.blocks = to.clone();
        obj.residency = Residency::Gpu;
        Ok(MovePlan { from, to })
    }

    /// STAGE a restore: validate + allocate the GPU destination, returning the
    /// H2D [`MovePlan`] (`from` = the still-owned CPU blocks, `to` = the fresh
    /// GPU blocks). The object is left UNCHANGED — it still owns its CPU blocks
    /// and reads [`Residency::Cpu`] — so a batch caller can stage EVERY page and
    /// only [`restore_commit`](Self::restore_commit) once all GPU allocs land.
    /// This makes a batch restore all-or-nothing for ANY per-page failure (not
    /// just the pool pre-check): on a mid-batch error the caller
    /// [`restore_abort`](Self::restore_abort)s the pages staged so far — no
    /// object's residency was flipped and no CPU block was freed, so nothing is
    /// left mapped-but-unpopulated (which would read silent garbage and defeat
    /// the caller's fail-loud guarantee). Finish with `restore_commit` after the
    /// H2D copy, or `restore_abort` to release the staged GPU blocks.
    pub fn restore_stage(&mut self, id: ObjectId) -> Result<MovePlan, ArenaError> {
        let (kind, size_blocks, residency, from) = {
            let obj = self.object(id)?;
            (obj.kind, obj.size_blocks, obj.residency, obj.blocks.clone())
        };
        if residency != Residency::Cpu {
            return Err(ArenaError::InvalidResidency {
                id,
                residency,
                expected: Residency::Cpu,
                op: "restore",
            });
        }
        let device = self.device;
        let to = {
            let pool = self.gpu_pool_mut(kind)?;
            pool.alloc(size_blocks).ok_or(ArenaError::OutOfBlocks {
                device,
                kind,
                requested: size_blocks,
                available: pool.available(),
            })?
        };
        // Object left UNCHANGED (still Cpu, still owns `from`) — the caller copies
        // `from`→`to`, then calls `restore_commit` to free `from` and repoint.
        Ok(MovePlan { from, to })
    }

    /// COMMIT a staged restore (call AFTER the H2D copy completes): free the CPU
    /// blocks and repoint the object to the staged GPU blocks `to`.
    pub fn restore_commit(&mut self, id: ObjectId, to: &[BlockId]) -> Result<(), ArenaError> {
        let from = {
            let obj = self.object(id)?;
            obj.blocks.clone()
        };
        self.cpu.free(&from);
        let obj = self.object_mut(id)?;
        obj.blocks = to.to_vec();
        obj.residency = Residency::Gpu;
        Ok(())
    }

    /// ABORT a staged restore: release the staged GPU blocks. The object was never
    /// changed (still owns its CPU blocks and reads `Residency::Cpu`), so this
    /// simply reclaims the GPU pool. `id` is only used to route the free to the
    /// object's GPU pool by kind.
    pub fn restore_abort(&mut self, id: ObjectId, to: &[BlockId]) -> Result<(), ArenaError> {
        let kind = self.object(id)?.kind;
        self.gpu_pool_mut(kind)?.free(to);
        Ok(())
    }

    /// Reserve fresh device blocks for a `Replayable` / `Missing` object so the
    /// caller can replay/recompute into them. Returns the new block ids.
    pub fn reserve_gpu(&mut self, id: ObjectId) -> Result<Vec<BlockId>, ArenaError> {
        let (kind, size_blocks, residency) = {
            let obj = self.object(id)?;
            (obj.kind, obj.size_blocks, obj.residency)
        };
        if residency.has_physical() {
            return Err(ArenaError::InvalidResidency {
                id,
                residency,
                expected: Residency::Replayable,
                op: "reserve_gpu",
            });
        }
        let device = self.device;
        let blocks = {
            let pool = self.gpu_pool_mut(kind)?;
            pool.alloc(size_blocks).ok_or(ArenaError::OutOfBlocks {
                device,
                kind,
                requested: size_blocks,
                available: pool.available(),
            })?
        };
        let obj = self.object_mut(id)?;
        obj.blocks = blocks.clone();
        obj.residency = Residency::Gpu;
        Ok(blocks)
    }

    /// Drop an evictable object's physical blocks, marking it `Replayable` (the
    /// caller holds a replay source — e.g. a snapshot blob). Refuses pinned or
    /// non-evictable objects.
    pub fn mark_replayable(&mut self, id: ObjectId) -> Result<(), ArenaError> {
        self.release_physical(id, Residency::Replayable)
    }

    /// Drop an object's physical blocks, marking it `Missing` (RS parity: needs
    /// lineage replay before reuse). Refuses pinned objects.
    pub fn mark_missing(&mut self, id: ObjectId) -> Result<(), ArenaError> {
        self.release_physical(id, Residency::Missing)
    }

    fn release_physical(&mut self, id: ObjectId, to: Residency) -> Result<(), ArenaError> {
        let (kind, residency, pinned, evictable, blocks) = {
            let obj = self.object(id)?;
            (
                obj.kind,
                obj.residency,
                obj.pinned,
                obj.evictable,
                obj.blocks.clone(),
            )
        };
        if pinned > 0 {
            return Err(ArenaError::Pinned(id));
        }
        if to == Residency::Replayable && !evictable {
            return Err(ArenaError::NotEvictable(id));
        }
        if let Some(pool) = self.pool_for_mut(kind, residency) {
            pool.free(&blocks);
        }
        let obj = self.object_mut(id)?;
        obj.blocks = Vec::new();
        obj.residency = to;
        Ok(())
    }

    // =========================================================================
    // Pinning (transaction safety)
    // =========================================================================

    /// Pin an object so it cannot be offloaded / evicted while a transaction
    /// holds it. Pins nest; each `pin` needs a matching `unpin`.
    pub fn pin(&mut self, id: ObjectId) -> Result<(), ArenaError> {
        let obj = self.object_mut(id)?;
        obj.pinned += 1;
        Ok(())
    }

    /// Release one pin.
    pub fn unpin(&mut self, id: ObjectId) -> Result<(), ArenaError> {
        let obj = self.object_mut(id)?;
        if obj.pinned > 0 {
            obj.pinned -= 1;
        }
        // If this drops the last pin on an object whose refcount already reached
        // 0 (its free was deferred by the pinned-check in `decref`), reclaim it
        // now — the in-flight forward that held it has retired.
        let reclaim = obj.pinned == 0 && obj.refcount == 0;
        if reclaim {
            self.free_object(id);
        }
        Ok(())
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Fresh value handle snapshotting the object's current state.
    pub fn handle(&self, id: ObjectId) -> Result<ArenaHandle, ArenaError> {
        let obj = self.object(id)?;
        Ok(ArenaHandle {
            kind: obj.kind,
            device: self.device,
            object_id: id,
            size_blocks: obj.size_blocks,
            residency: obj.residency,
        })
    }

    /// Physical block ids backing the object (empty if `Replayable`/`Missing`).
    pub fn blocks(&self, id: ObjectId) -> Result<&[BlockId], ArenaError> {
        Ok(&self.object(id)?.blocks)
    }

    pub fn residency(&self, id: ObjectId) -> Result<Residency, ArenaError> {
        Ok(self.object(id)?.residency)
    }

    pub fn refcount(&self, id: ObjectId) -> Result<u32, ArenaError> {
        Ok(self.object(id)?.refcount)
    }

    pub fn kind(&self, id: ObjectId) -> Result<ArenaKind, ArenaError> {
        Ok(self.object(id)?.kind)
    }

    pub fn size_blocks(&self, id: ObjectId) -> Result<u32, ArenaError> {
        Ok(self.object(id)?.size_blocks)
    }

    pub fn is_pinned(&self, id: ObjectId) -> Result<bool, ArenaError> {
        Ok(self.object(id)?.pinned > 0)
    }

    pub fn is_evictable(&self, id: ObjectId) -> Result<bool, ArenaError> {
        Ok(self.object(id)?.evictable)
    }

    /// Override the evictable flag (e.g. mark a GPU cache copy of snapshot-backed
    /// data evictable). Snapshot blobs default to non-evictable.
    pub fn set_evictable(&mut self, id: ObjectId, evictable: bool) -> Result<(), ArenaError> {
        self.object_mut(id)?.evictable = evictable;
        Ok(())
    }

    /// Number of live objects (any residency). Test/telemetry aid.
    pub fn live_objects(&self) -> usize {
        self.objects.len()
    }

    /// Free blocks available for `kind` in its device/CPU pool.
    pub fn available(&self, kind: ArenaKind) -> u32 {
        self.pool_for(kind).available()
    }

    /// Total block capacity for `kind`'s pool.
    pub fn capacity(&self, kind: ArenaKind) -> u32 {
        self.pool_for(kind).capacity()
    }

    /// Blocks currently in use for `kind`'s pool.
    pub fn used(&self, kind: ArenaKind) -> u32 {
        self.pool_for(kind).used()
    }

    // =========================================================================
    // Internals
    // =========================================================================

    fn validate_size(&self, kind: ArenaKind, size_blocks: u32) -> Result<(), ArenaError> {
        let ok = match kind {
            ArenaKind::KvPage => size_blocks == 1,
            _ => size_blocks >= 1,
        };
        if ok {
            Ok(())
        } else {
            Err(ArenaError::BadSize {
                id: ObjectId::MAX,
                kind,
                size: size_blocks,
            })
        }
    }

    fn fresh_id(&mut self) -> ObjectId {
        if let Some(id) = self.recycled_ids.pop() {
            id
        } else {
            let id = self.next_id;
            self.next_id += 1;
            id
        }
    }

    fn free_object(&mut self, id: ObjectId) {
        if let Some(obj) = self.objects.remove(&id) {
            if *ARENA_TRACE && matches!(obj.kind, ArenaKind::KvPage) {
                let bt = std::backtrace::Backtrace::force_capture();
                let s = format!("{bt}");
                let frame = s
                    .lines()
                    .filter(|l| l.contains("pie::") && !l.contains("free_object") && !l.contains("::decref"))
                    .take(3)
                    .map(|l| l.trim())
                    .collect::<Vec<_>>()
                    .join(" <- ");
                eprintln!("[ARENA] free   id={:?} blocks={:?} via {}", id, obj.blocks, frame);
            }
            if obj.residency.has_physical() {
                if let Some(pool) = self.pool_for_mut(obj.kind, obj.residency) {
                    pool.free(&obj.blocks);
                }
            }
            self.recycled_ids.push(id);
        }
    }

    fn object(&self, id: ObjectId) -> Result<&Object, ArenaError> {
        self.objects.get(&id).ok_or(ArenaError::UnknownObject(id))
    }

    fn object_mut(&mut self, id: ObjectId) -> Result<&mut Object, ArenaError> {
        self.objects
            .get_mut(&id)
            .ok_or(ArenaError::UnknownObject(id))
    }

    /// The device pool for a GPU-resident kind.
    fn gpu_pool_mut(&mut self, kind: ArenaKind) -> Result<&mut BlockPool, ArenaError> {
        match kind {
            ArenaKind::KvPage => Ok(&mut self.kv_gpu),
            ArenaKind::RsSlab => Ok(&mut self.rs_gpu),
            ArenaKind::Scratch => Ok(&mut self.scratch_gpu),
            _ => Err(ArenaError::InvalidKind {
                id: ObjectId::MAX,
                kind,
                op: "device alloc",
            }),
        }
    }

    /// The pool backing an object given its kind + current residency.
    fn pool_for_mut(&mut self, kind: ArenaKind, residency: Residency) -> Option<&mut BlockPool> {
        match residency {
            Residency::Gpu => match kind {
                ArenaKind::KvPage => Some(&mut self.kv_gpu),
                ArenaKind::RsSlab => Some(&mut self.rs_gpu),
                ArenaKind::Scratch => Some(&mut self.scratch_gpu),
                _ => None,
            },
            Residency::Cpu => Some(&mut self.cpu),
            Residency::Replayable | Residency::Missing => None,
        }
    }

    /// The pool a freshly-allocated object of `kind` draws from (for capacity
    /// queries).
    fn pool_for(&self, kind: ArenaKind) -> &BlockPool {
        match kind {
            ArenaKind::KvPage => &self.kv_gpu,
            ArenaKind::RsSlab => &self.rs_gpu,
            ArenaKind::Scratch => &self.scratch_gpu,
            ArenaKind::CpuStash | ArenaKind::SnapshotBlob => &self.cpu,
        }
    }
}

/// Internal CoW decision, before any refcount fold-in.
enum CowOutcome {
    InPlace,
    Copied {
        new_id: ObjectId,
        from: Vec<BlockId>,
        to: Vec<BlockId>,
    },
}

#[cfg(test)]
mod tests;
