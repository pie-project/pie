use crate::rt::Instant;
use std::collections::HashSet;
use std::sync::Arc;

use super::{ProcessId, ResidencyPlanner};

use crate::scheduler::{TierKind, TierMove};
use crate::store::kv::page_table::WorkingSetId;
use crate::store::kv::working_set::KvSuspendHandle;
use crate::store::kv::{KvResidencyTxn, KvSpillTxn, KvStoreError};
use crate::store::roles::{Owner, RoleMove, RolesHandle};
use crate::store::rs::RsResidencyTxn;
use ::engine::transfer::MemoryDomain;

fn spawn_watched(
    planner: Arc<ResidencyPlanner>,
    pid: ProcessId,
    label: &'static str,
    task: impl std::future::Future<Output = ()> + Send + 'static,
    on_fail: impl FnOnce(&Arc<ResidencyPlanner>, ProcessId) + Send + 'static,
) -> bool {
    if !crate::rt::has_runtime() {
        return false;
    };
    let handle = crate::rt::spawn(task);
    crate::rt::spawn(async move {
        if let Err(join_error) = handle.await {
            println!("[planner-exec] pid={pid} {label} task DIED: {join_error}");
            on_fail(&planner, pid);
        }
    });
    true
}

pub(super) fn spawn_evict(planner: Arc<ResidencyPlanner>, pid: ProcessId) {
    let task = evict(planner.clone(), pid);
    let spawned = spawn_watched(planner.clone(), pid, "evict", task, |planner, pid| {
        planner.eviction_failed(pid)
    });
    if !spawned {
        planner.eviction_failed(pid);
    }
}

/// No process reads them, so a failure only puts them back.
pub(super) fn spawn_move_out(
    planner: Arc<ResidencyPlanner>,
    txn: KvResidencyTxn,
    label: &'static str,
    done: fn(&Arc<ResidencyPlanner>, bool),
) {
    let (model, engine) = planner.locus();
    // The guard owns the move from here: a task that never runs drops it,
    // and that gives back what the move took.
    let guard = MoveGuard::new(
        model,
        engine,
        Move::Suspend {
            kv: Some(txn),
            rs: None,
            roles: None,
        },
    );
    let task = move_out(planner.clone(), guard, label, done);
    let spawned = spawn_watched(
        planner.clone(),
        ProcessId::nil(),
        label,
        task,
        move |planner, _| done(planner, false),
    );
    if !spawned {
        done(&planner, false);
    }
}

/// Idle cache on the host goes on down to disk; a failure only leaves it
/// on the host.
pub(super) fn spawn_spill(
    planner: Arc<ResidencyPlanner>,
    txn: KvSpillTxn,
    done: fn(&Arc<ResidencyPlanner>, bool),
) {
    let (model, engine) = planner.locus();
    // The guard owns the move from here: a task that never runs drops it,
    // and that gives back what the move took.
    let guard = SpillGuard {
        model,
        engine,
        txn: Some(txn),
        in_flight: None,
    };
    let task = {
        let planner = planner.clone();
        async move {
            let moved = spill(&planner, guard).await;
            if let Err(error) = &moved {
                tracing::warn!(%error, "planner: host to disk spill failed");
            }
            done(&planner, moved.is_ok());
        }
    };
    let spawned = spawn_watched(
        planner.clone(),
        ProcessId::nil(),
        "spill",
        task,
        move |planner, _| done(planner, false),
    );
    if !spawned {
        done(&planner, false);
    }
}

/// Parked role slots left on the device go on down a tier, the host or the
/// disk ([`crate::store::roles::Roles::prepare_park_down`]): their process is
/// evicted, so no fire reads them; a failure leaves them where they lie.
pub(super) fn spawn_park_down(
    planner: Arc<ResidencyPlanner>,
    pids: Vec<ProcessId>,
    roles: Option<RoleMove>,
    rs: Option<RsResidencyTxn>,
) {
    let (model, engine) = planner.locus();
    // The guard owns the move from here: a task that never runs drops it,
    // and that gives back what the move took.
    let guard = MoveGuard::new(
        model,
        engine,
        Move::Suspend {
            kv: None,
            rs,
            roles,
        },
    );
    let task = {
        let planner = planner.clone();
        async move {
            let moved = park_down(&planner, &pids, guard).await;
            if let Err(error) = &moved {
                tracing::warn!(%error, "planner: parked role slots could not move down");
            }
            planner.parked_down_done(moved.is_ok());
        }
    };
    let spawned = spawn_watched(
        planner.clone(),
        ProcessId::nil(),
        "park-down",
        task,
        |planner, _| planner.parked_down_done(false),
    );
    if !spawned {
        planner.parked_down_done(false);
    }
}

/// A list compaction ([`crate::store::kv::ListCompaction`]): the pages of
/// the device's emptiest list chunks move to free slots of the others, each
/// reader's sets fenced meanwhile, so whole chunks stand free; a failure
/// leaves every page where it lay.
pub(super) fn spawn_compact_lists(
    planner: Arc<ResidencyPlanner>,
    plan: crate::store::kv::ListCompaction,
) {
    let task = {
        let planner = planner.clone();
        async move {
            let moved = compact_lists(&planner, &plan).await;
            planner.list_compaction_done(moved);
        }
    };
    let spawned = spawn_watched(
        planner.clone(),
        ProcessId::nil(),
        "list-compaction",
        task,
        |planner, _| planner.list_compaction_done(Err("its task died".into())),
    );
    if !spawned {
        planner.list_compaction_done(Err("no runtime for its task".into()));
    }
}

/// Moves `plan`'s pages; the pages moved, or why none did. A reader with a
/// fire in flight (a lease held) is not waited for: its fence comes down
/// at once and the chunks its pages lie in stay, so no page a fire in flight
/// reads ever moves, and no wait here hangs on a fire that waits on room.
async fn compact_lists(
    planner: &ResidencyPlanner,
    plan: &crate::store::kv::ListCompaction,
) -> Result<usize, String> {
    let (model, engine) = planner.locus();
    // Only the running readers are fenced: a parked one's sets are fenced
    // already, and stay so (its restore waits for this to land).
    let mut fences = Vec::new();
    let mut fenced = HashSet::new();
    for &pid in &plan.readers {
        let handles = crate::inferlet::process::residency::kv_suspend_handles(pid, model, engine);
        let fence = FenceGuard::raise(handles, pid, (model, engine));
        if fence.handles.iter().all(|handle| handle.active_leases() == 0) {
            fenced.insert(pid);
            fences.push(fence);
        }
    }
    let stores = crate::store::registry::get(model, engine);
    let prepared = crate::store::registry::with_kv_lock(&stores.kv, "planner-compact", |kv| {
        kv.prepare_relocate(plan, &fenced)
    });
    let Some(txn) = prepared else {
        return Err("no chunk of the plan empties now".into());
    };
    tracing::debug!(
        pages = txn.page_count(),
        chunks = ?txn.chunks(),
        fenced = fenced.len(),
        "planner: a list compaction copies its pages device to device"
    );
    let (from, to) = txn.copy_plan();
    let copy = TierMove {
        kind: TierKind::kv(txn.shape()),
        tier: crate::scheduler::device_domain(engine),
        out: true,
        device: from,
        slots: to,
        source: None,
    };
    // The chunks the copy reads and writes: no chunk move takes them while
    // it does.
    let chunks: Vec<u32> = match copy.kind {
        TierKind::KvSlots(shape) => {
            let mut chunks = planner.port.copy_chunks(shape, &copy.device);
            chunks.extend(planner.port.copy_chunks(shape, &copy.slots));
            chunks
        }
        _ => Vec::new(),
    };
    planner.port.moving_out(&chunks);
    let started = Instant::now();
    let copied = match crate::scheduler::copy_tier_tracked(engine, &copy) {
        Ok(completion) => completion
            .wait()
            .await
            .map_err(|error| CopyFailure::of(&copy.label(), "copy", &error)),
        Err(error) => Err(CopyFailure::of(&copy.label(), "submit", &error)),
    };
    planner.port.moved_out(&chunks);
    planner.record_tier_copy(&copy, started.elapsed());
    let result = match copied {
        Ok(()) => crate::store::registry::with_kv_lock(&stores.kv, "planner-compact", |kv| {
            kv.commit_relocate(txn)
        })
        .map_err(|error| format!("its commit: {error}")),
        Err(failure) => {
            // Whatever of the copy the engine took has landed before the
            // slots it wrote go back.
            if !failure.untouched
                && let Err(error) = crate::scheduler::settle_copies(engine).await
            {
                tracing::error!(%error, "planner: a list compaction's copy did not settle; its slots stay out of use");
                return Err(failure.message);
            }
            crate::store::registry::with_kv_lock(&stores.kv, "planner-compact", |kv| {
                kv.abort_relocate(txn)
            });
            Err(failure.message)
        }
    };
    drop(fences);
    result
}

async fn park_down(
    planner: &ResidencyPlanner,
    pids: &[ProcessId],
    mut guard: MoveGuard,
) -> Result<(), String> {
    // The chunks its copy reads, fenced from a raise until it lands, as an
    // eviction's are ([`MovingOut`]).
    let reading = MovingOut::of_all(planner, pids);
    let copied = guard.copy(planner, ProcessId::nil()).await;
    drop(reading);
    if let Err(error) = copied {
        // Taken back from its sources, which still hold it (a copy that
        // did not land among them): it is tried again as room is asked.
        guard.take_back().await;
        return Err(error.into());
    }
    // Its instances' tables leave the device now: said parked again, kept
    // nowhere, before the commit hands their slots to anyone else. Refused
    // partway, the move is taken back and those told are parked where they
    // lie again.
    if let Err(refused) = park(planner.locus().1, guard.parked()).await {
        guard.take_back().await;
        let (model, engine) = planner.locus();
        undo_parks(engine, model, ProcessId::nil(), refused.told, true).await;
        return Err(refused.message);
    }
    let Move::Suspend { roles, rs, .. } = guard.take() else {
        unreachable!("a park down is a move down")
    };
    let (model, engine) = planner.locus();
    let stores = crate::store::registry::get(model, engine);
    if let (Some(txn), Some(store)) = (roles, &stores.roles) {
        store.lock().unwrap().commit(txn);
    }
    // An evicted process's state left on the device at its eviction.
    if let Some(txn) = rs {
        stores.rs.lock().unwrap().commit_suspend(txn);
    }
    Ok(())
}

/// Copies a spill's host slots to its disk slots (`copy_slots`, HostPinned
/// -> LocalDisk) and commits it: answers the pages moved.
pub(super) async fn spill(
    planner: &ResidencyPlanner,
    mut guard: SpillGuard,
) -> Result<usize, String> {
    let txn = guard.txn.as_ref().expect("spill present");
    let (host, disk) = txn.copy_plan();
    let copy = TierMove {
        kind: TierKind::KvSlots(txn.shape()),
        tier: MemoryDomain::LocalDisk,
        out: true,
        device: host,
        slots: disk,
        source: Some(MemoryDomain::HostPinned),
    };
    tracing::debug!(copy = %copy.label(), count = copy.device.len(), "planner: spill copy");
    let started = Instant::now();
    let completion = crate::scheduler::copy_tier_tracked(guard.engine, &copy)
        .map_err(|error| format!("{} submit: {error:#}", copy.label()))?;
    guard.in_flight = Some(completion.clone());
    let copied = completion.wait().await;
    guard.in_flight = None;
    planner.record_tier_copy(&copy, started.elapsed());
    copied.map_err(|error| format!("{} copy: {error}", copy.label()))?;
    let txn = guard.txn.take().expect("spill present");
    let stores = crate::store::registry::get(guard.model, guard.engine);
    Ok(crate::store::registry::with_kv_lock(
        &stores.kv,
        "planner-spill",
        |kv| kv.commit_spill(txn),
    ))
}

/// Gives back a spill's disk slots when it is dropped before it commits,
/// after any copy in flight.
pub(super) struct SpillGuard {
    model: usize,
    engine: usize,
    txn: Option<KvSpillTxn>,
    in_flight: Option<crate::scheduler::ControlCompletion>,
}

impl SpillGuard {
    pub(super) fn new(model: usize, engine: usize, txn: KvSpillTxn) -> Self {
        Self {
            model,
            engine,
            txn: Some(txn),
            in_flight: None,
        }
    }
}

impl Drop for SpillGuard {
    fn drop(&mut self) {
        let Some(txn) = self.txn.take() else {
            return;
        };
        let (model, engine) = (self.model, self.engine);
        let abort = move || {
            let stores = crate::store::registry::get(model, engine);
            crate::store::registry::with_kv_lock(&stores.kv, "spill-abort", |kv| {
                kv.abort_spill(txn)
            });
        };
        match self.in_flight.take() {
            None => abort(),
            Some(completion) if crate::rt::has_runtime() => {
                crate::rt::spawn(async move {
                    let _ = completion.wait().await;
                    abort();
                });
            }
            Some(_) => tracing::error!(
                model,
                engine,
                "spill dropped with an engine copy in flight and no runtime; \
                 keeping its slots to avoid reuse during the copy"
            ),
        }
    }
}

pub(super) fn spawn_reseat(planner: Arc<ResidencyPlanner>, pid: ProcessId) {
    let task = {
        let planner = planner.clone();
        async move {
            let seat = crate::inferlet::process::take_seat(pid).await;
            planner.reseated(pid, seat);
        }
    };
    let spawned = spawn_watched(planner.clone(), pid, "reseat", task, |planner, pid| {
        planner.reseat_failed(pid)
    });
    if !spawned {
        planner.reseat_failed(pid);
    }
}

pub(super) fn spawn_restore(
    planner: Arc<ResidencyPlanner>,
    pid: ProcessId,
    pages: super::grant::DevicePageReservation,
    slots: super::grant::RsSlotReservation,
) {
    let task = restore(planner.clone(), pid, pages, slots);
    let spawned = spawn_watched(planner.clone(), pid, "restore", task, |planner, pid| {
        planner.restore_failed(pid, "restore executor died")
    });
    if !spawned {
        planner.restore_faulted(pid, "no tokio runtime for the restore executor");
    }
}

struct FenceGuard {
    handles: Vec<KvSuspendHandle>,
    /// Whose sets: an eviction rolled back unfences every set its process
    /// holds then, one born fenced meanwhile included.
    pid: ProcessId,
    locus: (usize, usize),
    armed: bool,
}

impl FenceGuard {
    fn raise(handles: Vec<KvSuspendHandle>, pid: ProcessId, locus: (usize, usize)) -> Self {
        for handle in &handles {
            handle.fence();
        }
        Self {
            handles,
            pid,
            locus,
            armed: true,
        }
    }

    fn keep_raised(&mut self) {
        self.armed = false;
    }
}

impl Drop for FenceGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let (model, engine) = self.locus;
        for handle in crate::inferlet::process::residency::kv_suspend_handles(self.pid, model, engine)
        {
            handle.unfence();
        }
    }
}

/// A process's residency move: its kv and rs transactions and its instance
/// and channel slots (`roles`), prepared in the stores and committed once
/// the engine copies land, any of which may be absent when that store holds
/// nothing to move.
enum Move {
    Suspend {
        kv: Option<KvResidencyTxn>,
        rs: Option<RsResidencyTxn>,
        roles: Option<RoleMove>,
    },
    Restore {
        kv: Option<KvResidencyTxn>,
        rs: Option<RsResidencyTxn>,
        roles: Option<RoleMove>,
    },
}

impl Move {
    /// The engine copies the move takes: kv pages by the tier they are
    /// backed in, then the rs slots.
    fn copies(&self) -> Vec<TierMove> {
        let (out, kv, rs, roles) = match self {
            Move::Suspend { kv, rs, roles } => (true, kv, rs, roles),
            Move::Restore { kv, rs, roles } => (false, kv, rs, roles),
        };
        let mut copies: Vec<TierMove> = kv
            .iter()
            .flat_map(|txn| {
                let kind = TierKind::kv(txn.shape());
                txn.copy_plans()
                    .into_iter()
                    .map(move |(tier, device, slots)| TierMove {
                        kind,
                        tier,
                        out,
                        device,
                        slots,
                        source: None,
                    })
            })
            .collect();
        if let Some(rs) = rs.as_ref() {
            let plan = rs.copy_plan();
            // A shared pool's state moves shape by shape (CONTRACT-BC 22).
            // With no host row free (no host tier on unified memory) it
            // moves to the disk tier's slot file, laid out by shape too.
            let tiers = [
                (&plan.shared, MemoryDomain::HostPinned),
                (&plan.disk, MemoryDomain::LocalDisk),
            ];
            for (moves, tier) in tiers {
                for (shape, pairs) in moves {
                    let (device, slots) = pairs.iter().copied().unzip();
                    copies.push(TierMove {
                        kind: TierKind::RsSlots(*shape),
                        tier,
                        out,
                        device,
                        slots,
                        source: None,
                    });
                }
            }
        }
        // Instance tables and rings park with their process (never
        // dropped: no token rebuilds them).
        for copy in roles.iter().flat_map(RoleMove::copies) {
            copies.push(TierMove {
                kind: TierKind::RoleSlots(copy.shape),
                tier: if copy.disk {
                    MemoryDomain::LocalDisk
                } else {
                    MemoryDomain::HostPinned
                },
                out,
                device: copy.device.clone(),
                slots: copy.host.clone(),
                source: None,
            });
        }
        copies
    }
}

/// A dropped move whose copies were handed to the engine: aborted once
/// they have landed and the engine says it settled them (`settled`), or,
/// where it cannot, abandoned ([`abandon`]): either way it ends.
async fn settle_or_abandon(
    model: usize,
    engine: usize,
    in_flight: Vec<crate::scheduler::ControlCompletion>,
    settled: impl std::future::Future<Output = anyhow::Result<()>>,
    txn: Move,
) {
    let held = std::sync::Mutex::new(Some(txn));
    let aborted = crate::scheduler::abort_once_landed(in_flight, settled, || {
        if let Some(txn) = held.lock().unwrap().take() {
            abort(model, engine, txn);
        }
    })
    .await;
    if let Err(error) = aborted {
        if let Some(txn) = held.lock().unwrap().take() {
            abandon(model, engine, txn, &error);
        }
        return;
    }
    if let Some(planner) = crate::planner::planner_for(model, engine) {
        planner.pages_freed();
    }
}

/// A move whose copies the engine could not settle: its kv and state
/// pages stay out of reuse as before (a late copy may still write them),
/// and its role items end their move where it found them, the slots taken
/// for them leaked by name ([`crate::store::roles::Roles::abandon`]), so a
/// pair's other holders never wait on it.
fn abandon(model: usize, engine: usize, txn: Move, why: &str) {
    let roles = match txn {
        Move::Suspend { kv, rs, roles } | Move::Restore { kv, rs, roles } => {
            // Kept out of reuse: dropped without an abort, their pages and
            // slots are not given back.
            drop((kv, rs));
            roles
        }
    };
    let stores = crate::store::registry::get(model, engine);
    let (leaked, total) = match (roles, &stores.roles) {
        (Some(txn), Some(store)) => {
            let mut store = store.lock().unwrap();
            let leaked = store.abandon(txn);
            (leaked, store.leaked())
        }
        _ => (0, 0),
    };
    if let Some(planner) = crate::planner::planner_for(model, engine) {
        planner.note_leaked(total);
    }
    tracing::error!(
        model,
        engine,
        why,
        leaked_role_slots = leaked,
        "leaked: an engine could not settle the copies of an aborted move; its kv and \
         state pages and the role slots it took stay out of service, and its items end \
         their move where it found them"
    );
}

fn abort(model: usize, engine: usize, txn: Move) {
    let stores = crate::store::registry::get(model, engine);
    let roles = match txn {
        Move::Suspend { kv, rs, roles } => {
            if let Some(txn) = kv {
                crate::store::registry::with_kv_lock(&stores.kv, "planner-evict", |kv| {
                    kv.abort_suspend(txn)
                });
            }
            if let Some(txn) = rs {
                stores.rs.lock().unwrap().abort_suspend(txn);
            }
            roles
        }
        Move::Restore { kv, rs, roles } => {
            if let Some(txn) = kv {
                crate::store::registry::with_kv_lock(&stores.kv, "planner-restore", |kv| {
                    kv.abort_restore(txn)
                });
            }
            if let Some(txn) = rs {
                stores.rs.lock().unwrap().abort_restore(txn);
            }
            roles
        }
    };
    if let (Some(txn), Some(store)) = (roles, &stores.roles) {
        store.lock().unwrap().abort(txn);
    }
}

/// Aborts the move it still holds when dropped, after any copy in flight
/// has landed: a copy's completion may settle before it has written its
/// destinations (its sources go back at once where nothing can reach them
/// first), so the slots an abort gives back wait for every copy to land.
struct MoveGuard {
    model: usize,
    engine: usize,
    txn: Option<Move>,
    in_flight: Vec<crate::scheduler::ControlCompletion>,
    /// A copy of the move was handed to the engine.
    submitted: bool,
}

impl MoveGuard {
    fn new(model: usize, engine: usize, txn: Move) -> Self {
        Self {
            model,
            engine,
            txn: Some(txn),
            in_flight: Vec::new(),
            submitted: false,
        }
    }

    /// Submits every copy of the move, then waits for them all.
    async fn copy(&mut self, planner: &ResidencyPlanner, pid: ProcessId) -> Result<(), CopyFailure> {
        // Every device chunk the copies read or write is fenced until they
        // land: no evacuation moves it, no segment holding it is released,
        // no compaction or raise takes it (the L40S at 3 GiB on a0bfb512f:
        // CUDA 700 on an eviction's queued kv-to-disk copy).
        let fenced: Vec<u32> = self
            .txn
            .as_ref()
            .expect("move present")
            .copies()
            .iter()
            .filter(|copy| copy.source.is_none())
            .flat_map(|copy| match copy.kind {
                crate::scheduler::dispatch::TierKind::KvSlots(shape)
                | crate::scheduler::dispatch::TierKind::RsSlots(shape)
                | crate::scheduler::dispatch::TierKind::RoleSlots(shape) => {
                    planner.port.copy_chunks(shape, &copy.device)
                }
                crate::scheduler::dispatch::TierKind::Kv => Vec::new(),
            })
            .collect();
        planner.port.moving_out(&fenced);
        let copied = self.copy_fenced(planner, pid).await;
        planner.port.moved_out(&fenced);
        copied
    }

    async fn copy_fenced(&mut self, planner: &ResidencyPlanner, pid: ProcessId) -> Result<(), CopyFailure> {
        let mut waits = Vec::new();
        for copy in self.txn.as_ref().expect("move present").copies() {
            tracing::debug!(pid = %pid, copy = %copy.label(), count = copy.device.len(), "planner: residency copy");
            let started = Instant::now();
            let completion = crate::scheduler::copy_tier_tracked(self.engine, &copy)
                .map_err(|error| CopyFailure::of(&copy.label(), "submit", &error))?;
            self.submitted = true;
            self.in_flight.push(completion.clone());
            waits.push((copy, started, completion));
        }
        for (copy, started, completion) in waits {
            let copied = completion.wait().await;
            ptrace!(
                "tier-copy {} slots={} ms={}",
                copy.label(),
                copy.device.len(),
                started.elapsed().as_millis()
            );
            planner.record_tier_copy(&copy, started.elapsed());
            copied.map_err(|error| CopyFailure::of(&copy.label(), "copy", &error))?;
        }
        self.in_flight.clear();
        Ok(())
    }

    fn take(&mut self) -> Move {
        self.txn.take().expect("move present")
    }

    /// The instances its park moves down or leaves where they lie
    /// ([`RoleMove::parked`]).
    fn parked(&self) -> Vec<::engine::Rebind> {
        match self.txn.as_ref() {
            Some(Move::Suspend {
                roles: Some(roles), ..
            }) => roles.parked(),
            _ => Vec::new(),
        }
    }
}

/// The physical chunks a process's state holds, fenced against a raise's
/// taking them back while a copy of the process reads them; let go when
/// dropped ([`super::pool::PoolPort::moving_out`]).
struct MovingOut<'a> {
    planner: &'a ResidencyPlanner,
    chunks: Vec<u32>,
}

impl<'a> MovingOut<'a> {
    fn of(planner: &'a ResidencyPlanner, pid: ProcessId) -> Self {
        Self::of_all(planner, &[pid])
    }

    fn of_all(planner: &'a ResidencyPlanner, pids: &[ProcessId]) -> Self {
        let mut map = planner.port.chunk_occupants(pids);
        map.merge(planner.port.parked_occupants(super::residency_planner::SWEEP));
        let chunks: Vec<u32> = map
            .held
            .iter()
            .filter(|held| held.by.iter().any(|(holder, _)| pids.contains(holder)))
            .map(|held| held.chunk)
            .collect();
        planner.port.moving_out(&chunks);
        Self { planner, chunks }
    }
}

impl Drop for MovingOut<'_> {
    fn drop(&mut self) {
        self.planner.port.moved_out(&self.chunks);
    }
}

/// A residency move's copy that did not land: what it says, and whether
/// the engine refused it before reading or writing any of it (its state
/// lies where it lay, untouched: a placement the engine will not copy,
/// the runtime's own doing, never the process's fault).
#[derive(Debug)]
pub(super) struct CopyFailure {
    message: String,
    untouched: bool,
    /// The tier it copied down to is full ([`crate::scheduler::TierBelowFull`]):
    /// nothing on the device changed.
    tier_full: bool,
    /// A copy did not land ([`crate::scheduler::LandingFailed`]): its
    /// destination is not to be trusted, its sources hold what they held,
    /// and the move is taken back from them and tried again.
    landing_failed: bool,
}

impl CopyFailure {
    fn of(label: &str, step: &str, error: &anyhow::Error) -> Self {
        Self {
            message: format!("{label} {step}: {error:#}"),
            untouched: error
                .chain()
                .any(|cause| cause.downcast_ref::<crate::scheduler::RefusedUntouched>().is_some()),
            tier_full: error
                .chain()
                .any(|cause| cause.downcast_ref::<crate::scheduler::TierBelowFull>().is_some()),
            landing_failed: error
                .chain()
                .any(|cause| cause.downcast_ref::<crate::scheduler::LandingFailed>().is_some()),
        }
    }
}

impl std::fmt::Display for CopyFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl From<CopyFailure> for String {
    fn from(failure: CopyFailure) -> Self {
        failure.message
    }
}

impl MoveGuard {
    /// Takes the move back now, once every copy of it handed to the engine
    /// has landed: what follows reads the items where the move found them.
    /// Dropped instead, the take-back ran later on a task of its own, and
    /// the process said resident meanwhile was picked again, its second
    /// eviction finding its items still moving and parking none of them;
    /// the first move's take-back then left its instance bound on the
    /// device beside a process said evicted, and its pair was taken anew
    /// at its restore under it (the simulation's seed 1348: a bind over a
    /// live pair's run).
    async fn take_back(mut self) {
        let Some(txn) = self.txn.take() else {
            return;
        };
        let in_flight = std::mem::take(&mut self.in_flight);
        if !self.submitted {
            abort(self.model, self.engine, txn);
            return;
        }
        let settled = crate::scheduler::settle_copies(self.engine);
        settle_or_abandon(self.model, self.engine, in_flight, settled, txn).await;
    }
}

impl Drop for MoveGuard {
    fn drop(&mut self) {
        let Some(txn) = self.txn.take() else {
            return;
        };
        let in_flight = std::mem::take(&mut self.in_flight);
        if !self.submitted {
            abort(self.model, self.engine, txn);
            return;
        }
        let (model, engine) = (self.model, self.engine);
        if !crate::rt::has_runtime() {
            tracing::error!(
                model,
                engine,
                "residency move dropped with an engine copy in flight and no runtime; \
                 preserving its pages and slots to avoid reuse during the copy"
            );
            return;
        };
        crate::rt::spawn(async move {
            let settled = crate::scheduler::settle_copies(engine);
            settle_or_abandon(model, engine, in_flight, settled, txn).await;
        });
    }
}

async fn drain_detachable(pid: ProcessId) {
    let pipelines = crate::inferlet::process::residency::pipelines_of(pid);
    for fires in pipelines {
        let Some(_finalize_guard) = fires.try_finalize_guard() else {
            continue;
        };
        loop {
            let op = {
                let mut queue = fires.lock().unwrap();
                match queue.front() {
                    Some(op) if op.is_preemption_detachable() && op.is_settled() => {
                        queue.pop_front()
                    }
                    _ => None,
                }
            };
            let Some(op) = op else {
                break;
            };
            if let Err(error) = crate::pipeline::fire::finalize_op_detached(op).await {
                tracing::warn!(pid = %pid, %error, "planner: detachable finalize failed");
                break;
            }
        }
    }
}

pub(super) async fn evict(planner: Arc<ResidencyPlanner>, pid: ProcessId) {
    // Evicted again before its bind took what its restore held: withdrawn,
    // the reservation goes back.
    planner.port.release_restore(pid);
    let (model, engine) = planner.locus();
    let handles = crate::inferlet::process::residency::kv_suspend_handles(pid, model, engine);
    let working_sets: HashSet<WorkingSetId> =
        crate::inferlet::process::residency::kv_working_set_ids(pid, model, engine);
    let rs_working_sets =
        crate::inferlet::process::residency::rs_working_set_ids(pid, model, engine);
    // A process with no kv or state yet (a prompt not seated) may still hold
    // guest state on the device, the room a bind short of runs was chosen to
    // free (a chunk's occupant): it parks all the same, its guest scratch
    // given back and its tables and resident bytes copied down. Found
    // nothing, the M1 stress froze with eleven binds short of runs and every
    // chosen occupant "found no working set".
    let holds_guest = crate::store::registry::get(model, engine)
        .roles
        .as_ref()
        .is_some_and(|roles| roles.lock().unwrap().holds_device_parts(pid));
    if working_sets.is_empty() && rs_working_sets.is_empty() && !holds_guest {
        // Nothing to move now is nothing to move on the next pass either:
        // it is no victim again until the room changes (a restore clears
        // the blocks), or the plan picked it again at once, forever.
        tracing::debug!(pid = %pid, "planner: eviction found no working set");
        planner.eviction_found_nothing(pid);
        return;
    }
    let began = Instant::now();
    let mut fence = FenceGuard::raise(handles, pid, (model, engine));
    crate::scheduler::worker::notify_process_suspend(pid);
    // Each wait of the prelude said as it begins: an eviction that never
    // reports done names the wait it sits in (Mac Pro's beam mix: evicting
    // stuck at 2 after a bind-suspend, nothing said).
    tracing::debug!(%pid, "planner: an eviction drains its detachable fires");
    drain_detachable(pid).await;
    let drained = began.elapsed();
    // What it waits on, said as it begins: the fires holding its sets'
    // leases (submitted, or admitted and held in an unsealed frame), and the
    // engine's capped frames with the raise they wait on (a capped frame of
    // its own goes once raised: its raise stays queued through its
    // eviction, [`super::inner::Inner::begin_eviction`]).
    tracing::debug!(
        %pid,
        sets = fence.handles.len(),
        leases = fence.handles.iter().map(KvSuspendHandle::active_leases).sum::<usize>(),
        capped = %planner.capped_line(),
        "planner: an eviction waits for its sets' fires"
    );
    for handle in fence.handles.iter() {
        handle.quiesce().await;
    }
    tracing::debug!(%pid, "planner: an eviction waits for its state's device folds");
    let stores = crate::store::registry::get(model, engine);
    // A set a fire still folds by a count the device computes keeps its
    // state where the engine holds the record, whose length is the engine's
    // alone until that fold settles: the suspend waits for it, as it waits
    // for the fires themselves, and every way a fire ends wakes it.
    let rs_sets: Vec<_> = rs_working_sets.iter().copied().collect();
    crate::store::rs::device_folds_settled(&stores.rs, &rs_sets).await;
    let quiesced = began.elapsed();
    // Host room is for what is paused before what is cached: idle cache on
    // a host short of room goes on down to disk first, not dropped.
    let spilling = crate::store::registry::with_kv_lock(&stores.kv, "planner-evict", |kv| {
        kv.prepare_suspend_spill(&working_sets, crate::policy::retention::Installed)
    });
    if let Some(txn) = spilling
        && let Err(error) = spill(&planner, SpillGuard::new(model, engine, txn)).await
    {
        tracing::warn!(pid = %pid, %error, "planner: host to disk spill failed");
    }
    // A set released while the eviction waited above (its pipeline ended,
    // or failed and let it go) has nothing left to move: it is passed over,
    // never an error. Named an unknown set, the whole eviction failed, the
    // process went back resident and was picked again, its lanes held.
    // With every set gone it moves only what else it holds, and with
    // nothing it is "found nothing", its lanes released as any such.
    let (working_sets, prepared) =
        crate::store::registry::with_kv_lock(&stores.kv, "planner-evict", |kv| {
            let live: HashSet<WorkingSetId> = working_sets
                .iter()
                .copied()
                .filter(|&ws| kv.page_len(ws).is_ok())
                .collect();
            if live.len() < working_sets.len() {
                tracing::debug!(
                    pid = %pid,
                    gone = working_sets.len() - live.len(),
                    "planner: an eviction's sets released while it waited are passed over"
                );
            }
            let prepared = kv.prepare_suspend(&live, crate::policy::retention::Installed);
            (live, prepared)
        });
    let kv = match prepared {
        Ok(kv) => kv,
        // No room below for its kv, but guest parts of its own on the
        // device: its kv parks where it lies (marked suspended, as on a home
        // with no tier below; moved down later as room comes) and the park
        // goes on for what does move, its guest scratch given back and its
        // tables copied down. Blocked whole, a waiter holding chunks only its
        // guest parts filled kept them while every process waited (seed
        // 3392: a full disk, every resident's kv past its free slots).
        Err(KvStoreError::HostSwapFull { .. }) if holds_guest => {
            tracing::debug!(%pid, "planner: an eviction's kv parks in place, no room below");
            None
        }
        Err(error @ KvStoreError::HostSwapFull { .. }) => {
            tracing::warn!(pid = %pid, %error, "planner: eviction blocked on swap room");
            planner.eviction_failed_host_swap_full(pid);
            return;
        }
        Err(error) => {
            tracing::warn!(pid = %pid, %error, "planner: suspend prepare failed");
            planner.eviction_failed(pid);
            return;
        }
    };
    let (rs, rs_short) = {
        let mut store = stores.rs.lock().unwrap();
        let rs = store.prepare_suspend(&rs_working_sets);
        let short = if rs.is_none() {
            store.suspend_short(&rs_working_sets)
        } else {
            Default::default()
        };
        (rs, short)
    };
    // The process's instance tables and rings park with its state where the
    // host has room; what it lacks stays on the device, never dropped. Its
    // guest scratch, dead between fires, is given back whatever lies below:
    // a process with nothing else to move still frees its Instance chunks.
    // A bind of its on its way (taken, the engine's bind still to land)
    // lands first: its instance parks with the rest. Parked under it, the
    // instance stood on the device outside the park and its pair's slots
    // went to another bind ([`crate::store::roles::Roles::binds_on_way`]).
    let roles = match stores.roles.as_ref() {
        Some(roles) => loop {
            let landed = roles.lock().unwrap().bind_landed();
            let notified = landed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let mut held = roles.lock().unwrap();
                if !held.binds_on_way(pid) {
                    break held.prepare_park(pid);
                }
            }
            tracing::debug!(%pid, "planner: an eviction waits for its bind on its way");
            notified.await;
        },
        None => None,
    };
    if kv.is_none() && rs.is_none() && roles.is_none() {
        if !rs_short.is_zero() {
            // The host lacks rows for its state: nothing moves and nothing
            // is dropped; the head stays parked and the host is short.
            tracing::debug!(pid = %pid, %rs_short, "planner: eviction short of host rows");
            planner.note_host_short(rs_short);
            planner.eviction_failed_host_swap_full(pid);
            return;
        }
        tracing::debug!(pid = %pid, "planner: eviction found nothing reclaimable");
        planner.eviction_found_nothing(pid);
        return;
    }
    let prepared_at = began.elapsed();
    let mut guard = MoveGuard::new(model, engine, Move::Suspend { kv, rs, roles });
    // The chunks its copy reads: no raise takes them back for its
    // activation until the copy has landed or will not (Mac Pro at 5 GiB
    // on 5aa45805b: an eviction's copy of a lent chunk refused inside the
    // carve).
    let reading = MovingOut::of(&planner, pid);
    let copied_out = guard.copy(&planner, pid).await;
    drop(reading);
    if let Err(error) = copied_out {
        tracing::warn!(pid = %pid, %error, "planner: eviction copy failed");
        if error.tier_full {
            // The tier below is full: nothing on the device changed, the
            // move is taken back, the process stays resident, once the
            // take-back is done.
            guard.take_back().await;
            planner.eviction_tier_full(pid, &error.message);
        } else if error.untouched {
            // Refused untouched: its state lies where it lay, and the move
            // is taken back; the process stays as it was.
            guard.take_back().await;
            planner.eviction_copy_refused(pid, &error.message);
        } else if error.landing_failed {
            // A copy did not land: its state is still on the device, where
            // it was copied from. The move is taken back and the process
            // stays resident, its eviction tried again from there once the
            // pool moves.
            guard.take_back().await;
            planner.eviction_landing_failed(pid, &error.message);
        } else {
            guard.take_back().await;
            planner.eviction_copy_failed(pid, &error.message);
        }
        return;
    }
    let copied = began.elapsed();
    // Its instances said parked on its lane before the commit hands their
    // slots to anyone else: an engine still holding them at those slots
    // refuses the next owner's bind there (Vulkan on a0bfb512f, long
    // prompts: "Instance slots ... already hold instance N's run"), or
    // lays a parked pair over it after an evacuation (Metal). One it
    // refuses fails the process by name, its move taken back.
    // Refused partway, the move is taken back and those told are rebound
    // where they lie, live again, before the process fires.
    let told = guard.parked();
    if let Err(refused) = park(engine, told.clone()).await {
        tracing::warn!(pid = %pid, error = %refused.message, "planner: eviction's park not taken by the engine");
        guard.take_back().await;
        undo_parks(engine, model, pid, refused.told, false).await;
        planner.eviction_copy_failed(pid, &refused.message);
        return;
    }
    let Move::Suspend { kv, rs, roles } = guard.take() else {
        unreachable!("an eviction carries a suspend")
    };
    let freed = match kv {
        Some(txn) => {
            let freed = crate::store::registry::with_kv_lock(&stores.kv, "planner-evict", |kv| {
                kv.commit_suspend(txn)
            });
            match freed {
                Ok(freed) => freed,
                Err(error) => {
                    abort(
                        model,
                        engine,
                        Move::Suspend {
                            kv: None,
                            rs,
                            roles,
                        },
                    );
                    // Its instances were said parked: rebound where they
                    // lie, live again, before it fires.
                    undo_parks(engine, model, pid, told, false).await;
                    tracing::warn!(pid = %pid, %error, "planner: suspend commit failed");
                    planner.eviction_failed(pid);
                    return;
                }
            }
        }
        None => {
            crate::store::registry::with_kv_lock(&stores.kv, "planner-evict", |kv| {
                kv.mark_suspended(&working_sets)
            });
            0
        }
    };
    let rs_freed = rs.map_or(Default::default(), |txn| {
        stores.rs.lock().unwrap().commit_suspend(txn)
    });
    if let (Some(txn), Some(store)) = (roles, &stores.roles) {
        store.lock().unwrap().commit(txn);
    }
    fence.keep_raised();
    ptrace!(
        "evict-phases pid={} drained_ms={} quiesced_ms={} prepared_ms={} copied_ms={} done_ms={} freed={}",
        pid,
        drained.as_millis(),
        quiesced.as_millis(),
        prepared_at.as_millis(),
        copied.as_millis(),
        began.elapsed().as_millis(),
        freed
    );
    planner.report_evicted(pid, freed as u32, rs_freed);
}

/// Tells the engine where `pid`'s unparked instance tables and rings lie
/// now, on its lane ahead of any fire of theirs: each item's slots
/// (`Engine::rebind_slots`). An item it refuses stays stale, to be told
/// again on the next restore; one the lane closed since they were read is
/// not rebound ([`crate::store::roles::rebind`]).
async fn rebind(engine: usize, roles: &RolesHandle, pid: ProcessId) -> Result<(), String> {
    let stale = roles.lock().unwrap().stale(pid);
    // The lane marks each rebound and replays the host writes held for it.
    for rebind in stale {
        let owner = Owner::of(&rebind);
        crate::scheduler::rebind_slots(engine, rebind)
            .await
            .map_err(|error| format!("rebind {owner:?}: {error:#}"))?;
    }
    Ok(())
}

/// Says each of a park's instances parked to the engine, on its lane
/// ahead of any later bind or rebind ([`::engine::Rebind::Parked`]); one
/// the lane closed since is not said ([`crate::store::roles::rebind`]).
async fn park(engine: usize, parked: Vec<::engine::Rebind>) -> Result<(), ParkRefused> {
    let mut told = Vec::with_capacity(parked.len());
    for rebind in parked {
        let owner = Owner::of(&rebind);
        if let Err(error) = crate::scheduler::rebind_slots(engine, rebind.clone()).await {
            return Err(ParkRefused {
                message: format!("park {owner:?}: {error:#}"),
                told,
            });
        }
        told.push(rebind);
    }
    Ok(())
}

/// A park the engine refused partway: what it said, and the parks it was
/// told before the refusal, which the take-back must undo
/// ([`undo_parks`]).
struct ParkRefused {
    message: String,
    told: Vec<::engine::Rebind>,
}

/// The instances `told` names.
fn instances_of(told: &[::engine::Rebind]) -> Vec<::engine::InstanceId> {
    told.iter()
        .filter_map(|rebind| match rebind {
            ::engine::Rebind::Parked { id, .. } => Some(*id),
            _ => None,
        })
        .collect()
}

/// Parks said to the engine (`told`) whose move was then taken back: the
/// engine's view restored before `pid` fires again. An eviction's: each
/// instance rebound where it lies, live again ([`Roles::park_undone`]). A
/// park down's (`down`): each parked where it lies again, its tables kept
/// in place (`Rebind::Parked { kept: true }`), as before the move. One the
/// engine refuses is said by name; the process fails at its next fire on
/// the engine's own refusal rather than run on slots it believes parked.
async fn undo_parks(engine: usize, model: usize, pid: ProcessId, told: Vec<::engine::Rebind>, down: bool) {
    if told.is_empty() {
        return;
    }
    let Some(roles) = crate::store::registry::get(model, engine).roles else {
        return;
    };
    let ids = instances_of(&told);
    let undone = if down {
        let mut result = Ok(());
        for id in ids {
            if let Err(error) =
                crate::scheduler::rebind_slots(engine, ::engine::Rebind::Parked { id, kept: true }).await
            {
                result = Err(format!("park {id:?} again: {error:#}"));
                break;
            }
        }
        result
    } else {
        roles.lock().unwrap().park_undone(&ids);
        rebind(engine, &roles, pid).await
    };
    if let Err(error) = undone {
        tracing::error!(%pid, %error, "planner: a park taken back could not be undone on the engine");
    }
}

async fn move_out(
    planner: Arc<ResidencyPlanner>,
    mut guard: MoveGuard,
    label: &'static str,
    done: fn(&Arc<ResidencyPlanner>, bool),
) {
    let (model, engine) = planner.locus();
    if let Err(error) = guard.copy(&planner, ProcessId::nil()).await {
        tracing::warn!(%error, "planner: {label} copy failed");
        guard.take_back().await;
        done(&planner, false);
        return;
    }
    let Move::Suspend { kv: Some(txn), .. } = guard.take() else {
        unreachable!("a {label} carries a kv suspend")
    };
    let stores = crate::store::registry::get(model, engine);
    let moved = crate::store::registry::with_kv_lock(&stores.kv, "planner-move-out", |kv| {
        kv.commit_suspend(txn)
    });
    if let Err(error) = &moved {
        tracing::warn!(%error, "planner: {label} commit failed");
    }
    done(&planner, moved.is_ok());
}

async fn restore(
    planner: Arc<ResidencyPlanner>,
    pid: ProcessId,
    mut pages: super::grant::DevicePageReservation,
    mut slots: super::grant::RsSlotReservation,
) {
    let (model, engine) = planner.locus();
    let working_sets: HashSet<WorkingSetId> =
        crate::inferlet::process::residency::kv_working_set_ids(pid, model, engine);
    let rs_working_sets =
        crate::inferlet::process::residency::rs_working_set_ids(pid, model, engine);
    let stores = crate::store::registry::get(model, engine);
    let began = Instant::now();
    let mut kv = None;
    if !working_sets.is_empty() {
        // Another move may hold pages this process shares: it lands soon,
        // and this waits for it rather than for a clock.
        let ended = crate::store::registry::with_kv_lock(&stores.kv, "planner-restore", |kv| {
            kv.moves_ended()
        });
        loop {
            let notified = ended.notified();
            let mut notified = std::pin::pin!(notified);
            notified.as_mut().enable();
            let prepared =
                crate::store::registry::with_kv_lock(&stores.kv, "planner-restore", |kv| {
                    kv.prepare_restore(&working_sets, pages.lend())
                });
            match prepared {
                Ok(txn) => {
                    kv = Some(txn);
                    break;
                }
                // Another move holds pages it shares: it waits for that move
                // to end, the event, however long.
                Err(KvStoreError::MoveInFlight) => notified.await,
                // More of its pages were swapped out since the grant was
                // quoted: queue again, quoted anew, which is no failure of
                // this restore. It cannot loop on one quote: the serve
                // quotes a restore at the pages swapped out then
                // (`Step::ServeRestore`, its demand replaced when stale),
                // so a second mismatch needs a second move of its pages
                // after that quote, each one an event of its own, and the
                // pages it holds bound how many there can be.
                Err(KvStoreError::GrantMismatch { .. }) => {
                    planner.requeue_restore(pid);
                    return;
                }
                Err(error) => {
                    planner.restore_faulted(pid, &format!("kv prepare: {error}"));
                    return;
                }
            }
        }
    }
    drop(pages);
    let mut rs = None;
    if !rs_working_sets.is_empty() {
        let prepared = stores
            .rs
            .lock()
            .unwrap()
            .prepare_restore(&rs_working_sets, slots.lend());
        match prepared {
            Ok(txn) => rs = Some(txn),
            Err(error) => {
                abort(
                    model,
                    engine,
                    Move::Restore {
                        kv,
                        rs: None,
                        roles: None,
                    },
                );
                planner.restore_faulted(pid, &format!("rs prepare: {error}"));
                return;
            }
        }
    }
    drop(slots);
    // Taken before the unpark: a move of its pair that ends meanwhile is
    // not lost.
    let pair_watch = stores
        .roles
        .as_ref()
        .map(|roles| roles.lock().unwrap().room_watch());
    // The instance tables and rings parked with it come back into fresh
    // device slots, which the engine is told before the process fires.
    let roles = match stores
        .roles
        .as_ref()
        .map(|roles| roles.lock().unwrap().prepare_unpark(pid))
        .transpose()
    {
        Ok(roles) => {
            // Its unpark took its parts out of the room its serve held for
            // it: the placement every other waiter reads changed, and a
            // restore head the pass that sent this one found short (waiting
            // on this one in flight) may stand now. Planned on that, not on
            // this restore's landing (the simulation's lost wakes: "restore
            // short of runs: something in flight, not funded", its
            // placement standing once the restore sent before it unparked).
            if planner.returns.waiters.load(std::sync::atomic::Ordering::Acquire) != 0 {
                planner.poke();
            }
            roles.flatten()
        }
        Err(error) => {
            abort(
                model,
                engine,
                Move::Restore {
                    kv,
                    rs,
                    roles: None,
                },
            );
            // Room the budget holds but nothing has freed yet: the restore
            // waits, queued, as an admission does.
            if let Some(need) = error.unpark_runs() {
                tracing::debug!(
                    %pid,
                    %error,
                    ?error,
                    ?need,
                    "planner: a restore waits for room for its roles"
                );
                planner.requeue_restore_roles(pid, need, error.unpark_free_runs());
                planner.note_roles_short_why(pid, format!("{error:?}"));
                return;
            }
            // Its pair is on its way back with another process's restore:
            // it goes again once that lands, its instances rebound to it.
            // It goes again once that move ends (committed or aborted, the
            // roles say so), never at once: queued again straight away it
            // was served, short and queued again, round after round, as
            // long as the move took (Mac Pro's M5: 27702 rounds).
            // Said once a wait, with what the move carries; woken by the
            // move's end alone (a commit or an abort tells the roles' watch,
            // and the pair no longer moves), never re-planned meanwhile.
            // The move is a park's or a restore's copies of the instances
            // it names: it waits on no queued fire, so nothing queued
            // behind this restore (a holder's fire, an eviction) holds it.
            if let crate::store::roles::RolesError::PairParked(pair) = &error {
                let pair = *pair;
                let roles = stores.roles.as_ref().expect("a pair is a roles item");
                if let Some(moving) = roles.lock().unwrap().pair_moving(pair) {
                    tracing::debug!(
                        %pid,
                        %pair,
                        ?moving,
                        "planner: a restore waits for its pair's move: copies of the \
                         instances it moves with, landing on the engine's lane"
                    );
                }
                let mut watch = pair_watch;
                while let Some(waiting) = watch.take() {
                    let next = roles.lock().unwrap().room_watch();
                    if roles.lock().unwrap().pair_moving(pair).is_none() {
                        break;
                    }
                    waiting.freed().await;
                    watch = Some(next);
                }
                planner.requeue_restore(pid);
                return;
            }
            planner.restore_faulted(pid, &format!("roles unpark: {error}"));
            return;
        }
    };
    // Every instance its rebinds name must be placed once this unpark
    // commits: one left unbound (moving, parked apart, its pair away) is
    // waited for here, nothing committed, its process waiting and not in
    // flight; then it goes again, queued. Committed with one, it waited
    // Restoring on a rebind nothing made (Mac Pro at 2 GiB on 1e87ab171).
    if let Some(store) = stores.roles.as_ref() {
        let watch = store.lock().unwrap().room_watch();
        let unbound = store.lock().unwrap().unbound_after(pid, roles.as_ref());
        if !unbound.is_empty() {
            abort(model, engine, Move::Restore { kv, rs, roles });
            tracing::debug!(%pid, ?unbound, "planner: a restore waits, uncommitted, for its rebinds' parts");
            planner.rebind_waiting(pid, Some(format!("{unbound:?} (uncommitted)")));
            // Its pair on its way with another holder's move: woken by that
            // move's end alone, as a restore refused for it is. Woken by
            // room given back, two restores waiting on one pair woke each
            // other with their own aborts, round after round, as long as
            // the move took (the simulation with a bind's take and its engine
            // bind apart, seed 33: 43,051 rounds in one step).
            let moving = store.lock().unwrap().moving_pair_of(&unbound);
            match moving {
                Some(pair) => {
                    let mut watch = watch;
                    loop {
                        let next = store.lock().unwrap().room_watch();
                        if store.lock().unwrap().pair_moving(pair).is_none() {
                            break;
                        }
                        watch.freed().await;
                        watch = next;
                    }
                }
                None => watch.freed().await,
            }
            planner.rebind_waiting(pid, None);
            planner.requeue_restore(pid);
            return;
        }
    }
    // Suspended by a bind short of guest-state room, it comes back only
    // with room for that bind past what its own items took: else it would
    // bind short again at once. It waits, queued, as a restore short of
    // room for its roles.
    // Refused so, it says why, and its need is the bind's runs beside what
    // its own unpark lays at that length: read as the larger of the two
    // (bind (7, 2), unpark (7, 2): (7, 2)), the planner's placement stood
    // it while this check, after the unpark took its own, did not, and the
    // head was refused with no reason and no plan, round after round (Mac
    // Pro at 5 GiB on 4cef0c7d3: 40,098 no-step lines in 6 minutes, free
    // runs (7, 3) to (7, 4), need [(7, 2), (2, 2)]).
    // Read by the one placement, the takes' own, with what its serve held
    // for it open to it: never a count of its own.
    if let Some(need) = planner.pending_bind(pid)
        && let Some(free) = stores
            .roles
            .as_ref()
            .map(|roles| {
                roles.lock().unwrap().place_claimed(
                    pid,
                    &[crate::store::slots::OwnPiece::Chunks {
                        run: need.run,
                        pieces: need.pieces,
                    }],
                )
            })
            .filter(|placement| !placement.stands())
            .map(|placement| placement.free_runs(need.run))
    {
        abort(model, engine, Move::Restore { kv, rs, roles });
        let own = stores.roles.as_ref().map_or(0, |roles| {
            roles
                .lock()
                .unwrap()
                .unpark_need(pid)
                .iter()
                .filter(|&&(run, _)| run == need.run)
                .map(|&(_, pieces)| pieces)
                .sum::<u32>()
        });
        let why = format!(
            "PendingBindShort {{ run: {}, pieces: {}, free_runs_after_own_unpark: {free}, own_unpark_pieces: {own} }}",
            need.run, need.pieces
        );
        tracing::debug!(%pid, %why, "planner: a restore waits for room for its pending bind");
        planner.note_roles_short_why(pid, why);
        // The bind's own runs: its unpark's were placed first by the one
        // placement, so they are never added again (added each refusal,
        // the need grew 1, 3, 5, 7 until it fit nowhere: the simulation's
        // seed 13, 455 of 5,000 seeds).
        let _ = own;
        planner.requeue_restore_roles(pid, (need.run, need.pieces), Some(free));
        return;
    }
    let mut guard = MoveGuard::new(model, engine, Move::Restore { kv, rs, roles });
    if let Err(error) = guard.copy(&planner, pid).await {
        // Taken back before anything reads where it was: its state is
        // still below, where the copy read it from.
        guard.take_back().await;
        if error.landing_failed {
            // A copy did not land: what it wrote is not to be trusted, and
            // its sources still hold the state. The restore goes again
            // from them.
            tracing::warn!(pid = %pid, %error, "planner: a restore's copy did not land: tried again from its source");
            planner.requeue_restore(pid);
            return;
        }
        planner.restore_faulted(
            pid,
            &format!("its state could not be copied back to the device: {error}"),
        );
        return;
    }
    let Move::Restore { kv, rs, roles } = guard.take() else {
        unreachable!("a restore carries a restore")
    };
    if let Some(txn) = kv {
        let restored = crate::store::registry::with_kv_lock(&stores.kv, "planner-restore", |kv| {
            kv.commit_restore(txn)
        });
        if let Err(error) = restored {
            abort(
                model,
                engine,
                Move::Restore {
                    kv: None,
                    rs,
                    roles,
                },
            );
            planner.restore_failed(pid, &error.to_string());
            return;
        }
    }
    if let Some(txn) = rs {
        stores.rs.lock().unwrap().commit_restore(txn);
    }
    if let Some(store) = &stores.roles {
        if let Some(txn) = roles {
            store.lock().unwrap().commit(txn);
        }
        if let Err(error) = rebind(engine, store, pid).await {
            planner.restore_failed(pid, &error);
            return;
        }
        // An instance its rebinds could not name yet (its pair still on its
        // way back with another process's restore) is bound once that move
        // lands: the process is not resident until then, so no fire of it
        // runs on the places it gave back at its park. Each change of the
        // roles (a move committed or aborted) looks again.
        loop {
            let watch = store.lock().unwrap().room_watch();
            // Its restore is over once it is no longer restoring (parked
            // again, failed or gone meanwhile): nothing of it is rebound
            // or said restored from here.
            if planner.with_inner(|inner| {
                inner
                    .procs
                    .get(&pid)
                    .is_none_or(|proc| proc.state != super::proc::Residency::Restoring)
            }) {
                tracing::warn!(%pid, "planner: a restore waiting on its rebinds ended: its process is no longer restoring");
                return;
            }
            let unbound = store.lock().unwrap().unbound(pid);
            if unbound.is_empty() {
                break;
            }
            tracing::debug!(%pid, ?unbound, "planner: a restore waits for its instances' rebinds");
            planner.rebind_waiting(pid, Some(format!("{unbound:?}")));
            watch.freed().await;
            if let Err(error) = rebind(engine, store, pid).await {
                planner.restore_failed(pid, &error);
                return;
            }
        }
    }
    planner.rebind_waiting(pid, None);
    ptrace!(
        "restore-done pid={} ms={}",
        pid,
        began.elapsed().as_millis()
    );
    planner.report_restored(pid);
    crate::scheduler::nudge(engine);
}

/// Persists the sealed run `keys` (from token 0) on the disk tier of
/// `engine`: each resident page not on disk yet is copied to a fresh slot
/// and stays where it is, then the journal names it. Answers the pages
/// persisted; none when the tier is not kept, is full, or the run is not
/// held here.
pub(super) async fn persist(
    model: usize,
    engine: usize,
    keys: Vec<crate::store::kv::hash::Hash256>,
    retention: i64,
) -> Result<usize, String> {
    let stores = crate::store::registry::get(model, engine);
    let Some(txn) = crate::store::registry::with_kv_lock(&stores.kv, "persist", |kv| {
        kv.prepare_persist(&keys, retention)
    }) else {
        return Ok(0);
    };
    let (device, slots) = txn.copy_plan();
    let copy = TierMove {
        kind: TierKind::kv(txn.shape()),
        tier: MemoryDomain::LocalDisk,
        out: true,
        device,
        slots,
        source: None,
    };
    let mut guard = PersistGuard {
        model,
        engine,
        txn: Some(txn),
        in_flight: None,
    };
    let started = Instant::now();
    let completion = crate::scheduler::copy_tier_tracked(engine, &copy)
        .map_err(|error| format!("{} submit: {error:#}", copy.label()))?;
    guard.in_flight = Some(completion.clone());
    let copied = completion.wait().await;
    guard.in_flight = None;
    if let Some(planner) = crate::planner::planner_for(model, engine) {
        planner.record_tier_copy(&copy, started.elapsed());
    }
    copied.map_err(|error| format!("{} copy: {error}", copy.label()))?;
    let txn = guard.txn.take().expect("persist present");
    Ok(crate::store::registry::with_kv_lock(
        &stores.kv,
        "persist",
        |kv| kv.commit_persist(txn),
    ))
}

/// Gives back a persist's slots when it is dropped before it commits,
/// after any copy in flight.
struct PersistGuard {
    model: usize,
    engine: usize,
    txn: Option<crate::store::kv::KvPersistTxn>,
    in_flight: Option<crate::scheduler::ControlCompletion>,
}

impl Drop for PersistGuard {
    fn drop(&mut self) {
        let Some(txn) = self.txn.take() else {
            return;
        };
        let (model, engine) = (self.model, self.engine);
        let abort = move || {
            let stores = crate::store::registry::get(model, engine);
            crate::store::registry::with_kv_lock(&stores.kv, "persist-abort", |kv| {
                kv.abort_persist(txn)
            });
        };
        match self.in_flight.take() {
            None => abort(),
            Some(completion) if crate::rt::has_runtime() => {
                crate::rt::spawn(async move {
                    let _ = completion.wait().await;
                    abort();
                });
            }
            Some(_) => tracing::error!(
                model,
                engine,
                "persist dropped with an engine copy in flight and no runtime; \
                 keeping its slots to avoid reuse during the copy"
            ),
        }
    }
}

#[cfg(test)]
mod abandon_tests {
    use super::*;
    use crate::store::roles::tests::{FakeEngine, PROGRAM, holder, pools};
    use crate::store::roles::{RolesError, bind_instance, learn_program};

    /// A restore's move whose copies the engine cannot settle (its
    /// `settle_copies` fails): the dropped move takes the abandon path. It
    /// ends, each item where it lay; the slots it took are leaked by name and
    /// counted in the planner's status; and both holders' restores go after
    /// it, the planner serving on.
    #[tokio::test(flavor = "current_thread")]
    async fn a_move_the_engine_cannot_settle_ends_its_slots_leaked_and_counted() {
        let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
        let pools = pools();
        let model = crate::store::registry::register_model_with_chunks(
            16,
            &[crate::store::registry::EngineStores {
                shared: Some(pools.device.clone()),
                tiers: crate::store::shared::Tiers {
                    host: Some(pools.host.clone()),
                    disk_slots: 0,
                    disk_chunk_slots: 0,
                },
                max_context: 256,
                ..Default::default()
            }],
        );
        let planner = super::super::ResidencyPlanner::new(Arc::new(
            super::super::pool::RegistryPool::new(model, 0, true),
        ));
        super::super::global::init_planner(model, 0, planner);
        let roles = crate::store::registry::get(model, 0).roles.expect("roles");
        let mut engine = FakeEngine::default();
        learn_program(&roles, &mut engine, PROGRAM).unwrap();
        let mut ids = Vec::new();
        for by in [1, 2] {
            let binding = ::engine::InstanceBinding {
                program: PROGRAM,
                ..::engine::InstanceBinding::default()
            };
            ids.push(
                bind_instance(Some(&roles), &mut engine, &binding, Some(holder(by)), |_| false)
                    .unwrap()
                    .id,
            );
        }
        for by in [1, 2] {
            let park = roles.lock().unwrap().prepare_park(holder(by)).unwrap();
            roles.lock().unwrap().commit(park);
        }
        // The first's restore: its copies handed to the engine, never settled.
        let unpark = roles
            .lock()
            .unwrap()
            .prepare_unpark(holder(1))
            .unwrap()
            .expect("it unparks");
        let txn = Move::Restore {
            kv: None,
            rs: None,
            roles: Some(unpark),
        };
        settle_or_abandon(model, 0, Vec::new(), async {
            Err(anyhow::anyhow!("the device was lost"))
        }, txn)
        .await;
        let leaked = roles.lock().unwrap().leaked();
        assert!(leaked > 0, "the slots it took, leaked");
        let planner = super::super::global::planner_for(model, 0).expect("the planner");
        assert_eq!(
            planner.stats.leaked_role_slots.load(std::sync::atomic::Ordering::Relaxed),
            leaked,
            "counted in the status"
        );
        assert_eq!(planner.diagnostics().leaked_role_slots_total, leaked);
        // The move ended: both restores go.
        for by in [1, 2] {
            let back = roles.lock().unwrap().prepare_unpark(holder(by));
            assert!(
                !matches!(back, Err(RolesError::PairParked(_))),
                "nothing moves now"
            );
            let back = back.expect("it unparks").expect("it comes back");
            roles.lock().unwrap().commit(back);
        }
        for id in ids {
            roles.lock().unwrap().close_instance(id);
        }
        assert_eq!(roles.lock().unwrap().leaked(), leaked, "still out of service");
    }
}
