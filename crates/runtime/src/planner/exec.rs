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
            for (shape, pairs) in &plan.shared {
                let (device, slots) = pairs.iter().copied().unzip();
                copies.push(TierMove {
                    kind: TierKind::RsSlots(*shape),
                    tier: MemoryDomain::HostPinned,
                    out,
                    device,
                    slots,
                    source: None,
                });
            }
        }
        // Instance tables and rings park with their process (never
        // dropped: no token rebuilds them).
        for copy in roles.iter().flat_map(RoleMove::copies) {
            copies.push(TierMove {
                kind: TierKind::RoleSlots(copy.shape),
                tier: MemoryDomain::HostPinned,
                out,
                device: copy.device.clone(),
                slots: copy.host.clone(),
                source: None,
            });
        }
        copies
    }
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
    async fn copy(&mut self, planner: &ResidencyPlanner, pid: ProcessId) -> Result<(), String> {
        let mut waits = Vec::new();
        for copy in self.txn.as_ref().expect("move present").copies() {
            tracing::debug!(pid = %pid, copy = %copy.label(), count = copy.device.len(), "planner: residency copy");
            let started = Instant::now();
            let completion = crate::scheduler::copy_tier_tracked(self.engine, &copy)
                .map_err(|error| format!("{} submit: {error:#}", copy.label()))?;
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
            copied.map_err(|error| format!("{} copy: {error}", copy.label()))?;
        }
        self.in_flight.clear();
        Ok(())
    }

    fn take(&mut self) -> Move {
        self.txn.take().expect("move present")
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
            if let Err(error) =
                crate::scheduler::abort_once_landed(in_flight, settled, || abort(model, engine, txn))
                    .await
            {
                tracing::error!(
                    model,
                    engine,
                    %error,
                    "a residency move's copies did not settle; preserving its pages and \
                     slots to avoid reuse under a copy"
                );
                return;
            }
            if let Some(planner) = crate::planner::planner_for(model, engine) {
                planner.pages_freed();
            }
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
    let (model, engine) = planner.locus();
    let handles = crate::inferlet::process::residency::kv_suspend_handles(pid, model, engine);
    let working_sets: HashSet<WorkingSetId> =
        crate::inferlet::process::residency::kv_working_set_ids(pid, model, engine);
    let rs_working_sets =
        crate::inferlet::process::residency::rs_working_set_ids(pid, model, engine);
    // A process with no kv or state yet (a prompt not seated) may still hold
    // guest state on the device, the room a bind short of runs was chosen to
    // free (a chunk's occupant): it parks all the same, its guest scratch
    // given back. Found nothing, the M1 stress froze with eleven binds short
    // of runs and every chosen occupant "found no working set".
    let holds_guest = crate::store::registry::get(model, engine)
        .roles
        .as_ref()
        .is_some_and(|roles| roles.lock().unwrap().holds_guest_state(pid));
    if working_sets.is_empty() && rs_working_sets.is_empty() && !holds_guest {
        tracing::debug!(pid = %pid, "planner: eviction found no working set");
        planner.eviction_failed(pid);
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
    tracing::debug!(%pid, sets = fence.handles.len(), "planner: an eviction waits for its sets' fires");
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
    let prepared = crate::store::registry::with_kv_lock(&stores.kv, "planner-evict", |kv| {
        kv.prepare_suspend(&working_sets, crate::policy::retention::Installed)
    });
    let kv = match prepared {
        Ok(kv) => kv,
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
    let roles = stores
        .roles
        .as_ref()
        .and_then(|roles| roles.lock().unwrap().prepare_park(pid));
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
    if let Err(error) = guard.copy(&planner, pid).await {
        tracing::warn!(pid = %pid, %error, "planner: eviction copy failed");
        planner.eviction_copy_failed(pid, &error);
        return;
    }
    let copied = began.elapsed();
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

async fn move_out(
    planner: Arc<ResidencyPlanner>,
    mut guard: MoveGuard,
    label: &'static str,
    done: fn(&Arc<ResidencyPlanner>, bool),
) {
    let (model, engine) = planner.locus();
    if let Err(error) = guard.copy(&planner, ProcessId::nil()).await {
        tracing::warn!(%error, "planner: {label} copy failed");
        drop(guard);
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
        Ok(roles) => roles.flatten(),
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
                tracing::debug!(%pid, %error, "planner: a restore waits for room for its roles");
                planner.requeue_restore_roles(pid, need);
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
    // Suspended by a bind short of guest-state room, it comes back only
    // with room for that bind past what its own items took: else it would
    // bind short again at once. It waits, queued, as a restore short of
    // room for its roles.
    if let Some(need) = planner.pending_bind(pid)
        && stores
            .roles
            .as_ref()
            .is_some_and(|roles| roles.lock().unwrap().free_runs(need.run) < need.pieces)
    {
        abort(model, engine, Move::Restore { kv, rs, roles });
        tracing::debug!(%pid, "planner: a restore waits for room for its pending bind");
        planner.requeue_restore_roles(pid, (need.run, need.pieces));
        return;
    }
    let mut guard = MoveGuard::new(model, engine, Move::Restore { kv, rs, roles });
    if let Err(error) = guard.copy(&planner, pid).await {
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
    }
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
