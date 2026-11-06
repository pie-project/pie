use std::collections::{HashMap, HashSet};
use std::sync::{LazyLock, Mutex, RwLock, Weak};

use crate::pipeline::fire::{PendingFireQueue, PendingFires};
use crate::store::kv::page_table::{ReclaimQuote, WorkingSetId};
use crate::store::kv::working_set::KvSuspendHandle;
use crate::store::rs::RsWorkingSetId;

type WeakPendingFires = Weak<PendingFireQueue>;

pub(crate) struct ResidentPipeline {
    pub(crate) scope: crate::store::PipelineScope,
    pub(crate) fires: WeakPendingFires,
}

#[derive(Default)]
pub(crate) struct ProcessResidency {
    pub(crate) kv_working_sets:
        HashMap<(usize, crate::engine::EngineId, WorkingSetId), KvSuspendHandle>,
    pub(crate) rs_working_sets: HashSet<(usize, crate::engine::EngineId, RsWorkingSetId)>,
    pub(crate) pipelines: Vec<ResidentPipeline>,
}

#[derive(Clone)]
pub(crate) struct ResidencySnapshot {
    pub pipelines: Vec<PendingFires>,
    pub departed_pipeline_ids: Vec<uuid::Uuid>,
}

impl ProcessResidency {
    pub(crate) fn pipelines(&mut self) -> Vec<PendingFires> {
        let pipelines: Vec<_> = self
            .pipelines
            .iter()
            .filter_map(|pipeline| pipeline.fires.upgrade())
            .collect();
        self.pipelines
            .retain(|pipeline| pipeline.fires.strong_count() > 0);
        pipelines
    }

    pub(crate) fn teardown_snapshot(&mut self) -> ResidencySnapshot {
        let departed_pipeline_ids = self
            .pipelines
            .iter()
            .filter(|pipeline| pipeline.scope.close())
            .map(|pipeline| pipeline.scope.scheduler_id())
            .collect();
        ResidencySnapshot {
            pipelines: self.pipelines(),
            departed_pipeline_ids,
        }
    }
}

static RESIDENCIES: LazyLock<RwLock<HashMap<uuid::Uuid, Weak<Mutex<ProcessResidency>>>>> =
    LazyLock::new(Default::default);

pub(crate) fn register_residency(pid: uuid::Uuid, residency: Weak<Mutex<ProcessResidency>>) {
    RESIDENCIES.write().unwrap().insert(pid, residency);
}

pub(crate) fn unregister_residency(pid: uuid::Uuid) {
    RESIDENCIES.write().unwrap().remove(&pid);
}

fn with_residency<R: Default>(pid: uuid::Uuid, f: impl FnOnce(&mut ProcessResidency) -> R) -> R {
    let residency = {
        let residencies = RESIDENCIES.read().unwrap();
        residencies.get(&pid).and_then(Weak::upgrade)
    };
    match residency {
        Some(residency) => {
            let mut residency = residency.lock().unwrap();
            f(&mut residency)
        }
        None => R::default(),
    }
}

pub(crate) fn kv_working_set_ids(
    pid: uuid::Uuid,
    model: usize,
    engine: usize,
) -> HashSet<WorkingSetId> {
    with_residency(pid, |residency| {
        residency
            .kv_working_sets
            .keys()
            .filter_map(|&(m, d, ws)| (m == model && d == engine).then_some(ws))
            .collect()
    })
}

pub(crate) fn rs_working_set_ids(
    pid: uuid::Uuid,
    model: usize,
    engine: usize,
) -> HashSet<RsWorkingSetId> {
    with_residency(pid, |residency| {
        residency
            .rs_working_sets
            .iter()
            .filter_map(|&(m, d, ws)| (m == model && d == engine).then_some(ws))
            .collect()
    })
}

pub(crate) fn kv_suspend_handles(
    pid: uuid::Uuid,
    model: usize,
    engine: usize,
) -> Vec<KvSuspendHandle> {
    with_residency(pid, |residency| {
        residency
            .kv_working_sets
            .iter()
            .filter(|((m, d, _), _)| *m == model && *d == engine)
            .map(|(_, handle)| handle.clone())
            .collect()
    })
}

pub(crate) fn pipelines_of(pid: uuid::Uuid) -> Vec<PendingFires> {
    with_residency(pid, |residency| residency.pipelines())
}

pub(crate) fn kv_lease_quiescent(pid: uuid::Uuid, model: usize, engine: usize) -> bool {
    with_residency(pid, |residency| {
        residency
            .kv_working_sets
            .iter()
            .filter(|((m, d, _), _)| *m == model && *d == engine)
            .all(|(_, handle)| handle.active_leases() == 0)
    })
}

pub(crate) fn kv_working_sets_for(
    pids: &[uuid::Uuid],
    model: usize,
    engine: usize,
) -> Vec<Option<HashSet<WorkingSetId>>> {
    let residencies = RESIDENCIES.read().unwrap();
    pids.iter()
        .map(|pid| {
            let residency = residencies.get(pid)?.upgrade()?;
            let residency = residency.lock().unwrap();
            Some(
                residency
                    .kv_working_sets
                    .keys()
                    .filter_map(|&(m, d, ws)| (m == model && d == engine).then_some(ws))
                    .collect(),
            )
        })
        .collect()
}

pub(crate) fn quote_locked(
    kv: &crate::store::kv::KvStore,
    working_sets: Vec<Option<HashSet<WorkingSetId>>>,
    budget: u32,
) -> Vec<Option<ReclaimQuote>> {
    let known: Vec<HashSet<WorkingSetId>> = working_sets.iter().flatten().cloned().collect();
    let mut quotes = kv.reclaim_quotes(&known, budget).into_iter();
    working_sets
        .into_iter()
        .map(|entry| entry.and_then(|_| quotes.next()))
        .collect()
}

pub(crate) fn kv_reclaim_quotes(
    pids: &[uuid::Uuid],
    model: usize,
    engine: usize,
    budget: u32,
) -> Vec<Option<ReclaimQuote>> {
    let working_sets = kv_working_sets_for(pids, model, engine);
    let Some(stores) = crate::store::registry::try_get(model, engine) else {
        return vec![None; pids.len()];
    };
    crate::store::registry::with_kv_lock(&stores.kv, "planner-quotes", |kv| {
        quote_locked(kv, working_sets, budget)
    })
}
