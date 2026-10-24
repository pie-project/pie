use std::collections::HashSet;

use crate::store::rs::RsUnits;

use super::ProcessId;

pub(super) fn swapped_page_count(pid: ProcessId, model: usize, engine: usize) -> u32 {
    kv_page_count(pid, model, engine, |kv, ws| {
        kv.swapped_page_count(ws).unwrap_or(0)
    })
}

pub(super) fn held_page_count(pid: ProcessId, model: usize, engine: usize) -> u32 {
    kv_page_count(pid, model, engine, |kv, ws| {
        kv.held_page_count(ws).unwrap_or(0)
    })
}

pub(super) fn held_rs_units(pid: ProcessId, model: usize, engine: usize) -> RsUnits {
    rs_units(pid, model, engine, |rs, ws| rs.held(ws.iter().copied()))
}

pub(super) fn swapped_rs_units(pid: ProcessId, model: usize, engine: usize) -> RsUnits {
    rs_units(pid, model, engine, |rs, ws| rs.swapped_units(ws))
}

pub(super) fn suspendable_rs_units(pid: ProcessId, model: usize, engine: usize) -> RsUnits {
    rs_units(pid, model, engine, |rs, ws| rs.suspendable(ws))
}

fn rs_units(
    pid: ProcessId,
    model: usize,
    engine: usize,
    count: impl FnOnce(
        &crate::store::rs::RsStore,
        &HashSet<crate::store::rs::RsWorkingSetId>,
    ) -> RsUnits,
) -> RsUnits {
    let working_sets = crate::inferlet::process::residency::rs_working_set_ids(pid, model, engine);
    if working_sets.is_empty() {
        return RsUnits::default();
    }
    let Some(stores) = crate::store::registry::try_get(model, engine) else {
        return RsUnits::default();
    };
    count(&stores.rs.lock().unwrap(), &working_sets)
}

fn kv_page_count(
    pid: ProcessId,
    model: usize,
    engine: usize,
    count: impl FnOnce(
        &crate::store::kv::KvStore,
        &std::collections::HashSet<crate::store::kv::page_table::WorkingSetId>,
    ) -> usize,
) -> u32 {
    let working_sets = crate::inferlet::process::residency::kv_working_set_ids(pid, model, engine);
    if working_sets.is_empty() {
        return 0;
    }
    let Some(stores) = crate::store::registry::try_get(model, engine) else {
        return 0;
    };
    crate::store::registry::with_kv_lock(&stores.kv, "planner-held-pages", |kv| {
        count(kv, &working_sets) as u32
    })
}
