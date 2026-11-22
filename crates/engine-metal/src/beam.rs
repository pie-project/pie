//! Beam groups' page lists (`state-geometry.parents`,
//! `new-memory/designs/BEAM-PAGE-LISTS.md`): planned on the host from what
//! the host knows (each group's working set's reserve and frozen slots, the
//! slot its beams write, its lanes' staged tables), run on the device from
//! the group's resident state and the parents its last epilogue wrote
//! (`kernels_metal::runtime::beam`: `beam_step`, then `beam_copy_pages`),
//! in the fire's own command buffer ahead of its first append or read.
//!
//! The host stages each beam's page table as for any envelope lane; the
//! group's step reorders its lists by the parents (or primes them on its
//! first fire) and overwrites each beam's entries past `f0` and its rows'
//! write pages with the pages its list names, bounds-checked against the
//! group's pool and frozen slots; the copies it leaves in its state run
//! next. A lane out of bounds keeps the staged words (its own set's pages)
//! and is faulted by name at settle. The Metal port of engine-cuda's
//! `serve::pages`.

use crate::error::{Fault, Result};
use crate::inputs::{BeamAddresses, BeamRecords};
use crate::program::BeamLayout;
use kernels_metal::runtime::beam::{
    self as kbeam, BeamListRun, BeamRsRun, BeamStepRun, PageCopyRun, RsCopyRun,
};

/// A beam group of a fire: one instance's lanes.
#[derive(Clone, Debug)]
pub struct Group {
    pub instance: engine::InstanceId,
    /// Its parents cell and its state (GPU addresses), and its layout.
    pub parents: u64,
    pub state: u64,
    pub layout: BeamLayout,
    /// Its lanes in beam order.
    pub lanes: Vec<GroupLane>,
}

/// A beam's lane: its index in the fire, its seat in the staged tables,
/// its first row, the slots it held before the fire and the rows it lands.
#[derive(Clone, Copy, Debug)]
pub struct GroupLane {
    pub source: usize,
    /// Its seat in the staged tables (the composition's order).
    pub seat: usize,
    pub row_offset: u32,
    pub have: u32,
    pub rows: u32,
}

/// The pool pages a group of `lanes` beams can name at the fire whose
/// beams write `slot` (BEAM-PAGE-LISTS §3): the step hands out the lowest
/// free pages, so every page named or pending lies below
/// `B (c + 2) + B F + 1`. Only those are backed and staged.
#[must_use]
pub fn backed(lanes: u32, slot: u32, folded: u32, page: u32, frames: u32, reserve: u32) -> u32 {
    let page = page.max(1);
    let column = (slot / page).saturating_sub(folded / page);
    lanes
        .saturating_mul(column.saturating_add(2))
        .saturating_add(lanes.saturating_mul(frames.max(1)))
        .saturating_add(1)
        .min(reserve)
}

/// Why a lane's fault word faults it (BEAM-PAGE-LISTS §7).
#[must_use]
pub fn fault_reason(code: u32, lane: usize) -> String {
    let why = match code {
        kbeam::PAGES_PAST_RESERVE => "names a page past its working set's pool",
        kbeam::PAGES_SHORT => "is shorter than the pages the lane reads",
        kbeam::PAGES_BELOW_FLOOR => {
            "would land a row below its working set's frozen slots or past its reserve"
        }
        kbeam::PAGES_COPY_BOUNDS => {
            "states a page copy past its working set's pool, longer than a page, or into frozen \
             slots"
        }
        kbeam::PAGES_PARENT => "continues a parent past its beams",
        kbeam::PAGES_EXHAUSTED => {
            tracing::error!(
                lane,
                "engine-metal: BUG: a beam group found fewer free pages than it hands out; its \
                 pool is sized exactly (BEAM-PAGE-LISTS.md §3)"
            );
            "found no free page of its working set's pool left (a bug: the pool is sized exactly)"
        }
        _ => "faulted",
    };
    format!(
        "Overreach: lane {lane}'s page list {why}; nothing was written outside its own working \
         set, and the lane is failed"
    )
}

/// What [`plan`] reads of the fire as staged.
pub struct Staged<'a> {
    /// Per staged table: its lanes' `indptr` (a seat a lane, in lane order).
    pub indptr: Vec<&'a [i32]>,
    /// Per staged table: the GPU addresses of its indices, write pages and
    /// write offsets.
    pub tables: Vec<(u64, u64, u64)>,
    /// Whether any staged table is a windowed one (refused).
    pub windowed: bool,
    pub page: u32,
    pub addresses: BeamAddresses,
    pub planes: Vec<kbeam::PagePlane>,
}

/// Plans `groups` over the fire as `staged`: `translation(source)` a lane's
/// working set's whole table (slot refs) and `folded(source)` its frozen
/// slots; `pool(refs)` backs the pool pages a fire may name and gives their
/// kernel page ids and base addresses. Returns the records and, per fault
/// word, the lane and instance it faults.
pub fn plan(
    groups: &[Group],
    staged: &Staged<'_>,
    translation: &dyn Fn(usize) -> Vec<u32>,
    folded: &dyn Fn(usize) -> u32,
    pool: &mut dyn FnMut(&[u32]) -> Result<(Vec<u32>, Vec<u64>)>,
) -> Result<(BeamRecords, Vec<(usize, engine::InstanceId)>)> {
    let mut records = BeamRecords::default();
    let mut faults: Vec<(usize, engine::InstanceId)> = Vec::new();
    if groups.is_empty() {
        return Ok((records, faults));
    }
    if staged.windowed {
        return Err(Fault::program(
            "beam",
            "a beam group's page lists are resolved over full-context tables only, and this \
             load lays windowed ones",
        ));
    }
    let page = staged.page.max(1);
    records.planes.clone_from(&staged.planes);
    for group in groups {
        let first = group.lanes.first().ok_or_else(|| {
            Fault::program("beam", "a beam group of no lanes".to_string())
        })?;
        if group.layout.lanes > kbeam::MAX_LANES {
            return Err(Fault::program(
                "beam",
                format!(
                    "instance {}'s beam group runs {} beams and the head step holds at most {}",
                    group.instance,
                    group.layout.lanes,
                    kbeam::MAX_LANES
                ),
            ));
        }
        if group.lanes.len() as u32 != group.layout.lanes {
            return Err(Fault::program(
                "beam",
                format!(
                    "instance {}'s beam group binds {} lane(s) in this fire and states parents \
                     for {}; every beam runs every fire",
                    group.instance,
                    group.lanes.len(),
                    group.layout.lanes
                ),
            ));
        }
        let slot = first.have;
        if group.lanes.iter().any(|lane| lane.have != slot || lane.rows != 1) {
            return Err(Fault::program(
                "beam",
                format!(
                    "instance {}'s beams write different slots or more than one row; every beam \
                     writes its depth, one row a fire",
                    group.instance
                ),
            ));
        }
        let table_pages = translation(first.source);
        let folded = folded(first.source);
        let f0 = folded / page;
        let whole = table_pages.get(f0 as usize..).unwrap_or(&[]);
        let reserve = whole.len() as u32;
        if reserve > group.layout.cap {
            return Err(Fault::program(
                "beam",
                format!(
                    "instance {}'s working set reserves {reserve} pool page(s) past its frozen \
                     ones and its beam state is laid for {}",
                    group.instance, group.layout.cap
                ),
            ));
        }
        let backed = backed(group.layout.lanes, slot, folded, page, group.layout.frames, reserve);
        let (ids, bases) = pool(&whole[..backed as usize])?;
        let fault_at = faults.len() as u32;
        for lane in &group.lanes {
            faults.push((lane.source, group.instance));
        }
        let trans_at = records.trans.len() as u32;
        records.trans.extend(ids.iter().map(|&id| id as i32));
        let lists_at = records.lists.len() as u32;
        for (table, indptr) in staged.indptr.iter().enumerate() {
            let (indices, write_page, write_offset) = staged.tables[table];
            for (beam, lane) in group.lanes.iter().enumerate() {
                let from = *indptr.get(lane.seat).ok_or_else(|| {
                    Fault::program("beam", format!("lane {} has no seat", lane.source))
                })? as u32;
                let to = indptr.get(lane.seat + 1).copied().unwrap_or(0) as u32;
                let count = to.saturating_sub(from);
                records.lists.push(BeamListRun {
                    indices: indices + u64::from(from + f0.min(count)) * 4,
                    write_page: write_page + u64::from(lane.row_offset) * 4,
                    write_offset: write_offset + u64::from(lane.row_offset) * 4,
                    lane: beam as u32,
                    entries: count.saturating_sub(f0),
                    rows: lane.rows,
                    pad: 0,
                });
            }
        }
        let list_count = records.lists.len() as u32 - lists_at;
        let list_bytes = std::mem::size_of::<BeamListRun>() as u64;
        records.groups.push(BeamStepRun {
            state: group.state,
            parents: group.parents,
            trans: staged.addresses.trans + u64::from(trans_at) * 4,
            lists: staged.addresses.lists + u64::from(lists_at) * list_bytes,
            faults: staged.addresses.faults + u64::from(fault_at) * 4,
            lanes: group.layout.lanes,
            cap: group.layout.cap,
            reserve,
            page,
            folded,
            slot,
            list_count,
            frames: group.layout.frames,
            backed,
            rs_count: 0,
            rs: 0,
        });
        let bases_at = records.bases.len() as u32;
        records.bases.extend_from_slice(&bases);
        let copies = kbeam::beam_copies_word(
            group.layout.lanes,
            group.layout.cap,
            group.layout.frames,
        );
        for beam in 0..group.layout.lanes {
            records.copies.push(PageCopyRun {
                copy: group.state + (copies + 3 * u64::from(beam)) * 4,
                base: staged.addresses.bases + u64::from(bases_at) * 8,
                fault: staged.addresses.faults + u64::from(fault_at + beam) * 4,
                reserve: backed,
                page,
                floor_rel: folded - f0 * page,
                plane_at: 0,
                planes: records.planes.len() as u32,
                pad: 0,
            });
        }
    }
    records.lanes = faults.len() as u32;
    Ok((records, faults))
}

/// What [`plan_rs`] reads of a hybrid fire's rs buffer (BEAM-PAGE-LISTS
/// §11): each seat's buffer base and its set's rs pages (slab slots, every
/// one of its buffer), each seat's lane plan, the fire's rs page table (its
/// address and a row's ints), the rows a page holds, an rs page's bytes,
/// the slab its page ids address and each plane's `(offset, cell)`.
pub struct RsStaged<'a> {
    pub sets: &'a [Option<(u32, Vec<u32>)>],
    pub plans: &'a [engine::rs::LanePlan],
    pub table: u64,
    pub ints: u32,
    pub page_tokens: u32,
    pub page_bytes: u64,
    pub slab: u64,
    pub planes: Vec<kbeam::PagePlane>,
}

/// A hybrid's beams over the records [`plan`] laid (BEAM-PAGE-LISTS §11):
/// each group's set's rs entries, each beam's rs page table to resolve over
/// its list (its row of the fire's rs page table, from the buffer entry the
/// row's first slot holds) and each beam's rs copy over every rs plane. A
/// group whose lanes buffer no rows is left as it is. Refused by name where
/// a page's rows do not divide a kv page, or where the buffer does not
/// start on a row page (an entry would straddle two of a beam's pages).
pub fn plan_rs(
    records: &mut BeamRecords,
    groups: &[Group],
    rs: &RsStaged<'_>,
    addresses: &BeamAddresses,
) -> Result<()> {
    let tokens = rs.page_tokens.max(1);
    let plane_at = records.planes.len() as u32;
    records.planes.extend_from_slice(&rs.planes);
    let run_bytes = std::mem::size_of::<BeamRsRun>() as u64;
    for (group, step) in groups.iter().zip(records.groups.iter_mut()) {
        let first = group.lanes.first().ok_or_else(|| {
            Fault::program("beam", "a beam group of no lanes".to_string())
        })?;
        let Some(Some((base, set))) = rs.sets.get(first.seat) else {
            continue;
        };
        let (base, page) = (*base, step.page.max(1));
        if page % tokens != 0 || base % tokens != 0 {
            return Err(Fault::program(
                "beam",
                format!(
                    "instance {}'s beams buffer recurrent rows {tokens} a page from slot {base}, \
                     and their page lists page kv slots {page} a page: a hybrid's beams need \
                     the rs page's rows to divide the kv page's and its buffer to start on a \
                     row page (BEAM-PAGE-LISTS.md §11)",
                    group.instance
                ),
            ));
        }
        if step.slot < base {
            continue;
        }
        let bases_at = records.rs_bases.len() as u32;
        records.rs_bases.extend(set.iter().map(|&slot| slot as i32));
        let bases_len = set.len() as u32;
        let entries = (step.slot - base) / tokens + 1;
        let rs_at = records.rs.len() as u32;
        for (beam, lane) in group.lanes.iter().enumerate() {
            let plan = rs.plans.get(lane.seat).ok_or_else(|| {
                Fault::program("beam", format!("lane {} has no rs plan", lane.source))
            })?;
            // The row's first slot holds the buffer page of its first
            // extended row (`rs::page_table`).
            let Some(run) = plan.gather.as_ref().or(plan.scatter.as_ref()) else {
                return Err(Fault::program(
                    "beam",
                    format!(
                        "instance {}'s beam {beam} buffers no recurrent rows where beam 0 does",
                        group.instance
                    ),
                ));
            };
            let row = rs.table + (u64::from(lane.seat as u32) * u64::from(rs.ints) + 1) * 4;
            records.rs.push(BeamRsRun {
                pages: row,
                bases: addresses.rs_bases + u64::from(bases_at) * 4,
                lane: beam as u32,
                first: run.from / tokens,
                entries,
                base,
                tokens,
                bases_len,
            });
        }
        step.rs_count = records.rs.len() as u32 - rs_at;
        step.rs = addresses.rs + u64::from(rs_at) * run_bytes;
        let f0 = step.folded / page;
        let copies = kbeam::beam_copies_word(step.lanes, step.cap, step.frames);
        let fault_at = (step.faults - addresses.faults) / 4;
        for beam in 0..step.lanes {
            records.rs_copies.push(RsCopyRun {
                copy: step.state + (copies + 3 * u64::from(beam)) * 4,
                bases: addresses.rs_bases + u64::from(bases_at) * 4,
                fault: addresses.faults + (fault_at + u64::from(beam)) * 4,
                slab: rs.slab,
                reserve: step.backed,
                page,
                tokens,
                base,
                f0,
                floor_rel: step.folded - f0 * page,
                plane_at,
                planes: rs.planes.len() as u32,
                bases_len,
                page_bytes: u32::try_from(rs.page_bytes).map_err(|_| {
                    Fault::program("beam", "an rs page past 4 GiB".to_string())
                })?,
            });
        }
    }
    Ok(())
}
