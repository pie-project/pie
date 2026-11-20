//! Beam groups' page lists (`state-geometry.parents`,
//! `new-memory/designs/BEAM-PAGE-LISTS.md`): planned on the host from what
//! the host knows (each group's working set's reserve and frozen slots, the
//! slot its beams write, its lanes' staged tables), run on the device from
//! the group's resident state and the parents its last epilogue wrote
//! (`channel` `beam_step`, then `copy_pages`).
//!
//! The host stages each beam's page table as for any envelope lane; the
//! group's step then reorders its lists by the parents (or primes them on
//! its first fire), and overwrites each beam's entries past `f0` and its
//! rows' write pages with the pages its list names, each bounds-checked
//! against the group's pool and frozen slots; the copies it leaves in its
//! state run next, before the fire appends or reads. A lane out of bounds
//! keeps the staged words (its own set's pages) and is faulted by name at
//! settle.

use crate::error::{Fault, Result};
use kernels_cuda::runtime::channel::PagePlane;

/// A beam group of a fire: one instance's lanes, as `prepare` seated them.
#[derive(Clone, Debug)]
pub struct PageGroup {
    pub instance: engine::InstanceId,
    /// Its parents cell and its state (address, layout).
    pub parents: u64,
    pub state: u64,
    pub layout: crate::program::launch::BeamLayout,
    /// Its lanes in beam order: the lane's index in the fire, its seat in
    /// the staged page tables, its first row, the slots it held before the
    /// fire and the rows the fire lands.
    pub lanes: Vec<GroupLane>,
}

#[derive(Clone, Copy, Debug)]
pub struct GroupLane {
    pub source: usize,
    pub seat: usize,
    pub row_offset: u32,
    pub have: u32,
    pub rows: u32,
}

/// One group's step, as the device runs it.
#[derive(Clone, Copy, Debug)]
pub struct GroupPlan {
    pub state: u64,
    pub parents: u64,
    /// Where its pool's device ids start in [`PageWork::trans`].
    pub trans_at: u32,
    /// Where its lists start in [`PageWork::lists`], and how many.
    pub lists_at: u32,
    pub list_count: u32,
    /// Its first lane's fault word.
    pub fault_at: u32,
    pub lanes: u32,
    pub cap: u32,
    pub reserve: u32,
    pub page: u32,
    pub folded: u32,
    pub slot: u32,
    pub frames: u32,
    /// The pool pages staged from the first (`backed`).
    pub backed: u32,
    /// Where its beams' rs page tables start in [`PageWork::rs`], and how
    /// many (none off a hybrid).
    pub rs_at: u32,
    pub rs_count: u32,
}

/// The pool pages a group of `lanes` beams can name at the fire whose
/// beams write `slot` (`BEAM-PAGE-LISTS.md` §3): the step hands out the
/// lowest free pages, so every page named or pending lies below
/// `B (c + 2) + B F + 1`. Only those are seated and staged.
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

/// A beam's staged page table in one table.
#[derive(Clone, Copy, Debug)]
pub struct ListPlan {
    pub table: u32,
    pub lane: u32,
    /// Where its entry `f0` lies in the table's staged indices.
    pub indices_at: u32,
    pub write_row: u32,
    pub entries: u32,
    pub rows: u32,
}

/// A beam's copy over the planes a page holds: the kv planes over its
/// pool's kv pages, or on a hybrid the rs planes over its rs pages (§11).
#[derive(Clone, Copy, Debug)]
pub struct CopyPlan {
    pub fault: u32,
    /// Where its `[src, dst, tokens]` lies in its group's state.
    pub copy: u64,
    /// Where its group's pool's page addresses start in [`PageWork::bases`].
    pub bases_at: u32,
    pub reserve: u32,
    pub page: u32,
    pub floor_rel: u32,
    /// Its planes: `planes` of [`PageWork::planes`] from `plane_at`.
    pub plane_at: u32,
    pub planes: u32,
}

/// A beam's rs page table to resolve (a hybrid's, `BEAM-PAGE-LISTS.md`
/// §11): its column 0's entry in the fire's rs page table, its columns, and
/// where its group's rs pool pages' slot addresses start in
/// [`PageWork::bases`].
#[derive(Clone, Copy, Debug)]
pub struct RsListPlan {
    pub lane: u32,
    pub from: u32,
    pub entries: u32,
    pub bases_at: u32,
}

/// What the plan reads of a fire's rs buffer (a hybrid's): its lanes' rs
/// lanes by fire lane, the fire lane of each source lane, the slot address
/// each lane's table entry holds as staged, the rows a page holds, each
/// source lane's buffer base, and the planes a page's rows lie in.
pub struct RsFire<'a> {
    pub lanes: &'a [kernels_cuda::runtime::channel::RsLane],
    pub order: &'a [u32],
    pub slots: &'a [u64],
    pub page_tokens: u32,
    pub base: &'a dyn Fn(usize) -> Option<u32>,
    pub planes: Vec<PagePlane>,
}

/// What a fire's beam groups have the device run.
#[derive(Clone, Debug, Default)]
pub struct PageWork {
    pub groups: Vec<GroupPlan>,
    pub lists: Vec<ListPlan>,
    /// The beams' rs page tables (a hybrid's).
    pub rs: Vec<RsListPlan>,
    pub copies: Vec<CopyPlan>,
    pub trans: Vec<i32>,
    pub bases: Vec<u64>,
    pub planes: Vec<PagePlane>,
    /// Per fault word, the lane it faults.
    pub faults: Vec<(usize, engine::InstanceId)>,
}

impl PageWork {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }
}

/// Why a lane's fault word faults it (`BEAM-PAGE-LISTS.md` §7).
#[must_use]
pub fn fault_reason(code: u32, lane: usize) -> String {
    use kernels_cuda::runtime::channel::{
        PAGES_BELOW_FLOOR, PAGES_COPY_BOUNDS, PAGES_EXHAUSTED, PAGES_PARENT, PAGES_PAST_RESERVE,
        PAGES_SHORT,
    };
    let why = match code {
        PAGES_PAST_RESERVE => "names a page past its working set's pool",
        PAGES_SHORT => "is shorter than the pages the lane reads",
        PAGES_BELOW_FLOOR => "would land a row below its working set's frozen slots or past its reserve",
        PAGES_COPY_BOUNDS => {
            "states a page copy past its working set's pool, longer than a page, or into frozen \
             slots"
        }
        PAGES_PARENT => "continues a parent past its beams",
        PAGES_EXHAUSTED => {
            eprintln!(
                "engine-cuda: BUG: lane {lane}'s beam group found fewer free pages than it hands \
                 out; its pool is sized exactly (BEAM-PAGE-LISTS.md §3)"
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

/// Plans `groups` over the fire's staged `geometries` (`indptr` a seat each):
/// `translation(source)` a lane's working set's whole table (runtime page
/// ids) and `folded(source)` its frozen slots; `windowed` whether any table
/// is a windowed one (refused); `planes` the kv planes a page holds;
/// `device_ids` maps runtime page ids to the tables' device ids,
/// `page_base` gives a seated page's base address.
#[allow(clippy::too_many_arguments)]
pub fn plan(
    groups: &[PageGroup],
    geometries: &[poem_exec::store::kv::Geometry],
    windowed: bool,
    page: u32,
    translation: &dyn Fn(usize) -> Vec<u32>,
    folded: &dyn Fn(usize) -> u32,
    planes: &[PagePlane],
    rs: Option<&RsFire<'_>>,
    device_ids: &mut dyn FnMut(&mut [i32]) -> Result<()>,
    page_base: &mut dyn FnMut(u32) -> Result<u64>,
) -> Result<PageWork> {
    let mut work = PageWork::default();
    if groups.is_empty() {
        return Ok(work);
    }
    if windowed {
        return Err(Fault::program(
            "serve::pages",
            "a beam group's page lists are resolved over full-context tables only, and this \
             load lays windowed ones",
        ));
    }
    let page = page.max(1);
    work.planes.extend_from_slice(planes);
    let kv_planes = planes.len() as u32;
    let rs_plane_at = work.planes.len() as u32;
    if let Some(rs) = rs {
        work.planes.extend_from_slice(&rs.planes);
    }
    for group in groups {
        let first = group.lanes.first().ok_or_else(|| {
            Fault::program("serve::pages", "a beam group of no lanes".to_string())
        })?;
        if group.lanes.len() as u32 != group.layout.lanes {
            return Err(Fault::program(
                "serve::pages",
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
                "serve::pages",
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
        let backed = backed(group.layout.lanes, slot, folded, page, group.layout.frames, reserve);
        let pool = &whole[..backed as usize];
        if reserve > group.layout.cap {
            return Err(Fault::program(
                "serve::pages",
                format!(
                    "instance {}'s working set reserves {reserve} pool page(s) past its frozen \
                     ones and its beam state is laid for {}",
                    group.instance, group.layout.cap
                ),
            ));
        }
        let fault_at = work.faults.len() as u32;
        for lane in &group.lanes {
            work.faults.push((lane.source, group.instance));
        }
        let trans_at = work.trans.len() as u32;
        let mut ids: Vec<i32> = pool.iter().map(|&p| p as i32).collect();
        device_ids(&mut ids)?;
        work.trans.extend_from_slice(&ids);
        let lists_at = work.lists.len() as u32;
        for (table, geometry) in geometries.iter().enumerate() {
            for (beam, lane) in group.lanes.iter().enumerate() {
                let from = *geometry.indptr.get(lane.seat).ok_or_else(|| {
                    Fault::program("serve::pages", format!("lane {} has no seat", lane.source))
                })? as u32;
                let to = geometry.indptr.get(lane.seat + 1).copied().unwrap_or(0) as u32;
                let count = to.saturating_sub(from);
                work.lists.push(ListPlan {
                    table: table as u32,
                    lane: beam as u32,
                    indices_at: from + f0.min(count),
                    write_row: lane.row_offset,
                    entries: count.saturating_sub(f0),
                    rows: lane.rows,
                });
            }
        }
        let list_count = work.lists.len() as u32 - lists_at;
        // A hybrid's beams (§11): each beam's rs page table, column 0 at its
        // entry `f0 - base / S`, resolved over the same list; the group's rs
        // pool pages are the set's, as its lanes' tables stage them.
        let rs_at = work.rs.len() as u32;
        let mut rs_bases_at = None;
        if let Some(rs) = rs {
            rs_bases_at = rs_lists(&mut work, rs, group, f0, page, slot, backed)?;
        }
        let rs_count = work.rs.len() as u32 - rs_at;
        work.groups.push(GroupPlan {
            state: group.state,
            parents: group.parents,
            trans_at,
            lists_at,
            list_count,
            fault_at,
            lanes: group.layout.lanes,
            cap: group.layout.cap,
            reserve,
            page,
            folded,
            slot,
            frames: group.layout.frames,
            backed,
            rs_at,
            rs_count,
        });
        let bases_at = work.bases.len() as u32;
        for &p in pool {
            work.bases.push(page_base(p)?);
        }
        let copies = kernels_cuda::runtime::channel::beam_copies_word(
            group.layout.lanes,
            group.layout.cap,
            group.layout.frames,
        );
        for beam in 0..group.layout.lanes {
            work.copies.push(CopyPlan {
                fault: fault_at + beam,
                copy: group.state + (copies + 3 * u64::from(beam)) * 4,
                bases_at,
                reserve: backed,
                page,
                floor_rel: folded - f0 * page,
                plane_at: 0,
                planes: kv_planes,
            });
            if let (Some(rs), Some(rs_bases_at)) = (rs, rs_bases_at) {
                work.copies.push(CopyPlan {
                    fault: fault_at + beam,
                    copy: group.state + (copies + 3 * u64::from(beam)) * 4,
                    bases_at: rs_bases_at,
                    reserve: backed,
                    page,
                    floor_rel: folded - f0 * page,
                    plane_at: rs_plane_at,
                    planes: rs.planes.len() as u32,
                });
            }
        }
    }
    Ok(work)
}

/// A hybrid group's rs page tables (`BEAM-PAGE-LISTS.md` §11): each beam's
/// columns `0..=c` from its table's entry `f0 - base / S`, and its pool's
/// rs pages (the set's, from that entry on, as beam 0's table stages them)
/// in [`PageWork::bases`]; where their run starts there, or `None` for a
/// group whose lanes buffer no rows. Refused by name where a page's rows
/// are not a kv page's (one list pages both), where the buffer does not
/// start on a page, or where the set's rs pages do not reach the pool's
/// staged pages.
fn rs_lists(
    work: &mut PageWork,
    rs: &RsFire<'_>,
    group: &PageGroup,
    f0: u32,
    page: u32,
    slot: u32,
    backed: u32,
) -> Result<Option<u32>> {
    let lane_of = |source: usize| -> Option<kernels_cuda::runtime::channel::RsLane> {
        let fire_lane = *rs.order.get(source)? as usize;
        rs.lanes
            .get(fire_lane)
            .copied()
            .filter(|lane| lane.at != kernels_cuda::runtime::channel::RsLane::NONE)
    };
    let first = &group.lanes[0];
    let (Some(lane0), Some(base)) = (lane_of(first.source), (rs.base)(first.source)) else {
        return Ok(None);
    };
    if rs.page_tokens != page || base % page != 0 || base > f0 * page {
        return Err(Fault::program(
            "serve::pages",
            format!(
                "instance {}'s beams buffer recurrent rows {} a page from slot {base}, and \
                 one page list pages them with its kv slots ({page} a page, its beams from \
                 slot {}): a hybrid's beam group needs both on the same pages",
                group.instance,
                rs.page_tokens,
                f0 * page
            ),
        ));
    }
    let from = f0 - base / page;
    let staged = rs.slots.get(lane0.page_off as usize..).unwrap_or(&[]);
    let pool = staged.get(from as usize..).unwrap_or(&[]);
    if (pool.len() as u32) < backed {
        return Err(Fault::program(
            "serve::pages",
            format!(
                "instance {}'s working set holds {} rs buffer page(s) past slot {}, and its \
                 beams' pool stages {backed}: its rs pages are not yet reserved with its kv \
                 pool's (BEAM-PAGE-LISTS.md §3, §11)",
                group.instance,
                pool.len(),
                f0 * page
            ),
        ));
    }
    let bases_at = work.bases.len() as u32;
    work.bases.extend_from_slice(&pool[..backed as usize]);
    let entries = slot / page - f0 + 1;
    for (beam, lane) in group.lanes.iter().enumerate() {
        let rs_lane = lane_of(lane.source).ok_or_else(|| {
            Fault::program(
                "serve::pages",
                format!(
                    "instance {}'s beam {beam} buffers no recurrent rows where beam 0 does",
                    group.instance
                ),
            )
        })?;
        work.rs.push(RsListPlan {
            lane: beam as u32,
            from: rs_lane.page_off + from,
            entries,
            bases_at,
        });
    }
    Ok(Some(bases_at))
}
