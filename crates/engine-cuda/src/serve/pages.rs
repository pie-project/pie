//! Beam groups' page lists (`state-geometry.parents`): planned on the host from what
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
    /// Its first beam's entry of the fire's beam rows (`channel::RsRows`),
    /// its beams' in order from it.
    pub rows_at: u32,
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
    /// Its beams' entries of the fire's beam rows from this one, where they
    /// read rs rows (a hybrid's): the step points them at its lists.
    pub rows_at: Option<u32>,
}

/// The pool pages a group of `lanes` beams can name at the fire whose
/// beams write `slot`: the step hands out the
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
/// pool's kv pages, or on a hybrid the rs planes over its rs pages.
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

/// Where a lane's rs rows lie (`channel::RsRows`): its entry of the fire's
/// lane rows; a beam's (a hybrid's) list's row of
/// its group's state before the step points it (the state's first list:
/// valid memory; 0 for a lane alone on its set) and the list's columns, the group's first pool page `f0`
/// and kv page, the buffer's first slot and its rows a page, and where the
/// set's rs entries' slot addresses start in [`PageWork::bases`] and how
/// many.
#[derive(Clone, Copy, Debug)]
pub struct RsRowsPlan {
    pub entry: u32,
    pub list: u64,
    pub cap: u32,
    pub f0: u32,
    pub page: u32,
    pub base: u32,
    pub tokens: u32,
    pub bases_at: u32,
    pub bases_len: u32,
}

/// A beam's rs copy (a fork's rows on a hybrid): its group's copy, row by row of
/// each rs plane through the set's rs entries (`copy_rs_rows`).
#[derive(Clone, Copy, Debug)]
pub struct RsCopyPlan {
    pub fault: u32,
    pub copy: u64,
    pub bases_at: u32,
    pub bases_len: u32,
    pub reserve: u32,
    pub page: u32,
    pub tokens: u32,
    pub base: u32,
    pub f0: u32,
    pub floor_rel: u32,
    pub plane_at: u32,
    pub planes: u32,
}

/// What the plan reads of a fire's rs buffer (a hybrid's): its lanes' rs
/// lanes by fire lane, the fire lane of each source lane, the rows a page
/// holds, each source lane's buffer base and its set's rs pages, and the
/// planes a page's rows lie in.
pub struct RsFire<'a> {
    pub lanes: &'a [kernels_cuda::runtime::channel::RsLane],
    pub order: &'a [u32],
    pub page_tokens: u32,
    pub base: &'a dyn Fn(usize) -> Option<u32>,
    /// A source lane's set's rs pages' slot addresses, every one of its
    /// buffer (a beam's rows lie on any).
    pub set: &'a dyn Fn(usize) -> Result<Vec<u64>>,
    pub planes: Vec<PagePlane>,
}

/// What a fire's beam groups have the device run.
#[derive(Clone, Debug, Default)]
pub struct PageWork {
    pub groups: Vec<GroupPlan>,
    pub lists: Vec<ListPlan>,
    /// Where the lanes' rs rows lie: the beams' (a hybrid's) and the
    /// buffering lanes' alone on their sets.
    pub rs: Vec<RsRowsPlan>,
    pub copies: Vec<CopyPlan>,
    /// The beams' rs copies (a hybrid's).
    pub rs_copies: Vec<RsCopyPlan>,
    pub trans: Vec<i32>,
    /// The pool and rs pages' addresses: the lanes' alone on their sets
    /// first ([`PageWork::of_lanes`]), then the groups'.
    pub bases: Vec<u64>,
    /// Whether lanes alone on their sets read pages of `bases`.
    pub lanes: bool,
    pub planes: Vec<PagePlane>,
    /// Per fault word, the lane it faults.
    pub faults: Vec<(usize, engine::InstanceId)>,
    /// Each group's common slot word (its state's `header[3]`) and its
    /// instance: read with the fault words when the fire lands.
    pub commons: Vec<(u64, engine::InstanceId)>,
}

impl PageWork {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty() && !self.lanes
    }

    /// The work of a fire whose lanes alone on their sets read their rows
    /// through `slots` (the slot addresses of the pages they read or write,
    /// `RsLane::page_off` counting from the first), before its groups'.
    #[must_use]
    pub fn of_lanes(slots: Vec<u64>) -> PageWork {
        PageWork {
            lanes: !slots.is_empty(),
            bases: slots,
            ..PageWork::default()
        }
    }
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
    lane_pages: Vec<u64>,
) -> Result<PageWork> {
    let mut work = PageWork::of_lanes(lane_pages);
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
        // A hybrid's beams: each beam's rs rows, resolved row by row
        // through the same list; the group's rs pool pages are the set's.
        let mut rs_bases_at = None;
        if let Some(rs) = rs {
            rs_bases_at = rs_lists(&mut work, rs, group, f0, page, slot)?;
        }
        let rows_at = rs_bases_at.map(|_| group.rows_at);
        // The head step leaves the group's common slot in its header's fourth
        // word.
        work.commons.push((group.state + 12, group.instance));
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
            rows_at,
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
            if let (Some(rs), Some((bases_at, bases_len, base))) = (rs, rs_bases_at) {
                work.rs_copies.push(RsCopyPlan {
                    fault: fault_at + beam,
                    copy: group.state + (copies + 3 * u64::from(beam)) * 4,
                    bases_at,
                    bases_len,
                    reserve: backed,
                    page,
                    tokens: rs.page_tokens,
                    base,
                    f0,
                    floor_rel: folded - f0 * page,
                    plane_at: rs_plane_at,
                    planes: rs.planes.len() as u32,
                });
            }
        }
    }
    Ok(work)
}

/// A hybrid group's beams' rs rows: where each
/// beam's rows resolve, slot by slot through its list to the set's entry
/// holding the same rows, and the set's rs entries' slot addresses in
/// [`PageWork::bases`]; their run, its length and the buffer's base, or
/// `None` for a group whose lanes buffer no rows. Refused by name where a
/// page's rows do not divide a kv page, or where the buffer does not start
/// on a row page (an entry would straddle two of a beam's pages).
fn rs_lists(
    work: &mut PageWork,
    rs: &RsFire<'_>,
    group: &PageGroup,
    f0: u32,
    page: u32,
    slot: u32,
) -> Result<Option<(u32, u32, u32)>> {
    let lane_of = |source: usize| -> Option<kernels_cuda::runtime::channel::RsLane> {
        let fire_lane = *rs.order.get(source)? as usize;
        rs.lanes
            .get(fire_lane)
            .copied()
            .filter(|lane| lane.at != kernels_cuda::runtime::channel::RsLane::NONE)
    };
    let first = &group.lanes[0];
    let (Some(_lane0), Some(base)) = (lane_of(first.source), (rs.base)(first.source)) else {
        return Ok(None);
    };
    let tokens = rs.page_tokens.max(1);
    if page % tokens != 0 || base % tokens != 0 {
        return Err(Fault::program(
            "serve::pages",
            format!(
                "instance {}'s beams buffer recurrent rows {tokens} a page from slot {base}, \
                 and their page lists page kv slots {page} a page: a hybrid's beams need the \
                 rs page's rows to divide the kv page's and its buffer to start on a row page",
                group.instance
            ),
        ));
    }
    if slot < base {
        return Ok(None);
    }
    let set = (rs.set)(first.source)?;
    let bases_at = work.bases.len() as u32;
    work.bases.extend_from_slice(&set);
    let bases_len = set.len() as u32;
    for (beam, lane) in group.lanes.iter().enumerate() {
        lane_of(lane.source).ok_or_else(|| {
            Fault::program(
                "serve::pages",
                format!(
                    "instance {}'s beam {beam} buffers no recurrent rows where beam 0 does",
                    group.instance
                ),
            )
        })?;
        work.rs.push(RsRowsPlan {
            entry: group.rows_at + beam as u32,
            list: group.state
                + u64::from(kernels_cuda::runtime::channel::BEAM_HEADER + beam as u32 * group.layout.cap)
                    * 4,
            cap: group.layout.cap,
            f0,
            page,
            base,
            tokens,
            bases_at,
            bases_len,
        });
    }
    Ok(Some((bases_at, bases_len, base)))
}
