//! A beam group's head step on the device (`new-memory/designs/BEAM-PAGE-LISTS.md`):
//! `beam_step` (§5-6.3, then §6.5 over the lanes' staged tables) and
//! `beam_copy_pages` (§6.4), `runtime/beam.metal`. The run records mirror
//! the MSL structs; every address in them is a GPU virtual address.

use crate::encode::{Arg, Ctx, Fire, Grid};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "runtime/beam.metal";

/// `BEAM-PAGE-LISTS.md` §7's fault codes, one word a lane.
pub const PAGES_PAST_RESERVE: u32 = 1;
pub const PAGES_SHORT: u32 = 2;
pub const PAGES_BELOW_FLOOR: u32 = 3;
pub const PAGES_COPY_BOUNDS: u32 = 4;
pub const PAGES_PARENT: u32 = 5;
/// Fewer free pages than a fire hands out: unreachable with exact sizing,
/// so a bug where it happens.
pub const PAGES_EXHAUSTED: u32 = 6;
/// A group wider than [`MAX_LANES`]: every lane faulted, its step not run.
/// The host refuses such a group at bind and before a fire, so a fault
/// here is a bug where it happens.
pub const PAGES_TOO_WIDE: u32 = 7;

/// The most beams one group runs: `beam_step`'s threadgroup arrays
/// (`PIE_BEAM_MAX_LANES`), about 18 KiB with the common-ancestor table
/// (`MAX_LANES`² words), inside every Apple GPU's 32 KiB.
pub const MAX_LANES: u32 = 64;

const STEP_THREADS: u32 = 256;
const COPY_THREADS: u32 = 128;

/// The u32 words a group of `lanes` beams keeps for lists of `cap` columns
/// over `cap` pool pages and `frames` pending rings: a header of 4, the
/// lists twice (by the fire count's parity), the free pages, the ring, a
/// fire's copies `[src, dst, tokens]` a beam, and each pair of lanes' common
/// ancestor depth (BEAM-PAGE-LISTS §11.4).
#[must_use]
pub const fn beam_state_words(lanes: u32, cap: u32, frames: u32) -> u64 {
    beam_copies_word(lanes, cap, frames) + 3 * lanes as u64 + lanes as u64 * lanes as u64
}

/// Where a group's copies start in its state, in words.
#[must_use]
pub const fn beam_copies_word(lanes: u32, cap: u32, frames: u32) -> u64 {
    4 + 2 * lanes as u64 * cap as u64 + cap as u64 + frames as u64 * cap as u64
}

/// A lane's staged page table [`beam_step`] resolves over its group's list.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BeamListRun {
    pub indices: u64,
    pub write_page: u64,
    pub write_offset: u64,
    pub lane: u32,
    pub entries: u32,
    pub rows: u32,
    pub pad: u32,
}

/// One group's step.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BeamStepRun {
    pub state: u64,
    pub parents: u64,
    pub trans: u64,
    pub lists: u64,
    pub faults: u64,
    pub lanes: u32,
    pub cap: u32,
    pub reserve: u32,
    pub page: u32,
    pub folded: u32,
    pub slot: u32,
    pub list_count: u32,
    pub frames: u32,
    pub backed: u32,
    /// The hybrid lanes' rs page tables to resolve ([`BeamRsRun`]s at `rs`).
    pub rs_count: u32,
    pub rs: u64,
}

/// A hybrid lane's rs page table to resolve (`BEAM-PAGE-LISTS.md` §11.1):
/// its row's slot entries from the one holding buffer entry `first`
/// (`pages`, past the row's in-page header), entry `e` the rows of slot
/// `base + e * tokens` on, through `entries`, each resolved to the set's
/// entry (`bases`: its entries' slab slots, `bases_len` of them) holding
/// the same rows of its beam's list's page.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BeamRsRun {
    pub pages: u64,
    pub bases: u64,
    pub lane: u32,
    pub first: u32,
    pub entries: u32,
    pub base: u32,
    pub tokens: u32,
    pub bases_len: u32,
}

/// A beam's rs copy (`BEAM-PAGE-LISTS.md` §11.2): the `[src, dst, tokens]`
/// its group's step left at `copy`, moved row by row of each plane through
/// the set's rs entries (`bases`: slab slots of `page_bytes` pages in
/// `slab`); slots below the buffer have no row.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RsCopyRun {
    pub copy: u64,
    pub bases: u64,
    pub fault: u64,
    pub slab: u64,
    pub reserve: u32,
    pub page: u32,
    pub tokens: u32,
    pub base: u32,
    pub f0: u32,
    pub floor_rel: u32,
    pub plane_at: u32,
    pub planes: u32,
    pub bases_len: u32,
    pub page_bytes: u32,
}

/// A beam's copy over the kv planes a page holds.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PageCopyRun {
    pub copy: u64,
    pub base: u64,
    pub fault: u64,
    pub reserve: u32,
    pub page: u32,
    pub floor_rel: u32,
    pub plane_at: u32,
    pub planes: u32,
    pub pad: u32,
}

/// Where a plane lies in a page and the bytes a token holds in it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PagePlane {
    pub offset: u64,
    pub cell: u64,
}

/// A record's bytes as the MSL struct lays them (little-endian, no
/// padding past the stated fields).
pub trait Record {
    fn put(&self, out: &mut Vec<u8>);
}

/// `records` laid back to back.
#[must_use]
pub fn bytes<R: Record>(records: &[R]) -> Vec<u8> {
    let mut out = Vec::new();
    for record in records {
        record.put(&mut out);
    }
    out
}

fn put64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

impl Record for BeamListRun {
    fn put(&self, out: &mut Vec<u8>) {
        for v in [self.indices, self.write_page, self.write_offset] {
            put64(out, v);
        }
        for v in [self.lane, self.entries, self.rows, self.pad] {
            put32(out, v);
        }
    }
}

impl Record for BeamStepRun {
    fn put(&self, out: &mut Vec<u8>) {
        for v in [self.state, self.parents, self.trans, self.lists, self.faults] {
            put64(out, v);
        }
        for v in [
            self.lanes,
            self.cap,
            self.reserve,
            self.page,
            self.folded,
            self.slot,
            self.list_count,
            self.frames,
            self.backed,
            self.rs_count,
        ] {
            put32(out, v);
        }
        put64(out, self.rs);
    }
}

impl Record for BeamRsRun {
    fn put(&self, out: &mut Vec<u8>) {
        put64(out, self.pages);
        put64(out, self.bases);
        for v in [self.lane, self.first, self.entries, self.base, self.tokens, self.bases_len] {
            put32(out, v);
        }
    }
}

impl Record for RsCopyRun {
    fn put(&self, out: &mut Vec<u8>) {
        for v in [self.copy, self.bases, self.fault, self.slab] {
            put64(out, v);
        }
        for v in [
            self.reserve,
            self.page,
            self.tokens,
            self.base,
            self.f0,
            self.floor_rel,
            self.plane_at,
            self.planes,
            self.bases_len,
            self.page_bytes,
        ] {
            put32(out, v);
        }
    }
}

impl Record for PageCopyRun {
    fn put(&self, out: &mut Vec<u8>) {
        for v in [self.copy, self.base, self.fault] {
            put64(out, v);
        }
        for v in [self.reserve, self.page, self.floor_rel, self.plane_at, self.planes, self.pad] {
            put32(out, v);
        }
    }
}

impl Record for PagePlane {
    fn put(&self, out: &mut Vec<u8>) {
        put64(out, self.offset);
        put64(out, self.cell);
    }
}

impl Record for u32 {
    fn put(&self, out: &mut Vec<u8>) {
        put32(out, *self);
    }
}

impl Record for i32 {
    fn put(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

impl Record for u64 {
    fn put(&self, out: &mut Vec<u8>) {
        put64(out, *self);
    }
}

const _: () = {
    assert!(core::mem::size_of::<BeamListRun>() == 40);
    assert!(core::mem::size_of::<BeamStepRun>() == 88);
    assert!(core::mem::size_of::<BeamRsRun>() == 40);
    assert!(core::mem::size_of::<RsCopyRun>() == 72);
    assert!(core::mem::size_of::<PageCopyRun>() == 48);
    assert!(core::mem::size_of::<PagePlane>() == 16);
};

/// The pipelines the head step dispatches: a load builds them before
/// ready, where any fire may meet a beam group.
#[must_use]
pub fn pipelines() -> [Fire; 3] {
    [
        Fire::at(FILE, "beam_step"),
        Fire::at(FILE, "beam_copy_pages"),
        Fire::at(FILE, "copy_rs_rows"),
    ]
}

/// Runs `run_count` [`BeamStepRun`]s from `runs` (bytes), a threadgroup a
/// group.
pub fn beam_step(ctx: &Ctx<'_>, runs: Tensor, run_count: u32) -> Result<(), Error> {
    if run_count == 0 {
        return Ok(());
    }
    ctx.fire(
        Fire::at(FILE, "beam_step").apply(Grid::of(
            [STEP_THREADS * run_count, 1, 1],
            [STEP_THREADS, 1, 1],
        )),
        &[runs.arg(), run_count.arg()],
    )
}

/// Copies `run_count` [`PageCopyRun`]s from `runs` over their planes of
/// `planes`, a threadgroup a (run, plane); `max_planes` the most a run spans.
pub fn beam_copy_pages(
    ctx: &Ctx<'_>,
    runs: Tensor,
    planes: Tensor,
    run_count: u32,
    max_planes: u32,
) -> Result<(), Error> {
    if run_count == 0 || max_planes == 0 {
        return Ok(());
    }
    ctx.fire(
        Fire::at(FILE, "beam_copy_pages").apply(Grid::of(
            [COPY_THREADS * run_count, max_planes, 1],
            [COPY_THREADS, 1, 1],
        )),
        &[runs.arg(), planes.arg(), run_count.arg()],
    )
}

/// Copies `run_count` [`RsCopyRun`]s from `runs` over their planes of
/// `planes`, a threadgroup a (run, plane); `max_planes` the most a run spans
/// (`BEAM-PAGE-LISTS.md` §11.2).
pub fn copy_rs_rows(
    ctx: &Ctx<'_>,
    runs: Tensor,
    planes: Tensor,
    run_count: u32,
    max_planes: u32,
) -> Result<(), Error> {
    if run_count == 0 || max_planes == 0 {
        return Ok(());
    }
    ctx.fire(
        Fire::at(FILE, "copy_rs_rows").apply(Grid::of(
            [COPY_THREADS * run_count, max_planes, 1],
            [COPY_THREADS, 1, 1],
        )),
        &[runs.arg(), planes.arg(), run_count.arg()],
    )
}
