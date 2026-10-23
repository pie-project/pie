//! Lane phases as the engine runs them: the plan it builds from what an
//! entry states and the split it chose, where the phase's buffers live (the
//! PEs it runs on, its header, each bank's, plane's and page list's share of
//! a PE), and which indices a PE's block picks.

use dtype::Dtype;
use kernels_cerebras::error::{Error, refuse};
use kernels_cerebras::phase::{
    BankSplit, ColPlane, LaneKind, LanePhase, Lanes, LanesAt, PageSource, Reduce, Segment, Shard,
};
use kernels_cerebras::plan::LaneSplit;

use super::Program;

/// A lane phase as the host lays it out around the run: PE `p` runs lanes
/// `[l0, l1)` (rows `[r0, r1)`) over block `[a0, a1) × [b0, b1)`, read from
/// its row of `header` (a `[pes, 16]` i32 buffer: `l0, l1, held slots, a0,
/// a1, b0, b1, r0, r1, pg0, pages, 0…`), holding only its lanes' slot
/// blocks of `banks`, with `slots` rewritten to those local rows. What the
/// entry stated ([`Lanes`]) and the split the engine chose ([`LaneSplit`]),
/// together.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LanePlan {
    pub pes: u32,
    pub kind: LaneKind,
    pub header: String,
    /// The slot table (i32, one entry per row): lane `l`'s slot is entry
    /// `l` (per row) or entry `indptr[l]` (ragged).
    pub slots: String,
    /// Banks `(name, stride, split)`: `[slots_total, stride]` f32 planes; a
    /// PE holds `lanes_per_pe` local slot rows, each the whole `stride`, its
    /// block of `[a][b][w]`, or rows `[b0, b1)` of `[b][stride / b]`.
    pub banks: Vec<(String, u32, BankSplit)>,
    /// How each slot's state splits within a lane, when it does; `stride`
    /// then equals `a × b × w`.
    pub block: Option<BlockPlan>,
    /// The most lanes one PE runs; the local bank rows it holds.
    pub lanes_per_pe: u32,
    /// The PE order: page groups vary fastest, then the b axis's groups,
    /// then the a axis's, then the lane groups; `a_first` swaps the two
    /// block axes.
    pub a_first: bool,
    /// Outputs written row by row per lane `(name, width, a_stride,
    /// b_stride)` (see [`Lanes::row_outputs`]).
    pub row_outputs: Vec<(String, u32, u32, u32)>,
    /// Planes held by block (columns, or rows).
    pub cols: Vec<ColPlane>,
    /// How row outputs held by several PEs merge (page groups > 1).
    pub reduce: Vec<(String, Reduce)>,
    /// The lanes' rows clipped to `[start, start + len)` and counted from
    /// `start`.
    pub window: Option<(u32, u32)>,
    /// Paged banks instead of one slot per lane.
    pub pages: Option<PagePlan>,
}

/// How one slot's state `[a][b][w]` splits within a lane: `a` into
/// `a_groups` and `b` into `b_groups` ranges, `w` words contiguous.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockPlan {
    pub a: u32,
    pub b: u32,
    pub w: u32,
    pub a_groups: u32,
    pub b_groups: u32,
}

impl BlockPlan {
    /// Words one PE holds of one slot.
    pub fn words(&self) -> u64 {
        u64::from(self.a / self.a_groups) * u64::from(self.b / self.b_groups) * u64::from(self.w)
    }
}

/// The paged form of a lane's state: lane `l` holds pages
/// `indices[indptr[l]..indptr[l + 1]]` of the banks (each page `page_size`
/// rows of a bank plane). A PE holds `lanes_per_pe × max_pages` local
/// pages; it reads `indptr`, `indices` and the `rewritten` per-row page
/// tables with local page numbers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagePlan {
    pub source: PageSource,
    pub page_size: u32,
    pub max_pages: u32,
    /// Per-row page tables rewritten to local pages for the PE's rows.
    pub rewritten: Vec<String>,
    /// A lane's pages split over this many PEs (see [`LaneSplit`]): PE `p`
    /// holds pages `[pg0, pg0 + max_pages / page_groups)` of each of its
    /// lanes, with `pg0` (and the pages per PE) in its header words 9 and
    /// 10.
    pub page_groups: u32,
    /// The pool strided over the resident row, `strided_pages` local pages
    /// a PE (see [`LaneSplit`]).
    pub strided: bool,
    pub strided_pages: u32,
}

impl PagePlan {
    /// Pages one PE holds of one lane.
    pub fn pages_per_pe(&self) -> u32 {
        self.max_pages.max(1).div_ceil(self.page_groups.max(1))
    }
}

/// The plan of a lane phase an entry closed: what it stated, the split, and
/// the PEs and header the engine placed it on.
pub fn plan_of(phase: &LanePhase) -> LanePlan {
    let mut plan = plan(&phase.lanes, phase.split, phase.at.header.name.clone());
    plan.pes = phase.at.pes;
    plan
}

/// `lanes` split as `split` says, its header `header`, on the split's PEs.
fn plan(lanes: &Lanes, split: LaneSplit, header: String) -> LanePlan {
    let Lanes {
        kind,
        slots,
        banks,
        block,
        a_first,
        row_outputs,
        cols,
        reduce,
        window,
        pages,
    } = lanes.clone();
    LanePlan {
        pes: split.pes,
        kind,
        header,
        slots,
        banks,
        block: block
            .zip(split.block)
            .map(|(b, (a_groups, b_groups))| BlockPlan {
                a: b.a,
                b: b.b,
                w: b.w,
                a_groups,
                b_groups,
            }),
        lanes_per_pe: split.lanes_per_pe,
        a_first,
        row_outputs,
        cols,
        reduce,
        window,
        pages: pages.map(|p| PagePlan {
            source: p.source,
            page_size: p.page_size,
            max_pages: p.max_pages,
            rewritten: p.rewritten,
            page_groups: split.page_groups,
            strided: split.strided_pages.is_some(),
            strided_pages: split.strided_pages.unwrap_or(0),
        }),
    }
}

/// Places `lanes` split as `split` says on the program: under the resident
/// placement (`row`) on the whole row, its header named and declared `[pes,
/// 16]`, every buffer it holds by lane or block sharded to its share of a
/// PE.
pub fn place(
    program: &mut Program,
    row: Option<u32>,
    lanes: &Lanes,
    split: LaneSplit,
) -> Result<LanesAt, Error> {
    let header = program.unique("lanes");
    let plan = &plan(lanes, split, header);
    let pes = pes(row, plan)?;
    let header = program.declare(&plan.header, Dtype::I32, pes, 16, true)?;
    let mut shards = vec![(plan.header.clone(), Shard::Rows(pes))];
    for (bank, stride, split) in &plan.banks {
        shards.push((
            bank.clone(),
            Shard::Local(page_slots(plan) * slot_words(plan, *stride, *split)),
        ));
    }
    let planes: Vec<u64> = plan.cols.iter().map(|c| plane_words(plan, c)).collect();
    for (plane, words) in plan.cols.iter().zip(&planes) {
        shards.push((plane.name.clone(), Shard::Local(*words)));
    }
    if let Some(PagePlan {
        source: PageSource::Csr { indices, .. },
        ..
    }) = &plan.pages
    {
        shards.push((indices.clone(), Shard::Local(local_rows(plan))));
    }
    Ok(LanesAt {
        header,
        pes,
        shards,
        planes,
    })
}

/// The PEs a lane plan runs on: its own, or under the resident placement
/// the row (idle PEs past its own), where a lane whose state splits over
/// PEs merges its partials over the whole row or not at all.
fn pes(row: Option<u32>, plan: &LanePlan) -> Result<u32, Error> {
    let Some(pes) = row else {
        return Ok(plan.pes);
    };
    if plan.pes > pes {
        return Err(refuse(
            "lanes",
            format!(
                "the lane plan wants {} PEs; the resident row has {pes}",
                plan.pes
            ),
        ));
    }
    let whole_row = plan.reduce.iter().all(|(_, r)| {
        matches!(r, Reduce::FabricRoot { group } | Reduce::FabricSum { group } if *group == pes)
    });
    if !whole_row {
        return Err(refuse(
            "lanes",
            "the resident placement merges a lane's partials over the whole row or not at all",
        ));
    }
    Ok(pes)
}

/// Words one PE holds of one slot of `stride` (its block when the bank
/// splits by block).
fn slot_words(plan: &LanePlan, stride: u32, split: BankSplit) -> u64 {
    match (plan.block, split) {
        (Some(b), BankSplit::Block) => b.words(),
        (Some(b), BankSplit::ByB) => {
            u64::from(b.b / b.b_groups.max(1)) * u64::from(stride / b.b.max(1))
        }
        _ => u64::from(stride),
    }
}

/// Local slot (or page) rows one PE holds.
fn local_rows(plan: &LanePlan) -> u64 {
    u64::from(plan.lanes_per_pe) * u64::from(plan.pages.as_ref().map_or(1, PagePlan::pages_per_pe))
}

/// Bank slots (pages) one PE holds: its lanes' pages, or its share of the
/// pool under the resident placement.
fn page_slots(plan: &LanePlan) -> u64 {
    match &plan.pages {
        Some(p) if p.strided => u64::from(p.strided_pages.max(1)),
        _ => local_rows(plan),
    }
}

/// The words of `plane` one PE holds (the most over the PEs).
fn plane_words(plan: &LanePlan, plane: &ColPlane) -> u64 {
    let (rows, width) = (u64::from(plane.rows), u64::from(plane.width));
    let Some(b) = plan.block else {
        return rows * width;
    };
    if plane.row_block {
        let ai = b.a / b.a_groups.max(1);
        let bj = b.b / b.b_groups.max(1);
        let cols = picks(&plane.segments, (0, ai), (0, bj), plane.width).len() as u64;
        return u64::from(ai) * cols;
    }
    let bound = if plane.by_rows {
        plane.rows
    } else {
        plane.width
    };
    let (ai, bj) = (b.a / b.a_groups.max(1), b.b / b.b_groups.max(1));
    let picked = (0..b.a_groups.max(1))
        .flat_map(|ag| (0..b.b_groups.max(1)).map(move |bg| (ag, bg)))
        .map(|(ag, bg)| {
            picks(
                &plane.segments,
                (ag * ai, (ag + 1) * ai),
                (bg * bj, (bg + 1) * bj),
                bound,
            )
            .len() as u64
        })
        .max()
        .unwrap_or(0);
    if plane.by_rows {
        picked * width
    } else {
        rows * picked
    }
}

/// The indices below `bound` one segment picks for block `[a0, a1) × [b0,
/// b1)`, ascending and deduplicated.
fn segment_picks(s: &Segment, (a0, a1): (u32, u32), (b0, b1): (u32, u32), bound: u32) -> Vec<u32> {
    let mut seen = std::collections::BTreeSet::new();
    for i in a0..a1 {
        let ia = i.checked_div(s.a_group).unwrap_or(0);
        let js = if s.b_stride > 0 { b0..b1 } else { 0..1 };
        for j in js {
            for t in 0..s.span {
                let at = s.base + ia * s.a_stride + j * s.b_stride + t;
                if at < bound {
                    seen.insert(at);
                }
            }
        }
    }
    seen.into_iter().collect()
}

/// The indices `segments` pick together for block `a × b`, below `bound`,
/// ascending and deduplicated.
pub fn picks(segments: &[Segment], a: (u32, u32), b: (u32, u32), bound: u32) -> Vec<u32> {
    let mut all: Vec<u32> = segments
        .iter()
        .flat_map(|s| segment_picks(s, a, b, bound))
        .collect();
    all.sort_unstable();
    all.dedup();
    all
}
