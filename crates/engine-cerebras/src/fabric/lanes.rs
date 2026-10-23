//! Where a lane phase's buffers live: the PEs it runs on, its header, and
//! each bank's, plane's and page list's share of a PE.

use dtype::Dtype;
use kernels_cerebras::error::{Error, refuse};
use kernels_cerebras::phase::{
    BankSplit, ColPlane, LanePlan, LanesAt, PagePlan, PageSource, Reduce, Shard, picks,
};

use super::Program;

/// Places `plan` on the program: under the resident placement (`row`) on
/// the whole row, its header declared `[pes, 16]`, every buffer it holds
/// by lane or block sharded to its share of a PE.
pub fn place(program: &mut Program, row: Option<u32>, plan: &LanePlan) -> Result<LanesAt, Error> {
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
