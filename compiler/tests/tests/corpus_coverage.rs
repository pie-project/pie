//! What the corpus does *not* reach.
//!
//! Coverage was measured, not assumed: before `extended_traces` existed, the
//! goldens plus `synthetic_traces` reached 38 of the 55 ops in `OP_TABLE` and 5
//! of the 8 intrinsics, and never produced a `HierarchicalRow` region or an
//! `OnAttn` stage. Seventeen ops -- `neg`, `recip`, `abs`, `sign`, `max_elem`,
//! `min_elem`, `gt`, `ne`, `le`, `or`, `not`, `reduce_min`, `transpose`,
//! `cumsum`, `gather_row`, `scatter_add`, `rng` -- could have been emitted
//! wrongly, or not at all, without a single test noticing.
//!
//! This file is the tripwire that keeps that from creeping back. It is not a
//! correctness test; it is the answer to "is anything being compiled that
//! nothing looks at?"

#[path = "common/msl_corpus.rs"]
mod msl_corpus;

use std::collections::BTreeSet;

use msl_corpus::{corpus_stages, extended_stages};
use pie_ir::op::{IntrinsicId, OP_TABLE, Op};
use pie_ir::registry::Stage;
use pie_plan::{LibraryOp, RegionKind, ScheduleTemplate};

struct Coverage {
    ops: BTreeSet<u8>,
    intrinsics: BTreeSet<u16>,
    schedules: Vec<ScheduleTemplate>,
    libraries: Vec<LibraryOp>,
    stages: BTreeSet<Stage>,
}

fn measure() -> Coverage {
    let mut cov = Coverage {
        ops: BTreeSet::new(),
        intrinsics: BTreeSet::new(),
        schedules: Vec::new(),
        libraries: Vec::new(),
        stages: BTreeSet::new(),
    };
    for stage in corpus_stages().into_iter().chain(extended_stages()) {
        cov.stages.insert(stage.plan.normalized.stage);
        for op in &stage.plan.normalized.ops {
            cov.ops.insert(op.tag());
            if let Op::IntrinsicVal { intr, .. } = op {
                cov.intrinsics.insert(*intr as u16);
            }
        }
        for partition in [&stage.plan.singleton, &stage.plan.fused] {
            for region in &partition.regions {
                if !cov.schedules.contains(&region.schedule) {
                    cov.schedules.push(region.schedule);
                }
                if let RegionKind::Library(op) = region.kind
                    && !cov.libraries.contains(&op)
                {
                    cov.libraries.push(op);
                }
            }
        }
    }
    cov
}

/// Every op in `OP_TABLE` is compiled by at least one corpus trace.
///
/// If a new op lands, this fails until the corpus reaches it. That is the
/// point: an op the corpus never compiles is an op whose emitter output no
/// golden pins, so it can be broken by any rewrite in total silence.
#[test]
fn every_op_is_reached_by_the_corpus() {
    let covered = measure().ops;
    let missing: Vec<&str> = OP_TABLE
        .iter()
        .filter(|op| !covered.contains(&op.tag))
        .map(|op| op.name)
        .collect();
    assert!(
        missing.is_empty(),
        "{} of {} ops are compiled by no corpus trace, so nothing pins their \
         emitted code: {missing:?}. Add a trace to `msl_corpus::extended_traces`.",
        missing.len(),
        OP_TABLE.len()
    );
}

/// Same argument, for the intrinsics. This one has teeth: the Metal backend
/// bound *every* intrinsic to the logits buffer, and the corpus reached only
/// the intrinsics for which that happened to be correct.
#[test]
fn every_intrinsic_is_reached_by_the_corpus() {
    let covered = measure().intrinsics;
    let missing: Vec<&str> = IntrinsicId::ALL
        .iter()
        .filter(|intr| !covered.contains(&(**intr as u16)))
        .map(|intr| intr.name())
        .collect();
    assert!(
        missing.is_empty(),
        "intrinsics no corpus trace reaches: {missing:?}"
    );
}

/// Each schedule template is a different emitter path; an unreached one is an
/// unpinned one.
#[test]
fn every_schedule_and_library_op_is_reached() {
    let cov = measure();
    let schedules = [
        ScheduleTemplate::Effects,
        ScheduleTemplate::OneCtaPerRow,
        ScheduleTemplate::HierarchicalRow,
        ScheduleTemplate::Library,
    ];
    let missing: Vec<_> = schedules
        .iter()
        .filter(|s| !cov.schedules.contains(s))
        .collect();
    assert!(missing.is_empty(), "unreached schedules: {missing:?}");

    let libraries = [
        LibraryOp::NucleusSample,
        LibraryOp::TopK,
        LibraryOp::Sort,
        LibraryOp::Scan,
        LibraryOp::MatMul,
        LibraryOp::SecondParty,
    ];
    let missing: Vec<_> = libraries
        .iter()
        .filter(|l| !cov.libraries.contains(l))
        .collect();
    assert!(missing.is_empty(), "unreached library ops: {missing:?}");
}

/// All four stages, including `OnAttn` -- the only stage where `attn_score` is
/// legal, and one no golden ever entered.
#[test]
fn every_stage_is_reached_by_the_corpus() {
    let covered = measure().stages;
    let missing: Vec<_> = Stage::ALL
        .iter()
        .filter(|stage| !covered.contains(stage))
        .collect();
    assert!(missing.is_empty(), "unreached stages: {missing:?}");
}
