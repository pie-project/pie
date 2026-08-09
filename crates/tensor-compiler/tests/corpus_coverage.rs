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

use msl_corpus::{GOLDEN_NAMES, corpus_stages, extended_stages, golden_dir};
use tensor_compiler::plan::{LibraryOp, RegionKind, ScheduleTemplate};
use tensor_ir::op::{IntrinsicId, OP_TABLE, Op};
use tensor_ir::registry::Stage;

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

/// One level up from the rest of this file: is there a golden on disk that
/// nothing compiles?
///
/// `GOLDEN_NAMES` is what every emitter sweep enumerates and `tests/golden/`
/// is where the containers live, and nothing connected the two. A file added
/// to the directory was read by no test; a name left behind after a deletion
/// would fail late, in whichever sweep reached it first, with "missing".
///
/// The comparison is order-sensitive on purpose. The order is the order cases
/// appear in every golden dump, so it is part of what those dumps pin.
#[test]
fn every_golden_on_disk_is_named_by_the_corpus() {
    let mut on_disk: Vec<String> = std::fs::read_dir(golden_dir())
        .expect("compiler/tests/golden/")
        .map(|entry| entry.expect("golden dir entry").file_name())
        .filter_map(|name| {
            name.to_str()
                .and_then(|name| name.strip_suffix(".txt"))
                .map(str::to_string)
        })
        .collect();
    on_disk.sort();
    let named: Vec<String> = GOLDEN_NAMES.iter().map(|n| (*n).to_string()).collect();
    assert_eq!(
        on_disk, named,
        "compiler/tests/golden/ and GOLDEN_NAMES disagree. This corpus is \
         frozen -- new coverage belongs in `extended_traces` -- so a file here \
         that is not named is either a mistake or a decision to unfreeze it, \
         and a name with no file is a golden that was deleted without saying so"
    );
}
