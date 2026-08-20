//! Every region a plan declares reaches the CUDA driver as something it can
//! launch — or this file says why it does not.
//!
//! The CUDA driver has no second path. It launches compiled `PIE_KERNEL_FUSED`
//! kernels and nothing else: no tier-0 fallback for a generated region, no
//! native implementation of any `RegionKind::Library`. So a region the emitter
//! declines is a region that does not run, while the plan around it goes on
//! budgeting its scratch and reading its results.
//!
//! That is the failure this file exists for, and it is silent by construction.
//! `validate_generated_region` refused every library region for being one;
//! `emit_fused_region` therefore emitted nothing for the nucleus sampler;
//! `build_stage` read that as "the host declined on purpose" and skipped it;
//! `Prepared::build` still laid out the chain's thirteen values and zeroed
//! them. A softmax over zeros is uniform, a draw off a uniform row lands on
//! index 0, and every request at a nonzero temperature answered token 0 — with
//! no fault, no refusal, and a full green test suite. The goldens recorded it
//! the whole time, as `ok: false` on `nucleus_sample#0 fused#1`; nothing asked.
//!
//! So this asks. A refusal is allowed, but it has to be *named* here, with the
//! reason it is not a hole. [`REFUSED`] is that list, and
//! [`nothing_in_refused_is_actually_emitted`] keeps it from outliving its
//! entries. `.wiki/migration.md` §11.21 is the full account.

#[path = "common/msl_corpus.rs"]
mod msl_corpus;

use msl_corpus::{corpus_stages, extended_stages};
use tensor_compiler::codegen::cuda::{emit_fused_region, validate_generated_region};
use tensor_compiler::plan::{LibraryOp, Region, RegionKind};

/// Regions this backend is allowed not to emit, and why that is not a hole.
///
/// Keyed by what a region IS rather than by where it sits, so a corpus that
/// grows a case does not need an entry and a corpus that reorders one does not
/// invalidate any. Each entry is a claim that no plan a deployment builds can
/// route work through such a region.
const REFUSED: &[(&str, &str)] = &[
    (
        "single-op library lift",
        "a `RegionKind::Library` wrapping one op whose own tag IS the library op \
         — `top_k`, `sort_desc`, `cumsum`/`cumprod`, `matmul`, a second-party \
         call. The generic emitter would fall back to `ptir_m1_execute`, whose \
         single-threaded forms are O(len^2) or worse and do not return at a \
         real vocabulary, so refusing is the honest answer and the driver now \
         reports it at load rather than skipping it.",
    ),
];

/// One region's identity for the purposes of [`REFUSED`].
///
/// `None` when nothing here excuses it, which is what makes the assertion
/// below a question about the region rather than about this list.
///
/// IT TOOK THE STAGE TOO and read nothing off it. Every excuse in `REFUSED`
/// turns on the region's own shape -- its kind, its claimed library op, how
/// many nodes it carries -- so the stage it came from never entered the
/// answer, and passing it suggested an excuse could depend on context that no
/// excuse here does.
fn excuse(region: &Region) -> Option<&'static str> {
    let RegionKind::Library(claimed) = region.kind else {
        return None;
    };
    // The one multi-op lift emits; every other library op is a single node
    // carrying the boundary tag itself.
    if claimed == LibraryOp::NucleusSample {
        return None;
    }
    (region.nodes.len() == 1).then_some(REFUSED[0].0)
}

/// Every region of every corpus stage, with what the emitter said about it.
fn survey() -> Vec<(String, usize, bool, Option<&'static str>, String)> {
    let mut rows = Vec::new();
    for stage in corpus_stages().into_iter().chain(extended_stages()) {
        let plan = stage.plan;
        for (region_index, region) in plan.fused.regions.iter().enumerate() {
            let validated = validate_generated_region(&plan, region);
            let emitted = emit_fused_region("probe", &plan, region);
            let ok = validated.is_ok() && emitted.is_ok();
            let why = validated
                .err()
                .map(|e| e.to_string())
                .or_else(|| emitted.err().map(|e| e.to_string()))
                .unwrap_or_default();
            rows.push((
                format!("{}#{}", stage.golden, stage.stage_index),
                region_index,
                ok,
                excuse(region),
                why,
            ));
        }
    }
    rows
}

/// THE HEADLINE. A region that does not emit is a region that does not run.
#[test]
fn every_planned_region_emits_or_is_named_here() {
    let holes: Vec<String> = survey()
        .into_iter()
        .filter(|(_, _, ok, excuse, _)| !ok && excuse.is_none())
        .map(|(stage, region, _, _, why)| {
            format!("{stage} region {region}: {why}")
        })
        .collect();
    assert!(
        holes.is_empty(),
        "these regions are planned but not emitted, and nothing in `REFUSED` \
         covers them. The CUDA driver launches compiled regions and nothing \
         else, so each of these is work a plan expects to happen that will \
         not, leaving its results at the zeros `Prepared::build` wrote:\n  {}",
        holes.join("\n  ")
    );
}

/// The nucleus sampler in particular, named because it is the one that broke.
///
/// Covered by the test above, and asserted separately anyway: that test is a
/// survey whose failure names whatever it finds, and this is the specific
/// claim — the sampler every nonzero temperature runs through emits a kernel —
/// that a reader can check without reconstructing the survey.
#[test]
fn the_nucleus_sampler_emits_a_kernel() {
    let mut found = 0;
    for stage in corpus_stages().into_iter().chain(extended_stages()) {
        let plan = stage.plan;
        for region in plan.fused.regions.iter() {
            if region.kind != RegionKind::Library(LibraryOp::NucleusSample) {
                continue;
            }
            found += 1;
            let emitted = emit_fused_region("probe", &plan, region);
            assert!(
                validate_generated_region(&plan, region).is_ok() && emitted.is_ok(),
                "{}#{} holds a nucleus region the CUDA emitter will not emit, so \
                 top-p sampling does not run on this backend: {:?}",
                stage.golden,
                stage.stage_index,
                emitted.err(),
            );
        }
    }
    assert!(
        found > 0,
        "no corpus stage plans a nucleus region, so this file proves nothing \
         about the sampler. `nucleus_sample` is in `GOLDEN_NAMES`; if it stopped \
         producing a library region, that is the finding."
    );
}

/// An entry in [`REFUSED`] that no longer describes anything is a claim about
/// a shape the compiler no longer builds, and reading it later would mislead.
#[test]
fn nothing_in_refused_is_actually_emitted() {
    let mut used = vec![false; REFUSED.len()];
    let mut contradictions = Vec::new();
    for (stage, region, ok, excuse, _) in survey() {
        let Some(name) = excuse else { continue };
        let at = REFUSED.iter().position(|(key, _)| *key == name).unwrap();
        used[at] = true;
        if ok {
            contradictions.push(format!("{stage} region {region} is excused as `{name}`"));
        }
    }
    assert!(
        contradictions.is_empty(),
        "these regions emit, so excusing them records a refusal that does not \
         happen:\n  {}",
        contradictions.join("\n  ")
    );
    let dead: Vec<&str> = REFUSED
        .iter()
        .zip(&used)
        .filter(|(_, used)| !**used)
        .map(|((name, _), _)| *name)
        .collect();
    assert!(
        dead.is_empty(),
        "no corpus region matches these entries, so they excuse nothing and \
         should go: {dead:?}"
    );
}
