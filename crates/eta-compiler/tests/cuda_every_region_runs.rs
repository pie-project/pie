//! Every region a plan declares reaches the CUDA engine as something it can
//! launch — or this file says why it does not.
//!
//! The CUDA engine has no second path. It launches compiled `PIE_KERNEL_FUSED`
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
use eta_compiler::codegen::cuda::{emit_region, is_order_region, is_scan_region};
use eta_compiler::plan::{LibraryOp, Region, RegionKind};

/// Regions this backend is allowed not to emit, and why that is not a hole.
///
/// Keyed by what a region IS rather than by where it sits, so a corpus that
/// grows a case does not need an entry and a corpus that reorders one does not
/// invalidate any. Each entry is a claim that no plan a deployment builds can
/// route work through such a region.
const REFUSED: &[(&str, &str)] = &[(
    "single-op library lift with no CUDA kernel",
    "a `RegionKind::Library` wrapping one op whose own tag IS the library op, \
         and for which this backend has written no kernel. That is now \
         `matmul` and a second-party call, and nothing else. A second-party \
         call is a NAME the shell launches itself rather than a body an \
         emitter could write, and `build_stage` skips it before it reads the \
         slot; `matmul` has no generated kernel and no curated guest reaches \
         one through a library region. \
         \
         `top_k`, `sort_desc` AND `cumsum`/`cumprod` USED TO BE ON THIS LIST. \
         `codegen::cuda::order` and `codegen::cuda::scan` emit generated \
         kernels for them, which is what the two claim tests below assert. \
         They came off because the excuse was false: four curated inferlets \
         route work through exactly such a region — `beam-search` (`top_k`), \
         `locally-typical-sampling` and `tail-free-sampling` (`top_k` then \
         `cumsum`), and `mtp-speculative-decoding` (`cumprod`) — so `no plan a \
         deployment builds can route work through such a region` was a claim \
         about the plans nobody had checked. Each failed to register with \
         `stage 0 region N was declined by the emitter`, and the reason names \
         which boundary op it was.",
)];

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
    // The four that emit: `NucleusSample` is the one multi-op lift, and
    // `TopK` / `Sort` / `Scan` are the single-op lifts with kernels of their
    // own. What is left is a single node carrying a boundary tag with nothing
    // to run it with.
    if matches!(
        claimed,
        LibraryOp::NucleusSample | LibraryOp::TopK | LibraryOp::Sort | LibraryOp::Scan
    ) {
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
            // `emit_region` and not `emit_fused_region`: the question is what
            // reaches the engine's `KernelKind::Fused` slot, and choosing the
            // emitter that fills it is that function's job. Asking the fused
            // one directly would report a hole for every region served by a
            // library kernel — which is the answer this file wants for the ops
            // that have none and the wrong answer for `top_k`, which has one.
            let emitted = emit_region("probe", &plan, region);
            let ok = emitted.is_ok();
            let why = emitted.err().map(|e| e.to_string()).unwrap_or_default();
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
        .map(|(stage, region, _, _, why)| format!("{stage} region {region}: {why}"))
        .collect();
    assert!(
        holes.is_empty(),
        "these regions are planned but not emitted, and nothing in `REFUSED` \
         covers them. The CUDA engine launches compiled regions and nothing \
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
            let emitted = emit_region("probe", &plan, region);
            assert!(
                emitted.is_ok(),
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

/// The `top_k` ranking in particular, named for the same reason the sampler is.
///
/// It is the second region a curated inferlet actually routes work through and
/// the second that used to be skipped: `beam-search` cuts its candidates with
/// it every token, and `locally-typical-sampling` and `tail-free-sampling`
/// both materialise their candidate order with it. All three failed to
/// register on this backend with `stage 0 region 1 was declined by the emitter
/// (generated region contains a non-generated boundary (top_k))` until
/// `codegen::cuda::topk` existed.
#[test]
fn the_top_k_ranking_emits_a_kernel() {
    let mut found = 0;
    for stage in corpus_stages().into_iter().chain(extended_stages()) {
        let plan = stage.plan;
        for region in plan.fused.regions.iter() {
            if region.kind != RegionKind::Library(LibraryOp::TopK) {
                continue;
            }
            found += 1;
            assert!(
                is_order_region(&plan, region),
                "{}#{} claims `Library(TopK)` around something that is not a \
                 `top_k`, so the emitter would rank the wrong operand",
                stage.golden,
                stage.stage_index,
            );
            let emitted = emit_region("probe", &plan, region);
            assert!(
                emitted.is_ok(),
                "{}#{} holds a top_k region the CUDA emitter will not emit, so \
                 every ranking guest silently reads zeros or fails to register: \
                 {:?}",
                stage.golden,
                stage.stage_index,
                emitted.err(),
            );
        }
    }
    assert!(
        found > 0,
        "no corpus stage plans a `top_k` region, so this file proves nothing \
         about ranking. `beam_epilogue` and `pentathlon_iter` both hold one; if \
         they stopped producing a library region, that is the finding."
    );
}

/// The scan prefix, named for the same reason the sampler and the ranking are.
///
/// It is the boundary that outlived `top_k`: with the ranking emitting,
/// `locally-typical-sampling` and `tail-free-sampling` got one region further
/// and failed on `generated region contains a non-generated boundary (scan)`,
/// because both cut their candidate set with `cumsum(p) - p`.
/// `mtp-speculative-decoding` builds its accept prefix with `cumprod` and
/// failed the same way.
#[test]
fn the_scan_prefix_emits_a_kernel() {
    let mut found = 0;
    for stage in corpus_stages().into_iter().chain(extended_stages()) {
        let plan = stage.plan;
        for region in plan.fused.regions.iter() {
            if region.kind != RegionKind::Library(LibraryOp::Scan) {
                continue;
            }
            found += 1;
            assert!(
                is_scan_region(&plan, region),
                "{}#{} claims `Library(Scan)` around something that is not a \
                 `cumsum` or `cumprod`, so the emitter would prefix the wrong \
                 operand",
                stage.golden,
                stage.stage_index,
            );
            let emitted = emit_region("probe", &plan, region);
            assert!(
                emitted.is_ok(),
                "{}#{} holds a scan region the CUDA emitter will not emit, so \
                 every prefix-cutting guest silently reads zeros or fails to \
                 register: {:?}",
                stage.golden,
                stage.stage_index,
                emitted.err(),
            );
        }
    }
    assert!(
        found > 0,
        "no corpus stage plans a scan region, so this file proves nothing about \
         prefixes. `mtp_verify_tail` and `pentathlon_iter` both hold one; if \
         they stopped producing a library region, that is the finding."
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
