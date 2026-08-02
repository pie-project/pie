//! Every kernel a declaration STATES must be one its executor can resolve.
//!
//! This is the check that was missing on 2026-08-08, when a commit gave
//! gemma-4 and gpt-oss five seams each. A seam is a statement, and one of
//! them (`attn.qv`) states `pie_lora_qkv_correction` — a symbol gemma-4's
//! launcher registry had never heard of. An unknown symbol does not
//! degrade: `gemma4_validate_stated_kernels` refuses the whole plan at
//! model LOAD, so gemma-4's declared drive went silently dark and its
//! parity gate spent a day comparing the hand-written pass against
//! itself. It took a GPU and a checkpoint to notice.
//!
//! Nothing about the question needs either. The stated set comes from
//! tracing, which is pure; the resolvable set is a list of string
//! literals in a `.cpp`. So it is a unit test, and it fails in the second
//! it takes to run rather than in whatever week someone next boots that
//! family.
//!
//! Reading C++ as text is the same liberty `kernels_table.rs` takes, for
//! the same reason: the vocabulary genuinely lives on both sides of the
//! FFI, and a test that only reads one side cannot see a drift.
//!
//! # What this does NOT cover, and why it is not one more loop
//!
//! The seams commit broke gemma-4 TWICE, and only the first half is
//! checked here. The second was an op KIND: `HookSite`, which no arm in
//! either executor answered, so the first decode fire threw "op kind 25
//! has no emission rule" — the same model-LOAD failure, from a statement
//! rather than a symbol.
//!
//! The obvious extension (every `OpKind` in the plan must appear as a
//! `case PieForwardOpKind::` label) reports false positives, and the
//! reason is worth writing down. The executor does not walk the TRACED
//! op list; it walks what `lower()` returns, and lowering resolves some
//! kinds away. llama_like handles thirteen kinds and has no label for
//! `Guard`, `Peel` or `HookSite` — the first two because lowering turns
//! them into regions and row splits, and `HookSite` because that family
//! reads its sites as fire-level sidebands rather than as ops in the
//! walk. qwen3_5 and gemma-4 DO carry `HookSite` cases: it survives
//! lowering for them.
//!
//! So the honest form of that check compares against the LOWERED list,
//! per family, and the per-family answer is exactly what makes the naive
//! version wrong. Left undone rather than shipped wrong; the kernel half
//! below is verified against the real regression.

use model_compiler::trace::{ForwardPlan, OpKind};
use model_compiler::trace::FireClass;
use std::collections::BTreeSet;

/// The `if (k == "...")` / `if (kernel == "...")` chain a family's
/// `resolve_*_kernel` is written as. Deliberately dumb: a registry that
/// stops being a chain of literal compares should make this test fail
/// loudly (empty set) rather than pass by finding nothing to check.
fn registry(family_dir: &str) -> BTreeSet<String> {
    let path = format!(
        "{}/../driver-cuda/csrc/src/model/{family_dir}/declared_forward.cpp",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let mut out = BTreeSet::new();
    for (i, _) in text.match_indices("== \"") {
        let before = &text[..i];
        let is_kernel_compare = before.ends_with("k ")
            || before.ends_with("kernel ");
        if !is_kernel_compare {
            continue;
        }
        let rest = &text[i + 4..];
        if let Some(end) = rest.find('"') {
            out.insert(rest[..end].to_string());
        }
    }
    assert!(
        !out.is_empty(),
        "{family_dir}: found no `k == \"...\"` compares — the registry \
         changed shape and this test is no longer reading it"
    );
    out
}

fn stated(plan: &ForwardPlan) -> BTreeSet<String> {
    plan.ops
        .iter()
        .filter_map(|o| match &o.kind {
            OpKind::Launch { kernel, .. } => Some(kernel.clone()),
            _ => None,
        })
        .collect()
}

fn check(family: &str, family_dir: &str, plans: &[(FireClass, ForwardPlan)]) {
    let known = registry(family_dir);
    let mut missing: Vec<String> = Vec::new();
    for (class, plan) in plans {
        for k in stated(plan) {
            if !known.contains(&k) {
                missing.push(format!("{class:?}: {k}"));
            }
        }
    }
    missing.sort();
    missing.dedup();
    assert!(
        missing.is_empty(),
        "{family}: {} stated kernel(s) its executor cannot resolve, which \
         is a model-LOAD failure, not a fallback:\n  {}\n\
         Either give the executor an arm (a no-op arm is a real answer when \
         the family never serves the construct) or stop stating it.",
        missing.len(),
        missing.join("\n  ")
    );
}

#[test]
fn llama_like_states_only_what_it_can_execute() {
    use model::families::llama_like::forward as f;
    let facts = f::facts::LlamaLikeFacts::qwen3_0_6b();
    let cuda = f::facts::LlamaLikeCudaFacts::qwen3_0_6b_l40s();
    let plans: Vec<_> = [FireClass::Decode, FireClass::Prefill]
        .into_iter()
        .map(|c| (c, f::llama_like_cuda(&facts, &cuda, c)))
        .collect();
    check("llama_like", "llama_like", &plans);
}

#[test]
fn qwen3_5_states_only_what_it_can_execute() {
    use model::qwen_3_5::forward as f;
    let facts = f::facts::Qwen35HybridFacts::qwen3_5_0_8b();
    let cuda = f::facts::Qwen35CudaFacts::qwen3_5_0_8b_synthetic();
    let plans: Vec<_> = [FireClass::Decode, FireClass::Prefill]
        .into_iter()
        .map(|c| (c, f::qwen3_5_hybrid_cuda(&facts, &cuda, c)))
        .collect();
    check("qwen3_5", "qwen3_5", &plans);
}

#[test]
fn gemma4_states_only_what_it_can_execute() {
    use model::gemma_4::forward as f;
    let facts = f::facts::Gemma4Facts::gemma_4_e4b();
    let cuda = f::facts::Gemma4CudaFacts::gemma_4_e4b_synthetic();
    let plans: Vec<_> = [FireClass::Decode, FireClass::Prefill]
        .into_iter()
        .map(|c| (c, f::gemma4_cuda(&facts, &cuda, c)))
        .collect();
    check("gemma4", "gemma4", &plans);
}

#[test]
fn gpt_oss_states_only_what_it_can_execute() {
    use model::gpt_oss::forward as f;
    let facts = f::facts::GptOssFacts::gpt_oss_20b();
    let cuda = f::facts::GptOssCudaFacts::gpt_oss_20b_synthetic();
    let plans: Vec<_> = [FireClass::Decode, FireClass::Prefill]
        .into_iter()
        .map(|c| (c, f::gpt_oss_cuda(&facts, &cuda, c)))
        .collect();
    // gpt-oss's driver tree is named for the family it shares a pass with.
    check("gpt_oss", "mixtral", &plans);
}
