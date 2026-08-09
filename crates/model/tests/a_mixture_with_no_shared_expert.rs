//! A row whose mixture has no shared expert binds no shared expert.
//!
//! Five families compute a shared FFN beside the routed one and fold the
//! block away when the row states no shared intermediate. Every row this
//! build ships has one, so the fold had never been traced -- and a text
//! that kept the block binds `shared_gate`, `shared_up` and `shared_down`,
//! three weights the checkpoint of a no-shared-expert row does not
//! contain. The load fails naming a tensor the model is not supposed to
//! have, which reads as a corrupt checkpoint rather than as this bug.
//!
//! The families are asserted together because the texts are separate
//! statements of one rule and drift independently: gpt-oss's counterpart
//! lives beside its own text and caught nothing about these four.
//!
//! What is asserted is the WEIGHT NAMES the plan binds, not the op count.
//! An op count moves when a kernel is fused and says nothing about which
//! tensors the bind will look up, which is the thing that fails.

use model_compiler::trace::{FireClass, ForwardPlan, OpKind};

/// Every weight name the plan's ops name.
fn bound(plan: &ForwardPlan) -> Vec<String> {
    let mut out = Vec::new();
    for op in &plan.ops {
        collect(&op.kind, &mut out);
    }
    out
}

fn collect(kind: &OpKind, out: &mut Vec<String>) {
    let json = serde_json::to_value(kind).expect("an op kind serializes");
    walk(&json, out);
}

fn walk(v: &serde_json::Value, out: &mut Vec<String>) {
    match v {
        serde_json::Value::String(s) => out.push(s.clone()),
        serde_json::Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
        serde_json::Value::Object(o) => o.values().for_each(|x| walk(x, out)),
        _ => {}
    }
}

/// The `needle` has to be present in the STATED row and absent in the
/// zeroed one. Asserting the presence too is what stops this from
/// passing against a text that folded the block unconditionally, or
/// against a needle that never appears at all because it was misspelt.
fn folds(family: &str, needle: &str, with: &[String], without: &[String]) {
    assert!(
        with.iter().any(|n| n.contains(needle)),
        "{family}: the row that STATES the value does not name {needle:?}, \
         so the absence asserted below would hold for any text"
    );
    let leaked: Vec<&String> = without.iter().filter(|n| n.contains(needle)).collect();
    assert!(
        leaked.is_empty(),
        "{family}: a row that states none still names {leaked:?}"
    );
}

#[test]
fn kimi_k2_folds_the_shared_block_when_the_row_states_none() {
    use model::kimi_k2::forward::{
        facts::{KimiCudaFacts, KimiFacts},
        kimi_cuda,
    };
    let facts = KimiFacts::kimi_k2();
    let cuda = KimiCudaFacts::kimi_k2_synthetic();
    let mut none = facts.clone();
    none.moe.shared_intermediate = 0;
    folds(
        "kimi_k2",
        "shared",
        &bound(&kimi_cuda(&facts, &cuda, FireClass::Decode)),
        &bound(&kimi_cuda(&none, &cuda, FireClass::Decode)),
    );
}

#[test]
fn kimi_k3_folds_the_shared_block_when_the_row_states_none() {
    use model::kimi_k3::forward::{facts::KimiK3Facts, kimi_k3_cuda};
    let facts = KimiK3Facts::kimi_k3_synthetic();
    let mut none = facts.clone();
    none.moe.shared_intermediate = 0;
    folds(
        "kimi_k3",
        "shared",
        &bound(&kimi_k3_cuda(&facts, FireClass::Decode)),
        &bound(&kimi_k3_cuda(&none, FireClass::Decode)),
    );
}

#[test]
fn glm_5_folds_the_shared_block_when_the_row_states_none() {
    use model::glm_5::forward::{facts::Glm5Facts, glm5_cuda};
    let facts = Glm5Facts::glm5_106b_a12b();
    let mut none = facts.clone();
    none.moe.shared_intermediate = 0;
    folds(
        "glm_5",
        "shared",
        &bound(&glm5_cuda(&facts, FireClass::Decode)),
        &bound(&glm5_cuda(&none, FireClass::Decode)),
    );
}

#[test]
fn nemotron_h_folds_the_shared_block_when_the_row_states_none() {
    use model::nemotron_h::forward::{facts::NemotronHFacts, nemotron_h_cuda};
    let facts = NemotronHFacts::nemotron_h_synthetic();
    let mut none = facts.clone();
    none.moe.shared_intermediate = 0;
    folds(
        "nemotron_h",
        "shared",
        &bound(&nemotron_h_cuda(&facts, FireClass::Decode)),
        &bound(&nemotron_h_cuda(&none, FireClass::Decode)),
    );
}

/// Not a mixture, but the same shape of fold and the same reason it had
/// never run: `final_logit_softcapping` is a per-row measurement that
/// every gemma-4 row this build ships states as 30, and the generation
/// it descends from dropped the cap. A row that states 0 and still gets
/// capped logits is a silent numerical error -- no crash, no wrong
/// tensor, just a distribution that has been squashed.
#[test]
fn gemma_4_folds_the_softcap_when_the_row_states_none() {
    use model::gemma_4::forward::{
        facts::{Gemma4CudaFacts, Gemma4Facts},
        gemma4_cuda,
    };
    let facts = Gemma4Facts::gemma_4_e4b();
    let cuda = Gemma4CudaFacts::gemma_4_e4b_synthetic();
    let mut none = facts;
    none.logit_softcap = 0.0;
    folds(
        "gemma_4",
        "logit_softcap",
        &bound(&gemma4_cuda(&facts, &cuda, FireClass::Decode)),
        &bound(&gemma4_cuda(&none, &cuda, FireClass::Decode)),
    );
}
