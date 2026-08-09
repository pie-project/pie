//! EVERY FAMILY'S PLAN HAS THE SAME TWO ENDS.
//!
//! `cuda.md` §5.C3 asks for the goldens to be the equality — that a
//! *described* plan lowers to the same `Launch` list a coded one does.
//! The description does not exist yet (that is C2's endpoint), but part
//! of it does: the shared blocks in `model_compiler::dsl` are the
//! vocabulary a description would be written in, and this asks whether
//! they are a real STRUCTURE or only reused code.
//!
//! The claim: every family text begins with `embedded_prologue`'s pair —
//! the entry seam, then the embedding — and ends with
//! `logits_epilogue`'s — the readout, then the exit seam. Not "usually",
//! and not "the ones that happen to call the block": every one.
//!
//! It is worth pinning because both ends are ORDER contracts and neither
//! is locally checkable. The entry seam is where attached programs bind
//! their inputs, so a family that embedded first would hand them a value
//! they were supposed to influence. The exit seam is what sampling
//! attaches to, so a family that omitted it would trace a plan nothing
//! can read from — a failure with no symptom until something tries to
//! sample.
//!
//! A family that grows a new prologue is not forbidden; gemma-4's PLE
//! table and gemma-3n's altUp streams both build something before the
//! backbone. What they may not do is skip the seam, and the assertions
//! below are written to say exactly that.

#![cfg(feature = "forward")]

use model_compiler::trace::{FireClass, ForwardPlan, OpKind};

/// Every family this crate can trace for CUDA, by name and plan.
///
/// Deliberately a list rather than a registry walk: a family missing from
/// here is a family nobody checked, and a list makes that visible in a
/// diff. The `Decode` class alone, because both ends are class-invariant
/// — a prefill has the same prologue and the same readout.
fn plans() -> Vec<(&'static str, ForwardPlan)> {
    use model::families::llama_like::forward::facts::{LlamaLikeCudaFacts, LlamaLikeFacts};
    vec![
        (
            "llama_like",
            model::families::llama_like::forward::llama_like_cuda(
                &LlamaLikeFacts::qwen3_0_6b(),
                &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
                FireClass::Decode,
            ),
        ),
        (
            "gemma_2",
            model::gemma_2::forward::gemma2_cuda(
                &model::gemma_2::forward::facts::Gemma2Facts::gemma_2_9b(),
                FireClass::Decode,
            ),
        ),
        (
            "glm5",
            model::glm5::forward::glm5_cuda(
                &model::glm5::forward::facts::Glm5Facts::glm5_106b_a12b(),
                FireClass::Decode,
            ),
        ),
        (
            "kimi_k3",
            model::kimi_k3::forward::kimi_k3_cuda(
                &model::kimi_k3::forward::facts::KimiK3Facts::kimi_k3_synthetic(),
                FireClass::Decode,
            ),
        ),
        (
            "deepseek_v4",
            model::deepseek_v4::forward::dsv4_cuda(
                &model::deepseek_v4::forward::facts::Dsv4Facts::dsv4_synthetic(),
                FireClass::Decode,
            ),
        ),
    ]
}

/// The values a named seam publishes, if the plan states it.
///
/// `in` and `out` carry `op: None` deliberately — they are BOUNDARIES
/// rather than attachments, unlike `attn.q`/`attn.out` which point at the
/// statement they observe. So their POSITION is not representable and
/// this does not pretend otherwise; what is representable, and is the
/// stronger claim anyway, is WHICH VALUE the boundary exposes.
fn seam_values<'a>(plan: &'a ForwardPlan, name: &str) -> Option<&'a [model_compiler::trace::ValueId]> {
    plan.seams.iter().find(|s| s.seam == name).map(|s| s.values.as_slice())
}

#[test]
fn every_plan_states_both_boundaries() {
    for (name, plan) in plans() {
        assert!(
            seam_values(&plan, "in").is_some(),
            "{name}: no entry seam. It is where attached programs bind \
             their inputs, so a plan without one cannot be influenced — \
             and `dsl::embedded_prologue` is the pair that states it."
        );
        assert!(
            seam_values(&plan, "out").is_some(),
            "{name}: no exit seam. Sampling and host-visible emits attach \
             there, so this plan traces and then cannot be read from — a \
             failure with no symptom until something tries to sample."
        );
        assert_eq!(
            plan.ops.iter().filter(|o| matches!(o.kind, OpKind::Embed { .. })).count(),
            1,
            "{name}: a decode backbone embeds exactly once"
        );
        assert_eq!(
            plan.ops.iter().filter(|o| matches!(o.kind, OpKind::LmHead { .. })).count(),
            1,
            "{name}: a decode backbone reads out exactly once"
        );
    }
}

#[test]
fn the_exit_seam_publishes_what_the_readout_produced() {
    for (name, plan) in plans() {
        let published = seam_values(&plan, "out").expect("stated above");
        assert_eq!(published.len(), 1, "{name}: the exit seam names one value");
        let logits = published[0];

        // The value the seam names must be the LAST thing the epilogue
        // wrote: the readout's result, or the softcap's if the deployment
        // caps. Naming the readout's OPERAND instead is the mistake this
        // catches, and it is a quiet one — the plan traces, and the
        // sampler reads a pre-readout activation shaped like nothing.
        let head = plan
            .ops
            .iter()
            .rposition(|o| matches!(o.kind, OpKind::LmHead { .. }))
            .expect("counted above");
        let produced_after_head: Vec<_> = plan.ops[head..]
            .iter()
            .flat_map(|o| o.outputs.iter().copied())
            .collect();
        assert!(
            produced_after_head.contains(&logits),
            "{name}: the exit seam publishes a value the readout did not \
             produce. `dsl::logits_epilogue` seams the logits it just \
             wrote; a seam on anything earlier hands the sampler a \
             pre-readout activation."
        );
    }
}

/// THE THIRD ORDER CONTRACT, and the only one representable.
///
/// `attn.out` is the OnAttn site — where attached programs read the
/// attention's result and where a score consumer binds — so it must see
/// the value BEFORE `o_proj` consumes it. A family that projected first
/// would seam a value nothing else can reach.
///
/// Unlike `in`/`out`, this seam records the op it attaches to, so the
/// claim is checkable: the value it names must still be READ by some
/// statement after it. A seam on a value nothing later consumes is a
/// dead observation point, which is exactly what projecting first
/// produces.
///
/// Both orders trace and both lower, and the difference shows only when
/// something attaches — which is why this is a test rather than a
/// comment.
///
/// `llama_like` IS EXCLUDED, and the reason is structural rather than a
/// waiver. Its attention sits inside a value-producing GUARD, so the
/// arm's value and the region's output are different `ValueId`s by
/// construction — that is what `Lowering::region_outs` exists to
/// reconcile. The seam names the arm's; `o_proj` reads the region's; and
/// "is this value read later" cannot see through that. The families
/// below state their attention as a plain statement, where the question
/// means what it says.
#[test]
fn the_attention_seam_sees_a_value_something_later_consumes() {
    for (name, plan) in plans().into_iter().filter(|(n, _)| *n != "llama_like") {
        let sites: Vec<_> = plan
            .seams
            .iter()
            .filter(|s| s.seam == "attn.out")
            .filter_map(|s| s.op.map(|op| (op as usize, s.values.clone())))
            .collect();
        assert!(
            !sites.is_empty(),
            "{name}: no `attn.out` seam anywhere. Every layer with an \
             attention has one; a family with none has no OnAttn site and \
             no score consumer can bind."
        );
        for (at, values) in sites {
            for v in values {
                let consumed_later = plan.ops[at + 1..]
                    .iter()
                    .any(|o| o.inputs.contains(&v));
                assert!(
                    consumed_later,
                    "{name}: the `attn.out` seam at op {at} names a value \
                     nothing after it reads. `o_proj` is supposed to be \
                     that reader, so this family projected BEFORE seaming \
                     and the observation point is dead. \
                     `dsl::attention_landing` is the pair in the right \
                     order."
                );
            }
        }
    }
}
