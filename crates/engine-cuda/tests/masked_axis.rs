//! The `masked` axis, end to end — and the gate that says the C1 axis runs.
//!
//! **WHAT THIS FILE IS FOR.** `masked` is design §0's second supergraph axis
//! and the first one beyond decode/prefill: a per-lane fact the model
//! declares, a run-length mask on the submission, an `attention.masked` arm
//! over its own window. C1 wrote the bits path and then found the catalog
//! could not run it, for three reasons that were each somebody's to fix and
//! none of them the mask path's (build log 20). This file used to PIN those
//! three refusals so that the day one was fixed the test asserting it would
//! fail and say so. C1b fixed all three, so every one of them has flipped:
//! what is asserted now is the fix, in the same place and against the same
//! catalog.
//!
//! ```text
//! blocker            what it was                        what it is now
//! kv::probe          one geometry per kv SPACE, and     facts are keyed by
//!                    gemma states two                   ROW and by PLAN
//! the schedule       one plan_prefill read by two       gemma mints six, one
//!                    classes -> Fault::Straddled        per (reading x class)
//! the windowed arm   "fa2 has no custom+sliding"        it always had one:
//!                                                       VariantCustom IS it
//! ```
//!
//! **THE AXIS IS NO LONGER GEMMA'S ALONE, AND THIS FILE IS STILL GEMMA'S.**
//! `8cb1b6ce6` seated `attention.masked` in the qwen text as well, so the
//! catalog now has two declaring families and four documented-maskless ones.
//! The device gates below stay gemma's — it is the family that also states a
//! sliding window on most of its masked arms, so its arm exercises both terms
//! at once — but two things moved with the catalog: the sentinel names both
//! families and pins the four gaps, and the maskless-refusal gate boots
//! gpt-oss, because the qwen it used to call maskless is not one any more.
//!
//! ```text
//! cargo test -p engine-cuda --test masked_axis
//! cargo test -p engine-cuda --features cuda --test masked_axis -- --nocapture
//! ```

use model_dsl::Platform;
use model_ir::{Attention, Operation, Trace};

/// How many `attention.masked` arms a SKU's trace carries.
fn masked_arms(trace: &Trace) -> usize {
    trace.nodes
        .iter()
        .filter(|node| matches!(node.op, Operation::Attention(Attention::Masked { .. })))
        .count()
}

/// **WHO DECLARES THE AXIS, AND WHO IS DOCUMENTED AS NOT.**
///
/// Stated rather than assumed, because the device gates below are gemma's and
/// a reader is entitled to know that this is a choice about where the hardware
/// coverage is rather than a fact about the catalog. `masked` is a
/// model-declared fact (design §8), the bits are a runtime input, and a plan
/// with no `attention.masked` node has nowhere for them to go.
///
/// **TWO FAMILIES DECLARE IT.** Gemma always did. Qwen joined at `8cb1b6ce6`,
/// which added `Facts::masked` as a fifth fact FIRST in the priority split and
/// `ops::attn::masked` as a fourth arm of the attention merge. So "a qwen lane
/// carrying a mask is a mask nothing reads", which this gate used to say, is
/// no longer true — and the maskless-refusal gate below had to move off qwen
/// to keep meaning anything.
///
/// **THE OTHER FOUR ARE MASKLESS ON PURPOSE**, each for a written reason
/// (`8cb1b6ce6`'s report): gpt-oss and deepseek-v4 fold learned sinks — and
/// deepseek's pooled long-range merge — through the LSE, which
/// `Attention::Masked` does not export; glm-5 and kimi-k3 attend through MLA
/// latent caches with absorbed queries, and no masked MLA variant exists in
/// the vocabulary. This gate is the sentinel on that pair of claims: the day
/// one of those four grows the arm, or one of the two loses it, is the day
/// this fails and says which.
///
/// PURE CPU, and not `#[ignore]`d for that reason: it reads the catalog's
/// traces and loads no checkpoint and no device.
#[test]
fn the_masked_axis_is_declared_by_gemma_and_qwen_and_by_nobody_else() {
    // The families whose texts state `attention.masked`, by SKU prefix. The
    // qwen38 row joined with the qwen4 campaign: its hybrid keeps full
    // attention rows, and those rows carry the mask predicate qwen35 does.
    const DECLARE: [&str; 4] = ["gemma4-", "qwen35-", "qwen36-", "qwen38-"];
    // And the four with a written kernel gap, which must stay maskless.
    const GAPPED: [&str; 4] = ["dsv4-", "glm5-", "gptoss-", "kimik3-"];

    let mut declaring: Vec<(String, usize)> = Vec::new();
    let mut maskless: Vec<String> = Vec::new();
    for row in models::skus() {
        let (sku, trace) = (row.name.as_str(), row.trace);
        let arms = masked_arms(&trace(Platform::Cuda));
        if arms > 0 {
            declaring.push((sku.to_string(), arms));
        } else {
            maskless.push(sku.to_string());
        }
    }

    assert!(
        declaring
            .iter()
            .all(|(sku, _)| DECLARE.iter().any(|family| sku.starts_with(family))),
        "a family beyond gemma and qwen declares `attention.masked`, and the \
         device gates in this file were written against gemma: {declaring:?}"
    );
    assert!(
        !declaring.is_empty(),
        "no SKU declares `attention.masked` at all, and then the axis has no \
         model text to be exercised by"
    );

    // BOTH of them, and not just one. A text that DROPPED the arm would pass
    // the prefix check above by simply not appearing in the list.
    for family in DECLARE {
        assert!(
            declaring.iter().any(|(sku, _)| sku.starts_with(family)),
            "no `{family}*` SKU declares `attention.masked` any more, so the \
             axis lost a family: {declaring:?}"
        );
    }

    // And the documented gaps stay gaps. This is the half that gives the
    // maskless-refusal gate below an artifact to stand on: it boots
    // `gptoss-20b-*` precisely because this asserts gpt-oss bakes no arm.
    for family in GAPPED {
        let grew: Vec<&(String, usize)> = declaring
            .iter()
            .filter(|(sku, _)| sku.starts_with(family))
            .collect();
        assert!(
            grew.is_empty(),
            "`{family}*` grew an `attention.masked` arm, and a kernel gap was \
             written down as the reason it could not have one — the note and \
             the text now disagree: {grew:?}"
        );
        assert!(
            maskless.iter().any(|sku| sku.starts_with(family)),
            "no `{family}*` SKU is in the catalog at all, so this gate asserts \
             nothing about it"
        );
    }
}

// ── THE GATE: gemma, on a device, with all three classes co-firing ─────────

/// The load, shared with `serve_smoke` in shape and stated here rather than
/// imported because a test binary is its own crate.

/// The maskless rig: the family with a WRITTEN reason it cannot carry the arm.
///
/// **IT REPLACED A QWEN RIG THAT HAD EXACTLY ONE USER.** The refusal gate below
/// used to boot qwen through a `common` module, because qwen baked no masked
/// arm; `8cb1b6ce6` gave it one, and the module went from "the maskless rig" to
/// "a second masked rig" without a line of it changing. Nothing else in this
/// file read it — the device gates all stand on `gemma` — so it is gone rather
/// than kept beside this one, and the name here states the property the gate
/// actually depends on instead of the model that happened to have it.
///
/// gpt-oss folds a learned sink through the LSE, which `Attention::Masked` does
/// not export. That is a written reason rather than an accident, and the
/// sentinel at the top of this file fails the day it stops being true.
mod maskless {
    

    
    
    

}

// ─────────────────────────────────────────────────────────────────────────────
// THE DEVICE-GEOMETRY CLASS, AGAINST THE HOST-GEOMETRY FIRE IT MUST EQUAL
// ─────────────────────────────────────────────────────────────────────────────

/// The device-geometry gate's fixture: one guest program that is nothing but
/// descriptor ports, and the two fires it is compared through.
///
/// **THE PROGRAM HAS NO BODY ON PURPOSE.** What is under test is the
/// descriptor-port plane — `program::ports` reading committed cells and
/// `serve::prepare` using them — and a stage that computed anything would put
/// its own arithmetic between the seeds this module writes and the geometry
/// the fire resolves. The channels are seeded and the epilogue does nothing,
/// so the cell the port reads is the cell this file wrote, and a wrong logit
/// is the shell's reading of it.
mod devgeo {
    
    
    
    
    
    
    
    

}

/// The gemma load, and the greedy loop the gates above share.
mod gemma {
    
    

    
    
    
    

}
