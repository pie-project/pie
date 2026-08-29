//! The authoring-time half of palo build log 20's second blocker: an attention
//! SCHEDULE may only be read in the window it was built in.
//!
//! **WHY THIS IS A MODEL TEST AND NOT AN ENGINE ONE.** A schedule is not a
//! row-shaped table that slices — it is a carving. How many requests it
//! batches, where each request's query rows begin, how its work items split
//! the kv and how much of its grant it padded to are all fixed when the
//! builder walks the window it was dispatched in. The compiler then narrows a
//! prepare node by DEMAND to the union of the classes reading its struct
//! (design build log 7), which is the right answer for a shared tensor and the
//! wrong SHAPE for two windowed readers: an arm standing in a narrower window
//! hands the schedule its own rebased boundaries, and every work item past the
//! first request indexes a `qo_indptr` that has already ended. Nothing faults
//! on the device — the reads land in whatever follows a `[lanes + 1]` vector —
//! and the answer is wrong logits.
//!
//! So it is a property of the MODEL TEXT, and the cheapest instant to say so
//! is the sweep the author already runs. `resolve_classes` gives the per-node
//! class set outright (`ClassTable::node_mask`); regions are maximal runs of
//! EQUAL masks, so this predicate and `model_compiler`'s
//! `model_compiler::Error::Straddled` — which the shell's load path asks — are literally
//! the same comparison, asked one pass earlier and with no compiler in the
//! room.
//!
//! What a straddle costs the author is one line: build the second reader its
//! own schedule off its own arm — `ops::attn::plan_prefill(&arm)` (or
//! `plan_decode`, or `mla_plan`) over that arm of `inputs.split(..)`. The
//! class is then stated where the schedule is built, and the recorder refuses
//! a reader standing in another arm. Gemma builds six, one per
//! (reading × class), and gpt-oss four.

use model_dsl::{Attention, Def, Operands, Operation, Trace, Platform, resolve_classes};

/// Every platform a plan can be traced at: a model text may emit a different op
/// per platform, so one platform passing says nothing about the others.
const PLATFORMS: [Platform; 4] = [
    Platform::Cuda,
    Platform::Metal,
    Platform::Wgpu,
    Platform::Vulkan,
];

/// Whether a node DEFINES an attention schedule — the three plan ops, which
/// are exactly the `Ty::Struct` producers this IR has.
fn schedule(op: &Operation) -> bool {
    matches!(
        op,
        Operation::Attention(
            Attention::PlanDecode { .. } | Attention::PlanPrefill { .. } | Attention::MlaPlan { .. }
        )
    )
}

/// Every `(schedule, reader)` pair whose class sets differ, as sentences.
fn straddles(trace: &Trace) -> Vec<String> {
    let Ok(classes) = resolve_classes(trace) else {
        // A plan whose merges do not resolve is somebody else's test
        // (`every_class_resolves_every_merge`); saying it twice here would
        // only make one authoring mistake two failures.
        return Vec::new();
    };
    let mut found = Vec::new();
    let mut inputs = Vec::new();
    for (at, node) in trace.nodes.iter().enumerate() {
        inputs.clear();
        node.op.inputs(&mut inputs);
        for &read in &inputs {
            let Some(Def::Op(built_by)) = trace.values.get(read.0 as usize).map(|v| &v.def) else {
                continue;
            };
            if !schedule(&trace.nodes[*built_by as usize].op) {
                continue;
            }
            let planned = &classes.node_mask[*built_by as usize];
            let reader = &classes.node_mask[at];
            if planned != reader {
                found.push(format!(
                    "v{} is built by `{}` over classes {:?} and read by `{}` (node {at}) \
                     in classes {:?}",
                    read.0,
                    trace.nodes[*built_by as usize].op.name(),
                    planned.iter().collect::<Vec<_>>(),
                    node.op.name(),
                    reader.iter().collect::<Vec<_>>(),
                ));
            }
        }
    }
    found
}

#[test]
fn no_shipped_schedule_is_read_outside_the_window_it_was_built_in() {
    let mut faults = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            for line in straddles(&trace(platform)) {
                faults.push(format!("`{sku}` as {platform:?}: {line}"));
            }
        }
    }

    assert!(
        faults.is_empty(),
        "an attention schedule is read outside the window it was carved in. The fix is \
         model text: build each reader class its own schedule off its own arm of \
         `inputs.split(..)`, the way gemma_4 and gpt_oss do.\n{}\n",
        faults.join("\n"),
    );
}

/// And the net is not vacuous: gemma's own text, with the six schedules
/// collapsed back to two, is caught.
///
/// **THE OTHER HALF OF AN ASSERTION.** A check that passes on a catalog nobody
/// can straddle proves nothing about the check. This rebuilds the exact defect
/// C1 recorded — one `plan_prefill` shared by the prefill arm and the masked
/// arm — out of the shipped gemma trace, by rewriting every masked node to
/// read the prefill arm's schedule instead of its own, and asserts the sweep
/// says so.
#[test]
fn a_schedule_shared_by_two_classes_is_caught() {
    let mut trace = model::trace_of("gemma4-e4b-bf16-kv-bf16")
        .expect("the catalog ships gemma")(Platform::Cuda);
    assert!(
        straddles(&trace).is_empty(),
        "gemma ships straddle-free, which is what makes the rewrite below a defect"
    );

    // The prefill arm's schedule for each reading, found by walking the arms
    // themselves — the plan value ids are the trace's business, not this
    // test's.
    let mut prefill_plan_of_width = std::collections::HashMap::new();
    for node in &trace.nodes {
        if let Operation::Attention(Attention::Prefill { plan, head_dim, .. }) = &node.op {
            prefill_plan_of_width.insert(*head_dim, *plan);
        }
    }
    assert_eq!(
        prefill_plan_of_width.len(),
        2,
        "gemma prefills at two head widths, 256 sliding and 512 global"
    );

    let mut rewritten = 0usize;
    for node in &mut trace.nodes {
        if let Operation::Attention(Attention::Masked {
            plan: at, head_dim, ..
        }) = &mut node.op
        {
            *at = prefill_plan_of_width[head_dim];
            rewritten += 1;
        }
    }
    assert!(rewritten > 0, "gemma declares masked arms to rewrite");

    // Both readers of each now-shared schedule are named, which is the point:
    // demand widened the prepare node to the union {prefill, masked}, so the
    // prefill arm is straddling its own schedule too.
    let caught = straddles(&trace);
    assert_eq!(
        caught.len(),
        2 * rewritten,
        "each shared schedule is straddled from both sides — by the masked arm that \
         borrowed it and by the prefill arm it was carved for:\n{}",
        caught.join("\n"),
    );
    assert!(
        caught.iter().any(|line| line.contains("attention.masked"))
            && caught.iter().any(|line| line.contains("attention.prefill")),
        "both arms are named"
    );
}
