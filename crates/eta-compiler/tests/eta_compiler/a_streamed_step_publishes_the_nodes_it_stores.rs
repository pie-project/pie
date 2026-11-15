//! A Metal streamed kernel publishes, beside its dispatch words, the nodes
//! each step computes; an engine lays the region's scratch over them (a
//! value whose last reader's step has run gives its bytes to the next). So
//! the list must be what the step's dispatch stores: every node its `case`
//! computes is listed for that step, and no other. A later emitter change
//! that fuses more nodes into a pass, or moves one between steps, without
//! publishing it fails here rather than shrinking a layout under a kernel.

use eta_compiler::codegen::metal::step_node_table;
use eta_compiler::codegen::program::{Backend, KernelKind, emit_program};
use eta_compiler::plan::compile_bound;

use crate::msl_corpus::{
    GOLDEN_NAMES, extended_traces, golden_container, golden_served, synthetic_traces,
};

/// Whether a case's text names value `value`: its slot (`offsets[v]`), its
/// descriptor (`descriptors[v]`) or its register (`r_v`).
fn names(case: &str, value: u32) -> bool {
    reaches(case, value) || case.contains(format!("descriptors[{value}]").as_str())
}

/// Whether a case's text reaches value `value`'s bytes: its slot or its
/// register (a descriptor names its shape, not its bytes).
fn reaches(case: &str, value: u32) -> bool {
    [format!("offsets[{value}]"), format!("r_{value} ")]
        .iter()
        .any(|needle| case.contains(needle.as_str()))
}

/// Each `case N:` block of a streamed kernel's switch, in step order.
fn cases(source: &str) -> Vec<String> {
    let Some(switch) = source.find("switch (step.index)") else {
        return Vec::new();
    };
    let body = &source[switch..];
    let mut out = Vec::new();
    let mut step = 0usize;
    loop {
        let open = format!("  case {step}u: {{");
        let Some(start) = body.find(&open) else { break };
        let after = &body[start + open.len()..];
        let end = after
            .find(&format!("  case {}u: {{", step + 1))
            .or_else(|| after.find("  default:"))
            .unwrap_or(after.len());
        out.push(after[..end].to_string());
        step += 1;
    }
    out
}

#[test]
fn a_streamed_step_publishes_the_nodes_it_stores() {
    let mut traces: Vec<_> = GOLDEN_NAMES
        .iter()
        .map(|name| ((*name).to_string(), golden_container(name), golden_served(name)))
        .collect();
    for (name, container, served) in synthetic_traces().into_iter().chain(extended_traces()) {
        traces.push((name.to_string(), container, served));
    }
    let mut checked = 0usize;
    for (name, container, served) in traces {
        let Ok(bound) = served.bind(container) else {
            continue;
        };
        let stages = compile_bound(&bound);
        for kernel in emit_program(Backend::Metal, &stages, &bound) {
            if kernel.kind != KernelKind::Streamed || kernel.source.is_empty() {
                continue;
            }
            let (steps, nodes) = step_node_table(&kernel.steps);
            let Some(nodes) = nodes else {
                continue;
            };
            let where_ = format!(
                "{name} stage {} region {}",
                kernel.stage_index, kernel.region_index
            );
            let cases = cases(&kernel.source);
            assert_eq!(
                cases.len(),
                steps.len(),
                "{where_}: the switch has a case a dispatch word"
            );
            let stage = &stages[kernel.stage_index as usize];
            let mut result_of = Vec::with_capacity(stage.normalized.ops.len());
            let mut at = 0u32;
            for op in &stage.normalized.ops {
                result_of.push(at..at + op.result_count());
                at += op.result_count();
            }
            let region = &stage.fused.regions[kernel.region_index as usize];
            for node in region.nodes.iter().map(|node| node.get()) {
                let results = result_of[node as usize].clone();
                let steps_of: Vec<usize> = nodes
                    .iter()
                    .enumerate()
                    .filter(|(_, listed)| listed.contains(&node))
                    .map(|(step, _)| step)
                    .collect();
                // A node with no result writes nothing a layout holds.
                if results.is_empty() {
                    continue;
                }
                let Some(&first) = steps_of.first() else {
                    // In no step (a view its readers resolve through): then
                    // no dispatch may name a result of it.
                    for (step, case) in cases.iter().enumerate() {
                        assert!(
                            !results.clone().any(|value| reaches(case, value)),
                            "{where_} step {step}: names a result of node {node} {results:?}, \
                             which no step publishes; its case:\n{case}"
                        );
                    }
                    continue;
                };
                // Computed where it is listed: one of its steps names one of
                // its results (a reduction split over steps reads its input in
                // the first and writes its result in the last; the engine
                // holds every step between as one launch).
                assert!(
                    steps_of
                        .iter()
                        .any(|&step| results.clone().any(|value| names(&cases[step], value))),
                    "{where_}: node {node} is published for steps {steps_of:?}, none of whose \
                     dispatches names one of its results {results:?}"
                );
                // And nowhere earlier: no step before its first names a result
                // of it (a node fused into an earlier pass unpublished).
                for (step, case) in cases.iter().enumerate().take(first) {
                    assert!(
                        !results.clone().any(|value| reaches(case, value)),
                        "{where_} step {step}: names a result of node {node} {results:?}, \
                         which is first published for step {first}; its case:\n{case}"
                    );
                }
            }
            checked += 1;
        }
    }
    assert!(checked > 0, "no streamed kernel published its steps' nodes");
    eprintln!("{checked} streamed kernels: every step publishes the nodes it stores");
}
