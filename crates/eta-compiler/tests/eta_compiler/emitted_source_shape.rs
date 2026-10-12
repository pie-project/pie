use eta_compiler::codegen::program::{Backend, EmittedKernel, emit_program};
use eta_compiler::plan::compile_bound;
use eta_ir::validate::bind;

use crate::msl_corpus::{
    GOLDEN_NAMES, extended_traces, golden_container, golden_profile, synthetic_traces,
};

fn every_emitted_kernel() -> Vec<(String, Backend, EmittedKernel)> {
    let mut traces: Vec<_> = GOLDEN_NAMES
        .iter()
        .map(|name| {
            (
                (*name).to_string(),
                golden_container(name),
                golden_profile(name),
            )
        })
        .collect();
    for (name, container, profile) in synthetic_traces().into_iter().chain(extended_traces()) {
        traces.push((name.to_string(), container, profile));
    }

    let mut out = Vec::new();
    for (name, container, profile) in traces {
        let Ok(bound) = bind(container, profile) else {
            continue;
        };
        let stages = compile_bound(&bound);
        for &backend in Backend::ALL {
            for kernel in emit_program(backend, &stages, &bound) {
                out.push((name.clone(), backend, kernel));
            }
        }
    }
    out
}

#[test]
fn every_emitted_kernel_is_a_source_or_a_reason() {
    for (trace, backend, kernel) in every_emitted_kernel() {
        let where_ = format!(
            "{trace} {backend:?} stage {} region {}",
            kernel.stage_index, kernel.region_index
        );
        assert_ne!(
            kernel.source.is_empty(),
            kernel.error.is_empty(),
            "{where_}: source and error must be exactly one of the two \
             (source {} bytes, error {:?})",
            kernel.source.len(),
            kernel.error
        );
        if kernel.source.is_empty() {
            assert!(
                kernel.entry_name.is_empty(),
                "{where_}: a refusal still named an entry point {:?}",
                kernel.entry_name
            );
            continue;
        }
        assert!(
            !kernel.entry_name.is_empty(),
            "{where_}: emitted a source with no entry name"
        );
        assert!(
            kernel.source.contains(&kernel.entry_name),
            "{where_}: source does not define its own entry point {:?}",
            kernel.entry_name
        );
    }
}
